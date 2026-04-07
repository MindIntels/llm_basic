"""
mRoPE — Multimodal Rotary Position Embedding.

References
----------
- Qwen2-VL: https://arxiv.org/abs/2409.12191
  "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
- Original mRoPE concept: Wang et al. 2024, "RoPE to mRoPE"

Motivation
----------
Standard RoPE assigns a single integer position index to each token.
For multimodal sequences (text + images / video), each visual token occupies
a position in a *multi-dimensional* grid (time × height × width for video,
height × width for images) while text tokens receive a 1-D position.

mRoPE splits the head dimension into N independent RoPE "channels", one per
spatial/temporal axis, and applies a separate position encoding to each channel:

    For M axes and head_dim d:
      channel size c = d // M    (must be even per channel)
      Each channel encodes its own axis position independently.

Qwen2-VL uses 3 axes for images/video:  [temporal, height, width].
Text tokens receive the same position index on all three axes.

Position ID format
------------------
mRoPE expects ``position_ids`` of shape  [B, M, S]  where:
  - M = number of axes (e.g., 3)
  - position_ids[:, 0, :] = temporal / text positions
  - position_ids[:, 1, :] = height positions
  - position_ids[:, 2, :] = width positions

For text tokens set all M axes to the same monotonically increasing integer.
For image patches, each patch gets a (t, h, w) triple.

Implementation details
----------------------
Channel assignment (Qwen2-VL style, equivalent to concatenating M RoPEs):

    q_out[..., j*c:(j+1)*c] = apply_rotary(q[..., j*c:(j+1)*c],
                                            cos_j, sin_j)
    where cos_j, sin_j are computed from axis j's position_ids.

This module provides:
  * ``MultimodalRotaryEmbedding``  — M-axis RoPE cache + forward.
  * ``mRoPEAttention``             — MHA with built-in mRoPE.
  * ``make_text_position_ids``     — helper: construct 1-D position [B, M, S].
  * ``make_image_position_ids``    — helper: construct 2-D grid (h, w) ids.
  * ``make_video_position_ids``    — helper: construct 3-D grid (t, h, w) ids.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from .safe_softmax import safe_softmax
from .rope import rotate_half


# =========================================================================== #
#  Position-ID helpers                                                         #
# =========================================================================== #

def make_text_position_ids(
    seq_len: int,
    num_axes: int = 3,
    start: int = 0,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build position_ids for a pure-text sequence (all axes identical).

    Returns:
        [batch_size, num_axes, seq_len]  — positions 0..seq_len-1 on every axis.
    """
    ids = torch.arange(start, start + seq_len, device=device)  # [S]
    ids = ids.unsqueeze(0).expand(num_axes, -1)                  # [M, S]
    return ids.unsqueeze(0).expand(batch_size, -1, -1)           # [B, M, S]


def make_image_position_ids(
    height: int,
    width: int,
    text_len: int = 0,
    num_axes: int = 3,
    temporal_id: int = 0,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build position_ids for a single image patch grid (h × w).

    Layout: text tokens (positions 0..text_len-1) followed by image patches.
    Image patches are laid out in row-major order with:
      axis-0 (temporal) = temporal_id (constant)
      axis-1 (height)   = row index   [0, height)
      axis-2 (width)    = col index   [0, width)

    For a 2-axis model pass ``num_axes=2`` (height + width only).

    Returns:
        [batch_size, num_axes, text_len + height*width]
    """
    # Text prefix
    text_ids = _range_ids(text_len, delta=0, device=device)   #  [S_text] each

    # Image patch grid
    h_ids = torch.arange(height, device=device)
    w_ids = torch.arange(width,  device=device)
    grid_h, grid_w = torch.meshgrid(h_ids, w_ids, indexing="ij")
    grid_h = grid_h.reshape(-1) + text_len  # row-major, offset by text_len
    grid_w = grid_w.reshape(-1) + text_len  # same offset for spatial positions

    # Actually for spatial we want: h in [0..H-1], w in [0..W-1], not text_len offset.
    # The token sequence position is text_len + patch_index.
    # The *spatial* position is the grid coordinate.
    n_patch = height * width
    t_ids   = torch.full((n_patch,), temporal_id, dtype=torch.long, device=device)
    h_ids   = torch.arange(height, device=device).repeat_interleave(width)   # [H*W]
    w_ids   = torch.arange(width,  device=device).repeat(height)             # [H*W]

    # Build axis arrays (text part first)
    axis0_text = torch.full((text_len,), 0, dtype=torch.long, device=device)
    axis0_img  = t_ids
    axis1_text = torch.arange(text_len, dtype=torch.long, device=device)
    axis1_img  = h_ids
    axis2_text = torch.arange(text_len, dtype=torch.long, device=device)
    axis2_img  = w_ids

    all_axes = [
        torch.cat([axis0_text, axis0_img]),   # temporal
        torch.cat([axis1_text, axis1_img]),   # height
        torch.cat([axis2_text, axis2_img]),   # width
    ]
    # Trim to num_axes if fewer axes requested
    all_axes = all_axes[:num_axes]
    # Pad missing axes by repeating the last
    while len(all_axes) < num_axes:
        all_axes.append(all_axes[-1])

    pos = torch.stack(all_axes, dim=0).unsqueeze(0)          # [1, M, S]
    return pos.expand(batch_size, -1, -1)                     # [B, M, S]


def make_video_position_ids(
    num_frames: int,
    height: int,
    width: int,
    text_len: int = 0,
    num_axes: int = 3,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build position_ids for a video (T × H × W patches).

    Returns:
        [batch_size, num_axes, text_len + T*H*W]
    """
    n_patch = num_frames * height * width

    t_ids = torch.arange(num_frames, device=device).\
        repeat_interleave(height * width)                               # [T*H*W]
    h_ids = torch.arange(height, device=device).\
        repeat_interleave(width).repeat(num_frames)                     # [T*H*W]
    w_ids = torch.arange(width,  device=device).\
        repeat(num_frames * height)                                      # [T*H*W]

    axis0_text = torch.zeros(text_len, dtype=torch.long, device=device)
    axis1_text = torch.arange(text_len, dtype=torch.long, device=device)
    axis2_text = torch.arange(text_len, dtype=torch.long, device=device)

    all_axes = [
        torch.cat([axis0_text, t_ids]),
        torch.cat([axis1_text, h_ids]),
        torch.cat([axis2_text, w_ids]),
    ]
    all_axes = all_axes[:num_axes]
    while len(all_axes) < num_axes:
        all_axes.append(all_axes[-1])

    pos = torch.stack(all_axes, dim=0).unsqueeze(0)   # [1, M, S]
    return pos.expand(batch_size, -1, -1)              # [B, M, S]


def _range_ids(n: int, delta: int, device) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long, device=device) + delta


# =========================================================================== #
#  MultimodalRotaryEmbedding                                                   #
# =========================================================================== #

class MultimodalRotaryEmbedding(nn.Module):
    """M-axis Rotary Position Embedding for multimodal sequences.

    The head dimension is split evenly into ``num_axes`` channels.
    Each channel receives an independent 1-D RoPE using its corresponding
    axis position IDs.

    Parameters
    ----------
    head_dim : int
        Full head dimension.  Must satisfy ``head_dim % (2 * num_axes) == 0``.
    num_axes : int
        Number of independent position axes (default 3: time, height, width).
    base : float
        RoPE base frequency (default 10000).
    max_seq_len : int
        Pre-built table size; extended automatically.
    """

    def __init__(
        self,
        head_dim: int,
        num_axes: int = 3,
        base: float = 10_000.0,
        max_seq_len: int = 4096,
        device=None,
    ) -> None:
        super().__init__()
        assert head_dim % (2 * num_axes) == 0, (
            f"head_dim ({head_dim}) must be divisible by 2*num_axes ({2*num_axes})."
        )
        self.head_dim    = head_dim
        self.num_axes    = num_axes
        self.base        = float(base)
        self.channel_dim = head_dim // num_axes   # dimension per axis channel (must be even)
        self.max_seq_len = max_seq_len

        # Per-channel inv_freq: θᵢ = base^{-2i/c},  i=0..c/2-1
        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, self.channel_dim, 2, dtype=torch.float32, device=device)
                / self.channel_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len, device=device)

    # ------------------------------------------------------------------ #
    def _build_cache(self, max_pos: int, device=None) -> None:
        """Build cos/sin table up to max_pos."""
        inv = self.inv_freq
        t   = torch.arange(max_pos, dtype=torch.float32,
                            device=device if device else inv.device)
        freqs = torch.outer(t, inv)                  # [max_pos, c/2]
        emb   = torch.cat([freqs, freqs], dim=-1)    # [max_pos, c]  (c = channel_dim)
        self.register_buffer("cos_table", emb.cos(), persistent=False)   # [max_pos, c]
        self.register_buffer("sin_table", emb.sin(), persistent=False)
        self._cache_len = max_pos

    # ------------------------------------------------------------------ #
    def _gather(self, table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Lookup table rows by ids.

        Args:
            table : [max_pos, c]
            ids   : [B, S]  (integer positions)

        Returns:
            [B, S, c]
        """
        # ids may contain values up to max(ids); extend if needed
        max_id = int(ids.max().item()) + 1
        if max_id > self._cache_len:
            self._build_cache(max_id * 2, device=table.device)
            # re-fetch updated buffers
            table = self.cos_table if table is self.cos_table else self.sin_table
        return table[ids]   # [B, S, c]

    # ------------------------------------------------------------------ #
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply M-axis mRoPE.

        Args:
            q            : [B, num_heads,    S, head_dim]
            k            : [B, num_kv_heads, S, head_dim]
            position_ids : [B, num_axes,     S]  integer positions per axis.

        Returns:
            (q_rot, k_rot) — same shapes as q, k.
        """
        B, H, S, D = q.shape
        c = self.channel_dim  # = D // num_axes

        # Extend cache if seq_len > pre-built length
        if S > self._cache_len:
            self._build_cache(S * 2, device=q.device)

        q_rot_parts = []
        k_rot_parts = []

        for axis_idx in range(self.num_axes):
            ax_ids = position_ids[:, axis_idx, :]   # [B, S]

            # Ensure cache is large enough for this axis
            max_id = int(ax_ids.max().item()) + 1
            if max_id > self._cache_len:
                self._build_cache(max_id * 2, device=q.device)

            cos = self.cos_table[ax_ids]   # [B, S, c]
            sin = self.sin_table[ax_ids]   # [B, S, c]

            # Reshape for broadcast: [B, 1, S, c]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

            # Slice the corresponding channel out of q and k
            q_ch = q[..., axis_idx * c : (axis_idx + 1) * c].float()   # [B, H, S, c]
            k_ch = k[..., axis_idx * c : (axis_idx + 1) * c].float()

            q_ch_rot = q_ch * cos + rotate_half(q_ch) * sin
            k_ch_rot = k_ch * cos + rotate_half(k_ch) * sin

            q_rot_parts.append(q_ch_rot)
            k_rot_parts.append(k_ch_rot)

        q_out = torch.cat(q_rot_parts, dim=-1).type_as(q)   # [B, H, S, D]
        k_out = torch.cat(k_rot_parts, dim=-1).type_as(k)
        return q_out, k_out

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_axes={self.num_axes}, "
            f"channel_dim={self.channel_dim}, base={self.base}"
        )


# =========================================================================== #
#  mRoPEAttention                                                              #
# =========================================================================== #

class mRoPEAttention(nn.Module):
    """Multi-Head Attention with Multimodal Rotary Position Embedding (mRoPE).

    Supports both standard text sequences and multimodal (image / video)
    sequences via multi-axis position_ids.

    Parameters
    ----------
    hidden_size : int
    num_heads : int
    num_kv_heads : int | None
        GQA support.  None = MHA.
    head_dim : int | None
        Defaults to hidden_size // num_heads.
    num_axes : int
        Number of position axes (3 for Qwen2-VL style: time, height, width).
    base : float
        RoPE base frequency.
    max_seq_len : int
    dropout : float
    causal : bool
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        num_axes: int = 3,
        base: float = 10_000.0,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim     = head_dim or (hidden_size // num_heads)
        self.num_axes     = num_axes
        self.causal       = causal
        self.scale        = 1.0 / math.sqrt(self.head_dim)

        kv_hidden = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_hidden,   bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_hidden,   bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary = MultimodalRotaryEmbedding(
            head_dim=self.head_dim,
            num_axes=num_axes,
            base=base,
            max_seq_len=max_seq_len,
        )
        self.attn_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        causal: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x              : [B, S, hidden_size]
            position_ids   : [B, num_axes, S]  or None.
                             None → fall back to 1-D text positions (all axes same).
            attention_mask : additive mask [B, 1, S, S] or None.
            causal         : overrides self.causal if given.

        Returns:
            [B, S, hidden_size]
        """
        B, S, _ = x.shape
        use_causal = self.causal if causal is None else causal

        q = self.q_proj(x).view(B, S, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Build text-style position_ids if not provided
        if position_ids is None:
            position_ids = make_text_position_ids(
                seq_len=S, num_axes=self.num_axes,
                batch_size=B, device=x.device,
            )

        q, k = self.rotary(q, k, position_ids=position_ids)

        # GQA repeat
        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if use_causal:
            cm = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            scores = scores + cm.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            scores = scores + attention_mask

        w = safe_softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        w = self.attn_dropout(w)

        out = torch.matmul(w, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, num_axes={self.num_axes}"
        )
