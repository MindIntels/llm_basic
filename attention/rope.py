"""
RoPE — Rotary Position Embedding.

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position
Embedding" (2021).  https://arxiv.org/abs/2104.09864

Key idea
--------
Instead of adding position information to Q/K before the dot-product, RoPE
*rotates* the Q and K vectors in the complex plane using a position-dependent
rotation matrix.  The inner product then naturally encodes *relative* position:

    <RoPE(q, m), RoPE(k, n)> depends only on (q, k, m-n)

where m, n are the absolute positions of the query and key tokens.

Construction
------------
Split each head's d-dimensional vector into d/2 pairs (x₂ᵢ, x₂ᵢ₊₁).
For pair i at position pos, apply the 2×2 block rotation:

    [x₂ᵢ' ]   [cos(pos · θᵢ)   -sin(pos · θᵢ)] [x₂ᵢ  ]
    [x₂ᵢ₊₁'] = [sin(pos · θᵢ)    cos(pos · θᵢ)] [x₂ᵢ₊₁]

where θᵢ = base^(-2i/d),  base=10000 by default.

Efficient implementation: instead of building the rotation matrices, use the
complex-number trick:

    rotate(x, pos) = x  * cos(pos·Θ)  +  rotate_half(x) * sin(pos·Θ)

where rotate_half(x) = [-x₁, x₀, -x₃, x₂, ...] (swap pairs and negate first).

This module provides:
  * ``RotaryEmbedding``   — pre-computes and caches sinusoidal coefficients.
  * ``apply_rotary_emb``  — functional: apply RoPE to an already-shaped tensor.
  * ``RoPEAttention``     — drop-in MHA variant that applies RoPE to Q and K.

Variants included
-----------------
  * Standard RoPE (``base=10000``, original RoFormer).
  * LLaMA-style RoPE (same math, different weight init convention).
  * YaRN / long-context RoPE — scaling via ``rope_scaling`` dict
    (supported types: ``"linear"``, ``"dynamic"``).

Shape conventions
-----------------
  Q, K tensors entering ``apply_rotary_emb`` must be either:
    [B, num_heads, S, head_dim]   (4-D, canonical layout)
    [B, S, num_heads, head_dim]   (4-D, pre-transpose)
  cos/sin caches : [S, head_dim] or [1, 1, S, head_dim]
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from .safe_softmax import safe_softmax


# =========================================================================== #
#  Functional helpers                                                           #
# =========================================================================== #

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by d/2.

    Given [..., d], returns [..., d] where:
        result[..., :d/2]  = -x[..., d/2:]
        result[..., d/2:]  =  x[..., :d/2]

    Raises:
        ValueError: if the last dimension is odd (cannot split evenly).
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            f"rotate_half requires an even last dimension, got {x.shape[-1]}"
        )
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K using pre-computed cos/sin tables.

    Args:
        q   : [..., S, head_dim]  — queries (any leading dims OK).
        k   : [..., S, head_dim]  — keys.
        cos : [S, head_dim]  or  [1, 1, S, head_dim]  — cosine table.
        sin : same shape as cos   — sine table.

    Returns:
        (q_rot, k_rot) — rotated tensors with the same shapes as q, k.
    """
    # Broadcast cos/sin to match q/k shape
    # cos, sin shape: make sure last two dims are [S, head_dim]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# =========================================================================== #
#  RotaryEmbedding — sinusoidal cache                                          #
# =========================================================================== #

class RotaryEmbedding(nn.Module):
    """Pre-computes and caches RoPE cos/sin tables up to ``max_seq_len``.

    Parameters
    ----------
    head_dim : int
        Dimension of each attention head.  Must be even.
    base : float
        The base of the geometric frequency sequence (default 10000).
    max_seq_len : int
        Pre-computed sequence length.  Automatically extended if exceeded.
    rope_scaling : dict | None
        Optional scaling for long-context variants:
          ``{"type": "linear",  "factor": 4.0}`` — divide positions by factor.
          ``{"type": "dynamic", "factor": 4.0}`` — recompute θ dynamically.
        None = no scaling.
    device : torch.device | str | None
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0,
        max_seq_len: int = 4096,
        rope_scaling: dict | None = None,
        device=None,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.head_dim    = head_dim
        self.base        = float(base)
        self.max_seq_len = max_seq_len
        self.scaling     = rope_scaling

        # θ_i = base^{-2i/d},  i = 0..d/2-1
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                          / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute up to max_seq_len
        self._build_cache(max_seq_len, device=device)

    # ------------------------------------------------------------------ #
    def _get_inv_freq(self, seq_len: int) -> torch.Tensor:
        """Return (possibly re-scaled) inverse frequencies."""
        inv = self.inv_freq
        if self.scaling is None:
            return inv
        stype = self.scaling.get("type", "linear")
        factor = float(self.scaling.get("factor", 1.0))
        if stype == "linear":
            # Divide positions by factor (equivalent to stretching θ)
            return inv / factor
        elif stype == "dynamic":
            # Only rescale if seq_len exceeds the original max_seq_len
            if seq_len <= self.max_seq_len:
                return inv
            new_base = self.base * (
                (factor * seq_len / self.max_seq_len) - (factor - 1)
            ) ** (self.head_dim / (self.head_dim - 2))
            return 1.0 / (
                new_base ** (
                    torch.arange(0, self.head_dim, 2, dtype=torch.float32,
                                 device=inv.device) / self.head_dim
                )
            )
        else:
            raise ValueError(f"Unknown rope_scaling type: {stype!r}")

    # ------------------------------------------------------------------ #
    def _build_cache(self, seq_len: int, device=None) -> None:
        """Build cos/sin cache for positions [0, seq_len)."""
        inv = self._get_inv_freq(seq_len)
        t   = torch.arange(seq_len, dtype=torch.float32,
                            device=device if device else inv.device)
        # Outer product → [seq_len, head_dim/2]
        freqs = torch.outer(t, inv)
        # Repeat for [sin, cos] → [seq_len, head_dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        # Register as non-persistent buffers (not saved in state_dict by default)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0),
                             persistent=False)  # [1, 1, S, D]
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0),
                             persistent=False)
        self._cache_len = seq_len

    # ------------------------------------------------------------------ #
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE in-place style (returns new tensors).

        Args:
            q            : [B, num_heads, S, head_dim]
            k            : [B, num_kv_heads, S, head_dim]
            position_ids : [B, S] or None  (None → use 0..S-1).

        Returns:
            (q_rot, k_rot) — same shapes.
        """
        S = q.shape[-2]

        # Extend cache if needed
        if S > self._cache_len:
            self._build_cache(S * 2, device=q.device)

        if position_ids is None:
            cos = self.cos_cached[:, :, :S, :]   # [1, 1, S, D]
            sin = self.sin_cached[:, :, :S, :]
        else:
            # Gather rows by position id: [B, S] → [B, 1, S, D]
            cos = self.cos_cached[0, 0][position_ids].unsqueeze(1)
            sin = self.sin_cached[0, 0][position_ids].unsqueeze(1)

        q_rot, k_rot = apply_rotary_emb(q.float(), k.float(), cos, sin)
        return q_rot.type_as(q), k_rot.type_as(k)

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, base={self.base}, "
            f"max_seq_len={self.max_seq_len}, scaling={self.scaling}"
        )


# =========================================================================== #
#  RoPEAttention — MHA with built-in RoPE                                      #
# =========================================================================== #

class RoPEAttention(nn.Module):
    """Multi-Head Attention with Rotary Position Embedding.

    Fully self-contained: manages its own ``RotaryEmbedding`` cache and
    projects Q / K / V / output.

    Parameters
    ----------
    hidden_size : int
    num_heads : int
    num_kv_heads : int | None
        For Grouped-Query Attention (GQA).  None = MHA (num_kv_heads = num_heads).
    head_dim : int | None
        Defaults to ``hidden_size // num_heads``.
    base : float
        RoPE base frequency (default 10000).
    max_seq_len : int
        Pre-built cache length.
    rope_scaling : dict | None
        Long-context scaling config.
    dropout : float
    causal : bool
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        base: float = 10_000.0,
        max_seq_len: int = 4096,
        rope_scaling: dict | None = None,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim     = head_dim or (hidden_size // num_heads)
        self.causal       = causal
        self.scale        = 1.0 / math.sqrt(self.head_dim)

        kv_hidden = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_hidden,   bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_hidden,   bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary = RotaryEmbedding(
            head_dim=self.head_dim,
            base=base,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
        )
        self.attn_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x              : [B, S, hidden_size]
            attention_mask : additive mask [B, 1, S, S] or None.
            position_ids   : [B, S] integer positions or None (0..S-1).
            causal         : overrides self.causal if given.

        Returns:
            [B, S, hidden_size]
        """
        B, S, _ = x.shape
        use_causal = self.causal if causal is None else causal

        q = self.q_proj(x).view(B, S, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rotary(q, k, position_ids=position_ids)

        # GQA: repeat KV heads if needed
        if self.num_kv_heads != self.num_heads:
            r = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(r, dim=1)
            v = v.repeat_interleave(r, dim=1)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, H, S, S]

        if use_causal:
            causal_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            scores = scores + attention_mask

        w = safe_softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        w = self.attn_dropout(w)

        out = torch.matmul(w, v)                               # [B, H, S, D]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)  # [B, S, H]
        return self.o_proj(out)

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"causal={self.causal}"
        )
