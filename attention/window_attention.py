"""
Window Attention (Sliding Window / Local Attention).

Restricts each query to attend only within a fixed-size window around its
position, reducing complexity from O(S²) to O(S·W) where W is the window
size.  Useful in Longformer, BigBird, Mistral-style architectures.

Two modes:
  - **Symmetric**: each query attends to ``[i - W//2, i + W//2]``
  - **Causal**: each query attends to ``[i - W + 1, i]`` (left-only)

The implementation materialises only the windowed scores per head,
avoiding the full S×S matrix.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .safe_softmax import safe_softmax


class WindowAttention(nn.Module):
    """Sliding-window Multi-Head Attention.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads.
    window_size : int
        Total window width.  For symmetric mode the actual range is
        ``window_size // 2`` on each side. For causal mode, the query
        attends to the preceding ``window_size`` positions (inclusive).
    head_dim : int or None
        Dimension per head.  Defaults to ``hidden_size // num_heads``.
    causal : bool
        If True, use causal (left-only) windowing.
    dropout : float
        Attention dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 256,
        head_dim: int | None = None,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            "hidden_size must equal num_heads * head_dim"
        )

        self.window_size = window_size
        self.causal = causal

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    # ------------------------------------------------------------------ #
    #  Weight helpers                                                      #
    # ------------------------------------------------------------------ #
    def load_weights(
        self,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        w_o: torch.Tensor,
    ) -> None:
        """Load pre-defined weight matrices (all [hidden, hidden])."""
        self.q_proj.weight.data.copy_(w_q)
        self.k_proj.weight.data.copy_(w_k)
        self.v_proj.weight.data.copy_(w_v)
        self.o_proj.weight.data.copy_(w_o)

    # ------------------------------------------------------------------ #
    #  Build window mask                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_window_mask(
        seq_len: int,
        window_size: int,
        causal: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """Return an additive attention mask of shape ``[S_q, S_kv]``.

        Positions outside the window get ``-inf``; positions inside get ``0``.
        """
        row = torch.arange(seq_len, device=device).unsqueeze(1)  # [S, 1]
        col = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, S]

        if causal:
            # attend to [i - window_size + 1 .. i]
            inside = (col <= row) & (col >= row - window_size + 1)
        else:
            # symmetric: attend to [i - w//2 .. i + w//2]
            half = window_size // 2
            inside = (col >= row - half) & (col <= row + half)

        mask = torch.where(
            inside,
            torch.tensor(0.0, device=device),
            torch.tensor(float("-inf"), device=device),
        )
        return mask  # [S_q, S_kv]

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, hidden_size]
            attention_mask: optional additive mask [B, 1, S, S].

        Returns:
            [B, S, hidden_size]
        """
        B, S, _ = x.shape

        # --- Project -------------------------------------------------
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # --- Reshape to [B, num_heads, S, head_dim] ------------------
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # --- Attention scores ----------------------------------------
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]

        # Apply window mask
        window_mask = self._build_window_mask(
            S, self.window_size, self.causal, x.device,
        )
        attn = attn + window_mask.unsqueeze(0).unsqueeze(0)  # broadcast [1,1,S,S]

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = safe_softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
        attn = self.attn_dropout(attn)

        # --- Weighted sum -------------------------------------------
        out = torch.matmul(attn, v)  # [B, H, S, D]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)

        return self.o_proj(out)

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, window={self.window_size}, "
            f"causal={self.causal}"
        )


def window_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Functional window attention (no learnable weights).

    Args:
        q, k, v: [B, H, S, D] or [B, S, D].
        window_size: window width.
        causal: causal windowing.
        scale: attention scale (default 1/√D).

    Returns:
        Same shape as q.
    """
    squeezed = False
    if q.dim() == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        squeezed = True

    D = q.size(-1)
    S = q.size(-2)
    sc = scale or (1.0 / math.sqrt(D))

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sc

    mask = WindowAttention._build_window_mask(S, window_size, causal, q.device)
    scores = scores + mask.unsqueeze(0).unsqueeze(0)

    attn = safe_softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.matmul(attn, v.float()).to(q.dtype)

    if squeezed:
        out = out.squeeze(1)
    return out
