"""
Standard (naive) Multi-Head Attention — reference implementation.

This module materialises the full (seq_q × seq_k) attention score matrix
and is used **only** as a correctness baseline for the tiled / flash
variants.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .safe_softmax import safe_softmax


class StandardMHA(nn.Module):
    """Vanilla Multi-Head Attention (no TP / SP).

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension per head.  Defaults to ``hidden_size // num_heads``.
    dropout : float
        Attention dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, (
            "hidden_size must equal num_heads * head_dim"
        )

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
    #  Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:  [B, S, H]
            attention_mask: additive mask [B, 1, S, S] or None.
            causal: if True, apply an upper-triangular -inf mask.

        Returns:
            [B, S, H]
        """
        B, S, _ = x.shape

        # --- Project -------------------------------------------------
        q = self.q_proj(x)  # [B, S, H]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # --- Reshape to [B, num_heads, S, head_dim] ------------------
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # --- Attention scores ----------------------------------------
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]

        if causal:
            mask = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            attn = attn + mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = safe_softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
        attn = self.attn_dropout(attn)

        # --- Weighted sum --------------------------------------------
        out = torch.matmul(attn, v)  # [B, H, S, D]
        out = out.transpose(1, 2).contiguous().view(B, S, -1)

        # --- Output projection ---------------------------------------
        return self.o_proj(out)
