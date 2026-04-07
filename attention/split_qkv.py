"""
Split Q/K/V Multi-Head Attention.

Explicitly demonstrates how Q, K, V projections are **split** across
attention heads, with each head computing independent scaled dot-product
attention before being concatenated and passed through the output
projection.

Design goals
------------
1. Make head-level split **visible** (no fused batched-matmul across heads).
2. Allow per-head inspection / replacement (e.g. plug in Flash Attention for
   the per-head kernel while keeping the same outer structure).
3. Keep the interface compatible with ``StandardMHA`` for easy comparison.

Usage
-----
>>> attn = SplitQKVAttention(hidden_size=64, num_heads=4)
>>> y = attn(x)          # uses naive SDPA per head
>>> y = attn(x, use_flash=True, block_size=32)  # uses CPU Flash Attention
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .safe_softmax import safe_softmax


def _naive_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = False,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Naive scaled dot-product attention for **one head**.

    Args:
        q: [B, S_q, D]
        k: [B, S_k, D]
        v: [B, S_k, D]
        scale: 1/sqrt(D)
        causal: apply causal mask
        attn_mask: additive mask [B, S_q, S_k]

    Returns:
        [B, S_q, D]
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, Sq, Sk]

    if causal:
        S_q, S_k = scores.size(-2), scores.size(-1)
        mask = torch.triu(
            torch.full((S_q, S_k), float("-inf"), device=q.device, dtype=q.dtype),
            diagonal=1,
        )
        scores = scores + mask

    if attn_mask is not None:
        scores = scores + attn_mask

    weights = safe_softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
    return torch.matmul(weights, v)


class SplitQKVAttention(nn.Module):
    """Multi-Head Attention with **explicit per-head Q/K/V split**.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads.
    head_dim : int or None
        Dimension per head; defaults to ``hidden_size // num_heads``.
    dropout : float
        Attention dropout probability (applied after softmax in naive path).
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

        # --- Separate per-head Q/K/V projections ----------------------
        # We use a single nn.Linear for each of Q/K/V and then *split*
        # the output, making the split explicit and inspectable.
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
        self.q_proj.weight.data.copy_(w_q)
        self.k_proj.weight.data.copy_(w_k)
        self.v_proj.weight.data.copy_(w_v)
        self.o_proj.weight.data.copy_(w_o)

    # ------------------------------------------------------------------ #
    #  Core: split → per-head attention → concat                          #
    # ------------------------------------------------------------------ #
    def _split_heads(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Split [B, S, H] → list of num_heads tensors [B, S, head_dim]."""
        B, S, _ = x.shape
        # reshape to [B, S, num_heads, head_dim] then unbind on dim=2
        x = x.view(B, S, self.num_heads, self.head_dim)
        return list(x.unbind(dim=2))  # list of [B, S, D]

    def _concat_heads(self, heads: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate list of [B, S, head_dim] → [B, S, H]."""
        return torch.cat(heads, dim=-1)

    def _per_head_attention(
        self,
        q_h: torch.Tensor,
        k_h: torch.Tensor,
        v_h: torch.Tensor,
        causal: bool = False,
        attn_mask: torch.Tensor | None = None,
        use_flash: bool = False,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Compute attention for a single head.

        Args:
            q_h, k_h, v_h: [B, S, D]
            use_flash: if True, use CPU flash attention kernel.
            block_size: tile size for flash attention.

        Returns:
            [B, S, D]
        """
        if use_flash:
            from .flash_attention_cpu import flash_attention_forward
            return flash_attention_forward(
                q_h, k_h, v_h,
                scale=self.scale,
                causal=causal,
                block_size=block_size,
            )
        else:
            return _naive_sdpa(q_h, k_h, v_h, self.scale, causal, attn_mask)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
        use_flash: bool = False,
        block_size: int = 64,
    ) -> torch.Tensor:
        """
        Args:
            x:  [B, S, hidden_size]
            attention_mask: additive [B, S, S] or None (per-head broadcast).
            causal: apply causal (autoregressive) mask.
            use_flash: use CPU flash attention instead of naive SDPA.
            block_size: block/tile size for flash attention.

        Returns:
            [B, S, hidden_size]
        """
        # Step 1: Project
        q = self.q_proj(x)  # [B, S, H]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Step 2: Split into per-head tensors
        q_heads = self._split_heads(q)  # list of [B, S, D]
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)

        # Step 3: Per-head attention
        out_heads: list[torch.Tensor] = []
        for h_idx in range(self.num_heads):
            head_out = self._per_head_attention(
                q_heads[h_idx],
                k_heads[h_idx],
                v_heads[h_idx],
                causal=causal,
                attn_mask=attention_mask,
                use_flash=use_flash,
                block_size=block_size,
            )
            out_heads.append(head_out)

        # Step 4: Concatenate heads
        concat = self._concat_heads(out_heads)  # [B, S, H]

        # Step 5: Output projection
        return self.o_proj(concat)

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}"
        )
