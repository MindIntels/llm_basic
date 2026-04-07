"""
Flash Attention — CPU reference implementation.

Implements the **tiled online-softmax** algorithm from:
  *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*
  (Dao et al., 2022)

Key idea
--------
Instead of materialising the full (S_q × S_k) attention score matrix, we
process Q in row-blocks (size ``B_r``) and K/V in column-blocks (size
``B_c``).  A running log-sum-exp is maintained so that softmax is computed
**incrementally** — only one (B_r × B_c) tile is ever in memory at a time.

Complexity
----------
    Standard attention : O(S²·D)  memory for scores
    Flash attention    : O(B_r·B_c)  memory for the tile  ← independent of S

This file is a **pure-PyTorch CPU implementation** intended for correctness
verification and educational clarity; it is NOT optimised for speed.

Public API
----------
- ``flash_attention_forward(q, k, v, ...)`` — functional interface.
- ``FlashAttentionCPU``                     — nn.Module with Q/K/V projections.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
#  Functional Flash Attention kernel (CPU, pure PyTorch)
# ======================================================================

def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = False,
    block_size: int = 64,
) -> torch.Tensor:
    """Tiled flash attention — forward pass (CPU).

    Args:
        q: Query  — [B, S_q, D]  or  [B, H, S_q, D]
        k: Key    — [B, S_k, D]  or  [B, H, S_k, D]
        v: Value  — [B, S_k, D]  or  [B, H, S_k, D]
        scale: scaling factor; defaults to 1/sqrt(D).
        causal: if True apply a causal (lower-triangular) mask so that
                position i can only attend to positions ≤ i.
        block_size: tile size for both B_r (query block) and B_c (kv block).

    Returns:
        Output tensor with the same shape as *q*.

    Notes:
        * Supports both 3-D ``[B, S, D]`` and 4-D ``[B, H, S, D]`` layouts.
          For 4-D inputs the head dimension is treated as an extra batch dim.
        * ``S_q`` and ``S_k`` need **not** be multiples of ``block_size``;
          the last tile is simply smaller.
    """
    # --- Handle 3-D vs 4-D input -------------------------------------
    squeezed = False
    if q.dim() == 3:
        # Insert a dummy head dimension: [B, S, D] → [B, 1, S, D]
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        squeezed = True

    B, H, S_q, D = q.shape
    S_k = k.size(2)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    B_r = block_size  # query  block size (rows  of S matrix)
    B_c = block_size  # kv     block size (cols  of S matrix)

    # Number of blocks
    n_br = math.ceil(S_q / B_r)
    n_bc = math.ceil(S_k / B_c)

    # Collect per-query-block outputs in a list (autograd-friendly).
    block_outputs: list[torch.Tensor] = []

    for i in range(n_br):
        # ---- Q block: rows [i*B_r : (i+1)*B_r] ----------------------
        q_start = i * B_r
        q_end = min(q_start + B_r, S_q)
        br = q_end - q_start
        q_block = q[:, :, q_start:q_end, :]                       # [B, H, br, D]

        # Running accumulators for this Q-block (detached scalars)
        O_i = torch.zeros(B, H, br, D, dtype=q.dtype, device=q.device)
        m_i = torch.full((B, H, br, 1), float("-inf"), dtype=q.dtype,
                         device=q.device)
        l_i = torch.zeros((B, H, br, 1), dtype=q.dtype, device=q.device)

        for j in range(n_bc):
            # ---- K/V block: cols [j*B_c : (j+1)*B_c] ----------------
            k_start = j * B_c
            k_end = min(k_start + B_c, S_k)
            k_block = k[:, :, k_start:k_end, :]                   # [B, H, bc, D]
            v_block = v[:, :, k_start:k_end, :]                   # [B, H, bc, D]

            # S_ij = Q_i · K_j^T  (tile of attention scores)
            S_ij = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
            # S_ij: [B, H, br, bc]

            # ---- Causal masking (positions in q can only attend ≤ k) -
            if causal:
                row_idx = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                col_idx = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                causal_mask = row_idx < col_idx  # True where we should mask
                S_ij = S_ij.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # ---- Online softmax update --------------------------------
            # Local statistics for this tile
            m_ij = S_ij.max(dim=-1, keepdim=True).values            # [B, H, br, 1]
            # Guard against all -inf rows (fully masked)
            m_ij = torch.where(m_ij == float("-inf"),
                               torch.zeros_like(m_ij), m_ij)

            P_ij = torch.exp(S_ij - m_ij)                          # [B, H, br, bc]
            l_ij = P_ij.sum(dim=-1, keepdim=True)                  # [B, H, br, 1]

            # New running max
            m_new = torch.maximum(m_i, m_ij)                       # [B, H, br, 1]

            # Correction factors
            alpha = torch.exp(m_i - m_new)    # scale old accumulator
            beta  = torch.exp(m_ij - m_new)   # scale new tile

            # Update running sum-of-exp
            l_new = alpha * l_i + beta * l_ij                      # [B, H, br, 1]

            # Avoid division by zero (fully masked rows)
            l_safe = torch.where(l_new == 0, torch.ones_like(l_new), l_new)

            # Update output accumulator
            O_i = (alpha * l_i * O_i + beta * torch.matmul(P_ij, v_block)) / l_safe

            # Commit new stats
            m_i = m_new
            l_i = l_new

        block_outputs.append(O_i)

    # Concatenate along the sequence dimension
    O = torch.cat(block_outputs, dim=2)                            # [B, H, S_q, D]

    if squeezed:
        O = O.squeeze(1)

    return O


# ======================================================================
#  nn.Module wrapper (with learnable Q/K/V/O projections)
# ======================================================================

class FlashAttentionCPU(nn.Module):
    """Multi-Head Attention using the tiled flash-attention kernel on CPU.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension (= num_heads * head_dim).
    num_heads : int
        Number of attention heads.
    head_dim : int or None
        Dimension per head; defaults to ``hidden_size // num_heads``.
    dropout : float
        (Ignored in flash path; kept for API compat.)
    block_size : int
        Tile size for the flash attention algorithm.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        block_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size
        self.block_size = block_size

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

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

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        block_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, hidden_size]
            causal: apply causal mask.
            block_size: override default tile size.

        Returns:
            [B, S, hidden_size]
        """
        B, S, _ = x.shape
        bs = block_size or self.block_size

        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [B, num_heads, S, head_dim]
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Flash attention (operates on [B, H, S, D])
        attn_out = flash_attention_forward(
            q, k, v,
            scale=self.scale,
            causal=causal,
            block_size=bs,
        )

        # Reshape back → [B, S, hidden_size]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)

        return self.o_proj(attn_out)

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, block_size={self.block_size}"
        )
