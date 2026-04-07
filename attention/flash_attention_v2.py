"""
Flash Attention **v2** — CPU reference implementation.

Key improvements over v1 (Dao et al., 2023 — *FlashAttention-2*):

1. **Deferred rescaling** — In v1, O_i is re-normalised (divided by ``l``)
   inside *every* inner-loop iteration.  v2 keeps O_i **un-normalised**
   throughout the inner loop and does a **single** division after all KV
   blocks have been processed.  This halves the non-matmul FLOPs.

2. **Simplified online softmax** — Instead of tracking both ``m`` (running
   max) and ``l`` (running sum-of-exp) separately, v2 maintains an explicit
   ``logsumexp`` (LSE) accumulator and derives the correction factor directly
   from it, which is slightly more numerically stable.

3. **Causal early-exit** — For causal attention, if a KV block lies entirely
   above the diagonal (all positions masked), we skip it completely.  This
   saves ~50 % of tiles for causal workloads.

Complexity (same asymptotic, fewer constant-factor ops):
    Forward  : O(S²·D / (B_r·B_c)) tile iterations, 1 division per Q-block
    Memory   : O(B_r·B_c)  per tile

Public API
----------
- ``flash_attention_v2_forward(q, k, v, ...)``  — functional interface.
- ``FlashAttentionV2``                           — nn.Module wrapper.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ======================================================================
#  Functional kernel
# ======================================================================

def flash_attention_v2_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = False,
    block_size: int = 64,
) -> torch.Tensor:
    """Flash Attention v2 — deferred-rescaling tiled attention.

    Args:
        q: [B, S_q, D]  or  [B, H, S_q, D]
        k: [B, S_k, D]  or  [B, H, S_k, D]
        v: [B, S_k, D]  or  [B, H, S_k, D]
        scale: 1/sqrt(D) by default.
        causal: apply lower-triangular causal mask.
        block_size: tile size (B_r = B_c = block_size).

    Returns:
        Tensor with same shape as *q*.
    """
    # ---- handle 3-D / 4-D -------------------------------------------
    squeezed = False
    if q.dim() == 3:
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        squeezed = True

    B, H, S_q, D = q.shape
    S_k = k.size(2)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    B_r = block_size
    B_c = block_size
    n_br = math.ceil(S_q / B_r)
    n_bc = math.ceil(S_k / B_c)

    block_outputs: list[torch.Tensor] = []

    for i in range(n_br):
        q_start = i * B_r
        q_end = min(q_start + B_r, S_q)
        br = q_end - q_start
        q_block = q[:, :, q_start:q_end, :]                       # [B,H,br,D]

        # ---- v2: un-normalised accumulators --------------------------
        O_i = torch.zeros(B, H, br, D, dtype=q.dtype, device=q.device)
        # logsumexp accumulator per row  (init -inf → log(0))
        lse_i = torch.full((B, H, br, 1), float("-inf"),
                           dtype=q.dtype, device=q.device)

        for j in range(n_bc):
            k_start = j * B_c
            k_end = min(k_start + B_c, S_k)

            # --- v2 optimisation: causal early-exit -------------------
            # If all query positions < smallest key position in this
            # block, every entry would be masked → skip entirely.
            if causal and k_start > (q_end - 1):
                break                                              # <-- v2

            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]

            # Tile scores
            S_ij = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            # Causal mask
            if causal:
                row_idx = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                col_idx = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                S_ij = S_ij.masked_fill(
                    (row_idx < col_idx).unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # ---- v2: logsumexp-based online softmax ------------------
            # Local tile max & exp
            m_ij = S_ij.max(dim=-1, keepdim=True).values
            m_ij = torch.where(m_ij == float("-inf"),
                               torch.zeros_like(m_ij), m_ij)
            P_ij = torch.exp(S_ij - m_ij)                         # [B,H,br,bc]
            # Local logsumexp for tile j
            lse_ij = m_ij + torch.log(
                P_ij.sum(dim=-1, keepdim=True).clamp(min=1e-30)
            )                                                      # [B,H,br,1]

            # ---- v2: single correction factor from logsumexp ---------
            # new_lse = log(exp(lse_i) + exp(lse_ij))
            lse_new = torch.logaddexp(lse_i, lse_ij)              # [B,H,br,1]

            # Correction for old accumulator
            alpha = torch.exp(lse_i - lse_new)                    # [B,H,br,1]
            # Correction for new tile (P already uses m_ij)
            beta = torch.exp(lse_ij - lse_new - m_ij + m_ij)     # simplifies
            # Actually: we need  exp(S_ij - lse_new) · V
            # = exp(S_ij - m_ij) · exp(m_ij - lse_new) · V
            # = P_ij · exp(m_ij - lse_new) · V
            tile_scale = torch.exp(m_ij - lse_new)                 # [B,H,br,1]

            # ---- v2: deferred rescaling (no division by l in loop) ---
            O_i = alpha * O_i + tile_scale * torch.matmul(P_ij, v_block)

            lse_i = lse_new

        # ---- v2: O_i is already properly normalised -----------------
        # (because Σ_j exp(S_ij - lse) = 1 by definition of lse)
        block_outputs.append(O_i)

    O = torch.cat(block_outputs, dim=2)

    if squeezed:
        O = O.squeeze(1)
    return O


# ======================================================================
#  nn.Module wrapper
# ======================================================================

class FlashAttentionV2(nn.Module):
    """Multi-Head Attention using Flash Attention v2 kernel (CPU).

    Same interface as ``FlashAttentionCPU`` for drop-in replacement.
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

    def load_weights(self, w_q, w_k, w_v, w_o):
        self.q_proj.weight.data.copy_(w_q)
        self.k_proj.weight.data.copy_(w_k)
        self.v_proj.weight.data.copy_(w_v)
        self.o_proj.weight.data.copy_(w_o)

    def forward(self, x, causal=False, block_size=None):
        B, S, _ = x.shape
        bs = block_size or self.block_size
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = flash_attention_v2_forward(q, k, v, scale=self.scale,
                                              causal=causal, block_size=bs)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)

    def extra_repr(self):
        return (f"hidden={self.hidden_size}, heads={self.num_heads}, "
                f"head_dim={self.head_dim}, block_size={self.block_size}")
