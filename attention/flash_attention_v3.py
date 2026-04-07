"""
Flash Attention **v3** — CPU reference implementation.

Key improvements over v2 (Dao et al., 2024 — *FlashAttention-3*):

1. **Mixed-precision accumulation** — Q/K are kept in lower precision
   (float16 / bfloat16) for the matmul, while the output accumulator O
   and softmax statistics are maintained in float32.  This simulates the
   FP8-with-block-scaling approach in real hardware.

2. **Block-sparse attention** — An optional boolean block mask
   ``block_mask[n_br, n_bc]`` allows skipping entire tiles that are known
   to be zero (e.g. from a fixed sparse pattern or padding).  Combined
   with the causal early-exit from v2, this can skip >50 % of tiles.

3. **Two-pass softmax** — Instead of a single online pass, v3 offers an
   optional **two-pass** mode:
     Pass 1 — compute per-row logsumexp over *all* KV blocks.
     Pass 2 — use the exact logsumexp (no running correction) to compute
              ``softmax(S) · V`` in one shot per tile.
   This eliminates all rescaling multiplications, further reducing
   non-matmul FLOPs and improving numerical precision.

4. **Ping-pong pipelining** (simulated) — On real hardware v3 overlaps
   the softmax of tile j with the GEMM of tile j+1 using warp-
   specialisation.  We expose this as a ``pipeline=True`` flag that
   simply pre-fetches the *next* KV block before starting the current
   softmax, demonstrating the data-flow intent.

Complexity
----------
    Same O(S²·D / (B_r · B_c)) tile iterations as v2, but fewer actual
    tiles touched when ``block_mask`` or causal skipping is enabled, and
    fewer non-matmul FLOPs in two-pass mode.

Public API
----------
- ``flash_attention_v3_forward(q, k, v, ...)``
- ``FlashAttentionV3`` nn.Module
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ======================================================================
#  Helpers
# ======================================================================

def _to_compute_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast to a lower-precision compute dtype (simulation)."""
    if dtype is None or dtype == t.dtype:
        return t
    return t.to(dtype)


# ======================================================================
#  Two-pass logsumexp pre-computation
# ======================================================================

def _compute_logsumexp(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    causal: bool,
    block_size: int,
    block_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Pass 1: compute exact per-row logsumexp over all KV blocks.

    Returns:
        lse: [B, H, S_q, 1]
    """
    B, H, S_q, D = q.shape
    S_k = k.size(2)
    B_r, B_c = block_size, block_size
    n_br = math.ceil(S_q / B_r)
    n_bc = math.ceil(S_k / B_c)

    lse_blocks: list[torch.Tensor] = []

    for i in range(n_br):
        q_s = i * B_r
        q_e = min(q_s + B_r, S_q)
        q_block = q[:, :, q_s:q_e, :]

        block_lse = torch.full((B, H, q_e - q_s, 1), float("-inf"),
                               dtype=torch.float32, device=q.device)

        for j in range(n_bc):
            k_s = j * B_c
            k_e = min(k_s + B_c, S_k)

            # Block mask check
            if block_mask is not None and not block_mask[i, j]:
                continue
            # Causal early-exit (v2 style)
            if causal and k_s > (q_e - 1):
                break

            k_block = k[:, :, k_s:k_e, :]
            S_ij = torch.matmul(q_block.float(), k_block.float().transpose(-2, -1)) * scale

            if causal:
                row_idx = torch.arange(q_s, q_e, device=q.device).unsqueeze(1)
                col_idx = torch.arange(k_s, k_e, device=q.device).unsqueeze(0)
                S_ij = S_ij.masked_fill(
                    (row_idx < col_idx).unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # logsumexp of this tile
            tile_lse = torch.logsumexp(S_ij, dim=-1, keepdim=True)  # [B,H,br,1]

            # Merge with running lse (no in-place ops for autograd safety)
            block_lse = torch.logaddexp(block_lse, tile_lse)

        lse_blocks.append(block_lse)

    return torch.cat(lse_blocks, dim=2)


# ======================================================================
#  Functional kernel
# ======================================================================

def flash_attention_v3_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = False,
    block_size: int = 64,
    block_mask: torch.Tensor | None = None,
    two_pass: bool = False,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Flash Attention v3 — block-sparse, mixed-precision, two-pass.

    Args:
        q, k, v:  [B, S, D] or [B, H, S, D].
        scale:    1/sqrt(D).
        causal:   causal mask.
        block_size: tile size.
        block_mask: optional ``[n_br, n_bc]`` bool tensor. ``True`` = compute
                    this tile; ``False`` = skip (treat as zero attention).
        two_pass: use two-pass exact-logsumexp mode (pass 1 for LSE, pass 2
                  for P·V).
        compute_dtype: lower-precision dtype for Q·K^T matmul (e.g.
                       ``torch.float16``).  Accumulation stays in float32.

    Returns:
        Tensor with same shape as *q*.
    """
    squeezed = False
    if q.dim() == 3:
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        squeezed = True

    B, H, S_q, D = q.shape
    S_k = k.size(2)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    B_r, B_c = block_size, block_size
    n_br = math.ceil(S_q / B_r)
    n_bc = math.ceil(S_k / B_c)

    # ---- optional two-pass: pre-compute exact logsumexp ---------------
    exact_lse: torch.Tensor | None = None
    if two_pass:
        exact_lse = _compute_logsumexp(q, k, scale, causal, block_size, block_mask)

    block_outputs: list[torch.Tensor] = []

    for i in range(n_br):
        q_s = i * B_r
        q_e = min(q_s + B_r, S_q)
        br = q_e - q_s
        q_block = q[:, :, q_s:q_e, :]

        # v3: FP32 accumulator regardless of compute dtype
        O_i = torch.zeros(B, H, br, D, dtype=torch.float32, device=q.device)
        lse_i = torch.full((B, H, br, 1), float("-inf"),
                           dtype=torch.float32, device=q.device)

        # Pre-fetch handle (simulated pipeline – just reference next KV)
        # In real v3 on GPU, we'd issue async copies here.
        prefetch_k: torch.Tensor | None = None
        prefetch_v: torch.Tensor | None = None

        for j in range(n_bc):
            k_s = j * B_c
            k_e = min(k_s + B_c, S_k)

            # ---- v3: block-sparse skipping ----------------------------
            if block_mask is not None and not block_mask[i, j]:
                continue
            # Causal early-exit (v2 carry-over)
            if causal and k_s > (q_e - 1):
                break

            # ---- v3: pipelining – use prefetched or load fresh --------
            if prefetch_k is not None:
                k_block, v_block = prefetch_k, prefetch_v
            else:
                k_block = k[:, :, k_s:k_e, :]
                v_block = v[:, :, k_s:k_e, :]

            # Prefetch *next* KV block (simulated)
            nj = j + 1
            while nj < n_bc:
                nk_s = nj * B_c
                nk_e = min(nk_s + B_c, S_k)
                if block_mask is not None and not block_mask[i, nj]:
                    nj += 1
                    continue
                if causal and nk_s > (q_e - 1):
                    prefetch_k = prefetch_v = None
                    break
                prefetch_k = k[:, :, nk_s:nk_e, :]
                prefetch_v = v[:, :, nk_s:nk_e, :]
                break
            else:
                prefetch_k = prefetch_v = None

            # ---- v3: mixed-precision matmul ---------------------------
            q_lp = _to_compute_dtype(q_block, compute_dtype)
            k_lp = _to_compute_dtype(k_block, compute_dtype)
            S_ij = torch.matmul(q_lp.float(), k_lp.float().transpose(-2, -1)) * scale

            # Causal mask
            if causal:
                row_idx = torch.arange(q_s, q_e, device=q.device).unsqueeze(1)
                col_idx = torch.arange(k_s, k_e, device=q.device).unsqueeze(0)
                S_ij = S_ij.masked_fill(
                    (row_idx < col_idx).unsqueeze(0).unsqueeze(0), float("-inf")
                )

            if two_pass and exact_lse is not None:
                # ---- v3 two-pass: use exact logsumexp, no rescaling --
                # P_ij = exp(S_ij - lse)  already normalised
                row_lse = exact_lse[:, :, q_s:q_e, :]             # [B,H,br,1]
                P_ij = torch.exp(S_ij - row_lse)                  # [B,H,br,bc]
                O_i = O_i + torch.matmul(P_ij, v_block.float())
            else:
                # ---- v2-style online softmax with logsumexp ----------
                m_ij = S_ij.max(dim=-1, keepdim=True).values
                m_ij = torch.where(m_ij == float("-inf"),
                                   torch.zeros_like(m_ij), m_ij)
                P_ij = torch.exp(S_ij - m_ij)
                lse_ij = m_ij + torch.log(
                    P_ij.sum(dim=-1, keepdim=True).clamp(min=1e-30)
                )

                lse_new = torch.logaddexp(lse_i, lse_ij)
                alpha = torch.exp(lse_i - lse_new)
                tile_scale = torch.exp(m_ij - lse_new)

                O_i = alpha * O_i + tile_scale * torch.matmul(P_ij, v_block.float())
                lse_i = lse_new

        # Cast back to input dtype
        block_outputs.append(O_i.to(q.dtype))

    O = torch.cat(block_outputs, dim=2)
    if squeezed:
        O = O.squeeze(1)
    return O


# ======================================================================
#  nn.Module wrapper
# ======================================================================

class FlashAttentionV3(nn.Module):
    """MHA with Flash Attention v3 kernel (block-sparse + mixed precision)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        block_size: int = 64,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size
        self.block_size = block_size
        self.compute_dtype = compute_dtype

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

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        block_size: int | None = None,
        block_mask: torch.Tensor | None = None,
        two_pass: bool = False,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        bs = block_size or self.block_size
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = flash_attention_v3_forward(
            q, k, v,
            scale=self.scale,
            causal=causal,
            block_size=bs,
            block_mask=block_mask,
            two_pass=two_pass,
            compute_dtype=self.compute_dtype,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)

    def extra_repr(self):
        return (f"hidden={self.hidden_size}, heads={self.num_heads}, "
                f"head_dim={self.head_dim}, block_size={self.block_size}, "
                f"compute_dtype={self.compute_dtype}")
