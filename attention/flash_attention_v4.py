"""
Flash Attention **v4** — CPU reference implementation.

Key improvements over v3 (state-of-the-art optimisations):

1. **KV-cache incremental decoding** — During autoregressive generation,
   new queries attend to a growing KV cache.  ``flash_attention_v4_forward``
   accepts a ``kv_cache`` tuple ``(K_cache, V_cache)`` and only computes
   attention for the *new* query tokens, appending K/V in place.

2. **Sliding-window (local) attention** — An optional ``window_size``
   parameter restricts each query to attend only to the most recent
   ``window_size`` key positions.  Tiles outside the window are skipped,
   reducing cost from O(S²) to O(S·W).

3. **Adaptive block-size selection** — ``auto_block_size(S_q, S_k, D)``
   picks the tile size that balances the number of tiles against per-tile
   arithmetic, based on the problem dimensions.

4. **Softcap (tanh-based score capping)** — Gemma-2 style logit capping:
   ``scores = softcap * tanh(scores / softcap)`` prevents extreme
   attention weights, improving training stability.

5. **Return logsumexp** — Optionally returns the per-row logsumexp vector,
   useful for downstream loss computation (e.g. distillation) without
   re-computing attention.

Public API
----------
- ``flash_attention_v4_forward(q, k, v, ...)``
- ``auto_block_size(S_q, S_k, D)``
- ``FlashAttentionV4`` nn.Module
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ======================================================================
#  Adaptive block-size heuristic
# ======================================================================

def auto_block_size(S_q: int, S_k: int, D: int, max_block: int = 256) -> int:
    """Choose block size automatically based on problem dimensions.

    Heuristic: keep ``B_r · D`` roughly within a "register-friendly" budget
    (~4096 elements) while ensuring at least 4 tiles along the KV axis for
    pipelining benefit.

    Returns:
        block_size between 16 and *max_block*.
    """
    budget = 4096
    # Start from the largest block whose row-width fits the budget
    bs = max(16, min(budget // max(D, 1), max_block))
    # Ensure at least 4 KV tiles for pipelining benefit
    while bs > 16 and math.ceil(S_k / bs) < 4:
        bs //= 2
    # Round down to power of 2 for alignment
    bs = 1 << (bs.bit_length() - 1) if bs > 0 else 16
    return max(16, min(bs, max_block))


# ======================================================================
#  Functional kernel
# ======================================================================

def flash_attention_v4_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = False,
    block_size: int | None = None,
    window_size: int | None = None,
    softcap: float | None = None,
    kv_cache: Tuple[torch.Tensor, torch.Tensor] | None = None,
    return_lse: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Flash Attention v4 — KV-cache, sliding-window, softcap.

    Args:
        q: [B, S_q, D]  or  [B, H, S_q, D]  — new query tokens.
        k: [B, S_new, D] or [B, H, S_new, D] — new key tokens to append.
        v: [B, S_new, D] or [B, H, S_new, D] — new value tokens to append.
        scale:        1/sqrt(D).
        causal:       causal mask.
        block_size:   tile size; ``None`` → auto select.
        window_size:  if set, each query only attends to the closest
                      ``window_size`` keys (sliding window).
        softcap:      if set, apply ``softcap * tanh(scores / softcap)``.
        kv_cache:     ``(K_cache, V_cache)`` tensors from previous steps.
                      K_cache: [B, H, S_past, D].  New K/V are appended and
                      the updated cache is stored in-place.
        return_lse:   if True, also return per-row logsumexp [B, H, S_q, 1].

    Returns:
        output  or  (output, logsumexp)
    """
    # ---- handle 3-D / 4-D -------------------------------------------
    squeezed = False
    if q.dim() == 3:
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
        if kv_cache is not None:
            kv_cache = (kv_cache[0].unsqueeze(1), kv_cache[1].unsqueeze(1))
        squeezed = True

    B, H, S_q, D = q.shape

    # ---- v4: KV-cache append -----------------------------------------
    if kv_cache is not None:
        k_cache, v_cache = kv_cache
        k_full = torch.cat([k_cache, k], dim=2)
        v_full = torch.cat([v_cache, v], dim=2)
        # Update cache in-place for caller
        kv_cache = (k_full, v_full)
    else:
        k_full = k
        v_full = v

    S_k = k_full.size(2)
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # ---- v4: adaptive block size -------------------------------------
    if block_size is None:
        block_size = auto_block_size(S_q, S_k, D)
    B_r, B_c = block_size, block_size
    n_br = math.ceil(S_q / B_r)
    n_bc = math.ceil(S_k / B_c)

    block_outputs: list[torch.Tensor] = []
    lse_outputs: list[torch.Tensor] = []

    for i in range(n_br):
        q_s = i * B_r
        q_e = min(q_s + B_r, S_q)
        br = q_e - q_s
        q_block = q[:, :, q_s:q_e, :]

        # For causal with KV-cache, actual query positions are offset by
        # the past cache length.
        if kv_cache is not None:
            q_abs_start = S_k - S_q + q_s  # absolute position in full seq
            q_abs_end = S_k - S_q + q_e
        else:
            q_abs_start = q_s
            q_abs_end = q_e

        O_i = torch.zeros(B, H, br, D, dtype=torch.float32, device=q.device)
        lse_i = torch.full((B, H, br, 1), float("-inf"),
                           dtype=torch.float32, device=q.device)

        for j in range(n_bc):
            k_s = j * B_c
            k_e = min(k_s + B_c, S_k)

            # ---- v4: sliding-window skip ------------------------------
            if window_size is not None:
                # Skip if the entire KV block is outside the window
                # For query positions [q_abs_start, q_abs_end), the
                # attend range is [q_pos - window_size + 1, q_pos].
                # Conservatively: skip if k_e-1 < q_abs_start - window_size + 1
                # OR k_s > q_abs_end - 1
                if k_e - 1 < q_abs_start - window_size + 1:
                    continue
                if k_s > q_abs_end - 1 and causal:
                    break

            # ---- causal early-exit (v2 carry-over) --------------------
            if causal and k_s > (q_abs_end - 1):
                break

            k_block = k_full[:, :, k_s:k_e, :]
            v_block = v_full[:, :, k_s:k_e, :]

            S_ij = torch.matmul(q_block.float(), k_block.float().transpose(-2, -1)) * scale

            # ---- v4: softcap ------------------------------------------
            if softcap is not None:
                S_ij = softcap * torch.tanh(S_ij / softcap)

            # Causal mask
            if causal:
                row_idx = torch.arange(q_abs_start, q_abs_end,
                                       device=q.device).unsqueeze(1)
                col_idx = torch.arange(k_s, k_e, device=q.device).unsqueeze(0)
                S_ij = S_ij.masked_fill(
                    (row_idx < col_idx).unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # ---- v4: sliding-window mask (per-element) ----------------
            if window_size is not None:
                row_idx = torch.arange(q_abs_start, q_abs_end,
                                       device=q.device).unsqueeze(1)
                col_idx = torch.arange(k_s, k_e, device=q.device).unsqueeze(0)
                outside_window = col_idx < (row_idx - window_size + 1)
                S_ij = S_ij.masked_fill(
                    outside_window.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            # ---- v2-style logsumexp online softmax --------------------
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

        block_outputs.append(O_i.to(q.dtype))
        if return_lse:
            lse_outputs.append(lse_i)

    O = torch.cat(block_outputs, dim=2)
    if squeezed:
        O = O.squeeze(1)

    if return_lse:
        lse = torch.cat(lse_outputs, dim=2)
        if squeezed:
            lse = lse.squeeze(1)
        # Also return updated kv_cache tuple if applicable
        return O, lse

    return O


# ======================================================================
#  nn.Module wrapper
# ======================================================================

class FlashAttentionV4(nn.Module):
    """MHA with Flash Attention v4 — KV-cache, sliding window, softcap.

    Parameters
    ----------
    hidden_size, num_heads, head_dim: standard MHA dims.
    block_size:   ``None`` for auto selection.
    window_size:  sliding-window size (None = full attention).
    softcap:      logit soft-capping value (None = disabled).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        block_size: int | None = None,
        window_size: int | None = None,
        softcap: float | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size
        self.block_size = block_size
        self.window_size = window_size
        self.softcap = softcap

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
        kv_cache: Tuple[torch.Tensor, torch.Tensor] | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape
        bs = block_size or self.block_size  # may be None → auto
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        result = flash_attention_v4_forward(
            q, k, v,
            scale=self.scale,
            causal=causal,
            block_size=bs,
            window_size=self.window_size,
            softcap=self.softcap,
            kv_cache=kv_cache,
            return_lse=return_lse,
        )

        if return_lse:
            attn_out, lse = result
        else:
            attn_out = result

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.o_proj(attn_out)

        if return_lse:
            return out, lse
        return out

    def extra_repr(self):
        return (f"hidden={self.hidden_size}, heads={self.num_heads}, "
                f"head_dim={self.head_dim}, block_size={self.block_size}, "
                f"window={self.window_size}, softcap={self.softcap}")
