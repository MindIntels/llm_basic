"""
Tests for Flash Attention v4 — KV-cache, sliding window, softcap, auto block.

Verifies:
  1. Basic correctness (matches naive SDPA).
  2. KV-cache incremental decoding.
  3. Sliding-window attention.
  4. Softcap logit capping.
  5. Auto block-size selection.
  6. Return-logsumexp.
  7. Combined causal + sliding window + KV-cache.
  8. Module-level match with StandardMHA.
  9. Gradient flow.
"""

from __future__ import annotations

import math
import torch
import pytest

from attention.flash_attention_v4 import (
    flash_attention_v4_forward,
    auto_block_size,
    FlashAttentionV4,
)
from attention.standard_attention import StandardMHA
from attention.safe_softmax import safe_softmax


def _naive_sdpa(q, k, v, scale, causal=False, softcap=None, window_size=None):
    """Reference SDPA with optional softcap and sliding window."""
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)
    if causal:
        S_q, S_k = scores.size(-2), scores.size(-1)
        mask = torch.triu(
            torch.full((S_q, S_k), float("-inf"), dtype=scores.dtype, device=q.device),
            diagonal=1,
        )
        if scores.dim() == 3:
            scores = scores + mask
        else:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
    if window_size is not None:
        S_q, S_k = scores.size(-2), scores.size(-1)
        row_idx = torch.arange(S_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(S_k, device=q.device).unsqueeze(0)
        outside = col_idx < (row_idx - window_size + 1)
        if scores.dim() == 4:
            outside = outside.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(outside, float("-inf"))
    return torch.matmul(safe_softmax(scores, dim=-1), v.float()).to(q.dtype)


def _make_weights(H):
    torch.manual_seed(42)
    return torch.randn(H, H), torch.randn(H, H), torch.randn(H, H), torch.randn(H, H)


# ==================================================================
#  Basic correctness
# ==================================================================

class TestV4Basic:
    @pytest.mark.parametrize("block_size", [4, 8, 16])
    def test_matches_naive_3d(self, block_size):
        torch.manual_seed(0)
        B, S, D = 2, 32, 16
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v4_forward(q, k, v, scale=scale, block_size=block_size)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_matches_naive_4d(self):
        torch.manual_seed(1)
        B, H, S, D = 2, 4, 24, 16
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, H, S, D), torch.randn(B, H, S, D), torch.randn(B, H, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v4_forward(q, k, v, scale=scale, block_size=8)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_causal(self):
        torch.manual_seed(2)
        B, S, D = 1, 20, 8
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale, causal=True)
        out = flash_attention_v4_forward(q, k, v, scale=scale, causal=True, block_size=8)
        assert torch.allclose(ref, out, atol=1e-5)


# ==================================================================
#  KV-cache incremental decoding
# ==================================================================

class TestV4KVCache:
    def test_incremental_equals_full(self):
        torch.manual_seed(10)
        B, H, S, D = 1, 2, 16, 8
        scale = 1.0 / math.sqrt(D)

        q_full = torch.randn(B, H, S, D)
        k_full = torch.randn(B, H, S, D)
        v_full = torch.randn(B, H, S, D)

        ref = _naive_sdpa(q_full, k_full, v_full, scale)

        past = 12
        k_cache = k_full[:, :, :past, :]
        v_cache = v_full[:, :, :past, :]
        k_new = k_full[:, :, past:, :]
        v_new = v_full[:, :, past:, :]

        out = flash_attention_v4_forward(
            q_full, k_new, v_new, scale=scale, block_size=4,
            kv_cache=(k_cache, v_cache),
        )
        assert torch.allclose(ref, out, atol=1e-5), (
            f"max diff={(ref - out).abs().max().item():.6e}")

    def test_single_token_decode(self):
        torch.manual_seed(11)
        B, H, D = 1, 2, 8
        S_past = 20
        scale = 1.0 / math.sqrt(D)

        k_cache = torch.randn(B, H, S_past, D)
        v_cache = torch.randn(B, H, S_past, D)
        q_new = torch.randn(B, H, 1, D)
        k_new = torch.randn(B, H, 1, D)
        v_new = torch.randn(B, H, 1, D)

        k_all = torch.cat([k_cache, k_new], dim=2)
        v_all = torch.cat([v_cache, v_new], dim=2)
        ref = _naive_sdpa(q_new, k_all, v_all, scale)

        out = flash_attention_v4_forward(
            q_new, k_new, v_new, scale=scale, block_size=8,
            kv_cache=(k_cache, v_cache),
        )
        assert torch.allclose(ref, out, atol=1e-5)


# ==================================================================
#  Sliding-window attention
# ==================================================================

class TestV4SlidingWindow:
    def test_window_matches_reference(self):
        torch.manual_seed(20)
        B, S, D = 1, 24, 8
        W = 8
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale, window_size=W)
        out = flash_attention_v4_forward(q, k, v, scale=scale, window_size=W, block_size=4)
        assert torch.allclose(ref, out, atol=1e-5), (
            f"max diff={(ref - out).abs().max().item():.6e}")

    def test_causal_window(self):
        torch.manual_seed(21)
        B, S, D = 1, 20, 8
        W = 6
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale, causal=True, window_size=W)
        out = flash_attention_v4_forward(q, k, v, scale=scale, causal=True,
                                         window_size=W, block_size=4)
        assert torch.allclose(ref, out, atol=1e-5)


# ==================================================================
#  Softcap
# ==================================================================

class TestV4Softcap:
    def test_softcap_matches_reference(self):
        torch.manual_seed(30)
        B, S, D = 1, 16, 8
        cap = 10.0
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale, softcap=cap)
        out = flash_attention_v4_forward(q, k, v, scale=scale, softcap=cap, block_size=4)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_softcap_causal(self):
        torch.manual_seed(31)
        B, S, D = 1, 20, 8
        cap = 5.0
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale, causal=True, softcap=cap)
        out = flash_attention_v4_forward(q, k, v, scale=scale, causal=True,
                                         softcap=cap, block_size=8)
        assert torch.allclose(ref, out, atol=1e-5)


# ==================================================================
#  Auto block-size
# ==================================================================

class TestAutoBlockSize:
    def test_returns_power_of_2(self):
        bs = auto_block_size(128, 128, 64)
        assert bs > 0 and (bs & (bs - 1)) == 0  # power of 2

    def test_min_16(self):
        bs = auto_block_size(1, 1, 1)
        assert bs >= 16

    def test_auto_works_in_forward(self):
        torch.manual_seed(40)
        B, S, D = 1, 32, 16
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v4_forward(q, k, v, scale=scale, block_size=None)
        assert torch.allclose(ref, out, atol=1e-5)


# ==================================================================
#  Return logsumexp
# ==================================================================

class TestV4ReturnLSE:
    def test_returns_lse(self):
        torch.manual_seed(50)
        B, H, S, D = 1, 2, 16, 8
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, H, S, D), torch.randn(B, H, S, D), torch.randn(B, H, S, D)
        out, lse = flash_attention_v4_forward(q, k, v, scale=scale, block_size=4,
                                              return_lse=True)
        assert out.shape == (B, H, S, D)
        assert lse.shape == (B, H, S, 1)

    def test_lse_values_correct(self):
        torch.manual_seed(51)
        B, S, D = 1, 8, 4
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)

        _, lse = flash_attention_v4_forward(q, k, v, scale=scale, block_size=4,
                                            return_lse=True)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        ref_lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        assert torch.allclose(ref_lse, lse, atol=1e-5)


# ==================================================================
#  Module-level
# ==================================================================

class TestV4Module:
    @pytest.mark.parametrize("B,S,H,heads", [(1, 16, 32, 4), (2, 32, 64, 8)])
    def test_matches_standard(self, B, S, H, heads):
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)
        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o); ref.eval()
        v4 = FlashAttentionV4(H, heads, block_size=8)
        v4.load_weights(w_q, w_k, w_v, w_o); v4.eval()
        assert torch.allclose(ref(x), v4(x), atol=1e-3)

    def test_causal_module(self):
        B, S, H, heads = 2, 24, 64, 4
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)
        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o); ref.eval()
        v4 = FlashAttentionV4(H, heads, block_size=8)
        v4.load_weights(w_q, w_k, w_v, w_o); v4.eval()
        assert torch.allclose(ref(x, causal=True), v4(x, causal=True), atol=1e-3)


# ==================================================================
#  Gradient
# ==================================================================

class TestV4Grad:
    def test_backward(self):
        B, S, H, heads = 2, 16, 32, 4
        m = FlashAttentionV4(H, heads, block_size=8)
        x = torch.randn(B, S, H, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape

    def test_backward_with_window(self):
        B, S, H, heads = 1, 16, 32, 4
        m = FlashAttentionV4(H, heads, block_size=4, window_size=8)
        x = torch.randn(B, S, H, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None


# ==================================================================
#  Combined features
# ==================================================================

class TestV4Combined:
    def test_causal_window_softcap(self):
        torch.manual_seed(60)
        B, S, D = 1, 24, 8
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale, causal=True, softcap=8.0, window_size=10)
        out = flash_attention_v4_forward(q, k, v, scale=scale, causal=True,
                                         softcap=8.0, window_size=10, block_size=4)
        assert torch.allclose(ref, out, atol=1e-5)
