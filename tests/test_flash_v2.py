"""
Tests for Flash Attention v2 — deferred rescaling, causal early-exit.

Verifies:
  1. Numerical equivalence with naive SDPA (3-D and 4-D).
  2. Causal masking correctness.
  3. Causal early-exit (v2 optimisation) — result still correct.
  4. Various block sizes.
  5. Module-level match with StandardMHA.
  6. Gradient flow.
"""

from __future__ import annotations

import math
import torch
import pytest

from attention.flash_attention_v2 import flash_attention_v2_forward, FlashAttentionV2
from attention.standard_attention import StandardMHA
from attention.safe_softmax import safe_softmax


def _naive_sdpa(q, k, v, scale, causal=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        S_q, S_k = scores.size(-2), scores.size(-1)
        mask = torch.triu(
            torch.full((S_q, S_k), float("-inf"), dtype=q.dtype, device=q.device),
            diagonal=1,
        )
        if scores.dim() == 3:
            scores = scores + mask
        else:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
    return torch.matmul(safe_softmax(scores, dim=-1), v)


def _make_weights(H):
    torch.manual_seed(42)
    return torch.randn(H, H), torch.randn(H, H), torch.randn(H, H), torch.randn(H, H)


# ==================================================================
#  Functional kernel
# ==================================================================

class TestV2Kernel3D:
    @pytest.mark.parametrize("block_size", [4, 7, 16, 64])
    def test_matches_naive(self, block_size):
        torch.manual_seed(0)
        B, S, D = 2, 32, 16
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v2_forward(q, k, v, scale=scale, block_size=block_size)
        assert torch.allclose(ref, out, atol=1e-5), (
            f"bs={block_size}, max diff={( ref - out).abs().max().item():.6e}")

    def test_causal(self):
        torch.manual_seed(1)
        B, S, D = 1, 20, 8
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale, causal=True)
        out = flash_attention_v2_forward(q, k, v, scale=scale, causal=True, block_size=8)
        assert torch.allclose(ref, out, atol=1e-5)

    @pytest.mark.parametrize("S", [1, 3, 17])
    def test_various_seq(self, S):
        torch.manual_seed(2)
        B, D = 1, 16
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v2_forward(q, k, v, scale=scale, block_size=8)
        assert torch.allclose(ref, out, atol=1e-5)


class TestV2Kernel4D:
    def test_multihead(self):
        torch.manual_seed(3)
        B, H, S, D = 2, 4, 24, 16
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, H, S, D), torch.randn(B, H, S, D), torch.randn(B, H, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v2_forward(q, k, v, scale=scale, block_size=8)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_multihead_causal(self):
        torch.manual_seed(4)
        B, H, S, D = 1, 8, 32, 8
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, H, S, D), torch.randn(B, H, S, D), torch.randn(B, H, S, D)
        ref = _naive_sdpa(q, k, v, scale, causal=True)
        out = flash_attention_v2_forward(q, k, v, scale=scale, causal=True, block_size=8)
        assert torch.allclose(ref, out, atol=1e-5)


# ==================================================================
#  Module-level tests
# ==================================================================

class TestV2Module:
    @pytest.mark.parametrize("B,S,H,heads", [(1, 16, 32, 4), (2, 32, 64, 8)])
    def test_matches_standard(self, B, S, H, heads):
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)

        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o); ref.eval()

        v2 = FlashAttentionV2(H, heads, block_size=8)
        v2.load_weights(w_q, w_k, w_v, w_o); v2.eval()

        assert torch.allclose(ref(x), v2(x), atol=1e-3), (
            f"max diff = {(ref(x) - v2(x)).abs().max().item():.6e}")

    def test_causal(self):
        B, S, H, heads = 2, 24, 64, 4
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)
        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o); ref.eval()
        v2 = FlashAttentionV2(H, heads, block_size=8)
        v2.load_weights(w_q, w_k, w_v, w_o); v2.eval()
        assert torch.allclose(ref(x, causal=True), v2(x, causal=True), atol=1e-3)


class TestV2Grad:
    def test_backward(self):
        B, S, H, heads = 2, 16, 32, 4
        m = FlashAttentionV2(H, heads, block_size=8)
        x = torch.randn(B, S, H, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape
        assert m.q_proj.weight.grad is not None


class TestV2EdgeCases:
    def test_block_size_1(self):
        torch.manual_seed(6)
        B, S, D = 1, 8, 4
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v2_forward(q, k, v, scale=scale, block_size=1)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_block_larger_than_seq(self):
        torch.manual_seed(5)
        B, S, D = 1, 4, 8
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v2_forward(q, k, v, scale=scale, block_size=128)
        assert torch.allclose(ref, out, atol=1e-5)
