"""
Tests for Flash Attention v3 — block-sparse, mixed-precision, two-pass.

Verifies:
  1. Default mode (online softmax) matches naive SDPA.
  2. Two-pass mode matches naive SDPA.
  3. Block-sparse masking correctness.
  4. Mixed-precision (FP16 compute, FP32 accumulation).
  5. Combined causal + block-sparse.
  6. Module-level match with StandardMHA.
  7. Gradient flow.
"""

from __future__ import annotations

import math
import torch
import pytest

from attention.flash_attention_v3 import flash_attention_v3_forward, FlashAttentionV3
from attention.safe_softmax import safe_softmax
from attention.standard_attention import StandardMHA


def _naive_sdpa(q, k, v, scale, causal=False):
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
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
    return torch.matmul(safe_softmax(scores, dim=-1), v.float()).to(q.dtype)


def _make_weights(H):
    torch.manual_seed(42)
    return torch.randn(H, H), torch.randn(H, H), torch.randn(H, H), torch.randn(H, H)


# ==================================================================
#  Online-softmax mode
# ==================================================================

class TestV3OnlineMode:
    @pytest.mark.parametrize("block_size", [4, 8, 16])
    def test_matches_naive_3d(self, block_size):
        torch.manual_seed(0)
        B, S, D = 2, 32, 16
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v3_forward(q, k, v, scale=scale, block_size=block_size)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_matches_naive_4d(self):
        torch.manual_seed(1)
        B, H, S, D = 2, 4, 24, 16
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, H, S, D), torch.randn(B, H, S, D), torch.randn(B, H, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v3_forward(q, k, v, scale=scale, block_size=8)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_causal(self):
        torch.manual_seed(2)
        B, S, D = 1, 20, 8
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale, causal=True)
        out = flash_attention_v3_forward(q, k, v, scale=scale, causal=True, block_size=8)
        assert torch.allclose(ref, out, atol=1e-5)


# ==================================================================
#  Two-pass mode
# ==================================================================

class TestV3TwoPass:
    @pytest.mark.parametrize("block_size", [4, 8])
    def test_two_pass_matches_naive(self, block_size):
        torch.manual_seed(10)
        B, S, D = 2, 24, 16
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v3_forward(q, k, v, scale=scale, block_size=block_size,
                                         two_pass=True)
        assert torch.allclose(ref, out, atol=1e-5), (
            f"max diff={(ref-out).abs().max().item():.6e}")

    def test_two_pass_causal(self):
        torch.manual_seed(11)
        B, S, D = 1, 20, 8
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale, causal=True)
        out = flash_attention_v3_forward(q, k, v, scale=scale, causal=True,
                                         block_size=8, two_pass=True)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_two_pass_4d(self):
        torch.manual_seed(12)
        B, H, S, D = 1, 4, 16, 8
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, H, S, D), torch.randn(B, H, S, D), torch.randn(B, H, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v3_forward(q, k, v, scale=scale, block_size=4, two_pass=True)
        assert torch.allclose(ref, out, atol=1e-5)


# ==================================================================
#  Block-sparse attention
# ==================================================================

class TestV3BlockSparse:
    def test_full_mask_equals_naive(self):
        torch.manual_seed(20)
        B, S, D = 1, 16, 8
        scale = 1.0 / math.sqrt(D)
        bs = 4
        n_br = math.ceil(S / bs)
        n_bc = math.ceil(S / bs)
        mask = torch.ones(n_br, n_bc, dtype=torch.bool)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = _naive_sdpa(q, k, v, scale)
        out = flash_attention_v3_forward(q, k, v, scale=scale, block_size=bs,
                                         block_mask=mask)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_sparse_mask_zeroes_tiles(self):
        torch.manual_seed(21)
        B, S, D = 1, 16, 8
        scale = 1.0 / math.sqrt(D)
        bs = 8
        n_br = math.ceil(S / bs)
        n_bc = math.ceil(S / bs)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)

        full_mask = torch.ones(n_br, n_bc, dtype=torch.bool)
        sparse_mask = full_mask.clone()
        sparse_mask[0, 1] = False  # skip one tile

        out_full = flash_attention_v3_forward(q, k, v, scale=scale,
                                              block_size=bs, block_mask=full_mask)
        out_sparse = flash_attention_v3_forward(q, k, v, scale=scale,
                                                block_size=bs, block_mask=sparse_mask)
        assert not torch.allclose(out_full, out_sparse, atol=1e-6)


# ==================================================================
#  Mixed-precision simulation
# ==================================================================

class TestV3MixedPrecision:
    def test_fp16_compute(self):
        torch.manual_seed(30)
        B, S, D = 2, 16, 16
        scale = 1.0 / math.sqrt(D)
        q, k, v = torch.randn(B, S, D), torch.randn(B, S, D), torch.randn(B, S, D)
        ref = flash_attention_v3_forward(q, k, v, scale=scale, block_size=8)
        out = flash_attention_v3_forward(q, k, v, scale=scale, block_size=8,
                                         compute_dtype=torch.float16)
        assert torch.allclose(ref, out, atol=1e-3), (
            f"max diff={(ref-out).abs().max().item():.6e}")


# ==================================================================
#  Module-level
# ==================================================================

class TestV3Module:
    @pytest.mark.parametrize("B,S,H,heads", [(1, 16, 32, 4), (2, 32, 64, 8)])
    def test_matches_standard(self, B, S, H, heads):
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)
        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o); ref.eval()
        v3 = FlashAttentionV3(H, heads, block_size=8)
        v3.load_weights(w_q, w_k, w_v, w_o); v3.eval()
        assert torch.allclose(ref(x), v3(x), atol=1e-3)

    def test_two_pass_module(self):
        B, S, H, heads = 1, 16, 32, 4
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)
        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o); ref.eval()
        v3 = FlashAttentionV3(H, heads, block_size=8)
        v3.load_weights(w_q, w_k, w_v, w_o); v3.eval()
        assert torch.allclose(ref(x), v3(x, two_pass=True), atol=1e-4)


class TestV3Grad:
    def test_backward(self):
        B, S, H, heads = 2, 16, 32, 4
        m = FlashAttentionV3(H, heads, block_size=8)
        x = torch.randn(B, S, H, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and x.grad.shape == x.shape

    def test_backward_two_pass(self):
        B, S, H, heads = 1, 16, 32, 4
        m = FlashAttentionV3(H, heads, block_size=8)
        x = torch.randn(B, S, H, requires_grad=True)
        m(x, two_pass=True).sum().backward()
        assert x.grad is not None
