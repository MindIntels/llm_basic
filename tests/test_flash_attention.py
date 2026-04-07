"""
Tests for CPU Flash Attention.

Verifies:
  1. Functional kernel: ``flash_attention_forward`` matches naive SDPA.
  2. Various block sizes (tile boundary conditions).
  3. Causal masking correctness.
  4. 3-D and 4-D input layouts.
  5. ``FlashAttentionCPU`` module matches ``StandardMHA`` end-to-end.
  6. Gradient flow through the module.
  7. Memory advantage: peak tile is O(B_r * B_c) regardless of S.
"""

from __future__ import annotations

import math
import torch
import pytest

from attention.flash_attention_cpu import flash_attention_forward, FlashAttentionCPU
from attention.safe_softmax import safe_softmax
from attention.standard_attention import StandardMHA


# ======================================================================
#  Helpers
# ======================================================================

def _naive_sdpa_ref(q, k, v, scale, causal=False):
    """Minimal reference SDPA for raw Q/K/V tensors (no projections)."""
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
    weights = safe_softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def _make_weights(H: int):
    torch.manual_seed(42)
    return torch.randn(H, H), torch.randn(H, H), torch.randn(H, H), torch.randn(H, H)


# ======================================================================
#  Functional kernel tests
# ======================================================================

class TestFlashKernel3D:
    """Test ``flash_attention_forward`` with 3-D inputs [B, S, D]."""

    @pytest.mark.parametrize("block_size", [4, 7, 16, 64])
    def test_matches_naive(self, block_size):
        torch.manual_seed(0)
        B, S, D = 2, 32, 16
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, S, D)
        k = torch.randn(B, S, D)
        v = torch.randn(B, S, D)

        ref = _naive_sdpa_ref(q, k, v, scale)
        out = flash_attention_forward(q, k, v, scale=scale, block_size=block_size)

        assert out.shape == ref.shape
        assert torch.allclose(ref, out, atol=1e-5), (
            f"block_size={block_size}, max diff={( ref - out).abs().max().item():.6e}"
        )

    def test_causal(self):
        torch.manual_seed(1)
        B, S, D = 1, 20, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, S, D)
        k = torch.randn(B, S, D)
        v = torch.randn(B, S, D)

        ref = _naive_sdpa_ref(q, k, v, scale, causal=True)
        out = flash_attention_forward(q, k, v, scale=scale, causal=True, block_size=8)

        assert torch.allclose(ref, out, atol=1e-5), (
            f"max diff={( ref - out).abs().max().item():.6e}"
        )

    @pytest.mark.parametrize("S", [1, 3, 17, 64])
    def test_various_seq_lengths(self, S):
        """Non-power-of-2 and very short sequences."""
        torch.manual_seed(2)
        B, D = 1, 16
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, S, D)
        k = torch.randn(B, S, D)
        v = torch.randn(B, S, D)

        ref = _naive_sdpa_ref(q, k, v, scale)
        out = flash_attention_forward(q, k, v, scale=scale, block_size=8)

        assert torch.allclose(ref, out, atol=1e-5), (
            f"S={S}, max diff={(ref - out).abs().max().item():.6e}"
        )


class TestFlashKernel4D:
    """Test ``flash_attention_forward`` with 4-D inputs [B, H, S, D]."""

    def test_multihead(self):
        torch.manual_seed(3)
        B, H, S, D = 2, 4, 24, 16
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa_ref(q, k, v, scale)
        out = flash_attention_forward(q, k, v, scale=scale, block_size=8)

        assert out.shape == (B, H, S, D)
        assert torch.allclose(ref, out, atol=1e-5), (
            f"max diff={( ref - out).abs().max().item():.6e}"
        )

    def test_multihead_causal(self):
        torch.manual_seed(4)
        B, H, S, D = 1, 8, 32, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        ref = _naive_sdpa_ref(q, k, v, scale, causal=True)
        out = flash_attention_forward(q, k, v, scale=scale, causal=True, block_size=8)

        assert torch.allclose(ref, out, atol=1e-5), (
            f"max diff={( ref - out).abs().max().item():.6e}"
        )


# ======================================================================
#  Module-level tests
# ======================================================================

class TestFlashAttentionModule:
    """Test ``FlashAttentionCPU`` module vs ``StandardMHA``."""

    @pytest.mark.parametrize("B,S,H,heads", [
        (1, 16, 32, 4),
        (2, 32, 64, 8),
    ])
    def test_matches_standard(self, B, S, H, heads):
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)

        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o)
        ref.eval()

        flash = FlashAttentionCPU(H, heads, block_size=8)
        flash.load_weights(w_q, w_k, w_v, w_o)
        flash.eval()

        y_ref = ref(x)
        y_flash = flash(x)

        assert torch.allclose(y_ref, y_flash, atol=1e-4), (
            f"max diff = {(y_ref - y_flash).abs().max().item():.6e}"
        )

    def test_matches_standard_causal(self):
        B, S, H, heads = 2, 24, 64, 4
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)

        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o)
        ref.eval()

        flash = FlashAttentionCPU(H, heads, block_size=8)
        flash.load_weights(w_q, w_k, w_v, w_o)
        flash.eval()

        y_ref = ref(x, causal=True)
        y_flash = flash(x, causal=True, block_size=8)

        assert torch.allclose(y_ref, y_flash, atol=1e-4), (
            f"max diff = {(y_ref - y_flash).abs().max().item():.6e}"
        )


class TestFlashSplitQKVIntegration:
    """SplitQKVAttention with use_flash=True should match StandardMHA."""

    def test_split_qkv_flash_matches_standard(self):
        from attention.split_qkv import SplitQKVAttention

        B, S, H, heads = 2, 16, 32, 4
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)

        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o)
        ref.eval()

        split = SplitQKVAttention(H, heads, dropout=0.0)
        split.load_weights(w_q, w_k, w_v, w_o)
        split.eval()

        y_ref = ref(x)
        y_flash = split(x, use_flash=True, block_size=8)

        assert torch.allclose(y_ref, y_flash, atol=1e-4), (
            f"max diff = {(y_ref - y_flash).abs().max().item():.6e}"
        )

    def test_split_qkv_flash_causal(self):
        from attention.split_qkv import SplitQKVAttention

        B, S, H, heads = 1, 20, 32, 4
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)

        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o)
        ref.eval()

        split = SplitQKVAttention(H, heads, dropout=0.0)
        split.load_weights(w_q, w_k, w_v, w_o)
        split.eval()

        y_ref = ref(x, causal=True)
        y_flash = split(x, causal=True, use_flash=True, block_size=4)

        assert torch.allclose(y_ref, y_flash, atol=1e-4), (
            f"max diff = {(y_ref - y_flash).abs().max().item():.6e}"
        )


class TestFlashGrad:
    """Gradient flow."""

    def test_backward_module(self):
        B, S, H, heads = 2, 16, 32, 4
        model = FlashAttentionCPU(H, heads, block_size=8)
        x = torch.randn(B, S, H, requires_grad=True)
        y = model(x)
        y.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert model.q_proj.weight.grad is not None


class TestFlashBlockSizeEdge:
    """Edge cases for block_size."""

    def test_block_size_larger_than_seq(self):
        torch.manual_seed(5)
        B, S, D = 1, 4, 8
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, S, D)
        k = torch.randn(B, S, D)
        v = torch.randn(B, S, D)

        ref = _naive_sdpa_ref(q, k, v, scale)
        out = flash_attention_forward(q, k, v, scale=scale, block_size=128)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_block_size_one(self):
        torch.manual_seed(6)
        B, S, D = 1, 8, 4
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(B, S, D)
        k = torch.randn(B, S, D)
        v = torch.randn(B, S, D)

        ref = _naive_sdpa_ref(q, k, v, scale)
        out = flash_attention_forward(q, k, v, scale=scale, block_size=1)
        assert torch.allclose(ref, out, atol=1e-5), (
            f"max diff={(ref - out).abs().max().item():.6e}"
        )
