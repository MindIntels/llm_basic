"""
Tests for SplitQKVAttention — explicit per-head Q/K/V split MHA.

Verifies:
  1. Output shape correctness.
  2. Numerical equivalence with StandardMHA (same weights, same output).
  3. Per-head split/concat round-trip.
  4. Causal masking.
  5. Gradient flow.
"""

from __future__ import annotations

import torch
import pytest

from attention.standard_attention import StandardMHA
from attention.split_qkv import SplitQKVAttention


def _make_weights(H: int):
    torch.manual_seed(42)
    w_q = torch.randn(H, H)
    w_k = torch.randn(H, H)
    w_v = torch.randn(H, H)
    w_o = torch.randn(H, H)
    return w_q, w_k, w_v, w_o


class TestSplitQKVShape:
    def test_output_shape(self):
        B, S, H, heads = 2, 16, 64, 4
        model = SplitQKVAttention(H, heads, dropout=0.0)
        model.eval()
        x = torch.randn(B, S, H)
        y = model(x)
        assert y.shape == (B, S, H)

    def test_causal_output_shape(self):
        B, S, H, heads = 1, 32, 128, 8
        model = SplitQKVAttention(H, heads, dropout=0.0)
        model.eval()
        x = torch.randn(B, S, H)
        y = model(x, causal=True)
        assert y.shape == (B, S, H)


class TestSplitQKVCorrectness:
    @pytest.mark.parametrize("B,S,H,heads", [
        (1, 8, 32, 4),
        (2, 16, 64, 8),
        (1, 4, 16, 2),
    ])
    def test_matches_standard_mha(self, B, S, H, heads):
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)

        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o)
        ref.eval()

        split = SplitQKVAttention(H, heads, dropout=0.0)
        split.load_weights(w_q, w_k, w_v, w_o)
        split.eval()

        y_ref = ref(x)
        y_split = split(x)

        assert torch.allclose(y_ref, y_split, atol=1e-5), (
            f"max diff = {(y_ref - y_split).abs().max().item():.6e}"
        )

    def test_matches_standard_mha_causal(self):
        B, S, H, heads = 2, 16, 64, 4
        w_q, w_k, w_v, w_o = _make_weights(H)
        x = torch.randn(B, S, H)

        ref = StandardMHA(H, heads, dropout=0.0)
        ref.load_weights(w_q, w_k, w_v, w_o)
        ref.eval()

        split = SplitQKVAttention(H, heads, dropout=0.0)
        split.load_weights(w_q, w_k, w_v, w_o)
        split.eval()

        y_ref = ref(x, causal=True)
        y_split = split(x, causal=True)

        assert torch.allclose(y_ref, y_split, atol=1e-5), (
            f"max diff = {(y_ref - y_split).abs().max().item():.6e}"
        )


class TestSplitQKVHeadSplit:
    def test_split_concat_roundtrip(self):
        B, S, H, heads = 2, 8, 32, 4
        model = SplitQKVAttention(H, heads)
        x = torch.randn(B, S, H)

        heads_list = model._split_heads(x)
        assert len(heads_list) == heads
        for h in heads_list:
            assert h.shape == (B, S, H // heads)

        reconstructed = model._concat_heads(heads_list)
        assert torch.equal(x, reconstructed)

    def test_individual_heads_independent(self):
        B, S, H, num_heads = 1, 8, 32, 4
        model = SplitQKVAttention(H, num_heads, dropout=0.0)
        model.eval()

        x = torch.randn(B, S, H)

        q = model.q_proj(x)
        k = model.k_proj(x)
        v = model.v_proj(x)

        q_heads = model._split_heads(q)
        k_heads = model._split_heads(k)
        v_heads = model._split_heads(v)

        out_h0 = model._per_head_attention(q_heads[0], k_heads[0], v_heads[0])

        q_heads[1] = q_heads[1] + 100.0
        out_h0_after = model._per_head_attention(q_heads[0], k_heads[0], v_heads[0])

        assert torch.equal(out_h0, out_h0_after)


class TestSplitQKVGrad:
    def test_backward(self):
        B, S, H, heads = 2, 8, 32, 4
        model = SplitQKVAttention(H, heads, dropout=0.0)
        x = torch.randn(B, S, H, requires_grad=True)

        y = model(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert model.q_proj.weight.grad is not None
