"""
Tests for Window Attention and Cross-Attention.

Covers both the ``nn.Module`` wrappers and the functional APIs.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from attention import (
    WindowAttention,
    window_attention_forward,
    CrossAttention,
    cross_attention_forward,
    StandardMHA,
    safe_softmax,
)


# ====================================================================== #
#  Helpers                                                                 #
# ====================================================================== #

B, S, H, D = 2, 16, 4, 32
HIDDEN = H * D  # 128


def _rand(*shape, requires_grad=False):
    return torch.randn(*shape, requires_grad=requires_grad)


# ====================================================================== #
#  WindowAttention Module Tests                                            #
# ====================================================================== #

class TestWindowAttentionModule:
    def _make(self, window_size=8, causal=False):
        m = WindowAttention(HIDDEN, H, window_size=window_size, causal=causal)
        m.eval()
        return m

    @pytest.mark.parametrize("causal", [False, True])
    def test_output_shape(self, causal):
        m = self._make(causal=causal)
        x = _rand(B, S, HIDDEN)
        y = m(x)
        assert y.shape == (B, S, HIDDEN)

    def test_symmetric_mask(self):
        seq, w = 8, 4
        mask = WindowAttention._build_window_mask(seq, w, causal=False, device=torch.device("cpu"))
        assert mask.shape == (seq, seq)
        assert mask[0, 0].item() == 0.0
        assert mask[0, 2].item() == 0.0
        assert mask[0, 3].item() == float("-inf")
        for c in [2, 3, 4, 5, 6]:
            assert mask[4, c].item() == 0.0
        assert mask[4, 1].item() == float("-inf")
        assert mask[4, 7].item() == float("-inf")

    def test_causal_mask(self):
        seq, w = 8, 4
        mask = WindowAttention._build_window_mask(seq, w, causal=True, device=torch.device("cpu"))
        for c in [2, 3, 4, 5]:
            assert mask[5, c].item() == 0.0
        assert mask[5, 6].item() == float("-inf")
        assert mask[5, 1].item() == float("-inf")

    def test_causal_no_future(self):
        seq, w = 10, 10
        mask = WindowAttention._build_window_mask(seq, w, causal=True, device=torch.device("cpu"))
        for i in range(seq):
            for j in range(i + 1, seq):
                assert mask[i, j].item() == float("-inf"), f"row {i} sees future col {j}"

    def test_full_window_matches_standard(self):
        torch.manual_seed(42)
        m_win = WindowAttention(HIDDEN, H, window_size=S * 2)
        m_std = StandardMHA(HIDDEN, H)

        m_std.q_proj.weight.data.copy_(m_win.q_proj.weight.data)
        m_std.k_proj.weight.data.copy_(m_win.k_proj.weight.data)
        m_std.v_proj.weight.data.copy_(m_win.v_proj.weight.data)
        m_std.o_proj.weight.data.copy_(m_win.o_proj.weight.data)

        x = _rand(B, S, HIDDEN)
        m_win.eval(); m_std.eval()
        y_win = m_win(x)
        y_std = m_std(x)
        torch.testing.assert_close(y_win, y_std, atol=1e-4, rtol=1e-4)

    def test_gradient_flow(self):
        m = self._make()
        x = _rand(B, S, HIDDEN, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.all(x.grad == 0)

    def test_load_weights(self):
        m = self._make()
        wq = torch.eye(HIDDEN)
        wk = torch.eye(HIDDEN) * 2
        wv = torch.eye(HIDDEN) * 3
        wo = torch.eye(HIDDEN) * 4
        m.load_weights(wq, wk, wv, wo)
        torch.testing.assert_close(m.q_proj.weight.data, wq)
        torch.testing.assert_close(m.k_proj.weight.data, wk)
        torch.testing.assert_close(m.v_proj.weight.data, wv)
        torch.testing.assert_close(m.o_proj.weight.data, wo)

    def test_extra_repr(self):
        m = self._make(window_size=16, causal=True)
        r = m.extra_repr()
        assert "window=16" in r
        assert "causal=True" in r

    def test_dropout(self):
        m = WindowAttention(HIDDEN, H, window_size=8, dropout=0.1)
        assert m.attn_dropout.p == 0.1

    @pytest.mark.parametrize("w", [1, 4, 8, 16])
    def test_various_window_sizes(self, w):
        m = self._make(window_size=w)
        x = _rand(B, S, HIDDEN)
        y = m(x)
        assert y.shape == (B, S, HIDDEN)

    def test_window_size_1_symmetric(self):
        mask = WindowAttention._build_window_mask(8, 1, causal=False, device=torch.device("cpu"))
        for i in range(8):
            for j in range(8):
                if i == j:
                    assert mask[i, j].item() == 0.0
                else:
                    assert mask[i, j].item() == float("-inf")


# ====================================================================== #
#  window_attention_forward (functional) Tests                             #
# ====================================================================== #

class TestWindowAttentionFunctional:
    def test_4d_input(self):
        q = _rand(B, H, S, D)
        k = _rand(B, H, S, D)
        v = _rand(B, H, S, D)
        out = window_attention_forward(q, k, v, window_size=8)
        assert out.shape == (B, H, S, D)

    def test_3d_input(self):
        q = _rand(B, S, D)
        k = _rand(B, S, D)
        v = _rand(B, S, D)
        out = window_attention_forward(q, k, v, window_size=8)
        assert out.shape == (B, S, D)

    def test_causal(self):
        q = k = v = _rand(B, H, S, D)
        out = window_attention_forward(q, k, v, window_size=S, causal=True)
        assert out.shape == (B, H, S, D)

    def test_custom_scale(self):
        q = k = v = _rand(B, H, S, D)
        out = window_attention_forward(q, k, v, window_size=4, scale=0.5)
        assert out.shape == (B, H, S, D)

    def test_gradient_flow(self):
        q = _rand(B, H, S, D, requires_grad=True)
        k = _rand(B, H, S, D, requires_grad=True)
        v = _rand(B, H, S, D, requires_grad=True)
        out = window_attention_forward(q, k, v, window_size=8)
        out.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


# ====================================================================== #
#  CrossAttention Module Tests                                             #
# ====================================================================== #

class TestCrossAttentionModule:
    ENC_HIDDEN = 64
    S_KV = 12

    def _make(self, enc_hidden=None, bias=False):
        eh = enc_hidden or self.ENC_HIDDEN
        m = CrossAttention(HIDDEN, eh, num_heads=H, bias=bias)
        m.eval()
        return m

    def test_output_shape(self):
        m = self._make()
        dec = _rand(B, S, HIDDEN)
        enc = _rand(B, self.S_KV, self.ENC_HIDDEN)
        y = m(dec, enc)
        assert y.shape == (B, S, HIDDEN)

    def test_same_hidden_sizes(self):
        m = CrossAttention(HIDDEN, HIDDEN, num_heads=H)
        m.eval()
        dec = _rand(B, S, HIDDEN)
        enc = _rand(B, self.S_KV, HIDDEN)
        y = m(dec, enc)
        assert y.shape == (B, S, HIDDEN)

    def test_long_encoder(self):
        m = self._make()
        dec = _rand(B, 4, HIDDEN)
        enc = _rand(B, 32, self.ENC_HIDDEN)
        y = m(dec, enc)
        assert y.shape == (B, 4, HIDDEN)

    def test_gradient_flow_both(self):
        m = self._make()
        dec = _rand(B, S, HIDDEN, requires_grad=True)
        enc = _rand(B, self.S_KV, self.ENC_HIDDEN, requires_grad=True)
        y = m(dec, enc)
        y.sum().backward()
        assert dec.grad is not None and not torch.all(dec.grad == 0)
        assert enc.grad is not None and not torch.all(enc.grad == 0)

    def test_load_weights(self):
        m = self._make()
        inner = H * (HIDDEN // H)
        wq = _rand(inner, HIDDEN)
        wk = _rand(inner, self.ENC_HIDDEN)
        wv = _rand(inner, self.ENC_HIDDEN)
        wo = _rand(HIDDEN, inner)
        m.load_weights(wq, wk, wv, wo)
        torch.testing.assert_close(m.q_proj.weight.data, wq)
        torch.testing.assert_close(m.k_proj.weight.data, wk)

    def test_kv_cache(self):
        m = self._make()
        enc = _rand(B, self.S_KV, self.ENC_HIDDEN)
        kv = m.encode_kv(enc)
        assert len(kv) == 2
        assert kv[0].shape == (B, H, self.S_KV, HIDDEN // H)

        dec = _rand(B, S, HIDDEN)
        y = m(dec, enc, kv_cache=kv)
        assert y.shape == (B, S, HIDDEN)

    def test_kv_cache_matches_live(self):
        torch.manual_seed(7)
        m = self._make()
        dec = _rand(B, S, HIDDEN)
        enc = _rand(B, self.S_KV, self.ENC_HIDDEN)

        y_live = m(dec, enc)
        kv = m.encode_kv(enc)
        y_cached = m(dec, enc, kv_cache=kv)
        torch.testing.assert_close(y_live, y_cached, atol=1e-5, rtol=1e-5)

    def test_attention_mask(self):
        m = self._make()
        dec = _rand(B, S, HIDDEN)
        enc = _rand(B, self.S_KV, self.ENC_HIDDEN)
        mask = torch.zeros(B, 1, S, self.S_KV)
        y = m(dec, enc, attention_mask=mask)
        assert y.shape == (B, S, HIDDEN)

    def test_with_bias(self):
        m = self._make(bias=True)
        assert m.q_proj.bias is not None
        dec = _rand(B, S, HIDDEN)
        enc = _rand(B, self.S_KV, self.ENC_HIDDEN)
        y = m(dec, enc)
        assert y.shape == (B, S, HIDDEN)

    def test_extra_repr(self):
        m = self._make()
        r = m.extra_repr()
        assert "dec_hidden" in r
        assert "enc_hidden" in r

    def test_self_attend_via_cross(self):
        torch.manual_seed(99)
        m_cross = CrossAttention(HIDDEN, HIDDEN, num_heads=H)
        m_std = StandardMHA(HIDDEN, H)

        m_std.q_proj.weight.data.copy_(m_cross.q_proj.weight.data)
        m_std.k_proj.weight.data.copy_(m_cross.k_proj.weight.data)
        m_std.v_proj.weight.data.copy_(m_cross.v_proj.weight.data)
        m_std.o_proj.weight.data.copy_(m_cross.o_proj.weight.data)

        x = _rand(B, S, HIDDEN)
        m_cross.eval(); m_std.eval()
        y_cross = m_cross(x, x)
        y_std = m_std(x)
        torch.testing.assert_close(y_cross, y_std, atol=1e-4, rtol=1e-4)

    def test_custom_head_dim(self):
        m = CrossAttention(HIDDEN, 64, num_heads=H, head_dim=16)
        dec = _rand(B, S, HIDDEN)
        enc = _rand(B, 10, 64)
        y = m(dec, enc)
        assert y.shape == (B, S, HIDDEN)


# ====================================================================== #
#  cross_attention_forward (functional) Tests                              #
# ====================================================================== #

class TestCrossAttentionFunctional:
    S_KV = 12

    def test_basic_shape(self):
        q = _rand(B, H, S, D)
        k = _rand(B, H, self.S_KV, D)
        v = _rand(B, H, self.S_KV, D)
        out = cross_attention_forward(q, k, v)
        assert out.shape == (B, H, S, D)

    def test_custom_scale(self):
        q = _rand(B, H, S, D)
        k = _rand(B, H, self.S_KV, D)
        v = _rand(B, H, self.S_KV, D)
        out = cross_attention_forward(q, k, v, scale=0.1)
        assert out.shape == (B, H, S, D)

    def test_with_mask(self):
        q = _rand(B, H, S, D)
        k = _rand(B, H, self.S_KV, D)
        v = _rand(B, H, self.S_KV, D)
        mask = torch.zeros(B, 1, S, self.S_KV)
        out = cross_attention_forward(q, k, v, attention_mask=mask)
        assert out.shape == (B, H, S, D)

    def test_gradient_flow(self):
        q = _rand(B, H, S, D, requires_grad=True)
        k = _rand(B, H, 10, D, requires_grad=True)
        v = _rand(B, H, 10, D, requires_grad=True)
        out = cross_attention_forward(q, k, v)
        out.sum().backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_matches_manual(self):
        torch.manual_seed(0)
        q = _rand(B, H, S, D)
        k = _rand(B, H, 8, D)
        v = _rand(B, H, 8, D)
        sc = 1.0 / math.sqrt(D)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sc
        attn = safe_softmax(scores, dim=-1, dtype=torch.float32)
        expected = torch.matmul(attn, v.float())

        out = cross_attention_forward(q, k, v)
        torch.testing.assert_close(out.float(), expected.float(), atol=1e-5, rtol=1e-5)

    def test_equal_seq_len(self):
        q = k = v = _rand(B, H, S, D)
        out = cross_attention_forward(q, k, v)
        assert out.shape == (B, H, S, D)
