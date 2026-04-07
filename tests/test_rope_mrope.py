"""
Tests for RoPE and mRoPE implementations.

TestRotateHalf           — rotate_half() functional correctness
TestApplyRotaryEmb       — apply_rotary_emb() shape and numerical checks
TestRotaryEmbedding      — RotaryEmbedding cache, scaling, relative-position invariance
TestRoPEAttention        — full MHA with RoPE: shape, causal, GQA, gradient
TestPositionIDHelpers    — make_text/image/video_position_ids shapes and values
TestMultimodalRoPE       — MultimodalRotaryEmbedding axis independence, cache
TestmRoPEAttention       — mRoPEAttention shape, text fallback, GQA, gradient
TestRelativePositionInvariance — key property: <RoPE(q,m), RoPE(k,n)> depends on m-n
"""

from __future__ import annotations

import math
import torch
import pytest

from attention.rope import (
    rotate_half,
    apply_rotary_emb,
    RotaryEmbedding,
    RoPEAttention,
)
from attention.mrope import (
    MultimodalRotaryEmbedding,
    mRoPEAttention,
    make_text_position_ids,
    make_image_position_ids,
    make_video_position_ids,
)


# =========================================================================== #
#  TestRotateHalf                                                               #
# =========================================================================== #

class TestRotateHalf:

    def test_shape_preserved(self):
        x = torch.randn(2, 4, 8, 32)
        assert rotate_half(x).shape == x.shape

    def test_definition(self):
        """rotate_half([a, b]) == [-b, a] (splitting on last dim)."""
        x = torch.tensor([[1., 2., 3., 4.]])   # [1, 4]
        expected = torch.tensor([[-3., -4., 1., 2.]])
        torch.testing.assert_close(rotate_half(x), expected)

    def test_double_rotate_is_negation(self):
        """Applying rotate_half twice negates the vector."""
        x = torch.randn(2, 8, 32)
        torch.testing.assert_close(rotate_half(rotate_half(x)), -x, atol=1e-6, rtol=1e-6)

    def test_odd_last_dim_raises(self):
        with pytest.raises(Exception):
            rotate_half(torch.randn(2, 5))   # 5 is not divisible by 2

    def test_gradient_flows(self):
        x = torch.randn(2, 16, requires_grad=True)
        rotate_half(x).sum().backward()
        assert x.grad is not None


# =========================================================================== #
#  TestApplyRotaryEmb                                                          #
# =========================================================================== #

class TestApplyRotaryEmb:

    def _make_cos_sin(self, S, D, dtype=torch.float32):
        t = torch.arange(S, dtype=dtype)
        inv = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=dtype) / D))
        freqs = torch.outer(t, inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]   # [1,1,S,D]

    def test_output_shapes(self):
        B, H, S, D = 2, 4, 16, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        cos, sin = self._make_cos_sin(S, D)
        qr, kr = apply_rotary_emb(q, k, cos, sin)
        assert qr.shape == q.shape
        assert kr.shape == k.shape

    def test_cos_sin_1_matches_identity(self):
        """cos=1, sin=0 → output == input."""
        B, H, S, D = 1, 2, 8, 16
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        cos = torch.ones(1, 1, S, D)
        sin = torch.zeros(1, 1, S, D)
        qr, kr = apply_rotary_emb(q, k, cos, sin)
        torch.testing.assert_close(qr, q, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(kr, k, atol=1e-6, rtol=1e-6)

    def test_norm_preserved(self):
        """RoPE is an orthogonal transformation: ||RoPE(x)|| == ||x||."""
        B, H, S, D = 1, 1, 8, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        cos, sin = self._make_cos_sin(S, D)
        qr, kr = apply_rotary_emb(q.float(), k.float(), cos, sin)
        torch.testing.assert_close(
            qr.norm(dim=-1), q.float().norm(dim=-1), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            kr.norm(dim=-1), k.float().norm(dim=-1), atol=1e-5, rtol=1e-5)


# =========================================================================== #
#  TestRotaryEmbedding                                                         #
# =========================================================================== #

class TestRotaryEmbedding:

    def test_output_shapes(self):
        rope = RotaryEmbedding(head_dim=32)
        q = torch.randn(2, 4, 16, 32)
        k = torch.randn(2, 4, 16, 32)
        qr, kr = rope(q, k)
        assert qr.shape == q.shape
        assert kr.shape == k.shape

    def test_dtype_preserved(self):
        rope = RotaryEmbedding(head_dim=32)
        q = torch.randn(1, 2, 8, 32, dtype=torch.float32)
        k = torch.randn(1, 2, 8, 32, dtype=torch.float32)
        qr, kr = rope(q, k)
        assert qr.dtype == q.dtype
        assert kr.dtype == k.dtype

    def test_norm_preserved(self):
        """RoPE must not change vector norms."""
        rope = RotaryEmbedding(head_dim=64, base=10000)
        q = torch.randn(1, 1, 16, 64)
        k = torch.randn(1, 1, 16, 64)
        qr, kr = rope(q, k)
        torch.testing.assert_close(
            qr.float().norm(dim=-1), q.float().norm(dim=-1), atol=1e-4, rtol=1e-4)

    def test_cache_auto_extends(self):
        rope = RotaryEmbedding(head_dim=32, max_seq_len=8)
        q = torch.randn(1, 2, 32, 32)   # S=32 > max_seq_len=8
        k = torch.randn(1, 2, 32, 32)
        qr, kr = rope(q, k)
        assert qr.shape == q.shape

    def test_position_ids_applied(self):
        """position_ids must actually shift the rotation."""
        rope = RotaryEmbedding(head_dim=32)
        q = torch.randn(1, 1, 4, 32)
        k = torch.randn(1, 1, 4, 32)
        # Standard sequential
        qr1, _ = rope(q, k)
        # Reversed positions
        pos_rev = torch.tensor([[3, 2, 1, 0]])
        qr2, _ = rope(q, k, position_ids=pos_rev)
        # Should differ
        assert not torch.allclose(qr1, qr2)

    def test_linear_scaling(self):
        rope = RotaryEmbedding(head_dim=32, max_seq_len=16,
                               rope_scaling={"type": "linear", "factor": 2.0})
        q = torch.randn(1, 2, 16, 32)
        k = torch.randn(1, 2, 16, 32)
        qr, kr = rope(q, k)
        assert qr.shape == q.shape

    def test_dynamic_scaling(self):
        rope = RotaryEmbedding(head_dim=32, max_seq_len=8,
                               rope_scaling={"type": "dynamic", "factor": 2.0})
        q = torch.randn(1, 2, 32, 32)   # S=32 > max_seq_len
        k = torch.randn(1, 2, 32, 32)
        qr, kr = rope(q, k)
        assert qr.shape == q.shape

    def test_position_0_no_rotation(self):
        """At position 0, cos=1 sin=0, so output should equal input."""
        rope = RotaryEmbedding(head_dim=32)
        q = torch.randn(1, 1, 1, 32)
        k = torch.randn(1, 1, 1, 32)
        pos = torch.zeros(1, 1, dtype=torch.long)
        qr, kr = rope(q, k, position_ids=pos)
        torch.testing.assert_close(qr, q.float(), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
    def test_various_head_dims(self, head_dim):
        rope = RotaryEmbedding(head_dim=head_dim)
        q = torch.randn(1, 2, 8, head_dim)
        k = torch.randn(1, 2, 8, head_dim)
        qr, kr = rope(q, k)
        assert qr.shape == q.shape


# =========================================================================== #
#  TestRoPEAttention                                                           #
# =========================================================================== #

class TestRoPEAttention:

    def test_output_shape(self):
        m = RoPEAttention(64, 4)
        x = torch.randn(2, 16, 64)
        assert m(x).shape == (2, 16, 64)

    def test_causal_no_future_leak(self):
        m = RoPEAttention(32, 2, causal=True)
        m.eval()
        x = torch.randn(1, 12, 32)
        x_mod = x.clone()
        x_mod[0, 6:] += 100.0
        out1 = m(x)
        out2 = m(x_mod)
        torch.testing.assert_close(out1[0, :6], out2[0, :6], atol=1e-4, rtol=1e-4)

    def test_gqa_reduces_kv_params(self):
        m_mha = RoPEAttention(64, 4, num_kv_heads=4)
        m_gqa = RoPEAttention(64, 4, num_kv_heads=2)
        assert (sum(p.numel() for p in m_gqa.k_proj.parameters()) <
                sum(p.numel() for p in m_mha.k_proj.parameters()))

    def test_gqa_output_shape(self):
        m = RoPEAttention(64, 4, num_kv_heads=2)
        x = torch.randn(2, 16, 64)
        assert m(x).shape == (2, 16, 64)

    def test_position_ids_accepted(self):
        m = RoPEAttention(32, 2)
        x = torch.randn(1, 8, 32)
        pos = torch.arange(8).unsqueeze(0)  # [1, 8]
        out = m(x, position_ids=pos)
        assert out.shape == (1, 8, 32)

    def test_gradient_flows(self):
        m = RoPEAttention(64, 4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)

    def test_weight_grad_flows(self):
        m = RoPEAttention(32, 2)
        x = torch.randn(1, 4, 32, requires_grad=True)
        m(x).sum().backward()
        assert m.q_proj.weight.grad is not None

    def test_attention_mask_applied(self):
        m = RoPEAttention(32, 2)
        m.eval()
        x = torch.randn(1, 4, 32)
        mask = torch.zeros(1, 1, 4, 4)
        out_no_mask = m(x)
        out_mask = m(x, attention_mask=mask)
        torch.testing.assert_close(out_no_mask, out_mask, atol=1e-5, rtol=1e-5)

    def test_deterministic_eval(self):
        m = RoPEAttention(32, 2)
        m.eval()
        x = torch.randn(1, 8, 32)
        torch.testing.assert_close(m(x), m(x))

    @pytest.mark.parametrize("S", [1, 7, 16, 64])
    def test_various_seq_lengths(self, S):
        m = RoPEAttention(32, 2)
        m.eval()
        x = torch.randn(1, S, 32)
        assert m(x).shape == (1, S, 32)


# =========================================================================== #
#  TestPositionIDHelpers                                                       #
# =========================================================================== #

class TestPositionIDHelpers:

    def test_text_shape(self):
        ids = make_text_position_ids(seq_len=10, num_axes=3, batch_size=2)
        assert ids.shape == (2, 3, 10)

    def test_text_all_axes_equal(self):
        ids = make_text_position_ids(seq_len=8, num_axes=3)
        # All three axes should be identical
        torch.testing.assert_close(ids[0, 0], ids[0, 1])
        torch.testing.assert_close(ids[0, 0], ids[0, 2])

    def test_text_monotonic(self):
        ids = make_text_position_ids(seq_len=8, num_axes=2)
        diffs = ids[0, 0, 1:] - ids[0, 0, :-1]
        assert (diffs == 1).all()

    def test_image_shape(self):
        ids = make_image_position_ids(height=4, width=4, text_len=5,
                                      num_axes=3, batch_size=2)
        assert ids.shape == (2, 3, 5 + 4 * 4)

    def test_image_num_patches(self):
        H, W, T = 8, 6, 3
        ids = make_image_position_ids(height=H, width=W, text_len=T, num_axes=3)
        assert ids.shape[-1] == T + H * W

    def test_image_height_range(self):
        """Height axis of image patches should contain values 0..H-1."""
        H, W = 3, 4
        ids = make_image_position_ids(height=H, width=W, text_len=0, num_axes=3)
        # axis 1 = height
        h_vals = ids[0, 1, :]
        assert int(h_vals.min()) == 0
        assert int(h_vals.max()) == H - 1

    def test_image_width_range(self):
        H, W = 3, 4
        ids = make_image_position_ids(height=H, width=W, text_len=0, num_axes=3)
        w_vals = ids[0, 2, :]
        assert int(w_vals.min()) == 0
        assert int(w_vals.max()) == W - 1

    def test_video_shape(self):
        ids = make_video_position_ids(num_frames=2, height=4, width=4,
                                      text_len=3, num_axes=3, batch_size=2)
        assert ids.shape == (2, 3, 3 + 2 * 4 * 4)

    def test_video_temporal_range(self):
        T, H, W = 3, 2, 2
        ids = make_video_position_ids(num_frames=T, height=H, width=W,
                                      text_len=0, num_axes=3)
        t_vals = ids[0, 0, :]
        assert int(t_vals.min()) == 0
        assert int(t_vals.max()) == T - 1

    def test_2_axes_image(self):
        ids = make_image_position_ids(height=4, width=4, text_len=0, num_axes=2)
        assert ids.shape[-2] == 2   # M=2


# =========================================================================== #
#  TestMultimodalRoPE                                                          #
# =========================================================================== #

class TestMultimodalRoPE:

    def test_output_shapes(self):
        rope = MultimodalRotaryEmbedding(head_dim=48, num_axes=3)
        q = torch.randn(2, 4, 16, 48)
        k = torch.randn(2, 4, 16, 48)
        pos = make_text_position_ids(16, num_axes=3, batch_size=2)
        qr, kr = rope(q, k, pos)
        assert qr.shape == q.shape
        assert kr.shape == k.shape

    def test_norm_preserved(self):
        rope = MultimodalRotaryEmbedding(head_dim=48, num_axes=3)
        q = torch.randn(1, 2, 8, 48)
        k = torch.randn(1, 2, 8, 48)
        pos = make_text_position_ids(8, num_axes=3)
        qr, kr = rope(q, k, pos)
        torch.testing.assert_close(
            qr.float().norm(dim=-1), q.float().norm(dim=-1), atol=1e-4, rtol=1e-4)

    def test_different_axes_produce_different_results(self):
        """Changing axis positions should change the rotated output."""
        rope = MultimodalRotaryEmbedding(head_dim=48, num_axes=3)
        q = torch.randn(1, 2, 8, 48)
        k = torch.randn(1, 2, 8, 48)
        pos_text = make_text_position_ids(8, num_axes=3)
        pos_img  = make_image_position_ids(2, 4, num_axes=3)[:, :, :8]
        qr_text, _ = rope(q, k, pos_text)
        qr_img,  _ = rope(q, k, pos_img)
        assert not torch.allclose(qr_text, qr_img, atol=1e-4)

    def test_head_dim_must_be_divisible(self):
        with pytest.raises(AssertionError):
            MultimodalRotaryEmbedding(head_dim=32, num_axes=3)   # 32 % 6 != 0

    def test_head_dim_valid(self):
        MultimodalRotaryEmbedding(head_dim=48, num_axes=3)   # 48 % 6 == 0 ✓

    def test_cache_extends_automatically(self):
        rope = MultimodalRotaryEmbedding(head_dim=48, num_axes=3, max_seq_len=4)
        q = torch.randn(1, 2, 16, 48)
        k = torch.randn(1, 2, 16, 48)
        pos = make_text_position_ids(16, num_axes=3)
        qr, kr = rope(q, k, pos)
        assert qr.shape == q.shape

    def test_position_0_no_rotation(self):
        """All-zero positions → cos=1, sin=0 → no rotation."""
        rope = MultimodalRotaryEmbedding(head_dim=48, num_axes=3)
        q = torch.randn(1, 1, 1, 48)
        k = torch.randn(1, 1, 1, 48)
        pos = torch.zeros(1, 3, 1, dtype=torch.long)
        qr, kr = rope(q, k, pos)
        torch.testing.assert_close(qr, q.float(), atol=1e-5, rtol=1e-5)

    def test_extra_repr(self):
        rope = MultimodalRotaryEmbedding(head_dim=48, num_axes=3, base=10000)
        r = rope.extra_repr()
        assert "48" in r and "3" in r


# =========================================================================== #
#  TestmRoPEAttention                                                          #
# =========================================================================== #

class TestmRoPEAttention:

    def test_output_shape(self):
        m = mRoPEAttention(hidden_size=48, num_heads=4, num_axes=3)
        x = torch.randn(2, 16, 48)
        assert m(x).shape == (2, 16, 48)

    def test_text_fallback_no_position_ids(self):
        """With position_ids=None, falls back to sequential text positions."""
        m = mRoPEAttention(hidden_size=48, num_heads=4, num_axes=3)
        m.eval()
        x = torch.randn(1, 8, 48)
        out = m(x)
        assert out.shape == (1, 8, 48)
        assert torch.isfinite(out).all()

    def test_explicit_text_positions(self):
        m = mRoPEAttention(hidden_size=48, num_heads=4, num_axes=3)
        m.eval()
        x = torch.randn(1, 8, 48)
        pos = make_text_position_ids(8, num_axes=3)
        out = m(x, position_ids=pos)
        assert out.shape == (1, 8, 48)

    def test_image_positions(self):
        """Feed image position_ids to mRoPEAttention."""
        H, W, T_txt = 4, 4, 4
        total_len = T_txt + H * W
        m = mRoPEAttention(hidden_size=48, num_heads=4, num_axes=3)
        m.eval()
        x = torch.randn(1, total_len, 48)
        pos = make_image_position_ids(H, W, text_len=T_txt, num_axes=3)
        out = m(x, position_ids=pos)
        assert out.shape == (1, total_len, 48)

    def test_video_positions(self):
        T, H, W = 2, 3, 3
        n = T * H * W
        m = mRoPEAttention(hidden_size=48, num_heads=4, num_axes=3)
        m.eval()
        x = torch.randn(1, n, 48)
        pos = make_video_position_ids(T, H, W, text_len=0, num_axes=3)
        out = m(x, position_ids=pos)
        assert out.shape == (1, n, 48)

    def test_gqa_output_shape(self):
        m = mRoPEAttention(hidden_size=48, num_heads=4, num_kv_heads=2, num_axes=3)
        x = torch.randn(2, 8, 48)
        assert m(x).shape == (2, 8, 48)

    def test_causal_no_future_leak(self):
        m = mRoPEAttention(hidden_size=48, num_heads=4, num_axes=3, causal=True)
        m.eval()
        x = torch.randn(1, 12, 48)
        x_mod = x.clone()
        x_mod[0, 6:] += 100.0
        out1 = m(x)
        out2 = m(x_mod)
        torch.testing.assert_close(out1[0, :6], out2[0, :6], atol=1e-4, rtol=1e-4)

    def test_gradient_flows(self):
        m = mRoPEAttention(hidden_size=48, num_heads=4, num_axes=3)
        x = torch.randn(2, 8, 48, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)

    def test_deterministic_eval(self):
        m = mRoPEAttention(hidden_size=48, num_heads=4, num_axes=3)
        m.eval()
        x = torch.randn(1, 8, 48)
        torch.testing.assert_close(m(x), m(x))


# =========================================================================== #
#  TestRelativePositionInvariance                                              #
# =========================================================================== #

class TestRelativePositionInvariance:
    """Core RoPE property: <RoPE(q,m), RoPE(k,n)> = f(q, k, m-n)."""

    def _rope_dot(self, rope, q_head, k_head, pos_q, pos_k):
        """Compute RoPE-rotated dot product for single head, single position."""
        # q_head, k_head: [D]
        q = q_head[None, None, None, :]   # [1,1,1,D]
        k = k_head[None, None, None, :]
        pos_q_t = torch.tensor([[pos_q]])
        pos_k_t = torch.tensor([[pos_k]])
        qr, _ = rope(q, q, position_ids=pos_q_t)
        kr, _ = rope(k, k, position_ids=pos_k_t)
        return (qr * kr).sum().item()

    def test_relative_shift_invariance(self):
        """<q@m, k@n> == <q@(m+delta), k@(n+delta)> for any delta."""
        rope = RotaryEmbedding(head_dim=32, max_seq_len=128)
        torch.manual_seed(0)
        q = torch.randn(32)
        k = torch.randn(32)

        dot_0_5   = self._rope_dot(rope, q, k, 0,  5)
        dot_3_8   = self._rope_dot(rope, q, k, 3,  8)   # same relative offset 5
        dot_10_15 = self._rope_dot(rope, q, k, 10, 15)

        assert abs(dot_0_5 - dot_3_8)   < 1e-4, "RoPE not shift-invariant"
        assert abs(dot_0_5 - dot_10_15) < 1e-4, "RoPE not shift-invariant"

    def test_different_relative_positions_differ(self):
        """Different relative offsets must give different dot products (in general)."""
        rope = RotaryEmbedding(head_dim=32)
        torch.manual_seed(1)
        q = torch.randn(32)
        k = torch.randn(32)
        dot_1 = self._rope_dot(rope, q, k, 0, 1)
        dot_5 = self._rope_dot(rope, q, k, 0, 5)
        assert abs(dot_1 - dot_5) > 1e-3

    def test_rope_is_length_preserving(self):
        """RoPE rotation must not change the L2 norm of each token."""
        rope = RotaryEmbedding(head_dim=64)
        B, H, S, D = 2, 4, 16, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        qr, kr = rope(q, k)
        torch.testing.assert_close(
            qr.float().norm(dim=-1), q.float().norm(dim=-1), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            kr.float().norm(dim=-1), k.float().norm(dim=-1), atol=1e-4, rtol=1e-4)

    def test_mrope_text_equals_rope_on_first_axis(self):
        """mRoPE with text positions (all axes equal) should give same result
        as 1-D RoPE applied to the first channel only."""
        head_dim = 48   # divisible by 6 (3 axes × 2)
        rope   = RotaryEmbedding(head_dim=head_dim // 3)   # 16-dim RoPE
        mrope  = MultimodalRotaryEmbedding(head_dim=head_dim, num_axes=3)

        torch.manual_seed(2)
        q = torch.randn(1, 1, 8, head_dim)
        k = torch.randn(1, 1, 8, head_dim)

        pos_text = make_text_position_ids(8, num_axes=3)
        qr_m, kr_m = mrope(q, k, pos_text)

        # Compare first channel (axis-0) of mRoPE with vanilla RoPE on that slice
        q_ch = q[..., :head_dim//3]
        k_ch = k[..., :head_dim//3]
        qr_1d, kr_1d = rope(q_ch, k_ch)

        torch.testing.assert_close(
            qr_m[..., :head_dim//3].float(), qr_1d.float(), atol=1e-5, rtol=1e-5)
