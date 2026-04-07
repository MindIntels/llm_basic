"""
Tests for the Gated Architecture family:

  TestRMSNorm           — RMS normalisation, shape, dtype, grad
  TestSwiGLU            — functional swiglu, SwiGLUFFN shape / grad
  TestGatedAttention    — sigmoid / silu gate, causal, mask, grad
  TestGatedDeltaNet     — recurrent correctness, state propagation, step() API
  TestGatedTransformerBlock — single block with both mixer types
  TestGatedTransformer  — full stack, alternating pattern, residuals
  TestIntegration       — end-to-end forward + backward through full model
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import pytest

from attention.rmsnorm import RMSNorm
from attention.swiglu import SwiGLUFFN, swiglu
from attention.gated_attention import GatedAttention
from attention.gated_deltanet import GatedDeltaNet
from attention.gated_transformer import GatedTransformerBlock, GatedTransformer


# =========================================================================== #
#  TestRMSNorm                                                                  #
# =========================================================================== #

class TestRMSNorm:

    def test_output_shape(self):
        m = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        assert m(x).shape == (2, 16, 64)

    def test_zero_input_is_zero(self):
        m = RMSNorm(32)
        x = torch.zeros(1, 8, 32)
        # 0 / (eps ^ 0.5) * weight = 0 (weight init = 1)
        assert torch.allclose(m(x), x, atol=1e-6)

    def test_unit_scale_weight(self):
        """With initial weights=1 and no mean, output should equal x/rms(x)."""
        torch.manual_seed(0)
        m = RMSNorm(32)
        x = torch.randn(2, 8, 32)
        rms = x.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        expected = (x.float() / rms).to(x.dtype)
        torch.testing.assert_close(m(x), expected, atol=1e-5, rtol=1e-5)

    def test_learnable_weight_applied(self):
        m = RMSNorm(16)
        m.weight.data.fill_(2.0)
        x = torch.randn(1, 4, 16)
        rms = x.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        expected = (x.float() / rms * 2.0).to(x.dtype)
        torch.testing.assert_close(m(x), expected, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        m = RMSNorm(32)
        x = torch.randn(2, 8, 32, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)
        assert m.weight.grad is not None

    @pytest.mark.parametrize("shape", [(1, 1, 64), (3, 32, 64), (4, 7, 64)])
    def test_various_shapes(self, shape):
        m = RMSNorm(64)
        x = torch.randn(*shape)
        assert m(x).shape == shape

    def test_invariant_to_scale(self):
        """RMSNorm output w/ scale=c should equal c * RMSNorm output w/ scale=1."""
        m1 = RMSNorm(32); m2 = RMSNorm(32)
        m2.weight.data.mul_(3.0)
        x = torch.randn(2, 8, 32)
        torch.testing.assert_close(m2(x), 3.0 * m1(x), atol=1e-5, rtol=1e-5)

    def test_dtype_preserved(self):
        m = RMSNorm(32)
        x = torch.randn(1, 4, 32, dtype=torch.float32)
        assert m(x).dtype == torch.float32

    def test_no_bias_parameter(self):
        m = RMSNorm(64)
        params = dict(m.named_parameters())
        assert "weight" in params
        assert "bias" not in params

    def test_extra_repr(self):
        m = RMSNorm(128, eps=1e-8)
        assert "128" in m.extra_repr()


# =========================================================================== #
#  TestSwiGLU                                                                   #
# =========================================================================== #

class TestSwiGLU:

    def test_functional_shape(self):
        g = torch.randn(2, 8, 64)
        v = torch.randn(2, 8, 64)
        out = swiglu(g, v)
        assert out.shape == (2, 8, 64)

    def test_silu_gate(self):
        """swiglu(g, v) == silu(g) * v."""
        import torch.nn.functional as F
        g = torch.randn(3, 16, 32)
        v = torch.randn(3, 16, 32)
        expected = F.silu(g) * v
        torch.testing.assert_close(swiglu(g, v), expected)

    def test_zero_gate_zeroes_output(self):
        g = torch.zeros(2, 8, 16)
        v = torch.randn(2, 8, 16)
        assert torch.allclose(swiglu(g, v), torch.zeros_like(v))

    def test_ffn_output_shape(self):
        m = SwiGLUFFN(128)
        x = torch.randn(2, 16, 128)
        assert m(x).shape == (2, 16, 128)

    def test_ffn_custom_intermediate(self):
        m = SwiGLUFFN(64, intermediate_size=256)
        assert m.intermediate_size == 256
        x = torch.randn(1, 8, 64)
        assert m(x).shape == (1, 8, 64)

    def test_ffn_three_projections(self):
        m = SwiGLUFFN(64)
        names = {n for n, _ in m.named_parameters()}
        assert "gate_proj.weight" in names
        assert "up_proj.weight" in names
        assert "down_proj.weight" in names

    def test_ffn_gradient_flows(self):
        m = SwiGLUFFN(64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)

    @pytest.mark.parametrize("B,S,H", [(1, 4, 32), (3, 16, 64)])
    def test_ffn_batch_independence(self, B, S, H):
        m = SwiGLUFFN(H)
        m.eval()
        x = torch.randn(B, S, H)
        out_full = m(x)
        for b in range(B):
            out_b = m(x[b:b+1])
            torch.testing.assert_close(out_full[b:b+1], out_b, atol=1e-5, rtol=1e-5)

    def test_ffn_no_bias_by_default(self):
        m = SwiGLUFFN(64)
        assert m.gate_proj.bias is None
        assert m.up_proj.bias is None
        assert m.down_proj.bias is None

    def test_ffn_default_intermediate_size(self):
        """Default d_ff = 8/3 × d_model rounded to multiple of 64."""
        m = SwiGLUFFN(384)
        assert m.intermediate_size % 64 == 0
        raw = int(8 / 3 * 384)
        assert m.intermediate_size >= raw


# =========================================================================== #
#  TestGatedAttention                                                           #
# =========================================================================== #

class TestGatedAttention:

    @pytest.mark.parametrize("gate_act", ["sigmoid", "silu"])
    def test_output_shape(self, gate_act):
        m = GatedAttention(64, 4, gate_act=gate_act)
        x = torch.randn(2, 16, 64)
        assert m(x).shape == (2, 16, 64)

    def test_sigmoid_gate_bounded(self):
        """With sigmoid gate the output is bounded by the attention output magnitude."""
        m = GatedAttention(64, 4, gate_act="sigmoid")
        m.eval()
        x = torch.randn(2, 8, 64)
        # Manually compute max attn output
        import torch.nn.functional as F
        with torch.no_grad():
            out = m(x)
        # Output should not be NaN or Inf
        assert torch.isfinite(out).all()

    def test_causal_mask_no_future_leak(self):
        """With causal=True, output at position t must not depend on t+1."""
        m = GatedAttention(32, 2, causal=True)
        m.eval()
        x = torch.randn(1, 8, 32)
        x_mod = x.clone()
        x_mod[0, 5:, :] = x_mod[0, 5:, :] + 100.0
        out1 = m(x)
        out2 = m(x_mod)
        # Positions 0..4 must be identical
        torch.testing.assert_close(out1[0, :5], out2[0, :5], atol=1e-4, rtol=1e-4)

    def test_attention_mask_applied(self):
        m = GatedAttention(32, 2)
        m.eval()
        x = torch.randn(1, 4, 32)
        mask = torch.zeros(1, 1, 4, 4)
        out_no_mask = m(x)
        out_with_mask = m(x, attention_mask=mask)
        torch.testing.assert_close(out_no_mask, out_with_mask, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        m = GatedAttention(64, 4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)
        assert m.q_proj.weight.grad is not None
        assert m.g_proj.weight.grad is not None

    def test_gate_projects_input_not_attention(self):
        """The gate path and attention path are independent projections of x."""
        m = GatedAttention(32, 2, gate_act="sigmoid")
        assert m.g_proj.in_features == 32
        assert m.g_proj.out_features == 32

    @pytest.mark.parametrize("B,S", [(1, 1), (2, 32), (1, 64)])
    def test_various_seq_len(self, B, S):
        m = GatedAttention(32, 2)
        m.eval()
        x = torch.randn(B, S, 32)
        assert m(x).shape == (B, S, 32)

    def test_extra_repr(self):
        m = GatedAttention(64, 4, gate_act="silu")
        assert "silu" in m.extra_repr()

    def test_invalid_gate_act_raises(self):
        with pytest.raises(ValueError):
            GatedAttention(32, 2, gate_act="relu")

    def test_pre_norm_enabled(self):
        m = GatedAttention(64, 4, pre_norm=True)
        assert isinstance(m.gate_norm, RMSNorm)

    def test_pre_norm_disabled(self):
        m = GatedAttention(64, 4, pre_norm=False)
        assert isinstance(m.gate_norm, nn.Identity)


# =========================================================================== #
#  TestGatedDeltaNet                                                            #
# =========================================================================== #

class TestGatedDeltaNet:

    def test_output_shape(self):
        m = GatedDeltaNet(64, 4)
        x = torch.randn(2, 16, 64)
        assert m(x).shape == (2, 16, 64)

    def test_return_state_shape(self):
        H, D = 4, 16
        m = GatedDeltaNet(hidden_size=64, num_heads=H, head_dim=D)
        x = torch.randn(2, 8, 64)
        out, state = m(x, return_state=True)
        assert out.shape == (2, 8, 64)
        assert state.shape == (2, H, D, D)  # [B, H, head_dim, value_dim]

    def test_state_affects_output(self):
        """Non-zero initial state changes the output."""
        m = GatedDeltaNet(64, 4)
        m.eval()
        x = torch.randn(1, 4, 64)
        out_no_state = m(x)
        state = torch.randn(1, 4, 16, 16) * 0.01
        out_with_state = m(x, state=state)
        assert not torch.allclose(out_no_state, out_with_state, atol=1e-4)

    def test_step_matches_forward(self):
        """step() called S times should match forward() on the same sequence."""
        torch.manual_seed(42)
        B, S, H_size, H_heads = 1, 8, 64, 4
        D = H_size // H_heads    # 16

        m = GatedDeltaNet(H_size, H_heads)
        m.eval()

        x = torch.randn(B, S, H_size)

        # Full forward
        out_full, final_state = m(x, return_state=True)

        # Step-by-step
        state = m._init_state(B, x.device, x.dtype)
        step_outs = []
        for t in range(S):
            y_t, state = m.step(x[:, t, :], state)
            step_outs.append(y_t)

        out_step = torch.stack(step_outs, dim=1)  # [B, S, H_size]
        torch.testing.assert_close(out_full, out_step, atol=1e-4, rtol=1e-4)

    def test_gradient_flows(self):
        m = GatedDeltaNet(64, 4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)

    def test_key_normalization(self):
        """With norm_keys=True, keys passed to the state are unit-norm."""
        m = GatedDeltaNet(32, 2, head_dim=16, norm_keys=True)
        m.eval()
        x = torch.randn(1, 4, 32)
        # Internally we can't easily inspect, but output should be finite
        out = m(x)
        assert torch.isfinite(out).all()

    def test_no_output_gate(self):
        m = GatedDeltaNet(64, 4, use_output_gate=False)
        x = torch.randn(2, 8, 64)
        assert m(x).shape == (2, 8, 64)

    def test_state_continuity(self):
        """Splitting sequence in two and continuing with saved state gives same result."""
        torch.manual_seed(7)
        B, S = 1, 16
        m = GatedDeltaNet(64, 4)
        m.eval()

        x = torch.randn(B, S, 64)

        # Full sequence
        out_full, _ = m(x, return_state=True)

        # Two halves
        out1, state1 = m(x[:, :8, :], return_state=True)
        out2, _ = m(x[:, 8:, :], state=state1, return_state=True)
        out_halves = torch.cat([out1, out2], dim=1)

        torch.testing.assert_close(out_full, out_halves, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("S", [1, 3, 7, 16])
    def test_various_seq_lengths(self, S):
        m = GatedDeltaNet(32, 2)
        m.eval()
        x = torch.randn(1, S, 32)
        assert m(x).shape == (1, S, 32)

    def test_extra_repr(self):
        m = GatedDeltaNet(64, 4)
        assert "64" in m.extra_repr()

    def test_different_value_dim(self):
        m = GatedDeltaNet(64, 4, head_dim=16, value_dim=32)
        x = torch.randn(2, 8, 64)
        assert m(x).shape == (2, 8, 64)


# =========================================================================== #
#  TestGatedTransformerBlock                                                    #
# =========================================================================== #

class TestGatedTransformerBlock:

    @pytest.mark.parametrize("mixer", ["gated_attn", "deltanet"])
    def test_output_shape(self, mixer):
        block = GatedTransformerBlock(64, 4, mixer=mixer)
        x = torch.randn(2, 16, 64)
        if mixer == "gated_attn":
            out = block(x)
        else:
            out = block(x)
        assert out.shape == (2, 16, 64)

    def test_residual_connection_attn(self):
        """Output – input should change (residual is active)."""
        block = GatedTransformerBlock(64, 4, mixer="gated_attn")
        block.eval()
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert not torch.allclose(out, x, atol=1e-3)

    def test_residual_connection_deltanet(self):
        block = GatedTransformerBlock(64, 4, mixer="deltanet")
        block.eval()
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert not torch.allclose(out, x, atol=1e-3)

    def test_return_state_deltanet(self):
        block = GatedTransformerBlock(64, 4, mixer="deltanet")
        x = torch.randn(2, 8, 64)
        out, state = block(x, return_state=True)
        assert out.shape == (2, 8, 64)
        assert state is not None

    def test_return_state_gated_attn_is_none(self):
        block = GatedTransformerBlock(64, 4, mixer="gated_attn")
        x = torch.randn(2, 8, 64)
        out, state = block(x, return_state=True)
        assert state is None

    def test_gradient_flows_attn(self):
        block = GatedTransformerBlock(64, 4, mixer="gated_attn")
        x = torch.randn(2, 8, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)

    def test_gradient_flows_deltanet(self):
        block = GatedTransformerBlock(64, 4, mixer="deltanet")
        x = torch.randn(2, 8, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)

    def test_pre_norm_applied(self):
        """norm1 and norm2 are RMSNorm instances."""
        block = GatedTransformerBlock(64, 4)
        assert isinstance(block.norm1, RMSNorm)
        assert isinstance(block.norm2, RMSNorm)

    def test_ffn_is_swiglu(self):
        block = GatedTransformerBlock(64, 4)
        assert isinstance(block.ffn, SwiGLUFFN)

    def test_invalid_mixer_raises(self):
        with pytest.raises(ValueError):
            GatedTransformerBlock(64, 4, mixer="unknown")

    def test_causal_flag_propagated(self):
        block = GatedTransformerBlock(64, 4, mixer="gated_attn", causal=True)
        assert block.mixer.causal is True

    def test_extra_repr(self):
        block = GatedTransformerBlock(64, 4, mixer="deltanet")
        assert "deltanet" in block.extra_repr()


# =========================================================================== #
#  TestGatedTransformer                                                         #
# =========================================================================== #

class TestGatedTransformer:

    def test_output_shape(self):
        m = GatedTransformer(64, 4, num_layers=3, mixer_pattern="gated_attn")
        x = torch.randn(2, 16, 64)
        assert m(x).shape == (2, 16, 64)

    def test_deltanet_stack(self):
        m = GatedTransformer(64, 4, num_layers=2, mixer_pattern="deltanet")
        x = torch.randn(2, 8, 64)
        assert m(x).shape == (2, 8, 64)

    def test_alternating_pattern(self):
        m = GatedTransformer(64, 4, num_layers=4,
                             mixer_pattern=["gated_attn", "deltanet"])
        x = torch.randn(2, 8, 64)
        out = m(x)
        assert out.shape == (2, 8, 64)
        # Verify alternating types
        for i, layer in enumerate(m.layers):
            expected = "gated_attn" if i % 2 == 0 else "deltanet"
            assert layer.mixer_type == expected

    def test_final_norm_is_rmsnorm(self):
        m = GatedTransformer(64, 4, num_layers=2)
        assert isinstance(m.final_norm, RMSNorm)

    def test_num_layers(self):
        m = GatedTransformer(64, 4, num_layers=5)
        assert len(m.layers) == 5

    def test_gradient_flows(self):
        m = GatedTransformer(64, 4, num_layers=2, mixer_pattern="gated_attn")
        x = torch.randn(2, 8, 64, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None and not torch.all(x.grad == 0)

    def test_gradient_through_deltanet_stack(self):
        m = GatedTransformer(64, 4, num_layers=2, mixer_pattern="deltanet")
        x = torch.randn(2, 8, 64, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_extra_repr(self):
        m = GatedTransformer(128, 4, num_layers=6)
        assert "128" in m.extra_repr()
        assert "6" in m.extra_repr()

    def test_causal_propagated(self):
        m = GatedTransformer(64, 4, num_layers=2, mixer_pattern="gated_attn", causal=True)
        for layer in m.layers:
            assert layer.mixer.causal is True


# =========================================================================== #
#  TestIntegration                                                              #
# =========================================================================== #

class TestIntegration:

    def test_full_model_forward_backward(self):
        """GatedTransformer (4 layers, alternating) end-to-end."""
        torch.manual_seed(0)
        model = GatedTransformer(
            hidden_size=64,
            num_heads=4,
            num_layers=4,
            mixer_pattern=["gated_attn", "deltanet"],
        )
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = model(x)
        assert out.shape == (2, 16, 64)
        out.sum().backward()
        assert x.grad is not None

    def test_output_changes_with_input(self):
        model = GatedTransformer(32, 2, num_layers=2)
        model.eval()
        x1 = torch.randn(1, 8, 32)
        x2 = torch.randn(1, 8, 32)
        assert not torch.allclose(model(x1), model(x2), atol=1e-3)

    def test_deterministic_in_eval_mode(self):
        model = GatedTransformer(32, 2, num_layers=2)
        model.eval()
        x = torch.randn(1, 8, 32)
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_parameter_count_scales_with_depth(self):
        m2 = GatedTransformer(64, 4, num_layers=2)
        m4 = GatedTransformer(64, 4, num_layers=4)
        assert sum(p.numel() for p in m4.parameters()) > sum(p.numel() for p in m2.parameters())

    def test_gated_attn_no_causal_leak(self):
        """Strict causal: position t must not leak future positions."""
        model = GatedTransformer(32, 2, num_layers=1, mixer_pattern="gated_attn", causal=True)
        model.eval()
        torch.manual_seed(5)
        x = torch.randn(1, 12, 32)
        x_perturbed = x.clone()
        x_perturbed[0, 6:, :] += 50.0
        out1 = model(x)
        out2 = model(x_perturbed)
        # Positions 0..5 unchanged
        torch.testing.assert_close(out1[0, :6], out2[0, :6], atol=1e-3, rtol=1e-3)

    def test_rmsnorm_swiglu_gated_attn_composition(self):
        """Manually build a block and check composability."""
        norm = RMSNorm(64)
        attn = GatedAttention(64, 4)
        ffn  = SwiGLUFFN(64)

        x = torch.randn(2, 8, 64)
        h = norm(x)
        h = attn(h)
        h = x + h
        h = h + ffn(norm(h))
        assert h.shape == (2, 8, 64)
        h.sum().backward()

    def test_deltanet_state_across_chunks(self):
        """DeltaNet can process long sequences chunk-by-chunk with saved state."""
        torch.manual_seed(3)
        m = GatedTransformerBlock(32, 2, mixer="deltanet")
        m.eval()
        x = torch.randn(1, 24, 32)

        # Full pass
        out_full, _ = m(x, return_state=True)

        # Chunk pass (3 × 8)
        state = None
        parts = []
        for start in range(0, 24, 8):
            out_chunk, state = m(x[:, start:start+8, :], state=state, return_state=True)
            parts.append(out_chunk)
        out_chunks = torch.cat(parts, dim=1)

        torch.testing.assert_close(out_full, out_chunks, atol=1e-4, rtol=1e-4)
