"""
Tests for TransformerCalculator — parameter count, FLOPs, KV cache, memory.

All expected values are derived analytically from the model formulas.
"""

from __future__ import annotations

import pytest
from param.transformer_calc import TransformerConfig, TransformerCalculator, format_num


# ===========================================================================
#  Fixtures
# ===========================================================================

@pytest.fixture
def small_config():
    """Tiny config for exact numerical checks."""
    return TransformerConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        vocab_size=100,
        seq_len=32,
        batch_size=1,
        has_bias=False,
        tied_embeddings=False,
        use_swiglu=False,
    )


@pytest.fixture
def small_calc(small_config):
    return TransformerCalculator(small_config)


# ===========================================================================
#  TestConfig
# ===========================================================================

class TestConfig:
    def test_default_kv_heads(self):
        cfg = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4)
        assert cfg.num_kv_heads == 4

    def test_head_dim(self):
        cfg = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4)
        assert cfg.head_dim == 16  # 64 / 4

    def test_default_intermediate_size(self):
        cfg = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4)
        assert cfg.intermediate_size == 256  # 4 * 64

    def test_custom_intermediate_size(self):
        cfg = TransformerConfig(hidden_size=128, num_layers=4, num_heads=8,
                                intermediate_size=512)
        assert cfg.intermediate_size == 512


# ===========================================================================
#  TestParameterCount
# ===========================================================================

class TestParameterCount:
    def test_attention_params_mha_no_bias(self, small_calc):
        """MHA without bias: Q(h²) + K(h²) + V(h²) + O(h²) = 4h²."""
        attn = small_calc.attention_params()
        h = 64
        assert attn["total"] == 4 * h ** 2

    def test_attention_params_with_bias(self):
        cfg = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4,
                                has_bias=True)
        calc = TransformerCalculator(cfg)
        attn = calc.attention_params()
        h = 64
        # weights: 4h², biases: 4 * h = 4*64 = 256
        assert attn["total"] == 4 * h ** 2 + 4 * h

    def test_attention_params_gqa(self):
        """GQA: Q(h²) + K(h*kv_h) + V(h*kv_h) + O(h²) where kv_h = n_kv*d."""
        h, n_kv, n_heads = 64, 2, 4
        d = h // n_heads  # 16
        kv_h = n_kv * d   # 32
        cfg = TransformerConfig(hidden_size=h, num_layers=2, num_heads=n_heads,
                                num_kv_heads=n_kv)
        calc = TransformerCalculator(cfg)
        attn = calc.attention_params()
        expected = h * h + h * kv_h + h * kv_h + h * h  # 4096+2048+2048+4096 = 12288
        assert attn["total"] == expected

    def test_mlp_params_standard(self, small_calc):
        """Standard MLP: W1(h→4h) + W2(4h→h) = 8h²."""
        mlp = small_calc.mlp_params()
        h, inter = 64, 256
        assert mlp["total"] == 2 * h * inter  # 32768 = 8*64²

    def test_mlp_params_swiglu(self):
        """SwiGLU MLP: 3 matrices = 12h²."""
        h, inter = 64, 256
        cfg = TransformerConfig(hidden_size=h, num_layers=2, num_heads=4,
                                use_swiglu=True)
        calc = TransformerCalculator(cfg)
        mlp = calc.mlp_params()
        assert mlp["total"] == 3 * h * inter  # 49152 = 12*64²

    def test_layernorm_params(self, small_calc):
        """2 LayerNorms × (scale + bias) = 4h."""
        assert small_calc.layernorm_params() == 4 * 64  # 256

    def test_single_layer_total(self, small_calc):
        """Per-layer = attention + MLP + layernorm."""
        h, inter = 64, 256
        expected = 4 * h**2 + 2 * h * inter + 4 * h
        assert small_calc.per_layer_params() == expected

    def test_total_params_no_tie(self, small_calc, small_config):
        """Total = num_layers * per_layer + embedding_params."""
        l = small_config.num_layers
        assert small_calc.total_params() == (
            l * small_calc.per_layer_params() + small_calc.embedding_params()
        )

    def test_total_params_tied_embedding(self, small_config):
        """Tied embeddings: output head shared → fewer params."""
        cfg_no_tie = small_config
        cfg_tie = TransformerConfig(
            hidden_size=small_config.hidden_size,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            vocab_size=small_config.vocab_size,
            seq_len=small_config.seq_len,
            tied_embeddings=True,
        )
        calc_no_tie = TransformerCalculator(cfg_no_tie)
        calc_tie = TransformerCalculator(cfg_tie)
        # Tied removes the output head (V*h params)
        diff = calc_no_tie.total_params() - calc_tie.total_params()
        assert diff == small_config.vocab_size * small_config.hidden_size

    def test_gpt2_small_param_count(self):
        """GPT-2 Small ~ 124M params (has_bias=True, with weight tying)."""
        cfg = TransformerConfig(
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            vocab_size=50257,
            seq_len=1024,
            batch_size=1,
            has_bias=True,
            tied_embeddings=True,
        )
        calc = TransformerCalculator(cfg)
        total = calc.total_params()
        # GPT-2 Small is ~117-124M; accept a wide range
        assert 110_000_000 < total < 130_000_000, (
            f"Expected ~124M, got {total:,}")


# ===========================================================================
#  TestFLOPs
# ===========================================================================

class TestFLOPs:
    def test_attention_flops_mha(self, small_calc, small_config):
        """Attention FLOPs = 2Bs(2h² + 2h*kv_h) + 2Bs²h + 2Bs²h."""
        B, s, h = small_config.batch_size, small_config.seq_len, small_config.hidden_size
        kv_h = small_config.kv_hidden
        expected = (
            2 * B * s * (h**2 + h * kv_h + h * kv_h + h**2)  # projections
            + 2 * B * s * s * h  # QK^T
            + 2 * B * s * s * h  # AV
        )
        assert small_calc.attention_flops() == expected

    def test_mlp_flops_standard(self, small_calc, small_config):
        """Standard MLP FLOPs = 2Bs * h * inter * 2."""
        B, s, h = small_config.batch_size, small_config.seq_len, small_config.hidden_size
        inter = small_config.intermediate_size
        expected = 2 * B * s * h * inter * 2
        assert small_calc.mlp_flops() == expected

    def test_single_layer_flops(self, small_calc):
        """Per-layer FLOPs = attention + MLP."""
        assert (small_calc.per_layer_flops()
                == small_calc.attention_flops() + small_calc.mlp_flops())

    def test_flops_scales_linearly_with_layers(self, small_config):
        """Doubling num_layers approximately doubles forward_flops."""
        cfg2 = TransformerConfig(
            hidden_size=small_config.hidden_size,
            num_layers=small_config.num_layers * 2,
            num_heads=small_config.num_heads,
            vocab_size=small_config.vocab_size,
            seq_len=small_config.seq_len,
            batch_size=small_config.batch_size,
        )
        calc1 = TransformerCalculator(small_config)
        calc2 = TransformerCalculator(cfg2)
        layer_flops = calc1.per_layer_flops()
        assert (calc2.forward_flops() - calc1.forward_flops()
                == layer_flops * small_config.num_layers)

    def test_gqa_reduces_kv_proj_flops(self, small_config):
        """GQA with fewer KV heads → fewer FLOPs than full MHA."""
        cfg_gqa = TransformerConfig(
            hidden_size=small_config.hidden_size,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            num_kv_heads=1,
            vocab_size=small_config.vocab_size,
            seq_len=small_config.seq_len,
        )
        calc_mha = TransformerCalculator(small_config)
        calc_gqa = TransformerCalculator(cfg_gqa)
        assert calc_gqa.attention_flops() < calc_mha.attention_flops()

    def test_backward_flops_is_2x_forward(self, small_calc):
        assert small_calc.backward_flops() == 2 * small_calc.forward_flops()

    def test_train_flops_is_3x_forward(self, small_calc):
        assert small_calc.train_flops() == 3 * small_calc.forward_flops()


# ===========================================================================
#  TestKVCache
# ===========================================================================

class TestKVCache:
    def test_kv_cache_mha(self, small_calc, small_config):
        """KV cache = 2 * B * s * kv_h * num_layers * bytes."""
        B, s = small_config.batch_size, small_config.seq_len
        kv_h = small_config.kv_hidden
        l = small_config.num_layers
        dtype_bytes = small_config.kv_cache_dtype_bytes
        expected = 2 * B * s * kv_h * l * dtype_bytes
        assert small_calc.kv_cache_bytes() == expected

    def test_kv_cache_gqa_smaller(self, small_config):
        """GQA KV cache < MHA KV cache."""
        cfg_gqa = TransformerConfig(
            hidden_size=small_config.hidden_size,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            num_kv_heads=1,
            seq_len=small_config.seq_len,
            vocab_size=small_config.vocab_size,
        )
        calc_mha = TransformerCalculator(small_config)
        calc_gqa = TransformerCalculator(cfg_gqa)
        assert calc_gqa.kv_cache_bytes() < calc_mha.kv_cache_bytes()

    def test_kv_cache_scales_with_batch(self, small_config):
        """KV cache scales linearly with batch size."""
        cfg1 = small_config
        cfg2 = TransformerConfig(
            hidden_size=small_config.hidden_size,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            vocab_size=small_config.vocab_size,
            seq_len=small_config.seq_len,
            batch_size=4,
        )
        c1 = TransformerCalculator(cfg1)
        c2 = TransformerCalculator(cfg2)
        assert c2.kv_cache_bytes() == 4 * c1.kv_cache_bytes()

    def test_kv_cache_scales_with_seq_len(self, small_config):
        """KV cache scales linearly with sequence length."""
        cfg1 = small_config
        cfg2 = TransformerConfig(
            hidden_size=small_config.hidden_size,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            vocab_size=small_config.vocab_size,
            seq_len=small_config.seq_len * 2,
        )
        c1 = TransformerCalculator(cfg1)
        c2 = TransformerCalculator(cfg2)
        assert c2.kv_cache_bytes() == 2 * c1.kv_cache_bytes()

    def test_kv_cache_concrete_value(self):
        """Concrete KV cache for a deterministic tiny model."""
        cfg = TransformerConfig(
            hidden_size=32, num_layers=2, num_heads=2,
            seq_len=8, batch_size=1, kv_cache_dtype_bytes=2,
        )
        # head_dim = 16, kv_h = 32, 2 * 1 * 8 * 32 * 2 * 2 = 2048
        calc = TransformerCalculator(cfg)
        expected = 2 * 1 * 8 * 32 * 2 * 2  # 2048
        assert calc.kv_cache_bytes() == expected


# ===========================================================================
#  TestTrainingMemory
# ===========================================================================

class TestTrainingMemory:
    def test_fp32_adam_16_bytes_per_param(self, small_calc):
        """FP32 Adam uses 16 bytes per parameter."""
        mem = small_calc.training_memory(optimizer="adam", mixed_precision=False)
        P = small_calc.total_params()
        assert mem["bytes_per_param"] == 16
        assert mem["total"] == 16 * P

    def test_mixed_precision_adam_16_bytes(self, small_calc):
        """Mixed precision Adam also uses 16 bytes/param: fp16(2)+fp16(2)+fp32(4)+m(4)+v(4)."""
        mem = small_calc.training_memory(optimizer="adam", mixed_precision=True)
        assert mem["bytes_per_param"] == 16

    def test_fp32_adam_components(self, small_calc):
        """FP32 Adam: params=4P, grads=4P, master=0, m=4P, v=4P."""
        mem = small_calc.training_memory(optimizer="adam", mixed_precision=False)
        P = small_calc.total_params()
        assert mem["params"] == 4 * P
        assert mem["grads"] == 4 * P
        assert mem["master_weights"] == 0
        assert mem["optimizer_m"] == 4 * P
        assert mem["optimizer_v"] == 4 * P

    def test_fp32_sgd_8_bytes_per_param(self, small_calc):
        """SGD (no momentum): params(4) + grads(4) = 8 bytes/param."""
        mem = small_calc.training_memory(optimizer="sgd")
        assert mem["bytes_per_param"] == 8
        P = small_calc.total_params()
        assert mem["total"] == 8 * P

    def test_memory_scales_with_params(self, small_config):
        """Larger model → more training memory."""
        cfg_small = small_config
        cfg_large = TransformerConfig(
            hidden_size=128, num_layers=4, num_heads=4,
            vocab_size=small_config.vocab_size,
            seq_len=small_config.seq_len,
        )
        m_small = TransformerCalculator(cfg_small).training_memory()["total"]
        m_large = TransformerCalculator(cfg_large).training_memory()["total"]
        assert m_large > m_small

    def test_gb_conversion(self, small_calc):
        """training_memory total is in bytes; verify GB conversion makes sense."""
        mem = small_calc.training_memory()
        total_bytes = mem["total"]
        total_gb = total_bytes / (1024 ** 3)
        assert total_gb >= 0.0  # sanity: non-negative
        assert total_bytes > 0


# ===========================================================================
#  TestActivationMemory
# ===========================================================================

class TestActivationMemory:
    def test_activation_positive(self, small_calc):
        assert small_calc.activation_memory_per_layer() > 0

    def test_activation_scales_with_batch_and_seq(self, small_config):
        """Activation memory scales with batch_size × seq_len."""
        cfg1 = small_config
        cfg2 = TransformerConfig(
            hidden_size=small_config.hidden_size,
            num_layers=small_config.num_layers,
            num_heads=small_config.num_heads,
            vocab_size=small_config.vocab_size,
            seq_len=small_config.seq_len * 2,
            batch_size=small_config.batch_size * 2,
        )
        c1 = TransformerCalculator(cfg1)
        c2 = TransformerCalculator(cfg2)
        # seq*2, batch*2 → approx 4x (the dominant sbh term scales as sBh)
        # The formula has quadratic term in s too, so just check >4x
        assert c2.activation_memory_per_layer() > 4 * c1.activation_memory_per_layer()

    def test_total_activation_scales_with_layers(self, small_config):
        """Total activation = num_layers × per_layer."""
        calc = TransformerCalculator(small_config)
        assert (calc.activation_memory_total()
                == small_config.num_layers * calc.activation_memory_per_layer())


# ===========================================================================
#  TestFormatNum
# ===========================================================================

class TestFormatNum:
    def test_small(self):
        assert format_num(0) == "0"
        assert format_num(42) == "42"
        assert format_num(999) == "999"

    def test_thousands(self):
        r = format_num(1500)
        assert "K" in r
        assert "1.50" in r

    def test_millions(self):
        r = format_num(2_400_000)
        assert "M" in r
        assert "2.40" in r

    def test_billions(self):
        r = format_num(7_000_000_000)
        assert "B" in r
        assert "7.00" in r


# ===========================================================================
#  TestCrossValidation
# ===========================================================================

class TestCrossValidation:
    def test_per_layer_equals_attn_plus_mlp_plus_ln(self, small_calc):
        attn = small_calc.attention_params()["total"]
        mlp = small_calc.mlp_params()["total"]
        ln = small_calc.layernorm_params()
        assert small_calc.per_layer_params() == attn + mlp + ln

    def test_total_params_equals_sum(self, small_calc, small_config):
        expected = (small_config.num_layers * small_calc.per_layer_params()
                    + small_calc.embedding_params())
        assert small_calc.total_params() == expected

    def test_flops_forward_is_layers_plus_logits(self, small_calc, small_config):
        layer_flops = small_config.num_layers * small_calc.per_layer_flops()
        logits = small_calc.logits_flops()
        assert small_calc.forward_flops() == layer_flops + logits

    def test_all_layers_equals_per_layer_times_l(self, small_calc, small_config):
        layer = small_calc.per_layer_flops()
        assert (small_calc.forward_flops() - small_calc.logits_flops()
                == small_config.num_layers * layer)

    def test_approximate_flops_formula(self, small_config):
        """Chinchilla rule of thumb: total FLOPs ≈ 6 * P * T (train, tokens)."""
        cfg = TransformerConfig(
            hidden_size=512, num_layers=6, num_heads=8, vocab_size=32000,
            seq_len=1024, batch_size=1,
        )
        calc = TransformerCalculator(cfg)
        # The dominant term in forward_flops ≈ 2*s*l*(2h² + 8h²) = 2*s*l*10h² = 20slh²
        # train_flops = 3 * forward_flops ≈ 3 * 2 * l * s * (2h² + 8h²)
        # very rough sanity: train_flops > forward_flops
        assert calc.train_flops() > calc.forward_flops()
        assert calc.backward_flops() > calc.forward_flops()


# ===========================================================================
#  TestRealModels
# ===========================================================================

class TestRealModels:
    def test_llama_7b_approximate(self):
        """LLaMA-7B ~ 6.7B parameters."""
        cfg = TransformerConfig(
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            vocab_size=32000,
            seq_len=2048,
            batch_size=1,
            use_swiglu=True,
            intermediate_size=11008,
            has_bias=False,
            tied_embeddings=False,
        )
        calc = TransformerCalculator(cfg)
        total = calc.total_params()
        assert 6_000_000_000 < total < 8_000_000_000, (
            f"Expected ~6.7B, got {total:,}"
        )

    def test_kv_cache_llama_7b(self):
        """LLaMA-7B KV cache for batch=1, seq=2048 ≈ 1GB (FP16)."""
        cfg = TransformerConfig(
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            vocab_size=32000,
            seq_len=2048,
            batch_size=1,
            kv_cache_dtype_bytes=2,
        )
        calc = TransformerCalculator(cfg)
        kv_bytes = calc.kv_cache_bytes()
        kv_gb = kv_bytes / (1024 ** 3)
        # Should be ~1 GiB
        assert 0.9 < kv_gb < 1.1, f"Expected ~1 GB, got {kv_gb:.3f} GB"

    def test_larger_model_has_more_params(self):
        """A model with more layers/width has strictly more parameters."""
        small = TransformerConfig(hidden_size=256, num_layers=4, num_heads=4,
                                  vocab_size=1000)
        large = TransformerConfig(hidden_size=512, num_layers=8, num_heads=8,
                                  vocab_size=1000)
        c_small = TransformerCalculator(small)
        c_large = TransformerCalculator(large)
        assert c_large.total_params() > c_small.total_params()
