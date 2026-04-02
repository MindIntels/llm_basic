"""Transformer 参数量 / FLOPs / KV Cache / 显存 计算器 — 测试套件

测试策略:
  1. 基于已知公式验证计算结果（手动推导）
  2. 交叉验证不同方法得到的相同结果
  3. 边界值和特殊配置测试
  4. 与已知模型对比（GPT-2, LLaMA 等公开参数量）
"""

import pytest
from transformer_calc import TransformerConfig, TransformerCalculator, format_num


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_config():
    """小型配置，方便手算验证。h=64, l=2, n_h=4, V=100"""
    return TransformerConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        vocab_size=100,
        seq_len=32,
        batch_size=1,
        has_bias=False,
    )


@pytest.fixture
def small_config_with_bias():
    """带 bias 的小型配置。"""
    return TransformerConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        vocab_size=100,
        seq_len=32,
        batch_size=1,
        has_bias=True,
    )


@pytest.fixture
def gpt2_small_config():
    """GPT-2 Small 配置: h=768, l=12, n_h=12, V=50257"""
    return TransformerConfig(
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        vocab_size=50257,
        seq_len=1024,
        batch_size=1,
        has_bias=True,
    )


@pytest.fixture
def gqa_config():
    """GQA 配置: h=128, n_h=8, n_kv=2"""
    return TransformerConfig(
        hidden_size=128,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        vocab_size=100,
        seq_len=64,
        batch_size=1,
    )


@pytest.fixture
def swiglu_config():
    """SwiGLU MLP 配置。"""
    return TransformerConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        vocab_size=100,
        seq_len=32,
        batch_size=1,
        use_swiglu=True,
    )


# =============================================================================
# 1. 参数量测试
# =============================================================================

class TestParameterCount:
    """验证参数量计算的正确性。"""

    def test_attention_params_mha_no_bias(self, small_config):
        """标准 MHA, 无 bias: Q(h²) + K(h²) + V(h²) + O(h²) = 4h²"""
        calc = TransformerCalculator(small_config)
        attn = calc.attention_params()

        h = 64
        # Q投影: h×h = 64×64 = 4096
        assert attn["q_proj"] == h * h
        # K投影: h×h = 4096 (MHA, kv_dim = h)
        assert attn["k_proj"] == h * h
        # V投影: h×h = 4096
        assert attn["v_proj"] == h * h
        # O投影: h×h = 4096
        assert attn["o_proj"] == h * h
        # 权重总计: 4h² = 16384
        assert attn["weight_total"] == 4 * h * h
        # 无 bias
        assert attn["bias_total"] == 0
        # 总计 = 4h²
        assert attn["total"] == 4 * h * h

    def test_attention_params_with_bias(self, small_config_with_bias):
        """带 bias 的 MHA: 权重 4h² + bias 4h"""
        calc = TransformerCalculator(small_config_with_bias)
        attn = calc.attention_params()
        h = 64
        assert attn["weight_total"] == 4 * h * h
        assert attn["bias_total"] == 4 * h  # Q(h) + K(h) + V(h) + O(h)
        assert attn["total"] == 4 * h * h + 4 * h

    def test_attention_params_gqa(self, gqa_config):
        """GQA: K/V 投影更小。h=128, n_h=8, n_kv=2, d=16, kv_dim=32"""
        calc = TransformerCalculator(gqa_config)
        attn = calc.attention_params()
        h = 128
        kv_dim = 2 * 16  # n_kv × d_head = 2 × (128/8) = 32

        # Q投影: h×h = 128×128 = 16384
        assert attn["q_proj"] == h * h
        # K投影: h×kv_dim = 128×32 = 4096
        assert attn["k_proj"] == h * kv_dim
        # V投影: h×kv_dim = 4096
        assert attn["v_proj"] == h * kv_dim
        # O投影: h×h = 16384
        assert attn["o_proj"] == h * h

        # GQA 总参数 < MHA 总参数
        assert attn["total"] == h * h + h * kv_dim + h * kv_dim + h * h
        assert attn["total"] < 4 * h * h  # 比 MHA 少

    def test_mlp_params_standard(self, small_config):
        """标准 MLP (GeLU): W1(h×4h) + W2(4h×h) = 8h²"""
        calc = TransformerCalculator(small_config)
        mlp = calc.mlp_params()
        h = 64
        inter = 4 * h  # 256

        assert mlp["weight_total"] == 2 * h * inter  # h×inter + inter×h = 2×h×4h = 8h²
        assert mlp["weight_total"] == 8 * h * h
        assert mlp["bias_total"] == 0
        assert mlp["total"] == 8 * h * h

    def test_mlp_params_swiglu(self, swiglu_config):
        """SwiGLU MLP: gate(h×4h) + up(h×4h) + down(4h×h) = 12h²"""
        calc = TransformerCalculator(swiglu_config)
        mlp = calc.mlp_params()
        h = 64
        inter = 4 * h

        # SwiGLU: 3 个矩阵 → 3×h×inter = 3×64×256 = 49152
        assert mlp["weight_total"] == 3 * h * inter
        assert mlp["total"] == 3 * h * inter

    def test_layernorm_params(self, small_config):
        """每层 2 个 LayerNorm, 每个有 gamma(h) + beta(h) = 2h, 总计 4h"""
        calc = TransformerCalculator(small_config)
        ln = calc.layernorm_params()
        assert ln == 4 * 64  # 2 × (2 × 64) = 256

    def test_single_layer_total(self, small_config):
        """单层总参数 = Attention + MLP + LayerNorm = 4h² + 8h² + 4h = 12h² + 4h"""
        calc = TransformerCalculator(small_config)
        layer = calc.single_layer_params()
        h = 64
        expected = 12 * h * h + 4 * h  # 12×4096 + 256 = 49408
        assert layer["total"] == expected

    def test_total_params_no_tie(self, small_config):
        """完整模型（不共享 embedding）:
        Embedding(V×h) + l×layer + final_LN(2h) + lm_head(h×V)
        """
        calc = TransformerCalculator(small_config)
        total = calc.total_params()
        h, l, V = 64, 2, 100

        embedding = V * h                     # 6400
        per_layer = 12 * h * h + 4 * h        # 49408
        all_layers = l * per_layer             # 98816
        final_ln = 2 * h                       # 128
        output_head = h * V                    # 6400

        expected = embedding + all_layers + final_ln + output_head
        assert total["total"] == expected

    def test_total_params_tied_embedding(self):
        """共享 embedding 时, output_head = 0"""
        cfg = TransformerConfig(
            hidden_size=64, num_layers=2, num_heads=4,
            vocab_size=100, tied_embeddings=True,
        )
        calc = TransformerCalculator(cfg)
        total = calc.total_params()

        assert total["output_head"] == 0
        assert total["total"] == total["embedding"] + total["all_layers"] + total["final_layernorm"]

    def test_gpt2_small_param_count(self, gpt2_small_config):
        """GPT-2 Small 约 124M 参数（含 bias）。

        实际 GPT-2 Small 有 124,439,808 参数。
        关键：GPT-2 使用 tied embedding (output head 共享 embedding 权重)
        且有 learned position embedding (1024×768) 我们未计入。
        """
        # GPT-2 使用 tied embeddings
        gpt2_small_config.tied_embeddings = True
        calc = TransformerCalculator(gpt2_small_config)
        total = calc.total_params()["total"]

        # 实际 GPT-2: 124,439,808 (含 position embedding 1024×768=786,432)
        # 我们的计算不含 position embedding，所以会偏少约 0.8M
        # 允许 5% 误差范围
        assert 118_000_000 < total < 130_000_000, \
            f"GPT-2 Small should be ~124M params, got {total:,}"


# =============================================================================
# 2. FLOPs 测试
# =============================================================================

class TestFLOPs:
    """验证 FLOPs 计算的正确性。"""

    def test_attention_flops_mha(self, small_config):
        """标准 MHA Attention FLOPs:
        QKV投影: 6sh² + Attention计算: 4s²h + O投影: 2sh² = 8sh² + 4s²h
        """
        calc = TransformerCalculator(small_config)
        attn_flops = calc.attention_flops_per_layer()
        h, s = 64, 32

        # QKV 投影: 3 × 2sh² = 6sh²
        expected_qkv = 6 * s * h * h
        assert attn_flops["qkv_proj_total"] == expected_qkv

        # Attention 计算: QK^T (2s²h) + Attn×V (2s²h) = 4s²h
        expected_attn_compute = 4 * s * s * h
        assert attn_flops["attn_compute_total"] == expected_attn_compute

        # O 投影: 2sh²
        expected_o = 2 * s * h * h
        assert attn_flops["o_proj"] == expected_o

        # 总计: 8sh² + 4s²h
        expected_total = 8 * s * h * h + 4 * s * s * h
        assert attn_flops["total"] == expected_total

    def test_mlp_flops_standard(self, small_config):
        """标准 MLP FLOPs: W1(8sh²) + W2(8sh²) = 16sh²"""
        calc = TransformerCalculator(small_config)
        mlp_flops = calc.mlp_flops_per_layer()
        h, s = 64, 32
        inter = 4 * h

        # W_up: 2×s×h×inter + W_down: 2×s×inter×h
        expected = 2 * s * h * inter + 2 * s * inter * h
        assert mlp_flops["total"] == expected
        assert mlp_flops["total"] == 16 * s * h * h  # inter=4h → 16sh²

    def test_single_layer_flops(self, small_config):
        """单层 FLOPs = 24sh² + 4s²h (标准 MHA + 标准 MLP)"""
        calc = TransformerCalculator(small_config)
        layer_flops = calc.single_layer_flops()
        h, s = 64, 32

        expected = 24 * s * h * h + 4 * s * s * h
        assert layer_flops["total"] == expected

    def test_train_flops_is_3x_forward(self, small_config):
        """训练 FLOPs = 3 × 前向 FLOPs"""
        calc = TransformerCalculator(small_config)
        flops = calc.total_flops()

        assert flops["train_total"] == 3 * flops["forward"]

    def test_backward_flops_is_2x_forward(self, small_config):
        """反向 FLOPs ≈ 2 × 前向 FLOPs"""
        calc = TransformerCalculator(small_config)
        flops = calc.total_flops()

        assert flops["backward"] == 2 * flops["forward"]

    def test_flops_scales_linearly_with_layers(self):
        """FLOPs 应该与层数成正比（忽略 logits 部分）"""
        cfg1 = TransformerConfig(hidden_size=64, num_layers=4, num_heads=4, vocab_size=100, seq_len=32)
        cfg2 = TransformerConfig(hidden_size=64, num_layers=8, num_heads=4, vocab_size=100, seq_len=32)
        calc1 = TransformerCalculator(cfg1)
        calc2 = TransformerCalculator(cfg2)

        # 所有层的 FLOPs 应该是 2:1 比例
        f1 = calc1.total_flops()["all_layers_forward"]
        f2 = calc2.total_flops()["all_layers_forward"]
        assert f2 == 2 * f1

    def test_gqa_reduces_kv_proj_flops(self, gqa_config):
        """GQA 的 K/V 投影 FLOPs 应该小于 MHA"""
        calc_gqa = TransformerCalculator(gqa_config)

        # 创建等效 MHA 配置
        mha_config = TransformerConfig(
            hidden_size=128, num_layers=4, num_heads=8,
            num_kv_heads=8,  # MHA
            vocab_size=100, seq_len=64,
        )
        calc_mha = TransformerCalculator(mha_config)

        gqa_attn = calc_gqa.attention_flops_per_layer()
        mha_attn = calc_mha.attention_flops_per_layer()

        # K/V 投影 FLOPs: GQA < MHA
        assert gqa_attn["k_proj"] < mha_attn["k_proj"]
        assert gqa_attn["v_proj"] < mha_attn["v_proj"]

        # Q 和 O 投影不变
        assert gqa_attn["q_proj"] == mha_attn["q_proj"]
        assert gqa_attn["o_proj"] == mha_attn["o_proj"]

        # Attention score/multiply 不变 (仍然是 n_h 个 Q 头)
        assert gqa_attn["attn_score"] == mha_attn["attn_score"]


# =============================================================================
# 3. KV Cache 测试
# =============================================================================

class TestKVCache:
    """验证 KV Cache 显存计算。"""

    def test_kv_cache_mha(self, small_config):
        """标准 MHA KV Cache: 2 × B × l × s × h × dtype_bytes"""
        calc = TransformerCalculator(small_config)
        kv = calc.kv_cache_memory()

        B, l, s, h = 1, 2, 32, 64
        n_kv = 4  # MHA: n_kv = n_h
        d_head = 16  # h / n_h = 64 / 4

        # 单层: 2 × B × n_kv × s × d_head × 2 bytes
        per_layer_expected = 2 * B * n_kv * s * d_head * 2
        assert kv["per_layer_bytes"] == per_layer_expected

        # 注意 n_kv × d_head = h, 所以 per_layer = 2 × B × s × h × 2
        assert per_layer_expected == 2 * B * s * h * 2

        # 总计: l × per_layer
        assert kv["total_bytes"] == l * per_layer_expected

    def test_kv_cache_gqa_smaller(self, gqa_config):
        """GQA 的 KV Cache 应该小于 MHA"""
        calc_gqa = TransformerCalculator(gqa_config)

        mha_config = TransformerConfig(
            hidden_size=128, num_layers=4, num_heads=8,
            num_kv_heads=8, vocab_size=100, seq_len=64,
        )
        calc_mha = TransformerCalculator(mha_config)

        # GQA (n_kv=2) 的 KV Cache 应该是 MHA (n_kv=8) 的 1/4
        gqa_kv = calc_gqa.kv_cache_memory()["total_bytes"]
        mha_kv = calc_mha.kv_cache_memory()["total_bytes"]

        assert gqa_kv == mha_kv // 4

    def test_kv_cache_scales_with_seq_len(self):
        """KV Cache 与序列长度成正比"""
        cfg1 = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4, seq_len=32)
        cfg2 = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4, seq_len=64)

        kv1 = TransformerCalculator(cfg1).kv_cache_memory()["total_bytes"]
        kv2 = TransformerCalculator(cfg2).kv_cache_memory()["total_bytes"]

        assert kv2 == 2 * kv1

    def test_kv_cache_scales_with_batch(self):
        """KV Cache 与 batch size 成正比"""
        cfg1 = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4, batch_size=1)
        cfg2 = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4, batch_size=4)

        kv1 = TransformerCalculator(cfg1).kv_cache_memory()["total_bytes"]
        kv2 = TransformerCalculator(cfg2).kv_cache_memory()["total_bytes"]

        assert kv2 == 4 * kv1

    def test_kv_cache_concrete_value(self):
        """具体数值验证: h=128, l=1, s=10, B=1, FP16"""
        cfg = TransformerConfig(
            hidden_size=128, num_layers=1, num_heads=8,
            seq_len=10, batch_size=1, kv_cache_dtype_bytes=2,
        )
        calc = TransformerCalculator(cfg)
        kv = calc.kv_cache_memory()

        # 2(KV) × 1(B) × 8(n_kv=n_h) × 10(s) × 16(d_head) × 2(bytes)
        # = 2 × 1 × 8 × 10 × 16 × 2 = 5120 bytes
        assert kv["total_bytes"] == 5120


# =============================================================================
# 4. 训练显存测试
# =============================================================================

class TestTrainingMemory:
    """验证训练显存计算。"""

    def test_fp32_adam_16_bytes_per_param(self, small_config):
        """FP32 + Adam: 16 bytes per parameter"""
        calc = TransformerCalculator(small_config)
        mem = calc.training_memory(optimizer="adam", mixed_precision=False)

        assert mem["bytes_per_param"] == 16
        assert mem["total_bytes"] == 16 * mem["num_params"]

    def test_fp32_adam_components(self, small_config):
        """FP32 + Adam: 模型(4) + 梯度(4) + m(4) + v(4) = 16"""
        calc = TransformerCalculator(small_config)
        mem = calc.training_memory(optimizer="adam", mixed_precision=False)
        P = mem["num_params"]

        assert mem["model_fp32"] == 4 * P
        assert mem["grad_fp32"] == 4 * P
        assert mem["adam_m"] == 4 * P
        assert mem["adam_v"] == 4 * P

    def test_mixed_precision_adam_16_bytes(self, small_config):
        """Mixed Precision + Adam: 16 bytes per parameter
        FP16模型(2) + FP16梯度(2) + FP32 master(4) + m(4) + v(4) = 16
        """
        calc = TransformerCalculator(small_config)
        mem = calc.training_memory(optimizer="adam", mixed_precision=True)

        assert mem["bytes_per_param"] == 16
        P = mem["num_params"]
        expected = 2 * P + 2 * P + 4 * P + 4 * P + 4 * P  # 16P
        assert mem["total_bytes"] == expected

    def test_fp32_sgd_8_bytes_per_param(self, small_config):
        """FP32 + SGD (无 momentum): 8 bytes per parameter"""
        calc = TransformerCalculator(small_config)
        mem = calc.training_memory(optimizer="sgd", mixed_precision=False)

        assert mem["bytes_per_param"] == 8
        assert mem["total_bytes"] == 8 * mem["num_params"]

    def test_memory_scales_with_params(self):
        """显存应该与参数量成正比"""
        cfg1 = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4, vocab_size=100)
        cfg2 = TransformerConfig(hidden_size=64, num_layers=4, num_heads=4, vocab_size=100)

        mem1 = TransformerCalculator(cfg1).training_memory()
        mem2 = TransformerCalculator(cfg2).training_memory()

        # 更多层 → 更多参数 → 更多显存
        assert mem2["total_bytes"] > mem1["total_bytes"]

        # 比例应该接近参数量比例
        ratio_params = mem2["num_params"] / mem1["num_params"]
        ratio_mem = mem2["total_bytes"] / mem1["total_bytes"]
        assert abs(ratio_params - ratio_mem) < 0.01

    def test_gb_conversion(self, small_config):
        """验证 GB 换算正确"""
        calc = TransformerCalculator(small_config)
        mem = calc.training_memory()

        expected_gb = mem["total_bytes"] / (1024 ** 3)
        assert abs(mem["total_gb"] - expected_gb) < 1e-10


# =============================================================================
# 5. 中间激活测试
# =============================================================================

class TestActivationMemory:
    """验证中间激活显存计算。"""

    def test_activation_positive(self, small_config):
        """激活显存应为正数"""
        calc = TransformerCalculator(small_config)
        act = calc.activation_memory_per_layer()

        assert act["attention_bytes"] > 0
        assert act["mlp_bytes"] > 0
        assert act["total_bytes"] > 0

    def test_activation_scales_with_batch_and_seq(self):
        """激活显存与 batch_size × seq_len 成正比"""
        cfg1 = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4,
                                 seq_len=32, batch_size=1)
        cfg2 = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4,
                                 seq_len=32, batch_size=2)
        cfg3 = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4,
                                 seq_len=64, batch_size=1)

        act1 = TransformerCalculator(cfg1).activation_memory_per_layer()["total_bytes"]
        act2 = TransformerCalculator(cfg2).activation_memory_per_layer()["total_bytes"]
        act3 = TransformerCalculator(cfg3).activation_memory_per_layer()["total_bytes"]

        # 2× batch → 2× activation (近似, attention scores 的 s² 项会影响)
        assert act2 == 2 * act1

        # 2× seq_len → > 2× activation (因为 attention scores 是 s² 的)
        assert act3 > 2 * act1

    def test_total_activation_scales_with_layers(self, small_config):
        """总激活显存与层数成正比"""
        calc = TransformerCalculator(small_config)
        per_layer = calc.activation_memory_per_layer()["total_bytes"]
        total = calc.total_activation_memory()["total_bytes"]

        assert total == small_config.num_layers * per_layer


# =============================================================================
# 6. 配置和边界测试
# =============================================================================

class TestConfig:
    """测试配置的默认值和属性。"""

    def test_default_kv_heads(self):
        """默认 num_kv_heads = num_heads (MHA)"""
        cfg = TransformerConfig(num_heads=8)
        assert cfg.num_kv_heads == 8

    def test_default_intermediate_size(self):
        """默认 intermediate_size = 4 × hidden_size"""
        cfg = TransformerConfig(hidden_size=128)
        assert cfg.intermediate_size == 512

    def test_custom_intermediate_size(self):
        """自定义 intermediate_size"""
        cfg = TransformerConfig(hidden_size=128, intermediate_size=300)
        assert cfg.intermediate_size == 300

    def test_head_dim(self):
        """head_dim = hidden_size / num_heads"""
        cfg = TransformerConfig(hidden_size=128, num_heads=8)
        assert cfg.head_dim == 16


class TestFormatNum:
    """测试数字格式化工具。"""

    def test_billions(self):
        assert format_num(7e9) == "7.00B"

    def test_millions(self):
        assert format_num(124e6) == "124.00M"

    def test_thousands(self):
        assert format_num(50e3) == "50.00K"

    def test_small(self):
        assert format_num(42) == "42"


# =============================================================================
# 7. 交叉验证与一致性测试
# =============================================================================

class TestCrossValidation:
    """交叉验证：确保不同计算路径得到一致结果。"""

    def test_total_params_equals_sum(self, small_config):
        """总参数 = embedding + all_layers + final_ln + output_head"""
        calc = TransformerCalculator(small_config)
        total = calc.total_params()

        assert total["total"] == (
            total["embedding"] +
            total["all_layers"] +
            total["final_layernorm"] +
            total["output_head"]
        )

    def test_all_layers_equals_per_layer_times_l(self, small_config):
        """all_layers = per_layer × num_layers"""
        calc = TransformerCalculator(small_config)
        total = calc.total_params()

        assert total["all_layers"] == total["per_layer"] * small_config.num_layers

    def test_per_layer_equals_attn_plus_mlp_plus_ln(self, small_config):
        """per_layer = attention + mlp + layernorm"""
        calc = TransformerCalculator(small_config)
        layer = calc.single_layer_params()

        assert layer["total"] == layer["attention"] + layer["mlp"] + layer["layernorm"]

    def test_flops_forward_is_layers_plus_logits(self, small_config):
        """forward FLOPs = all_layers + logits"""
        calc = TransformerCalculator(small_config)
        flops = calc.total_flops()

        assert flops["forward"] == flops["all_layers_forward"] + flops["logits_forward"]

    def test_approximate_flops_formula(self):
        """近似公式验证: FLOPs_fwd ≈ 2Ps (忽略 attention s² 项和 logits)

        对于 h 较大、s 较小的情况，24sh² 主导 4s²h，
        而 24sh² = 2s × 12h² ≈ 2s × P_per_layer
        """
        cfg = TransformerConfig(
            hidden_size=1024, num_layers=24, num_heads=16,
            vocab_size=100, seq_len=64,
        )
        calc = TransformerCalculator(cfg)

        # 精确的 all_layers forward flops
        exact = calc.total_flops()["all_layers_forward"]

        # 近似: 2 × total_layer_params × seq_len
        # 每层参数约 12h² (忽略 bias/ln)
        approx_per_layer = 12 * 1024 * 1024
        approx_total = 2 * 24 * approx_per_layer * 64

        # 误差应该在 30% 以内 (s² 项的影响)
        ratio = exact / approx_total
        assert 0.7 < ratio < 1.3, f"Exact/Approx ratio = {ratio:.3f}"


# =============================================================================
# 8. 真实模型对比测试
# =============================================================================

class TestRealModels:
    """与已知模型参数量进行对比。"""

    def test_llama_7b_approximate(self):
        """LLaMA-7B: h=4096, l=32, n_h=32, V=32000
        实际约 6.7B 参数 (不共享 embedding)
        """
        cfg = TransformerConfig(
            hidden_size=4096, num_layers=32, num_heads=32,
            vocab_size=32000, seq_len=2048,
            has_bias=False, use_swiglu=True,
            intermediate_size=11008,  # LLaMA 使用 2/3 * 4h 的 SwiGLU
        )
        calc = TransformerCalculator(cfg)
        total = calc.total_params()["total"]

        # LLaMA-7B 实际约 6.7B (因为 SwiGLU intermediate=11008 而非 16384)
        assert 6_000_000_000 < total < 7_500_000_000, \
            f"LLaMA-7B should be ~6.7B params, got {total/1e9:.2f}B"

    def test_larger_model_has_more_params(self):
        """更大的模型应该有更多参数"""
        cfg_small = TransformerConfig(hidden_size=768, num_layers=12, num_heads=12)
        cfg_large = TransformerConfig(hidden_size=1024, num_layers=24, num_heads=16)

        p_small = TransformerCalculator(cfg_small).total_params()["total"]
        p_large = TransformerCalculator(cfg_large).total_params()["total"]

        assert p_large > p_small

    def test_kv_cache_llama_7b(self):
        """LLaMA-7B 在 s=2048, B=1 时的 KV Cache 估算"""
        cfg = TransformerConfig(
            hidden_size=4096, num_layers=32, num_heads=32,
            vocab_size=32000, seq_len=2048, batch_size=1,
            kv_cache_dtype_bytes=2,
        )
        calc = TransformerCalculator(cfg)
        kv = calc.kv_cache_memory()

        # 2 × 1 × 32 × 2048 × 4096 × 2 = 1,073,741,824 bytes = 1 GB
        expected = 2 * 1 * 32 * 2048 * 4096 * 2
        assert kv["total_bytes"] == expected
        assert abs(kv["total_gb"] - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
