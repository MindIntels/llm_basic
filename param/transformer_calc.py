"""Transformer 模型参数量 / FLOPs / KV Cache / 显存 计算器

基于知乎文章「分析transformer模型的参数量、计算量、中间激活、KV cache」的完整实现。
以标准 Transformer Decoder 层为例，逐步计算每个组件的参数量、计算量和显存占用。

References:
    - https://zhuanlan.zhihu.com/p/624740065
    - Kaplan et al., "Scaling Laws for Neural Language Models"
    - Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models"
"""

from __future__ import annotations

from dataclasses import dataclass, field


# =============================================================================
# 配置
# =============================================================================

@dataclass
class TransformerConfig:
    """Transformer 模型配置参数。

    所有计算基于标准 GPT-style Decoder-Only Transformer：
    - Pre-LayerNorm
    - Multi-Head Self-Attention (MHA) 或 Grouped-Query Attention (GQA)
    - MLP: h -> intermediate_size -> h (默认 intermediate_size = 4h)
    - 可选的 bias
    """
    # ---- 模型架构 ----
    hidden_size: int = 768           # h: 隐藏层维度
    num_layers: int = 12             # l: Transformer 层数
    num_heads: int = 12              # n_h: 注意力头数（Q 头数）
    num_kv_heads: int | None = None  # GQA 的 KV 头数，None 表示 MHA (= num_heads)
    intermediate_size: int | None = None  # MLP 中间维度，None 表示 4 * hidden_size
    vocab_size: int = 50257          # V: 词表大小

    # ---- 计算场景 ----
    seq_len: int = 2048              # s: 序列长度
    batch_size: int = 1              # B: 批大小

    # ---- 训练/推理选项 ----
    has_bias: bool = False           # 线性层是否有 bias
    tied_embeddings: bool = False    # embedding 和 output head 是否共享权重
    use_swiglu: bool = False         # 是否使用 SwiGLU（MLP 有 3 个矩阵）

    # ---- 数据精度 ----
    param_dtype_bytes: int = 2       # 模型参数精度：FP32=4, FP16/BF16=2
    kv_cache_dtype_bytes: int = 2    # KV cache 精度

    def __post_init__(self) -> None:
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度 d_head = h / n_h"""
        return self.hidden_size // self.num_heads

    @property
    def kv_hidden(self) -> int:
        """KV 投影的总维度 = n_kv_heads * head_dim"""
        return self.num_kv_heads * self.head_dim


# =============================================================================
# 核心计算器
# =============================================================================

class TransformerCalculator:
    """Transformer 模型的参数量、FLOPs、KV Cache、显存分析器。"""

    def __init__(self, config: TransformerConfig) -> None:
        self.cfg = config

    # =====================================================================
    # 1. 参数量计算
    # =====================================================================

    def attention_params(self) -> dict:
        """计算 Self-Attention 模块的参数量。

        标准 MHA (num_kv_heads == num_heads):
          Q: (h, h), K: (h, h), V: (h, h), O: (h, h) → 总共 4h²

        GQA (num_kv_heads < num_heads):
          Q: (h, h), K: (h, kv_h), V: (h, kv_h), O: (h, h)
          kv_h = num_kv_heads * head_dim
        """
        h = self.cfg.hidden_size
        kv_h = self.cfg.kv_hidden

        q_w = h * h
        k_w = h * kv_h
        v_w = h * kv_h
        o_w = h * h

        q_b = h if self.cfg.has_bias else 0
        k_b = kv_h if self.cfg.has_bias else 0
        v_b = kv_h if self.cfg.has_bias else 0
        o_b = h if self.cfg.has_bias else 0

        total = q_w + k_w + v_w + o_w + q_b + k_b + v_b + o_b
        return {
            "q_proj": q_w + q_b,
            "k_proj": k_w + k_b,
            "v_proj": v_w + v_b,
            "o_proj": o_w + o_b,
            "weights": q_w + k_w + v_w + o_w,
            "bias": q_b + k_b + v_b + o_b,
            "total": total,
        }

    def mlp_params(self) -> dict:
        """计算 MLP 模块的参数量。

        标准 MLP: W1(h→4h) + W2(4h→h)
          参数量 = h*4h + 4h*h = 8h²

        SwiGLU MLP: W1(h→4h) + W_gate(h→4h) + W2(4h→h)
          参数量 = 3 * h * intermediate_size
        """
        h = self.cfg.hidden_size
        inter = self.cfg.intermediate_size

        if self.cfg.use_swiglu:
            weights = 3 * h * inter
            bias = (2 * inter + h) if self.cfg.has_bias else 0
        else:
            weights = 2 * h * inter
            bias = (inter + h) if self.cfg.has_bias else 0

        return {
            "weights": weights,
            "bias": bias,
            "total": weights + bias,
        }

    def layernorm_params(self) -> int:
        """LayerNorm × 2 的参数量：每个 LN 有 scale(h) + bias(h) = 2h。"""
        return 4 * self.cfg.hidden_size  # 2 * 2 * h

    def per_layer_params(self) -> int:
        """单个 Transformer 层的参数量 = Attention + MLP + LayerNorm×2。"""
        return (
            self.attention_params()["total"]
            + self.mlp_params()["total"]
            + self.layernorm_params()
        )

    def embedding_params(self) -> int:
        """Embedding 层 + 输出 head + Final LayerNorm 的参数量。"""
        h = self.cfg.hidden_size
        vocab = self.cfg.vocab_size

        embed = vocab * h
        output_head = 0 if self.cfg.tied_embeddings else vocab * h
        final_ln = 2 * h  # scale + bias
        return embed + output_head + final_ln

    def total_params(self) -> int:
        """模型总参数量。"""
        return self.cfg.num_layers * self.per_layer_params() + self.embedding_params()

    # =====================================================================
    # 2. FLOPs 计算
    # =====================================================================

    def attention_flops(self) -> int:
        """单层 Attention 的前向 FLOPs（矩阵乘积计 2mkn）。

        每个 token：
          Q/K/V 投影: 2 * s * (h*h + h*kv_h + h*kv_h + h*h) = 2s(2h² + 2h*kv_h)
          QK^T:  2 * s² * h
          Attn×V: 2 * s² * h
          整体乘以 batch_size
        """
        B = self.cfg.batch_size
        s = self.cfg.seq_len
        h = self.cfg.hidden_size
        kv_h = self.cfg.kv_hidden

        proj_flops = 2 * B * s * (h * h + h * kv_h + h * kv_h + h * h)
        score_flops = 2 * B * s * s * h   # QK^T
        apply_flops = 2 * B * s * s * h   # Attn × V
        return proj_flops + score_flops + apply_flops

    def mlp_flops(self) -> int:
        """单层 MLP 的前向 FLOPs。"""
        B = self.cfg.batch_size
        s = self.cfg.seq_len
        h = self.cfg.hidden_size
        inter = self.cfg.intermediate_size

        num_matmuls = 3 if self.cfg.use_swiglu else 2
        # W1: (B,s,h)@(h,inter) = 2Bsh*inter; W2: (B,s,inter)@(inter,h) = 2Bs*inter*h
        # SwiGLU adds gate: same as W1
        return 2 * B * s * h * inter * num_matmuls

    def per_layer_flops(self) -> int:
        """单层前向 FLOPs = Attention + MLP（LN flops 忽略不计）。"""
        return self.attention_flops() + self.mlp_flops()

    def logits_flops(self) -> int:
        """输出 logits 层的 FLOPs: (B,s,h) @ (h,V)。"""
        B = self.cfg.batch_size
        s = self.cfg.seq_len
        h = self.cfg.hidden_size
        return 2 * B * s * h * self.cfg.vocab_size

    def forward_flops(self) -> int:
        """完整模型的前向 FLOPs。"""
        return self.cfg.num_layers * self.per_layer_flops() + self.logits_flops()

    def backward_flops(self) -> int:
        """反向传播 FLOPs ≈ 2 × 前向（对权重和输入各求一次梯度）。"""
        return 2 * self.forward_flops()

    def train_flops(self) -> int:
        """训练一步的总 FLOPs ≈ 3 × 前向。"""
        return 3 * self.forward_flops()

    # =====================================================================
    # 3. KV Cache
    # =====================================================================

    def kv_cache_bytes(self) -> int:
        """推理时的 KV Cache 显存（字节）。

        每层每 token 的 KV Cache:
          K: (B, s, n_kv, d_h) = B * s * kv_h
          V: (B, s, n_kv, d_h) = B * s * kv_h
          总计: 2 * B * s * kv_h * bytes_per_elem
        乘以层数 l。
        """
        B = self.cfg.batch_size
        s = self.cfg.seq_len
        kv_h = self.cfg.kv_hidden
        return 2 * B * s * kv_h * self.cfg.num_layers * self.cfg.kv_cache_dtype_bytes

    # =====================================================================
    # 4. 训练显存
    # =====================================================================

    def training_memory(
        self,
        optimizer: str = "adam",
        mixed_precision: bool = True,
    ) -> dict:
        """训练显存估算（字节）。

        FP32 + Adam:
          参数(4) + 梯度(4) + 一阶矩m(4) + 二阶矩v(4) = 16 bytes/param

        Mixed Precision + Adam:
          fp16参数(2) + fp16梯度(2) + fp32主权重(4) + m(4) + v(4) = 16 bytes/param

        FP32 + SGD (无 momentum):
          参数(4) + 梯度(4) = 8 bytes/param

        FP32 + SGD (有 momentum):
          参数(4) + 梯度(4) + 动量(4) = 12 bytes/param
        """
        P = self.total_params()
        opt = optimizer.lower()

        if opt == "adam":
            if mixed_precision:
                # fp16 params + fp16 grads + fp32 master + m + v
                params_bytes = 2 * P
                grads_bytes = 2 * P
                master_bytes = 4 * P
                m_bytes = 4 * P
                v_bytes = 4 * P
            else:
                # fp32 params + fp32 grads + m + v
                params_bytes = 4 * P
                grads_bytes = 4 * P
                master_bytes = 0
                m_bytes = 4 * P
                v_bytes = 4 * P
            total = params_bytes + grads_bytes + master_bytes + m_bytes + v_bytes
            return {
                "params": params_bytes,
                "grads": grads_bytes,
                "master_weights": master_bytes,
                "optimizer_m": m_bytes,
                "optimizer_v": v_bytes,
                "total": total,
                "bytes_per_param": total // P,
            }
        elif opt == "sgd":
            # Pure SGD (no momentum): params(4) + grads(4) = 8 bytes/param
            params_bytes = 4 * P
            grads_bytes = 4 * P
            total = params_bytes + grads_bytes
            return {
                "params": params_bytes,
                "grads": grads_bytes,
                "total": total,
                "bytes_per_param": total // P,
            }
        else:
            raise ValueError(f"Unknown optimizer: {optimizer!r}")

    # =====================================================================
    # 5. 中间激活显存
    # =====================================================================

    def activation_memory_per_layer(self) -> int:
        """单层中间激活显存（字节，FP16，需保留用于反向传播）。

        公式（来自 Korthikanti et al.）:
          sbh * (34 + 5 * n_h * s / h) bytes
        """
        s = self.cfg.seq_len
        b = self.cfg.batch_size
        h = self.cfg.hidden_size
        n_h = self.cfg.num_heads
        return int(s * b * h * (34 + 5 * n_h * s / h))

    def activation_memory_total(self) -> int:
        """所有层的中间激活显存（字节）。"""
        return self.cfg.num_layers * self.activation_memory_per_layer()

    # =====================================================================
    # 6. 综合报告
    # =====================================================================

    def summary(self) -> dict:
        """返回关键数字汇总。"""
        mem = self.training_memory()
        return {
            "total_params": self.total_params(),
            "per_layer_params": self.per_layer_params(),
            "forward_flops": self.forward_flops(),
            "backward_flops": self.backward_flops(),
            "train_flops": self.train_flops(),
            "kv_cache_bytes": self.kv_cache_bytes(),
            "activation_memory_bytes": self.activation_memory_total(),
            "training_memory_bytes": mem["total"],
        }

    def print_full_report(self) -> None:
        """打印格式化报告。"""
        cfg = self.cfg
        print(f"Model: hidden={cfg.hidden_size}, layers={cfg.num_layers}, "
              f"heads={cfg.num_heads}, kv_heads={cfg.num_kv_heads}, "
              f"vocab={cfg.vocab_size}, seq={cfg.seq_len}, batch={cfg.batch_size}")
        print("-" * 60)
        s = self.summary()
        for key, value in s.items():
            print(f"  {key:35s}: {format_num(value)}")


# =============================================================================
# 工具函数
# =============================================================================

def format_num(n: int | float) -> str:
    """将大数字格式化为可读字符串（K/M/B）。"""
    n = float(n)
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.2f}K"
    if n == int(n):
        return str(int(n))
    return f"{n:.4f}"
