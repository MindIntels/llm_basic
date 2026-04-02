"""Transformer 模型参数量 / FLOPs / KV Cache / 显存 计算器

基于知乎文章「分析transformer模型的参数量、计算量、中间激活、KV cache」的完整实现。
以标准 Transformer Decoder 层为例，逐步计算每个组件的参数量、计算量和显存占用。

References:
    - https://zhuanlan.zhihu.com/p/624740065
    - Kaplan et al., "Scaling Laws for Neural Language Models"
    - Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models"
"""

from dataclasses import dataclass


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

    def __post_init__(self):
        # GQA: 如果未指定 KV 头数，默认等于 Q 头数（标准 MHA）
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        # MLP 中间维度：默认 4 * hidden_size
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度 d_head = h / n_h"""
        return self.hidden_size // self.num_heads


# =============================================================================
# 核心计算器
# =============================================================================

class TransformerCalculator:
    """Transformer 模型的参数量、FLOPs、KV Cache、显存分析器。

    所有计算方法都附带详细注释，解释每一步推导。
    """

    def __init__(self, config: TransformerConfig):
        self.cfg = config

    # =====================================================================
    # 1. 参数量计算
    # =====================================================================

    def attention_params(self) -> dict:
        """计算 Self-Attention 模块的参数量。

        标准 Multi-Head Attention 包含 4 个线性投影：
          Q 投影: (h, h)        → h² 参数 (+h bias)
          K 投影: (h, kv_dim)   → h * kv_dim 参数
          V 投影: (h, kv_dim)   → h * kv_dim 参数
          O 投影: (h, h)        → h² 参数 (+h bias)

        对于标准 MHA (num_kv_heads == num_heads):
          kv_dim = h, 所以 Q/K/V/O 各 h², 总共 4h²

        对于 GQA (num_kv_heads < num_heads):
          kv_dim = num_kv_heads * head_dim, 可能 < h
          K/V 投影更小，节省参数
        """
        h = self.cfg.hidden_size
        n_kv = self.cfg.num_kv_heads
        d_head = self.cfg.head_dim

        # kv_dim: K/V 投影的输出维度
        # 标准 MHA: kv_dim = h
        # GQA: kv_dim = num_kv_heads * head_dim (可能 < h)
        kv_dim = n_kv * d_head

        # ---- 权重参数 ----
        # Q 投影: 输入 h 维 → 输出 h 维 (Q 头数 × d_head = h)
        q_weight = h * h

        # K 投影: 输入 h 维 → 输出 kv_dim 维
        k_weight = h * kv_dim

        # V 投影: 输入 h 维 → 输出 kv_dim 维
        v_weight = h * kv_dim

        # Output 投影: 输入 h 维 → 输出 h 维
        o_weight = h * h

        total_weight = q_weight + k_weight + v_weight + o_weight

        # ---- Bias 参数 ----
        if self.cfg.has_bias:
            # 每个投影有 output_dim 个 bias
            q_bias = h
            k_bias = kv_dim
            v_bias = kv_dim
            o_bias = h
            total_bias = q_bias + k_bias + v_bias + o_bias
        else:
            total_bias = 0

        return {
            "q_proj": q_weight,
            "k_proj": k_weight,
            "v_proj": v_weight,
            "o_proj": o_weight,
            "weight_total": total_weight,
            "bias_total": total_bias,
            "total": total_weight + total_bias,
        }

    def mlp_params(self) -> dict:
        """计算 MLP (Feed-Forward) 模块的参数量。

        标准 MLP (GeLU):
          W1: (h, 4h)  → 4h² 参数  (up projection)
          W2: (4h, h)  → 4h² 参数  (down projection)
          总计: 8h²

        SwiGLU MLP (多一个 gate 矩阵):
          W_gate: (h, 4h)  → 4h²
          W_up:   (h, 4h)  → 4h²
          W_down: (4h, h)  → 4h²
          总计: 12h² (或用实际 intermediate_size: 3 * h * inter)

        注意：实际中 SwiGLU 的 intermediate_size 通常调整为 2/3 * 4h
        以保持总参数量与标准 MLP 接近，即 intermediate = 8h/3
        """
        h = self.cfg.hidden_size
        inter = self.cfg.intermediate_size

        if self.cfg.use_swiglu:
            # SwiGLU: gate(h→inter) + up(h→inter) + down(inter→h) = 3 * h * inter
            w_gate = h * inter
            w_up = h * inter
            w_down = inter * h
            total_weight = w_gate + w_up + w_down
        else:
            # 标准 MLP: up(h→inter) + down(inter→h) = 2 * h * inter
            w_up = h * inter
            w_down = inter * h
            total_weight = w_up + w_down

        # Bias
        if self.cfg.has_bias:
            if self.cfg.use_swiglu:
                total_bias = inter + inter + h  # gate_bias + up_bias + down_bias
            else:
                total_bias = inter + h  # up_bias + down_bias
        else:
            total_bias = 0

        return {
            "weight_total": total_weight,
            "bias_total": total_bias,
            "total": total_weight + total_bias,
        }

    def layernorm_params(self) -> int:
        """计算一个 Transformer 层中 LayerNorm 的参数量。

        每个层有 2 个 LayerNorm（Attention 前 + MLP 前）
        每个 LayerNorm 有 gamma (scale) 和 beta (shift) 各 h 个参数
        RMSNorm 只有 gamma，没有 beta → 每个 h 参数

        总计: 2 × 2h = 4h (LayerNorm) 或 2 × h = 2h (RMSNorm)
        这里按标准 LayerNorm 计算: 4h
        """
        h = self.cfg.hidden_size
        # 每个 LayerNorm: gamma(h) + beta(h) = 2h
        # 2 个 LayerNorm per layer
        return 2 * (2 * h)

    def single_layer_params(self) -> dict:
        """计算单个 Transformer 层的总参数量。

        组成:
        - Self-Attention: 4h² (MHA) 或更少 (GQA)
        - MLP: 8h² (标准) 或 12h² (SwiGLU)
        - LayerNorm × 2: 4h
        - 总计约: 12h² + 4h ≈ 12h² (h 较大时)
        """
        attn = self.attention_params()
        mlp = self.mlp_params()
        ln = self.layernorm_params()

        return {
            "attention": attn["total"],
            "mlp": mlp["total"],
            "layernorm": ln,
            "total": attn["total"] + mlp["total"] + ln,
        }

    def total_params(self) -> dict:
        """计算完整模型的总参数量。

        完整模型 = Embedding + l × TransformerLayer + Final LN + Output Head

        公式:
          P = V×h + l×(12h²+4h) + 2h + V×h  (不共享 embedding)
          P = V×h + l×(12h²+4h) + 2h         (共享 embedding)

        其中:
          - V×h: token embedding 层
          - l×12h²: 所有 Transformer 层
          - 2h: final LayerNorm (gamma + beta)
          - V×h: output head (lm_head), 如果不共享权重
        """
        h = self.cfg.hidden_size
        l = self.cfg.num_layers
        v = self.cfg.vocab_size

        # 1. Token Embedding: (V, h)
        embedding = v * h

        # 2. 所有 Transformer 层
        per_layer = self.single_layer_params()["total"]
        all_layers = l * per_layer

        # 3. Final LayerNorm: gamma(h) + beta(h)
        final_ln = 2 * h

        # 4. Output Head (lm_head): (h, V)
        if self.cfg.tied_embeddings:
            output_head = 0  # 与 embedding 共享权重
        else:
            output_head = h * v

        total = embedding + all_layers + final_ln + output_head

        return {
            "embedding": embedding,
            "per_layer": per_layer,
            "all_layers": all_layers,
            "final_layernorm": final_ln,
            "output_head": output_head,
            "total": total,
        }

    # =====================================================================
    # 2. FLOPs 计算
    # =====================================================================

    def attention_flops_per_layer(self) -> dict:
        """计算单层 Self-Attention 的 FLOPs。

        矩阵乘法 (m,k)×(k,n) 的 FLOPs = 2×m×k×n
        （乘法 mkn 次 + 加法 mkn 次 = 2mkn）

        Attention 包含以下矩阵运算 (以序列为单位, 输入 (s, h)):
          1. Q投影: (s,h)×(h,h) → FLOPs = 2×s×h×h = 2sh²
          2. K投影: (s,h)×(h,kv_dim) → FLOPs = 2×s×h×kv_dim
          3. V投影: (s,h)×(h,kv_dim) → FLOPs = 2×s×h×kv_dim
          4. Attention Score: Q·Kᵀ → (s,h)×(h,s) → FLOPs = 2×s×h×s = 2s²h
             (对每个头: (s,d)×(d,s)=2s²d, 共 n_h 头 → 2s²×n_h×d = 2s²h)
          5. Attention × V: (s,s)×(s,kv_dim) 的等效
             实际是 (s,s)×(s,d) per head, n_h 头 → 2s²h
          6. Output投影: (s,h)×(h,h) → FLOPs = 2sh²

        注意：对于 GQA，步骤 4-5 中 KV 需要 repeat/broadcast 到 Q 的头数
        所以计算量仍然是 2s²h (因为实际计算时 Q 有 n_h 头)
        """
        h = self.cfg.hidden_size
        s = self.cfg.seq_len
        kv_dim = self.cfg.num_kv_heads * self.cfg.head_dim

        # Q 投影: (s,h) × (h,h) → 2sh²
        q_proj_flops = 2 * s * h * h

        # K 投影: (s,h) × (h,kv_dim) → 2s×h×kv_dim
        k_proj_flops = 2 * s * h * kv_dim

        # V 投影: (s,h) × (h,kv_dim) → 2s×h×kv_dim
        v_proj_flops = 2 * s * h * kv_dim

        # Attention Score (QK^T): 每头 (s,d)×(d,s)=2s²d, 共 n_h 头
        # 等效于 2s²h (因为 n_h × d = h)
        attn_score_flops = 2 * s * s * h

        # Attention × V: 每头 (s,s)×(s,d)=2s²d, 共 n_h 头
        # 等效于 2s²h
        attn_v_flops = 2 * s * s * h

        # Output 投影: (s,h) × (h,h) → 2sh²
        o_proj_flops = 2 * s * h * h

        # QKV 投影总计
        qkv_proj_flops = q_proj_flops + k_proj_flops + v_proj_flops

        # Attention 运算总计 (score + weighted sum)
        attn_compute_flops = attn_score_flops + attn_v_flops

        total = qkv_proj_flops + attn_compute_flops + o_proj_flops

        return {
            "q_proj": q_proj_flops,
            "k_proj": k_proj_flops,
            "v_proj": v_proj_flops,
            "qkv_proj_total": qkv_proj_flops,
            "attn_score": attn_score_flops,
            "attn_v_multiply": attn_v_flops,
            "attn_compute_total": attn_compute_flops,
            "o_proj": o_proj_flops,
            "total": total,
        }

    def mlp_flops_per_layer(self) -> dict:
        """计算单层 MLP 的 FLOPs。

        标准 MLP:
          W1 (up):   (s,h)×(h,4h) → 2×s×h×4h = 8sh²
          W2 (down): (s,4h)×(4h,h) → 2×s×4h×h = 8sh²
          总计: 16sh²

        SwiGLU MLP:
          W_gate: (s,h)×(h,inter) → 2×s×h×inter
          W_up:   (s,h)×(h,inter) → 2×s×h×inter
          gate * up (逐元素): s×inter
          W_down: (s,inter)×(inter,h) → 2×s×inter×h
          总计: 3×2×s×h×inter + s×inter
        """
        h = self.cfg.hidden_size
        s = self.cfg.seq_len
        inter = self.cfg.intermediate_size

        if self.cfg.use_swiglu:
            w_gate = 2 * s * h * inter
            w_up = 2 * s * h * inter
            elementwise = s * inter  # gate * up 逐元素乘
            w_down = 2 * s * inter * h
            total = w_gate + w_up + elementwise + w_down
        else:
            # 标准 MLP
            w_up = 2 * s * h * inter      # (s,h)×(h,inter)
            w_down = 2 * s * inter * h    # (s,inter)×(inter,h)
            total = w_up + w_down

        return {
            "total": total,
        }

    def single_layer_flops(self) -> dict:
        """计算单个 Transformer 层的前向 FLOPs。

        总计 = Attention FLOPs + MLP FLOPs
        对于标准 MHA + 标准 MLP:
          = (8sh² + 4s²h) + 16sh²
          = 24sh² + 4s²h

        注意：LayerNorm、激活函数、softmax 等的 FLOPs 相对很小，通常忽略
        """
        attn = self.attention_flops_per_layer()
        mlp = self.mlp_flops_per_layer()

        return {
            "attention": attn["total"],
            "mlp": mlp["total"],
            "total": attn["total"] + mlp["total"],
        }

    def total_flops(self) -> dict:
        """计算完整模型的 FLOPs。

        前向 FLOPs:
          - Embedding lookup: 忽略（查表操作，无矩阵乘法）
          - l 层 Transformer: l × single_layer_flops
          - Output logits: (s,h)×(h,V) → 2shV

        训练 FLOPs:
          - 前向: C_fwd
          - 反向 ≈ 2 × C_fwd
            (需要计算 loss 对输入的梯度 + loss 对权重的梯度，各约 C_fwd)
          - 总训练: 3 × C_fwd

        近似公式（忽略 attention s² 项和 logits）:
          FLOPs_fwd ≈ 2 × P × s  (每 token 2P FLOPs)
          FLOPs_train ≈ 6 × P × s  (每 token 6P FLOPs)
        """
        h = self.cfg.hidden_size
        s = self.cfg.seq_len
        l = self.cfg.num_layers
        v = self.cfg.vocab_size

        per_layer = self.single_layer_flops()["total"]
        all_layers = l * per_layer

        # Output logits: (s,h) × (h,V) → 2shV
        logits_flops = 2 * s * h * v

        # 前向总 FLOPs
        forward_flops = all_layers + logits_flops

        # 反向 ≈ 2× 前向
        backward_flops = 2 * forward_flops

        # 训练总 FLOPs (per sequence)
        train_flops = forward_flops + backward_flops

        return {
            "per_layer_forward": per_layer,
            "all_layers_forward": all_layers,
            "logits_forward": logits_flops,
            "forward": forward_flops,
            "backward": backward_flops,
            "train_total": train_flops,
        }

    # =====================================================================
    # 3. KV Cache 计算（推理）
    # =====================================================================

    def kv_cache_memory(self) -> dict:
        """计算推理时 KV Cache 的显存占用。

        KV Cache 存储每一层 Attention 中已计算的 Key 和 Value，
        避免每生成一个 token 都重新计算整个序列的 KV。

        每层的 KV 缓存:
          K cache: (B, n_kv_heads, s, d_head) → B × n_kv × s × d_head 个元素
          V cache: (B, n_kv_heads, s, d_head) → B × n_kv × s × d_head 个元素
          合计: 2 × B × n_kv × s × d_head 个元素

        对于标准 MHA (n_kv = n_h):
          2 × B × n_h × s × d_head = 2 × B × s × h 个元素

        对于 GQA (n_kv < n_h):
          2 × B × n_kv × s × d_head 个元素 (更少!)

        显存 = 元素数 × dtype_bytes

        全模型: l 层 × 单层 KV Cache
        """
        B = self.cfg.batch_size
        l = self.cfg.num_layers
        s = self.cfg.seq_len
        n_kv = self.cfg.num_kv_heads
        d_head = self.cfg.head_dim
        dtype = self.cfg.kv_cache_dtype_bytes

        # 单层 KV cache 元素数
        per_layer_elements = 2 * B * n_kv * s * d_head

        # 单层 KV cache 字节数
        per_layer_bytes = per_layer_elements * dtype

        # 全模型 KV cache
        total_bytes = l * per_layer_bytes

        return {
            "per_layer_elements": per_layer_elements,
            "per_layer_bytes": per_layer_bytes,
            "total_bytes": total_bytes,
            "total_gb": total_bytes / (1024 ** 3),
        }

    # =====================================================================
    # 4. 模型参数 / 梯度 / 优化器状态的显存计算
    # =====================================================================

    def training_memory(self, optimizer: str = "adam",
                        mixed_precision: bool = True) -> dict:
        """计算训练时模型参数、梯度、优化器状态的显存占用。

        === FP32 纯精度训练 ===

        Adam 优化器:
          模型参数 (FP32):  4Φ bytes  (每个参数 4 字节)
          梯度 (FP32):      4Φ bytes
          一阶矩 m (FP32):  4Φ bytes  (Adam 的 running mean)
          二阶矩 v (FP32):  4Φ bytes  (Adam 的 running variance)
          合计: 16Φ bytes

        SGD 优化器:
          模型参数 (FP32):  4Φ bytes
          梯度 (FP32):      4Φ bytes
          SGD无momentum:    0 额外
          SGD有momentum:    4Φ bytes (动量缓冲)
          合计: 8Φ 或 12Φ bytes

        === Mixed Precision 训练 (FP16/BF16) ===

        Adam 优化器:
          FP16 模型参数:    2Φ bytes  (前向/反向使用)
          FP16 梯度:        2Φ bytes  (反向传播产生)
          FP32 master 权重: 4Φ bytes  (optimizer step 中更新)
          FP32 一阶矩 m:   4Φ bytes
          FP32 二阶矩 v:   4Φ bytes
          合计: 16Φ bytes

        注：某些实现将梯度也以 FP32 累积 → 18Φ bytes
        """
        total_params = self.total_params()["total"]

        if mixed_precision:
            # Mixed precision + Adam
            if optimizer == "adam":
                # FP16 模型参数: 用于前向和反向计算
                model_fp16 = 2 * total_params
                # FP16 梯度: 反向传播输出
                grad_fp16 = 2 * total_params
                # FP32 master 权重: 用于精确的优化器更新
                model_fp32_master = 4 * total_params
                # Adam 一阶矩 (FP32): 梯度的指数移动平均
                adam_m = 4 * total_params
                # Adam 二阶矩 (FP32): 梯度平方的指数移动平均
                adam_v = 4 * total_params

                return {
                    "num_params": total_params,
                    "model_fp16": model_fp16,
                    "grad_fp16": grad_fp16,
                    "model_fp32_master": model_fp32_master,
                    "adam_m": adam_m,
                    "adam_v": adam_v,
                    "total_bytes": model_fp16 + grad_fp16 + model_fp32_master + adam_m + adam_v,
                    "bytes_per_param": 16,
                    "total_gb": (model_fp16 + grad_fp16 + model_fp32_master + adam_m + adam_v) / (1024 ** 3),
                }
            elif optimizer == "sgd":
                model_fp16 = 2 * total_params
                grad_fp16 = 2 * total_params
                model_fp32_master = 4 * total_params
                return {
                    "num_params": total_params,
                    "model_fp16": model_fp16,
                    "grad_fp16": grad_fp16,
                    "model_fp32_master": model_fp32_master,
                    "total_bytes": model_fp16 + grad_fp16 + model_fp32_master,
                    "bytes_per_param": 8,
                    "total_gb": (model_fp16 + grad_fp16 + model_fp32_master) / (1024 ** 3),
                }
        else:
            # Pure FP32
            if optimizer == "adam":
                # FP32 模型参数
                model_bytes = 4 * total_params
                # FP32 梯度
                grad_bytes = 4 * total_params
                # Adam 状态
                adam_m = 4 * total_params
                adam_v = 4 * total_params

                return {
                    "num_params": total_params,
                    "model_fp32": model_bytes,
                    "grad_fp32": grad_bytes,
                    "adam_m": adam_m,
                    "adam_v": adam_v,
                    "total_bytes": model_bytes + grad_bytes + adam_m + adam_v,
                    "bytes_per_param": 16,
                    "total_gb": (model_bytes + grad_bytes + adam_m + adam_v) / (1024 ** 3),
                }
            elif optimizer == "sgd":
                model_bytes = 4 * total_params
                grad_bytes = 4 * total_params
                return {
                    "num_params": total_params,
                    "model_fp32": model_bytes,
                    "grad_fp32": grad_bytes,
                    "total_bytes": model_bytes + grad_bytes,
                    "bytes_per_param": 8,
                    "total_gb": (model_bytes + grad_bytes) / (1024 ** 3),
                }

        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # =====================================================================
    # 5. 中间激活显存计算
    # =====================================================================

    def activation_memory_per_layer(self) -> dict:
        """计算单层 Transformer 的中间激活显存 (用于反向传播)。

        在训练中需要保存前向传播的中间结果用于反向传播计算梯度。
        以下按 FP16/BF16 (2 bytes) 计算。

        Self-Attention 中间激活:
          1. LayerNorm 输入:  s×h → 2sh bytes
          2. Q, K, V 矩阵:   3×s×h → 6sh bytes
             (GQA: s×h + 2×s×kv_dim)
          3. Attention Score (softmax 前): n_h×s×s → 2×n_h×s² bytes
          4. Attention Score (softmax 后/dropout): n_h×s×s → 2×n_h×s² bytes
          5. Attention 输出: s×h → 2sh bytes
          6. O投影后: s×h → 2sh bytes
          小计: 2sh×(4) + 4×n_h×s² = 8sh + 4n_h×s²  (bytes, FP16)
                即 s×(8h + 4n_h×s) bytes

        MLP 中间激活:
          1. LayerNorm 输入: s×h → 2sh bytes
          2. W1 输出 (up): s×inter → 2s×inter bytes
          3. 激活函数输入: s×inter → 2s×inter bytes
          4. W2 输出 (down, 用于 dropout): s×h → 2sh bytes
          小计: 2sh×2 + 2s×inter×2 = 4sh + 4s×inter bytes
                标准 MLP (inter=4h): 4sh + 16sh = 20sh bytes

        合计（标准 MHA + 标准 MLP, inter=4h):
          = (8sh + 4n_h×s²) + 20sh
          = 28sh + 4n_h×s² bytes (FP16)

        经典公式:
          Activation ≈ sbh × (34 + 5×n_h×s/h) bytes
          (包含 dropout mask 等额外开销)
        """
        s = self.cfg.seq_len
        h = self.cfg.hidden_size
        B = self.cfg.batch_size
        n_h = self.cfg.num_heads
        inter = self.cfg.intermediate_size
        dtype = 2  # FP16 = 2 bytes

        # ---- Attention 激活 ----
        # LayerNorm 输入保存
        attn_ln_input = B * s * h * dtype

        # Q, K, V 矩阵 (需要保存用于反向传播)
        qkv_activations = 3 * B * s * h * dtype

        # Attention scores (softmax 后, 每头 s×s)
        # shape: (B, n_h, s, s) → B × n_h × s × s 个元素
        attn_scores = B * n_h * s * s * dtype

        # Softmax 后 dropout mask (1 byte per element)
        attn_dropout_mask = B * n_h * s * s * 1

        # Attention 输出 (context)
        attn_output = B * s * h * dtype

        attention_total = attn_ln_input + qkv_activations + attn_scores + attn_dropout_mask + attn_output

        # ---- MLP 激活 ----
        # LayerNorm 输入保存
        mlp_ln_input = B * s * h * dtype

        # W1 输出 (up projection), 需保存用于激活函数反向
        mlp_up_output = B * s * inter * dtype

        # 激活函数输出 (GeLU需要保存输入用于求导)
        mlp_activation = B * s * inter * dtype

        # dropout mask
        mlp_dropout_mask = B * s * h * 1

        mlp_total = mlp_ln_input + mlp_up_output + mlp_activation + mlp_dropout_mask

        total = attention_total + mlp_total

        return {
            "attention_bytes": attention_total,
            "mlp_bytes": mlp_total,
            "total_bytes": total,
            "total_gb": total / (1024 ** 3),
        }

    def total_activation_memory(self) -> dict:
        """计算所有层的中间激活总显存。"""
        per_layer = self.activation_memory_per_layer()
        l = self.cfg.num_layers
        total = l * per_layer["total_bytes"]
        return {
            "per_layer_bytes": per_layer["total_bytes"],
            "total_bytes": total,
            "total_gb": total / (1024 ** 3),
        }

    # =====================================================================
    # 6. 报告输出
    # =====================================================================

    def print_full_report(self):
        """打印完整的分析报告。"""
        cfg = self.cfg
        h, l, s = cfg.hidden_size, cfg.num_layers, cfg.seq_len

        print("=" * 70)
        print("Transformer 模型分析报告")
        print("=" * 70)

        # 配置summary
        print(f"\n--- 模型配置 ---")
        print(f"  hidden_size (h)     = {h}")
        print(f"  num_layers (l)      = {l}")
        print(f"  num_heads (n_h)     = {cfg.num_heads}")
        print(f"  num_kv_heads (n_kv) = {cfg.num_kv_heads}")
        print(f"  head_dim (d)        = {cfg.head_dim}")
        print(f"  intermediate_size   = {cfg.intermediate_size}")
        print(f"  vocab_size (V)      = {cfg.vocab_size}")
        print(f"  seq_len (s)         = {s}")
        print(f"  batch_size (B)      = {cfg.batch_size}")
        print(f"  has_bias            = {cfg.has_bias}")
        print(f"  use_swiglu          = {cfg.use_swiglu}")

        # 参数量
        print(f"\n--- 参数量 ---")
        layer_p = self.single_layer_params()
        print(f"  单层 Attention 参数:  {self.attention_params()['total']:>15,}")
        print(f"  单层 MLP 参数:        {self.mlp_params()['total']:>15,}")
        print(f"  单层 LayerNorm 参数:  {self.layernorm_params():>15,}")
        print(f"  单层总参数:           {layer_p['total']:>15,}")

        total_p = self.total_params()
        print(f"  Embedding 参数:       {total_p['embedding']:>15,}")
        print(f"  所有层参数:           {total_p['all_layers']:>15,}")
        print(f"  Output Head 参数:     {total_p['output_head']:>15,}")
        print(f"  模型总参数:           {total_p['total']:>15,}  "
              f"({total_p['total']/1e9:.2f}B)")

        # FLOPs
        print(f"\n--- FLOPs (per sequence, s={s}) ---")
        flops = self.total_flops()
        print(f"  单层前向 FLOPs:       {flops['per_layer_forward']:>20,}")
        print(f"  前向总 FLOPs:         {flops['forward']:>20,}  "
              f"({flops['forward']/1e12:.2f} TFLOPs)")
        print(f"  反向 FLOPs (≈2×fwd):  {flops['backward']:>20,}")
        print(f"  训练总 FLOPs (≈3×fwd):{flops['train_total']:>20,}  "
              f"({flops['train_total']/1e12:.2f} TFLOPs)")

        # KV Cache
        print(f"\n--- KV Cache (推理, B={cfg.batch_size}, s={s}) ---")
        kv = self.kv_cache_memory()
        print(f"  单层 KV Cache:        {kv['per_layer_bytes']:>15,} bytes  "
              f"({kv['per_layer_bytes']/1024**2:.2f} MB)")
        print(f"  总 KV Cache:          {kv['total_bytes']:>15,} bytes  "
              f"({kv['total_gb']:.4f} GB)")

        # 训练显存
        print(f"\n--- 训练显存 (FP32 + Adam) ---")
        mem_fp32 = self.training_memory(optimizer="adam", mixed_precision=False)
        print(f"  模型参数 (FP32):      {mem_fp32['model_fp32']:>15,} bytes")
        print(f"  梯度 (FP32):          {mem_fp32['grad_fp32']:>15,} bytes")
        print(f"  Adam m (FP32):        {mem_fp32['adam_m']:>15,} bytes")
        print(f"  Adam v (FP32):        {mem_fp32['adam_v']:>15,} bytes")
        print(f"  合计: 16 bytes/param  {mem_fp32['total_bytes']:>15,} bytes  "
              f"({mem_fp32['total_gb']:.2f} GB)")

        print(f"\n--- 训练显存 (Mixed Precision + Adam) ---")
        mem_mp = self.training_memory(optimizer="adam", mixed_precision=True)
        print(f"  FP16 模型参数:        {mem_mp['model_fp16']:>15,} bytes")
        print(f"  FP16 梯度:            {mem_mp['grad_fp16']:>15,} bytes")
        print(f"  FP32 master 权重:     {mem_mp['model_fp32_master']:>15,} bytes")
        print(f"  Adam m (FP32):        {mem_mp['adam_m']:>15,} bytes")
        print(f"  Adam v (FP32):        {mem_mp['adam_v']:>15,} bytes")
        print(f"  合计: 16 bytes/param  {mem_mp['total_bytes']:>15,} bytes  "
              f"({mem_mp['total_gb']:.2f} GB)")

        # 激活显存
        print(f"\n--- 中间激活显存 (FP16, B={cfg.batch_size}, s={s}) ---")
        act = self.total_activation_memory()
        per_layer_act = self.activation_memory_per_layer()
        print(f"  单层激活 (Attention):  {per_layer_act['attention_bytes']:>15,} bytes")
        print(f"  单层激活 (MLP):        {per_layer_act['mlp_bytes']:>15,} bytes")
        print(f"  单层激活合计:          {per_layer_act['total_bytes']:>15,} bytes  "
              f"({per_layer_act['total_bytes']/1024**2:.2f} MB)")
        print(f"  全模型激活合计:        {act['total_bytes']:>15,} bytes  "
              f"({act['total_gb']:.2f} GB)")

        print(f"\n{'=' * 70}")


# =============================================================================
# 便捷函数
# =============================================================================

def format_num(n: int | float) -> str:
    """格式化大数字为可读字符串 (B/M/K)。"""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def quick_estimate(name: str, hidden: int, layers: int, heads: int,
                   vocab: int = 50257, seq_len: int = 2048):
    """快速估算常见模型的参数量和显存。"""
    cfg = TransformerConfig(
        hidden_size=hidden, num_layers=layers, num_heads=heads,
        vocab_size=vocab, seq_len=seq_len,
    )
    calc = TransformerCalculator(cfg)
    p = calc.total_params()["total"]
    mem = calc.training_memory()
    kv = calc.kv_cache_memory()
    print(f"{name}: params={format_num(p)}, "
          f"train_mem={mem['total_gb']:.1f}GB, "
          f"kv_cache={kv['total_gb']:.3f}GB")


if __name__ == "__main__":
    # === 示例 1: GPT-2 Small (124M) ===
    print("=" * 70)
    print("示例: GPT-2 Small (124M)")
    print("=" * 70)
    cfg = TransformerConfig(
        hidden_size=768, num_layers=12, num_heads=12,
        vocab_size=50257, seq_len=1024, batch_size=8,
    )
    calc = TransformerCalculator(cfg)
    calc.print_full_report()

    # === 示例 2: 常见模型快速对比 ===
    print("\n\n=== 常见模型参数量快速对比 ===")
    quick_estimate("GPT-2 Small  (124M)", 768, 12, 12)
    quick_estimate("GPT-2 Medium (355M)", 1024, 24, 16)
    quick_estimate("GPT-2 Large  (774M)", 1280, 36, 20)
    quick_estimate("GPT-2 XL    (1.5B)", 1600, 48, 25)
    quick_estimate("GPT-3       (175B)", 12288, 96, 96, vocab=50257, seq_len=2048)
    quick_estimate("LLaMA-7B         ", 4096, 32, 32, vocab=32000, seq_len=2048)
    quick_estimate("LLaMA-13B        ", 5120, 40, 40, vocab=32000, seq_len=2048)
    quick_estimate("LLaMA-65B        ", 8192, 80, 64, vocab=32000, seq_len=2048)
