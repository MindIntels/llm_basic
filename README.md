# Transformer 模型参数量 / FLOPs / 显存分析

## 1. Transformer 层结构回顾

一个标准 Transformer Decoder 层包含：
- **Multi-Head Self-Attention (MHA)**：Q/K/V 投影 + Attention 计算 + 输出投影
- **MLP (Feed-Forward Network)**：两个线性层 (h→4h→h)，中间 GeLU/SiLU
- **LayerNorm × 2**：分别在 Attention 和 MLP 前/后

## 2. 模型参数量计算

### 单个 Transformer 层

| 组件 | 权重形状 | 参数量（无 bias） | 参数量（含 bias） |
|------|----------|-------------------|-------------------|
| Q 投影 | (h, h) | h² | h² + h |
| K 投影 | (h, h) | h² | h² + h |
| V 投影 | (h, h) | h² | h² + h |
| O 投影 | (h, h) | h² | h² + h |
| MLP W1 (h→4h) | (h, 4h) | 4h² | 4h² + 4h |
| MLP W2 (4h→h) | (4h, h) | 4h² | 4h² + h |
| LayerNorm × 2 | (h,) × 2 | 4h | 4h |
| **合计** | | **12h²+4h** | **12h²+13h** |

> 近似：**每层 ≈ 12h²** 参数（h 较大时 bias 项可忽略）

### 完整模型

$$P_{\text{total}} = l \times 12h^2 + V \times h + 2h$$

其中 l=层数，V=词表大小，最后 2h 是 final LayerNorm

## 3. FLOPs 计算

矩阵乘法 `(m,k) × (k,n)` 的 FLOPs = 2mkn（乘法+加法各 mkn 次）

### 单个 Transformer 层（per token → per sequence）

| 操作 | FLOPs（一个序列，长度 s） |
|------|--------------------------|
| QKV 投影 | 3 × 2sh² = **6sh²** |
| Attention Score (Q·Kᵀ) | **2s²h** |
| Attention × V | **2s²h** |
| Output 投影 | **2sh²** |
| MLP W1 | **8sh²** |
| MLP W2 | **8sh²** |
| **合计** | **24sh² + 4s²h** |

### 训练总 FLOPs

- 前向 = $C_{\text{fwd}}$
- 反向 ≈ $2 \times C_{\text{fwd}}$（对输入和权重都要算梯度）
- **总训练 FLOPs** ≈ $3 \times C_{\text{fwd}}$

近似公式（忽略 attention 的 $s^2$ 项）：
$$F_{\text{train}} \approx 6 \times P \times s \times B \times N_{\text{tok}}$$

其中 $F_{\text{train}}$ 为训练 FLOPs，$N_{\text{tok}}$ 为总 token 数

## 4. KV Cache 计算（推理）

- 每层每 token 的 KV 缓存：$2 \times h \times b$ （$b$ = 每元素字节数）
- 总 KV Cache = $2 \times B \times l \times s \times h \times b$

对于 GQA（$n_{\text{kv}} < n_h$）：
$$\text{KV Cache} = 2 \times B \times l \times s \times n_{\text{kv}} \times d_h \times b$$

其中 $n_{\text{kv}}$ = KV 头数，$d_h$ = head dim，$b$ = dtype 字节数

## 5. 模型参数 / 梯度 / 优化器状态显存

### FP32 + Adam

| 组件 | 每参数字节 | 公式 |
|------|-----------|------|
| 模型参数 (FP32) | 4 | 4Φ |
| 梯度 (FP32) | 4 | 4Φ |
| Adam 一阶矩 m | 4 | 4Φ |
| Adam 二阶矩 v | 4 | 4Φ |
| **合计** | **16** | **16Φ** |

### Mixed Precision (FP16/BF16) + Adam

| 组件 | 每参数字节 | 公式 |
|------|-----------|------|
| FP16 模型参数 | 2 | 2Φ |
| FP16 梯度 | 2 | 2Φ |
| FP32 master 权重 | 4 | 4Φ |
| Adam 一阶矩 m (FP32) | 4 | 4Φ |
| Adam 二阶矩 v (FP32) | 4 | 4Φ |
| **合计** | **16** | **16Φ** |

> 注：有些实现中梯度累积用 FP32 (4Φ)，总计 18Φ

### SGD 优化器

- FP32 + SGD (无 momentum): 4Φ + 4Φ = **8Φ**
- FP32 + SGD (momentum): 4Φ + 4Φ + 4Φ = **12Φ**

## 6. 中间激活显存

每个 Transformer 层的中间激活（FP16，需要保留用于反向传播）：

$$\text{Activation} = sbh \times \left(34 + 5 \frac{n_h \times s}{h}\right) \text{ bytes}$$

其中 $s$ = 序列长度，$b$ = batch size，$h$ = hidden dim，$n_h$ = 注意力头数

---

## 7. 代码结构

```
model_basic/
├── README.md
├── attention/
│   ├── __init__.py               # attention 模块统一导出
│   ├── safe_softmax.py           # 数值稳定 softmax
│   ├── standard_attention.py     # 标准 MHA 基线实现
│   ├── split_qkv.py              # 显式 Q/K/V 拆分的多头注意力
│   ├── flash_attention_cpu.py    # Flash Attention v1 CPU 参考实现
│   ├── flash_attention_v2.py     # Flash Attention v2: deferred rescaling
│   ├── flash_attention_v3.py     # Flash Attention v3: block sparse / two-pass
│   ├── flash_attention_v4.py     # Flash Attention v4: KV-cache / window / softcap
│   ├── window_attention.py       # 滑窗注意力
│   ├── cross_attention.py        # 交叉注意力
│   ├── rope.py                   # RoPE 旋转位置编码（标准/线性缩放/动态缩放）
│   ├── mrope.py                  # mRoPE 多模态旋转位置编码（Qwen2-VL 风格）
│   ├── rmsnorm.py                # RMSNorm （无均値去除，无 bias）
│   ├── swiglu.py                 # SwiGLU FFN（三矩阵，SiLU 门控）
│   ├── gated_attention.py        # Gated Attention（输出元素闳门控）
│   ├── gated_deltanet.py         # Gated DeltaNet 线性循环层
│   └── gated_transformer.py      # 完整 Gated Transformer Block/Model
├── comm/
│   ├── __init__.py
│   └── comm.py               # 分布式通信原语实现
├── param/
│   ├── __init__.py
│   └── transformer_calc.py   # Transformer 参数量/FLOPs/显存 计算器
├── megatron_pp/
│   ├── core.py                   # 矩阵切分原语 + 通信模拟（numpy）
│   ├── matmul_parallel.py        # 矩阵乘的四种并行切分场景
│   ├── attention_parallel.py     # Attention DP / SP / TP 并行
│   └── linear_parallel.py        # Linear DP / SP / TP_Col / TP_Row / TP_ColRow
├── tests/
│   ├── test_comm.py              # 通信原语测试
│   ├── test_transformer_calc.py  # 计算器测试
│   ├── test_split_qkv.py         # SplitQKVAttention 测试
│   ├── test_flash_attention.py   # Flash Attention v1 测试
│   ├── test_flash_v2.py          # Flash Attention v2 测试
│   ├── test_flash_v3.py          # Flash Attention v3 测试
│   ├── test_flash_v4.py          # Flash Attention v4 测试
│   ├── test_window_cross_attention.py  # Window/Cross Attention 测试
│   ├── test_rope_mrope.py        # RoPE / mRoPE 测试（64 cases）
│   ├── test_megatron_pp.py       # Megatron 并行策略测试（37 cases）
│   └── test_gated_arch.py        # Gated 架构测试（80 cases）
└── conftest.py               # pytest sys.path 配置
```

## 8. 使用示例

```python
from param.transformer_calc import TransformerConfig, TransformerCalculator

# GPT-3 175B 配置
config = TransformerConfig(
    hidden_size=12288, num_layers=96, num_heads=96,
    vocab_size=50257, seq_len=2048, batch_size=1,
)
calc = TransformerCalculator(config)
calc.print_full_report()
```

---

# Attention 模块 (attention/)

`model_basic/attention/` 现已包含一套可直接运行的 attention 教学实现，覆盖：

- `StandardMHA`: 标准 scaled dot-product multi-head attention
- `SplitQKVAttention`: 显式拆分 head 级别的 Q/K/V 计算
- `FlashAttentionCPU` / `V2` / `V3` / `V4`: 从 tiled online softmax 到 KV-cache、滑窗、softcap 的渐进式实现
- `WindowAttention`: 本地滑窗 attention
- `CrossAttention`: encoder-decoder / 多模态 cross attention

```bash
cd model_basic
python -m pytest tests/ -q
# 或只跑 attention 测试
python -m pytest tests/test_split_qkv.py tests/test_flash_attention.py tests/test_flash_v2.py tests/test_flash_v3.py tests/test_flash_v4.py tests/test_window_cross_attention.py -v --tb=short
```

---

# RoPE / mRoPE (attention/rope.py 和 attention/mrope.py)

## RoPE — 旋转位置编码

**参考文献**：Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding"

**核心思想**：将位置信息编码为旋转角，使得 Q　1K 的点积自然编码**相对位置**：

$$\langle R_m q,\, R_n k \rangle = f(q, k, m-n)$$

实现时利用 complex-number trick，避免显式构造旋转矩阵：

$$\text{RoPE}(x, m) = x \cdot \cos(m\Theta) + \text{rotate\_half}(x) \cdot \sin(m\Theta)$$

其中 $\Theta_i = \text{base}^{-2i/d}$，$\text{base}=10000$。

| 类 / 函数 | 说明 |
|------------|------|
| `rotate_half(x)` | 将最后维分为两半并交换：`[-x2, x1]`，奇数统一抛异常 |
| `apply_rotary_emb(q,k,cos,sin)` | 对已成形 Q/K 应用预计算的 cos/sin |
| `RotaryEmbedding` | 缓存 cos/sin 表，支持动态扩展、`position_ids` 指定、缩放类型 |
| `RoPEAttention` | 内置 RoPE 的 MHA，支持 GQA（KV 头数 < Q 头数） |

**缩放直公**

| `rope_scaling` 类型 | 说明 | 用途 |
|---------------------|------|------|
| `None` | 标准 RoPE | 默认 |
| `"linear"` | 将位置除以 scale 因子 | LLaMA 长文本拓展 |
| `"dynamic"` | 超过 max_seq 时动态重计算 base | YaRN 风格 |

```python
from attention.rope import RotaryEmbedding, RoPEAttention

# 标准 RoPE
rope = RotaryEmbedding(head_dim=64)
cos, sin = rope(seq_len=512)

# GQA（8 Q 头 + 2 KV 头）
attn = RoPEAttention(hidden_size=512, num_heads=8, num_kv_heads=2)
out = attn(x)   # [B, S, 512]
```

---

## mRoPE — 多模态旋转位置编码

**参考文献**：Qwen2-VL (2024)、Wang et al. 2024 “RoPE to mRoPE”

**核心思想**：将 head_dim 分为 M 个通道，每个通道独立编码一个空间/时间轴：

$$\text{mRoPE}(x, \mathbf{p}) = \bigoplus_{j=0}^{M-1} \text{RoPE}(x_{[jc:(j+1)c]},\, p_j)$$

其中 $\mathbf{p} = (p_0, p_1, \ldots, p_{M-1})$ 是各轴的位置索引，$c = d/M$。

**Position ID 格式**：`[B, M, S]`

| 模态 | 轴配置 | 说明 |
|------|----------|------|
| 文本 | 所有轴相同 monotonic 整数 | 与标准 RoPE 等价 |
| 图像 | axis-0=时间, axis-1=行, axis-2=列 | H×W 块状网格 |
| 视频 | axis-0=帧, axis-1=行, axis-2=列 | T×H×W 块状网格 |

| 类 / 函数 | 说明 |
|------------|------|
| `make_text_position_ids(S, M)` | 纯文本一维位置 `[B,M,S]` |
| `make_image_position_ids(H,W,text_len)` | 文本 + 图像块 `[B,M,S]` |
| `make_video_position_ids(T,H,W,text_len)` | 文本 + 视频块 `[B,M,S]` |
| `MultimodalRotaryEmbedding` | M 轴缓存，调用时输入 `position_ids` |
| `mRoPEAttention` | 内置 mRoPE 的 MHA，支持 GQA |

```python
from attention.mrope import (
    MultimodalRotaryEmbedding, mRoPEAttention,
    make_text_position_ids, make_image_position_ids, make_video_position_ids,
)

# 3 轴（text, height, width）
mrope = MultimodalRotaryEmbedding(head_dim=64, num_axes=3)

# 图像：4×6 块，前 10 个 token 是文本
pos_ids = make_image_position_ids(height=4, width=6, text_len=10)  # [1,3,34]
cos, sin = mrope(pos_ids)

# mRoPE Attention MHA
attn = mRoPEAttention(hidden_size=256, num_heads=4, num_axes=3)
out = attn(x, position_ids=pos_ids)   # [B, S, 256]
```

## 测试内容（tests/test_rope_mrope.py，64 cases）

| 测试类 | Cases | 覆盖 |
|--------|-------|------|
| `TestRotateHalf` | 5 | 形状、定义、双旋转取负、奇数维异常、梯度 |
| `TestApplyRotaryEmb` | 3 | 输出形状、cos=1/sin=0 不旋转、模长不变 |
| `TestRotaryEmbedding` | 10 | 缓存形状、dtype、模长保持、动态扩展、pos_ids、linear/dynamic 缩放、位置0不旋转、多种 head_dim |
| `TestRoPEAttention` | 10 | 形状、因果无泄露、GQA 参数量减少、pos_ids、梯度、mask、确定性、多种序列长 |
| `TestPositionIDHelpers` | 9 | 文本/图像/视频 ID 形状、单调性、块数、高宽范围 |
| `TestMultimodalRoPE` | 8 | 届吏保持、多轴差异、head_dim 可被除性、缓存扩展、pos0不旋转 |
| `TestmRoPEAttention` | 9 | 纯文本/图像/视频 pos_ids、GQA、因果无泄露、梯度、确定性 |
| `TestRelativePositionInvariance` | 4 | 相对移位不变性、不同相对位置结果不同、模长保持、mRoPE 文本 ≡ RoPE |

```bash
cd model_basic
python -m pytest tests/test_rope_mrope.py -v --tb=short
# 64 passed
```

---

# Gated Architecture (attention/ 扩展)

`attention/` 中新增四个模块，构成完整的 **Gated Transformer** 架构：

| 模块 | 文件 | 核心内容 |
|------|------|----------|
| `RMSNorm` | `rmsnorm.py` | RMS 归一化，无均值去除，无 bias，仅 scale γ |
| `SwiGLUFFN` | `swiglu.py` | 三矩阵 FFN：gate@W_gate × up@W_up → down，SiLU 门控 |
| `GatedAttention` | `gated_attention.py` | MHA + 逐元素输出门 g=gate_act(x@W_g)，支持 sigmoid / SiLU |
| `GatedDeltaNet` | `gated_deltanet.py` | 线性循环层，delta-rule 快权重 + α 遗忘门 + β 写入门 + SiLU 输出门 |
| `GatedTransformerBlock` | `gated_transformer.py` | Pre-norm 块：RMSNorm + mixer + 残差 + RMSNorm + SwiGLU + 残差 |
| `GatedTransformer` | `gated_transformer.py` | N 层堆叠，支持 mixer_pattern 交替模式 + 最终 RMSNorm |

## RMSNorm

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \varepsilon}} \cdot \gamma$$

- 相比 LayerNorm 去掉了均值中心化和 β 参数
- 计算量约为 LayerNorm 的一半
- LLaMA / Mistral / Gemma 均采用此归一化

## SwiGLU

$$\text{SwiGLU}(x) = \text{SiLU}(x W_\text{gate}) \odot (x W_\text{up})$$
$$y = \text{SwiGLU}(x) \cdot W_\text{down}$$

默认中间维度 $d_{ff} = \lceil 8/3 \cdot d_{\text{model}} \rceil_{\times 64}$，参数量 = $3 d_{\text{model}} d_{ff}$。

## Gated Attention

```
gate = gate_act(x @ W_g)        # [B, S, H]  gate_act ∈ {sigmoid, SiLU}
out  = gate * MHA(x)            # 逐元素门控注意力输出
```

- `gate_act="sigmoid"` — 有界门 ∈ (0,1)，适合稳定训练
- `gate_act="silu"` — 无界门，表达力更强（Griffin / Hawk 风格）

## Gated DeltaNet

每步循环更新：

$$\alpha_t = \sigma(x_t W_\alpha), \quad \beta_t = \sigma(x_t W_\beta)$$
$$\hat{v}_t = S_{t-1} k_t$$
$$\delta_t = v_t - \alpha_t \hat{v}_t$$
$$S_t = \alpha_t S_{t-1} + \beta_t \, (k_t \otimes \delta_t)$$
$$y_t = S_t q_t$$

其中 $S_t \in \mathbb{R}^{H \times D \times D_v}$ 是快权重矩阵（fast-weight memory）。

- 支持 `forward(x)` 并行化（完整序列）
- 支持 `step(x_t, state)` 单步自回归推理
- 支持 `return_state=True` 跨块传递状态
- 分块推理正确性：chunk₁ 结尾状态传入 chunk₂，结果与全序列等价

## 完整块与模型

```python
from attention.gated_transformer import GatedTransformerBlock, GatedTransformer

# 单块：GatedAttention + SwiGLU
block = GatedTransformerBlock(hidden_size=256, num_heads=4, mixer="gated_attn")
out = block(torch.randn(2, 32, 256))   # [2, 32, 256]

# 完整模型：4 层交替 GatedAttention + GatedDeltaNet
model = GatedTransformer(
    hidden_size=256, num_heads=4, num_layers=4,
    mixer_pattern=["gated_attn", "deltanet"],   # 循环交替
    causal=True,
)
out = model(torch.randn(2, 32, 256))   # [2, 32, 256]
```

## 测试内容（tests/test_gated_arch.py，80 cases）

| 测试类 | Cases | 覆盖 |
|--------|-------|------|
| `TestRMSNorm` | 10 | 形状、零输入、学习权重、梯度、dtype、无 bias |
| `TestSwiGLU` | 10 | 函数正确性、零门、FFN 三矩阵、batch 独立、梯度 |
| `TestGatedAttention` | 11 | sigmoid/silu gate、因果无泄露、mask、梯度、pre-norm |
| `TestGatedDeltaNet` | 10 | 输出形状、状态传播、step() 等价性、分块连续性、梯度 |
| `TestGatedTransformerBlock` | 13 | 两种 mixer、残差、状态返回、范数类型、因果传播 |
| `TestGatedTransformer` | 9 | 单一/交替 pattern、层数、最终范数、梯度 |
| `TestIntegration` | 7 | 端到端 forward/backward、因果无泄露、分块 DeltaNet |

```bash
cd model_basic
python -m pytest tests/test_gated_arch.py -v --tb=short
# 80 passed
```

---

# 分布式通信原语 (comm.py)

> 单进程模拟实现，用 `list[Tensor]` 表示各 rank 的本地数据，无需启动多进程即可验证通信模式的正确性。

## 9. 支持的通信原语

### Point-to-point / Basic

| 原语 | 语义 | 通信量 |
|------|------|--------|
| `broadcast` | root 的数据广播到所有 rank | $(N-1) \cdot \lvert T \rvert$ |
| `scatter` | 将数据沿 dim 切分，分发给各 rank | $\frac{N-1}{N} \cdot \lvert T \rvert$ |
| `gather` | 将各 rank 的数据拼接到 root | $\frac{N-1}{N} \cdot \lvert T \rvert$ |

### Collective Operations

| 原语 | 语义 | 通信量 |
|------|------|--------|
| `all_gather` | 每个 rank 都拿到完整拼接结果 | $\frac{N-1}{N} \cdot \lvert T_\text{full} \rvert$ |
| `reduce` | 所有 rank 规约(求和)到 root | $(N-1) \cdot \lvert T \rvert$ |
| `all_reduce` | 规约后每个 rank 都拿到结果 | $\frac{2(N-1)}{N} \cdot \lvert T \rvert$ (Ring) |
| `reduce_scatter` | 先规约再切分，rank i 拿第 i 片 | $\frac{N-1}{N} \cdot \lvert T \rvert$ |
| `all_to_all` | 全交换 (chunk 矩阵转置) | $\frac{N-1}{N} \cdot \lvert T_\text{total} \rvert$ |

### Algorithmic Variants

| 原语 | 语义 | 特点 |
|------|------|------|
| `ring_all_reduce` | Ring 拓扑 AllReduce | 带宽最优: $\frac{2(N-1)}{N} \cdot \lvert T \rvert$ |
| `scatter_reduce` | 先切分再规约 (Ring Phase-1) | 等大小输入与 reduce_scatter 等价 |

## 10. 通信原语图示 (4 ranks)

```
broadcast(root=0):
  rank0: [ABCD] ──────> rank0: [ABCD]
                        rank1: [ABCD]
                        rank2: [ABCD]
                        rank3: [ABCD]

scatter(dim=0):
  rank0: [ABCD] ──────> rank0: [A]
                        rank1: [B]
                        rank2: [C]
                        rank3: [D]

gather(dim=0):
  rank0: [A] ─┐
  rank1: [B] ─┤────> root: [ABCD]
  rank2: [C] ─┤
  rank3: [D] ─┘

all_gather(dim=0):
  rank0: [A] ─┐        rank0: [ABCD]
  rank1: [B] ─┤──────> rank1: [ABCD]
  rank2: [C] ─┤        rank2: [ABCD]
  rank3: [D] ─┘        rank3: [ABCD]

reduce(sum, root=0):
  rank0: [A₀] ─┐
  rank1: [A₁] ─┤────> root: [A₀+A₁+A₂+A₃]
  rank2: [A₂] ─┤
  rank3: [A₃] ─┘

all_reduce(sum):
  rank0: [A₀] ─┐        rank0: [Σ]
  rank1: [A₁] ─┤──────> rank1: [Σ]   Σ = A₀+A₁+A₂+A₃
  rank2: [A₂] ─┤        rank2: [Σ]
  rank3: [A₃] ─┘        rank3: [Σ]

reduce_scatter(sum, dim=0):
  rank0: [A₀B₀C₀D₀] ─┐    rank0: [A₀+A₁+A₂+A₃]
  rank1: [A₁B₁C₁D₁] ─┤──> rank1: [B₀+B₁+B₂+B₃]
  rank2: [A₂B₂C₂D₂] ─┤    rank2: [C₀+C₁+C₂+C₃]
  rank3: [A₃B₃C₃D₃] ─┘    rank3: [D₀+D₁+D₂+D₃]

all_to_all(dim=0):
  rank0: [A₀B₀C₀D₀]    rank0: [A₀A₁A₂A₃]  (收集所有rank的chunk-0)
  rank1: [A₁B₁C₁D₁] -> rank1: [B₀B₁B₂B₃]
  rank2: [A₂B₂C₂D₂]    rank2: [C₀C₁C₂C₃]
  rank3: [A₃B₃C₃D₃]    rank3: [D₀D₁D₂D₃]
```

## 11. Ring AllReduce 算法详解

**带宽最优的 AllReduce 算法**, 通信量 $\frac{2(N-1)}{N} \times |T|$:

```
Phase 1 — Scatter-Reduce (N-1 步):
  每步: rank i 发送 chunk[(i-s) % N] 给 rank (i+1) % N
        接收方将收到的 chunk 累加到自己的对应位置
  N-1 步后: rank i 持有 chunk[(i+1) % N] 的全局规约结果

  示例 (4 ranks), Step 0:
    rank0 ──chunk0──> rank1 (rank1.chunk0 += rank0.chunk0)
    rank1 ──chunk1──> rank2
    rank2 ──chunk2──> rank3
    rank3 ──chunk3──> rank0

Phase 2 — All-Gather (N-1 步):
  已规约的 chunk 沿环传递，N-1 步后每个 rank 拥有完整结果

通信量:
  Phase 1: (N-1) 步 × |T|/N = (N-1)/N × |T|
  Phase 2: (N-1) 步 × |T|/N = (N-1)/N × |T|
  总计: 2(N-1)/N × |T|  → 与朴素 2(N-1)×|T| 相比, 减少 N 倍!
```

## 12. 原语之间的数学恒等关系

这些恒等关系在 `test_comm.py` 中逐一验证:

```
all_reduce    ≡  reduce + broadcast
              ≡  reduce_scatter + all_gather

all_gather    ≡  gather + broadcast

reduce_scatter ≡  reduce + scatter
               ≡  all_reduce 后每个 rank 取自己的 chunk

all_to_all     做两次 = 恒等 (自逆/involution)

ring_all_reduce ≡ all_reduce (结果一致, 通信模式不同)

scatter_reduce  ≡ reduce_scatter (等大小输入)
```

## 13. 大模型中的典型应用

| 并行策略 | 使用的通信原语 | 场景 |
|---------|---------------|------|
| **DDP (数据并行)** | `all_reduce` | 梯度同步: AllReduce 求和后取平均 |
| **TP 列并行** | `all_gather` | 拼接部分输出列: $[Y_0 , Y_1] = [XW_0 , XW_1]$ |
| **TP 行并行** | `all_reduce` | 求和部分结果: $Y = X_0 W_0 + X_1 W_1$ |
| **FSDP 前向** | `all_gather` | AllGather 权重分片 → 重建完整权重 |
| **FSDP 反向** | `reduce_scatter` | ReduceScatter 梯度 → 每个 rank 只保留自己的分片 |
| **MoE 专家并行** | `all_to_all` | Token dispatch: 按路由发到对应专家所在 GPU |
| **序列并行 ↔ 张量并行** | `all_gather` + `scatter` | SP [B,S/N,H] → AllGather → TP scatter → [B,S,H/N] |

## 14. test_comm.py 测试内容 (96 cases)

### 测试策略

| 类别 | 数量 | 说明 |
|------|------|------|
| 各原语基础测试 | 51 | 数值正确性, world_size=2/3/4/8, 多维张量, 边界值 |
| 恒等关系验证 | 5 | all_reduce≡reduce+broadcast, 等式见上方 |
| LLM 场景模拟 | 7 | DDP/TP/FSDP/MoE/SP 端到端通信模式 |
| 通信量分析 | 2 | Ring 带宽最优性验证 |

### 参数化维度

```python
world_size: [2, 3, 4, 8]            # 4 种分布式拓扑
tensor_shape: [(12,), (4,6), (2,3,8)]  # 1D / 2D / 3D
dim: [0, 1]                         # 不同切分维度
```

### 测试分类详情

**Broadcast**: 所有 rank 拿到相同数据, 克隆独立性, world_size=1

**Scatter**: dim=0/dim=1 切分, scatter→gather 还原恒等

**Gather**: 基础拼接, 多维, 单 rank

**AllGather**: 每个 rank 拿到完整数据, ≡ gather+broadcast

**Reduce**: 求和正确性, 已知值验证 (0+1+2+3=6)

**AllReduce**: 所有 rank 拿到求和, ≡ reduce+broadcast, DDP 梯度均值

**ReduceScatter**: ≡ reduce+scatter, FSDP 梯度分片, 2D dim=1

**All-to-All**: chunk 矩阵转置, 自逆性 (做两次=恒等), MoE dispatch

**Ring AllReduce**: ≡ naive all_reduce, world_size=1, 所有 rank 结果一致, 大张量

**Scatter-Reduce**: ≡ reduce_scatter (等大小输入), 2D

**LLM 场景**:
- `test_ddp_gradient_sync` — 4 GPU AllReduce 梯度取平均
- `test_tensor_parallel_column_linear` — 列并行: X@[W₀|W₁] + AllGather
- `test_tensor_parallel_row_linear` — 行并行: [X₀|X₁]@[W₀;W₁] + AllReduce
- `test_fsdp_forward_all_gather` — FSDP 前向: AllGather 权重分片
- `test_fsdp_backward_reduce_scatter` — FSDP 反向: ReduceScatter 梯度
- `test_moe_expert_parallel` — MoE: All-to-All dispatch → 专家计算 → All-to-All 收回
- `test_sequence_parallel_transition` — SP↔TP 转换: AllGather(seq) + scatter(hidden)

### 运行测试

```bash
cd model_basic
python -m pytest tests/test_comm.py -v --tb=short
# 或一次跑所有测试（包含 megatron_pp）
python -m pytest tests/ -q
```

---

# Megatron 并行策略 (megatron_pp/)

`model_basic/megatron_pp/` 用 **numpy 单机模拟** 分布式并行：每个"设备"对应一个 ndarray 分片，通信操作（AllReduce / AllGather / ReduceScatter）通过 Python 函数实现，数值结果与单机参考实现完全一致（atol=1e-4）。

## 模块结构

| 文件 | 功能 |
|------|------|
| `core.py` | 矩阵切分原语（`split_col/row/batch/seq`）+ 通信模拟（`all_reduce_sum`、`all_gather_*`、`broadcast`）+ 验证工具 |
| `matmul_parallel.py` | 矩阵乘 `Y = A @ B @ C` 的四种并行场景 |
| `attention_parallel.py` | Attention 层三种并行：DP / SP(Ring) / TP |
| `linear_parallel.py` | Linear 层五种并行：DP / SP / TP_Col / TP_Row / TP_ColRow |

## 矩阵并行四场景（matmul_parallel.py）

| 场景 | 切分方式 | 通信 | 适用 |
|------|----------|------|------|
| **场景一** `ScenarioA_BColumnSplit` | B 列切 | 正向 AllGather（可省） | 下游接列切分 |
| **场景二** `ScenarioB_ARowSplit`    | A 行切 | 正向 AllGather | batch/seq 拆分 |
| **场景三** `ScenarioC_AColBRowSplit`| A 列切 + B 行切 | 正向 AllReduce | 标准 TP（内积） |
| **场景四** `ScenarioD_BColCRowSplit`| B 列切 + C 行切 | 正向 AllReduce | 三矩阵 A@B@C（MLP） |

场景四等价于 Megatron MLP：`Y = X @ W₁ @ W₂`，W₁ 列切 → W₂ 行切 → 仅 1 次 AllReduce。

## Attention 并行（attention_parallel.py）

以 `[bs, heads, seq_len, head_dim]` 为例，支持 Q/K/V 投影 + SDPA + 输出投影全流程：

| 策略 | 切分维度 | 正向通信 | 反向通信 | 显存分布 |
|------|----------|----------|----------|----------|
| `AttentionDP` | batch | 0 | AllReduce(grad_W) | 权重全量 + 激活 ÷p |
| `AttentionSP` | seq（Ring Attention） | P2P × p | ReduceScatter | 权重全量 + 激活 ÷p |
| `AttentionTP` | heads（列切 W_QKV + 行切 W_O） | AllReduce(output) | AllReduce(input_grad) | 权重 ÷p + 激活全量 |

`AttentionSP` 实现了 **Ring Attention**：Q_local 不动，K/V 环形传递，配合 online softmax（log-sum-exp 递增）保证数值精确。

## Linear 层并行（linear_parallel.py）

以 `X:[bs, seq, h] @ W:[h, out] + b` 为例：

| 策略 | 切分 | 正向通信 | 适用 |
|------|------|----------|------|
| `LinearDP` | bs | 0 | 大 batch 吞吐 |
| `LinearSP` | seq | 0（token 独立） | 超长序列激活分摊 |
| `LinearTP_Col` | out（W 列切） | 0 / AllGather（可延迟） | 接 TP_Row 时零通信 |
| `LinearTP_Row` | hidden（W 行切） | AllReduce | 接在 TP_Col 之后 |
| `LinearTP_ColRow` | W₁ 列 + W₂ 行串联 | 1 次 AllReduce | Megatron MLP 完整策略 |

Megatron 经典 MLP 串联：`X @[W₁列切]→ 激活 @[W₂行切]→ AllReduce → Y`，两层共 1 次通信。

## 通信量估算（典型 LLM 维度 M=K=N=4096）

| 策略 | 通信量 |
|------|--------|
| Attention DP | 0（正向） |
| Attention SP Ring | ~128 MB × p |
| Attention TP | ~64 MB（AllReduce） |
| Linear SP | 0 |
| Linear TP_ColRow（MLP） | ~64 MB（1× AllReduce） |

## 测试内容（tests/test_megatron_pp.py，37 cases）

| 测试类 | Cases | 覆盖 |
|--------|-------|------|
| `TestMatmulParallel` | 11 | 四场景 × 多设备，含 3D 张量、MLP 仿真 |
| `TestAttentionParallel` | 9 | DP/SP/TP × 2/4 设备，Ring online softmax |
| `TestLinearParallel` | 12 | DP/SP/TP_Col/Row/ColRow × 多设备，分片形状验证 |
| `TestEdgeCases` | 4 | 1 设备退化、方阵、大值稳定性、DP/SP/TP 三路对齐 |
| `comm_report()` | — | 典型 LLM 维度通信量打印 |

```bash
cd model_basic
python -m pytest tests/test_megatron_pp.py -v --tb=short
# 37 passed
```
