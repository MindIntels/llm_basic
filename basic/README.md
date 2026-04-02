# Transformer 模型参数量 / FLOPs / 显存分析

> 基于 [知乎: 分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065) 的深度总结与代码实现

---

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

$$P_{total} = l \times 12h^2 + V \times h + 2h$$

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
$$\text{FLOPs}_{\text{train}} \approx 6 \times P \times s \times B \times \text{num\_tokens}$$

## 4. KV Cache 计算（推理）

- 每层每 token 的 KV 缓存：$2 \times h \times \text{bytes\_per\_element}$
- 总 KV Cache = $2 \times B \times l \times s \times h \times \text{dtype\_bytes}$

对于 GQA（$n_{\text{kv\_heads}} < n_{\text{heads}}$）：
$$\text{KV Cache} = 2 \times B \times l \times s \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{dtype\_bytes}$$

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
├── transformer_calc.py       # Transformer 参数量/FLOPs/显存 计算器
├── test_transformer_calc.py  # 计算器测试
├── comm.py                   # 分布式通信原语实现
└── test_comm.py              # 通信原语测试 (96 cases)
```

## 8. 使用示例

```python
from transformer_calc import TransformerConfig, TransformerCalculator

# GPT-3 175B 配置
config = TransformerConfig(
    hidden_size=12288, num_layers=96, num_heads=96,
    vocab_size=50257, seq_len=2048, batch_size=1,
)
calc = TransformerCalculator(config)
calc.print_full_report()
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
python -m pytest test_comm.py -v --tb=short
# 96 passed
```
