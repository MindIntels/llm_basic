"""
linear_parallel.py — 线性层的三种并行方式

输入: [bs, seq_len, hidden_size]

DP  — 切 bs
SP  — 切 seq_len
TP  — 切 hidden_size（列切分 W 或行切分 W）
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from abc import ABC, abstractmethod
from core import (
    split_batch, split_col, split_row, split_seq,
    all_reduce_sum, all_gather_col, all_gather_row,
    all_gather_batch, all_gather_seq, broadcast,
    assert_close
)


@dataclass
class LinearConfig:
    bs:          int = 2
    seq_len:     int = 8
    hidden_size: int = 32
    out_size:    int = 64


@dataclass
class LinearResult:
    output:    np.ndarray
    shards:    List[np.ndarray]
    comm_ops:  List[str]
    n_devices: int
    strategy:  str = ""


class LinearReference:
    """单机参考实现"""
    def __init__(self, cfg: LinearConfig, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((cfg.hidden_size, cfg.out_size)).astype(np.float32) * 0.02
        self.b = rng.standard_normal((cfg.out_size,)).astype(np.float32) * 0.01
        self.cfg = cfg

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: [bs, seq, hidden] → [bs, seq, out]"""
        return X @ self.W + self.b


class LinearParallelStrategy(ABC):
    def __init__(self, ref: LinearReference, n_devices: int):
        self.ref = ref
        self.cfg = ref.cfg
        self.n_devices = n_devices

    @abstractmethod
    def forward(self, X: np.ndarray) -> LinearResult:
        pass

    def verify(self, X: np.ndarray) -> bool:
        result = self.forward(X)
        ref_out = self.ref.forward(X)
        assert_close(result.output, ref_out, name=self.__class__.__name__)
        return True


# ─── DP：切 bs ────────────────────────────────────────────────────────────────

class LinearDP(LinearParallelStrategy):
    """
    数据并行 — 切 bs

    X_i: [bs/p, seq, hidden]
    W 完整副本在每设备

    正向: 零通信（各设备独立计算）
    反向: 梯度 AllReduce（权重梯度求和）

    适合: batch 较大时，线性扩展吞吐量
    """

    def forward(self, X: np.ndarray) -> LinearResult:
        X_shards = split_batch(X, self.n_devices)
        Y_shards = [X_shards[i] @ self.ref.W + self.ref.b for i in range(self.n_devices)]
        Y_full   = all_gather_batch(Y_shards)
        comm_ops = [
            "正向: 0 通信",
            f"反向: AllReduce(grad_W) = {self.ref.W.nbytes} bytes"
        ]
        return LinearResult(
            output=Y_full, shards=Y_shards,
            comm_ops=comm_ops, n_devices=self.n_devices, strategy="DP(batch)"
        )


# ─── SP：切 seq_len ───────────────────────────────────────────────────────────

class LinearSP(LinearParallelStrategy):
    """
    序列并行 — 切 seq_len

    X_i: [bs, seq/p, hidden]
    W 完整副本（或 TP 切分，见 SP+TP 组合）

    线性层的 token 之间完全独立（无 attention），因此 SP 切分完全无通信

    正向: 零通信
    反向: AllReduce(grad_W) 或 ReduceScatter（SP+TP 组合时）

    适合: 超长序列，激活显存 ÷ p
    """

    def forward(self, X: np.ndarray) -> LinearResult:
        X_shards = split_seq(X, self.n_devices, seq_axis=1)
        Y_shards = [X_shards[i] @ self.ref.W + self.ref.b for i in range(self.n_devices)]
        Y_full   = all_gather_seq(Y_shards, seq_axis=1)
        comm_ops = [
            "正向: 0 通信（token 独立，seq 切分完全无依赖）",
            f"反向: AllReduce(grad_W) = {self.ref.W.nbytes} bytes"
        ]
        return LinearResult(
            output=Y_full, shards=Y_shards,
            comm_ops=comm_ops, n_devices=self.n_devices, strategy="SP(seq_len)"
        )


# ─── TP 列切分（Column Parallel）──────────────────────────────────────────────

class LinearTP_Col(LinearParallelStrategy):
    """
    张量并行列切分 — 切 out_size

    W → [W₀|W₁|...|Wₚ₋₁]  每份 W_i:[hidden, out/p]
    b → [b₀|b₁|...|bₚ₋₁]

    每设备 i:
      Y_i = X @ W_i + b_i    [bs, seq, out/p]  无通信

    汇总（若需完整输出）:
      Y = AllGather(Y₀, ...) 沿列 → [bs, seq, out]

    适合: 后接行切分层（Megatron 列→行串联），这时不需要 AllGather
    """

    def forward(self, X: np.ndarray) -> LinearResult:
        W_shards = split_col(self.ref.W, self.n_devices)
        b_shards = np.array_split(self.ref.b, self.n_devices)

        Y_shards = [
            X @ W_shards[i] + b_shards[i]
            for i in range(self.n_devices)
        ]
        Y_full   = all_gather_col(Y_shards)
        comm_ops = [
            "正向: 0 通信（局部输出，不需要通信）",
            f"AllGather(col)（如需完整输出）= {Y_full.nbytes} bytes",
            "若下接行切分层，局部输出直接传入，总通信 0"
        ]
        return LinearResult(
            output=Y_full, shards=Y_shards,
            comm_ops=comm_ops, n_devices=self.n_devices, strategy="TP_Col(out)"
        )


# ─── TP 行切分（Row Parallel）─────────────────────────────────────────────────

class LinearTP_Row(LinearParallelStrategy):
    """
    张量并行行切分 — 切 hidden_size

    输入 X 已经是局部的（来自上一列切分层）
    W → [W₀;W₁;..;Wₚ₋₁]  每份 W_i:[hidden/p, out]

    每设备 i:
      P_i = X_i @ W_i          [bs, seq, out]  局部贡献（X_i 已经是局部的）

    汇总:
      Y = AllReduce SUM(P₀,..) + b   [bs, seq, out]

    通信: 1 次 AllReduce
    适合: 接在列切分层之后（Megatron MLP: 列→行）
    """

    def forward(self, X: np.ndarray) -> LinearResult:
        # X 已经是列切分输出，这里模拟：手动切 X 的最后维度
        X_shards = split_col(X, self.n_devices)   # 模拟接收来自列切分的局部激活
        W_shards = split_row(self.ref.W, self.n_devices)

        P_shards = [X_shards[i] @ W_shards[i] for i in range(self.n_devices)]
        Y_full   = all_reduce_sum(P_shards) + self.ref.b
        comm_ops = [f"AllReduce(sum): {Y_full.nbytes} bytes"]

        return LinearResult(
            output=Y_full, shards=[Y_full.copy() for _ in range(self.n_devices)],
            comm_ops=comm_ops, n_devices=self.n_devices, strategy="TP_Row(hidden)"
        )


# ─── TP 列→行串联（Megatron MLP）─────────────────────────────────────────────
# Megatron-LM 对 MLP 层采用 列并行（FC1）+ 行并行（FC2） 的策略
'''
├─────────────────────────────────────────────────────────────┤
|                                                             |
│                         MLP Layer                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input X [S, B, H]                                          │
│    │                                                        |
│    │  (Identity in forward, All-Reduce in backward)         │
│    ↓                                                        │
│  ┌──────────────────────────────────────────┐               │
│  │         FC1: Column Parallel              │              │
│  │  Weight: [4H/p, H] on each GPU           │               │
│  │  Output: [S, B, 4H/p] on each GPU        │               │
│  └──────────────────────────────────────────┘               │
│    │                                                        │
│    │  (No communication)                                    │
│    ↓                                                        │
│  ┌──────────────────────────────────────────┐               │
│  │         Activation Function               │              │
│  │  (GeLU, SwiGLU, etc.)                    │               │
│  └──────────────────────────────────────────┘               │
│    │                                                        │
│    │  (No communication)                                    │
│    ↓                                                        │
│  ┌──────────────────────────────────────────┐               │
│  │         FC2: Row Parallel                 │              │
│  │  Weight: [H, 4H/p] on each GPU           │               │
│  │  Output: [S, B, H] (partial) on each GPU │               │
│  └──────────────────────────────────────────┘               │
│    │                                                        │
│    │  All-Reduce (SUM) in forward                           │
│    ↓                                                        │
│  Output Y [S, B, H]                                         │
├─────────────────────────────────────────────────────────────┤                                                         
'''

class LinearTP_ColRow(LinearParallelStrategy):
    """
    列切 W₁ + 行切 W₂（两层串联，整体只需 1 次 AllReduce）

    等同于 ScenarioD：
      Y = X @ W₁ @ W₂
      W₁ 列切，W₂ 行切
      → 1 次 AllReduce
    """

    def __init__(self, ref: LinearReference, n_devices: int,
                 W2: np.ndarray, b2: np.ndarray):
        super().__init__(ref, n_devices)
        self.W2 = W2
        self.b2 = b2

    def forward(self, X: np.ndarray) -> LinearResult:
        W1_shards = split_col(self.ref.W, self.n_devices)
        b1_shards = np.array_split(self.ref.b, self.n_devices)
        W2_shards = split_row(self.W2, self.n_devices)

        Y_shards = []
        for i in range(self.n_devices):
            H_i = X @ W1_shards[i] + b1_shards[i]  # 列切，无通信
            H_i = np.maximum(H_i, 0)                 # ReLU（可替换为 GeLU）
            Y_i = H_i @ W2_shards[i]                 # 行切，局部贡献
            Y_shards.append(Y_i)

        Y_full   = all_reduce_sum(Y_shards) + self.b2
        comm_ops = [
            "W1 列切: 0 通信",
            f"W2 行切 AllReduce: {Y_full.nbytes} bytes",
            "整体只有 1 次 AllReduce"
        ]
        return LinearResult(
            output=Y_full, shards=[Y_full.copy() for _ in range(self.n_devices)],
            comm_ops=comm_ops, n_devices=self.n_devices, strategy="TP_ColRow(MLP)"
        )

    def verify(self, X: np.ndarray) -> bool:
        result = self.forward(X)
        # 参考：两层串联
        H = np.maximum(X @ self.ref.W + self.ref.b, 0)
        ref_out = H @ self.W2 + self.b2
        assert_close(result.output, ref_out, name=self.__class__.__name__)
        return True
