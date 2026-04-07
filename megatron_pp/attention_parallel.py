"""
attention_parallel.py — Attention 层的三种并行方式

输入张量: [bs, heads, seq_len, head_dim]

DP  (Data Parallel)     — 切 bs
SP  (Sequence Parallel) — 切 seq_len
TP  (Tensor Parallel)   — 切 heads

每种策略包含：
  * Q/K/V 投影
  * Scaled Dot-Product Attention
  * 输出投影
并说明各步的通信模式。
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from core import (
    split_batch, split_col, split_row, split_seq,
    all_reduce_sum, all_gather_col, all_gather_row,
    all_gather_batch, all_gather_seq, broadcast,
    assert_close
)


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Q: [..., seq_q, head_dim]
    K: [..., seq_k, head_dim]
    V: [..., seq_k, head_dim]
    返回: [..., seq_q, head_dim]
    """
    d = Q.shape[-1]
    scores = (Q @ K.swapaxes(-1, -2)) / math.sqrt(d)  # [..., seq_q, seq_k]
    if mask is not None:
        scores = scores + mask
    attn = softmax(scores, axis=-1)
    return attn @ V


@dataclass
class AttentionConfig:
    bs:       int = 2     # batch size
    heads:    int = 4     # 注意力头数
    seq_len:  int = 8     # 序列长度
    head_dim: int = 16    # 每个 head 的维度
    d_model:  int = 0     # 自动计算 = heads * head_dim

    def __post_init__(self):
        self.d_model = self.heads * self.head_dim


@dataclass
class AttentionResult:
    output:     np.ndarray           # 完整输出 [bs, seq_len, d_model]
    shards:     List[np.ndarray]     # 各设备局部输出
    comm_ops:   List[str]            # 通信操作记录
    n_devices:  int
    strategy:   str = ""


# ─── 参考实现（单机全量）────────────────────────────────────────────────────

class AttentionReference:
    """
    标准单机多头注意力（不并行），用于验证正确性。
    X: [bs, seq_len, d_model]
    权重: W_Q/W_K/W_V: [d_model, d_model], W_O: [d_model, d_model]
    """
    def __init__(self, cfg: AttentionConfig, seed: int = 42):
        rng = np.random.default_rng(seed)
        d = cfg.d_model
        self.cfg   = cfg
        self.W_Q   = rng.standard_normal((d, d)).astype(np.float32) * 0.02
        self.W_K   = rng.standard_normal((d, d)).astype(np.float32) * 0.02
        self.W_V   = rng.standard_normal((d, d)).astype(np.float32) * 0.02
        self.W_O   = rng.standard_normal((d, d)).astype(np.float32) * 0.02

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: [bs, seq_len, d_model] → [bs, seq_len, d_model]"""
        cfg = self.cfg
        bs, seq, d = X.shape

        Q = X @ self.W_Q                        # [bs, seq, d]
        K = X @ self.W_K
        V = X @ self.W_V

        # Reshape to multi-head: [bs, heads, seq, head_dim]
        def to_heads(T):
            return T.reshape(bs, seq, cfg.heads, cfg.head_dim).transpose(0, 2, 1, 3)

        Q, K, V = to_heads(Q), to_heads(K), to_heads(V)

        # Attention: [bs, heads, seq, head_dim]
        ctx = scaled_dot_product_attention(Q, K, V)

        # Merge heads: [bs, seq, d]
        ctx = ctx.transpose(0, 2, 1, 3).reshape(bs, seq, d)

        return ctx @ self.W_O


# ─── 基类 ─────────────────────────────────────────────────────────────────────

class AttentionParallelStrategy(ABC):
    def __init__(self, ref: AttentionReference, n_devices: int):
        self.ref = ref
        self.cfg = ref.cfg
        self.n_devices = n_devices

    @abstractmethod
    def forward(self, X: np.ndarray) -> AttentionResult:
        """并行前向，返回与参考相同的完整输出"""

    def verify(self, X: np.ndarray) -> bool:
        result = self.forward(X)
        ref_out = self.ref.forward(X)
        assert_close(result.output, ref_out, name=self.__class__.__name__)
        return True


# ─── DP：切 batch ─────────────────────────────────────────────────────────────

class AttentionDP(AttentionParallelStrategy):
    """
    数据并行（Data Parallel）— 切 bs

    每设备 i 持有 X_i: [bs/p, seq_len, d_model]
    完整权重 W_Q/W_K/W_V/W_O 在每设备上有副本（DP 标准做法）

    正向: 各设备独立完整计算，无需通信
    反向: 梯度 AllReduce（这里不实现反向，仅演示正向通信模式）

    通信:  正向 0 通信，反向 1 次 AllReduce（权重梯度）
    显存:  每设备 = 完整权重 + 1/p 激活
    """

    def forward(self, X: np.ndarray) -> AttentionResult:
        # 切 batch
        X_shards = split_batch(X, self.n_devices)   # [bs/p, seq, d]
        comm_ops = [f"无正向通信（DP）；反向梯度 AllReduce: {self.ref.W_Q.nbytes * 4} bytes"]

        # 每设备独立完成全量 attention
        Y_shards = [self.ref.forward(X_shards[i]) for i in range(self.n_devices)]

        # 完整输出 = AllGather batch
        Y_full = all_gather_batch(Y_shards)

        return AttentionResult(
            output=Y_full, shards=Y_shards,
            comm_ops=comm_ops, n_devices=self.n_devices, strategy="DP(batch)"
        )


# ─── SP：切 seq_len ───────────────────────────────────────────────────────────

class AttentionSP(AttentionParallelStrategy):
    """
    序列并行（Sequence Parallel）— 切 seq_len

    关键挑战: 注意力需要看到"完整序列"的 K 和 V（全局上下文）

    两种方案:
      SP-Naive  : 每设备只看 local K/V → 结果不等价（仅适合因果掩码的块对角注意力）
      SP-Ring   : Ring Attention — 每设备持有 Q_local，K/V 在设备间环形传递
                  → 结果完全等价，通信量 O(N×d)

    本实现: Ring Attention 等价模拟
      设备 i 持有 Q_i: [bs, seq/p, head_dim]
      K/V 环形传递，每设备累积计算 softmax(Q_i @ K_j^T) @ V_j
      使用 online softmax 保证数值等价
    """

    def forward(self, X: np.ndarray) -> AttentionResult:
        cfg = self.cfg
        bs, seq, d = X.shape
        p = self.n_devices
        local_seq = seq // p
        comm_ops  = []

        # ── QKV 投影（各设备对自己 seq 分片做投影，无通信）──────────────
        X_shards = split_seq(X, p, seq_axis=1)  # List[[bs, seq/p, d]]

        def proj_local(Xs, W):
            return Xs @ W   # [bs, seq/p, d]

        Q_shards = [proj_local(X_shards[i], self.ref.W_Q) for i in range(p)]
        K_shards = [proj_local(X_shards[i], self.ref.W_K) for i in range(p)]
        V_shards = [proj_local(X_shards[i], self.ref.W_V) for i in range(p)]

        # reshape to [bs, heads, seq/p, head_dim]
        def to_heads(T):
            b, s, _ = T.shape
            return T.reshape(b, s, cfg.heads, cfg.head_dim).transpose(0, 2, 1, 3)

        Q_h = [to_heads(q) for q in Q_shards]
        K_h = [to_heads(k) for k in K_shards]
        V_h = [to_heads(v) for v in V_shards]

        # ── Ring Attention：设备 i 的 Q_i 与所有 K_j/V_j 计算注意力 ─────
        # 真实 Ring 中每步将 K/V 传给下一设备
        # 这里模拟：设备 i 顺序接收所有 K/V 块并做 online softmax 累积
        comm_ops.append(
            f"Ring: {p} 步 P2P，每步传 K+V = "
            f"{2 * bs * cfg.heads * local_seq * cfg.head_dim * 4} bytes"
        )

        Y_shards = []
        for i in range(p):
            Q_i = Q_h[i]  # [bs, heads, seq/p, head_dim]

            # Online softmax 累积
            m = np.full((*Q_i.shape[:-1], 1), -np.inf, dtype=np.float32)  # 运行最大值
            s = np.zeros((*Q_i.shape[:-1], 1), dtype=np.float32)           # 归一化分母
            o = np.zeros_like(Q_i)                                          # 累积输出

            for j in range(p):
                K_j = K_h[j]  # [bs, heads, seq/p, head_dim]
                V_j = V_h[j]
                scale = 1.0 / math.sqrt(cfg.head_dim)

                # scores_ij: [bs, heads, seq/p_i, seq/p_j]
                scores_ij = Q_i @ K_j.swapaxes(-1, -2) * scale

                # Online softmax 更新
                m_new = np.maximum(m, scores_ij.max(axis=-1, keepdims=True))
                e_new = np.exp(scores_ij - m_new)        # 重新归一化当前块
                s_new = np.exp(m - m_new) * s + e_new.sum(axis=-1, keepdims=True)
                o     = np.exp(m - m_new) * o + e_new @ V_j
                m, s  = m_new, s_new

            Y_i = o / s   # [bs, heads, seq/p, head_dim]

            # merge heads → [bs, seq/p, d]
            Y_i = Y_i.transpose(0, 2, 1, 3).reshape(bs, local_seq, d)

            # 输出投影（各设备对局部 seq 做，无通信）
            Y_i = Y_i @ self.ref.W_O
            Y_shards.append(Y_i)

        # AllGather seq 方向
        Y_full = all_gather_seq(Y_shards, seq_axis=1)
        comm_ops.append(f"AllGather(seq): {Y_full.nbytes} bytes")

        return AttentionResult(
            output=Y_full, shards=Y_shards,
            comm_ops=comm_ops, n_devices=self.n_devices, strategy="SP(seq_len/Ring)"
        )


# ─── TP：切 heads ─────────────────────────────────────────────────────────────

class AttentionTP(AttentionParallelStrategy):
    """
    张量并行（Tensor Parallel）— 切 heads

    每设备 i 持有 heads_i = heads / p 个注意力 head

    W_Q/W_K/W_V 列切分（按 head 对应的列）：
      W_Q_i: [d_model, d_model/p]（负责 heads/p 个 head）

    W_O 行切分：
      W_O_i: [d_model/p, d_model]

    每设备 i:
      Q_i = X @ W_Q_i    [bs, seq, d/p]
      K_i = X @ W_K_i
      V_i = X @ W_V_i
      ctx_i = Attention(Q_i, K_i, V_i)  [bs, seq, d/p]（本地完整注意力）
      Y_i   = ctx_i @ W_O_i             [bs, seq, d]（局部贡献）

    AllReduce: Y = Σ Y_i

    通信: 1 次 AllReduce
    """

    def forward(self, X: np.ndarray) -> AttentionResult:
        cfg = self.cfg
        bs, seq, d = X.shape
        p = self.n_devices
        comm_ops = []

        # 列切 W_Q/W_K/W_V（按 head 维度）
        W_Q_shards = split_col(self.ref.W_Q, p)   # List[[d, d/p]]
        W_K_shards = split_col(self.ref.W_K, p)
        W_V_shards = split_col(self.ref.W_V, p)
        # 行切 W_O
        W_O_shards = split_row(self.ref.W_O, p)   # List[[d/p, d]]

        local_d    = d // p
        local_heads = cfg.heads // p

        Y_shards = []
        for i in range(p):
            # 投影（无通信，X 在每设备上有副本）
            Q_i = X @ W_Q_shards[i]   # [bs, seq, d/p]
            K_i = X @ W_K_shards[i]
            V_i = X @ W_V_shards[i]

            # reshape 到多头 [bs, local_heads, seq, head_dim]
            def to_heads(T):
                return T.reshape(bs, seq, local_heads, cfg.head_dim).transpose(0, 2, 1, 3)

            Q_h = to_heads(Q_i)
            K_h = to_heads(K_i)
            V_h = to_heads(V_i)

            # 本地注意力计算（只负责 local_heads 个 head）
            ctx_h = scaled_dot_product_attention(Q_h, K_h, V_h)  # [bs, local_heads, seq, head_dim]

            # merge heads → [bs, seq, d/p]
            ctx_i = ctx_h.transpose(0, 2, 1, 3).reshape(bs, seq, local_d)

            # 输出投影：局部贡献 [bs, seq, d]
            Y_i = ctx_i @ W_O_shards[i]
            Y_shards.append(Y_i)

        # AllReduce
        Y_full = all_reduce_sum(Y_shards)
        comm_ops.append(f"AllReduce(sum): {Y_full.nbytes} bytes")

        return AttentionResult(
            output=Y_full, shards=Y_shards,
            comm_ops=comm_ops, n_devices=self.n_devices, strategy="TP(heads)"
        )
