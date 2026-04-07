"""
matmul_parallel.py — ABC 矩阵运算的四种并行切分

目标计算: Y = A @ B @ C    (或 Y = A @ B 的变体)

场景一: B 矩阵列切   (Column Split on B)
场景二: A 矩阵行切   (Row Split on A)
场景三: A 列切 + B 行切  (Column-A × Row-B，结果局部，需 AllReduce)
场景四: B 列切 + C 行切  (Column-B × Row-C，用于三矩阵 A@B@C)

每个场景均提供:
  - ParallelStrategy 基类
  - 具体子类实现 forward()
  - 通信量 comm_volume() 估算
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
from core import (
    split_col, split_row, broadcast,
    all_reduce_sum, all_gather_col, all_gather_row,
    all_gather_batch, assert_close, matmul_ref
)


# ─── 基类 ─────────────────────────────────────────────────────────────────────

@dataclass
class ParallelResult:
    output:        np.ndarray          # 完整输出（用于验证）
    shards:        List[np.ndarray]    # 各设备局部结果
    comm_ops:      List[str]           # 通信操作记录
    n_devices:     int


class MatmulParallelStrategy(ABC):
    """所有矩阵并行策略的基类"""

    def __init__(self, n_devices: int):
        self.n_devices = n_devices

    @abstractmethod
    def forward(self, *matrices: np.ndarray) -> ParallelResult:
        """执行并行矩阵乘，返回结果"""

    @abstractmethod
    def comm_volume(self, *shapes) -> Dict[str, int]:
        """估算通信量（字节数），用于对比分析"""

    def verify(self, result: ParallelResult, *matrices: np.ndarray) -> bool:
        ref = matmul_ref(*matrices)
        assert_close(result.output, ref, name=self.__class__.__name__)
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}(n_devices={self.n_devices})"


# ─── 场景一: B 矩阵列切（Column Split on B）──────────────────────────────────

class ScenarioA_BColumnSplit(MatmulParallelStrategy):
    """
    目标: Y = A @ B       A:[M,K], B:[K,N]

    切分:
      B → [B₀|B₁|...|Bₚ₋₁]  每份 B_i:[K, N/p]
      A 广播到所有设备

    每设备 i:
      Y_i = A @ B_i           [M, N/p]  ← 无通信

    汇总:
      Y = AllGather(Y₀, Y₁, ...) 沿列拼接 → [M, N]
      或各设备保留局部输出（无需汇总，下游列切分时对接）

    通信: 1 次 AllGather（如需完整结果）
    """

    def forward(self, A: np.ndarray, B: np.ndarray) -> ParallelResult:
        comm_ops = []

        # 广播 A（实际中 A 本来就在每张卡上）
        A_copies = broadcast(A, self.n_devices)

        # 切分 B
        B_shards = split_col(B, self.n_devices)

        # 每设备独立计算：Y_i = A @ B_i
        Y_shards = [A_copies[i] @ B_shards[i] for i in range(self.n_devices)]

        # AllGather 得完整结果
        Y_full = all_gather_col(Y_shards)
        comm_ops.append(f"AllGather(col): {sum(s.nbytes for s in Y_shards)} bytes")

        return ParallelResult(
            output=Y_full, shards=Y_shards,
            comm_ops=comm_ops, n_devices=self.n_devices
        )

    def comm_volume(self, M: int, K: int, N: int, dtype_bytes: int = 4) -> Dict[str, int]:
        return {
            "AllGather(output)": M * N * dtype_bytes,
            "total": M * N * dtype_bytes,
            "note": "正向 0 通信（各设备保留局部），汇总时 AllGather"
        }


# ─── 场景二: A 矩阵行切（Row Split on A）─────────────────────────────────────

class ScenarioB_ARowSplit(MatmulParallelStrategy):
    """
    目标: Y = A @ B       A:[M,K], B:[K,N]

    切分:
      A → [A₀; A₁; ...; Aₚ₋₁]  每份 A_i:[M/p, K]
      B 广播到所有设备

    每设备 i:
      Y_i = A_i @ B              [M/p, N]  ← 无通信

    汇总:
      Y = AllGather(Y₀, Y₁, ...) 沿行拼接 → [M, N]

    通信: 1 次 AllGather（行方向）
    适用: A 是激活（按 batch 或 seq 切），B 是共享权重
    """

    def forward(self, A: np.ndarray, B: np.ndarray) -> ParallelResult:
        comm_ops = []

        # 切分 A（行）
        A_shards = split_row(A, self.n_devices)

        # 广播 B
        B_copies = broadcast(B, self.n_devices)

        # 每设备独立计算
        Y_shards = [A_shards[i] @ B_copies[i] for i in range(self.n_devices)]

        # AllGather 行方向
        Y_full = all_gather_row(Y_shards)
        comm_ops.append(f"AllGather(row): {sum(s.nbytes for s in Y_shards)} bytes")

        return ParallelResult(
            output=Y_full, shards=Y_shards,
            comm_ops=comm_ops, n_devices=self.n_devices
        )

    def comm_volume(self, M: int, K: int, N: int, dtype_bytes: int = 4) -> Dict[str, int]:
        return {
            "AllGather(output)": M * N * dtype_bytes,
            "total": M * N * dtype_bytes,
            "note": "适合 A 是 batch/seq 切分的激活，B 是权重"
        }


# ─── 场景三: A 列切 + B 行切（内积累加模式）──────────────────────────────────

class ScenarioC_AColBRowSplit(MatmulParallelStrategy):
    """
    目标: Y = A @ B       A:[M,K], B:[K,N]

    切分:
      A → [A₀|A₁|...|Aₚ₋₁]  沿 K 轴列切  每份 A_i:[M, K/p]
      B → [B₀; B₁; ...; Bₚ₋₁]  沿 K 轴行切  每份 B_i:[K/p, N]

    等价性: A @ B = Σᵢ A_i @ B_i  （外积累加，矩阵乘的分块性质）

    每设备 i:
      P_i = A_i @ B_i              [M, N]  局部贡献

    汇总:
      Y = AllReduce SUM(P₀, P₁, ...) → [M, N]
          每设备都得到完整的 Y

    通信: 1 次 AllReduce（数据量 M×N，最大的通信）
    适用: Megatron-LM 的列切 W₁ + 行切 W₂（MLP TP）
    """

    def forward(self, A: np.ndarray, B: np.ndarray) -> ParallelResult:
        comm_ops = []

        # A 列切（沿 K 轴）
        A_shards = split_col(A, self.n_devices)   # [M, K/p]

        # B 行切（沿 K 轴）
        B_shards = split_row(B, self.n_devices)   # [K/p, N]

        # 每设备：局部外积贡献
        P_shards = [A_shards[i] @ B_shards[i] for i in range(self.n_devices)]

        # AllReduce 求和
        Y_full = all_reduce_sum(P_shards)
        comm_ops.append(f"AllReduce(sum): {Y_full.nbytes} bytes")

        # 每设备都持有完整 Y（AllReduce 后）
        Y_shards = [Y_full.copy() for _ in range(self.n_devices)]

        return ParallelResult(
            output=Y_full, shards=Y_shards,
            comm_ops=comm_ops, n_devices=self.n_devices
        )

    def comm_volume(self, M: int, K: int, N: int, dtype_bytes: int = 4) -> Dict[str, int]:
        return {
            "AllReduce(output)": M * N * dtype_bytes,
            "total": M * N * dtype_bytes,
            "note": "输出需要 AllReduce，适合 MLP 的列切 W1 → 行切 W2 中间层"
        }


# ─── 场景四: B 列切 + C 行切（三矩阵 A@B@C）──────────────────────────────────

class ScenarioD_BColCRowSplit(MatmulParallelStrategy):
    """
    目标: Y = A @ B @ C      A:[M,K], B:[K,H], C:[H,N]

    切分:
      B → [B₀|B₁|...|Bₚ₋₁]  列切  每份 B_i:[K, H/p]
      C → [C₀; C₁; ...; Cₚ₋₁]  行切  每份 C_i:[H/p, N]

    每设备 i:
      T_i = A @ B_i              [M, H/p]   ← B 列切，无通信
      Y_i = T_i @ C_i            [M, N]     ← C 行切，局部贡献

    汇总:
      Y = AllReduce SUM(Y₀, Y₁, ...) → [M, N]

    等价性:
      A@B@C = A @ (B@C) = A @ Σᵢ(B_i@C_i) = Σᵢ(A@B_i@C_i)

    通信: 仅 1 次 AllReduce（正向），与单层 A@B 的 AllReduce 相同
    适用: Transformer MLP（A=激活, B=W₁列切, C=W₂行切）
    """

    def forward(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> ParallelResult:
        comm_ops = []

        # B 列切，C 行切
        B_shards = split_col(B, self.n_devices)  # [K, H/p]
        C_shards = split_row(C, self.n_devices)  # [H/p, N]
        A_copies = broadcast(A, self.n_devices)

        # 每设备两步计算
        Y_shards = []
        for i in range(self.n_devices):
            T_i = A_copies[i] @ B_shards[i]   # [M, H/p]，无通信
            Y_i = T_i @ C_shards[i]            # [M, N]，局部贡献
            Y_shards.append(Y_i)

        # AllReduce
        Y_full = all_reduce_sum(Y_shards)
        comm_ops.append(f"AllReduce(sum): {Y_full.nbytes} bytes")

        return ParallelResult(
            output=Y_full, shards=Y_shards,
            comm_ops=comm_ops, n_devices=self.n_devices
        )

    def comm_volume(self, M: int, K: int, H: int, N: int, dtype_bytes: int = 4) -> Dict[str, int]:
        return {
            "AllReduce(output)": M * N * dtype_bytes,
            "total": M * N * dtype_bytes,
            "note": "三矩阵整体只需 1 次 AllReduce，等同于两矩阵版本"
        }
