"""
core.py — 基础矩阵切分原语

所有"并行"在此用 numpy 单机模拟：
  - 每个"设备"对应一个 numpy ndarray 分片
  - 通信操作（AllReduce / AllGather / ReduceScatter）用显式 Python 函数实现
  - 结果与单机参考实现数值精确一致（atol=1e-5）

约定
----
  * split_col(X, n)  → 沿最后一维切 n 份，返回 List[ndarray]
  * split_row(X, n)  → 沿倒数第二维切 n 份，返回 List[ndarray]
  * all_reduce(parts) → 求和，返回单个 ndarray（等同 AllReduce SUM）
  * all_gather_col(parts) → 沿列拼接，返回完整 ndarray
  * all_gather_row(parts) → 沿行拼接，返回完整 ndarray
"""

import numpy as np
from typing import List


# ─── 切分辅助 ────────────────────────────────────────────────────────────────

def split_col(X: np.ndarray, n: int) -> List[np.ndarray]:
    """沿最后一轴（列方向）均匀切 n 份"""
    assert X.shape[-1] % n == 0, f"列数 {X.shape[-1]} 不能被 {n} 整除"
    return np.split(X, n, axis=-1)


def split_row(X: np.ndarray, n: int) -> List[np.ndarray]:
    """沿倒数第二轴（行方向）均匀切 n 份"""
    axis = X.ndim - 2
    assert X.shape[axis] % n == 0, f"行数 {X.shape[axis]} 不能被 {n} 整除"
    return np.split(X, n, axis=axis)


def split_batch(X: np.ndarray, n: int) -> List[np.ndarray]:
    """沿第 0 轴（batch）切 n 份"""
    assert X.shape[0] % n == 0, f"batch {X.shape[0]} 不能被 {n} 整除"
    return np.split(X, n, axis=0)


def split_seq(X: np.ndarray, n: int, seq_axis: int = 1) -> List[np.ndarray]:
    """沿序列轴切 n 份（默认 axis=1）"""
    assert X.shape[seq_axis] % n == 0
    return np.split(X, n, axis=seq_axis)


# ─── 通信原语（单机模拟）────────────────────────────────────────────────────

def all_reduce_sum(parts: List[np.ndarray]) -> np.ndarray:
    """AllReduce SUM：对所有分片求和，返回完整张量（每个设备都得到相同结果）"""
    result = np.zeros_like(parts[0])
    for p in parts:
        result = result + p
    return result


def all_gather_col(parts: List[np.ndarray]) -> np.ndarray:
    """AllGather 列方向：将分片沿最后一轴拼接成完整矩阵"""
    return np.concatenate(parts, axis=-1)


def all_gather_row(parts: List[np.ndarray]) -> np.ndarray:
    """AllGather 行方向：将分片沿倒数第二轴拼接成完整矩阵"""
    return np.concatenate(parts, axis=-2)


def all_gather_batch(parts: List[np.ndarray]) -> np.ndarray:
    """AllGather batch 方向：将分片沿 axis=0 拼接"""
    return np.concatenate(parts, axis=0)


def all_gather_seq(parts: List[np.ndarray], seq_axis: int = 1) -> np.ndarray:
    """AllGather 序列方向"""
    return np.concatenate(parts, axis=seq_axis)


def reduce_scatter_row(X: np.ndarray, n: int) -> List[np.ndarray]:
    """ReduceScatter（行方向）：先对 X 做规约，再沿行切分返回各设备的分片"""
    # 在真实分布式中 X 是各设备局部结果，这里接收的是已经 reduce 过的完整张量
    return split_row(X, n)


def broadcast(X: np.ndarray, n: int) -> List[np.ndarray]:
    """广播：给 n 个设备各发一份相同副本"""
    return [X.copy() for _ in range(n)]


# ─── 数值验证工具 ─────────────────────────────────────────────────────────────

def assert_close(a: np.ndarray, b: np.ndarray, name: str = "", atol: float = 1e-5):
    """断言两个张量数值接近（用于验证并行结果与单机参考一致）"""
    if not np.allclose(a, b, atol=atol):
        diff = np.abs(a - b).max()
        raise AssertionError(f"[FAIL] {name}: max_diff={diff:.2e} > atol={atol:.2e}")
    return True


def matmul_ref(*arrays) -> np.ndarray:
    """参考实现：顺序矩阵乘（用于对比验证）"""
    result = arrays[0]
    for arr in arrays[1:]:
        result = result @ arr
    return result
