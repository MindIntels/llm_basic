"""
大模型分布式通信原语 (Communication Primitives for Parallel Operations)
=====================================================================

单进程模拟实现，用 ``list[Tensor]`` 表示各 rank 的本地数据，
无需启动多进程即可验证通信模式的正确性。

支持的通信原语
--------------
Point-to-point / basic:
    broadcast   — 根节点广播到所有 rank
    scatter     — 将数据按维度切分，分发给各 rank
    gather      — 将各 rank 的数据拼接到根节点

Collective:
    all_gather      — 每个 rank 都拿到完整拼接结果
    reduce          — 规约(求和)到根节点
    all_reduce      — 规约后每个 rank 都拿到结果
    reduce_scatter  — 先规约再分片，rank i 拿到第 i 片
    all_to_all      — 全交换，每个 rank 发/收不同的片段

Algorithmic variants:
    ring_all_reduce — Ring 拓扑带宽最优 AllReduce 算法
    scatter_reduce  — 先分片再规约 (Ring AllReduce Phase-1)

原理示意图 (4 ranks)
--------------------

broadcast(root=0):
    rank0: [ABCD] ──broadcast──> rank0: [ABCD]
                                 rank1: [ABCD]
                                 rank2: [ABCD]
                                 rank3: [ABCD]

scatter(dim=0):
    rank0: [ABCD] ──scatter──> rank0: [A]
                               rank1: [B]
                               rank2: [C]
                               rank3: [D]

gather(dim=0):
    rank0: [A] ─┐
    rank1: [B] ─┤──gather──> root: [ABCD]
    rank2: [C] ─┤
    rank3: [D] ─┘

all_gather(dim=0):
    rank0: [A] ─┐              rank0: [ABCD]
    rank1: [B] ─┤──all_gather──> rank1: [ABCD]
    rank2: [C] ─┤              rank2: [ABCD]
    rank3: [D] ─┘              rank3: [ABCD]

reduce(sum, root=0):
    rank0: [A₀] ─┐
    rank1: [A₁] ─┤──reduce──> root: [A₀+A₁+A₂+A₃]
    rank2: [A₂] ─┤
    rank3: [A₃] ─┘

all_reduce(sum):
    rank0: [A₀] ─┐              rank0: [Σ]
    rank1: [A₁] ─┤──all_reduce──> rank1: [Σ]   Σ = A₀+A₁+A₂+A₃
    rank2: [A₂] ─┤              rank2: [Σ]
    rank3: [A₃] ─┘              rank3: [Σ]

reduce_scatter(sum, dim=0):
    rank0: [A₀B₀C₀D₀] ─┐                rank0: [A₀+A₁+A₂+A₃]
    rank1: [A₁B₁C₁D₁] ─┤──reduce_scatter──> rank1: [B₀+B₁+B₂+B₃]
    rank2: [A₂B₂C₂D₂] ─┤                rank2: [C₀+C₁+C₂+C₃]
    rank3: [A₃B₃C₃D₃] ─┘                rank3: [D₀+D₁+D₂+D₃]

all_to_all(dim=0):
    rank0: [A₀B₀C₀D₀]    rank0: [A₀A₁A₂A₃]   (收集所有rank的chunk-0)
    rank1: [A₁B₁C₁D₁] -> rank1: [B₀B₁B₂B₃]   (收集所有rank的chunk-1)
    rank2: [A₂B₂C₂D₂]    rank2: [C₀C₁C₂C₃]   (收集所有rank的chunk-2)
    rank3: [A₃B₃C₃D₃]    rank3: [D₀D₁D₂D₃]   (收集所有rank的chunk-3)

ring_all_reduce(sum):
    Phase 1 — Scatter-Reduce (N-1 steps):
      每步 rank i 把 chunk[(i-s)%N] 发给 rank (i+1)%N，接收方累加
      N-1 步后每个 rank 恰好持有一个 chunk 的完整规约结果

    Phase 2 — All-Gather (N-1 steps):
      已规约的 chunk 沿环传递，N-1 步后每个 rank 拥有完整结果

    通信量: 2(N-1)/N × data_size  (带宽最优)
"""

from __future__ import annotations

import torch
from typing import List


# ===================================================================
#  Point-to-point / Basic
# ===================================================================

def broadcast(tensor: torch.Tensor, root: int = 0, world_size: int = 1) -> List[torch.Tensor]:
    """Broadcast: root rank 的数据广播到所有 rank。

    原理:
      root 持有完整数据 T，广播后每个 rank 都持有 T 的一份副本。
      不涉及任何归约或切分。
      通信量: (N-1) × |T|   (tree broadcast 可降到 log₂N 轮)

    Args:
        tensor: root rank 上的张量
        root: 源 rank (模拟中不使用，但保留接口兼容性)
        world_size: rank 总数

    Returns:
        长度为 world_size 的列表，每个元素是 tensor 的克隆
    """
    return [tensor.clone() for _ in range(world_size)]


def scatter(tensor: torch.Tensor, dim: int = 0, world_size: int = 1) -> List[torch.Tensor]:
    """Scatter: 将 tensor 沿 dim 均匀切分，分发给各 rank。

    原理:
      将 T 沿指定维度切成 N 份: T = [C₀, C₁, ..., C_{N-1}]
      rank i 收到 Cᵢ
      通信量: (N-1)/N × |T|

    Args:
        tensor: 要切分的完整张量
        dim: 切分维度
        world_size: rank 总数

    Returns:
        长度为 world_size 的列表，元素 i 是第 i 个切片
    """
    chunks = torch.chunk(tensor, world_size, dim=dim)
    return [c.contiguous() for c in chunks]


def gather(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Gather: 将各 rank 的张量沿 dim 拼接，结果只在 root。

    原理:
      rank i 将自己的 Cᵢ 发送给 root
      root 拼接: T = cat(C₀, C₁, ..., C_{N-1})
      通信量: (N-1)/N × |T|

    Args:
        tensors: 各 rank 的本地张量列表
        dim: 拼接维度

    Returns:
        拼接后的单个张量 (在 root 上)
    """
    return torch.cat(tensors, dim=dim)


# ===================================================================
#  Collective Operations
# ===================================================================

def all_gather(tensors: List[torch.Tensor], dim: int = 0) -> List[torch.Tensor]:
    """AllGather: 每个 rank 都拿到完整拼接结果。

    原理:
      等价于 gather + broadcast:
        1) 所有 rank 的 Cᵢ 拼接成完整 T
        2) 完整 T 广播给每个 rank
      但实际实现用 ring 或 recursive-doubling 更高效
      通信量: (N-1)/N × |T_full|

    等价表达:
      all_gather = gather(to root) + broadcast(from root)
      但可以在 ring 上一次完成，无需经过 root

    Args:
        tensors: 各 rank 的本地张量列表
        dim: 拼接维度

    Returns:
        长度为 world_size 的列表，每个元素都是 cat(tensors, dim)
    """
    gathered = torch.cat(tensors, dim=dim)
    return [gathered.clone() for _ in tensors]


def reduce(tensors: List[torch.Tensor], root: int = 0) -> torch.Tensor:
    """Reduce: 所有 rank 的数据规约(求和)到 root。

    原理:
      T_root = Σᵢ Tᵢ
      只有 root 持有结果
      通信量: (N-1) × |T|   (tree reduce 可降到 log₂N 轮)

    Args:
        tensors: 各 rank 的本地张量列表
        root: 目标 rank

    Returns:
        所有输入的逐元素求和
    """
    return torch.stack(tensors).sum(dim=0)


def all_reduce(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """AllReduce: 所有 rank 的数据规约(求和), 每个 rank 都拿到结果。

    原理:
      等价于 reduce + broadcast:
        1) Σ = Σᵢ Tᵢ
        2) 每个 rank 都持有 Σ
      实际用 ring_all_reduce 可做到带宽最优: 2(N-1)/N × |T|

    大模型中的应用:
      - 数据并行 (DDP): AllReduce 梯度
      - 张量并行 (RowParallel): AllReduce 部分矩阵乘结果

    Args:
        tensors: 各 rank 的本地张量列表

    Returns:
        长度为 world_size 的列表，每个元素 = Σᵢ tensors[i]
    """
    total = torch.stack(tensors).sum(dim=0)
    return [total.clone() for _ in tensors]


def reduce_scatter(tensors: List[torch.Tensor], dim: int = 0) -> List[torch.Tensor]:
    """ReduceScatter: 先规约(求和)再沿 dim 切分, rank i 拿到第 i 片。

    原理:
      1) total = Σᵢ Tᵢ           (逐元素求和)
      2) chunks = split(total)    (沿 dim 切成 N 份)
      3) rank i 收到 chunks[i]

      等价于 reduce(to root) + scatter(from root)
      但实际可以和 ring 结合，更高效

    大模型中的应用:
      - FSDP (ZeRO-3): 反向传播中 ReduceScatter 梯度，每个 rank 只保留自己负责的分片

    通信量: (N-1)/N × |T|

    Args:
        tensors: 各 rank 的本地张量列表
        dim: 切分维度

    Returns:
        长度为 world_size 的列表, 元素 i = 第 i 个切片 of Σᵢ tensors[i]
    """
    total = torch.stack(tensors).sum(dim=0)
    world_size = len(tensors)
    chunks = torch.chunk(total, world_size, dim=dim)
    return [c.contiguous() for c in chunks]


def all_to_all(tensors: List[torch.Tensor], dim: int = 0) -> List[torch.Tensor]:
    """All-to-All: 全交换, 相当于通信矩阵的转置。

    原理:
      每个 rank i 的数据被切成 N 份: Tᵢ = [Cᵢ₀, Cᵢ₁, ..., Cᵢ_{N-1}]
      rank j 收到所有 rank 的 chunk j: recv_j = cat(C₀ⱼ, C₁ⱼ, ..., C_{N-1,j})

      这等价于 send_matrix[i][j] 的转置:
        send_matrix[i][j] = chunk j of rank i's data
        recv[j] = cat(send_matrix[0][j], ..., send_matrix[N-1][j])

    大模型中的应用:
      - 专家并行 (MoE): token 路由到不同的专家所在 GPU
      - 序列并行到张量并行的转换

    通信量: (N-1)/N × |T_total|

    Args:
        tensors: 各 rank 的本地张量列表, 形状相同
        dim: 切分和拼接维度

    Returns:
        长度为 world_size 的列表, 元素 j = cat(所有 rank 的 chunk-j)
    """
    world_size = len(tensors)
    send_chunks = [torch.chunk(t, world_size, dim=dim) for t in tensors]
    result = []
    for j in range(world_size):
        pieces = [send_chunks[i][j] for i in range(world_size)]
        result.append(torch.cat(pieces, dim=dim).contiguous())
    return result


# ===================================================================
#  Algorithmic Variants
# ===================================================================

def ring_all_reduce(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Ring AllReduce: 环形拓扑上的带宽最优 AllReduce 算法。

    原理:
      将每个 rank 的数据切成 N 份，在环上分两个阶段传递:

      Phase 1 — Scatter-Reduce (N-1 步):
        ┌───────────────────────────────────────────────┐
        │  Step s: rank i 发送 chunk[(i-s) % N] 给 rank (i+1) % N  │
        │          rank (i+1) 将收到的 chunk 累加到自己的对应位置     │
        └───────────────────────────────────────────────┘
        N-1 步后, rank i 的 chunk [(i+1) % N] 持有该chunk的全局规约结果

        示例 (4 ranks, 数据=[A,B,C,D]):
          初始:  rank0=[A₀,B₀,C₀,D₀]  rank1=[A₁,B₁,C₁,D₁]  ...
          step0: rank0→rank1 发送chunk0, rank1的chunk0 += rank0的chunk0
          step1: rank1→rank2 发送chunk0(已含rank0+rank1), rank2累加
          step2: rank2→rank3, 此时rank3的chunk0 = 全局sum
          同时其他chunk也在环上传递...

      Phase 2 — All-Gather (N-1 步):
        ┌───────────────────────────────────────────────┐
        │  已规约的 chunk 沿环传递, N-1 步后每个 rank 拥有完整结果    │
        └───────────────────────────────────────────────┘

      通信量分析:
        每步每个 rank 发送 |T|/N 数据
        Phase 1: (N-1) 步 → 每个 rank 发送 (N-1)/N × |T|
        Phase 2: (N-1) 步 → 每个 rank 发送 (N-1)/N × |T|
        总计: 2(N-1)/N × |T|  → 带宽最优!

    Args:
        tensors: 各 rank 的本地张量列表

    Returns:
        长度为 world_size 的列表, 每个元素 = Σᵢ tensors[i]
    """
    world_size = len(tensors)
    if world_size == 1:
        return [tensors[0].clone()]

    # 将每个 rank 的数据切成 N 份
    chunk_lists: List[List[torch.Tensor]] = [
        list(torch.chunk(t, world_size, dim=0)) for t in tensors
    ]
    # 可写的克隆
    buffers: List[List[torch.Tensor]] = [
        [c.clone() for c in cl] for cl in chunk_lists
    ]

    # Phase 1: Scatter-Reduce (N-1 步)
    for step in range(world_size - 1):
        new_buffers = [list(b) for b in buffers]
        for rank in range(world_size):
            send_idx = (rank - step) % world_size
            right = (rank + 1) % world_size
            # 右邻居把收到的 chunk 累加到自己的对应位置
            new_buffers[right][send_idx] = (
                buffers[right][send_idx] + buffers[rank][send_idx]
            )
        buffers = new_buffers

    # Phase 2: All-Gather (N-1 步)
    for step in range(world_size - 1):
        new_buffers = [list(b) for b in buffers]
        for rank in range(world_size):
            send_idx = (rank - step + 1) % world_size
            right = (rank + 1) % world_size
            # 右邻居直接覆盖 (已经是完整规约结果)
            new_buffers[right][send_idx] = buffers[rank][send_idx].clone()
        buffers = new_buffers

    return [torch.cat(b, dim=0) for b in buffers]


def scatter_reduce(tensors: List[torch.Tensor], dim: int = 0) -> List[torch.Tensor]:
    """Scatter-Reduce: 先切分再规约 (Ring AllReduce 的 Phase 1)。

    原理:
      1) 每个 rank 的数据沿 dim 切成 N 份
      2) rank j 收到所有 rank 的 chunk j 并求和

      与 reduce_scatter 的区别:
        reduce_scatter: 先 sum 所有 rank → 再 split
        scatter_reduce: 先 split 每个 rank → 再 sum 对应 chunk
        对等大小输入, 两者结果完全相同

    Args:
        tensors: 各 rank 的本地张量列表
        dim: 切分维度

    Returns:
        长度为 world_size 的列表, 元素 j = Σᵢ chunk_j(tensors[i])
    """
    world_size = len(tensors)
    all_chunks = [torch.chunk(t, world_size, dim=dim) for t in tensors]
    result = []
    for j in range(world_size):
        reduced = torch.stack([all_chunks[i][j] for i in range(world_size)]).sum(dim=0)
        result.append(reduced.contiguous())
    return result
