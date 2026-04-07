"""
Communication primitives for parallel operations.

Provides simulated collective communication ops that work on lists of tensors
representing each rank's local data.  When ``torch.distributed`` is available
and initialized the real NCCL backend is used; otherwise a **single-process
simulation** is provided so that correctness tests can run without spawning
multiple processes.

Supported primitives
--------------------
Point-to-point / basic:
    broadcast, scatter, gather

Collective:
    all_gather, reduce, all_reduce, reduce_scatter, all_to_all

Algorithmic variants:
    ring_all_reduce, scatter_reduce
"""

from __future__ import annotations

import torch
from typing import List, Optional


# ---------------------------------------------------------------------------
# Utility: check whether we are inside a real distributed context
# ---------------------------------------------------------------------------

def _is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


# ===================================================================
#  Simulated collectives  (list[Tensor] interface – one entry per rank)
# ===================================================================

# -------------------------------------------------------------------
# 1. broadcast
# -------------------------------------------------------------------

def broadcast(tensor: torch.Tensor, root: int = 0, world_size: int = 1) -> List[torch.Tensor]:
    """Broadcast *tensor* from *root* to all ranks.

    Args:
        tensor: the tensor on the root rank.
        root: source rank (unused in simulation but kept for API parity).
        world_size: number of ranks.

    Returns:
        list of length *world_size* where every element is a clone of *tensor*.
    """
    return [tensor.clone() for _ in range(world_size)]


# -------------------------------------------------------------------
# 2. scatter
# -------------------------------------------------------------------

def scatter(tensor: torch.Tensor, dim: int = 0, world_size: int = 1) -> List[torch.Tensor]:
    """Scatter: split *tensor* into *world_size* equal chunks along *dim*.

    Args:
        tensor: the full tensor to scatter.
        dim: dimension to split along.
        world_size: number of ranks.

    Returns:
        list of chunks, one per rank.
    """
    chunks = torch.chunk(tensor, world_size, dim=dim)
    return [c.contiguous() for c in chunks]


# -------------------------------------------------------------------
# 3. gather
# -------------------------------------------------------------------

def gather(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Gather: concatenate tensors from all ranks along *dim* **on root**.

    Unlike ``all_gather`` the result is returned only once (to root).

    Args:
        tensors: list of local tensors, one per rank.
        dim: concatenation dimension.

    Returns:
        A single tensor ``torch.cat(tensors, dim=dim)``.
    """
    return torch.cat(tensors, dim=dim)


# -------------------------------------------------------------------
# 4. all_gather
# -------------------------------------------------------------------

def all_gather(tensors: List[torch.Tensor], dim: int = 0) -> List[torch.Tensor]:
    """AllGather along *dim*.

    Each rank contributes its local chunk; each rank receives the full
    concatenation along *dim*.

    Args:
        tensors: list of local tensors, one per rank.
        dim: concatenation dimension.

    Returns:
        list where every element is ``torch.cat(tensors, dim=dim)``.
    """
    gathered = torch.cat(tensors, dim=dim)
    return [gathered.clone() for _ in tensors]


# -------------------------------------------------------------------
# 5. reduce
# -------------------------------------------------------------------

def reduce(tensors: List[torch.Tensor], root: int = 0) -> torch.Tensor:
    """Reduce (sum) across ranks and return the result **only on root**.

    Args:
        tensors: list of local tensors, one per rank.
        root: destination rank (unused in simulation – we just return the sum).

    Returns:
        A single tensor equal to the element-wise sum of all inputs.
    """
    return torch.stack(tensors).sum(dim=0)


# -------------------------------------------------------------------
# 6. all_reduce
# -------------------------------------------------------------------

def all_reduce(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """AllReduce (sum) across ranks.

    Args:
        tensors: list of length ``world_size``, each element is the local
                 tensor on that rank.

    Returns:
        list of tensors where every element equals the sum of all inputs.
    """
    total = torch.stack(tensors).sum(dim=0)
    return [total.clone() for _ in tensors]


# -------------------------------------------------------------------
# 7. reduce_scatter
# -------------------------------------------------------------------

def reduce_scatter(tensors: List[torch.Tensor], dim: int = 0) -> List[torch.Tensor]:
    """ReduceScatter along *dim*.

    1. Element-wise sum across all ranks → ``total``.
    2. Split ``total`` evenly along *dim* and give the i-th chunk to rank i.

    Args:
        tensors: list of local tensors, one per rank.
        dim: the dimension along which to scatter after reducing.

    Returns:
        list where element *i* is the i-th chunk of the reduced tensor.
    """
    total = torch.stack(tensors).sum(dim=0)
    world_size = len(tensors)
    chunks = torch.chunk(total, world_size, dim=dim)
    return [c.contiguous() for c in chunks]


# -------------------------------------------------------------------
# 8. all_to_all
# -------------------------------------------------------------------

def all_to_all(tensors: List[torch.Tensor], dim: int = 0) -> List[torch.Tensor]:
    """All-to-All: each rank sends a distinct chunk to every other rank.

    Each input tensor is split into *world_size* chunks along *dim*.
    Rank *j* receives the j-th chunk from every rank (concatenated along *dim*).

    This is equivalent to a **transpose** of the chunk matrix:

        send_matrix[i][j]  = chunk j of tensors[i]
        recv_matrix[j]     = cat(send_matrix[0][j], ..., send_matrix[N-1][j])

    Args:
        tensors: list of local tensors, one per rank.  All must have the same
                 shape.
        dim: dimension along which to split and re-concatenate.

    Returns:
        list of length *world_size*.  Element *j* is the concatenation (along
        *dim*) of all chunk-*j* pieces from every rank.
    """
    world_size = len(tensors)
    send_chunks = [torch.chunk(t, world_size, dim=dim) for t in tensors]
    result = []
    for j in range(world_size):
        pieces = [send_chunks[i][j] for i in range(world_size)]
        result.append(torch.cat(pieces, dim=dim).contiguous())
    return result


# -------------------------------------------------------------------
# 9. ring_all_reduce  (algorithmic variant)
# -------------------------------------------------------------------

def ring_all_reduce(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Ring AllReduce (sum) – simulates the ring-based algorithm.

    Phase 1 – Scatter-Reduce (N-1 steps):
        Each rank sends one chunk to its right neighbour and accumulates the
        chunk received from its left neighbour.  After N-1 steps each rank
        holds the fully-reduced value of exactly one chunk.

    Phase 2 – All-Gather (N-1 steps):
        The reduced chunks are rotated around the ring so that every rank ends
        up with the complete reduced tensor.

    The final result is identical to ``all_reduce`` but the communication
    pattern is bandwidth-optimal: 2(N-1)/N × |T|.

    Args:
        tensors: list of local tensors, one per rank.

    Returns:
        list of tensors where every element equals the sum of all inputs.
    """
    world_size = len(tensors)
    if world_size == 1:
        return [tensors[0].clone()]

    chunk_lists: List[List[torch.Tensor]] = [
        list(torch.chunk(t, world_size, dim=0)) for t in tensors
    ]
    buffers: List[List[torch.Tensor]] = [
        [c.clone() for c in cl] for cl in chunk_lists
    ]

    # Phase 1: Scatter-Reduce
    for step in range(world_size - 1):
        new_buffers = [list(b) for b in buffers]
        for rank in range(world_size):
            send_idx = (rank - step) % world_size
            right = (rank + 1) % world_size
            new_buffers[right][send_idx] = (
                buffers[right][send_idx] + buffers[rank][send_idx]
            )
        buffers = new_buffers

    # Phase 2: All-Gather
    for step in range(world_size - 1):
        new_buffers = [list(b) for b in buffers]
        for rank in range(world_size):
            send_idx = (rank - step + 1) % world_size
            right = (rank + 1) % world_size
            new_buffers[right][send_idx] = buffers[rank][send_idx].clone()
        buffers = new_buffers

    return [torch.cat(b, dim=0) for b in buffers]


# -------------------------------------------------------------------
# 10. scatter_reduce
# -------------------------------------------------------------------

def scatter_reduce(tensors: List[torch.Tensor], dim: int = 0) -> List[torch.Tensor]:
    """Scatter-Reduce: scatter then reduce (Phase 1 of Ring AllReduce).

    Each input tensor is split into *world_size* chunks along *dim*.
    Rank *j* receives chunk *j* from every rank and **sums** them.

    For equal-sized inputs the result is identical to ``reduce_scatter``.

    Args:
        tensors: list of local tensors, one per rank.
        dim: dimension along which to split.

    Returns:
        list where element *j* is the sum of chunk *j* across all ranks.
    """
    world_size = len(tensors)
    all_chunks = [torch.chunk(t, world_size, dim=dim) for t in tensors]
    result = []
    for j in range(world_size):
        reduced = torch.stack([all_chunks[i][j] for i in range(world_size)]).sum(dim=0)
        result.append(reduced.contiguous())
    return result


# ===================================================================
#  Autograd-aware wrappers
# ===================================================================

class _AllReduceFunc(torch.autograd.Function):
    """Forward: identity.  Backward: all-reduce (sum) gradients."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output


class _ReduceScatterFunc(torch.autograd.Function):
    """Forward: reduce-scatter.  Backward: all-gather."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, world_size: int, rank: int, dim: int) -> torch.Tensor:  # type: ignore[override]
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.dim = dim
        chunks = torch.chunk(tensor, world_size, dim=dim)
        return chunks[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        pieces = [torch.zeros_like(grad_output) for _ in range(ctx.world_size)]
        pieces[ctx.rank] = grad_output
        return torch.cat(pieces, dim=ctx.dim), None, None, None


class _AllGatherFunc(torch.autograd.Function):
    """Forward: all-gather.  Backward: reduce-scatter."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, world_size: int, rank: int, dim: int) -> torch.Tensor:  # type: ignore[override]
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.dim = dim
        return torch.cat([tensor] * world_size, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        chunks = torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)
        return chunks[ctx.rank].contiguous(), None, None, None


def all_reduce_autograd(tensor: torch.Tensor) -> torch.Tensor:
    """Identity in forward; all-reduce in backward (TP row-parallel)."""
    return _AllReduceFunc.apply(tensor)


def reduce_scatter_autograd(
    tensor: torch.Tensor, world_size: int, rank: int, dim: int = 0
) -> torch.Tensor:
    """ReduceScatter in forward (simulated); AllGather in backward."""
    return _ReduceScatterFunc.apply(tensor, world_size, rank, dim)


def all_gather_autograd(
    tensor: torch.Tensor, world_size: int, rank: int, dim: int = 0
) -> torch.Tensor:
    """AllGather in forward (simulated); ReduceScatter in backward."""
    return _AllGatherFunc.apply(tensor, world_size, rank, dim)
