"""comm package – distributed communication primitives (single-process simulation)."""

from .comm import (
    all_gather,
    all_gather_autograd,
    all_reduce,
    all_reduce_autograd,
    all_to_all,
    broadcast,
    gather,
    reduce,
    reduce_scatter,
    reduce_scatter_autograd,
    ring_all_reduce,
    scatter,
    scatter_reduce,
)

__all__ = [
    "broadcast",
    "scatter",
    "gather",
    "all_gather",
    "reduce",
    "all_reduce",
    "reduce_scatter",
    "all_to_all",
    "ring_all_reduce",
    "scatter_reduce",
    "all_reduce_autograd",
    "reduce_scatter_autograd",
    "all_gather_autograd",
]
