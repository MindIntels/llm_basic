"""
Numerically-stable (safe) softmax.

Standard ``softmax(x)`` computes ``exp(x_i) / sum(exp(x_j))``.  When the
entries of *x* are large in magnitude the ``exp`` can overflow to ``inf``
(for float16 / bfloat16 this is common even at moderate values).

**Safe softmax** subtracts the row-wise maximum before exponentiation:

    safe_softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

This is mathematically identical but keeps the exponent in [−∞, 0],
preventing overflow.  An additional guard replaces any residual ``nan``
(which can arise when an entire row is ``-inf``, producing ``0/0``) with
zero so that downstream matmuls stay clean.
"""

from __future__ import annotations

import torch


def safe_softmax(
    x: torch.Tensor,
    dim: int = -1,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Numerically-stable softmax with NaN→0 guard.

    Args:
        x: Input logits (arbitrary shape).
        dim: Dimension along which to compute softmax.
        dtype: If given, cast *x* to this dtype **before** the computation
               (useful to force float32 accumulation from float16 inputs).

    Returns:
        Tensor of the same shape as *x* (in *dtype* if provided, else same
        dtype as *x*).
    """
    if dtype is not None:
        x = x.to(dtype)

    # Subtract row-max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    out = exp_x / sum_exp

    # Guard: rows that are entirely -inf give 0/0 = nan → replace with 0
    out = torch.nan_to_num(out, nan=0.0)

    return out
