"""
RMSNorm — Root Mean Square Layer Normalization.

Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).
https://arxiv.org/abs/1910.07467

Unlike LayerNorm, RMSNorm:
  - Removes the mean-centering step.
  - Removes the bias (β) parameter.
  - Only keeps the learnable scale (γ).

Forward:
    rms(x)  = sqrt( mean(x²) + ε )
    out     = x / rms(x) * γ

Memory / compute advantage: roughly 2× fewer operations than LayerNorm and
no risk of β causing representation collapse in deep residual networks.

Used by: LLaMA, Mistral, Gemma, Falcon, and the Gated architectures here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Parameters
    ----------
    hidden_size : int
        Size of the last dimension of the input tensor.
    eps : float
        Small constant added inside the square root for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    # ------------------------------------------------------------------ #
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute x / rms(x), keeping dtype for stability."""
        # Cast to float32 for the variance computation to avoid fp16 overflow
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x_fp32 / rms).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [..., hidden_size]

        Returns:
            [..., hidden_size]  — same shape and dtype as input.
        """
        return self._norm(x) * self.weight

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
