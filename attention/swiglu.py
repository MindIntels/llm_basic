"""
SwiGLU feed-forward block.

Reference: Noam Shazeer, "GLU Variants Improve Transformer" (2020).
https://arxiv.org/abs/2002.05202

Architecture
------------
Standard FFN (2 matrices):
    FFN(x) = ReLU(x @ W1) @ W2

SwiGLU (3 matrices) — replaces ReLU with a gated activation:
    gate  = x @ W_gate     (shape [B, S, d_ff])
    value = x @ W_up       (shape [B, S, d_ff])
    h     = SiLU(gate) * value
    out   = h @ W_down     (shape [B, S, d_model])

where SiLU(x) = x * sigmoid(x)  (also called Swish-1).

Parameter count: 3 × d_model × d_ff  (vs 2 × d_model × d_ff for standard FFN).
To keep FLOPs equal, practitioners typically set d_ff = 8/3 * d_model ≈ 2.67 * d_model.
In LLaMA-2 7B for instance: d_model=4096, d_ff=11008 ≈ 2.69 × 4096.

This module contains:
  * ``swiglu``       — functional version (no weights).
  * ``SwiGLUFFN``    — Drop-in FFN replacement (3 Linear layers + optional RMSNorm).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Functional                                                                   #
# --------------------------------------------------------------------------- #

def swiglu(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Functional SwiGLU: SiLU(gate) * value.

    Args:
        gate  : [..., d_ff]  — the gating branch (passed through SiLU).
        value : [..., d_ff]  — the value branch (kept linear).

    Returns:
        [..., d_ff]
    """
    return F.silu(gate) * value


# --------------------------------------------------------------------------- #
#  Module                                                                       #
# --------------------------------------------------------------------------- #

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Parameters
    ----------
    hidden_size : int
        Input / output dimension d_model.
    intermediate_size : int | None
        d_ff.  If None, defaults to ``int(8/3 * hidden_size)`` rounded to the
        nearest multiple of 64 to keep memory-aligned weight shapes.
    bias : bool
        Whether to add bias terms to the linear layers.  Default: False
        (following LLaMA convention).
    dropout : float
        Dropout probability applied after the gated activation, before W_down.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        if intermediate_size is None:
            # 8/3 × d_model rounded up to multiple of 64
            raw = int(8 / 3 * hidden_size)
            intermediate_size = math.ceil(raw / 64) * 64
        self.intermediate_size = intermediate_size

        # Three weight matrices: gate, up (value), down
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, S, hidden_size]

        Returns:
            [B, S, hidden_size]
        """
        gate = self.gate_proj(x)   # [B, S, d_ff]
        up   = self.up_proj(x)     # [B, S, d_ff]
        h    = swiglu(gate, up)    # SiLU(gate) * up
        h    = self.dropout(h)
        return self.down_proj(h)   # [B, S, hidden_size]

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )
