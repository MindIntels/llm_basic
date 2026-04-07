"""
Gated Attention — gating mechanism over standard multi-head attention output.

Motivation
----------
Standard attention returns:
    out = Attention(Q, K, V) @ W_o

Gated Attention adds an element-wise gate that controls *how much* of the
attention output passes through:
    g   = sigmoid(x @ W_g + b_g)      OR    silu(x @ W_g)
    out = g * Attention(Q, K, V) @ W_o

This is used in:
  - RetNet (Sun et al. 2023): gates the recurrent (or parallel) attention
  - Griffin / Hawk (De et al. 2024): gates linear recurrences
  - Gated Linear Attention (Yang et al. 2023)
  - TransNormer, HGRN2, RWKV-6 gate variants

Implementation here
-------------------
We provide two gate activations:
  * ``GatedAttention(gate_act="sigmoid")`` — bounded gate in [0,1], stable.
  * ``GatedAttention(gate_act="silu")``    — unbounded gate, more expressive.

Architecture (per layer):
    # Projections
    q, k, v  = split(x @ W_qkv)           [B, S, H]  each
    q = reshape + RoPE-style scale (optional)
    k = reshape
    v = reshape

    # Attention
    attn   = softmax(q @ k^T / sqrt(d)) @ v           [B, H, S, D]
    attn   = merge_heads(attn)                          [B, S, hidden]
    attn   = attn @ W_o                                [B, S, hidden]

    # Gate
    g      = gate_act(x @ W_g)                         [B, S, hidden]
    out    = g * attn

The gate projection W_g maps x → hidden directly (same in/out dimensionality).
If ``pre_norm=True``, a RMSNorm is applied before the gate projection.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .safe_softmax import safe_softmax
from .rmsnorm import RMSNorm


class GatedAttention(nn.Module):
    """Multi-head attention with an element-wise output gate.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_heads : int
        Number of attention heads.
    head_dim : int | None
        Dimension per head.  Defaults to ``hidden_size // num_heads``.
    gate_act : str
        Activation function for the gate: ``"sigmoid"`` (bounded, LeCam-stable)
        or ``"silu"`` (unbounded, more expressive).
    pre_norm : bool
        If True, apply RMSNorm to x before the gate projection.
    dropout : float
        Attention dropout.
    causal : bool
        If True, apply a fixed causal mask in every forward call.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        gate_act: str = "sigmoid",
        pre_norm: bool = True,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = head_dim or (hidden_size // num_heads)
        assert self.head_dim * num_heads == hidden_size

        if gate_act not in ("sigmoid", "silu"):
            raise ValueError(f"gate_act must be 'sigmoid' or 'silu', got {gate_act!r}")
        self.gate_act = gate_act
        self.causal   = causal
        self.scale    = 1.0 / math.sqrt(self.head_dim)

        # Attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Gate projection: x → hidden  (with bias for sigmoid stability)
        self.g_proj = nn.Linear(hidden_size, hidden_size, bias=(gate_act == "sigmoid"))

        # Optional pre-norm before gate projection
        self.gate_norm = RMSNorm(hidden_size) if pre_norm else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    def _apply_gate(self, x: torch.Tensor) -> torch.Tensor:
        gate_input = self.gate_norm(x)
        g = self.g_proj(gate_input)
        if self.gate_act == "sigmoid":
            return torch.sigmoid(g)
        return F.silu(g)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x              : [B, S, hidden_size]
            attention_mask : additive mask [B, 1, S, S] or None.
            causal         : overrides ``self.causal`` if given.

        Returns:
            [B, S, hidden_size]
        """
        B, S, _ = x.shape
        use_causal = self.causal if causal is None else causal

        # ── Attention ──────────────────────────────────────────────────
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]

        if use_causal:
            causal_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = safe_softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.attn_dropout(attn_weights)

        ctx = torch.matmul(attn_weights, v)                          # [B, H, S, D]
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, -1)        # [B, S, H]
        attn_out = self.o_proj(ctx)                                   # [B, S, H]

        # ── Gate ───────────────────────────────────────────────────────
        g = self._apply_gate(x)   # [B, S, H]
        return g * attn_out

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, gate_act={self.gate_act!r}"
        )
