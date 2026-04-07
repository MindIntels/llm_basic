"""
GatedTransformerBlock — complete architecture block combining:

  1. RMSNorm  (pre-norm)
  2. GatedAttention  OR  GatedDeltaNet  (selectable via ``mixer`` argument)
  3. Residual connection
  4. RMSNorm  (pre-norm on FFN)
  5. SwiGLU FFN
  6. Residual connection

This mirrors the architecture used in modern efficient LLMs:
  - LLaMA 3   : RMSNorm + standard attention + SwiGLU FFN
  - Hawk/Griffin: RMSNorm + GatedAttention + SwiGLU FFN
  - GatedDeltaNet Transformer: RMSNorm + GatedDeltaNet + SwiGLU FFN

Additionally we expose a ``GatedTransformer`` (full model stack) that stacks
``N`` GatedTransformerBlock layers with a final RMSNorm before the logits.

Block diagram
-------------

    x ─────────────────────────────────────────────── +
    │                                                  │
    ┌──────────┐   ┌───────────────┐                  │
    │ RMSNorm  │──►│ Mixer (Attn   │                  │
    └──────────┘   │  or DeltaNet) │                  │
                   └───────────────┘                  │
                         │                            │
                         ▼ (residual) ────────────────┘
    x' ─────────────────────────────────────────────── +
    │                                                  │
    ┌──────────┐   ┌───────────────┐                  │
    │ RMSNorm  │──►│  SwiGLU FFN  │                  │
    └──────────┘   └───────────────┘                  │
                         │                            │
                         ▼ (residual) ────────────────┘
    out

Usage example
-------------
>>> from attention.gated_transformer import GatedTransformerBlock, GatedTransformer
>>> # Single block with GatedAttention + SwiGLU
>>> block = GatedTransformerBlock(hidden_size=256, num_heads=4, mixer="gated_attn")
>>> x = torch.randn(2, 32, 256)
>>> out = block(x)           # [2, 32, 256]

>>> # Full model (8 layers alternating GatedDeltaNet + GatedAttention)
>>> model = GatedTransformer(hidden_size=256, num_heads=4, num_layers=8,
...                          mixer_pattern=["deltanet", "gated_attn"])
>>> out = model(x)           # [2, 32, 256]
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .swiglu import SwiGLUFFN
from .gated_attention import GatedAttention
from .gated_deltanet import GatedDeltaNet


# --------------------------------------------------------------------------- #
#  Single GatedTransformerBlock                                                 #
# --------------------------------------------------------------------------- #

MixerType = Literal["gated_attn", "deltanet"]


class GatedTransformerBlock(nn.Module):
    """One Gated Transformer layer: pre-norm mixer + pre-norm SwiGLU FFN.

    Parameters
    ----------
    hidden_size : int
    num_heads : int
    mixer : str
        ``"gated_attn"`` — use GatedAttention.
        ``"deltanet"``   — use GatedDeltaNet.
    intermediate_size : int | None
        FFN expansion.  Defaults to 8/3 × hidden_size (SwiGLU convention).
    gate_act : str
        Gate activation for GatedAttention: ``"sigmoid"`` or ``"silu"``.
    ffn_dropout : float
    attn_dropout : float
    causal : bool
        Only relevant for ``mixer="gated_attn"``.
    head_dim : int | None
    value_dim : int | None
        Only relevant for ``mixer="deltanet"``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mixer: MixerType = "gated_attn",
        intermediate_size: int | None = None,
        gate_act: str = "sigmoid",
        ffn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        causal: bool = False,
        head_dim: int | None = None,
        value_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.mixer_type = mixer

        # Pre-norms
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

        # Token mixer
        if mixer == "gated_attn":
            self.mixer: nn.Module = GatedAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                gate_act=gate_act,
                pre_norm=True,
                dropout=attn_dropout,
                causal=causal,
            )
        elif mixer == "deltanet":
            self.mixer = GatedDeltaNet(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                value_dim=value_dim,
                use_output_gate=True,
                norm_keys=True,
            )
        else:
            raise ValueError(f"Unknown mixer: {mixer!r}. Choose 'gated_attn' or 'deltanet'.")

        # Feed-forward
        self.ffn = SwiGLUFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=False,
            dropout=ffn_dropout,
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x              : [B, S, hidden_size]
            attention_mask : additive mask (only used with gated_attn mixer).
            state          : recurrent state (only used with deltanet mixer).
            return_state   : whether to return updated recurrent state.

        Returns:
            out            : [B, S, hidden_size]
            state          : updated state (only if return_state=True).
        """
        # ── Mixer with pre-norm + residual ─────────────────────────────
        h = self.norm1(x)
        new_state = None

        if self.mixer_type == "gated_attn":
            m_out = self.mixer(h, attention_mask=attention_mask)
        else:  # deltanet
            if return_state:
                m_out, new_state = self.mixer(h, state=state, return_state=True)
            else:
                m_out = self.mixer(h, state=state, return_state=False)

        x = x + m_out

        # ── FFN with pre-norm + residual ────────────────────────────────
        x = x + self.ffn(self.norm2(x))

        if return_state:
            return x, new_state
        return x

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return f"mixer={self.mixer_type!r}"


# --------------------------------------------------------------------------- #
#  Full GatedTransformer model                                                  #
# --------------------------------------------------------------------------- #

class GatedTransformer(nn.Module):
    """Stack of GatedTransformerBlocks with a final RMSNorm.

    ``mixer_pattern`` can be a single string or a list of strings that is
    cycled across layers, allowing alternating architectures like:
        ["gated_attn", "deltanet", "gated_attn", "deltanet", ...]

    Parameters
    ----------
    hidden_size : int
    num_heads : int
    num_layers : int
    mixer_pattern : str | list[str]
        Mixer type per layer (cycled if shorter than num_layers).
    intermediate_size : int | None
    gate_act : str
    ffn_dropout : float
    attn_dropout : float
    causal : bool
    head_dim : int | None
    value_dim : int | None
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        mixer_pattern: str | list[str] = "gated_attn",
        intermediate_size: int | None = None,
        gate_act: str = "sigmoid",
        ffn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        causal: bool = False,
        head_dim: int | None = None,
        value_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        if isinstance(mixer_pattern, str):
            pattern = [mixer_pattern]
        else:
            pattern = list(mixer_pattern)

        self.layers = nn.ModuleList([
            GatedTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mixer=pattern[i % len(pattern)],
                intermediate_size=intermediate_size,
                gate_act=gate_act,
                ffn_dropout=ffn_dropout,
                attn_dropout=attn_dropout,
                causal=causal,
                head_dim=head_dim,
                value_dim=value_dim,
            )
            for i in range(num_layers)
        ])

        self.final_norm = RMSNorm(hidden_size)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x              : [B, S, hidden_size]
            attention_mask : additive mask or None.

        Returns:
            [B, S, hidden_size]
        """
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.final_norm(x)

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return f"hidden={self.hidden_size}, layers={self.num_layers}"
