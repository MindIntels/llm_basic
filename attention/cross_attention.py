"""
Cross-Attention (Encoder–Decoder Attention).

Cross-attention allows a *decoder* sequence to attend over a separate
*encoder* (or context) sequence.  Queries come from the decoder hidden
states while keys and values come from encoder hidden states.

This is the standard mechanism in:
  - Transformer encoder–decoder (Vaswani et al.)
  - Vision-Language models (Flamingo, LLaVA cross-attn variants)
  - Retrieval-augmented generation (RETRO)
  - Whisper / speech models

Supports:
  - Separate Q vs KV projections from different hidden sizes.
  - Optional causal masking (rarely needed but provided).
  - Gradient flow through both decoder and encoder inputs.
  - KV-cache for autoregressive decoding (pass pre-computed encoder KV).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .safe_softmax import safe_softmax


class CrossAttention(nn.Module):
    """Multi-Head Cross-Attention.

    Parameters
    ----------
    decoder_hidden_size : int
        Hidden dimension of the decoder (query source).
    encoder_hidden_size : int
        Hidden dimension of the encoder / context (key & value source).
        Can differ from decoder_hidden_size.
    num_heads : int
        Number of attention heads.
    head_dim : int or None
        Dimension per head.  Defaults to ``decoder_hidden_size // num_heads``.
    dropout : float
        Attention dropout probability.
    bias : bool
        Whether linear projections include bias.
    """

    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or decoder_hidden_size // num_heads
        inner_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(decoder_hidden_size, inner_dim, bias=bias)
        self.k_proj = nn.Linear(encoder_hidden_size, inner_dim, bias=bias)
        self.v_proj = nn.Linear(encoder_hidden_size, inner_dim, bias=bias)
        self.o_proj = nn.Linear(inner_dim, decoder_hidden_size, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    # ------------------------------------------------------------------ #
    #  Weight helpers                                                      #
    # ------------------------------------------------------------------ #
    def load_weights(
        self,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        w_o: torch.Tensor,
    ) -> None:
        """Load pre-defined weight matrices."""
        self.q_proj.weight.data.copy_(w_q)
        self.k_proj.weight.data.copy_(w_k)
        self.v_proj.weight.data.copy_(w_v)
        self.o_proj.weight.data.copy_(w_o)

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            decoder_hidden: [B, S_q, decoder_hidden_size]  — query source.
            encoder_hidden: [B, S_kv, encoder_hidden_size] — key/value source.
                Ignored if ``kv_cache`` is provided.
            attention_mask: optional additive mask [B, 1, S_q, S_kv].
            kv_cache: optional pre-computed (K, V) each
                [B, num_heads, S_kv, head_dim] to skip re-projection.

        Returns:
            [B, S_q, decoder_hidden_size]
        """
        B, S_q, _ = decoder_hidden.shape

        # --- Query always from decoder --------------------------------
        q = self.q_proj(decoder_hidden)  # [B, S_q, inner_dim]
        q = q.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        # q: [B, H, S_q, D]

        # --- Key/Value from encoder or cache --------------------------
        if kv_cache is not None:
            k, v = kv_cache  # already [B, H, S_kv, D]
        else:
            S_kv = encoder_hidden.size(1)
            k = self.k_proj(encoder_hidden)  # [B, S_kv, inner_dim]
            v = self.v_proj(encoder_hidden)
            k = k.view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # --- Attention scores -----------------------------------------
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # attn: [B, H, S_q, S_kv]

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = safe_softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
        attn = self.attn_dropout(attn)

        # --- Weighted sum ---------------------------------------------
        out = torch.matmul(attn, v)  # [B, H, S_q, D]
        out = out.transpose(1, 2).contiguous().view(B, S_q, -1)

        return self.o_proj(out)

    # ------------------------------------------------------------------ #
    #  Convenience: pre-compute encoder KV for caching                     #
    # ------------------------------------------------------------------ #
    def encode_kv(
        self,
        encoder_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project encoder hidden states to K, V for caching.

        Args:
            encoder_hidden: [B, S_kv, encoder_hidden_size]

        Returns:
            (K, V) each [B, num_heads, S_kv, head_dim]
        """
        B, S_kv, _ = encoder_hidden.shape
        k = self.k_proj(encoder_hidden).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_hidden).view(B, S_kv, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v

    def extra_repr(self) -> str:
        return (
            f"dec_hidden={self.decoder_hidden_size}, "
            f"enc_hidden={self.encoder_hidden_size}, "
            f"heads={self.num_heads}, head_dim={self.head_dim}"
        )


def cross_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional cross attention (no learnable weights).

    Args:
        q: [B, H, S_q, D] — queries (from decoder).
        k: [B, H, S_kv, D] — keys (from encoder).
        v: [B, H, S_kv, D] — values (from encoder).
        scale: attention scale (default 1/√D).
        attention_mask: optional additive mask [B, 1, S_q, S_kv].

    Returns:
        [B, H, S_q, D]
    """
    D = q.size(-1)
    sc = scale or (1.0 / math.sqrt(D))

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sc

    if attention_mask is not None:
        scores = scores + attention_mask

    attn = safe_softmax(scores, dim=-1, dtype=torch.float32)
    out = torch.matmul(attn, v.float()).to(q.dtype)
    return out
