"""
Attention module — Split Q/K/V Multi-Head Attention & CPU Flash Attention (v1–v4),
Window Attention, Cross-Attention, Gated Architecture, and Rotary Embeddings.

This package provides:
  - StandardMHA:          Naive scaled dot-product MHA (materialises full S×S matrix).
  - SplitQKVAttention:    Explicit split of Q/K/V projections with per-head compute.
  - FlashAttentionCPU:    v1 — tiled online softmax, no full S×S materialisation.
  - FlashAttentionV2:     v2 — deferred rescaling, causal early-exit.
  - FlashAttentionV3:     v3 — block-sparse, mixed-precision, two-pass softmax.
  - FlashAttentionV4:     v4 — KV-cache, sliding window, softcap, auto block size.
  - WindowAttention:      Sliding-window (local) attention — symmetric and causal.
  - CrossAttention:       Encoder–decoder (cross) attention.

  Gated Architecture family:
  - RMSNorm:              Root Mean Square Layer Normalization (no mean-centering).
  - SwiGLUFFN:            Feed-forward with SwiGLU gating (3 matrices, SiLU gate).
  - GatedAttention:       Multi-head attention with element-wise output gate.
  - GatedDeltaNet:        Linear recurrent layer with Gated Delta-Rule updates.
  - GatedTransformerBlock: Pre-norm block: mixer + SwiGLU FFN + residuals.
  - GatedTransformer:     Full N-layer stack with alternating mixer patterns.

  Rotary Position Embeddings:
  - rotate_half:          Helper — swap and negate second half of last dim.
  - apply_rotary_emb:     Functional RoPE application given precomputed cos/sin.
  - RotaryEmbedding:      Caches and applies standard 1-D RoPE (LLaMA style).
  - RoPEAttention:        MHA with built-in RoPE + GQA support.
  - MultimodalRotaryEmbedding: M-axis mRoPE for multimodal sequences (Qwen2-VL).
  - mRoPEAttention:       MHA with built-in mRoPE.
  - make_text_position_ids:  Helper — 1-D position IDs for text.
  - make_image_position_ids: Helper — 2-D grid (h, w) position IDs.
  - make_video_position_ids: Helper — 3-D grid (t, h, w) position IDs.
"""

from .split_qkv import SplitQKVAttention
from .standard_attention import StandardMHA
from .safe_softmax import safe_softmax
from .flash_attention_cpu import FlashAttentionCPU, flash_attention_forward
from .flash_attention_v2 import FlashAttentionV2, flash_attention_v2_forward
from .flash_attention_v3 import FlashAttentionV3, flash_attention_v3_forward
from .flash_attention_v4 import FlashAttentionV4, flash_attention_v4_forward, auto_block_size
from .window_attention import WindowAttention, window_attention_forward
from .cross_attention import CrossAttention, cross_attention_forward

# Gated architecture family
from .rmsnorm import RMSNorm
from .swiglu import SwiGLUFFN, swiglu
from .gated_attention import GatedAttention
from .gated_deltanet import GatedDeltaNet
from .gated_transformer import GatedTransformerBlock, GatedTransformer

# Rotary position embeddings
from .rope import (
    rotate_half,
    apply_rotary_emb,
    RotaryEmbedding,
    RoPEAttention,
)
from .mrope import (
    MultimodalRotaryEmbedding,
    mRoPEAttention,
    make_text_position_ids,
    make_image_position_ids,
    make_video_position_ids,
)

__all__ = [
    # Classic attention
    "SplitQKVAttention",
    "StandardMHA",
    "safe_softmax",
    "FlashAttentionCPU",
    "flash_attention_forward",
    "FlashAttentionV2",
    "flash_attention_v2_forward",
    "FlashAttentionV3",
    "flash_attention_v3_forward",
    "FlashAttentionV4",
    "flash_attention_v4_forward",
    "auto_block_size",
    "WindowAttention",
    "window_attention_forward",
    "CrossAttention",
    "cross_attention_forward",
    # Gated architecture
    "RMSNorm",
    "SwiGLUFFN",
    "swiglu",
    "GatedAttention",
    "GatedDeltaNet",
    "GatedTransformerBlock",
    "GatedTransformer",
    # RoPE
    "rotate_half",
    "apply_rotary_emb",
    "RotaryEmbedding",
    "RoPEAttention",
    # mRoPE
    "MultimodalRotaryEmbedding",
    "mRoPEAttention",
    "make_text_position_ids",
    "make_image_position_ids",
    "make_video_position_ids",
]
