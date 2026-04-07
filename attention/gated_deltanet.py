"""
Gated DeltaNet — linear recurrent layer with delta-rule updates and gating.

Background
----------
DeltaNet (Schlag et al. 2021, "Linear Transformers Are Secretly Fast Weight Programmers")
is a linear-complexity sequence model that approximates attention by maintaining a
fast-weight matrix S (the "state" or "memory") updated with the delta rule:

    β_t  = sigmoid(w_β · x_t)                      (per-head scalar)
    k_t  = normalize(K_t)                            (unit-norm key)
    S_t  = S_{t-1} + β_t * (v_t - S_{t-1} k_t) k_t^T   (delta update)
    y_t  = S_t q_t                                   (read)

This is equivalent to a recurrent neural network with per-step state updates.

**Gated DeltaNet** (Yang et al. 2024, "Gated DeltaNet") adds a forget gate α:

    α_t  = sigmoid(w_α · x_t)                      (forget factor ∈ (0,1))
    S_t  = α_t * S_{t-1} + β_t * (v_t - α_t * S_{t-1} k_t) k_t^T

This combines:
  * Selective forgetting (α controls memory decay, à la Mamba)
  * Delta-rule associative update (β + k + v as in DeltaNet)
  * Output gating (SiLU gate on the read output, as in GLA / RetNet)

Complexity: O(S · H · D²) recurrent  or  O(S² · H) chunk-parallel.
Here we implement the **recurrent** (step-by-step, O(S)) path which is
fully correct and easy to follow.

Notation
--------
  B   = batch size
  S   = sequence length
  H   = number of heads
  D   = head dimension  (hidden_size / num_heads)
  d_v = value head dim  (can equal D)

Shapes
------
  Input  x : [B, S, hidden_size]
  State  S : [B, H, D, d_v]    (updated per timestep)
  Output   : [B, S, hidden_size]
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear recurrent layer.

    Parameters
    ----------
    hidden_size : int
        Model dimension.
    num_heads : int
        Number of parallel heads.
    head_dim : int | None
        Per-head key/query dimension.  Defaults to ``hidden_size // num_heads``.
    value_dim : int | None
        Per-head value dimension.  Defaults to ``head_dim``.
    use_output_gate : bool
        If True (default), apply a SiLU output gate on y before o_proj.
    norm_keys : bool
        If True (default), L2-normalize the key vectors (required for the
        delta rule to be interpretable as a fast-weight update).
    eps : float
        Small value for key normalization denominator.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        value_dim: int | None = None,
        use_output_gate: bool = True,
        norm_keys: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size    = hidden_size
        self.num_heads      = num_heads
        self.head_dim       = head_dim or (hidden_size // num_heads)
        self.value_dim      = value_dim or self.head_dim
        self.use_output_gate = use_output_gate
        self.norm_keys      = norm_keys
        self.eps            = eps
        self.scale          = 1.0 / math.sqrt(self.head_dim)

        d_H  = self.head_dim
        d_v  = self.value_dim
        H    = num_heads

        # ── Projections ────────────────────────────────────────────────
        # Q, K projections: hidden → H * d_H
        self.q_proj = nn.Linear(hidden_size, H * d_H, bias=False)
        self.k_proj = nn.Linear(hidden_size, H * d_H, bias=False)
        # V projection: hidden → H * d_v
        self.v_proj = nn.Linear(hidden_size, H * d_v, bias=False)
        # Output projection: H * d_v → hidden
        self.o_proj = nn.Linear(H * d_v, hidden_size, bias=False)

        # ── Gating scalars (per head, shared across positions) ─────────
        # β: write gate (how much to update the state)
        self.beta_proj  = nn.Linear(hidden_size, H, bias=True)
        # α: forget gate (how much of the old state to keep)
        self.alpha_proj = nn.Linear(hidden_size, H, bias=True)

        # ── Optional output gate ───────────────────────────────────────
        if use_output_gate:
            self.out_gate_proj = nn.Linear(hidden_size, H * d_v, bias=False)

        # ── GroupNorm on output (one group = one head) ─────────────────
        self.out_norm = RMSNorm(H * d_v)

        # ── Initialization ─────────────────────────────────────────────
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        # Bias the forget gate toward 1 (remember more at the start)
        nn.init.constant_(self.alpha_proj.bias, 1.0)
        # Bias the write gate toward small values (sparse writes)
        nn.init.constant_(self.beta_proj.bias, -2.0)

    # ------------------------------------------------------------------ #
    def _init_state(self, B: int, device: torch.device, dtype: torch.dtype):
        """Zero initial fast-weight state [B, H, D, d_v]."""
        return torch.zeros(
            B, self.num_heads, self.head_dim, self.value_dim,
            device=device, dtype=dtype,
        )

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x           : [B, S, hidden_size]
            state       : [B, H, head_dim, value_dim] or None (zeros).
            return_state: if True, also return the final state.

        Returns:
            out         : [B, S, hidden_size]
            state       : [B, H, head_dim, value_dim]  (only if return_state=True)
        """
        B, S, _ = x.shape
        H, D, Dv = self.num_heads, self.head_dim, self.value_dim

        if state is None:
            state = self._init_state(B, x.device, x.dtype)

        # ── Project ────────────────────────────────────────────────────
        Q = self.q_proj(x).view(B, S, H, D)    # [B, S, H, D]
        K = self.k_proj(x).view(B, S, H, D)
        V = self.v_proj(x).view(B, S, H, Dv)

        # Gating scalars  — [B, S, H]
        alpha = torch.sigmoid(self.alpha_proj(x))   # forget ∈ (0,1)
        beta  = torch.sigmoid(self.beta_proj(x))    # write  ∈ (0,1)

        if self.norm_keys:
            K = F.normalize(K, p=2, dim=-1, eps=self.eps)

        # Scale queries
        Q = Q * self.scale

        # ── Recurrent loop over time ────────────────────────────────────
        # Each step t:
        #   S_t = α_t * S_{t-1} + β_t * (v_t - α_t * (S_{t-1} @ k_t)) ⊗ k_t
        #   y_t = S_t @ q_t
        outputs = []
        S_cur = state.clone()   # [B, H, D, Dv]

        for t in range(S):
            q_t = Q[:, t, :, :]      # [B, H, D]
            k_t = K[:, t, :, :]      # [B, H, D]
            v_t = V[:, t, :, :]      # [B, H, Dv]
            a_t = alpha[:, t, :]     # [B, H]
            b_t = beta[:, t, :]      # [B, H]

            # α decay: S = α * S
            # b-cast: a_t [B,H] → [B,H,1,1]
            a_t4 = a_t.unsqueeze(-1).unsqueeze(-1)   # [B, H, 1, 1]
            b_t2 = b_t.unsqueeze(-1)                  # [B, H, 1]

            # Predicted value from current state: [B, H, Dv]
            # S_cur @ k_t  →  einsum bhDd, bhD → bhd  (D=head_dim, d=value_dim)
            # k_t is [B,H,D], S_cur is [B,H,D,Dv]
            # S_cur k_t := k_t^T S_cur  →  matmul S_cur^T k_t  isn't quite right
            # Actually: v_hat_t = S_cur * k_t summed over D
            # i.e. einstein: B h D d, B h D -> B h d
            v_hat = torch.einsum("bhDd,bhD->bhd", S_cur, k_t)  # [B, H, Dv]

            # Delta: (v_t - α * v_hat)
            delta = v_t - a_t.unsqueeze(-1) * v_hat             # [B, H, Dv]

            # Outer product: k_t ⊗ delta  →  [B, H, D, Dv]
            update = torch.einsum("bhD,bhd->bhDd", k_t, delta)  # [B, H, D, Dv]

            # State update
            S_cur = a_t4 * S_cur + b_t2.unsqueeze(-1) * update  # [B, H, D, Dv]

            # Read: y_t = S_cur @ q_t
            y_t = torch.einsum("bhDd,bhD->bhd", S_cur, q_t)    # [B, H, Dv]
            outputs.append(y_t)

        # ── Stack outputs ───────────────────────────────────────────────
        out = torch.stack(outputs, dim=1)               # [B, S, H, Dv]
        out = out.view(B, S, H * Dv)                    # [B, S, H*Dv]

        # ── Normalise ───────────────────────────────────────────────────
        out = self.out_norm(out)

        # ── Optional output gate ────────────────────────────────────────
        if self.use_output_gate:
            g = F.silu(self.out_gate_proj(x))           # [B, S, H*Dv]
            out = g * out

        # ── Output projection ───────────────────────────────────────────
        out = self.o_proj(out)                           # [B, S, hidden_size]

        if return_state:
            return out, S_cur
        return out

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, value_dim={self.value_dim}, "
            f"output_gate={self.use_output_gate}"
        )

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step recurrent inference (auto-regressive decoding).

        Args:
            x_t   : [B, hidden_size]  — current input token embedding.
            state  : [B, H, head_dim, value_dim]

        Returns:
            (y_t, new_state)
            y_t  : [B, hidden_size]
            new_state : [B, H, head_dim, value_dim]
        """
        B = x_t.shape[0]
        H, D, Dv = self.num_heads, self.head_dim, self.value_dim

        q = self.q_proj(x_t).view(B, H, D) * self.scale
        k = self.k_proj(x_t).view(B, H, D)
        v = self.v_proj(x_t).view(B, H, Dv)

        alpha = torch.sigmoid(self.alpha_proj(x_t))   # [B, H]
        beta  = torch.sigmoid(self.beta_proj(x_t))    # [B, H]

        if self.norm_keys:
            k = F.normalize(k, p=2, dim=-1, eps=self.eps)

        a4 = alpha.unsqueeze(-1).unsqueeze(-1)        # [B, H, 1, 1]
        v_hat = torch.einsum("bhDd,bhD->bhd", state, k)
        delta = v - alpha.unsqueeze(-1) * v_hat
        update = torch.einsum("bhD,bhd->bhDd", k, delta)
        new_state = a4 * state + beta.unsqueeze(-1).unsqueeze(-1) * update

        y = torch.einsum("bhDd,bhD->bhd", new_state, q)   # [B, H, Dv]
        y = y.view(B, H * Dv)
        y_norm = self.out_norm(y)

        if self.use_output_gate:
            g = F.silu(self.out_gate_proj(x_t))
            y_norm = g * y_norm

        out = self.o_proj(y_norm)
        return out, new_state
