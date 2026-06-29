"""
model.py — Unified, interpretability-friendly transformer implementation.

Design notes:
  - Pre-LayerNorm residual stream (matches GPT-2, LLaMA, Gemma, most modern models).
  - Explicit W_Q / W_K / W_V / W_O weight matrices (no fused nn.Linear) for clean
    circuit-level decomposition, following TransformerLens naming conventions.
  - Configurable positional embeddings: none | learned | sinusoidal | rope.
  - Single TransformerConfig dataclass drives the whole architecture.
  - Loader helpers for HuggingFace (pretrained or random-weight) and TransformerLens.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

_VALID_POS_EMBED = {"none", "learned", "sinusoidal", "rope"}


@dataclass
class TransformerConfig:
    """
    Hyperparameters for a decoder-only transformer.

    Attributes:
        d_vocab:         Vocabulary size.
        d_model:         Residual stream / model dimension.
        n_heads:         Number of attention heads.
        d_head:          Per-head dimension. Inferred as d_model // n_heads when None.
        n_layers:        Number of transformer blocks.
        n_ctx:           Maximum sequence length (used for learned positional embeddings).
        d_mlp:           Hidden dimension of the MLP sub-layer.
                         Ignored when attn_only=True.
        attn_only:       If True, omit the MLP sub-layer in every block.
        use_layernorm:   If True, apply pre-LayerNorm (standard; see TransformerBlock).
        use_bias:        If True, add learnable bias terms to all attention projections
                         and MLP layers. Most modern models (LLaMA, Gemma) set this to
                         False for cleaner weight decomposition.
        pos_embed:       Positional encoding scheme:
                           "none"       — no positional information injected
                           "learned"    — trainable nn.Embedding lookup (GPT-2 / BERT)
                           "sinusoidal" — fixed, non-trainable (Vaswani et al. 2017)
                           "rope"       — Rotary Position Embedding (Su et al. 2021),
                                          applied inside attention to Q and K only
        use_causal_mask: If True, enforce autoregressive (lower-triangular) masking.
                         Set to False for bidirectional / encoder-style attention.
    """

    d_vocab:         int            = 16
    d_model:         int            = 64
    n_heads:         int            = 4
    d_head:          Optional[int]  = None   # inferred from d_model // n_heads if None
    n_layers:        int            = 1
    n_ctx:           int            = 128
    d_mlp:           int            = 256
    attn_only:       bool           = False
    use_layernorm:   bool           = True
    use_bias:        bool           = False
    pos_embed:       str            = "rope"
    use_causal_mask: bool           = True
    model_seed:      Optional[int]  = 42

    def __post_init__(self) -> None:
        if self.pos_embed not in _VALID_POS_EMBED:
            raise ValueError(
                f"pos_embed must be one of {_VALID_POS_EMBED}, got '{self.pos_embed}'"
            )
        if self.d_head is None:
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}) "
                    "when d_head is not specified."
                )
            self.d_head = self.d_model // self.n_heads


# ─────────────────────────────────────────────────────────────────────────────
# RoPE utilities
# ─────────────────────────────────────────────────────────────────────────────

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dimension for RoPE: split in half, negate the second
    half, then concatenate as (-second, first).  Matches the HuggingFace
    Llama/Gemma convention.
    """
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to Q and K.

    The cos/sin cache is built on-the-fly from the actual sequence length, so
    there is no hard n_ctx limit (unlike learned positional embeddings).

    Args:
        q, k: (batch, n_heads, n_ctx, d_head)

    Returns:
        Rotated q and k with the same shape.
    """
    n_ctx, d_head = q.shape[-2], q.shape[-1]

    inv_freq = 1.0 / (
        10_000.0 ** (
            torch.arange(0, d_head, 2, device=q.device, dtype=torch.float32) / d_head
        )
    )
    t = torch.arange(n_ctx, device=q.device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)            # (n_ctx, d_head // 2)
    emb   = torch.cat([freqs, freqs], dim=-1)   # (n_ctx, d_head)

    cos = emb.cos().to(q.dtype)[None, None]     # (1, 1, n_ctx, d_head)
    sin = emb.sin().to(q.dtype)[None, None]

    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal PE utility
# ─────────────────────────────────────────────────────────────────────────────

def _make_sinusoidal_pe(n_ctx: int, d_model: int) -> torch.Tensor:
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).
    Returns a (n_ctx, d_model) tensor — not a trainable parameter.
    """
    pe  = torch.zeros(n_ctx, d_model)
    pos = torch.arange(n_ctx, dtype=torch.float).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10_000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ─────────────────────────────────────────────────────────────────────────────
# MultiHeadAttention
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with explicit weight matrices for interpretability.

    Weight shapes (TransformerLens convention):
        W_Q, W_K, W_V  : (n_heads, d_model, d_head)
        W_O            : (n_heads, d_head,  d_model)
        b_Q, b_K, b_V  : (n_heads, d_head)           [present only when use_bias=True]
        b_O            : (d_model,)                   [present only when use_bias=True]

    Positional encoding is handled here (RoPE is applied inside attention; other
    schemes are injected at the embedding stage in Transformer.forward).
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H, D, Dh = cfg.n_heads, cfg.d_model, cfg.d_head

        self.W_Q = nn.Parameter(torch.empty(H, D, Dh))
        self.W_K = nn.Parameter(torch.empty(H, D, Dh))
        self.W_V = nn.Parameter(torch.empty(H, D, Dh))
        self.W_O = nn.Parameter(torch.empty(H, Dh, D))

        if cfg.use_bias:
            self.b_Q = nn.Parameter(torch.zeros(H, Dh))
            self.b_K = nn.Parameter(torch.zeros(H, Dh))
            self.b_V = nn.Parameter(torch.zeros(H, Dh))
            self.b_O = nn.Parameter(torch.zeros(D))

        self._init_weights()

    def _init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.cfg.d_model)
        for w in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.normal_(w, std=std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_ctx, d_model)

        Returns:
            attn_out:     (batch, n_ctx, d_model)
            attn_weights: (batch, n_heads, n_ctx, n_ctx)
        """
        # Project residual stream into Q, K, V  →  (batch, n_heads, n_ctx, d_head)
        Q = torch.einsum("btd,hde->bhte", x, self.W_Q)
        K = torch.einsum("btd,hde->bhte", x, self.W_K)
        V = torch.einsum("btd,hde->bhte", x, self.W_V)

        if self.cfg.use_bias:
            Q = Q + self.b_Q[None, :, None, :]
            K = K + self.b_K[None, :, None, :]
            V = V + self.b_V[None, :, None, :]

        if self.cfg.pos_embed == "rope":
            Q, K = _apply_rope(Q, K)

        # Scaled dot-product attention scores  →  (batch, n_heads, n_ctx, n_ctx)
        scores = torch.einsum("bhte,bhse->bhts", Q, K) / math.sqrt(self.cfg.d_head)

        if self.cfg.use_causal_mask:
            T    = x.shape[1]
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask[None, None], float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum over values, then project back to d_model
        head_out = torch.einsum("bhts,bhse->bhte", attn_weights, V)
        attn_out = torch.einsum("bhte,hed->btd", head_out, self.W_O)

        if self.cfg.use_bias:
            attn_out = attn_out + self.b_O

        return attn_out, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Two-layer feedforward network with GELU activation."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.W_in  = nn.Linear(cfg.d_model, cfg.d_mlp,  bias=cfg.use_bias)
        self.W_out = nn.Linear(cfg.d_mlp,   cfg.d_model, bias=cfg.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_out(F.gelu(self.W_in(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single transformer block using pre-LayerNorm:

        x ← x + Attn(LN(x))
        x ← x + MLP(LN(x))   [skipped when cfg.attn_only=True]
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg  = cfg
        self.attn = MultiHeadAttention(cfg)

        if cfg.use_layernorm:
            self.norm_attn = nn.LayerNorm(cfg.d_model)

        if not cfg.attn_only:
            self.mlp = MLP(cfg)
            if cfg.use_layernorm:
                self.norm_mlp = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_ctx, d_model)

        Returns:
            x:            updated residual stream (batch, n_ctx, d_model)
            attn_weights: (batch, n_heads, n_ctx, n_ctx)
        """
        attn_in      = self.norm_attn(x) if self.cfg.use_layernorm else x
        attn_out, attn_weights = self.attn(attn_in)
        x = x + attn_out

        if not self.cfg.attn_only:
            mlp_in = self.norm_mlp(x) if self.cfg.use_layernorm else x
            x      = x + self.mlp(mlp_in)

        return x, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Transformer (top-level model)
# ─────────────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Decoder-only transformer.

    Named parameters follow the TransformerLens convention:
        W_E        — token embedding          (d_vocab, d_model)
        W_pos      — positional embedding     (n_ctx,   d_model) [if pos_embed="learned"]
        blocks     — ModuleList of TransformerBlock
        norm_final — final LayerNorm before unembedding          [if use_layernorm]
        W_U        — unembedding              (d_model, d_vocab)  [via nn.Linear]

    Examples:
        # Minimal 1-layer attention-only model with RoPE, no causal mask
        cfg   = TransformerConfig(
            d_vocab=100, d_model=64, n_heads=4, n_layers=1,
            attn_only=True, pos_embed="rope", use_causal_mask=False,
        )
        model = Transformer(cfg)
        logits = model(tokens)                       # (batch, n_ctx, d_vocab)

        # With attention weights
        logits, attn_weights = model(tokens, return_attn_weights=True)

        # Architecture that mirrors a HuggingFace model (random weights, no download)
        model = Transformer.from_hf_config("google/gemma-2-2b-it")
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Embeddings ─────────────────────────────────────────────────────
        self.W_E = nn.Embedding(cfg.d_vocab, cfg.d_model)

        if cfg.pos_embed == "learned":
            self.W_pos = nn.Embedding(cfg.n_ctx, cfg.d_model)
        elif cfg.pos_embed == "sinusoidal":
            self.register_buffer(
                "pos_embed_fixed", _make_sinusoidal_pe(cfg.n_ctx, cfg.d_model)
            )

        # ── Transformer blocks ─────────────────────────────────────────────
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        # ── Final layer-norm (pre-unembed) ─────────────────────────────────
        if cfg.use_layernorm:
            self.norm_final = nn.LayerNorm(cfg.d_model)

        # ── Unembedding ────────────────────────────────────────────────────
        # Weight tying (W_U ≈ W_E.T) is common but not always desired;
        # left untied here for flexibility.
        self.W_U = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        torch.manual_seed(self.cfg.model_seed)
        std = 1.0 / math.sqrt(self.cfg.d_model)
        nn.init.normal_(self.W_E.weight, std=std)
        if self.cfg.pos_embed == "learned":
            nn.init.normal_(self.W_pos.weight, std=std)
        nn.init.normal_(self.W_U.weight, std=std)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
    ):
        """
        Args:
            x:                   (batch, n_ctx) integer token indices.
            return_attn_weights: If True, also return per-block attention weights.

        Returns:
            logits        — (batch, n_ctx, d_vocab)
            attn_weights  — list of (batch, n_heads, n_ctx, n_ctx), one per block
                            [only when return_attn_weights=True]
        """
        _, T = x.shape

        residual = self.W_E(x)                                  # (B, T, d_model)

        if self.cfg.pos_embed == "learned":
            residual = residual + self.W_pos(torch.arange(T, device=x.device))
        elif self.cfg.pos_embed == "sinusoidal":
            residual = residual + self.pos_embed_fixed[:T]

        all_attn_weights = []
        for block in self.blocks:
            residual, attn_w = block(residual)
            if return_attn_weights:
                all_attn_weights.append(attn_w)

        if self.cfg.use_layernorm:
            residual = self.norm_final(residual)

        logits = self.W_U(residual)                             # (B, T, d_vocab)

        if return_attn_weights:
            return logits, all_attn_weights
        return logits

    # ── Convenience helpers ─────────────────────────────────────────────────

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities over the vocabulary, shape (batch, n_ctx, d_vocab)."""
        return F.softmax(self.forward(x), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Weight initialisation utility
# ─────────────────────────────────────────────────────────────────────────────

def initialize_weights(module: nn.Module, std: float = 0.02, seed: int = 42) -> None:
    """
    Apply GPT-2 style weight initialisation to a module tree in-place.

    Typically called as:
        model.apply(initialize_weights)
    or with a custom std:
        model.apply(lambda m: initialize_weights(m, std=0.01))
    """
    torch.manual_seed(seed)
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)