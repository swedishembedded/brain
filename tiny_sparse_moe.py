#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""
tiny_sparse_moe.py
==================

A tiny — but complete and *correct* — sparse Mixture-of-Experts (MoE),
decoder-only Transformer for 64-token sequence modeling. It is small enough to
train on a CPU in seconds, yet contains every moving part of a real MoE LLM.

The code is organised top-to-bottom as a reading order:

    1. Config          – one dataclass describing the whole model.
    2. Layers          – RMSNorm, RoPE, attention, expert, sparse MoE, block.
    3. Model           – the full decoder-only Transformer + generation.
    4. Data            – how the train / eval token stream is built and batched.
    5. Train / eval    – the optimisation loop and (deterministic) evaluation.
    6. CLI             – `train`, `eval`, `generate` sub-commands.

Each section opens with a short explanation, and each layer's docstring states
*what it computes and why*.

--------------------------------------------------------------------------------
How the training / eval data is constructed (read this before changing it!)
--------------------------------------------------------------------------------
We model a 1-D stream of integer token IDs in ``[0, 64)``.

* Source (`make_toy_corpus` / `load_tokens`): either a synthetic corpus or an
  external ``.npy`` / raw-bytes file (bytes are taken modulo 64).
* Split  (`split_train_val`): the stream is cut sequentially — the first
  ``1 - val_frac`` is train, the tail is validation. No shuffling: for a single
  continuous stream that is the correct, leak-free split.
* Batch  (`get_batch`): we draw random length-``block_size`` windows; the input
  ``x`` is the window and the target ``y`` is the same window shifted by one
  (standard next-token prediction).

The synthetic task is deliberately a deterministic function of the *previous two
tokens only* — never of the absolute position ``i``. That is essential: training
crops random windows, so the model only ever sees token values (RoPE encodes
positions *within* a window, which always starts at 0). A rule depending on the
absolute index would be unobservable noise the model could never fit. See
`make_toy_corpus` for the full reasoning.

--------------------------------------------------------------------------------
Examples
--------------------------------------------------------------------------------
    python tiny_sparse_moe.py train    --steps 1000 --device cpu
    python tiny_sparse_moe.py eval     --ckpt moe.pt
    python tiny_sparse_moe.py generate --ckpt moe.pt --prompt 1,2,3,4 --max-new 64
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Config
# ==============================================================================

@dataclass
class ModelConfig:
    """All hyper-parameters that define the model architecture.

    Stored in the checkpoint so a model can be rebuilt exactly on load.
    """

    # --- Transformer backbone ---
    vocab_size: int = 64       # number of distinct token IDs
    block_size: int = 64       # maximum sequence length (context window)
    n_layers: int = 2          # number of stacked Transformer blocks
    d_model: int = 64          # residual-stream / embedding width
    n_heads: int = 4           # attention heads (head_dim = d_model // n_heads)
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    # --- Mixture-of-Experts feed-forward ---
    n_experts: int = 4         # total experts per MoE layer
    top_k: int = 2             # experts each token is routed to
    d_ff: int = 128            # hidden width inside each expert
    capacity_factor: float = 1.25   # head-room multiplier for per-expert capacity
    router_noise_std: float = 0.0   # exploration noise added to router logits (train only)
    aux_loss_coef: float = 0.01     # weight of the load-balancing loss
    z_loss_coef: float = 1e-4       # weight of the router z-loss
    expert_dropout: float = 0.0

    # --- Initialisation ---
    init_std: float = 0.02

    def validate(self) -> None:
        """Fail fast on impossible configurations."""
        assert self.vocab_size == 64, "This script is configured for a 64-token vocabulary."
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads."
        assert (self.d_model // self.n_heads) % 2 == 0, "RoPE requires an even head_dim."
        assert 1 <= self.top_k <= self.n_experts, "top_k must be in [1, n_experts]."
        assert self.block_size > 0
        assert self.capacity_factor > 0


# ==============================================================================
# 2. Layers
# ==============================================================================

class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm.

    Rescales each token vector to unit RMS, then applies a learned per-channel
    gain. Unlike LayerNorm it does not subtract the mean or add a bias, which
    makes it cheaper and is the norm used by most modern LLMs.

        y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Instead of adding a position vector, RoPE *rotates* each (even, odd) pair of
    channels in q and k by an angle proportional to the token position. The dot
    product of two rotated vectors then depends only on their *relative*
    distance, giving the model relative-position awareness for free.

    The cos/sin tables are precomputed once for all positions up to
    ``max_seq_len`` and cached as (non-persistent) buffers.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0
        # Frequencies: lower channels rotate fast, higher channels rotate slowly.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)            # [T, head_dim/2]
        self.register_buffer("cos", freqs.cos()[None, None], persistent=False)  # [1,1,T,head_dim/2]
        self.register_buffer("sin", freqs.sin()[None, None], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, H, T, head_dim]
        T = q.size(-2)
        cos = self.cos[:, :, :T, :].to(dtype=q.dtype)
        sin = self.sin[:, :, :T, :].to(dtype=q.dtype)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive channel pairs (x_even, x_odd) of ``x`` by the RoPE angle.

    x: [B, H, T, head_dim]; cos/sin: [1, 1, T, head_dim/2].
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    # Re-interleave the rotated pairs back into the original channel layout.
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    Standard scaled-dot-product attention with two specifics:
      * RoPE is applied to q and k so attention scores carry relative position.
      * ``is_causal=True`` masks each token from attending to future tokens,
        which is what makes this a left-to-right (autoregressive) model.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        # One fused projection produces q, k and v together, then we split it.
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = cfg.attn_dropout
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)
        self.rope = RotaryEmbedding(self.head_dim, cfg.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to q, k, v and reshape to per-head tensors [B, H, T, head_dim].
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Inject relative position into q and k.
        q, k = self.rope(q, k)

        # Fused, memory-efficient attention; is_causal handles the triangular mask.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        # Merge heads back to [B, T, C] and project out.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out(y))


class SwiGLUExpert(nn.Module):
    """A single feed-forward expert using the SwiGLU activation.

        h = SiLU(W_gate x) * (W_up x)     # gated activation
        y = W_down h

    SwiGLU consistently outperforms a plain ReLU/GELU MLP at the same parameter
    budget, which is why modern LLMs use it.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))


class SparseMoE(nn.Module):
    """Sparse top-k Mixture-of-Experts feed-forward block.

    Replaces the single dense FFN of a vanilla Transformer with ``n_experts``
    independent FFNs. A lightweight *router* picks the ``top_k`` experts for each
    token, so only a fraction of the parameters run per token — the model gets
    more capacity at roughly constant compute.

    Per token, per layer:
      1. Router produces a probability over experts (softmax of a linear map).
      2. Keep the top_k experts; renormalise their probabilities to sum to 1.
      3. Run *only* the selected experts on *only* their assigned tokens.
      4. Combine expert outputs weighted by the renormalised router weights.

    Two auxiliary losses keep training healthy (added to the main loss):
      * Load-balancing loss  — encourages tokens to spread across experts so a
        few experts don't dominate while others starve.
      * Router z-loss        — penalises large router logits for numerical
        stability (keeps the softmax well-behaved).

    Capacity: each expert processes at most
    ``ceil(capacity_factor * N * top_k / n_experts)`` token-assignments per
    batch. This bounds memory; assignments beyond capacity are dropped, keeping
    the highest-confidence ones. ``moe_drop_frac`` in the stats reports how many
    were dropped.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(cfg.d_model, cfg.d_ff, cfg.expert_dropout)
            for _ in range(cfg.n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        B, T, C = x.shape
        N = B * T                       # total tokens in the batch
        E = self.cfg.n_experts
        K = self.cfg.top_k
        x_flat = x.reshape(N, C)        # treat every token independently

        # --- 1. Route ---------------------------------------------------------
        router_logits = self.router(x_flat)                 # [N, E]
        if self.training and self.cfg.router_noise_std > 0.0:
            # Noisy gating: encourages exploration of under-used experts early on.
            router_logits = router_logits + torch.randn_like(router_logits) * self.cfg.router_noise_std

        router_probs = F.softmax(router_logits, dim=-1)      # [N, E]
        top_vals, top_idx = torch.topk(router_probs, k=K, dim=-1)   # [N, K]
        # Renormalise the kept probabilities so the combine weights sum to 1.
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # --- 2. Auxiliary losses ---------------------------------------------
        # Load balancing (Switch-Transformer form): product of the fraction of
        # tokens dispatched to each expert and the mean router probability mass
        # on it. The token-fraction term is treated as a constant (no_grad); the
        # gradient flows only through prob_per_expert. Minimised (~1.0) when both
        # are uniform across experts.
        with torch.no_grad():
            selected = F.one_hot(top_idx, num_classes=E).float()        # [N, K, E]
            tokens_per_expert = selected.sum(dim=(0, 1)) / float(N * K)  # [E], sums to 1
        prob_per_expert = router_probs.mean(dim=0)                       # [E], sums to 1
        aux_loss = E * torch.sum(tokens_per_expert.to(router_probs.dtype) * prob_per_expert)

        # z-loss: keeps logits from blowing up (logsumexp ~ "soft max" magnitude).
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        moe_loss = self.cfg.aux_loss_coef * aux_loss + self.cfg.z_loss_coef * z_loss

        # --- 3. Sparse dispatch ----------------------------------------------
        y_flat = torch.zeros_like(x_flat)
        capacity = max(1, math.ceil(self.cfg.capacity_factor * (N * K) / E))
        total_assignments = N * K
        kept_assignments = 0

        # Each expert processes only the tokens routed to it.
        for expert_id, expert in enumerate(self.experts):
            # Which (token, route-slot) pairs picked this expert?
            pos = (top_idx == expert_id).nonzero(as_tuple=False)   # [M, 2] -> (token_idx, kth_route)
            if pos.numel() == 0:
                continue

            token_indices = pos[:, 0]
            route_slots = pos[:, 1]
            weights = top_vals[token_indices, route_slots]         # combine weight per assignment

            # Capacity overflow: keep the highest-confidence assignments only.
            if token_indices.numel() > capacity:
                keep = torch.topk(weights, k=capacity, largest=True, sorted=False).indices
                token_indices = token_indices[keep]
                route_slots = route_slots[keep]
                weights = weights[keep]

            kept_assignments += int(token_indices.numel())
            expert_out = expert(x_flat[token_indices]) * weights.unsqueeze(-1)
            # Scatter-add: a token routed to K experts accumulates K weighted outputs.
            y_flat.index_add_(0, token_indices, expert_out)

        # --- 4. Diagnostics (not used for gradients) -------------------------
        drop_frac = 1.0 - (kept_assignments / float(total_assignments))
        with torch.no_grad():
            entropy = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            top1_usage = F.one_hot(top_idx[:, 0], num_classes=E).float().mean(dim=0)
            stats = {
                "moe_aux_raw": float(aux_loss.detach().item()),
                "moe_z_raw": float(z_loss.detach().item()),
                "moe_drop_frac": float(drop_frac),
                "moe_router_entropy": float(entropy.detach().item()),
                "moe_used_experts": float(int((top1_usage > 0).sum().item())),
                "moe_max_top1_usage": float(top1_usage.max().item()),
                "moe_min_top1_usage": float(top1_usage.min().item()),
                "moe_capacity": float(capacity),
            }

        return y_flat.view(B, T, C), moe_loss, stats


class TransformerBlock(nn.Module):
    """One pre-norm Transformer block: attention then MoE, each with a residual.

        x = x + Attention(RMSNorm(x))
        x = x + MoE(RMSNorm(x))

    Pre-norm (normalising the *input* to each sub-layer) gives stable gradients
    and is the standard choice for deep Transformers.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.moe = SparseMoE(cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        x = x + self.attn(self.norm1(x))
        moe_out, moe_loss, stats = self.moe(self.norm2(x))
        x = x + moe_out
        return x, moe_loss, stats


# ==============================================================================
# 3. Model
# ==============================================================================

class TinySparseMoETransformer(nn.Module):
    """Decoder-only Transformer: embed -> N blocks -> norm -> tied output head.

    The token embedding and the output projection share weights ("weight
    tying"), which saves parameters and usually improves small models.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.resid_dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.token_emb.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """idx: [B, T] token IDs. If ``targets`` is given, also returns the loss.

        Total loss = next-token cross-entropy + sum of every layer's MoE loss.
        """
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.cfg.block_size}")

        x = self.drop(self.token_emb(idx))                  # [B, T, d_model]

        total_moe_loss = x.new_zeros(())
        all_stats: List[Dict[str, float]] = []
        for block in self.blocks:
            x, moe_loss, stats = block(x)
            total_moe_loss = total_moe_loss + moe_loss
            all_stats.append(stats)

        x = self.norm(x)
        logits = self.lm_head(x)                            # [B, T, vocab_size]

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            loss = ce + total_moe_loss

        stats = average_stats(all_stats)
        if targets is not None:
            stats["loss_total"] = float(loss.detach().item())
            stats["loss_moe"] = float(total_moe_loss.detach().item())
        return logits, loss, stats

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively append ``max_new_tokens`` tokens to ``idx`` [B, T]."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]        # crop to context window
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)   # next-token logits
            if top_k is not None:
                # Restrict sampling to the top_k most likely tokens.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# ==============================================================================
# 4. Data
# ==============================================================================

def make_toy_corpus(num_tokens: int, vocab_size: int = 64, seed: int = 123) -> torch.Tensor:
    """Create a learnable but nontrivial token stream.

    Every token is a deterministic function of the two PRECEDING tokens (plus
    rare random resets), so the rule is fully recoverable from local context.

    Why "two preceding tokens only" and never the position ``i``:
        Training samples random fixed-length windows and the model only ever
        observes token *values* (RoPE encodes positions *within* a window, which
        always starts at 0). It never sees the absolute index ``i``. Any term
        that depends on ``i`` would therefore be pure noise the model cannot fit,
        artificially capping accuracy. So the recurrence depends only on previous
        token values.

    Why this particular map:
        ``t[i] = (t[i-2] + table[t[i-1]]) % vocab`` makes the state transition
        ``(t[i-2], t[i-1]) -> (t[i-1], t[i])`` a bijection on the V*V state
        space. A bijection has only long cycles, so the stream never collapses to
        a fixed point or short cycle — the failure mode a plain linear recurrence
        mod V can fall into (and the reason position-dependent terms were
        previously, but wrongly, used to mask it). ``table`` is a fixed
        seed-derived permutation, giving rich, learnable structure.
    """
    rng = np.random.default_rng(seed)
    table = rng.permutation(vocab_size).astype(np.int64)   # fixed pseudo-random mixing table
    data = np.zeros(num_tokens, dtype=np.int64)
    data[0] = rng.integers(0, vocab_size)
    data[1] = rng.integers(0, vocab_size)
    for i in range(2, num_tokens):
        if i % 257 == 0:
            data[i] = rng.integers(0, vocab_size)          # sparse reset: small irreducible noise
        else:
            data[i] = (data[i - 2] + table[data[i - 1]]) % vocab_size
    return torch.from_numpy(data)


def load_tokens(path: Optional[str], vocab_size: int, toy_tokens: int, seed: int) -> torch.Tensor:
    """Return a 1-D int64 tensor of token IDs in ``[0, vocab_size)``.

    ``path`` is None -> synthetic corpus; ``*.npy`` -> numpy array; otherwise the
    file is read as raw bytes. Any source is taken modulo ``vocab_size``.
    """
    if path is None:
        return make_toy_corpus(toy_tokens, vocab_size=vocab_size, seed=seed)
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
    arr = np.asarray(arr, dtype=np.int64).reshape(-1) % vocab_size
    if arr.size < 2:
        raise ValueError("Need at least 2 tokens.")
    return torch.from_numpy(arr.copy())


def split_train_val(data: torch.Tensor, val_frac: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cut the stream sequentially into (train, val). No shuffling: the tail is a
    genuine held-out continuation, so there is no train/val leakage."""
    n_val = max(1, int(len(data) * val_frac))
    n_train = len(data) - n_val
    if n_train < 2:
        raise ValueError("Dataset too small after split.")
    return data[:n_train], data[n_train:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Draw ``batch_size`` random next-token-prediction windows.

    ``x = data[i : i+block_size]`` and ``y = data[i+1 : i+block_size+1]``.

    The target ``y`` needs index ``i+block_size``, so the last valid start is
    ``len(data)-block_size-1``. ``randint``'s upper bound is exclusive, hence
    ``high = len(data)-block_size`` — this *includes* that last window (the
    previous code used ``...-1`` and silently never sampled it).

    Pass a seeded ``generator`` for reproducible batches (used by evaluation).
    """
    if len(data) < block_size + 1:
        raise ValueError("Data must be at least block_size + 1 tokens long")
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ==============================================================================
# 5. Train / eval helpers
# ==============================================================================

def average_stats(stats: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of stat dicts key-by-key (keys may differ between dicts)."""
    stats = list(stats)
    if not stats:
        return {}
    out: Dict[str, float] = {}
    for k in sorted(set().union(*(s.keys() for s in stats))):
        vals = [s[k] for s in stats if k in s]
        out[k] = float(sum(vals) / len(vals))
    return out


@torch.no_grad()
def estimate_loss(
    model: TinySparseMoETransformer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    eval_iters: int,
    device: str,
    eval_seed: int = 1234,
) -> Dict[str, float]:
    """Average loss + MoE stats over ``eval_iters`` batches per split.

    Evaluation uses a *fixed* RNG (``eval_seed``) so it samples the same windows
    every call. That makes the reported train/val numbers reproducible and
    directly comparable across training steps, instead of being a noisy estimate
    that wobbles with the global RNG state.
    """
    model.eval()
    result: Dict[str, float] = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        gen = torch.Generator().manual_seed(eval_seed)     # reset per split -> deterministic
        losses, stats_list = [], []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, batch_size, model.cfg.block_size, device, generator=gen)
            _, loss, stats = model(xb, yb)
            assert loss is not None
            losses.append(float(loss.item()))
            stats_list.append(stats)
        result[f"{split}_loss"] = float(sum(losses) / len(losses))
        for k, v in average_stats(stats_list).items():
            result[f"{split}_{k}"] = v
    model.train()
    return result


def save_checkpoint(path: str, model: TinySparseMoETransformer, optimizer: torch.optim.Optimizer, step: int) -> None:
    torch.save({
        "config": asdict(model.cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, path)


def load_checkpoint(path: str, device: str) -> Tuple[TinySparseMoETransformer, Optional[dict], int]:
    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt["config"])
    model = TinySparseMoETransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt.get("optimizer"), int(ckpt.get("step", 0))


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ==============================================================================
# 6. CLI commands
# ==============================================================================

def cmd_train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ModelConfig(
        vocab_size=64,
        block_size=args.block_size,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        attn_dropout=args.dropout,
        resid_dropout=args.dropout,
        n_experts=args.n_experts,
        top_k=args.top_k,
        d_ff=args.d_ff,
        capacity_factor=args.capacity_factor,
        router_noise_std=args.router_noise_std,
        aux_loss_coef=args.aux_loss_coef,
        z_loss_coef=args.z_loss_coef,
        expert_dropout=args.dropout,
    )
    cfg.validate()

    data = load_tokens(args.data, cfg.vocab_size, args.toy_tokens, args.seed)
    train_data, val_data = split_train_val(data, args.val_frac)

    model = TinySparseMoETransformer(cfg).to(args.device)
    total, trainable = count_parameters(model)
    print(f"config={json.dumps(asdict(cfg), sort_keys=True)}")
    print(f"params total={total:,} trainable={trainable:,}")
    print(f"tokens train={len(train_data):,} val={len(val_data):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    scaler_enabled = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(train_data, args.batch_size, cfg.block_size, args.device)
        optimizer.zero_grad(set_to_none=True)

        if scaler_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss, stats = model(xb, yb)
            assert loss is not None
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss, stats = model(xb, yb)
            assert loss is not None
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if step == 1 or step % args.log_every == 0:
            dt = time.time() - t0
            tok_s = step * args.batch_size * cfg.block_size / max(dt, 1e-9)
            print(
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"moe {stats.get('loss_moe', 0.0):.4f} | "
                f"drop {stats.get('moe_drop_frac', 0.0):.3f} | "
                f"used {stats.get('moe_used_experts', 0.0):.0f}/{cfg.n_experts} | "
                f"tok/s {tok_s:.0f}"
            )

        if step % args.eval_every == 0 or step == args.steps:
            metrics = estimate_loss(model, train_data, val_data, args.batch_size, args.eval_iters, args.device)
            print("eval " + json.dumps(metrics, sort_keys=True))
            save_checkpoint(args.ckpt, model, optimizer, step)
            print(f"saved {args.ckpt}")


def parse_prompt(prompt: str) -> List[int]:
    """Parse a comma/space separated prompt like "1,2,3" into a list of token IDs."""
    if not prompt.strip():
        return [0]
    vals = []
    for part in prompt.replace(" ", ",").split(","):
        if part == "":
            continue
        v = int(part)
        if not 0 <= v < 64:
            raise ValueError("Prompt tokens must be integers in [0, 63].")
        vals.append(v)
    return vals or [0]


def cmd_generate(args: argparse.Namespace) -> None:
    model, _, step = load_checkpoint(args.ckpt, args.device)
    model.eval()
    prompt = parse_prompt(args.prompt)
    idx = torch.tensor([prompt], dtype=torch.long, device=args.device)
    out = model.generate(idx, max_new_tokens=args.max_new, temperature=args.temperature, top_k=args.sample_top_k)
    print(f"checkpoint_step={step}")
    print(",".join(map(str, out[0].tolist())))


def cmd_eval(args: argparse.Namespace) -> None:
    model, _, step = load_checkpoint(args.ckpt, args.device)
    data = load_tokens(args.data, model.cfg.vocab_size, args.toy_tokens, args.seed)
    train_data, val_data = split_train_val(data, args.val_frac)
    metrics = estimate_loss(model, train_data, val_data, args.batch_size, args.eval_iters, args.device)
    print(f"checkpoint_step={step}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_export(args: argparse.Namespace) -> None:
    """Export a checkpoint to a flat binary the Rust/WGSL engine reads.

    Layout (little-endian):
        [u64 json_len][json_len bytes UTF-8 JSON][raw float32 tensor data]
    where JSON = {"config": {...},
                  "tensors": [{"name", "shape", "offset", "numel"}, ...]}
    and ``offset`` is measured in float32 elements from the start of the data
    section. Every parameter is stored row-major exactly as in PyTorch
    (Linear weights are [out_features, in_features]).
    """
    model, _, step = load_checkpoint(args.ckpt, "cpu")
    tensors, blobs, offset = [], [], 0
    for name, tensor in model.state_dict().items():
        flat = tensor.detach().to(torch.float32).reshape(-1).contiguous().numpy()
        tensors.append({"name": name, "shape": list(tensor.shape), "offset": offset, "numel": int(flat.size)})
        blobs.append(flat)
        offset += int(flat.size)

    header = json.dumps({"config": asdict(model.cfg), "tensors": tensors}).encode("utf-8")
    with open(args.out, "wb") as f:
        f.write(np.uint64(len(header)).tobytes())
        f.write(header)
        for blob in blobs:
            f.write(blob.tobytes())
    print(f"exported {len(tensors)} tensors ({offset:,} floats) from step {step} -> {args.out}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tiny fully-featured sparse MoE Transformer for 64-token sequences")
    sub = p.add_subparsers(required=True)

    # --- train ---
    tr = sub.add_parser("train")
    tr.add_argument("--data", type=str, default=None, help="Optional .npy or raw-binary token file; values are taken modulo 64")
    tr.add_argument("--toy-tokens", type=int, default=20_000)
    tr.add_argument("--val-frac", type=float, default=0.1)
    tr.add_argument("--ckpt", type=str, default="moe.pt")
    tr.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--seed", type=int, default=123)

    tr.add_argument("--steps", type=int, default=500)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight-decay", type=float, default=0.1)
    tr.add_argument("--grad-clip", type=float, default=1.0)
    tr.add_argument("--dropout", type=float, default=0.0)
    tr.add_argument("--amp", action="store_true", help="Use CUDA bfloat16 autocast")

    tr.add_argument("--block-size", type=int, default=64)
    tr.add_argument("--n-layers", type=int, default=2)
    tr.add_argument("--d-model", type=int, default=64)
    tr.add_argument("--n-heads", type=int, default=4)
    tr.add_argument("--d-ff", type=int, default=128)
    tr.add_argument("--n-experts", type=int, default=4)
    tr.add_argument("--top-k", type=int, default=2)
    tr.add_argument("--capacity-factor", type=float, default=1.25)
    tr.add_argument("--router-noise-std", type=float, default=0.0)
    tr.add_argument("--aux-loss-coef", type=float, default=0.01)
    tr.add_argument("--z-loss-coef", type=float, default=1e-4)

    tr.add_argument("--log-every", type=int, default=50)
    tr.add_argument("--eval-every", type=int, default=250)
    tr.add_argument("--eval-iters", type=int, default=20)
    tr.set_defaults(func=cmd_train)

    # --- eval ---
    ev = sub.add_parser("eval")
    ev.add_argument("--ckpt", type=str, default="moe.pt")
    ev.add_argument("--data", type=str, default=None)
    ev.add_argument("--toy-tokens", type=int, default=20_000)
    ev.add_argument("--val-frac", type=float, default=0.1)
    ev.add_argument("--batch-size", type=int, default=16)
    ev.add_argument("--eval-iters", type=int, default=50)
    ev.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ev.add_argument("--seed", type=int, default=123)
    ev.set_defaults(func=cmd_eval)

    # --- generate ---
    gen = sub.add_parser("generate")
    gen.add_argument("--ckpt", type=str, default="moe.pt")
    gen.add_argument("--prompt", type=str, default="0,1,2,3")
    gen.add_argument("--max-new", type=int, default=64)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--sample-top-k", type=int, default=None)
    gen.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    gen.set_defaults(func=cmd_generate)

    # --- export (for the Rust/WGSL engine) ---
    ex = sub.add_parser("export")
    ex.add_argument("--ckpt", type=str, default="moe.pt")
    ex.add_argument("--out", type=str, default="moe.weights")
    ex.set_defaults(func=cmd_export)

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
