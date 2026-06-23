#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""
tiny_sparse_moe.py

A tiny but complete sparse Mixture-of-Experts decoder-only Transformer for
64-token sequence modeling.

Features:
- True sparse top-k MoE dispatch: tokens are evaluated only by selected experts.
- Router softmax + top-k routing with normalized combine weights.
- Expert capacity factor with overflow dropping.
- Auxiliary load-balancing loss and router z-loss.
- Optional noisy router during training.
- Causal self-attention with RoPE positional encoding.
- RMSNorm, SwiGLU experts, residual blocks.
- Train/eval/generate CLI.
- Synthetic 64-token corpus or external .npy/.bin data.
- Checkpoint save/load.

Example:
  python tiny_sparse_moe.py train --steps 1000 --device cpu
  python tiny_sparse_moe.py generate --ckpt moe.pt --prompt 1,2,3,4 --max-new 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Config
# -----------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 64
    block_size: int = 64
    n_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0

    # MoE
    n_experts: int = 4
    top_k: int = 2
    d_ff: int = 128
    capacity_factor: float = 1.25
    router_noise_std: float = 0.0
    aux_loss_coef: float = 0.01
    z_loss_coef: float = 1e-4
    expert_dropout: float = 0.0

    # Init
    init_std: float = 0.02

    def validate(self) -> None:
        assert self.vocab_size == 64, "This script is configured for a 64-token vocabulary."
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads."
        assert (self.d_model // self.n_heads) % 2 == 0, "RoPE requires even head_dim."
        assert 1 <= self.top_k <= self.n_experts, "top_k must be in [1, n_experts]."
        assert self.block_size > 0
        assert self.capacity_factor > 0


# -----------------------------
# Layers
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x * rsqrt(mean(x^2)) * scale
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # [T, head_dim/2]
        self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k: [B, H, T, D]
        T = q.size(-2)
        cos = self.cos[:, :, :T, :].to(dtype=q.dtype)
        sin = self.sin[:, :, :T, :].to(dtype=q.dtype)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D], cos/sin: [1,1,T,D/2]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = cfg.attn_dropout
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)
        self.rope = RotaryEmbedding(self.head_dim, cfg.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        # PyTorch SDPA uses fast kernels where available. is_causal=True gives autoregressive masking.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out(y))


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))


class SparseMoE(nn.Module):
    """
    True sparse top-k Mixture-of-Experts feed-forward block.

    For each token:
      1. router produces probabilities over experts
      2. top-k experts are selected
      3. selected expert outputs are computed only for tokens routed to that expert
      4. outputs are weighted by normalized router weights and combined with index_add_

    Capacity handling:
      Each expert can process at most ceil(capacity_factor * N * top_k / n_experts)
      token-expert assignments. Overflow assignments are dropped by lowest router weight.
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
        N = B * T
        E = self.cfg.n_experts
        K = self.cfg.top_k
        x_flat = x.reshape(N, C)

        router_logits = self.router(x_flat)  # [N,E]
        if self.training and self.cfg.router_noise_std > 0.0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.cfg.router_noise_std

        router_probs = F.softmax(router_logits, dim=-1)  # [N,E]
        top_vals, top_idx = torch.topk(router_probs, k=K, dim=-1)  # [N,K]
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Load balancing loss. Minimum is ~1.0 when expert usage and router mass are balanced.
        with torch.no_grad():
            selected = F.one_hot(top_idx, num_classes=E).float()  # [N,K,E]
            tokens_per_expert = selected.sum(dim=(0, 1)) / float(N * K)  # [E], sums to 1
        prob_per_expert = router_probs.mean(dim=0)  # [E], sums to 1
        aux_loss = E * torch.sum(tokens_per_expert.to(router_probs.dtype) * prob_per_expert)

        # Router z-loss discourages very large router logits.
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        moe_loss = self.cfg.aux_loss_coef * aux_loss + self.cfg.z_loss_coef * z_loss

        # Actual sparse dispatch.
        y_flat = torch.zeros_like(x_flat)
        capacity = max(1, math.ceil(self.cfg.capacity_factor * (N * K) / E))
        total_assignments = N * K
        kept_assignments = 0

        # Loop over experts. This is intentionally sparse: each expert sees only its own tokens.
        for expert_id, expert in enumerate(self.experts):
            pos = (top_idx == expert_id).nonzero(as_tuple=False)  # [M,2] => token_index, kth_route
            if pos.numel() == 0:
                continue

            token_indices = pos[:, 0]
            route_slots = pos[:, 1]
            weights = top_vals[token_indices, route_slots]  # [M]

            # Capacity overflow: keep highest-confidence assignments for this expert.
            if token_indices.numel() > capacity:
                keep = torch.topk(weights, k=capacity, largest=True, sorted=False).indices
                token_indices = token_indices[keep]
                route_slots = route_slots[keep]
                weights = weights[keep]

            kept_assignments += int(token_indices.numel())
            expert_in = x_flat[token_indices]
            expert_out = expert(expert_in) * weights.unsqueeze(-1)
            y_flat.index_add_(0, token_indices, expert_out)

        drop_frac = 1.0 - (kept_assignments / float(total_assignments))
        with torch.no_grad():
            entropy = -(router_probs * router_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            hard_usage = F.one_hot(top_idx[:, 0], num_classes=E).float().mean(dim=0)
            used_experts = int((hard_usage > 0).sum().item())
            max_usage = float(hard_usage.max().item())
            min_usage = float(hard_usage.min().item())
            stats = {
                "moe_aux_raw": float(aux_loss.detach().item()),
                "moe_z_raw": float(z_loss.detach().item()),
                "moe_drop_frac": float(drop_frac),
                "moe_router_entropy": float(entropy.detach().item()),
                "moe_used_experts": float(used_experts),
                "moe_max_top1_usage": max_usage,
                "moe_min_top1_usage": min_usage,
                "moe_capacity": float(capacity),
            }

        return y_flat.view(B, T, C), moe_loss, stats


class TransformerBlock(nn.Module):
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


class TinySparseMoETransformer(nn.Module):
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
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.cfg.block_size}")

        x = self.drop(self.token_emb(idx))
        total_moe_loss = x.new_zeros(())
        all_stats: List[Dict[str, float]] = []

        for block in self.blocks:
            x, moe_loss, stats = block(x)
            total_moe_loss = total_moe_loss + moe_loss
            all_stats.append(stats)

        x = self.norm(x)
        logits = self.lm_head(x)

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
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# -----------------------------
# Data
# -----------------------------

def make_toy_corpus(num_tokens: int, vocab_size: int = 64, seed: int = 123) -> torch.Tensor:
    """
    Creates a learnable but nontrivial 64-token sequence.

    Every token is a deterministic function of the two PRECEDING tokens (plus
    rare random resets), so the rule is fully recoverable from local context.

    Why this matters: training samples random fixed-length windows and the model
    only ever observes token *values* (RoPE encodes positions within a window,
    which always starts at 0). It never sees the absolute index ``i``. Any term
    that depends on ``i`` would therefore be pure noise the model cannot fit,
    capping achievable accuracy. So the recurrence must depend only on previous
    token values.

    The map ``(t[i-2], t[i-1]) -> (t[i-1], t[i])`` with
    ``t[i] = (t[i-2] + table[t[i-1]]) % vocab`` is a bijection on the state
    space. That guarantees a long, mixing orbit and avoids the fixed-point /
    short-cycle collapse that a plain linear recurrence mod ``vocab`` can fall
    into (which is the trap the original position-dependent terms were papering
    over).
    """
    rng = np.random.default_rng(seed)
    table = rng.permutation(vocab_size).astype(np.int64)  # fixed pseudo-random mixing table
    data = np.zeros(num_tokens, dtype=np.int64)
    data[0] = rng.integers(0, vocab_size)
    data[1] = rng.integers(0, vocab_size)
    for i in range(2, num_tokens):
        if i % 257 == 0:
            data[i] = rng.integers(0, vocab_size)  # sparse random reset: small irreducible noise
        else:
            data[i] = (data[i - 2] + table[data[i - 1]]) % vocab_size
    return torch.from_numpy(data)


def load_tokens(path: Optional[str], vocab_size: int, toy_tokens: int, seed: int) -> torch.Tensor:
    if path is None:
        return make_toy_corpus(toy_tokens, vocab_size=vocab_size, seed=seed)
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        # Raw bytes become token IDs modulo vocab_size.
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
    arr = np.asarray(arr, dtype=np.int64).reshape(-1) % vocab_size
    if arr.size < 2:
        raise ValueError("Need at least 2 tokens.")
    return torch.from_numpy(arr.copy())


def split_train_val(data: torch.Tensor, val_frac: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
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
    # A window of length block_size needs its next-token target at i+block_size,
    # so the last valid start is len(data)-block_size-1. randint's upper bound is
    # exclusive, hence high = len(data)-block_size (this includes that last start).
    if len(data) < block_size + 1:
        raise ValueError("Data must be at least block_size + 1 tokens long")
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# -----------------------------
# Train/eval helpers
# -----------------------------

def average_stats(stats: Iterable[Dict[str, float]]) -> Dict[str, float]:
    stats = list(stats)
    if not stats:
        return {}
    out: Dict[str, float] = {}
    keys = sorted(set().union(*(s.keys() for s in stats)))
    for k in keys:
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
) -> Dict[str, float]:
    model.eval()
    result = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        stats_list = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, batch_size, model.cfg.block_size, device)
            _, loss, stats = model(xb, yb)
            assert loss is not None
            losses.append(float(loss.item()))
            stats_list.append(stats)
        avg = average_stats(stats_list)
        result[f"{split}_loss"] = float(sum(losses) / len(losses))
        for k, v in avg.items():
            result[f"{split}_{k}"] = v
    model.train()
    return result


def save_checkpoint(path: str, model: TinySparseMoETransformer, optimizer: torch.optim.Optimizer, step: int) -> None:
    payload = {
        "config": asdict(model.cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    torch.save(payload, path)


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


# -----------------------------
# CLI commands
# -----------------------------

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

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
            toks = step * args.batch_size * cfg.block_size
            tok_s = toks / max(dt, 1e-9)
            msg = (
                f"step {step:5d} | loss {loss.item():.4f} | "
                f"moe {stats.get('loss_moe', 0.0):.4f} | "
                f"drop {stats.get('moe_drop_frac', 0.0):.3f} | "
                f"used {stats.get('moe_used_experts', 0.0):.0f}/{cfg.n_experts} | "
                f"tok/s {tok_s:.0f}"
            )
            print(msg)

        if step % args.eval_every == 0 or step == args.steps:
            metrics = estimate_loss(model, train_data, val_data, args.batch_size, args.eval_iters, args.device)
            print("eval " + json.dumps(metrics, sort_keys=True))
            save_checkpoint(args.ckpt, model, optimizer, step)
            print(f"saved {args.ckpt}")


def parse_prompt(prompt: str) -> List[int]:
    if not prompt.strip():
        return [0]
    vals = []
    for part in prompt.replace(" ", ",").split(","):
        if part == "":
            continue
        v = int(part)
        if not 0 <= v < 64:
            raise ValueError("Prompt tokens must be integers in [0,63].")
        vals.append(v)
    return vals or [0]


def cmd_generate(args: argparse.Namespace) -> None:
    model, _, step = load_checkpoint(args.ckpt, args.device)
    model.eval()
    prompt = parse_prompt(args.prompt)
    idx = torch.tensor([prompt], dtype=torch.long, device=args.device)
    out = model.generate(idx, max_new_tokens=args.max_new, temperature=args.temperature, top_k=args.sample_top_k)
    toks = out[0].tolist()
    print(f"checkpoint_step={step}")
    print(",".join(map(str, toks)))


def cmd_eval(args: argparse.Namespace) -> None:
    model, _, step = load_checkpoint(args.ckpt, args.device)
    data = load_tokens(args.data, model.cfg.vocab_size, args.toy_tokens, args.seed)
    train_data, val_data = split_train_val(data, args.val_frac)
    metrics = estimate_loss(model, train_data, val_data, args.batch_size, args.eval_iters, args.device)
    print(f"checkpoint_step={step}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tiny fully-featured sparse MoE Transformer for 64-token sequences")
    sub = p.add_subparsers(required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data", type=str, default=None, help="Optional .npy token file or raw binary file; values are modulo 64")
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

    gen = sub.add_parser("generate")
    gen.add_argument("--ckpt", type=str, default="moe.pt")
    gen.add_argument("--prompt", type=str, default="0,1,2,3")
    gen.add_argument("--max-new", type=int, default=64)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--sample-top-k", type=int, default=None)
    gen.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    gen.set_defaults(func=cmd_generate)

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
