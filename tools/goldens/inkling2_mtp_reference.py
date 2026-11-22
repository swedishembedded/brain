#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Reference forward of the Inkling 2 MTP drafter (`mtp.safetensors`, the vendor's "DSpark" draft) and the golden files
brain's `dflash2` gates compare against.

Swedish Embedded AB implements speculative decoding - drafters, verifiers and the state rollback a looped or hybrid
stack needs - for its clients. If your team needs a drafter ported and gated against its reference then you can
procure our services by sending an email to info@swedishembedded.com.

The algorithm is the vendor's vLLM runtime (vLLM main 8c17e4692a7c, `vllm/models/inkling2/nvidia/dspark.py`,
`.../ops/dspark.py`, `speculator.py`, and the generic `vllm/v1/worker/gpu/spec_decode/{dflash,dspark}/speculator.py`
it extends), re-implemented here in plain torch at binary32 so the gate is independent of those CUDA kernels:

* **Context.** For every committed target position `p` the target's hidden states at its tapped logical depths
  (`target_layer_ids`: the carried state entering those visits, and the final normalised row at the full depth) are
  mixed per draft layer `l` with `softmax(fuse_weight[l])` over the taps, RMS-normed with `layers.l.dflare_norm`, and
  projected by `k_proj_ctx` / `v_proj_ctx`; the keys take the per-head `k_norm_ctx` and RoPE at `p`.
* **Block.** One block of `block_size` rows at positions `n .. n + block_size` (`n` = committed positions): row 0 is the
  anchor (the last sampled token, not yet consumed by the target), embedded by the target's embedding and its
  embedding norm; rows 1.. are the learned `mask_embed`. Each layer is pre-norm: q/k/v from `input_layernorm`, per-head
  q/k norms, RoPE (theta from the config, rotate-half), attention that is NOT causal: every row reads the whole block
  and the context (a local layer only the `sliding_window_size` positions before the block), then `o_proj`, the
  residual, `post_attention_layernorm` and a SwiGLU MLP. The block's own K/V join the keys.
* **Head.** Mask row `i` (1..) predicts the token at its own position. Its final-normed hidden `h_i` gives the base
  logits through the target's head. A recurrent Markov head corrects them left to right: `f = mix(features of position
  n - 1, softmax(fuse_weight[-1]))` normed by `markov_head.dflare_norm`; `state = 0`, `prev = anchor`; per step
  `raw = rnn_proj([state, markov_w1[prev], h_i, f])`, `gate = sigmoid(raw[:r])`, `state = gate * state + (1 - gate) *
  tanh(raw[r:2r])`, `bias = head(hidden_out(tanh(raw[2r:])))`, logits `= softcap(base + bias)`, the draft token its
  argmax, which becomes `prev`.

Two modes write `golden.safetensors` (f32):

    # a tiny random draft (draft/config.json, draft/mtp.safetensors) and its golden
    inkling2_mtp_reference.py tiny --out testdata/golden/inkling2/mtp-tiny

    # the released draft over real target features (the real golden's visit inputs at the tapped depths)
    inkling2_mtp_reference.py real --checkpoint MODELS/thinkingmachines/Inkling-2-Mini \\
        --features testdata/golden/inkling2/real-f32/golden.safetensors --out testdata/golden/inkling2/mtp-real
"""

import argparse
import json
import math
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_grad_enabled(False)


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos, theta):
    """Rotate-half RoPE over the last dim of `x` `[n, heads, d]` at positions `pos` `[n]`."""
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    ang = pos.double()[:, None] * inv[None]
    cos, sin = ang.cos().float()[:, None, :], ang.sin().float()[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class Draft:
    def __init__(self, cfg, w):
        self.c, self.w = cfg, w
        self.alpha = torch.softmax(w["fuse_weight"].float(), -1)

    def mix(self, feats, row, norm):
        """`feats` `[n, taps * d]`: the alpha-weighted tap sum, RMS-normed."""
        c = self.c
        t = feats.view(feats.shape[0], len(c["target_layer_ids"]), c["hidden_size"])
        mixed = (t * self.alpha[row][None, :, None]).sum(1)
        return rms(mixed, norm, c["rms_norm_eps"])

    def context_kv(self, feats, positions):
        c, w = self.c, self.w
        hd, nkv = c["head_dim"], c["num_key_value_heads"]
        out = []
        for l in range(c["num_hidden_layers"]):
            p = f"layers.{l}."
            ctx = self.mix(feats, l, w[p + "dflare_norm.weight"])
            k = (ctx @ w[p + "self_attn.k_proj_ctx.weight"].T).view(-1, nkv, hd)
            k = rope(rms(k, w[p + "self_attn.k_norm_ctx.weight"], c["rms_norm_eps"]), positions, c["rope_theta"])
            v = (ctx @ w[p + "self_attn.v_proj_ctx.weight"].T).view(-1, nkv, hd)
            out.append((k, v))
        return out

    def block(self, ctx_kv, anchor_row, n):
        """The block at positions `n ..`: per-layer outputs (pre-norm) and the final-normed rows."""
        c, w = self.c, self.w
        b, d = c["block_size"], c["hidden_size"]
        hd, nh, nkv, eps = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"], c["rms_norm_eps"]
        x = torch.cat([anchor_row[None], w["mask_embed"][None].expand(b - 1, d)], 0)
        pos = torch.arange(n, n + b)
        layers = []
        for l in range(c["num_hidden_layers"]):
            p = f"layers.{l}."
            h = rms(x, w[p + "input_layernorm.weight"], eps)
            q = rope(rms((h @ w[p + "self_attn.q_proj.weight"].T).view(b, nh, hd), w[p + "self_attn.q_norm.weight"], eps), pos, c["rope_theta"])
            k = rope(rms((h @ w[p + "self_attn.k_proj.weight"].T).view(b, nkv, hd), w[p + "self_attn.k_norm.weight"], eps), pos, c["rope_theta"])
            v = (h @ w[p + "self_attn.v_proj.weight"].T).view(b, nkv, hd)
            lo = max(0, n - c["sliding_window_size"]) if l in c["local_layer_ids"] else 0
            ck, cv = ctx_kv[l]
            keys = torch.cat([ck[lo:n], k], 0).repeat_interleave(nh // nkv, 1)
            vals = torch.cat([cv[lo:n], v], 0).repeat_interleave(nh // nkv, 1)
            s = torch.einsum("qhd,khd->hqk", q, keys) * hd**-0.5
            a = torch.einsum("hqk,khd->qhd", s.softmax(-1), vals).reshape(b, nh * hd)
            x = x + a @ w[p + "self_attn.o_proj.weight"].T
            h = rms(x, w[p + "post_attention_layernorm.weight"], eps)
            g, u = h @ w[p + "mlp.gate_proj.weight"].T, h @ w[p + "mlp.up_proj.weight"].T
            x = x + (torch.nn.functional.silu(g) * u) @ w[p + "mlp.down_proj.weight"].T
            layers.append(x.clone())
        return layers, rms(x, w["norm.weight"], eps)

    def markov(self, hidden, last_feats, anchor, head, cap, valid):
        """Greedy Markov walk over the mask rows' normed hidden `[b - 1, d]`: tokens and per-step logits."""
        c, w = self.c, self.w
        r = c["markov_rank"]
        f = self.mix(last_feats[None], -1, w["markov_head.dflare_norm.weight"])[0]
        W, bias = w["markov_head.rnn_proj.weight"], w["markov_head.rnn_proj.bias"]
        state, prev = torch.zeros(r), anchor
        tokens, logits = [], []
        for i in range(hidden.shape[0]):
            h = hidden[i]
            raw = W @ torch.cat([state, w["markov_head.markov_w1.weight"][prev], h, f]) + bias
            gate = torch.sigmoid(raw[:r])
            state = gate * state + (1 - gate) * torch.tanh(raw[r : 2 * r])
            latent = torch.tanh(raw[2 * r :])
            corr = w["markov_head.hidden_out.weight"] @ latent + w["markov_head.hidden_out.bias"]
            lg = head[:valid] @ (h + corr)
            lg = cap * torch.tanh(lg / cap)
            prev = int(lg.argmax())
            tokens.append(prev)
            logits.append(lg)
        return tokens, torch.stack(logits)


def case(draft, feats, anchor, embed_row, head, cap, valid, prefix, out):
    """One draft round after `feats.shape[0]` committed positions, the golden tensors under `prefix`."""
    n = feats.shape[0]
    kv = draft.context_kv(feats, torch.arange(n))
    layers, normed = draft.block(kv, embed_row, n)
    tokens, logits = draft.markov(normed[1:], feats[n - 1], anchor, head, cap, valid)
    out[prefix + "taps"] = feats
    out[prefix + "anchor"] = torch.tensor([float(anchor)])
    for l, x in enumerate(layers):
        out[f"{prefix}layer.{l}"] = x
    out[prefix + "ctx_k.0"] = kv[0][0].reshape(n, -1)
    out[prefix + "ctx_v.0"] = kv[0][1].reshape(n, -1)
    out[prefix + "hidden"] = normed[1:]
    out[prefix + "logits"] = logits
    out[prefix + "path"] = torch.tensor([float(t) for t in tokens])
    print(f"{prefix}: n={n} anchor={anchor} path={tokens}")


TINY = {
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 48,
    "max_position_embeddings": 4096,
    "torch_dtype": "float32",
    "q_bias": False,
    "o_bias": False,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "local_layer_ids": [0, 2],
    "sliding_window_size": 6,
    "block_size": 5,
    "markov_rank": 8,
    "markov_head_type": "rnn_hidden",
    "target_layer_ids": [1, 3, 5, 6],
}
TINY_VOCAB = 64
TINY_CAP = 5.0


def tiny_weights(cfg, vocab, gen):
    d, hd = cfg["hidden_size"], cfg["head_dim"]
    nh, nkv, ff, r = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["intermediate_size"], cfg["markov_rank"]
    taps = len(cfg["target_layer_ids"])

    def m(*shape, scale=None):
        s = scale if scale is not None else 1.0 / math.sqrt(shape[-1])
        return torch.randn(*shape, generator=gen) * s

    def gain(n):
        return 1.0 + 0.2 * torch.randn(n, generator=gen)

    w = {"fuse_weight": m(cfg["num_hidden_layers"] + 1, taps, scale=1.0), "mask_embed": m(d, scale=1.0), "norm.weight": gain(d)}
    for l in range(cfg["num_hidden_layers"]):
        p = f"layers.{l}."
        w.update({
            p + "dflare_norm.weight": gain(d),
            p + "input_layernorm.weight": gain(d),
            p + "post_attention_layernorm.weight": gain(d),
            p + "self_attn.q_proj.weight": m(nh * hd, d),
            p + "self_attn.k_proj.weight": m(nkv * hd, d),
            p + "self_attn.v_proj.weight": m(nkv * hd, d),
            p + "self_attn.o_proj.weight": m(d, nh * hd),
            p + "self_attn.k_proj_ctx.weight": m(nkv * hd, d),
            p + "self_attn.v_proj_ctx.weight": m(nkv * hd, d),
            p + "self_attn.q_norm.weight": gain(hd),
            p + "self_attn.k_norm.weight": gain(hd),
            p + "self_attn.k_norm_ctx.weight": gain(hd),
            p + "mlp.gate_proj.weight": m(ff, d),
            p + "mlp.up_proj.weight": m(ff, d),
            p + "mlp.down_proj.weight": m(d, ff),
        })
    w.update({
        "markov_head.dflare_norm.weight": gain(d),
        "markov_head.rnn_proj.weight": m(3 * r, 2 * r + 2 * d, scale=0.5),
        "markov_head.rnn_proj.bias": m(3 * r, scale=0.3),
        "markov_head.hidden_out.weight": m(d, r, scale=1.0),
        "markov_head.hidden_out.bias": m(d, scale=0.3),
        "markov_head.markov_w1.weight": m(vocab, r, scale=1.0),
    })
    return w


def run_tiny(args):
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    print(f"seed {seed}")
    gen = torch.Generator().manual_seed(seed)
    cfg, vocab = dict(TINY), TINY_VOCAB
    w = tiny_weights(cfg, vocab, gen)
    draft = Draft(cfg, w)
    d, taps = cfg["hidden_size"], len(cfg["target_layer_ids"])
    # The target's side: its normed embedding rows and its head (logits soft-capped at TINY_CAP).
    embed = torch.randn(vocab, d, generator=gen)
    head = torch.randn(vocab, d, generator=gen) / math.sqrt(d) * 2.0
    feats_b = torch.randn(17, taps * d, generator=gen) * 1.5
    out = {"embed": embed, "head": head}
    # Round a after 14 committed positions; round b after 17, sharing the first 14 (the context grows).
    case(draft, feats_b[:14].clone(), 9, embed[9], head, TINY_CAP, vocab, "a.", out)
    case(draft, feats_b, 23, embed[23], head, TINY_CAP, vocab, "b.", out)
    os.makedirs(os.path.join(args.out, "draft"), exist_ok=True)
    save_file({k: v.contiguous() for k, v in w.items()}, os.path.join(args.out, "draft", "mtp.safetensors"))
    with open(os.path.join(args.out, "draft", "config.json"), "w") as f:
        json.dump({"mtp_config": cfg, "vocab_size": vocab, "final_logit_softcapping": TINY_CAP}, f, indent=1)
    save_file({k: v.float().contiguous() for k, v in out.items()}, os.path.join(args.out, "golden.safetensors"))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"kind": "tiny", "seed": seed, "cases": {"a": 14, "b": 17}, "softcap": TINY_CAP}, f, indent=1)


def run_real(args):
    ckpt = args.checkpoint
    with open(os.path.join(ckpt, "config.json")) as f:
        full = json.load(f)
    cfg, text = full["mtp_config"], full["text_config"]
    w = {}
    with safe_open(os.path.join(ckpt, "mtp.safetensors"), "pt") as f:
        for k in f.keys():
            w[k] = f.get_tensor(k).float()
    with safe_open(os.path.join(ckpt, "model-00001-of-00001.safetensors"), "pt") as f:
        embed = f.get_tensor("model.llm.embed.weight")
        embed_norm = f.get_tensor("model.llm.embed_norm.weight").float()
        head = f.get_tensor("model.llm.unembed.weight").float()
    draft = Draft(cfg, w)
    valid, cap, eps = text["unpadded_vocab_size"], text["final_logit_softcapping"], text["rms_norm_eps"]
    visits = cfg["target_layer_ids"]
    out = {}
    with safe_open(args.features, "pt") as g:
        for i in range(args.prompts):
            p = f"p{i}."
            tokens = [int(t) for t in g.get_tensor(p + "tokens").tolist()]
            # Features: the visit inputs at the tapped depths; the full depth's final normalised row is not in the real
            # golden, so the last tap reads the coda's output there (any hidden-like row: this gate is the drafter given
            # its inputs; the served engine's end-to-end acceptance is the gate of its real taps).
            total = sum(1 for k in g.keys() if k.startswith(p + "visit") and k.endswith(".in"))
            rows = [g.get_tensor(f"{p}visit{v}.in") if v < total else g.get_tensor(f"{p}visit{total - 1}.out") for v in visits]
            feats = torch.cat(rows, -1).float()
            n = feats.shape[0] - 1
            anchor = tokens[n]
            anchor_row = rms(embed[anchor].float(), embed_norm, eps)
            case(draft, feats[:n].clone(), anchor, anchor_row, head, cap, valid, p, out)
            out[p + "anchor_embed"] = anchor_row
    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.float().contiguous() for k, v in out.items()}, os.path.join(args.out, "golden.safetensors"))
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"kind": "real", "features": os.path.basename(os.path.dirname(args.features)), "prompts": args.prompts, "softcap": cap, "valid_vocab": valid}, f, indent=1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "tiny":
        ap = argparse.ArgumentParser(prog="inkling2_mtp_reference.py tiny")
        ap.add_argument("--out", required=True)
        ap.add_argument("--seed", type=int, default=None)
        run_tiny(ap.parse_args(sys.argv[2:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "real":
        ap = argparse.ArgumentParser(prog="inkling2_mtp_reference.py real")
        ap.add_argument("--checkpoint", required=True)
        ap.add_argument("--features", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--prompts", type=int, default=3)
        run_real(ap.parse_args(sys.argv[2:]))
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
