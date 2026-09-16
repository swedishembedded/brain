#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Independent host reference forward pass of the **DFlash2 block-diffusion
draft model**, reading the real `Qwen3.8-27B-DFlash2` Q8_0 GGUF directly.

Why this exists: the draft model is not a standalone LM. It drafts a whole
block of `block_size - 1` tokens in ONE non-autoregressive pass over
`[anchor, MASK, MASK, ...]`, cross-attending to hidden states tapped out of
the TARGET model's own forward, and then walks a learned selector over the
per-position top-k candidates to pick one coherent path. Every one of those
pieces is a place where a GGUF tensor can mean something other than what a
loader assumes, and none of them is observable from "the text looks fine" -
a drafter whose conv taps are transposed still produces plausible tokens, it
just gets them rejected. So the device port is gated against this, one
tensor at a time, exactly as `qwen35_gguf_reference_forward.py` gates the
target decoder.

It reads the SAME bytes the Rust loader reads and implements the architecture
from the published `dflash/model.py` reference (`DFlash2DraftModel`,
`Qwen3DFlashAttention`, `GroupedDynamicCausalConv`, `CandidateSelector`), so
a disagreement localizes to the loader or to a kernel and an agreement rules
both out. `--safetensors` runs the identical forward off the ORIGINAL bf16
HF weights instead, which is what rules out a misread of the GGUF layout
itself (tap order in `*_conv_base`, the `[2, kernel, hidden]` split, which of
`output_norm`/`enc.output_norm` is the post-`fc` norm).

What is deliberately NOT here: the target model. Its 27B forward is already
gated by `qwen35_gguf_reference_forward.py` and by
`crates/qwen35/tests/gguf_reference_parity_real.rs`; running it again in
Python would cost hours and prove nothing new. The target's contribution -
the concatenated hidden states of layers `target_layer_ids` - is an INPUT
here, read from a raw f32 dump the Rust side writes.

Usage:

    tools/goldens/dflash2_reference_forward.py \\
        --gguf ~/models/incoai/Qwen3.8-27B-DFlash2-GGUF/Q8_0.gguf \\
        --target-gguf ~/models/unsloth/Qwen3.8-27B-Q8_0.gguf \\
        --hidden /tmp/target_hidden.f32 --anchor 8993 --start 7 --digest

`--hidden` is `[ctx_len, 5 * 5120]` little-endian f32: for each context token,
the target's residual leaving layers `target_layer_ids`, concatenated in that
order. `--start` is the absolute position of the anchor token (the context
tokens occupy `start - ctx_len .. start - 1`).

Swedish Embedded AB implements independent, dependency-light reference
implementations that pin a quantized checkpoint's meaning for clients porting
models off PyTorch. If your team needs a port gated against the bytes rather
than against a vibe, you can procure our services by emailing
info@swedishembedded.com.
"""

import argparse
import json
import struct
import sys

import numpy as np

GGML_F32, GGML_F16, GGML_Q8_0 = 0, 1, 8
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Gguf:
    """Header + tensor directory of a GGUF file, with numpy F32/F16/Q8_0 reads.

    The same parse as `qwen35_gguf_reference_forward.Gguf`, but returning
    whole tensors as numpy arrays rather than one Python-list row at a time:
    this model's `fc` is `[5120, 25600]` and its head is `[248320, 5120]`, and
    a per-row CPython loop over those is hours, not seconds.
    """

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        rd = lambda fmt: struct.unpack("<" + fmt, self.f.read(struct.calcsize(fmt)))[0]
        rstr = lambda: self.f.read(rd("Q")).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, n = rd("I"), rd("Q")
                if et == 8:
                    return [rstr() for _ in range(n)]
                if et == 9:
                    return [rval(9) for _ in range(n)]
                return [rd(_SCALAR[et]) for _ in range(n)]
            return rd(_SCALAR[t])

        if self.f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        rd("I")
        n_tensors, n_kv = rd("Q"), rd("Q")
        self.kv = {}
        for _ in range(n_kv):
            k = rstr()
            self.kv[k] = rval(rd("I"))
        self.t = {}
        for _ in range(n_tensors):
            name = rstr()
            ne = [rd("Q") for _ in range(rd("I"))]
            self.t[name] = (ne, rd("I"), rd("Q"))
        align = self.kv.get("general.alignment", 32)
        self.data = (self.f.tell() + align - 1) // align * align
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def shape(self, name):
        """Logical (row-major) shape: GGUF `ne` is fastest-varying first."""
        return tuple(reversed(self.t[name][0]))

    def rows(self, name, idx):
        """Just rows `idx` of a `[rows, k]` tensor, as `[len(idx), k]` f32.

        The target's embedding table is `[248320, 5120]` - 5.1 GB dequantized -
        and this script needs eight rows of it. Reading the whole thing to
        throw 99.997% of it away is the difference between a 30-second oracle
        run and a several-minute one, on every invocation.
        """
        ne, tt, off = self.t[name]
        k = ne[0]
        base = self.data + off
        out = np.empty((len(idx), k), dtype=np.float32)
        for i, r in enumerate(idx):
            if tt == GGML_F32:
                start = base + r * k * 4
                out[i] = np.frombuffer(self.mm[start : start + k * 4].tobytes(), dtype="<f4")
                continue
            if tt != GGML_Q8_0:
                raise SystemExit(f"{name}: unsupported ggml type {tt} for a row read")
            nb = k // 32
            start = base + r * nb * 34
            raw = np.frombuffer(self.mm[start : start + nb * 34].tobytes(), dtype=np.uint8).reshape(nb, 34)
            sc = raw[:, :2].copy().view("<f2").astype(np.float32)
            qs = raw[:, 2:].copy().view(np.int8).astype(np.float32)
            out[i] = (qs * sc).ravel()
        return out

    def tensor(self, name):
        """One whole tensor as `float32`, in row-major logical shape."""
        ne, tt, off = self.t[name]
        shape = tuple(reversed(ne))
        n = int(np.prod(shape))
        base = self.data + off
        if tt == GGML_F32:
            raw = self.mm[base : base + n * 4]
            return np.frombuffer(raw.tobytes(), dtype="<f4").reshape(shape)
        if tt == GGML_F16:
            raw = self.mm[base : base + n * 2]
            return np.frombuffer(raw.tobytes(), dtype="<f2").reshape(shape).astype(np.float32)
        if tt != GGML_Q8_0:
            raise SystemExit(f"{name}: unsupported ggml type {tt}")
        k = ne[0]
        if k % 32:
            raise SystemExit(f"{name}: Q8_0 needs a multiple of 32 columns, got {k}")
        rows, nb = n // k, k // 32
        raw = np.frombuffer(self.mm[base : base + rows * nb * 34].tobytes(), dtype=np.uint8)
        raw = raw.reshape(rows * nb, 34)
        scales = raw[:, :2].copy().view("<f2").astype(np.float32)
        qs = raw[:, 2:].copy().view(np.int8).astype(np.float32)
        return (qs * scales).reshape(shape)


# ------------------------------------------------------------------- weights


class Weights:
    """The draft model's tensors, under the reference implementation's own
    names, so the forward below reads like `dflash/model.py` does.

    Loading from the GGUF is the case that matters (it is what the Rust port
    reads); `from_safetensors` exists to cross-check the GGUF's own layout
    conventions against the original checkpoint, which is the only way to
    settle questions like "which axis of `attn_conv_base` is the tap".
    """

    LAYER = {
        "input_layernorm": "attn_norm.weight",
        "post_attention_layernorm": "ffn_norm.weight",
        "q_proj": "attn_q.weight",
        "k_proj": "attn_k.weight",
        "v_proj": "attn_v.weight",
        "o_proj": "attn_output.weight",
        "q_norm": "attn_q_norm.weight",
        "k_norm": "attn_k_norm.weight",
        "gate_proj": "ffn_gate.weight",
        "up_proj": "ffn_up.weight",
        "down_proj": "ffn_down.weight",
        "attn_conv_base": "attn_conv_base",
        "attn_conv_proj": "attn_conv_proj.weight",
        "mlp_conv_base": "ffn_conv_base",
        "mlp_conv_proj": "ffn_conv_proj.weight",
    }

    def __init__(self):
        self.layers = []
        self.cfg = {}

    @classmethod
    def from_gguf(cls, path):
        g = Gguf(path)
        w = cls()
        w.cfg = dict(
            n_layers=int(g.kv["dflash.block_count"]),
            d_model=int(g.kv["dflash.embedding_length"]),
            d_ff=int(g.kv["dflash.feed_forward_length"]),
            n_heads=int(g.kv["dflash.attention.head_count"]),
            n_kv_heads=int(g.kv["dflash.attention.head_count_kv"]),
            head_dim=int(g.kv["dflash.attention.key_length"]),
            eps=float(g.kv["dflash.attention.layer_norm_rms_epsilon"]),
            rope_theta=float(g.kv["dflash.rope.freq_base"]),
            window=int(g.kv["dflash.attention.sliding_window"]),
            block_size=int(g.kv["dflash.block_size"]),
            conv_kernel=int(g.kv["dflash.conv_kernel_size"]),
            conv_group=int(g.kv["dflash.conv_group_size"]),
            selector_rank=int(g.kv["dflash.selector_rank"]),
            selector_top_k=int(g.kv["dflash.selector_top_k"]),
            mask_token_id=int(g.kv["tokenizer.ggml.mask_token_id"]),
            # GGUF carries the +1 `hidden_states`-tuple index (0 is the
            # embedding output), the HF config carries the 0-based DECODER
            # layer id. Store the latter, which is what a forward taps.
            target_layers=[int(i) - 1 for i in g.kv["dflash.target_layers"]],
            causal=bool(g.kv["dflash.attention.causal"]),
        )
        w.fc = g.tensor("fc.weight")
        w.hidden_norm = g.tensor("enc.output_norm.weight")
        w.norm = g.tensor("output_norm.weight")
        w.sel_hidden = g.tensor("selector_hidden.weight")
        w.sel_pred = g.tensor("selector_predecessor.weight")
        w.sel_succ = g.tensor("selector_successor.weight")
        for i in range(w.cfg["n_layers"]):
            w.layers.append({k: g.tensor(f"blk.{i}.{v}") for k, v in cls.LAYER.items()})
        return w

    @classmethod
    def from_safetensors(cls, path, config_path):
        # Via torch, not `safetensors.numpy`: the released checkpoint is
        # bf16, which numpy has no dtype for.
        import torch
        from safetensors.torch import load_file

        st = {k: v.float().numpy() for k, v in load_file(path).items()}
        del torch
        cfg = json.load(open(config_path))
        d = cfg["dflash_config"]
        w = cls()
        w.cfg = dict(
            n_layers=cfg["num_hidden_layers"],
            d_model=cfg["hidden_size"],
            d_ff=cfg["intermediate_size"],
            n_heads=cfg["num_attention_heads"],
            n_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            eps=cfg["rms_norm_eps"],
            rope_theta=float(cfg["rope_parameters"]["rope_theta"]),
            window=cfg["sliding_window"],
            block_size=d["block_size"],
            conv_kernel=d["conv_kernel_size"],
            conv_group=d["conv_group_size"],
            selector_rank=d["selector_rank"],
            selector_top_k=d["selector_top_k"],
            mask_token_id=d["mask_token_id"],
            target_layers=d["target_layer_ids"],
            causal=cfg["is_causal"],
        )
        f32 = lambda n: st[n].astype(np.float32)
        w.fc = f32("fc.weight")
        w.hidden_norm = f32("hidden_norm.weight")
        w.norm = f32("norm.weight")
        w.sel_hidden = f32("candidate_selector.hidden_projection.weight")
        w.sel_pred = f32("candidate_selector.predecessor_codebook")
        w.sel_succ = f32("candidate_selector.successor_codebook")
        pre = "layers.{}."
        name = {
            "input_layernorm": "input_layernorm.weight",
            "post_attention_layernorm": "post_attention_layernorm.weight",
            "q_proj": "self_attn.q_proj.weight",
            "k_proj": "self_attn.k_proj.weight",
            "v_proj": "self_attn.v_proj.weight",
            "o_proj": "self_attn.o_proj.weight",
            "q_norm": "self_attn.q_norm.weight",
            "k_norm": "self_attn.k_norm.weight",
            "gate_proj": "mlp.gate_proj.weight",
            "up_proj": "mlp.up_proj.weight",
            "down_proj": "mlp.down_proj.weight",
            "attn_conv_base": "attention_conv.base_kernel",
            "attn_conv_proj": "attention_conv.kernel_projection.weight",
            "mlp_conv_base": "mlp_conv.base_kernel",
            "mlp_conv_proj": "mlp_conv.kernel_projection.weight",
        }
        for i in range(w.cfg["n_layers"]):
            w.layers.append({k: f32(pre.format(i) + v) for k, v in name.items()})
        return w


# ------------------------------------------------------------------- forward


def rmsnorm(x, weight, eps):
    """Plain `x/rms(x) * w`. NOT the `(1 + w)` fold the Qwen3.5 TARGET's GGUF
    carries: the draft uses `transformers`' stock `Qwen3RMSNorm`, whose weight
    is ones-initialized, and the file's values scatter around 0.5-2.6 rather
    than clustering on 1.0."""
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * weight


def rope_tables(positions, head_dim, theta):
    """Qwen3's stock rotary: FULL rotation over every one of `head_dim`
    channels with the half-split `(d, d + head_dim/2)` pairing.

    Not partial. The TARGET decoder rotates only `rope.dimension_count = 64`
    of its 256 head channels, so the natural assumption to carry over is that
    the draft is partial too; it is not. Its config says `rope_type: default`
    with no `partial_rotary_factor`, and its GGUF says
    `rope.dimension_sections = [64, 0, 0, 0]`, which sums to `head_dim / 2` -
    the mrope spelling of a FULL rotation, with every section on the text
    axis."""
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    freqs = np.outer(np.asarray(positions, dtype=np.float64), inv)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x, cos, sin):
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


def grouped_dynamic_convolve(hidden, dynamic, base, group_size):
    """`dflash.model._grouped_dynamic_convolve`, one block at a time.

    `out[t, g, j] = sum_tap (base[tap, g*gs + j] + dynamic[t, tap, g])
                            * hidden[t - tap, g, j]`

    with zero padding before row 0. The taps are `0 = the current token`,
    `1 = the previous one`, which the checkpoint states outright: the tap-0
    slab of every `*_conv_base` has mean ~1.0 and the tap-1 slab mean ~0.0,
    i.e. an identity initialization on the current token.

    The convolution is CAUSAL inside the block even though the attention is
    not, and its history starts at zero for every block - the reference runs
    it over the freshly built `[anchor, MASK...]` rows with nothing carried
    from the previous round, so there is no conv state to cache.
    """
    length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.reshape(length, groups, group_size)
    out = np.zeros_like(blocks)
    for tap in range(base.shape[0]):
        if tap == 0:
            values = blocks
        else:
            values = np.concatenate([np.zeros((tap, groups, group_size), np.float32), blocks[:-tap]], axis=0)
        out += base[tap].reshape(1, groups, group_size) * values
        out += dynamic[:, tap, :, None] * values
    return out.reshape(length, hidden_size)


def conv_prepare(hidden, base_kernel, projection, kernel_size, group_size):
    """`GroupedDynamicCausalConv.prepare`: project the PRE-convolution hidden
    into `2 * kernel_size * groups` dynamic taps, spend the first half on the
    convolution that feeds the sublayer, and hand the second half back to be
    spent on the convolution that closes it."""
    groups = hidden.shape[-1] // group_size
    dyn = (hidden @ projection.T).reshape(-1, 2, kernel_size, groups)
    return grouped_dynamic_convolve(hidden, dyn[:, 0], base_kernel[0], group_size), dyn[:, 1]


def conv_finish(hidden, dynamic, base_kernel, group_size):
    return grouped_dynamic_convolve(hidden, dynamic, base_kernel[1], group_size)


def silu(x):
    return x / (1.0 + np.exp(-x))


def draft_forward(w, ctx, noise_embedding, ctx_positions, q_positions):
    """The whole draft stack: `DFlashDraftModel.forward` with DFlash2's convs.

    `ctx` is the already-`fc`-projected, already-`hidden_norm`-ed target
    context (one row per context token); `noise_embedding` is the target's raw
    embedding of `[anchor, MASK, ...]`. Returns the post-`norm` hidden for
    every block row, the anchor row included (the caller drops it).
    """
    c = w.cfg
    hd, nh, nkv = c["head_dim"], c["n_heads"], c["n_kv_heads"]
    eps, group = c["eps"], c["conv_group"]
    ctx_len, q_len = ctx.shape[0], noise_embedding.shape[0]
    scale = hd**-0.5

    cos_k, sin_k = rope_tables(list(ctx_positions) + list(q_positions), hd, c["rope_theta"])
    cos_q, sin_q = cos_k[ctx_len:], sin_k[ctx_len:]

    # Non-causal INSIDE the sliding window, in both directions: the block's
    # MASK rows are denoised jointly, so a row sees the ones after it.
    kpos = np.asarray(list(ctx_positions) + list(q_positions))[None, :]
    qpos = np.asarray(q_positions)[:, None]
    visible = np.abs(qpos - kpos) < c["window"]
    if c["causal"]:
        visible &= kpos <= qpos
    bias = np.where(visible, 0.0, -np.inf).astype(np.float32)

    x = noise_embedding.astype(np.float32)
    for layer in w.layers:
        residual = x
        h = rmsnorm(x, layer["input_layernorm"], eps)
        h, dyn = conv_prepare(h, layer["attn_conv_base"], layer["attn_conv_proj"], c["conv_kernel"], group)

        q = rmsnorm((h @ layer["q_proj"].T).reshape(q_len, nh, hd), layer["q_norm"], eps)
        kv_rows = np.concatenate([ctx, h], axis=0)
        k = rmsnorm((kv_rows @ layer["k_proj"].T).reshape(ctx_len + q_len, nkv, hd), layer["k_norm"], eps)
        v = (kv_rows @ layer["v_proj"].T).reshape(ctx_len + q_len, nkv, hd)
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)

        # GQA: every `nh / nkv` query heads share one kv head.
        k = np.repeat(k, nh // nkv, axis=1)
        v = np.repeat(v, nh // nkv, axis=1)
        scores = np.einsum("qhd,khd->hqk", q, k) * scale + bias[None]
        scores -= scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        attn = np.einsum("hqk,khd->qhd", p, v).reshape(q_len, nh * hd)
        h = conv_finish(attn @ layer["o_proj"].T, dyn, layer["attn_conv_base"], group)
        x = residual + h

        residual = x
        h = rmsnorm(x, layer["post_attention_layernorm"], eps)
        h, dyn = conv_prepare(h, layer["mlp_conv_base"], layer["mlp_conv_proj"], c["conv_kernel"], group)
        h = (silu(h @ layer["gate_proj"].T) * (h @ layer["up_proj"].T)) @ layer["down_proj"].T
        x = residual + conv_finish(h, dyn, layer["mlp_conv_base"], group)

    return x, rmsnorm(x, w.norm, eps)


def select_path(w, hidden, logits, anchor_id):
    """`CandidateSelector.select` at temperature 0.

    Per position: take the head's top-`selector_top_k` candidates, then score
    each against the token actually chosen at the PREVIOUS position through a
    rank-`selector_rank` three-way Hadamard product

        score[c] = logit[c] + <pred_codebook[prev] * (hidden @ W_sel), succ_codebook[c]>

    and take the argmax. The walk is GREEDY left to right, not a Viterbi pass
    over the lattice - the chosen token becomes the next position's
    predecessor, and an earlier position is never revisited.
    """
    top_k = w.cfg["selector_top_k"]
    proj = hidden @ w.sel_hidden.T
    path, cands = [], []
    prev = anchor_id
    for pos in range(hidden.shape[0]):
        idx = np.argpartition(-logits[pos], top_k - 1)[:top_k]
        unary = logits[pos][idx]
        pair = (w.sel_pred[prev] * proj[pos]) @ w.sel_succ[idx].T
        prev = int(idx[int(np.argmax(unary + pair))])
        path.append(prev)
        cands.append(idx)
    return path, cands


# ---------------------------------------------------------------------- main

# The target's own token list, for readable diagnostics. A one-element list so
# `main` can fill it after parsing without threading it through every printer.
G_TOKENS = [None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", help="the DFlash2 Q8_0 GGUF")
    ap.add_argument("--safetensors", help="load the draft weights from the original HF checkpoint instead")
    ap.add_argument("--config", help="config.json beside --safetensors")
    ap.add_argument("--target-gguf", required=True, help="the TARGET Qwen3.8-27B GGUF (its embedding and head)")
    ap.add_argument("--hidden", required=True, help="raw f32 [ctx_len, 5*5120] target hidden dump")
    ap.add_argument("--anchor", type=int, required=True, help="the anchor token id (the block's first row)")
    ap.add_argument("--start", type=int, required=True, help="absolute position of the anchor")
    ap.add_argument("--block", type=int, default=0, help="block size (default: the checkpoint's own)")
    ap.add_argument("--dump-draft-hidden", help="write the post-norm draft hidden [block-1, 5120] here")
    ap.add_argument("--dump-ctx", help="write the fc+norm projected context [ctx_len, 5120] here")
    ap.add_argument("--digest", action="store_true", help="print per-row rms/sum digests")
    args = ap.parse_args()

    if args.safetensors:
        w = Weights.from_safetensors(args.safetensors, args.config)
        source = args.safetensors
    elif args.gguf:
        w = Weights.from_gguf(args.gguf)
        source = args.gguf
    else:
        raise SystemExit("one of --gguf / --safetensors is required")
    c = w.cfg
    block = args.block or c["block_size"]

    tg = Gguf(args.target_gguf)
    G_TOKENS[0] = tg.kv.get("tokenizer.ggml.tokens")

    raw = np.fromfile(args.hidden, dtype="<f4")
    wide = len(c["target_layers"]) * c["d_model"]
    if raw.size % wide:
        raise SystemExit(f"{args.hidden}: {raw.size} floats is not a multiple of {wide}")
    target_hidden = raw.reshape(-1, wide)
    ctx_len = target_hidden.shape[0]

    print(f"# draft weights : {source}")
    print(f"# target weights: {args.target_gguf}")
    print(f"# ctx_len={ctx_len} block={block} anchor={args.anchor} start={args.start} mask={c['mask_token_id']}")
    print(f"# layers={c['n_layers']} d={c['d_model']} heads={c['n_heads']}/{c['n_kv_heads']}x{c['head_dim']} "
          f"causal={c['causal']} window={c['window']} taps={c['conv_kernel']} group={c['conv_group']} "
          f"target_layers={c['target_layers']}")

    ctx = rmsnorm(target_hidden @ w.fc.T, w.hidden_norm, c["eps"])
    if args.dump_ctx:
        ctx.astype("<f4").tofile(args.dump_ctx)

    ids = [args.anchor] + [c["mask_token_id"]] * (block - 1)
    noise = tg.rows("token_embd.weight", ids)
    positions = list(range(args.start, args.start + block))
    ctx_positions = list(range(args.start - ctx_len, args.start))

    pre, post = draft_forward(w, ctx, noise, ctx_positions, positions)
    pre, hidden = pre[1:], post[1:]
    if args.dump_draft_hidden:
        hidden.astype("<f4").tofile(args.dump_draft_hidden)

    head = tg.tensor("output.weight")
    logits = hidden @ head.T
    del head
    path, cands = select_path(w, hidden, logits, args.anchor)

    if args.digest:
        print("\n# ctx after fc + hidden_norm")
        for i in (0, ctx_len // 2, ctx_len - 1):
            print(f"  ctx[{i:>4}] rms={np.sqrt((ctx[i]**2).mean()):.6f} sum={ctx[i].sum():+.6f} first={ctx[i][:4]}")
        print("\n# draft hidden BEFORE the final norm (mask rows only) - what the device returns")
        for i in range(pre.shape[0]):
            print(f"  row[{i}] rms={np.sqrt((pre[i]**2).mean()):.6f} sum={pre[i].sum():+.6f} first={pre[i][:4]}")
        print("\n# draft hidden after the final norm (mask rows only) - what the head and the selector see")
        for i in range(hidden.shape[0]):
            print(f"  row[{i}] rms={np.sqrt((hidden[i]**2).mean()):.6f} sum={hidden[i].sum():+.6f} first={hidden[i][:4]}")
        tok = G_TOKENS[0]
        show = (lambda i: repr(tok[i])) if tok else (lambda i: "")
        print("\n# per-row argmax vs selector choice")
        for i in range(hidden.shape[0]):
            am = int(np.argmax(logits[i]))
            print(f"  row[{i}] argmax={am:<7} (logit {logits[i][am]:+.4f}) {show(am):<14} selector={path[i]:<7} "
                  f"(logit {logits[i][path[i]]:+.4f}) {show(path[i]):<14} {'same' if am == path[i] else 'MOVED'}")

    print("\nproposed:", " ".join(str(t) for t in path))
    print("argmax  :", " ".join(str(int(np.argmax(logits[i]))) for i in range(hidden.shape[0])))


if __name__ == "__main__":
    sys.exit(main())
