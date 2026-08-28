#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Independent reference forward pass of the Qwen3.5/3.8 dense hybrid
Gated-DeltaNet/GQA decoder, reading a REAL Q8_0 GGUF directly.

Why this exists: `qwen35_dump_reference.py` drives the HF module at TINY
random dims, which validates the graph but says nothing about whether a
llama.cpp-converted checkpoint's tensors mean what the loader thinks they
mean. This script answers that second question - it is a second, independent
implementation of the same architecture that reads the same real bytes, so a
disagreement localizes to the loader and an agreement rules the loader out.

It is deliberately dependency-free: pure CPython, no torch, no numpy, no
transformers, no llama.cpp. It parses the GGUF header itself, dequantizes
Q8_0 blocks itself, and implements the decoder exactly as the published
`modeling_qwen3_5.py` reference does (cross-checked line by line against it,
and against the identical formulas in its `qwen3_next` sibling):

  * `Qwen3_5GatedDeltaNet.forward` - separate `in_proj_qkv`/`in_proj_z`/
    `in_proj_b`/`in_proj_a` projections (this family does NOT fuse them the
    way `qwen3_next` does, so llama.cpp's `attn_qkv`/`attn_gate`/`ssm_beta`/
    `ssm_alpha` are 1:1 renames with no de-interleave to get wrong), then
    depthwise causal conv1d (kernel 4,
    weight `[channel][tap]`, tap `K-1` multiplies the current token), SiLU
    AFTER the conv, q|k|v split, L2-norm of q/k (sum-of-squares + 1e-6),
    `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`,
    `query *= 1/sqrt(head_k_dim)`, q/k repeat_interleave to the value heads,
    the recurrent gated delta rule, gated RMSNorm (norm, then `* silu(z)`).
  * `Qwen3_5Attention.forward` - the doubled `q_proj` split PER HEAD into
    `[query | gate]` (rows `h*2*hd .. h*2*hd+hd` are the query), q/k RMSNorm
    over `head_dim`, partial RoPE over the first `rope.dimension_count`
    channels with the half-split pairing `(d, d+rot/2)`, GQA head sharing,
    then `attn_out * sigmoid(gate)`.
  * pre-norm residual layers, SwiGLU MLP, final norm, `output.weight` head.

Weight-value conventions taken from the file (not assumed): llama.cpp has
ALREADY applied Qwen3.5's `(1+w)` plain-RMSNorm fold - the reference's plain
`RMSNorm` is `x * (1 + w)` with `w` zero-initialized, and the file's values
cluster on 1.0 rather than on 0.0 - and has NOT applied it to the gated
`RMSNormGated`, which is `x * w` with `w` ones-initialized; `ssm_a` holds
`-exp(A_log)`, so `A = -ssm_a`. Interleaved M-RoPE collapses to plain RoPE
for a text-only prompt (all three position axes advance together), so the
mrope section layout does not enter here.

Usage (positions are decoded one token at a time, exactly as
`qwen35::int8_gguf_resident` drives the real stack):

    tools/goldens/qwen35_gguf_reference_forward.py \\
        --gguf ~/models/unsloth/Qwen3.8-27B-Q8_0.gguf \\
        --layers 4 --tokens 760,6511 --digest

`--digest` prints the per-position `rms`/`sum`/first values of the residual
leaving the last requested layer - the numbers
`crates/qwen35/tests/gguf_reference_parity_real.rs` pins. `--head` also runs
the final norm and the `output.weight` projection and prints the top-10
logits, which is how the whole 64-layer stack's actual prediction was
obtained. `--procs` splits every mat-vec row range over a process pool;
64 layers of the real 27B shape cost ~3 s/layer/token at `--procs 40`.
"""

import argparse
import math
import multiprocessing as mp
import struct
import sys
import time
from operator import mul

# ---------------------------------------------------------------- GGUF reader

GGML_F32, GGML_Q8_0 = 0, 8
_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class Gguf:
    """Header + tensor directory of a GGUF file, plus F32/Q8_0 reads."""

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

    def f32(self, name):
        ne, tt, off = self.t[name]
        if tt != GGML_F32:
            raise SystemExit(f"{name}: expected F32, got ggml type {tt}")
        n = 1
        for e in ne:
            n *= e
        self.f.seek(self.data + off)
        return list(struct.unpack("<%df" % n, self.f.read(n * 4)))

    def row(self, name, r):
        """One row of a `[rows, k]` tensor (GGUF `ne = [k, rows]`)."""
        ne, tt, off = self.t[name]
        k = ne[0]
        if tt == GGML_F32:
            self.f.seek(self.data + off + r * k * 4)
            return list(struct.unpack("<%df" % k, self.f.read(k * 4)))
        nb = k // 32
        self.f.seek(self.data + off + r * nb * 34)
        buf = self.f.read(nb * 34)
        out = []
        for bi in range(nb):
            sc = struct.unpack_from("<e", buf, bi * 34)[0]
            out.extend([sc * q for q in struct.unpack_from("<32b", buf, bi * 34 + 2)])
        return out


# ------------------------------------------------------------ parallel matvec
#
# `out[o] = sum_i W[o, i] * x[i]`. A GGUF matmul weight is `ne = [in, out]`,
# i.e. row-major `[out, in]` with the contraction axis contiguous - the same
# `out = x @ W^T` convention brain's own `model::ops` uses.

_G = None


def _init(path):
    global _G
    _G = Gguf(path)


def _rows(args):
    off, tt, k, r0, r1, xb = args
    f = _G.f
    if tt == GGML_F32:
        f.seek(off + r0 * k * 4)
        big = f.read((r1 - r0) * k * 4)
        x = [v for blk in xb for v in blk]
        return [sum(map(mul, struct.unpack_from("<%df" % k, big, r * k * 4), x)) for r in range(r1 - r0)]
    nb = len(xb)
    rowbytes = nb * 34
    f.seek(off + r0 * rowbytes)
    big = f.read((r1 - r0) * rowbytes)
    q8 = struct.Struct("<32b").unpack_from
    scale = struct.Struct("<e").unpack_from
    out = []
    for r in range(r1 - r0):
        base = r * rowbytes
        acc = 0.0
        for bi in range(nb):
            o = base + bi * 34
            acc += scale(big, o)[0] * sum(map(mul, q8(big, o + 2), xb[bi]))
        out.append(acc)
    return out


class Model:
    def __init__(self, gguf, pool):
        self.g = gguf
        self.pool = pool
        self.cache = {}

    def w32(self, name):
        if name not in self.cache:
            self.cache[name] = self.g.f32(name)
        return self.cache[name]

    def matvec(self, name, x):
        ne, tt, off = self.g.t[name]
        k, rows = ne[0], ne[1]
        assert k == len(x), (name, k, len(x))
        xb = [x[i * 32:(i + 1) * 32] for i in range(k // 32)]
        base = self.g.data + off
        n = self.pool._processes if self.pool else 1
        step = max(64, (rows + n - 1) // n)
        tasks = [(base, tt, k, r0, min(r0 + step, rows), xb) for r0 in range(0, rows, step)]
        parts = self.pool.map(_rows, tasks) if self.pool else map(_rows, tasks)
        out = []
        for p in parts:
            out.extend(p)
        return out


# ------------------------------------------------------------------- the math


def rmsnorm(x, w, eps=1e-6):
    s = 1.0 / math.sqrt(sum(t * t for t in x) / len(x) + eps)
    return [x[i] * s * w[i] for i in range(len(x))]


def silu(x):
    return [t / (1.0 + math.exp(-t)) if t > -80 else 0.0 for t in x]


def sigmoid(t):
    return 1.0 / (1.0 + math.exp(-t)) if t > -80 else 0.0


def softplus(t):
    return math.log1p(math.exp(-abs(t))) + max(t, 0.0)


def l2norm_heads(x, nh, hd, eps=1e-6):
    out = []
    for h in range(nh):
        seg = x[h * hd:(h + 1) * hd]
        inv = 1.0 / math.sqrt(sum(t * t for t in seg) + eps)
        out.extend([t * inv for t in seg])
    return out


class Cfg:
    """Shapes read from the GGUF's own KV and tensor shapes, never assumed."""

    def __init__(self, g):
        a = lambda k: g.kv["qwen35." + k]
        self.d = a("embedding_length")
        self.n_heads = a("attention.head_count")
        self.n_kv = a("attention.head_count_kv")
        self.head_dim = a("attention.key_length")
        self.rot = a("rope.dimension_count")
        self.theta = a("rope.freq_base")
        self.interval = a("full_attention_interval")
        self.eps = a("attention.layer_norm_rms_epsilon")
        self.n_layers = a("block_count") - g.kv.get("qwen35.nextn_predict_layers", 1)
        self.khd = a("ssm.state_size")
        self.vhd = g.t["blk.0.ssm_norm.weight"][0][0]
        self.value_dim = g.t["blk.0.attn_gate.weight"][0][1]
        self.conv_dim = g.t["blk.0.attn_qkv.weight"][0][1]
        self.key_dim = (self.conv_dim - self.value_dim) // 2
        self.nkh = self.key_dim // self.khd
        self.nvh = self.value_dim // self.vhd
        self.kw = a("ssm.conv_kernel")

    def is_full(self, l):
        return (l + 1) % self.interval == 0


def gdn_layer(m, c, l, x, st):
    p = f"blk.{l}."
    xn1 = rmsnorm(x, m.w32(p + "attn_norm.weight"), c.eps)
    qkv = m.matvec(p + "attn_qkv.weight", xn1)
    cw = m.w32(p + "ssm_conv1d.weight")  # [channel][tap], tap fastest
    K = c.kw
    hist = st.setdefault("hist", [[0.0] * (K - 1) for _ in range(c.conv_dim)])
    conv = []
    for ch in range(c.conv_dim):
        h = hist[ch]
        acc = qkv[ch] * cw[ch * K + K - 1]
        for j in range(K - 1):
            acc += h[j] * cw[ch * K + j]
        conv.append(acc)
        h[:] = h[1:] + [qkv[ch]]
    act = silu(conv)
    q = l2norm_heads(act[0:c.key_dim], c.nkh, c.khd)
    k = l2norm_heads(act[c.key_dim:2 * c.key_dim], c.nkh, c.khd)
    v = act[2 * c.key_dim:]
    beta = [sigmoid(t) for t in m.matvec(p + "ssm_beta.weight", xn1)]
    a = m.matvec(p + "ssm_alpha.weight", xn1)
    A = [-t for t in m.w32(p + "ssm_a")]  # llama.cpp stores -exp(A_log)
    dtb = m.w32(p + "ssm_dt.bias")
    g = [-A[i] * softplus(a[i] + dtb[i]) for i in range(c.nvh)]
    scale = 1.0 / math.sqrt(c.khd)
    S = st.setdefault("S", [[0.0] * (c.khd * c.vhd) for _ in range(c.nvh)])
    group = c.nvh // c.nkh
    out = [0.0] * c.value_dim
    for j in range(c.nvh):
        kh = j // group  # repeat_interleave, not tile
        qs = q[kh * c.khd:(kh + 1) * c.khd]
        ks = k[kh * c.khd:(kh + 1) * c.khd]
        Sj = S[j]
        decay = math.exp(g[j])
        for i in range(len(Sj)):
            Sj[i] *= decay
        kv = [0.0] * c.vhd
        for dk in range(c.khd):
            kk = ks[dk]
            if kk:
                base = dk * c.vhd
                for dv in range(c.vhd):
                    kv[dv] += Sj[base + dv] * kk
        b = beta[j]
        delta = [(v[j * c.vhd + dv] - kv[dv]) * b for dv in range(c.vhd)]
        o = [0.0] * c.vhd
        for dk in range(c.khd):
            kk = ks[dk]
            qq = qs[dk] * scale
            base = dk * c.vhd
            for dv in range(c.vhd):
                Sj[base + dv] += kk * delta[dv]
                o[dv] += Sj[base + dv] * qq
        out[j * c.vhd:(j + 1) * c.vhd] = o
    z = silu(m.matvec(p + "attn_gate.weight", xn1))
    nw = m.w32(p + "ssm_norm.weight")  # gated norm: NOT (1+w)-folded
    gated = []
    for j in range(c.nvh):
        normed = rmsnorm(out[j * c.vhd:(j + 1) * c.vhd], nw, c.eps)
        gated.extend(normed[i] * z[j * c.vhd + i] for i in range(c.vhd))
    y = m.matvec(p + "ssm_out.weight", gated)
    return [x[i] + y[i] for i in range(c.d)]


def rope(c, vec, nheads, pos):
    half = c.rot // 2
    for h in range(nheads):
        base = h * c.head_dim
        for d in range(half):
            ang = pos * c.theta ** (-2.0 * d / c.rot)
            cs, sn = math.cos(ang), math.sin(ang)
            x1, x2 = vec[base + d], vec[base + d + half]
            vec[base + d] = x1 * cs - x2 * sn
            vec[base + d + half] = x2 * cs + x1 * sn


def gqa_layer(m, c, l, x, st, pos):
    p = f"blk.{l}."
    hd = c.head_dim
    xn1 = rmsnorm(x, m.w32(p + "attn_norm.weight"), c.eps)
    qg = m.matvec(p + "attn_q.weight", xn1)
    qn, kn = m.w32(p + "attn_q_norm.weight"), m.w32(p + "attn_k_norm.weight")
    q, gate = [], []
    for h in range(c.n_heads):  # per-head [query | gate], not a global chunk
        b = h * 2 * hd
        q.extend(rmsnorm(qg[b:b + hd], qn, c.eps))
        gate.extend(qg[b + hd:b + 2 * hd])
    kk = m.matvec(p + "attn_k.weight", xn1)
    kh = []
    for h in range(c.n_kv):
        kh.extend(rmsnorm(kk[h * hd:(h + 1) * hd], kn, c.eps))
    vv = m.matvec(p + "attn_v.weight", xn1)
    rope(c, q, c.n_heads, pos)
    rope(c, kh, c.n_kv, pos)
    kc = st.setdefault("k", [])
    vc = st.setdefault("v", [])
    kc.append(kh)
    vc.append(vv)
    scale = 1.0 / math.sqrt(hd)
    grp = c.n_heads // c.n_kv
    ctx = []
    for h in range(c.n_heads):
        hkv = h // grp
        qs = q[h * hd:(h + 1) * hd]
        sc = [sum(map(mul, qs, kc[t][hkv * hd:(hkv + 1) * hd])) * scale for t in range(len(kc))]
        mx = max(sc)
        ex = [math.exp(s - mx) for s in sc]
        z = sum(ex)
        o = [0.0] * hd
        for t in range(len(kc)):
            wgt = ex[t] / z
            vs = vc[t][hkv * hd:(hkv + 1) * hd]
            for d in range(hd):
                o[d] += wgt * vs[d]
        ctx.extend(o)
    y = m.matvec(p + "attn_output.weight", [ctx[i] * sigmoid(gate[i]) for i in range(len(ctx))])
    return [x[i] + y[i] for i in range(c.d)]


def mlp(m, c, l, r1):
    p = f"blk.{l}."
    xn2 = rmsnorm(r1, m.w32(p + "post_attention_norm.weight"), c.eps)
    hg = silu(m.matvec(p + "ffn_gate.weight", xn2))
    hu = m.matvec(p + "ffn_up.weight", xn2)
    dv = m.matvec(p + "ffn_down.weight", [hg[i] * hu[i] for i in range(len(hg))])
    return [r1[i] + dv[i] for i in range(c.d)]


def rms(v):
    return math.sqrt(sum(t * t for t in v) / len(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", required=True, help="comma-separated token ids, decoded one at a time")
    ap.add_argument("--layers", type=int, default=4, help="how many decoder blocks to run (from 0)")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--digest", action="store_true", help="print rms/sum/first values per position")
    ap.add_argument("--head", action="store_true", help="also run the final norm + output.weight, print top-10")
    ap.add_argument("--save", default="", help="directory to write the residual after every layer into")
    args = ap.parse_args()

    g = Gguf(args.gguf)
    c = Cfg(g)
    pool = mp.Pool(args.procs, initializer=_init, initargs=(args.gguf,)) if args.procs > 1 else None
    if pool is None:
        _init(args.gguf)
    m = Model(g, pool)
    toks = [int(t) for t in args.tokens.split(",")]
    nl = min(args.layers, c.n_layers)
    print(f"{args.gguf}: {c.n_layers} layers ({nl} requested), d_model {c.d}, "
          f"{c.n_heads}/{c.n_kv} heads x {c.head_dim}, rot {c.rot} theta {c.theta:g}, "
          f"GDN {c.nkh}/{c.nvh} heads x {c.khd}/{c.vhd}")
    states = [dict() for _ in range(nl)]
    t0 = time.time()
    for pos, tid in enumerate(toks):
        x = g.row("token_embd.weight", tid)
        for l in range(nl):
            r1 = gqa_layer(m, c, l, x, states[l], pos) if c.is_full(l) else gdn_layer(m, c, l, x, states[l])
            x = mlp(m, c, l, r1)
            if args.save:
                with open(f"{args.save}/ref_p{pos}_l{l + 1}.txt", "w") as fh:
                    fh.write(" ".join("%.7g" % t for t in x))
            print("  pos %d layer %2d rms %.6f (%.1fs)" % (pos, l, rms(x), time.time() - t0))
            sys.stdout.flush()
        if args.digest:
            print("POS %d tok %d layers %d: rms %.6f sum %.5f first4 %s" %
                  (pos, tid, nl, rms(x), sum(x), [round(t, 6) for t in x[:4]]))
        if args.head:
            lg = m.matvec("output.weight", rmsnorm(x, m.w32("output_norm.weight"), c.eps))
            top = sorted(range(len(lg)), key=lambda i: -lg[i])[:10]
            vocab = g.kv["tokenizer.ggml.tokens"]
            print("  TOP10 %s" % [(i, round(lg[i], 3), vocab[i]) for i in top])
        sys.stdout.flush()


if __name__ == "__main__":
    main()
