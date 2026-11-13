# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the Volume VI project "Comparing Language Model Blocks": an engine of interchangeable blocks, a cost model, a
parity harness and a controlled comparison.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the project's own description. The libraries
of the earlier chapters are imported, never copied: the decoder of Chapter 1 (`modern_decoder`, the reference the dense block is
compared with), the gated delta recurrence in its token-by-token and chunked forms (`linear_attention`, Chapter 5), the router, the
balancing loss and its gradient (`mixture_of_experts`, Chapter 4), the ledger of parameters, FLOPs and cache (`model_cost`,
Chapter 10), the measures, the rounding to 32 bits and the safetensors writer of the parity chapter (`importing_models`, Chapter 2),
the paired interval and the verdict rule of the research chapter (`spectral_lm`, Chapter 9), and from Volume V the RMS norm and the
rotary rotation (`transformer_block`, `positions`), the learning-rate curve (`schedules`) and the AdamW step (`training_gpt`).
What this module writes is what makes them one engine:

* a block interface (initialise, forward, backward, state) with four implementations: grouped-query attention with a SwiGLU network,
  the same with a mixture-of-experts network, a gated delta network layer and a sliding-window attention layer, each with a
  central-difference gradient check;
* the cost of every block, counted by running it and by formula;
* goldens of every block, three measures, a tolerance counted from the roundings, and planted bugs the harness must catch;
* a synthetic task that needs local and long-range information, a registered set of predictions and the paired comparison of the
  blocks and of a hybrid, each run repeated from slightly different starting points so that only the digits that survive are printed.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints its seeds. A dot product is
`math.sumprod`, which accumulates in extended precision, so that its last bit does not depend on the order of the additions; every
printed digit stays clear of that last bit.

Swedish Embedded AB implements solutions for comparing neural network architectures on equal terms. If your team needs expertise in
architecture evaluation, cost models and parity testing of language models, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import math
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import importing_models as im
import linear_attention as la
import mixture_of_experts as moe
import model_cost as mc
import modern_decoder as md
import positions as pos_mod
import schedules
import spectral_lm as sl
import stable
import training_gpt as tg
import transformer_block as tb

# ------------------------------------------------------------------------------------------------ formatting


def _fx(v, d=4):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _int(n):
    return f"{n:,}"


def _sci(x, d=2):
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}e{int(e)}"


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


def _below(x, limit):
    m, e = f"{limit:.0e}".split("e")
    return f"below {m}e{int(e)}" if x < limit else _sci(x, 2)


# ------------------------------------------------------------------------------------------------ small linear algebra

OPS = {"matmul": 0, "attention": 0, "recurrence": 0}   # multiply-adds, by kind, since the last reset


def reset_ops():
    for k in OPS:
        OPS[k] = 0


def mm(A, B):
    """A [n, k] times B [k, m]; every entry is a `math.sumprod`. Counts n k m multiply-adds."""
    cols = list(zip(*B))
    OPS["matmul"] += len(A) * len(B) * len(cols)
    return [[math.sumprod(r, c) for c in cols] for r in A]


def tr(A):
    return [list(r) for r in zip(*A)]


def add(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def silu(a):
    return a / (1.0 + math.exp(-a))


def dsilu(a):
    s = 1.0 / (1.0 + math.exp(-a))
    return s * (1.0 + a * (1.0 - s))


# ------------------------------------------------------------------------------------------------ the shared shape

# Every block works on the width D = 16 of the model. The attention blocks have H = 4 query heads and HKV = 2 key-value heads of
# width DH = 4 (so H DH = D); the gated delta layer has HG = 2 heads with key width DK = 4 and value width DV = 7, chosen so that
# its layer holds one parameter more than the attention layer (809 against 808); the dense network has hidden width F = 24; the
# mixture has E = 4 experts of hidden width FE = 6, of which K = 2 are used; the window is 4 keys.
CFG = {"V": 12, "D": 16, "H": 4, "HKV": 2, "DH": 4, "F": 24, "BASE": 100.0, "WINDOW": 4,
       "E": 4, "K": 2, "FE": 6, "ALPHA": 0.01, "HG": 2, "DK": 4, "DV": 7, "CB": -3.0, "CHUNK": 4}
MIXERS = ("gqa", "swa", "gdn")
FFNS = ("swiglu", "moe")


def cfg_with(**kw):
    c = dict(CFG)
    c.update(kw)
    return c


# ------------------------------------------------------------------------------------------------ initialisation


def init_layer(g, cfg, spec):
    """One layer drawn from generator g, each matrix row by row. Draws, in this order: the gain of norm 1 (D numbers, each
    `1 + 0.1 normal()`); the mixer's tensors; the gain of norm 2; the network's tensors.
    gqa and swa: W_Q [D, H DH], W_K and W_V [D, HKV DH] (each `normal() / sqrt(D)`), the QK-norm gains of queries then keys (DH
    numbers each, `1 + 0.1 normal()`), W_O [H DH, D] (`normal() / sqrt(H DH)`).
    gdn: W_Q and W_K [D, HG DK], W_V [D, HG DV], W_a and W_b [D, HG] (each `normal() / sqrt(D)`), the decay bias (HG numbers, all
    CB, no draws), the output-norm gain (DV numbers, `1 + 0.1 normal()`), W_O [HG DV, D] (`normal() / sqrt(HG DV)`).
    swiglu: W_gate and W_up [D, F] (`normal() / sqrt(D)`), W_down [F, D] (`normal() / sqrt(F)`).
    moe: the router [D, E] (`normal() / sqrt(D)`), then for each expert W_gate and W_up [D, FE], W_down [FE, D] with the scales of
    swiglu."""
    mixer, ffn = spec
    D = cfg["D"]

    def mat(r, c, s):
        return [[s * g.normal() for _ in range(c)] for _ in range(r)]

    def gain(n):
        return [1.0 + 0.1 * g.normal() for _ in range(n)]

    def swiglu(f):
        return {"Wg": mat(D, f, 1 / math.sqrt(D)), "Wu": mat(D, f, 1 / math.sqrt(D)), "Wd": mat(f, D, 1 / math.sqrt(f))}

    lay = {"g1": gain(D)}
    if mixer in ("gqa", "swa"):
        H, HKV, DH = cfg["H"], cfg["HKV"], cfg["DH"]
        lay["Wq"], lay["Wk"], lay["Wv"] = mat(D, H * DH, 1 / math.sqrt(D)), mat(D, HKV * DH, 1 / math.sqrt(D)), mat(D, HKV * DH, 1 / math.sqrt(D))
        lay["qn"], lay["kn"] = gain(DH), gain(DH)
        lay["Wo"] = mat(H * DH, D, 1 / math.sqrt(H * DH))
    else:
        HG, DK, DV = cfg["HG"], cfg["DK"], cfg["DV"]
        lay["Wq"], lay["Wk"], lay["Wv"] = mat(D, HG * DK, 1 / math.sqrt(D)), mat(D, HG * DK, 1 / math.sqrt(D)), mat(D, HG * DV, 1 / math.sqrt(D))
        lay["Wa"], lay["Wb"] = mat(D, HG, 1 / math.sqrt(D)), mat(D, HG, 1 / math.sqrt(D))
        lay["cb"] = [cfg["CB"]] * HG
        lay["on"] = gain(DV)
        lay["Wo"] = mat(HG * DV, D, 1 / math.sqrt(HG * DV))
    lay["g2"] = gain(D)
    if ffn == "swiglu":
        lay.update(swiglu(cfg["F"]))
    else:
        lay["Wr"] = mat(D, cfg["E"], 1 / math.sqrt(D))
        lay["experts"] = [swiglu(cfg["FE"]) for _ in range(cfg["E"])]
    return lay


def init_model(name, cfg, specs):
    """The token table [V, D] as `normal() / sqrt(D)`, then the layers in order (`init_layer`), then the final gain (D numbers,
    `1 + 0.1 normal()`). The head is the table used again."""
    g = bookrng.generator(name)
    s = 1 / math.sqrt(cfg["D"])
    p = {"emb": [[s * g.normal() for _ in range(cfg["D"])] for _ in range(cfg["V"])], "layers": [init_layer(g, cfg, s) for s in specs]}
    p["gn"] = [1.0 + 0.1 * g.normal() for _ in range(cfg["D"])]
    return p


def leaves(p):
    """Every tensor of p as (path, rows) in storage order; a vector is wrapped as a one-row matrix that shares its storage."""
    out = []

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, path + (k,))
        elif x and isinstance(x[0], dict):
            for i, v in enumerate(x):
                walk(v, path + (i,))
        else:
            out.append((path, x if isinstance(x[0], list) else [x]))

    walk(p, ())
    return out


def parameter_count(p):
    return sum(len(r) for _, rows in leaves(p) for r in rows)


def zeros_like(x):
    if isinstance(x, dict):
        return {k: zeros_like(v) for k, v in x.items()}
    if x and isinstance(x[0], (dict, list)):
        return [zeros_like(v) for v in x]
    return [0.0] * len(x)


def add_into(dst, src):
    if isinstance(dst, dict):
        for k in dst:
            add_into(dst[k], src[k])
    elif dst and isinstance(dst[0], (dict, list)):
        for a, b in zip(dst, src):
            add_into(a, b)
    else:
        for i, v in enumerate(src):
            dst[i] += v


def scale_by(x, s):
    if isinstance(x, dict):
        for v in x.values():
            scale_by(v, s)
    elif x and isinstance(x[0], (dict, list)):
        for v in x:
            scale_by(v, s)
    else:
        for i in range(len(x)):
            x[i] *= s


# ------------------------------------------------------------------------------------------------ sublayers: attention


def attention_forward(lay, N, cfg, window, bug=None):
    """Grouped-query attention with QK-norm and half-split rotary positions over the rows of N [n, D], causal; with a window w a
    row sees only the last w rows including itself. Returns (output [n, D], trace). `bug` names a planted mistake: "group_map"
    (query head h uses key-value head h mod HKV instead of h // group), "window_off_by_one" (the window holds w + 1 keys),
    "no_qk_norm_gain" (the gains of the QK-norm are not applied), "rotate_keys_only" (queries are not rotated)."""
    H, HKV, DH = cfg["H"], cfg["HKV"], cfg["DH"]
    n, group, scale = len(N), cfg["H"] // cfg["HKV"], 1.0 / math.sqrt(DH)
    ws = pos_mod.omegas(DH, cfg["BASE"])
    Q0, K0, V0 = mm(N, lay["Wq"]), mm(N, lay["Wk"]), mm(N, lay["Wv"])
    qn = [1.0] * DH if bug == "no_qk_norm_gain" else lay["qn"]
    kn = [1.0] * DH if bug == "no_qk_norm_gain" else lay["kn"]
    Q1, cq = tb.rms_forward([Q0[t][h * DH:(h + 1) * DH] for t in range(n) for h in range(H)], qn)
    K1, ck = tb.rms_forward([K0[t][h * DH:(h + 1) * DH] for t in range(n) for h in range(HKV)], kn)
    q2 = [[Q1[t * H + h] if bug == "rotate_keys_only" else pos_mod.rotate(Q1[t * H + h], t, ws, "half") for h in range(H)] for t in range(n)]
    k2 = [[pos_mod.rotate(K1[t * HKV + h], t, ws, "half") for h in range(HKV)] for t in range(n)]
    v = [[V0[t][h * DH:(h + 1) * DH] for h in range(HKV)] for t in range(n)]
    vcols = [[[v[j][kv][d] for j in range(n)] for d in range(DH)] for kv in range(HKV)]
    w = None if window is None else window + (1 if bug == "window_off_by_one" else 0)
    O, P = [], []
    for t in range(n):
        lo = 0 if w is None else max(0, t - w + 1)
        row, prow = [], []
        for h in range(H):
            kv = h % HKV if bug == "group_map" else h // group
            pr = softmax([math.sumprod(q2[t][h], k2[j][kv]) * scale for j in range(lo, t + 1)])
            OPS["attention"] += 2 * DH * (t + 1 - lo)
            row += [math.sumprod(pr, vcols[kv][d][lo:t + 1]) for d in range(DH)]
            prow.append(pr)
        O.append(row)
        P.append(prow)
    A = mm(O, lay["Wo"])
    return A, {"N": N, "cq": cq, "ck": ck, "q2": q2, "k2": k2, "v": v, "P": P, "O": O, "w": w}


def attention_backward(lay, c, dA, cfg):
    """Backward of `attention_forward`: returns (dN, gradients of this sublayer's tensors)."""
    H, HKV, DH = cfg["H"], cfg["HKV"], cfg["DH"]
    n, group, scale = len(dA), cfg["H"] // cfg["HKV"], 1.0 / math.sqrt(DH)
    ws = pos_mod.omegas(DH, cfg["BASE"])
    g = {"Wo": mm(tr(c["O"]), dA)}
    dO = mm(dA, tr(lay["Wo"]))
    q2, k2, v, P, w = c["q2"], c["k2"], c["v"], c["P"], c["w"]
    dq2 = [[[0.0] * DH for _ in range(H)] for _ in range(n)]
    dk2 = [[[0.0] * DH for _ in range(HKV)] for _ in range(n)]
    dv = [[[0.0] * DH for _ in range(HKV)] for _ in range(n)]
    for t in range(n):
        lo = 0 if w is None else max(0, t - w + 1)
        for h in range(H):
            kv = h // group
            doh, pr = dO[t][h * DH:(h + 1) * DH], P[t][h]
            dp = [math.sumprod(doh, v[j][kv]) for j in range(lo, t + 1)]
            rowsum = math.sumprod(pr, dp)
            for idx, j in enumerate(range(lo, t + 1)):
                ds = scale * pr[idx] * (dp[idx] - rowsum)
                for d in range(DH):
                    dq2[t][h][d] += ds * k2[j][kv][d]
                    dk2[j][kv][d] += ds * q2[t][h][d]
                    dv[j][kv][d] += pr[idx] * doh[d]
    dq1 = [pos_mod.rotate(dq2[t][h], -t, ws, "half") for t in range(n) for h in range(H)]
    dk1 = [pos_mod.rotate(dk2[t][h], -t, ws, "half") for t in range(n) for h in range(HKV)]
    dq0, g["qn"] = tb.rms_backward(dq1, c["cq"])
    dk0, g["kn"] = tb.rms_backward(dk1, c["ck"])
    dQ0 = [sum((dq0[t * H + h] for h in range(H)), []) for t in range(n)]
    dK0 = [sum((dk0[t * HKV + h] for h in range(HKV)), []) for t in range(n)]
    dV0 = [sum((dv[t][h] for h in range(HKV)), []) for t in range(n)]
    Nt = tr(c["N"])
    g["Wq"], g["Wk"], g["Wv"] = mm(Nt, dQ0), mm(Nt, dK0), mm(Nt, dV0)
    dN = add(add(mm(dQ0, tr(lay["Wq"])), mm(dK0, tr(lay["Wk"]))), mm(dV0, tr(lay["Wv"])))
    return dN, g


# ------------------------------------------------------------------------------------------------ sublayers: gated delta network


def _l2_backward(x, dy, eps=1e-6):
    """Backward of y = x / sqrt(x.x + eps): dx = r dy - r^3 x (x.dy)."""
    r = 1.0 / math.sqrt(math.sumprod(x, x) + eps)
    c = r ** 3 * math.sumprod(x, dy)
    return [r * d - c * xi for d, xi in zip(dy, x)]


def gdn_head_forward(Qr, Kr, V, g, beta, dk, form, chunk, bug=None):
    """One head of the recurrence of Chapter 5 on raw (q, k) rows: L2 normalisation of both, the scale 1/sqrt(dk) on the queries,
    then the tokenwise recurrence (`form` "recurrent", which also returns the state after every token) or the chunked form
    (`form` "chunked" with chunk size `chunk`; the states are not returned). `bug` "chunk_state_dropped" starts every chunk of
    the chunked form from a zero state; "decay_after_read" is applied in `gdn_forward`."""
    Q, K = la._prepare(Qr, Kr, dk)
    if form == "recurrent":
        outs, S, states = la.gated_delta_tokenwise(Q, K, V, g, beta)
        return outs, (Q, K, states)
    if bug == "chunk_state_dropped":
        outs = []
        for s0 in range(0, len(K), chunk):
            o, _ = la.gated_delta_chunked(Q[s0:s0 + chunk], K[s0:s0 + chunk], V[s0:s0 + chunk], g[s0:s0 + chunk], beta[s0:s0 + chunk], chunk)
            outs += o
        return outs, (Q, K, None)
    outs, _ = la.gated_delta_chunked(Q, K, V, g, beta, chunk)
    return outs, (Q, K, None)


def gdn_forward(lay, N, cfg, form="recurrent", chunk=None, bug=None):
    """The gated delta layer over the rows of N [n, D]: projections to q, k, v and to the decay and write gates, the recurrence
    per head, an RMS norm of every head's output and the output projection. The decay is g = -softplus(a + bias) (so A = 1 in the
    notation of Chapter 5) and the write strength beta = sigmoid(b). The short convolution and the output gate of the full layer
    are not part of this block. `bug`: "no_q_scale" (the 1/sqrt(dk) on the queries left out), "gate_sign" (g = +softplus)."""
    HG, DK, DV = cfg["HG"], cfg["DK"], cfg["DV"]
    n = len(N)
    Q0, K0, V0, A0, B0 = mm(N, lay["Wq"]), mm(N, lay["Wk"]), mm(N, lay["Wv"]), mm(N, lay["Wa"]), mm(N, lay["Wb"])
    heads, rows = [], []
    for h in range(HG):
        Qr = [Q0[t][h * DK:(h + 1) * DK] for t in range(n)]
        Kr = [K0[t][h * DK:(h + 1) * DK] for t in range(n)]
        V = [V0[t][h * DV:(h + 1) * DV] for t in range(n)]
        pre = [A0[t][h] + lay["cb"][h] for t in range(n)]
        g = [(1.0 if bug == "gate_sign" else -1.0) * la._softplus(a) for a in pre]
        beta = [la._sigmoid(B0[t][h]) for t in range(n)]
        if bug == "no_q_scale":
            Qr = [[x * math.sqrt(DK) for x in q] for q in Qr]
        outs, aux = gdn_head_forward(Qr, Kr, V, g, beta, DK, form, chunk, bug)
        OPS["recurrence"] += la.tokenwise_macs(n, DK, DV) if form == "recurrent" else la.chunk_macs_formula(n, chunk, DK, DV)
        heads.append({"Qr": Qr, "Kr": Kr, "V": V, "pre": pre, "g": g, "beta": beta, "aux": aux, "outs": outs})
        rows += outs
    On, cn = tb.rms_forward(rows, lay["on"])
    Ocat = [sum((On[h * n + t] for h in range(HG)), []) for t in range(n)]
    A = mm(Ocat, lay["Wo"])
    return A, {"N": N, "heads": heads, "cn": cn, "Ocat": Ocat, "A0": A0, "B0": B0}


def gdn_head_backward(h, dO, dk, dv):
    """Backward through the recurrence of one head by the reverse of its steps. With S' = alpha S_{t-1}, pred = S'^T k,
    e = beta (v - pred), S_t = S' + k e^T and o_t = S_t^T q: the gradient of S_t from the future, plus q do^T from the read,
    gives dk through e, dv and dbeta through e, dpred through S', dg through alpha, and the gradient of S_{t-1} is alpha times that
    of S'. Returns (dQ, dK, dV, dg, dbeta) for the normalised q (scaled) and k."""
    Q, K, states = h["aux"]
    V, g, beta = h["V"], h["g"], h["beta"]
    n = len(K)
    dQ, dK, dV, dg, db = [], [None] * n, [None] * n, [0.0] * n, [0.0] * n
    dS = [[0.0] * dv for _ in range(dk)]
    dQ = [None] * n
    for t in reversed(range(n)):
        S = states[t]
        Sprev = states[t - 1] if t else [[0.0] * dv for _ in range(dk)]
        a = math.exp(g[t])
        Sp = [[a * x for x in row] for row in Sprev]
        pred = [math.sumprod([Sp[i][c] for i in range(dk)], K[t]) for c in range(dv)]
        e = [beta[t] * (V[t][c] - pred[c]) for c in range(dv)]
        dQ[t] = [math.sumprod(S[i], dO[t]) for i in range(dk)]
        for i in range(dk):
            for c in range(dv):
                dS[i][c] += Q[t][i] * dO[t][c]
        dK[t] = [math.sumprod(dS[i], e) for i in range(dk)]
        de = [math.sumprod([dS[i][c] for i in range(dk)], K[t]) for c in range(dv)]
        db[t] = sum(de[c] * (V[t][c] - pred[c]) for c in range(dv))
        dV[t] = [beta[t] * x for x in de]
        dpred = [-beta[t] * x for x in de]
        for i in range(dk):
            for c in range(dv):
                dS[i][c] += K[t][i] * dpred[c]
        for i in range(dk):
            dK[t][i] += math.sumprod(Sp[i], dpred)
        dg[t] = a * sum(Sprev[i][c] * dS[i][c] for i in range(dk) for c in range(dv))
        dS = [[a * x for x in row] for row in dS]
    return dQ, dK, dV, dg, db


def gdn_backward(lay, c, dA, cfg):
    HG, DK, DV = cfg["HG"], cfg["DK"], cfg["DV"]
    n = len(dA)
    g = {"Wo": mm(tr(c["Ocat"]), dA)}
    dOcat = mm(dA, tr(lay["Wo"]))
    dOn = [dOcat[t][h * DV:(h + 1) * DV] for h in range(HG) for t in range(n)]
    dO, g["on"] = tb.rms_backward(dOn, c["cn"])
    dQ0 = [[0.0] * (HG * DK) for _ in range(n)]
    dK0 = [[0.0] * (HG * DK) for _ in range(n)]
    dV0 = [[0.0] * (HG * DV) for _ in range(n)]
    dA0 = [[0.0] * HG for _ in range(n)]
    dB0 = [[0.0] * HG for _ in range(n)]
    g["cb"] = [0.0] * HG
    scale = 1.0 / math.sqrt(DK)
    for h, hd in enumerate(c["heads"]):
        dQ, dK, dV, dg, db = gdn_head_backward(hd, dO[h * n:(h + 1) * n], DK, DV)
        for t in range(n):
            dq = _l2_backward(hd["Qr"][t], [scale * x for x in dQ[t]])
            dk = _l2_backward(hd["Kr"][t], dK[t])
            dQ0[t][h * DK:(h + 1) * DK] = dq
            dK0[t][h * DK:(h + 1) * DK] = dk
            dV0[t][h * DV:(h + 1) * DV] = dV[t]
            sig = la._sigmoid(hd["pre"][t])               # d softplus(z) / dz
            dpre = -sig * dg[t]
            dA0[t][h] = dpre
            g["cb"][h] += dpre
            sb = hd["beta"][t]
            dB0[t][h] = sb * (1.0 - sb) * db[t]
    Nt = tr(c["N"])
    g["Wq"], g["Wk"], g["Wv"], g["Wa"], g["Wb"] = mm(Nt, dQ0), mm(Nt, dK0), mm(Nt, dV0), mm(Nt, dA0), mm(Nt, dB0)
    dN = mm(dQ0, tr(lay["Wq"]))
    for dX, W in ((dK0, "Wk"), (dV0, "Wv"), (dA0, "Wa"), (dB0, "Wb")):
        dN = add(dN, mm(dX, tr(lay[W])))
    return dN, g


# ------------------------------------------------------------------------------------------------ sublayers: feed-forward networks


def swiglu_forward(X, w):
    G, U = mm(X, w["Wg"]), mm(X, w["Wu"])
    Hd = [[silu(a) * u for a, u in zip(rg, ru)] for rg, ru in zip(G, U)]
    return mm(Hd, w["Wd"]), (X, G, U, Hd)


def swiglu_backward(w, c, dY):
    X, G, U, Hd = c
    g = {"Wd": mm(tr(Hd), dY)}
    dH = mm(dY, tr(w["Wd"]))
    dG = [[dh * u * dsilu(a) for dh, u, a in zip(rh, ru, ra)] for rh, ru, ra in zip(dH, U, G)]
    dU = [[dh * silu(a) for dh, a in zip(rh, ra)] for rh, ra in zip(dH, G)]
    Xt = tr(X)
    g["Wg"], g["Wu"] = mm(Xt, dG), mm(Xt, dU)
    return add(mm(dG, tr(w["Wg"])), mm(dU, tr(w["Wu"]))), g


def moe_forward(lay, N, cfg, bug=None):
    """The mixture of experts over the rows of N: router logits N W_r, softmax, the K experts with the largest probability (a tie
    to the lower index, `moe.route`), their probabilities renormalised to sum to 1 as the combine weights, and the sum of the
    experts' outputs weighted by them. Returns (output, trace, aux) where aux is the balancing loss ALPHA E sum f_e pbar_e of the
    rows of this call. `bug` "no_renorm": the combine weights are the raw probabilities."""
    E, K, alpha = cfg["E"], cfg["K"], cfg["ALPHA"]
    n = len(N)
    logits = mm(N, lay["Wr"])
    routes = [moe.route(row, K, "softmax", renorm=(bug != "no_renorm")) for row in logits]
    members = [[] for _ in range(E)]
    for t, (chosen, w) in enumerate(routes):
        for e, we in zip(chosen, w):
            members[e].append((t, we))
    out = [[0.0] * cfg["D"] for _ in range(n)]
    caches = {}
    for e in range(E):
        if not members[e]:
            continue
        Y, cc = swiglu_forward([N[t] for t, _ in members[e]], lay["experts"][e])
        caches[e] = (cc, Y)
        for (t, we), y in zip(members[e], Y):
            out[t] = [o + we * v for o, v in zip(out[t], y)]
    f, pbar, _ = moe.balance_terms(logits, K)
    aux = moe.balance_loss(f, pbar, alpha)
    return out, {"N": N, "logits": logits, "routes": routes, "members": members, "caches": caches, "f": f,
                 "bug": bug}, aux


def moe_backward(lay, c, dOut, cfg):
    """Backward of `moe_forward`: through the combine weights (the probabilities of the chosen experts, renormalised), through
    the softmax and the router, and into each expert; plus the gradient of the balancing loss with the loads held fixed."""
    E, K, alpha = cfg["E"], cfg["K"], cfg["ALPHA"]
    N, logits, routes, members = c["N"], c["logits"], c["routes"], c["members"]
    n = len(N)
    g = {"experts": [], "Wr": None}
    dN = [[0.0] * cfg["D"] for _ in range(n)]
    dgate = [dict() for _ in range(n)]
    for e in range(E):
        if not members[e]:
            g["experts"].append({"Wg": [[0.0] * len(lay["experts"][e]["Wg"][0]) for _ in range(cfg["D"])],
                                 "Wu": [[0.0] * len(lay["experts"][e]["Wu"][0]) for _ in range(cfg["D"])],
                                 "Wd": [[0.0] * cfg["D"] for _ in range(len(lay["experts"][e]["Wd"]))]})
            continue
        cc, Y = c["caches"][e]
        dY = [[we * x for x in dOut[t]] for t, we in members[e]]
        for (t, _), y in zip(members[e], Y):
            dgate[t][e] = math.sumprod(dOut[t], y)
        dX, ge = swiglu_backward(lay["experts"][e], cc, dY)
        g["experts"].append(ge)
        for (t, _), row in zip(members[e], dX):
            dN[t] = [a + b for a, b in zip(dN[t], row)]
    dlog = moe.router_regulariser_grad(logits, c["f"], alpha, 0.0)
    for t in range(n):
        p = moe._softmax(logits[t])
        chosen = routes[t][0]
        z = sum(p[e] for e in chosen) if c["bug"] != "no_renorm" else 1.0
        sdp = sum(dgate[t][e] * p[e] for e in chosen)
        dp = [0.0] * E
        for e in chosen:
            dp[e] = dgate[t][e] / z - (sdp / (z * z) if c["bug"] != "no_renorm" else 0.0)
        pd = math.sumprod(p, dp)
        for e in range(E):
            dlog[t][e] += p[e] * (dp[e] - pd)
    g["Wr"] = mm(tr(N), dlog)
    dN = add(dN, mm(dlog, tr(lay["Wr"])))
    return dN, g


# ------------------------------------------------------------------------------------------------ the layer and the model


def layer_forward(lay, X, cfg, spec, form="recurrent", chunk=None, bug=None):
    """Pre-norm layer: X + mixer(norm1(X)), then + ffn(norm2(.)). Returns (output, trace, aux)."""
    mixer, ffn = spec
    N1, cn1 = tb.rms_forward(X, lay["g1"])
    if mixer == "gdn":
        A, mc = gdn_forward(lay, N1, cfg, form, chunk, bug if bug in ("no_q_scale", "gate_sign", "chunk_state_dropped") else None)
    else:
        window = cfg["WINDOW"] if mixer == "swa" else None
        A, mc = attention_forward(lay, N1, cfg, window, bug if bug in ("group_map", "window_off_by_one", "no_qk_norm_gain", "rotate_keys_only") else None)
    X1 = add(X, A)
    N2, cn2 = tb.rms_forward(X1, lay["g2"])
    aux = 0.0
    if ffn == "swiglu":
        Fo, fc = swiglu_forward(N2, lay)
    else:
        Fo, fc, aux = moe_forward(lay, N2, cfg, bug if bug == "no_renorm" else None)
    return add(X1, Fo), {"cn1": cn1, "mc": mc, "cn2": cn2, "fc": fc}, aux


def layer_backward(lay, c, dY, cfg, spec):
    """Returns (dX, gradients of the layer's tensors in the layout of `init_layer`)."""
    mixer, ffn = spec
    g = {}
    if ffn == "swiglu":
        dN2, gf = swiglu_backward(lay, c["fc"], dY)
    else:
        dN2, gf = moe_backward(lay, c["fc"], dY, cfg)
    dX1n, g["g2"] = tb.rms_backward(dN2, c["cn2"])
    dX1 = add(dY, dX1n)
    dN1, gm = (gdn_backward if mixer == "gdn" else attention_backward)(lay, c["mc"], dX1, cfg)
    dXn, g["g1"] = tb.rms_backward(dN1, c["cn1"])
    out = {"g1": g["g1"]}
    for k in lay:
        if k == "g1":
            continue
        out[k] = g["g2"] if k == "g2" else (gm[k] if k in gm else gf[k])
    return add(dX1, dXn), out


def forward(p, cfg, specs, ids, form="recurrent", chunk=None, bug=None, bug_layer=None, keep=False, taps=None):
    """Logits [n, V] of the text `ids`. `bug` plants one of the mistakes of `attention_forward`, `gdn_forward` or `moe_forward` in
    layer `bug_layer` (every layer when None). A list `taps` receives the hidden states: the embeddings, the output of each layer
    and the output of the final norm. With keep, also the traces and the sum of the layers' balancing losses."""
    X = [list(p["emb"][t]) for t in ids]
    if taps is not None:
        taps.append([list(x) for x in X])
    traces, aux = [], 0.0
    for l, (lay, spec) in enumerate(zip(p["layers"], specs)):
        X, trc, a = layer_forward(lay, X, cfg, spec, form, chunk, bug if bug_layer in (None, l) else None)
        traces.append(trc)
        aux += a
        if taps is not None:
            taps.append([list(x) for x in X])
    Hn, cnf = tb.rms_forward(X, p["gn"])
    if taps is not None:
        taps.append(Hn)
    Z = [[math.sumprod(h, row) for row in p["emb"]] for h in Hn]
    OPS["matmul"] += len(Hn) * len(p["emb"]) * len(p["gn"])
    return (Z, {"layers": traces, "cnf": cnf, "Hn": Hn, "aux": aux}) if keep else Z


def loss_grad(p, cfg, specs, ids, positions):
    """Mean cross-entropy over the predictions made at `positions` (the logits at t predict ids[t + 1]), plus the balancing losses
    of the mixture layers, and the gradient of that sum. Returns (cross-entropy, balancing loss, gradient, per-position losses)."""
    Z, tr_ = forward(p, cfg, specs, ids, keep=True)
    n, V = len(ids), cfg["V"]
    m = len(positions)
    dZ = [[0.0] * V for _ in range(n)]
    ce, per = 0.0, {}
    for t in positions:
        pr = softmax(Z[t])
        loss_t = -math.log(pr[ids[t + 1]])
        per[t] = loss_t
        ce += loss_t / m
        for r in range(V):
            dZ[t][r] = (pr[r] - (1.0 if r == ids[t + 1] else 0.0)) / m
    emb = p["emb"]
    dHn = mm(dZ, emb)
    g = zeros_like(p)
    g["emb"] = mm(tr(dZ), tr_["Hn"])
    dX, g["gn"] = tb.rms_backward(dHn, tr_["cnf"])
    for l in reversed(range(len(specs))):
        dX, g["layers"][l] = layer_backward(p["layers"][l], tr_["layers"][l], dX, cfg, specs[l])
    for t, tok in enumerate(ids):
        row = g["emb"][tok]
        for d in range(len(row)):
            row[d] += dX[t][d]
    return ce, tr_["aux"], g, per


# ------------------------------------------------------------------------------------------------ the task

TASK = {"pairs": 6, "symbols": 6}
SEQ_LEN = 1 + 2 * TASK["pairs"]                       # the header, then six (x, y) pairs
PREDICT_AT = [2 * i - 1 for i in range(1, TASK["pairs"] + 1)]   # the position of x_i: its logits predict y_i


def draw_sequence(g):
    """One sequence of the task from generator g: a header h = integer(6), shown as the token 6 + h at position 0; then for
    i = 1 .. 6 a symbol x_i = integer(6) (token x_i) followed by y_i = (x_i + h) mod 6 (token y_i). Seven draws."""
    S = TASK["symbols"]
    h = g.integer(S)
    ids = [S + h]
    for _ in range(TASK["pairs"]):
        x = g.integer(S)
        ids += [x, (x + h) % S]
    return ids


# ------------------------------------------------------------------------------------------------ training

NAME = "project_architecture_comparison"
SW = ("swiglu",)
ARMS = {
    "dense": [("gqa", "swiglu")] * 2,
    "moe": [("gqa", "moe")] * 2,
    "gdn": [("gdn", "swiglu")] * 2,
    "swa": [("swa", "swiglu")] * 2,
    "hybrid": [("gdn", "swiglu"), ("gqa", "swiglu")],
}
RECIPE = {"B": 8, "steps": 120, "warmup": 10, "floor": 0.1, "clip": 1.0, "b1": 0.9, "b2": 0.95, "eps": 1e-8, "n_eval": 120}


def eval_losses(p, cfg, specs, seed):
    """Mean cross-entropy of the six predictions on 120 sequences of generator `<NAME>.eval.<seed>`, and the same per prediction."""
    g = bookrng.generator(f"{NAME}.eval.{seed}")
    per = [0.0] * len(PREDICT_AT)
    n = RECIPE["n_eval"]
    for _ in range(n):
        ids = draw_sequence(g)
        Z = forward(p, cfg, specs, ids)
        for k, t in enumerate(PREDICT_AT):
            per[k] += -math.log(softmax(Z[t])[ids[t + 1]]) / n
    return sum(per) / len(per), per


def train(job):
    """Train the arm for RECIPE["steps"] Adam steps. job = (arm, seed index, peak rate, nudge). The nudge is added to the first
    weight of the token table before the first step. The batches come from generator `<NAME>.data.<seed>`, one `draw_sequence`
    after another, eight to a step. Returns the final evaluation, the mean training loss of the last 10 steps and the fraction of
    mixture slots the busiest expert took on the evaluation sequences (1 / E for a perfectly balanced router)."""
    arm, seed, lr, nudge = job
    cfg, specs = CFG, ARMS[arm]
    p = init_model(f"{NAME}.init.{arm}.{seed}", cfg, specs)
    p["emb"][0][0] += nudge
    ts = [rows for _, rows in leaves(p)]
    mom = [([[0.0] * len(r) for r in t], [[0.0] * len(r) for r in t]) for t in ts]
    o = {"b1": RECIPE["b1"], "b2": RECIPE["b2"], "eps": RECIPE["eps"], "wd": 0.0}
    rate = schedules.linear_decay(lr, lr * RECIPE["floor"], RECIPE["warmup"], RECIPE["steps"])
    data = bookrng.generator(f"{NAME}.data.{seed}")
    B = RECIPE["B"]
    tail = []
    for step in range(RECIPE["steps"]):
        total = zeros_like(p)
        ce_sum = 0.0
        for _ in range(B):
            ce, aux, g, _ = loss_grad(p, cfg, specs, draw_sequence(data), PREDICT_AT)
            add_into(total, g)
            ce_sum += ce
        scale_by(total, 1.0 / B)
        gs = [rows for _, rows in leaves(total)]
        norm = tg.global_norm(gs)
        clip = min(1.0, RECIPE["clip"] / norm) if norm > 0 else 1.0
        tg.adamw_update(ts, gs, mom, step + 1, rate(step), clip, [False] * len(ts), o)
        if step >= RECIPE["steps"] - 10:
            tail.append(ce_sum / B)
    loss, per = eval_losses(p, cfg, specs, seed)
    return {"job": job, "loss": loss, "per": per, "train": sum(tail) / len(tail), "busiest": busiest_share(p, cfg, specs, seed)}


def busiest_share(p, cfg, specs, seed):
    """For each mixture layer, the share of the expert slots on 30 evaluation sequences that went to the busiest expert (1 / E when
    balanced); an empty list for a stack without a mixture."""
    if not any(s[1] == "moe" for s in specs):
        return []
    g = bookrng.generator(f"{NAME}.eval.{seed}")
    counts = {l: [0] * cfg["E"] for l, s in enumerate(specs) if s[1] == "moe"}
    for _ in range(30):
        _, t = forward(p, cfg, specs, draw_sequence(g), keep=True)
        for l in counts:
            for chosen, _ in t["layers"][l]["fc"]["routes"]:
                for e in chosen:
                    counts[l][e] += 1
    return [max(c) / sum(c) for c in counts.values()]


# ------------------------------------------------------------------------------------------------ milestone 1: the blocks

BLOCKS = {"dense": ("gqa", "swiglu"), "moe": ("gqa", "moe"), "gdn": ("gdn", "swiglu"), "swa": ("swa", "swiglu")}
BLOCK_LABELS = {"dense": "dense: grouped-query attention, SwiGLU", "moe": "mixture: grouped-query attention, 4 experts, top 2",
                "gdn": "gated delta network, SwiGLU", "swa": "sliding window of 4 keys, SwiGLU",
                "hybrid": "hybrid: gated delta network then dense"}
NAME_BLOCKS = f"{NAME}.blocks"
NAME_REF = f"{NAME}.reference"
REF_TOKENS = 13                      # the length of a task sequence


@lru_cache(maxsize=None)
def ref_ids():
    return draw_sequence(bookrng.generator(NAME_REF))


def block_model(arm):
    """The reference weights of an arm, from the generator `<NAME>.blocks.<arm>` in the order of `init_model`."""
    return init_model(f"{NAME_BLOCKS}.{arm}", CFG, ARMS[arm])


def layer_parameters(spec):
    return parameter_count({"layers": [init_layer(bookrng.generator("x"), CFG, spec)]})


def blocks():
    rows = []
    base = parameter_count(block_model("dense"))
    for arm in ("dense", "moe", "gdn", "swa", "hybrid"):
        p = block_model(arm)
        n = parameter_count(p)
        layer = [parameter_count({"layers": [lay]}) for lay in p["layers"]]
        rows.append((f"`{arm}`", BLOCK_LABELS[arm], " + ".join(_int(x) for x in layer), _int(n), f"{100 * (n - base) / base:+.2f}%"))
    return ("The five stacks of the project, two layers each, at width D = 16 and vocabulary 12 (the token table is tied to the "
            "head). Weights come from `bookrng.generator(\"" + NAME_BLOCKS + ".<arm>\")` in the order of the initialization routine "
            "(seeds " + ", ".join(f"`{a}` {_seed(f'{NAME_BLOCKS}.{a}')}" for a in ARMS) + "). Parameters are counted by walking the "
            "tensors that were drawn.\n\n"
            + _table(["arm", "layers", "parameters in each layer", "parameters of the model", "against dense"], rows, "llrrr"))


def coordinates(name, p):
    """For every tensor of p, two coordinates (row, column) drawn from generator `name`: row = integer(rows), column = integer(columns), twice."""
    g = bookrng.generator(name)
    out = []
    for path, rows in leaves(p):
        for _ in range(2):
            i = g.integer(len(rows))
            out.append((path, i, g.integer(len(rows[0]))))
    return out


def _task_loss(p, specs, ids):
    Z, t = forward(p, CFG, specs, ids, keep=True)
    tot = 0.0
    for pos in PREDICT_AT:
        tot += -math.log(softmax(Z[pos])[ids[pos + 1]]) / len(PREDICT_AT)
    return tot + t["aux"]


CHECK_STEP = 1e-5
CRITERION = 1e-6


def gradient_checks():
    ids = ref_ids()
    rows = []
    for arm in ("dense", "moe", "gdn", "swa", "hybrid"):
        specs, p = ARMS[arm], block_model(arm)
        ce, aux, g, _ = loss_grad(p, CFG, specs, ids, PREDICT_AT)
        gl = {path: r for path, r in leaves(g)}
        pts = coordinates(f"{NAME}.gradcheck.{arm}", p)
        by_path = {path: r for path, r in leaves(p)}
        worst = big = 0.0
        for path, i, j in pts:
            r = by_path[path]
            old = r[i][j]
            r[i][j] = old + CHECK_STEP
            hi = _task_loss(p, specs, ids)
            r[i][j] = old - CHECK_STEP
            lo = _task_loss(p, specs, ids)
            r[i][j] = old
            ana = gl[path][i][j]
            worst = max(worst, abs((hi - lo) / (2 * CHECK_STEP) - ana))
            big = max(big, abs(ana))
        rows.append((f"`{arm}`", f"{ce + aux:.4f}", len(by_path), len(pts), f"{big:.3f}", _below(worst, CRITERION)))
    return (f"Central differences with step {CHECK_STEP:g} against the hand-written backward pass, in double precision, for each stack on the "
            f"reference sequence (ids {' '.join(str(i) for i in ref_ids())}, drawn from `{NAME_REF}`, seed {_seed(NAME_REF)}). "
            "The loss is the mean cross-entropy (nats) of the six scored predictions of milestone 4 plus the balancing loss. Two coordinates of every tensor are "
            "checked, drawn from the generator `" + NAME + ".gradcheck.<arm>`: for each tensor in storage order, a row `integer(rows)` "
            "then a column `integer(columns)`, twice. Storage order is the order of the draws of milestone 1 with the token table first and "
            "the final gain last, layer by layer (the dense layer: g1, W_Q, W_K, W_V, the two QK gains, W_O, g2, W_gate, W_up, W_down; "
            "the gated delta layer: g1, W_Q, W_K, W_V, W_a, W_b, the decay bias c, the output-norm gain, W_O, g2, then the network; the "
            "mixture's network: the router, then each expert's W_gate, W_up, W_down); the bias c, which has no draws, sits between W_b and "
            "the output-norm gain. A vector (a gain or the bias) is a matrix of one row: its row draw is `integer(1)`, which consumes a draw "
            "and gives 0, and then comes the column draw. Criterion: every difference below 1e-6.\n\n"
            + _table(["stack", "loss", "tensors", "coordinates", "largest gradient checked", "largest difference"], rows))


NAME_CHUNK = f"{NAME}.chunked"
U64 = 2.0 ** -53


def chunk_tolerance(cfg, L, n, C, z_max):
    """2 L (n + C) (DK + DV + 2) u z_max: a first-order count. The recurrence rewrites the state at each of the n steps with a map of
    norm at most 1 (decay in (0, 1] and a unit-key delta rule), so the rounding of a step (DK + DV + 2 products and sums) is added
    and never amplified; the chunked form adds the C steps of its triangular solve."""
    return 2 * L * (n + C) * (cfg["DK"] + cfg["DV"] + 2) * U64 * z_max


def chunked_recurrent():
    arm, n = "gdn", 12
    p, ids = block_model(arm), ref_ids()[:n]
    ref = forward(p, CFG, ARMS[arm], ids)
    z_max = max(abs(x) for r in ref for x in r)
    rows = []
    for C in (1, 2, 3, 4, 6, 12):
        got = forward(p, CFG, ARMS[arm], ids, form="chunked", chunk=C)
        d = im.max_abs(got, ref)
        tol = chunk_tolerance(CFG, len(ARMS[arm]), n, C, z_max)
        rows.append((C, n // C, _below(d, 1e-13), _sci(tol, 2), "yes" if d < tol else "NO"))
    bad = forward(p, CFG, ARMS[arm], ids, form="chunked", chunk=4, bug="chunk_state_dropped")
    d = im.max_abs(bad, ref)
    tol = chunk_tolerance(CFG, len(ARMS[arm]), n, 4, z_max)
    rows.append(("4, state dropped at chunk boundaries (planted)", 3, f"{d:.3f}", _sci(tol, 2), "yes" if d < tol else "NO"))
    return (f"The `gdn` stack on the first {n} ids of the reference sequence: the logits ({n} x 12) of the recurrent form and of the chunked form of "
            f"Chapter 5 (`linear_attention.gated_delta_chunked`) at six chunk sizes, with the largest absolute difference and the tolerance "
            f"$2L(n + C)(d_k + d_v + 2)\\,u\\,z_{{\\max}}$ with $L = 2$, $n = {n}$, $d_k = 4$, $d_v = 7$, $u = 2^{{-53}}$ and the largest logit "
            f"$z_{{\\max}}$ = {z_max:.3f}. The last row starts every chunk of the chunked form from a zero state.\n\n"
            + _table(["chunk size C", "chunks", "largest difference of logits", "tolerance", "within tolerance"], rows))


# ------------------------------------------------------------------------------------------------ milestone 2: the cost model

BYTES = 2                                   # numbers stored in bfloat16
NO_WINDOW = 10 ** 18


def layer_record(spec, cfg):
    """The record of one layer for `model_cost.ledger`: (mixer parameters, feed-forward parameters stored, feed-forward parameters
    a token uses, gain parameters, cached numbers per token, window, width of the scores)."""
    mixer, ffn = spec
    D = cfg["D"]
    if ffn == "swiglu":
        stored, flops = moe.layer_costs(D, cfg["F"], 1, 1)
    else:
        stored, flops = moe.layer_costs(D, cfg["FE"], cfg["E"], cfg["K"])
    if mixer == "gdn":
        HG, DK, DV = cfg["HG"], cfg["DK"], cfg["DV"]
        return (2 * D * HG * DK + 2 * D * HG * DV + 2 * D * HG, stored, flops // 2, 2 * D + HG + DV, 0, 0, 0)
    H, HKV, DH = cfg["H"], cfg["HKV"], cfg["DH"]
    window = cfg["WINDOW"] if mixer == "swa" else NO_WINDOW
    return (2 * D * H * DH + 2 * D * HKV * DH, stored, flops // 2, 2 * D + 2 * DH, 2 * HKV * DH, window, H * DH)


def recurrence_flops(spec, cfg):
    """FLOPs per token of the gated delta recurrence (decay, read, write and read-out of a state of DK x DV numbers per head)."""
    return 8 * cfg["HG"] * cfg["DK"] * cfg["DV"] if spec[0] == "gdn" else 0


def state_numbers(spec, cfg):
    return cfg["HG"] * cfg["DK"] * cfg["DV"] if spec[0] == "gdn" else 0


def costs(specs, cfg, T, b=BYTES):
    """(parameters P, parameters a token uses N_act, decode FLOPs of a token that sees T keys, cache bytes of a sequence of T tokens,
    state bytes that do not grow) from the records of the layers, by `model_cost.ledger`."""
    layers = [layer_record(s, cfg) for s in specs]
    V, D = cfg["V"], cfg["D"]
    P, flops, cache = mc.ledger(layers, V * D, D, V * D, T, b)
    n_act = V * D + sum(m + fu for m, _, fu, *_ in layers)
    flops += sum(recurrence_flops(s, cfg) for s in specs)
    return P, n_act, flops, cache, sum(state_numbers(s, cfg) for s in specs) * b


def prefill_flops(specs, cfg, T):
    """FLOPs of a prompt of T tokens, token i seeing i keys: the sum over i of the decode FLOPs at i."""
    return sum(costs(specs, cfg, i)[2] for i in range(1, T + 1))


def counted_flops(arm, n):
    """Multiply-adds counted while running the forward pass on the first n ids of the reference sequence, times 2."""
    reset_ops()
    forward(block_model(arm), CFG, ARMS[arm], ref_ids()[:n])
    return 2 * sum(OPS.values())


def retained_numbers(arm):
    """Numbers a decode step must keep after the forward pass over the reference sequence, counted from the objects it built: the
    keys and values of the last WINDOW (or all) positions for an attention layer, the final state of every head for a gated delta layer."""
    p, specs = block_model(arm), ARMS[arm]
    _, trc = forward(p, CFG, specs, ref_ids(), keep=True)
    total_attn = total_state = 0
    for spec, t in zip(specs, trc["layers"]):
        mc_ = t["mc"]
        if spec[0] == "gdn":
            total_state += sum(len(h["aux"][2][-1]) * len(h["aux"][2][-1][0]) for h in mc_["heads"])
        else:
            keep = len(mc_["k2"]) if mc_["w"] is None else min(len(mc_["k2"]), mc_["w"])
            total_attn += sum(len(v) for row in mc_["k2"][-keep:] for v in row) + sum(len(v) for row in mc_["v"][-keep:] for v in row)
    return total_attn, total_state


def cost_model():
    T = REF_TOKENS
    rows = []
    for arm in ("dense", "moe", "gdn", "swa", "hybrid"):
        specs, p = ARMS[arm], block_model(arm)
        P, n_act, dec, cache, state = costs(specs, CFG, T)
        pre = prefill_flops(specs, CFG, T)
        pre_c = counted_flops(arm, T)
        dec_c = pre_c - counted_flops(arm, T - 1)
        assert P == parameter_count(p) and pre == pre_c and dec == dec_c, (arm, P, parameter_count(p), pre, pre_c, dec, dec_c)
        a_n, s_n = retained_numbers(arm)
        assert (a_n * BYTES, s_n * BYTES) == (cache, state), (arm, a_n, s_n, cache, state)
        rows.append((f"`{arm}`", _int(P), _int(n_act), _int(pre // T), _int(dec), _int(cache), _int(state), _int(cache + state)))
    return (f"Costs at the sizes of the project (D = 16, 2 layers, vocabulary 12, tied head) for a sequence of T = {T} tokens, numbers kept in "
            f"bfloat16 ({BYTES} bytes). Parameters are the tensors that exist; active parameters are the ones in the matrix products of a "
            "token (the table lookup is not a product; the tied head counts once; the mixture counts the router and the 2 experts used). "
            "FLOPs count 2 per multiply-add in the matrix products, in the products of attention (scores and weighted values, over the keys "
            f"a token sees) and in the recurrence (8 times the heads times $d_k d_v$ per token and layer); norms, softmax, gates and additions are not counted. "
            f"'Prefill' is the mean per token over a prompt of {T} tokens (token $i$ sees $i$ keys; the total for the prompt divided by {T} and rounded down to a whole number, so the window's {prefill_flops(ARMS['swa'], CFG, T) / T:,.2f} is printed {prefill_flops(ARMS['swa'], CFG, T) // T:,}), 'decode' the cost of the "
            f"{T}th token alone. Each FLOP count was checked against the multiply-adds counted while the forward pass ran (the total for the "
            f"prompt, and the {T}th token as the difference between the counts for {T} and {T - 1} tokens), each parameter count against the "
            "tensors, and each cache against the numbers the forward pass holds.\n\n"
            + _table(["stack", "parameters", "active parameters", "prefill FLOPs per token", "decode FLOPs", "cache bytes", "state bytes", "bytes held"], rows))


def cost_scaling():
    """Decode FLOPs and bytes held at four contexts, by the ledger, for the stacks at the project's sizes; and the contexts at which
    the gated delta stack becomes the cheaper one."""
    ctxs = (13, 256, 4096, 65536)
    rows = []
    for arm in ("dense", "moe", "gdn", "swa", "hybrid"):
        specs = ARMS[arm]
        row = [f"`{arm}`"]
        for c in ctxs:
            _, _, f, cache, state = costs(specs, CFG, c)
            row += [_int(f), _int(cache + state)]
        rows.append(row)
    cross = {}
    for what, idx in (("decode FLOPs", 2), ("bytes held", None)):
        c = 1
        while True:
            d, g = costs(ARMS["dense"], CFG, c), costs(ARMS["gdn"], CFG, c)
            if (d[2] > g[2]) if idx == 2 else (d[3] + d[4] > g[3] + g[4]):
                break
            c += 1
        cross[what] = c
    heads = ["stack"] + [x for c in ctxs for x in (f"FLOPs at {_int(c)}", f"bytes at {_int(c)}")]
    return (f"The same ledger at four contexts (tokens a decode step sees; sizes of the project; bfloat16). The attention stacks' decode "
            "FLOPs and cache grow with the context; the sliding window stops growing at its 4 keys; the gated delta stack and the "
            "state of a hybrid do not grow at all. The gated delta stack needs fewer decode FLOPs than the dense one from a context of "
            f"{cross['decode FLOPs']} tokens, and holds fewer bytes from a context of {cross['bytes held']} tokens.\n\n"
            + _table(heads, rows))


BIG = cfg_with(V=32000, D=1024, H=16, HKV=4, DH=64, F=2816, WINDOW=1024, E=16, K=2, FE=704, HG=16, DK=64, DV=64)
BIG_LAYERS = 24


def big_specs(arm):
    if arm == "hybrid":
        return [("gdn" if (i + 1) % 4 else "gqa", "swiglu") for i in range(BIG_LAYERS)]
    return [BLOCKS[arm]] * BIG_LAYERS


def big_shape():
    rows = []
    for arm in ("dense", "moe", "gdn", "swa", "hybrid"):
        specs = big_specs(arm)
        P, n_act, f, cache, state = costs(specs, BIG, 131072)
        rows.append((f"`{arm}`", f"{P / 1e6:,.1f}", f"{n_act / 1e6:,.1f}", f"{costs(specs, BIG, 4096)[2] / 1e6:,.1f}", f"{f / 1e6:,.1f}",
                     f"{(cache + state) / 2 ** 20:,.1f}"))
    return (f"The ledger at a size where the blocks differ in what they cost: width 1,024, 24 layers, vocabulary 32,000, 16 query heads and 4 "
            "key-value heads of width 64, SwiGLU hidden width 2,816, a window of 1,024 keys, a mixture of 16 experts of hidden width 704 "
            "with 2 used, and a gated delta layer of 16 heads with $d_k = d_v = 64$; the hybrid has a gated delta layer in three of every "
            "four positions and a dense layer in the fourth. The sizes are not matched between the stacks (the gated delta layer holds more "
            "parameters than the attention layer here); the table reports what each costs. Millions of parameters and of FLOPs, bfloat16, "
            "MiB for one sequence.\n\n"
            + _table(["stack", "parameters (M)", "active (M)", "decode MFLOPs at 4,096", "decode MFLOPs at 131,072", "MiB held at 131,072"], rows))


# ------------------------------------------------------------------------------------------------ milestone 3: the parity harness

U32 = 2.0 ** -24


def r32_params(p):
    """A copy of p with every weight rounded to the nearest 32-bit float (the checkpoint a published model would ship)."""
    def go(x):
        if isinstance(x, dict):
            return {k: go(v) for k, v in x.items()}
        if x and isinstance(x[0], (dict, list)):
            return [go(v) for v in x]
        return [im.r32(v) for v in x]
    return go(p)


def layer_stages(spec, cfg, T):
    """The stages of one layer, [(stage, roundings along the longest chain of dependent operations)], by the rule of Chapter 2: a sum
    of n products costs n roundings, each further operation on its result one. T is the number of keys a token sees (the window
    limits it for a sliding window), and for the gated delta layer the number of tokens along the chain of the state."""
    mixer, ffn = spec
    D, DH, H = cfg["D"], cfg["DH"], cfg["H"]
    st = [("RMS norm 1", D + 6)]
    if mixer == "gdn":
        DK, DV, HG = cfg["DK"], cfg["DV"], cfg["HG"]
        st += [("projection (dot of length D)", D),
               ("L2 normalisation of q and k", DK + 4),
               (f"recurrence, {T} tokens of d_k + 5", T * (DK + 5)),
               ("read-out (dot of length d_k)", DK),
               ("output norm over d_v", DV + 6),
               ("W_O of the recurrent layer (dot of length 2 d_v)", HG * DV)]
    else:
        keys = min(T, cfg["WINDOW"]) if mixer == "swa" else T
        st += [("projection (dot of length D)", D),
               ("QK-norm", DH + 6),
               ("rotation", 4),
               ("scores (dot of length d_h, scale)", DH + 1),
               ("softmax over the keys", keys + 3),
               ("weighted sum of values", keys),
               ("W_O of the attention layer (dot of length 4 d_h)", H * DH)]
    st += [("residual add (mixer)", 1), ("RMS norm 2", D + 6)]
    if ffn == "swiglu":
        st += [("gate or up projection (dot of length D)", D), ("silu and product with up", 4),
               ("W_down (dot of length F)", cfg["F"])]
    else:
        st += [("router (dot of length D)", D), ("router softmax over E experts", cfg["E"] + 3),
               ("renormalisation of the kept pair", 2),
               ("expert gate or up (dot of length D)", D), ("silu and product with up", 4),
               ("expert W_down (dot of length FE)", cfg["FE"]), ("weighted sum of the K experts", cfg["K"])]
    st.append(("residual add (network)", 1))
    return st


def layer_roundings(spec, cfg, T):
    return sum(c for _, c in layer_stages(spec, cfg, T))


def rounding_stages():
    T = REF_TOKENS
    kinds = (("dense", ("gqa", "swiglu")), ("swa", ("swa", "swiglu")), ("moe", ("gqa", "moe")), ("gdn", ("gdn", "swiglu")))
    per = {k: layer_stages(sp, CFG, T) for k, sp in kinds}
    order = ("RMS norm 1", "projection (dot of length D)", "L2 normalisation of q and k", "QK-norm", "rotation",
             "scores (dot of length d_h, scale)", "softmax over the keys", "weighted sum of values", f"recurrence, {T} tokens of d_k + 5",
             "read-out (dot of length d_k)", "output norm over d_v", "W_O of the attention layer (dot of length 4 d_h)", "W_O of the recurrent layer (dot of length 2 d_v)",
             "residual add (mixer)", "RMS norm 2", "gate or up projection (dot of length D)", "router (dot of length D)",
             "router softmax over E experts", "renormalisation of the kept pair", "expert gate or up (dot of length D)",
             "silu and product with up", "W_down (dot of length F)", "expert W_down (dot of length FE)",
             "weighted sum of the K experts", "residual add (network)")
    names = sorted({n for k, _ in kinds for n, _ in per[k]}, key=order.index)
    rows = []
    for n in names:
        row = [n]
        for k, _ in kinds:
            c = dict(per[k]).get(n)
            row.append("-" if c is None else str(c))
        rows.append(row)
    rows.append(["**layer total**"] + [f"**{sum(c for _, c in per[k])}**" for k, _ in kinds])
    head = (CFG["D"] + 6) + CFG["D"]
    tap_rows = []
    for arm, specs in ARMS.items():
        n = [layer_roundings(sp, CFG, T) for sp in specs]
        tap_rows.append((f"`{arm}`", n[0], n[1], head, 1, n[0] + 1, n[0] + n[1] + 1, n[0] + n[1] + head + 1))
    return (f"The roundings counted along the longest chain of one layer, stage by stage, for a token that sees {T} keys (the window "
            f"of `swa` keeps {CFG['WINDOW']} of them in the softmax and the weighted sum; the recurrence runs along the {T} tokens of the sequence). "
            "The rule is that of Chapter 2: a dot product of length $n$ costs $n$, an RMS norm over $n$ numbers $n + 6$, and each further "
            "operation on a result one. The L2 normalisation is the sum of $d_k$ squares, the addition of $\\varepsilon$, the square root, the reciprocal and the "
            "product, $d_k + 4$; a step of the recurrence is the decay product, the dot of length $d_k$, the subtraction, the product with $\\beta$, the product "
            "with the key and the addition to the state, $d_k + 5$; the QK-norm is $d_h + 6$ as in Chapter 2; the router's renormalisation is a "
            "sum and a division, and the weighted sum of the $K$ experts is $K$. A dash means the layer has no such stage. The softplus of the decay and the sigmoid of the write strength "
            "are on branches shorter than the chain through the L2-normalised key and do not enter. The hybrid has the gated delta layer "
            "first and the dense layer second, so its layers cost those of the `gdn` and `dense` columns.\n\n"
            + _table(["stage", "dense layer", "window layer", "mixture layer", "gated delta layer"], rows, "lrrrr")
            + f"\n\nThe head adds a final RMS norm ({CFG['D']} + 6) and the output projection ({CFG['D']}), {head} in all, and the golden's own "
            "storage adds 1. The bound on a tap is $(N + 1)\\,2^{-24}$ with $N$ the sum of the counts of the layers up to the tap (and of the "
            "head for the logits); the embeddings tap, before any layer, has $N = 0$. Per stack, in units of $2^{-24}$:\n\n"
            + _table(["stack", "layer 0", "layer 1", "head", "embeddings tap", "layer 0 tap", "layer 1 tap", "logits tap"], tap_rows, "l" + "r" * 7))


def roundings(specs, cfg, T, layers=None, head=True):
    k = len(specs) if layers is None else layers
    n = sum(layer_roundings(s, cfg, T) for s in specs[:k])
    return n + ((cfg["D"] + 6) + cfg["D"] if head else 0)


def golden_blob(arm):
    """The golden of a stack on the reference sequence as a safetensors file: the ids (I32), the hidden state after the embeddings
    and after each layer, the output of the final norm and the logits (F32 each, a 64-bit result rounded to 32 bits)."""
    p, specs, ids = block_model(arm), ARMS[arm], ref_ids()
    taps = []
    Z = forward(p, CFG, specs, ids, taps=taps)
    items = [("ids", "I32", [len(ids)], list(ids))]
    for i, t in enumerate(taps[:-1]):
        items.append((f"hidden_{i:02d}", "F32", [len(ids), CFG["D"]], [im.r32(x) for r in t for x in r]))
    items.append(("final_norm", "F32", [len(ids), CFG["D"]], [im.r32(x) for r in taps[-1] for x in r]))
    items.append(("logits", "F32", [len(ids), CFG["V"]], [im.r32(x) for r in Z for x in r]))
    meta = {"generator": f"{NAME_BLOCKS}.{arm}", "arithmetic": "64-bit, stored as 32-bit",
            "tolerance_basis": "rounding count x 2^-24"}
    return im.write_safetensors(items, meta)[0]


def digest(blob):
    import hashlib
    return hashlib.sha256(blob).hexdigest()[:16]


def read_golden(blob):
    _, tensors, _ = im.read_safetensors(blob)
    taps = [tensors[k][2] for k in sorted(tensors) if k.startswith("hidden_")]
    return taps, tensors["final_norm"][2], tensors["logits"][2]


def flat(taps):
    return [x for r in taps for x in r]


def tolerance(specs, k):
    """The relative bound after k layers (k == len(specs) + 1 means the logits): (N + 1) 2^-24, N the roundings, 1 for the golden's storage."""
    L = len(specs)
    n = roundings(specs, CFG, REF_TOKENS, layers=min(k, L), head=k > L)
    return (n + 1) * U32


def measures(got, want):
    g, w = flat(got) if isinstance(got[0], list) else got, flat(want) if isinstance(want[0], list) else want
    return im.max_abs(g, w), im.rel_l2(g, w), im.cosine(g, w)


def check_against_golden(p, arm, blob, bug=None, bug_layer=None):
    """The harness: run the candidate parameters p, compare every tap and the logits with the golden file. Returns [(tap name,
    max_abs, rel_l2, 1 - cosine, bound)] and the verdict of each measure on the logits."""
    specs, ids = ARMS[arm], ref_ids()
    gold_taps, gold_norm, gold_logits = read_golden(blob)
    taps = []
    Z = forward(p, CFG, specs, ids, bug=bug, bug_layer=bug_layer, taps=taps)
    got = [flat(t) for t in taps[:-1]]
    rows = []
    names = ["embeddings"] + [f"layer {i}" for i in range(len(specs))]
    for i, (g, w) in enumerate(zip(got, gold_taps)):
        m = measures(g, w)
        rows.append((names[i], m[0], m[1], 1 - m[2], tolerance(specs, i)))
    m = measures(flat(Z), gold_logits)
    rows.append(("logits", m[0], m[1], 1 - m[2], tolerance(specs, len(specs) + 1)))
    return rows


def verdict_of(row, gold_scale):
    """(max_abs, rel_l2, cosine) tests derived from the one relative bound eps: rel_l2 <= eps, max_abs <= eps ||golden||, 1 - cos <= eps^2 / 2."""
    _, mx, rel, onecos, eps = row
    return (mx <= eps * gold_scale, rel <= eps, onecos <= eps * eps / 2)


def golden_files():
    import json
    import struct
    rows = []
    for arm in ("dense", "moe", "gdn", "swa", "hybrid"):
        blob = golden_blob(arm)
        _, tensors, _ = im.read_safetensors(blob)
        rows.append((f"`{arm}`", len(tensors), _int(len(blob)), digest(blob)))
    blob = golden_blob("dense")
    _, tensors, _ = im.read_safetensors(blob)
    names = ", ".join(f"`{k}`" for k in tensors)
    blob0 = golden_blob("dense")
    (hlen,) = struct.unpack("<Q", blob0[:8])
    header = json.loads(blob0[8:8 + hlen].decode("utf-8"))
    meta = header["__metadata__"]
    entries = [f'"{k}":{{"dtype":"{v["dtype"]}","shape":{json.dumps(v["shape"], separators=(",", ":"))},'
               f'"data_offsets":{json.dumps(v["data_offsets"], separators=(",", ":"))}}}'
               for k, v in header.items() if k != "__metadata__"]
    meta_text = json.dumps({"__metadata__": meta}, separators=(",", ":"))[:-1]
    pieces = [meta_text[:meta_text.index('"arithmetic"')], meta_text[meta_text.index('"arithmetic"'):meta_text.index('"tolerance_basis"')],
              meta_text[meta_text.index('"tolerance_basis"'):] + ","] + [e + ("," if i < len(entries) - 1 else "}") for i, e in enumerate(entries)]
    return (f"Goldens of the five stacks on the reference sequence (ids {' '.join(str(i) for i in ref_ids())}), each a safetensors file written "
            "by `importing_models.write_safetensors` (Chapter 2): the ids as 32-bit integers, then the hidden state after the embeddings and after "
            "each layer, the output of the final norm and the logits as 32-bit floats (the 64-bit result rounded to the nearest 32-bit "
            f"float), little-endian, back to back in this order. The `dense` file holds {names}. The file is an 8-byte little-endian length "
            f"prefix, the header text, and the {len(blob0) - 8 - hlen:,} bytes of data. The header is compact JSON (separators `,` and `:`, no other spaces), "
            f"in the key order below, with the metadata first, padded with spaces on the right to a multiple of 8 bytes: {hlen} bytes for every stack "
            f"(the text is {len(json.dumps(header, separators=(',', ':')))} bytes for `dense`, and the padding absorbs the different "
            "lengths of the other arms' names). The header text of `dense` is these lines joined with no separator (the line breaks are "
            "for the page); the other stacks differ only in the last word of the generator name, which is the arm:\n\n"
            "```text\n" + "\n".join(pieces) + "\n```\n\n"
            "Digest: the first 16 hexadecimal digits of the SHA-256 of the whole file. "
            f"Stack, tensors, bytes and digest:\n\n"
            + _table(["stack", "tensors", "file bytes", "digest"], rows))


def _scale(blob):
    return math.sqrt(sum(x * x for x in read_golden(blob)[2]))


def parity_table():
    rows = []
    for arm in ("dense", "moe", "gdn", "swa", "hybrid"):
        blob = golden_blob(arm)
        p32 = r32_params(block_model(arm))
        res = check_against_golden(p32, arm, blob)
        lg = res[-1]
        t = verdict_of(lg, _scale(blob))
        N = roundings(ARMS[arm], CFG, REF_TOKENS)
        rows.append((f"`{arm}`", N, _sci(lg[4], 2), _below(lg[2], 1e-7), _below(lg[1], 1e-6), "yes" if all(t) else "NO", f"{lg[4] / max(lg[2], 1e-12):.0f}"))
    p = block_model("dense")
    ref = md.forward(p, {"V": 12, "D": 16, "L": 2, "H": 4, "HKV": 2, "DH": 4, "F": 24, "BASE": 100.0}, ref_ids())
    ours = forward(p, CFG, ARMS["dense"], ref_ids())
    rel = im.rel_l2(ours, ref)
    return (f"The harness on correct candidates. The candidate is the stack with every weight rounded to 32 bits and the arithmetic in 64 "
            f"bits; the golden is the file above, so the two differ by the rounded weights and by the golden's own rounding. The bound is "
            f"$\\tau_{{\\mathrm{{par}}}} = (N_{{\\mathrm{{rnd}}}} + 1)\\,2^{{-24}}$ on the relative error of the logits, where $N_{{\\mathrm{{rnd}}}}$ counts the roundings along the longest chain of dependent "
            "operations by the rule of Chapter 2 (a sum of $n$ products costs $n$, each further operation one; the stage counts are in the table above, "
            "and the bound of each tap in the second table), and the one that is added is the golden's storage. "
            "The tests derived from that one bound are rel_l2 at most the bound, max_abs at most the bound times the length of the golden, and 1 - cosine at "
            "most half the bound squared.\n\n"
            + _table(["stack", "$N_{\\mathrm{rnd}}$", "bound $\\tau_{\\mathrm{par}}$", "rel_l2 of logits", "max_abs of logits", "all three tests pass", "bound over rel_l2"], rows)
            + f"\n\nThe dense stack against the decoder of Chapter 1 (`modern_decoder.forward`, same weights, 64 bits both): rel_l2 {_below(rel, 1e-14)}.")


def planted_bugs():
    cases = (("dense", "group_map", 1, "query head h reads key-value head h mod 2, not h // 2"),
             ("dense", "rotate_keys_only", 1, "queries are not rotated"),
             ("dense", "no_qk_norm_gain", 1, "the QK-norm gains are not applied"),
             ("swa", "window_off_by_one", 1, "the window holds 5 keys, not 4"),
             ("moe", "no_renorm", 1, "the combine weights are the raw probabilities"),
             ("gdn", "gate_sign", 1, "the decay gate is +softplus, so the state grows"),
             ("hybrid", "group_map", 1, "as above, in the dense layer of the hybrid"),
             ("dense", "final_gain_doubled", None, "the final norm gain is doubled: every logit is twice as large"),
             ("dense", "weight_1e-9", None, "the weight in row 0, column 0 of W_Q ($[16, 16]$) of layer 0, after the rounding to 32 bits, is increased by 1e-9"),
             ("dense", "weight_1e-3", None, "the same weight is increased by 1e-3"))
    rows = []
    for arm, bug, layer, what in cases:
        blob = golden_blob(arm)
        p = r32_params(block_model(arm))
        engine_bug = bug
        if bug == "final_gain_doubled":
            p["gn"] = [2 * x for x in p["gn"]]
            engine_bug = None
        elif bug.startswith("weight_"):
            p["layers"][0]["Wq"][0][0] += float(bug[len("weight_"):])
            engine_bug = None
        res = check_against_golden(p, arm, blob, engine_bug, layer)
        t = verdict_of(res[-1], _scale(blob))
        first = next((r[0] for r in res if r[2] > r[4]), "none")
        lg = res[-1]
        rows.append((f"`{arm}`", f"`{bug}`" + (f" in layer {layer}" if layer is not None else ""), _sci(lg[2], 2), _sci(lg[3], 2),
                     " ".join("caught" if not ok else "-" for ok in t), first))
    return ("Mistakes planted in a stack that otherwise passes (the candidate of the previous table), compared with its golden. Each row "
            "lists the relative error of the logits, 1 - cosine, which of the three tests (max_abs, rel_l2, 1 - cosine) failed, and the "
            "first tap whose rel_l2 exceeded its bound: the embeddings, then layer 0, layer 1, then the logits. A planted mistake in layer 1 "
            "leaves the taps before it inside their bounds.\n\n"
            + "\n".join("- " + f"`{b}`: {w}" for _, b, _, w in cases) + "\n\n"
            + _table(["stack", "planted mistake", "rel_l2", "1 - cosine", "tests failed (max_abs, rel_l2, cosine)", "first tap beyond bound"], rows, "llrrll"))


# ------------------------------------------------------------------------------------------------ milestone 4: the controlled comparison

SELECT_SEED = "s0"
SEEDS = tuple(f"r{i}" for i in range(1, 7))
RATES = (0.01, 0.02, 0.04)
MEMO_PATH = None
CLEAR_RATIO = 1.1   # the runner-up's selection loss must exceed the best's by this factor for the choice to count as clear
LN_SYMBOLS = math.log(TASK["symbols"])


def reach_after(layers, window, t):
    """The positions that can influence position t after `layers` layers of a causal window of `window` keys."""
    reach = {t}
    for _ in range(layers):
        reach = {j for r in reach for j in range(max(0, r - window + 1), r + 1)}
    return reach


def swa_floor(layers, window):
    """Lower bound of the mean loss of a windowed stack of `layers` layers on the task: a prediction whose position cannot reach the
    header (position 0) can do no better than ln 6, one that can reach it may reach 0."""
    blind = sum(1 for t in PREDICT_AT if 0 not in reach_after(layers, window, t))
    return blind * LN_SYMBOLS / len(PREDICT_AT), blind


def task_statement():
    g = bookrng.generator(f"{NAME}.example")
    ex = [draw_sequence(g) for _ in range(3)]
    rows = []
    for ids in ex:
        h = ids[0] - TASK["symbols"]
        rows.append((" ".join(str(i) for i in ids), h, " ".join(str(ids[2 * i - 1]) for i in range(1, 7)), " ".join(str(ids[2 * i]) for i in range(1, 7))))
    floor, blind = swa_floor(len(ARMS["swa"]), CFG["WINDOW"])
    floor1, blind1 = swa_floor(1, CFG["WINDOW"])
    return ("The task needs two kinds of information. A sequence has 13 tokens: a header token $6 + h$ at position 0, with $h$ drawn from "
            "$\\{0, \\dots, 5\\}$, then six pairs $(x_i, y_i)$ in which $x_i$ is a symbol drawn from $\\{0, \\dots, 5\\}$ and $y_i = (x_i + h) \\bmod 6$. "
            f"Draws from `bookrng.generator(\"{NAME}.data.<seed>\")` (the example below: `{NAME}.example`, seed {_seed(NAME + '.example')}), seven per sequence in this "
            "order: `integer(6)` for $h$, then `integer(6)` for each of $x_1, \\dots, x_6$. The model reads the sequence and at the position of $x_i$ "
            "(positions 1, 3, 5, 7, 9, 11) predicts $y_i$; only these six predictions are trained and scored. The symbol $x_i$ is the previous token, "
            "the header is up to eleven positions back. A model that knows $x_i$ but not $h$ can do no better than $\\ln 6 = " + _fx(LN_SYMBOLS, 4) + "$ nats; "
            "one that finds the header reaches 0.\n\n"
            + _table(["sequence (ids)", "h", "x_1 .. x_6", "y_1 .. y_6"], rows, "lrll")
            + f"\n\nA window of 4 keys cannot see the header from the position of $x_i$ once $2i - 1 - 3 > 0$. After {len(ARMS['swa'])} layers a position "
            f"reaches back $2 \\cdot 3 = 6$ positions, so {blind} of the six predictions (those for $i = 4, 5, 6$) cannot see the header at all, and "
            f"a windowed stack's mean loss is at least {blind} / 6 of $\\ln 6$ = {_fx(floor, 4)} nats, even if it passes the header through the first layer to the second "
            f"(one layer alone: {blind1} of 6, {_fx(floor1, 4)}).")


HYPOTHESES = (
    ("H1", "swa", "dense", "ge", 0.5, "the window cannot see the header: swa - dense >= +0.5"),
    ("H2", "gdn", "dense", "le", 0.1, "the state can carry the header: gdn - dense <= +0.1"),
    ("H3", "moe", "dense", "ge", 0.05, "experts of width 6 are narrower than one of 24: moe - dense >= +0.05"),
    ("H4", "hybrid", "dense", "le", 0.1, "one attention layer is enough: hybrid - dense <= +0.1"),
    ("H5", "hybrid", "gdn", "le", 0.05, "the hybrid is no worse than the pure recurrent stack: hybrid - gdn <= +0.05"),
)


def registered():
    rows = [(n, claim) for n, _, _, _, _, claim in HYPOTHESES]
    return ("Predictions registered before the six runs of each arm were made, from the mechanisms and not from their outcome, each as a "
            "claim about a paired difference of the mean loss over the six scored predictions (nats), the first arm minus the second, and a "
            "margin. The verdict rule: 'supported' when the 95 percent interval lies wholly on the predicted side of the margin, 'refuted' when "
            "wholly on the other, 'undecided' otherwise.\n\n" + _table(["claim", "statement"], rows, "ll"))


@lru_cache(maxsize=None)
def results():
    """All runs, grouped as {(kind, arm, seed, rate): [base record, nudged records...]}. The selection runs (one seed, three rates,
    no nudge) and the main runs (six seeds at the selected rate, each with its four nudged copies) are spread over four processes."""
    import hashlib
    import json
    from pathlib import Path
    src = Path(__file__).read_text(encoding="utf-8")
    h = hashlib.sha256(src[:src.index("\n# ------------------------------------------------------------------------------------------------ milestone 1")].encode("utf-8"))
    h.update(repr((SEEDS, RATES, SELECT_SEED, RECIPE, CFG)).encode("utf-8"))
    for mod in (sl, stable, bookrng, tb, la, moe, tg, schedules, pos_mod):
        h.update(Path(mod.__file__).read_bytes())
    key = h.hexdigest()
    memo = Path(__file__).resolve().parents[3] / "build" / "book" / "project_architecture_comparison-results.json"
    if memo.is_file():
        try:
            data = json.loads(memo.read_text(encoding="utf-8"))
            if data.get("digest") == key:
                return data["results"]
        except (ValueError, KeyError):
            pass
    with get_context("fork").Pool(4) as pool:
        sel = pool.map(train, [(arm, SELECT_SEED, lr, 0.0) for arm in ARMS for lr in RATES], chunksize=1)
        best = {}
        for arm in ARMS:
            runs = [r for r in sel if r["job"][0] == arm]
            best[arm] = min(runs, key=lambda r: (r["loss"], r["job"][2]))["job"][2]
        main = pool.map(train, [(arm, sd, best[arm], nd) for arm in ARMS for sd in SEEDS for nd in (0.0, *stable.NUDGES)], chunksize=1)
    out = {"select": [dict(r, job=list(r["job"])) for r in sel], "best": best, "main": [dict(r, job=list(r["job"])) for r in main]}
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text(json.dumps({"digest": key, "results": out}), encoding="utf-8")
    return out


def selected_rate(arm):
    return results()["best"][arm]


def run_of(arm, seed, k=None):
    """The record of an arm and seed: the original run (k None) or its k-th nudged copy."""
    nd = 0.0 if k is None else stable.NUDGES[k]
    return next(r for r in results()["main"] if r["job"][0] == arm and r["job"][1] == seed and r["job"][3] == nd)


def values(arm, key="loss", k=None, index=None):
    out = []
    for sd in SEEDS:
        v = run_of(arm, sd, k)[key]
        out.append(v if index is None else v[index])
    return out


def stable_text(f, d=3, signed=False):
    """f(k) -> a float from the run (k None) or its k-th nudged copy; text by the stable-digits rule. When not even one digit survives
    but every run is small, the text says how small: 'below b' (or 'within b of zero' for a signed quantity) for the smallest b of
    0.05, 0.1, 0.2, 0.5 that bounds all five runs."""
    fmt = (lambda x: f"{x:+.{d}f}") if signed else (lambda x: _fx(x, d))
    vals = [f(None)] + [f(k) for k in range(len(stable.NUDGES))]
    t = stable.text(vals[0], vals[1:], fmt)
    if t != "unstable":
        return t
    for b in (0.05, 0.1, 0.2, 0.5):
        if all(abs(v) < b for v in vals):
            return f"within {b:g} of zero" if signed else f"below {b:g}"
    return t

# ------------------------------------------------------------------------------------------------ tables of the comparison


def _interval(lo, hi):
    if lo.startswith(("below", "within", "unstable")) or hi.startswith(("below", "within", "unstable")):
        return "digits of the ends do not survive"
    return f"{lo} to {hi}"


def selection():
    best = results()["best"]
    sel = results()["select"]
    rows = []
    for arm in ARMS:
        runs = {r["job"][2]: r["loss"] for r in sel if r["job"][0] == arm}
        ordered = sorted(runs.values())
        ratio = ordered[1] / ordered[0]
        rows.append((f"`{arm}`", f"{best[arm]:g}", f"{ratio:.1f}", "yes" if ratio > CLEAR_RATIO else "no"))
    return (f"Recipe: Adam with the moments of Volume III (0.9, 0.95, epsilon 1e-8, 1-based step in the bias corrections), no weight decay, "
            f"{RECIPE['steps']} steps numbered $t = 0, \\dots, {RECIPE['steps'] - 1}$ with the rate $\\eta_t = \\eta_{{\\max}}\\,(t + 1) / {RECIPE['warmup']}$ for "
            f"$t < {RECIPE['warmup']}$ (so step {RECIPE['warmup'] - 1} is the first at the peak) and "
            f"$\\eta_t = \\eta_{{\\max}}\\,(1 - 0.9\\,(t - {RECIPE['warmup']}) / {RECIPE['steps'] - RECIPE['warmup']})$ for $t \\ge {RECIPE['warmup']}$ "
            f"(a linear fall to a tenth of the peak at step {RECIPE['steps']}, which is never reached; step {RECIPE['steps'] - 1} uses "
            f"$1 - 0.9 \\cdot {RECIPE['steps'] - 1 - RECIPE['warmup']} / {RECIPE['steps'] - RECIPE['warmup']}$ of it), a clip of the batch gradient's global norm "
            f"(the square root of the sum of squares of every entry) at {RECIPE['clip']:g}: when the norm exceeds it the gradient is multiplied by clip over norm, then "
            f"{RECIPE['steps']} steps of {RECIPE['B']} sequences, each sequence the next seven draws of the data generator, the gradient being the mean over the "
            "batch. The loss that is differentiated is the mean over the six scored predictions plus, for a mixture, the balancing loss with weight 0.01; "
            "the 'training loss' columns below report the cross-entropy part alone (the balancing term is a regulariser of the "
            "router and is not a quality). Each arm chose its peak rate from "
            f"{', '.join(f'{r:g}' for r in RATES)} by the held-out cross-entropy on {RECIPE['n_eval']} sequences after one run of {RECIPE['steps']} steps on the selection seed (`{SELECT_SEED}`, "
            f"data from `{NAME}.data.{SELECT_SEED}`, held-out set from `{NAME}.eval.{SELECT_SEED}`, initial weights from `{NAME}.init.<arm>.{SELECT_SEED}`, "
            f"no nudge), a seed none of the reported runs uses; a tie goes to the lower rate. The ratio column is the runner-up's loss over the best loss, "
            f"to one digit (these are single runs that end at a few hundredths, whose further digits are not reproducible); the last column says whether it exceeds {CLEAR_RATIO:g}, "
            "so that the choice does not hinge on a small difference.\n\n"
            + _table(["arm", "peak rate chosen", "runner-up loss over best loss", f"ratio above {CLEAR_RATIO:g}"], rows, "lrrr"))


def arm_summary(arm, k=None):
    v = values(arm, "loss", k)
    mu, sd = sl.mean_sd(v)
    half = sl.ex._t95(len(v) - 1) * sd / math.sqrt(len(v))
    return mu, sd, mu - half, mu + half


def comparison():
    rows = []
    for arm in ARMS:
        mu = stable_text(lambda k, arm=arm: arm_summary(arm, k)[0], 3)
        sd = stable_text(lambda k, arm=arm: arm_summary(arm, k)[1], 3)
        lo = stable_text(lambda k, arm=arm: arm_summary(arm, k)[2], 3)
        hi = stable_text(lambda k, arm=arm: arm_summary(arm, k)[3], 3)
        tr_ = stable_text(lambda k, arm=arm: sum(values(arm, "train", k)) / len(SEEDS), 3)
        rows.append((f"`{arm}`", f"{selected_rate(arm):g}", _int(costs(ARMS[arm], CFG, 1)[0]), mu, sd, _interval(lo, hi), tr_))
    return (f"Held-out loss (mean cross-entropy of the six scored predictions, nats) after {RECIPE['steps']} steps, on the {len(SEEDS)} reported seeds "
            f"({', '.join(f'`{s}`' for s in SEEDS)}; data `{NAME}.data.<seed>`, held-out `{NAME}.eval.<seed>`, weights `{NAME}.init.<arm>.<seed>`, "
            f"the seed of a stream is a function of its full name by the rule of Volume I, Chapter 17: `{NAME}.data.r1` has the seed {_seed(f'{NAME}.data.r1')}, "
            f"`{NAME}.data.r2` {_seed(f'{NAME}.data.r2')}, and every other stream named here has the seed of its own name). All arms see the same sequences in the same order "
            "for a seed. Mean over seeds, sample standard deviation (divisor $n - 1$), the 95 percent interval of the mean (Student $t$ with "
            f"{len(SEEDS) - 1} degrees of freedom, multiplier {_fx(sl.ex._t95(len(SEEDS) - 1), 4)}), and the training loss: the batch cross-entropy of the last 10 steps (the balancing term excluded), averaged over the steps and the seeds. " + stable.RULE + "\n\n"
            + _table(["arm", "peak rate", "parameters", "mean", "sd", "95 percent interval of the mean", "training loss, last 10 steps"], rows))


def _verdict_text(a, b, side, margin):
    vs = []
    for k in [None, *range(len(stable.NUDGES))]:
        mu, sd, se, lo, hi = sl.paired_interval(values(a, "loss", k), values(b, "loss", k))
        vs.append(sl._verdict(lo, hi, side, margin))
    return vs[0] if all(v == vs[0] for v in vs) else "unstable"


def contrasts():
    rows = []
    for name, a, b, side, margin, claim in HYPOTHESES:
        f = lambda k, a=a, b=b, i=0: sl.paired_interval(values(a, "loss", k), values(b, "loss", k))[i]
        mu = stable_text(lambda k, f=f: f(k, i=0), 3, signed=True)
        lo = stable_text(lambda k, f=f: f(k, i=3), 3, signed=True)
        hi = stable_text(lambda k, f=f: f(k, i=4), 3, signed=True)
        rows.append((name, f"`{a}` - `{b}`", f"{side} {margin:+g}", mu, _interval(lo, hi), _verdict_text(a, b, side, margin)))
    return ("The registered contrasts, paired by seed (every arm of a seed shares the sequences and the held-out set): the mean of the six "
            "differences first arm minus second, and the 95 percent t interval of that mean (5 degrees of freedom, multiplier "
            f"{_fx(sl.ex._t95(len(SEEDS) - 1), 4)}). A negative difference favours the first arm. 'ge' claims the difference is at least the "
            "margin, 'le' at most. A verdict that differs between a run and its four nudged copies would print 'unstable'.\n\n"
            + _table(["claim", "contrast", "predicted", "mean difference", "95 percent interval", "verdict"], rows, "lllrrl"))


def by_position():
    arms = ("dense", "swa", "gdn", "moe", "hybrid")
    rows = []
    for i in range(len(PREDICT_AT)):
        row = [f"$y_{i + 1}$ (position {PREDICT_AT[i]}, header {PREDICT_AT[i]} back)"]
        for arm in arms:
            row.append(stable_text(lambda k, arm=arm, i=i: sum(values(arm, "per", k, i)) / len(SEEDS), 2))
        rows.append(row)
    return ("Held-out loss of each of the six predictions, mean over the six seeds (nats). A prediction at position $t = 2i - 1$ has the header "
            f"$t$ positions back; the bound for a window of 4 keys in two layers is $t \\le 6$.\n\n"
            + _table(["prediction", *[f"`{a}`" for a in arms]], rows, "l" + "r" * len(arms)))


def moe_balance():
    busiest = [stable_text(lambda k, l=l: sum(run_of("moe", sd, k)["busiest"][l] for sd in SEEDS) / len(SEEDS), 2) for l in range(2)]
    return ("Share of the expert slots taken by the busiest of the 4 experts on the first 30 sequences of the held-out stream "
            f"`{NAME}.eval.<seed>` (the first 30 draws of sequences, all 13 positions of each, so 390 tokens and 780 slots per layer), mean over the six seeds, "
            f"for the two mixture layers of the `moe` arm; a slot is one (token, chosen expert) pair and the share is the largest of the four counts over their sum: layer 0 {busiest[0]}, layer 1 {busiest[1]} (1 / 4 = 0.25 when the router is perfectly balanced, 0.5 when "
            "two experts take everything). The balancing loss has weight 0.01.")


def cost_axis(arm):
    return costs(ARMS[arm], CFG, REF_TOKENS)[2]


def quality_data():
    """[(arm, decode FLOPs of the 13th token, mean loss, lo, hi)] for the figure."""
    out = []
    for arm in ARMS:
        mu, _, lo, hi = arm_summary(arm)
        out.append((arm, cost_axis(arm), mu, lo, hi))
    return out


def hybrid_tradeoff():
    rows = []
    for arm in ("dense", "gdn", "hybrid"):
        mu = stable_text(lambda k, arm=arm: arm_summary(arm, k)[0], 3)
        _, _, d13, c13, s13 = costs(ARMS[arm], CFG, REF_TOKENS)
        _, _, d4k, c4k, s4k = costs(ARMS[arm], CFG, 4096)
        rows.append((f"`{arm}`", mu, _int(d13), _int(c13 + s13), _int(d4k), _int(c4k + s4k)))
    return ("The hybrid beside the two pure stacks it is made of (two layers each; the hybrid has the gated delta layer first and the dense layer second). "
            "Quality is the held-out loss of the previous table; costs are the ledger's, at the task's 13 tokens and at 4,096 tokens by the same formulas.\n\n"
            + _table(["stack", "held-out loss", "decode FLOPs at 13", "bytes held at 13", "decode FLOPs at 4,096", "bytes held at 4,096"], rows))


def acceptance_fingerprint():
    rows = []
    gd = gradient_checks_summary()
    rows.append(("1", "parameters of dense, moe, gdn, swa, hybrid", " / ".join(_int(parameter_count(block_model(a))) for a in ARMS), "exact"))
    rows.append(("1", "gradient check, largest difference over every tensor of the five stacks", gd, "criterion"))
    rows.append(("1", "chunked against recurrent, chunk sizes 1 to 12; with the state dropped", "all within tolerance; 3.490", "tolerance; 3 digits"))
    for arm in ("dense", "moe", "gdn", "swa", "hybrid"):
        P, n_act, dec, cache, state = costs(ARMS[arm], CFG, REF_TOKENS)
        rows.append(("2", f"`{arm}`: active parameters; prefill FLOPs per token (rounded down); decode FLOPs; bytes held", f"{_int(n_act)}; {_int(prefill_flops(ARMS[arm], CFG, REF_TOKENS) // REF_TOKENS)}; {_int(dec)}; {_int(cache + state)}", "exact"))
    for arm in ("dense", "gdn"):
        rows.append(("3", f"golden digest, `{arm}`", digest(golden_blob(arm)), "exact"))
    return _table(["milestone", "quantity", "value", "tolerance"], rows, "lllr")


def gradient_checks_summary():
    ids = ref_ids()
    worst = 0.0
    for arm in ARMS:
        specs, p = ARMS[arm], block_model(arm)
        _, _, g, _ = loss_grad(p, CFG, specs, ids, PREDICT_AT)
        gl = {path: r for path, r in leaves(g)}
        by_path = {path: r for path, r in leaves(p)}
        for path, i, j in coordinates(f"{NAME}.gradcheck.{arm}", p):
            r = by_path[path]
            old = r[i][j]
            r[i][j] = old + CHECK_STEP
            hi = _task_loss(p, specs, ids)
            r[i][j] = old - CHECK_STEP
            lo = _task_loss(p, specs, ids)
            r[i][j] = old
            worst = max(worst, abs((hi - lo) / (2 * CHECK_STEP) - gl[path][i][j]))
    return _below(worst, CRITERION)
