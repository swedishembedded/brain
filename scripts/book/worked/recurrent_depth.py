# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Looped and Recurrent-Depth Transformers": one block applied several times.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. The block, its backward pass and the two-hop task are those of Volume V, Chapter 9
(`transformer_block.block_forward`, `block_backward`, `two_hop_example`, imported, not copied); what this module adds is
the loop around them:

* the parameters and the FLOPs of a stack of distinct blocks, of one block applied several times and of a mixture layer, and
  of the two released models whose crates the chapter reads (counted from their shapes);
* the key-value cache of a looped and of a recurrent-depth model, in bytes;
* a one-block decoder applied two times, trained on the two-hop task and evaluated with one to eight loops;
* the exit distribution of Ouro, by hand on four passes, with its gradient checked against finite differences, and the same
  objective trained on the toy (three passes, a gate on the final-normed state of the last position);
* a state that is pulled towards the input at every step (the injection of the recurrent-depth design), iterated past the range.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums of
products are plain left-to-right additions (never Python's `sum`, which may compensate). Numbers of trained models are
printed by the stable-digits rule of `stable.py`.

Swedish Embedded AB implements solutions for weight-shared and adaptive-depth language models. If your team needs expertise in
looped transformers and their serving cost, you can procure our services by sending an email to info@swedishembedded.com.
"""
import hashlib
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import bookrng
import mixture_of_experts as moe
import stable
import transformer_block as tb

# ------------------------------------------------------------------------------------------------ formatting


def _fx(v, d=3):
    """Fixed decimals, rounded half away from zero (the book's round rule) on the exact binary value, so only a true tie moves."""
    out = str(Decimal(v).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)) if math.isfinite(v) else f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _int(n):
    return f"{n:,}"


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


def _msum(xs):
    """Left-to-right sum, the order a C loop would use."""
    total = 0.0
    for x in xs:
        total += x
    return total


# ------------------------------------------------------------------------------------------------ counting: stack, loop, mixture

D_OURO, F_OURO, H_OURO, L_OURO, V_OURO, PASSES_OURO = 2048, 5632, 16, 48, 49152, 4
CONTEXT = 2048   # the cached tokens the attention of a decoded token reads, in the FLOP counts


def block_weight_flops(D, F, T):
    """FLOPs of one block for one decoded token that reads T cached tokens: 2 per weight of the four attention matrices and the
    three SwiGLU matrices, and 4 D T for the scores and the weighted sum of the values (norms, softmax and adds left out)."""
    attn_w, ffn_w, _ = tb.block_parameters(D, "swiglu", F)
    return 2 * (attn_w + ffn_w) + 4 * D * T


def stack_or_loop():
    """Four blocks three ways, and a mixture layer: parameters against FLOPs per token."""
    D, F = D_OURO, F_OURO
    attn_w, ffn_w, norms = tb.block_parameters(D, "swiglu", F)
    one = attn_w + ffn_w + norms
    f_one = block_weight_flops(D, F, CONTEXT)
    n = 4
    # A mixture layer in place of the feed-forward network: E experts of the same hidden width, k used per token.
    e, k = 32, 2
    p_exp = moe.expert_params(D, F)
    moe_p = attn_w + norms + e * p_exp + D * e
    moe_f = 2 * attn_w + 4 * D * CONTEXT + 2 * (k * p_exp + D * e)
    rows = [
        ("one block", 1, one, f_one),
        (f"{n} distinct blocks", n, n * one, n * f_one),
        (f"one block applied {n} times", n, one, n * f_one),
        (f"one block, feed-forward network replaced by {e} experts, {k} used", 1, moe_p, moe_f),
    ]
    table = _table(["layer or stack", "block applications per token", "parameters", "FLOPs per token", "FLOPs per parameter"],
                   [(a, b, _int(c), _int(d), _fx(d / c, 2)) for a, b, c, d in rows])
    return (f"A block of width D = {D}, 16 heads of width 128 (the attention matrices are four {D} by {D}), SwiGLU hidden width F = {F}, "
            f"two RMS norm gains, no biases: attention {_int(attn_w)}, feed-forward network {_int(ffn_w)}, gains {_int(norms)} parameters. "
            f"FLOPs of one decoded token that reads {CONTEXT} cached tokens: 2 per weight of the seven matrices, plus 4 D T = "
            f"{_int(4 * D * CONTEXT)} for the scores and the weighted sum of the values (norms, softmax and adds left out). The mixture "
            f"row keeps the attention and replaces the three feed-forward matrices by {e} experts of the same shape and a router "
            f"of {D} by {e}; a token uses {k} experts. Exact integers.\n\n" + table)


def params_flops_data():
    """Series of the figure: (label, [(FLOPs per token in millions, parameters in millions)]) for one feed-forward layer of the
    chapter on mixtures (D = 2048): the dense layer widened (distinct layers added), the same dense layer applied more times,
    and the mixture with more experts. The starting point is a dense layer of hidden width 4096, which is 8 experts' width."""
    d, f_dense = moe.D_Q, 4096
    p0, f0 = moe.layer_costs(d, f_dense, 1, 1)
    distinct = [(n * f0 / 1e6, n * p0 / 1e6) for n in (1, 2, 4, 8, 16)]
    looped = [(n * f0 / 1e6, p0 / 1e6) for n in (1, 2, 4, 8, 16)]
    mixture = []
    for e in (8, 16, 32, 64, 128, 256):
        p, fl = moe.layer_costs(d, moe.F_Q, e, 8)
        mixture.append((fl / 1e6, p / 1e6))
    return [("n distinct dense layers (n = 1 to 16)", distinct), ("one dense layer applied n times (n = 1 to 16)", looped),
            ("one mixture layer, 8 experts of 8 used, then up to 256 experts", mixture)]


def ouro_layer_params(D, F):
    """Parameters of one Ouro layer (brain's `param_list`): four RMS norm gains, wq, wk, wv, wo (D by D each as 16 heads of
    128 give D), the SwiGLU gate, up and down matrices."""
    return 4 * D + 4 * D * D + 3 * D * F


def ouro_released():
    """The released Ouro-2.6B-Thinking shape counted from the tensor list the config declares: the total must be the
    2,667,974,657 of the checkpoint header."""
    D, F, L, V, T = D_OURO, F_OURO, L_OURO, V_OURO, PASSES_OURO
    layer = ouro_layer_params(D, F)
    embed, head = V * D, V * D
    final_norm, gate = D, D + 1
    total = embed + L * layer + final_norm + gate + head
    assert total == 2_667_974_657
    f_layer = 2 * (4 * D * D + 3 * D * F)
    rows = []
    for label, layers, passes in (("the released model: 48 layers, 4 passes", L, T), ("the same layers, 1 pass", L, 1),
                                  ("192 distinct layers (the compute of 4 passes)", L * T, 1)):
        params = embed + head + final_norm + gate + layers * layer
        flops = layers * passes * f_layer + 2 * V * D
        rows.append((label, layers, layers * passes, _int(params), _fx(flops / 1e9, 2), _fx(flops / params, 2)))
    return ("Counted from the tensors the config declares (embedding 49,152 by 2,048; per layer four gains, the four attention matrices, "
            "gate, up and down; final norm; the exit gate's 2,048 weights and one bias; an untied head). The total of the first row "
            f"is {_int(total)}, the number in the checkpoint's header. FLOPs per decoded token: 2 per weight of the layers' seven matrices "
            "for every application, plus 2 V D for the head; the scores and softmax are left out, so the row is a lower bound.\n\n"
            + _table(["model", "distinct layers", "layer applications per token", "parameters", "GFLOPs per token", "FLOPs per parameter"],
                     rows))


HUGINN = {"V": 65536, "D": 5280, "F": 17920, "heads": 55, "prelude": 2, "core": 4, "coda": 2, "mean_recurrence": 32, "block_size": 4096}


def huginn_block_params(D, F):
    """One block of huginn-0125: four norm gains, the fused QKV (3 D by D), the output matrix (D by D), the Q and K bias vectors
    (D each), the fused gate-and-up matrix (2 F by D) and the projection (D by F)."""
    return 4 * D + 3 * D * D + D * D + 2 * D + 2 * F * D + F * D


def huginn_released():
    """huginn-0125 counted from its tensor list, and the layers a token passes through as the number of core steps r grows."""
    h = HUGINN
    D, F, V = h["D"], h["F"], h["V"]
    blk = huginn_block_params(D, F)
    n_blocks = h["prelude"] + h["core"] + h["coda"]
    adapter, embed, ln_f = D * 2 * D, V * D, D
    total = embed + ln_f + adapter + n_blocks * blk
    f_blk = 2 * (4 * D * D + 3 * D * F)
    rows = []
    for r in (1, 4, 8, 16, 32, 64):
        eff = h["prelude"] + h["core"] * r + h["coda"]
        flops = (h["prelude"] + h["core"] * r + h["coda"]) * f_blk + r * 2 * adapter + 2 * V * D
        rows.append((r, eff, _fx(flops / 1e9, 1), _fx(flops / total, 2)))
    return (f"Counted from the tensors the config declares: embedding (tied to the head) {_int(embed)}, adapter {D} by {2 * D} = {_int(adapter)}, "
            f"final norm {D}, and {n_blocks} blocks of {_int(blk)} parameters each (four gains, fused QKV, output matrix, two bias vectors, "
            f"fused gate-and-up, projection), in all {_int(total)} parameters, whatever r is. One decoded token costs, per core step, the "
            f"adapter and the {h['core']} core blocks; the prelude and the coda run once; the head costs 2 V D once. FLOPs count 2 per weight "
            "of the seven matrices of every block, 2 per weight of the adapter and 2 V D for the head (scores left out).\n\n"
            + _table(["core steps r", "layers a token passes through", "GFLOPs per token", "FLOPs per parameter"], rows))


def kv_cache_bytes():
    """Bytes of cached keys and values per token and for a full context, for the layouts of the two crates."""
    D, L, T = D_OURO, L_OURO, PASSES_OURO
    h = HUGINN
    ctx = 4096
    stages = h["prelude"] + h["core"] * h["mean_recurrence"] + h["coda"]
    ring = 8
    planes = h["prelude"] + ring + h["coda"]
    # (layout, caches, numbers per token = 2 (key and value) x key width x caches)
    layouts = [
        ("48 layers, 1 pass", L, 2 * D * L),
        ("Ouro, 4 passes", L * T, 2 * D * L * T),
        ("Ouro, one cache per layer for all 4 passes", L, 2 * D * L),
        (f"Huginn, 32 core steps", stages, 2 * h["D"] * stages),
        (f"Huginn, ring of {ring} core histories", planes, 2 * h["D"] * planes),
    ]
    rows = [(a, b, _int(2 * n), _int(2 * n * ctx), _int(4 * n * ctx)) for a, b, n in layouts]
    return ("Per token a layer stores a key and a value, each of the model's key width: 2,048 numbers for Ouro "
            "(16 heads of 128, no sharing between heads) and 5,280 for huginn-0125 (55 heads of 96). The columns on the right are a full "
            f"context of {_int(ctx)} tokens (the positional capacity of huginn-0125) at 2 bytes per number (bf16) and at 4 bytes per number "
            "(the float32 the two engines of brain keep their caches in). Exact integers.\n\n"
            + _table(["layout", "caches", "bytes per token (2-byte)", "4,096 tokens (2-byte)", "4,096 tokens (4-byte)"], rows))


# ------------------------------------------------------------------------------------------------ the exit distribution by hand

EXIT_Z = (-1.0, 0.0, 1.0)             # gate logits of passes 1 .. 3 (the gate of the last pass is never used)
EXIT_CE = (2.0, 1.2, 0.9, 0.8)        # cross-entropy of the head after each of the four passes, in nats
EXIT_QS = (0.3, 0.5, 0.9, 1.0)


def exit_probs(zs, total):
    """Exit distribution over `total` passes from the gate logits of the first total - 1: q_t = lambda_t prod_{j<t}(1 - lambda_j),
    the last pass takes the remaining mass. Returns (lambdas, q)."""
    lam = [tb.sigmoid(z) for z in zs]
    rem, q = 1.0, []
    for t in range(total):
        if t + 1 < total:
            q.append(lam[t] * rem)
            rem *= 1.0 - lam[t]
        else:
            q.append(rem)
    return lam, q


def exit_loss(zs, ces):
    _, q = exit_probs(zs, len(ces))
    return _msum([a * b for a, b in zip(q, ces)])


def exit_grad(zs, ces):
    """dL/dz_t of L = sum_j q_j c_j by the product form: dL/dlambda_t = c_t prod_{k<t}(1 - lambda_k) - sum_{j>t} c_j q'_j where
    q'_j is the product of the other survival factors times lambda_j (1 for the last pass), then times lambda_t (1 - lambda_t)."""
    total = len(ces)
    lam = [tb.sigmoid(z) for z in zs]
    out = []
    for t in range(total - 1):
        d = 0.0
        for j in range(t, total):
            prod = 1.0
            for k in range(j):
                if k != t:
                    prod *= 1.0 - lam[k]
            coef = 1.0 if j == t else (lam[j] if j + 1 < total else 1.0)
            term = ces[j] * prod * coef
            d += term if j == t else -term
        out.append(d * lam[t] * (1.0 - lam[t]))
    return out


def exit_rule(q, threshold):
    """The first pass (0-based) whose cumulative exit probability reaches the threshold; the last pass when none does."""
    cdf = 0.0
    for t, v in enumerate(q):
        cdf += v
        if cdf >= threshold:
            return t
    return len(q) - 1


def exit_by_hand():
    zs, ces = EXIT_Z, EXIT_CE
    total = len(ces)
    lam, q = exit_probs(zs, total)
    rem, cdf, rows = 1.0, 0.0, []
    for t in range(total):
        rem_before = rem
        if t + 1 < total:
            rem *= 1.0 - lam[t]
        cdf += q[t]
        rows.append((t + 1, _fx(zs[t], 1) if t + 1 < total else "unused", _fx(lam[t], 4) if t + 1 < total else "unused", _fx(rem_before, 4),
                     _fx(q[t], 4), _fx(cdf, 4), _fx(ces[t], 1)))
    loss = exit_loss(zs, ces)
    mean_passes = _msum([(t + 1) * q[t] for t in range(total)])
    rule = [(f"{th}", exit_rule(q, th) + 1) for th in EXIT_QS]
    grad = exit_grad(zs, ces)
    h = 1e-6
    worst = 0.0
    for t in range(total - 1):
        zp, zm = list(zs), list(zs)
        zp[t] += h
        zm[t] -= h
        fd = (exit_loss(zp, ces) - exit_loss(zm, ces)) / (2 * h)
        worst = max(worst, abs(fd - grad[t]))
    assert worst < 1e-6
    return ("Four passes. Gate logits z of passes 1 to 3 and the cross-entropy c of the head after each pass are given. "
            "lambda = sigmoid(z); q_t = lambda_t times the mass remaining; the last pass takes what remains. Tolerance: 4 decimals.\n\n"
            + _table(["pass t", "gate logit z", "lambda", "mass remaining before t", "exit probability q", "cumulative", "c"], rows)
            + f"\n\nThe weighted loss sum q_t c_t = {_fx(loss, 4)} nats. The expected number of passes sum t q_t = {_fx(mean_passes, 4)} of {total}. "
            "The first pass whose cumulative probability reaches the threshold: "
            + "; ".join(f"threshold {a}: pass {b}" for a, b in rule) + ". The gradient of the loss with respect to z_1, z_2, z_3 by the "
            f"product form is ({', '.join(_fx(g, 4) for g in grad)}); it agrees with central differences at step 1e-6 to below 1e-6.")


# ------------------------------------------------------------------------------------------------ the toy: a decoder of one block

D_T, H_T, F_T, N_KEYS = 16, 2, 24, 4
STEPS_LOOP, STEPS_EXIT, BATCH, LR = 600, 800, 16, 0.01
LOOPS_TRAINED, PASSES_EXIT, TEST_LOOPS = 2, 3, 8
BETA1, BETA2, EPS_ADAM, N_TEST, WINDOW = 0.9, 0.999, 1e-8, 400, 20
GATE_BIAS0, GATE_SCALE = -1.5, 0.1
CHECKPOINTS = (100, 200, 400)
NAME_LOOP, NAME_EXIT, NAME_TEST = "recurrent_depth.loop", "recurrent_depth.exit", "recurrent_depth.test"


def init_loop(name, with_gate):
    """Draws from `bookrng.generator(name)`, in this order: the embedding table E (20 by 16, every entry `normal()`), the one
    block (W_Q, W_K, W_V, W_O of 16 by 16, W_gate, W_up of 16 by 24, W_down of 24 by 16, each entry `normal() / sqrt(rows)`),
    the head W_C (16 by 4, `normal() / sqrt(16)`), and for the exit model the gate weights (16 entries `0.1 normal()`). The gains
    start at 1; the gate bias starts at -1.5."""
    rng = bookrng.generator(name)
    p = {"E": tb._normal_matrix(rng, tb.VOCAB, D_T, 1.0)}
    p["b"] = tb.init_block(rng, D_T, F_T, 1.0)
    p["Wc"] = tb._normal_matrix(rng, D_T, N_KEYS, 1.0 / math.sqrt(D_T))
    p["gf"] = [[1.0] * D_T]
    if with_gate:
        p["gw"] = [[GATE_SCALE * rng.normal() for _ in range(D_T)]]
        p["gb"] = [[GATE_BIAS0]]
    return p


def tensors(p):
    out = [p["E"]] + [p["b"][k] for k in tb.BLOCK_KEYS] + [p["gf"], p["Wc"]]
    if "gw" in p:
        out += [p["gw"], p["gb"]]
    return out


def _add_into(acc, g):
    for k in tb.BLOCK_KEYS:
        for ra, rb in zip(acc[k], g[k]):
            for j in range(len(ra)):
                ra[j] += rb[j]


def _softmax_ce(logits, y):
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    z = _msum(e)
    pr = [v / z for v in e]
    return -math.log(pr[y]), pr


def loop_forward(p, ids, loops):
    """The decoder with its one block applied `loops` times to the stream (starting from the embeddings of the ids), then the
    gain-only RMS norm and the head on the last position. Returns (logits, caches, streams, normed, norm cache) where
    streams[k] is the stream after k + 1 loops."""
    x = [list(p["E"][t]) for t in ids]
    caches, streams = [], []
    for _ in range(loops):
        x, c = tb.block_forward(p["b"], x, "pre", H_T)
        caches.append(c)
        streams.append(x)
    xn, cf = tb.rms_forward(x, p["gf"][0])
    return tb.matmul([xn[-1]], p["Wc"])[0], caches, streams, xn, cf


def head_after(p, x):
    """The logits of the last position for a stream x: the gain-only RMS norm, then the head."""
    xn, _ = tb.rms_forward(x, p["gf"][0])
    return tb.matmul([xn[-1]], p["Wc"])[0]


def loop_loss_grad(p, ids, y, loops, bug=None):
    """(loss, gradients in the order of `tensors`): the softmax cross-entropy of the last position after `loops` applications of
    the one block; the block's gradient is the sum over the uses."""
    logits, caches, _, xn, cf = loop_forward(p, ids, loops)
    loss, pr = _softmax_ce(logits, y)
    dl = [pr[c] - (1.0 if c == y else 0.0) for c in range(N_KEYS)]
    d = D_T
    dxn = [[0.0] * d for _ in ids]
    dxn[-1] = tb.matmul([dl], tb.transpose(p["Wc"]))[0]
    dWc = tb.matmul(tb.transpose([xn[-1]]), [dl])
    dx, dgf = tb.rms_backward(dxn, cf)
    acc = None
    for c in reversed(caches):
        dx, g = tb.block_backward(c, dx)
        if acc is None:
            acc = g
        elif bug != "last_use_only":   # the wrong backward: a later use overwrites, it does not add to, the earlier one
            _add_into(acc, g)
    dE = [[0.0] * d for _ in range(tb.VOCAB)]
    for t, tok in enumerate(ids):
        for j in range(d):
            dE[tok][j] += dx[t][j]
    return loss, [dE] + [acc[k] for k in tb.BLOCK_KEYS] + [[dgf], dWc]


def exit_forward(p, ids, passes):
    """Ouro's wiring on the toy: pass 1 starts from the embeddings, pass t > 1 from the final-normed output of pass t - 1; each
    pass ends in the gain-only RMS norm (shared), the head on the last position and the gate z = w . h + b of the last position.
    Returns, per pass, (block cache, norm cache, normed rows, gate logit, head logits)."""
    xin = [list(p["E"][t]) for t in ids]
    out = []
    for _ in range(passes):
        xo, c = tb.block_forward(p["b"], xin, "pre", H_T)
        hs, cf = tb.rms_forward(xo, p["gf"][0])
        z = _msum([a * b for a, b in zip(hs[-1], p["gw"][0])]) + p["gb"][0][0]
        out.append((c, cf, hs, z, tb.matmul([hs[-1]], p["Wc"])[0]))
        xin = hs
    return out


def exit_loss_grad(p, ids, y, passes, bug=None):
    """(loss, gradients in the order of `tensors`) of the released objective on the toy: sum over passes of q_t times the
    cross-entropy of the head after pass t, the exit distribution q from the gate as in `exit_probs`. The gradient reaches
    the head, the shared norm, the block (summed over the passes), the embeddings and the gate."""
    steps = exit_forward(p, ids, passes)
    d = D_T
    zs = [s[3] for s in steps[:-1]]
    _, q = exit_probs(zs, passes)
    ces, prs = [], []
    for s in steps:
        c, pr = _softmax_ce(s[4], y)
        ces.append(c)
        prs.append(pr)
    loss = _msum([a * b for a, b in zip(q, ces)])
    dz = exit_grad(zs, ces) + [0.0]
    acc, dgf, dWc, dgw, dgb = None, [0.0] * d, [[0.0] * N_KEYS for _ in range(d)], [[0.0] * d], [[0.0]]
    carry = [[0.0] * d for _ in ids]
    for t in reversed(range(passes)):
        c, cf, hs, _, _ = steps[t]
        dl = [q[t] * (prs[t][k] - (1.0 if k == y else 0.0)) for k in range(N_KEYS)]
        dhs = [list(r) for r in carry]
        dlast = tb.matmul([dl], tb.transpose(p["Wc"]))[0]
        for j in range(d):
            dhs[-1][j] += dlast[j] + dz[t] * p["gw"][0][j]
        for a in range(d):
            for b in range(N_KEYS):
                dWc[a][b] += hs[-1][a] * dl[b]
            dgw[0][a] += dz[t] * hs[-1][a]
        dgb[0][0] += dz[t]
        dxo, dg = tb.rms_backward(dhs, cf)
        for j in range(d):
            dgf[j] += dg[j]
        carry, g = tb.block_backward(c, dxo)
        if acc is None:
            acc = g
        elif bug != "last_use_only":
            _add_into(acc, g)
    dE = [[0.0] * d for _ in range(tb.VOCAB)]
    for t, tok in enumerate(ids):
        for j in range(d):
            dE[tok][j] += carry[t][j]
    return loss, [dE] + [acc[k] for k in tb.BLOCK_KEYS] + [[dgf], dWc, dgw, dgb]


def _train(p, lossf, data_name, steps):
    """Adam (no weight decay, no clipping, no warmup) on every tensor, batches of 16 consecutive examples of the stream
    `data_name`, the gradient of a step the mean of the 16 example gradients; returns the per-step mean losses."""
    ts = tensors(p)
    m = [[[0.0] * len(r) for r in t] for t in ts]
    v = [[[0.0] * len(r) for r in t] for t in ts]
    rng = bookrng.generator(data_name)
    losses = []
    for s in range(1, steps + 1):
        acc, total = None, 0.0
        for _ in range(BATCH):
            ids, y = tb.two_hop_example(rng)
            loss, grads = lossf(p, ids, y)
            total += loss / BATCH
            if acc is None:
                acc = [[[x / BATCH for x in r] for r in t] for t in grads]
            else:
                for a, t in zip(acc, grads):
                    for ra, rt in zip(a, t):
                        for j, x in enumerate(rt):
                            ra[j] += x / BATCH
        losses.append(total)
        for ti, t in enumerate(ts):
            for ri, r in enumerate(t):
                for j in range(len(r)):
                    gv = acc[ti][ri][j]
                    m[ti][ri][j] = BETA1 * m[ti][ri][j] + (1 - BETA1) * gv
                    v[ti][ri][j] = BETA2 * v[ti][ri][j] + (1 - BETA2) * gv * gv
                    r[j] -= LR * (m[ti][ri][j] / (1 - BETA1 ** s)) / (math.sqrt(v[ti][ri][j] / (1 - BETA2 ** s)) + EPS_ADAM)
    return losses


def _argmax(logits):
    return max(range(N_KEYS), key=lambda c: (logits[c], -c))


def _fn_of(ids):
    f = {tok // N_KEYS: tok % N_KEYS for tok in ids[:-1]}
    return f, ids[-1] - N_KEYS * N_KEYS


def _rms_all(x):
    return math.sqrt(_msum([v * v for r in x for v in r]) / (len(x) * len(x[0])))


def _distance_rel(a, b):
    num = _msum([(u - w) ** 2 for ra, rb in zip(a, b) for u, w in zip(ra, rb)])
    den = _msum([w * w for rb in b for w in rb])
    return math.sqrt(num / den)


def _job(spec):
    kind, nudge = spec
    rng_t = bookrng.generator(NAME_TEST)
    if kind == "loop":
        p = init_loop(f"{NAME_LOOP}.init", False)
        p["E"][0][0] += nudge
        losses = _train(p, lambda pp, ids, y: loop_loss_grad(pp, ids, y, LOOPS_TRAINED), f"{NAME_LOOP}.data", STEPS_LOOP)
        right, change, rms, step = [0] * TEST_LOOPS, [0.0] * TEST_LOOPS, [0.0] * TEST_LOOPS, [0.0] * TEST_LOOPS
        for _ in range(N_TEST):
            ids, y = tb.two_hop_example(rng_t)
            _, _, streams, _, _ = loop_forward(p, ids, TEST_LOOPS)
            prev = [list(p["E"][t]) for t in ids]
            for k in range(1, TEST_LOOPS + 1):
                right[k - 1] += 1 if _argmax(head_after(p, streams[k - 1])) == y else 0
                change[k - 1] += _distance_rel(streams[k - 1], prev) / N_TEST
                rms[k - 1] += _rms_all(streams[k - 1]) / N_TEST
                step[k - 1] += _rms_all([[u - w for u, w in zip(ra, rb)] for ra, rb in zip(streams[k - 1], prev)]) / N_TEST
                prev = streams[k - 1]
        return {"kind": kind, "nudge": nudge, "curve": {c: _msum(losses[c - WINDOW:c]) / WINDOW for c in CHECKPOINTS},
                "right": right, "change": change, "rms": rms, "step": step}
    p = init_loop(f"{NAME_EXIT}.init", True)
    p["E"][0][0] += nudge
    losses = _train(p, lambda pp, ids, y: exit_loss_grad(pp, ids, y, PASSES_EXIT), f"{NAME_EXIT}.data", STEPS_EXIT)
    n_p = PASSES_EXIT
    right, ce, qsum = [0] * n_p, [0.0] * n_p, [0.0] * n_p
    easy_n, easy_q, hard_q = 0, [0.0] * n_p, [0.0] * n_p
    rule = {th: [0, 0.0] for th in EXIT_QS_TOY}   # threshold -> [right, passes run]
    for _ in range(N_TEST):
        ids, y = tb.two_hop_example(rng_t)
        steps = exit_forward(p, ids, n_p)
        _, q = exit_probs([s[3] for s in steps[:-1]], n_p)
        for t in range(n_p):
            right[t] += 1 if _argmax(steps[t][4]) == y else 0
            ce[t] += _softmax_ce(steps[t][4], y)[0] / N_TEST
            qsum[t] += q[t] / N_TEST
        f, x = _fn_of(ids)
        easy = f[f[x]] == f[x]   # one hop already gives the answer
        easy_n += 1 if easy else 0
        for t in range(n_p):
            (easy_q if easy else hard_q)[t] += q[t]
        for th in EXIT_QS_TOY:
            t_exit = exit_rule(q, th)
            rule[th][0] += 1 if _argmax(steps[t_exit][4]) == y else 0
            rule[th][1] += t_exit + 1
    hard_n = N_TEST - easy_n
    return {"kind": kind, "nudge": nudge, "curve": {c: _msum(losses[c - WINDOW:c]) / WINDOW for c in CHECKPOINTS},
            "right": right, "ce": ce, "q": qsum, "easy_n": easy_n, "easy_q": [v / easy_n for v in easy_q],
            "hard_q": [v / hard_n for v in hard_q], "rule": {th: (r / N_TEST, passes / N_TEST) for th, (r, passes) in rule.items()}}


EXIT_QS_TOY = (0.3, 0.5, 0.8, 0.95)


# The ten trainings take minutes, and the tables and the figure of the chapter need the same numbers, so the results are kept in a file
# of the build directory under a digest of every source they depend on; a changed module, or any module it imports, recomputes them.
MEMO = Path(__file__).resolve().parents[3] / "build" / "book" / "recurrent_depth-results.json"


def _digest():
    h = hashlib.sha256()
    for mod in (None, tb, tb.sa, stable, bookrng):
        path = Path(__file__ if mod is None else mod.__file__)
        h.update(path.read_bytes())
    return h.hexdigest()


def _encode(res):
    out = {}
    for kind, (base, twins) in res.items():
        runs = []
        for r in [base, *twins]:
            r = dict(r)
            r["curve"] = {str(k): v for k, v in r["curve"].items()}
            if "rule" in r:
                r["rule"] = {str(k): list(v) for k, v in r["rule"].items()}
            runs.append(r)
        out[kind] = runs
    return out


def _decode(raw):
    out = {}
    for kind, runs in raw.items():
        dec = []
        for r in runs:
            r = dict(r)
            r["curve"] = {int(k): v for k, v in r["curve"].items()}
            if "rule" in r:
                r["rule"] = {float(k): tuple(v) for k, v in r["rule"].items()}
            dec.append(r)
        out[kind] = [dec[0], dec[1:]]
    return out


@lru_cache(maxsize=None)
def results():
    """{kind: (result, [nudged results])}; the ten trainings are spread over four worker processes, and kept in MEMO."""
    digest = _digest()
    if MEMO.is_file():
        try:
            memo = json.loads(MEMO.read_text(encoding="utf-8"))
            if memo.get("digest") == digest:
                return _decode(memo["results"])
        except (ValueError, KeyError):
            pass
    jobs = [(k, d) for k in ("exit", "loop") for d in (0.0, *stable.NUDGES)]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    out = {}
    for r in res:
        entry = out.setdefault(r["kind"], [None, []])
        if r["nudge"] == 0.0:
            entry[0] = r
        else:
            entry[1].append(r)
    MEMO.parent.mkdir(parents=True, exist_ok=True)
    MEMO.write_text(json.dumps({"digest": digest, "results": _encode(out)}), encoding="utf-8")
    return out


def _cell(kind, f, digits=3):
    base, twins = results()[kind]
    return stable.text(f(base), [f(t) for t in twins], lambda x: _fx(x, digits))


def _span(kind, f, digits=3):
    """The interval, over the run and its four copies, that contains every value, widened outward to a grid of 10^-digits.
    A chaotic run has no reproducible value, only a scatter; the outward rounding keeps only the digits on which the copies lie
    together, and the claims under the table (not the interval) are what another implementation must reproduce."""
    base, twins = results()[kind]
    vals = [f(base)] + [f(t) for t in twins]
    scale = 10 ** digits
    lo = math.floor(min(vals) * scale + 1e-9) / scale
    hi = math.ceil(max(vals) * scale - 1e-9) / scale
    return f"{lo:.{digits}f} to {hi:.{digits}f}"


def _count(kind, pred):
    """In how many of the five copies (the run and its four nudged twins) the claim `pred(result)` holds, as 'k of 5'."""
    base, twins = results()[kind]
    runs = [base, *twins]
    return f"{sum(1 for r in runs if pred(r))} of {len(runs)}"


# ------------------------------------------------------------------------------------------------ the task as a pointer chase


def chase_reference():
    """For k = 1 .. 8: the fraction of the 256 functions times 4 queries for which f^k(x) equals f^2(x), by enumeration: the accuracy
    against the two-hop target of a predictor that follows the pointer exactly k times."""
    n = N_KEYS
    out = []
    for k in range(1, TEST_LOOPS + 1):
        agree = 0
        for code in range(n ** n):
            f = [(code // n ** a) % n for a in range(n)]
            for x in range(n):
                u = x
                for _ in range(k):
                    u = f[u]
                agree += 1 if u == f[f[x]] else 0
        out.append(agree / (n ** n * n))
    return out


def one_hop_enough():
    """The share of (function, query) pairs for which f(x) is a fixed point of f, so that one hop already gives f(f(x))."""
    n = N_KEYS
    hit = 0
    for code in range(n ** n):
        f = [(code // n ** a) % n for a in range(n)]
        for x in range(n):
            hit += 1 if f[f[x]] == f[x] else 0
    return hit / (n ** n * n)


def loop_setup():
    rng = bookrng.generator(f"{NAME_LOOP}.init")
    p = init_loop(f"{NAME_LOOP}.init", False)
    params_block = sum(len(r) for k in tb.BLOCK_KEYS for r in p["b"][k])
    params_all = sum(len(r) for t in tensors(p) for r in t)
    del rng
    chase = chase_reference()
    return (f"The task is the two-hop task of Volume V, Chapter 9 (`transformer_block.two_hop_example`, drawn exactly as there): four pairs "
            f"a -> f(a) of a random function on {{0, 1, 2, 3}} in shuffled order and a query x, the target f(f(x)); ids 0 .. 15 are the pair tokens, "
            f"ids 16 .. 19 the query tokens, T = 5, no position information. The decoder is that chapter's with ONE block: embedding table E [20, {D_T}], "
            f"the pre-norm block of width {D_T} with {H_T} heads and SwiGLU hidden width {F_T} (causal attention), a gain-only final RMS norm "
            f"(epsilon 1e-6) and a head W_C [{D_T}, 4] on the last position. The block is applied {LOOPS_TRAINED} times in training and the loss is the "
            f"cross-entropy of the last position after the second application. Parameters: {_int(params_block)} in the block, {_int(params_all)} in all "
            f"(a decoder with two distinct blocks of Chapter 9 has 4,816). Initialization draws, in this order from one generator: the embedding table E "
            f"(20 by {D_T}, row by row, every entry `normal()`), then W_Q, W_K, W_V, W_O ({D_T} by {D_T}), W_gate, W_up ({D_T} by {F_T}) and W_down ({F_T} by {D_T}), row by row, "
            f"each entry `normal() / sqrt(rows)`, then W_C ({D_T} by 4, `normal() / sqrt({D_T})`); both gains and the final gain start at 1. A predictor that knows the pointer and follows it exactly k times is right "
            f"with probability {', '.join(_fx(v, 4) for v in chase)} for k = 1 .. {TEST_LOOPS} (enumerated over the 256 functions and 4 queries); "
            f"in {_fx(one_hop_enough() * 100, 1)} percent of the pairs one hop already gives the answer, because f(x) is a fixed point.")


def loop_accuracy_data():
    """Series of the figure: (label, [(loops, accuracy)]) with the trained model and the exact pointer chaser."""
    base, _ = results()["loop"]
    model = [(k, base["right"][k - 1] / N_TEST) for k in range(1, TEST_LOOPS + 1)]
    chase = [(k, v) for k, v in enumerate(chase_reference(), start=1)]
    return model, chase


def loop_training():
    rows = []
    chase = chase_reference()
    for k in range(1, TEST_LOOPS + 1):
        acc = _span("loop", lambda r, k=k: r["right"][k - 1] / N_TEST, 2)
        chg = _cell("loop", lambda r, k=k: r["change"][k - 1], 3)
        rms = _cell("loop", lambda r, k=k: r["rms"][k - 1], 2)
        stp = _cell("loop", lambda r, k=k: r["step"][k - 1], 2)
        rows.append((k, acc, _fx(chase[k - 1], 3), chg, rms, stp))
    curve = [_cell("loop", lambda r, c=c: r["curve"][c], 3) for c in CHECKPOINTS]
    text = (f"Initialization: draws from `{NAME_LOOP}.init` ({_seed(NAME_LOOP + '.init')}) in the order of the setup table. Training: the examples of the stream "
            f"`{NAME_LOOP}.data` ({_seed(NAME_LOOP + '.data')}), step s of {STEPS_LOOP} takes the examples 16 (s - 1) to 16 s - 1; the gradient of a step "
            f"is the mean of the 16 example gradients, with the block's gradient the sum over its two uses; Adam on every tensor at rate {LR}, "
            f"beta 0.9 and 0.999, epsilon 1e-8, no weight decay, clipping or warmup. Mean training loss over the 20 steps up to step 100, 200, 400: "
            f"{curve[0]}, {curve[1]}, {curve[2]}. Test: the first {N_TEST} examples of `{NAME_TEST}` ({_seed(NAME_TEST)}), the largest logit "
            "(lowest id on a tie) against the target, with k = 1 .. 8 applications of the block. The relative change is the mean over the test "
            "examples of ||x_k - x_(k-1)|| / ||x_(k-1)|| for the stream matrices [5, 16] (x_0 the embeddings), the RMS of the stream the mean over the "
            "test examples of the root mean square of the 80 entries of x_k of that example (an average of per-example RMS values, not the RMS of "
            "all entries pooled) and the RMS of the change the same average for the entries of x_k - x_(k-1). " + stable.RULE + " The test accuracy of this run is chaotic in its second digit: where the five values differ, "
            "the cell of the accuracy column gives the hundredths that contain the values of all five copies, written 'a to b', and the claims below the table "
            "(not the interval) are what another implementation must reproduce.\n\n")
    claims = [
        ("accuracy is highest at k = 2, the number of applications it was trained with", _count("loop", lambda r: max(range(TEST_LOOPS), key=lambda k: r["right"][k]) == 1)),
        ("accuracy at k = 2 is above 0.9", _count("loop", lambda r: r["right"][1] / N_TEST > 0.9)),
        ("accuracy at k = 1 is below 0.7", _count("loop", lambda r: r["right"][0] / N_TEST < 0.7)),
        ("accuracy falls at every k from 3 to 8", _count("loop", lambda r: all(r["right"][k] < r["right"][k - 1] for k in range(2, TEST_LOOPS)))),
        ("accuracy at k = 8 is at least 0.1 below accuracy at k = 2", _count("loop", lambda r: r["right"][1] - r["right"][7] >= 0.1 * N_TEST)),
        ("the relative change of the stream falls at every k", _count("loop", lambda r: all(r["change"][k] < r["change"][k - 1] for k in range(1, TEST_LOOPS)))),
        ("the RMS of the stream grows at every k", _count("loop", lambda r: all(r["rms"][k] > r["rms"][k - 1] for k in range(1, TEST_LOOPS)))),
    ]
    return (text + _table(["loops k", "test accuracy", "exact pointer follower", "relative change", "RMS of the stream", "RMS of the change"], rows)
            + "\n\nWhat holds in every copy:\n\n" + _table(["claim", "copies in which it holds"], claims, "lr"))


def exit_training():
    n_p = PASSES_EXIT
    rows = []
    for t in range(n_p):
        rows.append((t + 1, _span("exit", lambda r, t=t: r["right"][t] / N_TEST, 1), _span("exit", lambda r, t=t: r["ce"][t], 1),
                     _span("exit", lambda r, t=t: r["q"][t], 1), _span("exit", lambda r, t=t: r["easy_q"][t], 1),
                     _span("exit", lambda r, t=t: r["hard_q"][t], 1)))
    easy = results()["exit"][0]["easy_n"]
    rule_rows = [(th, _span("exit", lambda r, th=th: r["rule"][th][1], 1), _span("exit", lambda r, th=th: r["rule"][th][0], 1)) for th in EXIT_QS_TOY]
    curve = [_cell("exit", lambda r, c=c: r["curve"][c], 3) for c in CHECKPOINTS]
    text = (f"The same task and decoder with the objective of Ouro: {PASSES_EXIT} passes of the one block, pass t > 1 starting from the final-normed "
            f"output of pass t - 1, each pass ending in the shared gain-only RMS norm, the head on the last position and the gate "
            f"z = w . h + b of the last position (w [16] drawn as 0.1 normal() after W_C, b = {GATE_BIAS0}); q from the gates of passes 1 and 2 as in "
            f"the hand example; the loss of an example is sum q_t c_t. Initialization from `{NAME_EXIT}.init` ({_seed(NAME_EXIT + '.init')}), "
            f"the data stream `{NAME_EXIT}.data` ({_seed(NAME_EXIT + '.data')}), {STEPS_EXIT} steps, the optimizer and the test examples of the previous "
            f"table. Mean training loss over the 20 steps up to step 100, 200, 400: {curve[0]}, {curve[1]}, {curve[2]}. Of the {N_TEST} test examples, "
            f"{easy} have a fixed point at f(x) ('one hop enough'); the last two columns are the mean exit probability of that group and of the rest. "
            + stable.RULE + " The test numbers of this run scatter in their first digit (the copies differ by far more than the nudge: the run is "
            "chaotic after about 400 steps), and the cells of the tables are the tenths that contain the values of all five, written 'a to b'; they are "
            "a guide to the size, not numbers to reproduce. What another implementation must reproduce are the claims below the tables, which "
            "hold in every copy with a margin.\n\n")
    text += _table(["pass t", "accuracy of the head after pass t", "cross-entropy (nats)", "mean exit probability q_t", "q_t, one hop enough",
                    "q_t, the rest"], rows)
    text += ("\n\nThe exit rule: leave at the first pass whose cumulative exit probability reaches the threshold (the last pass when none does).\n\n"
             + _table(["threshold", "mean passes run (of 3)", "accuracy at the exit pass"], rule_rows))
    claims = [
        ("the head after pass 1 is right on fewer than 0.6 of the examples", _count("exit", lambda r: r["right"][0] / N_TEST < 0.6)),
        ("the heads after passes 2 and 3 are right on more than 0.9", _count("exit", lambda r: min(r["right"][1:]) / N_TEST > 0.9)),
        ("most of the exit probability is on the last pass (q_3 above 0.5)", _count("exit", lambda r: r["q"][2] > 0.5)),
        ("q_1 of the 'one hop enough' group is at least 3 times q_1 of the rest", _count("exit", lambda r: r["easy_q"][0] >= 3 * r["hard_q"][0])),
        ("at threshold 0.5 fewer than 2.9 passes are run on average", _count("exit", lambda r: r["rule"][0.5][1] < 2.9)),
        ("at threshold 0.5 accuracy is within 0.03 of the accuracy of the last pass", _count("exit", lambda r: r["rule"][0.5][0] >= r["right"][2] / N_TEST - 0.03)),
    ]
    return text + "\n\nWhat holds in every copy:\n\n" + _table(["claim", "copies in which it holds"], claims, "lr")


# ------------------------------------------------------------------------------------------------ an injected state forgets where it started


INJ_GAINS = (0.5, 0.9, 1.0, 1.1)
INJ_INPUT, INJ_STARTS = 1.0, (-3.0, 0.0, 5.0)
INJ_STEPS = (0, 1, 2, 4, 8, 16, 32)


def injection_iterates():
    """The scalar recurrence s <- a s + (1 - a) u_in, a pull towards the injected input u_in = 1, from three starting states, for four
    gains a; the same recurrence with the input removed, s <- a s, for the last column. Both are the simplest maps of the form
    s <- g(s, input) that a recurrent-depth model applies; the gain a is the factor by which one step shrinks a difference."""
    rows = []
    for a in INJ_GAINS:
        spreads = []
        for steps in (4, 16, 32):
            finals = []
            for s0 in INJ_STARTS:
                s = s0
                for _ in range(steps):
                    s = a * s + (1 - a) * INJ_INPUT
                finals.append(s)
            spreads.append(max(finals) - min(finals))
        with_input, without = 0.0, 0.0
        for _ in range(32):
            with_input = a * with_input + (1 - a) * INJ_INPUT
            without = a * without
        rows.append((a, *[f"{c:.3e}" for c in spreads], f"{with_input:.4g}", f"{without:.4g}"))
    return ("Starting states s_0 = -3, 0 and 5; input u_in = 1. For each gain a the table gives the spread (largest minus smallest) of the "
            "three states after 4, 16 and 32 steps of s <- a s + (1 - a) u_in (the spread of the recurrence without the input is the same, "
            "because a difference is multiplied by a either way), and the state reached from s_0 = 0 after 32 steps with the input and with "
            "the input removed. Tolerance: four significant digits.\n\n"
            + _table(["gain a", "spread after 4", "spread after 16", "spread after 32", "from 0 after 32, with input", "from 0 after 32, no input"],
                     rows, "rrrrrr"))


# ------------------------------------------------------------------------------------------------ gradient check of a shared block

def _entries(t):
    rows, cols = len(t), len(t[0])
    return sorted({(0, 0), (rows // 2, cols // 2), (rows - 1, cols - 1)})


def _worst(p, lossf, grads):
    ts = tensors(p)
    h, worst = 1e-5, 0.0
    for ti, t in enumerate(ts):
        for r, c in _entries(t):
            old = t[r][c]
            t[r][c] = old + h
            lp = lossf(p)
            t[r][c] = old - h
            lm = lossf(p)
            t[r][c] = old
            a, n = grads[ti][r][c], (lp - lm) / (2 * h)
            worst = max(worst, abs(a - n) / max(abs(a), abs(n), 1e-3))
    return worst


def shared_gradient_check():
    """Central differences (step 1e-5) of the loss of one example on three entries of every tensor (the first, the middle and the last
    entry), against the analytic gradient that sums the uses of the shared block, and against the wrong gradient that keeps only the
    last use. The largest relative error |a - n| / max(|a|, |n|, 1e-3) is reported against its criterion."""
    rng = bookrng.generator("recurrent_depth.check.example")
    ids, y = tb.two_hop_example(rng)
    rows = []
    for label, with_gate, passes, lossgrad in (("loop: block applied 2 times, loss after the last", False, 2, loop_loss_grad),
                                               ("exit: 3 passes, loss weighted by q", True, 3, exit_loss_grad)):
        p = init_loop(f"recurrent_depth.check.{'exit' if with_gate else 'loop'}", with_gate)
        loss_only = lambda pp, lg=lossgrad, n=passes: lg(pp, ids, y, n)[0]
        good = lossgrad(p, ids, y, passes)[1]
        bad = lossgrad(p, ids, y, passes, bug="last_use_only")[1]
        n_ent = sum(len(_entries(t)) for t in tensors(p))
        rows.append((label, n_ent, "below 1e-6" if _worst(p, loss_only, good) < 1e-6 else "FAILS", f"{_worst(p, loss_only, bad):.2f}"))
    return ("One example of the two-hop task from `recurrent_depth.check.example` (" + _seed("recurrent_depth.check.example") + "), parameters from "
            "`recurrent_depth.check.loop` and `recurrent_depth.check.exit`, three entries of every tensor (the first, the middle and the last). "
            "Conventions. Parameters: drawn as in the training initialization (the embedding table E [20, 16], each entry `normal()`; then W_Q, W_K, "
            "W_V, W_O [16, 16], W_gate, W_up [16, 24] and W_down [24, 16], each row by row with `normal() / sqrt(rows)`; then W_C [16, 4] with "
            "`normal() / sqrt(16)`; for the exit model last the gate weights, 16 values of 0.1 `normal()`), the gains 1, the gate bias -1.5, from the "
            "two generators named above (one per model). A tensor is a matrix [rows, cols] (a gain or gate-weight vector is one row; the bias is [1, 1]); "
            "the entries checked are (0, 0), (rows // 2, cols // 2) and (rows - 1, cols - 1) with duplicates counted once, so a vector of 16 gives its "
            "entries 0, 8 and 15 and the bias one entry. Numerical derivative: (L(w + h) - L(w - h)) / (2h) with h = 1e-5 on the loss of that one example, "
            "in double precision. Relative error: |a - n| / max(|a|, |n|, 1e-3) with a the analytic and n the numerical derivative (the 1e-3 keeps a "
            "gradient that is zero from dividing by zero); the table gives the largest over the entries. 'Last use only' is the wrong backward that "
            "keeps, for each weight of the block, the gradient of the last application alone (the first one the backward pass meets) and "
            "discards the contributions of the earlier uses, while the gradient with respect to the input of each use still flows on to the earlier "
            "ones as usual. Criterion: the largest relative error below 1e-6.\n\n"
            + _table(["model", "entries checked", "largest relative error, sum over the uses", "same, last use only"], rows, "lrrr"))
