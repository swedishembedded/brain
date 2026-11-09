# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Mixture of Experts": how a model can hold many parameters and use few of them per token.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. Each public function without arguments returns a markdown string that the chapter embeds between
`<!-- worked: mixture_of_experts.<name> -->` markers; the `*_data` functions supply the numbers of the chapter's figures, so a
picture and the table beside it cannot disagree.

* the parameters and FLOPs per token of a dense feed-forward network and of mixture-of-experts layers of the same expert width,
  and the parameters of Qwen3.5-35B-A3B counted in total and per token;
* one token through a router by hand: softmax scores, top-k, renormalisation, the weighted sum of expert outputs, the shared
  expert, and the sigmoid router with a selection bias;
* the dispatch of a batch: histogram, exclusive scan and stable emission of the expert-major order, the experts run on their
  tokens and combined, compared with evaluating every expert and masking;
* an untrained router on a batch whose tokens share a component, and the capacity factor that drops the overflow;
* the load-balancing loss, its value on three loads and its gradient checked against central differences;
* a tiny mixture of experts trained with three strengths of the balancing loss (the stable-digits rule of `stable.py`);
* the bytes that expert parallelism exchanges, and the bytes a decode step reads as the batch grows.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints the seed. Sums of products
are plain left-to-right additions, the order a C loop would use (never Python's `sum`, which may compensate).

Swedish Embedded AB implements solutions for serving sparse language models within a memory budget. If your team needs expertise
in mixture-of-experts inference and training, you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import stable

# ------------------------------------------------------------------------------------------------ formatting


def _int(n):
    return f"{n:,}"


def _fx(v, d=4):
    out = f"{v:.{d}f}"
    return out[1:] if out.startswith("-") and not any(c in "123456789" for c in out) else out


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed_line(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`)."


def _vec(v, d=3):
    return "(" + ", ".join(_fx(x, d) for x in v) + ")"


# ------------------------------------------------------------------------------------------------ arithmetic


def _dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def _silu(z):
    return z * _sigmoid(z)


def _silu_slope(z):
    s = _sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


def _softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = 0.0
    for v in e:
        s += v
    return [v / s for v in e]


def _top_k(scores, k):
    """The k indices of the largest scores, largest first; a tie goes to the lower index (a stable descending argsort)."""
    return sorted(range(len(scores)), key=lambda e: (-scores[e], e))[:k]


def _max_abs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _vecmat(x, W):
    """x W for x of length m and W of shape [m, n]: entry j is the sum over i = 0 .. m-1 of x[i] W[i][j], left to right."""
    out = []
    for j in range(len(W[0])):
        s = 0.0
        for i in range(len(x)):
            s += x[i] * W[i][j]
        out.append(s)
    return out


def _expert(x, w):
    """SwiGLU feed-forward network of one expert: (silu(x Wgate) * (x Wup)) Wdown, with w = (Wgate, Wup, Wdown)."""
    gate_pre = _vecmat(x, w[0])
    up = _vecmat(x, w[1])
    h = [_silu(a) * b for a, b in zip(gate_pre, up)]
    return _vecmat(h, w[2])


def route(logits, k, scoring="softmax", bias=None, scale=1.0, renorm=True):
    """The router's decision for one token: (selected experts in rank order, their combine weights in the same order).

    softmax: scores are softmax(logits), selection by score. sigmoid: scores are sigmoid(logits); selection is by score plus the
    bias (if any), but the combine weight comes from the score alone. renorm divides the chosen weights by their sum."""
    e = len(logits)
    scores = _softmax(logits) if scoring == "softmax" else [_sigmoid(z) for z in logits]
    choice = scores if bias is None else [scores[i] + bias[i] for i in range(e)]
    chosen = _top_k(choice, k)
    total = 0.0
    for i in chosen:
        total += scores[i]
    if renorm:
        w = [scale * scores[i] / total for i in chosen]
    else:
        w = [scale * scores[i] for i in chosen]
    return chosen, w


# ------------------------------------------------------------------------------------------------ capacity without compute

# The expert of the large configurations: SwiGLU with model width D = 2,048 and hidden width F = 512 (Qwen3.5-35B-A3B).
D_Q, F_Q, E_Q, K_Q, LAYERS_Q = 2048, 512, 256, 8, 40


def expert_params(d, f):
    return 3 * d * f


def layer_costs(d, f, e, k, shared=0):
    """(parameters, FLOPs per token) of a mixture-of-experts feed-forward layer: e experts of hidden width f, k of them used
    per token, `shared` always-used experts, and a router of d x e weights when e > 1. A dense layer is e = 1, k = 1."""
    p_exp = expert_params(d, f)
    router_p = d * e if e > 1 else 0
    params = e * p_exp + shared * p_exp + router_p
    flops = 2 * (k * p_exp + shared * p_exp + router_p)
    return params, flops


def layer_counts_data():
    """Rows (label, params, flops, ratio) of the dense-versus-sparse table."""
    d, f = D_Q, F_Q
    rows = []
    for label, (e, k, sh), ff in (
        ("dense, hidden width 512", (1, 1, 0), 512),
        ("dense, hidden width 4,096 (8 experts' width)", (1, 1, 0), 4096),
        ("dense, hidden width 131,072 (256 experts' width)", (1, 1, 0), 131072),
        ("mixture, 256 experts, 8 per token", (256, 8, 0), 512),
        ("mixture, 256 experts, 8 per token, 1 shared", (256, 8, 1), 512),
    ):
        p, fl = layer_costs(d, ff, e, k, sh)
        rows.append((label, p, fl))
    return rows


def layer_counts():
    rows = []
    for label, p, fl in layer_counts_data():
        rows.append([label, _int(p), _int(fl), _fx(p / fl * 2, 2)])
    return ("Model width D = 2,048; an expert is a SwiGLU network with three matrices, 3 D F parameters and 6 D F FLOPs per token "
            "(2 FLOPs per parameter that takes part in a product). The router of a mixture is a D by E matrix, 2 D E FLOPs per "
            "token. 'Parameters per used parameter' is the total parameters divided by the parameters a token uses (FLOPs "
            "divided by 2).\n\n"
            + _table(["layer", "parameters", "FLOPs per token", "parameters per used parameter"], rows))


# Layer parameter counts of Qwen3.5-35B-A3B that are inputs here (the token mixers are Chapters 1, 3 and 5's subject):
# 30 linear-attention layers and 10 full-attention layers, vocabulary 248,320, untied embedding and output tables.
LINEAR_ATTN_PARAMS = 33_718_464
FULL_ATTN_PARAMS = 27_263_488
V_Q = 248_320


def qwen_counts_data():
    d, f, e, k, n_layers = D_Q, F_Q, E_Q, K_Q, LAYERS_Q
    p_exp = expert_params(d, f)
    moe_total = e * d + e * p_exp + p_exp + d          # router, routed experts, shared expert, shared expert's gate vector
    moe_used = e * d + k * p_exp + p_exp + d           # the router is read in full; 8 routed experts; the shared expert
    mixers = 30 * LINEAR_ATTN_PARAMS + 10 * FULL_ATTN_PARAMS
    norms = n_layers * 2 * d + d
    embed_in = V_Q * d
    head = V_Q * d
    total = mixers + norms + embed_in + head + n_layers * moe_total
    used = mixers + norms + head + n_layers * moe_used  # a lookup of the input table is not a product
    return {"p_exp": p_exp, "moe_total": moe_total, "moe_used": moe_used, "mixers": mixers, "norms": norms, "embed_in": embed_in,
            "head": head, "total": total, "used": used}


def qwen_counts():
    c = qwen_counts_data()
    n = LAYERS_Q
    rows = [
        ["token mixers (30 linear-attention layers of 33,718,464, 10 full-attention layers of 27,263,488)", _int(c["mixers"]), _int(c["mixers"])],
        ["norm gains (2 per layer and the final one)", _int(c["norms"]), _int(c["norms"])],
        ["input embedding table (248,320 x 2,048)", _int(c["embed_in"]), "0"],
        ["output matrix (2,048 x 248,320)", _int(c["head"]), _int(c["head"])],
        [f"routers ({n} x 256 x 2,048 and {n} shared-expert gate vectors of 2,048)", _int(n * (E_Q * D_Q + D_Q)), _int(n * (E_Q * D_Q + D_Q))],
        [f"shared experts ({n} x {_int(c['p_exp'])})", _int(n * c["p_exp"]), _int(n * c["p_exp"])],
        [f"routed experts ({n} x 256 x {_int(c['p_exp'])}; a token uses 8 of the 256)", _int(n * E_Q * c["p_exp"]), _int(n * K_Q * c["p_exp"])],
        ["total", _int(c["total"]), _int(c["used"])],
    ]
    return ("Parameters of Qwen3.5-35B-A3B by group, and how many of them take part in the products of one token. The expert "
            "has D = 2,048 and F = 512, so 3,145,728 parameters; there are 256 per layer and 8 are used. The two token-mixer counts "
            "are inputs: a linear-attention layer is 8,192 x 2,048 + 4,096 x 2,048 + 2 x 32 x 2,048 + 8,192 x 4 + 2 x 32 + 128 + "
            "2,048 x 4,096 parameters, a full-attention layer 8,192 x 2,048 + 2 x 512 x 2,048 + 2 x 256 + 2,048 x 4,096.\n\n"
            + _table(["group", "parameters", "used per token"], rows)
            + f"\n\nTotal over used: {_fx(c['total'] / c['used'], 2)}. In billions: {c['total'] / 1e9:.2f} and {c['used'] / 1e9:.2f}.")


def params_flops_data():
    """Series for the figure: (label, [(active FLOPs per token in millions, total parameters in millions)]) for one layer."""
    d, f = D_Q, F_Q
    dense = []
    for ff in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072):
        p, fl = layer_costs(d, ff, 1, 1)
        dense.append((fl / 1e6, p / 1e6))
    series = [("dense (hidden width 512 to 131,072)", dense)]
    for k in (2, 8):
        pts = []
        for e in (8, 16, 32, 64, 128, 256):
            p, fl = layer_costs(d, f, e, k)
            pts.append((fl / 1e6, p / 1e6))
        series.append((f"mixture, {k} of E experts used (E = 8 to 256)", pts))
    return series


# ------------------------------------------------------------------------------------------------ one token by hand

HAND_X = [1.0, -1.0, 0.5]
HAND_ROUTER = [[0.8, 0.1, -0.2], [-0.4, 0.9, 0.3], [0.5, -0.6, 0.7], [0.2, 0.2, 0.2]]   # row e is the router's column for expert e
HAND_K = 2
# Width-1 SwiGLU experts, so that every number can be printed: (gate column, up column, down row), each of length D = 3.
HAND_EXPERTS = [
    ([1.0, 0.0, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, -1.0]),
    ([0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ([0.5, -0.5, 0.5], [1.0, 1.0, 1.0], [-1.0, 0.5, 1.0]),
    ([1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]),
]
HAND_SHARED = ([0.2, 0.4, -0.2], [1.0, 0.0, 1.0], [0.5, -0.5, 0.5])
HAND_SHARED_GATE = [0.3, 0.3, 0.6]
HAND_BIAS = [-0.3, 0.0, 0.0, 0.45]


def _hand_expert(x, w):
    """One width-1 expert: a = x . gate column, u = x . up column, h = silu(a) u, output h times the down row."""
    a, u = _dot(x, w[0]), _dot(x, w[1])
    h = _silu(a) * u
    return a, u, h, [h * c for c in w[2]]


def router_scores():
    logits = [_dot(HAND_X, r) for r in HAND_ROUTER]
    p = _softmax(logits)
    chosen, w = route(logits, HAND_K)
    zsel = _softmax([logits[i] for i in chosen])
    rows = [[str(e), _vec(HAND_ROUTER[e], 1), _fx(logits[e]), _fx(math.exp(logits[e] - max(logits))), _fx(p[e]),
             f"rank {chosen.index(e) + 1}" if e in chosen else "-"] for e in range(4)]
    tot = sum(p[i] for i in chosen)
    assert _max_abs(w, zsel) < 1e-12
    return ("Token x = (1.0, -1.0, 0.5), model width D = 3, E = 4 experts, k = 2. The router has one weight vector per expert (the "
            "columns of $\\mathbf{W}_{\\mathrm{r}}$, written as rows here); the logit of expert $e$ is the dot product $x \\cdot r_e$. "
            "'exp(z - max)' is the numerator of the stable softmax of Volume III, Chapter 3.\n\n"
            + _table(["e", "router vector $r_e$", "logit z", "exp(z - max)", "softmax p", "chosen"], rows)
            + f"\n\nThe two chosen experts are {chosen[0]} and {chosen[1]}, with probabilities {_fx(p[chosen[0]])} and "
            f"{_fx(p[chosen[1]])}, which sum to {_fx(tot)}. Dividing each by that sum gives the combine weights "
            f"({_fx(w[0])}, {_fx(w[1])}). The softmax of the two chosen logits alone is ({_fx(zsel[0])}, {_fx(zsel[1])}); the two "
            f"agree to below 1e-12 in absolute value, as the algebra of the chapter says they must.")


def combine_by_hand():
    logits = [_dot(HAND_X, r) for r in HAND_ROUTER]
    chosen, w = route(logits, HAND_K)
    rows, y = [], [0.0, 0.0, 0.0]
    for e, we in zip(chosen, w):
        a, u, h, out = _hand_expert(HAND_X, HAND_EXPERTS[e])
        rows.append([str(e), _fx(a), _fx(u), _fx(_silu(a)), _fx(h), _vec(out), _fx(we), _vec([we * o for o in out])])
        for i in range(3):
            y[i] += we * out[i]
    a, u, h, sh_out = _hand_expert(HAND_X, HAND_SHARED)
    s_logit = _dot(HAND_X, HAND_SHARED_GATE)
    s_gate = _sigmoid(s_logit)
    unweighted = [y[i] + sh_out[i] for i in range(3)]
    gated = [y[i] + s_gate * sh_out[i] for i in range(3)]
    return ("The experts of the example have hidden width 1: expert $e$ has a gate column, an up column and a down row of 3 numbers "
            "each (gate columns (1,0,1), (0,1,0), (0.5,-0.5,0.5), (1,1,1); up columns (0.5,0,1), (1,0,1), (1,1,1), (0,1,0); down "
            "rows (1,0,-1), (0,1,0), (-1,0.5,1), (0.5,0.5,0.5), for experts 0 to 3). With one hidden unit, $a = x \\cdot$ gate "
            "column, $u = x \\cdot$ up column, $h = \\mathrm{silu}(a)\\,u$ and the expert's output is $h$ times the down row. The "
            "chosen experts are those of the router table above, summed in rank order.\n\n"
            + _table(["e", "a", "u", "silu(a)", "h", "expert output", "weight", "weight x output"], rows)
            + f"\n\nRouted sum $y = \\sum_e g_e\\,\\mathrm{{ffn}}_e(x)$ = {_vec(y)}.\n\n"
            f"The shared expert (gate column (0.2,0.4,-0.2), up column (1,0,1), down row (0.5,-0.5,0.5)) has a = {_fx(a)}, "
            f"u = {_fx(u)}, h = {_fx(h)} and output {_vec(sh_out)}. Added unweighted: {_vec(unweighted)}. Added with the sigmoid "
            f"gate of Qwen3.5 (gate vector (0.3,0.3,0.6), logit {_fx(s_logit)}, sigmoid {_fx(s_gate)}): {_vec(gated)}.")


def sigmoid_router():
    logits = [_dot(HAND_X, r) for r in HAND_ROUTER]
    scores = [_sigmoid(z) for z in logits]
    chosen, w = route(logits, HAND_K, "sigmoid")
    chosen_b, w_b = route(logits, HAND_K, "sigmoid", bias=HAND_BIAS)
    ch_s, w_s = route(logits, HAND_K, "softmax")
    choice = [scores[e] + HAND_BIAS[e] for e in range(4)]
    rows = [[str(e), _fx(logits[e]), _fx(scores[e]), _fx(HAND_BIAS[e], 2), _fx(choice[e]),
             f"rank {chosen_b.index(e) + 1}" if e in chosen_b else "-"] for e in range(4)]
    return ("The same logits as the softmax router. A sigmoid router scores each expert on its own, $s_e = \\sigma(z_e)$, so the "
            "scores need not sum to 1. Selection uses $s_e + b_e$ with the bias vector $b$ = (-0.30, 0, 0, 0.45) below; the "
            "combine weight uses $s_e$ alone, divided by the sum of the chosen scores.\n\n"
            + _table(["e", "logit z", "score s = sigmoid(z)", "bias b", "selection score s + b", "chosen with bias"], rows)
            + f"\n\nWithout the bias the sigmoid router chooses experts {chosen[0]} and {chosen[1]} with weights "
            f"({_fx(w[0])}, {_fx(w[1])}); the softmax router chose {ch_s[0]} and {ch_s[1]} with weights ({_fx(w_s[0])}, "
            f"{_fx(w_s[1])}). With the bias it chooses {chosen_b[0]} and {chosen_b[1]} with weights ({_fx(w_b[0])}, "
            f"{_fx(w_b[1])}): the bias moved expert 0 out and expert 3 in, and neither weight contains it.")


# ------------------------------------------------------------------------------------------------ dispatching a batch

NAME_DISPATCH = "mixture_of_experts.dispatch"
DISP_N, DISP_D, DISP_F, DISP_E, DISP_K = 8, 4, 3, 4, 2


def _draw_matrix(g, rows, cols, scale):
    return [[scale * g.normal() for _ in range(cols)] for _ in range(rows)]


@lru_cache(maxsize=None)
def dispatch_setup():
    """The layer, the batch and the routing of the dispatch example, drawn in the order the chapter prints."""
    g = bookrng.generator(NAME_DISPATCH)
    wr = _draw_matrix(g, DISP_D, DISP_E, 0.5)
    experts = [(_draw_matrix(g, DISP_D, DISP_F, 0.5), _draw_matrix(g, DISP_D, DISP_F, 0.5), _draw_matrix(g, DISP_F, DISP_D, 0.5))
               for _ in range(DISP_E)]
    xs = _draw_matrix(g, DISP_N, DISP_D, 1.0)
    ids, wts = [], []
    for x in xs:
        chosen, w = route(_vecmat(x, wr), DISP_K)
        ids.append(chosen)
        wts.append(w)
    return {"wr": wr, "experts": experts, "xs": xs, "ids": ids, "wts": wts}


def histogram_scan_emit(slot_experts, n_experts):
    """The counting sort of Volume IV, Chapter 7 on the expert of every slot: counts, exclusive offsets, and the expert-major
    permutation (slot indices, ascending within an expert)."""
    counts = [0] * n_experts
    for e in slot_experts:
        counts[e] += 1
    offsets, run = [], 0
    for c in counts:
        offsets.append(run)
        run += c
    fill = list(offsets)
    perm = [0] * len(slot_experts)
    for s, e in enumerate(slot_experts):
        perm[fill[e]] = s
        fill[e] += 1
    return counts, offsets, perm


def moe_dense_masked(s):
    """Evaluate every expert on every token and multiply by the gate (zero for an expert the token did not choose)."""
    out = []
    for x, chosen, w in zip(s["xs"], s["ids"], s["wts"]):
        gate = [0.0] * DISP_E
        for e, we in zip(chosen, w):
            gate[e] = we
        y = [0.0] * DISP_D
        for e in range(DISP_E):
            o = _expert(x, s["experts"][e])
            for j in range(DISP_D):
                y[j] += gate[e] * o[j]
        out.append(y)
    return out


def moe_dispatched(s):
    """Group the slots by expert, run each expert on its own tokens, and add the weighted rows back, expert by expert."""
    k = DISP_K
    slot_expert = [e for chosen in s["ids"] for e in chosen]
    counts, offsets, perm = histogram_scan_emit(slot_expert, DISP_E)
    out = [[0.0] * DISP_D for _ in s["xs"]]
    for e in range(DISP_E):
        for q in range(offsets[e], offsets[e] + counts[e]):
            slot = perm[q]
            tok, rank = slot // k, slot % k
            o = _expert(s["xs"][tok], s["experts"][e])
            for j in range(DISP_D):
                out[tok][j] += s["wts"][tok][rank] * o[j]
    return out


def dispatch_trace():
    s = dispatch_setup()
    k = DISP_K
    slot_expert = [e for chosen in s["ids"] for e in chosen]
    counts, offsets, perm = histogram_scan_emit(slot_expert, DISP_E)
    dense, disp = moe_dense_masked(s), moe_dispatched(s)
    same = all(a == b for ra, rb in zip(dense, disp) for a, b in zip(ra, rb))
    assert same
    route_rows = [[str(i), _vec(s["xs"][i], 2), f"{s['ids'][i][0]}, {s['ids'][i][1]}", _vec(s["wts"][i], 3)] for i in range(DISP_N)]
    slot_rows = [[str(sl), str(sl // k), str(slot_expert[sl])] for sl in range(DISP_N * k)]
    group_rows = []
    for e in range(DISP_E):
        sl = perm[offsets[e]:offsets[e] + counts[e]]
        group_rows.append([str(e), str(counts[e]), str(offsets[e]), ", ".join(str(q) for q in sl), ", ".join(str(q // k) for q in sl)])
    return (_seed_line(NAME_DISPATCH) + f" A layer of E = {DISP_E} experts, k = {k}, model width D = {DISP_D}, expert hidden width F = "
            f"{DISP_F}, and a batch of n = {DISP_N} tokens. Draws, in order: the router $\\mathbf{{W}}_{{\\mathrm{{r}}}}$ ({DISP_D} x "
            f"{DISP_E}, row-major), then for e = 0 to {DISP_E - 1} the matrices $\\mathbf{{W}}_{{\\mathrm{{gate}}}}$ ({DISP_D} x "
            f"{DISP_F}), $\\mathbf{{W}}_{{\\mathrm{{up}}}}$ ({DISP_D} x {DISP_F}) and $\\mathbf{{W}}_{{\\mathrm{{down}}}}$ ({DISP_F} x "
            f"{DISP_D}), each row-major, each entry 0.5 `normal()`, then the batch ({DISP_N} x {DISP_D}, row-major, `normal()`). "
            "Routing is the softmax router: top-2 of the softmax, renormalised.\n\n"
            + _table(["token", "x", "chosen experts", "combine weights"], route_rows, "lrrr")
            + "\n\nSlot $s = k\\,i + r$ is the $r$-th choice of token $i$ (rank 0 is the largest score). The expert of each slot, in slot "
            "order, is the input of the counting sort:\n\n"
            + _table(["slot", "token", "expert"], slot_rows)
            + "\n\nHistogram of the experts, its exclusive scan, and the permutation that lists the slots expert by expert (ascending "
            "slot number within an expert, so the order is the same on every run):\n\n"
            + _table(["expert", "count", "offset", "slots in expert-major order", "tokens"], group_rows, "lrrrr")
            + f"\n\nEach expert runs once on its own tokens ({DISP_N * k} expert evaluations in all). Evaluating every expert on every "
            f"token and multiplying by a gate that is zero for the unchosen ones takes {DISP_N * DISP_E}. The two layer outputs are "
            f"bit for bit equal: {'yes' if same else 'no'} (both add the experts' contributions to a token in the order e = 0, 1, 2, 3).")


# ------------------------------------------------------------------------------------------------ an untrained router

NAME_COLLAPSE = "mixture_of_experts.collapse"
COL_N, COL_D, COL_E, COL_K = 1024, 16, 8, 2
COL_SHARES = (0.0, 0.5, 1.0, 2.0)


@lru_cache(maxsize=None)
def collapse_loads():
    """{s: slots per expert} for tokens x_i = s mu + eps_i routed by one untrained router (the same draws for every s)."""
    g = bookrng.generator(NAME_COLLAPSE)
    wr = _draw_matrix(g, COL_D, COL_E, 1.0)
    mu = [g.normal() for _ in range(COL_D)]
    eps = _draw_matrix(g, COL_N, COL_D, 1.0)
    out = {}
    for s in COL_SHARES:
        counts = [0] * COL_E
        for i in range(COL_N):
            x = [s * mu[d] + eps[i][d] for d in range(COL_D)]
            for e in _top_k(_vecmat(x, wr), COL_K):
                counts[e] += 1
        out[s] = counts
    return out


def collapse():
    loads = collapse_loads()
    ideal = COL_N * COL_K // COL_E
    rows = []
    for s, c in loads.items():
        top2 = sum(sorted(c, reverse=True)[:2])
        starved = sum(1 for v in c if v < ideal / 4)
        rows.append([_fx(s, 1)] + [str(v) for v in c] + [_fx(max(c) / ideal, 2), f"{100 * top2 / (COL_N * COL_K):.1f}%", str(starved)])
    return (_seed_line(NAME_COLLAPSE) + f" E = {COL_E}, k = {COL_K}, D = {COL_D}, n = {COL_N} tokens. Draws, in order: the router "
            f"$\\mathbf{{W}}_{{\\mathrm{{r}}}}$ ({COL_D} x {COL_E}, row-major, `normal()`), the shared direction $\\mu$ ({COL_D} "
            f"values), then the noise $\\varepsilon_i$ of the {COL_N} tokens ({COL_N} x {COL_D}, row-major). Token $i$ is $x_i = s\\mu + "
            f"\\varepsilon_i$, and the same draws are used for every $s$. The selection is the top 2 of $x_i \\mathbf{{W}}_{{\\mathrm{{r}}}}$ "
            f"(the softmax does not change the order). A balanced router sends {ideal} slots to every expert; 'starved' counts the experts "
            f"with fewer than {ideal // 4}.\n\n"
            + _table(["s"] + [f"e{e}" for e in range(COL_E)] + ["largest / ideal", "slots to the 2 busiest", "starved"], rows))


# ------------------------------------------------------------------------------------------------ capacity and dropped tokens

CAP_FACTORS = (1.0, 1.25, 2.0)


def capacity_of(factor, n, k, e):
    return math.ceil(factor * n * k / e)


def capacity():
    loads = collapse_loads()
    rows = []
    for s, c in loads.items():
        row = [_fx(s, 1)]
        for f in CAP_FACTORS:
            cap = capacity_of(f, COL_N, COL_K, COL_E)
            row.append(str(sum(max(0, v - cap) for v in c)))
        rows.append(row)
    caps = ", ".join(f"{_fx(f, 2)}: {capacity_of(f, COL_N, COL_K, COL_E)}" for f in CAP_FACTORS)
    return (f"The loads of the previous table with a capacity of ceil(factor x n k / E) slots per expert; with n = {COL_N}, k = {COL_K}, "
            f"E = {COL_E} the capacity is, by factor, {caps}. A slot beyond an expert's capacity is dropped (its token goes on without "
            f"that expert). Entries are the number of slots dropped out of {COL_N * COL_K}.\n\n"
            + _table(["s"] + [f"factor {_fx(f, 2)}" for f in CAP_FACTORS], rows))


# ------------------------------------------------------------------------------------------------ the balancing loss

NAME_AUX = "mixture_of_experts.aux"
AUX_N, AUX_E, AUX_K = 6, 4, 2
AUX_ALPHA, AUX_Z = 0.5, 0.1


def balance_terms(logits, k):
    """(f, mean probability, probabilities) of a batch of logit rows: f_e is the share of the n k slots that chose expert e."""
    n, e = len(logits), len(logits[0])
    probs = [_softmax(row) for row in logits]
    counts = [0] * e
    for row in logits:
        for c in _top_k(row, k):
            counts[c] += 1
    f = [c / (n * k) for c in counts]
    pbar = []
    for j in range(e):
        s = 0.0
        for i in range(n):
            s += probs[i][j]
        pbar.append(s / n)
    return f, pbar, probs


def balance_loss(f, pbar, alpha):
    """alpha E sum_e f_e pbar_e, the sum taken over e ascending."""
    s = 0.0
    for fe, pe in zip(f, pbar):
        s += fe * pe
    return alpha * len(f) * s


def _lse(row):
    m = max(row)
    s = 0.0
    for v in row:
        s += math.exp(v - m)
    return m + math.log(s)


def router_regulariser(logits, f, alpha, z_coef):
    """balance_loss with f held fixed, plus z_coef times the mean over tokens of lse(logits)^2."""
    n = len(logits)
    probs = [_softmax(r) for r in logits]
    pbar = []
    for j in range(len(f)):
        s = 0.0
        for i in range(n):
            s += probs[i][j]
        pbar.append(s / n)
    z = 0.0
    for r in logits:
        z += _lse(r) ** 2
    return balance_loss(f, pbar, alpha) + z_coef * z / n


def router_regulariser_grad(logits, f, alpha, z_coef):
    """The analytic gradient of router_regulariser with respect to the logits: through the mean probability (the softmax rule)
    and through the z term."""
    n, e = len(logits), len(logits[0])
    grad = []
    for r in logits:
        p = _softmax(r)
        dp = [alpha * e * f[j] / n for j in range(e)]
        pd = 0.0
        for j in range(e):
            pd += p[j] * dp[j]
        zt = z_coef * 2.0 * _lse(r) / n
        grad.append([p[j] * (dp[j] - pd) + zt * p[j] for j in range(e)])
    return grad


def balance_values():
    rows = []
    for label, f, pbar in (
        ("balanced: every expert gets a quarter of the slots and of the probability", [0.25] * 4, [0.25] * 4),
        ("two experts take every slot, probabilities follow", [0.5, 0.5, 0.0, 0.0], [0.45, 0.45, 0.05, 0.05]),
        ("the same loads, probabilities moved off the busy experts", [0.5, 0.5, 0.0, 0.0], [0.05, 0.05, 0.45, 0.45]),
    ):
        rows.append([label, _vec(f, 2), _vec(pbar, 2), _fx(balance_loss(f, pbar, 1.0), 3)])
    return ("E = 4, k = 2, $\\alpha_{\\mathrm{bal}} = 1$: the loss $E\\sum_e f_e\\bar p_e$ on three pairs of load shares $f$ and mean "
            "probabilities $\\bar p$.\n\n" + _table(["case", "f", "$\\bar p$", "loss"], rows, "lrrr"))


def balance_gradient():
    g = bookrng.generator(NAME_AUX)
    logits = _draw_matrix(g, AUX_N, AUX_E, 1.0)
    f, pbar, _ = balance_terms(logits, AUX_K)
    ana = router_regulariser_grad(logits, f, AUX_ALPHA, AUX_Z)
    h = 1e-6
    num = []
    for i in range(AUX_N):
        row = []
        for j in range(AUX_E):
            up = [r[:] for r in logits]
            dn = [r[:] for r in logits]
            up[i][j] += h
            dn[i][j] -= h
            row.append((router_regulariser(up, f, AUX_ALPHA, AUX_Z) - router_regulariser(dn, f, AUX_ALPHA, AUX_Z)) / (2 * h))
        num.append(row)
    worst = max(abs(a - b) for ra, rb in zip(ana, num) for a, b in zip(ra, rb))
    assert worst < 1e-6
    rows = [[str(e), _fx(f[e], 4), _fx(pbar[e], 4), _fx(logits[0][e], 4), _fx(ana[0][e], 6), _fx(num[0][e], 6)] for e in range(AUX_E)]
    return (_seed_line(NAME_AUX) + f" n = {AUX_N} tokens, E = {AUX_E}, k = {AUX_K}; the logits are the {AUX_N} x {AUX_E} draws of "
            f"`normal()`, row-major. $\\alpha_{{\\mathrm{{bal}}}} = {AUX_ALPHA}$ and the z-loss coefficient is {AUX_Z}. The shares $f_e$ "
            "are counted once and held fixed in the loss. Token 0's row of the gradient with respect to the logits, from the formula "
            "of the chapter and from central differences with step 1e-6.\n\n"
            + _table(["e", "f", "$\\bar p$", "logit of token 0", "formula", "central difference"], rows)
            + f"\n\nLoss {_fx(router_regulariser(logits, f, AUX_ALPHA, AUX_Z), 6)} (balancing term "
            f"{_fx(balance_loss(f, pbar, AUX_ALPHA), 6)}). Over all {AUX_N * AUX_E} entries the largest difference between formula and "
            "central difference is below 1e-6 (criterion: below 1e-6).")


# ------------------------------------------------------------------------------------------------ training a tiny mixture

NAME_TRAIN_DATA = "mixture_of_experts.train_data"
TR_N, TR_D, TR_C, TR_E, TR_K, TR_F = 128, 6, 6, 8, 2, 4
TR_SPREAD, TR_NOISE, TR_INIT, TR_LR, TR_STEPS = 1.0, 0.3, 0.3, 0.02, 300
TR_ALPHAS = (0.0, 0.05, 0.5)
TR_SEEDS = (0, 1, 2)
ADAM = (0.9, 0.999, 1e-8)


def train_init_name(seed):
    return f"mixture_of_experts.train_init.{seed}"


@lru_cache(maxsize=None)
def train_data():
    """128 tokens of 6 numbers from 6 clusters, each token's target a cluster-specific linear map of it."""
    g = bookrng.generator(NAME_TRAIN_DATA)
    centers = _draw_matrix(g, TR_C, TR_D, TR_SPREAD)
    maps = [_draw_matrix(g, TR_D, TR_D, 1.0 / math.sqrt(TR_D)) for _ in range(TR_C)]
    xs, ts = [], []
    for _ in range(TR_N):
        c = g.integer(TR_C)
        x = [centers[c][d] + TR_NOISE * g.normal() for d in range(TR_D)]
        xs.append(x)
        ts.append(_vecmat(x, maps[c]))
    return xs, ts


def _zeros_like(params):
    return [[[0.0] * len(r) for r in m] for m in params]


def _expert_backward(x, w, gw, cache, dy):
    """Backward of one SwiGLU expert: add its weight gradients to gw; cache is (a, u, h) of the forward."""
    a, u, h = cache
    d_in, f = len(x), len(a)
    for i in range(f):
        for r in range(len(dy)):
            gw[2][i][r] += h[i] * dy[r]
    for i in range(f):
        dh = 0.0
        for r in range(len(dy)):
            dh += w[2][i][r] * dy[r]
        da = dh * u[i] * _silu_slope(a[i])
        du = dh * _silu(a[i])
        for d in range(d_in):
            gw[0][d][i] += x[d] * da
            gw[1][d][i] += x[d] * du


def _expert_forward(x, w):
    a = _vecmat(x, w[0])
    u = _vecmat(x, w[1])
    h = [_silu(ai) * ui for ai, ui in zip(a, u)]
    return (a, u, h), _vecmat(h, w[2])


def train_moe(job):
    """Train the tiny mixture for TR_STEPS full-batch Adam steps. job = (alpha, init seed index, nudge). The nudge is added to
    the first router weight before the first step. Returns the final loads (slots per expert) and the final loss."""
    alpha, seed, nudge = job
    xs, ts = train_data()
    g = bookrng.generator(train_init_name(seed))
    router = _draw_matrix(g, TR_D, TR_E, TR_INIT)
    experts = []
    for _ in range(TR_E):
        experts.append([_draw_matrix(g, TR_D, TR_F, TR_INIT), _draw_matrix(g, TR_D, TR_F, TR_INIT), _draw_matrix(g, TR_F, TR_D, TR_INIT)])
    router[0][0] += nudge
    n, k, e_n = TR_N, TR_K, TR_E
    flat = [router] + [m for ex in experts for m in ex]
    m1, m2 = _zeros_like(flat), _zeros_like(flat)
    b1, b2, eps = ADAM
    loss = 0.0
    for step in range(1, TR_STEPS + 1):
        grads = _zeros_like(flat)
        g_router = grads[0]
        g_exp = [[grads[1 + 3 * e], grads[2 + 3 * e], grads[3 + 3 * e]] for e in range(e_n)]
        cache, counts, loss = [], [0] * e_n, 0.0
        for i in range(n):
            x = xs[i]
            p = _softmax(_vecmat(x, router))
            chosen = _top_k(p, k)
            z = 0.0
            for e in chosen:
                z += p[e]
            gate = {e: p[e] / z for e in chosen}
            out, outs = [0.0] * TR_D, {}
            for e in sorted(chosen):
                c, y = _expert_forward(x, experts[e])
                outs[e] = (c, y)
                counts[e] += 1
                for r in range(TR_D):
                    out[r] += gate[e] * y[r]
            err = [out[r] - ts[i][r] for r in range(TR_D)]
            for r in range(TR_D):
                loss += 0.5 * err[r] * err[r] / n
            cache.append((p, chosen, z, gate, outs, err))
        f = [c / (n * k) for c in counts]
        for i in range(n):
            p, chosen, z, gate, outs, err = cache[i]
            dout = [v / n for v in err]
            dgate = {}
            for e in chosen:
                s = 0.0
                for r in range(TR_D):
                    s += dout[r] * outs[e][1][r]
                dgate[e] = s
                _expert_backward(xs[i], experts[e], g_exp[e], outs[e][0], [gate[e] * v for v in dout])
            sdp = 0.0
            for e in chosen:
                sdp += dgate[e] * p[e]
            dp = [alpha * e_n * f[e] / n for e in range(e_n)]
            for e in chosen:
                dp[e] += dgate[e] / z - sdp / (z * z)
            pd = 0.0
            for e in range(e_n):
                pd += p[e] * dp[e]
            for e in range(e_n):
                dl = p[e] * (dp[e] - pd)
                for d in range(TR_D):
                    g_router[d][e] += xs[i][d] * dl
        for pi in range(len(flat)):
            for r in range(len(flat[pi])):
                for c in range(len(flat[pi][r])):
                    gr = grads[pi][r][c]
                    m1[pi][r][c] = b1 * m1[pi][r][c] + (1 - b1) * gr
                    m2[pi][r][c] = b2 * m2[pi][r][c] + (1 - b2) * gr * gr
                    flat[pi][r][c] -= TR_LR * (m1[pi][r][c] / (1 - b1 ** step)) / (math.sqrt(m2[pi][r][c] / (1 - b2 ** step)) + eps)
    counts = [0] * e_n
    for x in xs:
        for e in _top_k(_vecmat(x, router), k):
            counts[e] += 1
    return {"job": job, "loads": counts, "loss": loss}


@lru_cache(maxsize=None)
def train_results():
    """{(alpha, seed): (base run, [nudged runs])}, 45 runs spread over four processes."""
    jobs = [(a, s, n) for a in TR_ALPHAS for s in TR_SEEDS for n in (0.0, *stable.NUDGES)]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(train_moe, jobs, chunksize=1)
    out = {(a, s): [None, []] for a in TR_ALPHAS for s in TR_SEEDS}
    for r in res:
        a, s, n = r["job"]
        if n == 0.0:
            out[(a, s)][0] = r
        else:
            out[(a, s)][1].append(r)
    return out


def balance_data():
    """Final loads of the un-nudged runs: {alpha: {seed: loads}}, for the figure."""
    res = train_results()
    return {a: {s: res[(a, s)][0]["loads"] for s in TR_SEEDS} for a in TR_ALPHAS}


def balance_training():
    res = train_results()
    ideal = TR_N * TR_K // TR_E
    rows = []
    for a in TR_ALPHAS:
        for s in TR_SEEDS:
            base, twins = res[(a, s)]
            cells = [stable.text(base["loads"][e], [t["loads"][e] for t in twins], lambda x: f"{x:.0f}") for e in range(TR_E)]
            loss = stable.text(base["loss"], [t["loss"] for t in twins], lambda x: _fx(x, 4))
            rows.append([_fx(a, 2), str(s)] + cells + [loss])
    return (_seed_line(NAME_TRAIN_DATA) + f" The data: 6 cluster centers, each 6 draws of `normal()` (scale {TR_SPREAD:g}); then 6 maps "
            f"of 6 x 6 entries, each `normal() / sqrt(6)`, row by row; then for each of the {TR_N} tokens the cluster `integer(6)` and 6 "
            f"values of `{TR_NOISE:g} * normal()` added to the center. The initial weights are `{TR_INIT:g} * normal()`. "
            f"Slots per expert after {TR_STEPS} steps, on the {TR_N} training tokens, for three strengths of the balancing loss and "
            f"three initialisations (each initialisation is the generator `mixture_of_experts.train_init.<seed>`, with seed "
            + ", ".join(f"{s}: {bookrng.seed_of(train_init_name(s)):#018x}" for s in TR_SEEDS)
            + f"). A balanced router has {ideal} slots per expert. The last column is the training loss, "
            "half the squared error summed over the 6 outputs and averaged over tokens, as computed in the forward pass of step "
            f"{TR_STEPS} (with the weights before that step's update, not the final weights); the slots are counted with the final router. " + stable.RULE + "\n\n"
            + _table(["$\\alpha_{\\mathrm{bal}}$", "seed"] + [f"e{e}" for e in range(TR_E)] + ["loss"], rows))


# ------------------------------------------------------------------------------------------------ expert parallelism

BYTES = 2                      # bf16
A2A_T = 4096                   # tokens per device in the large example
A2A_DEVICES = (2, 4, 8, 16, 32)


def all_to_all_tiny():
    s = dispatch_setup()
    p_dev, per_dev_tokens, per_dev_experts = 2, DISP_N // 2, DISP_E // 2
    sent = [[0] * p_dev for _ in range(p_dev)]
    for tok, chosen in enumerate(s["ids"]):
        for e in chosen:
            sent[tok // per_dev_tokens][e // per_dev_experts] += 1
    rows = [[f"device {a}"] + [str(sent[a][b]) for b in range(p_dev)] for a in range(p_dev)]
    remote = sum(sent[a][b] for a in range(p_dev) for b in range(p_dev) if a != b)
    slot_bytes = D_Q * BYTES
    return ("The routing of the dispatch example split over 2 devices: tokens 0 to 3 live on device 0 and tokens 4 to 7 on device 1; "
            "experts 0 and 1 live on device 0 and experts 2 and 3 on device 1. Entry (a, b) counts the slots of tokens on device a "
            "whose expert is on device b.\n\n"
            + _table(["tokens on", "experts on device 0", "experts on device 1"], rows)
            + f"\n\n{remote} of the {DISP_N * DISP_K} slots cross between devices. At a model width of {_int(D_Q)} in bf16 a slot is "
            f"{_int(slot_bytes)} bytes, so the exchange sends {_int(remote * slot_bytes)} bytes to the experts and receives the "
            f"same number of bytes of results back.")


def all_to_all_data():
    """Rows (P, experts per device, fraction of slots that leave, MiB sent per direction, MiB of expert weights per device,
    both-ways bytes over weight bytes, crossover tokens) for the large example, uniform routing."""
    t, k, e, d, f = A2A_T, K_Q, E_Q, D_Q, F_Q
    out = []
    for p in A2A_DEVICES:
        frac = (p - 1) / p
        sent = t * k * frac * d * BYTES
        weights = (e // p) * expert_params(d, f) * BYTES
        out.append((p, e // p, frac, sent / 2 ** 20, weights / 2 ** 20, 2 * sent / weights, 3 * e * f / (2 * k * (p - 1))))
    return out


def all_to_all():
    rows = [[str(p), str(ep), _fx(fr, 4), _fx(sent, 1), _fx(w, 1), _fx(ratio, 3), f"{cross:,.0f}"] for p, ep, fr, sent, w, ratio, cross in all_to_all_data()]
    layer = 40
    p8 = next(r for r in all_to_all_data() if r[0] == 8)
    return (f"The expert layer of Qwen3.5-35B-A3B (E = 256, k = 8, D = 2,048, F = 512) on $n_{{\\mathrm{{dev}}}}$ devices, each holding "
            f"E / $n_{{\\mathrm{{dev}}}}$ experts and a batch of n = {_int(A2A_T)} tokens, uniform routing, bf16 (2 bytes). A token's k slots go "
            "to the devices of their experts; a slot stays home with probability $1 / n_{\\mathrm{dev}}$. 'MiB sent' is per device, per "
            "layer, one direction (the results return the same number of bytes). 'Both ways / weights' divides twice that by the expert "
            "weights a device holds. The last column is the n at which that ratio is 1: $n^\\ast = 3EF / (2k(n_{\\mathrm{dev}}-1))$.\n\n"
            + _table(["devices", "experts per device", "share of slots leaving", "MiB sent", "MiB of expert weights", "both ways / weights", "$n^\\ast$"], rows)
            + f"\n\nWith 8 devices one pass over all {layer} layers moves {_fx(2 * p8[3] * layer / 1024, 2)} GiB through each device's "
            "network port.")


# ------------------------------------------------------------------------------------------------ what a decode step reads

DEC_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)


def expected_distinct(e, k, n):
    """Expected number of distinct experts that n tokens choose when each token picks k of e uniformly: e (1 - (1 - k/e)^n)."""
    return e * (1.0 - (1.0 - k / e) ** n)


def decode_layer(n, e=E_Q, k=K_Q, d=D_Q, f=F_Q):
    """(bytes read, FLOPs) of one decode step of n tokens through one expert layer with one shared expert, bf16: the router,
    the shared expert and the expected distinct routed experts are read once; every token uses k routed experts and the shared one."""
    p_exp = expert_params(d, f)
    read = (expected_distinct(e, k, n) + 1) * p_exp * BYTES + e * d * BYTES
    flops = n * (2 * (k + 1) * p_exp + 2 * e * d)
    return read, flops


def decode_data():
    """Rows (n, distinct experts, MiB read, MoE FLOPs per byte, dense FLOPs per byte)."""
    rows = []
    for n in DEC_BATCHES:
        read, flops = decode_layer(n)
        dense_read = (K_Q + 1) * expert_params(D_Q, F_Q) * BYTES
        dense_flops = n * 2 * (K_Q + 1) * expert_params(D_Q, F_Q)
        rows.append((n, expected_distinct(E_Q, K_Q, n), read / 2 ** 20, flops / read, dense_flops / dense_read))
    return rows


def tokens_for_intensity(target):
    """The smallest n whose decode step reaches `target` FLOPs per byte, for the dense layer of the same active width and the mixture."""
    dense = target
    n = 1
    while decode_layer(n)[1] / decode_layer(n)[0] < target:
        n += 1
    return dense, n


def decode_bytes():
    rows = [[_int(n), _fx(dist, 1), _fx(mib, 1), _fx(imoe, 2), _fx(idense, 2), _fx(imoe / idense, 3)] for n, dist, mib, imoe, idense in decode_data()]
    need = [[str(t), str(tokens_for_intensity(t)[0]), str(tokens_for_intensity(t)[1])] for t in (16, 64, 256)]
    c = qwen_counts_data()
    return ("One expert layer of Qwen3.5-35B-A3B at decode (E = 256, k = 8, one shared expert, D = 2,048, F = 512, bf16), n sequences "
            "each producing one token, uniform routing. Bytes read: the router, the shared expert and the expected number of distinct "
            "routed experts, $256\\,(1 - (1 - 8/256)^n)$, each once. FLOPs: every token uses 8 routed experts, the shared expert and the "
            "router. The dense layer has the same active width (9 experts' worth, 28,311,552 parameters), reads all of it and does "
            "$2$ FLOPs per parameter per token, so its intensity is exactly n FLOPs per byte.\n\n"
            + _table(["n", "distinct routed experts", "MiB read", "mixture FLOPs per byte", "dense FLOPs per byte", "ratio"], rows)
            + "\n\nTokens per step needed to reach a given arithmetic intensity (FLOPs per byte of weights read):\n\n"
            + _table(["target intensity", "n, dense", "n, mixture"], need)
            + f"\n\nWhole model, n = 1: a token's products read {_int(c['used'] * BYTES)} bytes of weights ({_fx(c['used'] * BYTES / 2 ** 30, 2)} GiB) "
            f"while {_int(c['total'] * BYTES)} bytes ({_fx(c['total'] * BYTES / 2 ** 30, 2)} GiB) must be resident.")
