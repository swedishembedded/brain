# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Training a GPT from Scratch": raw text to a trained generative model.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. The pieces of earlier chapters are imported, never copied: the pre-tokenizer and byte-pair
merges of Chapter 2 (`tokenization`), the pre-norm transformer block with its backward pass of Chapter 9
(`transformer_block`, whose attention is the layer of Chapter 7), the learning-rate curve of Volume III (`schedules`) and the
model device of Volume IV (`roofline`). What makes the pipeline one pipeline is written here:

* a small generated corpus, its tokenizer (with a compact vocabulary and a document separator) and the packing of the token
  stream into fixed rows with an input, a target, a loss mask and a document id for every position;
* the GPT itself: token embedding plus learned positions, `L` blocks, a final RMS norm and a head tied to the embedding
  table, with a forward and a backward pass (checked against central differences) in which attention may or may not cross a
  document boundary;
* a masked mean cross-entropy, AdamW with warmup, cosine decay and a clip on the global gradient norm, validation after every
  few steps, greedy text from saved checkpoints, and a copy task that shows what a wrong loss mask costs;
* the arithmetic of training as work: FLOPs per token (counted from the shapes and compared with six times the parameters),
  the token budget of a GPT-2 small run, the fraction of a model device that a given utilisation means, and the memory of the
  weights, the optimizer and the activations.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints its seeds. Sums of
products are plain left-to-right additions, the order a C loop would use. Numbers of trained models are printed by the
stable-digits rule of `stable.py`: each run is repeated from five starting points that differ by 1e-12 in one weight.

Swedish Embedded AB implements solutions for training language models from raw text, from the tokenizer to the checkpoint. If your
team needs expertise in building, training and costing generative transformers, you can procure our services by sending an email
to info@swedishembedded.com.
"""
import math
import re
from collections import Counter
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import roofline
import schedules
import self_attention as sa
import stable
import tokenization as tok
import transformer_block as tb

matmul = sa.matmul
transpose = sa.transpose
_sum = sa._sum

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


# ------------------------------------------------------------------------------------------------ the corpus

# One document is one sentence of seven slots; every slot is drawn uniformly from its list, in this order.
DETS = ["the", "a"]
ADJS = ["small", "red", "old", "quiet"]
NOUNS = ["cat", "dog", "fox", "hen", "owl"]
VERBS = ["sat", "ran", "slept"]
PREPS = ["on", "near", "by"]
PLACES = ["mat", "barn", "pond", "hill"]
SLOTS = [DETS, ADJS, NOUNS, VERBS, PREPS, DETS, PLACES]
N_TRAIN_DOCS, N_VAL_DOCS = 150, 40
NAME_TRAIN, NAME_VAL = "training_gpt.corpus.train", "training_gpt.corpus.val"
SENTENCE_NATS = math.log(math.prod(len(s) for s in SLOTS))   # entropy of one sentence in nats


def make_docs(name, n):
    g = bookrng.generator(name)
    docs = []
    for _ in range(n):
        w = [s[g.integer(len(s))] for s in SLOTS]
        docs.append(f"{w[0]} {w[1]} {w[2]} {w[3]} {w[4]} {w[5]} {w[6]}.")
    return docs


@lru_cache(maxsize=None)
def train_docs():
    return make_docs(NAME_TRAIN, N_TRAIN_DOCS)


@lru_cache(maxsize=None)
def val_docs():
    return make_docs(NAME_VAL, N_VAL_DOCS)


# ------------------------------------------------------------------------------------------------ the tokenizer

MERGES = 48
EOT = 0   # the document separator, id 0 of the compact vocabulary


class Tokenizer:
    """Byte-pair merges of Chapter 2 learned on the training documents only, over a compact id space: id 0 is the document
    separator, then the bytes that occur in the training text in increasing order, then the merges in the order learned."""

    def __init__(self, docs=None, merges=MERGES):
        docs = train_docs() if docs is None else docs
        table = Counter()
        for d in docs:
            table.update(tok.word_table(d.encode()))
        self.merges = tok.train_bpe(table, merges)
        self.ranks = tok.ranks_of(self.merges)
        self.pieces_raw = tok.token_bytes(self.merges)
        used = sorted({b for d in docs for b in d.encode()})
        raw = used + [256 + k for k in range(len(self.merges))]
        self.to_compact = {r: 1 + i for i, r in enumerate(raw)}
        self.to_raw = [None] + raw
        self.vocab = 1 + len(raw)
        self.n_bytes = len(used)

    def encode(self, doc):
        return [self.to_compact[i] for i in tok.encode(doc.encode(), self.ranks)]

    def piece(self, i):
        return b"" if i == EOT else self.pieces_raw[self.to_raw[i]]

    def name(self, i):
        return "EOT" if i == EOT else tok.show(self.piece(i))

    def decode(self, ids):
        return b"".join(self.piece(i) for i in ids).decode("utf-8", errors="replace")


@lru_cache(maxsize=None)
def tokenizer():
    return Tokenizer()


# ------------------------------------------------------------------------------------------------ the stream and the rows

ROW = 16   # the block size T: positions per row


def token_stream(docs, tk):
    """(ids, doc): every document is its separator then its tokens; doc[i] is the index of the document that position i opens
    or continues."""
    ids, doc = [], []
    for k, d in enumerate(docs):
        t = [EOT] + tk.encode(d)
        ids += t
        doc += [k] * len(t)
    return ids, doc


def pack_rows(ids, doc, T=ROW):
    """Fixed rows of T positions: row r has inputs ids[rT .. rT+T-1] and targets ids[rT+1 .. rT+T], so every token but the
    first is a target exactly once. The last row is padded with the separator as input and given a loss mask of 0 (and the
    document id -1) on its padding. A row is {x, y, m, doc}."""
    n = len(ids)
    rows = []
    for r in range(-(-(n - 1) // T)):
        a = r * T
        k = min(T, n - 1 - a)
        pad = T - k
        rows.append({"x": ids[a:a + k] + [EOT] * pad, "y": ids[a + 1:a + 1 + k] + [EOT] * pad,
                     "m": [1] * k + [0] * pad, "doc": doc[a:a + k] + [-1] * pad})
    return rows


@lru_cache(maxsize=None)
def data():
    tk = tokenizer()
    out = {}
    for split, docs in (("train", train_docs()), ("val", val_docs())):
        ids, doc = token_stream(docs, tk)
        out[split] = {"ids": ids, "doc": doc, "rows": pack_rows(ids, doc)}
    return out


def corpus_floor(split):
    """The entropy of the generator in nats per target token: the documents of the split times the entropy of one sentence,
    over the number of targets. No model can do better in expectation."""
    d = data()[split]
    return (len(val_docs() if split == "val" else train_docs()) * SENTENCE_NATS) / (len(d["ids"]) - 1)


def corpus_and_tokens():
    tk = tokenizer()
    d = data()
    docs = train_docs()
    rows = []
    for k in range(3):
        ids = tk.encode(docs[k])
        rows.append((k, f"`{docs[k]}`", " ".join(f"`{tk.name(i)}`" for i in ids), len(ids)))
    merges = ", ".join(f"`{tok.show(tk.pieces_raw[256 + k])}`" for k in range(len(tk.merges)))
    lengths = [len(tk.encode(x)) + 1 for x in docs]
    leak = len(set(val_docs()) & set(docs))
    return (f"Slots (each drawn uniformly, in this order): determiner {DETS}, adjective {ADJS}, noun {NOUNS}, verb {VERBS}, "
            f"preposition {PREPS}, determiner {DETS}, place {PLACES}; the document is the seven words joined by single spaces "
            f"followed by a full stop. Draws: one `integer(n)` per slot, `n` the size of the slot's list, in the order above, "
            f"for each document in turn. {N_TRAIN_DOCS} training documents from `{NAME_TRAIN}` (seed {_seed(NAME_TRAIN)}); "
            f"{N_VAL_DOCS} validation documents from `{NAME_VAL}` (seed {_seed(NAME_VAL)}). A sentence has "
            f"{math.prod(len(s) for s in SLOTS):,} possible spellings, so its entropy is ln of that, {_fx(SENTENCE_NATS)} nats, "
            f"and {leak} of the {N_VAL_DOCS} validation documents also occur among the training documents.\n\n"
            f"The tokenizer is the byte-pair encoder of Chapter 2 with its pre-tokenizer (a word carries the space before it; "
            f"the first word of a document has no space before it, since the text starts with it; every later word carries "
            f"the space before it and the full stop has none), "
            f"learned with {MERGES} merges on the training documents only. The vocabulary has id 0 for the document separator "
            f"`EOT`, then the {tk.n_bytes} bytes that occur in the training text in increasing order, then the {len(tk.merges)} merges: "
            f"V = 1 + {tk.n_bytes} + {len(tk.merges)} = {tk.vocab}. Merges in the order learned (a space is written `_`): {merges}.\n\n"
            + _table(["document", "text", "tokens", "token count (with its separator)"],
                     [(a, b, c, e + 1) for a, b, c, e in rows], "rlll")
            + f"\n\nThe {N_TRAIN_DOCS} training documents have {_int(len(d['train']['ids']))} tokens with their separators "
            f"(mean {_fx(_sum([float(x) for x in lengths]) / len(lengths), 2)} per document, longest {max(lengths)}); the validation "
            f"documents {_int(len(d['val']['ids']))}. The entropy of the generator is {_fx(corpus_floor('train'))} nats per target "
            f"token on the training stream and {_fx(corpus_floor('val'))} on the validation stream, perplexities "
            f"{_fx(math.exp(corpus_floor('train')))} and {_fx(math.exp(corpus_floor('val')))}.")


# ------------------------------------------------------------------------------------------------ the GPT

def causal(i, j, doc=None):
    return j <= i


def same_document(doc):
    """The block-diagonal causal mask of a row: position i may read j when j <= i and both belong to one document."""
    return lambda i, j: j <= i and doc[i] == doc[j]


def attention(X, p, H, allowed):
    """Multi-head attention (Chapter 7) over the rows of X with the mask `allowed(i, j)`; the result and the cache are the
    ones `self_attention.mha_forward` returns, so `self_attention.mha_backward` differentiates it unchanged."""
    T = len(X)
    Wq, Wk, Wv, Wo = p["Wq"], p["Wk"], p["Wv"], p["Wo"]
    dk = len(Wq[0]) // H
    Q, K, V = matmul(X, Wq), matmul(X, Wk), matmul(X, Wv)
    scale = 1.0 / math.sqrt(dk)
    heads = []
    for h in range(H):
        Qh, Kh, Vh = sa._cols(Q, h * dk, (h + 1) * dk), sa._cols(K, h * dk, (h + 1) * dk), sa._cols(V, h * dk, (h + 1) * dk)
        S = [[sa._dot(Qh[i], Kh[j]) * scale if allowed(i, j) else -math.inf for j in range(T)] for i in range(T)]
        P = [sa._softmax(row) for row in S]
        heads.append({"Q": Qh, "K": Kh, "V": Vh, "S": S, "P": P, "O": matmul(P, Vh)})
    O = sa._hcat([hd["O"] for hd in heads])
    return matmul(O, Wo), {"X": X, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "H": H, "dk": dk, "dv": dk, "heads": heads, "O": O,
                           "uniform": False}


def block(p, X, H, allowed):
    """The pre-norm block of Chapter 9 (RMS norm, attention, SwiGLU, two residual adds) with an arbitrary mask. Same cache
    as `transformer_block.block_forward(..., "pre", ...)`, so `transformer_block.block_backward` differentiates it."""
    c = {"layout": "pre", "X": X}
    c["n1"], c["cn1"] = tb.rms_forward(X, p["g1"][0])
    c["a"], c["ca"] = attention(c["n1"], p, H, allowed)
    c["x1"] = tb._add(X, c["a"])
    c["n2"], c["cn2"] = tb.rms_forward(c["x1"], p["g2"][0])
    c["f"], c["cf"] = tb.ffn_forward(p, c["n2"])
    c["x2"] = tb._add(c["x1"], c["f"])
    return c["x2"], c


CFG = {"V": None, "D": 16, "H": 2, "FF": 24, "L": 2, "T": ROW}   # V is the tokenizer's, filled in by `config()`
INIT_STD = 0.02
BLOCK_KEYS = tb.BLOCK_KEYS


def config(**kw):
    c = dict(CFG)
    c["V"] = tokenizer().vocab
    c.update(kw)
    return c


def init_gpt(name, cfg, std=INIT_STD):
    """Parameters of the GPT, every matrix entry `std * normal()` from `bookrng.generator(name)`, drawn row by row in this
    order: the embedding table E [V, D]; the position table P [T, D]; then for each block W_Q, W_K, W_V, W_O (D by D), W_gate,
    W_up (D by F) and W_down (F by D), the two output matrices W_O and W_down multiplied by 1 / sqrt(2 L). All gains are 1."""
    g = bookrng.generator(name)
    V, D, T, F, L = cfg["V"], cfg["D"], cfg["T"], cfg["FF"], cfg["L"]
    out = 1.0 / math.sqrt(2 * L)
    mat = lambda r, c, s: [[std * s * g.normal() for _ in range(c)] for _ in range(r)]
    p = {"E": mat(V, D, 1.0), "P": mat(T, D, 1.0), "blocks": []}
    for _ in range(L):
        b = {}
        for key, (r, c), s in (("Wq", (D, D), 1.0), ("Wk", (D, D), 1.0), ("Wv", (D, D), 1.0), ("Wo", (D, D), out),
                               ("Wg", (D, F), 1.0), ("Wu", (D, F), 1.0), ("Wd", (F, D), out)):
            b[key] = mat(r, c, s)
        b["g1"], b["g2"] = [[1.0] * D], [[1.0] * D]
        p["blocks"].append(b)
    p["gf"] = [[1.0] * D]
    return p


def tensors(p):
    """Every parameter matrix in a fixed order (the order of the gradients): E, P, each block's tensors, the final gain."""
    out = [p["E"], p["P"]]
    for b in p["blocks"]:
        out += [b[k] for k in BLOCK_KEYS]
    out.append(p["gf"])
    return out


def tensor_names(cfg):
    return ["E", "P"] + [f"{k}{l}" for l in range(cfg["L"]) for k in BLOCK_KEYS] + ["gf"]


def parameter_count(p):
    return sum(len(r) for t in tensors(p) for r in t)


def forward(p, ids, cfg, allowed):
    """(logits [T, V], cache) for one row of token ids: x = E[ids] + P, L blocks, the final RMS norm, and the head, which is
    the embedding table used a second time: logits = xn E^T."""
    H = cfg["H"]
    x = [[a + b for a, b in zip(p["E"][t], p["P"][i])] for i, t in enumerate(ids)]
    caches = []
    for b in p["blocks"]:
        x, c = block(b, x, H, allowed)
        caches.append(c)
    xn, cf = tb.rms_forward(x, p["gf"][0])
    logits = matmul(xn, transpose(p["E"]))
    return logits, {"ids": ids, "caches": caches, "xn": xn, "cf": cf}


def softmax_row(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = _sum(e)
    return [v / s for v in e], m + math.log(s)


def losses(logits, y):
    """Per-position cross-entropy -log softmax(logits)[y] (natural log), one number per position."""
    out = []
    for z, t in zip(logits, y):
        _, lse = softmax_row(z)
        out.append(lse - z[t])
    return out


def backward(p, cache, logits, y, m, denom, cfg):
    """Gradients (in the order of `tensors`) of sum_i m_i * ce_i / denom for one row. The gradient of the logits is
    m_i (softmax - onehot) / denom; the head's gradient reaches E through E^T, the embedding lookup scatters the stream's
    gradient into the rows of E, and the position table receives it unchanged."""
    V, D = cfg["V"], cfg["D"]
    dlog = []
    for z, t, mi in zip(logits, y, m):
        pr, _ = softmax_row(z)
        s = mi / denom
        dlog.append([(pr[j] - (1.0 if j == t else 0.0)) * s for j in range(V)])
    dxn = matmul(dlog, p["E"])
    dE = matmul(transpose(dlog), cache["xn"])
    dx, dgf = tb.rms_backward(dxn, cache["cf"])
    block_grads = []
    for c in reversed(cache["caches"]):
        dx, g = tb.block_backward(c, dx)
        block_grads.append(g)
    block_grads.reverse()
    dP = [list(r) for r in dx]
    for i, t in enumerate(cache["ids"]):
        row = dE[t]
        for j in range(D):
            row[j] += dx[i][j]
    grads = [dE, dP]
    for g in block_grads:
        grads += [g[k] for k in BLOCK_KEYS]
    grads.append([dgf])
    return grads


def mask_for(row, mode):
    return same_document(row["doc"]) if mode == "block" else causal


def batch_loss(p, rows, cfg, mode="causal", grad=False):
    """(mean loss over the supervised positions of the rows, gradients or None). The denominator is the number of
    supervised positions of the whole batch, as a data-parallel trainer divides by it."""
    denom = float(max(1, sum(sum(r["m"]) for r in rows)))
    total, acc = 0.0, None
    for r in rows:
        logits, cache = forward(p, r["x"], cfg, mask_for(r, mode))
        ce = losses(logits, r["y"])
        for c, mi in zip(ce, r["m"]):
            total += c * mi
        if grad:
            g = backward(p, cache, logits, r["y"], r["m"], denom, cfg)
            if acc is None:
                acc = g
            else:
                for ta, tg in zip(acc, g):
                    for ra, rg in zip(ta, tg):
                        for j, x in enumerate(rg):
                            ra[j] += x
    return total / denom, acc


# ------------------------------------------------------------------------------------------------ the optimizer and the loop

TRAIN = {"B": 8, "steps": 300, "peak": 0.01, "floor": 0.001, "warmup": 20, "wd": 0.1, "clip": 1.0,
         "b1": 0.9, "b2": 0.95, "eps": 1e-8}
EVAL_EVERY = 10
CHECKPOINTS = (0, 10, 25, 50, 100, 200, 300)
NAME_INIT, NAME_ORDER = "training_gpt.init", "training_gpt.batches"


def decayed(cfg):
    return [not (n.startswith("g")) for n in tensor_names(cfg)]


def global_norm(grads):
    return math.sqrt(_sum([_sum([x * x for r in t for x in r]) for t in grads]))


def adamw_update(ts, grads, mom, step, lr, scale, decay, o):
    """One AdamW step (Volume III) on every tensor: gradient times `scale` (the clip coefficient), Adam's moments with bias
    correction at the 1-based step, then p <- p - lr wd p - lr mhat / (sqrt(vhat) + eps); the decay skips the gains."""
    bc1, bc2 = 1.0 - o["b1"] ** step, 1.0 - o["b2"] ** step
    for t, g, (m, v), dec in zip(ts, grads, mom, decay):
        for r, gr, mr, vr in zip(t, g, m, v):
            for j in range(len(r)):
                gj = gr[j] * scale
                mr[j] = o["b1"] * mr[j] + (1.0 - o["b1"]) * gj
                vr[j] = o["b2"] * vr[j] + (1.0 - o["b2"]) * gj * gj
                if dec:
                    r[j] -= lr * o["wd"] * r[j]
                r[j] -= lr * (mr[j] / bc1) / (math.sqrt(vr[j] / bc2) + o["eps"])


def batch_order(n_rows, B, steps, name=NAME_ORDER):
    """The rows of each step: shuffled epochs. Draws from `bookrng.generator(name)`: one `shuffle` of 0 .. n_rows-1 per
    epoch; consecutive groups of B rows are the batches and an incomplete last group is dropped."""
    g = bookrng.generator(name)
    out = []
    while len(out) < steps:
        idx = g.shuffle(list(range(n_rows)))
        out += [idx[k:k + B] for k in range(0, n_rows - B + 1, B)]
    return out[:steps]


def mean_loss(p, rows, cfg, mode="causal"):
    return batch_loss(p, rows, cfg, mode)[0]


def greedy(p, cfg, prompt, max_new, tk):
    """Greedy continuation: the largest logit at the last position (the first on a tie), the context cropped to the last T
    tokens; stops after a separator or after max_new tokens. Returns the text before the stop."""
    ctx = list(prompt)
    out = []
    for _ in range(max_new):
        window = ctx[-cfg["T"]:]
        logits, _ = forward(p, window, cfg, causal)
        last = logits[-1]
        nxt = max(range(len(last)), key=lambda j: (last[j], -j))
        if nxt == EOT:
            break
        ctx.append(nxt)
        out.append(nxt)
    return tk.decode(out)


def run(job):
    """One training run: (mode, nudge). Returns the numbers the chapter prints, all computed inside the run so that the
    nudged copies can be compared digit by digit."""
    mode, nudge = job
    cfg, o = config(), TRAIN
    tk, d = tokenizer(), data()
    p = init_gpt(NAME_INIT, cfg)
    p["E"][0][0] += nudge
    ts = tensors(p)
    mom = [([[0.0] * len(r) for r in t], [[0.0] * len(r) for r in t]) for t in ts]
    decay = decayed(cfg)
    lr_of = schedules.cosine(o["peak"], o["floor"], o["warmup"], o["steps"])
    order = batch_order(len(d["train"]["rows"]), o["B"], o["steps"])
    val_rows = d["val"]["rows"]
    out = {"mode": mode, "nudge": nudge, "train": [], "val": {}, "gnorm": [], "clipped": 0, "text": {}, "ckpt": {}}
    out["val"][0] = mean_loss(p, val_rows, cfg, mode)
    out["text"][0] = greedy(p, cfg, [EOT], 24, tk)
    for step in range(o["steps"]):
        rows = [d["train"]["rows"][i] for i in order[step]]
        loss, grads = batch_loss(p, rows, cfg, mode, grad=True)
        norm = global_norm(grads)
        scale = min(1.0, o["clip"] / (norm + 1e-6))
        out["clipped"] += 1 if scale < 1.0 else 0
        adamw_update(ts, grads, mom, step + 1, lr_of(step), scale, decay, o)
        out["train"].append(loss)
        out["gnorm"].append(norm)
        done = step + 1
        if done % EVAL_EVERY == 0 or done in CHECKPOINTS:
            out["val"][done] = mean_loss(p, val_rows, cfg, mode)
        if done in CHECKPOINTS:
            out["text"][done] = greedy(p, cfg, [EOT], 24, tk)
    out["final"] = analyse(p, cfg, mode)
    return out


def alone_rows(docs, tk, T=ROW):
    """Every document as a row of its own (the layout of one example per row): inputs the separator and the document's
    tokens, targets the tokens and a closing separator, the rest of the row padding with a mask of 0."""
    rows = []
    for d in docs:
        t = [EOT] + tk.encode(d)
        k = len(t)
        rows.append({"x": t + [EOT] * (T - k), "y": t[1:] + [EOT] * (T - k + 1), "m": [1] * k + [0] * (T - k), "doc": [0] * k + [-1] * (T - k)})
    return rows


def attention_mass(p, cfg, rows):
    """Mean attention weight that a position of the packed rows puts on positions of another document, per (layer, head), over
    all real positions and over the positions that open a document; and the same mean if every visible position had the same
    weight. Returns (per head: all, opening), uniform (all, opening)."""
    L, H = cfg["L"], cfg["H"]
    acc = {(l, h): [0.0, 0, 0.0, 0] for l in range(L) for h in range(H)}
    uni = [0.0, 0, 0.0, 0]
    for r in rows:
        _, cache = forward(p, r["x"], cfg, causal)
        doc, x = r["doc"], r["x"]
        for i in range(len(doc)):
            if doc[i] < 0:
                continue
            opening = x[i] == EOT
            other = [j for j in range(i + 1) if doc[j] != doc[i]]
            share = len(other) / (i + 1)
            uni[0] += share
            uni[1] += 1
            if opening:
                uni[2] += share
                uni[3] += 1
            for l, c in enumerate(cache["caches"]):
                for h, hd in enumerate(c["ca"]["heads"]):
                    mass = _sum([hd["P"][i][j] for j in other])
                    a = acc[(l, h)]
                    a[0] += mass
                    a[1] += 1
                    if opening:
                        a[2] += mass
                        a[3] += 1
    heads = {k: (a[0] / a[1], a[2] / a[3]) for k, a in acc.items()}
    return heads, (uni[0] / uni[1], uni[2] / uni[3])


def analyse(p, cfg, mode):
    """What is measured on a trained model: the validation loss under each way of letting attention cross documents, with
    each document alone, the attention mass that crosses documents, and the loss under three loss denominators."""
    d, tk = data(), tokenizer()
    val = d["val"]["rows"]
    alone = alone_rows(val_docs(), tk)
    final = {"causal": mean_loss(p, val, cfg, "causal"), "block": mean_loss(p, val, cfg, "block"),
             "alone": mean_loss(p, alone, cfg, "causal")}
    final["mass"], final["mass_uniform"] = attention_mass(p, cfg, val)
    rows = val[-4:]
    total, positions = 0.0, 0
    every = 0.0
    supervised = 0
    for r in rows:
        logits, _ = forward(p, r["x"], cfg, causal)
        ce = losses(logits, r["y"])
        total += _sum([c * m for c, m in zip(ce, r["m"])])
        every += _sum(ce)
        supervised += sum(r["m"])
        positions += len(ce)
    final["demo"] = {"supervised": supervised, "positions": positions, "masked": total / supervised,
                     "diluted": total / positions, "every": every / positions}
    return final


# ------------------------------------------------------------------------------------------------ a copy task for the loss mask

COPY_ALPHABET, COPY_LEN = "abcd", 3
COPY_CHARS = list(COPY_ALPHABET) + ["=", "\n"]   # ids 0 .. 5
COPY_CFG = {"V": len(COPY_CHARS), "D": 16, "H": 2, "FF": 24, "L": 1, "T": 2 * COPY_LEN + 1}
COPY_TRAIN = {"B": 8, "steps": 600, "peak": 0.01, "floor": 0.001, "warmup": 10, "wd": 0.1, "clip": 1.0,
              "b1": 0.9, "b2": 0.95, "eps": 1e-8}
COPY_MASKS = {"answer": [0, 0, 0, 1, 1, 1, 1], "every": [1] * 7, "shifted": [0, 0, 0, 0, 1, 1, 1]}
COPY_LABEL = {"answer": "the answer targets", "every": "every target", "shifted": "answer targets, off by one"}
NAME_COPY_TRAIN, NAME_COPY_TEST, NAME_COPY_INIT = "training_gpt.copy.train", "training_gpt.copy.test", "training_gpt.copy.init"
N_COPY_TEST = 200


def copy_line(g):
    """One line S=S followed by a line feed: S is COPY_LEN draws of `integer(4)` (letters a .. d), in order."""
    s = [g.integer(len(COPY_ALPHABET)) for _ in range(COPY_LEN)]
    return s + [4] + s + [5]


def copy_row(line, mask):
    return {"x": line[:-1], "y": line[1:], "m": list(mask), "doc": [0] * (len(line) - 1)}


COPY_EVAL = (150, 300, 600)


def copy_evaluate(p):
    """(cross-entropy at each of the four answer targets, fraction of test lines whose four answer targets are all the
    largest logit), teacher forced, over the N_COPY_TEST lines of `NAME_COPY_TEST`."""
    gt = bookrng.generator(NAME_COPY_TEST)
    per = [0.0] * 4
    right = 0
    for _ in range(N_COPY_TEST):
        row = copy_row(copy_line(gt), COPY_MASKS["answer"])
        logits, _ = forward(p, row["x"], COPY_CFG, causal)
        ce = losses(logits, row["y"])
        ok = True
        for k in range(4):
            per[k] += ce[3 + k] / N_COPY_TEST
            z = logits[3 + k]
            ok = ok and max(range(len(z)), key=lambda j: (z[j], -j)) == row["y"][3 + k]
        right += 1 if ok else 0
    return per, right / N_COPY_TEST


def copy_run(job):
    """One copy-task run: (variant, nudge). Batches are 8 fresh lines per step from `NAME_COPY_TRAIN`, the same lines for every
    variant; only the loss mask differs. Returns the loss of the last 20 steps under the variant's own mask, and the
    held-out answer losses and exact-match fraction after COPY_EVAL steps."""
    variant, nudge = job
    cfg, o = COPY_CFG, COPY_TRAIN
    p = init_gpt(NAME_COPY_INIT, cfg)
    p["E"][0][0] += nudge
    ts = tensors(p)
    mom = [([[0.0] * len(r) for r in t], [[0.0] * len(r) for r in t]) for t in ts]
    decay = decayed(cfg)
    lr_of = schedules.cosine(o["peak"], o["floor"], o["warmup"], o["steps"])
    g = bookrng.generator(NAME_COPY_TRAIN)
    mask = COPY_MASKS[variant]
    seen, at = [], {}
    for step in range(o["steps"]):
        rows = [copy_row(copy_line(g), mask) for _ in range(o["B"])]
        loss, grads = batch_loss(p, rows, cfg, "causal", grad=True)
        norm = global_norm(grads)
        adamw_update(ts, grads, mom, step + 1, lr_of(step), min(1.0, o["clip"] / (norm + 1e-6)), decay, o)
        seen.append(loss)
        if step + 1 in COPY_EVAL:
            at[step + 1] = {"reported": _sum(seen[-20:]) / 20.0, "eval": copy_evaluate(p)}
    return {"variant": variant, "nudge": nudge, "at": at}


@lru_cache(maxsize=None)
def copy_results():
    jobs = [(v, n) for v in COPY_MASKS for n in (0.0, *stable.NUDGES)]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(copy_run, jobs, chunksize=1)
    out = {v: [None, []] for v in COPY_MASKS}
    for r in res:
        if r["nudge"] == 0.0:
            out[r["variant"]][0] = r
        else:
            out[r["variant"]].__getitem__(1).append(r)
    return out


def copy_cell(variant, f, digits=4):
    """A cell of the copy table: below 0.01 every run is written 'below 0.01' (the digits of such a loss are noise), otherwise
    the digits that survive the nudges."""
    base, twins = copy_results()[variant]
    vals = [f(base)] + [f(t) for t in twins]
    if all(v < 0.01 for v in vals):
        return "below 0.01"
    return stable.text(vals[0], vals[1:], lambda x: _fx(x, digits))


# ------------------------------------------------------------------------------------------------ running the experiments

MODES = ("causal", "block")
MODE_LABEL = {"causal": "attention crosses documents (causal mask of the row)", "block": "block-diagonal mask (one document at a time)"}


@lru_cache(maxsize=None)
def results():
    """{mode: (base run, [nudged runs])}: both ways of masking, each trained from the same parameters and the same batches
    and repeated from the four nudged starting points of the stable-digits rule; the runs are spread over four processes."""
    jobs = [(m, n) for m in MODES for n in (0.0, *stable.NUDGES)]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=1)
    out = {m: [None, []] for m in MODES}
    for r in res:
        if r["nudge"] == 0.0:
            out[r["mode"]][0] = r
        else:
            out[r["mode"]][1].append(r)
    return out


@lru_cache(maxsize=None)
def base_run(mode="causal"):
    """The un-nudged run alone, for figures, which need no stable digits."""
    return run((mode, 0.0))


def cell(mode, f, digits=4):
    base, twins = results()[mode]
    return stable.text(f(base), [f(t) for t in twins], lambda x: _fx(x, digits))


def window_loss(r, step, window=EVAL_EVERY):
    """Mean of the training batch losses of the `window` steps that end at `step` (a loss of one batch is noisy)."""
    xs = r["train"][max(0, step - window):step]
    return _sum(xs) / len(xs)


def text_cell(mode, step):
    base, twins = results()[mode]
    texts = {r["text"][step] for r in [base, *twins]}
    shown = re.sub(r" {4,}", lambda m: f"<{len(m.group())} spaces>", base["text"][step])
    return f"`{shown}`" if len(texts) == 1 else "differs between the nudged runs"


# ------------------------------------------------------------------------------------------------ packing


def packed_rows():
    tk, d = tokenizer(), data()["train"]
    r = d["rows"][1]
    T = len(r["x"])
    docs_in_row = sorted({x for x in r["doc"] if x >= 0})
    rows = [(i, f"`{tk.name(r['x'][i])}`", f"`{tk.name(r['y'][i])}`", r["m"][i], r["doc"][i]) for i in range(T)]
    spans = []
    for k in range(4):
        row = d["rows"][k]
        ds = sorted({x for x in row["doc"] if x >= 0})
        spans.append((k, k * T, f"{k * T + 1} .. {k * T + T}", ", ".join(str(x) for x in ds)))
    return (f"The training stream has {_int(len(d['ids']))} tokens: every document is its separator `EOT` followed by its tokens. "
            f"Rows have T = {T} positions. Row r takes the inputs `ids[{T} r .. {T} r + {T - 1}]` and the targets `ids[{T} r + 1 .. {T} r + {T}]`, "
            f"so each token except the first is a target exactly once and the stream gives ceil(({len(d['ids'])} - 1) / {T}) = "
            f"{len(d['rows'])} rows; the last row is padded with the separator and a loss mask of 0 (the loss mask holds a 0 or a 1 for every position: 1 where the model is scored on the target, 0 where it is not, as in the column m below). The first four rows touch these documents:\n\n"
            + _table(["row r", "first input at", "targets are tokens", "documents touched"], spans, "rrrl")
            + f"\n\nRow 1 position by position (`doc` is the index of the document the position belongs to; a position belongs to the "
            f"document whose separator or tokens it holds):\n\n"
            + _table(["position i", "input x", "target y", "mask m", "doc"], rows, "rllrr")
            + f"\n\nThe documents of row 1 are {docs_in_row[0]} and {docs_in_row[1]}: the row ends the first document, with the "
            f"target `EOT` at position {r['doc'].index(docs_in_row[1]) - 1}, and opens the second.")


def packing_waste():
    tk, d = tokenizer(), data()["train"]
    T = ROW
    alone = alone_rows(train_docs(), tk)
    a_pos = len(alone) * T
    a_sup = sum(sum(r["m"]) for r in alone)
    p_pos = len(d["rows"]) * T
    p_sup = sum(sum(r["m"]) for r in d["rows"])
    rows = [("one document per row, padded", len(alone), _int(a_pos), _int(a_sup), _fx(100 * (a_pos - a_sup) / a_pos, 1)),
            ("packed rows", len(d["rows"]), _int(p_pos), _int(p_sup), _fx(100 * (p_pos - p_sup) / p_pos, 1))]
    return ("Training documents of the corpus, rows of T = " + str(T) + " positions. A position is computed whether or not its target "
            "is supervised, so the share of positions with mask 0 is arithmetic that is run and thrown away. (The one-document-per-row "
            "layout also supervises a closing separator after each document, which the packed stream has only between documents.)\n\n"
            + _table(["layout", "rows", "positions computed", "supervised positions", "wasted positions, percent"], rows))


def block_mask_counts():
    tk, d = tokenizer(), data()["train"]
    r = d["rows"][1]
    T = len(r["x"])
    allowed = same_document(r["doc"])
    rows, tc, tb_ = [], 0, 0
    for i in range(T):
        nb = sum(1 for j in range(T) if allowed(i, j))
        tc += i + 1
        tb_ += nb
        rows.append((i, f"`{tk.name(r['x'][i])}`", r["doc"][i], i + 1, nb))
    return ("Row 1 of the training rows. For each query position i: how many keys it may read under the causal mask of the row "
            "(i + 1) and under the block-diagonal mask (the positions j <= i of the same document).\n\n"
            + _table(["i", "input", "doc", "keys visible, causal", "keys visible, block-diagonal"], rows, "rlrrr")
            + f"\n\nAllowed (query, key) pairs: {tc} causal, {tb_} block-diagonal, of {T * T} in the full square.")


# ------------------------------------------------------------------------------------------------ the architecture


def gelu_tanh(x):
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_forms():
    xs = [-3.0, -1.0, -0.5, 0.5, 1.0, 3.0]
    rows = [(_fx(x, 1), _fx(tb.gelu(x), 6), _fx(gelu_tanh(x), 6), _fx(gelu_tanh(x) - tb.gelu(x), 6)) for x in xs]
    grid = [k / 100.0 for k in range(-600, 601)]
    worst = max(grid, key=lambda x: abs(gelu_tanh(x) - tb.gelu(x)))
    return ("The exact GELU, x times the Gaussian cumulative distribution (library `erf`), and the tanh form 0.5 x (1 + tanh(sqrt(2/pi) "
            "(x + 0.044715 x^3))) that GPT-2 used (library `tanh`).\n\n"
            + _table(["x", "exact", "tanh form", "tanh form - exact"], rows)
            + f"\n\nOver x = -6.00, -5.99, ..., 6.00 the largest difference is {_fx(abs(gelu_tanh(worst) - tb.gelu(worst)), 6)}, at x = {_fx(worst, 2)}.")


V_GPT2, T_GPT2, D_GPT2, L_GPT2, F_GPT2, H_GPT2 = 50257, 1024, 768, 12, 3072, 12


def gpt2_block_parameters():
    D, F = D_GPT2, F_GPT2
    return [("two LayerNorms", "4 D", 4 * D), ("attention q, k, v", "3 D^2 + 3 D", 3 * D * D + 3 * D),
            ("attention output", "D^2 + D", D * D + D), ("feed-forward fc", "D F + F", D * F + F),
            ("feed-forward proj", "F D + D", F * D + D)]


def gpt_parameters():
    V, T, D, L = V_GPT2, T_GPT2, D_GPT2, L_GPT2
    per_block = gpt2_block_parameters()
    blk = sum(n for _, _, n in per_block)
    total_tied = V * D + T * D + L * blk + 2 * D
    rows = [("token embedding", "V D", _int(V * D)), ("position table", "T D", _int(T * D))]
    rows += [(f"{n} (each block)", f, _int(c)) for n, f, c in per_block]
    rows += [("one block", "", _int(blk)), (f"{L} blocks", f"{L} x", _int(L * blk)), ("final LayerNorm", "2 D", _int(2 * D)),
             ("total, head tied to the token embedding", "", _int(total_tied)),
             ("separate head", "V D", _int(V * D)), ("total, separate head", "", _int(total_tied + V * D))]
    cfg = config()
    p = init_gpt("training_gpt.count", cfg)
    Dt, Ff, Vt, Tt, Lt = cfg["D"], cfg["FF"], cfg["V"], cfg["T"], cfg["L"]
    b0 = p["blocks"][0]
    cnt = lambda t: sum(len(r) for r in t)
    blk_t = sum(cnt(b0[k]) for k in BLOCK_KEYS)
    trows = [("W_emb (token embedding, also the head)", f"{Vt} x {Dt}", _int(cnt(p["E"]))), ("W_pos (positions)", f"{Tt} x {Dt}", _int(cnt(p["P"])))]
    trows += [(n, f"{len(b0[k])} x {len(b0[k][0])}", _int(cnt(b0[k]))) for n, k in
              (("W_Q", "Wq"), ("W_K", "Wk"), ("W_V", "Wv"), ("W_O", "Wo"), ("W_gate", "Wg"), ("W_up", "Wu"), ("W_down", "Wd"))]
    trows += [("two RMS gains", f"2 x {Dt}", _int(cnt(b0["g1"]) + cnt(b0["g2"]))), ("one block", "", _int(blk_t)),
              (f"{Lt} blocks", f"{Lt} x", _int(Lt * blk_t)), ("final RMS gain", f"1 x {Dt}", _int(cnt(p["gf"])))]
    tiny_total = parameter_count(p)
    trows.append(("total, head tied to W_emb", "", _int(tiny_total)))
    return ("GPT-2 small as published (Radford and colleagues, 2019): vocabulary V = 50,257, T = 1,024, D = 768, L = 12 blocks, "
            "12 heads, F = 4 D = 3,072, LayerNorm with a gain and a shift, a GELU feed-forward network, biases on every linear map.\n\n"
            + _table(["part", "formula", "parameters"], rows, "llr")
            + f"\n\nThe tiny model of this chapter: V = {Vt}, T = {Tt}, D = {Dt}, L = {Lt}, {cfg['H']} heads, "
            f"SwiGLU of hidden width F = {Ff}, RMS norms (epsilon 1e-6) with a gain and no shift, no biases, the head tied to W_emb.\n\n"
            + _table(["tensor", "shape", "parameters"], trows, "llr")
            + f"\n\nA separate head would add V D = {_int(Vt * Dt)} parameters.")


def init_scales():
    cfg = config()
    rows_d = data()["train"]["rows"][:8]
    out = []
    for std in (0.02, 0.1, 0.5, 1.0):
        p = init_gpt("training_gpt.init_scale", cfg, std)
        total, n, sq = 0.0, 0, 0.0
        for r in rows_d:
            logits, _ = forward(p, r["x"], cfg, causal)
            ce = losses(logits, r["y"])
            total += _sum(ce)
            n += len(ce)
            sq += _sum([_sum([v * v for v in z]) / len(z) for z in logits])
        sd = math.sqrt(sq / n)
        out.append((_fx(std, 2), _fx(std * math.sqrt(cfg["D"]), 3), _fx(sd, 3), _fx(total / n, 4)))
    V = cfg["V"]
    blocks = []
    for L in (2, 12, 24, 48):
        blocks.append((L, 2 * L, _fx(1 / math.sqrt(2 * L), 4), _fx(0.02 / math.sqrt(2 * L), 5), _fx(2 * L * (1 / math.sqrt(2 * L)) ** 2, 1)))
    return (f"The tiny model ({_int(parameter_count(init_gpt('training_gpt.count', cfg)))} parameters, V = {V}) with every matrix drawn "
            f"as `std * normal()` from `training_gpt.init_scale` (seed {_seed('training_gpt.init_scale')}; the draw order is that of the "
            f"initialization section), no training. Loss: mean cross-entropy over the supervised positions of the first 8 training "
            f"rows. The final RMS norm gives a vector of root mean square 1 (squared length D), so a logit is a sum of D products "
            f"with an entry of W_emb and has standard deviation std sqrt(D); 'measured' is the root mean square of all logits.\n\n"
            + _table(["std", "std sqrt(D)", "measured logit rms", "initial loss"], out)
            + f"\n\nA model that guesses uniformly has loss ln V = {_fx(math.log(V), 4)}.\n\n"
            "Residual scaling: each of the L blocks writes twice to the stream (attention, feed-forward), so 2 L writes; the output "
            "matrices are multiplied by 1/sqrt(2 L) and the stream receives 2 L contributions of variance proportional to the square.\n\n"
            + _table(["blocks L", "writes 2 L", "factor 1/sqrt(2 L)", "std of an output matrix, 0.02 times the factor", "(2 L) times factor squared"], blocks))


# ------------------------------------------------------------------------------------------------ the gradient check

GC_CFG = {"V": 7, "D": 8, "H": 2, "FF": 12, "L": 2, "T": 6}
NAME_GC, NAME_GC_INIT = "training_gpt.check", "training_gpt.check.init"
GC_STEP = 1e-5


def _diffs(a, n):
    """(largest absolute difference, largest relative difference over the entries whose gradient is above 1e-3)."""
    worst_abs, worst_rel = 0.0, 0.0
    for ra, rn in zip(a, n):
        for x, y in zip(ra, rn):
            worst_abs = max(worst_abs, abs(x - y))
            if max(abs(x), abs(y)) > 1e-3:
                worst_rel = max(worst_rel, abs(x - y) / max(abs(x), abs(y)))
    return worst_abs, worst_rel


GROUPS = (("W_emb (token embedding and head)", "E"), ("W_pos (positions)", "P"), ("W_Q", "Wq"), ("W_K", "Wk"), ("W_V", "Wv"), ("W_O", "Wo"),
          ("W_gate", "Wg"), ("W_up", "Wu"), ("W_down", "Wd"), ("block gains", "g"), ("final gain", "gf"))


def _group_of(name):
    if name in ("E", "P", "gf"):
        return {"E": GROUPS[0][0], "P": GROUPS[1][0], "gf": GROUPS[10][0]}[name]
    if name.startswith("g"):
        return GROUPS[9][0]
    for label, pre in GROUPS[2:9]:
        if name.startswith(pre) and name[len(pre):].isdigit():
            return label
    raise KeyError(name)


def check_rows(cfg):
    """Two rows of T positions drawn from `training_gpt.check`: x and y are T draws of `integer(V)` each (x first). Row 0 has
    two documents (0 0 0 1 1 1) and every position supervised; row 1 has documents 0 0 1 1 and two padding positions
    (document -1, mask 0)."""
    g = bookrng.generator(NAME_GC)
    T, V = cfg["T"], cfg["V"]
    rows = []
    for doc, m in (([0, 0, 0, 1, 1, 1], [1] * 6), ([0, 0, 1, 1, -1, -1], [1, 1, 1, 1, 0, 0])):
        rows.append({"x": [g.integer(V) for _ in range(T)], "y": [g.integer(V) for _ in range(T)], "m": m, "doc": doc})
    return rows


@lru_cache(maxsize=None)
def gradient_results():
    cfg = dict(GC_CFG)
    rows = check_rows(cfg)
    out = {}
    for mode in MODES:
        p = init_gpt(NAME_GC_INIT, cfg, std=0.3)
        _, ana = batch_loss(p, rows, cfg, mode, grad=True)
        names = tensor_names(cfg)
        per = {label: [0, 0.0, 0.0] for label, _ in GROUPS}
        for name, t, a in zip(names, tensors(p), ana):
            num = []
            for r in range(len(t)):
                row = []
                for j in range(len(t[r])):
                    keep = t[r][j]
                    t[r][j] = keep + GC_STEP
                    hi = batch_loss(p, rows, cfg, mode)[0]
                    t[r][j] = keep - GC_STEP
                    lo = batch_loss(p, rows, cfg, mode)[0]
                    t[r][j] = keep
                    row.append((hi - lo) / (2 * GC_STEP))
                num.append(row)
            wa, wr = _diffs(a, num)
            e = per[_group_of(name)]
            e[0] += sum(len(r) for r in t)
            e[1], e[2] = max(e[1], wa), max(e[2], wr)
        out[mode] = per
    # the block of this module with the causal mask against the block of Chapter 9
    g = bookrng.generator("training_gpt.block")
    b = tb.init_block(g, 8, 12)
    X = [[g.normal() for _ in range(8)] for _ in range(6)]
    mine, _ = block(b, X, 2, causal)
    ref, _ = tb.block_forward(b, X, "pre", 2)
    out["block_difference"] = max(abs(x - y) for r1, r2 in zip(mine, ref) for x, y in zip(r1, r2))
    return out


def gradient_check():
    res = gradient_results()
    cfg = GC_CFG
    rows = []
    for label, _ in GROUPS:
        a = res["causal"][label]
        b = res["block"][label]
        rows.append((label, _int(a[0]), tb._under_criterion(a[1]), tb._under_criterion(a[2]), tb._under_criterion(b[1]), tb._under_criterion(b[2])))
    return (f"A GPT with V = {cfg['V']}, T = {cfg['T']}, D = {cfg['D']}, {cfg['H']} heads, F = {cfg['FF']}, L = {cfg['L']}, every matrix "
            f"`0.3 * normal()` from `{NAME_GC_INIT}` ({_seed(NAME_GC_INIT)}), in the draw order of the initialization section, with W_O and W_down of "
            f"each block then multiplied by 1/sqrt(2 L) = {1 / math.sqrt(2 * cfg['L']):.4f} as in the main run (so their entries have standard "
            f"deviation {0.3 / math.sqrt(2 * cfg['L']):.2f}); all gains 1 and every RMS norm at epsilon 1e-6. Two rows "
            f"from `{NAME_GC}` ({_seed(NAME_GC)}): x (6 draws of `integer(7)`) then y (6 draws), for row 0 then row 1; row 0 has documents "
            "0 0 0 1 1 1 and mask 1 everywhere; row 1 has documents 0 0 1 1 and two padding positions (document -1, mask 0). The "
            "loss is the mean cross-entropy over the 10 supervised positions. Central differences with step "
            f"{GC_STEP:g} on every entry of every parameter. A cell says 'below 1e-6' when the largest difference meets the criterion of 1e-6 "
            "(the digits of a smaller difference are finite-difference noise); the absolute difference is |a - n|, the relative difference "
            "|a - n| / max(|a|, |n|) is taken over entries whose gradient is above 1e-3.\n\n"
            + _table(["parameters", "entries", "causal: absolute", "causal: relative",
                      "block-diagonal: absolute", "block-diagonal: relative"], rows)
            + f"\n\nThe forward pass of this module's block with the causal mask against the block of Chapter 9 on a random "
            f"6 by 8 input: the largest absolute difference is {_fx(res['block_difference'], 1)}.")


# ------------------------------------------------------------------------------------------------ the training run

SHOW_STEPS = (10, 20, 50, 100, 150, 200, 250, 300)


def setup_text():
    o, cfg = TRAIN, config()
    d = data()
    return (f"Model: V = {cfg['V']}, T = {cfg['T']}, D = {cfg['D']}, L = {cfg['L']}, {cfg['H']} heads, SwiGLU width {cfg['FF']}, initialized "
            f"from `{NAME_INIT}` ({_seed(NAME_INIT)}) with std {INIT_STD} (output matrices times 1/sqrt(2 L)). Batches: {o['B']} rows per step "
            f"in shuffled epochs of the {len(d['train']['rows'])} training rows (one `shuffle` of 0 .. {len(d['train']['rows']) - 1} per epoch from "
            f"`{NAME_ORDER}`, {_seed(NAME_ORDER)}; consecutive groups of {o['B']}; an incomplete last group is dropped), the same batches for every "
            f"run below. {o['steps']} steps of AdamW (beta1 {o['b1']}, beta2 {o['b2']}, epsilon {o['eps']:g}, weight decay {o['wd']} on every "
            f"matrix except the gains), the learning rate warming up linearly over {o['warmup']} steps to {o['peak']} and following half a cosine "
            f"to {o['floor']} at step {o['steps']} (`schedules.cosine`, 0-based: step s uses the rate at index s - 1), the global gradient norm "
            f"clipped to {o['clip']} by the factor min(1, {o['clip']} / (norm + 1e-6)); the 'gradient norm' column is that norm for the step that ends at "
            f"the row, before the clip (the square root of the sum of the squares of every entry of every gradient, the gains and the position "
            f"table included), and the 'learning rate' column is the rate of that step's update. Every RMS norm has epsilon 1e-6 and the "
            f"gradient is that of the mean over the supervised positions of the 8 rows of the step. Validation: every one of the {len(d['val']['rows'])} "
            f"validation rows, after every {EVAL_EVERY} steps. A training loss is the mean of the batch losses of the {EVAL_EVERY} steps that end "
            "at the step. " + stable.RULE)


def training_curve():
    o = TRAIN
    lr_of = schedules.cosine(o["peak"], o["floor"], o["warmup"], o["steps"])
    rows = [(0, "-", "-", "-", cell("causal", lambda r: r["val"][0]), cell("causal", lambda r: math.exp(r["val"][0]), 3))]
    for s in SHOW_STEPS:
        rows.append((s, _fx(lr_of(s - 1), 5), cell("causal", lambda r, s=s: r["gnorm"][s - 1]),
                     cell("causal", lambda r, s=s: window_loss(r, s)), cell("causal", lambda r, s=s: r["val"][s]),
                     cell("causal", lambda r, s=s: math.exp(r["val"][s]), 3)))
    base, twins = results()["causal"]
    clipped = {r["clipped"] for r in [base, *twins]}
    best = lambda r: min(r["val"], key=lambda s: (r["val"][s], s))
    bests = {best(r) for r in [base, *twins]}
    d = data()
    return (setup_text() + "\n\n"
            + _table(["step", "learning rate", "gradient norm", "training loss", "validation loss", "validation perplexity"], rows)
            + f"\n\nThe clip engaged in {sorted(clipped)[0] if len(clipped) == 1 else 'a varying number of'} of the {o['steps']} steps. "
            f"Validation perplexity is exp of the validation loss. A model that guesses uniformly has loss ln V = {_fx(math.log(config()['V']))}; "
            f"the entropy of the generator is {_fx(corpus_floor('val'))} nats per target token on the validation stream "
            f"(perplexity {_fx(math.exp(corpus_floor('val')), 3)}). The lowest validation loss of the steps evaluated is "
            f"{cell('causal', lambda r: min(r['val'].values()))}, at step {sorted(bests)[0] if len(bests) == 1 else 'a step that differs between the runs'}.")


def training_curve_data():
    """(steps, train loss window means, validation steps, validation losses) of the base run, for the figure."""
    r = base_run("causal")
    steps = list(range(EVAL_EVERY, TRAIN["steps"] + 1, EVAL_EVERY))
    vs = sorted(r["val"])
    return steps, [window_loss(r, s) for s in steps], vs, [r["val"][s] for s in vs]


def generations():
    rows = []
    for s in CHECKPOINTS:
        rows.append((s, cell("causal", lambda r, s=s: r["val"][s]), text_cell("causal", s)))
    return ("Greedy generation from the separator: at each step the token with the largest logit (the first on a tie) of the last position, "
            f"the context cropped to the last {ROW} tokens, stopping at the first separator or after 24 tokens. Parameters saved at the "
            "listed step of the run of the previous table; the text is printed when the run and its four nudged copies agree on it.\n\n"
            + _table(["step", "validation loss", "text generated"], rows, "rrl"))


def mask_choice():
    rows = []
    for mode in MODES:
        rows.append((MODE_LABEL[mode], cell(mode, lambda r: r["final"]["causal"]), cell(mode, lambda r: r["final"]["block"]),
                     cell(mode, lambda r: r["final"]["alone"])))
    head = [("all positions", 0), ("document-opening positions", 1)]
    mrows = []
    cfg = config()
    base, twins = results()["causal"]
    for l in range(cfg["L"]):
        for h in range(cfg["H"]):
            mrows.append((f"layer {l}, head {h}",
                          *[stable.text(base["final"]["mass"][(l, h)][k], [t["final"]["mass"][(l, h)][k] for t in twins], lambda x: _fx(x, 3))
                            for k in (0, 1)]))
    mrows.append(("uniform weights, for comparison", *[_fx(base["final"]["mass_uniform"][k], 3) for k in (0, 1)]))
    return ("Validation loss (mean cross-entropy per target token) of the two models of the previous section after 300 steps, "
            "evaluated three ways: on the packed validation rows with the causal mask of the row, on the same rows with the "
            "block-diagonal mask, and with each validation document alone in a row (positions from 0). "
            + stable.RULE + "\n\n"
            + _table(["trained with", "evaluated on packed rows, causal", "packed rows, block-diagonal", "each document alone"], rows, "lrrr")
            + "\n\nFor the model trained with the causal mask, the attention weight that a position puts on positions of an earlier "
            "document (mean over the validation rows), per layer and head, over all positions and over the positions that hold a "
            "document's opening separator; the last row is the same mean if every visible position had the same weight. For a position i of a "
            "validation row (padding positions excluded) the mass is the sum, over the keys j <= i that belong to another document, of the "
            "attention weight of that head and layer; a cell is the mean of that mass over all the real positions of all the validation rows "
            "(first column) or over those whose input token is the separator (second column), each position counting once. The uniform row "
            "uses the same positions with the mass (number of other-document keys) / (i + 1).\n\n"
            + _table(["", "mass on other documents, all positions", "mass on other documents, opening positions"], mrows, "lrr"))


def loss_denominator():
    base, twins = results()["causal"]
    d = base["final"]["demo"]
    rows = [("mean over the supervised positions (the loss)", cell("causal", lambda r: r["final"]["demo"]["masked"]), "sum / supervised"),
            ("sum over supervised positions divided by all positions", cell("causal", lambda r: r["final"]["demo"]["diluted"]), "sum / (B T)"),
            ("every position counted, padding targets included", cell("causal", lambda r: r["final"]["demo"]["every"]), "(sum + padding) / (B T)")]
    return (f"The model after 300 steps on the last 4 validation rows: B T = {d['positions']} positions of which {d['supervised']} are supervised "
            f"(the last row carries {d['positions'] - d['supervised']} positions of padding). Three denominators for the same per-position "
            "cross-entropies.\n\n" + _table(["quantity", "value", "formula"], rows, "lrl"))


def copy_task():
    rows = []
    for v in COPY_MASKS:
        for s in COPY_EVAL:
            per = [copy_cell(v, lambda r, s=s, k=k: r["at"][s]["eval"][0][k], 3) for k in range(4)]
            rows.append((COPY_LABEL[v] if s == COPY_EVAL[0] else "", s, copy_cell(v, lambda r, s=s: r["at"][s]["reported"], 4),
                         *per, copy_cell(v, lambda r, s=s: r["at"][s]["eval"][1], 3)))
    o = COPY_TRAIN
    return (f"Lines `S=S` and a line feed, S three letters from `abcd` (draws: three `integer(4)` per line, in order, from `{NAME_COPY_TRAIN}`, "
            f"{_seed(NAME_COPY_TRAIN)}); ids a .. d = 0 .. 3, `=` = 4, line feed = 5. A row is the 7 inputs `S = S` and the 7 targets that follow, so the "
            "targets are s2 s3 = a1 a2 a3 line-feed. Model: V = 6, T = 7, D = 16, one block, 2 heads, SwiGLU width 24, from "
            f"`{NAME_COPY_INIT}` ({_seed(NAME_COPY_INIT)}), the same start for every row below, drawn as in the main run: standard deviation "
            f"{INIT_STD} for every matrix, W_O and W_down times 1/sqrt(2 L) = {1 / math.sqrt(2 * COPY_CFG['L']):.4f}, the table W_emb [6, 16] "
            "also the head, a position table of 7 rows, all gains 1, RMS norms at epsilon 1e-6. "
            f"{o['B']} fresh lines per step (a batch of 8 rows, loss and gradient averaged over the counted targets of the batch), {o['steps']} steps "
            f"of AdamW (beta1 {o['b1']}, beta2 {o['b2']}, epsilon {o['eps']:g}, weight decay {o['wd']} on every matrix except the gains, peak {o['peak']}, "
            f"floor {o['floor']}, warmup {o['warmup']}, half a cosine over the {o['steps']} steps, clip {o['clip']} by the factor min(1, {o['clip']} / "
            "(norm + 1e-6)): the settings of the main run with these steps). The three masks "
            "differ only in which of the 7 targets count (1 = counted): 'every target' 1 1 1 1 1 1 1; 'the answer targets' (the four after the "
            "input `=`, a1 a2 a3 and the line feed) 0 0 0 1 1 1 1; 'off by one' (the mask of the answer tokens, which are tokens 4 to 7 of the "
            "line, laid on the input positions 4 to 6 instead of on the target positions 3 to 6) 0 0 0 0 1 1 1. 'Reported loss' is the training loss of the last 20 steps under the run's own mask. The "
            f"other columns are measured on {N_COPY_TEST} held-out lines of `{NAME_COPY_TEST}` ({_seed(NAME_COPY_TEST)}), teacher forced, whatever "
            "the mask: the cross-entropy at the four answer targets (a1, a2, a3, line feed) and the fraction of lines whose four answer targets "
            "all have the largest logit. Guessing a letter costs ln 4 = " + _fx(math.log(4), 3) + ". " + stable.RULE + "\n\n"
            + _table(["mask", "step", "reported loss", "a1", "a2", "a3", "line feed", "answers all right"], rows, "lrrrrrrr"))


# ------------------------------------------------------------------------------------------------ training as work

FP32 = 4   # bytes per number


def count_macs(n, a, b):
    """Multiply-adds of the three products of one linear map Y = X W, W [a, b], on n rows, counted by running the loops:
    the forward Y = X W, the weight gradient dW = X^T dY and the input gradient dX = dY W^T."""
    fwd = sum(1 for i in range(n) for j in range(b) for k in range(a))
    dw = sum(1 for k in range(a) for j in range(b) for i in range(n))
    dx = sum(1 for i in range(n) for k in range(a) for j in range(b))
    return fwd, dw, dx


def linear_flops():
    n, a, b = 5, 4, 3
    f, w, x = count_macs(n, a, b)
    rows = [("forward Y = X W", f, 2 * f), ("weight gradient dW = X^T dY", w, 2 * w), ("input gradient dX = dY W^T", x, 2 * x),
            ("all three", f + w + x, 2 * (f + w + x))]
    cfg = config()
    D, F, tok = cfg["D"], cfg["FF"], TRAIN["B"] * cfg["T"]
    return (f"One linear map with W of shape [{a}, {b}] ({a * b} parameters) on n = {n} rows, a multiply-add counted as two FLOPs, the "
            "multiply-adds counted by running the loops of each product.\n\n"
            + _table(["product", "multiply-adds", "FLOPs"], rows)
            + f"\n\nSo the three products cost {2 * (f + w + x)} = 6 n a b = 6 x {n} x {a * b} FLOPs: six times the number of rows times the "
            f"number of parameters of the map. For the W_gate of the tiny model ([{D}, {F}], {D * F} parameters) on the {tok} tokens of one "
            f"step ({TRAIN['B']} rows of {cfg['T']}) that is 6 x {tok} x {D * F} = {_int(6 * tok * D * F)} FLOPs.")


def tiny_ops(cfg):
    """(name, shape text, forward FLOPs of one row of T tokens) of every matrix product of the tiny model."""
    T, D, F, H, L, V = cfg["T"], cfg["D"], cfg["FF"], cfg["H"], cfg["L"], cfg["V"]
    dk = D // H
    per_layer = [("W_Q, W_K, W_V", f"3 x [{T},{D}] x [{D},{D}]", 3 * 2 * T * D * D),
                 ("scores Q K^T", f"{H} x [{T},{dk}] x [{dk},{T}]", H * 2 * T * T * dk),
                 ("weights times V", f"{H} x [{T},{T}] x [{T},{dk}]", H * 2 * T * T * dk),
                 ("W_O", f"[{T},{D}] x [{D},{D}]", 2 * T * D * D),
                 ("W_gate, W_up", f"2 x [{T},{D}] x [{D},{F}]", 2 * 2 * T * D * F),
                 ("W_down", f"[{T},{F}] x [{F},{D}]", 2 * T * F * D)]
    return per_layer, ("head logits = xn W_emb^T", f"[{T},{D}] x [{D},{V}]", 2 * T * D * V)


def flops_per_token_formula(V, D, F, L, T, mlp_matrices, attention_full=True):
    """The forward FLOPs per token: twice the parameters of the matrix products plus 4 L T D for the two attention products."""
    n = L * (4 * D * D + mlp_matrices * D * F) + V * D
    return n, 2 * n + 4 * L * T * D


def tiny_flops():
    cfg = config()
    T, D, F, L, V = cfg["T"], cfg["D"], cfg["FF"], cfg["L"], cfg["V"]
    per_layer, head = tiny_ops(cfg)
    layer_total = sum(f for _, _, f in per_layer)
    rows = [(n, s, _int(f)) for n, s, f in per_layer]
    rows += [("one block", "", _int(layer_total)), (f"{L} blocks", "", _int(L * layer_total)), (head[0], head[1], _int(head[2]))]
    total = L * layer_total + head[2]
    n, per_tok = flops_per_token_formula(V, D, F, L, T, 3)
    assert total == per_tok * T, "the counted FLOPs must equal 2 N + 4 L T D per token"
    train_tok = 3 * per_tok
    attn = 12 * L * T * D
    return (f"Forward FLOPs of the matrix products of the tiny model on one row of T = {T} tokens (V = {V}, D = {D}, F = {F}, L = {L}, "
            "2 heads), a multiply-add counted as two FLOPs, the whole T by T square of scores counted. Norms, the softmax, the SwiGLU "
            "gate, the embedding lookup and the residual adds are not matrix products and are left out.\n\n"
            + _table(["product", "shapes", "forward FLOPs per row"], rows, "llr")
            + f"\n\nTotal {_int(total)} FLOPs per row, {_int(per_tok)} per token. The parameters that take part in a matrix product are "
            f"N = L (4 D^2 + 3 D F) + V D = {_int(n)} (the position table and the gains do not); then 2 N = {_int(2 * n)} and the attention "
            f"products add 4 L T D = {_int(4 * L * T * D)}, and {_int(2 * n)} + {_int(4 * L * T * D)} = {_int(per_tok)}. Training costs three times "
            f"the forward: {_int(train_tok)} FLOPs per token, of which 6 N = {_int(6 * n)} and the attention term 12 L T D = {_int(attn)}, "
            f"{_fx(100 * attn / train_tok, 1)} percent of the total.")


def gpt2_flops_per_token():
    V, T, D, L, F = V_GPT2, T_GPT2, D_GPT2, L_GPT2, F_GPT2
    n, fwd = flops_per_token_formula(V, D, F, L, T, 2)
    return n, fwd, 3 * fwd


def gpt2_budget():
    V, T, D, L, F = V_GPT2, T_GPT2, D_GPT2, L_GPT2, F_GPT2
    n, fwd, train = gpt2_flops_per_token()
    peak_gflops, bw_gbs = roofline.MACHINES["GPU"]
    peak = peak_gflops * 1e9
    batch_rows, tokens_step, budget = 8, 8 * T, 10 ** 10
    rows = [("parameters in matrix products, N = L (4 D^2 + 2 D F) + V D", _int(n)),
            ("forward FLOPs per token, 2 N + 4 L T D", _int(fwd)),
            ("of which the attention products, 4 L T D", _int(4 * L * T * D)),
            ("training FLOPs per token, three times the forward", _int(train)),
            ("6 N alone", _int(6 * n)),
            ("training FLOPs per token over 6 N", _fx(train / (6 * n), 4)),
            (f"training FLOPs of one step of {batch_rows} rows of {T} = {_int(tokens_step)} tokens", _sci(train * tokens_step, 3)),
            (f"training FLOPs of a budget of {_int(budget)} tokens", _sci(train * budget, 3))]
    total = train * budget
    util = []
    for u in (0.1, 0.3, 0.5):
        secs = total / (u * peak)
        util.append((_fx(u, 1), _sci(u * peak, 2), _fx(secs / 86400.0, 2)))
    return (f"GPT-2 small (V = {_int(V)}, T = {_int(T)}, D = {D}, L = {L}, 12 heads, F = {_int(F)}, a GELU network of two matrices, the head "
            "counted as a matrix product with the embedding table). The token budget is a stated example and not the budget of any published run.\n\n"
            + _table(["quantity", "value"], rows, "lr")
            + f"\n\nThe model device of Volume IV has a peak of {_sci(peak, 2)} FLOP per second and a memory bandwidth of {_sci(bw_gbs * 1e9, 2)} bytes per "
            "second. Model FLOPs utilization is the training FLOPs per second the run achieves divided by that peak, with the FLOPs counted as above "
            "(a recomputed forward is not counted). The time is what the arithmetic needs at that utilization on that device, a consequence of "
            "the counts and not a measurement.\n\n"
            + _table(["utilization", "achieved FLOP per second", "device-days for the budget"], util))


def intensity_vs_tokens():
    V, T, D, L, F = V_GPT2, T_GPT2, D_GPT2, L_GPT2, F_GPT2
    n, _, train = gpt2_flops_per_token()
    peak_g, bw_g = roofline.MACHINES["GPU"]
    ridge = peak_g / bw_g
    rows = []
    for tokens in (64, 512, 8192, 65536):
        intensity = tokens / 2.0
        rows.append((_int(tokens), _fx(intensity, 1), _fx(min(1.0, intensity / ridge), 3), _fx(28.0 * ridge / (6.0 * tokens), 4)))
    return ("Weights are read once for the forward products and once for the input gradients and a gradient is written once, 3 passes "
            f"of 4 bytes per parameter (fp32): 12 bytes per parameter, for 6 FLOPs per parameter per token. With n tokens in the step the "
            f"intensity of the matrix products is 6 n P / 12 P = n / 2 FLOP per byte (activation traffic left out). The ridge of the model "
            f"device is {_fx(ridge, 0)} FLOP per byte. The optimizer reads the weight, the gradient and the two moments and writes the weight and "
            "the moments: 7 numbers of 4 bytes, 28 bytes per parameter per step whatever n is; the last column is its bytes divided by the "
            "bandwidth over the matrix FLOPs divided by the peak, 28 x ridge / (6 n).\n\n"
            + _table(["tokens per step n", "intensity of the matrix products, FLOP per byte", "attainable fraction of peak", "optimizer time over matrix-product time"], rows))


def training_memory():
    V, T, D, L, F, H = V_GPT2, T_GPT2, D_GPT2, L_GPT2, F_GPT2, H_GPT2
    tied = V * D + T * D + L * sum(n for _, _, n in gpt2_block_parameters()) + 2 * D
    untied = tied + V * D
    layer = [("residual entering the block", D), ("LayerNorm 1 output", D), ("q, k, v", 3 * D), ("attention scores", H * T), ("attention weights", H * T),
             ("attention output before W_O", D), ("residual after attention", D), ("LayerNorm 2 output", D), ("feed-forward pre-activation", F), ("GELU output", F)]
    per_layer = sum(n for _, n in layer)
    tail = [("last residual", D), ("final LayerNorm output", D), ("logits", V), ("softmax statistics and per-position loss", 3)]
    per_token = L * per_layer + sum(n for _, n in tail)
    tokens = 8 * T
    boundaries = (L + 1) * D
    recompute = boundaries + (per_layer - D) + sum(n for _, n in tail[1:])
    rows = [(n, _int(c), _int(L * c)) for n, c in layer] + [("one block", _int(per_layer), _int(L * per_layer))]
    rows += [(n, _int(c), _int(c)) for n, c in tail]
    rows.append(("all, kept for the backward pass", "", _int(per_token)))
    gb = lambda b: _fx(b / 1e9, 3)
    state = [("parameters", 1, tied, untied), ("gradients", 1, tied, untied), ("Adam moments m and v", 2, tied, untied)]
    srows = [(n, gb(k * FP32 * a), gb(k * FP32 * b)) for n, k, a, b in state]
    srows.append(("weights, gradients and optimizer state", gb(4 * FP32 * tied), gb(4 * FP32 * untied)))
    arows = [(f"activations kept, {_int(tokens)} tokens", gb(per_token * tokens * FP32), gb(per_token * tokens * FP32)),
             (f"activations with recomputation, {_int(tokens)} tokens", gb(recompute * tokens * FP32), gb(recompute * tokens * FP32))]
    return ("Floats kept per token for the backward pass by the layer of `crates/gpt2` (the buffers of its `Layer` and the residual snapshots), "
            f"GPT-2 small, T = {T}: the score and weight tensors are H T numbers per token each.\n\n"
            + _table(["tensor", "floats per token per block", "floats per token, all blocks"], rows)
            + f"\n\nWith recomputation only the {L + 1} residual snapshots are kept and one block's buffers are rebuilt before its backward: "
            f"{_int(boundaries)} + {_int(per_layer - D)} + {_int(sum(n for _, n in tail[1:]))} = {_int(recompute)} floats per token, against "
            f"{_int(per_token)}, at the price of one more forward: 8 N instead of 6 N FLOPs per token (a factor {_fx(8 / 6, 3)}, plus the attention "
            "term). Memory in gigabytes of 10^9 bytes, fp32 everywhere.\n\n"
            + _table(["", "head tied to the embedding (124,439,808 parameters)", f"separate head ({_int(untied)} parameters)"], srows + arows, "lrr"))


# ------------------------------------------------------------------------------------------------ checkpoints and resuming

RESUME_AT, RESUME_TO = 20, 40


def advance(p, mom, order, first, last, counter):
    """Train steps first .. last - 1 of the batch order from the state (p, mom). `counter` is the optimizer's step counter at
    step `first`: it sets both the learning rate of the schedule and the bias correction of Adam, and advances with the steps."""
    cfg, o, d = config(), TRAIN, data()
    lr_of = schedules.cosine(o["peak"], o["floor"], o["warmup"], o["steps"])
    ts, decay = tensors(p), decayed(cfg)
    for k, step in enumerate(range(first, last)):
        rows = [d["train"]["rows"][i] for i in order[step]]
        _, grads = batch_loss(p, rows, cfg, "causal", grad=True)
        scale = min(1.0, o["clip"] / (global_norm(grads) + 1e-6))
        adamw_update(ts, grads, mom, counter + k + 1, lr_of(counter + k), scale, decay, o)


def zero_moments(p):
    return [([[0.0] * len(r) for r in t], [[0.0] * len(r) for r in t]) for t in tensors(p)]


@lru_cache(maxsize=None)
def resume_results():
    import copy
    cfg, o, d = config(), TRAIN, data()
    order = batch_order(len(d["train"]["rows"]), o["B"], o["steps"])
    p = init_gpt(NAME_INIT, cfg)
    mom = zero_moments(p)
    advance(p, mom, order, 0, RESUME_AT, 0)
    saved_p, saved_m = copy.deepcopy(p), copy.deepcopy(mom)
    straight_p, straight_m = copy.deepcopy(p), copy.deepcopy(mom)
    advance(straight_p, straight_m, order, RESUME_AT, RESUME_TO, RESUME_AT)
    variants = [("weights, moments and step counter", copy.deepcopy(saved_p), copy.deepcopy(saved_m), RESUME_AT),
                ("weights and step counter, moments set to zero", copy.deepcopy(saved_p), zero_moments(saved_p), RESUME_AT),
                ("weights and moments, step counter restarted at 0", copy.deepcopy(saved_p), copy.deepcopy(saved_m), 0),
                ("weights only: moments zero and step counter restarted", copy.deepcopy(saved_p), zero_moments(saved_p), 0)]
    out = []
    val = d["val"]["rows"]
    ref_val = mean_loss(straight_p, val, cfg)
    for label, vp, vm, counter in variants:
        advance(vp, vm, order, RESUME_AT, RESUME_TO, counter)
        diff = max(abs(a - b) for ta, tbb in zip(tensors(vp), tensors(straight_p)) for ra, rb in zip(ta, tbb) for a, b in zip(ra, rb))
        out.append((label, diff, mean_loss(vp, val, cfg)))
    return ref_val, out


def checkpoint_resume():
    ref_val, rows = resume_results()
    table = [("never stopped (reference)", "0", _fx(ref_val, 4))]
    table += [(label, "0" if diff == 0.0 else tb._bound(diff), _fx(v, 4)) for label, diff, v in rows]
    o = TRAIN
    return (f"The run of the main experiment (same parameters, same batch order, steps 1 .. {RESUME_TO}) is stopped after step {RESUME_AT} and "
            f"continued from what a checkpoint holds. The step counter is the number of applied optimizer steps: it fixes the learning rate "
            f"(warmup {o['warmup']}, then the cosine to step {o['steps']}) and Adam's bias correction; the batches continue at step {RESUME_AT + 1} "
            "in every case. 'Largest difference' is the largest absolute difference of any parameter from the reference after step "
            f"{RESUME_TO}, as the power of ten it stays below (0: identical); the validation loss is the mean over all validation rows.\n\n"
            + _table(["resumed from", "largest parameter difference below", f"validation loss after step {RESUME_TO}"], table))
