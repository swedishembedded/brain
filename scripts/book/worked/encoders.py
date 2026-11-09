# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Bidirectional Encoders and Embedding Models": models that read a whole text at once.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. The pieces of earlier chapters are imported, never copied: the greedy WordPiece matcher of
Chapter 2 (`tokenization`), the transformer block with an arbitrary attention mask and its backward pass of Chapters 7 to 10
(`training_gpt.block`, `forward`, `backward`, which call `transformer_block` and `self_attention`), the AdamW step and the
learning-rate curve of Volume III, and the power-iteration principal directions of Chapter 3 (`embeddings`). What makes an
encoder is written here:

* two toy languages with known entropy (a barnyard sentence in which the verb gives the noun away, and a language of
  synonyms in which two sentences share a meaning and no content word) and the exact bag-of-words baseline on the second;
* the full and the causal attention mask side by side, and the proof that a causal row cannot read what is to its right;
* BERT's masking rule (selection, then 80 percent mask, 10 percent random token, 10 percent unchanged), a masked language
  model trained with it, and a causal model trained on the same sentences, read against the exact floors of the language;
* padding in a bidirectional encoder (an unmasked pad key changes every real position), CLS and mean pooling;
* InfoNCE with in-batch negatives and a temperature, its gradient through the cosine, a contrastive run that is repeated
  from nudged starting points, retrieval with recall at k, and a linear head on frozen vectors;
* the arithmetic of encoding as work (FLOPs and weight bytes of one pass, the band of a local attention layer, the cost of
  scanning an index) and the two brain details that are pure arithmetic: T5's relative-position buckets and the scale at which
  a gate that is swapped with its up projection cannot be seen.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints its seeds. Sums of
products are plain left-to-right additions, the order a C loop would use. Numbers of trained models are printed by the
stable-digits rule of `stable.py`: each run is repeated from five starting points that differ by 1e-12 in one weight.

Swedish Embedded AB implements solutions for search and classification models that read a whole text at once, from the
masking recipe to the retrieval metric. If your team needs expertise in training and evaluating encoders and embedding models,
you can procure our services by sending an email to info@swedishembedded.com.
"""
import math
import struct
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import embeddings as emb
import schedules
import self_attention as sa
import stable
import tokenization as tok
import training_gpt as tg
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


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


def _sci(x, d=2):
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}e{int(e)}"


def _tick(b):
    return "1" if b else "0"


def _dot(a, b):
    return _sum([x * y for x, y in zip(a, b)])


def _cos(a, b):
    return _dot(a, b) / math.sqrt(_dot(a, a) * _dot(b, b))


def _unit(v):
    n = math.sqrt(_dot(v, v))
    return [x / n for x in v], n


def _softmax_list(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = _sum(e)
    return [v / s for v in e]


# ------------------------------------------------------------------------------------------------ language A: the barnyard

NOUNS_A = ["cat", "dog", "hen", "owl", "cow", "fox"]
STEMS_A = ["purr", "bark", "cluck", "hoot", "moo", "yelp"]        # the verb of noun k is STEMS_A[k] + "ed"
PLACES_A = ["mat", "barn", "pond", "hill"]
SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
WP_A = SPECIALS + ["the", "near", "."] + NOUNS_A + STEMS_A + ["##ed"] + PLACES_A
ID_A = {w: i for i, w in enumerate(WP_A)}
PAD, CLS, SEP, MASK = (ID_A[w] for w in SPECIALS[:4])
V_A = len(WP_A)
T_A = 10                       # [CLS] the NOUN STEM ##ed near the PLACE . [SEP]
CFG_A = {"V": V_A, "D": 16, "H": 2, "FF": 24, "L": 1, "T": T_A}
NAME_A_INIT, NAME_A_SENT, NAME_A_MASK = "encoders.mlm.init", "encoders.mlm.sentences", "encoders.mlm.mask"
SLOT_NAMES = {1: "the", 2: "noun", 3: "verb stem", 4: "##ed", 5: "near", 6: "the", 7: "place", 8: "full stop"}
FULL = lambda i, j: True   # noqa: E731  (every position may read every other: the encoder's mask)
CAUSAL = lambda i, j: j <= i   # noqa: E731


def words_a(noun, place):
    return ["the", NOUNS_A[noun], STEMS_A[noun] + "ed", "near", "the", PLACES_A[place], "."]


def ids_a(noun, place):
    """The ids of one barnyard sentence: [CLS], the WordPiece pieces of its seven words (Chapter 2's greedy longest match
    against WP_A), [SEP]."""
    out = [CLS]
    for w in words_a(noun, place):
        out += [ID_A[p] for p in tok.wordpiece_encode(w, WP_A)]
    return out + [SEP]


def pieces(ids):
    return " ".join(WP_A[i] for i in ids)


def vocabulary_a():
    rows = [(f"{noun}, {place}", f"`{' '.join(words_a(noun, place))}`", " ".join(f"`{p}`" for p in
            [WP_A[i] for i in ids_a(noun, place)]), len(ids_a(noun, place))) for noun, place in ((0, 0), (4, 2))]
    return (f"Vocabulary ({V_A} entries, ids in this order): " + ", ".join(f"`{w}`" for w in WP_A) + ". A sentence is `the N Ved near "
            "the P .` with N one of six nouns, V the verb of that noun (`cat purred`, `dog barked`, `hen clucked`, `owl hooted`, "
            "`cow mooed`, `fox yelped`) and P one of four places, written with the greedy longest-match WordPiece of Chapter 2 "
            "(a verb is its stem and `##ed`), and framed by `[CLS]` and `[SEP]`.\n\n"
            + _table(["noun, place (indices)", "sentence", "tokens", "length"], rows, "llll")
            + f"\n\nThere are 6 x 4 = 24 sentences, so the entropy of the language is ln 24 = {_fx(math.log(24))} nats: ln 6 = "
            f"{_fx(math.log(6))} for the noun (the verb follows from it) and ln 4 = {_fx(math.log(4))} for the place.")


# ------------------------------------------------------------------------------------------------ the two masks

def mask_rows(allowed, n):
    return [[1 if allowed(i, j) else 0 for j in range(n)] for i in range(n)]


BLANK_EXAMPLE = ["the", "[MASK]", "sat", "on", "the", "mat"]


def masks_matrices():
    n = len(BLANK_EXAMPLE)
    return mask_rows(CAUSAL, n), mask_rows(FULL, n)


def hidden_rows(p, ids, allowed):
    """The rows of the final normalised stream [T, D] of the encoder for `ids` under the mask `allowed`."""
    cfg = CFG_A
    x = [[a + b for a, b in zip(p["E"][t], p["P"][i])] for i, t in enumerate(ids)]
    for b in p["blocks"]:
        x, _ = tg.block(b, x, cfg["H"], allowed)
    return tb.rms_forward(x, p["gf"][0])[0]


def blank_example():
    n = len(BLANK_EXAMPLE)
    causal, full = masks_matrices()
    rows = [(i, BLANK_EXAMPLE[i], " ".join(BLANK_EXAMPLE[j] for j in range(n) if causal[i][j]),
             " ".join(BLANK_EXAMPLE[j] for j in range(n) if full[i][j])) for i in range(n)]
    pairs = (sum(map(sum, causal)), sum(map(sum, full)))
    p = tg.init_gpt(NAME_A_INIT, CFG_A)
    a = [CLS, ID_A["the"], MASK, ID_A["purr"], ID_A["##ed"], ID_A["near"], ID_A["the"], ID_A["mat"], ID_A["."], SEP]
    b = [CLS, ID_A["the"], MASK, ID_A["hoot"], ID_A["##ed"], ID_A["near"], ID_A["the"], ID_A["pond"], ID_A["."], SEP]
    out = []
    for name, allowed in (("causal (j <= i)", CAUSAL), ("full (every pair)", FULL)):
        ha, hb = hidden_rows(p, a, allowed), hidden_rows(p, b, allowed)
        d = math.sqrt(_sum([(x - y) ** 2 for x, y in zip(ha[2], hb[2])]))
        out.append((name, "0" if d < 1e-12 else _fx(d, 4)))
    return (f"Tokens `{' '.join(BLANK_EXAMPLE)}` (positions 0 to {n - 1}); row i lists the positions the mask lets position i read.\n\n"
            + _table(["i", "token", "causal mask reads", "full mask reads"], rows, "rlll")
            + f"\n\nAllowed pairs: {pairs[0]} for the causal mask (n (n + 1) / 2 for n = {n}) and {pairs[1]} for the full mask "
            f"(n^2). Now a model with random weights ({_seed(NAME_A_INIT)} for `{NAME_A_INIT}`, D = 16, one block of 2 heads, learned "
            "positions) reads the two ten-token inputs `" + pieces(a) + "` and `" + pieces(b) + "`, which differ only to the right of "
            "the blank (the verb, the place). The Euclidean distance between the two final vectors at the blank (position 2):\n\n"
            + _table(["mask", "distance between the two vectors at position 2"], out, "lr"))


# ------------------------------------------------------------------------------------------------ BERT's corruption rule

def corrupt(ids, g, rate, vocab=V_A, specials=(CLS, SEP, PAD)):
    """Selection and replacement of the corruption rule. Positions are visited in order; a special token is skipped without a
    draw. For every other position one `uniform()` decides selection (selected when it is below `rate`). A selected position
    gets a second `uniform()` r: r < 0.8 makes the input `[MASK]`, r < 0.9 makes it `integer(vocab)` (one more draw, any id,
    the true token included), otherwise the input is left unchanged. Returns (inputs, targets, weights) with the target the
    original id at a selected position and the weight 1 there, 0 elsewhere. `outcome` lists the kind chosen per position."""
    x, y, m, kinds = list(ids), list(ids), [0] * len(ids), [""] * len(ids)
    for i, t in enumerate(ids):
        if t in specials:
            continue
        if g.uniform() >= rate:
            continue
        m[i] = 1
        r = g.uniform()
        if r < 0.8:
            x[i], kinds[i] = MASK, "mask"
        elif r < 0.9:
            x[i], kinds[i] = g.integer(vocab), "random"
        else:
            kinds[i] = "keep"
    return x, y, m, kinds


NAME_CORRUPT = "encoders.corruption"
RATE_BERT, RATE_BRAIN, RATE_TOY = 0.15, 0.30, 0.40


def corruption_example_data():
    """(tried, ids, inputs, weights, kinds) of the first sentence of the `NAME_CORRUPT` stream, corrupted at RATE_TOY, in which
    all three outcomes occur and the random token differs from the original."""
    g = bookrng.generator(NAME_CORRUPT)
    tried = 0
    while True:
        ids = ids_a(g.integer(6), g.integer(4))
        x, y, m, kinds = corrupt(ids, g, RATE_TOY)
        tried += 1
        changed = all(x[i] != ids[i] for i in range(len(ids)) if kinds[i] == "random")
        if {"mask", "random", "keep"} <= set(kinds) and changed:
            return tried, ids, x, m, kinds


def corruption_example():
    """The first sentence with at least one position of each kind, and the counts over 20,000 sentences."""
    tried, ids, x, m, kinds = corruption_example_data()
    rows = [(i, WP_A[ids[i]], WP_A[x[i]], kinds[i] or "not selected", WP_A[ids[i]] if m[i] else "-", m[i]) for i in range(len(ids))]
    gen = bookrng.generator(NAME_CORRUPT + ".count")
    n = 20000
    tot = {k: 0 for k in ("mask", "random", "keep")}
    sel = 0
    for _ in range(n):
        ids2 = ids_a(gen.integer(6), gen.integer(4))
        _, _, m2, k2 = corrupt(ids2, gen, RATE_TOY)
        sel += sum(m2)
        for k in k2:
            if k:
                tot[k] += 1
    rows_c = []
    for rate_name, rate in (("0.15 (BERT)", RATE_BERT), ("0.30 (brain's default)", RATE_BRAIN), ("0.40 (this chapter's runs)", RATE_TOY)):
        gr = bookrng.generator(NAME_CORRUPT + ".rate." + str(rate))
        s = 0
        for _ in range(n):
            ids2 = ids_a(gr.integer(6), gr.integer(4))
            _, _, m2, _ = corrupt(ids2, gr, rate)
            s += sum(m2)
        rows_c.append((rate_name, _fx(s / n, 3), _fx(s / n / 8, 3)))
    return (f"{_seed(NAME_CORRUPT)} for `{NAME_CORRUPT}`. Draws: for each sentence `integer(6)` (the noun) then `integer(4)` (the place); "
            "then, positions in order with the specials `[CLS]` and `[SEP]` skipped, one `uniform()` for selection (selected when below "
            f"the rate, here {RATE_TOY}), and for a selected position a second `uniform()` r (r < 0.8: `[MASK]`; r < 0.9: the input becomes "
            "`integer(25)`, which may by chance be the true id; otherwise unchanged). The first sentence of the stream in which all "
            f"three outcomes occur, the random token being a different one, is sentence {tried}:\n\n"
            + _table(["position", "original", "input to the model", "what happened", "target", "loss weight"], rows, "rlllll")
            + f"\n\nOver {_int(n)} sentences from `{NAME_CORRUPT}.count` at rate {RATE_TOY}, the selected positions divide into "
            + ", ".join(f"{_fx(tot[k] / sel, 3)} {k}" for k in ("mask", "random", "keep"))
            + " (the expected 0.8, 0.1 and 0.1; a `random` draw that lands on the true id counts as `random`). Selected positions per "
            "sentence, of the 8 that are not special, from `" + NAME_CORRUPT + ".rate.<rate>`:\n\n"
            + _table(["rate", "selected per sentence", "share of the 8 positions"], rows_c, "lrr"))


def seen_token_odds():
    """P(the token at a supervised position is the right one) under the 80/10/10 rule: 0.1 + 0.1 / V."""
    keep = 0.1 + 0.1 / V_A
    return keep, 0.8


# ------------------------------------------------------------------------------------------------ training the two models

TRAIN_A = {"B": 16, "steps": 400, "peak": 0.03, "floor": 0.003, "warmup": 20, "wd": 0.1, "clip": 1.0, "b1": 0.9, "b2": 0.95,
           "eps": 1e-8}
KINDS_A = ("encoder", "causal")


def _zero_like(ts):
    return [([[0.0] * len(r) for r in t], [[0.0] * len(r) for r in t]) for t in ts]


def _add_grads(acc, g):
    if acc is None:
        return g
    for ta, tg_ in zip(acc, g):
        for ra, rg in zip(ta, tg_):
            for j, x in enumerate(rg):
                ra[j] += x
    return acc


def probe_ce(p, ids, allowed, masked_positions, targets_at):
    """Cross-entropy of the true tokens at `targets_at`, with the encoder reading `ids` whose `masked_positions` are `[MASK]`."""
    x = list(ids)
    for i in masked_positions:
        x[i] = MASK
    logits, _ = tg.forward(p, x, CFG_A, allowed)
    return [tg.losses([logits[i]], [ids[i]])[0] for i in targets_at], logits


def evaluate_a(p, kind):
    """Per-slot cross-entropy over all 24 sentences. Encoder: exactly one position holds `[MASK]` (the inputs of the other
    positions are the true tokens). Causal: the loss of predicting the token at position i from positions 0 .. i-1. Also, for
    the encoder, the noun and the verb stem both masked, and the probabilities of the demonstration blanks."""
    per = [0.0] * T_A
    both = [0.0, 0.0, 0.0]
    for noun in range(6):
        for place in range(4):
            ids = ids_a(noun, place)
            if kind == "encoder":
                for i in range(1, T_A - 1):
                    ce, _ = probe_ce(p, ids, FULL, [i], [i])
                    per[i] += ce[0] / 24
                ce, logits = probe_ce(p, ids, FULL, [2, 3], [2, 3])
                both[0] += ce[0] / 24
                both[1] += ce[1] / 24
                pn = _softmax_list(logits[2])
                ps = _softmax_list(logits[3])
                both[2] += _sum([pn[ID_A[NOUNS_A[k]]] * ps[ID_A[STEMS_A[k]]] for k in range(6)]) / 24
            else:
                logits, _ = tg.forward(p, ids[:-1], CFG_A, CAUSAL)
                ce = tg.losses(logits, ids[1:])
                for i in range(1, T_A):
                    per[i] += ce[i - 1] / 24
    return per, both


def fill_blanks(p, kind):
    """Probabilities of the six nouns for the blank in `the [MASK] purred near the mat .` and of the four places for the blank in
    `the cat purred near the [MASK] .`, as the model gives them (raw softmax entries, not renormalised)."""
    noun_ids = ids_a(0, 0)
    place_ids = ids_a(0, 0)
    out = []
    for ids, pos, pool in ((noun_ids, 2, NOUNS_A), (place_ids, 7, PLACES_A)):
        if kind == "encoder":
            x = list(ids)
            x[pos] = MASK
            logits, _ = tg.forward(p, x, CFG_A, FULL)
            z = logits[pos]
        else:
            logits, _ = tg.forward(p, ids[:pos], CFG_A, CAUSAL)
            z = logits[pos - 1]
        pr = _softmax_list(z)
        out.append([pr[ID_A[w]] for w in pool])
    return out


def train_a(job):
    """One run of a model of language A: (kind, nudge). Both start from `NAME_A_INIT` (one block, D = 16, 2 heads, F = 24,
    V = 25, T = 10; `training_gpt.init_gpt`, every entry 0.02 normal(), tied head); `nudge` is added to E[0][0]. Every step
    takes TRAIN_A['B'] sentences from `NAME_A_SENT` (noun then place per sentence, the same sentences for both kinds). The
    encoder corrupts each with `corrupt` at RATE_TOY from `NAME_A_MASK` and is scored at the selected positions only (full mask);
    the causal model reads positions 0 .. T-2 under the causal mask and is scored at every position on the next token. The loss
    is the sum over the batch's supervised positions divided by their number; AdamW with warmup, cosine decay and a clip."""
    kind, nudge = job
    o = TRAIN_A
    p = tg.init_gpt(NAME_A_INIT, CFG_A)
    p["E"][0][0] += nudge
    ts = tg.tensors(p)
    mom = _zero_like(ts)
    decay = tg.decayed(CFG_A)
    lr_of = schedules.cosine(o["peak"], o["floor"], o["warmup"], o["steps"])
    gs, gm = bookrng.generator(NAME_A_SENT), bookrng.generator(NAME_A_MASK)
    allowed = FULL if kind == "encoder" else CAUSAL
    losses = []
    for step in range(o["steps"]):
        rows = []
        for _ in range(o["B"]):
            ids = ids_a(gs.integer(6), gs.integer(4))
            if kind == "encoder":
                x, y, m, _ = corrupt(ids, gm, RATE_TOY)
            else:
                x, y, m = ids[:-1], ids[1:], [1] * (len(ids) - 1)
            rows.append((x, y, m))
        denom = float(max(1, sum(sum(r[2]) for r in rows)))
        acc, total = None, 0.0
        for x, y, m in rows:
            logits, cache = tg.forward(p, x, CFG_A, allowed)
            ce = tg.losses(logits, y)
            total += _sum([c * w for c, w in zip(ce, m)])
            acc = _add_grads(acc, tg.backward(p, cache, logits, y, m, denom, CFG_A))
        norm = tg.global_norm(acc)
        tg.adamw_update(ts, acc, mom, step + 1, lr_of(step), min(1.0, o["clip"] / (norm + 1e-6)), decay, o)
        losses.append(total / denom)
    per, both = evaluate_a(p, kind)
    return {"kind": kind, "nudge": nudge, "per": per, "both": both, "fill": fill_blanks(p, kind),
            "first": _sum(losses[:10]) / 10, "last": _sum(losses[-20:]) / 20}


@lru_cache(maxsize=None)
def results_a():
    """{kind: (base run, [nudged runs])}: both models, each repeated from the four nudged starting points of the stable-digits
    rule, spread over four processes."""
    jobs = [(k, n) for k in KINDS_A for n in (0.0, *stable.NUDGES)]
    with get_context("fork").Pool(4) as pool:
        res = pool.map(train_a, jobs, chunksize=1)
    out = {k: [None, []] for k in KINDS_A}
    for r in res:
        if r["nudge"] == 0.0:
            out[r["kind"]][0] = r
        else:
            out[r["kind"]][1].append(r)
    return out


def cell_a(kind, f, digits=3):
    base, twins = results_a()[kind]
    vals = [f(base)] + [f(t) for t in twins]
    if all(v < 0.01 for v in vals):
        return "below 0.01"
    return stable.text(vals[0], vals[1:], lambda x: _fx(x, digits))


def slot_floor(kind, slot):
    if slot == 7:
        return math.log(4)
    if slot == 2 and kind == "causal":
        return math.log(6)
    return 0.0


def slot_table():
    rows = []
    for slot in range(1, 9):
        rows.append((slot, SLOT_NAME_TEXT[slot], _fx(slot_floor("encoder", slot), 3), cell_a("encoder", lambda r, s=slot: r["per"][s]),
                     _fx(slot_floor("causal", slot), 3), cell_a("causal", lambda r, s=slot: r["per"][s])))
    cfg = CFG_A
    return (f"Both models: V = {V_A}, T = {T_A}, D = {cfg['D']}, {cfg['H']} heads, one block of Chapter 9 (RMS norm, SwiGLU of width "
            f"{cfg['FF']}), learned positions, head tied to the embedding table, initial weights from `{NAME_A_INIT}` "
            f"({_seed(NAME_A_INIT)}); sentences from `{NAME_A_SENT}` ({_seed(NAME_A_SENT)}), corruption draws from `{NAME_A_MASK}` "
            f"({_seed(NAME_A_MASK)}). {TRAIN_A['steps']} steps of {TRAIN_A['B']} sentences, AdamW (peak {TRAIN_A['peak']}, floor "
            f"{TRAIN_A['floor']}, {TRAIN_A['warmup']} warmup steps, cosine decay, weight decay {TRAIN_A['wd']}, betas "
            f"{TRAIN_A['b1']} and {TRAIN_A['b2']}, epsilon {TRAIN_A['eps']}, global clip {TRAIN_A['clip']}). The encoder is scored "
            "after training with exactly one `[MASK]` in the sentence, each of the 8 word positions in turn, over all 24 sentences; the "
            "causal model is scored on the token at each position from the tokens before it. Cross-entropy in nats; the floor is the "
            f"entropy of the slot given what the model reads. {stable.RULE}\n\n"
            + _table(["position", "slot", "encoder floor", "encoder", "causal floor", "causal"], rows, "rlrrrr"))


SLOT_NAME_TEXT = {1: "the", 2: "noun", 3: "verb stem", 4: "`##ed`", 5: "near", 6: "the", 7: "place", 8: "full stop"}


def supervision_counts():
    sel = RATE_TOY * 8
    return sel, 9


def joint_blanks():
    rows = []
    for name, f in (("noun", lambda r: r["both"][0]), ("verb stem", lambda r: r["both"][1])):
        rows.append((name, _fx(math.log(6), 3), cell_a("encoder", f)))
    cons = cell_a("encoder", lambda r: r["both"][2])
    return (f"The encoder of the table above with both the noun (position 2) and the verb stem (position 3) set to `[MASK]` at once, "
            "over the 24 sentences. Cross-entropy of each blank, and the floor ln 6 each: with both gone, nothing in the sentence "
            "says which animal it is.\n\n"
            + _table(["blank", "floor", "encoder"], rows, "lrr")
            + f"\n\nIf each blank is filled by a draw from the model's own distribution at that position, independently, the chance "
            f"that the noun and the verb belong together is the sum over the six nouns of P(noun) P(its verb stem), averaged over "
            f"the sentences: {cons} (a coin between six animals would give 1/6 = {_fx(1 / 6, 3)}).")


def fill_table():
    nouns_enc = []
    for k, name in enumerate(NOUNS_A):
        nouns_enc.append((name, cell_a("encoder", lambda r, k=k: r["fill"][0][k], 3), cell_a("causal", lambda r, k=k: r["fill"][0][k], 3)))
    places = []
    for k, name in enumerate(PLACES_A):
        places.append((name, cell_a("encoder", lambda r, k=k: r["fill"][1][k], 3), cell_a("causal", lambda r, k=k: r["fill"][1][k], 3)))
    return ("Probabilities the two trained models give to each candidate, for the blank in `the [MASK] purred near the mat .` "
            "(the encoder reads the whole sentence with the blank in position 2; the causal model reads `[CLS] the` and predicts position 2), "
            "and for the blank in `the cat purred near the [MASK] .` (position 7):\n\n"
            + _table(["noun", "encoder", "causal"], nouns_enc, "lrr") + "\n\n" + _table(["place", "encoder", "causal"], places, "lrr"))


# ------------------------------------------------------------------------------------------------ language B: synonyms

ANIMALS_B = [("cat", "feline"), ("dog", "hound"), ("hen", "fowl"), ("fox", "vixen")]
PLACES_B = [("barn", "shed"), ("pond", "lake"), ("hill", "mound"), ("mat", "rug")]
VERBS_B = ["sat", "ran", "slept"]
PREPS_B = ["near", "by"]
DETS_B = ["the", "a"]
WORDS_B = ["[PAD]", "[CLS]"] + DETS_B + [w for g in ANIMALS_B for w in g] + VERBS_B + PREPS_B + [w for g in PLACES_B for w in g]
ID_B = {w: i for i, w in enumerate(WORDS_B)}
V_B, T_B = len(WORDS_B), 7                 # [CLS] det animal verb preposition det place
CFG_B = {"V": V_B, "D": 16, "H": 2, "FF": 24, "L": 1, "T": T_B}
HELD_OUT = ((0, 1), (1, 2), (2, 3), (3, 0))                     # (animal group, place group) pairs never trained on
MEANINGS = [(a, b) for a in range(4) for b in range(4)]
TRAIN_MEANINGS = [m for m, ab in enumerate(MEANINGS) if ab not in HELD_OUT]
HELD_MEANINGS = [m for m, ab in enumerate(MEANINGS) if ab in HELD_OUT]
NAME_B_INIT, NAME_B_STREAM, NAME_B_EVAL = "encoders.nce.init", "encoders.nce.stream", "encoders.nce.retrieval"


def realize(m, wa, wb, d1, v, pr, d2):
    """The ids of the sentence `[CLS] det animal verb prep det place` for meaning m = 4 a + b: the animal word is word `wa` of
    animal group a, the place word is word `wb` of place group b, the others are the indices d1, v, pr, d2 into their lists."""
    a, b = MEANINGS[m]
    words = [DETS_B[d1], ANIMALS_B[a][wa], VERBS_B[v], PREPS_B[pr], DETS_B[d2], PLACES_B[b][wb]]
    return [ID_B["[CLS]"]] + [ID_B[w] for w in words]


def text_b(ids):
    return " ".join(WORDS_B[i] for i in ids[1:])


def _noise(g):
    return g.integer(2), g.integer(3), g.integer(2), g.integer(2)


def draw_pair(g, m):
    """A paraphrase pair of meaning m. Draws: `integer(2)` for the animal word index wa, `integer(2)` for the place word index
    wb, then the four noise indices (det, verb, preposition, det) of the first sentence, then of the second. The first sentence
    uses words (wa, wb), the second the other synonyms (1 - wa, 1 - wb): the pair shares no animal word and no place word."""
    wa, wb = g.integer(2), g.integer(2)
    n1, n2 = _noise(g), _noise(g)
    return realize(m, wa, wb, *n1), realize(m, 1 - wa, 1 - wb, *n2)


@lru_cache(maxsize=None)
def eval_set():
    """32 documents (two per meaning, animal and place word 0 of their groups) and 16 queries (one per meaning, word 1). Draws
    from `NAME_B_EVAL`: for each meaning in order, two documents with the four noise indices each; then for each meaning in order
    one query with the four noise indices. A document is relevant to the query of its meaning."""
    g = bookrng.generator(NAME_B_EVAL)
    docs, doc_m = [], []
    for m in range(16):
        for _ in range(2):
            docs.append(realize(m, 0, 0, *_noise(g)))
            doc_m.append(m)
    queries = [realize(m, 1, 1, *_noise(g)) for m in range(16)]
    return docs, doc_m, queries


def vocabulary_b():
    rows = [("determiners", ", ".join(DETS_B)), ("animals, by group", "; ".join(" = ".join(g) for g in ANIMALS_B)),
            ("verbs", ", ".join(VERBS_B)), ("prepositions", ", ".join(PREPS_B)),
            ("places, by group", "; ".join(" = ".join(g) for g in PLACES_B))]
    docs, doc_m, queries = eval_set()
    ex = realize(5, 0, 0, 0, 0, 0, 0)
    ex2 = realize(5, 1, 1, 1, 2, 1, 1)
    return (f"A sentence is `[CLS] det animal verb prep det place`, T = {T_B} tokens, V = {V_B} (`[PAD]`, `[CLS]`, then the words below). "
            "The meaning of a sentence is its pair (animal group, place group), 4 x 4 = 16 meanings numbered m = 4 a + b; a word that "
            "is written `x = y` is one group (two synonyms). The determiners, the verb and the preposition carry no meaning.\n\n"
            + _table(["kind", "words"], rows, "ll")
            + f"\n\nMeaning 5 is (animal group 1, place group 1): `{text_b(ex)}` and `{text_b(ex2)}` are two sentences of that meaning "
            f"with no content word in common. Meanings {HELD_MEANINGS} (animal group, place group) = "
            + ", ".join(str(ab) for ab in HELD_OUT)
            + f" are held out: no training pair has them. The collection used for retrieval has {len(docs)} documents (two per "
            f"meaning, always the first synonym of each group) and {len(queries)} queries (one per meaning, always the second synonym), "
            f"from `{NAME_B_EVAL}` ({_seed(NAME_B_EVAL)}); a document is relevant to the query of its meaning.")


# ------------------------------------------------------------------------------------------------ encoding, pooling

def encode(p, ids, pool, allowed=FULL, cfg=CFG_B):
    """(vector, cache) of one sentence: x = E[ids] + P, the blocks under the mask `allowed`, the final RMS norm, then pooling.
    `pool` "cls" returns row 0 of the normalised stream; "mean" returns the mean of its rows over the positions `allowed`
    lets position 0 read (all rows when the mask is full)."""
    x = [[a + b for a, b in zip(p["E"][t], p["P"][i])] for i, t in enumerate(ids)]
    caches = []
    for b in p["blocks"]:
        x, c = tg.block(b, x, cfg["H"], allowed)
        caches.append(c)
    xn, cf = tb.rms_forward(x, p["gf"][0])
    n, d = len(ids), len(xn[0])
    if pool == "mean":
        v = [_sum([r[j] for r in xn]) / n for j in range(d)]
    else:
        v = list(xn[0])
    return v, {"ids": ids, "caches": caches, "cf": cf, "n": n, "pool": pool, "xn": xn}


def encode_backward(p, cache, dv):
    """Gradients (in the order of `training_gpt.tensors`) of a loss with respect to the parameters, given dv = dL/dv."""
    n, d = cache["n"], len(dv)
    if cache["pool"] == "mean":
        dxn = [[dv[j] / n for j in range(d)] for _ in range(n)]
    else:
        dxn = [[0.0] * d for _ in range(n)]
        dxn[0] = list(dv)
    dx, dgf = tb.rms_backward(dxn, cache["cf"])
    block_grads = []
    for c in reversed(cache["caches"]):
        dx, g = tb.block_backward(c, dx)
        block_grads.append(g)
    block_grads.reverse()
    dE = [[0.0] * d for _ in range(len(p["E"]))]
    for i, t in enumerate(cache["ids"]):
        for j in range(d):
            dE[t][j] += dx[i][j]
    dP = [[0.0] * d for _ in range(len(p["P"]))]
    for i in range(n):
        dP[i] = list(dx[i])
    grads = [dE, dP]
    for g in block_grads:
        grads += [g[k] for k in tg.BLOCK_KEYS]
    grads.append([dgf])
    return grads


PAD_EXAMPLE = [ID_B["[CLS]"], ID_B["the"], ID_B["cat"], ID_B["sat"], ID_B["near"], ID_B["the"], ID_B["barn"]]
NAME_PAD_INIT = "encoders.padding.init"


def padding_example():
    """The seven-token sentence alone, and padded to ten positions with `[PAD]`: pad keys readable, then pad keys masked."""
    cfg = dict(CFG_B, T=10)
    p = tg.init_gpt(NAME_PAD_INIT, cfg)
    n = len(PAD_EXAMPLE)
    padded = PAD_EXAMPLE + [ID_B["[PAD]"]] * 3
    alone_n = encode(p, PAD_EXAMPLE, "mean", FULL, cfg)[1]["xn"]
    rows = []
    pooled = {}
    for name, allowed in (("padded, pad positions readable", FULL), ("padded, pad positions masked", lambda i, j: j < n)):
        xn = encode(p, padded, "mean", allowed, cfg)[1]["xn"]
        worst = max(abs(a - b) for r1, r2 in zip(alone_n, xn[:n]) for a, b in zip(r1, r2))
        rows.append((name, "below 1e-12" if worst < 1e-12 else _fx(worst, 4)))
        pooled[name] = xn
    alone_v = [_sum([r[j] for r in alone_n]) / n for j in range(16)]
    prow = []
    for name, xn in pooled.items():
        over_real = [_sum([r[j] for r in xn[:n]]) / n for j in range(16)]
        over_all = [_sum([r[j] for r in xn]) / 10 for j in range(16)]
        prow.append((name, _fx(_cos(alone_v, over_real), 4), _fx(_cos(alone_v, over_all), 4)))
    return (f"Sentence `{text_b(PAD_EXAMPLE)}` (with `[CLS]`, 7 tokens). Random weights from `{NAME_PAD_INIT}` ({_seed(NAME_PAD_INIT)}): "
            "`init_gpt` with T = 10, V = 25, D = 16, one block of 2 heads, the same weights in all three encodings (the alone "
            "encoding uses the first 7 rows of the position table). Padded: three `[PAD]` tokens (id 0) after the sentence. "
            "Largest absolute difference, over the 7 real positions and the 16 coordinates, between the final normalised stream of the "
            "padded encoding and that of the encoding alone:\n\n"
            + _table(["encoding", "largest difference at real positions"], rows, "lr")
            + "\n\nThe mean-pooled vector of the sentence alone against the mean pooled over the 7 real positions of the padded "
            "stream and against the mean over all 10 positions of the padded stream (cosine similarity):\n\n"
            + _table(["encoding", "cosine, mean over the 7 real positions", "cosine, mean over all 10 positions"], prow, "lrr"))


# ------------------------------------------------------------------------------------------------ InfoNCE

def infonce(U, V, tau):
    """(loss, dU, dV) of the symmetric InfoNCE loss for unit rows U, V [B, D]: S = U V^T / tau; the loss is the mean over the 2 B
    cross-entropies of the rows of S and of the columns of S against the diagonal, and dU, dV are its gradients."""
    B, d = len(U), len(U[0])
    S = [[_dot(U[i], V[j]) / tau for j in range(B)] for i in range(B)]
    R = [_softmax_list(r) for r in S]
    C = [_softmax_list([S[i][j] for i in range(B)]) for j in range(B)]
    loss = _sum([-math.log(R[i][i]) - math.log(C[i][i]) for i in range(B)]) / (2 * B)
    dS = [[(R[i][j] + C[j][i] - (2.0 if i == j else 0.0)) / (2 * B) for j in range(B)] for i in range(B)]
    dU = [[_sum([dS[i][j] * V[j][k] for j in range(B)]) / tau for k in range(d)] for i in range(B)]
    dV = [[_sum([dS[i][j] * U[i][k] for i in range(B)]) / tau for k in range(d)] for j in range(B)]
    return loss, dU, dV


def unit_backward(du, u, norm):
    """Gradient through v -> v / |v| given the gradient du at the unit vector u and the length |v|: (du - (du . u) u) / |v|."""
    c = _dot(du, u)
    return [(x - c * y) / norm for x, y in zip(du, u)]


EXAMPLE_COS = [[0.9, 0.3, -0.1], [0.2, 0.8, 0.4], [0.0, 0.5, 0.6]]


def infonce_example():
    S = EXAMPLE_COS
    B = len(S)
    rows = []
    for tau in (1.0, 0.5, 0.1, 0.05):
        per = []
        for i in range(B):
            z = [s / tau for s in S[i]]
            per.append(-math.log(math.exp(z[i]) / _sum([math.exp(v) for v in z])))
            ce = tg.losses([z], [i])[0]
            assert abs(ce - per[-1]) < 1e-12
        rows.append((f"{tau:g}", *[_fx(v, 4) for v in per], _fx(_sum(per) / B, 4)))
    p1 = _softmax_list([s / 0.1 for s in S[1]])
    return ("A batch of B = 3 pairs; the matrix of cosines s_ij between the anchor of pair i (row) and the positive of pair j (column) is\n\n"
            + _table(["", "j = 0", "j = 1", "j = 2"], [(f"i = {i}", *[_fx(v, 1) for v in S[i]]) for i in range(B)], "lrrr")
            + "\n\nThe loss of row i is the cross-entropy of the softmax of (s_i0, s_i1, s_i2) / tau against the label i (the library "
            "cross-entropy and the explicit -log(e^{s_ii/tau} / sum_j e^{s_ij/tau}) agree to 1e-12 in every cell):\n\n"
            + _table(["tau", "row 0", "row 1", "row 2", "mean (one direction)"], rows, "lrrrr")
            + f"\n\nWith every similarity equal the loss is ln 3 = {_fx(math.log(3), 4)}. At tau = 0.1 the softmax of row 1 is "
            + _vec_text(p1, 4) + f", so the gradient of the row's loss with respect to s_11 is (p_11 - 1) / tau = {_fx((p1[1] - 1) / 0.1, 3)} "
            f"and with respect to the wrong cosine s_12 it is p_12 / tau = {_fx(p1[2] / 0.1, 3)}.")


def _vec_text(v, d=3):
    return "(" + ", ".join(_fx(x, d) for x in v) + ")"


def false_negatives():
    """Probability that 8 meanings drawn with replacement from the 16 contain a repeat, and the floor a repeat puts on the loss."""
    keep = 1.0
    for k in range(8):
        keep *= (16 - k) / 16
    floor = 2 * math.log(2) / 8
    return (f"Drawing the 8 pairs of a batch with replacement from the 16 meanings gives a batch with two pairs of one meaning with "
            f"probability 1 - (16 x 15 x ... x 9) / 16^8 = {_fx(1 - keep, 4)}. The anchors of such a pair see two equally good "
            f"positives, so each of their two rows can do no better than a coin between two columns: row cross-entropy at least "
            f"ln 2 = {_fx(math.log(2), 4)} on 2 of 8 rows, which raises the mean of one direction by at least 2 ln 2 / 8 = {_fx(floor, 4)} "
            f"however good the encoder is. The runs below draw the 8 meanings without replacement (one `shuffle` of the training "
            f"meanings, the first 8 taken), so the floor does not occur.")


# ------------------------------------------------------------------------------------------------ the contrastive runs

TRAIN_B = {"B": 8, "steps": 100, "peak": 0.01, "floor": 0.001, "warmup": 10, "wd": 0.1, "clip": 1.0, "b1": 0.9, "b2": 0.95,
           "eps": 1e-8}
CHECKPOINTS_B = (10, 20, 40, 100)
CONFIGS_B = (("mean", 0.1), ("cls", 0.1), ("mean", 1.0), ("mean", 0.02))
KS = (2, 4, 8)
RULE_AGAIN = "Digits follow the stable-digits rule stated with the training table above (each run repeated from four nudged starting points)."


def retrieval_metrics(vq, vd, doc_m):
    """Metrics of 16 query vectors against 32 document vectors by cosine: the mean cosine of a query with its relevant documents
    and with the others, and recall at k (relevant documents among the k best, over the 2 relevant) averaged over all
    queries, over those of trained meanings and over those of held-out meanings. Ties keep the lower document index."""
    pos, neg, rec = [], [], {k: [] for k in KS}
    for m, q in enumerate(vq):
        sims = [_cos(q, d) for d in vd]
        for i, s in enumerate(sims):
            (pos if doc_m[i] == m else neg).append(s)
        order = sorted(range(len(vd)), key=lambda i: -sims[i])
        for k in KS:
            rec[k].append(sum(1 for i in order[:k] if doc_m[i] == m) / 2.0)
    held = set(HELD_MEANINGS)
    out = {"pos": _sum(pos) / len(pos), "neg": _sum(neg) / len(neg)}
    for k in KS:
        out[f"r{k}"] = _sum(rec[k]) / 16
        out[f"r{k}_held"] = _sum([rec[k][m] for m in HELD_MEANINGS]) / len(held)
    return out


def evaluate_b(p, pool):
    docs, doc_m, queries = eval_set()
    vd = [encode(p, d, pool)[0] for d in docs]
    vq = [encode(p, q, pool)[0] for q in queries]
    return retrieval_metrics(vq, vd, doc_m)


def bag_of_words():
    docs, doc_m, queries = eval_set()
    vec = lambda ids: [float(ids.count(w)) for w in range(2, V_B)]   # noqa: E731  (the counts of the words, [PAD] and [CLS] left out)
    return retrieval_metrics([vec(q) for q in queries], [vec(d) for d in docs], doc_m)


def train_b(job):
    """One contrastive run: (pool, tau, nudge). Weights from `NAME_B_INIT` (`init_gpt`, V = 25, T = 7, D = 16, one block); `nudge`
    is added to E[0][0]. Each of the 100 steps draws one `shuffle` of the 12 trained meanings from `NAME_B_STREAM`, takes the
    first 8, and for them in order one `draw_pair`; every sentence is encoded with the full mask and pooled (`pool`), the vectors
    are scaled to length 1, the loss is the symmetric InfoNCE at temperature `tau`, the gradient goes through the pooling and the
    block, and AdamW (peak 0.01, floor 0.001, 10 warmup steps, cosine decay, decay 0.1, betas 0.9 and 0.95, clip 1) steps.
    Returns the metrics of the encoder on the retrieval set after the steps of CHECKPOINTS_B, the mean batch loss of the ten
    steps ending there, and the final parameters."""
    pool, tau, nudge = job
    o = TRAIN_B
    p = tg.init_gpt(NAME_B_INIT, CFG_B)
    p["E"][0][0] += nudge
    ts = tg.tensors(p)
    mom = _zero_like(ts)
    decay = tg.decayed(CFG_B)
    lr_of = schedules.cosine(o["peak"], o["floor"], o["warmup"], o["steps"])
    g = bookrng.generator(NAME_B_STREAM)
    losses, at = [], {0: dict(evaluate_b(p, pool), loss=None)}
    for step in range(o["steps"]):
        ms = [TRAIN_MEANINGS[i] for i in g.shuffle(list(range(len(TRAIN_MEANINGS))))[:o["B"]]]
        prs = [draw_pair(g, m) for m in ms]
        ea = [encode(p, a, pool) for a, _ in prs]
        eb = [encode(p, b, pool) for _, b in prs]
        ua, ub = [_unit(v) for v, _ in ea], [_unit(v) for v, _ in eb]
        loss, dU, dV = infonce([u for u, _ in ua], [u for u, _ in ub], tau)
        acc = None
        for (u, n), (_, c), du in zip(ua, ea, dU):
            acc = _add_grads(acc, encode_backward(p, c, unit_backward(du, u, n)))
        for (u, n), (_, c), dv in zip(ub, eb, dV):
            acc = _add_grads(acc, encode_backward(p, c, unit_backward(dv, u, n)))
        norm = tg.global_norm(acc)
        tg.adamw_update(ts, acc, mom, step + 1, lr_of(step), min(1.0, o["clip"] / (norm + 1e-6)), decay, o)
        losses.append(loss)
        if step + 1 in CHECKPOINTS_B:
            at[step + 1] = dict(evaluate_b(p, pool), loss=_sum(losses[-10:]) / 10)
    return {"job": (pool, tau), "nudge": nudge, "at": at, "p": p}


@lru_cache(maxsize=None)
def results_b():
    """{(pool, tau): (base run, [nudged runs])}: every configuration repeated from the four nudged starting points."""
    jobs = [(pool, tau, n) for pool, tau in CONFIGS_B for n in (0.0, *stable.NUDGES)]
    with get_context("fork").Pool(4) as pool_:
        res = pool_.map(train_b, jobs, chunksize=1)
    out = {c: [None, []] for c in CONFIGS_B}
    for r in res:
        if r["nudge"] == 0.0:
            out[r["job"]][0] = r
        else:
            out[r["job"]][1].append(r)
    return out


@lru_cache(maxsize=None)
def base_run_b(pool="mean", tau=0.1):
    """The un-nudged run alone, for the figure."""
    return train_b((pool, tau, 0.0))


def cell_b(config, step, f, digits=4):
    base, twins = results_b()[config]
    return stable.text(f(base["at"][step]), [f(t["at"][step]) for t in twins], lambda x: _fx(x, digits))


def untrained_b(pool="mean"):
    return evaluate_b(tg.init_gpt(NAME_B_INIT, CFG_B), pool)


def contrastive_training():
    cfg, o = CFG_B, TRAIN_B
    base = untrained_b()
    rows = [("0", f"ln 8 = {_fx(math.log(8), 4)} (chance)", _fx(base["pos"], 4), _fx(base["neg"], 4), _fx(base["r2"], 4), _fx(base["r2_held"], 4))]
    c = ("mean", 0.1)
    for s in CHECKPOINTS_B:
        rows.append((str(s), cell_b(c, s, lambda m: m["loss"]), cell_b(c, s, lambda m: m["pos"]), cell_b(c, s, lambda m: m["neg"]),
                     cell_b(c, s, lambda m: m["r2"]), cell_b(c, s, lambda m: m["r2_held"])))
    return (f"Language B, one block (V = {cfg['V']}, T = {cfg['T']}, D = {cfg['D']}, {cfg['H']} heads, SwiGLU width {cfg['FF']}) with mean "
            f"pooling and tau = 0.1, initial weights from `{NAME_B_INIT}` ({_seed(NAME_B_INIT)}), pairs from `{NAME_B_STREAM}` "
            f"({_seed(NAME_B_STREAM)}), {o['steps']} steps of {o['B']} pairs, AdamW (peak {o['peak']}, floor {o['floor']}, {o['warmup']} "
            f"warmup steps, cosine decay, decay {o['wd']}, betas {o['b1']} and {o['b2']}, clip {o['clip']}). After the step, on the "
            "collection of the retrieval set: the loss (mean of the ten steps ending there), the mean cosine of a query with its two relevant "
            "documents, with the 30 others, and recall at 2 over all 16 queries and over the 4 held-out meanings. " + stable.RULE
            + "\n\n" + _table(["step", "batch loss", "cosine, relevant", "cosine, others", "recall at 2", "recall at 2, held-out"], rows,
                              "rrrrrr"))


def retrieval_table():
    docs, doc_m, queries = eval_set()
    bow = bag_of_words()
    rows = [("random order (expected)", *[_fx(k / 32.0, 4) for k in KS], _fx(2 / 32.0, 4)),
            ("bag of words, cosine of word counts", *[_fx(bow[f"r{k}"], 4) for k in KS], _fx(bow["r2_held"], 4))]
    un = untrained_b()
    rows.append(("encoder, random weights, mean pooling", *[_fx(un[f"r{k}"], 4) for k in KS], _fx(un["r2_held"], 4)))
    for pool, tau in (("mean", 0.1), ("cls", 0.1)):
        c = (pool, tau)
        rows.append((f"contrastive, {pool} pooling, tau = {tau}", *[cell_b(c, 100, lambda m, k=k: m[f"r{k}"]) for k in KS],
                     cell_b(c, 100, lambda m: m["r2_held"])))
    return ("32 documents, 16 queries, 2 relevant documents each (the collection of the vocabulary section). Recall at k is the share of "
            "the query's 2 relevant documents that are among its k highest cosines, averaged over the queries; ties keep the lower "
            "document index. The last column is recall at 2 on the 4 queries whose meaning no training pair had. The bag of words "
            "counts the 23 words other than `[PAD]` and `[CLS]` in each sentence. " + RULE_AGAIN + "\n\n"
            + _table(["ranker", "recall at 2", "recall at 4", "recall at 8", "recall at 2, held-out"], rows, "lrrrr"))


def temperature_table():
    rows = []
    for tau in (1.0, 0.1, 0.02):
        c = ("mean", tau)
        rows.append((f"{tau:g}", cell_b(c, 10, lambda m: m["loss"]), cell_b(c, 100, lambda m: m["loss"]), cell_b(c, 10, lambda m: m["pos"] - m["neg"]),
                     cell_b(c, 100, lambda m: m["pos"] - m["neg"]), cell_b(c, 10, lambda m: m["r2"]), cell_b(c, 100, lambda m: m["r2"])))
    return ("Mean pooling, three temperatures, everything else as in the training run above. The gap is the mean cosine of a query "
            "with its relevant documents minus its mean cosine with the others. " + RULE_AGAIN + "\n\n"
            + _table(["tau", "loss, step 10", "loss, step 100", "gap, step 10", "gap, step 100", "recall at 2, step 10", "recall at 2, step 100"],
                     rows, "rrrrrrr"))


# ------------------------------------------------------------------------------------------------ the picture of the space

PCA_MEANINGS = (0, 5, 10, 15)               # (animal group, place group) = (0, 0), (1, 1), (2, 2), (3, 3)
NAME_PCA = "encoders.pca"


def pca_sentences():
    """For each of four meanings, 8 sentences: the four combinations of synonyms (wa, wb), two draws of the noise each. Draws
    from `NAME_PCA`: for each meaning in order, for each (wa, wb) in (0,0), (0,1), (1,0), (1,1), two groups of four noise indices."""
    g = bookrng.generator(NAME_PCA)
    out = []
    for m in PCA_MEANINGS:
        out.append([realize(m, wa, wb, *_noise(g)) for wa in (0, 1) for wb in (0, 1) for _ in range(2)])
    return out


def pca_points(stage):
    """Principal-direction coordinates (embeddings.`_top2`) of the 32 pooled vectors of PCA_MEANINGS, before or after training,
    as four lists of (x, y), and the share of variance the two directions keep."""
    p = tg.init_gpt(NAME_B_INIT, CFG_B) if stage == "before" else base_run_b()["p"]
    sets = pca_sentences()
    vecs = [[encode(p, ids, "mean")[0] for ids in grp] for grp in sets]
    flat = [v for grp in vecs for v in grp]
    xc, dirs = emb._top2(flat)
    pts = [(_dot(r, dirs[0][1]), _dot(r, dirs[1][1])) for r in xc]
    total = _sum([_dot(r, r) for r in xc]) / len(xc)
    groups = [pts[i * 8:(i + 1) * 8] for i in range(len(sets))]
    return groups, (dirs[0][0] + dirs[1][0]) / total


def pca_summary():
    rows = []
    for stage in ("before", "after"):
        groups, kept = pca_points(stage)
        spread = []
        for grp in groups:
            cx = _sum([x for x, _ in grp]) / 8
            cy = _sum([y for _, y in grp]) / 8
            spread.append(math.sqrt(_sum([(x - cx) ** 2 + (y - cy) ** 2 for x, y in grp]) / 8))
        cents = [(_sum([x for x, _ in grp]) / 8, _sum([y for _, y in grp]) / 8) for grp in groups]
        dmin = min(math.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(cents) for b in cents[:i])
        rows.append((stage + " training", _fx(100 * kept, 1) + "%", _fx(_sum(spread) / 4, 3), _fx(dmin, 3)))
    names = ", ".join(f"{ANIMALS_B[MEANINGS[m][0]][0]} in the {PLACES_B[MEANINGS[m][1]][0]}" for m in PCA_MEANINGS)
    return (f"Four meanings (animal group, place group) = (0,0), (1,1), (2,2), (3,3), named by the first synonym: {names}. Eight sentences "
            f"each: the four combinations of synonyms, two draws of the noise each, from `{NAME_PCA}` ({_seed(NAME_PCA)}). Each sentence "
            "is encoded with the full mask and mean-pooled (the run of the training table, un-nudged, and the same weights before "
            "training). The 32 vectors are centred and projected on the first two principal directions (power iteration of Chapter 3). "
            "The table gives the variance kept, the mean distance of a point from its meaning's centre in the plane, and the smallest "
            "distance between two meanings' centres.\n\n"
            + _table(["encoder", "variance in 2 directions", "mean spread within a meaning", "smallest gap between meanings"], rows, "lrrr"))


def pooling_table():
    rows = []
    for step in (20, 40, 100):
        rows.append((str(step), *[x for pool in ("mean", "cls") for x in (cell_b((pool, 0.1), step, lambda m: m["loss"]),
                                                                          cell_b((pool, 0.1), step, lambda m: m["r2"]))]))
    return ("The same run with the pooled vector taken as the mean of the seven rows of the final stream or as row 0 (`[CLS]`), tau = 0.1, "
            "everything else equal. Loss is the mean of the ten steps ending at the step; recall at 2 is over all 16 queries. "
            + RULE_AGAIN + "\n\n"
            + _table(["step", "loss, mean pooling", "recall at 2, mean", "loss, CLS pooling", "recall at 2, CLS"], rows, "rrrrr"))


def tau_floor():
    """The smallest value of the one-direction loss of a batch of 8 at tau = 1: cosines are at most 1 for the positive and at
    least -1 for the 7 others."""
    return math.log(1 + 7 * math.exp(-2.0))


def tau_floor_text():
    rows = []
    for tau in (1.0, 0.1, 0.02):
        rows.append((f"{tau:g}", _fx(math.log(1 + 7 * math.exp(-2.0 / tau)), 6)))
    return ("The smallest possible loss of one direction of a batch of 8 at temperature tau: the positive at cosine 1 and the 7 others at "
            "cosine -1, which is ln(1 + 7 e^{-2/tau}).\n\n" + _table(["tau", "smallest loss"], rows, "rr"))


# ------------------------------------------------------------------------------------------------ a linear head on frozen vectors

NAME_PROBE = "encoders.probe"
PROBE_STEPS, PROBE_LR = 300, 2.0


def probe_data(name, n):
    """n sentences of language B with the animal group as the label. Draws from `name`: per sentence `integer(16)` for the
    meaning, `integer(2)` for wa, `integer(2)` for wb, then the four noise indices."""
    g = bookrng.generator(name)
    out = []
    for _ in range(n):
        m = g.integer(16)
        wa, wb = g.integer(2), g.integer(2)
        out.append((realize(m, wa, wb, *_noise(g)), MEANINGS[m][0]))
    return out


def fit_probe(X, y, classes, steps=PROBE_STEPS, lr=PROBE_LR):
    """Softmax regression W [d, classes], b [classes] on rows X, starting from zeros, `steps` full-batch gradient steps of rate lr
    on the mean cross-entropy."""
    d, n = len(X[0]), len(X)
    W = [[0.0] * classes for _ in range(d)]
    b = [0.0] * classes
    for _ in range(steps):
        gW = [[0.0] * classes for _ in range(d)]
        gb = [0.0] * classes
        for xi, yi in zip(X, y):
            z = [_sum([xi[j] * W[j][c] for j in range(d)]) + b[c] for c in range(classes)]
            pr = _softmax_list(z)
            for c in range(classes):
                e = (pr[c] - (1.0 if c == yi else 0.0)) / n
                gb[c] += e
                for j in range(d):
                    gW[j][c] += e * xi[j]
        for j in range(d):
            for c in range(classes):
                W[j][c] -= lr * gW[j][c]
        for c in range(classes):
            b[c] -= lr * gb[c]
    return W, b


def probe_accuracy(p, n_labeled, pool="mean"):
    """Accuracy on 200 test sentences of a softmax regression fitted on the vectors (scaled to length 1) of the first n_labeled
    sentences of the training set, labels: animal group."""
    train, test = probe_data(NAME_PROBE + ".train", 64), probe_data(NAME_PROBE + ".test", 200)
    train = train[:n_labeled]
    enc = lambda s: _unit(encode(p, s, pool)[0])[0]   # noqa: E731
    X, y = [enc(s) for s, _ in train], [c for _, c in train]
    W, b = fit_probe(X, y, 4)
    hit = 0
    for s, c in test:
        v = enc(s)
        z = [_sum([v[j] * W[j][k] for j in range(len(v))]) + b[k] for k in range(4)]
        hit += 1 if max(range(4), key=lambda k: (z[k], -k)) == c else 0
    return hit / len(test)


PROBE_LABELS = (8, 16, 64)


def probe_table():
    base, twins = results_b()[("mean", 0.1)]
    un = tg.init_gpt(NAME_B_INIT, CFG_B)
    rows = []
    for n in PROBE_LABELS:
        accs = [probe_accuracy(r["p"], n) for r in [base, *twins]]
        rows.append((str(n), _fx(probe_accuracy(un, n), 3), stable.text(accs[0], accs[1:], lambda x: _fx(x, 3))))
    return (f"A linear head on frozen vectors. Task: the animal group (4 classes) of a sentence. The head is a softmax regression "
            f"(a 16 by 4 matrix and 4 biases, zeros at the start) on the mean-pooled vector scaled to length 1, fitted by {PROBE_STEPS} "
            f"full-batch gradient steps of rate {PROBE_LR:g} on the mean cross-entropy of the first n labelled sentences, and scored on 200 test "
            f"sentences. Sentences: per sentence `integer(16)` for the meaning, `integer(2)` for wa, `integer(2)` for wb, then the four noise "
            f"indices, from `{NAME_PROBE}.train` ({_seed(NAME_PROBE + '.train')}) and `{NAME_PROBE}.test` ({_seed(NAME_PROBE + '.test')}). "
            "Accuracy on the test sentences; chance is 0.25. The trained encoder is the contrastive run above (mean pooling, tau = 0.1).\n\n"
            + _table(["labelled sentences", "encoder with random weights", "contrastive encoder"], rows, "rrr"))


# ------------------------------------------------------------------------------------------------ the cost of one pass

MB = {"V": 50368, "D": 1024, "L": 28, "H": 16, "FF": 2624, "window": 64, "every": 3}   # ModernBERT-large, config.rs
T5 = {"V": 32128, "D": 4096, "L": 24, "H": 64, "dkv": 64, "FF": 10240, "buckets": 32}   # T5-XXL v1.1 encoder, config.rs


def mb_layer_is_full(l):
    return l % MB["every"] == 0


def mb_parameters():
    c = MB
    D, F = c["D"], c["FF"]
    total = c["V"] * D + D                                     # token table and the embedding norm
    for l in range(c["L"]):
        total += (D if l > 0 else 0) + 3 * D * D + D * D + D + 2 * F * D + D * F   # attn_norm (not layer 0), qkv, proj, mlp_norm, wi, wo
    return total + D                                           # final norm


def mb_matrix_parameters():
    D, F = MB["D"], MB["FF"]
    return MB["L"] * (4 * D * D + 3 * D * F)


def band_pairs(T, w):
    """Number of pairs (i, j) with |i - j| <= w among T positions."""
    return sum(min(i, w) + min(T - 1 - i, w) + 1 for i in range(T))


def attention_flops(T, pairs, D):
    """Score products and weighted sum of values: 2 FLOPs per multiply-add, twice, over `pairs` query-key pairs of width D."""
    return 4 * pairs * D


def mb_flops(T, alternating=True):
    D = MB["D"]
    dense = 2 * T * mb_matrix_parameters()
    att = 0
    for l in range(MB["L"]):
        pairs = T * T if (mb_layer_is_full(l) or not alternating) else band_pairs(T, MB["window"])
        att += attention_flops(T, pairs, D)
    return dense, att


def t5_parameters():
    c = T5
    D, F, inner = c["D"], c["FF"], c["H"] * c["dkv"]
    per_block = D + 3 * D * inner + inner * D + D + 2 * D * F + F * D
    return c["V"] * D + c["buckets"] * c["H"] + c["L"] * per_block + D


def cost_of_a_pass():
    rows = []
    P = mb_parameters()
    bytes_w = 4 * P
    for T in (32, 512, 2048, 8192):
        dense, att = mb_flops(T)
        _, att_full = mb_flops(T, alternating=False)
        total = dense + att
        rows.append((_int(T), _sci(dense, 3), _sci(att, 3), _sci(att_full, 3), _fx(total / bytes_w, 1), _fx(total / (T * bytes_w), 2)))
    pairs = band_pairs(8192, MB["window"])
    full_layers = sum(1 for l in range(MB["L"]) if mb_layer_is_full(l))
    return (f"ModernBERT-large as `ModernBertConfig::modernbert_large` states it: V = {MB['V']}, D = {MB['D']}, {MB['L']} layers, "
            f"{MB['H']} heads, GeGLU width {MB['FF']}, full attention in layers 0, 3, 6, ... ({full_layers} of {MB['L']}), a window of "
            f"{MB['window']} positions to each side in the others. Parameters counted from the tensor manifest: {_int(mb_parameters())}; "
            f"in the {MB['L']} layers' matrices (4 D^2 + 3 D F each): {_int(mb_matrix_parameters())}; fp32 weights {_fx(bytes_w / 1e9, 3)} GB "
            f"({_fx(16 * P / 1e9, 2)} GB with a gradient and two AdamW moments). A pass over T tokens costs 2 T (matrix parameters) FLOPs in the "
            "matrices and 4 D (pairs) FLOPs per layer in attention, pairs = T^2 in a full layer and the pairs with |i - j| <= 64 in a local "
            f"one (at T = 8192 that is {_int(pairs)} of {_int(8192 * 8192)}). FLOPs per weight byte: all FLOPs of the pass over the bytes of "
            "the weights, read once; the last column divides by T to give what T separate one-token passes would reach.\n\n"
            + _table(["T", "matrix FLOPs", "attention FLOPs, alternating", "attention FLOPs, all layers full",
                      "FLOP per weight byte, one pass", "FLOP per weight byte, token by token"], rows, "rrrrrr"))


def t5_cost():
    c = T5
    P = t5_parameters()
    D, F, inner = c["D"], c["FF"], c["H"] * c["dkv"]
    matrix = c["L"] * (3 * D * inner + inner * D + 3 * D * F)
    T = 512
    dense = 2 * T * matrix
    att = c["L"] * attention_flops(T, T * T, inner)
    return (f"T5-XXL v1.1 encoder as `T5Config::xxl` states it: V = {c['V']}, D = {D}, {c['L']} blocks, {c['H']} heads of width {c['dkv']} "
            f"(attention width {inner}), gated feed-forward width {F}, one table of {c['buckets']} buckets x {c['H']} heads for the "
            f"relative-position bias. Parameters {_int(P)} ({_fx(P / 1e9, 3)} billion); fp32 weights {_fx(4 * P / 1e9, 2)} GB. One pass over "
            f"T = {T} tokens: matrix FLOPs {_sci(dense, 3)}, attention FLOPs {_sci(att, 3)} (full attention in every block), "
            f"{_fx(100 * att / (att + dense), 1)} percent of the total.")


def scan_cost():
    N, D, T = 1_000_000, 1024, 32
    dense, att = mb_flops(T)
    query = dense + att
    scan_flops, scan_bytes = 2 * N * D, 4 * N * D
    docs_dense, docs_att = mb_flops(256)
    rows = [("embed a query of 32 tokens", _sci(query, 3), _sci(4 * mb_parameters(), 3), _fx(query / (4 * mb_parameters()), 1)),
            ("score it against 1,000,000 stored vectors of 1024 fp32 numbers", _sci(scan_flops, 3), _sci(scan_bytes, 3), _fx(scan_flops / scan_bytes, 2)),
            ("embed one document of 256 tokens (once, at index time)", _sci(docs_dense + docs_att, 3), _sci(4 * mb_parameters(), 3),
             _fx((docs_dense + docs_att) / (4 * mb_parameters()), 1)),
            ("embed 1,000,000 such documents", _sci(N * (docs_dense + docs_att), 3), "weights stay resident", "-")]
    return ("Retrieval with the encoder above as the embedding model (D = 1024), counting FLOPs and the bytes that must be read at least "
            "once: for an embedding pass the weights, for a scan the stored vectors.\n\n"
            + _table(["step", "FLOPs", "bytes read", "FLOP per byte"], rows, "lrrr"))


# ------------------------------------------------------------------------------------------------ T5's buckets

def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def t5_bucket(rel, num_buckets=32, max_distance=128, single=False):
    """T5's bidirectional relative-position bucket of rel = key - query: half = num_buckets / 2 is added when rel > 0; offsets
    |rel| < max_exact = half / 2 are their own bucket; larger ones go to max_exact + trunc(ln(|rel| / max_exact) /
    ln(max_distance / max_exact) (half - max_exact)), at most half - 1. With `single` every step of the logarithmic branch is
    rounded to float32 (the engine's rule)."""
    half = num_buckets // 2
    exact = half // 2
    a = abs(rel)
    base = half if rel > 0 else 0
    if a < exact:
        return base + a
    if single:
        denom = f32(math.log(max_distance / exact))
        big = f32(f32(f32(math.log(f32(a / exact))) / denom) * float(half - exact))
    else:
        big = math.log(a / exact) / math.log(max_distance / exact) * (half - exact)
    return base + min(exact + int(big), half - 1)


def t5_buckets():
    offsets = (0, 1, 2, 7, 8, 9, 12, 16, 24, 32, 48, 64, 96, 127, 128, 200, 1000)
    rows = [(o, t5_bucket(-o), t5_bucket(o) if o > 0 else "-") for o in offsets]
    ranges = []
    for b in range(8, 16):
        offs = [a for a in range(1, 400) if t5_bucket(-a) == b]
        last = b == 15           # the last bucket also takes every larger distance
        ranges.append((b, f"{min(offs)} and beyond" if last else f"{min(offs)} to {max(offs)}", "all larger" if last else len(offs)))
    disagree = [r for r in range(-4000, 4001) if t5_bucket(r) != t5_bucket(r, single=True)]
    T = 128
    table = [[t5_bucket(j - i) for j in range(T)] for i in range(T)]
    seen = sorted({v for row in table for v in row})
    missing = [b for b in range(32) if b not in seen]
    return ("T5's encoder: 32 buckets, maximum distance 128, so half = 16 buckets for each direction and offsets below max_exact = 8 are "
            "their own bucket. The offset is key - query: `behind` is a key before the query (offset -d), `ahead` a key after it.\n\n"
            + _table(["distance d", "bucket, key d behind", "bucket, key d ahead"], rows, "rrr")
            + "\n\nThe eight buckets that share the logarithmic range, as distances behind the query (those ahead are 16 higher):\n\n"
            + _table(["bucket", "distances", "count"], ranges, "rlr")
            + f"\n\nFor all offsets from -4000 to 4000, evaluating the logarithmic branch in float32 (every operation rounded to a "
            f"32-bit float, as the engine's note says it does) or in double gives the same bucket: {len(disagree)} offsets differ. "
            f"On a window of T = 128 positions the table holds bucket numbers {seen[0]} to {seen[-1]} except {missing} "
            f"(bucket 16 would be 'ahead by distance 0', which does not exist); query 0 sees key 127 in bucket {table[0][127]} "
            f"and query 127 sees key 0 in bucket {table[127][0]}.")


# ------------------------------------------------------------------------------------------------ the gate that cannot be seen

NAME_GEGLU = "encoders.geglu"


def geglu_swap():
    """Mean |gelu(a) b - gelu(b) a| over mean |gelu(a) b| for a, b drawn with standard deviation s. The exchanged product is what a
    mistaken GeGLU computes when it activates the second half instead of the first."""
    g = bookrng.generator(NAME_GEGLU)
    n = 20000
    z = [(g.normal(), g.normal()) for _ in range(n)]
    rows = []
    for s in (0.01, 0.1, 1.0, 3.0):
        diff = _sum([abs(tb.gelu(s * a) * s * b - tb.gelu(s * b) * s * a) for a, b in z]) / n
        ref = _sum([abs(tb.gelu(s * a) * s * b) for a, b in z]) / n
        rows.append((f"{s:g}", _sci(ref, 3), _sci(diff, 3), _sci(diff / ref, 3)))
    return (f"{_seed(NAME_GEGLU)} for `{NAME_GEGLU}`. Draws: {n:,} pairs, each `normal()` for a then `normal()` for b; the pair used at "
            "scale s is (s a, s b). GELU is the exact Gaussian-cumulative form (library `erf`).\n\n"
            + _table(["s", "mean |gelu(a) b|", "mean |gelu(a) b - gelu(b) a|", "ratio"], rows, "rrrr"))


if __name__ == "__main__":
    for f in (vocabulary_a, blank_example, corruption_example, slot_table, joint_blanks, fill_table, vocabulary_b, padding_example,
              infonce_example, false_negatives, contrastive_training, retrieval_table, temperature_table, pooling_table, tau_floor_text, probe_table,
              pca_summary, cost_of_a_pass, t5_cost, scan_cost, t5_buckets, geglu_swap):
        print(f"--- {f.__name__}")
        print(f())
