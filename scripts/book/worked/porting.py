# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Porting a Model from Its Reference".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each
public function without arguments returns a markdown string that the chapter embeds between `<!-- worked: porting.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.

* two import mistakes in a 6-layer decoder, compared with its golden layer by layer in two ways: chained (the implementation
  runs on its own outputs) and isolated (every layer is fed the golden's own input to that layer);
* which mistakes a golden can see, as a function of the text it was generated on;
* the precision at which a golden is generated, and what that does to its bytes and its digest;
* the identity record that pairs a golden with the checkpoint it was generated from;
* the two gates of a port, forward parity and the gradient check, and which mistakes each one sees;
* two mistakes that the chapter's exercises ask about: a wrong wire between two correct stages (seen by the chained comparison and by
  no isolated one) and a factor of 2 on an attention output or on the queries before their norm (seen at a tap before the norm).

Everything is reused by import, never copied: the decoder of Volume VI, Chapter 1 and its initialisation (`modern_decoder`), the
forward pass in 64-bit and emulated 32-bit arithmetic, the bf16 rounding of the published weights, the rounding counts and the
safetensors writer (`importing_models`), and the two-layer network with its backward pass and finite differences
(`gradient_checking`). No number here comes from training or sampling, so none needs the nudge rule of `stable.py`.

Swedish Embedded AB implements solutions for porting published models onto its clients' own engines and proving them correct. If
your team needs expertise in reference goldens and layer-by-layer parity, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import hashlib
import struct

import bookrng
import gradient_checking as gc
import importing_models as im
import modern_decoder as md

U = im.U32                      # 2^-24, the unit round-off of the 32-bit format
T_TEXT = len(md.IDS)            # 8 positions of the text of Volume VI, Chapter 1
BUG_KV_LAYER = 1                # keys and values exchanged in this layer
BUG_GU_LAYER = 4                # gate and up exchanged in this layer
NAME_REDRAWN = "porting.redrawn"
NAME_GATES = "porting.two_gates"
GATE_DIMS = (3, 5, 7, 2)        # B, D, H, O of the two-gates network: all different (Volume III, Chapter 6)
GATE_H = 1e-5                   # the central-difference step of the per-entry check
_table, _sci = im._table, im._sci


def _fmt(v, d=2):
    return "0" if v == 0 else _sci(v, d)


# ------------------------------------------------------------------------------------------------ mistakes made on purpose


def exchanged(p, a, b, layers=None):
    """A copy of the parameters p with the tensors a and b of the listed layers (all layers when None) exchanged."""
    q = dict(p)
    q["layers"] = [dict(lay) for lay in p["layers"]]
    for i, lay in enumerate(q["layers"]):
        if layers is None or i in layers:
            lay[a], lay[b] = lay[b], lay[a]
    return q


def _times(v, factor):
    return [_times(x, factor) for x in v] if isinstance(v, list) else factor * v


def scaled(p, name, factor, layers=None):
    """A copy of p with the vector or matrix `name` of the listed layers (all layers when None) multiplied by `factor`."""
    q = dict(p)
    q["layers"] = [dict(lay) for lay in p["layers"]]
    for i, lay in enumerate(q["layers"]):
        if layers is None or i in layers:
            lay[name] = _times(lay[name], factor)
    return q


def two_mistakes(p):
    """The port of the 6-layer model with keys and values exchanged in layer 1 and gate and up exchanged in layer 4."""
    return exchanged(exchanged(p, "Wk", "Wv", [BUG_KV_LAYER]), "Wg", "Wu", [BUG_GU_LAYER])


# ------------------------------------------------------------------------------------------------ chained and isolated parity


def stored(taps):
    """Hidden states rounded to 32 bits, as a golden file stores them."""
    return [[[im.r32(x) for x in row] for row in t] for t in taps]


def golden_taps(p, cfg, ids):
    """The golden: the 64-bit forward pass on `ids`, every hidden state stored in 32 bits (embeddings, each layer's output,
    the final norm), followed by the stored logits."""
    taps = []
    logits = im.run(p, cfg, list(ids), taps=taps)
    return stored(taps) + stored([logits])


def chained_taps(p, cfg, ids):
    """The implementation under test run end to end on its own outputs: 32-bit, sums left to right."""
    taps = []
    logits = im.run(p, cfg, list(ids), r=im.r32, order="forward", taps=taps)
    return taps + [logits]


def stage_functions(p, cfg):
    """The stages of the implementation under test as functions of their input state, in 32-bit arithmetic with sums left to
    right: layer k alone, then the final norm alone, then the head alone. A layer or the final norm is run as a decoder of its own
    whose embedding table is the state it is given (row t at position t, so the rotation sees the same positions)."""
    L = len(p["layers"])

    def alone(layers):
        def f(state):
            one = {"emb": state, "layers": layers, "gn": p["gn"], "out": p["out"]}
            taps = []
            im.run(one, cfg, list(range(len(state))), r=im.r32, order="forward", taps=taps)
            return taps[1]
        return f

    def head(state):
        return [[im.dot(h, p["out"][i], im.r32, "forward") for i in range(cfg["V"])] for h in state]

    return [alone([lay]) for lay in p["layers"]] + [alone([]), head]


def isolated_parity(stage, want, bound):
    """The routine of the chapter's pseudocode: every stage s (in increasing order) whose output on the golden's stored input
    want[s] differs from the golden's stored output want[s + 1] by a rel_l2 above bound[s]."""
    assert len(stage) == len(bound) and len(want) == len(stage) + 1, "one golden state before and after every stage"
    above = []
    for s, f in enumerate(stage):
        got = f(want[s])
        assert len(got) == len(want[s + 1]), "a stage changed the shape of the golden state"
        if im.rel_l2(got, want[s + 1]) > bound[s]:
            above.append(s)
    return above


def isolated_taps(p, cfg, gold):
    """Each stage of the implementation under test fed the golden's own input to that stage. Position 0 is the embedding lookup,
    compared directly; stage s + 1 is the stage function s run on the stored golden state s (the layers, then the final norm, then
    the head)."""
    out = [[list(p["emb"][t]) for t in md.IDS]]
    out += [f(gold[s]) for s, f in enumerate(stage_functions(p, cfg))]
    return out


def stage_names(L):
    return ["embeddings"] + [f"layer {k}" for k in range(L)] + ["final norm", "logits"]


def chained_bounds(cfg, T):
    """Bound on the relative error of each stored state of a chained run: roundings of everything before it, plus one for the
    stored golden, times u."""
    layer, head = im.stages(cfg, T)
    n_layer, n_norm, n_head = sum(c for _, c in layer), head[0][1], head[1][1]
    L = cfg["L"]
    b = [im.bound_at(cfg, T, k) for k in range(L + 1)]
    b.append((L * n_layer + n_norm + 1) * U)
    b.append((L * n_layer + n_norm + n_head + 1) * U)
    return b


def isolated_bounds(cfg, T):
    """Bound for an isolated stage: one rounding for its stored input, its own roundings, one for its stored output."""
    layer, head = im.stages(cfg, T)
    n_layer, n_norm, n_head = sum(c for _, c in layer), head[0][1], head[1][1]
    return [U] + [(n_layer + 2) * U] * cfg["L"] + [(n_norm + 2) * U, (n_head + 2) * U]


def two_mistakes_data():
    """[(stage, chained correct, chained mistaken, chained bound, isolated correct, isolated mistaken, isolated bound)]."""
    cfg, p = im.deep_setup()
    gold = golden_taps(p, cfg, md.IDS)
    bad = two_mistakes(p)
    cg, cb = chained_taps(p, cfg, md.IDS), chained_taps(bad, cfg, md.IDS)
    ig, ib = isolated_taps(p, cfg, gold), isolated_taps(bad, cfg, gold)
    cbd, ibd = chained_bounds(cfg, T_TEXT), isolated_bounds(cfg, T_TEXT)
    rows = []
    for i, nm in enumerate(stage_names(cfg["L"])):
        rows.append((nm, im.rel_l2(cg[i], gold[i]), im.rel_l2(cb[i], gold[i]), cbd[i],
                     im.rel_l2(ig[i], gold[i]), im.rel_l2(ib[i], gold[i]), ibd[i]))
    return rows


def two_mistakes_table():
    cfg, _ = im.deep_setup()
    layer, head = im.stages(cfg, T_TEXT)
    n_layer, n_norm, n_head = sum(c for _, c in layer), head[0][1], head[1][1]
    rows = two_mistakes_data()
    body = []
    for nm, cg, cb, cbd, ig, ib, ibd in rows:
        body.append((nm, _fmt(cg), _fmt(cb), _fmt(cbd), "above" if cb > cbd else "within",
                     _fmt(ig), _fmt(ib), _fmt(ibd), "above" if ib > ibd else "within"))
    chained_first = next(r[0] for r in rows if r[2] > r[3])
    chained_above = sum(1 for r in rows if r[2] > r[3])
    cfg, p = im.deep_setup()
    gold = golden_taps(p, cfg, md.IDS)
    names = stage_names(cfg["L"])[1:]
    isolated_above = [names[s] for s in isolated_parity(stage_functions(two_mistakes(p), cfg), gold, isolated_bounds(cfg, T_TEXT)[1:])]
    seed = bookrng.seed_of(im.NAME_BIS)
    return (f"The 6-layer tiny decoder (V = 11, D = 6, H = 4, 2 key-value heads of width 4, F = 10, L = 6, rotary base 100, eps 1e-6; "
            f"weights from the generator `{im.NAME_BIS}`, seed {seed:#018x}, drawn in the order of Volume VI, Chapter 1 and rounded "
            "to bf16) on the 8-token text 3, 9, 1, 7, 4, 4, 10, 2. The golden is the 64-bit forward pass with every hidden state "
            "and the logits stored in 32 bits; each stored state is a tap, a state at which two runs can be compared. The implementation "
            "under test is the 32-bit forward pass (every operation rounded, the gated product of the feed-forward block evaluated as "
            "fl(fl(g / fl(1 + fl(exp(-g)))) * up), that is silu(g) = g / (1 + exp(-g)) with exp, add, divide and the product with up "
            f"rounded in that order, the four roundings of Volume VI, Chapter 2; sums left to right) of an import with keys and values exchanged in layer {BUG_KV_LAYER} and gate and up exchanged in "
            f"layer {BUG_GU_LAYER}; the correct import is the same pass without the two mistakes. Chained: the implementation runs "
            "end to end and each of its states is compared with the golden's. Isolated: each stage is run alone on the golden's "
            "stored input to that stage (layer k on the stored output of layer k - 1, the final norm on the stored output of layer "
            "5, the head on the stored final norm). Every number is rel_l2 against the golden state. A layer counts "
            f"{n_layer} roundings, the final norm {n_norm} and the head {n_head} (the counting rule of Volume VI, Chapter 2: a sum "
            "of n products costs n, each further operation 1). Chained bound after k stages: (roundings of all stages so far + 1) x "
            f"2^-24; isolated bound: (roundings of the stage + 2) x 2^-24, one rounding for the stored input and one for the stored "
            "output (the embedding lookup has no arithmetic and its bound is 2^-24).\n\n"
            + _table(["stage", "chained, correct", "chained, mistaken", "chained bound", "", "isolated, correct",
                      "isolated, mistaken", "isolated bound", ""], body, "lrrrlrrrl")
            + f"\n\nChained: {chained_above} of the {len(rows)} states are above their bound, the first at {chained_first}. "
            f"Isolated: above their bound are {', '.join(isolated_above)}, and nothing else.")


# ------------------------------------------------------------------------------------------------ what a golden's text can see


GOLDEN_TEXTS = [("1 token: 3", (3,)), ("2 tokens: 3, 9", (3, 9)), ("8 tokens: the text", md.IDS)]


def text_mistakes(p, cfg):
    """(label, how to run the mistaken model on ids) for the five mistakes of the golden-input table."""
    return [
        ("rotary base 10000 for the published 100", lambda ids: im.run(p, cfg, list(ids), base=10000.0)),
        ("query-norm and key-norm gains exchanged", lambda ids: im.run(exchanged(p, "qn", "kn"), cfg, list(ids))),
        ("scores divided by d_h, not sqrt(d_h)", lambda ids: im.run(scaled(p, "qn", 0.5), cfg, list(ids))),
        ("keys and values exchanged", lambda ids: im.run(exchanged(p, "Wk", "Wv"), cfg, list(ids))),
        ("gate and up exchanged", lambda ids: im.run(exchanged(p, "Wg", "Wu"), cfg, list(ids))),
    ]


def golden_text_data():
    """{(mistake, text label): rel_l2 of the logits}, the bound of each text, and the mistake labels."""
    cfg, p, _ = im.published()
    res = {}
    for label, ids in GOLDEN_TEXTS:
        gold = im.run(p, cfg, list(ids))
        for name, f in text_mistakes(p, cfg):
            res[(name, label)] = im.rel_l2(f(ids), gold)
    bounds = {label: im.bound_logits(cfg, len(ids)) for label, ids in GOLDEN_TEXTS}
    return res, bounds, [n for n, _ in text_mistakes(p, cfg)]


def golden_text():
    res, bounds, names = golden_text_data()
    body = []
    for name in names:
        cells = []
        for label, _ in GOLDEN_TEXTS:
            v = res[(name, label)]
            cells.append("exactly 0" if v == 0 else (_fmt(v) + (" (below bound)" if v <= bounds[label] else "")))
        body.append([name] + cells)
    body.append(["parity tolerance of the logits"] + [_fmt(bounds[label]) for label, _ in GOLDEN_TEXTS])
    cfg, _, _ = im.published()
    counts = ", ".join(f"{im.bound_logits(cfg, len(ids)) / im.U32:.0f}" for _, ids in GOLDEN_TEXTS)
    seed = bookrng.seed_of(md.NAME_TINY)
    return ("The published 2-layer tiny decoder of Volume VI, Chapter 2 (V = 11, D = 6, H = 4, 2 key-value heads of width 4, "
            f"F = 10, rotary base 100, eps 1e-6; weights from the generator `{md.NAME_TINY}`, seed {seed:#018x}, drawn in the order "
            "of Volume VI, Chapter 1 and rounded to bf16). Three goldens are generated, one for each text in the columns (prefixes "
            "of the text 3, 9, 1, 7, 4, 4, 10, 2). Each row makes one mistake in every layer of the port and reports rel_l2 of its "
            "logits against the golden's, both in 64-bit arithmetic, so the number is the mistake alone. The query-norm gain is "
            "halved to divide every score by d_h = 4 instead of sqrt(d_h) = 2, which multiplies each score by 0.5. The last row is "
            "the parity tolerance of Volume VI, Chapter 2 for that text (the rounding count of a 32-bit pass times 2^-24; it grows "
            "with the number of positions T through the softmax, which costs T + 3 roundings, and the weighted sum of values, which "
            f"costs T, so each layer counts 90 + 2T; the counts N of the three columns are {counts}).\n\n"
            + _table(["mistake in the port"] + [label for label, _ in GOLDEN_TEXTS], body, "lrrr"))


# ------------------------------------------------------------------------------------------------ the precision of a golden


def _f32_bytes(values):
    return b"".join(struct.pack("<f", v) for v in values)


def _golden_values(p, cfg, r, order):
    """Every hidden state and the logits of one reference run on the 8-token text, flattened in tap order and stored in 32 bits."""
    taps = []
    logits = im.run(p, cfg, list(md.IDS), r=r, order=order, taps=taps)
    return [im.r32(x) for t in taps + [logits] for x in im._flatten(t)]


REFERENCE_RUNS = [("64-bit", im._ident, "forward"), ("64-bit", im._ident, "reverse"), ("64-bit", im._ident, "pairwise"),
                  ("32-bit", im.r32, "forward"), ("32-bit", im.r32, "reverse"), ("32-bit", im.r32, "pairwise")]
ORDER_NAME = {"forward": "left to right", "reverse": "right to left", "pairwise": "pairwise"}


def golden_precision_data():
    cfg, p, _ = im.published()
    base = _golden_values(p, cfg, im._ident, "forward")
    out = []
    for arith, r, order in REFERENCE_RUNS:
        vals = _golden_values(p, cfg, r, order)
        diff = sum(1 for a, b in zip(vals, base) if struct.pack("<f", a) != struct.pack("<f", b))
        rel = max(abs(a - b) / abs(b) for a, b in zip(vals, base) if b != 0)
        out.append((arith, ORDER_NAME[order], diff, len(vals), rel, hashlib.sha256(_f32_bytes(vals)).hexdigest()[:16]))
    return out


def golden_precision():
    rows = golden_precision_data()
    n = rows[0][3]
    body = [(a, o, f"{d} of {nn}", _fmt(rel), f"`{h}`") for a, o, d, nn, rel, h in rows]
    return ("Six reference runs of the published 2-layer tiny decoder (the model of the previous table) on the 8-token text, each "
            f"stored as a golden of {n} 32-bit numbers: the 8 x 6 embeddings, the outputs of the two layers, the final norm (each "
            "8 x 6) and the 8 x 11 logits, in that order, little-endian. A run is either 64-bit or 32-bit arithmetic (every operation "
            "rounded to 32 bits, as in Volume VI, Chapter 2) with one of the three summation orders of that chapter, applied to every "
            "sum. Each stored golden is compared, number by number, with the first one (64-bit, left to right): how many stored "
            "32-bit patterns differ, and the largest relative difference |a - b| / |b| over the numbers. The digest is the first 16 "
            "hex digits of the SHA-256 (Secure Hash Algorithm, 256 bits) of the stored bytes.\n\n"
            + _table(["reference arithmetic", "summation order", "stored numbers that differ", "largest relative difference",
                      "digest"], body, "llrrl"))


# ------------------------------------------------------------------------------------------------ pairing a golden with its checkpoint

IDENTITY_FIELDS = ("V", "D", "L", "H", "HKV", "DH", "F")


def identity_mismatches(recorded, actual):
    """The routine of the chapter's pseudocode: the names of the fields of `actual` (checked in the order listed) whose integers
    differ from those in `recorded`, in the same order; a field that `recorded` does not hold is an error."""
    bad = []
    for name, value in actual:
        found = False
        for rname, rvalue in recorded:
            if rname == name:
                found = True
                if rvalue != value:
                    bad.append(name)
        if not found:
            raise KeyError(f"the golden does not record the field {name}, which the test needs")
    return bad


def _checkpoint_digest(p, cfg):
    blob, _, _ = im.write_safetensors([(n, "BF16", s, v) for n, s, v in im.reference_layout(p, cfg)], {"format": "pt"})
    return hashlib.sha256(blob).hexdigest()


def pairing_data():
    """For each checkpoint offered to the golden of the 6-layer model: (name, identity mismatches, digest equal, rel_l2 at the
    embeddings and of the logits when the comparison runs)."""
    cfg, p = im.deep_setup()
    gold = golden_taps(p, cfg, md.IDS)
    recorded = {k: cfg[k] for k in IDENTITY_FIELDS}
    recorded_pairs = list(recorded.items())
    digest = _checkpoint_digest(p, cfg)
    tiny_cfg, tiny_p, _ = im.published()
    re_p = im.bf16_params(md.init_decoder(cfg, name=NAME_REDRAWN))
    out = []
    for name, c, q in (("the 6-layer model the golden was generated from", cfg, p),
                       ("the 2-layer tiny model (same widths, other depth and weights)", tiny_cfg, tiny_p),
                       (f"6 layers, the same widths, weights drawn again", cfg, re_p)):
        bad = identity_mismatches(recorded_pairs, [(k, c[k]) for k in IDENTITY_FIELDS])
        mism = [f"{k}: golden {recorded[k]}, checkpoint {c[k]}" for k in bad]
        same = _checkpoint_digest(q, c) == digest
        if mism:
            out.append((name, mism, same, None, None))
        else:
            taps = chained_taps(q, c, md.IDS)
            out.append((name, mism, same, im.rel_l2(taps[0], gold[0]), im.rel_l2(taps[-1], gold[-1])))
    return out, recorded, digest


def pairing():
    rows, recorded, digest = pairing_data()
    body = []
    for name, mism, same, r_emb, r_log in rows:
        ident = "agrees" if not mism else "refused: " + "; ".join(mism)
        cmp_ = "not run" if r_emb is None else f"embeddings {_fmt(r_emb)}, logits {_fmt(r_log)}"
        body.append((name, ident, "same" if same else "different", cmp_))
    rec = ", ".join(f"{k} {v}" for k, v in recorded.items())
    return ("The golden of the 6-layer model of the two-mistakes table records, beside its tensors, the identity of the checkpoint "
            f"it was generated from (the integers that fix every tensor shape: {rec}) and the SHA-256 of that checkpoint's file "
            f"(the {8 * recorded['L'] + 3} tensors of the published layout of Volume VI, Chapter 2 in bf16, layer after layer in the order of that chapter's tables (input norm, fused q-k-v projection, query norm, key norm, "
            f"output projection, post-attention norm, fused gate-up projection, down projection; each matrix stored [out, in]), then the "
            f"final norm and the head, written as one safetensors "
            "file exactly as that chapter writes one: the compact JSON (JavaScript Object Notation) header with `__metadata__` `{\"format\":\"pt\"}` first, "
            "padded with spaces to a multiple of 8, then the data; "
            f"digest beginning `{digest[:16]}`). Three checkpoints are offered to it. The identity is compared field by field "
            "and a mismatch refuses the pairing by name; the digest is compared only to say which file it is. When the identity "
            "agrees, the 32-bit port of that checkpoint is run on the text and compared with the golden (rel_l2). The redrawn "
            f"weights come from the generator `{NAME_REDRAWN}`, seed {bookrng.seed_of(NAME_REDRAWN):#018x}, in the same order and "
            "with the same scales, then rounded to bf16.\n\n"
            + _table(["checkpoint offered", "identity check", "file digest", "comparison"], body, "llll"))


# ------------------------------------------------------------------------------------------------ the two gates of a port


def _outputs(P, X, T, act):
    """The network's outputs Y = E + T (E is the error the forward pass returns), flattened."""
    _, _, _, E = gc.forward(P, X, T, gc.IDENT, act)
    return [e + t for re, rt in zip(E, T) for e, t in zip(re, rt)]


def two_gates_data():
    B, D, H, O = GATE_DIMS
    P, X, T = gc._random_problem(NAME_GATES, B, D, H, O)
    y_ref = _outputs(P, X, T, "tanh")
    ports = [("none (a correct port)", "tanh", "none"),
             ("ReLU in place of tanh, in the forward and the backward pass", "relu", "none"),
             ("backward: tanh derivative taken from the pre-activation", "tanh", "deriv_from_preact")]
    out = []
    for label, act, bug in ports:
        y = _outputs(P, X, T, act)
        par = im.rel_l2(y, y_ref)
        g = gc.grad_vector(gc.backward(P, X, T, bug, act))
        loss = lambda Q, act=act: gc.loss_of(Q, X, T, gc.IDENT, act)
        n = gc.numeric_gradient(loss, P, GATE_H)
        fails = sum(1 for a, b in zip(g, n) if not gc.within(a, b, gc.F64_ATOL, gc.F64_RTOL))
        out.append((label, par, fails, len(g)))
    return out


def two_gates():
    B, D, H, O = GATE_DIMS
    rows = two_gates_data()
    body = [(lab, _fmt(par), "pass" if par == 0 else "FAIL", f"{f} of {n}", "pass" if f == 0 else "FAIL") for lab, par, f, n in rows]
    seed = bookrng.seed_of(NAME_GATES)
    return (f"The two-layer network of Volume III, Chapter 6 (B = {B} examples, D = {D} inputs, H = {H} hidden units, O = {O} outputs, "
            f"{rows[0][3]} parameters, mean squared error), as the reference with tanh. Its inputs, targets and weights come from "
            f"the generator `{NAME_GATES}`, seed {seed:#018x}, drawn in the order X (row by row), T (row by row), W1 (row by row), b1, "
            "W2 (row by row), b2, each entry `normal()`, and the four tensors W1, b1, W2, b2 (the last four of the six) multiplied "
            f"by 0.5; W1 has {H} rows of {D} entries and W2 has {O} rows of {H}, so the hidden layer is tanh(W1 x + b1) and "
            "the outputs are W2 a + b2, and the loss is the sum of the squared errors divided by B x O. Every port runs in 64-bit arithmetic in the same "
            "order of operations as the reference, so a correct port's outputs are equal to the reference's and the forward gate "
            "demands rel_l2 = 0 exactly; a real port, with its own order of operations, would use the rounding bound instead. The "
            f"gradient check compares the port's backward pass with the central difference of the port's own loss, step h = {GATE_H:g} "
            f"on every parameter, under the gate |a - n| <= {gc.F64_ATOL:g} + {gc.F64_RTOL:g} max(|a|, |n|) of Volume III, Chapter 6.\n\n"
            + _table(["mistake in the port", "forward: rel_l2 of the outputs", "forward gate", "gradient check: entries failing",
                      "gradient gate"], body, "lrlrl"))


# ------------------------------------------------------------------------------------------------ a mistake of composition

COMPOSITION_LAYER = 3           # the layer that is wired to the wrong state
COMPOSITION_SOURCE = 1          # ... and reads the output of this layer instead of the output of layer COMPOSITION_LAYER - 1


def wired_taps(p, cfg, ids, reads):
    """The implementation under test run end to end like `chained_taps`, with stage s reading the state `reads[s]` (0 is the
    embeddings, k + 1 the output of layer k) instead of the state before it. Every stage is the correct stage function; only the
    plumbing between them differs. Returns the same list as `chained_taps`: the embeddings, each layer's output, the final norm,
    the logits."""
    fs = stage_functions(p, cfg)
    states = [[list(p["emb"][t]) for t in ids]]
    for s, f in enumerate(fs):
        states.append(f(states[reads[s]]))
    return states


def composition_data():
    """[(stage, chained, chained bound, isolated, isolated bound)] for the port whose layer COMPOSITION_LAYER reads the output of
    layer COMPOSITION_SOURCE, and the stages above their bound in the isolated comparison."""
    cfg, p = im.deep_setup()
    gold = golden_taps(p, cfg, md.IDS)
    S = cfg["L"] + 2
    right = list(range(S))
    wrong = list(right)
    wrong[COMPOSITION_LAYER] = COMPOSITION_SOURCE + 1
    assert wired_taps(p, cfg, md.IDS, right) == chained_taps(p, cfg, md.IDS), "stage functions and the end-to-end run must agree"
    chained = wired_taps(p, cfg, md.IDS, wrong)
    iso = isolated_taps(p, cfg, gold)          # the stages are right: the test chooses their inputs, not the port
    cbd, ibd = chained_bounds(cfg, T_TEXT), isolated_bounds(cfg, T_TEXT)
    rows = [(nm, im.rel_l2(chained[i], gold[i]), cbd[i], im.rel_l2(iso[i], gold[i]), ibd[i])
            for i, nm in enumerate(stage_names(cfg["L"]))]
    return rows, isolated_parity(stage_functions(p, cfg), gold, ibd[1:])


def composition():
    rows, iso_above = composition_data()
    body = [(nm, _fmt(c), _fmt(cb), "above" if c > cb else "within", _fmt(i), _fmt(ib), "above" if i > ib else "within")
            for nm, c, cb, i, ib in rows]
    first = next(r[0] for r in rows if r[1] > r[2])
    n_above = sum(1 for r in rows if r[1] > r[2])
    return (f"The 6-layer decoder, text, golden and bounds of the two-mistakes table, ported with every stage correct and one wire wrong: "
            f"layer {COMPOSITION_LAYER} reads the output of layer {COMPOSITION_SOURCE} instead of the output of layer "
            f"{COMPOSITION_LAYER - 1} (layer {COMPOSITION_LAYER - 1} is still computed and its output is discarded). Each stage is "
            "the 32-bit stage function of the two-mistakes table, so the same code run with every wire right is the chained run of "
            "that table to the last bit. Chained: the wired port end to end against the golden's states. Isolated: every stage fed the "
            "golden's own input, which the test supplies and so bypasses the wire. Every number is rel_l2 against the golden "
            "state.\n\n"
            + _table(["stage", "chained", "chained bound", "", "isolated", "isolated bound", ""], body, "lrrlrrl")
            + f"\n\nChained: {n_above} of the {len(rows)} states are above their bound, the first at {first}. Isolated: "
            f"{len(iso_above)} stages are above their bound.")


# ------------------------------------------------------------------------------------------------ a scale mistake and a norm

SCALE_LAYER = 0                 # the layer that carries the scale mistake
SCALE_FACTOR = 2.0


def _layer_taps(p, cfg, ids):
    """64-bit taps of layer SCALE_LAYER: the queries before the QK-norm, the attention output before the residual, the input of
    the second RMS norm (after the residual add), the output of the layer and the logits. The attention output is the difference
    between the layer's output and its input when W_down is zero, so that the MLP adds nothing."""
    X = [list(p["emb"][t]) for t in ids]
    lay = p["layers"][SCALE_LAYER]
    q0 = im.matmul(im.rms(X, lay["g1"], im._ident, "forward", md.tb.EPS), lay["Wq"], im._ident, "forward")
    no_mlp = dict(lay, Wd=[[0.0] * len(row) for row in lay["Wd"]])
    taps = []
    im.run({"emb": X, "layers": [no_mlp], "gn": p["gn"], "out": p["out"]}, cfg, list(range(len(X))), taps=taps)
    x1 = taps[1]
    attn = [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(x1, X)]
    taps = []
    logits = im.run(p, cfg, list(ids), taps=taps)
    return {"queries": q0, "attention": attn, "norm2_input": x1, "layer": taps[SCALE_LAYER + 1], "logits": logits}


def scale_data():
    """[(mistake, tap, rel_l2, bound)] for the two scale mistakes on the published 2-layer model, 64-bit arithmetic."""
    cfg, p, _ = im.published()
    layer, _ = im.stages(cfg, T_TEXT)
    through = lambda n: (sum(c for _, c in layer[:n]) + 1) * U       # the bound after the first n stages of one layer
    bounds = {"queries": through(2), "attention": through(8), "norm2_input": through(9),
              "layer": im.bound_at(cfg, T_TEXT, SCALE_LAYER + 1), "logits": im.bound_logits(cfg, T_TEXT)}
    gold = _layer_taps(p, cfg, md.IDS)
    mistakes = [("attention output (W_O) times " + f"{SCALE_FACTOR:g}", scaled(p, "Wo", SCALE_FACTOR, [SCALE_LAYER]),
                 ["attention", "norm2_input", "layer", "logits"]),
                ("queries before the QK-norm (W_Q) times " + f"{SCALE_FACTOR:g}", scaled(p, "Wq", SCALE_FACTOR, [SCALE_LAYER]),
                 ["queries", "layer", "logits"])]
    taps_name = {"queries": "queries before the QK-norm", "attention": "attention output before the residual",
                 "norm2_input": "input of RMS norm 2 (after the residual)", "layer": f"output of layer {SCALE_LAYER}",
                 "logits": "logits"}
    rows = []
    for label, q, names in mistakes:
        got = _layer_taps(q, cfg, md.IDS)
        for nm in names:
            rows.append((label, taps_name[nm], im.rel_l2(got[nm], gold[nm]), bounds[nm]))
    return rows


def scale():
    rows = scale_data()
    body = [(m, t, _fmt(v), _fmt(b), "above" if v > b else "within") for m, t, v, b in rows]
    cfg, _, _ = im.published()
    layer, _ = im.stages(cfg, T_TEXT)
    seed = bookrng.seed_of(md.NAME_TINY)
    return (f"The published 2-layer tiny decoder of the golden-text table (weights from `{md.NAME_TINY}`, seed {seed:#018x}, text of "
            f"{T_TEXT} tokens), reference and mistaken port both in 64-bit arithmetic, so each number is the mistake alone. The "
            f"mistake is a factor {SCALE_FACTOR:g} on the matrix named, in layer {SCALE_LAYER} only; the product with a matrix is "
            "linear, so a factor on W_O scales the attention output and a factor on W_Q scales the queries, exactly. Taps: the "
            "queries before the QK-norm (the input of the norm of Volume VI, Chapter 1, 8 x 4 per head); the attention output "
            "before the residual add (read as the layer's output minus its input when W_down is zero, so that the MLP adds "
            "nothing); the input of RMS norm 2, which is the residual sum; the output of the layer; the logits. Every number is "
            "rel_l2 against the reference's tap. The bound of a tap is (the roundings of the stages of the layer up to it, by the "
            f"count of the two-mistakes table, plus 1) x 2^-24: {sum(c for _, c in layer[:2]) + 1} after the projection to queries, "
            f"{sum(c for _, c in layer[:8]) + 1} after W_O, {sum(c for _, c in layer[:9]) + 1} after the residual add, and for the "
            "layer output and the logits as in the golden-text table.\n\n"
            + _table(["mistake", "tap", "rel_l2", "bound", ""], body, "llrrl"))
