# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the capstone of Volume XVI, "From Raw Data to a Served Model": one small system that takes raw text to
answers served under continuous batching, and the reference that every stage of it is compared with.

Plain Python 3 on floats (IEEE double precision), standard library only, written from the chapter's own description and
independent of any engine code. The pieces of the earlier volumes are imported, never copied:

* the cleaning rule and the epoch order of Volume XIII, Chapter 1 (`training_data.normalise`, `training_data.epoch_order`);
* the byte-pair tokenizer with its compact vocabulary, the GPT with its forward and backward passes and the one-example rows
  of Volume V, Chapter 10 (`training_gpt`), whose block is the one of Volume V, Chapter 9 (`transformer_block`);
* the cached forward pass `extend` of Volume V, Chapter 12 (`kv_cache`), which counts the multiply-adds it runs;
* the low-rank factors of Volume XIII, Chapter 4 (`peft._lora_weight`, `peft._lora_gradients`);
* the 64-bit FNV-1a digest of Volume XIII, Chapter 7 (`checkpoints_and_recovery.fnv1a64`) and the Adam step, the generations
  (values, state, meta last, rename) and their reader of the Volume XIII project (`project_scaling_training`).

What this module writes is what makes them one path: the raw corpus and its cleaning, the artifacts that cross from training to
serving (tokenizer, base, adapter, manifest) with the checks a loader applies to them, the server with its two batching
policies and the work each one does, the planted mistakes at every seam, and the comparison of the whole system with the
simplest run that computes the same answers.

Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`); each block prints its seeds. Sums of
products are plain left-to-right additions, the order a C loop would use. Trained numbers are printed by the stable-digits rule
of `stable.py`: each run is repeated from four starting points that differ by 1e-12 or 3e-12 in one weight.

Swedish Embedded AB implements solutions for machine learning systems that are verified from the raw data to the served answer.
If your team needs expertise in building training and serving pipelines whose stages provably agree, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import json
import math
import struct
from functools import lru_cache
from multiprocessing import get_context

import bookrng
import checkpoints_and_recovery as cr
import kv_cache as kv
import peft
import project_gpt as pg
import project_scaling_training as pst
import stable
import tokenization as tok
import training_data as td
import training_gpt as tg

# ------------------------------------------------------------------------------------------------ formatting


def _table(headers, rows, align=None):
    align = align or "l" + "r" * (len(headers) - 1)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _int(n):
    return f"{n:,}"


def _seed(name):
    return f"{bookrng.seed_of(name)} (seed_of '{name}')"


def _hex(blob):
    return f"{cr.fnv1a64(blob):016x}"


def _f64(xs):
    return struct.pack(f"<{len(xs)}d", *xs)


def _text(ids, tk):
    return "`" + tk.decode(ids).replace(" ", "_") + "`" if ids else "(nothing)"


# ------------------------------------------------------------------------------------------------ the constants of the path

NAME_RAW, NAME_RAW_FT = "capstone.raw", "capstone.raw_ft"
NAME_INIT, NAME_ADAPTERS = "capstone.init", "capstone.adapters"
NAME_ORDER, NAME_ORDER_FT = "capstone.order", "capstone.order_ft"
NAME_GC, NAME_GC_B = "capstone.gradcheck", "capstone.gradcheck_b"

NAMES, COLOURS, THINGS = pg.NAMES, pg.COLORS, pg.THINGS      # the word lists of the Volume V project
N_RAW, N_RAW_FT = 200, 200              # raw documents of the base corpus and of the fine-tuning corpus
VAL_MOD = 5                             # a document is held out when its digest is 0 modulo 5
MERGES = 80                             # merges asked of the byte-pair learner (it stops when no pair is left)
T = 12                                  # positions of a row, of the position table and of a served sequence
CFG = {"D": 16, "H": 2, "FF": 24, "L": 2, "T": T}
STD = 0.2                               # the scale of every initial matrix entry
B = 8                                   # draws of one update
PRE_STEPS, PRE_LR, PRE_EVERY, PRE_KILL = 120, 0.01, 20, 107
FT_STEPS, FT_LR, FT_EVERY, FT_KILL = 60, 0.03, 10, 47
RANK, ALPHA = 2, 2.0
SCALE = ALPHA / RANK
SITES = [(0, "Wo"), (0, "Wd"), (1, "Wo"), (1, "Wd")]   # (block, matrix) of each adapter, in the order of the adapter vector
SLOTS = 3                               # sequences the server runs at once
ARRIVALS = (0, 0, 0, 1, 3, 4, 4, 7)     # iteration at which request j reaches the server
MAX_NEW = (6, 3, 6, 6, 4, 6, 3, 6)      # tokens request j asks for at most
GC_COORDS, GC_STEP = 40, 1e-5


# ------------------------------------------------------------------------------------------------ stage 1: raw text to documents

def _dirty(text, case, space):
    if case == 1:
        text = text[0].upper() + text[1:]
    elif case == 2:
        text = text.upper()
    if space == 1:
        text = text.replace(". ", ".  ", 1)
    elif space == 2:
        text = " " + text + "  "
    return text


def _raw(name, n, second):
    """n raw documents: per document integer(6) name, integer(5) colour, integer(6) thing, integer(3) case, integer(3) spacing."""
    g = bookrng.generator(name)
    out = []
    for _ in range(n):
        a, c, t = NAMES[g.integer(6)], COLOURS[g.integer(5)], THINGS[g.integer(6)]
        case, space = g.integer(3), g.integer(3)
        out.append(_dirty(f"{a} has a {c} {t}. " + second.format(t=t, c=c), case, space))
    return out


def _held_out(doc):
    return cr.fnv1a64(doc.encode()) % VAL_MOD == 0


def _clean(raw):
    """(training, validation, number of duplicates): normalise, keep the first copy of each text, split by digest."""
    seen, docs = set(), []
    for r in raw:
        d = td.normalise(r)
        if d not in seen:
            seen.add(d)
            docs.append(d)
    return [d for d in docs if not _held_out(d)], [d for d in docs if _held_out(d)], len(raw) - len(docs)


@lru_cache(maxsize=None)
def corpora():
    raw = _raw(NAME_RAW, N_RAW, "the {t} is {c}.")
    raw_ft = _raw(NAME_RAW_FT, N_RAW_FT, "a {t} is {c}.")
    return {"raw": raw, "raw_ft": raw_ft, "base": _clean(raw), "ft": _clean(raw_ft)}


def _index_split(raw):
    """The planted mistake: every fifth raw document (index mod 5 = 4) held out first, then each part cleaned and deduplicated."""
    val = {td.normalise(r) for i, r in enumerate(raw) if i % VAL_MOD == VAL_MOD - 1}
    train = {td.normalise(r) for i, r in enumerate(raw) if i % VAL_MOD != VAL_MOD - 1}
    return train, val


def raw_and_clean():
    c = corpora()
    raw = c["raw"]
    tr, va, dup = c["base"]
    ftr, fva, fdup = c["ft"]
    rows = [(i, "`" + raw[i].replace(" ", "_") + "`", "`" + td.normalise(raw[i]).replace(" ", "_") + "`",
             "held out" if _held_out(td.normalise(raw[i])) else "training") for i in range(6)]
    itr, iva = _index_split(raw)
    leak = len(iva & itr)
    listing = "\n".join(td.normalise(d) for d in tr).encode()
    return (f"The generator `{NAME_RAW}` (seed {bookrng.seed_of(NAME_RAW)}) writes {N_RAW} raw documents; for each, in this order, "
            f"integer(6) picks a name of {NAMES}, integer(5) a colour of {COLOURS}, integer(6) a thing of {THINGS}, integer(3) the case "
            f"(0 as written, 1 the first letter upper case, 2 every letter upper case) and integer(3) the spacing (0 single spaces, 1 two "
            f"spaces after the first full stop, 2 one space before the text and two after it). The text before the dirt is "
            f"'<name> has a <colour> <thing>. the <thing> is <colour>.'. Cleaning lower-cases the text, splits it at runs of white space "
            f"and joins the pieces with single spaces; a cleaned text equal to an earlier one is dropped; a cleaned text is held out when "
            f"the 64-bit FNV-1a digest of its UTF-8 bytes is 0 modulo {VAL_MOD}. A space is written `_` in the table.\n\n"
            + _table(["raw document", "as drawn", "cleaned", "split"], rows, "rlll")
            + f"\n\n| corpus | raw documents | duplicates dropped | training | held out |\n|---|---:|---:|---:|---:|\n"
            f"| base (`{NAME_RAW}`, seed {bookrng.seed_of(NAME_RAW)}) | {N_RAW} | {dup} | {len(tr)} | {len(va)} |\n"
            f"| fine-tuning (`{NAME_RAW_FT}`, seed {bookrng.seed_of(NAME_RAW_FT)}, second sentence 'a <thing> is <colour>.') | {N_RAW_FT} | "
            f"{fdup} | {len(ftr)} | {len(fva)} |\n\n"
            f"The {len(tr)} cleaned training documents of the base corpus, joined by one newline byte (10) in their order, have the "
            f"FNV-1a digest {_hex(listing)}. No held-out document is also a training document, in either corpus, since a document's "
            f"split is a function of its text. The planted mistake holds out every fifth raw document (raw index modulo 5 equal to 4) "
            f"first and cleans and deduplicates each part after: {len(iva)} distinct held-out texts, of which {leak} are also training "
            f"texts.")


# ------------------------------------------------------------------------------------------------ stage 2: the tokenizer and its artifact

@lru_cache(maxsize=None)
def tokenizer():
    return tg.Tokenizer(corpora()["base"][0], MERGES)


def tokenizer_bytes(tk):
    """u32 count of bytes, the bytes (one u8 each, ascending), u32 count of merges, each merge as two u32 (BPE ids), all little-endian."""
    used = tk.to_raw[1:1 + tk.n_bytes]
    out = struct.pack("<I", len(used)) + bytes(used) + struct.pack("<I", len(tk.merges))
    return out + b"".join(struct.pack("<II", a, b) for a, b in tk.merges)


def tokenizer_from_bytes(blob):
    """The tokenizer the server rebuilds from its artifact: the same fields as `training_gpt.Tokenizer`, from bytes alone."""
    n = struct.unpack_from("<I", blob, 0)[0]
    used = list(blob[4:4 + n])
    m = struct.unpack_from("<I", blob, 4 + n)[0]
    merges = [struct.unpack_from("<II", blob, 8 + n + 8 * k) for k in range(m)]
    if len(blob) != 8 + n + 8 * m:
        raise ValueError("tokenizer artifact has trailing or missing bytes")
    t = tg.Tokenizer.__new__(tg.Tokenizer)
    t.merges = [tuple(x) for x in merges]
    t.ranks = tok.ranks_of(t.merges)
    t.pieces_raw = tok.token_bytes(t.merges)
    raw = used + [256 + k for k in range(m)]
    t.to_compact = {r: 1 + i for i, r in enumerate(raw)}
    t.to_raw = [None] + raw
    t.vocab = 1 + len(raw)
    t.n_bytes = n
    return t


def tokenizer_stage():
    tk = tokenizer()
    c = corpora()
    tr, va, _ = c["base"]
    ftr, fva, _ = c["ft"]
    blob = tokenizer_bytes(tk)
    back = tokenizer_from_bytes(blob)
    every = tr + va + ftr + fva
    trip = sum(1 for d in every if back.decode(back.encode(d)) == d and back.encode(d) == tk.encode(d))
    lengths = {len(tk.encode(d)) + 1 for d in tr + va}
    ft_lengths = {len(tk.encode(d)) + 1 for d in ftr + fva}
    pieces = ", ".join(f"`{tk.name(i)}`" for i in range(1 + tk.n_bytes, tk.vocab))
    rows = []
    for label, d in (("base, training document 0", tr[0]), ("fine-tuning, training document 0", ftr[0])):
        ids = tk.encode(d)
        rows.append((label, f"`{d.replace(' ', '_')}`", " ".join(f"`{tk.name(i)}`" for i in ids), " ".join(map(str, ids))))
    return (f"The byte-pair learner of Volume V, Chapter 2, asked for {MERGES} merges on the {len(tr)} cleaned training documents of "
            f"the base corpus, stops after {len(tk.merges)} (no adjacent pair is left in any word). The compact vocabulary of Volume V, "
            f"Chapter 10 has id 0 for the separator `EOT`, then the {tk.n_bytes} bytes that occur in the training text in increasing "
            f"order, then the merges in the order learned: V = 1 + {tk.n_bytes} + {len(tk.merges)} = {tk.vocab}. Ids "
            f"{1 + tk.n_bytes} to {tk.vocab - 1} are the merges ({pieces}; a space is written `_`).\n\n"
            + _table(["document", "text", "tokens", "ids"], rows, "llll")
            + f"\n\nEvery base document is {', '.join(map(str, sorted(lengths)))} tokens with its separator and every fine-tuning document "
            f"{', '.join(map(str, sorted(ft_lengths)))}, so one row of {T} positions holds any of them. The artifact "
            f"`tokenizer.bin` (a u32 count of bytes, the bytes one u8 each, a u32 count of merges, each merge as two u32 BPE ids (a byte is its value 0 to 255, merge number r is 256 + r), all "
            f"little-endian) is {len(blob)} bytes with the FNV-1a digest {_hex(blob)}. The tokenizer rebuilt from those bytes alone gives "
            f"the same ids as the one that learned them, and decodes them back to the text, for {trip} of the {len(every)} cleaned "
            f"documents of both corpora.")


# ------------------------------------------------------------------------------------------------ the model as a flat vector

@lru_cache(maxsize=None)
def config():
    c = dict(CFG)
    c["V"] = tokenizer().vocab
    return c


def _template():
    return tg.init_gpt(NAME_INIT, config(), std=STD)


def _flat(p):
    return [x for t in tg.tensors(p) for r in t for x in r]


def _model(flat):
    """A fresh GPT parameter dict holding the numbers of `flat` (in the order of `training_gpt.tensors`)."""
    p = _template()
    k = 0
    for t in tg.tensors(p):
        for r in t:
            n = len(r)
            r[:] = flat[k:k + n]
            k += n
    assert k == len(flat)
    return p


def _tensor_index():
    return {name: i for i, name in enumerate(tg.tensor_names(config()))}


def _site_shapes():
    """(in, out) of each adapted matrix as the GPT stores it ([in, out], acting on row vectors)."""
    p = _template()
    return [(len(p["blocks"][l][k]), len(p["blocks"][l][k][0])) for l, k in SITES]


def _adapters_from(flat):
    """[(A [r, in], B [out, r])] per site, from the flat adapter vector (A then B, row-major, site by site)."""
    out, k = [], 0
    for n_in, n_out in _site_shapes():
        A = [flat[k + q * n_in:k + (q + 1) * n_in] for q in range(RANK)]
        k += RANK * n_in
        Bm = [flat[k + o * RANK:k + (o + 1) * RANK] for o in range(n_out)]
        k += n_out * RANK
        out.append((A, Bm))
    assert k == len(flat)
    return out


def _tr(M):
    return [list(c) for c in zip(*M)]


def _merged(base_flat, ad_flat, flip=True):
    """The served or trained network: every adapted matrix replaced by W0 + s (B A)^T (the transpose because the stored matrix is
    [in, out] and the adapter of Volume XIII, Chapter 4 acts on [out, in])."""
    p = _model(base_flat)
    if ad_flat is None:
        return p
    for (l, key), (A, Bm) in zip(SITES, _adapters_from(ad_flat)):
        p["blocks"][l][key] = _tr(peft._lora_weight(_tr(p["blocks"][l][key]), A, Bm, SCALE))
    return p


def _init_adapters(b_name=None):
    """A = normal() / sqrt(in) from `capstone.adapters`, site by site, row-major; B = 0 (or normal() / 10 from b_name)."""
    g = bookrng.generator(NAME_ADAPTERS)
    gb = bookrng.generator(b_name) if b_name else None
    flat = []
    for n_in, n_out in _site_shapes():
        flat += [g.normal() / math.sqrt(n_in) for _ in range(RANK * n_in)]
        flat += [gb.normal() / 10.0 if gb else 0.0 for _ in range(n_out * RANK)]
    return flat


def _adapter_grad(grads, ad_flat, flip=True):
    """The adapter gradient from the gradients of the whole stored matrices G ([in, out]): (dA, dB) = (s B^T G^T, s G^T A^T).
    flip=False is the planted mistake that forgets the transpose (G in place of G^T)."""
    idx = _tensor_index()
    out = []
    for (l, key), (A, Bm) in zip(SITES, _adapters_from(ad_flat)):
        G = grads[idx[f"{key}{l}"]]
        dA, dB = peft._lora_gradients(_tr(G) if flip else G, A, Bm, SCALE)
        out += [x for r in dA for x in r] + [x for r in dB for x in r]
    return out


@lru_cache(maxsize=None)
def rows():
    tk = tokenizer()
    c = corpora()
    return {"pre": tg.alone_rows(c["base"][0], tk, T), "pre_val": tg.alone_rows(c["base"][1], tk, T),
            "ft": tg.alone_rows(c["ft"][0], tk, T), "ft_val": tg.alone_rows(c["ft"][1], tk, T)}


# ------------------------------------------------------------------------------------------------ the model and its gradient check

def _central(loss, vec, j, h=GC_STEP):
    keep = vec[j]
    vec[j] = keep + h
    hi = loss(vec)
    vec[j] = keep - h
    lo = loss(vec)
    vec[j] = keep
    return (hi - lo) / (2 * h)


def _check_rows():
    r = rows()["pre"]
    return [r[td.epoch_order(bookrng.seed_of(NAME_ORDER), len(r), 0)[0]]]


@lru_cache(maxsize=None)
def _gradient_results():
    cfg = config()
    base = _flat(_template())
    rr = _check_rows()
    _, g = tg.batch_loss(_model(base), rr, cfg, grad=True)
    gflat = [x for t in g for r in t for x in r]
    sizes = [sum(len(r) for r in t) for t in tg.tensors(_template())]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    gen = bookrng.generator(NAME_GC)
    shapes = [(len(t), len(t[0])) for t in tg.tensors(_template())]
    worst_base = 0.0
    loss = lambda v: tg.batch_loss(_model(v), rr, cfg)[0]
    vec = list(base)
    for _ in range(GC_COORDS):
        k = gen.integer(len(shapes))
        i, j = gen.integer(shapes[k][0]), gen.integer(shapes[k][1])
        pos = starts[k] + i * shapes[k][1] + j
        n = _central(loss, vec, pos)
        worst_base = max(worst_base, abs(n - gflat[pos]) / max(1.0, abs(gflat[pos])))
    ad = _init_adapters(NAME_GC_B)
    _, g2 = tg.batch_loss(_merged(base, ad), rr, cfg, grad=True)
    ana = _adapter_grad(g2, ad)
    loss_ad = lambda v: tg.batch_loss(_merged(base, v), rr, cfg)[0]
    adv = list(ad)
    num = [_central(loss_ad, adv, j) for j in range(len(ad))]
    worst_ad = max(abs(a - n) / max(1.0, abs(a)) for a, n in zip(ana, num))
    # the planted mistake on the square sites (Wo): the transpose forgotten
    per_site, k = [], 0
    for (l, key), (n_in, n_out) in zip(SITES, _site_shapes()):
        size = RANK * n_in + n_out * RANK
        per_site.append((l, key, k, size))
        k += size
    wrong = []
    idx = _tensor_index()
    for (l, key, k0, size), (A, Bm) in zip(per_site, _adapters_from(ad)):
        G = g2[idx[f"{key}{l}"]]
        if len(G) != len(G[0]):
            wrong.append((l, key, None))
            continue
        dA, dB = peft._lora_gradients(G, A, Bm, SCALE)
        flat = [x for r in dA for x in r] + [x for r in dB for x in r]
        wrong.append((l, key, max(abs(a - n) / max(1.0, abs(a)) for a, n in zip(flat, num[k0:k0 + size]))))
    return worst_base, worst_ad, wrong, len(ad)


def _crit(x, bound):
    return f"passes (below {bound:g})".replace("e-0", "e-") if x < bound else "FAILS"


def model_and_check():
    cfg = config()
    p = _template()
    base = _flat(p)
    wb, wa, wrong, n_ad = _gradient_results()
    shapes = _site_shapes()
    names = tg.tensor_names(cfg)
    ts = tg.tensors(p)
    inventory = [("E", f"[{cfg['V']}, {cfg['D']}]", len(ts[0]) * len(ts[0][0]), "token embedding, also the head (logits = x E^T)"),
                 ("P", f"[{cfg['T']}, {cfg['D']}]", cfg["T"] * cfg["D"], "learned positions 0 to 11"),
                 ("W_Q, W_K, W_V, W_O", f"4 x [{cfg['D']}, {cfg['D']}]", 4 * cfg["D"] ** 2, "attention of one block, 2 heads of width 8"),
                 ("g1, g2", f"2 x [1, {cfg['D']}]", 2 * cfg["D"], "gains of the block's two RMS norms (start at 1)"),
                 ("W_gate, W_up, W_down", f"2 x [{cfg['D']}, {cfg['FF']}], [{cfg['FF']}, {cfg['D']}]", 3 * cfg["D"] * cfg["FF"], "SwiGLU network of one block"),
                 ("g_f", f"[1, {cfg['D']}]", cfg["D"], "gain of the final RMS norm (starts at 1)")]
    inv = _table(["tensor", "shape (as stored, [in, out])", "numbers", "what it is"],
                 [(a, b, _int(c) if i not in (2, 3, 4) else f"{_int(c)} per block", d) for i, (a, b, c, d) in enumerate(inventory)], "llrl")
    ad_rows = [(f"block {l}, {'W_O' if key == 'Wo' else 'W_down'}", f"[{RANK}, {n_in}]", f"[{n_out}, {RANK}]", RANK * n_in + n_out * RANK)
               for (l, key), (n_in, n_out) in zip(SITES, shapes)]
    bad = "; ".join(f"block {l} {'W_O' if key == 'Wo' else 'W_down'}: " + ("the shapes do not agree, so it cannot be formed" if w is None else
                                            f"largest deviation {w:.2g}, FAILS") for l, key, w in wrong)
    return (f"The GPT of Volume V, Chapter 10 with V = {cfg['V']}, D = {cfg['D']}, H = {cfg['H']}, F = {cfg['FF']}, L = {cfg['L']} and T = "
            f"{cfg['T']}: token embedding plus learned position, {cfg['L']} pre-norm blocks (RMS norm with epsilon 1e-6, causal attention, "
            f"SwiGLU), a final RMS norm and a head tied to E; {_int(len(base))} numbers. `init_gpt` of Volume V, Chapter 14 with every "
            f"0.02 replaced by {STD}: the generator `{NAME_INIT}` (seed {bookrng.seed_of(NAME_INIT)}) draws normal() for every entry, "
            f"row by row, in the order E, P, then for each block W_Q, W_K, W_V, W_O, W_gate, W_up, W_down, each entry times {STD} and "
            f"W_O and W_down also times 1/sqrt(2 L) = 0.5; the gains start at 1.0 and take no draw. The flat vector holds the tensors in "
            f"the order E, P, then for each block W_Q, W_K, W_V, W_O, g1, g2, W_gate, W_up, W_down, then g_f, each row by row; written as "
            f"64-bit floats it has the FNV-1a digest {_hex(_f64(base))}, and its first three numbers are "
            f"{', '.join(f'{x:.8f}' for x in base[:3])}.\n\n" + inv
            + f"\n\nThe adapters (rank r = {RANK}, alpha = {ALPHA:g}, scale s = alpha / r = {SCALE:g}) sit on W_O and W_down of both blocks. "
            f"A has shape [r, in] and B [out, r] in the orientation of Volume XIII, Chapter 4, so the stored matrix becomes "
            f"W0 + s (B A)^T. The generator `{NAME_ADAPTERS}` (seed {bookrng.seed_of(NAME_ADAPTERS)}) draws, site by site in the order of "
            f"the table, normal() / sqrt(in) for every entry of A, row by row; B starts at 0.0. The adapter vector is A then B of each site, "
            f"row-major: {n_ad} numbers.\n\n"
            + _table(["site", "A", "B", "numbers"], ad_rows, "lllr")
            + f"\n\nGradient checks, by central differences of step {GC_STEP:g} on the loss of one row (the first draw of the first epoch "
            f"of the order `{NAME_ORDER}`), the deviation being |numeric - analytic| / max(1, |analytic|), criterion below 1e-7. The base: "
            f"{GC_COORDS} coordinates of the initial network, drawn from `{NAME_GC}` (seed {bookrng.seed_of(NAME_GC)}) as integer(21) "
            f"for the tensor (in the order of the flat vector), then integer(rows) and integer(columns) of that tensor (a tensor of one row, a gain, still draws integer(1)): "
            f"{_crit(wb, 1e-7)}. The adapters: all {n_ad} adapter numbers on the initial network, with A drawn as above and B drawn, "
            f"site by site and row by row, as normal() / 10 from `{NAME_GC_B}` (seed {bookrng.seed_of(NAME_GC_B)}) (a B of zero would "
            f"make the gradient of A zero and test nothing): {_crit(wa, 1e-7)}. The planted mistake forms (dA, dB) from G instead of G^T: {bad}.")


# ------------------------------------------------------------------------------------------------ training: the two stages

def _pre_cfg():
    return {"rows": rows()["pre"], "seed": bookrng.seed_of(NAME_ORDER), "lr": PRE_LR}


def _ft_cfg(base):
    return {"rows": rows()["ft"], "seed": bookrng.seed_of(NAME_ORDER_FT), "lr": FT_LR, "base": base}


def _batch(cfg, d0):
    r = cfg["rows"]
    n = len(r)
    return [r[td.epoch_order(cfg["seed"], n, (d0 + b) // n)[(d0 + b) % n]] for b in range(B)]


def _update(cfg, st):
    """One update: B draws from the epoch order, the pooled gradient of `training_gpt.batch_loss`, then Adam (`pst._adam`)."""
    batch = _batch(cfg, st["d"])
    if "base" in cfg:
        _, g = tg.batch_loss(_merged(cfg["base"], st["p"]), batch, config(), grad=True)
        grad = _adapter_grad(g, st["p"])
    else:
        _, g = tg.batch_loss(_model(st["p"]), batch, config(), grad=True)
        grad = [x for t in g for r in t for x in r]
    st["t"] += 1
    st["d"] += B
    pst._adam(st["p"], st["m"], st["v"], grad, st["t"], lr=cfg["lr"])


def _advance(cfg, st, until, every=None, disk=None, owner=None):
    while st["t"] < until:
        _update(cfg, st)
        if every and st["t"] % every == 0:
            for op in pst._commit_ops(st, owner):
                pst._apply(disk, op)
            pst._prune(disk, owner)
    return st


def _copy(st):
    return {k: (list(v) if isinstance(v, list) else v) for k, v in st.items()}


def _nudged(nudge):
    flat = _flat(_template())
    flat[0] += nudge
    return flat


def _owner_pre():
    return _hex(tokenizer_bytes(tokenizer()))


def _answer_ids(p, prompt, max_new):
    """The reference answer: greedy, the whole context run through `training_gpt.forward` at every token, the largest logit (the
    lowest id on a tie), stopping at EOT (not kept) or after max_new tokens."""
    cfg = config()
    ctx, out = list(prompt), []
    while len(out) < max_new:
        logits, _ = tg.forward(p, ctx, cfg, tg.causal)
        nxt = kv._argmax(logits[-1])
        if nxt == tg.EOT:
            break
        out.append(nxt)
        if len(ctx) == T:
            break
        ctx.append(nxt)
    return out


def _prompt_and_expected(doc):
    first, second = doc.split(". ", 1)
    return [tg.EOT] + tokenizer().encode(first + "."), " " + second


def _right(p, docs):
    return sum(1 for d in docs if tokenizer().decode(_answer_ids(p, _prompt_and_expected(d)[0], T)) == _prompt_and_expected(d)[1])


def _pre_eval(flat):
    p = _model(flat)
    c = corpora()
    return {"val": tg.mean_loss(p, rows()["pre_val"], config()), "right": _right(p, c["base"][1])}


def _ft_eval(base, ad):
    p = _merged(base, ad)
    c = corpora()
    return {"val": tg.mean_loss(p, rows()["ft_val"], config()), "right": _right(p, c["ft"][1]),
            "right_old": _right(p, c["base"][1]), "val_old": tg.mean_loss(p, rows()["pre_val"], config())}


def _job(job):
    """One process of the parallel runs. ('runs', nudge): both stages straight (no kill), evaluated, and the server run on the
    result. ('kill_pre',): the base run killed after PRE_KILL updates and resumed. Each 'runs' job also resumes its own base run from
    the same generation with the moments set to zero.
    ('kill_ft', base, owner): the adapter run on `base` killed after FT_KILL updates and resumed."""
    kind = job[0]
    if kind == "runs":
        nudge = job[1]
        st = pst._fresh(_nudged(nudge))
        evals = {0: _pre_eval(st["p"])}
        cfg = _pre_cfg()
        for stop in range(PRE_EVERY, PRE_STEPS + 1, PRE_EVERY):
            _advance(cfg, st, stop)
            evals[stop] = _pre_eval(st["p"])
            if stop == PRE_KILL // PRE_EVERY * PRE_EVERY:
                restart = _copy(st)                                 # the generation a kill after PRE_KILL updates resumes from
        base = list(st["p"])
        restart["m"], restart["v"] = [0.0] * len(base), [0.0] * len(base)
        no_moments = _advance(cfg, restart, PRE_STEPS)
        ft = pst._fresh(_init_adapters())
        fevals = {0: _ft_eval(base, ft["p"])}
        fcfg = _ft_cfg(base)
        for stop in range(FT_EVERY, FT_STEPS + 1, FT_EVERY):
            _advance(fcfg, ft, stop)
            fevals[stop] = _ft_eval(base, ft["p"])
        served = _serve_summary(_merged(base, ft["p"]))
        out = {"nudge": nudge, "pre": evals, "base": base, "ft": fevals, "ad": list(ft["p"]), "served": served,
               "base_after": list(fcfg["base"]),
               "no_moments": {"p": no_moments["p"], "right": _pre_eval(no_moments["p"])["right"]}}
        if nudge == 0.0:
            out["kill_ft"] = _job(("kill_ft", base, _hex(_f64(base))))
        return out
    if kind == "kill_pre":
        disk, owner = {}, _owner_pre()
        st = _advance(_pre_cfg(), pst._fresh(_nudged(0.0)), PRE_KILL, PRE_EVERY, disk, owner)
        st = None                                                   # the process is gone; only the disk is left
        on_disk = pst._generations(disk)
        got, gen = pst._restore(disk, owner)
        resumed = _advance(_pre_cfg(), got, PRE_STEPS)
        return {"on_disk": on_disk, "gen": gen, "resumed": resumed}
    if kind == "kill_ft":
        base, owner = job[1], job[2]
        disk = {}
        st = _advance(_ft_cfg(base), pst._fresh(_init_adapters()), FT_KILL, FT_EVERY, disk, owner)
        st = None
        on_disk = pst._generations(disk)
        got, gen = pst._restore(disk, owner)
        resumed = _advance(_ft_cfg(base), got, FT_STEPS)
        return {"on_disk": on_disk, "gen": gen, "resumed": resumed, "disk": disk}
    raise ValueError(kind)


@lru_cache(maxsize=None)
def results():
    """(runs by nudge, the killed base run, the killed adapter run); at most four processes at a time."""
    with get_context("fork").Pool(4) as pool:
        first = pool.map(_job, [("runs", n) for n in (0.0, *stable.NUDGES)] + [("kill_pre",)], chunksize=1)
    runs = {r["nudge"]: r for r in first[:5]}
    kill_pre = first[5]
    assert _bits(kill_pre["resumed"]["p"], runs[0.0]["base"]), "the resumed base must be the straight base, bit for bit"
    return runs, kill_pre, runs[0.0]["kill_ft"]


def _stable(f, full=lambda x: f"{x:.4g}"):
    runs = results()[0]
    return stable.text(f(runs[0.0]), [f(runs[n]) for n in stable.NUDGES], full)


def _same(f):
    """An integer outcome that all five runs share (asserted), printed as it is."""
    runs = results()[0]
    vals = {f(runs[n]) for n in (0.0, *stable.NUDGES)}
    assert len(vals) == 1, vals
    return vals.pop()


def _maxdev(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _bits(a, b):
    return _f64(a) == _f64(b)


def training_stages():
    runs, kp, kf = results()
    c = corpora()
    n_val, n_fval = len(c["base"][1]), len(c["ft"][1])
    pre = [(t, _stable(lambda r, t=t: r["pre"][t]["val"]), f"{_same(lambda r, t=t: r['pre'][t]['right'])} of {n_val}")
           for t in range(0, PRE_STEPS + 1, PRE_EVERY)]
    ft = [(t, _stable(lambda r, t=t: r["ft"][t]["val"]), f"{_same(lambda r, t=t: r['ft'][t]['right'])} of {n_fval}",
           f"{_same(lambda r, t=t: r['ft'][t]['right_old'])} of {n_val}") for t in range(0, FT_STEPS + 1, FT_EVERY)]
    ref = runs[0.0]
    frozen = all(_bits(r["base"], r["base_after"]) for r in runs.values())
    t1 = _table(["updates", "held-out loss (nats per target token)", "held-out answers right"], pre, "rrr")
    t2 = _table(["updates", "held-out loss on the new form", "new-form answers right", "old-form answers right"], ft, "rrrr")
    dev = [_maxdev(r["no_moments"]["p"], r["base"]) for r in (runs[0.0], *(runs[n] for n in stable.NUDGES))]
    assert min(dev) > 1e-9, dev
    deviation = stable.text(dev[0], dev[1:], lambda x: f"{x:.2g}")
    restored = (f"The base run with a generation every {PRE_EVERY} updates (the commit of the Volume XIII project: `values.bin`, "
                f"`state.bin`, `meta.json` last, then the rename; two newest generations kept; the meta's owner field is the digest of "
                f"`tokenizer.bin`), killed after {PRE_KILL} updates, leaves {', '.join(kp['on_disk'])}; the reader accepts {kp['gen']}, "
                f"and the second process repeats the updates from there to {PRE_STEPS}: its {_int(len(ref['base']))} numbers are "
                f"{'the same 64-bit patterns as' if _bits(kp['resumed']['p'], ref['base']) else 'NOT the same bits as'} the run that was "
                f"never killed. Resumed with the moments set to zero, it ends {deviation} from it at the largest entry "
                f"(FAILS against 1e-9) and answers {_same(lambda r: r['no_moments']['right'])} of {n_val} held-out prompts right. The adapter run with a "
                f"generation every {FT_EVERY} updates (owner field: the digest of the base), killed after {FT_KILL} updates, leaves "
                f"{', '.join(kf['on_disk'])}; the reader accepts {kf['gen']}, and the resumed adapter numbers are "
                f"{'the same 64-bit patterns as' if _bits(kf['resumed']['p'], ref['ad']) else 'NOT the same bits as'} the straight run's.")
    return (f"The base run: {PRE_STEPS} updates, each of {B} draws, draw d being entry d mod n of `training_data.epoch_order` of epoch "
            f"d // n over the n = {len(rows()['pre'])} training rows, with the seed {_seed(NAME_ORDER)}. A row is one document (the "
            f"separator, then its tokens; targets shifted by one, the last target the closing separator, padding beyond with loss mask 0). "
            f"The loss of an update is the mean cross-entropy (natural log) over the supervised positions of its {B} rows together, its "
            f"gradient that of `training_gpt.batch_loss`, and the update is `adam_update` of Volume XIII, Chapter 4 at the rate {PRE_LR} "
            f"(bias corrections by repeated multiplication from 1.0, no decay, no clip, constant rate). The held-out loss is the same "
            f"pooled mean over the {len(rows()['pre_val'])} held-out rows; an answer is right when the greedy continuation of the first "
            f"sentence and its full stop (the reference rule: the whole context through the network at every token, the largest "
            f"logit, the lowest id on a tie; EOT ends the answer and is not kept, and so does a full row of {T}, whose logits still choose a last token that is kept) decodes to the held-out second "
            f"sentence. {stable.RULE} All five runs agree on every count of answers (the resume without moments included: each run is also resumed without its moments from its own generation 100).\n\n" + t1
            + f"\n\nThe adapter run on the frozen base after {PRE_STEPS} updates: {FT_STEPS} updates of {B} draws from the "
            f"{len(rows()['ft'])} fine-tuning rows (seed {_seed(NAME_ORDER_FT)}), the forward pass with every adapted matrix merged "
            f"as W0 + s (B A)^T, the adapter gradient (dA, dB) = (s B^T G^T, s G^T A^T) from the gradient G of the stored matrix, Adam "
            f"at the rate {FT_LR} on the {len(ref['ad'])} adapter numbers only. 'New form' is the held-out fine-tuning corpus ("
            f"'... a <thing> is <colour>.'), 'old form' the held-out base corpus ('... the <thing> is <colour>.').\n\n" + t2
            + f"\n\nAfter the adapter run the base's digest is {'unchanged' if frozen else 'CHANGED'} "
            f"({_hex(_f64(ref['base']))}). " + restored)


# ------------------------------------------------------------------------------------------------ the artifacts and the loader

def _manifest(tok_blob, base_blob, ad_blob, trained_on):
    body = {"adapter": {"alpha": ALPHA, "digest": _hex(ad_blob), "length": len(ad_blob), "rank": RANK,
                        "sites": [f"{k}{l}" for l, k in SITES], "trained_on": trained_on},
            "base": {"digest": _hex(base_blob), "length": len(base_blob)},
            "config": {k: config()[k] for k in ("D", "FF", "H", "L", "T", "V")},
            "tokenizer": {"digest": _hex(tok_blob), "length": len(tok_blob)}}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def package(base, ad, tk=None, trained_on=None):
    tk = tk or tokenizer()
    files = {"tokenizer.bin": tokenizer_bytes(tk), "base.bin": _f64(base), "adapter.bin": _f64(ad)}
    files["manifest.json"] = _manifest(files["tokenizer.bin"], files["base.bin"], files["adapter.bin"],
                                       trained_on or _hex(files["base.bin"]))
    return files


def load(files, check=True, apply_adapter=True):
    """(tokenizer, merged network) rebuilt from the four files alone, or the reason the loader refuses them."""
    try:
        meta = json.loads(files["manifest.json"])
    except (KeyError, ValueError):
        return "no manifest that parses"
    if check:
        for name, key in (("tokenizer.bin", "tokenizer"), ("base.bin", "base"), ("adapter.bin", "adapter")):
            blob = files.get(name)
            if blob is None or len(blob) != meta[key]["length"] or _hex(blob) != meta[key]["digest"]:
                return f"{name} is not the file the manifest names"
        if meta["adapter"]["trained_on"] != meta["base"]["digest"]:
            return "the adapter was trained on another base"
        if meta["config"]["V"] != tokenizer_from_bytes(files["tokenizer.bin"]).vocab:
            return "the tokenizer's vocabulary is not the model's"
    tk = tokenizer_from_bytes(files["tokenizer.bin"])
    unpack = lambda blob: list(struct.unpack(f"<{len(blob) // 8}d", blob))
    base = unpack(files["base.bin"])
    p = _merged(base, unpack(files["adapter.bin"]) if apply_adapter else None)
    return tk, p


# ------------------------------------------------------------------------------------------------ the server

def requests(tk=None):
    """The 8 requests: the first sentence of each of the first 8 held-out fine-tuning documents, as text."""
    docs = corpora()["ft"][1][:len(ARRIVALS)]
    return [(j, ARRIVALS[j], d.split(". ", 1)[0] + ".", MAX_NEW[j], " " + d.split(". ", 1)[1]) for j, d in enumerate(docs)]


def serve(p, tk, reqs, slots=SLOTS, policy="continuous", bug=None):
    """Run the requests to completion. Each iteration: (continuous) admit queued requests into free slots, lowest slot first, each
    prefilled in this iteration; (static) admit only when every slot is free. Then every sequence admitted earlier decodes one
    token. A sequence ends at EOT (not kept) or at its max_new tokens, and its slot is free from the next iteration.
    bug='batch position': every decode row of an iteration is given the position of the decode row in the lowest-numbered slot."""
    cfg = config()
    queue, slot, done, timeline = [], [None] * slots, {}, []
    flops = weight_bytes = kv_bytes = iterations = rows_run = 0
    n_weights = len(_flat(p))
    k = 0
    while len(done) < len(reqs):
        queue += [r for r in reqs if r[1] == k]
        meter = kv.Meter()
        ran = []
        if policy == "continuous" or all(s is None for s in slot):
            for s in range(slots):
                if slot[s] is None and queue:
                    j, arr, text, mx, _ = queue.pop(0)
                    seq = {"id": j, "arr": arr, "adm": k, "cache": kv.Cache(cfg["L"]), "out": [], "mx": mx, "end": False}
                    ids = [tg.EOT] + tk.encode(text)
                    logits = kv.extend(p, cfg, seq["cache"], ids, meter)
                    rows_run += len(ids)
                    nxt = kv._argmax(logits)
                    seq["end"] = nxt == tg.EOT
                    if not seq["end"]:
                        seq["out"].append(nxt)
                    slot[s] = seq
                    ran.append((s, j, "p"))
        decode = [s for s in range(slots) if slot[s] is not None and not slot[s]["end"] and slot[s]["adm"] < k
                  and len(slot[s]["out"]) < slot[s]["mx"] and slot[s]["cache"].length() < T]
        lead = slot[decode[0]]["cache"].length() if decode else None
        for s in decode:
            seq = slot[s]
            c = seq["cache"].length()
            pos = [lead] if bug == "batch position" else None
            kv_bytes += 2 * cfg["L"] * (c + 1) * cfg["D"] * 8
            logits = kv.extend(p, cfg, seq["cache"], [seq["out"][-1]], meter, positions=pos)
            rows_run += 1
            nxt = kv._argmax(logits)
            if nxt == tg.EOT:
                seq["end"] = True
            else:
                seq["out"].append(nxt)
            ran.append((s, seq["id"], "d"))
        if ran:
            iterations += 1
            weight_bytes += 8 * n_weights
        flops += meter.flops
        kinds = {s: kind for s, _, kind in ran}
        timeline.append([(slot[s]["id"], kinds.get(s, "held")) if slot[s] else None for s in range(slots)])
        for s in range(slots):
            seq = slot[s]
            if seq is not None and (seq["end"] or len(seq["out"]) >= seq["mx"] or seq["cache"].length() >= T) and seq["id"] not in done:
                done[seq["id"]] = {"tokens": seq["out"], "adm": seq["adm"], "fin": k, "arr": seq["arr"]}
            if policy == "continuous" and seq is not None and seq["id"] in done:
                slot[s] = None
        if policy == "static" and all(s is None or s["id"] in done for s in slot):
            slot = [None] * slots
        k += 1
    generated = sum(len(d["tokens"]) for d in done.values())
    return {"done": done, "timeline": timeline, "flops": flops, "weight_bytes": weight_bytes, "kv_bytes": kv_bytes,
            "iterations": iterations, "rows": rows_run, "generated": generated, "clock": k}


def _offline(p, tk, reqs):
    return {j: _answer_ids(p, [tg.EOT] + tk.encode(text), mx) for j, _, text, mx, _ in reqs}


def _serve_summary(p):
    tk = tokenizer()
    reqs = requests()
    ref = _offline(p, tk, reqs)
    out = {}
    for label, kw in (("continuous", {}), ("static", {"policy": "static"}), ("bug", {"bug": "batch position"})):
        r = serve(p, tk, reqs, **kw)
        out[label] = sum(1 for j in ref if r["done"][j]["tokens"] == ref[j])
    return out


@lru_cache(maxsize=None)
def _models():
    """The served network of each of the five runs (the un-nudged one first)."""
    runs = results()[0]
    return tuple(_merged(runs[n]["base"], runs[n]["ad"]) for n in (0.0, *stable.NUDGES))


def _over_runs(f):
    """f(model) for the five runs; the outcome is printed only if all five agree (asserted)."""
    vals = [f(p) for p in _models()]
    assert all(v == vals[0] for v in vals), vals
    return vals[0]


@lru_cache(maxsize=None)
def served_reference():
    runs = results()[0]
    ref = runs[0.0]
    return _merged(ref["base"], ref["ad"])


def _opening_rows(p):
    tk = tokenizer()
    reqs = requests()
    ref = _offline(p, tk, reqs)
    good = serve(p, tk, reqs)
    bad = serve(p, tk, reqs, bug="batch position")
    rows_ = [(j, arr, mx, _text(ref[j], tk), "same" if good["done"][j]["tokens"] == ref[j] else _text(good["done"][j]["tokens"], tk),
              "same" if bad["done"][j]["tokens"] == ref[j] else _text(bad["done"][j]["tokens"], tk))
             for j, arr, _, mx, _ in reqs]
    agree_good = sum(1 for j in ref if good["done"][j]["tokens"] == ref[j])
    agree_bad = sum(1 for j in ref if bad["done"][j]["tokens"] == ref[j])
    alone = sum(1 for j, a, t, m, e in reqs if serve(p, tk, [(j, 0, t, m, e)], bug="batch position")["done"][j]["tokens"] == ref[j])
    together = serve(p, tk, [(j, 0, t, m, e) for j, a, t, m, e in reqs], slots=len(reqs), bug="batch position")
    agree_together = sum(1 for j in ref if together["done"][j]["tokens"] == ref[j])
    return rows_, agree_good, agree_bad, alone, agree_together


def opening():
    rows_, agree_good, agree_bad, alone, agree_together = _over_runs(_opening_rows)
    n = len(rows_)
    return (f"The adapted model after both stages of training, eight requests (the first sentences of the first eight held-out "
            f"fine-tuning documents), {SLOTS} slots, greedy decoding. 'Offline' is the reference answer, the whole context through the "
            f"network at every token; the two servers run the cached forward pass with continuous batching, the first giving each new "
            f"token the position equal to its own cache length and the second giving every decode row of an iteration (the step of one already running sequence) the position of "
            f"the row in the lowest-numbered slot. A space is written `_`. The five runs of the stable-digits rule (a trained number is printed only to the digits that survive a nudge of the first weight, explained below) give the same tokens "
            f"in every cell.\n\n"
            + _table(["request", "arrives at iteration", "at most tokens", "offline answer", "correct server", "one position per batch"],
                     rows_, "rrrlll")
            + f"\n\nAgreement with the offline answers: {agree_good} of {n} for the correct server, {agree_bad} of {n} "
            f"for the other. The same faulty server agrees on {alone} of {n} when each request is sent alone, and on "
            f"{agree_together} of {n} when all eight arrive at iteration 0 and it has {n} slots.")


def _schedule_rows(p):
    tk = tokenizer()
    reqs = requests()
    ref = _offline(p, tk, reqs)
    out = []
    per = {}
    for policy in ("continuous", "static"):
        r = serve(p, tk, reqs, policy=policy)
        per[policy] = r
        d = r["done"]
        same = sum(1 for j in ref if d[j]["tokens"] == ref[j])
        wait = sum(d[j]["adm"] - d[j]["arr"] for j in d) / len(d)
        span = sum(d[j]["fin"] - d[j]["arr"] + 1 for j in d) / len(d)
        out.append((policy, r["iterations"], r["rows"], r["generated"], _int(r["flops"]), _int(r["weight_bytes"]),
                    f"{r['weight_bytes'] / r['generated']:.1f}", _int(r["kv_bytes"]), f"{wait:.3f}", f"{span:.3f}", f"{same} of {len(reqs)}"))
    c = per["continuous"]["done"]
    s = per["static"]["done"]
    req_rows = [(j, arr, mx, c[j]["adm"], c[j]["fin"], len(c[j]["tokens"]), s[j]["adm"], s[j]["fin"]) for j, arr, _, mx, _ in reqs]
    return out, req_rows


def schedules():
    out, req_rows = _over_runs(_schedule_rows)
    n_w = len(_flat(served_reference()))
    return ("The eight requests of the opening on the same server under both policies. Continuous: in each iteration the free slots "
            "are filled from the queue (first come, first served; the lowest free slot first) and each admitted request is prefilled "
            "in that iteration, which gives its first token; every sequence admitted earlier then decodes one token. Static: the "
            f"server admits only when all {SLOTS} slots are free, and a slot whose sequence has ended stays empty until the whole "
            "batch has ended. A sequence ends at EOT (not counted as a token) or at its limit; its slot is free from the next iteration. "
            "Iterations count from 0. The five runs of the stable-digits rule give the same numbers in both tables.\n\n"
            + _table(["request", "arrives", "at most tokens", "continuous: admitted", "continuous: last token", "tokens",
                      "static: admitted", "static: last token"], req_rows, "rrrrrrrr")
            + "\n\nWork, counted by these rules: an iteration is one in which at least one row runs; a row is one token through the "
            "network (a prefill of 7 tokens is 7 rows); FLOPs are 2 for each multiply-add of the cached forward pass `extend` of "
            "Volume V, Chapter 12 (every matrix product, every attention score and every weighted sum of values); every iteration reads "
            f"all {_int(n_w)} served weights once, 8 bytes each; a decode row at cache length c reads the c + 1 key rows and value "
            "rows of every layer (the new token's own included), 8 bytes a number. A request's wait is its admission iteration minus "
            "its arrival iteration, and its span is the iteration of its last token minus its arrival iteration, plus 1; the means "
            "are over the 8 requests, exact multiples of 1/8 printed to 3 decimals.\n\n"
            + _table(["policy", "iterations", "rows", "tokens", "FLOPs", "weight bytes read", "weight bytes per token",
                      "cache bytes read", "mean wait", "mean span", "same as offline"], out, "lrrrrrrrrrr"))


def schedule_grids():
    """For the figure: (policy, timeline of slot occupancy, done) for both policies."""
    p = served_reference()
    tk = tokenizer()
    reqs = requests()
    return [(policy, serve(p, tk, reqs, policy=policy)) for policy in ("continuous", "static")]


# ------------------------------------------------------------------------------------------------ artifacts, the loader and the whole path

def _wrap_at_commas(text, width=72):
    """Lines of at most `width` characters where possible, each cut right after a comma."""
    lines, cur = [], ""
    for piece in text.split(","):
        piece = piece + ","
        if cur and len(cur) + len(piece) > width:
            lines.append(cur)
            cur = ""
        cur += piece
    lines.append(cur[:-1])
    return "\n".join(lines)


def artifacts():
    runs, kp, kf = results()
    ref = runs[0.0]
    files = package(kp["resumed"]["p"], kf["resumed"]["p"])
    rows_ = [(name, _int(len(blob)), _hex(blob)) for name, blob in files.items()]
    tk = tokenizer()
    c = corpora()
    every = tokenizer_from_bytes(tokenizer_bytes(tg.Tokenizer(c["base"][0] + c["base"][1], MERGES)))
    cases = []
    cases.append(("the four files as written", load(files)))
    swapped = dict(files)
    swapped["tokenizer.bin"] = tokenizer_bytes(every)
    cases.append(("tokenizer.bin rebuilt from the training and held-out documents of the base corpus", load(swapped)))
    other = package(runs[0.0]["no_moments"]["p"], kf["resumed"]["p"], trained_on=_hex(_f64(kp["resumed"]["p"])))
    cases.append(("base.bin from the run resumed without its moments, manifest rewritten for it", load(other)))
    flipped = dict(files)
    blob = bytearray(files["adapter.bin"])
    blob[100] ^= 1
    flipped["adapter.bin"] = bytes(blob)
    cases.append(("one bit of adapter.bin flipped (byte 100, bit 0)", load(flipped)))
    lines = [(label, "accepted" if isinstance(res, tuple) else f"refused: {res}") for label, res in cases]
    meta = _wrap_at_commas(files["manifest.json"].decode())
    return ("The four files that cross from training to serving, written from the resumed runs of both stages: `tokenizer.bin` "
            "(stage 2), `base.bin` (the 64-bit floats of the trained base in the order of the flat vector, little-endian), "
            "`adapter.bin` (the adapter vector, likewise) and `manifest.json`, whose text, keys sorted and no spaces, is one line; it "
            "is printed here cut after commas, and the file is these lines joined with nothing between them (no line break):\n\n"
            f"```text\n{meta}\n```\n\n"
            + _table(["file", "bytes", "FNV-1a digest"], rows_, "lrl")
            + "\n\nThe digest of `tokenizer.bin` depends on no floating-point operation, and the lengths depend on none of the trained "
              "numbers. Those of `base.bin`, `adapter.bin` and `manifest.json` depend on the last bit of every trained number, which "
              "depends on the order of the sums of the backward pass and on how the library functions round, and neither is fixed by "
              "this book: the digests printed for them are those of the module's run, and an implementation checks its own files "
              "against its own manifest, built from its own three digests in the form above."
            + "\n\nThe loader reads the manifest, then accepts the files only if each of the three has the recorded length and digest, the "
              "adapter's `trained_on` equals the base's digest and the tokenizer's vocabulary equals the configuration's V; it then "
              "rebuilds the tokenizer from its bytes and the network by merging the adapter into the base.\n\n"
            + _table(["what the server is given", "loader"], lines, "ll"))


def _path_counts(base, ad, reqs, every, no_moments):
    """The counts of the whole-path table that depend only on the trained numbers: for each build, the answers equal to the
    reference's offline answers and the answers that are the held-out second sentence (requests allowed 6 tokens)."""
    tk = tokenizer()
    offline = _offline(_merged(base, ad), tk, reqs)
    files = package(base, ad)

    def counts(p, tkz, bug=None):
        r = serve(p, tkz, reqs, bug=bug)
        same = sum(1 for j in offline if r["done"][j]["tokens"] == offline[j])
        right = sum(1 for j, _, _, mx, exp in reqs if mx == 6 and tkz.decode(r["done"][j]["tokens"]) == exp)
        return same, right

    tk_s, p_s = load(files)
    swapped = dict(files)
    swapped["tokenizer.bin"] = tokenizer_bytes(every)
    tk_x, p_x = load(swapped, check=False)
    tk_n, p_n = load(files, apply_adapter=False)
    tk_o, p_o = load(package(no_moments, ad, trained_on=_hex(_f64(base))), check=False)
    return {"no moments": counts(p_o, tk_o), "system": counts(p_s, tk_s), "tokenizer": counts(p_x, tk_x),
            "unmerged": counts(p_n, tk_n), "eight-bit": counts(quantized(p_s), tk_s),
            "position": counts(p_s, tk_s, "batch position")}


def whole_path():
    runs, kp, kf = results()
    ref = runs[0.0]
    reqs = requests()
    c = corpora()
    every = tokenizer_from_bytes(tokenizer_bytes(tg.Tokenizer(c["base"][0] + c["base"][1], MERGES)))
    per_run = [_path_counts(runs[n]["base"], runs[n]["ad"], reqs, every, runs[n]["no_moments"]["p"])
               for n in (0.0, *stable.NUDGES)]
    assert all(x == per_run[0] for x in per_run), per_run
    k = per_run[0]
    resumed = (kp["resumed"]["p"], kf["resumed"]["p"])
    same_numbers = _bits(resumed[0], ref["base"]) and _bits(resumed[1], ref["ad"])
    sys_counts = _path_counts(resumed[0], resumed[1], reqs, every, ref["no_moments"]["p"])["system"]
    n, n_full = len(reqs), sum(1 for m in MAX_NEW if m == 6)
    pair = lambda xy: (f"{xy[0]} of {n}", f"{xy[1]} of {n_full}")
    rows_ = [("the reference: straight runs, the answers computed offline one request at a time", "-", "-", f"{n} of {n}",
              f"{k['system'][1]} of {n_full}"),
             ("the system: both stages killed and resumed, four files, the loader, continuous batching",
              "same bits" if same_numbers else "DIFFER", "accepted", *pair(sys_counts)),
             ("tokenizer rebuilt by the server from training and held-out documents", "same bits", "refused (tokenizer.bin)",
              *pair(k["tokenizer"])),
             ("base resumed without its moments, served with the adapter of the right base", "FAILS (base)",
              "refused (trained on another base)", *pair(k["no moments"])),
             ("the adapter not merged by the server", "same bits", "accepted", *pair(k["unmerged"])),
             ("the system serving 8-bit weights (a tolerance, not a mistake)", "same bits", "accepted", *pair(k["eight-bit"])),
             ("one position for every decode row of an iteration", "same bits", "accepted", f"{k['position'][0]} of {n}", "-")]
    return ("The reference trains both stages straight in one process and computes each answer offline; the system kills each "
            f"stage once (after {PRE_KILL} and {FT_KILL} updates), resumes from the newest generation that verifies, writes the four "
            "files, and answers from what its loader rebuilt from them, under continuous batching with the arrivals of the opening. "
            "Each planted mistake changes one seam of the system. 'Trained numbers' compares the base and adapter numbers the server "
            "is given with the reference's (the bits, or the criterion 1e-9 at the largest entry); 'loader' is what the loader with "
            f"its checks says; the third row's tokenizer is learned, with the same {MERGES} merges, on the training and held-out documents of the base corpus ({len(c['base'][0]) + len(c['base'][1])} of them) and its ids are compared with the reference's ids; the last two columns are what the server answers when the files are loaded without the checks: answers "
            f"equal to the reference's offline answers, and answers that are the held-out second sentence, among the {n_full} requests "
            "allowed 6 tokens. The five runs of the stable-digits rule give the same counts in every row (the resume without moments is "
            "taken from each run's own generation 100).\n\n"
            + _table(["build", "trained numbers", "loader", "same as the reference", "right answers"], rows_, "lllrr"))


# ------------------------------------------------------------------------------------------------ the work of the whole path

def _row_macs(cfg):
    """Multiply-adds of `training_gpt.forward` on one row of T positions, the products as the code runs them: the four projections,
    all T x T scores and weighted sums of every head (the masked ones included), the three SwiGLU products and the head."""
    T_, D, F, V, L, H = cfg["T"], cfg["D"], cfg["FF"], cfg["V"], cfg["L"], cfg["H"]
    dk = D // H
    return L * (4 * T_ * D * D + 2 * H * T_ * T_ * dk + 3 * T_ * D * F) + T_ * D * V


def work_bill():
    cfg = config()
    runs, kp, kf = results()
    ref = runs[0.0]
    macs = _row_macs(cfg)
    step = 3 * 2 * macs * B
    sched = serve(served_reference(), tokenizer(), requests())
    n_base, n_ad = len(ref["base"]), len(ref["ad"])
    pre_bytes = 8 * (3 * n_base) + 16
    ft_bytes = 8 * (3 * n_ad) + 16
    rows_ = [("base run", PRE_STEPS, _int(step), _int(PRE_STEPS * step), f"{PRE_STEPS // PRE_EVERY} generations of {_int(pre_bytes)} bytes of values and state"),
             ("adapter run", FT_STEPS, _int(step), _int(FT_STEPS * step), f"{FT_STEPS // FT_EVERY} generations of {_int(ft_bytes)} bytes of values and state"),
             ("serving the 8 requests", sched["iterations"], "-", _int(sched["flops"]), f"{_int(sched['weight_bytes'])} bytes of weights read")]
    return (f"FLOPs are 2 for every multiply-add of a matrix product, the products as the code runs them. A forward pass of one row is "
            f"{_int(macs)} multiply-adds (for each block the 4 projections, T x D x D each, the scores and weighted sums of the 2 heads "
            f"over all T x T pairs, masked ones included, and the 3 SwiGLU products; then the head, T x D x V), and the backward pass is "
            f"counted as twice the forward; norms, softmax and the optimizer are not counted. An update of {B} rows is then "
            f"3 x 2 x {_int(macs)} x {B} FLOPs, in both stages: the adapter run here forms the gradient of every stored matrix and keeps "
            f"the part that reaches the adapters. A generation's values and state are 8 bytes for every trained number, its first and "
            f"its second moment, and 16 bytes of counts.\n\n"
            + _table(["stage", "updates or iterations", "FLOPs per update", "FLOPs", "bytes moved"], rows_, "lrrrl"))


# ------------------------------------------------------------------------------------------------ serving in 8 bits

QUANT_KEYS = ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd")


def _quantize_matrix(M):
    """Each row of a stored matrix to 8-bit codes: scale = max |entry| / 127, code = the quotient rounded half away from zero (and
    within -127 .. 127 by construction), read back as code * scale. A row of zeros keeps the scale 0 and the codes 0."""
    out = []
    for row in M:
        big = max(abs(x) for x in row)
        scale = big / 127.0
        out.append([peft._round_half_away(x / scale) * scale if scale > 0.0 else 0.0 for x in row])
    return out


def quantized(p):
    """The served network with E, P and every matrix of every block in 8 bits per entry (one 64-bit scale per row); gains stay."""
    q = _model(_flat(p))
    q["E"] = _quantize_matrix(p["E"])
    q["P"] = _quantize_matrix(p["P"])
    for b_q, b_p in zip(q["blocks"], p["blocks"]):
        for key in QUANT_KEYS:
            b_q[key] = _quantize_matrix(b_p[key])
    return q


def _served_bytes(p, eight_bit):
    """Bytes of the served weights: 8 per number in 64 bits; in 8 bits one byte per entry of a quantized matrix plus 8 per row for
    its scale, and 8 per gain."""
    if not eight_bit:
        return 8 * len(_flat(p))
    mats = [p["E"], p["P"]] + [b[k] for b in p["blocks"] for k in QUANT_KEYS]
    gains = [p["gf"]] + [b[k] for b in p["blocks"] for k in ("g1", "g2")]
    return sum(len(m) * len(m[0]) + 8 * len(m) for m in mats) + sum(8 * len(g[0]) for g in gains)


def _margins(p):
    q = quantized(p)
    tk = tokenizer()
    reqs = requests()
    ref = _offline(p, tk, reqs)
    served = serve(q, tk, reqs)
    agree = sum(1 for j in ref if served["done"][j]["tokens"] == ref[j])
    cfg = config()
    margin, change, steps, held, kept = math.inf, 0.0, 0, 0, 0
    for j, _, text, mx, _ in reqs:
        ctx = [tg.EOT] + tk.encode(text)
        for t in ref[j] + ([tg.EOT] if len(ref[j]) < mx else []):
            zp = tg.forward(p, ctx, cfg, tg.causal)[0][-1]
            zq = tg.forward(q, ctx, cfg, tg.causal)[0][-1]
            top = sorted(zp, reverse=True)
            step_margin = top[0] - top[1]
            step_change = max(abs(a - b) for a, b in zip(zp, zq))
            margin, change, steps = min(margin, step_margin), max(change, step_change), steps + 1
            held += 1 if step_change < step_margin / 2 else 0
            kept += 1 if kv._argmax(zq) == kv._argmax(zp) else 0
            if t == tg.EOT or len(ctx) == T:
                break
            ctx = ctx + [t]
    return {"agree": agree, "margin": margin, "change": change, "steps": steps, "held": held, "kept": kept,
            "n": len(ref), "b64": _served_bytes(p, False), "b8": _served_bytes(p, True)}


def quantized_serving():
    per = [_margins(p) for p in _models()]
    for key in ("agree", "steps", "held", "kept", "n", "b64", "b8"):
        assert all(x[key] == per[0][key] for x in per), key
    z = per[0]
    agree, steps, held, kept, b64, b8 = z["agree"], z["steps"], z["held"], z["kept"], z["b64"], z["b8"]
    fmt = lambda x: f"{x:.3g}"
    margin = stable.text(z["margin"], [x["margin"] for x in per[1:]], fmt)
    change = stable.text(z["change"], [x["change"] for x in per[1:]], fmt)
    whole = "holds" if all(x["change"] < x["margin"] / 2 for x in per) else ("does not hold" if not any(
        x["change"] < x["margin"] / 2 for x in per) else "is unstable")
    rows_ = [("64-bit weights", _int(b64), f"{z['n']} of {z['n']}", "-"),
             ("8-bit codes, one scale per row", _int(b8), f"{agree} of {z['n']}", change)]
    return ("The served network of the opening, and the same network with E, P and every matrix of both blocks stored as 8-bit codes: "
            "for each row of a stored matrix the scale is its largest magnitude divided by 127, the code is the entry divided by "
            "the scale and rounded half away from zero, and the network computes with code times scale; the gains stay in 64 bits. "
            f"Both serve the eight requests of the opening under continuous batching. The margin is the largest logit minus the second "
            f"largest of the 64-bit network at each of the {steps} steps of the offline answers (the steps whose logits choose a "
            "token, the final EOT included); the change is the largest absolute difference between the two networks' logits at the "
            "same steps.\n\n"
            + _table(["served weights", "bytes", "same as offline", "largest logit change"], rows_, "lrrr")
            + f"\n\nThe smallest margin over those steps is {margin} and the largest change {change}, so the bound taken over "
            f"all steps at once (the largest change below half of the smallest margin) {whole}. Step by "
            f"step, the change is below half of that step's own margin at {held} of the {steps} steps, and the largest logit of the 8-bit "
            f"network is the reference's token at {kept} of the {steps} steps. The margin and the change are trained numbers, printed by "
            f"the stable-digits rule; the counts are the same in all five runs.")
