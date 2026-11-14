# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Data Pipelines for Large Models".

Plain Python 3, standard library only, written from the chapter's own description. Nothing here is trained: every number is a
count or an exact ratio, so every digit is stable and another implementation reproduces it.

* The record format: a conversation becomes three parallel arrays, the token ids, a supervision mask and the start offset of each
  record, under a byte tokenizer with two special tokens and the ChatML framing of Volume V.
* Three row layouts of the same validation records (a packed stream, one record per row at a fixed length, one record per row at
  the longest record's length), counted for padding, for how many other records a supervised position can read, and for how often
  a record's exact answer stands earlier in its own row.
* Batches of variable-length records, padded to the longest of the batch, with and without sorting inside a pool of batches.
* The mixture of sources: how many times each group is listed so that a share of the draws is kept, and what a share of records is
  as a share of tokens.
* Duplicates: exact hashes, the Jaccard similarity of word shingles, MinHash signatures with banding, and the decontamination of
  a validation set.
* The order of examples: a shuffle per epoch that is a function of the run seed and the epoch alone, what a resumed run needs, and
  the shard each worker keeps.

Each public function without a leading underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: training_data.<name> -->` markers. Underscore functions and the constants are shared with the chapter's figures.

Swedish Embedded AB implements training data pipelines that are counted before they are trusted: records, masks, rows, mixtures and
duplicates. If your team needs expertise in preparing data for fine-tuning and pre-training, you can procure our services by sending
an email to info@swedishembedded.com.
"""
import hashlib
import math
from functools import lru_cache

import bookrng
import training_gpt as tg

M64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15

# ------------------------------------------------------------------------------------------------ formatting


def _int(n):
    return f"{n:,}"


def _fx(v, d=4):
    return f"{v:.{d}f}"


def _pct(v, d=1):
    return f"{100.0 * v:.{d}f}"


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed(name):
    return f"{bookrng.seed_of(name):#018x}"


# ------------------------------------------------------------------------------------------------ the record format

IM_START, IM_END, NEWLINE = 256, 257, 10
SPECIAL = {IM_START: "[start]", IM_END: "[end]"}


def _piece(i):
    """The printed form of one token id: id 256 as `[start]` (the text `<|im_start|>`), id 257 as `[end]` (`<|im_end|>`), a byte by its
    character (a space as `_`, a line feed as `\\n`)."""
    if i in SPECIAL:
        return SPECIAL[i]
    if i == NEWLINE:
        return "\\n"
    return "_" if i == 32 else chr(i)


def encode_record(messages, convention):
    """`messages` is a list of (role, content, trained). Each message is IM_START, the role's bytes, a line feed (the header), the
    content's bytes, IM_END and a line feed (the tail). Returns (ids, mask, answer_spans): `mask[i]` is True when token i is
    supervised. Convention 'whole' supervises every token of a trained message, header included; 'asked' supervises the content and
    the tail only (the header is part of the prompt the model is asked from). `answer_spans` lists (start, end) of the content of
    each trained message."""
    ids, mask, spans = [], [], []
    for role, content, trained in messages:
        header = [IM_START] + list(role.encode("utf-8")) + [NEWLINE]
        body = list(content.encode("utf-8"))
        tail = [IM_END, NEWLINE]
        flags = [trained and convention == "whole"] * len(header) + [trained] * (len(body) + len(tail))
        if trained:
            spans.append((len(ids) + len(header), len(ids) + len(header) + len(body)))
        ids += header + body + tail
        mask += flags
    return ids, mask, spans


def _runs(xs):
    """Consecutive integers as ranges: [3, 4, 5, 9] -> '3 to 5 and 9'."""
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]} to {xs[j]}")
        i = j + 1
    return ", ".join(out[:-1]) + (" and " if len(out) > 1 else "") + out[-1]


def record_format():
    one = [("user", "Hello", False), ("assistant", "Hi", True)]
    other = [("user", "2+2?", False), ("assistant", "4", True)]
    lines = ["Token 256 is the text `<|im_start|>` and is written `[start]` in the table; token 257 is `<|im_end|>`, written `[end]`; "
             "every other id is a byte, written as its character, with `\\n` for the line feed (10). The record is `user: Hello`, "
             "`assistant: Hi`, the second message trained; the second record is `user: 2+2?`, `assistant: 4`.", ""]
    ids_w, mask_w, _ = encode_record(one, "whole")
    _, mask_a, _ = encode_record(one, "asked")
    rows = [(i, f"`{_piece(t)}`", t, int(mask_w[i]), int(mask_a[i])) for i, t in enumerate(ids_w)]
    lines.append(_table(["position i", "token", "id", "supervised, whole message", "supervised, as asked"], rows))
    lines.append("")
    arrays = []
    starts = 0
    offsets = []
    stream = {"whole": [], "asked": []}
    ids_all = []
    for rec in (one, other):
        offsets.append(len(ids_all))
        ids, _, _ = encode_record(rec, "whole")
        ids_all += ids
        for c in ("whole", "asked"):
            stream[c] += encode_record(rec, c)[1]
    for k, rec in enumerate((one, other)):
        n = len(encode_record(rec, "whole")[0])
        arrays.append((k, n, offsets[k], sum(encode_record(rec, "whole")[1]), sum(encode_record(rec, "asked")[1])))
    lines.append(_table(["record", "tokens", "start offset", "supervised, whole message", "supervised, as asked"], arrays))
    lines.append("")
    asked = stream["asked"]
    targets = [t for t in range(len(ids_all) - 1) if asked[t + 1]]
    lines.append(f"The dataset of the two records has {len(ids_all)} token ids, a mask of {len(asked)} flags and the starts {offsets}. "
                 f"Under the as-asked mask the positions t whose target is supervised (the mask read at t + 1) are {_runs(targets)}: "
                 f"{len(targets)} positions; the mask at position 0 of the stream, which is never a target, is {int(asked[0])}.")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ three layouts

NAMES = ["Ada", "Ben", "Cleo", "Dan", "Eva", "Finn", "Gus", "Hana", "Ivo", "Jon", "Kai", "Lea"]
CITIES = ["Oslo", "Rome", "Lima", "Kyiv", "Bern", "Riga", "Baku", "Doha", "Faro", "Graz", "Hue", "Nice"]
QUESTIONS = ["Where does {n} live?", "Which city is home to {n}?", "{n} lives in which city?"]
FACT_TRAIN = "training_data.facts.train"
FACT_VAL = "training_data.facts.val"
ROW = 512


def _facts(name, count):
    g = bookrng.generator(name)
    out = []
    for _ in range(count):
        f = g.integer(len(NAMES))
        q = g.integer(len(QUESTIONS))
        out.append((f, q))
    return out


def _record_messages(f, q):
    return [("user", QUESTIONS[q].format(n=NAMES[f]), False), ("assistant", f"{NAMES[f]} lives in {CITIES[f]}.", True)]


@lru_cache(maxsize=None)
def val_records():
    recs = []
    for f, q in _facts(FACT_VAL, 28):
        ids, mask, spans = encode_record(_record_messages(f, q), "asked")
        recs.append({"fact": f, "ids": ids, "mask": mask, "ans": spans[0]})
    return recs


def _layout_rows(recs, layout):
    """Rows of real input tokens for one layout: a list of dicts with `ids` (the row's inputs), `doc` (the record of each input),
    `sup` (1 where the target of that position is supervised, as asked) and `ans` (for each record whose answer lies inside the
    row, its (start, end) in row coordinates)."""
    if layout == "packed":
        ids, doc, mask, ans_of = [], [], [], {}
        for k, r in enumerate(recs):
            ans_of[k] = (len(ids) + r["ans"][0], len(ids) + r["ans"][1])
            ids += r["ids"]
            doc += [k] * len(r["ids"])
            mask += r["mask"]
        rows = []
        for r_i, row in enumerate(tg.pack_rows(ids, doc, ROW)):
            k = sum(row["m"])
            a = r_i * ROW
            sup = [int(mask[a + i + 1]) for i in range(k)]
            inside = {d: (s - a, e - a) for d, (s, e) in ans_of.items() if s >= a and e <= a + k}
            rows.append({"ids": row["x"][:k], "doc": row["doc"][:k], "sup": sup, "ans": inside, "pad": ROW - k})
        return rows
    block = ROW if layout == "fixed" else max(len(r["ids"]) for r in recs)
    rows = []
    for k, r in enumerate(recs):
        n = len(r["ids"])
        sup = [int(r["mask"][t + 1]) for t in range(n - 1)] + [0]
        rows.append({"ids": r["ids"], "doc": [k] * n, "sup": sup, "ans": {k: r["ans"]}, "pad": block - n})
    return rows


def _layout_numbers(layout):
    recs = val_records()
    rows = _layout_rows(recs, layout)
    block = ROW if layout in ("packed", "fixed") else max(len(r["ids"]) for r in recs)
    computed = len(rows) * block
    real = sum(len(r["ids"]) for r in rows)
    supervised = sum(sum(r["sup"]) for r in rows)
    visible, counted = 0, 0
    inside = seen = 0
    for r in rows:
        for i, s in enumerate(r["sup"]):
            if s:
                own = r["doc"][i + 1] if i + 1 < len(r["doc"]) else r["doc"][i]
                visible += len(set(r["doc"][: i + 1]) - {own})
                counted += 1
        for d, (s, e) in r["ans"].items():
            inside += 1
            ans = r["ids"][s:e]
            before = r["ids"][:s]
            if any(before[j:j + len(ans)] == ans for j in range(len(before) - len(ans) + 1)):
                seen += 1
    return {"rows": len(rows), "block": block, "computed": computed, "real": real, "supervised": supervised,
            "visible": visible / counted, "inside": inside, "seen": seen}


def layouts():
    recs = val_records()
    stream = sum(len(r["ids"]) for r in recs)
    lines = [f"28 validation records from `{FACT_VAL}` (seed {_seed(FACT_VAL)}); for each, in order, `integer(12)` picks the fact "
             f"and then `integer(3)` the wording of the question; the user message is the question and the assistant message is "
             f"`<name> lives in <city>.`, name i living in city i, the names being {', '.join(NAMES)} and the cities "
             f"{', '.join(CITIES)}, and the three wordings of the question (index 0, 1, 2) being "
             f"{'; '.join('`' + q + '`' for q in QUESTIONS)} with `{{n}}` the name. "
             f"The 28 records hold {_int(stream)} tokens (mean {stream / 28:.2f}, "
             f"longest {max(len(r['ids']) for r in recs)}), of which "
             f"{_int(sum(sum(r['mask']) for r in recs))} are supervised under the as-asked mask. "
             f"A row has {ROW} positions in the first two layouts: position i of a packed row holds the stream token it starts at "
             f"plus i, its target is the next token, and the last row is padded; in a one-record row all the record's tokens "
             f"are inputs and the last has no target. 'Positions with a real input' are the positions that are not padding. "
             f"'Mean other records readable' is, over the supervised targets, the number of records other than the target's own "
             f"that have a token at or before the position in its row. 'Answers already in the row' counts, among the records "
             f"whose answer text (the bytes of `<name> lives in <city>.`) lies wholly in the inputs of one row, those whose answer "
             f"occurs as a run of tokens in the same row before it begins.".replace("{}", "n"),
             ""]
    rows = []
    for name, layout in (("packed stream, rows of 512", "packed"), ("one record per row, rows of 512", "fixed"),
                         ("one record per row, rows of the longest record", "longest")):
        n = _layout_numbers(layout)
        rows.append((name, n["rows"], n["block"], _int(n["computed"]), _int(n["real"]), _pct(1 - n["real"] / n["computed"]),
                     _int(n["supervised"]), _fx(n["visible"], 2), f"{n['seen']} of {n['inside']}"))
    lines.append(_table(["layout", "rows", "row length", "positions computed", "positions with a real input", "padding, percent",
                         "supervised targets", "mean other records readable", "answers already in the row"], rows))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ batches of unequal length

LENGTHS = "training_data.lengths"
BATCHES = "training_data.batches"
N_RECORDS = 240
BATCH = 8
POOLS = (1, 2, 3, 5, 6, 10, 15, 30)


@lru_cache(maxsize=None)
def record_lengths():
    """240 record lengths: for each record in turn one `uniform()`; below 0.2 the record is long and its length is 96 + `integer(129)`,
    otherwise it is short and its length is 16 + `integer(33)`. The supervised tokens of a record are its length divided by 2,
    rounded down."""
    g = bookrng.generator(LENGTHS)
    out = []
    for _ in range(N_RECORDS):
        if g.uniform() < 0.2:
            out.append(96 + g.integer(129))
        else:
            out.append(16 + g.integer(33))
    return out


def bucket_batches(lengths, batch, pool, g):
    """Shuffle the record indices; cut the shuffled list into chunks of pool * batch records; sort each chunk by length ascending
    (equal lengths keep their shuffled order); cut it into batches of `batch` records; shuffle the list of batches. pool = 1 is plain
    shuffled batching, and a pool as large as the whole data is one global sort."""
    order = list(range(len(lengths)))
    g.shuffle(order)
    batches = []
    for c in range(0, len(order), pool * batch):
        chunk = sorted(order[c:c + pool * batch], key=lambda i: lengths[i])
        batches += [chunk[b:b + batch] for b in range(0, len(chunk), batch)]
    g.shuffle(batches)
    return batches


def _batch_numbers(pool):
    L = record_lengths()
    batches = bucket_batches(L, BATCH, pool, bookrng.generator(BATCHES))
    computed = sum(len(b) * max(L[i] for i in b) for b in batches)
    sup = [sum(L[i] // 2 for i in b) for b in batches]
    return batches, computed, sup


def length_batches():
    L = record_lengths()
    useful = sum(L)
    lmax = max(L)
    lines = [f"{N_RECORDS} record lengths from `{LENGTHS}` (seed {_seed(LENGTHS)}), drawn in order, each record by one `uniform()` and then one `integer`: "
             f"when the uniform is below 0.2 the record is long and its length is 96 + `integer(129)`, otherwise it is short and its "
             f"length is 16 + `integer(33)`; {sum(1 for x in L if x >= 96)} are long, the longest is {lmax}, the shortest {min(L)}, together {_int(useful)} tokens "
             f"(mean {useful / len(L):.2f}). Batches of {BATCH} records, so {N_RECORDS // BATCH} batches; each batch is computed at the "
             f"length of its longest record, and a position is one token of one row. For every pool size the generator "
             f"`{BATCHES}` (seed {_seed(BATCHES)}) starts again: first the shuffle of the {N_RECORDS} indices, then, after the sorting "
             f"and cutting, the shuffle of the list of batches. The weight of a supervised token is 1 divided by the supervised tokens "
             f"of its batch, since the loss of a batch is a per-token mean; a record of length L has L // 2 (integer division) supervised "
             f"tokens.", ""]
    rows = [("every row at the longest record", "-", _int(N_RECORDS * lmax), _pct(1 - useful / (N_RECORDS * lmax)), "-", "-")]
    for p in POOLS:
        batches, computed, sup = _batch_numbers(p)
        rows.append((f"pool of {p} batch{'' if p == 1 else 'es'}" + (" (plain shuffled batches)" if p == 1 else " (one global sort)" if p == 30 else ""), p,
                     _int(computed), _pct(1 - useful / computed), f"{min(sup)} to {max(sup)}", _fx(max(sup) / min(sup), 2)))
    lines.append(_table(["batching", "pool", "positions computed", "padding, percent", "supervised tokens per batch",
                         "heaviest / lightest batch"], rows))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ mixtures

MIX_MAX_SCALE = 16
MIX_TOLERANCE = 0.01


def _round_half_up(x):
    return math.floor(x + 0.5)


def mix_repeats(groups):
    """`groups` is [(records, share)], the dataset first (its share is what the others leave). Each group's weight per record is its
    share over its records; the repeats are those weights over the smallest, times a whole scale from 1 to 16, rounded half away
    from zero and at least 1; the first scale at which every realised share is within 0.01 of its share is used (the closest scale
    when none is). A group with no records is listed once. Returns (repeats, scale, deviation)."""
    others = sum(s for _, s in groups[1:])
    shares = [(groups[0][0], 1.0 - others)] + [(n, s) for n, s in groups[1:]]
    weights = [s / n if n > 0 else None for n, s in shares]
    least = min(w for w in weights if w is not None)

    def at(scale):
        return [1 if w is None else max(1, _round_half_up(w / least * scale)) for w in weights]

    def deviation(rep):
        rows = sum(r * n for r, (n, _) in zip(rep, shares))
        return max(abs(r * n / rows - s) for r, (n, s) in zip(rep, shares) if n > 0)

    best = None
    for scale in range(1, MIX_MAX_SCALE + 1):
        rep = at(scale)
        off = deviation(rep)
        if off <= MIX_TOLERANCE:
            return rep, scale, off
        if best is None or off < best[2]:
            best = (rep, scale, off)
    return best


def mixture():
    cases = [("a replay set nine times the dataset", [(100, 0.0), (900, 0.25)]),
             ("a replay set a tenth of the dataset", [(100, 0.0), (10, 0.5)]),
             ("a ratio that rounds badly at scale 1", [(278, 0.0), (150, 0.25)]),
             ("two replay groups", [(278, 0.0), (60, 0.25), (93, 0.25)])]
    rows = []
    for name, groups in cases:
        rep, scale, off = mix_repeats(groups)
        listed = [r * n for r, (n, _) in zip(rep, groups)]
        total = sum(listed)
        asked = [1.0 - sum(s for _, s in groups[1:])] + [s for _, s in groups[1:]]
        rows.append((name, ", ".join(str(n) for n, _ in groups), ", ".join(f"{s:.2f}" for s in asked), ", ".join(str(r) for r in rep),
                     scale, _int(total), ", ".join(f"{x / total:.4f}" for x in listed), _fx(off, 4)))
    lines = [_table(["case", "records per group", "share asked", "times listed", "scale", "rows listed", "share realised",
                     "largest deviation"], rows, "lrrrrrrr"), ""]
    # A share of rows is not a share of tokens.
    a_len, b_len = 40, 200
    def row_share_for_token_share(t):
        return t * a_len / (b_len * (1 - t) + t * a_len)

    lines.append(f"Two sources: every record of A has {a_len} tokens and every record of B has {b_len}. A share w of "
                 f"B among the rows drawn is a share w * {b_len} / ((1 - w) * {a_len} + w * {b_len}) of the tokens drawn.")
    lines.append("")
    body = []
    for w in (0.05, 0.1, 0.2, 0.5):
        t = w * b_len / ((1 - w) * a_len + w * b_len)
        body.append((_fx(w, 2), _fx(t, 4)))
    lines.append(_table(["share of B among the rows", "share of B among the tokens"], body))
    lines.append("")
    body = []
    for t in (0.05, 0.1, 0.2, 0.5):
        w = row_share_for_token_share(t)
        body.append((_fx(t, 2), _fx(w, 4)))
    lines.append(_table(["share of B wanted among the tokens", "share of B to ask among the rows"], body))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ duplicates

DOCS = "training_data.docs"
PLANT = "training_data.plant"
FRESH = "training_data.fresh"
MINHASH = "training_data.minhash"
VOCAB = 150
DOC_WORDS = 60
N_ORIGINAL = 200
SHINGLE = 3
RATES = (0.1, 0.2, 0.3)
K_HASHES = 64


def _word(i):
    return f"w{i:03d}"


def _draw_doc(g):
    return [g.integer(VOCAB) for _ in range(DOC_WORDS)]


def _corrupt(words, rate, g):
    """Every word in turn: one `uniform()`; below `rate` the word is replaced, and then one `integer(150)` gives the new word."""
    out = []
    for w in words:
        if g.uniform() < rate:
            out.append(g.integer(VOCAB))
        else:
            out.append(w)
    return out


@lru_cache(maxsize=None)
def corpus():
    """Returns (texts, kind, source): the raw text of each document, its kind ('original', 'copy', 'format', 'rate 0.1' ...) and the
    index of the original it was made from (-1 for an original).

    200 originals from `training_data.docs`: 60 `integer(150)` draws each, word i written `w` and its three digits. Then the
    generator `training_data.plant` shuffles the list 0 .. 199; its first 30 entries are copied exactly, the next 30 are copied
    with every word in capitals and two spaces between words, and the 20 entries after those, then the next 20, then the next 20 are
    corrupted at the rates 0.1, 0.2 and 0.3 (the same generator continues, in that order, one entry after another)."""
    g = bookrng.generator(DOCS)
    base = [_draw_doc(g) for _ in range(N_ORIGINAL)]
    texts = [" ".join(_word(w) for w in d) for d in base]
    kinds = ["original"] * N_ORIGINAL
    source = [-1] * N_ORIGINAL
    p = bookrng.generator(PLANT)
    order = list(range(N_ORIGINAL))
    p.shuffle(order)
    for i in order[:30]:
        texts.append(texts[i])
        kinds.append("copy")
        source.append(i)
    for i in order[30:60]:
        texts.append("  ".join(_word(w).upper() for w in base[i]))
        kinds.append("format")
        source.append(i)
    for r, rate in enumerate(RATES):
        for i in order[60 + 20 * r:80 + 20 * r]:
            texts.append(" ".join(_word(w) for w in _corrupt(base[i], rate, p)))
            kinds.append(f"rate {rate}")
            source.append(i)
    return texts, kinds, source


def normalise(text):
    """Lower-case, split on runs of white space, join with single spaces."""
    return " ".join(text.lower().split())


def shingles(text):
    """The set of word 3-shingles of the normalised text: each run of three consecutive words joined by single spaces."""
    w = normalise(text).split()
    return {" ".join(w[i:i + SHINGLE]) for i in range(len(w) - SHINGLE + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b)


def fingerprint(text):
    """The first 8 bytes of the SHA-256 digest of the UTF-8 text as one big-endian integer (the rule of `seed_of`)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


@lru_cache(maxsize=None)
def minhash_seeds():
    g = bookrng.generator(MINHASH)
    return [g.next_u64() for _ in range(K_HASHES)]


def minhash(sh):
    """The signature of a shingle set: entry i is the smallest over the shingles of mix(fingerprint(shingle) xor s_i), with s_i the
    i-th raw value of the generator `training_data.minhash` and mix the three scrambling lines of the SplitMix64 step."""
    fps = [fingerprint(s) for s in sh]
    return [min(bookrng._mix(f ^ s) for f in fps) for s in minhash_seeds()]


def estimate(sig_a, sig_b):
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)


@lru_cache(maxsize=None)
def _sets():
    texts, kinds, source = corpus()
    sh = [shingles(t) for t in texts]
    sigs = [minhash(s) for s in sh]
    return sh, sigs


def _first_copies(keys):
    """Documents that repeat an earlier key: how many, when every document after the first of its key is dropped."""
    seen = set()
    drop = 0
    for k in keys:
        if k in seen:
            drop += 1
        seen.add(k)
    return drop


def duplicates():
    texts, kinds, source = corpus()
    sh, sigs = _sets()
    n = len(texts)
    planted = [i for i in range(n) if source[i] >= 0]
    lines = [f"{N_ORIGINAL} originals of {DOC_WORDS} words from `{DOCS}` (seed {_seed(DOCS)}), planted copies from `{PLANT}` "
             f"(seed {_seed(PLANT)}) in this way: {n} documents, {len(planted)} of them planted. "
             f"An original is {DOC_WORDS} draws `integer({VOCAB})` of `{DOCS}`, document after document, word k written `w` and its "
             f"three digits (`w007`) and the words joined by single spaces. The generator `{PLANT}` shuffles the list 0 .. "
             f"{N_ORIGINAL - 1} once (the Fisher-Yates shuffle of the randomness chapter), so the originals are chosen without repeats; "
             f"its first 30 entries name the "
             f"originals copied exactly, the next 30 the originals copied in capitals (`W007`) with two spaces between words, and "
             f"the next three runs of 20 the originals corrupted at the rates 0.1, 0.2 and 0.3. A corrupted copy takes its words in "
             f"turn: one `uniform()` from `{PLANT}`, and when it is below the rate one `integer({VOCAB})` from the same generator "
             f"gives the new word (the copies and the format variants use no draws). A shingle is three consecutive words of the "
             f"normalised text joined by single spaces (`w007 w101 w044`); a document has {DOC_WORDS - SHINGLE + 1} shingles at most.", ""]
    raw = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    norm = [hashlib.sha256(normalise(t).encode("utf-8")).hexdigest() for t in texts]
    lines.append(_table(["rule for calling a document a duplicate of an earlier one", "documents dropped", "of the planted"],
                        [("SHA-256 of the raw text", _first_copies(raw), "30 copies"),
                         ("SHA-256 of the normalised text", _first_copies(norm), "30 copies and 30 format variants")], "lrl"))
    lines.append("")
    # exact Jaccard of every planted document with its original
    by_kind = {}
    for i in planted:
        by_kind.setdefault(kinds[i], []).append(i)
    rows = []
    for kind in ("copy", "format", "rate 0.1", "rate 0.2", "rate 0.3"):
        ids = by_kind[kind]
        js = [jaccard(sh[i], sh[source[i]]) for i in ids]
        es = [estimate(sigs[i], sigs[source[i]]) for i in ids]
        err = sum(abs(a - b) for a, b in zip(js, es)) / len(ids)
        rows.append((kind, len(ids), _fx(sum(js) / len(js), 4), _fx(min(js), 4), _fx(sum(es) / len(es), 4), _fx(err, 4)))
    lines.append(_table(["planted kind", "documents", "mean Jaccard with the original", "smallest", "mean MinHash estimate (64 hashes)",
                         "mean absolute error of the estimate"], rows))
    lines.append("")
    # all pairs
    pairs = [(i, j) for j in range(n) for i in range(j)]
    truth = {(min(i, source[i]), max(i, source[i])) for i in planted}
    j_all = {p: jaccard(sh[p[0]], sh[p[1]]) for p in pairs}
    e_all = {p: estimate(sigs[p[0]], sigs[p[1]]) for p in pairs}
    rows = []
    for tau in (0.5, 0.3, 0.2):
        fj = {p for p, v in j_all.items() if v >= tau}
        fe = {p for p, v in e_all.items() if v >= tau}

        def rate(flag, kinds_wanted):
            ids = [i for i in planted if kinds[i] in kinds_wanted]
            return sum(1 for i in ids if (min(i, source[i]), max(i, source[i])) in flag), len(ids)

        k1, n1 = rate(fj, ("copy", "format"))
        k2, n2 = rate(fj, ("rate 0.1",))
        k3, n3 = rate(fj, ("rate 0.2",))
        k4, n4 = rate(fj, ("rate 0.3",))
        rows.append((tau, f"{k1} of {n1}", f"{k2} of {n2}", f"{k3} of {n3}", f"{k4} of {n4}", len(fj - truth), len(fe - truth),
                     len(truth - fe)))
    lines.append(f"All {_int(len(pairs))} pairs, a pair being flagged when its similarity is at least the threshold.")
    lines.append("")
    lines.append(_table(["threshold", "copies and variants found (exact Jaccard)", "rate 0.1", "rate 0.2", "rate 0.3",
                         "false pairs (exact Jaccard)", "false pairs (estimate)", "planted pairs missed (estimate)"], rows))
    return "\n".join(lines)


def banding():
    """The S-curve of banded signatures and the candidate pairs of the corpus."""
    texts, kinds, source = corpus()
    sh, sigs = _sets()
    n = len(texts)
    settings = ((32, 2), (16, 4), (8, 8))
    lines = [f"A signature of {K_HASHES} entries is cut into b bands of r consecutive entries (b r = {K_HASHES}); two documents are a "
             f"candidate pair when all r entries of at least one band are equal. A pair of Jaccard similarity J is a candidate with "
             f"probability 1 - (1 - J^r)^b. The {K_HASHES} seeds of the signatures are the first {K_HASHES} raw values (`next`) of "
             f"the generator `{MINHASH}` (seed {_seed(MINHASH)}); band k holds entries k r to k r + r - 1.", ""]
    rows = []
    for J in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        rows.append([_fx(J, 1)] + [_fx(1 - (1 - J ** r) ** b, 4) for b, r in settings])
    lines.append(_table(["J"] + [f"b = {b}, r = {r}" for b, r in settings], rows))
    lines.append("")
    pairs = [(i, j) for j in range(n) for i in range(j)]
    truth = {(min(i, source[i]), max(i, source[i])) for i in range(n) if source[i] >= 0}
    rows = []
    for b, r in settings:
        cand = set()
        for i, j in pairs:
            if any(sigs[i][k * r:(k + 1) * r] == sigs[j][k * r:(k + 1) * r] for k in range(b)):
                cand.add((i, j))
        good = {p for p in cand if jaccard(sh[p[0]], sh[p[1]]) >= 0.3}
        check = sum(min(len(sh[i]), len(sh[j])) for i, j in cand)
        rows.append((f"b = {b}, r = {r}", len(cand), _fx(len(cand) / len(pairs), 4), len(truth & cand), len(good), len(cand) - len(good),
                     _int(check)))
    lines.append(_table(["banding", "candidate pairs", "share of all pairs", "planted pairs among them", "candidates with exact J >= 0.3",
                         "candidates below 0.3", "lookups to check the candidates"], rows))
    lines.append("")
    total_sh = sum(len(x) for x in sh)
    exact = sum(min(len(sh[i]), len(sh[j])) for i, j in pairs)
    lines.append(f"Counting rule for the work: one lookup of a shingle in another document's shingle set is one operation, one hash "
                 f"evaluation of a shingle under one of the {K_HASHES} seeds is one operation, and one entry written into a band table "
                 f"is one operation. Comparing every pair exactly costs the smaller shingle count of each pair in lookups, "
                 f"{_int(exact)} here. Building the signatures costs {_int(total_sh)} shingles times {K_HASHES} seeds, "
                 f"{_int(total_sh * K_HASHES)} hash evaluations; filling the band tables costs one entry per document and band, "
                 f"{_int(n * 32)}, {_int(n * 16)} and {_int(n * 8)} for the three bandings; checking the candidates exactly costs "
                 f"the last column of the table above.")
    return "\n".join(lines)


def contamination():
    texts, kinds, source = corpus()
    sh, _ = _sets()
    g = bookrng.generator(FRESH)
    base_words = [t.split(" ") for t in texts[:N_ORIGINAL]]
    train_sets = sh[:N_ORIGINAL]
    order = list(range(N_ORIGINAL))
    g.shuffle(order)
    val_texts, truth = [], []
    for i in order[:6]:
        words = [int(w[1:]) for w in base_words[i]]
        val_texts.append(" ".join(_word(w) for w in _corrupt(words, 0.1, g)))
        truth.append(i)
    for _ in range(24):
        val_texts.append(" ".join(_word(w) for w in _draw_doc(g)))
    val_sets = [shingles(t) for t in val_texts]
    best = []
    for v in val_sets:
        js = [jaccard(v, t) for t in train_sets]
        m = max(js)
        best.append((m, js.index(m)))
    raw_train = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts[:N_ORIGINAL]}
    flagged = [k for k, (m, _) in enumerate(best) if m >= 0.3]
    drop = {best[k][1] for k in flagged}
    after = [max(jaccard(v, t) for i, t in enumerate(train_sets) if i not in drop) for v in val_sets]
    lines = [f"A validation set of 30 documents from `{FRESH}` (seed {_seed(FRESH)}) against the {N_ORIGINAL} original documents as "
             f"the training set. The generator first shuffles 0 .. {N_ORIGINAL - 1}; for the first 6 entries, in order, the training "
             f"document of that index is corrupted at rate 0.1 to give validation documents 0 to 5 (each word in turn: one `uniform()` from `{FRESH}`, and below 0.1 "
             f"one `integer({VOCAB})` from the same generator gives the new word); then 24 fresh documents "
             f"of {DOC_WORDS} `integer(150)` draws each follow.", ""]
    lines.append(_table(["check", "validation documents flagged"],
                        [("an exact SHA-256 match with a training text", sum(1 for t in val_texts if hashlib.sha256(t.encode("utf-8")).hexdigest() in raw_train)),
                         ("a training document with Jaccard at least 0.3", f"{len(flagged)} (numbers {', '.join(map(str, flagged))})"),
                         ("the same, after dropping the training documents those matched", sum(1 for x in after if x >= 0.3))], "lr"))
    lines.append("")
    lines.append(f"The largest Jaccard similarities of the flagged validation documents with the training set run from "
                 f"{_fx(min(best[k][0] for k in flagged), 4)} to {_fx(max(best[k][0] for k in flagged), 4)}; of the 24 fresh documents the "
                 f"largest is {_fx(max(m for m, _ in best[6:]), 4)}. The training documents dropped are {len(drop)}, and the best match of "
                 f"every flagged validation document is the training document it was made from: "
                 f"{all(best[k][1] == truth[k] for k in range(6))}.")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the order of examples

EPOCH_SEED = "training_data.epochs"


def epoch_order(seed, n, epoch):
    """The order the n examples are trained in during `epoch`: the list 0 .. n - 1 shuffled by the generator whose seed is `seed xor
    ((epoch + 1) * 0x9E3779B97F4A7C15 mod 2^64)`."""
    g = bookrng.SplitMix64(seed ^ (((epoch + 1) * GAMMA) & M64))
    order = list(range(n))
    g.shuffle(order)
    return order


def _stream(seed, n, start, count):
    """The picks of draws start .. start + count - 1: draw d is entry d mod n of the order of epoch d // n."""
    return [epoch_order(seed, n, d // n)[d % n] for d in range(start, start + count)]


def epochs():
    seed = bookrng.seed_of(EPOCH_SEED)
    n = 7
    lines = [f"7 examples, 0 to 6, and the run seed {_seed(EPOCH_SEED)} (the seed of the name `{EPOCH_SEED}`).", ""]
    rows = [(e, " ".join(map(str, epoch_order(seed, n, e)))) for e in range(3)]
    lines.append(_table(["epoch", "order of the 7 examples"], rows, "rl"))
    lines.append("")
    straight = _stream(seed, n, 0, 21)
    resumed = _stream(seed, n, 10, 11)
    restarted = epoch_order(seed, n, 0)[10 % n:] + epoch_order(seed, n, 1)
    lines.append(_table(["run", "draws 10 to 20"],
                        [("straight run, read from draw 10", " ".join(map(str, straight[10:]))),
                         ("resumed at draw 10: epoch 10 // 7, entry 10 mod 7, then on", " ".join(map(str, resumed))),
                         ("resumed with the epoch forgotten: entry 3 of epoch 0, then epoch 1", " ".join(map(str, restarted[:11])))], "ll"))
    lines.append("")
    lines.append(f"The resumed run equals the straight run: {resumed == straight[10:]}. Every epoch holds each example once: "
                 f"{all(sorted(epoch_order(seed, n, e)) == list(range(n)) for e in range(50))} (epochs 0 to 49).")
    lines.append("")
    world = 3
    rows = []
    for rank in range(world):
        mine = [(d, straight[d]) for d in range(21) if d % world == rank]
        rows.append((rank, " ".join(f"{d}:{p}" for d, p in mine)))
    lines.append(f"With {world} workers, draw d belongs to worker d mod {world}; every worker computes all 21 draws and keeps its own "
                 f"(draw:example).")
    lines.append("")
    lines.append(_table(["worker", "draws it keeps"], rows, "rl"))
    return "\n".join(lines)
