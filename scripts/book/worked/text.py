# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Text as Data": characters, code points and bytes.

The encoder and the decoder below are written from the chapter's own description of UTF-8: the prefix
patterns of the four sequence lengths, the bit arithmetic that fills them, and a decoder that is a table
of states (`DECODER_TABLE`, the same table the chapter's state-machine figure draws). Python's own codecs
(`str.encode`, `bytes.decode`) and `unicodedata` appear only as the reference the independent
implementation is checked against; every check that draws random inputs uses the book's SplitMix64
(`bookrng.generator`) and prints its seed. Tables are plain ASCII so that they typeset: a character is
named by its code point, never printed, because the book's fonts lack most scripts.

Swedish Embedded AB implements solutions for getting text into models without silent corruption. If your
team needs expertise in text encodings, normalization and tokenizer pipelines, you can procure our
services by sending an email to info@swedishembedded.com.
"""
import unicodedata

import bookrng

MAX_CODE_POINT = 0x10FFFF
SURROGATES = (0xD800, 0xDFFF)
REPLACEMENT = 0xFFFD

# The sample string of the chapter: one character of each UTF-8 length.
SAMPLE = "Aé猫\U0001F600"

# The same greeting in several scripts, as the chapter's inputs for the bytes-per-character counts.
PHRASES = [
    ("English", "Hello, world"),
    ("Swedish", "Hej, världen"),
    ("Russian", "Привет, мир"),
    ("Greek", "Γεια σου, κόσμε"),
    ("Hindi", "नमस्ते दुनिया"),
    ("Chinese", "你好，世界"),
    ("Japanese", "こんにちは、世界"),
    ("Emoji", "\U0001F44B\U0001F30D"),
]


# ------------------------------------------------------------------ small helpers

def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _n(v):
    return f"{v:,}"


def cp_hex(cp):
    """A code point as the book writes it: U+ and at least four hex digits."""
    return f"U+{cp:04X}"


def cps(text):
    return [ord(c) for c in text]


def byte_hex(bs):
    return " ".join(f"{b:02X}" for b in bs)


def cp_list(values):
    return " ".join(f"{v:04X}" if v <= 0xFFFF else f"{v:X}" for v in values)


def _seed_line(name):
    return (f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} "
            f"(`seed_of` of the name `{name}`, defined on the pseudocode page).")


def _name(cp):
    return unicodedata.name(chr(cp), f"<no name, category {unicodedata.category(chr(cp))}>")


# ------------------------------------------------- the encoder, from the chapter's rules

def is_scalar_value(cp):
    return 0 <= cp <= MAX_CODE_POINT and not (SURROGATES[0] <= cp <= SURROGATES[1])


def encode_utf8(code_points):
    """UTF-8 of a list of Unicode scalar values: the pattern of the sequence length that holds them."""
    out = bytearray()
    for cp in code_points:
        if not is_scalar_value(cp):
            raise ValueError(f"{cp:#x} is not a Unicode scalar value")
        if cp < 0x80:
            out.append(cp)
        elif cp < 0x800:
            out += bytes([0xC0 | (cp >> 6), 0x80 | (cp & 0x3F)])
        elif cp < 0x10000:
            out += bytes([0xE0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F)])
        else:
            out += bytes([0xF0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3F), 0x80 | ((cp >> 6) & 0x3F),
                          0x80 | (cp & 0x3F)])
    return bytes(out)


def utf8_length(cp):
    return 1 if cp < 0x80 else 2 if cp < 0x800 else 3 if cp < 0x10000 else 4


def encode_utf16(code_points):
    """UTF-16 code units: one unit below U+10000, else a surrogate pair of the 20 bits above 0x10000."""
    units = []
    for cp in code_points:
        if not is_scalar_value(cp):
            raise ValueError(f"{cp:#x} is not a Unicode scalar value")
        if cp < 0x10000:
            units.append(cp)
        else:
            v = cp - 0x10000
            units += [0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF)]
    return units


# ------------------------------------------- the decoder, as a state machine

# A state is (need, lo, hi): the continuation bytes still expected, and the range the NEXT one must fall in.
# After a lead byte the range is 80..BF except where the shortest-form and range rules narrow it.
START = (0, 0x80, 0xBF)
STATE_NAMES = {START: "start", (1, 0x80, 0xBF): "need1", (2, 0x80, 0xBF): "need2", (3, 0x80, 0xBF): "need3",
               (2, 0xA0, 0xBF): "after E0", (2, 0x80, 0x9F): "after ED",
               (3, 0x90, 0xBF): "after F0", (3, 0x80, 0x8F): "after F4"}
_LEAD_MASK = {2: 0x1F, 3: 0x0F, 4: 0x07}


def step(state, b):
    """One byte: (next state, action), or None when the byte is invalid in this state.

    Actions: 'ascii' (a character by itself), 'lead2/3/4' (start a value from the low 5/4/3 bits),
    'cont' (shift in six more bits)."""
    need, lo, hi = state
    if need > 0:
        return ((need - 1, 0x80, 0xBF), "cont") if lo <= b <= hi else None
    if b <= 0x7F:
        return START, "ascii"
    if 0xC2 <= b <= 0xDF:
        return (1, 0x80, 0xBF), "lead2"
    if 0xE0 <= b <= 0xEF:
        return (2, 0xA0 if b == 0xE0 else 0x80, 0x9F if b == 0xED else 0xBF), "lead3"
    if 0xF0 <= b <= 0xF4:
        return (3, 0x90 if b == 0xF0 else 0x80, 0x8F if b == 0xF4 else 0xBF), "lead4"
    return None


class Utf8Decoder:
    """The state machine of the chapter, fed any number of byte chunks.

    `feed` returns the code points completed so far and keeps the unfinished sequence; `finish` ends the
    stream. With `strict`, an invalid byte raises; otherwise each maximal invalid prefix becomes U+FFFD
    and decoding restarts at the byte that broke it."""

    def __init__(self, strict=False):
        self.strict, self.state, self.value = strict, START, 0

    def feed(self, chunk):
        out = []
        for b in chunk:
            while True:
                nxt = step(self.state, b)
                if nxt is not None:
                    self.state, action = nxt
                    if action == "ascii":
                        out.append(b)
                    elif action == "cont":
                        self.value = (self.value << 6) | (b & 0x3F)
                        if self.state == START:
                            out.append(self.value)
                    else:
                        self.value = b & _LEAD_MASK[int(action[-1])]
                    break
                if self.strict:
                    raise ValueError(f"byte {b:#04x} is invalid in state {STATE_NAMES[self.state]!r}")
                out.append(REPLACEMENT)
                if self.state == START:
                    break  # the offending byte stood alone: it is consumed
                self.state = START  # a broken sequence: the byte starts over
        return out

    def finish(self):
        if self.state == START:
            return []
        if self.strict:
            raise ValueError("the stream ends inside a sequence")
        self.state = START
        return [REPLACEMENT]


def decode_utf8(data, strict=False):
    d = Utf8Decoder(strict)
    return d.feed(data) + d.finish()


def python_decode(data):
    """Python's reference decoding with replacement, as code points."""
    return cps(data.decode("utf-8", errors="replace"))


# -------------------------------------------------------------- worked examples

def sample_string():
    rows = []
    totals = [0, 0, 0]
    for ch in SAMPLE:
        cp = ord(ch)
        u8, u16, u32 = encode_utf8([cp]), encode_utf16([cp]), [cp]
        totals[0] += len(u8)
        totals[1] += len(u16)
        totals[2] += len(u32)
        rows.append([cp_hex(cp), _name(cp), f"{cp:b}", byte_hex(u8), " ".join(f"{u:04X}" for u in u16), f"{cp:08X}"])
    table = _table(["code point", "name", "binary", "UTF-8 bytes", "UTF-16 units", "UTF-32 unit"], rows, "lllllr")
    b16 = len(SAMPLE.encode("utf-16-le"))
    return (f"The string has {len(SAMPLE)} code points; it is written out below.\n\n{table}\n\n"
            f"Counted: {totals[2]} UTF-32 units ({4 * totals[2]} bytes), {totals[1]} UTF-16 units ({2 * totals[1]} bytes), "
            f"{totals[0]} UTF-8 bytes. Python's `len` of the string is {len(SAMPLE)}, its UTF-16 encoding has "
            f"{b16 // 2} units, its UTF-8 encoding {len(SAMPLE.encode('utf-8'))} bytes.")


def ascii_layout():
    rows = []
    for label, code in [("NUL", 0), ("LINE FEED", 10), ("SPACE", 32), ("DIGIT ZERO", 0x30), ("DIGIT NINE", 0x39),
                        ("LATIN CAPITAL A", 0x41), ("LATIN SMALL A", 0x61), ("LATIN CAPITAL Z", 0x5A),
                        ("LATIN SMALL Z", 0x7A), ("DELETE", 127)]:
        rows.append([label, code, f"{code:02X}", f"{code:07b}"])
    control = len([c for c in range(128) if c < 32 or c == 127])
    printable = 128 - control
    flip = [(c, c ^ 0x20) for c in (0x41, 0x4D, 0x5A)]
    flips = ", ".join(f"{a:07b} xor 0100000 = {b:07b} ({a} to {b})" for a, b in flip)
    return (_table(["character", "decimal", "hex", "7-bit pattern"], rows, "lrrl") +
            f"\n\nOf the 128 codes, {control} are control codes (0 to 31 and 127) and {printable} print (32 to 126). "
            f"Flipping bit 5 (value 32) changes the case of a letter: {flips}.")


def mojibake():
    rows = []
    for label, text in [("e with acute", "é"), ("A, e with acute", "Aé"), ("CJK ideograph for cat", "猫"),
                        ("Greek small alpha", "α")]:
        raw = text.encode("utf-8")
        wrong = [ord(c) for c in raw.decode("latin-1")]
        rows.append([label, byte_hex(raw), cp_list(cps(text)), " ".join(cp_hex(c) for c in wrong)])
    return (_table(["text", "UTF-8 bytes", "intended code points", "the same bytes read as ISO 8859-1"], rows, "llll") +
            "\n\nReading a byte as a Latin-1 character is a table lookup that never fails: byte b is code point b.")


def unicode_space():
    cats = {"letters (L*)": 0, "marks (M*)": 0, "numbers (N*)": 0, "punctuation (P*)": 0, "symbols (S*)": 0,
            "separators (Z*)": 0, "controls and format (Cc, Cf)": 0, "private use (Co)": 0, "surrogates (Cs)": 0,
            "unassigned (Cn)": 0}
    for cp in range(MAX_CODE_POINT + 1):
        c = unicodedata.category(chr(cp))
        if c[0] == "L":
            cats["letters (L*)"] += 1
        elif c[0] == "M":
            cats["marks (M*)"] += 1
        elif c[0] == "N":
            cats["numbers (N*)"] += 1
        elif c[0] == "P":
            cats["punctuation (P*)"] += 1
        elif c[0] == "S":
            cats["symbols (S*)"] += 1
        elif c[0] == "Z":
            cats["separators (Z*)"] += 1
        elif c in ("Cc", "Cf"):
            cats["controls and format (Cc, Cf)"] += 1
        elif c == "Co":
            cats["private use (Co)"] += 1
        elif c == "Cs":
            cats["surrogates (Cs)"] += 1
        else:
            cats["unassigned (Cn)"] += 1
    total = MAX_CODE_POINT + 1
    scalar = total - (SURROGATES[1] - SURROGATES[0] + 1)
    head = (f"17 planes of 65,536 code points: {_n(17 * 65536)} code points in U+0000 to U+10FFFF; minus the "
            f"{SURROGATES[1] - SURROGATES[0] + 1:,} surrogates U+D800 to U+DFFF leaves {_n(scalar)} scalar values.\n\n")
    rows = [[k, _n(v)] for k, v in cats.items()]
    rows.append(["all", _n(sum(cats.values()))])
    return (head + _table(["general category", "code points"], rows) +
            f"\n\nCounted with `unicodedata.category` of Python's Unicode database, version {unicodedata.unidata_version}. "
            "The rows for private use and surrogates are fixed by the standard; the others grow as Unicode assigns "
            "more characters, so another version prints other numbers.")


def graphemes():
    items = [
        ("e with acute, precomposed", [0xE9], 1),
        ("e then combining acute", [0x65, 0x301], 1),
        ("Swedish flag (two regional indicators)", [0x1F1F8, 0x1F1EA], 1),
        ("thumbs up with a skin-tone modifier", [0x1F44D, 0x1F3FD], 1),
        ("family: man, woman, girl joined by ZWJ", [0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467], 1),
        ("Devanagari NA with vowel sign I", [0x0928, 0x093F], 1),
        ("the word 'Hej'", cps("Hej"), 3),
    ]
    rows = []
    for label, seq, seen in items:
        s = "".join(map(chr, seq))
        rows.append([label, cp_list(seq), len(seq), len(s.encode("utf-8")), len(encode_utf16(seq)), seen])
    return (_table(["text", "code points (hex)", "code points", "UTF-8 bytes", "UTF-16 units", "seen by a reader"], rows, "llrrrr") +
            "\n\nThe last column is not computed: it is what the rules of Unicode text segmentation (UAX 29) "
            "and a reader agree on.")


def utf8_layout():
    """Pattern, capacity and the number of scalar values of each sequence length (counted by encoding them all)."""
    count = [0, 0, 0, 0, 0]
    for cp in range(MAX_CODE_POINT + 1):
        if is_scalar_value(cp):
            count[utf8_length(cp)] += 1
    rows = []
    lows = {1: 0, 2: 0x80, 3: 0x800, 4: 0x10000}
    for n, payload in [(1, 7), (2, 11), (3, 16), (4, 21)]:
        pattern = layout_pattern(n)
        hi = (1 << payload) - 1
        top = min(hi, MAX_CODE_POINT)
        rows.append([n, pattern, payload, f"{cp_hex(lows[n])} to {cp_hex(top)}", _n(count[n])])
    return (_table(["bytes", "bit pattern", "payload bits", "scalar values it encodes", "how many"], rows, "rlrlr") +
            f"\n\nIn total {_n(sum(count[1:]))} scalar values. A 4-byte sequence could carry up to "
            f"{cp_hex((1 << 21) - 1)}, but nothing above {cp_hex(MAX_CODE_POINT)} is a code point.")


def layout_pattern(n):
    """The bit pattern of an n-byte sequence, lead byte first, 'x' for a payload bit."""
    if n == 1:
        return "0xxxxxxx"
    lead = "1" * n + "0" + "x" * (7 - n)
    return " ".join([lead] + ["10xxxxxx"] * (n - 1))


def layout_rows():
    """For the layout figure: for each length, the bytes as lists of bit characters."""
    return [[list(b) for b in layout_pattern(n).split()] for n in (1, 2, 3, 4)]


def utf8_bits():
    lines = []
    for ch in SAMPLE:
        cp = ord(ch)
        n = utf8_length(cp)
        payload = [7, 11, 16, 21][n - 1]
        bits = f"{cp:0{payload}b}"
        if n == 1:
            groups = [bits]
        else:
            first = 7 - n
            groups = [bits[:first]] + [bits[first + 6 * i:first + 6 * (i + 1)] for i in range(n - 1)]
        pat = layout_pattern(n).split()
        filled = []
        for p, g in zip(pat, groups):
            it = iter(g)
            filled.append("".join(next(it) if c == "x" else c for c in p))
        out = bytes(int(b, 2) for b in filled)
        assert out == encode_utf8([cp]) == ch.encode("utf-8")
        lines.append([cp_hex(cp), f"{cp:b}", f"{bits}", " / ".join(groups), " ".join(filled), byte_hex(out)])
    return _table(["code point", "binary", "padded to the payload width", "payload split by the pattern", "bytes in binary",
                   "hex"], lines, "llllll")


def utf16_pair():
    cp = 0x1F600
    v = cp - 0x10000
    hi, lo = 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF)
    be = bytes([hi >> 8, hi & 255, lo >> 8, lo & 255])
    le = bytes([hi & 255, hi >> 8, lo & 255, lo >> 8])
    assert [hi, lo] == encode_utf16([cp]) and be == chr(cp).encode("utf-16-be") and le == chr(cp).encode("utf-16-le")
    rows = [
        ["code point", cp_hex(cp)],
        ["v = code point - 0x10000", f"0x{v:05X} = {v:020b}"],
        ["high 10 bits of v, v >> 10", f"0x{v >> 10:03X} = {v >> 10:010b}"],
        ["low 10 bits of v, v and 0x3FF", f"0x{v & 0x3FF:03X} = {v & 0x3FF:010b}"],
        ["high surrogate, 0xD800 + high", f"0x{hi:04X} = {hi:016b}"],
        ["low surrogate, 0xDC00 + low", f"0x{lo:04X} = {lo:016b}"],
        ["bytes, big-endian (UTF-16BE)", byte_hex(be)],
        ["bytes, little-endian (UTF-16LE)", byte_hex(le)],
        ["byte order mark U+FEFF, as UTF-16BE / UTF-16LE / UTF-8", f"{byte_hex(bytes([0xFE, 0xFF]))} / {byte_hex(bytes([0xFF, 0xFE]))} / {byte_hex(encode_utf8([0xFEFF]))}"],
    ]
    return _table(["step", "value"], rows, "ll")


# ----------------------------------------------------------- checks against the reference

def _scalar_draw(gen):
    k = gen.integer(MAX_CODE_POINT + 1 - 0x800)
    return k if k < SURROGATES[0] else k + 0x800


def check_encoder():
    name = "text.check_encoder"
    gen = bookrng.generator(name)
    n = 20000
    drawn = [_scalar_draw(gen) for _ in range(n)]
    ok_enc = ok_round = 0
    by_len = [0] * 5
    for cp in drawn:
        mine = encode_utf8([cp])
        ok_enc += mine == chr(cp).encode("utf-8")
        ok_round += decode_utf8(mine, strict=True) == [cp]
        by_len[len(mine)] += 1
    inv = [0xD800, 0xDFFF, 0x110000, -1]
    rejected = 0
    for cp in inv:
        try:
            encode_utf8([cp])
        except ValueError:
            rejected += 1
    edge = [0x00, 0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x10FFFF]
    edge_ok = sum(encode_utf8([c]) == chr(c).encode("utf-8") for c in edge)
    rows = [
        ["scalar values drawn", f"{n:,}"],
        ["of 1 / 2 / 3 / 4 bytes", " / ".join(str(x) for x in by_len[1:])],
        ["encoder equals Python's `str.encode('utf-8')`", f"{ok_enc:,} of {n:,}"],
        ["strict decoding of the encoding gives the code point back", f"{ok_round:,} of {n:,}"],
        [f"boundary values {', '.join(cp_hex(c) for c in edge)}", f"{edge_ok} of {len(edge)} equal"],
        ["encoder refuses U+D800, U+DFFF, U+110000, -1", f"{rejected} of {len(inv)} refused"],
    ]
    return (_seed_line(name) + " Each draw is `k = integer(1112064)`, then the scalar value `k` if `k < 0xD800` and `k + 0x800` otherwise "
            "(which skips the 2,048 surrogates and is uniform over the 1,112,064 scalar values).\n\n" +
            _table(["check", "result"], rows, "ll"))


def invalid_inputs():
    cases = [
        ("lone continuation byte", b"\x80"),
        ("byte never used (FF)", b"\xff"),
        ("overlong NUL, 2 bytes", b"\xc0\x80"),
        ("overlong slash, 2 bytes", b"\xc0\xaf"),
        ("overlong NUL, 3 bytes", b"\xe0\x80\x80"),
        ("overlong NUL, 4 bytes", b"\xf0\x80\x80\x80"),
        ("surrogate U+D800", b"\xed\xa0\x80"),
        ("beyond U+10FFFF (110000)", b"\xf4\x90\x80\x80"),
        ("lead byte F5, never valid", b"\xf5\x80\x80\x80"),
        ("truncated at the end", b"A\xe7\x8c"),
        ("interrupted by an ASCII byte", b"\xe7\x8cA"),
        ("interrupted by a lead byte", b"\xe7\xf0\x9f\x98\x80"),
        ("valid text for contrast", encode_utf8(cps(SAMPLE))),
    ]
    rows, agree = [], 0
    for why, data in cases:
        mine = decode_utf8(data)
        ref = python_decode(data)
        try:
            data.decode("utf-8")
            ref_strict = "accepts"
        except UnicodeDecodeError:
            ref_strict = "rejects"
        try:
            decode_utf8(data, strict=True)
            mine_strict = "accepts"
        except ValueError:
            mine_strict = "rejects"
        agree += (mine == ref) and (mine_strict == ref_strict)
        rows.append([why, byte_hex(data), mine_strict, cp_list(mine)])
    return (_table(["input", "bytes", "strict decoder", "decoded with replacement (code points)"], rows, "llll") +
            f"\n\nThe replacement output and the strict verdict equal Python's `bytes.decode('utf-8')` "
            f"(`errors='replace'` and strict) in {agree} of {len(cases)} rows.")


def _random_bytes(gen):
    """A byte string biased toward the cases a decoder must handle: ASCII, lead, continuation and any byte."""
    n = 1 + gen.integer(8)
    out = bytearray()
    for _ in range(n):
        kind = gen.integer(4)
        out.append(gen.integer(128) if kind == 0 else 0x80 + gen.integer(64) if kind == 1
                   else 0xC0 + gen.integer(64) if kind == 2 else gen.integer(256))
    return bytes(out)


def check_decoder():
    name = "text.check_decoder"
    gen = bookrng.generator(name)
    n = 50000
    same = valid = same_verdict = 0
    for _ in range(n):
        data = _random_bytes(gen)
        mine = decode_utf8(data)
        ref = python_decode(data)
        same += mine == ref
        try:
            data.decode("utf-8")
            ok = True
        except UnicodeDecodeError:
            ok = False
        valid += ok
        try:
            decode_utf8(data, strict=True)
            mine_ok = True
        except ValueError:
            mine_ok = False
        same_verdict += mine_ok == ok
    rows = [
        ["strings", f"{n:,}"],
        ["valid UTF-8 (strict accepts)", f"{valid:,}"],
        ["invalid (replacement used)", f"{n - valid:,}"],
        ["decoding with replacement equals Python's", f"{same:,} of {n:,}"],
        ["strict verdict (accept or reject) equals Python's", f"{same_verdict:,} of {n:,}"],
    ]
    return (_seed_line(name) + " Each string has length `1 + integer(8)`; each byte draws `kind = integer(4)` and then "
            "`integer(128)` (kind 0), `128 + integer(64)` (kind 1), `192 + integer(64)` (kind 2) or `integer(256)` (kind 3).\n\n" +
            _table(["check", "result"], rows, "ll"))


def check_stream():
    name = "text.check_stream"
    gen = bookrng.generator(name)
    n = 20000
    same = max_pending = 0
    for _ in range(n):
        data = _random_bytes(gen)
        d = Utf8Decoder()
        out, at = [], 0
        while at < len(data):
            step = 1 + gen.integer(3)
            out += d.feed(data[at:at + step])
            at += step
        out += d.finish()
        same += out == decode_utf8(data)
    # how many bytes can a stream hold back? feed prefixes of a 4-byte character
    held = []
    for prefix in range(1, 4):
        d = Utf8Decoder()
        emitted = d.feed(encode_utf8([0x1F600])[:prefix])
        held.append((prefix, len(emitted), STATE_NAMES[d.state]))
    rows = [
        ["strings", f"{n:,}"],
        ["chunked decoding equals one-shot decoding", f"{same:,} of {n:,}"],
    ]
    held_rows = [[p, e, s] for p, e, s in held]
    return (_seed_line(name) + " Each string is drawn by the rule of `text.check_decoder`; then chunk sizes `1 + integer(3)` "
            "are drawn, one per chunk, until the string is used up.\n\n" + _table(["check", "result"], rows, "ll") +
            "\n\nBytes of U+1F600 (F0 9F 98 80) fed one prefix at a time: nothing is emitted until the last byte arrives.\n\n" +
            _table(["bytes fed", "code points emitted", "state afterwards"], held_rows, "rrl"))


def order_preserved():
    name = "text.order_preserved"
    gen = bookrng.generator(name)
    n = 20000
    agree = 0
    for _ in range(n):
        a, b = _scalar_draw(gen), _scalar_draw(gen)
        agree += (a < b) == (encode_utf8([a]) < encode_utf8([b])) and (a == b) == (encode_utf8([a]) == encode_utf8([b]))
    return (_seed_line(name) + f"\n\nPairs (a, b) of scalar values, each drawn by `_scalar_draw` (a first, then b): {n:,}. "
            f"`a < b` agrees with comparing the two byte strings byte by byte in {agree:,} of {n:,}.")


def resync():
    data = encode_utf8(cps(SAMPLE))
    rows = []
    for i in range(len(data)):
        cut = data[:i] + data[i + 1:]
        got = decode_utf8(cut)
        rows.append([i, f"{data[i]:02X}", cp_list(got), sum(1 for c in got if c == REPLACEMENT)])
    start = []
    for k in range(len(data)):
        got = decode_utf8(data[k:])
        start.append(f"{k}: {cp_list(got)}")
    return ("The text is " + byte_hex(data) + " (code points " + cp_list(cps(SAMPLE)) + ").\n\n" +
            _table(["deleted byte index", "its value", "decoded code points (FFFD is the replacement)", "replacements"], rows, "rlll") +
            "\n\nStarting the decoder at byte k instead of byte 0 (code points, FFFD is the replacement):\n\n" +
            "\n".join(f"- start at {s}" for s in start))


# ---------------------------------------------------------------- normalization

def _forms(seq):
    s = "".join(map(chr, seq))
    return {f: cps(unicodedata.normalize(f, s)) for f in ("NFC", "NFD", "NFKC", "NFKD")}


def same_look_different_bytes():
    a = [0x63, 0x61, 0x66, 0xE9]
    b = [0x63, 0x61, 0x66, 0x65, 0x301]
    sa, sb = "".join(map(chr, a)), "".join(map(chr, b))
    rows = []
    for label, s in [("composed", sa), ("decomposed", sb)]:
        rows.append([label, cp_list(cps(s)), len(s), byte_hex(s.encode("utf-8")), len(s.encode("utf-8"))])
    t = _table(["spelling", "code points", "count", "UTF-8 bytes", "byte count"], rows, "llrlr")
    eq = [
        ["raw strings", sa == sb],
        ["after NFC", unicodedata.normalize("NFC", sa) == unicodedata.normalize("NFC", sb)],
        ["after NFD", unicodedata.normalize("NFD", sa) == unicodedata.normalize("NFD", sb)],
        ["after NFKC", unicodedata.normalize("NFKC", sa) == unicodedata.normalize("NFKC", sb)],
        ["after NFKD", unicodedata.normalize("NFKD", sa) == unicodedata.normalize("NFKD", sb)],
    ]
    t2 = _table(["compared", "equal"], [[k, str(v).lower()] for k, v in eq], "lr")
    return t + "\n\n" + t2


def normalization_forms():
    items = [
        ("e with acute (U+00E9)", [0xE9]),
        ("e, combining acute", [0x65, 0x301]),
        ("ANGSTROM SIGN (U+212B)", [0x212B]),
        ("LATIN SMALL LIGATURE FI (U+FB01)", [0xFB01]),
        ("SUPERSCRIPT TWO (U+00B2)", [0xB2]),
        ("FULLWIDTH LATIN CAPITAL A (U+FF21)", [0xFF21]),
        ("long s with dot above, dot below (U+1E9B U+0323)", [0x1E9B, 0x323]),
    ]
    rows = []
    for label, seq in items:
        f = _forms(seq)
        rows.append([label, cp_list(seq), cp_list(f["NFC"]), cp_list(f["NFD"]), cp_list(f["NFKC"]), cp_list(f["NFKD"])])
    return (_table(["input", "code points", "NFC", "NFD", "NFKC", "NFKD"], rows, "llllll") +
            f"\n\nProduced by `unicodedata.normalize` (Unicode {unicodedata.unidata_version}); code points in hex.")


def reordering():
    a = [0x73, 0x307, 0x323]
    b = [0x73, 0x323, 0x307]
    rows = []
    for label, seq in [("s, dot above, dot below", a), ("s, dot below, dot above", b)]:
        s = "".join(map(chr, seq))
        classes = " ".join(str(unicodedata.combining(c)) for c in s)
        rows.append([label, cp_list(seq), classes, cp_list(cps(unicodedata.normalize("NFD", s))),
                     cp_list(cps(unicodedata.normalize("NFC", s)))])
    sa, sb = ("".join(map(chr, x)) for x in (a, b))
    eq = unicodedata.normalize("NFD", sa) == unicodedata.normalize("NFD", sb)
    return (_table(["input", "code points", "combining classes", "NFD", "NFC"], rows, "lllll") +
            f"\n\nThe two inputs differ as raw strings ({str(sa == sb).lower()} for equality) and are equal after NFD ({str(eq).lower()}).")


S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
L_COUNT, V_COUNT, T_COUNT = 19, 21, 28
N_COUNT = V_COUNT * T_COUNT
S_COUNT = L_COUNT * N_COUNT


def hangul_decompose(s):
    """The arithmetic decomposition of a precomposed Hangul syllable into 2 or 3 conjoining jamo."""
    i = s - S_BASE
    out = [L_BASE + i // N_COUNT, V_BASE + (i % N_COUNT) // T_COUNT]
    t = i % T_COUNT
    if t:
        out.append(T_BASE + t)
    return out


def hangul():
    rows = []
    for s in (0xAC00, 0xD55C, 0xD7A3):
        i = s - S_BASE
        d = hangul_decompose(s)
        rows.append([cp_hex(s), i, i // N_COUNT, (i % N_COUNT) // T_COUNT, i % T_COUNT, cp_list(d)])
    agree = sum(hangul_decompose(S_BASE + i) == cps(unicodedata.normalize("NFD", chr(S_BASE + i))) for i in range(S_COUNT))
    return (_table(["syllable", "index i", "L index", "V index", "T index", "jamo (hex)"], rows, "lrrrrl") +
            f"\n\nWith N = {N_COUNT} (vowels times trailing slots) and T = {T_COUNT}: L index = i // N, V index = (i mod N) // T, T index = i mod T. "
            f"Constants: first syllable U+AC00, first leading jamo U+1100, first vowel U+1161, one before the first trailing "
            f"consonant U+11A7; {L_COUNT} leading consonants, {V_COUNT} vowels, {T_COUNT} trailing slots (slot 0 is none), so "
            f"{S_COUNT:,} syllables. The formula equals `unicodedata.normalize('NFD', ...)` for {agree:,} of {S_COUNT:,} syllables.")


def case_folding():
    items = [
        ("sharp s (U+00DF)", [0xDF]),
        ("Greek capitals omicron, delta, omicron, sigma", [0x39F, 0x394, 0x39F, 0x3A3]),
        ("capital I with dot above (U+0130)", [0x130]),
        ("LATIN SMALL LIGATURE FI (U+FB01)", [0xFB01]),
        ("LATIN CAPITAL LETTER A", [0x41]),
    ]
    rows = []
    for label, seq in items:
        s = "".join(map(chr, seq))
        rows.append([label, cp_list(seq), cp_list(cps(s.lower())), cp_list(cps(s.casefold())), cp_list(cps(s.upper()))])
    return _table(["input", "code points", "lower", "casefold", "upper"], rows, "lllll")


# -------------------------------------------------- bytes per character by script

def languages():
    inputs = "\n".join(f"- {n}: `{cp_list(cps(t))}`" for n, t in PHRASES)
    rows = []
    for n, t in PHRASES:
        c = cps(t)
        u8 = len(t.encode("utf-8"))
        rows.append([n, len(c), u8, f"{u8 / len(c):.3f}", 2 * len(encode_utf16(c)), 4 * len(c)])
    return ("Inputs, as code points in hex:\n\n" + inputs + "\n\n" +
            _table(["text", "code points", "UTF-8 bytes", "UTF-8 bytes per code point", "UTF-16 bytes", "UTF-32 bytes"], rows) +
            "\n\nByte counts are exact; the ratio is rounded to three decimals.")


def cumulative_bytes(text):
    """Bytes of the UTF-8 encoding after reading 0, 1, 2, ... code points, for the figure."""
    total, out = 0, [0]
    for c in cps(text):
        total += utf8_length(c)
        out.append(total)
    return out


def transition_rows():
    """(state, first byte, last byte, next state, action) for every byte range the decoder accepts, found by
    trying all 256 bytes in every state reachable from the start; the state-machine figure draws these."""
    seen, queue, rows = {START}, [START], []
    while queue:
        st = queue.pop(0)
        run = None
        for b in range(257):
            r = step(st, b) if b < 256 else None
            key = (r[0], r[1]) if r else None
            if run and run[0] == key and b < 256:
                run[2] = b
                continue
            if run and run[0] is not None:
                nxt, action = run[0]
                rows.append((STATE_NAMES[st], run[1], run[2], STATE_NAMES[nxt], action))
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
            run = [key, b, b]
    return rows
