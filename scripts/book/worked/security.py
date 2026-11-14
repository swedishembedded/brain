# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Security of Machine Learning Services".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a leading
underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: security.<name> -->` markers. Helpers with a leading underscore are shared with the chapter's figure.

Everything here is integer arithmetic on byte counts, the bytes of small hand-built files and enumerations of small walks, so
another implementation reproduces every printed digit. No generator is used.

Swedish Embedded AB implements solutions for inference services that stay up and keep their clients apart when the requests
and the model files they receive are hostile, for its clients. If your team needs expertise in hardening machine learning
servers then you can procure our services by sending an email to info@swedishembedded.com.
"""
import struct
import zlib

import number_formats as nf

_table = nf._table

KIB, MIB, GIB = 1 << 10, 1 << 20, 1 << 30

# The ceilings of the chapter, with the names they have in brain.
MAX_IMAGE_PIXELS = 36_000_000        # capability::blob::MAX_IMAGE_PIXELS, one image
MAX_REQUEST_PIXELS = 64_000_000      # capability::blob::MAX_REQUEST_PIXELS, all images of one request
MAX_IMAGES = 8                       # capability::blob::MAX_IMAGES
FLOAT_BYTES_PER_PIXEL = 12           # three 32-bit floats: red, green and blue
MAX_CONCURRENT_DECODES = 4           # apiserve::preprocess::MAX_CONCURRENT_DECODES
BODY_DIALECT = 8 * MIB               # apiserve::MAX_BODY_BYTES (chat, embeddings, images)
BODY_UPLOAD = 64 * MIB               # capabilities::MAX_BODY_BYTES and video::MAX_BODY_BYTES
BODY_DECISIONS = 2 * MIB             # systemone::MAX_BODY_BYTES
HEADER_DEADLINE_S = 10               # ConnectionLimits::header_read
BODY_DEADLINE_S = 300                # ConnectionLimits::body
MAX_CONNECTIONS = 1024               # ConnectionLimits::max_connections, per surface
EDGE_CONCURRENCY = 256               # residency::admission::EDGE_CONCURRENCY
MAX_FD_BLOB = 256 * MIB              # dbus::fd::MAX_FD_BLOB_BYTES
MAX_FDS_PER_CALL = 16                # dbus service MAX_FDS_PER_CALL
MAX_FD_CALL = 1 * GIB                # dbus service MAX_FD_CALL_BYTES
DEFLATE_MAX_RATIO = 1032             # 258 bytes per 2 bits of a dynamic-Huffman deflate stream (RFC 1951)


def _crc_chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def png_header_only(w, h):
    """The bytes of a PNG that declares w x h 8-bit RGB pixels and carries nothing of them: the 8-byte signature, an IHDR
    chunk (width, height, bit depth 8, colour type 2, compression 0, filter 0, interlace 0) and an IDAT chunk holding only
    the two bytes of a zlib stream header (0x78 0x01)."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return sig + _crc_chunk(b"IHDR", ihdr) + _crc_chunk(b"IDAT", b"\x78\x01")


def _b64_len(n):
    return 4 * ((n + 2) // 3)


def _ratio(a, b):
    r = a / b
    return f"{r:,.0f}" if r >= 100 else f"{r:.3g}"


# ------------------------------------------------------------------ what a declared size costs

def png_bomb():
    sizes = [(6000, 6000), (6001, 6000), (65535, 65535)]
    rows = []
    for w, h in sizes:
        f = png_header_only(w, h)
        px = w * h
        floats = px * FLOAT_BYTES_PER_PIXEL
        verdict = "passes the header check" if px <= MAX_IMAGE_PIXELS else "refused from the header"
        rows.append((f"{w:,} x {h:,}", f"{len(f)}", f"{_b64_len(len(f))}", f"{px:,}", f"{3 * px:,}", f"{floats:,}",
                     _ratio(floats, len(f)), verdict))
    f = png_header_only(65535, 65535)
    out = [f"A PNG that declares w x h 8-bit RGB (red, green, blue) pixels and holds no pixel data is {len(f)} bytes: the 8-byte signature, "
           "a 25-byte IHDR (image header) chunk (4 bytes of length, 4 of type, 13 of data: width and height as big-endian 32-bit "
           "integers, bit depth 8, colour type 2, compression, filter and interlace 0, and a 4-byte CRC-32, the chunk's cyclic "
           "redundancy check) and a 14-byte IDAT (image data) chunk "
           "whose data are the 2 bytes 0x78 0x01 of a zlib stream header. Base64 makes 4 characters of every 3 bytes. A "
           f"decoded pixel is three 32-bit floats, {FLOAT_BYTES_PER_PIXEL} bytes. The ratio is the float bytes over the file "
           f"bytes, rounded to the nearest integer. The header check refuses more than {MAX_IMAGE_PIXELS:,} pixels per image; "
           "exactly that many pass it.",
           "",
           f"The 65,535 x 65,535 file, in hexadecimal: `{f.hex(' ')}`.",
           "",
           _table(["declared size", "file bytes", "base64 characters", "pixels", "bytes as 8-bit RGB", "bytes as f32 RGB",
                   "f32 bytes per file byte", "header check"], rows, "lrrrrrrl")]
    return "\n".join(out)


def _raw_png_stream(w, h):
    return h * (1 + 3 * w)          # one filter byte per row, three bytes per pixel


def deflate_floor():
    rows = []
    for w, h in [(1000, 1000), (6000, 6000), (8000, 4500)]:
        raw = _raw_png_stream(w, h)
        least = -(-raw // DEFLATE_MAX_RATIO)
        floats = w * h * FLOAT_BYTES_PER_PIXEL
        rows.append((f"{w:,} x {h:,}", f"{w * h:,}", f"{raw:,}", f"{least:,}", f"{_b64_len(least):,}", f"{floats:,}",
                     f"{floats / least:,.1f}"))
    out = ["The pixel stream of an 8-bit RGB PNG with no interlacing is h x (1 + 3 w) bytes, one filter byte before every "
           "row. A deflate stream that encodes a run of equal bytes as matches of length 258 at distance 1 spends at least "
           "2 bits per match under a dynamic Huffman code (a 1-bit length code and a 1-bit distance code), so no stream is "
           f"shorter than the pixel stream divided by {DEFLATE_MAX_RATIO:,}, rounded up; the few bytes of headers and the "
           "first literal are left out. The last column is the decoded f32 bytes over that least compressed size.",
           "",
           _table(["image", "pixels", "pixel stream bytes", "least deflate bytes", "base64 characters", "bytes as f32 RGB",
                   "f32 bytes per compressed byte"], rows)]
    return "\n".join(out)


# ------------------------------------------------------------------ a budget checked before or after the work

def _decode_sequence(images, rule):
    """Pixels decoded for the declared sizes `images` (pixels per image) under `rule`: "after" decodes an image (if it is
    within the per-image limit) and then refuses the request when the total is over the request budget; "before" refuses
    an image from its header when it is over min(per-image limit, budget left)."""
    used = 0
    decoded = 0
    for k, px in enumerate(images):
        if k == MAX_IMAGES:
            return decoded, k, f"refused at image {k + 1}: more than {MAX_IMAGES} images"
        if rule == "after":
            if px > MAX_IMAGE_PIXELS:
                return decoded, k, f"refused at image {k + 1}: over the per-image limit"
            decoded += px
            used += px
            if used > MAX_REQUEST_PIXELS:
                return decoded, k + 1, f"refused after decoding image {k + 1}"
        else:
            left = MAX_REQUEST_PIXELS - used
            if px > min(MAX_IMAGE_PIXELS, left):
                return decoded, k, f"refused at the header of image {k + 1}"
            decoded += px
            used += px
    return decoded, len(images), "served"


PIXEL_SEQUENCES = [
    ("36,000,000 then 27,999,999 then 36,000,000", [36_000_000, 27_999_999, 36_000_000]),
    ("36,000,000 twice", [36_000_000, 36_000_000]),
    ("8 images of 8,000,000", [8_000_000] * 8),
    ("9 images of 1,000", [1_000] * 9),
]


def pixel_budget():
    rows = []
    for label, seq in PIXEL_SEQUENCES:
        for rule, name in [("after", "check the total after each decode"), ("before", "check each header against what is left")]:
            px, n, verdict = _decode_sequence(seq, rule)
            rows.append((label, name, f"{n}", f"{px:,}", f"{px * FLOAT_BYTES_PER_PIXEL:,}", verdict))
    out = [f"Per-image limit {MAX_IMAGE_PIXELS:,} pixels, request budget {MAX_REQUEST_PIXELS:,} pixels, at most "
           f"{MAX_IMAGES} images, {FLOAT_BYTES_PER_PIXEL} bytes per decoded pixel. Images are taken in the order listed. "
           "'Check after' decodes an image within the per-image limit, adds it and refuses the request when the total is "
           "over the budget; 'check before' reads each image's declared size from its header and refuses it, before "
           "decoding, when it is over the smaller of the per-image limit and the budget left. 'Decoded' counts the "
           "images whose pixels were decoded before the outcome.",
           "",
           _table(["images declared (pixels)", "rule", "decoded", "pixels decoded", "f32 bytes held", "outcome"], rows,
                  "llrrrl")]
    return "\n".join(out)


# ------------------------------------------------------------------ the ceilings of one surface

def surface_ceilings():
    per_request = MAX_REQUEST_PIXELS * FLOAT_BYTES_PER_PIXEL
    rows = [
        ("request body, chat, embeddings and image routes", f"{BODY_DIALECT:,} bytes",
         f"base64 media in it: at most {BODY_DIALECT * 3 // 4:,} bytes"),
        ("request body, `/v1/run`, `/v1/jobs`, `/videos`", f"{BODY_UPLOAD:,} bytes",
         f"base64 media in it: at most {BODY_UPLOAD * 3 // 4:,} bytes"),
        ("request body, `/v1/systemone`", f"{BODY_DECISIONS:,} bytes", ""),
        ("decoded image pixels, one request", f"{MAX_REQUEST_PIXELS:,} pixels", f"{per_request:,} bytes of f32"),
        ("media decodes at once, whole process", f"{MAX_CONCURRENT_DECODES}",
         f"{MAX_CONCURRENT_DECODES * per_request:,} bytes of f32 at most"),
        ("request head after connecting or between requests", f"{HEADER_DEADLINE_S} seconds", ""),
        ("whole request body", f"{BODY_DEADLINE_S} seconds",
         f"{BODY_UPLOAD:,} bytes in {BODY_DEADLINE_S} seconds is {BODY_UPLOAD * 8 // BODY_DEADLINE_S:,} bits per second"),
        ("connections held, one surface", f"{MAX_CONNECTIONS:,}",
         f"to keep all filled without a key: {MAX_CONNECTIONS / HEADER_DEADLINE_S:g} new connections per second"),
        ("requests in flight at the edge", f"{EDGE_CONCURRENCY}", "one more is answered 503 at once"),
        ("one input fd over D-Bus", f"{MAX_FD_BLOB:,} bytes", ""),
        ("input fds of one D-Bus call", f"{MAX_FDS_PER_CALL} fds, {MAX_FD_CALL:,} bytes together",
         f"{MAX_FDS_PER_CALL} x {MAX_FD_BLOB:,} = {MAX_FDS_PER_CALL * MAX_FD_BLOB:,}, so the total binds first"),
    ]
    out = ["The ceilings are brain's constants (named in the text); the derived column is arithmetic on them. Base64 "
           "carries 3 bytes in 4 characters, so a body of b bytes holds at most floor(3 b / 4) bytes of media; a bit rate is "
           "the bytes times 8 divided by the seconds, rounded down.",
           "",
           _table(["what is bounded", "ceiling", "what follows from it"], rows, "lrl")]
    return "\n".join(out)


# ------------------------------------------------------------------ counts a file declares

def _reserve(n, left, min_item):
    return min(n, left // max(min_item, 1))


def count_reservation():
    rows = []
    cases = [
        ("1,024-byte file declaring 2^40 tensors", 1024, 24, 2 ** 40, 24),
        ("1,024-byte file declaring 3 tensors", 1024, 24, 3, 24),
        ("1 MiB file declaring a metadata array of 2^62 items", MIB, 4096, 2 ** 62, 1),
    ]
    for label, length, pos, n, m in cases:
        left = length - pos
        naive = n * 24
        rows.append((label, f"{left:,}", f"{m}", f"{n:,}", f"{naive:,}", f"{_reserve(n, left, m):,}"))
    out = ["A reader reserves room for the n items a header declares before it reads them. 'Bytes left' is the file length "
           "minus the position where the declared items begin (byte 24 for the tensor records of a GGUF file with no metadata entries, after its 4-byte magic, "
           "4-byte version and two 8-byte counts; byte 4,096 for the array); every declared item takes at least the given number of bytes of the file (a "
           "tensor record of a GGUF file at least 24: an 8-byte name length, a 4-byte dimension count, a 4-byte type and "
           "an 8-byte offset; a metadata array item at least 1). The naive reservation is n records of 24 bytes; the "
           "bounded one is min(n, floor(bytes left / least bytes per item)) records.",
           "",
           _table(["header", "bytes left", "least bytes per item", "items declared", "naive reservation (bytes)",
                   "bounded reservation (items)"], rows)]
    depth_rows = []
    for length in [KIB, MIB, GIB]:
        depth_rows.append((f"{length:,}", f"{length // 12:,}", "8"))
    out += ["", "A nested metadata array costs the file 12 bytes per level (a 4-byte element type and an 8-byte count), so a "
            "file of L bytes can spell floor(L / 12) levels; a reader that recurses once per level needs that many stack "
            "frames unless it stops at a fixed depth.", "",
            _table(["file bytes", "levels the file can spell", "levels read before refusing"], depth_rows)]
    return "\n".join(out)


M64 = 1 << 64


def _walk(records, length, checked, steps):
    """Positions visited by a walk over length-prefixed records: at each position the record's declared size is added.
    `records` maps a position to its declared size. With `checked`, a sum past 2^64 - 1 or past the end of the file stops
    the walk with an error; otherwise the sum wraps modulo 2^64 and only 'past the end' is tested."""
    at, seen = 0, []
    for _ in range(steps):
        if at + 8 > length:
            return seen, "ends: the walk reached the end of the file"
        size = records[at]
        seen.append(at)
        nxt = at + size
        if checked:
            if nxt >= M64 or nxt > length:
                return seen, f"refused: the record at byte {at} overruns the file"
        else:
            nxt %= M64
            if nxt > length:
                return seen, f"refused: the record at byte {at} overruns the file"
        at = nxt
    return seen, f"still walking after {steps} records"


def wrapping_walk():
    length = 40
    records = {0: 24, 24: M64 - 24}
    rows = []
    for checked, name in [(False, "sum wraps modulo 2^64"), (True, "sum checked")]:
        seen, verdict = _walk(records, length, checked, 8)
        rows.append((name, ", ".join(str(s) for s in seen), verdict))
    out = [f"A {length}-byte file of two length-prefixed records: at byte 0 a record of 24 bytes, at byte 24 a record whose "
           f"64-bit size is 2^64 - 24 = {M64 - 24:,}. The walk starts at byte 0, records each position it visits and moves by "
           "the declared size; a record must lie inside the file. Eight steps are shown.",
           "",
           _table(["arithmetic", "positions visited", "outcome"], rows, "lll")]
    return "\n".join(out)


# ------------------------------------------------------------------ a tensor view a pickle declares

def _view(size, stride, offset, storage):
    numel = 1
    for d in size:
        numel *= d
        if numel >= M64:
            return None, None, "refused: the element count overflows 64 bits"
    if numel > storage:
        return numel, None, f"refused: {numel:,} elements over a storage of {storage:,}"
    if numel == 0:
        return 0, None, "accepted (empty)"
    last = offset + sum((d - 1) * s for d, s in zip(size, stride))
    if last >= storage:
        return numel, last, f"refused: reads element {last:,}, past the storage"
    return numel, last, "accepted"


VIEWS = [([2, 3], [3, 1], 0, 6), ([3], [2], 0, 5), ([4], [0], 0, 4), ([3], [3], 0, 6), ([2, 2], [2, 1], 2, 6),
         ([1 << 20, 1 << 20], [0, 0], 0, 1), ([1 << 32, 1 << 32, 0], [0, 0, 0], 0, 1)]


def stride_views():
    rows = []
    for size, stride, off, st in VIEWS:
        numel, last, verdict = _view(size, stride, off, st)
        rows.append((str(size), str(stride), f"{off}", f"{st:,}", "over 2^64" if numel is None else f"{numel:,}",
                     "-" if last is None else f"{last:,}",
                     verdict))
    out = ["A tensor of a pickled checkpoint is a view of a storage: a size, a stride per axis and an offset, all chosen by "
           "the file. The element count is the product of the sizes, formed axis by axis, and a partial product of 2^64 or "
           "more refuses the view; the last element read is offset + sum over the axes "
           "of (size - 1) x stride. A view is accepted when its count is at most the storage's elements and its last "
           "element is inside the storage; the count is checked first, so an overflowing or oversized view is refused "
           "before anything is allocated.",
           "",
           _table(["size", "stride", "offset", "storage elements", "elements declared", "last element read", "outcome"],
                  rows, "lllrrrl")]
    return "\n".join(out)


# ------------------------------------------------------------------ a pickle is a program

def _pickle_call(module, name, arg):
    """Protocol 2: PROTO 2, GLOBAL module name, the argument tuple, REDUCE, STOP. `arg` is None for an empty tuple."""
    b = b"\x80\x02" + b"c" + module.encode() + b"\n" + name.encode() + b"\n"
    if arg is None:
        b += b")"
    else:
        s = arg.encode()
        b += b"X" + struct.pack("<I", len(s)) + s + b"\x85"
    return b + b"R."


_OPS = {0x80: "PROTO", 0x63: "GLOBAL", 0x58: "BINUNICODE", 0x85: "TUPLE1", 0x29: "EMPTY_TUPLE", 0x52: "REDUCE",
        0x2E: "STOP"}

ALLOWED = {("collections", "OrderedDict"): "an empty ordered mapping",
           ("torch._utils", "_rebuild_tensor_v2"): "a tensor view (needs four arguments)",
           ("torch._utils", "_rebuild_tensor"): "a tensor view (needs four arguments)",
           ("torch._utils", "_rebuild_parameter"): "the tensor it wraps"}


def _disassemble(b):
    ops, i = [], 0
    while i < len(b):
        op = b[i]
        name = _OPS[op]
        if op == 0x80:
            ops.append((i, name, f"protocol {b[i + 1]}"))
            i += 2
        elif op == 0x63:
            j = b.index(b"\n", i + 1)
            k = b.index(b"\n", j + 1)
            ops.append((i, name, f"`{b[i + 1:j].decode()}` `{b[j + 1:k].decode()}`"))
            i = k + 1
        elif op == 0x58:
            n = struct.unpack("<I", b[i + 1:i + 5])[0]
            ops.append((i, name, f"{n} bytes: `{b[i + 5:i + 5 + n].decode()}`"))
            i += 5 + n
        else:
            ops.append((i, name, ""))
            i += 1
    return ops


def _allowlist_reader(b):
    """The reader of the chapter: a stack machine over the opcodes above that never imports or calls anything. GLOBAL
    pushes the pair (module, name); REDUCE applies an allowed pair as a data constructor and turns any other pair into an
    empty record."""
    stack = []
    for _, name, arg in _disassemble(b):
        if name == "GLOBAL":
            mod, nm = [s.strip("`") for s in arg.split(" ")]
            stack.append(("global", mod, nm))
        elif name == "BINUNICODE":
            stack.append(("text",))
        elif name == "EMPTY_TUPLE":
            stack.append(("tuple", 0))
        elif name == "TUPLE1":
            stack.pop()
            stack.append(("tuple", 1))
        elif name == "REDUCE":
            args = stack.pop()
            g = stack.pop()
            key = (g[1], g[2])
            if key == ("collections", "OrderedDict"):
                stack.append("an empty ordered mapping")
            elif key in ALLOWED:
                stack.append(f"refused: {g[2]} needs four arguments, got {args[1]}")
            else:
                stack.append("an empty record (nothing imported, nothing called)")
        elif name == "STOP":
            return stack.pop()
    return "no STOP"


def pickle_program():
    progs = [("builtins", "print", "hello"), ("collections", "OrderedDict", None)]
    out = ["Two pickles of protocol 2, byte offsets from 0. Opcodes: PROTO (0x80) and its version byte; GLOBAL (the byte "
           "`c`, then a module name and an attribute name, each ended by a newline byte); BINUNICODE (`X`, a 4-byte "
           "little-endian length, that many bytes of UTF-8); TUPLE1 (0x85, the top value becomes a 1-tuple); EMPTY_TUPLE "
           "(`)`); REDUCE (`R`, pop the arguments and the callable, push the result of the call); STOP (`.`, the value on "
           "top of the stack is the result)."]
    rows = []
    for mod, name, arg in progs:
        b = _pickle_call(mod, name, arg)
        out += ["", f"`{mod}.{name}` with {'the argument `' + arg + '`' if arg else 'no arguments'}: {len(b)} bytes, "
                    f"`{b.hex(' ')}`.", ""]
        out.append(_table(["offset", "opcode", "operand"], [(f"{o}", n, a) for o, n, a in _disassemble(b)], "rll"))
        full = (f"imports `{mod}` and calls `{name}`" + (f"('{arg}')" if arg else "()"))
        rows.append((f"`{mod}.{name}`", full, _allowlist_reader(b)))
    out += ["", _table(["global", "a general unpickler", "the allow-list reader"], rows, "lll")]
    return "\n".join(out)


# ------------------------------------------------------------------ who is calling

KEY_PREFIX = "sk-brain-"
KEY_HEX = 32


def key_guessing():
    total = len(KEY_PREFIX) + KEY_HEX
    rows = []
    for k in [0, 9, 20, 40, 41]:
        early = min(k + 1, total)
        rows.append((f"{k}", f"{early}", f"{total}"))
    worst = 16 * KEY_HEX
    expected = KEY_HEX * (16 + 1) / 2
    out = [f"A key is `{KEY_PREFIX}` followed by {KEY_HEX} hexadecimal digits ({4 * KEY_HEX} random bits), {total} "
           "characters in all. An early-exit comparison stops at the first character that differs, so it compares "
           "min(k + 1, length) characters of a guess whose first k characters are right; a constant-time comparison "
           "compares every character whatever the guess.",
           "",
           _table(["leading characters right", "compared, early exit", "compared, constant time"], rows),
           "",
           f"With a perfect timing oracle (the comparison count is observed exactly) the digits are found one at a time: "
           f"at most 16 guesses per digit, {worst} in all, and {expected:g} on average when the right digit is equally "
           f"likely to be any of the 16 tried in a fixed order. Without the oracle a guess is one of 16^{KEY_HEX} = 2^"
           f"{4 * KEY_HEX} = {16 ** KEY_HEX:,} keys, and {16 ** KEY_HEX // 2:,} guesses are needed on average."]
    return "\n".join(out)


def job_ids():
    live = 1024
    p = live / 2 ** 122
    rows = [("a counter from 1, no owner", "1 for every number below its own", "yes"),
            ("random, 122 bits (a version 4 UUID), one scope for every key-holder",
             f"{live:,} / 2^122 = {p:.3e} per guess", "yes"),
            ("random, 122 bits, looked up with the caller's scope", f"{p:.3e} per guess", "no: another scope's job is 'no such job'")]
    out = [f"A caller tries to name another caller's job. {live:,} jobs are live (the most one HTTP surface holds). The "
           "first column is how the server numbers jobs and finds them; the second is the chance that one guess names a "
           "live job; the third is whether naming it is enough to read or cancel it.",
           "",
           _table(["identifiers", "chance a guess names a live job", "a named job can be touched"], rows, "lll")]
    return "\n".join(out)


def caller_shares():
    callers = [("assistant", "inference", 16), ("batch", "inference", 4), ("operator", "admin", 2)]
    total = sum(m for _, _, m in callers)
    rows = [(n, r, f"{m}", f"{100 * m / EDGE_CONCURRENCY}") for n, r, m in callers]
    out = [f"Three named callers and their in-flight budgets. A call holds one slot of its caller's budget from the moment "
           f"it is let in until it is settled, a background job's whole run included. The edge admits {EDGE_CONCURRENCY} "
           f"requests at once across all callers. The share is the budget over {EDGE_CONCURRENCY} as a percentage, printed exactly.",
           "",
           _table(["caller", "role", "in-flight budget", "share of the edge (%)"], rows, "llrr"),
           "",
           f"The budgets add to {total}, below {EDGE_CONCURRENCY}, so the calls these budgets let run cannot fill the edge "
           "by themselves. A request over its caller's budget is refused with 429 at once, and holds an edge permit only for "
           "the moment that takes."]
    return "\n".join(out)


# ------------------------------------------------------------------ the wire

def _is_loopback(host):
    if ":" in host:
        return host == "::1"
    return host.split(".")[0] == "127"


def may_serve_plaintext(hosts, tls, flag):
    if tls or flag:
        return True
    return all(_is_loopback(h) for h in hosts)


def exposure():
    cases = [(["127.0.0.1"], False, False), (["127.0.0.1", "::1"], False, False), (["127.8.9.10"], False, False),
             (["0.0.0.0"], False, False), (["::"], False, False), (["::ffff:127.0.0.1"], False, False),
             (["127.0.0.1", "192.168.1.5"], False, False), (["192.168.1.5"], True, False),
             (["192.168.1.5"], False, True)]
    rows = []
    for hosts, tls, flag in cases:
        ok = may_serve_plaintext(hosts, tls, flag)
        rows.append((", ".join(f"`{h}`" for h in hosts), "yes" if tls else "no", "yes" if flag else "no",
                     "serves" if ok else "refused before anything loads"))
    out = ["An IPv4 (Internet Protocol version 4) address is loopback when its first byte is 127; an IPv6 (version 6) address is loopback only when it is `::1` (an "
           "IPv4 address written in IPv6 form, `::ffff:127.0.0.1`, is not). The surfaces are served when TLS is configured, "
           "or the operator states that something else encrypts the path, or every address is loopback.",
           "",
           _table(["addresses bound", "TLS", "plaintext stated", "outcome"], rows, "llll")]
    return "\n".join(out)
