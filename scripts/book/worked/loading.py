# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Loading Models".

Plain Python 3, standard library only, written from the chapter's own description. Each public function without a leading
underscore that takes no arguments returns a markdown string that the chapter embeds between
`<!-- worked: loading.<name> -->` markers. Helpers with a leading underscore, or that take arguments, are shared with the
chapter's figures so a picture and the table beside it cannot disagree.

Everything here is integer arithmetic on byte counts, closed-form sums and enumerations of small state machines, so another
implementation reproduces every printed digit. No generator is used.

Swedish Embedded AB implements solutions for loading multi-gigabyte model files quickly and within a fixed memory budget for
its clients. If your team needs expertise in model storage, memory-mapped loading and model distribution then you can procure
our services by sending an email to info@swedishembedded.com.
"""
import hashlib
import json

import number_formats as nf

_table = nf._table

KIB, MIB, GIB = 1 << 10, 1 << 20, 1 << 30
PAGE = 4096                      # bytes per page assumed by the page-counting example (the usual size on Linux for x86-64)


def _gib(b):
    return f"{b / GIB:.4g}"


def _gb(b):
    return f"{b / 1e9:.4g}"


def _lcm(a, b):
    from math import gcd
    return a // gcd(a, b) * b


def _ceil_div(a, b):
    return -(-a // b)


# ------------------------------------------------------------------ pages of a mapped file

WIDTH = {"F32": 4, "BF16": 2, "F16": 2, "U32": 4}

MODEL = [("embed.weight", "BF16", [32000, 4096]),
         ("layers.0.attn.q.weight", "BF16", [4096, 4096]),
         ("layers.0.attn.norm.weight", "F32", [4096]),
         ("layers.0.mlp.up.weight", "BF16", [11008, 4096])]


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def safetensors_layout(tensors, pad=8):
    """Header text and offsets of a safetensors file holding `tensors` (name, dtype, shape) back to back in the order given.
    The header is the compact JSON object `{name: {"dtype", "shape", "data_offsets": [start, end]}}`, padded with spaces to a
    multiple of `pad` bytes (pad 1: no padding). Returns (header length, {name: (absolute start, absolute end)}, file size);
    absolute offsets count from the first byte of the file, the data starting at 8 + header length."""
    header, pos = {}, 0
    for name, dtype, shape in tensors:
        n = _numel(shape) * WIDTH[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [pos, pos + n]}
        pos += n
    text = json.dumps(header, separators=(",", ":"))
    length = len(text) + (-len(text) % pad)
    base = 8 + length
    spans = {name: (base + h["data_offsets"][0], base + h["data_offsets"][1]) for name, h in header.items()}
    return length, spans, base + pos


def _pages(a, b):
    """The pages (first, last, count) that the byte range [a, b) of a file touches, with PAGE-byte pages from offset 0."""
    first, last = a // PAGE, (b - 1) // PAGE
    return first, last, last - first + 1


def pages_touched():
    length, spans, size = safetensors_layout(MODEL)
    total_pages = _ceil_div(size, PAGE)
    row = 8192                                            # bytes of one row of embed.weight: 4096 values of 2 bytes
    e0 = spans["embed.weight"][0]
    reads = [("open: the length prefix and the header", 0, 8 + length),
             ("embed.weight, rows 100 to 102 (3 rows)", e0 + 100 * row, e0 + 103 * row),
             ("layers.0.attn.norm.weight, whole", *spans["layers.0.attn.norm.weight"]),
             ("layers.0.attn.q.weight, whole", *spans["layers.0.attn.q.weight"]),
             ("every tensor, whole", 8 + length, size)]
    rows = []
    for label, a, b in reads:
        first, last, n = _pages(a, b)
        rows.append((label, f"{a:,}", f"{b:,}", f"{first:,} to {last:,}", f"{n:,}", f"{100 * n / total_pages:.4g}"))
    out = [f"The file holds {len(MODEL)} tensors, header text {length:,} bytes (padded to a multiple of 8), data starting at byte "
           f"{8 + length:,}, size {size:,} bytes = {total_pages:,} pages of {PAGE:,} bytes.", ""]
    for name, dtype, shape in MODEL:
        a, b = spans[name]
        out.append(f"- `{name}`: {dtype} {shape}, bytes {a:,} to {b:,}")
    out.append("")
    out.append(_table(["read", "first byte", "end byte (excluded)", "pages touched", "page count", "% of the file's pages"], rows))
    return "\n".join(out)


# ------------------------------------------------------------------ resident bytes of four ways to read a checkpoint

STRATEGIES = ("slurp", "map", "map + release", "populate + release")


def resident_during(strategy, n, s, d, i):
    """Resident bytes of the process while tensor i (0-based) of n equal tensors is being decoded, as (heap, file-backed pages).
    s: stored bytes of a tensor, d: bytes of its decoded f32 form. The decoded tensors of 0..i (tensor i being built) are heap
    in every strategy; S = n s is the whole file."""
    S = n * s
    heap = (i + 1) * d
    if strategy == "slurp":
        return heap + S, 0           # the file read into a buffer of its own: heap, whole, from the first byte
    if strategy == "map":
        return heap, (i + 1) * s     # a touched page stays resident
    if strategy == "map + release":
        return heap, s               # tensors 0..i-1 were released after decoding; tensor i's pages are resident
    if strategy == "populate + release":
        return heap, S - i * s       # every page was faulted in at open; tensors 0..i-1 were released
    raise ValueError(strategy)


def resident_peak(strategy, n, s, d):
    """(peak of heap + file-backed pages, heap at that peak, file-backed pages at that peak, number of tensors decoded by then); when
    several tensors reach the same peak, the last one is reported."""
    best = None
    for i in range(n):
        h, f = resident_during(strategy, n, s, d, i)
        if best is None or h + f >= best[0]:
            best = (h + f, h, f, i + 1)
    return best


CASE_F32 = dict(n=32, s=128 * MIB, d=128 * MIB)            # 32 tensors of 128 MiB stored as f32, decoded to f32
CASE_BF16 = dict(n=32, s=500_000_000, d=1_000_000_000)     # 32 tensors of 5e8 stored bytes as bf16 (16e9 bytes, 8e9 values)


def resident_bytes():
    rows = []
    for strategy in STRATEGIES:
        a = resident_peak(strategy, **CASE_F32)
        b = resident_peak(strategy, **CASE_BF16)
        rows.append((strategy, _gib(a[0]), _gib(a[1]), _gib(a[2]), _gb(b[0]), _gb(b[1]), _gb(b[2])))
    return _table(["strategy", "f32: peak (GiB)", "heap", "file-backed", "bf16: peak (GB)", "heap", "file-backed"], rows)


def resident_series(strategy):
    """Resident GiB while each of the 32 tensors of the f32 case is decoded (the figure's data)."""
    return [sum(resident_during(strategy, CASE_F32["n"], CASE_F32["s"], CASE_F32["d"], i)) / GIB for i in range(CASE_F32["n"])]


# ------------------------------------------------------------------ alignment and ranges

ALIGN_TENSORS = [("a.weight", "F32", [3, 5]), ("b.bias", "BF16", [3]), ("c.weight", "F32", [2, 2]), ("d.scale", "U32", [8])]


def zero_copy():
    rows = []
    for pad in (8, 1):
        length, spans, size = safetensors_layout(ALIGN_TENSORS, pad)
        for name, dtype, shape in ALIGN_TENSORS:
            a, b = spans[name]
            can = dtype in ("F32", "U32") and a % 4 == 0
            rows.append((f"pad {pad}" if name == ALIGN_TENSORS[0][0] else "", f"{length}" if name == ALIGN_TENSORS[0][0] else "",
                         f"`{name}`", f"{dtype} {shape}", f"{a}", f"{a % 4}", "yes" if can else "no"))
    t1 = _table(["header padding", "header bytes", "tensor", "type and shape", "first byte in the file", "mod 4", "lent as words"], rows,
                "lrlrrrr")
    chunk = 1 << 20                                    # words per chunk of the device upload (4 MiB of f32)
    big = [("11,008 x 4,096", 11008 * 4096), ("32,000 x 4,096", 32000 * 4096), ("151,936 x 4,096", 151936 * 4096)]
    rows2 = []
    for name, n in big:
        c = _ceil_div(n, chunk)
        rows2.append((name, f"{n:,}", f"{4 * n:,}", f"{c:,}", f"{4 * chunk:,}", f"{n - (c - 1) * chunk:,}"))
    t2 = _table(["tensor of shape (decoded to f32)", "elements", "bytes if decoded whole", "chunks of 1,048,576 elements",
                 "bytes of the one scratch buffer", "elements of the last chunk"], rows2)
    return t1 + "\n\n" + t2


# ------------------------------------------------------------------ the descriptor of a packed tensor

def _ggml(elems, nbytes, name):
    return dict(kind="ggml", elems=elems, bytes=nbytes, name=name, scales=[])


def _plain(bits, name, scales=()):
    return dict(kind="plain", bits=bits, name=name, scales=list(scales))


ENC = {
    "bf16": _plain(16, "bf16"),
    "fp8 tiles": _plain(8, "e4m3, one bf16 scale per 128 x 128 tile", [("tile", 128, 128, 16)]),
    "int8 block 32": _plain(8, "int8, one f32 scale per 32", [("block", 32, 32)]),
    "nf4 block 32": _plain(4, "4-bit codebook, one f32 scale per 32", [("block", 32, 32)]),
    "nvfp4": _plain(4, "e2m1, e4m3 per 16, f32 per tensor", [("tensor", 0, 32), ("block", 16, 8)]),
    "mxfp4": _plain(4, "e2m1, e8m0 per 32", [("block", 32, 8)]),
    "Q8_0": _ggml(32, 34, "ggml Q8_0"),
    "Q4_0": _ggml(32, 18, "ggml Q4_0"),
    "Q4_K": _ggml(256, 144, "ggml Q4_K"),
    "Q6_K": _ggml(256, 210, "ggml Q6_K"),
}
# scales: ("tensor", _, bits) | ("block", n, bits) | ("row", _, bits) | ("tile", rows, cols, bits)


def _geometry(e):
    """(elements, bytes) of the smallest independently addressable unit of the payload."""
    if e["kind"] == "ggml":
        return e["elems"], e["bytes"]
    if e["bits"] < 8:
        return 8 // e["bits"], 1
    return 1, e["bits"] // 8


def _scale_count(e, level, shape):
    s, n = e["scales"][level], _numel(shape)
    inner = shape[-1]
    if s[0] == "tensor":
        return 1
    if s[0] == "block":
        return n // s[1]
    if s[0] == "row":
        return n // inner
    return _ceil_div(shape[0], s[1]) * _ceil_div(shape[1], s[2])


def _scale_bits(e, level):
    return e["scales"][level][-1]


def tile_quantum(e, shape):
    q = _geometry(e)[0]
    for s in e["scales"]:
        if s[0] == "block":
            q = _lcm(q, s[1])
        elif s[0] in ("row", "tile"):
            q = _lcm(q, shape[-1])
    return q


def encode_quantum(e, shape):
    q = tile_quantum(e, shape)
    for s in e["scales"]:
        if s[0] == "tile":
            q = _lcm(q, s[1] * shape[-1])
    return q


def payload_bytes(e, shape):
    ge, gb = _geometry(e)
    return _ceil_div(_numel(shape), ge) * gb


def scale_bytes(e, level, shape):
    return _scale_count(e, level, shape) * _scale_bits(e, level) // 8


def bits_per_weight(e, shape):
    total = payload_bytes(e, shape) + sum(scale_bytes(e, k, shape) for k in range(len(e["scales"])))
    return 8 * total / _numel(shape)


SHAPE = [4096, 4096]


def encodings():
    rows = []
    for name, e in ENC.items():
        pay = payload_bytes(e, SHAPE)
        sc = [scale_bytes(e, k, SHAPE) for k in range(len(e["scales"]))]
        total = pay + sum(sc)
        rows.append((name, e["name"] if e["kind"] == "plain" else "own interleaved scales", f"{pay:,}", " + ".join(f"{x:,}" for x in sc) or "0",
                     f"{total:,}", f"{8 * total / _numel(SHAPE):.7g}", f"{tile_quantum(e, SHAPE):,}", f"{encode_quantum(e, SHAPE):,}"))
    return (f"The tensor is 4096 x 4096 = {_numel(SHAPE):,} elements, row-major, innermost axis 4096.\n\n" +
            _table(["encoding", "codes and scales", "payload bytes", "scale bytes per level", "total bytes", "bits per element",
                    "tile quantum", "encode quantum"], rows, "llrrrrrr"))


def _payload_range(e, a, b):
    ge, gb = _geometry(e)
    return a // ge * gb, _ceil_div(b, ge) * gb


def _scale_range(e, level, shape, a, b):
    s = e["scales"][level]
    inner = shape[-1]
    if s[0] == "tensor":
        return 0, 1
    if s[0] == "block":
        return a // s[1], _ceil_div(b, s[1])
    if s[0] == "row":
        return a // inner, _ceil_div(b, inner)
    per_band = _ceil_div(shape[1], s[2])
    r0, r1 = a // inner, _ceil_div(b, inner)
    return (r0 // s[1]) * per_band, ((r1 - 1) // s[1] + 1) * per_band


TILES = [("nvfp4", [8, 64], 16, 48), ("Q4_K", [4, 512], 256, 512), ("int8 block 32", [4, 64], 32, 96),
         ("fp8 tiles", [256, 256], 0, 256 * 128)]


def tile_ranges():
    rows = []
    for name, shape, a, b in TILES:
        e = ENC[name]
        q = tile_quantum(e, shape)
        assert a % q == 0 and (b % q == 0 or b == _numel(shape))
        pr = _payload_range(e, a, b)
        sr = [_scale_range(e, k, shape, a, b) for k in range(len(e["scales"]))]
        rows.append((name, str(shape), f"{a} to {b}", f"{q}", f"{pr[0]} to {pr[1]}", "; ".join(f"level {k}: {x} to {y}" for k, (x, y) in enumerate(sr)) or "none"))
    return _table(["encoding", "shape", "tile (flat elements)", "tile quantum", "payload bytes", "scale elements (half-open)"], rows, "llrrrl")


# ------------------------------------------------------------------ the tiled load plan

def _tile_bytes(e, elems):
    pay = _payload_range(e, 0, elems)
    total = pay[1] - pay[0]
    for k, s in enumerate(e["scales"]):
        a, b = _scale_range(e, k, [elems, elems], 0, elems) if s[0] != "tile" else (0, 0)
        total += (b - a) * _scale_bits(e, k) // 8
    return total


def staging_bytes(src, dst, op, elems):
    """Bytes staged for one tile of `elems` elements: Preserve, the tile as stored; Repack, the stored tile and the repacked tile;
    Convert, the stored tile, the decoded f32 values (4 bytes each) and twice the encoded tile (the packers build a word buffer
    before the bytes)."""
    if op == "Preserve":
        return _tile_bytes(src, elems)
    if op == "Repack":
        return _tile_bytes(src, elems) + _tile_bytes(dst, elems)
    return _tile_bytes(src, elems) + 4 * elems + 2 * _tile_bytes(dst, elems)


def plan_tiles(shape, src_name, dst_name, op, tile_cap, capacity):
    """(tile elements, tiles, staging bytes per tile) or the refusal text. The tile starts at the quantum and grows by one quantum at
    a time while it stays inside the tensor and its staging bytes stay within min(tile_cap, capacity)."""
    src, dst = ENC[src_name], ENC[dst_name]
    if op == "Preserve":
        dst = src
    quantum = tile_quantum(src, shape)
    if op != "Preserve":
        quantum = _lcm(quantum, encode_quantum(dst, shape))
    cap = min(tile_cap, capacity)
    n = _numel(shape)
    if staging_bytes(src, dst, op, quantum) > cap:
        return f"refused: the smallest tile ({quantum} elements, {staging_bytes(src, dst, op, quantum)} bytes) exceeds {cap} bytes"
    tile = quantum
    while tile + quantum <= n and staging_bytes(src, dst, op, tile + quantum) <= cap:
        tile += quantum
    tile = min(tile, n)
    return tile, _ceil_div(n, tile), staging_bytes(src, dst, op, tile)


def load_plan():
    cases = [("Q8_0 to Q8_0", "Preserve", [4096, 4096], "Q8_0", "Q8_0", 4 * MIB, 1 << 40),
             ("Q8_0 to int8 block 32", "Repack", [4096, 4096], "Q8_0", "int8 block 32", 4 * MIB, 1 << 40),
             ("int8 block 32 to nf4 block 32", "Convert", [4096, 4096], "int8 block 32", "nf4 block 32", 4 * MIB, 1 << 40),
             ("bf16 to nf4 block 32 (256 x 32768)", "Convert", [256, 32768], "bf16", "nf4 block 32", 128 * KIB, 512 * KIB),
             ("bf16 to int8 block 32, 64-byte budget", "Convert", [4, 4096], "bf16", "int8 block 32", 1 * MIB, 64)]
    rows = []
    for label, op, shape, a, b, cap, capacity in cases:
        r = plan_tiles(shape, a, b, op, cap, capacity)
        if isinstance(r, str):
            rows.append((label, op, f"{_numel(shape):,}", f"{cap:,}", f"{capacity:,}" if capacity < 1 << 40 else "unbounded", r, "", ""))
        else:
            rows.append((label, op, f"{_numel(shape):,}", f"{cap:,}", f"{capacity:,}" if capacity < 1 << 40 else "unbounded",
                         f"{r[0]:,}", f"{r[1]:,}", f"{r[2]:,}"))
    t1 = _table(["load", "step", "elements", "tile byte cap", "budget capacity (bytes)", "elements per tile", "tiles", "staging bytes per tile"],
                rows, "llrrrrrr")
    rows2 = []
    for k in (22, 26, 30):
        n = 1 << k
        shape = [n // 4096, 4096]
        r = plan_tiles(shape, "bf16", "nf4 block 32", "Convert", 4 * MIB, 1 << 40)
        rows2.append((f"2^{k}", f"{n:,}", f"{2 * n:,}", f"{r[0]:,}", f"{r[1]:,}", f"{r[2]:,}"))
    t2 = _table(["elements", "count", "bytes stored as bf16", "elements per tile", "tiles", "staging bytes per tile"], rows2)
    return t1 + "\n\n" + t2


# ------------------------------------------------------------------ one lease or two

def lease_schedules():
    capacity = 100

    def run(steps_for_load):
        # state: positions, bytes held by each load; free = capacity - sum(held)
        def go(pos, held):
            used = held[0] + held[1]
            moves = []
            for who in (0, 1):
                if pos[who] < len(steps_for_load):
                    st = steps_for_load[pos[who]]
                    if st[0] == "acquire" and used + st[1] > capacity:
                        continue
                    moves.append(who)
            if not moves:
                done = all(p == len(steps_for_load) for p in pos)
                return (1, 1, 0) if done else (1, 0, 1)
            tot = ok = stuck = 0
            for who in moves:
                st = steps_for_load[pos[who]]
                p2, h2 = list(pos), list(held)
                p2[who] += 1
                h2[who] = h2[who] + st[1] if st[0] == "acquire" else 0
                t, o, s = go(p2, h2)
                tot, ok, stuck = tot + t, ok + o, stuck + s
            return tot, ok, stuck
        return go([0, 0], [0, 0])

    per_buffer = [("acquire", 40), ("acquire", 40), ("release",)]
    one_lease = [("acquire", 80), ("release",)]
    a, b = run(per_buffer), run(one_lease)
    rows = [("two leases of 40 bytes, taken one after the other", *[f"{x}" for x in a]),
            ("one lease of 80 bytes", *[f"{x}" for x in b])]
    return _table(["each of the two loads asks for", "interleavings", "both loads finish", "stuck with both waiting"], rows)


# ------------------------------------------------------------------ the model reference grammar

QUANTS = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_K"]
RESERVED = ("brain", "local", "test")


def _strip_quant(repo):
    """(base, token) when repo ends in "-" + one token and a non-empty base remains, longest token tried first; else None."""
    for tok in sorted(QUANTS, key=lambda t: -len(t)):
        if repo.endswith(tok):
            base = repo[:-len(tok)]
            if base.endswith("-") and len(base) > 1:
                return base[:-1], tok
    return None


def parse_ref(s):
    """The reference grammar. Returns (vendor, repo, quant, adapter) with adapter None or (owner, name, tag), or the name of the
    refusal as a string."""
    if s == "" or "/" not in s:
        return "no vendor"
    vendor, rest = s.split("/", 1)
    if vendor == "" or rest == "":
        return "empty segment"
    if "/" in rest:
        return "too many segments"
    if any(c.isspace() for c in vendor + rest):
        return "bad character"
    adapter = None
    if "+" in rest:
        repo_part, plus = rest.split("+", 1)
        if ":" in repo_part or "+" in plus:
            return "bad adapter"
        name, tag = plus.split(":", 1) if ":" in plus else (plus, "latest")
        if name == "" or tag == "" or ":" in tag:
            return "bad adapter"
        if name in (".", "..") or tag in (".", ".."):
            return "path traversal"
        adapter, rest = ("local", name, tag), repo_part
    elif ":" in rest:
        repo_part, adapter_part = rest.split(":", 1)
        segs = adapter_part.split(":")
        if len(segs) != 3 or "" in segs:
            return "bad adapter"
        if any(x in (".", "..") for x in segs):
            return "path traversal"
        adapter, rest = tuple(segs), repo_part
    if rest == "":
        return "empty segment"
    quant = None
    cut = _strip_quant(rest)
    if cut:
        if _strip_quant(cut[0]):
            return "two quantizations"
        rest, quant = cut
    if vendor in (".", "..") or rest in (".", ".."):
        return "path traversal"
    return vendor, rest, quant, adapter


def _show_ref(r):
    if isinstance(r, str):
        return f"refused: {r}"
    vendor, repo, quant, adapter = r
    text = f"{vendor}/{repo}" + (f"-{quant}" if quant else "") + (f":{':'.join(adapter)}" if adapter else "")
    return f"vendor `{vendor}`, repo `{repo}`, quant {quant or 'none'}, adapter {':'.join(adapter) if adapter else 'none'}; prints as `{text}`"


REFS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-Q8_0", "qwen/qwen3-0.6b", "Qwen/Qwen3-0.6B-q8_0",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-Q8_0", "Qwen/Qwen3-0.6B-Q8_0-Q4_0",
        "Qwen/Qwen3-8B+maren", "Qwen/Qwen3-8B-Q4_K_M+maren:v2", "Qwen/Qwen3-8B:swedishembedded-com:generic-sft:latest",
        "Qwen/Qwen3-8B:local:maren", "Qwen/Qwen3-8B+a:b:c", "Qwen/Qwen3-8B:local:..:latest",
        "Qwen", "/Qwen3", "Qwen/Qwen3/x", "../Qwen3", "Qwen/..", "Qwen/Qwen3 8B", "Q4_K_M/-Q4_K_M", "Qwen/-Q8_0"]


def reference_grammar():
    rows = [(f"`{s}`", _show_ref(parse_ref(s))) for s in REFS]
    return _table(["text", "result"], rows, "ll")


def reserved_vendors():
    rows = [(f"`{s}`", "never fetched" if parse_ref(s)[0] in RESERVED else "may be fetched") for s in ("brain/demo", "local/qwen", "test/x", "Qwen/Qwen3-0.6B")]
    return _table(["reference", "network"], rows, "ll")


# ------------------------------------------------------------------ where models live

def models_dir(published=None, env=None, flag=None):
    """The models directory: a non-empty flag; else the published data root plus /models; else BRAIN_MODELS_DIR; else
    XDG_DATA_HOME/brain/models; else HOME/.local/share/brain/models; else None. A variable that is empty counts as unset."""
    env = env or {}
    if flag:
        return flag
    if published:
        return published.rstrip("/") + "/models"
    if env.get("BRAIN_MODELS_DIR"):
        return env["BRAIN_MODELS_DIR"]
    if env.get("XDG_DATA_HOME"):
        return env["XDG_DATA_HOME"] + "/brain/models"
    if env.get("HOME"):
        return env["HOME"] + "/.local/share/brain/models"
    return None


LADDER = [("nothing set", None, {}, None),
          ("HOME only", None, {"HOME": "<home>"}, None),
          ("XDG_DATA_HOME and HOME", None, {"XDG_DATA_HOME": "<xdg>", "HOME": "<home>"}, None),
          ("BRAIN_MODELS_DIR, XDG_DATA_HOME and HOME", None, {"BRAIN_MODELS_DIR": "<m>", "XDG_DATA_HOME": "<xdg>", "HOME": "<home>"}, None),
          ("an empty BRAIN_MODELS_DIR and HOME", None, {"BRAIN_MODELS_DIR": "", "HOME": "<home>"}, None),
          ("a published data root `<data>` and BRAIN_MODELS_DIR", "<data>", {"BRAIN_MODELS_DIR": "<m>"}, None),
          ("the flag `<flag>`, a data root and BRAIN_MODELS_DIR", "<data>", {"BRAIN_MODELS_DIR": "<m>"}, "<flag>")]


def store_directories():
    rows = []
    for label, published, env, flag in LADDER:
        r = models_dir(published, env, flag)
        rows.append((label, f"`{r}`" if r else "none: no store"))
    ref = parse_ref("Qwen/Qwen3-0.6B-Q8_0")
    layout = [f"`<models>/{ref[0]}/{ref[1]}/`  holds the base, its tokenizer and its `{ref[2]}` quantization (`{ref[2]}.gguf`)",
              "`<models>/Qwen/Qwen3-8B/adapters/local/maren/latest/adapter.brain.safetensors`  holds the adapter of `Qwen/Qwen3-8B+maren`"]
    return _table(["what is set", "the models directory"], rows, "ll") + "\n\n" + "\n".join(f"- {x}" for x in layout)


def quant_candidates():
    ref = parse_ref("Qwen/Qwen3-0.6B-Q8_0")
    v, repo, q = ref[0], ref[1], ref[2]
    rows = [("1", f"{v}/{repo}-GGUF", f"{repo}-{q}.gguf"), ("2", f"{v}/{repo}", f"{repo}-{q}.gguf"), ("3", f"{v}/{repo}-{q}", f"{repo}-{q}.gguf")]
    return _table(["order", "repo listed", "file looked for"], [(a, f"`{b}`", f"`{c}`") for a, b, c in rows], "rll")


# ------------------------------------------------------------------ a file that arrives whole or not at all

DISK_RESERVE = GIB


def room_for(total, free):
    return total + DISK_RESERVE <= free


def _stream_to_file(fs, dest, data, total, expected, die_after=None):
    """The download of `data` to `dest` in the file system `fs` (a dict from path to bytes): chunks of 64 KiB (here the data is
    shorter) are written to dest + ".part-1"; more bytes than `total` declared, or a digest that is not `expected`, removes the
    part file and fails; a process killed after `die_after` bytes leaves the part file; success renames it onto dest."""
    tmp = dest + ".part-1"
    fs[tmp] = b""
    got = 0
    while got < len(data):
        chunk = data[got:got + 65536]
        if die_after is not None and got + len(chunk) > die_after:
            fs[tmp] = data[:die_after]
            return "killed"
        got += len(chunk)
        if total is not None and got > total:
            del fs[tmp]
            return "error: runs past its declared length"
        fs[tmp] += chunk
    if expected is not None and hashlib.sha256(fs[tmp]).hexdigest() != expected:
        del fs[tmp]
        return "error: digest differs"
    fs[dest] = fs.pop(tmp)
    return "ok"


def atomic_download():
    abc = hashlib.sha256(b"abc").hexdigest()
    cases = [("10 bytes arrive, 4 declared", b"0123456789", 4, None, None),
             ("b\"abc\", digest 00", b"abc", 3, "00", None),
             ("b\"abc\", 3 declared, digest of b\"abc\"", b"abc", 3, abc, None),
             ("b\"abcdef\", killed after 2 bytes", b"abcdef", 6, None, 2)]
    rows = []
    for label, data, total, expected, die in cases:
        fs = {}
        r = _stream_to_file(fs, "m/model.safetensors", data, total, expected, die)
        rows.append((label, r, "yes" if "m/model.safetensors" in fs else "no", "yes" if any(k.endswith(".part-1") for k in fs) else "no"))
    t1 = _table(["transfer", "outcome", "`model.safetensors` exists", "part file left"], rows, "llll")
    rows2 = [(f"{t:,}", f"{f:,}", "yes" if room_for(t, f) else "no") for t, f in
             ((8 * GIB, 9 * GIB), (8 * GIB, 9 * GIB - 1), (8 * GIB, 100 * GIB), (100 * GIB, 99 * GIB))]
    t2 = _table(["download (bytes)", "free on the file system (bytes)", "starts"], rows2)
    return (f"The SHA-256 digest of `abc` is the 64 hexadecimal digits `{abc[:32]}` followed by `{abc[32:]}`.\n\n" + t1 + "\n\n" + t2)


def resume_by_file():
    shards = [4 * GIB] * 6
    killed_shard, got = 3, 3 * GIB // 2              # the fourth shard (index 3) is killed after 1.5 GiB
    first = sum(shards[:killed_shard]) + got         # what the first attempt moved before it died
    final = sum(shards)                              # what a complete fetch must hold
    rows = []
    retry_skip = sum(shards[killed_shard:])          # the files that do not exist yet
    rows.append(("six shards of 4 GiB, killed in the fourth after 1.5 GiB", "skip a file that exists", first, retry_skip, final))
    rows.append(("same kill", "download everything again", first, final, final))
    one = 9 * GIB
    rows.append(("one file of 9 GiB, killed after 4.5 GiB", "skip a file that exists", one // 2, one, one))
    out = [(a, b, _gib(c), _gib(d), _gib(c + d), _gib(c + d - f)) for a, b, c, d, f in rows]
    return _table(["interrupted fetch", "retry rule", "first attempt (GiB)", "retry (GiB)", "both attempts (GiB)", "moved twice (GiB)"], out, "llrrrr")


ALLOWED = ("huggingface.co", "hf.co")


def host_allowed(host, override=None):
    suffixes = list(ALLOWED) + ([override] if override else [])
    return any(host == d or host.endswith("." + d) for d in suffixes)


def hosts():
    cases = [("huggingface.co", None), ("cdn-lfs.huggingface.co", None), ("us.aws.cdn.hf.co", None), ("evilhuggingface.co", None),
             ("huggingface.co.evil.example", None), ("hf-mirror.example", None), ("hf-mirror.example", "hf-mirror.example")]
    rows = [(f"`{h}`", f"`{o}`" if o else "none", "yes" if host_allowed(h, o) else "no") for h, o in cases]
    return _table(["host a redirect names", "mirror named by the operator", "followed"], rows, "llr")


# ------------------------------------------------------------------ the three download policies

def download_policy():
    rows = []
    for policy in ("Offline", "IfMissing", "AlwaysCheck"):
        for local in (True, False):
            if policy == "Offline":
                act = "use the local copy" if local else "stop with an error that names `brain pull`"
                hub = "no"
            elif local:
                act, hub = "use the local copy", "no"
            else:
                act, hub = "list, download, convert, then use it", "yes"
            rows.append((policy, "yes" if local else "no", hub, act))
    return _table(["policy", "reference already in the store", "hub contacted", "result"], rows, "llll")


# ------------------------------------------------------------------ the inputs and rules printed above each table
# Each worked function below returns its text and table; the sentence that states the inputs and the counting rules of a table is
# kept here, in one place, and prepended to the table so that the page cannot state different inputs from the ones computed.

_INTRO = {
    "pages_touched": (
        "The page size is 4,096 bytes, an assumption of the example (the usual size on Linux for x86-64; a system reports its own). "
        "Pages are numbered from 0 at the first byte of the file, and the byte range [a, b) touches pages a // 4096 through "
        "(b - 1) // 4096. No data is created: only the arithmetic of the layout is computed. The header is the compact JSON object "
        "with one entry per tensor in the order listed, each `{\"dtype\", \"shape\", \"data_offsets\": [start, end]}`, with offsets "
        "counted from the first byte after the header, padded with spaces to a multiple of 8 bytes. A row of `embed.weight` is "
        "4,096 x 2 = 8,192 bytes."),
    "resident_bytes": (
        "The resident bytes of a process while the tensors of a checkpoint are decoded to 32-bit floats one after another. Heap is "
        "memory the process owns: the decoded tensors, and in the first strategy a buffer that holds the whole file. File-backed is "
        "the pages of the mapped file that are resident. While tensor i (counted from 0) is decoded, the decoded tensors 0 to i are "
        "heap in every strategy. slurp: the file was read into a buffer first, so a heap copy of the whole file exists throughout. "
        "map: the file is mapped and every page that was touched stays resident, tensors 0 to i. map + release: a tensor's pages are "
        "dropped after it is decoded, so only tensor i's are resident. populate + release: every page was loaded when the mapping "
        "was made, and tensors 0 to i - 1 have been dropped, so the pages of tensors i to 31 remain. Two cases of 32 equal tensors: "
        "f32, 134,217,728 bytes stored (as f32) and 134,217,728 decoded each, 4 GiB in all; bf16, 500,000,000 bytes stored and "
        "1,000,000,000 decoded each, 16 x 10^9 bytes of file. The peak is the largest value of heap plus file-backed over the 32 "
        "tensors (the last tensor that reaches it is reported); GiB = 2^30 bytes, GB = 10^9 bytes, four significant digits."),
    "zero_copy": (
        "Four tensors in this order, `a.weight` F32 [3, 5] (60 bytes), `b.bias` BF16 [3] (6 bytes), `c.weight` F32 [2, 2] (16 bytes) and "
        "`d.scale` U32 [8] (32 bytes), in a file whose header is a compact JSON object of the form of the first example, padded with spaces "
        "to a multiple of 8 bytes (pad 8) or not padded (pad 1). Offsets count from the first byte of the file; a mapping begins on a "
        "page boundary, so a byte's alignment in memory is its offset in the file mod 4. A tensor is lent as 32-bit words when its "
        "type is F32 or U32 and its first byte is a multiple of 4. The second table is the chunking of the upload: chunks of "
        "1,048,576 elements, each decoded into one scratch buffer of 4 bytes per element that is reused; the last chunk holds the "
        "remainder. The first two shapes are those of the first example; the third is a hypothetical vocabulary table."),
    "encodings": (
        "Counts for a tensor of 4096 x 4096 elements. The payload is ceil(elements / elements per block) blocks of the block's bytes "
        "(an element of 4 bits: 2 elements per byte). A scale level adds its count (1 per tensor; elements / n for a scale per n "
        "elements; ceil(4096 / 128)^2 = 1,024 for 128 x 128 tiles) times its width in bytes. Bits per element = 8 x total bytes / "
        "elements. The tile quantum is the least number of elements that is a multiple of the payload block and of every scale "
        "block (a scale per row or per tile counts the row length 4096); the encode quantum adds whole bands of tile rows."),
    "tile_ranges": (
        "A tile is a range [a, b) of the flat, row-major elements of a tensor, a multiple of the tile quantum (the last tile may end at "
        "the end of the tensor). Its payload bytes are [a // g x B, ceil(b / g) x B) for a payload block of g elements and B bytes; "
        "a scale per n elements covers scale elements [a // n, ceil(b / n)); a per-tensor scale is element 0 only; a scale per "
        "128 x 128 tile covers the bands of 128 rows that the tile's rows touch, ceil(256 / 128) = 2 scales per band here."),
    "load_plan": (
        "The plan of one tensor. The tile starts at the quantum (the step's tile quantum: the least common multiple of the quanta of "
        "the stored and the target encoding) and grows by one quantum at a time while it stays inside the tensor and its staging "
        "bytes are at most the cap, the smaller of the tile byte cap and the budget capacity; a tensor whose first tile exceeds the "
        "cap is refused. Staging bytes of a tile of e elements: Preserve, the stored tile; Repack, the stored tile plus the repacked "
        "tile; Convert, the stored tile, 4 e bytes of decoded f32 values and twice the encoded tile. The tile of a stored or target "
        "encoding is its payload bytes plus its scale bytes for e elements counted from the start of a tensor. Encodings: Q8_0 is 32 "
        "elements per 34 bytes; int8 block 32 is 1 byte per element plus 4 bytes per 32; nf4 block 32 is 4 bits per element plus "
        "4 bytes per 32; bf16 is 2 bytes per element. The first table has a cap of 4 MiB; the second repeats the bf16 to nf4 "
        "load for tensors of 2^22, 2^26 and 2^30 elements."),
    "lease_schedules": (
        "Two loads share a budget of 100 bytes. A load that needs 80 bytes in all either asks for them in two leases of 40 bytes, "
        "one after the other, and gives both back when it is done, or asks for one lease of 80. A move is one step by one load; a "
        "request for bytes that are not free is not a move (the load waits). An interleaving is a maximal sequence of moves, counted "
        "by exhaustive search; two sequences that differ in the order of the moves are two interleavings."),
    "reference_grammar": (
        "The fifteen quantization tokens are `Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K "
        "Q8_K`, uppercase only; `BF16`, `F16`, `F32` and `FP8` are not tokens. The checks run in the order the text gives, and "
        "the first that fails names the refusal."),
    "reserved_vendors": "The vendors `brain`, `local` and `test` are reserved.",
    "store_directories": (
        "The models directory is the first of: a non-empty flag; the data root published by a flag, followed by `/models`; the "
        "non-empty variable BRAIN_MODELS_DIR; XDG_DATA_HOME followed by `/brain/models`; HOME followed by "
        "`/.local/share/brain/models`. A variable that is set to the empty text counts as unset. Angle brackets stand for a directory "
        "the machine names, `<home>` for the value of HOME."),
    "quant_candidates": (
        "For a reference with a quantization, repos are listed in this order and the first that holds the file is downloaded "
        "(`<repo>` is the repo name without the suffix, `<Q>` the token)."),
    "atomic_download": (
        "A download writes its bytes to a part file beside the destination and renames the part file onto the destination only when "
        "every check has passed. The first three rows are the three outcomes of the transfer of `model.safetensors` through an "
        "in-memory file system; in the fourth the process is killed after 2 bytes, so nothing runs that could remove the part "
        "file. The second table asks whether a download of the first size may start when the file system has the second size "
        "free: it needs the download plus a reserve of 2^30 bytes."),
    "resume_by_file": (
        "A repository of six shards of 4 GiB each, 24 GiB in all, is fetched file by file, and the process is killed in the fourth "
        "shard after 1.5 GiB of it was written. The part file is not continued by a retry (the retry opens it afresh), and a file "
        "that exists is complete. GiB = 2^30 bytes. 'Moved twice' is the bytes of both attempts minus the bytes of the complete "
        "repository."),
    "hosts": (
        "A redirect is followed only to a host that is a listed domain or a subdomain of one (the host equals the domain, or ends "
        "in a dot and the domain). The listed domains are `huggingface.co` and `hf.co`, plus the one domain an operator may name "
        "for a mirror."),
    "download_policy": (
        "What the three policies do for a default checkpoint, from the order of their checks. The last column assumes the hub "
        "answers."),
}


def _install_intros():
    import functools
    g = globals()
    for name, text in _INTRO.items():
        fn = g[name]

        def make(fn, text):
            @functools.wraps(fn)
            def wrapped():
                return text + "\n\n" + fn()
            return wrapped
        g[name] = make(fn, text)


_install_intros()
