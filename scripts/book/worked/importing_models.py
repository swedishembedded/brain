# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Worked examples for the chapter "Importing Pretrained Models and Proving Parity".

Plain Python 3, standard library only, written from the chapter's own description and independent of any engine code. Each public
function without arguments returns a markdown string that the chapter embeds between `<!-- worked: importing_models.<name> -->`
markers; the `*_data` functions supply the numbers of the chapter's figures, so a picture and the table beside it cannot disagree.

* a safetensors file written byte by byte (length prefix, JSON header, raw little-endian data) and parsed back, the decoding of
  its 16-bit formats, and a sharded checkpoint with its index;
* the tiny decoder of "Anatomy of a Modern Decoder" (`modern_decoder`) published in a reference layout (fused projections,
  matrices stored as [out, in], bf16) and imported back through a name-mapping plan, with every mistake of the mapping run
  through the model and measured;
* a configuration with omitted fields and what each wrong default does to the output;
* the rounding-error budget of a 32-bit implementation against a 64-bit one, the three comparison measures, golden files and the
  layer-by-layer search for the first tensor that disagrees.

Everything the chapter reuses is imported, never copied: the decoder and its initialisation (`modern_decoder`), the half-split
rotary rates (`positions`), the formats (`number_formats`) and the table helpers. 32-bit arithmetic is emulated by rounding a
64-bit result to the nearest 32-bit number after every operation (`struct`), which is exactly the correctly rounded 32-bit result
for one addition, subtraction, multiplication, division or square root of 32-bit operands; exp, cos and sin are the library's
64-bit value rounded once to 32 bits. Every random quantity comes from the book's SplitMix64 (`bookrng.generator(name)`).

Swedish Embedded AB implements solutions for bringing published model weights onto its clients' own inference engines. If your
team needs expertise in checkpoint import and numerical parity testing, you can procure our services by sending an email to
info@swedishembedded.com.
"""
import json
import math
import struct

import bookrng
import modern_decoder as md
import number_formats as nf
import positions as pos_mod

U32 = 2.0 ** -24          # unit round-off of the 32-bit format (Volume I, Chapter 15)
COS_FLOOR = 0.9999       # the row-cosine floor of the DeepSeek parity ladder
EPS_PUBLISHED = md.tb.EPS  # the epsilon of the RMS norms in the published configuration (1e-6, as in Chapter 1)
EPS_READER_DEFAULT = 1e-5  # the epsilon a reader falls back on when the configuration omits it
BASE_PUBLISHED = md.CFG["BASE"]
SHARD_BYTES = 1024        # the byte budget of one shard of the sharded example
NAME_DRAWS = "importing_models.draws"         # weights of the eight draws of the error budget
NAME_SQUARE = "importing_models.square"       # weights of the square-matrix model

# ------------------------------------------------------------------------------------------------ formatting


def _int(n):
    return f"{n:,}"


def _table(headers, rows, align=None):
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _sci(v, d=2):
    if v == 0:
        return "0"
    return f"{v:.{d}e}".replace("e-0", "e-").replace("e+0", "e+")


def _seed_line(name):
    return f"Generator: the book's SplitMix64, seed {bookrng.seed_of(name):#018x} (`seed_of` of the name `{name}`)."


# ------------------------------------------------------------------------------------------------ safetensors by hand

WIDTH = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4}


def encode_values(dtype, values):
    """The little-endian bytes of `values` stored as `dtype` (F32, F16, BF16 or I32)."""
    out = bytearray()
    for v in values:
        if dtype == "F32":
            out += struct.pack("<f", v)
        elif dtype == "I32":
            out += struct.pack("<i", v)
        else:
            out += struct.pack("<H", nf.encode(v, "fp16" if dtype == "F16" else "bf16"))
    return bytes(out)


def decode_values(dtype, raw):
    """The numbers stored in `raw` as `dtype`, in order."""
    w = WIDTH[dtype]
    out = []
    for i in range(0, len(raw), w):
        chunk = raw[i:i + w]
        if dtype == "F32":
            out.append(struct.unpack("<f", chunk)[0])
        elif dtype == "I32":
            out.append(struct.unpack("<i", chunk)[0])
        else:
            out.append(nf.decode(struct.unpack("<H", chunk)[0], "fp16" if dtype == "F16" else "bf16"))
    return out


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def write_safetensors(tensors, metadata=None):
    """tensors: list of (name, dtype, shape, values). Returns (file bytes, header bytes, offsets) where the header is the JSON text
    padded with spaces to a multiple of 8 so that the data starts at an offset divisible by 8, and the data is the tensors back to
    back in the order given."""
    header, data, offsets = {}, bytearray(), {}
    if metadata:
        header["__metadata__"] = metadata
    for name, dtype, shape, values in tensors:
        raw = encode_values(dtype, values)
        assert len(raw) == numel(shape) * WIDTH[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(data), len(data) + len(raw)]}
        offsets[name] = (len(data), len(data) + len(raw))
        data += raw
    text = json.dumps(header, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text + bytes(data), text, offsets


def read_safetensors(blob, source="buffer"):
    """Parse a safetensors buffer into (metadata, {name: (dtype, shape, values)}, data start), refusing a header that misdescribes
    its data: a length prefix past the end, offsets that are not an ordered pair inside the data, or an extent that is not the
    shape's bytes."""
    if len(blob) < 8:
        raise ValueError(f"{source}: shorter than the 8-byte length prefix")
    (n,) = struct.unpack("<Q", blob[:8])
    if n > len(blob) - 8:
        raise ValueError(f"{source}: header length {n} exceeds the {len(blob) - 8} bytes that follow the prefix")
    header = json.loads(blob[8:8 + n].decode("utf-8"))
    data = blob[8 + n:]
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        if not 0 <= start <= end <= len(data):
            raise ValueError(f"{source}: {name}: data_offsets {start}..{end} lie outside the {len(data)} bytes of data")
        want = numel(meta["shape"]) * WIDTH[meta["dtype"]]
        if end - start != want:
            raise ValueError(f"{source}: {name}: shape {meta['shape']} of {meta['dtype']} is {want} bytes, its data is {end - start}")
        out[name] = (meta["dtype"], list(meta["shape"]), decode_values(meta["dtype"], data[start:end]))
    return header.get("__metadata__", {}), out, 8 + n


TOY = [("w", "F32", [2, 3], [1.0, -2.0, 0.5, 0.25, 3.0, -0.125]),
       ("b", "BF16", [3], [1.0, -2.5, 0.3]),
       ("h", "F16", [2], [0.1, 65504.0])]


def _hex_rows(blob, width=16):
    return [f"{i:04x}  " + " ".join(f"{b:02x}" for b in blob[i:i + width]) for i in range(0, len(blob), width)]


def safetensors_file():
    blob, header, offsets = write_safetensors(TOY, {"format": "pt"})
    n = len(header)
    meta, back, start = read_safetensors(blob)
    seg = [("0 .. 8", "8", "header length N, unsigned 64-bit little-endian", f"N = {n}"),
           (f"8 .. {8 + n}", str(n), "header: JSON text, then spaces up to a multiple of 8", f"{len(header.rstrip())} characters of JSON, {n - len(header.rstrip())} of padding")]
    for name, dtype, shape, _ in TOY:
        a, b = offsets[name]
        seg.append((f"{start + a} .. {start + b}", str(b - a), f"data of `{name}` ({dtype} {shape})", f"data offsets {a} .. {b}"))
    rows = []
    for name, dtype, shape, vals in TOY:
        a, b = offsets[name]
        rows.append((f"`{name}`", dtype, str(shape), f"{a} .. {b}", start + a, (start + a) % 4,
                     ", ".join(f"{v:g}" for v in back[name][2])))
    return ("A safetensors file of three tensors: `w` is F32 with shape [2, 3] holding 1, -2, 0.5, 0.25, 3, -0.125 "
            "(row by row); `b` is BF16 with shape [3] holding 1, -2.5, 0.3; `h` is F16 with shape [2] holding 0.1, 65504. The header is "
            "the JSON of `{\"__metadata__\": {\"format\": \"pt\"}}` and the three entries in that order, written without spaces, with "
            "each entry's keys in the order dtype, shape, data_offsets, and padded with spaces to a multiple of 8. "
            f"The file is {len(blob)} bytes. Its bytes, 16 to a row, with the offset of each row at the left:\n\n```text\n"
            + "\n".join(_hex_rows(blob)) + "\n```\n\nThe header text, which is one line in the file and is broken here after each entry, with the padding spaces at its end not shown, is\n\n```text\n" + header.decode("utf-8").rstrip().replace("},\"", "},\n\"") + "\n```\n\n"
            + "The segments of the file:\n\n" + _table(["bytes", "length", "what", "detail"], seg, "lrll")
            + "\n\nParsed back (data of `h` at its stored precision): the tensors, with the offsets that the header declares, the "
            "absolute offset of the first byte of each tensor, that offset modulo 4 (a 32-bit value read in place needs 0), and the "
            "decoded numbers.\n\n"
            + _table(["name", "dtype", "shape", "offsets", "absolute start", "mod 4", "decoded"], rows, "lllrrrl"))


def dtype_decoding():
    xs = [0.3, -2.5, 1.0e-3, 70000.0, 6.0e-8]
    rows = []
    for x in xs:
        b32 = nf.f32_bits(x)
        cb = nf.encode(x, "bf16")
        ch = nf.encode(x, "fp16")
        db, dh = nf.decode(cb, "bf16"), nf.decode(ch, "fp16")
        x32 = nf.f32(x)

        def rel(d):
            return "inf" if math.isinf(d) else _sci(abs(d - x32) / abs(x32), 2)
        rows.append((f"{x:g}", f"0x{b32:08X}", f"0x{cb:04X}", f"{db:.9g}", rel(db), f"0x{ch:04X}",
                     "inf" if math.isinf(dh) else f"{dh:.9g}", rel(dh)))
    return ("Five numbers stored in the 16-bit formats of Volume IV, Chapter 1. Each is first rounded to 32 bits (its pattern in the "
            "second column), then to bf16 (1 sign, 8 exponent, 7 mantissa bits, ties to even) and to fp16 (1, 5, 10, ties to even); "
            "the decoded value is shown with 9 significant digits and its relative difference from the 32-bit number. A bf16 "
            "pattern is the top 16 bits of a 32-bit one after rounding; an fp16 pattern needs its exponent re-biased, so 70000 "
            "overflows to infinity and 6e-8 is a subnormal.\n\n"
            + _table(["x", "fp32 pattern", "bf16 pattern", "bf16 value", "bf16 relative difference", "fp16 pattern", "fp16 value",
                      "fp16 relative difference"], rows))


# ------------------------------------------------------------------------------------------------ the reference layout

HF = "model.layers.{l}."


def _transpose(M):
    return [list(r) for r in zip(*M)]


def _flat(M):
    return [x for row in M for x in row]


def _bf16(x):
    return nf.quantize(x, "bf16")


def bf16_params(p):
    """The parameters with every number rounded to bf16 (what a checkpoint stored in bf16 holds)."""
    def conv(v):
        if isinstance(v, list):
            return [conv(x) for x in v]
        return _bf16(v)
    return {k: ([{kk: conv(vv) for kk, vv in lay.items()} for lay in v] if k == "layers" else conv(v)) for k, v in p.items()}


def reference_layout(p, cfg):
    """The published layout of the parameters p: a list of (name, shape, values) in the order a file lists them. Every matrix is
    stored [out, in] row-major (the transpose of the book's x @ W layout); queries, keys and values are one fused matrix with the
    rows of W_Q^T, then those of W_K^T, then those of W_V^T; gate and up are one fused matrix (rows of W_gate^T, then W_up^T)."""
    D, V = cfg["D"], cfg["V"]
    out = [("model.embed_tokens.weight", [V, D], _flat(p["emb"]))]
    for l, lay in enumerate(p["layers"]):
        pre = HF.format(l=l)
        qkv = _transpose(lay["Wq"]) + _transpose(lay["Wk"]) + _transpose(lay["Wv"])
        gu = _transpose(lay["Wg"]) + _transpose(lay["Wu"])
        out += [(pre + "input_layernorm.weight", [D], list(lay["g1"])),
                (pre + "self_attn.qkv_proj.weight", [len(qkv), D], _flat(qkv)),
                (pre + "self_attn.q_norm.weight", [cfg["DH"]], list(lay["qn"])),
                (pre + "self_attn.k_norm.weight", [cfg["DH"]], list(lay["kn"])),
                (pre + "self_attn.o_proj.weight", [D, cfg["H"] * cfg["DH"]], _flat(_transpose(lay["Wo"]))),
                (pre + "post_attention_layernorm.weight", [D], list(lay["g2"])),
                (pre + "mlp.gate_up_proj.weight", [len(gu), D], _flat(gu)),
                (pre + "mlp.down_proj.weight", [D, cfg["F"]], _flat(_transpose(lay["Wd"])))]
    out += [("model.norm.weight", [D], list(p["gn"])), ("lm_head.weight", [V, D], _flat(p["out"]))]
    return out


def published(cfg=None):
    """(cfg, bf16 parameters, file bytes) of the published tiny model."""
    cfg = cfg or md.CFG
    p = bf16_params(md.init_decoder(cfg))
    tensors = [(n, "BF16", s, v) for n, s, v in reference_layout(p, cfg)]
    blob, _, _ = write_safetensors(tensors, {"format": "pt"})
    return cfg, p, blob


# ------------------------------------------------------------------------------------------------ storage order


def storage_order():
    W = [[1, 2, 3], [4, 5, 6]]                       # the published matrix: out = 2 rows, in = 3 columns
    flat = _flat(W)
    x = [1, 2, 3]
    y_ref = [sum(w * v for w, v in zip(row, x)) for row in W]                    # y = W x, the published convention
    book = _transpose(W)                                                          # the book's layout: x @ W', W' is [3, 2]
    y_book = [sum(x[k] * book[k][j] for k in range(3)) for j in range(2)]
    reshaped = [flat[0:2], flat[2:4], flat[4:6]]                                  # the same buffer cut into the book's [3, 2]
    y_wrong = [sum(x[k] * reshaped[k][j] for k in range(3)) for j in range(2)]
    return ("A published matrix of 2 outputs and 3 inputs, W = [[1, 2, 3], [4, 5, 6]], stored row by row as the six numbers "
            f"{', '.join(map(str, flat))}. The published convention is y = W x, and the input is x = (1, 2, 3). The book's layout "
            "computes y = x W' with W' of shape [3, 2].\n\n"
            + _table(["how W' is made from the stored buffer", "W' (rows)", "shape check [3, 2]", "y"],
                     [("transpose of W", str(book), "passes", str(y_book)),
                      ("the stored shape [2, 3] kept", str(W), "refused: x has 3 entries, W' has 2 rows", "none"),
                      ("the buffer cut again into [3, 2]", str(reshaped), "passes", str(y_wrong))], "llll")
            + f"\n\nThe published y is {tuple(y_ref)}. The correct transpose agrees with it; the third row has the right shape and the "
            "right count of numbers and computes something else.")


# ------------------------------------------------------------------------------------------------ shards


def shard_plan(tensors, budget=SHARD_BYTES):
    """Greedy sharding: tensors in file order, a new shard when the next tensor would take the current one past `budget` bytes
    (a tensor larger than the budget gets a shard of its own). Returns a list of lists of (name, dtype, shape, values)."""
    shards, cur, size = [], [], 0
    for t in tensors:
        nbytes = numel(t[2]) * WIDTH[t[1]]
        if cur and size + nbytes > budget:
            shards.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += nbytes
    if cur:
        shards.append(cur)
    return shards


def sharded_checkpoint(tensors, budget=SHARD_BYTES):
    """(files {name: bytes}, index dict) of a sharded checkpoint."""
    shards = shard_plan(tensors, budget)
    n = len(shards)
    files, weight_map, total = {}, {}, 0
    for i, sh in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        files[fname], _, _ = write_safetensors(sh, {"format": "pt"})
        for t in sh:
            weight_map[t[0]] = fname
            total += numel(t[2]) * WIDTH[t[1]]
    return files, {"metadata": {"total_size": total}, "weight_map": weight_map}


def read_sharded(files, index):
    """The tensors of a sharded checkpoint, shards read in sorted order; refuses a missing shard, a name that two shards both
    hold and a name the index lists that no shard holds."""
    for f in sorted(set(index["weight_map"].values())):
        if f not in files:
            raise ValueError(f"index names shard {f}, which is missing")
    out = {}
    for f in sorted(set(index["weight_map"].values())):
        _, tensors, _ = read_safetensors(files[f], f)
        for k, v in tensors.items():
            if k in out:
                raise ValueError(f"tensor {k} is in more than one shard")
            out[k] = v
    for k in index["weight_map"]:
        if k not in out:
            raise ValueError(f"index lists {k}, which no shard holds")
    return out


def sharded_index():
    cfg, p, _ = published()
    tensors = [(n, "BF16", s, v) for n, s, v in reference_layout(p, cfg)]
    files, index = sharded_checkpoint(tensors)
    back = read_sharded(files, index)
    rows = []
    for f in sorted(files):
        names = [k for k, v in index["weight_map"].items() if v == f]
        nbytes = sum(numel(back[k][1]) * 2 for k in names)
        rows.append((f"`{f}`", len(files[f]), len(names), nbytes, f"`{names[0]}`" + (f" .. `{names[-1]}`" if len(names) > 1 else "")))
    same = all(back[n][2] == v for n, _, _, v in tensors)
    missing = dict(files)
    del missing[sorted(files)[1]]
    errs = []
    for label, fn in (("shard 2 deleted", lambda: read_sharded(missing, index)),
                      ("a name pointing at the wrong shard", lambda: read_sharded(
                          files, {"weight_map": {**index["weight_map"], "ghost.weight": sorted(files)[0]}}))):
        try:
            fn()
            errs.append((label, "accepted"))
        except ValueError as e:
            errs.append((label, str(e)))
    total = index["metadata"]["total_size"]
    return (f"The 19 tensors of the published tiny model in bf16 ({_int(total)} bytes of tensor data), in file order, cut into shards "
            f"by the greedy rule (a new shard when the next tensor would take the current one past {SHARD_BYTES} bytes). Each shard is a "
            "safetensors file of its own, in exactly the format of the single file above (the 8-byte length N, the compact JSON header "
            "with `__metadata__` `{\"format\":\"pt\"}` first and then one entry per tensor of the shard with offsets counted from the start of "
            "that shard's data, padded with spaces to a multiple of 8, then the data), so the file size is 8 + N + tensor bytes; the "
            "index lists, for every tensor name, the shard that holds it, and the total size. The tensors, in file order, each BF16:\n\n"
            + _table(["tensor", "shape", "shard"], [(f"`{n}`", " x ".join(str(d) for d in s_), int(index["weight_map"][n].split("-")[1]))
                                                    for n, _, s_, _ in tensors], "llr")
            + "\n\n" + _table(["shard", "file bytes", "tensors", "tensor bytes", "first .. last tensor"], rows, "lrrrl")
            + "\n\nThe index file, abbreviated to its first three names:\n\n```text\n"
            + "{\n \"metadata\": {\"total_size\": " + str(total) + "},\n \"weight_map\": {\n"
            + ",\n".join(f"  \"{k}\": \"{v}\"" for k, v in list(index["weight_map"].items())[:3]) + ",\n  ...\n }\n}\n```\n\n"
            + f"Reading every shard in sorted order returns {len(back)} tensors, each once, with values equal to the originals: "
            + ("yes" if same else "NO") + ". The two failures a reader must refuse:\n\n"
            + _table(["damage", "result"], errs, "ll"))


# ------------------------------------------------------------------------------------------------ the import plan


def layer_names(l, cfg):
    return [("g1", f"gain 1 (layer {l})"), ("Wq", f"W_Q (layer {l})"), ("Wk", f"W_K (layer {l})"), ("Wv", f"W_V (layer {l})"),
            ("qn", f"QK-norm gain, queries (layer {l})"), ("kn", f"QK-norm gain, keys (layer {l})"), ("Wo", f"W_O (layer {l})"),
            ("g2", f"gain 2 (layer {l})"), ("Wg", f"W_gate (layer {l})"), ("Wu", f"W_up (layer {l})"), ("Wd", f"W_down (layer {l})")]


class Step:
    """How one book tensor is made from the checkpoint: source name, a row range [r0, r1) of the source matrix (None for all of
    it), and whether the rows are transposed to the book's [in, out] layout."""

    def __init__(self, dest, src, rows=None, transpose=False):
        self.dest, self.src, self.rows, self.transpose = dest, src, rows, transpose

    def op(self):
        parts = []
        if self.rows:
            parts.append(f"rows {self.rows[0]} .. {self.rows[1] - 1}")
        if self.transpose:
            parts.append("transpose")
        return ", ".join(parts) if parts else "copy"


def import_plan(cfg, bug=None):
    """The steps that build the book's tensors from the published layout. `bug` is None for the correct plan or one of
    "no_transpose", "gate_up_exchanged", "key_value_exchanged", "head_tied" (a mistake the chapter measures)."""
    D, H, HKV, DH, F = cfg["D"], cfg["H"], cfg["HKV"], cfg["DH"], cfg["F"]
    q0, q1 = 0, H * DH
    k0, k1 = q1, q1 + HKV * DH
    v0, v1 = k1, k1 + HKV * DH
    tr = bug != "no_transpose"
    steps = [Step("W_emb", "model.embed_tokens.weight")]
    for l in range(cfg["L"]):
        pre = HF.format(l=l)
        n = dict(layer_names(l, cfg))
        kr, vr = ((v0, v1), (k0, k1)) if bug == "key_value_exchanged" else ((k0, k1), (v0, v1))
        gr, ur = ((F, 2 * F), (0, F)) if bug == "gate_up_exchanged" else ((0, F), (F, 2 * F))
        steps += [Step(n["g1"], pre + "input_layernorm.weight"),
                  Step(n["Wq"], pre + "self_attn.qkv_proj.weight", (q0, q1), tr),
                  Step(n["Wk"], pre + "self_attn.qkv_proj.weight", kr, tr),
                  Step(n["Wv"], pre + "self_attn.qkv_proj.weight", vr, tr),
                  Step(n["qn"], pre + "self_attn.q_norm.weight"), Step(n["kn"], pre + "self_attn.k_norm.weight"),
                  Step(n["Wo"], pre + "self_attn.o_proj.weight", None, tr),
                  Step(n["g2"], pre + "post_attention_layernorm.weight"),
                  Step(n["Wg"], pre + "mlp.gate_up_proj.weight", gr, tr),
                  Step(n["Wu"], pre + "mlp.gate_up_proj.weight", ur, tr),
                  Step(n["Wd"], pre + "mlp.down_proj.weight", None, tr)]
    steps.append(Step("last gain", "model.norm.weight"))
    # A tied head is not read from the file: the plan names the checkpoint's own head as dropped, which is what a reader does when
    # its configuration says the head is tied.
    steps.append(Step(None, "lm_head.weight") if bug == "head_tied" else Step("W_out", "lm_head.weight"))
    return steps


def expected_shapes(cfg):
    """The shape of every book tensor, by name (vectors as [n])."""
    D, H, HKV, DH, F, V = (cfg[k] for k in ("D", "H", "HKV", "DH", "F", "V"))
    out = {"W_emb": [V, D], "last gain": [D], "W_out": [V, D]}
    for l in range(cfg["L"]):
        n = dict(layer_names(l, cfg))
        out.update({n["g1"]: [D], n["Wq"]: [D, H * DH], n["Wk"]: [D, HKV * DH], n["Wv"]: [D, HKV * DH], n["qn"]: [DH],
                    n["kn"]: [DH], n["Wo"]: [H * DH, D], n["g2"]: [D], n["Wg"]: [D, F], n["Wu"]: [D, F], n["Wd"]: [F, D]})
    return out


def apply_plan(steps, source, cfg):
    """Build the book's parameter structure from `source` ({name: (dtype, shape, flat values)}) by the plan, checking before it
    builds anything that every destination is produced once, every source tensor is used, and every result has the book's shape.
    Returns (params, None) or (None, the first error)."""
    want = expected_shapes(cfg)
    if "W_out" not in {s.dest for s in steps}:
        want = {k: v for k, v in want.items() if k != "W_out"}
    built, used = {}, set()
    for s in steps:
        if s.src not in source:
            return None, f"missing source tensor {s.src}"
        _, shape, vals = source[s.src]
        used.add(s.src)
        if s.dest is None:
            continue
        if s.rows and s.rows[1] > shape[0]:
            return None, f"{s.dest}: rows {s.rows[0]} .. {s.rows[1] - 1} of {s.src}, which has {shape[0]} rows"
        if len(shape) == 2:
            rows = [vals[i * shape[1]:(i + 1) * shape[1]] for i in range(shape[0])]
            if s.rows:
                rows = rows[s.rows[0]:s.rows[1]]
            if s.transpose:
                rows = _transpose(rows)
            built[s.dest] = rows
        else:
            built[s.dest] = list(vals)
    for dest, shape in want.items():
        if dest not in built:
            return None, f"no step produces {dest}"
        got = [len(built[dest])] + ([len(built[dest][0])] if isinstance(built[dest][0], list) else [])
        if got != shape:
            return None, f"{dest}: shape {got}, the model needs {shape}"
    unused = [n for n in source if n not in used]
    if unused:
        return None, f"source tensors never used: {unused}"
    p = {"emb": built["W_emb"], "layers": [], "gn": built["last gain"]}
    for l in range(cfg["L"]):
        n = dict(layer_names(l, cfg))
        p["layers"].append({k: built[v] for k, v in n.items()})
    p["out"] = built["W_out"] if "W_out" in built else built["W_emb"]
    return p, None


def name_mapping():
    cfg, p, blob = published()
    _, source, _ = read_safetensors(blob)
    steps = import_plan(cfg)
    shape_of = {s: source[s][1] for s in source}
    rows = []
    for st in steps:
        if st.dest.endswith("(layer 1)"):
            continue
        sshape = shape_of[st.src]
        if st.rows:
            sshape = f"{st.rows[1] - st.rows[0]} of {sshape[0]} rows x {sshape[1]}"
        else:
            sshape = " x ".join(str(d) for d in sshape)
        dshape = expected_shapes(cfg)[st.dest]
        rows.append((f"`{st.src.replace('model.layers.0.', 'model.layers.N.')}`", sshape, st.dest.replace(" (layer 0)", ""),
                     " x ".join(str(d) for d in dshape), st.op()))
    got, err = apply_plan(steps, source, cfg)
    same = got is not None and md.tensors(got) == md.tensors(p)
    ntens = len(source)
    nbook = len(expected_shapes(cfg))
    return ("The plan that builds the tiny decoder of Chapter 1 (V = 11, D = 6, L = 2, H = 4, 2 key-value heads of width 4, F = 10) from "
            f"its published layout. The checkpoint holds {ntens} tensors; the plan makes {nbook}. Layer 0 is shown (layer 1 is the same with "
            "`N` = 1); \"rows a .. b\" is a range of rows of the source matrix, and \"transpose\" turns a row-major [out, in] matrix "
            "into the [in, out] layout of the book's `x @ W`.\n\n"
            + _table(["checkpoint tensor", "stored as", "book tensor", "shape", "operation"], rows, "llllc")
            + f"\n\nRun on the file, the plan passes its own checks (every destination once, every source used, every shape the model's) "
            f"and builds parameters equal to the ones that were written: {'yes' if same else 'NO: ' + str(err)}.")


# ------------------------------------------------------------------------------------------------ the forward pass in two precisions


def r32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _ident(x):
    return x


def sum_terms(terms, r, order):
    """Sum of the list `terms` in the stated order, rounding with r after every addition. 'forward' adds left to right, 'reverse'
    right to left, 'pairwise' adds the sum of the first half to the sum of the second half, recursively (the middle of n terms is
    n // 2)."""
    if order == "reverse":
        terms = terms[::-1]
    if order in ("forward", "reverse"):
        s = terms[0]
        for t in terms[1:]:
            s = r(s + t)
        return s

    def pw(lo, hi):
        if hi - lo == 1:
            return terms[lo]
        mid = lo + (hi - lo) // 2
        return r(pw(lo, mid) + pw(mid, hi))
    return pw(0, len(terms))


def dot(a, b, r, order):
    return sum_terms([r(x * y) for x, y in zip(a, b)], r, order)


def matmul(A, B, r, order):
    cols = list(zip(*B))
    return [[dot(row, col, r, order) for col in cols] for row in A]


def rms(X, g, r, order, eps):
    D = len(g)
    e = r(eps)
    out = []
    for x in X:
        ms = r(dot(x, x, r, order) / D)
        inv = r(1.0 / r(math.sqrt(r(ms + e))))
        out.append([r(gj * r(xj * inv)) for gj, xj in zip(g, x)])
    return out


def run(p, cfg, ids, r=_ident, order="forward", eps=md.tb.EPS, base=None, taps=None):
    """The forward pass of the decoder of Chapter 1 on the whole text, with every operation rounded by r (the identity gives 64-bit
    arithmetic, r32 gives 32-bit) and every sum of products added in `order`. `taps`, when a list, receives the hidden states:
    the embeddings, then the output of every layer, then the output of the final norm, as lists of rows. Returns the logits."""
    D, L, H, HKV, DH = (cfg[k] for k in ("D", "L", "H", "HKV", "DH"))
    base = cfg["BASE"] if base is None else base
    ws = [r(w) for w in pos_mod.omegas(DH, base)]
    group, scale = H // HKV, r(1.0 / math.sqrt(DH))
    n = len(ids)
    X = [list(p["emb"][t]) for t in ids]
    if taps is not None:
        taps.append([list(x) for x in X])

    def rotate(x, pos):
        out = list(x)
        for m, w in enumerate(ws):
            ang = r(pos * w)
            c, s = r(math.cos(ang)), r(math.sin(ang))
            a, b = m, m + DH // 2
            out[a] = r(r(x[a] * c) - r(x[b] * s))
            out[b] = r(r(x[a] * s) + r(x[b] * c))
        return out

    for lay in p["layers"]:
        N1 = rms(X, lay["g1"], r, order, eps)
        Q0, K0, V0 = (matmul(N1, lay[k], r, order) for k in ("Wq", "Wk", "Wv"))
        q, k, v = [], [], []
        for t in range(n):
            q.append([rotate(rms([Q0[t][h * DH:(h + 1) * DH]], lay["qn"], r, order, eps)[0], t) for h in range(H)])
            k.append([rotate(rms([K0[t][h * DH:(h + 1) * DH]], lay["kn"], r, order, eps)[0], t) for h in range(HKV)])
            v.append([V0[t][h * DH:(h + 1) * DH] for h in range(HKV)])
        O = []
        for t in range(n):
            row = []
            for h in range(H):
                kv = h // group
                sc = [r(dot(q[t][h], k[j][kv], r, order) * scale) for j in range(t + 1)]
                m = max(sc)
                e = [r(math.exp(r(s - m))) for s in sc]
                den = sum_terms(e, r, order)
                pr = [r(x / den) for x in e]
                for d in range(DH):
                    row.append(sum_terms([r(pr[j] * v[j][kv][d]) for j in range(t + 1)], r, order))
            O.append(row)
        A = matmul(O, lay["Wo"], r, order)
        X1 = [[r(x + a) for x, a in zip(rx, ra)] for rx, ra in zip(X, A)]
        N2 = rms(X1, lay["g2"], r, order, eps)
        G, U = matmul(N2, lay["Wg"], r, order), matmul(N2, lay["Wu"], r, order)
        Hd = [[r(r(g / r(1.0 + r(math.exp(-g)))) * u) for g, u in zip(rg, ru)] for rg, ru in zip(G, U)]
        F = matmul(Hd, lay["Wd"], r, order)
        X = [[r(x + f) for x, f in zip(rx, rf)] for rx, rf in zip(X1, F)]
        if taps is not None:
            taps.append([list(x) for x in X])
    Hn = rms(X, p["gn"], r, order, eps)
    if taps is not None:
        taps.append(Hn)
    W = p["out"]
    return [[dot(h, W[i], r, order) for i in range(cfg["V"])] for h in Hn]


# ------------------------------------------------------------------------------------------------ the three measures


def _flatten(x):
    return [v for row in x for v in row] if isinstance(x[0], list) else list(x)


def max_abs(got, want):
    return max(abs(a - b) for a, b in zip(_flatten(got), _flatten(want)))


def rel_l2(got, want):
    g, w = _flatten(got), _flatten(want)
    num = math.sqrt(sum((a - b) ** 2 for a, b in zip(g, w)))
    return num / math.sqrt(sum(b * b for b in w))


def cosine(got, want):
    g, w = _flatten(got), _flatten(want)
    return sum(a * b for a, b in zip(g, w)) / (math.sqrt(sum(a * a for a in g)) * math.sqrt(sum(b * b for b in w)))


def argmax_agree(got, want):
    return sum(1 for a, b in zip(got, want) if a.index(max(a)) == b.index(max(b)))


# ------------------------------------------------------------------------------------------------ mistakes of the mapping


def golden_logits(p, cfg, **kw):
    return run(p, cfg, list(md.IDS), **kw)


def _measure(got, want):
    return (max_abs(got, want), rel_l2(got, want), cosine(got, want), argmax_agree(got, want))


def square_setup():
    """The degenerate model: every matrix square (D = H DH = HKV DH = F = 8; H = HKV = 4, DH = 2)."""
    cfg = md.cfg_with(D=8, H=4, HKV=4, DH=2, F=8)
    p = bf16_params(md.init_decoder(cfg, name=NAME_SQUARE))
    return cfg, p


def _imported(p, cfg, bug):
    tensors = {n: (d, s, v) for n, d, s, v in [(n, "BF16", s, v) for n, s, v in reference_layout(p, cfg)]}
    q, err = apply_plan(import_plan(cfg, bug), tensors, cfg)
    return q, err


def import_mistakes():
    cfg, p, _ = published()
    scfg, sp = square_setup()
    gold, sgold = golden_logits(p, cfg), golden_logits(sp, scfg)
    rows = []
    for label, c, pp, g, bug in (("none (the correct plan)", cfg, p, gold, None),
                                 ("transposes left out", cfg, p, gold, "no_transpose"),
                                 ("transposes left out", scfg, sp, sgold, "no_transpose"),
                                 ("keys and values exchanged", cfg, p, gold, "key_value_exchanged"),
                                 ("gate and up exchanged", cfg, p, gold, "gate_up_exchanged"),
                                 ("head dropped, tied to the embedding", cfg, p, gold, "head_tied")):
        q, err = _imported(pp, c, bug)
        model = "tiny (rectangular)" if c is cfg else "square"
        if q is None:
            rows.append((label, model, "refused: " + err, "", "", "", ""))
            continue
        m = _measure(run(q, c, list(md.IDS)), g)
        rows.append((label, model, "runs", _sci(m[0], 3), _sci(m[1], 3), f"{m[2]:.6f}", f"{m[3]} of {len(md.IDS)}"))
    return ("Each mistake of the mapping, made in the import plan and run through the 64-bit forward pass on the text of Chapter 1 "
            "(ids 3, 9, 1, 7, 4, 4, 10, 2, eps 1e-6, the epsilon of the published configuration, base 100). The published numbers are the bf16 weights; the golden is the logits of "
            "those weights, and the three measures compare the 8 x 11 logits of the imported model with it (max_abs, rel_l2 and cosine "
            "are defined below, with the goldens). \"Square\" is a model whose matrices are all square (V = 11, D = 8, L = 2, H = HKV = 4, "
            f"head width 2, F = 8; weights from the generator `{NAME_SQUARE}`, seed {bookrng.seed_of(NAME_SQUARE):#018x}, drawn in "
            "the order of Chapter 1). A plan checks the shape of each tensor it builds before it runs anything.\n\n"
            + _table(["mistake", "model", "outcome", "max_abs", "rel_l2", "cosine", "argmax agrees"], rows, "llllrrr"))


# ------------------------------------------------------------------------------------------------ configuration

PUBLISHED_CONFIG = {"architectures": ["TinyForCausalLM"], "vocab_size": 11, "hidden_size": 6, "num_hidden_layers": 2,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4, "intermediate_size": 10,
                    "rms_norm_eps": EPS_PUBLISHED, "rope_theta": BASE_PUBLISHED, "tie_word_embeddings": False}
LIBRARY_DEFAULTS = {"rms_norm_eps": EPS_READER_DEFAULT, "rope_theta": 10000.0, "tie_word_embeddings": False}   # the reader's own defaults


def resolve_config(cfg_json):
    """The model the reader builds from a configuration: omitted fields take the reader's defaults (kv heads: the number of query
    heads; the others: LIBRARY_DEFAULTS). Returns a dict of the fields that matter."""
    c = dict(cfg_json)
    heads = c["num_attention_heads"]
    return {"H": heads, "HKV": c.get("num_key_value_heads", heads), "eps": c.get("rms_norm_eps", LIBRARY_DEFAULTS["rms_norm_eps"]),
            "base": c.get("rope_theta", LIBRARY_DEFAULTS["rope_theta"]),
            "tied": c.get("tie_word_embeddings", LIBRARY_DEFAULTS["tie_word_embeddings"])}


def config_defaults():
    cfg, p, blob = published()
    _, source, _ = read_safetensors(blob)
    gold = golden_logits(p, cfg, eps=EPS_PUBLISHED)
    rows = []
    shown = json.dumps(PUBLISHED_CONFIG, indent=1)
    for field in ("num_key_value_heads", "rms_norm_eps", "rope_theta", "tie_word_embeddings"):
        trimmed = {k: v for k, v in PUBLISHED_CONFIG.items() if k != field}
        r_ = resolve_config(trimmed)
        default = {"num_key_value_heads": f"{r_['HKV']} (= the query heads)", "rms_norm_eps": f"{r_['eps']:g}",
                   "rope_theta": f"{r_['base']:g}", "tie_word_embeddings": str(r_["tied"]).lower()}[field]
        truth = {"num_key_value_heads": "2", "rms_norm_eps": f"{EPS_PUBLISHED:g}", "rope_theta": f"{BASE_PUBLISHED:g}",
                 "tie_word_embeddings": "false"}[field]
        if field == "num_key_value_heads":
            c2 = md.cfg_with(HKV=r_["HKV"])
            _, err = apply_plan(import_plan(c2), source, c2)
            outcome = "refused: " + err if err else "runs"
            rows.append((f"`{field}`", truth, default, outcome, "", ""))
            continue
        got = golden_logits(p, cfg, eps=r_["eps"], base=r_["base"])
        m = _measure(got, gold)
        if field == "tie_word_embeddings":
            rows.append((f"`{field}`", truth, default, "runs (the default is the truth)", "0", "1.000000"))
            q, _ = _imported(p, cfg, "head_tied")
            m = _measure(golden_logits(q, cfg, eps=EPS_PUBLISHED), gold)
            rows.append((f"`{field}`, read by a reader whose default is true", truth, "true", "runs, `lm_head.weight` dropped",
                         _sci(m[1], 2), f"{m[2]:.6f}"))
        else:
            rows.append((f"`{field}`", truth, default, "runs", _sci(m[1], 2), f"{m[2]:.6f}"))
    return ("The published configuration of the tiny model, with the fields a reader needs:\n\n```json\n" + shown
            + "\n```\n\nOne field at a time is removed and the reader's default is used instead (the reader defaults the key-value "
            "heads to the query heads, the epsilon to 1e-5, the rotary base to 10000 and the tied head to false). The weights and "
            "the text are those of the previous table; the golden is the 64-bit forward pass with the published values "
            "(eps 1e-6, base 100); rel_l2 and cosine compare the logits computed with the default against it.\n\n"
            + _table(["omitted field", "published", "default", "outcome", "rel_l2 of logits", "cosine"], rows, "lllllr"))


def eps_scale_table():
    cfg, p, _ = published()
    rows = []
    for s in (1.0, 0.1, 0.01):
        q = dict(p)
        q["emb"] = [[s * x for x in row] for row in p["emb"]]
        gold = golden_logits(q, cfg, eps=EPS_PUBLISHED)
        got = golden_logits(q, cfg, eps=EPS_READER_DEFAULT)
        rows.append((f"{s:g}", _sci(rel_l2(got, gold), 2), _sci(max_abs(got, gold), 2)))
    return ("The same wrong epsilon (1e-5 for the published 1e-6) at three scales of the activations: the embedding table is multiplied "
            "by the factor in the first column, the weights are unchanged, and the golden is the 64-bit forward pass with eps 1e-6 on "
            "the same scaled table.\n\n" + _table(["factor on the embedding table", "rel_l2 of logits", "max_abs of logits"], rows))


# ------------------------------------------------------------------------------------------------ the error budget


def stages(cfg, T):
    """(stage, roundings along the longest chain of dependent operations) for one layer of T positions, and for the head. Rule: a
    sum of n products costs n roundings (n multiplications and n - 1 additions, each at most u, by the bound of Chapter 15 of
    Volume I); one more rounding for each further operation applied to its result."""
    D, H, DH, F = cfg["D"], cfg["H"], cfg["DH"], cfg["F"]
    layer = [("RMS norm 1 (sum of D squares, divide by D, add eps, sqrt, reciprocal, product with x, product with the gain)", D + 6),
             ("projection to q, k, v (dot of length D)", D),
             ("QK-norm (sum of d_h squares and the same six operations)", DH + 6),
             ("rotation (angle, cosine or sine, product, add)", 4),
             ("scores (dot of length d_h, scale)", DH + 1),
             ("softmax (subtract, exp, sum of T terms, divide)", T + 3),
             ("weighted sum of values (T products added)", T),
             ("W_O (dot of length H d_h)", H * DH),
             ("residual add", 1),
             ("RMS norm 2 (as norm 1)", D + 6),
             ("gate or up projection (dot of length D)", D),
             ("silu, product with up (exp, add, divide, multiply)", 4),
             ("W_down (dot of length F)", F),
             ("residual add", 1)]
    head = [("final RMS norm (as norm 1)", D + 6), ("output projection (dot of length D)", D)]
    return layer, head


def budget(cfg, T, layers):
    layer, head = stages(cfg, T)
    return sum(c for _, c in layer) * layers


def bound_at(cfg, T, k):
    """Bound of the relative error after k layers: roundings of k layers (golden storage adds one) times u."""
    return (budget(cfg, T, k) + 1) * U32


def bound_logits(cfg, T):
    layer, head = stages(cfg, T)
    return (budget(cfg, T, cfg["L"]) + sum(c for _, c in head) + 1) * U32


def statistical_logits(cfg, T):
    """The random-walk estimate: the roundings as independent errors, so the count enters under a square root."""
    layer, head = stages(cfg, T)
    n = budget(cfg, T, cfg["L"]) + sum(c for _, c in head) + 1
    return math.sqrt(n) * U32


def draws_cfg_p(i):
    cfg = md.CFG
    p = bf16_params(md.init_decoder(cfg, name=f"{NAME_DRAWS}.{i}"))
    return cfg, p


def error_budget():
    cfg = md.CFG
    T = len(md.IDS)
    layer, head = stages(cfg, T)
    rows = [(s, c) for s, c in layer] + [("one layer, total", sum(c for _, c in layer))]
    rows += [(s, c) for s, c in head] + [("storing the golden in 32 bits", 1)]
    total = budget(cfg, T, cfg["L"]) + sum(c for _, c in head) + 1
    meas = []
    for i in range(8):
        c, p = draws_cfg_p(i)
        gold = run(p, c, list(md.IDS))
        row = [i]
        for order in ("forward", "reverse", "pairwise"):
            g32 = run(p, c, list(md.IDS), r=r32, order=order)
            stored = [[r32(x) for x in rw] for rw in gold]
            row.append(rel_l2(g32, stored))
        meas.append(row)
    worst = max(max(r[1:]) for r in meas)
    mrows = [(f"{bookrng.seed_of(f'{NAME_DRAWS}.{r[0]}'):#018x}", _sci(r[1], 2), _sci(r[2], 2), _sci(r[3], 2)) for r in meas]
    return ("Roundings along the longest chain of dependent operations of the tiny decoder for the last of T = "
            f"{T} positions (D = 6, H = 4, head width 4, F = 10), counted by the rule that a sum of n products costs n roundings and "
            "each further operation on a result costs one. The RMS norm is the one of the book's pseudocode: `inv_std = 1 / sqrt(mean(x^2) + eps)`, "
            "`xhat = x * inv_std`, `y = gamma * xhat`, with eps = 1e-6, so the two products follow the reciprocal on one chain (D "
            "roundings for the sum of D squares, then the divide, the addition of eps, the square root, the reciprocal and the two "
            "products: D + 6):\n\n" + _table(["stage", "roundings"], rows, "lr")
            + f"\n\nTwo layers, the head and the stored golden give N = {total} roundings, so the bound on the relative error of the "
            f"logits is N u = {total} x 2^-24 = {_sci(bound_logits(cfg, T), 2)}, and the random-walk estimate sqrt(N) u is "
            f"{_sci(statistical_logits(cfg, T), 2)}. Measured: eight draws of the weights of the tiny model "
            f"(generators `{NAME_DRAWS}.0` to `{NAME_DRAWS}.7`, same shapes, scales and draw order as Chapter 1, rounded to bf16), "
            "each run in 32-bit arithmetic with the three summation orders against the 64-bit logits stored in 32 bits. The 32-bit run "
            "rounds to the nearest 32-bit number after every operation: each product and each addition of every sum (the sum of "
            "squares of a norm, every dot product, the softmax denominator, the weighted sum of values), each division, square root "
            "and reciprocal (the gated product is fl(fl(g / fl(1 + fl(exp(-g)))) * up): the exp, the add, the divide, then the product "
            "with up, the four roundings of the table), each exp, cos and sin (the 64-bit library value rounded once), and the constants eps, the rotary rates "
            "and 1/sqrt(d_h) before they are used; the bf16 weights are exactly representable, and the maximum of the softmax is exact. "
            "The 'left to right' order adds the terms of a sum one after another from the first; 'right to left' from the last; "
            "'pairwise' adds the sum of the first half to the sum of the second half, recursively, the middle of n terms being "
            "n // 2. rel_l2 of the "
            "8 x 11 logits:\n\n" + _table(["seed of the draw", "left to right", "right to left", "pairwise"], mrows)
            + f"\n\nThe largest measured error is {_sci(worst, 2)}, {bound_logits(cfg, T) / worst:.0f} times below the bound "
            f"and {worst / statistical_logits(cfg, T):.2f} times the random-walk estimate.")


# ------------------------------------------------------------------------------------------------ the measures disagree


def metric_blindness():
    cfg, p, _ = published()
    T = len(md.IDS)
    gold = golden_logits(p, cfg)
    gold5 = golden_logits(p, cfg, eps=EPS_PUBLISHED)
    tol = bound_logits(cfg, T)
    cases = []
    q = dict(p)
    q["gn"] = [1.05 * x for x in p["gn"]]
    cases.append(("final gain scaled by 1.05 (a dropped or doubled scale)", golden_logits(q, cfg), gold))
    cases.append(("logits scaled by 0.5", [[0.5 * x for x in rw] for rw in gold], gold))
    for label, bug in (("gate and up exchanged", "gate_up_exchanged"), ("keys and values exchanged", "key_value_exchanged"),
                       ("head tied to the embedding", "head_tied")):
        q, _ = _imported(p, cfg, bug)
        cases.append((label, golden_logits(q, cfg), gold))
    cases.append(("rotary base 10000 for the published 100", golden_logits(p, cfg, eps=EPS_PUBLISHED, base=10000.0), gold5))
    cases.append(("epsilon 1e-5 for the published 1e-6", golden_logits(p, cfg, eps=EPS_READER_DEFAULT), gold5))
    cases.append(("32-bit arithmetic, left to right", run(p, cfg, list(md.IDS), r=r32), gold))
    rows = []
    for label, got, want in cases:
        m = _measure(got, want)
        cos_pass = "pass" if m[2] >= COS_FLOOR else "fail"
        rl_pass = "pass" if m[1] <= tol else "fail"
        loose = "pass" if (m[3] == T and m[0] < 1.0) else "fail"
        rows.append((label, "below 1e-12" if 1 - m[2] < 1e-12 else _sci(1 - m[2], 2), cos_pass, _sci(m[1], 2), rl_pass, _sci(m[0], 2), loose))
    return (f"Eight outputs compared with the 64-bit logits of the published tiny model (the two rows about the rotary base and the "
            f"epsilon: with the logits at the published eps 1e-6, base 100) by the three measures and by three tests: cosine at least "
            f"{COS_FLOOR} (the floor of the DeepSeek ladder described below), rel_l2 at most the bound {_sci(tol, 2)} derived below "
            "from the count of roundings, and the gate of Chapter 1's forward-parity test (the top prediction agrees at all 8 positions and max_abs is below "
            "1.0). 1 - cosine is printed because cosine is 1 to six places for most rows.\n\n"
            + _table(["output", "1 - cosine", "cosine test", "rel_l2", "rel_l2 test", "max_abs", "gate of Chapter 1"], rows, "lrlrlrl"))


def bf16_weights_effect():
    cfg = md.CFG
    p64 = md.init_decoder(cfg)
    pbf = bf16_params(p64)
    g64, gbf = golden_logits(p64, cfg), golden_logits(pbf, cfg)
    n = md.parameter_count(p64)
    ws = [x for _, rows in md.tensors(p64) for row in rows for x in row]
    rel = max(abs(nf.quantize(x, "bf16") - x) / abs(x) for x in ws if x != 0)
    return (f"The weights of the tiny model as drawn (64 bits) and rounded to bf16, {_int(n)} numbers. Largest relative change of one "
            f"weight {_sci(rel, 2)} (the bound for round to nearest is 2^-8 = {_sci(2.0 ** -8, 2)}). Logits of the two models on the text of "
            f"Chapter 1: relative L2 difference {_sci(rel_l2(gbf, g64), 2)}, largest absolute difference {_sci(max_abs(gbf, g64), 2)}, "
            f"cosine similarity {cosine(gbf, g64):.6f}. The bound on the rounding of a 32-bit forward pass (the error budget "
            f"below), {_sci(bound_logits(cfg, len(md.IDS)), 2)}, is {rel_l2(gbf, g64) / bound_logits(cfg, len(md.IDS)):.0f} "
            "times smaller than this change, which is why a golden has to be computed from the published weights and not from "
            "the weights before they were rounded.")


# ------------------------------------------------------------------------------------------------ goldens and bisection

BIS_L = 6
BIS_BUG_LAYER = 3
NAME_BIS = "importing_models.deep"


def deep_setup():
    cfg = md.cfg_with(L=BIS_L)
    p = bf16_params(md.init_decoder(cfg, name=NAME_BIS))
    return cfg, p


def golden_file(p, cfg):
    """The golden of a model on md.IDS as a safetensors file (bytes): the ids as I32, every hidden state and the logits as F32."""
    taps = []
    logits = run(p, cfg, list(md.IDS), taps=taps)
    T = len(md.IDS)
    items = [("ids", "I32", [T], list(md.IDS))]
    for i, t in enumerate(taps[:-1]):
        items.append((f"hidden_{i:02d}", "F32", [T, cfg["D"]], [r32(x) for x in _flatten(t)]))
    items.append(("final_norm", "F32", [T, cfg["D"]], [r32(x) for x in _flatten(taps[-1])]))
    items.append(("logits", "F32", [T, cfg["V"]], [r32(x) for x in _flatten(logits)]))
    meta = {"generator": NAME_BIS, "eps": f"{EPS_PUBLISHED:g}", "rope_base": f"{cfg['BASE']:g}", "arithmetic": "64-bit, stored as 32-bit",
            "tolerance_basis": "rounding count x 2^-24"}
    blob, header, offsets = write_safetensors(items, meta)
    return blob, header, offsets, items


def golden_contents():
    cfg, p = deep_setup()
    blob, header, offsets, items = golden_file(p, cfg)
    _, back, start = read_safetensors(blob)
    rows = []
    for name, dtype, shape, _ in items:
        a, b = offsets[name]
        if name.startswith("hidden_"):
            what = "the embeddings" if name == "hidden_00" else f"output of layer {int(name[-2:]) - 1}"
        else:
            what = {"ids": "the input text", "final_norm": "output of the last RMS norm, before the head", "logits": "the output"}[name]
        rows.append((f"`{name}`", dtype, " x ".join(str(d) for d in shape), b - a, what))
    total = sum(r[3] for r in rows)
    return ("The golden for a 6-layer version of the tiny decoder (V = 11, D = 6, H = 4, 2 key-value heads of width 4, F = 10, "
            f"L = 6; weights from the generator `{NAME_BIS}`, seed {bookrng.seed_of(NAME_BIS):#018x}, drawn in the order of Chapter 1 "
            "and rounded to bf16), written as one safetensors file. It holds the text, the hidden state "
            "after the embedding and after each layer, the output of the final norm and the logits, each tensor stored in the order "
            "listed below with the data back to back. The header is the compact JSON of the three-tensor file above (`__metadata__` first, "
            "then the tensors in that order), padded with spaces to a multiple of 8, and its `__metadata__` is exactly "
            f"`{json.dumps(json.loads(header.decode())['__metadata__'], separators=(',', ':'))}`, "
            f"so the header is {len(header)} bytes. "
            f"The file is {_int(len(blob))} bytes ({_int(total)} of tensor data).\n\n"
            + _table(["tensor", "dtype", "shape", "bytes", "what"], rows, "lllrl"))


def bisection_data():
    """[(tap name, rel_l2 of the correct 32-bit import, rel_l2 of the 32-bit import with gate and up exchanged in layer 3, bound)]
    for every tap, against the 64-bit golden stored in 32 bits."""
    cfg, p = deep_setup()
    T = len(md.IDS)
    gold_taps = []
    run(p, cfg, list(md.IDS), taps=gold_taps)
    gold_taps = [[[r32(x) for x in rw] for rw in t] for t in gold_taps]
    good_taps = []
    run(p, cfg, list(md.IDS), r=r32, order="forward", taps=good_taps)
    bad = dict(p)
    bad["layers"] = [dict(lay) for lay in p["layers"]]
    lay = bad["layers"][BIS_BUG_LAYER]
    lay["Wg"], lay["Wu"] = lay["Wu"], lay["Wg"]
    bad_taps = []
    run(bad, cfg, list(md.IDS), r=r32, order="forward", taps=bad_taps)
    names = ["embeddings"] + [f"layer {i}" for i in range(cfg["L"])] + ["final norm"]
    out = []
    for i, nm in enumerate(names):
        k = i if i <= cfg["L"] else cfg["L"]
        layer, head = stages(cfg, T)
        if i == len(names) - 1:
            bound = (budget(cfg, T, cfg["L"]) + head[0][1] + 1) * U32
        else:
            bound = bound_at(cfg, T, i)
        out.append((nm, rel_l2(good_taps[i], gold_taps[i]), rel_l2(bad_taps[i], gold_taps[i]), bound))
    return out


def first_divergence(rows):
    """The first tap whose rel_l2 exceeds its bound, or None."""
    for nm, _, bad, bound in rows:
        if bad > bound:
            return nm
    return None


def bisection():
    rows = bisection_data()
    first = first_divergence(rows)
    layer, head = stages(md.cfg_with(L=BIS_L), len(md.IDS))
    n_layer, n_norm = sum(c for _, c in layer), head[0][1]
    body = [(nm, _sci(g, 2) if g else "0", _sci(b, 2) if b else "0", _sci(bound, 2), "above" if b > bound else "within")
            for nm, g, b, bound in rows]
    return ("Layer-by-layer comparison for the 6-layer model whose golden is listed above. The implementation under test is the 32-bit forward "
            "pass (summation left to right); the golden is the stored 64-bit one. The mistake is an import in which gate and up are "
            f"exchanged in layer {BIS_BUG_LAYER} only. For each hidden state, the rel_l2 of the correct import, the rel_l2 of the "
            "mistaken one, and the bound (roundings of the layers so far, plus one for the stored golden, times 2^-24). Every layer "
            f"counts {n_layer} roundings (the table above, with eps 1e-6 and the "
            f"same 32-bit rules), so the bound after k layers is (k x that + 1) x 2^-24; the final norm adds its own "
            f"{n_norm} roundings (D + 6), so its bound is ({BIS_L} x {n_layer} + {n_norm} + 1) x 2^-24.\n\n"
            + _table(["hidden state", "rel_l2, correct import", "rel_l2, mistaken import", "bound", "mistaken import is"], body, "lrrrl")
            + f"\n\nThe first hidden state above its bound is the output of {first}.")


# ------------------------------------------------------------------------------------------------ data of the figures


def layout_data():
    """The segments of the toy file of `safetensors_file`: (label, first byte, last byte + 1) of the length prefix, the header and
    each tensor, and the header text."""
    blob, header, offsets = write_safetensors(TOY, {"format": "pt"})
    start = 8 + len(header)
    segs = [("length N", 0, 8), ("header", 8, start)]
    segs += [(name, start + a, start + b) for name, (a, b) in offsets.items()]
    return segs, header.decode("utf-8"), len(blob)


def mapping_data():
    """The steps of layer 0 of the import plan as (source name, source shape, dest name, dest shape, operation)."""
    cfg, p, blob = published()
    _, source, _ = read_safetensors(blob)
    shapes = expected_shapes(cfg)
    return [(s.src, source[s.src][1], s.dest, shapes[s.dest], s.op()) for s in import_plan(cfg)
            if s.dest and (s.dest.endswith("(layer 0)"))]
