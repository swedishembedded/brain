#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump Wan2.1 umT5-XXL text-encoder reference goldens for brain parity tests.

Runs the OFFICIAL `Wan2.1/wan/modules/t5.py` (CPU, fp32) on a handful of fixed
prompts and dumps every stage brain has to reproduce:

  t5/tokens.safetensors    token ids + attention mask + lengths for N prompts,
                           each padded to `text_len = 512`
  t5/encoder.safetensors   bucket table, the 24 per-block relative-position
                           tables, two materialised `[heads, 512, 512]` biases,
                           the embedding, block-0 internals, every block output,
                           `last_hidden_state`, the zero-padded context the DiT
                           actually consumes, and the UNMASKED run for contrast
  t5/manifest.json         prompts, shapes, sha256 per file, versions

Everything is saved as f32 (brain's safetensors reader is F32/F16/BF16-only);
token ids cast exactly.

## Importing the reference

`import wan.modules.t5` executes `wan/__init__.py`, which drags in the whole
model stack, so both modules are loaded BY FILE PATH under a synthetic package
so that `t5.py`'s `from .tokenizers import HuggingfaceTokenizer` resolves.

Two shims are needed and both are recorded in the manifest:

* `t5.py`'s `T5EncoderModel.__init__` has `device=torch.cuda.current_device()`
  as a DEFAULT ARGUMENT, which is evaluated when the class body executes, i.e.
  at import. On a CPU-only torch that raises, so `torch.cuda.current_device` is
  patched to return `'cpu'` for the duration of the import only.
* `tokenizers.py` imports `ftfy`, which is not installed here. It is stubbed
  with an identity `fix_text`. `ftfy.fix_text` repairs mojibake; it is the
  identity on well-formed UTF-8, and every prompt below is well-formed, so the
  cleaning path that remains (`html.unescape` twice, then the `\\s+ -> ' '`
  collapse and strip) is the reference's own code, unmodified.

## Self-validation (two independent paths, asserted before anything is written)

1. **Tokenizer**: the HF tokenizer the reference wraps vs a from-scratch
   Unigram Viterbi written here against `tokenizer.json` (normalizer, Metaspace
   pre-tokenizer, lattice, `fuse_unk`, `</s>` post-processor). This second path
   is the executable spec `crates/data/src/unigram.rs` is written from, so the
   two are checked against each other rather than one being trusted.
2. **Relative position bias**: each block's `T5RelativeEmbedding.forward` output
   vs a bucket table recomputed here from the formula, gathered through the
   block's own `[num_buckets, heads]` table.
3. **Per-block, not shared**: the 24 tables are asserted pairwise DISTINCT. umT5
   passes `shared_pos=False` (`t5.py:456-466`) where the class default is
   `True`; a port that shares block 0's table produces plausible embeddings and
   subtly wrong video, so the fixture itself has to be able to tell them apart.
4. **Encoder**: the reference wrapper's own `T5EncoderModel.__call__` (tokenize
   -> forward -> trim to `seq_len`) vs the manual `model(ids, mask)` path used
   for the taps.
5. **Truncation equivalence**: `model(ids[:, :L], None)` equals the masked
   512-wide run's first L rows. Nothing in this encoder mixes rows except
   attention, and the mask removes the pad keys entirely, so the two must agree
   to fp32 reassociation. This is what makes the `text_len` pad a pure host-side
   operation on brain's side.

Usage:
  python tools/goldens/wan_t5_dump_reference.py \\
      --weights /path/to/models_t5_umt5-xxl-enc-bf16.pth \\
      --tokenizer /path/to/google/umt5-xxl \\
      --out testdata/golden/wan/t5
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import types

import torch
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer

# `wan/configs/shared_config.py`: text_len = 512 for every Wan2.1 task.
TEXT_LEN = 512

# Prompts every golden is built from. Multilingual on purpose - the 256k
# vocabulary is the whole reason umT5 replaces T5 v1.1 here, and an
# ASCII-only fixture would exercise none of it.
ENCODER_PROMPTS = [
    "A belgian malinois running on a paved highway, cinematic lighting",
    "两只可爱的橘猫戴着墨镜,在阳光下的沙滩上散步。",
]

# Tokenizer-only prompts: ids are cheap, an 11 GB encoder forward is not.
# Includes `wan/configs/shared_config.py`'s real negative prompt, the empty
# string (the unconditional branch when no negative prompt is given), scripts
# outside the BMP, and text that forces the unknown-piece path.
TOKENIZER_PROMPTS = ENCODER_PROMPTS + [
    "Un cafe au bord de la Seine a Paris, sous la pluie, ambiance cinematographique",
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走",
    "",
    "Ελληνικά, русский, العربية, हिन्दी, 한국어, 日本語のテキスト",
    "a rocket \U0001F680 and a cat \U0001F408 with a beaker \U0001F9EA",
    "\U00010437\U00010437 deseret, \ufb01 ligature, e\u0301 combining",
    "   collapsed    whitespace   and\ttabs   ",
]


# --------------------------------------------------------------------------
# loading the reference by file path
# --------------------------------------------------------------------------

def load_reference(t5_path):
    """Import `wan/modules/t5.py` (and its sibling `tokenizers.py`) by path."""
    root = os.path.dirname(os.path.abspath(t5_path))
    pkg = types.ModuleType("wan_ref")
    pkg.__path__ = [root]
    sys.modules["wan_ref"] = pkg

    if "ftfy" not in sys.modules:
        ftfy = types.ModuleType("ftfy")
        ftfy.fix_text = lambda s: s
        sys.modules["ftfy"] = ftfy

    def load(name, path):
        spec = importlib.util.spec_from_file_location(f"wan_ref.{name}", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"wan_ref.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    tok = load("tokenizers", os.path.join(root, "tokenizers.py"))
    saved = getattr(torch.cuda, "current_device", None)
    torch.cuda.current_device = lambda: "cpu"
    try:
        t5 = load("t5", t5_path)
    finally:
        if saved is not None:
            torch.cuda.current_device = saved
    return t5, tok


# --------------------------------------------------------------------------
# the independent tokenizer path (the executable spec for the Rust port)
# --------------------------------------------------------------------------

class RefUnigram:
    """SentencePiece-unigram encoder read straight from `tokenizer.json`.

    Deliberately independent of `tokenizers`: normalizer, Metaspace
    pre-tokenizer, Viterbi lattice, `fuse_unk` and the `</s>` post-processor are
    all reimplemented here so that agreeing with the library means something.
    """

    UNK_PENALTY = 10.0
    METASPACE = "▁"

    def __init__(self, path):
        j = json.load(open(path, encoding="utf-8"))
        m = j["model"]
        assert m["type"] == "Unigram", m["type"]
        assert not m.get("byte_fallback"), "byte_fallback would need the byte pieces"
        assert j["normalizer"] == {
            "type": "Sequence",
            "normalizers": [{"type": "Replace", "pattern": {"Regex": " {2,}"}, "content": " "}],
        }, j["normalizer"]
        pre = j["pre_tokenizer"]
        assert pre["type"] == "Metaspace" and pre["replacement"] == self.METASPACE, pre
        # Two spellings of the same Metaspace in the wild: the current
        # `prepend_scheme`/`split` pair, and the legacy `add_prefix_space` bool
        # that `tokenizers` widens to exactly `always` + `split` when true.
        # `google/umt5-xxl` ships the legacy form, the diffusers export the new
        # one, and the two encode identically.
        if "prepend_scheme" in pre:
            assert pre["prepend_scheme"] == "always" and pre["split"], pre
        else:
            assert pre["add_prefix_space"] is True, pre
        self.unk_id = m["unk_id"]
        self.piece = {}
        for i, (s, sc) in enumerate(m["vocab"]):
            self.piece.setdefault(s, (i, sc))
        self.unk_score = min(sc for _, sc in m["vocab"]) - self.UNK_PENALTY
        self.max_bytes = max(len(s.encode()) for s, _ in m["vocab"])
        self.added = {a["content"]: a["id"] for a in j["added_tokens"]}
        self.added_re = re.compile(
            "|".join(sorted((re.escape(k) for k in self.added), key=len, reverse=True)))
        post = j["post_processor"]
        assert post["type"] == "TemplateProcessing", post["type"]
        self.suffix = [t["SpecialToken"]["id"] for t in post["single"] if "SpecialToken" in t]
        self.suffix = [post["special_tokens"][s]["ids"][0] for s in self.suffix]

    def _viterbi(self, text):
        b = text.encode()
        n = len(b)
        best = [None] * (n + 1)
        best[0] = (0.0, None, None)
        i = 0
        while i < n:
            cur = best[i][0]
            c = b[i]
            mblen = 1 if c < 0x80 else (2 if c < 0xE0 else (3 if c < 0xF0 else 4))
            single = False
            for ln in range(1, min(self.max_bytes, n - i) + 1):
                try:
                    s = b[i:i + ln].decode()
                except UnicodeDecodeError:
                    continue
                e = self.piece.get(s)
                if e is None:
                    continue
                idx, sc = e
                cand = cur + sc
                t = best[i + ln]
                if t is None or cand > t[0]:
                    best[i + ln] = (cand, i, idx)
                if ln == mblen:
                    single = True
            if not single:
                t = best[i + mblen]
                cand = cur + self.unk_score
                if t is None or cand > t[0]:
                    best[i + mblen] = (cand, i, self.unk_id)
            i += mblen
        path, k = [], n
        while k > 0:
            _, st, idx = best[k]
            path.append((st, k, idx))
            k = st
        path.reverse()
        # fuse_unk: consecutive unknown spans become ONE unk token.
        out = []
        for st, en, idx in path:
            if idx == self.unk_id and out and out[-1][2] == self.unk_id:
                out[-1] = (out[-1][0], en, idx)
            else:
                out.append((st, en, idx))
        return [i for _, _, i in out]

    def encode(self, text):
        text = re.sub(" {2,}", " ", text)
        ids, pos, segs = [], 0, []
        for m in self.added_re.finditer(text):
            if m.start() > pos:
                segs.append((text[pos:m.start()], None))
            segs.append((m.group(), self.added[m.group()]))
            pos = m.end()
        if pos < len(text):
            segs.append((text[pos:], None))
        for seg, sid in segs:
            if sid is not None:
                ids.append(sid)
                continue
            if seg == "":
                continue
            s = seg.replace(" ", self.METASPACE)
            if not s.startswith(self.METASPACE):
                s = self.METASPACE + s
            # Metaspace(split=True) is SplitDelimiterBehavior::MergedWithNext:
            # every piece begins with the replacement character.
            parts, cur = [], ""
            for ch in s:
                if ch == self.METASPACE:
                    if cur:
                        parts.append(cur)
                    cur = self.METASPACE
                else:
                    cur += ch
            if cur:
                parts.append(cur)
            for p in parts:
                ids.extend(self._viterbi(p))
        return ids + self.suffix


# --------------------------------------------------------------------------
# the independent relative-position bias path
# --------------------------------------------------------------------------

def bucket_table(t, num_buckets, max_dist=128):
    """`T5RelativeEmbedding._relative_position_bucket`, bidirectional, rewritten
    from the formula (`t5.py:245-264`) as scalar integer math."""
    half = num_buckets // 2
    max_exact = half // 2
    denom = math.log(max_dist / max_exact)
    out = torch.zeros(t, t, dtype=torch.long)
    for i in range(t):
        for jj in range(t):
            rel = jj - i
            direction = half if rel > 0 else 0
            a = abs(rel)
            if a < max_exact:
                off = a
            else:
                big = max_exact + int(math.log(a / max_exact) / denom * (half - max_exact))
                off = min(big, half - 1)
            out[i, jj] = direction + off
    return out


# --------------------------------------------------------------------------

def save(out, name, tensors, manifest):
    tensors = {k: v.detach().to(torch.float32).clone().contiguous() for k, v in tensors.items()}
    path = os.path.join(out, name)
    save_file(tensors, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest[name] = {"sha256": h, "tensors": {k: list(v.shape) for k, v in tensors.items()}}
    total = sum(v.numel() * 4 for v in tensors.values())
    print(f"wrote {name}: {len(tensors)} tensors, {total / 1e6:.1f} MB", flush=True)


def agree(name, a, b, tol=2e-5):
    """Assert two paths agree RELATIVE to the reference tensor's own scale."""
    d = (a.double() - b.double()).abs().max().item()
    scale = max(1e-6, b.double().abs().max().item())
    rel = d / scale
    print(f"  self-validate {name}: max abs {d:.3e} / scale {scale:.3g} = {rel:.2e} "
          f"(tol {tol:g})", flush=True)
    assert rel < tol, f"{name}: the two paths disagree by {rel:.3e} relative"
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="models_t5_umt5-xxl-enc-bf16.pth")
    ap.add_argument("--tokenizer", required=True, help="google/umt5-xxl directory")
    ap.add_argument("--reference", default="scratchpad/reference/wan/Wan2.1/wan/modules/t5.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-only", action="store_true",
                    help="skip the 11 GB encoder forward; write tokens.safetensors only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    t5, ref_tok = load_reference(args.reference)
    print(f"reference: {args.reference}", flush=True)

    manifest = {
        "run": {
            "text_len": TEXT_LEN,
            "encoder_prompts": ENCODER_PROMPTS,
            "tokenizer_prompts": TOKENIZER_PROMPTS,
            "tokenizer": os.path.basename(os.path.normpath(args.tokenizer)),
            "clean": "whitespace",
            "shims": [
                "torch.cuda.current_device -> 'cpu' during the t5.py import only",
                "ftfy.fix_text stubbed to the identity (mojibake repair; the prompts "
                "are well-formed UTF-8)",
            ],
        },
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "python": sys.version.split()[0]},
    }

    # ---- tokenizer: the checkpoint's own file vs the independent Viterbi ----
    # `HuggingfaceTokenizer` supplies the cleaning (`whitespace_clean` after
    # `basic_clean`) and the pad-to-`text_len`; the ids themselves come from the
    # shipped `tokenizer.json` read two independent ways.
    hf = ref_tok.HuggingfaceTokenizer(
        name=args.tokenizer, seq_len=TEXT_LEN, clean="whitespace")
    raw = Tokenizer.from_file(os.path.join(args.tokenizer, "tokenizer.json"))
    spec = RefUnigram(os.path.join(args.tokenizer, "tokenizer.json"))
    print(f"tokenizer: vocab {hf.vocab_size}, unk {spec.unk_id}, "
          f"max piece {spec.max_bytes} bytes, unk_score {spec.unk_score:.4f}", flush=True)

    hf_ids, hf_mask = hf(TOKENIZER_PROMPTS, return_mask=True, add_special_tokens=True)
    assert hf_ids.shape == (len(TOKENIZER_PROMPTS), TEXT_LEN), hf_ids.shape
    lens = hf_mask.gt(0).sum(dim=1).long()
    ids = torch.zeros_like(hf_ids)
    mask = torch.zeros_like(hf_mask)
    drift = 0
    for i, p in enumerate(TOKENIZER_PROMPTS):
        # No prompt here reaches `text_len`, so HF's truncation never engages
        # and the paths can be compared as pure encodings.
        assert int(lens[i]) < TEXT_LEN, f"prompt {i} fills the whole 512-token window"
        cleaned = hf._clean(p)
        mine = spec.encode(cleaned)
        theirs = raw.encode(cleaned).ids
        assert mine == theirs, (
            f"prompt {i}: independent unigram path differs\n  file {theirs[:40]}\n"
            f"  mine {mine[:40]}")
        # `transformers` rebuilds the backend model from the vocab rather than
        # loading it verbatim, and the version installed here gets `unk_id`
        # wrong (2 = `<s>` instead of the file's 3 = `<unk>`). It is an UNK-only
        # divergence, so it cannot touch a real prompt; the goldens follow the
        # FILE, and the divergence is counted rather than hidden.
        transformers_ids = hf_ids[i, :lens[i]].tolist()
        assert len(transformers_ids) == len(mine), (transformers_ids, mine)
        for a, b in zip(transformers_ids, mine):
            if a != b:
                assert b == spec.unk_id, f"prompt {i}: {a} vs {b} is not an unk substitution"
                drift += 1
        n = len(mine)
        ids[i, :n] = torch.tensor(mine)
        mask[i, :n] = 1
        assert hf_ids[i, lens[i]:].eq(0).all(), "pad id is not 0"
        print(f"  prompt {i}: {n:3d} tokens, two paths agree "
              f"({p[:44]!r}{'...' if len(p) > 44 else ''})", flush=True)
    manifest["run"]["transformers_unk_substitutions"] = drift
    if drift:
        print(f"NOTE: transformers {transformers.__version__} substituted a different "
              f"unk id at {drift} position(s); the goldens follow tokenizer.json",
              flush=True)

    save(args.out, "tokens.safetensors", {
        "input_ids": ids.to(torch.float32),
        "attention_mask": mask.to(torch.float32),
        "seq_lens": lens.to(torch.float32),
    }, manifest)

    if args.tokenizer_only:
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"wrote {args.out}/manifest.json (tokenizer only)", flush=True)
        return

    # ---- the encoder -------------------------------------------------------
    t0 = time.time()
    model = t5.umt5_xxl(encoder_only=True, return_tokenizer=False,
                        dtype=torch.float32, device="cpu").eval().requires_grad_(False)
    print(f"built umt5_xxl encoder in {time.time() - t0:.0f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.3f} B params, "
          f"shared_pos={model.shared_pos})", flush=True)
    assert not model.shared_pos, "umt5_xxl must set shared_pos=False"

    t0 = time.time()
    sd = torch.load(args.weights, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(sd, strict=True)
    del sd
    print(f"loaded {args.weights} in {time.time() - t0:.0f}s (strict)", flush=True)

    cfg = {"vocab": 256384, "dim": model.dim, "dim_attn": model.dim_attn,
           "dim_ffn": model.dim_ffn, "heads": model.num_heads,
           "layers": model.num_layers, "buckets": model.num_buckets}
    manifest["run"]["config"] = cfg
    print(f"config: {cfg}", flush=True)

    e_ids, e_mask = hf(ENCODER_PROMPTS, return_mask=True, add_special_tokens=True)
    e_lens = e_mask.gt(0).sum(dim=1).long()
    # The encoder prompts contain no unknown pieces, so the wrapper's ids and
    # the file's are the same tensor; assert it rather than assume it.
    assert torch.equal(e_ids, ids[:len(ENCODER_PROMPTS)].to(e_ids.dtype))
    print(f"encoder batch: B={e_ids.shape[0]}, T={e_ids.shape[1]}, lens={e_lens.tolist()}",
          flush=True)

    # ---- the 24 relative-position tables, and that they really differ -------
    tables = torch.stack([b.pos_embedding.embedding.weight for b in model.blocks])
    assert tables.shape == (cfg["layers"], cfg["buckets"], cfg["heads"]), tables.shape
    worst = min(
        (tables[i] - tables[j]).abs().max().item()
        for i in range(cfg["layers"]) for j in range(i + 1, cfg["layers"]))
    print(f"per-block relative-position tables: {cfg['layers']} of "
          f"{tuple(tables.shape[1:])}, closest pair differs by {worst:.3e}", flush=True)
    assert worst > 1e-3, "the 24 tables are near-identical: shared_pos would be undetectable"

    buckets = bucket_table(TEXT_LEN, cfg["buckets"])
    b0_bias = model.blocks[0].pos_embedding(TEXT_LEN, TEXT_LEN)
    b23_bias = model.blocks[cfg["layers"] - 1].pos_embedding(TEXT_LEN, TEXT_LEN)
    assert b0_bias.shape == (1, cfg["heads"], TEXT_LEN, TEXT_LEN), b0_bias.shape
    for name, l, got in (("b0", 0, b0_bias), ("b23", cfg["layers"] - 1, b23_bias)):
        mine = tables[l][buckets].permute(2, 0, 1).unsqueeze(0)
        agree(f"{name}_position_bias vs the recomputed bucket table", got, mine, tol=1e-12)
    spread = (b0_bias - b23_bias).abs().max().item()
    print(f"  block 0 and block 23 biases differ by max_abs {spread:.3e}", flush=True)
    assert spread > 1e-3, "block 0 and 23 biases coincide: the fixture could not see shared_pos"

    # ---- hooks: one real forward, every tap captured -----------------------
    taps = {}

    def watch(name, module, want_input=False):
        def hook(_m, i, o):
            taps[name] = (i[0] if want_input else o).detach().clone()
        return module.register_forward_hook(hook)

    handles = [watch("embed", model.token_embedding)]
    b0 = model.blocks[0]
    handles += [
        watch("b0_attn_norm", b0.norm1),
        watch("b0_q", b0.attn.q),
        watch("b0_k", b0.attn.k),
        watch("b0_v", b0.attn.v),
        watch("b0_attn_ctx", b0.attn.o, want_input=True),
        watch("b0_attn_out", b0.attn),
        watch("b0_ff_norm", b0.norm2),
        watch("b0_wi0", b0.ffn.gate[0]),
        watch("b0_wi1", b0.ffn.fc1),
        watch("b0_gated", b0.ffn.fc2, want_input=True),
        watch("b0_ff_out", b0.ffn),
    ]
    for l, blk in enumerate(model.blocks):
        handles.append(watch(f"block{l}_out", blk))

    t0 = time.time()
    hidden = model(e_ids, e_mask)
    for h in handles:
        h.remove()
    print(f"masked forward: {time.time() - t0:.0f}s -> {tuple(hidden.shape)}", flush=True)
    assert hidden.shape == (len(ENCODER_PROMPTS), TEXT_LEN, cfg["dim"]), hidden.shape
    # `T5SelfAttention` is x + attn(norm1(x)); the residual is the tap brain
    # calls `attn_res` and is not a module output anywhere.
    taps["b0_attn_res"] = taps["embed"] + taps["b0_attn_out"]

    # ---- self-validation: the reference's own wrapper -----------------------
    # `T5EncoderModel.__call__` tokenizes, forwards and trims to seq_len. It is
    # the path `text2video.py` actually calls, so the taps above are only worth
    # dumping if they reproduce it.
    trimmed = [u[:v] for u, v in zip(hidden, e_lens)]
    enc = t5.T5EncoderModel.__new__(t5.T5EncoderModel)
    enc.text_len, enc.dtype, enc.device = TEXT_LEN, torch.float32, torch.device("cpu")
    enc.model, enc.tokenizer = model, hf
    t0 = time.time()
    wrapper = enc(ENCODER_PROMPTS, torch.device("cpu"))
    print(f"wrapper forward: {time.time() - t0:.0f}s", flush=True)
    for i, (a, b) in enumerate(zip(wrapper, trimmed)):
        assert a.shape == b.shape, (a.shape, b.shape)
        agree(f"wrapper vs manual prompt {i}", a, b, tol=1e-9)

    # ---- the DiT-facing context: zero-padded back to text_len ---------------
    # `wan/modules/model.py:552-558`: the DiT re-pads each trimmed context with
    # `u.new_zeros(text_len - u.size(0), dim)`. The pad rows are EXACTLY ZERO,
    # not the encoder's output at the pad positions.
    context = torch.stack([
        torch.cat([u, u.new_zeros(TEXT_LEN - u.size(0), u.size(1))]) for u in wrapper])
    for i, n in enumerate(e_lens.tolist()):
        assert context[i, n:].abs().max().item() == 0.0
        pad_out = hidden[i, n:].abs().max().item()
        print(f"  prompt {i}: encoder output on pad rows peaks at {pad_out:.3e}; "
              f"the DiT sees zeros there", flush=True)

    # ---- porting.md section 6: does the mask matter? -----------------------
    t0 = time.time()
    unmasked = model(e_ids, None)
    print(f"unmasked forward: {time.time() - t0:.0f}s", flush=True)
    for i, n in enumerate(e_lens.tolist()):
        d = (hidden[i, :n] - unmasked[i, :n]).abs().max().item()
        print(f"  prompt {i}: masked vs unmasked differs by max_abs {d:.3e} on the "
              f"{n} CONTENT rows", flush=True)
        assert d > 1e-3, "the mask would be a no-op, which contradicts text2video.py"

    # ---- self-validation: truncation equivalence ---------------------------
    # Masking the pad KEYS out entirely is the same computation as never having
    # them: nothing in this encoder mixes rows except attention. Proving it here
    # is what lets brain treat the 512 pad as a host-side operation.
    for i, n in enumerate(e_lens.tolist()):
        short = model(e_ids[i:i + 1, :n], None)
        agree(f"truncated(T={n}) vs masked prompt {i}", short[0], hidden[i, :n], tol=1e-5)

    tensors = {
        "input_ids": e_ids.to(torch.float32),
        "attention_mask": e_mask.to(torch.float32),
        "seq_lens": e_lens.to(torch.float32),
        "relative_position_bucket": buckets.to(torch.float32),
        "pos_emb_tables": tables,
        "b0_position_bias": b0_bias[0],
        "b23_position_bias": b23_bias[0],
        "last_hidden_state": hidden,
        "last_hidden_state_unmasked": unmasked,
        "context_padded": context,
    }
    tensors.update({k: v for k, v in taps.items()})
    save(args.out, "encoder.safetensors", tensors, manifest)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"wrote {args.out}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
