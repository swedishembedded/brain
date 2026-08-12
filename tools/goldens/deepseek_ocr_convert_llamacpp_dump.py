#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Convert llama.cpp graph-eval tensor dumps into brain's REAL-weight DeepSeek-OCR goldens.

`deepseek_ocr_dump_reference.py` produces the CHECKPOINT-FREE tiny golden (a
random miniature pipeline, dumped through PyTorch). This script produces the
complementary one: per-stage taps of the **real Q8_0 checkpoint**, captured from
llama.cpp - the upstream reference implementation the GGUF format targets -
running against the very same
`ggml-org/DeepSeek-OCR-GGUF/{DeepSeek-OCR-Q8_0.gguf,mmproj-DeepSeek-OCR-Q8_0.gguf}`
files brain imports. Both sides therefore dequantize the *same* Q8_0 blocks, so
the expected agreement is fp32-tight, not merely "close".

Files written under `--out` (default `testdata/deepseek-ocr/real`):

  vision.safetensors   the SAM ViT-B tower at its production 1024x1024 shape:
                       every one of the 12 blocks' outputs, the neck+compressor
                       output, and CLIP's assembled input rows.
  decoder.safetensors  the MoE decoder on a two-token text-only prompt: token
                       embeddings, per-layer attention output and residual
                       stream, the full router internals of every MoE layer,
                       the final norm and the final logits over the whole vocab.
  manifest-real.json   shapes + sha256 per tensor, the capture commands, and the
                       recorded semantic findings.

  (`testdata/` is gitignored; these are re-derived, never committed.)

Reproducing the INPUT dumps
---------------------------
The two source directories this script reads are raw `NNNNN_<name>.f32` blobs
plus a `manifest.jsonl` (one JSON object per tensor: `index`, `name`, `op`,
`ne`, `file`), written by llama.cpp's graph-eval callback. Upstream's built-in
callback prints a TRUNCATED preview; a full dump needs `common/debug.cpp`'s
`cb_eval` patched to write each tensor's complete f32 data to a file. With that
patch, in a llama.cpp build tree:

    # SAM tower (mmproj / vision):
    ./build/bin/llama-mtmd-debug -m <lm.gguf> --mmproj <mmproj.gguf> \
        -p encode --image gray -n 1024

    # decoder (language model only, no image, no CLIP):
    ./build/bin/llama-eval-callback -m <lm.gguf> --prompt "Hello" -n 1

Known upstream limitation, deliberately NOT worked around here: with the
debug callback active, the CLIP-block half of the vision graph segfaults
immediately after the cls-token concat (`node_848` below), reproducibly and
independently of the dump patch. The production path (`llama-mtmd-cli`) runs
that same graph fine, so it is a defect in llama.cpp's debug tooling for this
model, not in the model or the weights. Consequently there are NO CLIP-internal
taps and no image+decoder end-to-end tap in this golden - `clip_input_tokens`
is exactly as far as the capture gets, and it is placed here precisely so the
SAM -> compressor -> CLIP-input handoff can still be proven.

The vision input, verified against llama.cpp's own source
--------------------------------------------------------
`-p encode --image gray -n 1024` does NOT go through image preprocessing.
`mtmd-debug.cpp` builds a `[1024][1024*3]` vector filled with a constant `0.5f`
and hands it to `mtmd_debug_encode_image` (`tools/mtmd/mtmd.cpp`), which copies
it verbatim into a `clip_image_f32` and calls `clip_image_encode` - there is no
`clip_image_preprocess` call and no mean/std normalization on this path. The
network therefore sees pixel values of exactly 0.5 everywhere, which is NOT what
DeepSeek-OCR's real preprocessing would produce for a gray image (that
normalizes with mean=std=(0.5,0.5,0.5), mapping gray to 0.0). A consumer must
feed brain's SAM tower a constant 0.5 tensor directly, with no normalization
step. Because the fill is spatially and channel-wise constant, its HWC-vs-CHW
layout is immaterial - only the scalar matters. The OUTPUT is emphatically not
constant (SAM's learned absolute position embedding and its decomposed
relative-position bias both break translation invariance), so this remains a
real test of the architecture against the real weights.

Layouts, verified from the bytes rather than assumed
----------------------------------------------------
ggml's `ne` is fastest-axis-first, so a row-major (safetensors) shape is `ne`
reversed with trailing 1s dropped. That much is mechanical; what is NOT is which
axis carries what, and it differs between the transformer-row tensors and the
conv-graph tensors in the same dump:

  * `sam_layer_out-N`  ne=[768,64,64] -> [64,64,768] = (H, W, C), channel
    fastest. Confirmed by correlation structure: adjacent patches' 768-vectors
    correlate at 0.997 while opposite corners correlate at 0.71.
  * `sam_output`       ne=[16,16,1024] -> [1024,16,16] = (C, H, W), i.e. NCHW,
    channel SLOWEST. Same test, opposite answer (0.998 adjacent under NCHW; the
    NLC reading gives 0.27, i.e. noise).
  * `node_848`         ne=[1024,257] -> [257,1024] = 257 rows of width 1024.

The last of these is checked structurally as well, and the check pinned a fact:
rows 1..=256 of `node_848` are **bit-identical** (max abs difference exactly
0.0) to `sam_output` reinterpreted NCHW -> NLC. So CLIP's input assembly is
`concat(class_embd_row, flatten_nlc(sam_output))` with the class token at row 0
and NO position embedding and NO `pre_ln` applied yet - those come after the
concat, inside the (uncapturable) CLIP graph. `input_cls_row` is split out as
its own tensor so an importer can compare it directly against the mmproj's
`v.class_embd`.

Router semantics, now confirmed on the REAL weights
---------------------------------------------------
`ffn_moe_probs-N` sums to 1.0 across all 64 routed experts (plain softmax, no
sigmoid scoring), while `ffn_moe_weights-N` - the gate values the layer actually
multiplies its expert outputs by - sums to between 0.31 and 0.82 across the 6
selected experts and equals the corresponding `ffn_moe_probs` entries exactly.
That is `norm_topk_prob = false` and `routed_scaling_factor = 1.0`, observed
rather than inferred from an absent GGUF key.

Usage
-----
    tools/goldens/deepseek_ocr_convert_llamacpp_dump.py \
        --sam-dump <dir> --decoder-dump <dir> [--out testdata/deepseek-ocr/real]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from safetensors.numpy import save_file

# The exact token ids llama.cpp's own tokenizer produced for the prompt
# "Hello" under this checkpoint (0 = BOS), taken from its tokenizer log. Pinned
# as data so a consumer replays them directly and tokenizer correctness stays
# out of scope for a decoder-parity test.
DECODER_TOKENS = [0, 19923]

# The constant the vision capture feeds as pixel values (see the module
# docstring: no preprocessing runs on this path).
VISION_INPUT_FILL = 0.5


def read_dump(path):
    """A dump directory -> {name: (ne, np.ndarray of f32)}."""
    manifest = os.path.join(path, "manifest.jsonl")
    if not os.path.isfile(manifest):
        sys.exit(f"no manifest.jsonl under {path}")
    out = {}
    with open(manifest) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data = np.fromfile(os.path.join(path, rec["file"]), dtype="<f4")
            ne = [int(n) for n in rec["ne"]]
            want = int(np.prod(ne))
            if data.size != want:
                sys.exit(f"{rec['name']}: {data.size} floats on disk, ne {ne} wants {want}")
            out[rec["name"]] = (ne, data)
    return out


def row_major(ne):
    """ggml `ne` (fastest axis first) -> a row-major shape, trailing 1s dropped."""
    shape = [n for n in reversed(ne)]
    while len(shape) > 1 and shape[0] == 1:
        shape = shape[1:]
    return shape


def as_tensor(entry):
    ne, data = entry
    return data.reshape(row_major(ne))


def build_vision(src):
    """SAM tower taps, plus the CLIP-input handoff."""
    out = {}
    for layer in range(12):
        out[f"sam_blk{layer:02d}_out"] = as_tensor(src[f"sam_layer_out-{layer}"])
    out["sam_output"] = as_tensor(src["sam_output"])
    tokens = as_tensor(src["node_848"])
    out["clip_input_cls"] = tokens[0].copy()
    out["clip_input_tokens"] = tokens.copy()
    out["input_fill"] = np.asarray([VISION_INPUT_FILL], dtype=np.float32)

    # Self-checks. These are the assertions that make the layout claims in the
    # module docstring facts rather than restatements of an intention.
    blk0 = out["sam_blk00_out"]
    assert blk0.shape == (64, 64, 768), blk0.shape
    near = np.corrcoef(blk0[10, 10], blk0[10, 11])[0, 1]
    far = np.corrcoef(blk0[0, 0], blk0[63, 63])[0, 1]
    assert near > 0.99 > far, f"sam block output is not (H,W,C): near {near} far {far}"

    sam_out = out["sam_output"]
    assert sam_out.shape == (1024, 16, 16), sam_out.shape
    near = np.corrcoef(sam_out[:, 5, 5], sam_out[:, 5, 6])[0, 1]
    assert near > 0.99, f"sam_output is not NCHW: adjacent-column correlation {near}"

    # The handoff: CLIP's spatial rows ARE the compressor output, flattened
    # NCHW -> NLC, with no transform in between. Bit-identical, not merely close.
    nlc = sam_out.reshape(1024, 256).T
    delta = float(np.abs(nlc - out["clip_input_tokens"][1:]).max())
    assert delta == 0.0, f"clip_input_tokens[1:] != flatten_nlc(sam_output): max abs {delta}"

    # The output must not be spatially uniform, or the whole capture would be
    # vacuous for a constant input.
    spread = float(sam_out.std(axis=(1, 2)).mean())
    assert spread > 1e-4, f"sam_output is spatially flat ({spread}); the test would prove nothing"
    return out


def build_decoder(src):
    """Decoder taps for the two-token text-only prompt.

    Every tensor here is stored `(positions, ...)`, positions FIRST and always
    present even when there is only one. That is deliberate: ggml only needs the
    final position's activations once the graph reaches the output head, so the
    dump's own sequence extent shrinks from 2 to 1 partway down the stack
    (`l_out-11` onwards, and layer 11's router). Letting `row_major` drop a
    leading 1 would make some layers 2-D and others 1-D for the same quantity,
    and a consumer would have to special-case the tail. `rows` reshapes off the
    *trailing* (fastest) ggml axes instead, so the leading position axis is
    whatever is left over.
    """

    def rows(name, *trailing):
        _ne, data = src[name]
        return data.reshape(-1, *trailing)

    out = {"tokens": np.asarray(DECODER_TOKENS, dtype=np.float32)}
    d_model = src["embd"][0][0]
    out["embd"] = rows("embd", d_model)
    for layer in range(12):
        out[f"attn_out_L{layer:02d}"] = rows(f"attn_out-{layer}", d_model)
        out[f"l_out_L{layer:02d}"] = rows(f"l_out-{layer}", d_model)
        if f"ffn_moe_probs-{layer}" not in src:
            continue  # layer 0 is dense
        n_experts = src[f"ffn_moe_probs-{layer}"][0][0]
        top_k = src[f"ffn_moe_topk-{layer}"][0][0]
        moe_ff = src[f"ffn_moe_gate-{layer}"][0][0]
        out[f"moe_probs_L{layer:02d}"] = rows(f"ffn_moe_probs-{layer}", n_experts)
        # ggml stores the top-k selection as I32; the dump wrote the same words
        # back out as f32, so the values are exact small integers.
        topk = rows(f"ffn_moe_topk-{layer}", top_k)
        assert np.all(topk == np.floor(topk)), f"layer {layer} top-k indices are not integral"
        out[f"moe_topk_L{layer:02d}"] = topk
        # ne [1, top_k, positions] -- the fastest axis is the degenerate one.
        out[f"moe_weights_L{layer:02d}"] = rows(f"ffn_moe_weights-{layer}", top_k)
        out[f"moe_gate_L{layer:02d}"] = rows(f"ffn_moe_gate-{layer}", top_k, moe_ff)
    out["result_norm"] = rows("result_norm", d_model)
    out["result_output"] = rows("result_output", src["result_output"][0][0])

    findings = {}
    # The raw-gate fact, measured on the real weights.
    sums = []
    for layer in range(1, 12):
        probs = out[f"moe_probs_L{layer:02d}"]
        weights = out[f"moe_weights_L{layer:02d}"]
        topk = out[f"moe_topk_L{layer:02d}"].astype(np.int64)
        assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5), f"layer {layer} probs do not sum to 1"
        for row in range(probs.shape[0]):
            gathered = probs[row][topk[row]]
            assert np.allclose(gathered, weights[row], atol=0, rtol=0), (
                f"layer {layer} row {row}: the used gate is not the RAW softmax prob "
                f"({gathered} vs {weights[row]}) - routed_scaling is not 1.0, or the "
                f"gate is renormalized"
            )
            s = float(weights[row].sum())
            assert s < 0.999, f"layer {layer} row {row}: gate sum {s} - this would be renormalized"
            sums.append(s)
        # The selection really is the top-k of the softmax, not some other rule.
        assert set(topk[0].tolist()) == set(np.argsort(-probs[0])[: topk.shape[-1]].tolist())
    findings["router_gate_sums"] = {"min": min(sums), "max": max(sums), "n": len(sums)}
    findings["norm_topk_prob"] = False
    findings["routed_scaling_factor"] = 1.0
    findings["n_routed_experts"] = int(out["moe_probs_L01"].shape[-1])
    findings["top_k"] = int(out["moe_topk_L01"].shape[-1])
    findings["greedy_next_token"] = int(np.asarray(out["result_output"]).reshape(-1).argmax())
    return out, findings


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).hexdigest()


def write(path, tensors):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in tensors.items()}, path)
    return {k: {"shape": list(v.shape), "sha256": digest(v)} for k, v in tensors.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sam-dump", required=True, help="llama-mtmd-debug '-p encode' dump directory")
    ap.add_argument("--decoder-dump", required=True, help="llama-eval-callback dump directory")
    ap.add_argument("--out", default="testdata/deepseek-ocr/real")
    args = ap.parse_args()

    vision, vfind = build_vision(read_dump(args.sam_dump)), {}
    decoder, dfind = build_decoder(read_dump(args.decoder_dump))

    manifest = {
        "source": "llama.cpp graph-eval dumps of ggml-org/DeepSeek-OCR-GGUF (Q8_0 pair)",
        "capture": {
            "vision": "llama-mtmd-debug -m <lm.gguf> --mmproj <mmproj.gguf> -p encode --image gray -n 1024",
            "decoder": 'llama-eval-callback -m <lm.gguf> --prompt "Hello" -n 1',
            "note": "the debug callback segfaults inside the CLIP block graph; there are no CLIP-internal taps",
        },
        "vision_input": {
            "fill": VISION_INPUT_FILL,
            "shape": [1, 3, 1024, 1024],
            "preprocessing": "none - mtmd_debug_encode_image feeds the constant buffer straight to clip_image_encode",
        },
        "decoder_input": {"prompt": "Hello", "tokens": DECODER_TOKENS},
        "findings": {**vfind, **dfind},
        "files": {},
    }
    manifest["files"]["vision.safetensors"] = write(os.path.join(args.out, "vision.safetensors"), vision)
    manifest["files"]["decoder.safetensors"] = write(os.path.join(args.out, "decoder.safetensors"), decoder)
    with open(os.path.join(args.out, "manifest-real.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {len(vision)} vision + {len(decoder)} decoder tensors under {args.out}")
    print(json.dumps(manifest["findings"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
