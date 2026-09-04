#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump the Qwen3-TTS Talker logit golden for `crates/qwen3tts/tests/parity.rs`.

The reference is the UPSTREAM `qwen-tts` package's `Qwen3TTSTalkerModel` (see
`qwen3tts_ref.py` for why it cannot come from a released `transformers`), driven
in fp32 on the CPU against the real `Qwen/Qwen3-TTS-12Hz-0.6B-Base` weights.

Exactly which sub-computation
-----------------------------
`TalkerModel::logits_all` on the Rust side is the Talker decoder and its untied
output head and nothing else, so this dumper reproduces precisely that:

    logits = codec_head( talker.model( inputs_embeds = codec_embedding(ids) ) )

with `position_ids = None` (so the reference builds `arange(T)` and expands it
to all three M-RoPE sections), `attention_mask = None` (so it builds a plain
causal mask), and `use_cache = False`. No text projection, no MTP code
predictor, no speech tokenizer, no prompt assembly - none of which
`logits_all` models. `inputs_embeds` is passed rather than `input_ids` because
`Qwen3TTSTalkerModel.forward` embeds through `self.embed_tokens`, an attribute
that only exists after `set_input_embeddings`; the table the model really owns
is `codec_embedding`, and the reference's own generation path likewise calls
`get_input_embeddings()` itself and hands the result in as `inputs_embeds`.

Two boundary details this dumper asserts rather than assumes, because both are
places brain's port could silently disagree:

  * M-RoPE collapses to plain half-split RoPE here. The Talker config declares
    `rope_scaling = {interleaved: true, mrope_section: [24,20,20]}`, but a pure
    codec stream gives all three sections the SAME position index, so
    `apply_multimodal_rotary_pos_emb` reduces to `q*cos + rotate_half(q)*sin`
    over `cat(freqs, freqs)` - which is what brain's shared Qwen3 backbone
    applies. Checked bit-for-bit against the reference's own two functions.
  * `sliding_window` is null, so the mask really is full causal.

Weights are the `talker.model.*` / `talker.codec_head.weight` subset of the
released checkpoint, cast bf16 -> fp32 (exact, no rounding) and loaded with
`strict=True`. brain's importer dequantizes the same bf16 blocks to fp32 too,
so the two sides evaluate identical numbers and the expected agreement is
fp32-tight rather than merely "close".

The token sequence, and why it is synthetic
-------------------------------------------
`--tokens` positions of codebook-0 ids: `codec_bos_id` at position 0 followed by
ids drawn from the acoustic range `[0, codebook_size)` by the small LCG below.
That is the shape of a real teacher-forced codebook-0 stream, and it is
byte-identically reproducible from `--seed` alone on any machine holding the
checkpoint - no audio clip, no codec, no network.

Real ids off the 12 Hz speech tokenizer were considered and deliberately not
used. In production the Talker's input row for a frame is the SUM of sixteen
embeddings (`codec_embedding(cb0)` plus the code predictor's fifteen residual
tables), so the moment this gate isolates `codec_embedding(cb0)` alone the input
hidden states are off the production distribution no matter where the ids come
from. What is left to measure is decoder and head NUMERICS, for which the ids
only have to be in-vocab and varied - and a self-contained deterministic
sequence buys reproducibility that a downloaded clip does not.

Output (`<out>/talker_ref/`, `<out>` defaults to `testdata/tts/dumps`):

  tokens.u32   flat `[T]` little-endian u32 - the ids that were fed in
  logits.f32   flat `[T, vocab]` little-endian f32, row-major
  meta.json    shapes, the token sequence, the `source` provenance block, and
               the measured logit statistics

Both `.bin`-style files are bare little-endian arrays with no count prefix and
no header: the whole file is data and the shape comes from context, which is the
layout `parity.rs`'s `read_u32`/`read_f32` parse (`T = tokens.len()`, vocab
fixed at 3072). `testdata/` is gitignored; this dump is re-derived, never
committed.

Environment
-----------
`qwen3tts_ref.bootstrap()` fetches `qwen-tts==0.1.1` and installs its pinned
`transformers==4.57.3` (plus `huggingface_hub==0.35.3`, `tokenizers==0.22.1`)
into a private directory - it never touches the ambient interpreter's packages
and needs no `trust_remote_code`, because the published checkpoint carries no
remote modelling code. Point `$QWEN_TTS_REF_DIR` at that directory to control
where it lands. Only `torch`, `numpy` and `safetensors` have to be present
already; CPU-only torch is enough and is what this was dumped with.

Usage
-----
    python3 tools/goldens/qwen3tts_dump_talker_reference.py \
        --ckpt testdata/tts/ckpt/Qwen3-TTS-12Hz-0.6B-Base \
        [--out testdata/tts/dumps] [--tokens 64] [--seed 20260904]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_source import source_block  # noqa: E402
from qwen3tts_ref import load_talker  # noqa: E402

CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TALKER_PREFIX = "talker.model."
HEAD_TENSOR = "talker.codec_head.weight"
# The acoustic codebook the Talker's codebook-0 ids come from. The Talker vocab
# is wider (3072) because it also carries the special ids - bos/eos/pad/think -
# which live above this range and are not drawn as ordinary frames.
CODEBOOK_SIZE = 2048
# Floor on the smallest top-1 margin the dumped logits may carry. `parity.rs`
# asserts top-1 agreement at EVERY position, so a position whose best and
# second-best logits sit closer together than the two implementations can agree
# would turn that assertion into a coin flip. Measured agreement between this
# dump and brain's decoder is max-abs 5e-4 over all 64x3072 logits, so a 0.01
# floor leaves an order of magnitude of headroom on each side of the gap. If a
# seed trips this, pick another rather than relaxing the floor.
MIN_TOP1_MARGIN = 0.01


def lcg_ids(count, seed, modulus):
    """`count` ids in `[0, modulus)` from a 32-bit LCG (the glibc multiplier).

    An explicit recurrence rather than `numpy.random`: the sequence is then a
    property of this file and this seed alone, not of whichever RNG version the
    dumping machine happened to install."""
    out = []
    state = seed & 0xFFFFFFFF
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0xFFFFFFFF
        # The high bits of an LCG are the well-mixed ones.
        out.append(int((state >> 16) % modulus))
    return out


def check_mrope_collapses(mod, model, seq_len, head_dim, mrope_section):
    """Assert that M-RoPE here IS plain half-split RoPE, using the reference's
    own `apply_multimodal_rotary_pos_emb` and `rotate_half`.

    brain applies the single-section rotation; this is the fact that makes that
    correct for a codec stream, and it is cheap enough to re-derive on every
    dump instead of trusting a comment."""
    import torch

    position_ids = torch.arange(seq_len)[None, None, :].expand(3, 1, -1)
    dummy = torch.zeros(1, seq_len, model.config.hidden_size)
    cos, sin = model.rotary_emb(dummy, position_ids)
    q = torch.randn(1, 2, seq_len, head_dim, generator=torch.Generator().manual_seed(0))
    got, _ = mod.apply_multimodal_rotary_pos_emb(q, q, cos, sin, mrope_section, True)
    # The single-section rotation: section 0's cos/sin, unsqueezed over heads.
    want = q * cos[0].unsqueeze(1) + mod.rotate_half(q) * sin[0].unsqueeze(1)
    delta = float((got - want).abs().max())
    if delta != 0.0:
        raise SystemExit(
            f"M-RoPE does not collapse to half-split RoPE (max abs {delta}); brain's "
            "shared Qwen3 rotation would not be parity-equivalent for this checkpoint"
        )
    return delta


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, help="Qwen3-TTS-12Hz-*-Base directory")
    ap.add_argument("--out", default=os.path.join("testdata", "tts", "dumps"),
                    help="dump root; the golden lands in <out>/talker_ref")
    ap.add_argument("--tokens", type=int, default=64, help="sequence length T")
    ap.add_argument("--seed", type=int, default=20260904, help="LCG seed for the ids")
    args = ap.parse_args()

    if args.tokens < 2:
        raise SystemExit("--tokens must be at least 2 for the gate to exercise attention")

    config_class, model_class, mod = load_talker()
    import torch

    weights = os.path.join(args.ckpt, "model.safetensors")
    with open(os.path.join(args.ckpt, "config.json")) as fh:
        root = json.load(fh)
    cfg = config_class(**root["talker_config"])
    # Eager attention: the mask is then an explicit additive tensor and the
    # softmax is the documented one, with no kernel-dispatch variability
    # between torch builds to explain away in a parity number.
    cfg._attn_implementation = "eager"
    if cfg.sliding_window is not None:
        raise SystemExit(
            f"talker sliding_window={cfg.sliding_window}: the reference would build a "
            "windowed mask, which is not what `TalkerModel::logits_all` computes"
        )

    from transformers.modeling_utils import no_init_weights

    with no_init_weights():  # every parameter is overwritten below
        model = model_class(cfg)
    model.eval()

    from safetensors import safe_open

    state, head = {}, None
    with safe_open(weights, "pt") as fh:
        for key in fh.keys():
            if key.startswith(TALKER_PREFIX):
                state[key[len(TALKER_PREFIX):]] = fh.get_tensor(key).float()
            elif key == HEAD_TENSOR:
                head = fh.get_tensor(key).float()
    if head is None:
        raise SystemExit(f"{weights} has no {HEAD_TENSOR}")
    model.load_state_dict(state, strict=True)

    rope_delta = check_mrope_collapses(
        mod, model, args.tokens, cfg.head_dim, cfg.rope_scaling["mrope_section"]
    )

    tokens = [int(cfg.codec_bos_id)] + lcg_ids(args.tokens - 1, args.seed, CODEBOOK_SIZE)
    if max(tokens) >= cfg.vocab_size:
        raise SystemExit(f"token {max(tokens)} is outside the talker vocab {cfg.vocab_size}")
    ids = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        # The one boundary this whole golden is about: embed, decode, project.
        inputs_embeds = model.codec_embedding(ids)
        hidden = model(inputs_embeds=inputs_embeds, use_cache=False).last_hidden_state
        logits = torch.nn.functional.linear(hidden, head)[0]

    if tuple(logits.shape) != (args.tokens, cfg.vocab_size):
        raise SystemExit(f"logits {tuple(logits.shape)} != [{args.tokens}, {cfg.vocab_size}]")
    logits = logits.numpy().astype(np.float32)
    # A flat head would make top-1 agreement a coin flip rather than a test.
    margins = np.sort(logits, axis=-1)
    margin = float((margins[:, -1] - margins[:, -2]).min())
    if margin < MIN_TOP1_MARGIN:
        raise SystemExit(
            f"smallest top-1 margin is {margin}, under the {MIN_TOP1_MARGIN} floor: the argmax "
            "at some position is a near tie and top-1 agreement would not be a meaningful "
            "assertion. Re-dump with a different --seed."
        )

    out_dir = os.path.join(args.out, "talker_ref")
    os.makedirs(out_dir, exist_ok=True)
    np.asarray(tokens, dtype="<u4").tofile(os.path.join(out_dir, "tokens.u32"))
    np.ascontiguousarray(logits, dtype="<f4").tofile(os.path.join(out_dir, "logits.f32"))
    meta = {
        "tokens": tokens,
        "shape": [args.tokens, int(cfg.vocab_size)],
        "seed": args.seed,
        "dtype": "weights bf16 -> fp32, forward fp32, cpu",
        "attn_implementation": "eager",
        "mrope_vs_half_split_rope_max_abs": rope_delta,
        "greedy_next_tokens": [int(v) for v in logits.argmax(axis=-1)],
        "logit_stats": {
            "min": float(logits.min()),
            "max": float(logits.max()),
            "std": float(logits.std()),
            "min_top1_margin": margin,
        },
        "source": source_block(
            checkpoint=CHECKPOINT,
            files=[weights],
            identity={
                "num_hidden_layers": cfg.num_hidden_layers,
                "hidden_size": cfg.hidden_size,
                "head_dim": cfg.head_dim,
                "num_attention_heads": cfg.num_attention_heads,
                "num_key_value_heads": cfg.num_key_value_heads,
                "intermediate_size": cfg.intermediate_size,
                "vocab_size": cfg.vocab_size,
            },
        ),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    print(f"talker golden: [{args.tokens}, {cfg.vocab_size}] logits -> {out_dir}")
    print(f"  logit std {meta['logit_stats']['std']:.4f}, min top-1 margin {margin:.4f}")
    print(f"  m-rope vs half-split rope: max abs {rope_delta}")


if __name__ == "__main__":
    main()
