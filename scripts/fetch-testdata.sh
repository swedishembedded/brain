#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Populate the gitignored `testdata/` tree that parity / integration tests read.
#
# Design goals:
#   * Idempotent — only ever fetches files that are NOT already present.
#   * Offline-first — prefers a local mirror directory (fast, no network); falls
#     back to a URL download only when a mirror file is absent and a URL is known.
#   * Zero extra disk on the same filesystem — mirror files are HARD-LINKED into
#     `testdata/` when possible (instant, shares blocks), else copied.
#   * Never bakes an absolute path into the source tree: tests resolve their
#     inputs from `$BRAIN_TESTDATA` (default `<repo>/testdata`); this script is the
#     ONE place a machine-specific mirror location may appear, and it is an
#     overridable variable, not source code.
#
# Layout produced (a proper tree, mirroring each model's asset namespace):
#   testdata/asr/nemotron/hf/…         Nemotron 3.5 ASR 0.6B HF checkpoint
#   testdata/asr/qwen-asr/hf/…         Qwen3-ASR 1.7B HF checkpoint
#   testdata/asr/golden/{nemotron,qwen_encoder,qwen_decode,frontend}/…  dumped goldens
#   testdata/asr/audio/…               test waveforms
#
# Usage:
#   make fetch/testdata                      # populate everything missing
#   BRAIN_ASR_MIRROR=/path make fetch/testdata
#   BRAIN_TESTDATA=/scratch/td make fetch/testdata
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
DEST="${BRAIN_TESTDATA:-$ROOT/testdata}"

# Local mirrors of the raw assets on a dev box (override each with its env var).
# These are the "copy from the local absolute path" sources the tests used to
# hardcode — the ONE place a machine-specific path may still appear.
ASR_MIRROR="${BRAIN_ASR_MIRROR:-/data/workspace/resources/asr}"
VL_MIRROR="${BRAIN_VL_MIRROR:-/data/workspace/resources/vl}"
TTS_MIRROR="${BRAIN_TTS_MIRROR:-/data/workspace/tmp/qwen3-tts-resources}"
GOLDEN_MIRROR="${BRAIN_GOLDEN_MIRROR:-/data/workspace/resources/brain-goldens}"
SAM2_MIRROR="${BRAIN_SAM2_MIRROR:-/data/workspace/resources/sam2}"

added=0 skipped=0 missing=0

# _link_from <mirror-root> <mirror-subdir> <dest-subdir> — hard-link (or copy)
# every file under <mirror-root>/<mirror-subdir> into $DEST/<dest-subdir>,
# creating only what's missing.
_link_from() {
  local root="$1" sub_src="$2" sub_dst="$3"
  local src="$root/$sub_src" dst="$DEST/$sub_dst"
  if [ ! -d "$src" ]; then
    echo "  · $sub_dst: mirror '$src' absent — skipping (point its BRAIN_*_MIRROR at a copy, or add a URL)"
    missing=$((missing + 1))
    return 0
  fi
  mkdir -p "$dst"
  while IFS= read -r -d '' f; do
    local rel="${f#"$src"/}"
    local out="$dst/$rel"
    if [ -e "$out" ]; then
      skipped=$((skipped + 1))
      continue
    fi
    mkdir -p "$(dirname "$out")"
    ln "$f" "$out" 2>/dev/null || cp "$f" "$out"
    added=$((added + 1))
  done < <(find "$src" -type f -print0)
  echo "  ✓ $sub_dst"
}
asr_tree() { _link_from "$ASR_MIRROR" "$1" "$2"; }
golden_tree() { _link_from "$GOLDEN_MIRROR" "$1" "$2"; }
vl_tree()  { _link_from "$VL_MIRROR"  "$1" "$2"; }
tts_tree() { _link_from "$TTS_MIRROR" "$1" "$2"; }
sam2_tree() { _link_from "$SAM2_MIRROR" "$1" "$2"; }

echo "brain: populating testdata at $DEST"
echo "       mirrors: asr=$ASR_MIRROR vl=$VL_MIRROR tts=$TTS_MIRROR golden=$GOLDEN_MIRROR sam2=$SAM2_MIRROR"

# --- ASR (Nemotron 3.5 ASR, Qwen3-ASR) --------------------------------------
asr_tree "nemotron/hf"     "asr/nemotron/hf"
asr_tree "qwen3-asr/hf"    "asr/qwen-asr/hf"
asr_tree "golden"          "asr/golden"
asr_tree "audio"           "asr/audio"

# --- Model-parity goldens (lfm / qwen encoder / vae / zimage) ----------------
# Small dumped fixtures the staged parity tests read (regenerable from the
# reference checkpoints via tools/*_dump_reference.py; never committed).
golden_tree "lfm"          "golden/lfm"
golden_tree "qwen"         "golden/qwen"
golden_tree "vae"          "golden/vae"
golden_tree "zimage"       "golden/zimage"

# --- SAM 2.1 (promptable segmentation, image path) ---------------------------
# The reference CHECKPOINTS (+ their hydra yaml) only. The stage goldens
# (`{input,trunk,neck,case_*}.safetensors`, ~1 GB per variant) are NOT mirrored:
# regenerate them next to the checkpoint with
#   python3 tools/sam2_dump_reference.py --code <sam2 repo> \
#       --config testdata/sam2/hiera-tiny/sam2.1_hiera_t.yaml \
#       --ckpt   testdata/sam2/hiera-tiny/sam2.1_hiera_tiny.pt \
#       --out    testdata/sam2/hiera-tiny
# `crates/sam2/tests/parity.rs` skips itself while they are absent.
sam2_tree "weights/sam2.1-hiera-large" "sam2/hiera-large"
sam2_tree "weights/sam2.1-hiera-tiny"  "sam2/hiera-tiny"

# --- Vision-language (FastVLM, Moondream3, Qwen3-VL) -------------------------
vl_tree  "fastvlm/hf"      "vl/fastvlm/hf"
vl_tree  "moondream3/hf"   "vl/moondream3/hf"
vl_tree  "qwen3-vl"        "vl/qwen3-vl"
vl_tree  "parity"          "vl/parity"

# --- Audio codec / TTS / speaker (Qwen3-TTS) --------------------------------
tts_tree "ckpt"            "tts/ckpt"
tts_tree "dumps"           "tts/dumps"
# loose reference files at the mirror root (e.g. the voice-clone example wav)
if [ -d "$TTS_MIRROR" ]; then
  for f in "$TTS_MIRROR"/*.wav; do
    [ -e "$f" ] || continue
    out="$DEST/tts/$(basename "$f")"
    [ -e "$out" ] && { skipped=$((skipped + 1)); continue; }
    mkdir -p "$DEST/tts"; ln "$f" "$out" 2>/dev/null || cp "$f" "$out"; added=$((added + 1))
  done
fi

echo "brain: testdata ready — $added new, $skipped already present, $missing groups unavailable"
