#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

#
# Populate the gitignored `testdata/` tree that parity / integration tests read.
#
# Design goals:
#   * Idempotent - only ever fetches files that are NOT already present.
#   * Offline-first - reads a local mirror directory (fast, no network). There is
#     currently no URL-download fallback; a missing mirror is reported and
#     skipped (see the `missing` counter at the end of a run). `brain fetch`
#     (`crates/cli/src/fetch.rs`) is the network-download path for actual model
#     weights - this script is test-fixture plumbing, not a general fetcher.
#   * Zero extra disk on the same filesystem - mirror files are HARD-LINKED into
#     `testdata/` when possible (instant, shares blocks), else copied.
#   * Never bakes an absolute path into the source tree: tests resolve their
#     inputs from `$BRAIN_TESTDATA` (default `<repo>/testdata`); this script is the
#     ONE place a machine-specific mirror location may appear, and it is an
#     overridable variable, not source code.
#   * `testdata/` holds test INPUTS AND GOLDENS ONLY - never a `.git` directory
#     (stripped unconditionally, see `_link_from` below), never upstream
#     source/notebooks/docs a test doesn't read (see each tree's extra-exclude),
#     and, with two named exceptions below, never a model checkpoint - those go
#     to the model store, addressed by fully-qualified `<vendor>/<repo>` name,
#     the same place `brain fetch` writes and `brain_testutil::model_dir`
#     resolves - this separation is what an earlier testdata-layout audit
#     of this script recommended. The exceptions are the antelopev2 ONNX pair
#     and the Qwen3-TTS checkpoint, whose tests resolve them through
#     `brain_testutil::testdata` and so must find them under `testdata/`.
#
# Layout produced (a proper tree, mirroring each model's asset namespace):
#   testdata/asr/golden/{nemotron,qwen_encoder,qwen_decode,frontend}/…  dumped goldens
#   testdata/asr/audio/…               test waveforms
#   testdata/face/antelopev2/…         the two released insightface ONNX graphs
#   testdata/tts/ckpt/…                the Qwen3-TTS base + speech-tokenizer ckpt
#
# The checkpoints the model crates import are NOT produced here: they are
# reported present-or-absent in the model store (`<models-dir>/<vendor>/<repo>`,
# resolved like brain_modelstore::default_root: BRAIN_MODELS_DIR, else
# $XDG_DATA_HOME/brain/models, else $HOME/.local/share/brain/models) and
# `brain fetch <vendor>/<repo>` is what puts one there.
#
# Usage:
#   make fetch/testdata                      # populate everything missing
#   BRAIN_ASR_MIRROR=/path make fetch/testdata
#   BRAIN_TESTDATA=/scratch/td make fetch/testdata
#   BRAIN_MODELS_DIR=/scratch/models make fetch/testdata
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
DEST="${BRAIN_TESTDATA:-$ROOT/testdata}"
if [ -n "${BRAIN_MODELS_DIR:-}" ]; then
  MODELS_DIR="$BRAIN_MODELS_DIR"
elif [ -n "${XDG_DATA_HOME:-}" ]; then
  MODELS_DIR="$XDG_DATA_HOME/brain/models"
else
  MODELS_DIR="$HOME/.local/share/brain/models"
fi

# Local mirrors of the raw assets on a dev box (override each with its env var).
# These are the "copy from the local absolute path" sources the tests used to
# hardcode - the ONE place a machine-specific path may still appear.
#
# There are TWO kinds, and they resolve differently:
#
#   * CHECKPOINTS - `$MODEL_MIRROR`. A populated model store IS the checkpoint
#     mirror: it is laid out as `<vendor>/<repo>` (`crates/modelstore`), so every
#     checkpoint below is addressed by exactly the reference `brain fetch` takes,
#     and one variable covers all of them. This replaces a per-domain
#     `<root>/<domain>/weights/…` arrangement that predates the store and no
#     longer exists on any box. It resolves through `$BRAIN_MODELS_DIR` first
#     precisely because that IS the store on a configured box - the absolute
#     default is the fallback for a shell that has not set it. Only the handful
#     of checkpoints whose tests read them from `testdata/` are copied out of it;
#     the rest are reported present-or-absent and left where they lie.
#   * DUMPED GOLDENS and RAW TEST MEDIA - the `*_MIRROR` variables under it.
#     These are NOT checkpoints, so they are not in the model store and have no
#     canonical address; each one names a directory to hard-link from if you
#     have it. A tree whose mirror is absent is reported by name together with
#     where its contents come from (`_origin` below), because "absent" is the
#     normal state for these - they are regenerated per box, not distributed.
#     Their defaults are the directories these trees were last produced in on
#     this box; none of them exists here today, and none is guessed at - a wrong
#     guess would be a fixture that never arrives, reported as though it might.
MODEL_MIRROR="${BRAIN_MODEL_MIRROR:-${BRAIN_MODELS_DIR:-/data/workspace/resources}}"
ASR_MIRROR="${BRAIN_ASR_MIRROR:-/data/workspace/resources/asr}"
VL_MIRROR="${BRAIN_VL_MIRROR:-/data/workspace/resources/vl}"
TTS_MIRROR="${BRAIN_TTS_MIRROR:-/data/workspace/tmp/qwen3-tts-resources}"
GOLDEN_MIRROR="${BRAIN_GOLDEN_MIRROR:-/data/workspace/resources/brain-goldens}"

added=0 skipped=0 missing=0
# Where the tree being linked right now comes from when its mirror is absent -
# one line, printed as part of the skip message so a run says what is missing
# AND how to get it, not merely that a directory wasn't there. Set by the caller
# immediately before each group; `_link_from`/`_link_files` clear it after use so
# a stale origin can never be attributed to the next tree.
ORIGIN=""
_origin() {
  if [ -n "$ORIGIN" ]; then
    echo "      from: $ORIGIN"
  fi
  ORIGIN=""
}

# _link_from <mirror-root> <mirror-subdir> <dest-subdir> [extra-exclude-ere] -
# hard-link (or copy) every file under <mirror-root>/<mirror-subdir> into
# $DEST/<dest-subdir>, creating only what's missing.
#
# ALWAYS excludes `.git` directory contents - no test consumer ever needs one,
# and a cloned mirror's `.git` is pure duplicated disk (a checked-out working
# tree plus its own packed history of the same blobs) - and a mirror's
# `.cache/huggingface/` (download bookkeeping `huggingface_hub`/`hf` leave
# behind: `.gitignore`, `CACHEDIR.TAG`, per-file `.metadata`, tree manifests -
# never a weight, never read by a test). An optional extended regex (matched
# against the path relative to <mirror-subdir>) excludes whatever else that
# particular tree's mirror carries but no test reads - see the Qwen3-TTS tree,
# whose one mirror repo holds two differently-named destinations.
_link_from() {
  local root="$1" sub_src="$2" sub_dst="$3" extra_exclude="${4:-}"
  local src="$root/$sub_src" dst="$DEST/$sub_dst"
  if [ ! -d "$src" ]; then
    echo "  · $sub_dst: mirror '$src' absent - skipping (point its BRAIN_*_MIRROR at a copy, or add a URL)"
    _origin
    missing=$((missing + 1))
    return 0
  fi
  ORIGIN=""
  mkdir -p "$dst"
  local excluded=0
  while IFS= read -r -d '' f; do
    local rel="${f#"$src"/}"
    case "$rel" in
      .git/*|*/.git/*|.cache/huggingface/*|*/.cache/huggingface/*) excluded=$((excluded + 1)); continue ;;
    esac
    if [ -n "$extra_exclude" ] && [[ "$rel" =~ $extra_exclude ]]; then
      excluded=$((excluded + 1))
      continue
    fi
    local out="$dst/$rel"
    if [ -e "$out" ]; then
      skipped=$((skipped + 1))
      continue
    fi
    mkdir -p "$(dirname "$out")"
    ln "$f" "$out" 2>/dev/null || cp "$f" "$out"
    added=$((added + 1))
  done < <(find "$src" -type f -print0)
  if [ "$excluded" -gt 0 ]; then
    echo "  ✓ $sub_dst ($excluded excluded - .git and/or the tree's extra-exclude pattern)"
  else
    echo "  ✓ $sub_dst"
  fi
}
# _link_files <mirror-root> <mirror-subdir> <dest-subdir> <file>… - same as
# _link_from but for a NAMED subset, used where the mirror holds more models
# than a test needs (antelopev2 ships five .onnx; the face crates read two).
_link_files() {
  local root="$1" sub_src="$2" sub_dst="$3"
  shift 3
  local src="$root/$sub_src" dst="$DEST/$sub_dst"
  if [ ! -d "$src" ]; then
    echo "  · $sub_dst: mirror '$src' absent - skipping (point its BRAIN_*_MIRROR at a copy, or add a URL)"
    _origin
    missing=$((missing + 1))
    return 0
  fi
  mkdir -p "$dst"
  local incomplete=0
  for rel in "$@"; do
    if [ ! -e "$src/$rel" ]; then
      echo "  · $sub_dst/$rel: not in mirror - skipping"
      incomplete=1
      missing=$((missing + 1))
      continue
    fi
    if [ -e "$dst/$rel" ]; then
      skipped=$((skipped + 1))
      continue
    fi
    ln "$src/$rel" "$dst/$rel" 2>/dev/null || cp "$src/$rel" "$dst/$rel"
    added=$((added + 1))
  done
  # A mirror that exists but is missing a NAMED file is a half-populated tree,
  # not a success: report the origin there too, or the per-file "not in mirror"
  # lines above say what is absent without saying where to get it.
  if [ "$incomplete" -ne 0 ]; then
    _origin
  else
    echo "  ✓ $sub_dst"
    ORIGIN=""
  fi
}
asr_tree() { _link_from "$ASR_MIRROR" "$1" "$2"; }
golden_tree() { _link_from "$GOLDEN_MIRROR" "$1" "$2"; }
vl_tree()  { _link_from "$VL_MIRROR"  "$1" "$2"; }
tts_tree() { _link_from "$TTS_MIRROR" "$1" "$2"; }
# ckpt_tree <vendor>/<repo> <dest-subdir> [extra-exclude] - hard-link one model
# store entry out of $MODEL_MIRROR into the testdata tree, for the two families
# whose tests resolve a checkpoint through `brain_testutil::testdata` rather than
# `model_dir` and so cannot read it from the store directly.
ckpt_tree() { _link_from "$MODEL_MIRROR" "$1" "$2" "${3:-}"; }

echo "brain: populating testdata at $DEST, models at $MODELS_DIR"
echo "       checkpoint mirror: $MODEL_MIRROR"
echo "       fixture mirrors: asr=$ASR_MIRROR vl=$VL_MIRROR tts=$TTS_MIRROR golden=$GOLDEN_MIRROR"

# --- Checkpoints the parity tests import ------------------------------------
# REPORTED, never fetched. Each is an upstream HF repo the model crates read out
# of the model store under exactly the reference `brain fetch` takes, and
# `brain fetch` is the thing that puts it there - this script is test-fixture
# plumbing, not a general fetcher (see the header). Copying them here would also
# be the one operation in this script that is NOT nearly free: `$MODELS_DIR` is
# very often on a different filesystem from any mirror, where the hard link
# fails and the fallback `cp` duplicates tens of gigabytes.
# Only the ONE variant per family the crates import is listed - a store also
# carries sibling sizes (FastVLM-1.5B/7B, sam2.1-hiera-{small,base-plus}) that
# nothing here reads.
for ref in \
  nvidia/nemotron-3.5-asr-streaming-0.6b \
  Qwen/Qwen3-ASR-1.7B \
  apple/FastVLM-0.5B \
  moondream/moondream3-preview \
  Qwen/Qwen3-VL-4B-Instruct \
  facebook/sam2.1-hiera-tiny \
  facebook/sam2.1-hiera-large \
  Lightricks/LTX-2.5
do
  if [ -d "$MODELS_DIR/$ref" ]; then
    echo "  ✓ $ref (model store)"
  else
    echo "  · $ref: absent from the model store - skipping"
    ORIGIN="\`brain fetch $ref\` (network), which writes it to $MODELS_DIR/$ref"
    _origin
    missing=$((missing + 1))
  fi
done

# --- ASR goldens + waveforms -------------------------------------------------
ORIGIN="dump against the checkpoints in $MODELS_DIR with tools/goldens/asr_dump_reference.py; the waveforms are LibriSpeech clips"
asr_tree "golden"          "asr/golden"
ORIGIN="LibriSpeech test clips (clip.wav, librispeech_mr_quilter.wav)"
asr_tree "audio"           "asr/audio"

# --- Model-parity goldens (lfm / qwen encoder / vae / zimage) ----------------
# Small dumped fixtures the staged parity tests read (regenerable from the
# reference checkpoints via tools/goldens/*_dump_reference.py; never committed).
ORIGIN="tools/goldens/lfm2_dump_reference.py"
golden_tree "lfm"          "golden/lfm"
ORIGIN="tools/goldens/qwen_encoder_dump_reference.py"
golden_tree "qwen"         "golden/qwen"
ORIGIN="tools/goldens/vae_dump_reference.py and tools/goldens/sdxl_dump_vae_decode.py"
golden_tree "vae"          "golden/vae"
ORIGIN="tools/goldens/s3dit_model_dump_reference.py (and its block/real siblings)"
golden_tree "zimage"       "golden/zimage"
ORIGIN="tools/goldens/rrdbnet_dump_reference.py"
golden_tree "esrgan"       "golden/esrgan"
ORIGIN="tools/goldens/qwen3omnimoe_dump_reference.py"
golden_tree "omni"         "golden/omni"
ORIGIN="tools/goldens/ltxv_{vae,audio,upsampler,duration_head,na_decoder,schedule,dit,av_dit}_dump_reference.py, against Lightricks/LTX-2.5 in the store"
golden_tree "ltxv"         "golden/ltxv"
ORIGIN="tools/goldens/gemma4_dump_reference.py, against Lightricks/LTX-2.5's gemma4-12b-with-proj text encoder in the store"
golden_tree "gemma4"       "golden/gemma4"

# --- SAM 2.1 stage goldens (promptable segmentation, image path) -------------
# The CHECKPOINTS come from the store above (`facebook/sam2.1-hiera-*`), which is
# where `crates/sam2/tests/parity.rs` looks when `testdata/sam2/<variant>/` has
# no local copy. Only the stage goldens
# (`{input,trunk,neck,case_*}.safetensors`, ~1 GB per variant) belong here, and
# they are NOT mirrored anywhere: regenerate them with
#   python3 tools/goldens/sam2_dump_reference.py --code <sam2 repo> \
#       --config <sam2 repo>/sam2/configs/sam2.1/sam2.1_hiera_t.yaml \
#       --ckpt   <models-dir>/facebook/sam2.1-hiera-tiny/sam2.1_hiera_tiny.pt \
#       --out    testdata/sam2/hiera-tiny
# `crates/sam2/tests/parity.rs` skips itself while they are absent.

# --- Face recognition (insightface antelopev2: SCRFD + ArcFace) --------------
# The two released ONNX files the face crates import - `crates/scrfd` reads the
# detector, `crates/arcface` the embedder (and the detector too, for its aligned
# path). The stage goldens (`{arcface,arcface_blocks,scrfd,align,e2e}.safetensors`
# + `manifest*.json`) are NOT mirrored - regenerate them next to the weights with
#   python3 tools/goldens/scrfd_dump_reference.py \
#       --weights testdata/face/antelopev2 --out testdata/face/antelopev2
#   python3 tools/goldens/arcface_dump_reference.py \
#       --weights testdata/face/antelopev2 --out testdata/face/antelopev2 \
#       [--photos a.jpg b.jpg c.jpg]
# Both crates' `tests/parity.rs` skip themselves while they are absent.
# These two graphs are a checkpoint, but the tests resolve them through
# `brain_testutil::testdata`, so the store copy is linked INTO testdata rather
# than read where it lies.
ORIGIN="the insightface antelopev2 release, in the store as DIAMONIK7777/antelopev2 (\`brain fetch\` it)"
_link_files "$MODEL_MIRROR" "DIAMONIK7777/antelopev2" "face/antelopev2" \
  glintr100.onnx scrfd_10g_bnkps.onnx

# --- CLIP tokenizer (SDXL) ---------------------------------------------------
# `vocab.json` + `merges.txt` for `data::clip_bpe::ClipBpe`. SDXL's `tokenizer/`
# and `tokenizer_2/` ship BYTE-IDENTICAL copies (they differ only in
# `pad_token`), so one copy is linked and the test builds both tokenizers from
# it. The id golden `clip/tokenizer/ids.safetensors` comes from
# `tools/clip_dump_reference.py`, not from here.
ORIGIN="the SDXL base repo's tokenizer/ - \`brain fetch stabilityai/stable-diffusion-xl-base-1.0\`"
_link_files "$MODEL_MIRROR" "stabilityai/stable-diffusion-xl-base-1.0/tokenizer" "clip/tokenizer" \
  vocab.json merges.txt

# --- Vision-language parity dumps (FastVLM, Moondream3, Qwen3-VL) ------------
# The checkpoints come from the store above; only the decoder/vision reference
# dumps `crates/{fastvlm,moondream3,qwen3vl}/src/parity.rs` read live here.
ORIGIN="tools/goldens/{fastvlm_decoder,fastvlm_vision,fastvlm_caption,moondream3_decoder,qwen3vl_decoder}_dump_reference.py"
vl_tree  "parity"          "vl/parity"

# --- Audio codec / TTS / speaker (Qwen3-TTS) --------------------------------
# The checkpoint halves are in the store as one repo: the talker/base config +
# weights at its root, the 12 Hz speech tokenizer in `speech_tokenizer/`.
# `crates/{qwen3tts,ecapatdnn,mimi}`'s tests read both through
# `brain_testutil::testdata` under the names below, so they are linked in rather
# than read from the store. `brain_tts/` (the pre-split talker/mtp/codec/speaker
# safetensors) is excluded: nothing under `testdata/tts/` reads it - the TTS
# tests import their weights to `out/tts/` instead - and `speech_tokenizer/` is
# excluded from the root copy because it is linked separately, under the name
# the mimi tests use.
ORIGIN="\`brain fetch Qwen/Qwen3-TTS-12Hz-0.6B-Base\`"
ckpt_tree "Qwen/Qwen3-TTS-12Hz-0.6B-Base" "tts/ckpt/Qwen3-TTS-12Hz-0.6B-Base" \
  '^(brain_tts|speech_tokenizer)/'
ORIGIN="\`brain fetch Qwen/Qwen3-TTS-12Hz-0.6B-Base\` (its speech_tokenizer/ subdir)"
ckpt_tree "Qwen/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer" "tts/ckpt/Qwen3-TTS-Tokenizer-12Hz"
ORIGIN="tools/goldens/qwen3tts_{codec,speaker}_dump_reference.py (they fetch the upstream reference themselves)"
tts_tree "dumps"           "tts/dumps"
# loose reference files at the mirror root (e.g. the voice-clone example wav)
if [ -d "$TTS_MIRROR" ]; then
  for f in "$TTS_MIRROR"/*.wav; do
    [ -e "$f" ] || continue
    out="$DEST/tts/$(basename "$f")"
    [ -e "$out" ] && { skipped=$((skipped + 1)); continue; }
    mkdir -p "$DEST/tts"; ln "$f" "$out" 2>/dev/null || cp "$f" "$out"; added=$((added + 1))
  done
elif [ ! -e "$DEST/tts/voice-clone-example-voice.wav" ]; then
  # Silence here would be the one group a run never mentions: the voice-clone
  # tests name this file directly, so an absent mirror has to say so.
  echo "  · tts/voice-clone-example-voice.wav: mirror '$TTS_MIRROR' absent - skipping"
  ORIGIN="the upstream Qwen3-TTS reference implementation's voice-clone example"
  _origin
  missing=$((missing + 1))
fi

echo "brain: testdata ready - $added new, $skipped already present, $missing groups unavailable"
