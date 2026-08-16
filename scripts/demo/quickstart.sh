#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Runs every one-liner in README.md's Quick start, in order, against real
# auto-fetched weights, and writes the images/text it produces into
# docs/quickstart/img/ - the source of truth the README embeds and links to.
# Driven by `make docs/quickstart`.
#
# Design:
#   * Idempotent - a step whose output file already exists is skipped, so a
#     killed/resumed run (this fetches tens of GB) picks up where it left off.
#     Delete a file under $IMG_DIR (or the whole dir) to force a step to redo.
#   * No hidden state - every command here is copy-pasteable verbatim from
#     README.md; this script's only job beyond running them is capturing
#     stdout into the .txt files the README quotes and copying images into
#     $IMG_DIR under their published names.
#   * Chained - each step's real output feeds the next: the text2image call
#     is the ONE seed image every later step (detect, segment, depth,
#     caption, inpaint, vlm) reads.
#   * Network-bound, not compute-bound - the checkpoints this pulls are
#     multi-GB; set HF_TOKEN (or `hf auth login` first) to avoid HF's
#     unauthenticated rate limit, and expect a cold run to take a while.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

BRAIN="${BRAIN:-$REPO/target/release/brain}"
[ -x "$BRAIN" ] || { echo "quickstart: no brain binary at $BRAIN -- run 'make release' first" >&2; exit 1; }

: "${BRAIN_MODELS_DIR:=$REPO/.quickstart-models}"
export BRAIN_MODELS_DIR
mkdir -p "$BRAIN_MODELS_DIR"

IMG_DIR="$REPO/docs/quickstart/img"
mkdir -p "$IMG_DIR"

# Every real HF token this box has, so a cold run does not eat the anonymous
# rate limit - `hf auth login` (or `HF_TOKEN=... make docs/quickstart`) sets
# this up ahead of time; unset is fine too, just slower.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
fi

step() { echo "== $* =="; }
need() { [ -s "$1" ]; }

# ---------------------------------------------------------------- 1. text

step "1. text generation (qwen3, auto-fetches Qwen/Qwen3-0.6B)"
if ! need "$IMG_DIR/qwen3-infer.txt"; then
  "$BRAIN" infer qwen3 --prompt "The capital of France is" --max-new 12 \
    2>/dev/null | tail -1 > "$IMG_DIR/qwen3-infer.txt"
fi

# ---------------------------------------------------------------- 2. image

step "2. text-to-image (s3dit, auto-fetches Tongyi-MAI/Z-Image-Turbo - ~33 GB, the slow step)"
if ! need "$IMG_DIR/seed.png"; then
  "$BRAIN" --device gpu s3dit text2image \
    --prompt "a golden retriever dog and a red apple on a wooden table, photorealistic, natural lighting" \
    --width 512 --height 512 --seed 7 --steps 8 --precision int8 \
    --out image="$IMG_DIR/seed.png"
fi

# ---------------------------------------------------------------- 3. upscale, then detect

step "3a. super-resolution (rrdbnet, auto-fetches schwgHao/RealESRGAN_x4plus)"
if ! need "$IMG_DIR/upscaled.png"; then
  # --tile 0 (whole image, no tiling) OOMs at this size: RealESRGAN's 4x
  # upscale of a 512x512 source produces a 2048x2048 canvas, and an
  # untiled forward materializes per-layer feature-map buffers around this
  # backend's 2047 MiB single-buffer binding cap. --tile 256 (1280x1280
  # output tiles) still crosses it; --tile 128 (512x512 output tiles,
  # measured) comfortably does not.
  "$BRAIN" rrdbnet upscale --in image="$IMG_DIR/seed.png" --tile 128 \
    --out image="$IMG_DIR/upscaled.png"
fi

step "3b. object detection (yolov8, auto-fetches Ultralytics/YOLOv8) + draw_boxes, on the upscaled image"
if ! need "$IMG_DIR/detected.png"; then
  DETECTIONS="$("$BRAIN" yolov8 detect \
    --weights "$BRAIN_MODELS_DIR/Ultralytics/YOLOv8/model.brain.safetensors" \
    --image "$IMG_DIR/upscaled.png" 2>/dev/null)"
  BOXES="$(python3 -c '
import json, sys
out = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("["):
        continue
    x1, y1, x2, y2, conf, cls = json.loads(line)
    out.append({"bbox": [x1, y1, x2, y2], "conf": conf, "class": int(cls)})
print(json.dumps(out))
' <<< "$DETECTIONS")"
  echo "$DETECTIONS" > "$IMG_DIR/yolov8-detect.txt"
  "$BRAIN" imageops draw_boxes --in image="$IMG_DIR/upscaled.png" --boxes "$BOXES" \
    --out image="$IMG_DIR/detected.png"
fi

# ---------------------------------------------------------------- 4. segment

step "4. promptable segmentation (sam2, auto-fetches facebook/sam2.1-hiera-tiny) + colorize"
if ! need "$IMG_DIR/segmented.png"; then
  "$BRAIN" sam2 segment --in image="$IMG_DIR/seed.png" --points "220,180" --labels "1" \
    --out mask="$IMG_DIR/.mask-raw.png" --json
  "$BRAIN" imageops colorize --in image="$IMG_DIR/.mask-raw.png" --colormap gray \
    --out image="$IMG_DIR/segmented.png"
  rm -f "$IMG_DIR/.mask-raw.png"
fi

# ---------------------------------------------------------------- 4b. depth

step "4b. monocular depth (zipdepth) + turbo colormap"
if ! need "$IMG_DIR/depth.png"; then
  # No HuggingFace host exists for this checkpoint - the upstream project
  # ships it directly in its own GitHub repo, so this is the one step that
  # does not go through brain's own model-store auto-fetch.
  ZIPDEPTH_PTH="$BRAIN_MODELS_DIR/brain/zipdepth/zipdepth_base.pth"
  if ! need "$ZIPDEPTH_PTH"; then
    mkdir -p "$(dirname "$ZIPDEPTH_PTH")"
    curl -sSL -o "$ZIPDEPTH_PTH" \
      "https://github.com/fabiotosi92/ZipDepth/raw/main/checkpoints/zipdepth_base.pth"
  fi
  # zipdepth's CLI reads/writes PPM directly (not the shared --in/--out
  # image= codec path every other action here uses), so this is the one
  # step that round-trips through PPM by hand.
  python3 -c "
from PIL import Image
Image.open('$IMG_DIR/seed.png').convert('RGB').save('$IMG_DIR/.seed.ppm')
"
  DISPLAY= "$BRAIN" zipdepth --image "$IMG_DIR/.seed.ppm" --weights "$ZIPDEPTH_PTH" \
    --headless --view depth --colormap turbo --out "$IMG_DIR/.depth.ppm"
  python3 -c "
from PIL import Image
Image.open('$IMG_DIR/.depth.ppm').save('$IMG_DIR/depth.png')
"
  rm -f "$IMG_DIR/.seed.ppm" "$IMG_DIR/.depth.ppm"
fi

# ---------------------------------------------------------------- 5. image + text -> text (two VLMs)

step "5a. image + text -> text, fast captioning (fastvlm, auto-fetches apple/FastVLM-0.5B)"
if ! need "$IMG_DIR/fastvlm-caption.txt"; then
  # `brain fastvlm caption` reliably segfaults on exit AFTER correctly
  # writing its output (real, reproducible bug -- not this script working
  # around a computation error). `|| true` keeps the real crash from
  # aborting the rest of the quickstart; the `need` check right after
  # still verifies the output file is real and non-empty.
  "$BRAIN" fastvlm caption --in image="$IMG_DIR/seed.png" \
    --prompt "What is in this image? Answer in one sentence." --max_new 40 \
    --out text="$IMG_DIR/fastvlm-caption.txt" || true
  need "$IMG_DIR/fastvlm-caption.txt" || { echo "fastvlm caption produced no output" >&2; exit 1; }
fi

step "5b. image + text -> text, general VQA (qwen3vl, auto-fetches Qwen/Qwen3-VL-4B-Instruct)"
if ! need "$IMG_DIR/qwen3vl-caption.txt"; then
  "$BRAIN" qwen3vl generate --in image="$IMG_DIR/seed.png" \
    --prompt "What is in this image? Answer in one sentence." --max_new 40 \
    --out text="$IMG_DIR/qwen3vl-caption.txt"
fi

# ---------------------------------------------------------------- 6. inpaint

step "6. masked inpainting (s3dit inpaint) - replace the apple with a slice of chocolate cake"
if ! need "$IMG_DIR/inpainted.png"; then
  "$BRAIN" imageops mask_rect --width 512 --height 512 --x 150 --y 300 --w 210 --h 190 \
    --out mask="$IMG_DIR/.inpaint-mask.png"
  "$BRAIN" --device gpu s3dit inpaint --in image="$IMG_DIR/seed.png" --in mask="$IMG_DIR/.inpaint-mask.png" \
    --prompt "a slice of chocolate cake on the table" --strength 0.85 --steps 8 --precision int8 \
    --out image="$IMG_DIR/inpainted.png"
  rm -f "$IMG_DIR/.inpaint-mask.png"
fi

# ---------------------------------------------------------------- 7. tts -> asr -> text (round trip)

step "7. text -> speech -> text -> text (qwen3tts synth, auto-fetches Qwen/Qwen3-TTS-12Hz-0.6B-Base; nemotronasr transcribe, auto-fetches nvidia/nemotron-3.5-asr-streaming-0.6b; qwen3 infer)"
if ! need "$IMG_DIR/roundtrip.txt"; then
  SENTENCE="Brain trains and runs neural networks from scratch, in Rust."
  "$BRAIN" qwen3tts synth --text "$SENTENCE" --out "$IMG_DIR/.spoken.wav"
  TRANSCRIPT="$("$BRAIN" nemotronasr transcribe --in audio="$IMG_DIR/.spoken.wav" --json 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["text"].strip())')"
  RESPONSE="$("$BRAIN" infer qwen3 --prompt "$TRANSCRIPT" --max-new 24 2>/dev/null | tail -1)"
  {
    echo "synthesized: $SENTENCE"
    echo "transcribed: $TRANSCRIPT"
    echo "---"
    echo "$RESPONSE"
  } > "$IMG_DIR/roundtrip.txt"
  rm -f "$IMG_DIR/.spoken.wav"
fi

# ---------------------------------------------------------------- 8. TTS LoRA

step "8. audio LoRA (qwen3tts finetune, synthetic text->codes data)"
if ! need "$IMG_DIR/qwen3tts-lora.txt"; then
  "$BRAIN" data gen tts --out "$BRAIN_MODELS_DIR/.quickstart-tts-data" --n 4000 --seed 1337
  "$BRAIN" qwen3tts finetune \
    --base "$BRAIN_MODELS_DIR/Qwen/Qwen3-TTS-12Hz-0.6B-Base/brain_tts/talker.safetensors" \
    --data "$BRAIN_MODELS_DIR/.quickstart-tts-data" \
    --out "$BRAIN_MODELS_DIR/.quickstart-talker-lora.safetensors" --steps 200 \
    2>&1 | tail -2 > "$IMG_DIR/qwen3tts-lora.txt"
fi

# ---------------------------------------------------------------- 9. serving

step "9. local OpenAI-compatible API (brain serve)"
if ! need "$IMG_DIR/serve.txt"; then
  "$BRAIN" infer qwen3 --prompt "warmup" --max-new 1 >/dev/null 2>&1 || true
  "$BRAIN" serve --openai 8799 > "$IMG_DIR/.serve.log" 2>&1 &
  SERVE_PID=$!
  for _ in $(seq 1 30); do
    grep -q "APIKEY openai" "$IMG_DIR/.serve.log" 2>/dev/null && break
    sleep 1
  done
  APIKEY="$(grep "APIKEY openai" "$IMG_DIR/.serve.log" | awk '{print $3}')"
  # The APIKEY log line prints once the key is generated, slightly before
  # the HTTP listener is guaranteed ready to accept connections -- poll
  # with a real request instead of trusting a fixed extra sleep.
  RESPONSE=""
  for _ in $(seq 1 10); do
    RESPONSE="$(curl -s --max-time 30 http://127.0.0.1:8799/v1/chat/completions \
      -H "Authorization: Bearer $APIKEY" -H 'Content-Type: application/json' \
      -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Say hello in exactly five words."}]}')"
    echo "$RESPONSE" | python3 -c 'import json,sys; json.load(sys.stdin)["choices"]' >/dev/null 2>&1 && break
    sleep 2
  done
  kill "$SERVE_PID" 2>/dev/null || true
  wait "$SERVE_PID" 2>/dev/null || true
  echo "$RESPONSE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' \
    > "$IMG_DIR/serve.txt" || { echo "serve: unexpected response: $RESPONSE" >&2; exit 1; }
  rm -f "$IMG_DIR/.serve.log"
fi

step "done -- assets in $IMG_DIR"
