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
[ -x "$BRAIN" ] || { echo "quickstart: no brain binary at $BRAIN -- run 'make build/release' first" >&2; exit 1; }

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

step "1b. bidirectional text embedding (lfm2, auto-fetches LiquidAI/LFM2.5-350M)"
if ! need "$IMG_DIR/lfm2-embed.txt"; then
  "$BRAIN" lfm2 embed --text "Brain trains and runs neural networks from scratch, in Rust." \
    2>/dev/null | grep "^embedding" > "$IMG_DIR/lfm2-embed.txt"
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
if ! need "$IMG_DIR/segmented.png" || ! need "$IMG_DIR/dog-mask.png"; then
  "$BRAIN" sam2 segment --in image="$IMG_DIR/seed.png" --points "220,180" --labels "1" \
    --out mask="$IMG_DIR/dog-mask.png" --json
  "$BRAIN" imageops colorize --in image="$IMG_DIR/dog-mask.png" --colormap gray \
    --out image="$IMG_DIR/segmented.png"
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

step "6. masked inpainting (s3dit inpaint) - regenerate everything BUT the dog, from sam2's own mask"
if ! need "$IMG_DIR/inpainted.png"; then
  # Invert step 4's real segmentation mask (white = dog) into an inpaint mask
  # (white = regenerate): everything except the dog is fair game, so this
  # keeps the dog itself pixel-anchored while the wood table, the apple, and
  # the background all change together - a real demonstration of "hold part
  # of the image fixed", not just a small object swap inside a hand-picked
  # rectangle.
  python3 -c "
from PIL import Image, ImageOps
ImageOps.invert(Image.open('$IMG_DIR/dog-mask.png').convert('L')).save('$IMG_DIR/.bg-mask.png')
"
  # feather=0 (a hard mask edge) is intentional here, not a fallback: the
  # sam2 mask already follows the dog's organic fur silhouette (unlike a
  # synthetic rectangle, it needs no softening to hide a straight edge), and
  # any positive feather blends a few percent of the ORIGINAL pixels back in
  # at every sampling step for cells near the boundary. The apple used to
  # sit right against the dog's chin, so that leak reinforced an apple-shaped
  # ghost into the regenerated cake every step, regardless of step count
  # (confirmed: 8 vs 16 steps at feather=3 produced near-identical ghosting,
  # mean diff 7.9/255 -- ruling out "just needs more steps"). feather=0
  # removes the leak entirely and the boundary still reads as clean because
  # the mask shape itself is already the right shape.
  "$BRAIN" --device gpu s3dit inpaint --in image="$IMG_DIR/seed.png" --in mask="$IMG_DIR/.bg-mask.png" \
    --prompt "a golden retriever dog sitting behind a slice of chocolate cake on a white marble kitchen countertop, bright natural daylight, blurred modern kitchen background, photorealistic" \
    --strength 1.0 --feather 0 --steps 8 --precision int8 \
    --out image="$IMG_DIR/inpainted.png"
  rm -f "$IMG_DIR/.bg-mask.png"
fi

# ---------------------------------------------------------------- 6b. text -> video

step "6b. text-to-video (wan, auto-fetches Wan-AI/Wan2.1-T2V-1.3B - ~17.6 GB)"
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "   skipped: wan needs the ffmpeg CLI, both to write a container and to tile the strip below"
elif ! need "$IMG_DIR/wan-strip.png"; then
  # Deliberately NOT the model's own sampling defaults (81 frames at 832x480
  # over 50 steps), which occupy a Tesla P40 for the better part of an hour:
  # this script is re-run by `make docs/quickstart` and asserted on by
  # `make test/e2e/quickstart`, and an hour-long step in it would simply stop
  # being run. Half-scale 16:9 (416x240) at 9 frames and 20 steps is the
  # smallest setting measured here that still produces a recognizable subject
  # moving across the frame. Most of its wall clock is the umT5-XXL text
  # encode, which runs on the CPU (22.72 GB in fp32 does not fit the card) and
  # costs the same at every size, so shrinking further buys very little.
  "$BRAIN" --device gpu wan t2v \
    --prompt "a golden retriever running along a sandy beach at sunset, waves in the background, cinematic" \
    --frames 9 --width 416 --height 240 --steps 20 --seed 7 \
    --output-path "$IMG_DIR/wan.mp4"
  # Every other step here publishes a PNG the README embeds; a video cannot be
  # embedded, so publish every other frame as one contact strip instead. The
  # .mp4 is kept beside it - the strip is the still that proves motion, the
  # file is the thing the command actually produced.
  ffmpeg -y -v error -i "$IMG_DIR/wan.mp4" \
    -vf "select='not(mod(n\,2))',tile=5x1" -frames:v 1 "$IMG_DIR/wan-strip.png"
fi

# ---------------------------------------------------------------- 7. tts -> asr -> text (round trip)

step "7. text -> speech -> text -> text (qwen3tts synth, auto-fetches Qwen/Qwen3-TTS-12Hz-0.6B-Base; nemotronasr + qwen3asr transcribe, two independent ASR models on the same audio; qwen3 infer)"
if ! need "$IMG_DIR/roundtrip.txt"; then
  SENTENCE="Brain trains and runs neural networks from scratch, in Rust."
  "$BRAIN" qwen3tts synth --text "$SENTENCE" --out "$IMG_DIR/.spoken.wav"
  TRANSCRIPT="$("$BRAIN" nemotronasr transcribe --in audio="$IMG_DIR/.spoken.wav" --json 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["text"].strip())')"
  TRANSCRIPT2="$("$BRAIN" qwen3asr transcribe --in audio="$IMG_DIR/.spoken.wav" --json 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["text"].strip())')"
  RESPONSE="$("$BRAIN" infer qwen3 --prompt "$TRANSCRIPT" --max-new 24 2>/dev/null | tail -1)"
  {
    echo "synthesized: $SENTENCE"
    echo "transcribed (nemotronasr): $TRANSCRIPT"
    echo "transcribed (qwen3asr):    $TRANSCRIPT2"
    echo "---"
    echo "$RESPONSE"
  } > "$IMG_DIR/roundtrip.txt"
  rm -f "$IMG_DIR/.spoken.wav"
fi

# ---------------------------------------------------------------- 7b. document OCR round trip

step "7b. text -> document image -> text (deepseek2ocr, auto-fetches ggml-org/DeepSeek-OCR-GGUF)"
if ! need "$IMG_DIR/ocr.txt"; then
  SENTENCE="Brain trains and runs neural networks from scratch, in Rust."
  python3 -c "
from PIL import Image, ImageDraw, ImageFont
img = Image.new('RGB', (640, 320), 'white')
d = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 28)
d.multiline_text((40, 120), 'Brain trains and runs neural\nnetworks from scratch, in Rust.', fill='black', font=font, spacing=16)
img.save('$IMG_DIR/doc.png')
"
  OCR_TEXT="$("$BRAIN" deepseek2ocr generate --in image="$IMG_DIR/doc.png" \
    --prompt "<|grounding|>Convert the document to markdown." 2>/dev/null \
    | python3 -c 'import json,re,sys
for line in sys.stdin:
    line = line.strip()
    if line.startswith("text:"):
        raw = json.loads(line.split(":", 1)[1].strip())
        print(re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", "", raw).strip())
        break')"
  {
    echo "rendered:    $SENTENCE"
    echo "ocr'd:       $OCR_TEXT"
  } > "$IMG_DIR/ocr.txt"
fi

# ---------------------------------------------------------------- 7c. forecasting

# The CSV is committed (examples/forecast/synthetic_hourly.csv), regenerated by
# hand with tools/forecast/make_synthetic_ohlcv.py -- so this step is a forecast,
# not a data-generation run, and it costs no network.
step "7c. OHLCV forecasting (kronos, auto-fetches NeoQuasar/Kronos-base + NeoQuasar/Kronos-Tokenizer-base - ~407 MB)"
if ! need "$IMG_DIR/kronos-forecast.png" || ! need "$IMG_DIR/kronos-forecast.txt"; then
  if ! command -v gnuplot >/dev/null 2>&1; then
    echo "   skipped: the forecast chart needs the gnuplot CLI (apt-get install gnuplot)"
  else
    # Minutes on CPU, the longest compute-bound step on this page: a 506-bar
    # prefill per origin is the cost, and 16 origins is 16 disjoint held-out
    # windows rather than one draw. Worth the minutes - the same measurement
    # over 8 origins moves by 14 points, so a cheaper run would be reporting
    # luck. The horizon is 6 because that is where this checkpoint's skill
    # actually is: it beats persistence on CRPS at 6 bars and loses at 12 and
    # 24, and both numbers are in the README next to this one.
    "$BRAIN" forecast predict --csv examples/forecast/synthetic_hourly.csv \
      --horizon 6 --samples 16 --origins 16 --gnuplot "$IMG_DIR/kronos-forecast.png" \
      2>/dev/null | grep -v '^  chart:' > "$IMG_DIR/kronos-forecast.txt"
  fi
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
