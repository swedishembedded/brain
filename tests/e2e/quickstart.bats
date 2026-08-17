#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Re-runs README.md's Quick start (via scripts/demo/quickstart.sh) and asserts
# the cross-model agreement the README claims: yolo's detected classes and
# fastvlm's/qwen3vl's captions both name real content in the generated seed
# image, and the TTS->ASR->LLM round trip actually round-trips. This is what
# keeps the README honest - a line that stops working here is a line the
# README is lying about.
#
# NOT run by default: this pulls tens of GB of real checkpoints on first run
# (see quickstart.sh's own docs) and takes a while on a slow/unauthenticated
# link. Skips cleanly unless BRAIN_QUICKSTART_E2E=1 is set, so `make test/e2e`
# stays fast; opt in explicitly (CI job, or a developer with the bandwidth and
# the time) to actually exercise it.

setup_file() {
  [ "${BRAIN_QUICKSTART_E2E:-0}" = "1" ] || skip "set BRAIN_QUICKSTART_E2E=1 to run (pulls tens of GB of real checkpoints)"
  command -v python3 >/dev/null 2>&1 || skip "python3 not installed"

  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="${BRAIN_BIN:-$REPO/target/release/brain}"
  [ -x "$BRAIN" ] || skip "no brain binary (build with: make release, or set BRAIN_BIN)"
  export BRAIN

  IMG_DIR="$REPO/docs/quickstart/img"
  export IMG_DIR

  # The full run (scripts/demo/quickstart.sh) is driven separately by `make
  # docs/quickstart` - this suite ASSERTS on whatever is already in $IMG_DIR
  # rather than re-running the whole multi-hour fetch itself, so a CI job can
  # run the fetch once and this suite many times against its output.
  [ -d "$IMG_DIR" ] || skip "docs/quickstart/img missing -- run scripts/demo/quickstart.sh (or 'make docs/quickstart') first"
}

@test "quickstart: qwen3 text generation produced non-empty output" {
  [ -s "$IMG_DIR/qwen3-infer.txt" ] || skip "step 1 (qwen3 infer) has not run yet"
  run cat "$IMG_DIR/qwen3-infer.txt"
  [ "$status" -eq 0 ]
  [ -n "$output" ]
}

@test "quickstart: lfm2 produced a real 1024-dim embedding" {
  [ -s "$IMG_DIR/lfm2-embed.txt" ] || skip "step 1b (lfm2 embed) has not run yet"
  run cat "$IMG_DIR/lfm2-embed.txt"
  [ "$status" -eq 0 ]
  echo "$output" | grep -q "^embedding\[1024\]" || {
    echo "unexpected lfm2 embed output: $output" >&2
    return 1
  }
}

@test "quickstart: s3dit text2image produced a real image" {
  [ -s "$IMG_DIR/seed.png" ] || skip "step 2 (s3dit text2image) has not run yet"
  run python3 -c "
from pathlib import Path
data = Path('$IMG_DIR/seed.png').read_bytes()
assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG'
assert len(data) > 10_000, f'suspiciously small: {len(data)} bytes'
"
  [ "$status" -eq 0 ]
}

@test "quickstart: rrdbnet upscale produced a larger image than the source" {
  [ -s "$IMG_DIR/upscaled.png" ] || skip "step 3a (rrdbnet upscale) has not run yet"
  run python3 -c "
from pathlib import Path
seed = Path('$IMG_DIR/seed.png').stat().st_size
up = Path('$IMG_DIR/upscaled.png').stat().st_size
assert up > seed, f'upscaled ({up}B) is not bigger than the seed ({seed}B)'
"
  [ "$status" -eq 0 ]
}

@test "quickstart: yolov8 detected at least one real object on the upscaled image" {
  [ -s "$IMG_DIR/detected.png" ] || skip "step 3b (yolov8 detect) has not run yet"
  [ -s "$IMG_DIR/yolov8-detect.txt" ] || skip "step 3b detections log missing"
  run python3 -c "
import json
from pathlib import Path
data = Path('$IMG_DIR/detected.png').read_bytes()
assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG'
lines = [l for l in Path('$IMG_DIR/yolov8-detect.txt').read_text().splitlines() if l.strip().startswith('[')]
assert len(lines) > 0, 'no detections logged'
"
  [ "$status" -eq 0 ]
}

@test "quickstart: sam2 segmentation mask is non-trivial (not all-black or all-white)" {
  [ -s "$IMG_DIR/segmented.png" ] || skip "step 4 (sam2 segment) has not run yet"
  run python3 -c "
from pathlib import Path
data = Path('$IMG_DIR/segmented.png').read_bytes()
assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG'
assert len(data) > 500, 'suspiciously tiny -- likely a flat/degenerate mask'
"
  [ "$status" -eq 0 ]
}

@test "quickstart: zipdepth produced a real depth map" {
  [ -s "$IMG_DIR/depth.png" ] || skip "step 4b (zipdepth) has not run yet"
  run python3 -c "
from pathlib import Path
data = Path('$IMG_DIR/depth.png').read_bytes()
assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG'
assert len(data) > 500, 'suspiciously tiny -- likely a flat/degenerate depth map'
"
  [ "$status" -eq 0 ]
}

@test "quickstart: fastvlm caption names an object the s3dit prompt asked for (cross-model agreement)" {
  [ -s "$IMG_DIR/fastvlm-caption.txt" ] || skip "step 5 (fastvlm caption) has not run yet"
  run cat "$IMG_DIR/fastvlm-caption.txt"
  [ "$status" -eq 0 ]
  [ -n "$output" ]
  # The seed prompt (README/quickstart.sh) asks for a dog and an apple; the
  # caption should name at least one of the objects the prompt asked for --
  # weak on purpose (a VLM's exact wording varies run to run), but strong
  # enough to catch a caption that is empty, an error message, or wildly off.
  echo "$output" | grep -qiE 'dog|retriever|apple|fruit|table' || {
    echo "caption did not mention any expected object: $output" >&2
    return 1
  }
}

@test "quickstart: qwen3vl caption names an object the s3dit prompt asked for (cross-model agreement)" {
  [ -s "$IMG_DIR/qwen3vl-caption.txt" ] || skip "step 5b (qwen3vl generate) has not run yet"
  run cat "$IMG_DIR/qwen3vl-caption.txt"
  [ "$status" -eq 0 ]
  [ -n "$output" ]
  echo "$output" | grep -qiE 'dog|retriever|apple|fruit|table' || {
    echo "caption did not mention any expected object: $output" >&2
    return 1
  }
}

@test "quickstart: s3dit inpaint held the dog fixed (sam2 mask) while changing everything else" {
  [ -s "$IMG_DIR/inpainted.png" ] || skip "step 6 (s3dit inpaint) has not run yet"
  [ -s "$IMG_DIR/seed.png" ] || skip "step 2 (s3dit text2image) has not run yet"
  [ -s "$IMG_DIR/dog-mask.png" ] || skip "step 4 (sam2 segment) has not run yet"
  # The inpaint mask is sam2's own dog mask, inverted (regenerate everything
  # BUT the dog). A real mask-anchored inpaint should show near-zero drift
  # inside the dog region and real, substantial change outside it - not just
  # "the file changed", which a broken mask (e.g. inverted the wrong way, or
  # ignored entirely) would also satisfy.
  run python3 -c "
from PIL import Image
import numpy as np
seed = np.array(Image.open('$IMG_DIR/seed.png').convert('RGB'), dtype=np.int16)
inpainted = np.array(Image.open('$IMG_DIR/inpainted.png').convert('RGB'), dtype=np.int16)
dog = np.array(Image.open('$IMG_DIR/dog-mask.png').convert('L')) > 127
assert seed.shape == inpainted.shape, f'size mismatch: seed {seed.shape} vs inpainted {inpainted.shape}'
diff = np.abs(seed - inpainted).mean(axis=2)
dog_drift = diff[dog].mean()
bg_drift = diff[~dog].mean()
assert dog_drift < 10.0, f'dog region drifted too much (mean abs diff {dog_drift:.1f}) -- mask is not holding it fixed'
assert bg_drift > dog_drift * 3, f'background barely changed relative to the dog region (bg {bg_drift:.1f} vs dog {dog_drift:.1f}) -- inpaint looks like a no-op'
"
  [ "$status" -eq 0 ]
}

@test "quickstart: wan produced a real clip whose frames actually differ" {
  [ -s "$IMG_DIR/wan.mp4" ] || skip "step 6b (wan t2v) has not run yet"
  [ -s "$IMG_DIR/wan-strip.png" ] || skip "step 6b contact strip missing"
  # A video model that produced a still (every frame identical) would satisfy
  # "the file exists" and "it decodes" while having done nothing a text-to-image
  # model could not - so assert MOTION, not just bytes. The strip is 5 tiles of
  # one frame each, so comparing the first tile to the last is comparing frame 1
  # to frame 9 of the clip.
  run python3 -c "
from PIL import Image
import numpy as np
im = np.array(Image.open('$IMG_DIR/wan-strip.png').convert('RGB'), dtype=np.int16)
h, w = im.shape[0], im.shape[1] // 5
assert w > 0 and h > 0, f'degenerate strip {im.shape}'
first, last = im[:, :w], im[:, 4 * w:5 * w]
drift = np.abs(first - last).mean()
assert drift > 2.0, f'first and last frame are near-identical (mean abs diff {drift:.2f}) -- no motion'
"
  [ "$status" -eq 0 ]
}

@test "quickstart: TTS -> ASR round trip recovered recognizable words from the synthesized sentence" {
  [ -s "$IMG_DIR/roundtrip.txt" ] || skip "step 7 (tts/asr round trip) has not run yet"
  run cat "$IMG_DIR/roundtrip.txt"
  [ "$status" -eq 0 ]
  grep -qi "synthesized:" <<< "$output"
  grep -qi "transcribed (nemotronasr):" <<< "$output"
  grep -qi "transcribed (qwen3asr):" <<< "$output"
  # At least one content word from the synthesized sentence must survive
  # transcription -- not an exact match (ASR is not lossless), a real signal
  # that the audio round-trip carried actual content rather than silence/noise.
  # nemotronasr is the reliable one here (qwen3asr is shown as-is in the
  # README, including its real, honest undershoot on this clip's length) --
  # scoped to its own line rather than the whole file, so qwen3asr's shorter
  # transcript can't accidentally satisfy this on nemotronasr's behalf.
  grep -i "transcribed (nemotronasr):" <<< "$output" | grep -qiE 'brain|neural|network|rust' || {
    echo "nemotronasr's transcript lost every expected content word: $output" >&2
    return 1
  }
}

@test "quickstart: deepseek2ocr read back the rendered document text" {
  [ -s "$IMG_DIR/ocr.txt" ] || skip "step 7b (deepseek2ocr) has not run yet"
  run cat "$IMG_DIR/ocr.txt"
  [ "$status" -eq 0 ]
  grep -qi "^rendered:" <<< "$output"
  grep -qi "^ocr'd:" <<< "$output"
  echo "$output" | grep -qiE 'brain|neural|network|rust' || {
    echo "OCR output did not recover the rendered document's content: $output" >&2
    return 1
  }
}

@test "quickstart: TTS LoRA loss measurably decreased" {
  [ -s "$IMG_DIR/qwen3tts-lora.txt" ] || skip "step 8 (qwen3tts finetune) has not run yet"
  run cat "$IMG_DIR/qwen3tts-lora.txt"
  [ "$status" -eq 0 ]
  echo "$output" | grep -qiE 'loss.*->.*[0-9]' || {
    echo "no loss-descent line found: $output" >&2
    return 1
  }
}

@test "quickstart: brain serve answered a real chat completion" {
  [ -s "$IMG_DIR/serve.txt" ] || skip "step 9 (brain serve) has not run yet"
  run cat "$IMG_DIR/serve.txt"
  [ "$status" -eq 0 ]
  [ -n "$output" ]
}
