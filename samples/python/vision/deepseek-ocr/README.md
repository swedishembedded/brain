# sample: vision/deepseek-ocr (D-Bus)

`ocr_document.py` drives DeepSeek-OCR's streaming `generate` action over
D-Bus: a document image and an instruction in, decoded text out - plain
text, markdown, or (with the `<|grounding|>` marker its default prompt
carries) text annotated with the boxes it was read from. The script is 60
lines of client; the image travels as a sealed memfd rather than bytes
marshalled through D-Bus.

## v1 - `deepseek-ai/DeepSeek-OCR`

```bash
BRAIN_DEEPSEEK_OCR_DIR=<dir with both DeepSeek-OCR GGUFs> \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/deepseek-ocr/ocr_document.py --image page.ppm --max-new 8
```

## v2 - `deepseek-ai/DeepSeek-OCR-2`

DeepSeek-OCR-2 reuses the exact same decoder and the same `generate` action
shape as v1 - only the vision front end changed - so this repo ships ONE
copy of `ocr_document.py` rather than a near-duplicate that would drift out
of sync over a three-line difference. To point it at v2 instead of v1,
edit two things in the script:

1. `MODEL = "deepseek-ai/DeepSeek-OCR"` (near the top) -> 
   `MODEL = "deepseek-ai/DeepSeek-OCR-2"`
2. Serve the model with `BRAIN_DEEPSEEKOCR2_DIR` instead of
   `BRAIN_DEEPSEEK_OCR_DIR`.

```bash
BRAIN_DEEPSEEKOCR2_DIR=<dir with both DeepSeek-OCR-2 GGUFs> \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/deepseek-ocr/ocr_document.py --image page.ppm --max-new 8
```

No vendor-published GGUF exists for DeepSeek-OCR-2 yet - obtain one
yourself and point `BRAIN_DEEPSEEKOCR2_DIR` at the directory holding both
files (see the model catalog's DeepSeek-OCR-2 entry for what a compatible
checkpoint pair must carry). v2 also serves its **global view only** today
(a document image fit into one 1024x1024 view) - the multi-tile layout v1
doesn't serve over D-Bus either needs the same SAM position-embedding work
either model would need.

## What to expect, and why the script prints seconds

**Both versions are slow, structurally.** The model runs on the CPU
backend (the SAM tower is not correct on wgpu at 1024^2, a tracked bug),
holds ~22 GiB resident, and its decoder has **no KV cache** - each
generated token re-runs the whole sequence through 12 MoE layers. Measured
on 22 cores: **~22 s per token**, so `--max-new 10` is about five minutes
and `--max-new 32` about twelve. `ocr_document.py` prints the wall time
beside every streamed fragment, which is what makes that concrete instead
of a warning you skim. The page itself is encoded once, before the first
token, not per step.

The first request on a fresh server additionally pays activation:
importing the mmproj, expanding the decoder to fp32 on disk the first time
ever (~12 GB, cached beside the checkpoint), and uploading ~22 GiB of
weights. Later requests reuse the resident instance.

## What it demonstrates

`generate` also works with no bus at all, and - because the manifest is
chat-capable-shaped - over the OpenAI and Anthropic HTTP surfaces
`brain serve` exposes, with the same streaming and real `prompt_tokens` /
`completion_tokens` / `finish_reason`:

```bash
BRAIN_DEEPSEEK_OCR_DIR=<dir> \
  brain deepseek2ocr generate \
    --prompt "<|grounding|>Convert the document to markdown." \
    --max_new 8 --in image=page.ppm --json
```

## What it needs

`BRAIN_DEEPSEEK_OCR_DIR` (v1) or `BRAIN_DEEPSEEKOCR2_DIR` (v2), each
pointing at a directory holding **both** shipped GGUFs for that version.
Input is a binary PPM (P6); `brain caps deepseek-ai/DeepSeek-OCR[-2]`
prints the full option list, and the model catalog's own page carries the
honest limits - single global view, greedy only, batch 1, no early stop at
EOS.

## Options

| flag | default |
|---|---|
| `--image PATH` | *required* - binary PPM (P6) of the page |
| `--prompt TEXT` | `<\|grounding\|>Convert the document to markdown.` |
| `--max-new N` | `8` (every token is a full recompute) |
