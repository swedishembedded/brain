# sample: vision/deepseek-ocr (D-Bus)

`ocr_document.py` drives DeepSeek-OCR's streaming `generate` action over
D-Bus: a document image and an instruction in, decoded text out - plain
text, markdown, or (with the `<|grounding|>` marker its default prompt
carries) text annotated with the boxes it was read from. The script is 60
lines of client; the image travels as a sealed memfd rather than bytes
marshalled through D-Bus.

## v1 - `deepseek-ai/DeepSeek-OCR`

```bash
brain pull ggml-org/DeepSeek-OCR-GGUF
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
2. `brain pull deepseek-ai/DeepSeek-OCR-2`; the server resolves it from the
   model store, no variable needed.

```bash
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/vision/deepseek-ocr/ocr_document.py --image page.ppm --max-new 8
```

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

The model pulled into the model store (`brain pull deepseek-ai/DeepSeek-OCR` or
`deepseek-ai/DeepSeek-OCR-2`).
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
