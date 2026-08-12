# DeepSeek-OCR - a page in, text out

DeepSeek-OCR reads a document image and writes back what it says: plain text,
markdown, or - with the `<|grounding|>` marker its default prompt carries -
text annotated with the boxes it was read from. It is driven through the same
`Run`/`Subscribe` methods on `com.swedishembedded.Brain1` as every other model,
with no model-specific code in the transport; `ocr_document.py` here is 60 lines
of client, and the image travels as a sealed memfd rather than bytes marshalled
through D-Bus.

| model | action | weights env |
|---|---|---|
| `deepseek-ai/DeepSeek-OCR` | `generate` - image + instruction → streamed text | `BRAIN_DEEPSEEK_OCR_DIR` (the directory holding **both** shipped GGUFs) |

```bash
brain caps deepseek-ai/DeepSeek-OCR
```

## Run it

A private session bus needs no system configuration:

```bash
BRAIN_DEEPSEEK_OCR_DIR=<dir> \
  dbus-run-session -- bash -c '
    brain serve --dbus & sleep 5
    python3 examples/vision/deepseek-ocr/ocr_document.py --image page.ppm --max-new 8'
```

Inputs are binary PPM (P6) - brain's image convention; `brain_py.image.load_ppm`
turns one into the HWC-f32 blob the wire format carries. The same action works
with no bus at all:

```bash
BRAIN_DEEPSEEK_OCR_DIR=<dir> \
  brain do deepseek-ai/DeepSeek-OCR generate \
    --prompt "<|grounding|>Convert the document to markdown." \
    --max_new 8 --in image=page.ppm --json
```

...and, because the manifest is chat-capable-shaped, over the OpenAI and
Anthropic HTTP surfaces `brain serve` exposes, with the same streaming and real
`prompt_tokens` / `completion_tokens` / `finish_reason`.

## What to expect, and why the script prints seconds

**This model is slow, structurally.** It runs on the CPU backend (the SAM tower
is not correct on wgpu at 1024², a tracked bug), it holds ~22 GiB resident, and
its decoder has **no KV cache** - each generated token re-runs the whole
sequence through 12 MoE layers. Measured on 22 cores: **~22 s per token**, so
`--max-new 10` is about five minutes and `--max-new 32` about twelve.
`ocr_document.py` prints the wall time beside every streamed fragment, which is
what makes that concrete instead of a warning you skim. The page itself is
encoded once, before the first token, not per step.

The first request on a fresh server additionally pays activation: importing the
mmproj, expanding the decoder to fp32 on disk the first time ever (~12 GB,
cached beside the checkpoint), and uploading ~22 GiB of weights. Later requests
reuse the resident instance.

`brain caps deepseek-ai/DeepSeek-OCR` prints the full option list; the model's
own page in the documentation tree (the model catalog's DeepSeek-OCR entry)
carries the honest limits - single global view, greedy only, batch 1, no early
stop at EOS.
