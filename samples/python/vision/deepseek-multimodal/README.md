<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: vision/deepseek-multimodal

Ask DeepSeek-VL or Janus-Pro about an image, or have Janus-Pro draw one, over
brain's OpenAI-compatible HTTP surface.

```bash
brain pull deepseek-ai/deepseek-vl-7b-chat
python3 samples/python/vision/deepseek-multimodal/deepseek_multimodal.py chat --image photo.png

brain pull deepseek-ai/Janus-Pro-7B
python3 samples/python/vision/deepseek-multimodal/deepseek_multimodal.py chat --model brain/januspro
python3 samples/python/vision/deepseek-multimodal/deepseek_multimodal.py draw --prompt "A red apple on a wooden table." --out /tmp/apple.png
```

Each launches (and stops) its own `brain serve --openai`. Point at a server
you already launched instead with `--base-url` and `--keys-file`.

## What it demonstrates

* An image as an OpenAI `image_url` data URL beside the question: brain places
  the model's image placeholder where the image part stands in the message.
* Text to image through `/v1/images/generations` at `384x384`, the one size
  Janus-Pro generates; any other size is refused.
* DeepSeek-VL needs two 24 GB cards (its towers on one, the decoder on the
  other); Janus-Pro serves each of its two builds from one.

`--model brain/mock` runs either mode offline against the weight-free mock.

## Options

| flag | default |
|---|---|
| `chat` / `draw` | required |
| `--model ID` | `brain/deepseekvl` (chat), `brain/januspro` (draw) |
| `--image PATH` | the quickstart's seed image |
| `--question TEXT` | `What is in this image? Answer in one sentence.` |
| `--max-tokens N` | `128` |
| `--prompt TEXT` | `A red apple on a wooden table.` |
| `--seed N` | `7` |
| `--out PATH` | `/tmp/janus.png` |
| `--base-url URL` | unset (self-launches a server instead) |
| `--api-key KEY` / `--keys-file PATH` | required with `--base-url` |
| `--brain PATH` | `./target/release/brain` (self-launch mode only) |
| `--port N` | `8788` (self-launch mode only) |
