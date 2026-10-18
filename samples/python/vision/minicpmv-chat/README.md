<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: vision/minicpmv-chat

Ask MiniCPM-V 4.6 about one or more images, or have it call a tool, over
brain's OpenAI-compatible HTTP surface.

```bash
brain pull openbmb/MiniCPM-V-4.6
python3 samples/python/vision/minicpmv-chat/minicpmv_chat.py --image photo.png
python3 samples/python/vision/minicpmv-chat/minicpmv_chat.py --image a.png --image b.png --question "How do these differ?"
python3 samples/python/vision/minicpmv-chat/minicpmv_chat.py --tool
```

Each launches (and stops) its own `brain serve --openai`. Point at a server
you already launched instead with `--base-url` and `--keys-file`.

## What it demonstrates

* Images as OpenAI `image_url` data URLs beside the question: brain lays out
  MiniCPM-V's placeholders - each image's source view and, for a large
  image, its slices - where the image part stands.
* Several images in one message.
* Tool calling: the model answers with an XML function call that brain
  returns as a structured `tool_calls` entry.

`--model brain/mock` runs offline against the weight-free mock.

## Options

| flag | default |
|---|---|
| `--model ID` | `brain/minicpmv` |
| `--image PATH` (repeatable) | the quickstart's seed image |
| `--question TEXT` | `What is in this image? Answer in one sentence.` |
| `--tool` | off |
| `--max-tokens N` | `128` |
| `--base-url URL` | unset (self-launches a server instead) |
| `--api-key KEY` / `--keys-file PATH` | required with `--base-url` |
| `--brain PATH` | `./target/release/brain` (self-launch mode only) |
| `--port N` | `8789` (self-launch mode only) |
