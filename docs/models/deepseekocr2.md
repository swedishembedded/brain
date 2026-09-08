# DeepSeek-OCR-2 (document image → text/markdown)

The successor to [DeepSeek-OCR](deepseek2ocr.md), reusing the same
DeepSeek-V2-family MoE decoder unmodified. Its vision front end is new: SAM
ViT-B feeds a 24-layer Qwen2-shaped GQA tower run as a learned-query
resampler under a prefix-LM attention mask (image tokens attend to each
other bidirectionally, learned query tokens attend causally over everything
before them), then a single linear projector - replacing v1's SAM + 16x
conv compressor + CLIP-L/14 arrangement.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| CLI (`brain <arch> <action>`)      | [x] |
| HTTP API              | [x] |
| D-Bus                 | [x] |
| Batched serving       | [ ] |

Global view only today - a document image is fit into the model's one
`1024x1024` view; the multi-tile "Gundam" layout needs a SAM
position-embedding resample not yet implemented (`.agents/roadmap/
deepseekocr2.md`). Decode is full-recompute rather than KV-cached (see
`deepseekocr2::caps`'s own doc for why and what would change that).

## Getting the weights

Model id: `deepseek-ai/DeepSeek-OCR-2`. Point `BRAIN_DEEPSEEKOCR2_DIR` at a
directory holding both:

```text
mmproj-deepseek-ocr-2-q8_0.gguf
deepseek-ocr-2-q8_0.gguf
```

No vendor-published GGUF exists for this model as of this writing (only
community conversions of the upstream `deepseek-ai/DeepSeek-OCR-2`
safetensors checkpoint) - `brain pull`/auto-fetch is not wired to one, so
the pair must be placed manually.

## Running it

```
BRAIN_DEEPSEEKOCR2_DIR=<dir> brain deepseekocr2 generate --in image=doc.png --prompt "Free OCR"
```

Also reachable through `brain caps`, D-Bus, and the HTTP API via the
generic `capability::Provider` surface.

## Full content pending

This page is a minimal stub, required by `check-arch-names.sh` once the
architecture registered. The full options table, hardware/limits section,
and a worked example are M11's job - see `.agents/roadmap/deepseekocr2.md`.
