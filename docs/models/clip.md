# CLIP (text & image embeddings)

Turns text or images into vector embeddings you can compare for search,
similarity, and classification - a CLIP-family encoder combining CLIP-L and
OpenCLIP-bigG (two text towers) with an EVA-CLIP image tower.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| CLI (`brain <arch> <action>`)       | [x] |
| HTTP API               | [ ] `/v1/embeddings` dispatches the literal action `embed`; these are `embed_text`/`embed_image` (per-tower), so the route would 404 the model rather than mis-dispatch it |
| D-Bus                  | [x] |
| Batched serving        | [x] |

## Getting the weights

Model id: `brain/clip`. Set `BRAIN_CLIP_DIR` to a checkpoint root in the SDXL
layout: `text_encoder/` (CLIP-L) and/or `text_encoder_2/` (OpenCLIP-bigG)
weight directories, `tokenizer/` and/or `tokenizer_2/` BPE directories (at
least one tokenizer must be present), and the EVA-CLIP image tower file
(`EVA02_CLIP_L_336_psz14_s6B.pt`) at the root if you want image embeddings.

## Running it

```bash
brain caps brain/clip
BRAIN_CLIP_DIR=<ckpt-root> brain clip embed_text --text "a photo of a cat" \
    --tower clip_l --out embedding=text.bin --json
BRAIN_CLIP_DIR=<ckpt-root> brain clip embed_image \
    --in image=photo.ppm --out embedding=img.bin --json
```

Both actions are also reachable over D-Bus.

## Options

- `embed_text` - `text` (required), `tower` (`clip_l`, 768-d, default; or
  `openclip_bigg`, 1280-d). Returns an f32 embedding.
- `embed_image` - an input `image` (resized to the tower's 336² input).
  Returns an L2-normalised f32 embedding.

## Hardware and limits

Text embedding batches efficiently (many strings in one forward pass); image
embedding processes one image per call. There is no fine-tuning/LoRA path
exposed on the CLI, and this model is not reachable through the HTTP
embeddings endpoint - use `brain do` or D-Bus.
