# januspro - roadmap

Janus-Pro-7B (`crates/januspro`): understanding through `brain-deepseekvl`'s
composite with one SigLIP-L tower, and text to image by classifier-free
guided sampling of 576 VQ-16 tokens on the serving engine at bf16. Both
match the pinned reference (`tests/reference_parity.rs`, `tests/t2i_real.rs`)
and are served as `brain/januspro` (`generate`, `text2image`;
`crates/cli/src/resident_januspro.rs`, `tests/e2e/januspro.bats`).

## Outstanding

- **Serving a fine-tune.** `brain januspro finetune` writes an adapter
  and either an aligner or the generation heads; the served builds cannot
  attach them yet (no adapter store layout for the composite, and the
  engine folds adapters only for the plain qwen3 residents).
- **Generation training beyond one card and one image per example.** The
  trainer holds a bf16 decoder on one card, one image's 576 tokens per step,
  and does not guide its loss: a sampled image is the check, not a held-out
  number.
- **More than one image per request.** `TextToImage` draws `parallel`
  images in one batch; the served action builds it for one, and the API
  loops over `n`.
- **Concurrent requests** each run their own 576 steps; merging them into
  one batch of guided pairs would share the decode.
- **The chat build's context** is what one card leaves beside its tower and
  decoder (about 2k tokens on a 24 GB card). Splitting it across two cards
  would triple that, but the scheduler never evicts a multi-device resident
  to make room for another, so the drawing build could then not swap in.
