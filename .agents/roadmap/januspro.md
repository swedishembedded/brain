# januspro - roadmap

Janus-Pro-7B (`crates/januspro`): understanding through `brain-deepseekvl`'s
composite with one SigLIP-L tower, and text to image by classifier-free
guided sampling of 576 VQ-16 tokens on the serving engine at bf16. Both
match the pinned reference (`tests/reference_parity.rs`, `tests/t2i_real.rs`)
and are served as `brain/januspro` (`generate`, `text2image`;
`crates/cli/src/resident_januspro.rs`, `tests/e2e/januspro.bats`).

## Outstanding

- **Training.** Understanding fine-tuning as for DeepSeek-VL, plus the
  generation objective: cross-entropy of the generation head's logits on
  VQ-16-encoded target images, back through the decoder into a LoRA
  adapter, with a trainable generation head and aligner. Needs the decoder
  to take a gradient at its hidden rows from a head other than its own.
- **More than one image per request.** `TextToImage` draws `parallel`
  images in one batch; the served action builds it for one, and the API
  loops over `n`.
- **Concurrent requests** each run their own 576 steps; merging them into
  one batch of guided pairs would share the decode.
- **The chat build's context** is what one card leaves beside its tower and
  decoder (about 2k tokens on a 24 GB card). Splitting it across two cards
  would triple that, but the scheduler never evicts a multi-device resident
  to make room for another, so the drawing build could then not swap in.
