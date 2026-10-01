# deepseekvl - roadmap

DeepSeek-VL-7B-chat (`crates/deepseekvl`): SAM-B at 1024 px and SigLIP-L at
384 px, a split aligner, and a 30-layer Llama decoder at the checkpoint's own
fp16. The composite matches the pinned reference stage by stage and to 16
greedy tokens (`tests/composite_parity.rs`), and is served as
`brain/deepseekvl` with its towers and decoder on separate cards
(`crates/cli/src/resident_deepseekvl.rs`, `tests/e2e/deepseek_vl.bats`).

## Outstanding

- **Fine-tuning beyond one card and one image.** The trainer holds a
  bf16 decoder on one card, which bounds the context (each image is 576
  rows). A decoder split across cards needs the decoder hook to hand the
  image rows' gradient across a pipeline stage boundary, and an example
  with several images needs the splice to take several regions.
- **Serial decoding.** Each request prefills and decodes alone on
  `qwen3::Qwen`. The serving engine now has what batching needs (a
  half-precision tier and `Engine::prefill_mixed`), so concurrent requests
  could share its paged decode.
- **Single-card hosts.** The towers need about 8 GB (SAM's attention at 1024
  px) beside a 15 GB decoder, so a 24 GB card alone does not serve the model.
  An int8 decoder, or towers that free their activations between images,
  would make it fit.
