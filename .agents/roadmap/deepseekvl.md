# deepseekvl - roadmap

DeepSeek-VL-7B-chat (`crates/deepseekvl`): SAM-B at 1024 px and SigLIP-L at
384 px, a split aligner, and a 30-layer Llama decoder at the checkpoint's own
fp16. The composite matches the pinned reference stage by stage and to 16
greedy tokens (`tests/composite_parity.rs`), and is served as
`brain/deepseekvl` with its towers and decoder on separate cards
(`crates/catalog/src/resident_deepseekvl.rs`, `tests/e2e/deepseek_vl.bats`).
