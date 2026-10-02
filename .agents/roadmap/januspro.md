# januspro - roadmap

Janus-Pro-7B (`crates/januspro`): understanding through `brain-deepseekvl`'s
composite with one SigLIP-L tower, and text to image by classifier-free
guided sampling of 576 VQ-16 tokens on the serving engine at bf16. Both
match the pinned reference (`tests/reference_parity.rs`, `tests/t2i_real.rs`)
and are served as `brain/januspro` (`generate`, `text2image`;
`crates/catalog/src/resident_januspro.rs`, `tests/e2e/januspro.bats`).
