# T5-XXL / umT5-XXL encoder (not yet servable)

The text encoder [FLUX.1](flux1.md) conditions on, and the one
[Wan2.1 video](wan.md) conditions on: a bidirectional encoder-only T5
(RMSNorm, no bias, learned relative-position bucket bias, gated-GELU FFN), in
two variants that share one implementation.

|  | T5-XXL v1.1 | umT5-XXL |
|---|---|---|
| used by | FLUX.1 / FLUX.2 | Wan2.1 / 2.2 |
| vocabulary | 32128 | 256384 (multilingual) |
| relative bias | one table, shared by all 24 blocks | **one table per block** |
| attention mask | none (right pad is attended) | **key padding, 512-token window** |
| parameters | 4.762 B (19.05 GB fp32) | 5.681 B (22.72 GB fp32) |
| tensors | 219 -> 171 | 242 -> 194 |

Both are real, verified ports: imported with two-way coverage and
forward-parity-gated per stage (T5-XXL: 42/42 stages, worst cosine
0.9999999992). The umT5 side additionally has a SentencePiece **unigram
tokenizer** (`data::unigram`, the first non-BPE tokenizer in the workspace),
gated to exact-id equality against the real `google/umt5-xxl` tokenizer over a
nine-prompt multilingual corpus.

Two things are worth knowing before trusting a change here. The per-block
relative bias is a **silent** difference - a port that shares block 0's table
produces plausible embeddings and subtly wrong video, so the parity test checks
block 0 and block 23 separately (they differ by max_abs 53 in the released
checkpoint). And the 512-token pad is applied **after** the encoder as hard
zeros, not taken from the encoder's output at the pad positions, which is not
small: that output peaks at 0.87.

Still missing: INT8, the serving contract, and training for the umT5 variant
(the trainer folds one shared bias gradient across the block stack and attends
over every key, so it refuses a per-block or masked config rather than
returning a wrong gradient). umT5 does not fit a 24 GB card in fp32, so its
gate is a `BRAIN_DEVICE=cpu` test today.

Package: `brain-t5encoder`.
