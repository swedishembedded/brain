<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 188. A near-unit codebook hides a missing normalize

LlamaGen's VQ-16, the image tokenizer in Janus-Pro, is trained with
`codebook_l2_norm=True`. Every read of `quantize.embedding.weight` goes
through `F.normalize`, both in the nearest-neighbour search and in the
decode-side `get_codebook_entry`. The stored table is therefore never used
raw. Training with the constraint also left its row norms close enough
to 1 that the raw table looks nearly correct.

Decoding the real Janus-Pro weights with the raw table gave these numbers
against the reference:

| stage | cosine | rel L2 |
|---|---|---|
| `post_quant_conv` | 0.99953 | 4.8e-2 |
| `decoder.mid.2` | 0.99938 | 4.1e-2 |
| `decoder.conv_blocks.2.upsample` | 0.99954 | 3.5e-2 |

At a cosine floor of 0.999, every stage up to the head passes. With
normalization every stage reaches 1-cos ≤ 5.8e-12 and rel L2 ≤ 3.4e-6. The
only visible failure was at `conv_out`, cosine 0.97, and it had a different
cause: the missing head SiLU.

Three rules follow:

- Gate on relative L2 as well as cosine. A missing normalization leaves
  the direction almost unchanged. The magnitude error is 4e-2, far above
  a 1e-4 ceiling.
- Normalize a trained, constraint-shaped weight on every read that the
  reference normalizes. Do not assume the stored copy already satisfies
  the constraint because it looks close.
- Where the value of `beta` goes is configuration, not a detail. `basicsr`
  applies it to the codebook term (`vqgan_arch.py:55`). LlamaGen applies it
  to the commitment term. A finite-difference check cannot distinguish the
  two, because it tests the backward pass against the forward that was
  emitted. `vqgan::BetaTerm` records the choice.
  `train::tests::beta_weights_the_term_its_config_names` pins it by the exact
  `beta` ratio of the codebook gradient. A forward feature missing from the
  training graph is also invisible to finite differences. It is pinned by
  `llamagen_trainer_forward_matches_the_inference_graph`.
