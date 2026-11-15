<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 461. A placement plan holds only where every handle lands: no policy in a library run, and a new kernel set on CUDA opened the ambient card

**Lesson:** Byte estimates that match the measured peak do not make a placement fit. The plan also has to be the
one the run uses, and every device handle a part builds has to land on the card the plan gave that part. Two gaps
broke this for FLUX.1 and PuLID while the estimates themselves were within a few percent of the real figures.

## What happened

FLUX.1-dev's real-weight cancellation proof (int8 DiT, 256 px, two Tesla P40s) ran out of memory with automatic
placement, so the test pinned `te_device` to the second card. Measured on the card (driver figure, the context
excluded): the int8 DiT took 13,948 MiB against a 12,978 MiB estimate (the 13,032 MiB of buffers requested plus
the driver's block rounding, which the placement headroom's allocator share covers), and T5-XXL at 512 tokens took
22,280 MiB against 22,199 MiB. Neither estimate was wrong. The causes were elsewhere:

- **No policy was installed.** `gpu_core::devices::place` with no `Placer` answers every part with the ambient card,
  ignoring sizes and `Apart`. A model crate's own test (and any embedder that skips `loader::install_default_placer`)
  therefore put the DiT and T5-XXL on gpu0 together, and the run printed `placement dit=gpu0 te=gpu0` before failing.
- **With the policy installed, T5-XXL still missed a card.** Its 22,199 MiB exceeded the 20.5 GiB an idle 24 GiB card
  budgets after headroom, so the plan sent it to the host tier although the card held it. Most of that figure was
  per-block scratch only a parity test reads: built with one block's scratch shared by all blocks
  (`t5encoder::model::Taps::Output`), the encoder measures 18,600 MiB, and the plan splits the two parts over the
  two cards by itself.
- **PuLID's identity stack landed on the encoder's card.** It is built with `gpu.new_like(..)` from a handle scoped to
  the DiT's card. The CUDA backend implemented `share` but not `new_like`, and the facade's fallback for a backend
  without it is a fresh `Gpu::new` on the AMBIENT card, which was the text encoder's. That card then held about
  4.6 GiB the plan never charged it (23,412 MiB peak next to an 18,600 MiB encoder).

## What to do

- A real-weight test that claims to run "the automatic plan" installs the production policy first
  (`loader::install_default_placer`, a dev-dependency), or it tests the no-policy default instead.
- `Backend::new_like` is part of the device contract: a backend that can open its own card again implements it, so a
  handle built from another never wanders to the ambient card.
- Price an encoder by the build the pipeline makes, from the encoder crate's own account of its allocations
  (`t5encoder::model::device_bytes`), and pin that account to a measured peak.

## Evidence

`crates/flux1/tests/it/placement_real.rs` (both estimates against measured peaks, `-5%..+ALLOC_OVERHEAD_PERCENT`),
`flux1::pipeline::placement_tests::an_int8_run_splits_over_two_idle_cards_and_spills_the_encoder_on_one`,
`crates/gpu-core/tests/it/cuda_new_like.rs`, `t5encoder` `smoke::an_output_only_build_matches_the_per_block_build_bit_for_bit`,
and the two mid-run cancellation proofs (`flux1::caps`, `pulid::caps`) now running on the automatic plan.
