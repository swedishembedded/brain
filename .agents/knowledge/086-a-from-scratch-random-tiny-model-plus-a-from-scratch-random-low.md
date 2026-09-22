<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 86. A from-scratch-random tiny model plus a from-scratch-random low-rank LoRA adapter can converge to a real, non-trivial local minimum within ~100 steps - repeatably, across rank/lr/clipping - and that is not automatically a wiring bug

Writing `qwen3vl::train_smoke`'s LoRA convergence check, a single-example
overfit plateaued at a materially non-zero loss with gradients that were
genuinely shrinking toward zero (confirmed by dumping `read_decoder_grad`
per parameter), unmoved across LoRA rank 4/8/16, learning rates 1e-2 to
1e-1, and with/without gradient clipping - all landing on essentially the
SAME plateau. Ruled out as a `Qwen3Vl`/`crate::finetune` wiring bug by
swapping the identical composite to FULL (non-LoRA) training on the same
single example: it collapsed to ~0 within the same ~100 steps. The real
cause: LoRA's forward computes `dA` via `B` (`lora_fwd`/`proj_bwd`,
`crates/qwen3/src/model.rs`) - `A`'s gradient is exactly zero at the
standard `B = 0` init (already documented on `gradcheck::check_qwen_lora`)
and stays SMALL for as long as `B` stays small, so a from-scratch random
architecture with a frozen, TIED, `d_model < vocab` head can settle into a
local minimum of the LOW-RANK-CONSTRAINED loss landscape before `A` ever
moves far enough to escape it. A real pretrained checkpoint does not have
this failure mode nearly as often (its base representations are already
organized, so a small LoRA nudge finds a much better-conditioned
gradient), which is exactly why this shows up in a from-scratch synthetic
smoke test and did not show up in `fastvlm::train_smoke`'s FULL-fine-tune
precedent.

**Rule going forward**: before treating a stuck synthetic LoRA
convergence test as a forward/backward bug, dump per-parameter gradient
NORMS (not just the loss) across a few steps, and run the SAME composite
under FULL (non-LoRA) training on the SAME data as a control - if full
training collapses the loss and LoRA does not, the composite's math is
fine and the fix is in the test's own assertion (a modest, honest bound
like "loss falls below 90% of its start"), not in the model code.
