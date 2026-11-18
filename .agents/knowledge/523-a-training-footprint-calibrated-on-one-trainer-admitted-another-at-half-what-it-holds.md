<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 523. A training footprint calibrated on one trainer admitted another at half what it holds

**Lesson:** A reservation is the trainer's own buffers, itemised from the code that allocates them and held to the
device's allocation accounting by a test, not a per-token rate measured on a different trainer. The rate hides
which terms grow with the row (boundaries, scratch, logits), so it cannot be right for a trainer that keeps them
differently.

## What happened

Served LoRA training of Qwen3-4B (bf16 base, every projection, `max_block` 32768) was admitted at 25.9 GiB and ran
out of memory on a 96 GB GH200. At 16384 tokens one seed ran out of memory on its first step until the server ran
with `--block-cache-mb 0`.

`brain::ChatFineTune::footprint` priced every chat trainer's activations by one rate:
`model::memplan::calibrated_activation_bytes`, 1.4 MiB a token at 64 x 5120, measured on the 27B's streamed
trainer and scaled by depth times width. For Qwen3-4B that is 0.39 MiB a token.

The qwen3 LoRA trainer recomputes each layer from its fp32 boundary. It keeps 37 boundaries of `[n, 2560]`, one
layer's scratch and the backward scratch. That is about 0.88 MB a token. It also keeps one 3,520-row chunk of
logits and their gradient over the 151,936-token vocabulary, 4.3 GB at any row length past 3,520 tokens. The rate
counted no logits at all. Nothing counted the freed blocks the CUDA backend keeps (up to 4 GiB on a 96 GB card).
These blocks are what `--block-cache-mb 0` gave back.

## Fix

- `qwen3::footprint::lora_training_demands` itemises a LoRA training build: the base at its tier with the adapter's
  values, the adapter's gradients and AdamW moments, the boundaries in the saved format, the shared layer and
  backward scratch, the attention scores where the device runs no flash kernels, and the head's logits chunk. Four
  buffer groups the inference estimate had also missed are now counted there too: the projection outputs, the id
  rows, the final norm's output and the per-head norm statistics.
- `brain::ChatFineTune::footprint` takes it for `qwen3`, and its tier choice does too. Every footprint adds the one
  rule `model::memplan::train_reservation`: 10 %, the device context, and on a card the backend's freed-block cap
  (`gpu_core::runtime::block_cache_cap`, the CUDA backend's own rule). The refusal quotes the itemised breakdown.
- Qwen3-4B at 32768 tokens is now reserved at about 49 GiB (39 GiB of buffers), and at 16384 at about 34 GiB.

Tests:

- `qwen3` `lora_footprint` builds Qwen3-4B-ratio models on the CPU backend and runs a whole step: forward,
  backward and update. The device's peak of live bytes is at most the estimate, and the estimate is within 15 % of
  that peak. This holds at three row lengths and the three saved-activation formats. A longer row adds what the
  estimate says, to 0.3 %.
- `qwen3` `footprint` `a_long_row_of_qwen3_4b_holds_its_boundaries_scratch_and_logits` pins the Qwen3-4B numbers.

## Not covered

- A card's own additions are not measured: the GEMM library's workspace and the tensor-core operand copies. The
  ignored test `on_a_card_a_qwen3_4b_lora_step_takes_no_more_than_its_reservation` holds a 36-layer Qwen3-4B step at
  4K and 16K tokens to the reservation. It has not run on a card yet.
- The qwen35 and Ripple trainers still take the calibrated rate.
