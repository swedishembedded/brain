<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 320. The SDK chat fine-tune reaches the 27B, and a trained adapter is served only under its store file name

Question: can a caller of the SDK (`brain::ChatFineTune`, the path a campaign's `train` step takes)
train a Qwen3.8-27B LoRA from a chat dataset and have `brain serve` offer it beside the base? Before
this entry the SDK refused `qwen35` ("does not provide chat fine-tuning yet") although the CLI trainer
and the served adapter bank both existed; the pieces were never joined.

What was missing and is now joined (all VERIFIED-BY-RUN 2026-10-18, brain 5c1aa4863 plus the lane
change):

* `ChatFineTune` dispatches over a `ChatFamily` (qwen3 or qwen35) for the checkpoint vocabulary, the
  LoRA trainer and the adapter save; everything else (dataset validation, packing, mix, cancel and
  exact resume, record and digests) is the one shared path. `Qwen35` now exposes its AdamW moments
  through `model::Model` (`optimized_params`, `read_moments`, `write_moments`), which is what exact
  resume needs; a cancelled qwen35 run continues to the byte-identical adapter
  (`crates/sdk/tests/chat_finetune_qwen35.rs`, tiny qwen35, CPU).
* Real 27B, NF4 base, one card: 20 invented fact records, rank 16, 12 steps, loss 2.599 to 0.499,
  467 MB adapter written with its card, base digest and dataset digest.
* Served logits equal the training forward for a file the trainer wrote
  (`crates/qwen35/tests/adapter_bank.rs`, `an_adapter_the_trainer_wrote_is_served`).

The trap: `brain serve` finds an adapter only as `<models-dir>/<vendor>/<repo>/adapters/<owner>/<name>/<tag>/adapter.brain.safetensors`.
The SDK writes `adapter.safetensors` into the run's out directory; a server pointed at a store where the
file kept that name started cleanly and listed only the base. Install under the store name, and give the
fine-tune an `adapter_id` that is an adapter reference of the served quantisation
(`unsloth/Qwen3.8-27B-GGUF-Q8_0:<owner>:<name>:<tag>`), or the card id does not match the directory.

Not done: held-out scoring on qwen35 (`score_chat` is qwen3's), int8 tier, 27B DPO.
