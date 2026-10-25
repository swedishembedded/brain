<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# chat-full-finetune

Full-parameter fine-tuning of a chat model with `brain::ChatFullFineTune`:
every weight of the base trains on a chat dataset, and the run writes a
checkpoint directory that `brain serve` serves and a later fine-tune (full or
LoRA) continues.

## What it needs

- A base: a checkpoint file or directory, or a `vendor/repo` name in the
  model store (`brain pull Qwen/Qwen3-1.7B`).
- A chat dataset: one `{"messages": [{"role": "user", "content": "...",
  "train": false}, {"role": "assistant", "content": "...", "train": true}]}`
  object per line; `train` says which messages' tokens are supervised.
- A card that holds the model's weights and gradients in fp32 (eight bytes
  per parameter) plus one step's activations; the AdamW moments and master
  weights (twelve bytes per parameter) live in host memory. The run refuses
  before loading when the cards cannot hold its plan.

## Run

```bash
make samples/text/chat-full-finetune/build
target/release/sample-text-chat-full-finetune --base Qwen/Qwen3-1.7B \
    --dataset train.jsonl --out runs/full-1 --steps 50 --device gpu0
```

It prints the losses before the first step and after the last, the training
block, the seed (pass it back with `--seed` to repeat the batch order) and
where the checkpoint and its training record were written. Run it under
`/usr/bin/time -v` for the host peak and beside `nvidia-smi
--query-gpu=memory.used -lms 500` for the card's.
