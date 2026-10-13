<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 215. Qwen3-8B bf16 LoRA (rank 16) fits 4096 tokens on two P40s and nothing longer was measured

What was asked: peak device memory and seconds per optimizer step for a LoRA
SFT of Qwen3-8B (bf16 frozen base, rank 16) at 4096, 8192, 16384 and 24576
tokens, on one P40 and on both.

Setup. Two Tesla P40 (24 GiB each, 21.9 GiB reported free by the placer),
driver 570.195.03, Vulkan backend, both cards idle before every run (0-1 MiB
used, no compute process) and back to 0 MiB after every run. The base is the
store's fp32 `model.brain.safetensors`; `--base-dtype bf16` converts the
frozen linears on load, the adapters and AdamW state stay fp32. Batch 1, rank
16, alpha 32, all 7 projections, AdamW, clip 1.0, one training example per
run: a user turn `Continue.` and an assistant answer of `T - 16` repeated
` the` tokens, so the rendered example is exactly `T` tokens and `--block T`
is the row. A debug-profile `brain` binary was used (no release binary existed), so
host-side time (load, conversion,
first-step setup) is not representative; the post-load step time is dominated
by device work.

Command (the global `--device gpu0` pins one card; omitting it lets the
placer choose, and it printed the layout it chose):

```
brain [--device gpu0] qwen3 finetune --lora 16 --base-dtype bf16 \
  --weights Qwen/Qwen3-8B --adapter <owner>/<name>:<tag> --dataset <dir> \
  --models-dir <dir> --steps <1|2> --batch 1 --block <T> --seed 1
```

Peak MiB is the maximum of `nvidia-smi` `memory.used` per card, sampled once a
second over the run. Seconds per step is the post-load time of a 2-step run
minus that of a 1-step run (the load finishes at the `qwen3 load ... 100%`
line; the difference removes load, setup and the first step's one-time
costs). That is one pair per length, so it carries run-to-run noise that was
not separately measured. The method was checked first on Qwen3-0.6B at 2048
tokens: 4.3 s per step, the same order as the 3.61 s a release
`qwen_bench lora-train` reports for that length (entry 193).

| T | cards | outcome | peak MiB gpu0 / gpu1 | s/step |
|---|---|---|---|---|
| 4096 | one P40 (`--device gpu0`) | out of memory while building the trainer ("this device has requested 24151.5 MiB across 2569 buffers") | 24240 / - | none |
| 4096 | both (placer: 2 stages, layers 0..18 and 18..36) | 2 steps completed, loss 0.0357 -> 0.0243 | 15352 / 17670 (1-step run: 15349 / 17670) | 61.8 (66 tok/s) |
| 8192 | both (same layout) | failed in both of 2 attempts, 218 s and 221 s in, before the load finished: backend guard panic "4331569152 bytes of device buffers were dropped without an intervening poll_wait()" | 22134 / 1 and 15894 / 1 (gpu1 never built) | none |
| 16384, 24576 | - | not run: the ladder stops at the first failure | - | absent |

A tiny 0.6B run (512 tokens, one card) between the two 8192 attempts
succeeded, so the 8192 failure is not a wedged driver.

What this says and does not say.

- On one P40 the model does not train even at 4096 tokens: the pinned device
  skips the placer, and the trainer asks for 24151 MiB of a card that holds
  about 22.4 GiB.
- With both cards the placer splits the model into two pipeline stages and
  4096 tokens trains with 6 GiB of headroom on gpu0 and 4 GiB on gpu1.
- 8192 tokens does not train on the two-card layout today. The failure is the
  buffer-reclamation guard, not a reported out-of-memory, and gpu0's reading
  reached 22 GiB in one attempt, so capacity is the likely cause but was not
  separated from the guard. No length between 4096 and 8192 was tried.
- Nothing is claimed for 8192 and above, and no number for them exists here.

Rule: a plan that "fits" by the placer's estimate is not a measured fit. The
first rung that fails gets a second attempt and a small control run before it
is written down as a limit.
