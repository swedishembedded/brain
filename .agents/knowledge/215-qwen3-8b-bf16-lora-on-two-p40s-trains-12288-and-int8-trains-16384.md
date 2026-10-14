<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 215. Qwen3-8B LoRA (rank 16) on two P40s: bf16 base trains 12288 tokens, int8 base trains 16384

What was asked: peak device memory and seconds per optimizer step of a LoRA
SFT of Qwen3-8B at 4096, 8192, 16384 and 24576 tokens on the two cards, and
where it stops.

## Setup

Two Tesla P40 (24 GiB each, 21.9 GiB reported free by the placer), Vulkan
backend. Release `brain` (`make build/release`). Batch 1, rank 16, alpha 32,
all 7 projections, AdamW, clip 1.0, one example per run that renders to
exactly `T` tokens (user turn `Continue.`, answer of `T - 16` ` the` tokens),
`--block T`. The base is the store's fp32 `model.brain.safetensors`;
`--base-dtype bf16|int8` converts the frozen linears on load, adapters and
AdamW state stay fp32. The backward already recomputes each layer from its
saved residual (one layer of activations plus a residual per boundary), and
attention is the flash kernel, so there is no further activation-saving flag:
the knobs that exist are `--base-dtype` and the number of cards.

Harness: `tools/bench/bench_qwen3_lora_memory.py` (standard library only).
Per length it runs a 2-step and a 1-step run; seconds per step is the
post-load time of the 2-step run minus that of the 1-step run. It reads
`nvidia-smi` before every run and waits for idle cards, samples
`memory.used` per card once a second, discards and repeats a run during which
a second compute process was seen, kills a run whose placer homes a stage on
the host CPU (no layout over the cards: the training kernels do not run
there), and records the memory still held after the run (0 MiB every time).

```
python3 tools/bench/bench_qwen3_lora_memory.py --brain target/release/brain \
  --models-dir DIR --workdir DIR --lengths 2048,4096,8192 --base-dtype bf16 --repeats 2
```

## Results (release binary, both cards, placer-chosen layout)

| T | base | outcome | peak MiB (card A / card B) | s/step (pairs) |
|---|---|---|---|---|
| 2048 | bf16 | trains, whole model on ONE card | 21141 / - | 30.8, 28.2 |
| 4096 | bf16 | trains, 2 stages | 11893 / 14216 | 64.7, 83.7 |
| 8192 | bf16 | trains, 2 stages | 15606 / 17992 | 153.7, 129.2 |
| 12288 | bf16 | trains, 2 stages | 21079 / 20007 (1-step run, no other process) | about 267 (one pair, the 2-step run shared the cards with another process in all four tries) |
| 16384 | bf16 | refused by the placer: no all-GPU layout (estimate 21.9 / 25.9 GiB uneven, about 24 GiB per card even, against 20.9 usable) | - | - |
| 16384 | int8 | trains, 2 stages, 2 steps, loss finite | 22587 / 21001 | 418 (one pair) |
| 24576 | int8 | refused by the placer (estimate 25.7 / 29.6 GiB for two stages, past 2 x 21.9 GiB) | - | - |

The per-step spread at 4096 and 8192 (about 20 percent) is run-to-run noise
of a card another process also touched between runs; the pairs are all
reported rather than averaged. 16384 and 12288 have a single clean pair
because one step is 7 to 25 minutes. 24576 was not run: the placer's
estimate is within 1 GiB of the measured peaks at 4096 and 8192 (11.5 / 15.5
estimated against 11.9 / 14.2 measured, 15.0 / 19.0 against 15.6 / 18.0), so
a 29.6 GiB stage is a genuine capacity limit, not a conservative one.

One P40 does not hold bf16 at 4096 tokens in the earlier measurement
(24151 MiB requested); at 2048 the placer fits the whole model on one card at
21.1 GiB.

## Root cause of the 8192-token failure (first measurement, 3 of 3 attempts)

The panic "4331569152 bytes of device buffers were dropped without an
intervening poll_wait()" was neither a capacity limit nor a missing poll in a
loop. `Qwen::ops_act` built an int8 packing (`Ops::act`: a packed copy of the
activation plus its scales) for every linear's input whenever the base was
not entirely fp32. A bf16 or f16 weight reads the raw fp32 rows
(`Ops::matmul`), so the packing was never read. Each packing is `n * k` bytes
and is dropped as soon as the matmul step is recorded, while the recorded
step still pins the underlying wgpu buffer through its bind group, so the
bytes counted as dropped-but-unreclaimed although the device had not freed
them. Forward and backward tapes together summed to 4.03 GiB at 8192 rows on
an 18-layer stage, past the device's 4292870144-byte single-buffer ceiling,
and the next allocation tripped the guard (a backtrace from a lowered
`BRAIN_GPU_RECLAIM_CEILING` on Qwen3-0.6B at 8192 tokens ends in
`layer_fwd_steps -> Ops::act -> track`; a bf16 build of a tiny model left
24064 bytes pending where the fp32 build left 0). The same bytes were also
real device memory and two extra dispatches per linear.

Fixed: `ops_act` packs only when some weight is int8, q4 or k-quant
(`reads_f32_rows`), regression test
`building_a_bf16_base_leaves_no_dropped_device_buffers_behind`. Peak memory at
4096 tokens fell from 15352 / 17670 MiB (first measurement) to 11893 / 14216.

Remaining limitation, not changed: the guard counts a buffer as pending when
the last `DeviceBuffer` handle drops even if a recorded step still pins it
(`DeviceBuffer::is_unique` documents that steps hold no handle). Making steps
hold handles would also pin scratch-arena buffers and change reuse for every
model, so any other tape that creates and drops a large temporary per op can
still trip the guard.

## Why 12288 and 16384 needed a second fix

With the packing gone, bf16 at 12288 tokens was still refused and the placer
homed stage 1 on the host CPU, where a 2-hour attempt died on a kernel the
CPU backend cannot run. The cause was the cut: `plan_lora_layout` cut by
parameter count, but the head stage also carries the logits and
cross-entropy scratch and every stage a row-length residual per boundary, so
the second stage was about 4 GiB heavier (18.5 / 22.5 GiB estimated at 12288).
The cut is now the exact minimum of the largest `estimate_vram_bytes` over
contiguous stages (`plan_by_bytes`), test
`the_cut_balances_stage_bytes_so_a_long_row_fits_two_cards`. That made 12288
bf16 and 16384 int8 placeable, and both trained.

## Configuration that trains 16384 tokens

`--base-dtype int8` on both cards (placer chooses two stages). bf16 does not
place at 16384 (about 24 GiB per card against 20.9 usable), and int8 at
24576 estimates 25.7 / 29.6 GiB per stage.

Rule: a plan that "fits" by the placer's estimate is not a measured fit, and a
guard failure is not a capacity limit until the buffer it counted has been
identified. The first rung that fails gets a backtrace and a small control run
before it is written down as a limit.
