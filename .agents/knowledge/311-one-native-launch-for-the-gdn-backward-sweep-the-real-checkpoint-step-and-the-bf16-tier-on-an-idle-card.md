<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 311. One native launch for the GDN backward sweep; the real-checkpoint 27B step and the bf16 tier on an idle card

Question: after knowledge 310 (550-560 tokens/s at W8K NF4), what do a native
reverse-sweep kernel for Gated DeltaNet, larger bf16 long-row settings and kept freed blocks
for a bf16 base add, and what does the real `brain qwen35 finetune` path (real
Qwen3.8-27B-FP8 checkpoint) reach? Every figure is **VERIFIED-BY-RUN** unless it says
**DERIVED**. Which figures are real-checkpoint and which are probe is stated per row; a probe
row (random weights, `qwen35_train_probe`) is never a statement about the checkpoint.

## Conditions

One GH200, `gpu` lock held. "Idle" means `nvidia-smi --query-compute-apps` listed no other
compute context at process start (`other_contexts_start=0`, `other_contexts_end=0` in the probe
line; the CLI runs print the context list before the run and it was empty). Between 15:10 and
15:45 two other lanes' processes (a model server at 29-69% SM and a controller at 68-88% SM)
were on the card; the runs taken then are labelled "shared" and are not rates of the code.

- Probe W8K: `qwen35_train_probe --layers 64 --tokens 8192 --rank 16 --seed 7 --bf16-residuals on`,
  random weights, real 248,320 vocabulary, with the segment and MLP rows named in the row.
- Real: `brain --backend cuda qwen35 finetune <dir> --base Qwen3.8-27B-FP8 --mode lora
  --base-dtype nf4|bf16 --seq-len 8192 --batch 1 --seed 7` with every other setting at the
  CLI default (rank 8, segment 2048 / MLP rows 2048 / bf16 residuals on at 8192 tokens). The
  corpus is 100,000 random token ids (timing only: token content does not change step time);
  `tokens/s` is 8192 over the logged step seconds of steps 2-3 (step 1 carries kernel
  compilation). The CLI does not print dispatches or peak memory; peak is the maximum of
  `nvidia-smi memory.used` sampled every 2 s.

## The sweep kernel (`gdn_bwd_sweep_f32`)

The reverse sweep of `gdn_chunk_bwd` carries the state gradient from chunk to chunk and
issued six dispatches per chunk (two `bmm`, the decay-scale gradient, the state decay, two
`bmm_acc`). Both products with the state gradient contract over a different axis of it, so,
unlike the forward recurrence, it cannot be split over value columns. One block per head
(256, then 512 threads) keeps the whole 128 x 128 state gradient in dynamic shared memory
(padded to 129 columns, 82.7 KB; opted in past 48 KiB) for all chunks and runs the four
products and the scalar reduction itself; it needs `CudaLaunch.dynamic_shared_bytes`, which
the provider now passes through `Gpu::native_kernel_launched`. The device offers it
automatically (capability, `dk = dv = 128`, chunk at most 64 rows, the block's shared memory
granted) and otherwise the host walk of knowledge 310 runs unchanged. Numerics: each product
is the ascending-k fp32 sum, but the two state-update products are folded into the state
element in one accumulator, so the result differs from the dispatch sequence by the fp32
summation bound, not bitwise (`gdn_bwd_sweep_native.rs`: f64 reference, 64-row and 5-row
chunks, `d_v_new` and `d_g_cs` accumulated into, only each chunk's last `d_g_cs` row changes).

Probe W8K NF4 (segment 2048, MLP rows 2048), idle, steps 1-2 unprofiled:

| | dispatches | step 2 s | tokens/s | peak MiB | loss step 1 / 2 |
|---|---:|---:|---:|---:|---|
| before (knowledge 310, c305c3ab) | 114,207 | 14.7-14.9 | 550-560 | 51.7 GiB | 13.46 (falls 13.46, 13.40, 13.32) |
| sweep kernel, 256 threads | 77,151 | 13.64 | 600 | 49,973 | 13.4607 / 13.4011 |
| sweep kernel, 512 threads | 77,151 | 13.42 | 611 | 49,973 | 13.4607 / 13.4012 |

Device time per kernel kind (one timed step, idle; before in knowledge 310): `bmm_tiled`
1,671 ms / 29,760 calls to 1,011 ms / 5,184 calls; the sweep kernel 536 ms in 192 calls
(2.8 ms a call; 512 threads against 256 moved it by 0.1 ms, so it is bound by its
staging and synchronisation, not by thread count); sum of kernel durations 12.0 s to 10.7 s
of a 13.4 s step, i.e. the host gap is now about 2.7 s (it was 5 s). The loss is the same to
the fourth digit as before. The W4 bf16 probe, steps 1-3, reads 13.4252 / 13.4198 / 13.4143, the
values recorded in knowledge 310, at 3,640 dispatches instead of 4,798.

## Real checkpoint, idle, CLI defaults (this is the figure to quote)

| base | step 2 / step 3 s | tokens/s | load s | peak (sampled) | commit |
|---|---|---:|---:|---:|---|
| NF4 | 13.2 / 13.2 | **621** | 33.9 | 44.4 GiB | 6edaafde |
| bf16, freed blocks off | 19.4 / 19.5 | **422** | 57.0 | 65.8 GiB | 6edaafde |
| bf16, freed blocks kept | 12.8 / 12.8 | **640** | 59.2 | 79.8 GiB | 38141fe5 |

For comparison on the shared card (two other contexts, same code and settings): NF4 step 2
41.9 s (195 tokens/s) for the new binary and 85.3 s (96) for the binary of c305c3ab. Those two
runs were not simultaneous, so their ratio is not a speed-up.

## The bf16 tier (probe W8K, idle)

| settings | freed blocks | dispatches | s/step | tokens/s | peak MiB |
|---|---|---:|---:|---:|---:|
| segment 512, MLP rows 1024 (earlier CLI default) | off | 214,927 | 35.05 | 234 | 67,335 |
| segment 1024, rows 1024 | off | 129,295 | 25.63 | 320 | 67,335 |
| segment 2048, rows 2048 | off | 74,191 | 19.92 | 411 | 68,763 |
| segment 2048, rows 2048 | **kept** | 74,191 | 13.07 | **627** | 83,381 |

The peak did not depend on the segment (the head and the layer stack set it), so the bf16 base
takes the 2048/2048 settings of the 4-bit base. With the freed blocks off, every released
activation is a driver call that waits for the device (knowledge 310 item 1), worth 7 s of a
20 s step; keeping them costs 14.6 GiB (81.4 GiB peak of the 95), so a bf16 27B step now
needs the card to itself. That is a change of the default, taken because a 27B step at these
sizes has no other use for the room.

## What is not done

1,000 tokens/s is not reached: the best real-checkpoint figure is 621 (NF4), the best probe
probe figures 627 (bf16) and 611 (NF4); the real-checkpoint bf16 step is 640 (the best, at 79.8 GiB). Of the 13.4 s NF4 step the device kinds are: library GEMMs
2.6 s (the card's practical bf16 rate, not tunable), the three GDN recurrence kinds 1.8 s
(`chunk_loop` fwd and train, `bwd_sweep`), `bmm_tiled` 1.0 s, layout permutes and `l2norm`
0.9 s, and the host gap 2.7 s. The ordered levers, unchanged: a fused GDN whole-tensor
backward (the remaining `bmm_tiled` and about 55 whole-tensor dispatches per
segment-layer), the recurrence kernels' occupancy (192 blocks over 132 multiprocessors), per-layer graphs or a
parameter ABI for the 77,151 launches (about 35 us of host time each), Q2 (about 3,000
drains), K4, K7, the shared-input A concatenation of K3 and a fused tensor-core flash
attention. `Frozen::capture_inner` dropped the dynamic shared memory and cluster of the
kernels it captured (the eager path already passed them); fixed in dd75146b, no test reaches
that path with a dynamic-memory kernel yet.
