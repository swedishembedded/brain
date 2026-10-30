<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 389. The training step waited on the driver to map its activations again after every synchronisation, because the pool kept only 4 GiB, so a training loop raises the retention to its working set

Question: knowledge 388 left the 27B W8K NF4 LoRA probe at 1037-1051 tok/s with `graph_replays=0` and 6.6 s of a 7.8 s step as host
time. Where does the card wait for the host, and what of that is host work as opposed to the host waiting for the card?

Measured (gpu1, `qwen35_train_probe --layers 64 --tokens 8192 --rank 16 --seed 7 --steps 4 --residuals bf16 --base nf4
--gdn-segment 2048 --mlp-rows 2048`, nsys `-t cuda,nvtx --cuda-graph-trace=node`, warm step 3, card busy = union of kernel, memset and
memcpy intervals, host API time from the runtime table):

1. **Base: card busy 67 %, idle 2.6 s of 7.9 s.** 46,624 launches, 37,822 `cuMemAllocFromPoolAsync` calls costing 2.03 s of host time
   (avg 54 us, 12 calls of 17-30 ms a step, hundreds of 0.5-1.5 ms), and 196 `cuStreamSynchronize` (3 per layer: the MLP row-chunk
   drains in the forward and the backward, the rebuilt-activation drain) that the host spends 4.8 s inside. The idle card time was in
   gaps of 1-30 ms that each began at one of those synchronisations or at a pool allocation: 1.27 s next to a synchronisation, 1.07 s
   in 1056 gaps over 100 us elsewhere, 0.26 s in sub-20 us launch gaps.
2. **Cause: the stream-ordered pool's release threshold is the block-cache cap, `total / 16` clamped to 256 MiB..4 GiB** (serving's
   temporaries are a few hundred MB). The driver trims a pool down to its release threshold at every stream synchronisation. A training
   step cycles about 40 GB of activations and drains 196 times, so after each drain the next layer's allocations had to map physical
   memory again (`cuMemAllocFromPoolAsync` 0.5-30 ms, in front of the next kernel). The host was not computing in the gaps; it was inside
   the allocation call.
3. **Fix: `Gpu::retain_freed_blocks(bytes)`** (`Backend::retain_freed_blocks`, CUDA: cap and pool release threshold raised, clamped to the
   device memory; an allocation that would fail still trims, so retention cannot run the card out). The qwen35 training build calls it
   with `u64::MAX` where it already asks to hold freed blocks. Nothing else changed.

Result (same session, base binary and new binary alternating, 2 runs each, warm steps 2-4): NF4 1051 / 1049 base, 1449 / 1443 new tok/s
(7.8 s -> 5.67 s a step); bf16 base 1120 -> 1540; peak memory unchanged (39,988 MiB NF4, 71,956 MiB bf16); losses equal to the fourth digit.
nsys of the new step: card busy 95.5 %, host allocation time 2.03 s -> 35 ms, idle 2.6 s -> 0.26 s (sub-20 us launch gaps), gaps over
100 us: 1056 -> 6. The 199 synchronisations are still there but the host builds a layer in less time than the card runs it, so they no
longer cost card time (14 ms of idle adjacent to them).

Rule: a loop whose working set exceeds the default retention must raise it (`Gpu::retain_freed_blocks`); a host that is 85 % "busy"
inside a driver call is not computing, so read the gap's host calls (nsys runtime table) before calling a step host-bound.

Three more host-path and glue items found by the same profile (same session, same flags, nsys of the probe after each):

4. **A frozen NF4 weight was decoded to bf16 for every product that read it.** `Ops::for_each_tile` decoded the whole weight into a scratch per
   call: forward, rebuild and input gradient of a layer, and the MLP's four 2048-row chunks each time: 51 decodes of a layer's eight to seven
   weights per layer, `weight_dequant_bf16` 3020 launches, 428 ms (7.8 % of the step). `Ops::decoded_weights()` is a scope (nests; the outermost
   guard releases the copies) in which a whole-weight window is decoded by the first product and shared by the rest; qwen35 opens one per layer
   in the forward and in the backward. 972 launches, 105 ms; +0.1 GiB peak. The contract is the caller's: the weights do not change while the
   scope is open, and the work is submitted in the order it is built.
5. **`mul`, `splice_add` and `splice` ran the generated one-element-a-thread form at 0.7-1 TB/s.** They are the GDN output gate and the MLP
   row-chunk copies. Native forms (four elements a thread, 16-byte accesses, scalar tail and unaligned base handled), bit-identical to the
   references (`elementwise_native`): 192 -> 53 ms, 174 -> 46 ms.
6. **Every submission of a layer paid its own parameter copy (14.5 k HtoD copies a step).** A layer's forward and backward in `pass_scope`
   (the prefill seam of knowledge 381) issue with one upload: copies 14.5 k -> 0.65 k, `submit_s` 0.40 -> 0.09, +1.6 GiB peak.

Cumulative (probe, gpu1, same session): 1051 -> 1449 -> 1517 -> 1559 -> 1597 tok/s; step 7.8 -> 5.19 s; nsys of the last: card busy 96.1 %,
idle 0.21 s (98 ms in sub-5 us launch gaps, 83 ms in 271 gaps of 0.1-1 ms where the host builds the next layer), kernel time 4.8 s.

What is left, by measured share of the 5.19 s step (the host is no longer the bound; the card is):
- Library bf16 GEMMs 2.2 s (42 %), about 600 TFLOP/s each; an MLP chunk at 2048 or 8192 rows gives the same rate (`--mlp-rows 8192`
  1559 = 1559 tok/s, which also drops the row-chunk copies, so those are not the bound). Only a narrower precision moves them.
- The LoRA products are 0.6 s (12 %) of cutlass TF32 kernels (`Kernel2`); the ff-wide ones (`mid[M,16] @ B[ff,16]^T` accumulated into gate
  and up, 100 us each) are at the memory roofline of reading and writing the [2048, 17408] fp32 output, so only fusing them into the
  consumer (`silu_mul`) helps, about 1.5 %.
- Gated DeltaNet backward and forward-train 0.74 s (15 %): `gdn_chunk_bwd_post_tc` 290 ms, `gdn_bwd_sweep_tc` 218, `gdn_chunk_loop_train_tc` 151,
  `gdn_chunk_prep_tok_train_tc` 82 (knowledge 384: staging, barriers and shared memory, not the MMA).
- Training attention is the blockwise library path (`brain_attn_softmax_fwd` 100 ms, `brain_attn_dscore` 58, TF32 score GEMMs ~160 ms), not
  the flash trio: a fused tensor-core forward and backward at head_dim 256 would save about 250 ms (5 %).
- The whole step as one recorded graph: the remaining launch-gap loss is 0.2 s (4 %), but the replay needs the arena with lifetime-planned
  offsets (the step cycles 1.5 TB of requested buffers; a tape holds every buffer it binds) and graph-safe library GEMMs (knowledge 356: replay
  loses them), so it is serving-runtime.md B/C and item 5, not a training-only change.
