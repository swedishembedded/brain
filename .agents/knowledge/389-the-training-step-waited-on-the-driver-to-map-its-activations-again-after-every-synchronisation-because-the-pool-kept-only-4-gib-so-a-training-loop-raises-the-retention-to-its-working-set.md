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
(7.8 s -> 5.67 s a step); bf16 base 1120 -> 1540; peak memory unchanged (40.0 GiB NF4, 72.0 GiB bf16); losses equal to the fourth digit.
nsys of the new step: card busy 95.5 %, host allocation time 2.03 s -> 35 ms, idle 2.6 s -> 0.26 s (sub-20 us launch gaps), gaps over
100 us: 1056 -> 6. The 199 synchronisations are still there but the host builds a layer in less time than the card runs it, so they no
longer cost card time (14 ms of idle adjacent to them).

Rule: a loop whose working set exceeds the default retention must raise it (`Gpu::retain_freed_blocks`); a host that is 85 % "busy"
inside a driver call is not computing, so read the gap's host calls (nsys runtime table) before calling a step host-bound.
