<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 428. The 8K x 16 throughput gap to vLLM was host stalls on the serving thread, and a pageable snapshot copy was the largest

Question (2026-11-02): brain served the 27B FP8 model at 8K x 16 below vLLM at matched settings (bf16 KV, f32 state) although a decode step was within 8 % of it. Where did the time go?

## Finding

- nsys over one 8K x 16 cell: the GPU was busy 25.3 s for brain against 25.2 s for vLLM, but brain's GPU sat idle in gaps longer than 1 ms for 3.66 s of a 29.5 s cell; vLLM's gaps summed to 0.01 s. The gap was the host, not a kernel.
- CPU sampling during the gaps (nsys `--sample=process-tree --sampling-period=281250 --backtrace=dwarf` on the `profiling` cargo profile; nsys did not resolve the PIE symbols, so the load base was recovered from return addresses and the frames resolved with addr2line):
  - `finish_slice` -> `snapshot_prefill_state` -> `copy_between` held 42 % of the samples. The prefix-cache snapshot copied device state into PAGEABLE host memory, which the driver copies synchronously: ~16 ms of a stopped serving thread per prompt.
  - `qwen3vl::mrope::mrope_tables_scaled` held ~11 %: it evaluated `theta.powf(-2d/head_dim)` for every row and channel each round.
  - The head top-k read (~13 %) and unified-pool VMM calls held most of the rest.
- Fixes: `AllocPolicy::Pinned` (system memory page-locked with `cuMemHostRegister`, so the copy runs on the copy engine) used by prefix snapshots where the backend supports it. M-RoPE frequencies are computed once per table and each angle goes through `sin_cos`; the output is bit-identical and the 8192-row table takes 1.3 ms instead of 3.5 ms.
- One nsys run with Pinned: gaps longer than 1 ms fell from 3.66 s to 1.43 s, and 278 -> 302 tok/s. A served A/B over two rounds (averages): 8K x 16 269.7 -> 290.6 tok/s, 16K x 16 150.3 -> 164.8, 8K x 64 331.4 -> 351.7; TTFT medians lower in every cell. The 8K x 1 cell was unchanged.

## Rules

- When GPU busy time matches the reference but wall time does not, look for idle gaps on the GPU timeline and sample the host inside them before tuning kernels.
- A device-to-host copy into pageable memory is a synchronous stop of the calling thread. Host buffers that a serving loop copies into belong in page-locked memory.
- Serving latency measurements need a quiet CPU: confine benchmarks and other CPU-heavy work to disjoint sockets (`taskset`). A concurrent 160-core job moved 16-stream median TTFT by up to 50 %.
