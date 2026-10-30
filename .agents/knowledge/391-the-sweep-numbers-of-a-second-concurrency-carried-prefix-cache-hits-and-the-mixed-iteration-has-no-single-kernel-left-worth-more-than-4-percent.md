<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 391. The sweep numbers of a second concurrency carried prefix-cache hits, and the mixed iteration has no single kernel left worth more than 4 %

Question: after knowledge 385-390 c=64 serves 851 tok/s against 1241 for a reference server, and c=16 519-581 against 759. Which part of the mixed
iteration (about 768 rows, 61 decode + ~700 prompt, ~93 ms) is worth taking, and is the comparison fair?

Measured (gpu1 and gpu0 in the same session, one binary per card, FP8 27B, int8 KV, `vllm bench serve` random 1024 in / 128 out, `--ignore-eos`):

1. **A sweep that runs c=16 and then c=64 with the same `--seed` replays the c=16 prompts at c=64.** The server's prefix cache holds their KV, so a quarter of the
   c=64 requests skip their prefill. The same binary served c=64 at 851-882 tok/s, ITL p50 24.5 ms in that sweep and at **735-745 tok/s, ITL p50 90 ms, TTFT p50 0.5 s
   with a seed per concurrency** (a fresh server or a seed of its own). Every earlier c=64 figure of this ledger that came out of a shared-seed sweep (851, 857,
   882) is a cache-assisted figure; the clean figure of the same code is ~740. The reference server's 1241 was taken the same way, so the real gap at c=64 is not known
   until it is re-run with distinct seeds (its start failed in this session: the engine core refused to initialise).
2. **Real iterations equal the profiled ones.** A trace of every mixed iteration (rows, duration; temporary, not kept) on an un-profiled server: 310 iterations, 61 decode
   + 715 prompt rows, 92-94 ms mean, 130 us a row; the nsys iteration is 90-94 ms. Throughput is `128 / (1024 x cost per prompt row)` per request: 735 tok/s is 130
   us a row, and 1241 would be ~103.
3. **The iteration (nsys, c=64, median of 300, ms):** FP8 GEMM 48.4 (400 launches, ~864 TFLOPS at 768 rows: at the isolated kernel's 890, so the GEMM is not
   the lever without a new GEMM), `gdn_recur_pool` 6.5 (the 61 decode rows' state: 3.1 MB a layer read and written at ~3 TB/s, at the memory roof for an f32
   state), `quant_epilogue_cluster` 6.6, GDN prep + chunk loop + conv 8.5 (0.226 us a row against 0.146 for one 8K prompt, 12 chunks of 64 a layer in a handful of
   segments), `quant_e4m3_128` 2.8, `add_rms_quant` 2.4, flash prefill/decode + combine 4.8, `concat_split` + `splice` + pool gather/scatter 3.3, nvjet bf16 gate GEMMs 1.5.
   Decode-only step (24.5 ms): swap GEMMs 15.2 (weights 27 GB: 6.7 ms at 4 TB/s, so 2.2x the roof), `gdn_recur_pool` 6.6.
4. **Host time that nsys shows is partly nsys.** Under nsys the first 4 ms of an iteration held a 2.3 ms hole in front of the second kernel: eight
   `cuMemAllocFromPoolAsync` calls of ~145 us each (the pool trimmed to its 256 MiB retention at each of the iteration's 15 stream synchronisations, knowledge 389's
   mechanism) plus 0.5 ms of M-RoPE table construction (`powf` and `sincosf` for 768 x 32 channels). Retaining the working set (`retain_freed_blocks(u64::MAX)`) removed the
   hole in the trace (mixed span 93.8 -> 90.0 ms) and a position-row cache removed the table time, but un-profiled, base against retained, side by side
   on two cards, the iteration was 92.6 -> 92.2 ms and the served rate 743 -> 752 tok/s (+1 %, inside the run-to-run band). Not landed.
5. **Landed: `silu_mul_quant_e4m3_128`** (`cu/silu_mul_quant_e4m3_128.cu`): the MLP product and its E4M3 rows in one warp-per-128-block pass (no shared memory, 256-thread
   blocks), used for the FP8 `down` projection when it reads E4M3 at that row count. It is `quant_epilogue` + `quant_e4m3_128` with the product not read back: 82 + ~22
   us -> 48 us a layer (nsys: 3.1 ms for 64 launches against 6.6 + 1.5 ms), bit-identical to the pair (`silu_mul_quant_e4m3_native`, 12 shapes). Served c=16 / 64 with a seed each,
   two runs each, base on gpu0 and new on gpu1 at the same time: 519.9 / 523.5 -> 521.2 / 521.1 and 739.7 / 732.4 -> 745.0 / 735.0 tok/s (+0.5 %): the kernel time saved
   (-3.6 ms of 92) is largely absorbed by a GEMM that ran 2 ms slower in the same profile (power/clock variation of ~2 ms between nsys runs is the noise floor of this comparison).

What stays open, in the order of size (each is 1.5-4 % of the iteration, none reaches the 23 % a match with 103 us a row needs): a GEMM that reaches 1.1+ PFLOPS at 768 rows (needs
N = 192-256 per wgmma on one accumulator set, knowledge 387); 61 decode rows' GDN state at bf16 (halves 6.5 ms, changes numerics); the 96 pool scatters + 48 gathers around the
slices (1.4 ms; a `fresh` segment needs no gather); the epilogue for `add_rms_quant` (0.7 ms against 0.77 ms saved, knowledge 385); asynchronous scheduling so the 1.6 ms between
iterations and ~3 ms of pre-forward host work overlap the previous iteration.
