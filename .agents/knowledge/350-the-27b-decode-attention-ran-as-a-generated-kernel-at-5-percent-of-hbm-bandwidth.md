<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 350. The 27B decode attention ran as a generated kernel at 5 percent of HBM bandwidth

Symptom: in the served 27B (Q8_0 GGUF, int8 paged KV), `paged_flash_decode_gqa_hd256_GRP_6_kv_int8` took 122 us per GQA
layer at 12.8k context (26 MB of keys and values, about 215 GB/s): 1.95 ms of a 14 ms token, about 6 ms at 48k. Cutting the
keys into 4x more splits (400 instead of 104 blocks) gained only 4-8 percent.

Cause: on CUDA that kernel was the GENERATED tier (WGSL -> CUDA C++), not a hand-written one: no `native_upgrade` row
named it. Its WGSL design gives one key to one thread (a 16-byte piece of a 256-byte row per load, 32 lanes on 32 rows),
widens every int8 with a convert and merges each tile through five barriers; on the CUDA backend that is bound by
instruction latency, not by memory. A census of b27's nsys windows shows it is the only generated kernel that matters on
that tape: the others (`rmsnorm_rows`, the decode combine, `argmax_part`, `rope2d_partial`, `mul`) are about 3 percent of
a decode token and 4 percent of a prefill round; the GEMVs, GEMMs, GDN and fused glue kernels are all native.

A CUDA-core rewrite (half-warp per key row, 16-byte loads, queries in registers, int8 widened by the exponent trick) was
correct but still latency bound: 212 registers, two blocks per SM, about 1,500 dependent FMAs per key and kv head for the
scores and as many for the values (ncu: 69 us at 12.8k context).

Fix: `paged_flash_decode_gqa_hd256_kv8` (kernels-cuda, redirected to by name from `#GRP=6#kv=int8`) runs both products on
`mma.sync.m16n8k16`: S = Q . K^T with the group padded to 16 rows, then O^T += V^T . P^T, whose B operand is the first
product's accumulator, so P stays in registers. int8 is exact in f16; the f32 query and probabilities are split into an f16
high part and the f16 rounding of the remainder, so the products keep about 22 bits (worst relative error to exact f64 at
48k context 1.1e-5, the gate's bound is 2e-4). Each warp walks its own 16-key chunks with its own running softmax and the
four warps merge once. The order of channels in a dot product and of keys in a chunk is free, so it is chosen to make every
lane's loads contiguous 16-byte reads. The native grid targets two resident blocks per SM (192 registers), and the
partial-record slab is `storage_uninit` (every record is written).

Measured (ncu, card 1 shared with another job, so ranges, split pass only; generated -> native): batch 1 at 12.8k keys
127.9 -> 28.8 us (0.93 TB/s on the KV bytes), batch 1 at 48k 276.6 -> 57.1 us (1.75 TB/s), batch 8 at 12.8k 372.6 ->
131.0 us, batch 8 at 48k 988.1 -> 371.2 us (2.17 TB/s, 55 percent of the 4.0 TB/s spec). Gate:
`native_int8_decode_is_exact_at_agent_contexts_over_a_shared_prefix_on_cuda` (crates/model/tests/kv_tier.rs, CUDA opened
explicitly - the ambient `Gpu::new_gpu` may be another backend, on which the test's nativity assertion would never run);
timing report `int8_decode_attention_timing_report` (ignored).

In the served 27B (nsys decode window, cached 10.5k-token prompt, 300 tokens, card 1 otherwise idle): the split pass
25.0 us and the combine 10.9 us per layer, 0.58 ms of attention per token against 2.06 ms before (b27's window at
12.8k: 122.3 + 7.1 us), 4.6 percent of the token against 14.7; 13.8 ms per token. Server A/B against the previous main
(three runs each on a card shared with other jobs): 38k-context decode 52.9-58.7 -> 61.4-67.6 tok/s, an 8-conversation
recorded-trajectory replay 73.6-77.4 -> 70.3-71.5 s wall; 1k and 10k contexts within the run-to-run noise.

Open: the combine (still generated) is now a third of the attention time; batch 1 at 12.8k is at 23 percent of the spec bandwidth (one chunk's loads are not overlapped with the previous
chunk's products; a cp.async double buffer per warp would). The bf16/fp8 tiers and group 8 (the 35B-A3B) still run the
generated kernel.
