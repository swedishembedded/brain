<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 485. A decode step's small launches cost their length each, so two tensors of one layer go in one launch

**Lesson:** in a recorded decode step of a few dozen rows, a glue kernel is a few microseconds no matter how little it
moves: the launch, the ramp and the tail are the kernel. Two launches over two tensors of the same layer (the query's
and the key's QK norm, the K and the V append) cost two of those; one launch over both costs about one. Look for the
same op run back to back on sibling tensors before tuning either.

## What happened

After the bf16 norms (ledger, 2026-11-16) a Qwen3-4B decode step at c=64 (bf16 weights, int8 KV, GH200, nsys graph-node
trace) spent 1.11 ms in glue, 72 launches a step each of `qknorm_rope_rows` (3.6 us) and `kv_append_i8_clipped`
(3.6 us). The query (32 heads) and key (8 heads) norms read the same positions; the K and V appends write the same
slots.

## What to do

- `qknorm_rope_qk` (`library::qk_norm_rope_pair`) runs a layer's query and key heads in one launch, the key's groups of
  four heads after the query's for each token; `kv_append_i8_clipped_kv` (`provider::cuda::kv_append_i8_kv_step`) runs
  the K and V appends in one, the first half of its warps on K. Both are bit-identical to the two launches they
  replace (gates `qknorm_rope_native`, `kv_append_i8_pair_native`); the qwen3 engine's int8 KV tier takes them.
- `engine_glue` holds a bf16 tape to one QK-norm launch and no portable append per layer.

## Evidence

Same workload, c=64, decode-phase device time per step: QK norm 256 -> 151 us, KV append 259 -> 122 us (c=16:
237 -> 125 us and 242 -> 103 us). Prefill per 8K-row prompt: 13.0 -> 12.9 ms and 2.2 -> 2.0 ms.
