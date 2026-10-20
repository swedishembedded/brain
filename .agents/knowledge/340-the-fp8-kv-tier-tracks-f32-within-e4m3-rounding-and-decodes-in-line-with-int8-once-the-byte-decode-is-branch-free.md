<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 340. The fp8 KV tier tracks f32 within E4M3 rounding and decodes in line with int8 once the byte decode is branch-free

Question: what does an FP8 E4M3 KV cache (one byte per element, one f32 scale per (token, kv-head) row, scale = absmax
/ 448, the int8 tier's footprint) cost in accuracy and in decode time against f32, bf16 and int8?

Accuracy (CPU JIT and GH200, random mixed-magnitude rows, `crates/model/tests/kv_tier.rs`, `qwen35/tests/kv_tier.rs`):
- every finite E4M3 value round-trips exactly through append and the scores kernel (254 codes, scale 1);
- attention output (scores, softmax, apply) against the original f32 rows, relative L2: bf16 3.1e-3, int8 1.4e-2,
  fp8 3.4e-2 (one sequence of 120 keys, 4 heads, head_dim 64);
- qwen35 logits against the f32 cache on the tiny hybrid: bf16 3.4e-6, int8 9.2e-6, fp8 3.7e-5 (prefill) and 7.4e-5
  (decode top-k); at the real attention dims (24/4 heads, head_dim 256, 4 layers): bf16 2.1e-4, int8 8.6e-4, fp8
  3.3e-3 prefill and 2.0e-3 decode. fp8 is about four times int8's error and below the 27B's own numerics floor
  (knowledge 328: 0.09 between its own tapes).

Cost (`qwen_bench serve 16 12 2048 i8w kv8|kvfp8 --model 8b`, one card shared with two 27B servers, so per-call
times vary 2.5x between runs): the first fp8 decode kernel branched on the exponent and ran 0.64 ms per call against
int8's 0.15 (runs/qkv/bench8b-kvfp8-r16-c2048.log, bench8b-kv8-r16-c2048.log). The branch-free decode (rebias the
exponent and mantissa bits into an f32, a select for the subnormal range, the sign ORed in) measured 0.44 against
int8's 0.37 in an interleaved pair (bench8b-*-b.log); the whole 16-row step was 17.1 ms (fp8) against 21.6 ms (int8)
there, which is the noise of the shared card, not a gain: fp8 and int8 are indistinguishable at the step level.

Not done: no tensor-core prefill kernel reads fp8 rows (the portable fused prefill runs), and the FP4/INT4 tiers are
not started (the evidence above says fp8 is usable, int4 needs its own accuracy campaign).
