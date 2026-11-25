<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 588. An int8 KV cache makes the served policy trace the wrong behaviour for the clipped surrogate

**Lesson:** a served reply records the behaviour log-probability its serving engine computed over its own KV cache.
Qwen3's engine keeps that cache in int8 by default (`--kv-cache`); the trainer scores the same tokens over
full-precision activations. At the first step of an online run, where the policy is the one that sampled (version
1, staleness 0, every ratio ideally 1), the int8 cache put some tokens' importance ratios anywhere from 0.0 to 2.8:
the clipped surrogate clipped 8% of the tokens it trained, and their gradients are the quantisation's, not the
policy's. With `--kv-cache f32` the same first step had every ratio between 0.90 and 1.09 and clipped none.
Invariant 1 of the asynchronous roadmap (the recorded behaviour is the distribution that drew the token) therefore
needs a lossless KV cache on the serving side, or a correction the trainer does not have. The online measurements
serve with `--kv-cache f32` (the qwen3 engine offers no bf16 cache; fp8 not measured). Open: whether a stream job
should refuse, or warn about, a policy served over a lossy KV cache (no silent degradation), and what fp8 costs.

## Evidence (2026-11-25, lane agrpo-real, brain 760c04d3b, one GH200)

Qwen3-0.6B base, LoRA rank 16, an orchestrator's online stage over integer-arithmetic tasks, temperature 1.0
untruncated, thinking off. The first step frame's `pg` (version 1, `staleness_mean` 0):

| run | KV cache | policy tokens | ratio mean | ratio min | ratio max | clip fraction |
|---|---|---|---|---|---|---|
| smoke0 | int8 (default) | 143 | 0.991 | 0.032 | 1.600 | 0.105 |
| a34 | int8 (default) | 1265 | 0.998 | 0.053 | 2.772 | 0.078 |
| a34-kv-int8b | int8 (default) | 1139 | 0.993 | 0.000 | 2.442 | 0.080 |
| a34-kv-f32 | f32 | 1069 | 0.999 | 0.897 | 1.090 | 0.000 |

The f32 comparison runs that followed (cmp-S-1, cmp-A-1) clipped 0.000 to 0.001 at every step. Qwen3-4B
(`--kv-cache f32 --weight-format bf16`, the trainer's base in bf16) had ratios of 0.68 to 1.10 (clip 0.006, 168
tokens) and 0.81 to 1.72 (clip 0.008, 243 tokens) at version 1. The rest of the spread comes from the serving
weights (fp32 for 0.6B, bf16 for 4B) against the trainer's bf16 base. The run directories were node-local and were
lost in a container restart the same day; these figures were read from their step frames before that. The run
settings are in `~/project/shared/tmp/agrpo/agrpo-real/cfg`.
