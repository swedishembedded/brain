<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 192. Classifier-free guidance multiplies a quantized decoder's error

Janus-Pro samples each image token from `u + w * (c - u)`. Here `c` and `u`
are the generation head's logits for the conditional and the unconditional
sequence, and `w = 5`. The two branches see nearly the same weights and differ
by a small amount, so the blend is dominated by `w` times a small difference.
Any error in either branch lands in that difference at five times its size,
while the difference itself is small.

Measured on the real Janus-Pro-7B checkpoint, teacher-forced against the fp32
reference over its first four steps, with the serving engine's int8 linears
(group-wise weights, dynamic int8 activations):

| logits | cosine to the reference |
|---|---|
| conditional branch alone | 0.9987 to 0.9994 |
| unconditional branch alone | 0.9960 to 0.9998 |
| guided blend, step 0 | **0.848** |
| guided blend, steps 1-7 | 0.974 to 0.995 |

Each branch looks healthy on its own, and the quantity that is actually
sampled from does not. A gate on the branch logits would pass a decoder that
samples from a visibly different distribution.

The fix was not a tolerance. The serving engine gained a half-precision
weight tier (`Engine::from_map_tier` with `BF16` or `F16`): linears stored at
the checkpoint's own precision and decoded inline in the GEMM. A 7B decoder
then fits a 24 GB card in about 14 GB. At bf16 the guided logits match the
reference at cosine >= 0.9999 on every step.

Rule: gate a guided sampler on the guided quantity, and serve it at the
checkpoint's own precision. The same arithmetic applies to any `cfg_blend`
consumer that runs a quantized model, including the diffusion samplers'
velocity or noise predictions.
