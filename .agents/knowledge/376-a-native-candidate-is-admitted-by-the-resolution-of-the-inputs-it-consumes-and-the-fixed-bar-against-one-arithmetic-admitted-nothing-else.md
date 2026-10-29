<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 376. A native candidate is admitted by the resolution of the inputs it consumes, and a fixed bar against one arithmetic admits nothing else

Question: `gpu_core::native_tune` checked every candidate of a kernel slot against an f64 oracle of the INT8 semantics (int8
codes of the activations times int8 weight codes, scaled per 32) with a fixed bar of `2e-5 * rms`. A bf16 tensor-core
candidate consumes the activations as fp32 rounded to bf16 and the weights as `code * scale` rounded to bf16: its error
against that oracle is the bf16 rounding of its inputs, about 2^-9 of every term (2e-3), a hundred times the bar. What rule
admits a candidate of a different arithmetic without admitting a wrong one?

Mechanism (`native_tune::admit`, `native_upgrade::Input`): every candidate row declares the form of its inputs - the
activations as packed int8 with a scale per 32 (`ActFormat::Int8Group32`) or as bf16 rows (`ActFormat::Bf16`), the weights
as stored (codes and scales consumed separately) or as `code * scale` rounded to bf16 (`WeightRead`). The tuner builds the
activations as fp32 "as produced" and every form from them (the int8 group form as `max_abs_row` and `quant_pack` do, the
bf16 form as `cast_f32_bf16` does), and for every sampled output computes in f64

- the reference: fp32 activations against the weights as stored (`code * scale`, exact in f64);
- the resolution error: the same sum over the inputs rounded to the candidate's declared forms.

A candidate is admitted when its measured error against the reference stays within alpha = 8 times that resolution error,
both as a root mean square and as a maximum over the sampled outputs (a wrong tile edge passes an rms bound), and never
under a floor of the fp32 accumulation error (2^-24 of the sum of the absolute terms), so a candidate that rounds nothing is
held to an achievable bound. The bound is derived from the declared formats and from nothing else: a new candidate states its
formats and needs no tolerance of its own. A kernel that is wrong is orders of magnitude over the line; one that only rounds
its inputs sits on it (ratio 1), and the 8 leaves room for the arithmetic around the rounding (the bf16 kernels round the
product of code and scale once more than the f64 emulation).

What it admits, GH200, Qwen3.8-27B grid (120 classes plus the ragged shapes of `i8_dequant_native.rs`): every int8 candidate
that was admitted before, and every dequantize-in-register candidate (#377); nothing was refused. The measured error over the
bound's denominator (candidate error / resolution error, 736 candidate runs): the int8 candidates 1.00 (their output is the f64
emulation of their inputs to float noise), the bf16 ones 0.99-1.83 in rms (median 1.19) and 0.75-2.33 in the maximum (median 1.27)
against a limit of 8. The bf16 weight is `bf16(code * bf16(scale))`, two roundings where the f64 emulation rounds once, which is
where the excess over 1 comes from.

What it does not say: that a model keeps its quality. The rule bounds the LOCAL error of one linear; a model-level gate stays
separate. `qwen35_gguf_logit_check` (real prompts, the released 27B Q8_0) compares logits across kernel sets: rel-L2, KL and
the argmax flips with their margins, against the tape's own numerics floor (decode tape against chunk tape, same kernels).
Results on the released 27B Q8_0 (new build against the old, same prompts and device):

- decode batches of 48 and 72 streams against the same streams alone (rows 33-64 and 65-96 on the new candidates): mean rel-L2
  9.89e-3 -> 9.70e-3 and 9.47e-3 -> 9.23e-3, both 1.0x the tape's own floor (9.8e-3, 9.5e-3); mean KL 1.6e-4 -> 1.8e-4 and 1.55e-4 ->
  1.44e-4; argmax flips 2 -> 5 and 2 -> 4 of 384 and 432 steps, every one at a margin of 0.08 or less.
- chunked prefill of four real 3000-token prompts and 24 greedy steps (the Hopper dequantize GEMM): on the contexts that agree,
  mean rel-L2 1.95e-2 against a floor of 1.09e-2, mean KL 8.4e-4, 4 argmax flips in 93 rows at margins 0.005-0.12; two of the four
  greedy streams part at a margin of 0.02 and 0.06. This distance is the int8 activation quantisation of the OLD path as much as
  the bf16 rounding of the new one, so it is read beside the next line.
- perplexity of 9283 tokens of real source text (`qwen35_gguf_score`, eight 4800-character records): 7.7415 -> 7.7486 (+0.09 %),
  per-token logprob delta -0.0009 +- 0.0005 nats, top-1 against the recorded token 54.66 % -> 54.76 %, argmax identical to the old
  path in 98.8 % of positions.

Rule: a tolerance that is written next to a kernel is a statement about one arithmetic. State the formats a candidate consumes
and let the line follow from them; keep the model-level check as its own gate, run on the real checkpoint.

Traps: `Tuned::choice` answers per caller: a dispatch that can hand over only packed int8 activations (a redirect of the WGSL
pipeline) must not be given a candidate that reads bf16 rows, so the fastest candidate that reads the supplied form is chosen
among the class's measured results, not the class's overall winner. The persisted table's key includes ALPHA, so a changed rule
retunes.
