<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 129. An exact algebraic identity is not an implementation, and random weights cannot tell you which one you have

**Lesson:** Exactness in exact arithmetic says nothing about conditioning, and random inputs sample
a numerically benign corner of the input space. "Passes at random weights" is not evidence of
numerical correctness for anything whose stability depends on the correlation structure of its
inputs.

## What happened
A GDN prefill optimization replaced a triangular solve with a mathematically EXACT reformulation.
`attn0` is nilpotent, so `(I - attn0)^-1 = sum_k attn0^k = prod_m (I + attn0^(2^m))`, computable in
17 batched-GEMM dispatches instead of 64 tiny ones. The identity is correct.

Every gate in the suite stayed green. On the real checkpoint, prompts past ~2 chunked-prefill rounds
came out as `"-regexp-regexp-regexp..."`.

Two smaller things cost time here:

- The existing oracle comparison asserted on the FIRST element over tolerance. It reported `2.7e-3`
  for a failure whose worst element was ten orders of magnitude larger.
- The localization pointed at "one layer, one round boundary". That is where the corruption became
  VISIBLE: a bad chunk poisons the recurrent state, which the next round then multiplies into every
  token. It is not where the corruption was produced.

## Why
Nothing about the transformation is wrong. What changed is how much cancellation stands between the
inputs and the output.

The answer is `O(1)` and the intermediates the reformulation must pass through reach `1e17`. The sum
telescopes, but only if you can still see the low digits, and fp32 has seven.

The suite could not see it for a second reason. `attn0[i,j] = -beta_i*(k_i.k_j)*decay`, and its
magnitude is set by how CORRELATED the keys are.

- Independent random keys in `Dk` dimensions are near-orthogonal (`k_i.k_j ~ 1/sqrt(Dk)`).
- A random-weight test therefore leaves `|attn0| << 1`, where every way of forming the inverse
  agrees to round-off.
- Real trained weights on repetitive text drive `k_i.k_j` towards 1.

## What to do
- Before trading a solve for a series, ask what the intermediates are worth relative to the result,
  not only whether the terms sum to it.
- Build a synthetic gate for this class of defect to be adversarial on purpose.
- Compare on the worst element, not the first, because a conditioning failure is spread unevenly.
- When a defect compounds through carried state, treat the layer and round the diagnostic names as
  the amplifier, not the source.

## Evidence
The gate that now pins it is not expensive or checkpoint-bound: near-parallel keys at the production
chunk width, 0.3 s, no weights at all.
