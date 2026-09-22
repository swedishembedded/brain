<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 129. An exact algebraic identity is not an implementation, and random weights cannot tell you which one you have

A GDN prefill optimization replaced a triangular solve with a mathematically
EXACT reformulation: `attn0` is nilpotent, so `(I - attn0)^-1 =
sum_k attn0^k = prod_m (I + attn0^(2^m))`, computable in 17 batched-GEMM
dispatches instead of 64 tiny ones. The identity is correct. Every gate in
the suite stayed green. On the real checkpoint, prompts past ~2 chunked-
prefill rounds came out as `"-regexp-regexp-regexp..."`.

Exactness in exact arithmetic says nothing about conditioning. Here the
answer is `O(1)` and the intermediates the reformulation must pass through
reach `1e17` - the sum telescopes, but only if you can still see the low
digits, and fp32 has seven. Nothing about the transformation is wrong; what
changed is how much cancellation stands between the inputs and the output.
Before trading a solve for a series, ask what the intermediates are worth
relative to the result, not only whether the terms sum to it.

The second half is why the suite could not see it. `attn0[i,j] =
-beta_i*(k_i.k_j)*decay`, and its magnitude is set by how CORRELATED the
keys are. Independent random keys in `Dk` dimensions are near-orthogonal
(`k_i.k_j ~ 1/sqrt(Dk)`), so a random-weight test leaves `|attn0| << 1`,
where every way of forming the inverse agrees to round-off. Real trained
weights on repetitive text drive `k_i.k_j` towards 1. Random inputs sample a
numerically BENIGN corner of the input space for anything whose stability
depends on the correlation structure of its inputs - so "passes at random
weights" is not evidence of numerical correctness, and a synthetic gate for
this class of defect has to be built to be adversarial on purpose. The gate
that now pins it is not expensive or checkpoint-bound: near-parallel keys at
the production chunk width, 0.3 s, no weights at all.

Two smaller things that cost time here. The existing oracle comparison
asserted on the FIRST element over tolerance, which reported `2.7e-3` for a
failure whose worst element was ten orders of magnitude larger - a
conditioning failure is spread unevenly, so compare on the worst element, not
the first. And the localization pointed at "one layer, one round boundary",
which is where the corruption became VISIBLE (a bad chunk poisons the
recurrent state, which the next round then multiplies into every token), not
where it was produced. When a defect compounds through carried state, the
layer and round the diagnostic names are the amplifier, not the source.
