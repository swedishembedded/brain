<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 47. A tap measured downstream of a normalization cannot gate a scale error upstream of it

The int8 tier added to the Gemma-4 text encoder was gated by comparing a real
layer's fp32 and int8 outputs on the real checkpoint, asserting cosine AND
`rel_l2` on the layer output and cosine on the self-attention tap. That gate
was then mutation-verified by halving every per-output-channel weight scale -
the single most likely packing bug in a per-channel int8 tier, and one that
makes the arithmetic wrong by a factor of two.

**The gate passed.** Two independent reasons, and both generalise:

* On the self-attention tap, cosine came back **bit-identical** to the
  unmutated run (0.998157036 either way). That is #2 restated with a real
  number: cosine is scale-invariant, so the one error a per-channel *scale*
  can make is exactly the error cosine cannot see. Only `rel_l2` moved, and it
  moved enormously - 0.061 to 0.503.
* On the layer OUTPUT, *nothing* moved outside tolerance, `rel_l2` included
  (0.067 to 0.082). The attention branch is followed by
  `post_attention_layernorm` and then added to a residual: the norm divides
  the injected factor straight back out, and the residual dominates what
  survives. A metric that can see a scale error still cannot see it if it is
  measured after something that removes scale.

So the gate needed `rel_l2` **on the pre-normalization tap specifically**, and
that assertion is what now fails the mutation. The general rule: when choosing
where to tap a quantized-vs-reference comparison, ask what lies between the
error's entry point and the tap. Any normalization in between (RMSNorm,
LayerNorm, a softmax, an L2 normalize) makes everything downstream blind to a
whole error class, and a residual add makes it blind to magnitude generally.
Tap the sublayer's own output, not the block's.

The corollary is about process rather than numerics: this gate looked complete
and was written by someone who had just read #2. Only deliberately breaking
the thing it guards showed which of its four assertions were load bearing
(one) and which were decoration (three). A gate nobody has seen fail is a
hypothesis - including when the person writing it was actively trying to
avoid this exact defect.
