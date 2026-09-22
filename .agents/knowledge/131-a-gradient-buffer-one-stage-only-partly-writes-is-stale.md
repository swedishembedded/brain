<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 131. A gradient buffer one stage only PARTLY writes is stale everywhere else, and it trains

The decision model's head writes the encoder's seed buffer in two pieces: the
key/value path ASSIGNS the state rows, and the `[CLS]` scatter ACCUMULATES
onto one row per option. Every other row of that buffer - the non-`[CLS]`
rows of every option's slot - is written by neither. The encoder's reverse
pass reads the WHOLE buffer, so it was fed a correct gradient plus the
previous step's gradient on those rows.

WHAT IT LOOKED LIKE. Training "worked". The loss moved, no NaN, no crash,
and the model answered every question. It reached 0.231 on an eight-way task
against a chance of 0.125 - above chance, so plainly learning something, and
easy to read as "undertrained, needs more steps". Clearing the buffer took
the SAME 600 steps to **0.962**, with mean confidence going from 0.000 to
0.973. A quarter of the ceiling looked like a small model warming up; it was
the encoder being corroded by noise every step.

WHAT POINTED AT IT. Not the loss curve, which was merely noisy. The tell was
that the trained head RELOADED onto the untouched encoder answered sensibly
while the in-process model - same head weights, plus a fine-tuned encoder -
returned a near-uniform distribution. When a checkpoint outperforms the live
model it was just saved from, the difference is the part that was not saved,
and here the only such part was the encoder. That located the corruption in
the encoder's input gradient rather than anywhere in the head.

THE RULE. When one stage produces a buffer another stage consumes WHOLLY,
the producer owes every element of it, and "assign a region, accumulate onto
some rows" owes the rest a zero. `Gpu::submit`'s `clears` argument is for
exactly this and costs nothing - and it must ride on the SAME submit as the
steps that follow it, or it races them across handles (lesson #130).

A gradcheck cannot see this. It builds one model, sets one batch, and
differences a scalar loss; the buffer is zero on the first use, so the stale
term is zero and the check passes. Only a SECOND step has anything stale to
inherit. A finite-difference check proves an adjoint, not that the tensor it
is handed was fully written.
