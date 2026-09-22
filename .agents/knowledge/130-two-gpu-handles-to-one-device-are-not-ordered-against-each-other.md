<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 130. Two `Gpu` handles to one device are not ordered against each other, and the failure is silent zeros

A model built from two halves - an encoder and a head - gave a falling
training loss and a BELOW-CHANCE accuracy: 0.0 correct where chance was
0.125. Both halves were independently proven. The encoder was parity-gated
against its reference to 1.3e-5 and the head's adjoint was
finite-difference checked. Neither could see this, because neither runs the
other.

The head held its own `Gpu` handle, taken with `share()`. `submit` on one
handle is not ordered against `submit` on another, so the head dispatched
against the encoder's hidden-state buffer BEFORE the encoder had written
it, and read zeros. `Gpu::poll_wait` on the producing handle between the
two is the fix; the same hazard exists in reverse on the backward, where
the encoder reads the seed buffer the head writes.

WHY IT SURVIVED. Zeros are a fixed point that looks like arithmetic. Every
downstream stage computed happily on them, the softmax of an all-zero score
vector is a clean uniform distribution, the loss fell (the one free
parameter, the shared bias, still moved), and the model answered every
question. Nothing was NaN and nothing crashed. A cross-handle read that
failed loudly would have been found in a minute.

WHAT FOUND IT, AND WHAT DID NOT. Three hypotheses were wrong before the
right one: an uninformative `[CLS]`, a broken option-to-score mapping, and
cross-handle buffer visibility - that last one killed by a six-line probe
that allocates on one handle and gathers on another, which PASSES. What
actually localized it was a per-stage norm of the head's forward: every
stage that read the encoder's buffer was exactly 0.0 and every stage that
did not was alive. Then the shape of one number closed it - `probs` read
2.828 = sqrt(8), which is the norm of a uniform softmax over exactly the 9
state rows that request had, i.e. a softmax of all-zero scores over the
right geometry. The right geometry with the wrong contents is a
synchronization failure, not a wiring one.

THE GENERAL RULE. A test that asserts a model LEARNS is not the same test as
one that asserts its parts are connected, and the first cannot substitute
for the second: loss fell here for 400 steps against a model that could not
see its own input. What pins this now is cheap and weights-free - different
options must get different scores, and permuting the option list must
permute the scores - and either would have failed on the first run.
