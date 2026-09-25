<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 158. A conditioning scale that can run away costs more than the capability it buys

`samples/learning/visualclick` needed one thing its architecture could not
express. A decision head scores each on-screen control from a query row built
as `Wopt . control + b(instruction)`. That is ADDITIVE, so the instruction
contributes the same vector to every option and the coefficient on any feature
of the control - its position, say - is identical whatever was asked.
`leftmost` and `rightmost` need that coefficient's SIGN to differ. Adding a
shared constant cannot produce that, at any width, with any amount of training.

The standard fix is a multiplicative (FiLM-style) scale: `s(instruction) * Wopt
. control + b(instruction)`. The capability argument is correct and the
implementation was correct - it gradient-checked against finite differences on
every parameter. Written the obvious way, as `s = 1 + Wscale . cls`, it still
made the model **worse than the layer it replaced**.

Three arms, same seed, same 8000 steps, same 500 held-out screens, chance 7.6%:

| slot-row conditioning | overall | attribute | spatial | ECE | click error |
|---|---|---|---|---|---|
| additive `Wopt . c + b` | 84.4% | 97.8% | 63.0% | 0.043 | 22.2 px |
| unbounded `(1 + g) * Wopt . c + b` | **74.2%** | 83.3% | 58.0% | 0.108 | 37.7 px |
| bounded `(1 + 2 tanh g) * Wopt . c + b` | **85.0%** | 96.7% | 65.2% | 0.046 | 21.8 px |

`Wscale` was zero-initialised in BOTH multiplicative arms, so at step 0 the
scale is exactly 1 and all three arms are the same function. Nothing about the
task, the data, the seed or the initialisation differs. The only variable is
whether the scale is free to grow.

## Why the unbounded one loses

The scale sits between two things being trained at once. Every step it moves,
the query rows it produces are rescaled, and the head downstream is re-fitting
a target that shifted underneath it for reasons that have nothing to do with
the example. The damage shows up where the model was ALREADY good - attribute
instructions fell 97.8% -> 83.3% - because those had a converged solution to
lose. The capability the scale was added for barely moved.

The loss curve said so before the eval did: single-example losses spiking to
**5.69** and **5.01** against a chance of 2.64, mid-run, after the loss had
already reached 1e-4. A training loss that goes far ABOVE chance after
converging is not a hard example. It is the model being rescaled out from under
itself.

## The rule

**Bound anything that multiplies a representation, and start it at the
identity.** `1 + 2 tanh(x)` gives `(-1, 3)`: wide enough to INVERT a channel,
which was the entire reason for adding it, and incapable of running away. At
`x = 0` it is exactly 1, so training starts from the additive layer it replaces
and can only deviate by earning it. That single change recovered all 10.8
points and then some.

The same reasoning applies to any learned quantity that multiplies rather than
adds - attention temperature, a gate, a residual scale, a loss weight. An
additive term perturbs; a multiplicative one rescales everything downstream of
it, so its dynamic range is a design decision and not a detail to leave to the
optimizer.

## What the bound actually bought

The bounded scale is worth its place because it is free (it cannot be worse
than the layer it replaces) and because the capability is genuinely
unrepresentable without it. It is not what MADE spatial reference work:
spatial went 63.0% -> 65.2%, which is a mechanism becoming representable, not
an optimizer exploiting it. A +0.6 overall improvement is the conditioning
ceasing to be harmful, not the conditioning working, and reading it as the
latter would have stopped the search at 85% with the real cause untouched.

**The real cause was two layers away, and it was found by measuring rather
than by trying more conditioning.** Two further changes each looked as
principled as this one and each made things WORSE or did nothing - whitening
the instruction embedding (85.0% -> 83.6%) and feeding the head the
instruction's token rows instead of a pooled vector (85.6% -> 82.0%). What
finally moved it was a measurement of the frozen encoder itself: two
instructions differing only in a short function word sat at **cosine 0.9837**,
against 0.8510 for two differing in a colour. Nothing downstream recovers a
distinction the encoder has already collapsed, so the fix was to stop asking
it to - name the thing with a content word, and give it a visible correlate in
the pixels. That took the same model from 87.2% to **96.0%** with no
architectural change at all.

The general lesson is the ordering: **when a conditioning change does not buy
what it should, measure whether the signal it is conditioning on survives the
encoder before designing a bigger conditioner.** Every architectural attempt
above was downstream of an input distinction that was not there to begin with.
`samples/learning/visualclick --validate` now checks exactly that and fails
loudly if the words the answer turns on are not separable.
