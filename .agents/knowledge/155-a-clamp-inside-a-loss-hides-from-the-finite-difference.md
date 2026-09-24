<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 155. A clamp inside a loss hides from the finite-difference check that would catch it

`rlcd::scoring::decision_loss_soft` floored every probability before it could
enter a log or a reciprocal:

```rust
let pc: Vec<f64> = p.iter().map(|&pi| (pi as f64).max(1e-12)).collect();
```

This is the ordinary defensive move against `ln(0)`, and it was wrong in the
one place it mattered. The focal/CE slope is `-1/p`, and the gradient
chains it through the softmax jacobian as `p_i * slope_i`. Where `p` is
clamped, that product is no longer `-1`: it is `-p/1e-12`, which goes to
zero as the real `p` shrinks. Cross-entropy's gradient is supposed to be
`p - target` at every logit gap:

| logit gap | real `p[0]` | gradient as written | `p - target` |
|---|---|---|---|
| 10 | 4.5e-5 | -0.800 | -0.800 |
| 27 | 1.9e-12 | -0.800 | -0.800 |
| 30 | 9.4e-14 | **-0.075** | -0.800 |
| 100 | 3.7e-44 | **-3.0e-32** | -0.800 |

So the correction vanished for exactly the predictions that needed the
largest one: a model that has become confidently WRONG gets no gradient to
climb back out with. The loss saturates at `-ln(1e-12)` at the same time, so
a run shows a plateau and no reason for it.

**The finite-difference gate could not see this, and would not have.** The
repo's own `the_gradient_matches_finite_differences` perturbs the scores and
compares. But the clamp flattens the LOSS by exactly the amount it flattens
the analytic gradient, so in the clamped region the numeric derivative is
also ~0. Analytic and numeric agree perfectly, and both are wrong. The check
did not fail to run and did not get the tolerance wrong - it was structurally
incapable of detecting a defect in the quantity it differentiates.

THE RULE. **A finite-difference check is only independent where the loss is
smooth. Any clamp, floor, `max`, or saturating branch inside a loss is a
region the FD gate has been blinded in, and needs a CLOSED FORM to check
against instead.** Here that form is exact and was already known:
cross-entropy's gradient is `p - target`, at every gap, forever.

THE FIX, and why it is not another clamp. Carry `ln p` alongside `p` from one
max-shifted pass (`p` underflows to zero around a gap of 745; `ln p` is
merely `-745` and still exact), and form `p * slope` as a single quantity
rather than dividing and re-multiplying:

```text
p * d/dp[(1-p)^g * -ln p]  =  g*(1-p)^(g-1) * p*ln p  -  (1-p)^g
```

At `g = 0` that is exactly `-1` for every `p`, with no division performed and
nothing to floor. The `1/p` was never needed - it was formed only to be
cancelled one line later - and the clamp existed to protect a division that
should not have been there. Taking the limit analytically also makes
`gamma < 1` total, which the clamped form left as an `inf * 0`.

Related: [[145-a-held-out-split-drawn-like-the-training-set-cannot-see-a]] and
[[153-a-value-that-fits-its-own-trajectory-cannot-rank-the]] are gates that
cannot see a defect because of the DISTRIBUTION they measure on. This one is
narrower and nastier: the gate measures on the right input and is defeated by
the arithmetic of the thing it measures.
