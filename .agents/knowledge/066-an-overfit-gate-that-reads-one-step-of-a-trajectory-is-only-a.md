<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 66. An overfit gate that reads ONE step of a trajectory is only a gate if the trajectory has converged - above the optimizer's stability threshold it reads the limit cycle's phase

`crates/supir/src/finetune.rs::full_backbone_overfits_a_single_sample`
asserted `last < l0 * 0.35` after 120 Adam steps at `lr = 0.03` and failed
deterministically at `7.06e-1 -> 4.68e-1`. Its sibling
`adaptor_only_overfits_a_single_sample`, same helper, same lr, same step
count, only a different freeze predicate, passed. That asymmetry reads like a
backward bug in the parameters only full-backbone trains, and it is not one.

**What the trajectory actually did.** The gate printed every twentieth step,
which showed a clean monotone descent (`3.70e-1, 3.12e-1, 2.80e-1, 2.52e-1,
2.38e-1` at steps 20..100) and hid everything. Printing every step:

| step | 103 | 104 | 105 | 115 | 116 | 117 | 119 | 121 |
|---|---|---|---|---|---|---|---|---|
| loss | 2.29e-1 | 2.40e-1 | 3.88e-1 | 2.90e-1 | **9.85e-1** | 4.83e-1 | 4.68e-1 | 5.76e-1 |

The run is a limit cycle: a slow descent punctuated by excursions that reach
**1.40x `l0`** - above where training started - and step 119 is simply one of
them. Steps 1..5 already read `2.73e0, 1.24e0, 5.64e-1, 2.78e0` against an
`l0` of `7.06e-1`, so it was never in a stable regime at all. The passing
sibling is in the same cycle with smaller excursions (peak 0.44-0.54x `l0`)
and happened to be sampled on a descending step.

**Two things this cost, both worth stating separately.**

* **The backward was ruled out the expensive way, and the ruling-out is a
  reusable gap.** `gradcheck::check_supir` deliberately checks only
  `control_model.*` and `project_modules.*`, on the stated grounds that
  `check_unet` already covers the backbone's own adjoints. True - but it
  means the exact parameter set that distinguishes the two modes is the one
  set no SUPIR gate touches, so "is full-backbone's gradient wrong?" had no
  cheap answer. A directional FD sweep over all **263** unprefixed tensors
  found nothing above the fp32 FD noise floor (worst `rel = 2.36e-1`, on
  `up_blocks.0.resnets.0.conv1.bias` at `|g| = 4.5e-3` where a central
  difference at `eps = 2.5e-4` against a loss of 0.7 is pure rounding), and
  the worst offenders were `up_blocks.*` tensors the PASSING mode trains too.
* **`lr` is not "how fast", it is "how big a step relative to the weight".**
  `model::lora::adam`'s update is scale-free: each entry moves by about `lr`
  per step whatever its gradient's magnitude. `UNetConfig::tiny`'s convs
  initialise at `1/sqrt(fan_in)` ~ 0.06, so `lr = 0.03` moves every weight by
  half its own distribution's width per step. The stability threshold that
  crosses is a property of WHICH parameters are unfrozen, not how many:
  unfreezing the backbone encoder moves the input every later stage is
  conditioned on, so full-backbone leaves the stable regime first *because*
  it has more capacity, not despite it. "Strictly more trainable capacity
  must overfit at least as well" is true at the optimum and says nothing
  about a fixed-`lr` path to it.

**The fix made the gate stronger, not looser.** Re-measured on this box (Tesla
P40, Vulkan, `SupirConfig::tiny`, `H=W=8`, 120 steps, both seeds, both modes):

| lr | adaptor 101 | adaptor 202 | full 101 | full 202 |
|---|---|---|---|---|
| 0.03 | 0.253 | 0.312 | 0.242 | **0.664** (peak 1.395) |
| 0.01 | 4.5e-5 | 4.8e-7 | 5.0e-6 | 9.8e-7 |
| 0.005 | 1.9e-7 | 2.0e-7 | 2.3e-7 | 2.8e-7 |

(each cell `last / l0`.) At `lr = 0.005` every combination reaches a literal
near-zero floor with no post-warm-up step above `0.008 * l0`, so the gate
moved from `last < 0.35 * l0` to `last < 1e-4 * l0` - three and a half orders
of magnitude tighter, on a trajectory that no longer has a phase to sample.
The old comment's framing ("this gate asserts clear, substantial descent
rather than a literal near-zero floor") was describing the divergence, not the
model.

And because "converged" is the property the old gate could not see, the new
one now asserts it directly and cheaply: `overfit_single` returns the WORST
loss after a 20-step Adam warm-up alongside the last one, and every caller
asserts that it stayed below `l0`. That is the check that would have caught
this on the step it happened rather than by luck 14 steps later, it costs one
`max` per step, and it is what makes the endpoint threshold meaningful instead
of a sample. Measured margin on the shipping configuration: worst post-warm-up
loss is `4.6e-3` / `5.4e-3` / `6.3e-2` against an `l0` of `7.4e-1` / `7.1e-1`
/ `7.9e-1`.

**And the calibration named a box that no longer exists.** The comment said it
was measured "on this machine's real Vulkan iGPU backend"; this machine is now
a discrete P40. A single-point assertion on a non-converged trajectory cannot
survive that, because the excursions move - which is lesson #58's "a number in
a comment has no version" with teeth: the number was not just stale, it was
*only ever true for one phase of one run*.

The general shape: before calibrating a threshold against a training run, look
at the run at full resolution and confirm it CONVERGED. A descending
every-20th-step summary is compatible with divergence. If the trajectory
oscillates, the honest options are to fix the hyperparameter until it does not
(preferred - it usually tightens the gate by orders of magnitude, as here) or
to assert on a statistic the phase cannot move (`min` and a hold bound, the
shape `crates/flux2/tests/lora_train.rs` and `supir::lora` already use, and
which each says in-line WHY it is a floor-and-hold rather than an endpoint).
Never re-fit the threshold to the endpoint the unstable run happened to land
on.
