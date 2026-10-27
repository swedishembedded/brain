<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 366. A finite-difference probe through a discrete selection measures the jump, not the slope

Symptom: the first gradient check of EdgeRock's anchor memory (a bank that keeps the
top-scoring records, chosen by a sort) failed on every seed with the selection
changing between the plus and minus evaluations. Where it did not fail outright, a
passing probe proved nothing: a swapped pair of records moves the loss by a finite
amount, which the probe divides by `2 * eps` and reports as a huge "derivative".

Cause: the loss is piecewise smooth in every parameter that reaches the scorer,
and a directional probe moves ALL weights of a tensor at once. Even `tok.weight`
at the workspace step of 5e-3 moved a token's embedding by a quarter of its norm,
enough to reorder scores a few percent apart. A second slip hid behind it: the
admission scorer was listed as trainable (`param_names`) with no gradient buffer,
so `read_grad` panicked before any comparison.

Fix: the check pins the selection. `gradcheck::check_edgerock` wraps the model so
every forward also reads the banks and fails, naming the tensor last written, when
they differ from the first forward's. The harness makes the probe valid rather
than the assertion softer: a wide embedding (a step is then a small relative
move), a smaller step (5e-4), scorer weights wide enough to separate the scores,
and a frozen scorer. The scorer is `Role::Frozen`: no gradient reaches a discrete
sort, and neither optimizer step nor weight decay may change it.

Rule: a gradient check through a discrete decision must assert that the decision
did not move, and fail loudly when it did; never skip, never widen the tolerance.
A parameter with no gradient by design is frozen in the store, so every list of
"trained parameters" can be derived from it.
