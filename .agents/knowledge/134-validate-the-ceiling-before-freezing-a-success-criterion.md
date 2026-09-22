<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 134. Validate the ceiling before freezing a success criterion

Freezing success criteria before the experiment is right: it stops a threshold
drifting to meet whatever came out. Freezing an UNVALIDATED one is worse than
not freezing at all, because it locks in a number nobody can reach and then
reads the failure as a property of the method.

A control sample's criteria were frozen at "beat 71%". Seven hand-written
policies were then measured on the same game and **the one that ignored the
observation won**, at 63.7%; every policy that read range, target type or
health did worse. The ceiling WAS the naive policy. Three training runs, a
minibatching bug and a missing critic were all diagnosed against a bar no
policy could clear.

THE MISSING STEP. Before setting any threshold, measure a LADDER of
hand-written policies and require that one which reads the observation beats
one that ignores it, by a wide margin. If that gap does not exist, the
environment cannot demonstrate learning and no result on it is interpretable -
whatever the learning curve looks like. It is now an assertion
(`the_policy_ladder_shows_the_headroom`), so a broken testbed fails loudly
instead of producing numbers that mean nothing.

Redesigned so target choice decided the fight rather than fire rate, the same
ladder spread 35 points, and the same PPO went from "cannot beat the teacher"
to +35 points over it.

TWO MEASUREMENT TRAPS FOUND WHILE VALIDATING THE REPLACEMENT, both of the same
family:

* **A baseline that is a lottery.** The untrained-policy bar read 74% on the
  default seed and 0-5% on seven others. A greedy untrained head is
  deterministic - it prefers one kind of option forever - so it is usually
  catastrophic and occasionally lands on a near-optimal rule by accident. Any
  bar quoted from ONE initialisation is a coin flip; quote a median.
* **A baseline silently broken by an unrelated fix.** Switching the update to
  minibatches cut a behaviour-cloning warm start from 600 optimizer steps to
  nine. It produced 38.5% on one seed and 1.0% on another, and the first looked
  right because it was near the teacher it was cloning. That number was the
  justification for a frozen threshold, so fixing it invalidated the criterion.

THE RULE FOR THAT LAST ONE. A threshold is only as good as the baseline it was
set against, and a baseline is a measurement that can be wrong. When one turns
out to be, append an ERRATUM saying which criteria it invalidates and which now
carry the claim - do not edit the frozen table, and do not quietly let the
weakened criterion stand in for the strong one.
