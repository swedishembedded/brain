<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 137. A shaped reward can teach exactly the behaviour you are trying to prevent

Reaching the exit of a DOOM level pays once, hundreds of decisions after the
choices that earned it, so the reward was shaped with progress toward it:
`r += w * (previous_distance - current_distance)`. That is textbook
potential-based shaping (Ng, Harada & Russell 1999) and it provably leaves the
OPTIMAL policy unchanged, which is why it looked safe.

It is not safe, because the guarantee is about the optimum and not about what
is learned on the way to it. **The straight line to the exit goes through
walls.** Measured on E1M1: a trajectory spent 90 of its 120 decisions shuttling
between two spots - walk at the wall the exit is behind, back out, walk at it
again - and was PAID for every one of them. The agent was being taught to walk
into walls by the reward function, and the scripted teacher whose episodes were
being cloned was doing the same thing for the same reason.

Nothing looked wrong. Returns were positive, the loss fell, the critic fitted.
The only visible symptom was a distinct-cells-visited count of 15 over 120
decisions, which nobody was looking at because it was not a metric anybody had
thought to record.

THE RULE. When a shaping term is a heuristic about the GOAL rather than a fact
about the environment, ask what it pays for in the states where the heuristic
is wrong - not whether it preserves the optimum. A potential that is a straight
line in a space with obstacles pays for pressing against the obstacle.

WHAT REPLACED IT. A count-based exploration bonus: the first visit to a patch
of floor in an episode pays, the n-th pays `1/sqrt(n)`. It is the cheap,
network-free member of the intrinsic-motivation family (ICM, RND) that the
sparse-reward navigation literature converged on, and it is a fact about where
the agent has BEEN rather than a guess about where it should go - so there is
no state in which it rewards the wrong thing. Walking into a wall discovers
nothing and earns nothing.

The `1/sqrt(n)` tail rather than a one-shot first-visit bonus is deliberate: a
strictly one-shot bonus makes an already-walked corridor worth exactly zero, so
an agent that must cross one to reach anything new is charged for the crossing.

RELATED, SAME CHANGE. A scripted teacher's episodes were being cloned
wholesale. A heuristic teacher is not uniformly good - it is good in the
situations it was written for and arbitrary everywhere else, and its bad
episodes are bad in a specific, learnable way. Keeping only the best fraction
by return (`ControlSpec::warmup_keep`, what the imitation-learning literature
calls filtered behaviour cloning) costs nothing a run does not already have,
and stops the policy gradient having to spend its samples unlearning something
it was deliberately taught. On the first real run it dropped 5 of 12 episodes,
the worst kept scoring +2.87 against the best at +11.06.
