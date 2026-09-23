<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 153. A value that fits its own trajectory cannot rank the alternatives to it

`samples/decision/rubiks` learns which macro to play from a set where every
member is admissible, so the choice cannot make the solve wrong - only long
or short. The obvious method is to fit cost-to-go on finished solves (labels
are exact: every state is labelled by the moves that actually followed it),
then at each step rank the candidates by `macro length + V(child)`.

Fitted on episodes from a RANDOM chooser it worked, partly: 213 moves per
solve against 356 for choosing randomly and 211 for a hand-written tiebreak.
So the obvious next step was policy iteration - regenerate episodes with the
improved chooser, refit, repeat, each round moving the target from "cost of
behaving randomly" toward "cost of behaving well".

Four rounds made it **worse**: 228 moves, against 213 after one round.

The tell was in the fit, not the result:

| round | driver | value MSE |
|---|---|---|
| 0 | random | 3.56 |
| 1 | learned, 20% random | 0.13 |
| 2 | learned, 20% random | **0.02** |
| 3 | learned, 20% random | **0.02** |

A near-zero training error on a task this hard is not success, it is a
warning. A greedy driver visits a narrow band of states, and each round
narrowed it further, so the network fit that band almost exactly.

**The states it fits are not the states it is asked about.** At decision time
the value ranks the CHILDREN of the current state - a median of 59 of them -
and most of those children are exactly the ones the greedy driver did not
take. They are off the trajectory by construction. So every round of
iteration improved the fit on the visited states and degraded it on the
compared ones, which are the only ones the decision rule reads.

THE RULE. **Fit the value on the states you will ASK about, not on the states
you will visit.** For a chooser that ranks candidates, the training
distribution must include the candidates - which means labelling rejected
children too, not only the accepted one. Keeping exploration up is a weak
substitute: at 20% random this still collapsed, because the rejected children
of a good chooser are not reachable by occasionally acting randomly, they are
reachable only by being enumerated.

Two ways to see it coming, both cheap:

1. **Watch the training error for collapse.** An error that falls two orders
   of magnitude while the measured task gets worse is the signature of a
   model fitting a distribution narrower than the one it is used on.
2. **Report the decision-time metric every round, not the loss.** Here that
   is moves per solve. The loss said the model was getting better at exactly
   the rate the solves were getting longer.

Related: [[145-a-held-out-split-drawn-like-the-training-set-cannot-see-a]] is
the same family - a number drawn from the training distribution that cannot
see the task. This entry is its sharper form: here the mismatch is not
between train and test, it is between the states a policy VISITS and the
states it COMPARES, and those differ even when both come from the same run.
