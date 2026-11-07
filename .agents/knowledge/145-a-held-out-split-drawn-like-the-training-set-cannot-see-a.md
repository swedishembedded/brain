<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 145. A held-out split drawn like the training set cannot see a skewed curriculum

**Lesson:** A held-out split drawn the same way as the training set prices the training skew in
rather than reporting it. For anything that acts in sequence, also score the policy on the states
its own behaviour reaches.

## What happened
`samples/decision/rubiks` generates its labels by walking scrambled cubes home along optimal
solutions. The move that undoes each step is, by construction, a move that gets closer, so labels
are free and the planner only has to verify them.

The fitted policy reproduces the skew exactly:

| distance from solved | examples | first pick on a shortest path |
|---|---|---|
| 1 | 83 | 100% |
| 2 | 70 | 100% |
| 3 | 56 | 92.9% |
| 4 | 46 | 76.1% |
| 5 | 29 | 48.3% |
| 6 | 16 | 43.8% |

The held-out score over that same split was **87%**, and it was not wrong. It is the correct
average over the distribution it was drawn from. It is simply not a number about the task.

Let the policy DRIVE and the task shows up. A solve spends most of its turns at the deep end, and
every mistake moves it deeper. The same model that scored 87% held out picked a shortest move on
**37%** of the turns it took, and finished **9 of 50** cubes. An 87% decision model that solves 18%
of its problems is not a contradiction; it is two questions with different answers.

Every signal available said the model was fine: the loss descended, the held-out accuracy was high,
and the per-distance breakdown (which did contain the whole story) reads as a normal difficulty
curve unless you also look at the example COUNTS beside it.

## Why
A walk from depth *d* yields one state at every distance from *d* down to 1, so distance 1 appears
in EVERY walk and the deepest distance only in the walks that started there.

Nobody chose that curriculum. It is an artifact of the generation method, and it is roughly 5:1
shallow. Held-out data drawn the same way as training data is skewed the same way, so it prices the
skew in rather than reporting it.

The general shape is distribution shift, which is textbook.

## What to do
When a generation METHOD implies a distribution (walking home, replaying a log, scraping whatever
was convenient), that distribution is a curriculum choice nobody made, and a held-out split drawn
from it will not report it. Two defences, and they are cheap next to the training run they protect:

1. **Balance the axis the method skews, or state why you did not.** A quota per bucket is usually a
   few lines. Here the shallow end was already at 100%, so additional shallow data had no marginal
   value at all. The skew was not merely unhelpful, it was spending most of the budget on the part
   that was finished.
2. **Score the policy on the states its OWN behaviour reaches, not only on a held-out draw.** For
   anything that acts in sequence, these are different distributions, because errors compound into
   states the training draw never visited. Report both, side by side. A single held-out number for
   a sequential task is the measurement that lets this ship.
