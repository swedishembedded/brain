<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 145. A held-out split drawn like the training set cannot see a skewed curriculum

`samples/decision/rubiks` generates its labels by walking scrambled cubes home
along optimal solutions - the move that undoes each step is, by construction,
a move that gets closer, so labels are free and the planner only has to verify
them. A walk from depth *d* yields one state at every distance from *d* down
to 1, so distance 1 appears in EVERY walk and the deepest distance only in the
walks that started there.

Nobody chose that curriculum. It is an artifact of the generation method, and
it is roughly 5:1 shallow. The fitted policy reproduces it exactly:

| distance from solved | examples | first pick on a shortest path |
|---|---|---|
| 1 | 83 | 100% |
| 2 | 70 | 100% |
| 3 | 56 | 92.9% |
| 4 | 46 | 76.1% |
| 5 | 29 | 48.3% |
| 6 | 16 | 43.8% |

The held-out score over that same split was **87%**, and it was not wrong - it
is the correct average over the distribution it was drawn from. It is simply
not a number about the task. Held-out data drawn the same way as training data
is skewed the same way, so it prices the skew in rather than reporting it.

What the task actually asks is different, and one measurement says so: let the
policy DRIVE. A solve spends most of its turns at the deep end - and every
mistake moves it deeper - so the same model that scored 87% held out picked a
shortest move on **37%** of the turns it took, and finished **9 of 50** cubes.
An 87% decision model that solves 18% of its problems is not a contradiction;
it is two questions with different answers.

THE RULE. When a generation METHOD implies a distribution - walking home, replaying
a log, scraping whatever was convenient - that distribution is a curriculum
choice nobody made, and a held-out split drawn from it will not report it.
Two defences, and they are cheap next to the training run they protect:

1. **Balance the axis the method skews, or state why you did not.** A quota
   per bucket is usually a few lines. Here the shallow end was already at
   100% - additional shallow data had no marginal value at all, so the skew
   was not merely unhelpful, it was spending most of the budget on the part
   that was finished.
2. **Score the policy on the states its OWN behaviour reaches, not only on a
   held-out draw.** For anything that acts in sequence, these are different
   distributions - errors compound into states the training draw never
   visited. Report both, side by side. A single held-out number for a
   sequential task is the measurement that lets this ship.

The general shape is distribution shift, which is textbook. What is worth
writing down is that **every signal available said the model was fine**: the
loss descended, the held-out accuracy was high, and the per-distance
breakdown - which did contain the whole story - reads as a normal difficulty
curve unless you also look at the example COUNTS beside it.
