<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# rubiks - a decision model turning a Rubik's cube, with a planner that knows the answer

A decision model returns a calibrated distribution over options the caller
supplies at run time. This sample puts that in front of a puzzle where every
answer is checkable, and then checks it.

The shape is the one `laya-mlx` uses for its Snake demo - *"a deterministic
planner describes legal directions ... Laya receives those descriptions"*, and
a safety layer *"executes the model's highest-probability admissible
direction"*, counting interventions:

1. an **exact planner** (`search.rs`) describes this turn's candidate moves;
2. the **model** picks one, from their text;
3. a **shield** plays the model's highest-probability *admissible* pick -
   admissible meaning **provably on a shortest solution** - and counts an
   intervention whenever the model's own first choice was not.

So the cube is always solved, in the optimal number of moves, and the number
this sample reports is not "it worked" but **how often the model's first pick
was one the planner would have made**, against the chance rate for the same
question.

## Run it

```bash
brain pull convaiinnovations/laya           # once

make samples/decision/rubiks/build
./target/release/sample-decision-rubiks --cubes 3 --scramble 6
./target/release/sample-decision-rubiks --cubes 3 --scramble 6 --hints off
./target/release/sample-decision-rubiks --cubes 3 --scramble 4 --unassisted
```

`--model NAME|DIR` (`laya` by default), `--cubes`, `--scramble 1..8`,
`--options` (candidates offered per turn), `--seed`, `--hints on|off`,
`--unassisted`, `--quiet`, plus the usual `--device`/`--backend`.

## What a run looks like

```text
cube 1 - scrambled by D2 F D' U2 B (undone by B' U2 D F' D2), 5 moves from solved;
         one optimal solution is B' U2 D F' D2
   5 left | model picks R' turn the right face a quarter turn counter-clockwise p=0.330 NOT on a shortest path | plays B'
   4 left | model picks F' turn the front face a quarter turn counter-clockwise p=0.385 NOT on a shortest path | plays U2
   3 left | model picks F  turn the front face a quarter turn clockwise         p=0.259 NOT on a shortest path | plays D
   2 left | model picks F' turn the front face a quarter turn counter-clockwise p=0.248 on a shortest path     | plays F'
   1 left | model picks R  turn the right face a quarter turn clockwise         p=0.210 NOT on a shortest path | plays D2
  solved in 5 moves (optimal is 5)
      UUU
      UUU
      UUU
  LLL FFF RRR BBB
  LLL FFF RRR BBB
  LLL FFF RRR BBB
      DDD
      DDD
      DDD
```

Every line is checkable: the scramble, its own undo, the planner's optimal
solution (asserted against the cube before it is printed), what the model
wanted, what the shield played, and the finished net.

## What is actually guaranteed

**The cube engine is derived, not typed in.** Each of the 54 facelets knows
where it sits and which way it faces; a move rotates the coordinates of one
layer and looks up which facelet now occupies that place. Six hand-written
permutation tables would be six chances to transpose a pair of digits and get
a thing that turns plausibly and is not a cube - and a solver on top of it
would still "solve" its own broken group. The tests are the group laws: every
face has order 4, opposite faces commute and adjacent ones do not, the sexy
move `(R U R' U')` has order 6, a T-perm is an involution.

**The planner is exact, not heuristic.** A breadth-first sweep from the solved
cube records the true distance of every state within four moves of it; a
depth-limited search from the scramble only has to reach that shell. Because
the sweep is complete, a state missing from it is known to be further than
four moves away - which prunes the forward search to `bound - 4` deep and is
the difference between this suite running in 0.07 s and in 25 minutes (it did,
before that line existed).

**The shield's promise is tested without a model in the loop**: against an
adversary that always names the option the planner likes least, every cube
still comes out solved in exactly the optimal number of moves.

Scope, stated: scrambles up to **8 moves**, solved **optimally**. A 20-move
worst case wants Kociemba's two-phase algorithm with pattern databases, which
is a different and much larger piece of work, and this sample does not claim
it.

## What the model contributes

Ten cubes, six-move scrambles, `convaiinnovations/laya` zero-shot, 57 turns
decided in each configuration:

| | first pick on a shortest path | chance | mean confidence | cubes solved |
|---|---|---|---|---|
| options say what they do (`--hints on`) | **26%** (15 of 57) | 18% | 0.121 | 10 of 10, in 57 of 57 optimal moves |
| options say only which turn they are (`--hints off`) | **11%** (6 of 57) | 18% | 0.076 | 10 of 10, in 57 of 57 optimal moves |
| no shield (`--unassisted`, 4 cubes) | 26% (9 of 35) | 18% | 0.135 | **0 of 4** - 35 moves spent on cubes needing 16 |

Read that carefully, because it says three different things.

**The cube is always solved, and that is the shield's doing, not the
model's.** Ten of ten, in exactly the optimal number of moves, while the
model's own first choice was admissible about a quarter of the time. Take the
shield away and it is zero of four: the model walks cubes further from solved
than they started, sometimes past the eight moves this planner can even
measure.

**With the effect spelled out in the options it reads a little**: 26% against
18% chance. Pooled over both hint-on runs (24 of 92 turns) that is a +8 point
lift at roughly p = 0.02 one-sided - real, small, and not what anyone would
call solving a cube. At 57 turns alone it is not distinguishable from chance,
and this sample prints the chance rate next to the result precisely so that
cannot be glossed over.

**Without the hints it is AT or below chance** (11% against 18%), which is
the honest measure of what this checkpoint knows about a Rubik's cube: nothing.
Its confidence says so too - 0.076 mean, on a scale where the same model
routes a support ticket at 0.85.

### Three things that were wrong before they were the model's fault

Each of these looked like "the model cannot read the options" and was not.
Every one was found by sending the sample's own question through
`samples/decision/json` and changing one thing at a time.

| what changed | P(the right option) |
|---|---|
| option = hint + the move spelled out again ("turn the right face a quarter turn clockwise") | 0.16 |
| option = the hint alone, move identity left to the label | **0.61** |
| state = per-face census ("top face 4/9 solved, right face 5/9 solved, ...") | 0.10 |
| state = one short sentence | **0.52** |

Boilerplate repeated across every option drowns the phrase that separates
them; a wall of near-identical numeric clauses in the state competes with the
options for the same packed sequence. Both are the caller's bug, not the
model's, and both were worth more than any change to the model would have
been.

The third one is the model's, and it is worth knowing: the SAME options that
read at 0.52 behind a neutral state read at 0.12 once the state mentions a
Rubik's cube. Its ability to read its options is not robust to the domain of
the state - which is the argument for a design where a verifier, not the
model, decides what actually gets played.

## Why the options are drawn, not listed

A cube has eighteen moves, and offering all of them every turn would run the
model's own packed-sequence budget out and silently shorten the option
descriptions - which are the thing being read. So each turn offers a random
subset (six by default) that always contains at least one admissible move, in
random order.

That also keeps the question honest: a fixed list in a fixed order is a list
whose right answer can be found by POSITION rather than by reading it, which
is the one thing a decision model must not be allowed to do.
`samples/decision/intents` is the sample that exists to test exactly that
property.

## Cost

26 brain crates - it names one surface (`decision`), like every other sample
in this directory.

---

Swedish Embedded AB builds the layer that makes a model's judgement safe to
act on: a planner that bounds it, a shield that enforces the bound, and the
measurement that says what the model contributed. If your team needs that, you
can procure our services by sending an email to info@swedishembedded.com.
