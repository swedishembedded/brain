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

The planner describes nothing to the model. It supplies the cube, and it
judges the answer; the model decides in between.

So the cube is always solved, in the optimal number of moves, and the number
this sample reports is not "it worked" but **how often the model's first pick
was one the planner would have made**, against the chance rate for the same
question.

## Run it

```bash
brain pull convaiinnovations/laya           # once

make samples/decision/rubiks/build
./target/release/sample-decision-rubiks --cubes 3 --scramble 6
./target/release/sample-decision-rubiks --cubes 3 --scramble 4 --unassisted
./target/release/sample-decision-rubiks --model minilm --train 2000 --eval 40
```

`--model NAME|DIR` (`laya` by default), `--cubes`, `--scramble 1..8`,
`--seed`, `--encode compact|rows`, `--head FILE`, `--unassisted`, `--quiet`,
the training flags below, and the usual `--device`/`--backend`.

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

## What the model is asked

Every turn it gets the **complete cube** - all 54 stickers, in a fixed order -
and **all eighteen moves**, described identically:

```text
state    U:WWYWWWWWG R:RRRBRRRRO F:GGGOGGWGG D:YYRYYYYYB L:OOOOOOBOO B:BBBBBBGBY
ask      Which turn brings this cube closer to solved?
options  R: right face clockwise | R2: right face half turn | R': right face anticlockwise | ... (18)
```

Nothing in that text says which move helps. Whether one does is decided
afterwards by the exact planner, and that verdict is used for exactly two
things: as the LABEL when training a head, and as the SCORE when measuring
one. It never reaches the model's input.

An earlier version of this sample annotated each option with the planner's
own verdict ("one step closer to solved, 2 moves left"). Picking that is
reading an oracle, not deciding anything, and it could never have constituted
solving a cube. Two tests keep it out: one scans the option text for verdict
words and for differences in shape, and one proves the state text is COMPLETE
by parsing it back into a cube - a summary could not be inverted.

## Training a head on cube decisions

The labels are free. Walk AWAY from the solved cube and the move that undoes
each step is, by construction, a move that gets closer - the standard
backward-generation scheme, and it needs no solver. The one thing it can get
wrong is a walk that doubles back on itself, after which "undo the last move"
is still legal but no longer a shortest step, so each state is checked against
the planner (`distance == steps taken`) and the walk is abandoned the moment
that stops holding. The planner VERIFIES the label; it never produces it.
That runs at **54,000 examples/s**, and depth is drawn with a curriculum
weighted shallow, because the end of every solve is a shallow state.

```bash
rubiks --model minilm --train 4000 --examples 8000 --scramble 2 --eval 40 --cubes 10 --unassisted
```

`--train` trains, `--eval` scores held-out decisions (different seed) broken
down by distance-from-solved, and the run then tries to solve cubes with no
shield at all. Training needs a `decide`-shaped encoder (`--model minilm`): a
Laya checkpoint ships its head pretrained and has no optimizer loop in this
SDK yet, and the sample says so by name rather than failing obscurely.

## What it does, and does not, do yet

Measured on this box, `convaiinnovations/laya` zero-shot and a MiniLM head
trained for 200 steps:

| | first pick gets closer | chance |
|---|---|---|
| untrained head, held out | 1.7% | 5.9% |
| head after 200 steps, held out | 3.3% | 6.1% |
| the same head, on the states its own play walked into | **25%** | 7% |
| **cubes solved unassisted** | **0 of 10** | - |

Read honestly: the model has learned SOMETHING - 25% against 7% over 120
turns is far outside noise - and it is nowhere near enough to close a solve.
The reason is not subtle. Training runs at **batch size one**, and one
example gives a gradient estimate so noisy that the loss wanders (2.49, 1.07,
2.82, 2.09, 2.87 against a chance level of ln 18 = 2.89) instead of
descending. 200 steps is what 24 minutes buys here.

**The step is host-bound, not device-bound**: it costs ~2.4 s on this
machine's Intel Arc iGPU and ~2.4 s on its 22-core CPU backend, and two
numbers that equal cannot both be the arithmetic. For contrast
`samples/decision/triage` records 24 ms/step on a Tesla P40 after an earlier
optimisation pass. Batching examples - which the decide crate's own notes name
as the next real win, and which fixes the gradient noise at the same time - is
the work that has to land before this measurement means anything.

So: **the shield solves every cube, optimally, always. The model does not
solve one yet.** Both numbers are printed by every run, side by side, which
is the only arrangement in which the first is not mistaken for the second.

## Watching it

```bash
rubiks --window                      # a real window, if there is a display
rubiks --frames /tmp/frames --fps 12 # every frame as a PNG, headless
rubiks --record run.mp4              # the same, encoded (needs ffmpeg)
```

The cube is drawn as what it is - 26 plastic cubies, each with a sticker on
the faces that reach the surface - through a perspective camera, sorted back
to front, with the turning layer animated through the same angle the engine's
own move applies. It is drawn from `cube::facelet_geometry`, the same layout
the engine turns, so the picture cannot drift out of agreement with the
state; a test pins the end of every animation to where the discrete move
lands. Beside it: the state the model was given, every option with its
probability, which one it picked, which one was played, and the running
counts.

## Cost

26 brain crates - it names one surface (`decision`), like every other sample
in this directory.

---

Swedish Embedded AB builds the layer that makes a model's judgement safe to
act on: a planner that bounds it, a shield that enforces the bound, and the
measurement that says what the model contributed. If your team needs that, you
can procure our services by sending an email to info@swedishembedded.com.
