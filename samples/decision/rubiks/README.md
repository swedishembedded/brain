<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# rubiks - putting a model in control of something you can verify

You have a model whose judgement is useful and not trustworthy, and an action
it wants to take. Ship it and you inherit its worst decision. Don't ship it and
you inherit none of its good ones.

This sample is the third option, built end to end on a puzzle where every
decision has a checkable right answer: **a verifier bounds what the model is
allowed to do, the model chooses inside that bound, and you get a number for
what the model actually contributed.** The cube is the test rig. The
arrangement is the product.

```text
        ┌──────────┐   candidate moves    ┌───────┐   ranked picks   ┌────────┐
state → │ planner  │ ───────────────────► │ model │ ───────────────► │ shield │ → action
        │  exact   │                      │ picks │                  │ bounds │
        └──────────┘                      └───────┘                  └────────┘
              └──────────── judges the result, never tells the model ──────────┘
```

Three properties, and each is worth something different:

- **The cube always gets solved, optimally, whatever the model says.** The
  shield plays the model's highest-ranked move *among those provably on a
  shortest solution*. A model having a bad day costs you optimality of nothing
  - it costs you the model's contribution, and the run says so.
- **What the model is worth is measured, not asserted.** Every run reports how
  often the model's own first pick was one the planner would have made, beside
  the rate for guessing. `--unassisted` removes the shield entirely, which is
  the measurement that says what the model is worth *without* one.
- **The model is never told the answer.** The planner's verdict is used as a
  training label and as a score. It never enters the model's input.

## Run it

```bash
brain pull sentence-transformers/all-MiniLM-L6-v2   # the encoder, once

make samples/decision/rubiks/build
./target/release/sample-decision-rubiks --cubes 3 --scramble 6
```

That solves three cubes with an untrained model behind the shield - useful on
its own, because it shows the shield carrying a model that contributes nothing.
To see a trained one decide for itself, train a policy first (below) and run it
with no shield:

```bash
./target/release/sample-decision-rubiks --model out/rubiks-model \
    --cubes 20 --scramble 4 --unassisted
```

## What a run looks like

```text
cube 1 - scrambled by D2 F D' U2 B (undone by B' U2 D F' D2), 5 moves from solved; one optimal solution is B' U2 D F' D2
   5 left | model picks B' turn the back face a quarter turn counter-clockwise p=0.915 on a shortest path | plays B'
   4 left | model picks D' turn the bottom face a quarter turn counter-clockwise p=0.701 NOT on a shortest path | plays D
   3 left | model picks D2 turn the bottom face a half turn p=0.386 NOT on a shortest path | plays U2
   2 left | model picks F' turn the front face a quarter turn counter-clockwise p=0.980 on a shortest path | plays F'
   1 left | model picks D2 turn the bottom face a half turn p=0.999 on a shortest path | plays D2
  solved in 5 moves (optimal is 5)
```

Every line is checkable: the scramble, its own undo, the planner's optimal
solution (asserted against the cube before it is printed), what the model
wanted, what the shield played, and the finished net.

## Train a policy

The labels are free. Walk *away* from a solved cube and the move that undoes
each step is, by construction, a move that gets closer - so a cube is an
endless supply of decisions with a known right answer, generated far faster
than training consumes them: the whole 200,000-example set the command below
uses is built in seconds, against a fit that then runs for over an hour. The
planner *verifies* each label (a walk that doubles back is abandoned the moment
"undo the last move" stops being a shortest step); it never produces one.

```bash
./target/release/sample-decision-rubiks --model minilm \
    --train 8000 --batch 32 --examples 200000 --scramble 6 \
    --eval 300 --save out/rubiks-model \
    --cubes 50 --unassisted
```

`--train` trains, `--eval` scores held-out decisions (a different seed) broken
down by distance-from-solved, `--save` writes a complete checkpoint directory,
and the run then tries to solve cubes with no shield at all. Reuse the result
by passing it as the model:

```bash
./target/release/sample-decision-rubiks --model out/rubiks-model --cubes 10 --unassisted
```

Two flags decide whether this works at all, and neither is a tuning detail:

- **`--batch`** is how many decisions are accumulated into each optimizer step.
  One decision's gradient is a very noisy estimate of the objective's, and a
  run at batch 1 spends its budget chasing that noise - the loss wanders
  instead of descending. The default is 16.
- **`--scramble`** sets how deep the training cubes go, and a policy only
  learns the distances it was shown. Training data is drawn along optimal
  solutions rather than from fresh scrambles, because that is the distribution
  a solve actually walks through: every run ends in the shallow states, so a
  policy trained only on deep ones has never seen the last move of a solve.

Training needs a `decide`-shaped encoder (`--model minilm`). A Laya checkpoint
ships its head pretrained and the sample says so by name rather than failing
obscurely.

## What it achieves

Two numbers, and they answer different questions. Both are printed by every
run, side by side, which is the only arrangement in which the first is not
mistaken for the second.

**With the shield, every cube is solved in the optimal number of moves, at any
depth this planner covers** - including the cubes the model on its own cannot
finish. That is a property of the arrangement, not of the model: it holds
against an adversary that always names the worst option, and a test asserts it
with no model in the loop.

**Without the shield, the model drives.** A policy trained by the command
above - a MiniLM encoder, an hour and a half of labelled decisions, no search
at inference and one forward pass per move - solves 200 random cubes per depth:

| scramble depth | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| solved, no shield | 100% | 100% | **96%** | 78% | 45% | 16% |

At three moves it is reliable, and beyond that it degrades for a reason worth
understanding before you copy the pattern: **a greedy policy has to be right
every turn.** The same run's held-out decisions show accuracy falling with
distance from solved - 100% one move out, 94% at three, 78% at four, 44% at
six - and a solve has to string a whole run of those together, every wrong turn
moving the cube further out and drawing the next decision from the harder end
of that curve. The gap between the two views is the whole point: 44% of
individual decisions right at distance six, and 16% of six-move cubes
finished. Accuracy that looks respectable per decision is not the same as
finishing.

Where the deep end is capped is not something this sample establishes, but
the architecture suggests where to look first: the state and the options never
meet inside the encoder, only in the single cross-attention layer that scores
them, so "would turning this face help" has exactly one layer in which to be
computed. Solving a 20-move worst case is a
different design - a value network with search, in the DeepCubeA shape - and
this sample does not claim it.

Which is the point of the shield. A model that is excellent near the goal and
weak far from it is exactly the kind you can still ship, provided something
else holds the floor.

## Watching it

```bash
rubiks=./target/release/sample-decision-rubiks   # the shorthand below

$rubiks --window                     # a real window, if there is a display
$rubiks --frames DIR --fps 12        # every frame as a PNG, headless
$rubiks --record run.mp4             # the same, encoded (needs ffmpeg)
```

Both halves of the result above are worth watching, and both record headlessly
on a machine with no display:

```bash
# the model driving, alone, on twelve random three-move cubes
$rubiks --model out/rubiks-model --cubes 12 --scramble 3 --unassisted --record solo.mp4

# the shield carrying it at a depth the model cannot finish by itself
$rubiks --model out/rubiks-model --cubes 6 --scramble 8 --record shielded.mp4
```

The cube is drawn as what it is - 26 plastic cubies, each with a sticker on the
faces that reach the surface - through a perspective camera, sorted back to
front, with the turning layer animated through the same angle the engine's own
move applies. It is drawn from `cube::facelet_geometry`, the same layout the
engine turns, so the picture cannot drift out of agreement with the state.
Beside it: the state the model was given, every option with its probability,
which one it picked, which one was played, and the running counts.

## What is actually guaranteed

**The cube engine is derived, not typed in.** Each of the 54 facelets knows
where it sits and which way it faces; a move rotates the coordinates of one
layer and looks up which facelet now occupies that place. Six hand-written
permutation tables would be six chances to transpose a pair of digits and get a
thing that turns plausibly and is not a cube - and a solver on top of it would
still "solve" its own broken group. The tests are the group laws: every face
has order 4, opposite faces commute and adjacent ones do not, the sexy move
`(R U R' U')` has order 6, a T-perm is an involution.

**The planner is exact, not heuristic.** A breadth-first sweep from the solved
cube records the true distance of every state within four moves of it; a
depth-limited search from the scramble only has to reach that shell. Because
the sweep is complete, a state missing from it is known to be further than four
moves away. That turns the forward search from one that must reach the solved
state into one that only has to reach the shell, halving the depth it explores
of a tree that branches eighteen ways - which is what lets the whole suite,
exact solutions and all, run in a fraction of a second.

**The shield's promise is tested without a model in the loop.** Against an
adversary that always names the option the planner likes least, every cube
still comes out solved in exactly the optimal number of moves.

**The model cannot read the answer.** An earlier version of this sample
annotated each option with the planner's verdict ("one step closer to solved").
Picking that is reading an oracle, not deciding anything. Two tests keep it
out: one scans the option text for verdict words and for differences in shape,
and one proves the state text is complete by parsing it back into a cube - a
summary could not be inverted.

## What the model is asked

Every turn it gets the **complete cube** - all 54 stickers, in a fixed order -
and **all eighteen moves**, described identically:

```text
state    U U U R R F D D D  R R R B R R R R O  ...   (54, one per sticker)
ask      Which turn brings this cube closer to solved?
options  R: right face clockwise | R2: right face half turn | R': right face anticlockwise | ... (18)
```

`--encode` chooses how the cube is written down. The default, `grid`, gives
each sticker its own whitespace-separated symbol, so the tokenizer emits
exactly one token per sticker and row *i* is always sticker *i*. That matters
because the encoder is a WordPiece model: pack the stickers into runs
(`U:WWYWWWWWG`, the `compact` encoding) and it cuts them at whatever boundaries
its vocabulary happens to have, so a state's token count moves with its
content, every sticker after the first difference shifts row, and the learned
position embedding - the only thing that says which sticker is which - reads a
different sticker at every row.

## Adapting it to your problem

The cube is replaceable; the arrangement is not. To put this pattern behind
your own decision, you need three things:

1. **A verifier that is cheap relative to the decision** - here an exact
   meet-in-the-middle search. It has to answer "is this action admissible?"
   without needing the model.
2. **An admissibility rule that always leaves at least one option.** The shield
   is only safe because every unsolved cube has a move that helps; if your rule
   can empty the option set, you need a defined fallback before it does.
3. **Labels from the verifier, kept out of the input.** The same verdict that
   scores a decision can train on it - which is what makes the labels free -
   but the moment it reaches the model's input you are measuring a reading
   comprehension task instead.

`search.rs` is the verifier, `policy.rs` is the state/option/label contract,
and `main.rs` is the shield and the tally. Those three files are the parts you
would rewrite; `cube.rs` and `view.rs` are the puzzle and its picture.

## Cost

It names two surfaces where most samples in this directory name one:
`decision` to score the options, and `viewport` because it draws the cube.
Dropping the window and the recorder drops the second. The linked-crate
budget is enforced from `Cargo.toml`, which is where to read it - a count
copied into prose drifts silently as the workspace grows.

---

Swedish Embedded AB builds the layer that makes a model's judgement safe to act
on: a planner that bounds it, a shield that enforces the bound, and the
measurement that says what the model contributed. If your team needs that, you
can procure our services by sending an email to info@swedishembedded.com.
