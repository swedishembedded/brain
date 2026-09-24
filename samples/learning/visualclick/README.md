<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# visualclick - can a row the encoder never produced drive a decision?

`Decide` answers a runtime-supplied question from a **text** state:
tokenize, encode, cross-attend. It has no image input, no non-text input
hook of any kind. This sample validates one specific, minimal claim about
what it would take to add one: that a row spliced directly into the head's
cross-attention state - never touched by the encoder, produced by a small
trainable projector instead - can drive the model to a spatially grounded
answer it could not otherwise give.

Three arms answer the SAME 16-way "which grid cell" question over the SAME
synthetic scenes (3 colored rectangles on a 4x4 grid; the instruction names
one color; the answer is the cell it occupies), differing only in what
reaches the model:

```text
blind   nothing but the instruction               -> must read chance
text    scene facts serialized to text              existing, UNMODIFIED
        (Decide::train_step_with)                    Decide path
pixels  16 rows spliced into the cross-attention     the NEW path:
        state (Decide::accumulate_kept)               Features::from_parts
```

`blind` at chance rules out label leakage through the option names. `text`
well above chance proves the task, the grid, the question shape, and the
training loop are sound BEFORE any new capability is on trial - see "Two
designs that looked right and were not," below, for what that caught.
`pixels` decisively above `blind` is the verdict on the splice mechanism
itself; two ablations (`--ablate noise`, `--ablate shuffle`) must both
collapse `pixels` back to chance or the result is not trustworthy - a
statistic an ablation also produces is not evidence for the thing under
test.

## Where the splice actually is

`Decide`'s head is a **cross-attention** over the encoder's own hidden
states: every option's query attends over the state's rows (`crates/decide/
src/head.rs`). `Decide::accumulate_kept` (`crates/decide/src/decide.rs`)
exists so a frozen encoder's hidden states can be supplied from a cache
instead of a live forward pass - and nothing in it requires that cache to
have come from the encoder at all. `Features::from_parts` (added to
`crates/decide` for this sample) builds one directly: state rows come from a
small host-side `Linear -> LayerNorm` (`src/patches.rs`) over each grid
cell's color identity, never from a token embedding. Training reaches the
projector through the SAME reverse pass that trains the head - the head's
backward writes `dL/d(state row)` into the encoder's own seed buffer
regardless of whether the encoder itself is frozen, and that buffer is read
back out on the host to run the projector's own Adam step. No change to
`crates/decide`'s embedding path, no new kernel, no GPU code in the
projector at all.

## Two designs that looked right and were not

Both were caught by real measured runs, not by review, and both are worth
keeping in mind before extending this sample:

1. **The question has to carry the per-example content, not the state.**
   A first version put the instruction ("click the red rectangle") in
   `state` and left `Question::instructions` a single string shared across
   every example. `Decide`'s head computes an option's query from the
   option's OWN slot text (`"{instructions} [SEP] {option}"`), never from
   the state it attends over - so with a shared `instructions`, "cell 7"'s
   query was nearly identical across every example, and cross-attention had
   no channel to compare a retrieved state fact against a color it never
   saw. `text` trained flat at chance for 2000-3000 steps, even with the
   encoder UNFROZEN (ruling out "just needs more encoder capacity"). Fixed
   by building a fresh `Question` per example with the color in
   `instructions` - see `question_for` in `src/main.rs`.
2. **A random projector needs a fair fight against the head's other
   option, or it never gets gradient.** Even after fix 1, `pixels` trained
   flat at chance through 15000 steps. Reading `Head::read_grad` on the
   first few steps showed the head's own weights getting a normal-scale
   gradient while the SLICE of it reaching the image rows specifically was
   two orders of magnitude smaller: the cross-attention head was spending
   its budget on a placeholder text state (constant, familiar-looking to a
   frozen encoder, informationally useless) rather than the image rows
   (novel at init, actually informative) - a cold-start trap with little
   gradient reaching the one thing that needed to learn. Fixed by dropping
   the placeholder text from the state entirely, so the image rows are the
   ONLY thing left to attend to - see `spliced_features`'s own doc.

A raw-RGB projector and a missing output LayerNorm were ALSO tried and fixed
before these two - see `src/patches.rs`'s own module doc for the measured
scale mismatch (projector rows RMS ~0.17-0.27 against real rows' ~0.36-0.63)
and why continuous RGB asks the head to learn a cross-modal alignment CLIP's
own contrastive pretraining exists to avoid learning from scratch.

## A measured run

`sentence-transformers/all-MiniLM-L6-v2` (`decide`'s MiniLM-L6 encoder, 6
layers, d=384), frozen, one CPU/iGPU-class run, seed 11, chance = 1/16 =
0.0625:

```
blind   (3000 steps)   held-out accuracy 0.050          ECE 0.014
text    (3000 steps)   held-out accuracy 0.270          ECE 0.090
pixels  (15000 steps)  held-out accuracy 0.273          ECE 0.097
  ablate noise   (15000 steps)  held-out accuracy 0.068   (collapses to chance)
  ablate shuffle (15000 steps)  held-out accuracy 0.062   (collapses to chance)
```

`pixels` matches `text` and clears chance by more than 4x; both ablations
land within noise of chance, which is the falsifiability check this sample
exists to pass. `text`'s loss shows a clean downward trend from ~2.9 by step
1000-1500; `pixels`'s loss stays flat near `ln(16) = 2.77` through 3000-6000
steps and only starts trending down around step 4000-4400 - see "Two designs
that looked right and were not" for why that gap is real and expected, not a
tuning accident.

Full console output for each arm, `--shot`'s rendered PNGs (outline
correctly on the named color's rectangle after a `Canvas::text` scale bug
was found and fixed - its `px` argument is a multiplier on a 5x7 font, not a
pixel height), and the gradient-probing session that found design flaw 2 are
not reproduced here; every number above came from a real run of the binary
described in "Run it," below.

## Run it

```text
make samples/learning/visualclick/run ARGS="--shot out/scenes"
make samples/learning/visualclick/run ARGS="--arm blind"
make samples/learning/visualclick/run ARGS="--arm text"
make samples/learning/visualclick/run ARGS="--arm pixels"
make samples/learning/visualclick/run ARGS="--arm pixels --ablate noise"
make samples/learning/visualclick/run ARGS="--arm pixels --ablate shuffle"
```

`pixels` defaults to 15000 training scenes where `blind`/`text` default to
3000 - see `default_train_n` in `src/main.rs` for the measured reason a
random projector needs a longer budget than a head-only bootstrap.

## What is not claimed

- **This tests a linear projection into one cross-attention layer, not deep
  multimodal encoder fusion.** A positive result here earns the right to
  attempt splicing rows earlier, into the embedding stream itself
  (`crates/decide/src/model.rs`'s `x[0]`, which would need a new device-side
  scatter - see the design note this sample's plan recorded before building
  it); it does not demonstrate that path works.
- **The color signal is a discrete one-hot over a six-color vocabulary, not
  a real image embedding.** No CLIP or other vision-tower checkpoint was
  available in the environment this was built in (`~/.local/share/brain/
  models/` had no CLIP weights, confirmed before building) - see the
  producer-swap seam in `patches::Projector`'s doc for where a real vision
  encoder's patch tokens would plug in behind the same interface.
- **The scenes are synthetic solid rectangles.** Nothing about real UI
  screenshots, occlusion, overlapping text, or visual clutter is claimed.
- **A 4x4 grid is coarse localization, not pixel-accurate clicking.**
- **`pixels` needed 5x the training budget `text` did.** That is reported as
  a real, measured cost of bootstrapping a brand-new, unaligned input
  channel from nothing (see design flaw 2, above), not smoothed into a
  single shared step count.

## Cost

34 brain crates (`decision` + `viewport`), matching `samples/decision/doom`'s
budget for the same feature combination.

---

Swedish Embedded AB builds decision systems that ground a probability in
whatever evidence actually bears on it, image or text, and reports exactly
how that claim was checked - including the two designs that looked right and
were not - rather than assuming it. If your team needs a grounded decision
system audited before it ships, you can procure our services by sending an
email to info@swedishembedded.com.
