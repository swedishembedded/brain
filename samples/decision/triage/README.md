<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# triage - train a decision model, then keep deciding with it

A decision model returns a **probability distribution over options the caller
supplies at request time**, instead of text. This sample trains one on
BANKING77 customer-service messages and then answers new ones, as a single
expression:

```rust
DecisionPipeline::from_pretrained(encoder)
    .train(spec)       // fine-tune on labelled examples
    .evaluate()        // score the held-out split
    .save(head)        // write the trained head
    .tui()             // keep answering until end of input
    .report()          // what every stage did
    .finish()?         // the one error site in the program
```

`train`, `evaluate`, `save`, `ask`, `tui`, `report` and `finish` are the stages
**every** brain pipeline has. What each one MEANS is supplied by the
architecture, including the shape of its training specification - a decision
model trains on labelled options, an image model does not, and neither has to
pretend otherwise to share the chain. A failed stage stops the chain and every
later stage skips, keeping the original cause, so five stages have one error
site rather than five.

## Run it

```bash
brain pull sentence-transformers/all-MiniLM-L6-v2   # the encoder, once
make fetch/testdata                                  # BANKING77, once

make samples/decision/triage/build
make samples/decision/triage/run ARGS="--steps 600 --intents 8"
```

That trains, evaluates, saves the head, and drops into a prompt. Then reuse it
without retraining:

```bash
make samples/decision/triage/run ARGS="--head out/triage-head.safetensors --ask 'my card never arrived'"
```

## What it shows

**The option set lives in the call, not in the weights.** The same trained model
answers a three-option question and a seventy-seven-option one with no reload,
because there is no final layer whose width is the answer space. That is also
why `--head` has to be told what the options are: weights alone do not say what
the model is deciding between.

**It returns a distribution, not a label.** Every answer carries the full
probabilities and a confidence, so the CALLER decides what is confident enough
to act on - per action, not one threshold for the whole system. The sample
prints the top three and flags anything under 0.3 as something a real system
would escalate.

**Options are sampled during training.** Each step scores its example against a
random subset of the intents, at a random position, always including the
correct one. A model always shown the same list in the same order can reach the
right answer from the POSITION without reading an option at all - and then
cannot answer a question whose options it has never seen, which is the one
thing this model is for.

## A measured run

600 steps, 8 intents, one Tesla P40, fine-tuning the encoder and a fresh head:

```
train:    600 steps, final loss 0.102, 14.2s (24 ms/step)
evaluate: accuracy 0.938 over 160 items, chance 0.125, mean confidence 0.959
ask:      "my card has not arrived yet, it has been three weeks"
          -> card arrival  (confidence 1.00)
```

That step used to take **1187 ms**, and what it was spending it on is worth
recording, because none of it was the model:

| | ms/step | why |
|---|---|---|
| gradient-norm clipping | 1163 | one thread, serially, over a 11.7M-float embedding table |
| dispatch re-recording | ~10 | the packed layout moves with every message, so the tape was rebuilt each step |
| embedding scatter | 5.7 | an invocation per (vocab row, channel) when at most ~130 rows were looked up |
| cross-attention `d_scores` | 4.2 | one thread per query row, walking its whole key axis twice |

Every one of those was a kernel or a seam that already existed and was simply
not selected: the cooperative grad-norm pair, `emb_bwd_uniq`,
`attn_bwd_dscores_cross_rows`, and `Gpu::enable_step_cache`. `decide_bench`
prints the table this came from -
`cargo run --release -p brain-decide --bin decide_bench -- 8 12 8`.

What remains is genuine arithmetic at an awkward width: the reverse pass'
GEMMs run at ~8% of this card's roof because a training step is ONE example,
134 packed rows. Batching examples is the next real win, and it is an
architectural change rather than a kernel one.

## Cost

26 brain crates, the fewest of any sample here, because it names one surface
(`decision`) and the SDK links only what that surface needs.

---

Swedish Embedded AB builds realtime decision layers - calibrated,
bounded-output models that choose among actions a system defines at run time -
for its clients. If your team needs judgment in the loop without handing
control flow to a text generator, you can procure our services by sending an
email to info@swedishembedded.com.
