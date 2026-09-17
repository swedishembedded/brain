<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# salesagent - what are the odds this deal closes, right now

This sample scores a sales conversation **while it is still happening**. After
every turn it answers one proposition - *will this end in a closed deal* - with
a probability, and attaches a confidence that says whether to act on the number
or get a second opinion.

Nothing is generated. The output is a float in `[0, 1]` and a routing decision,
and the conversation never leaves the machine.

It implements two published methods end to end:

* **SalesRLAgent** ([arXiv:2503.23303][p1]) - conversion prediction as a
  sequential decision problem, trained supervised-then-reinforcement.
* **Confidence-Aware Routing** ([arXiv:2510.01237][p2]) - three confidence
  signals combined into one score that picks one of four pathways.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Run it](#2-run-it)
3. [Reading the output](#3-reading-the-output)
4. [Using a trained model](#4-using-a-trained-model)
5. [How it works](#5-how-it-works)
6. [Interpreting the numbers](#6-interpreting-the-numbers)
7. [Faithful, adapted, not done](#7-faithful-adapted-not-done)
8. [Command reference](#8-command-reference)

---

## 1. Prerequisites

Three things, fetched once. Nothing else is downloaded at run time - the sample
tells you what is missing and exits rather than starting a download.

### 1.1 The encoder checkpoint (~90 MB)

```bash
brain pull sentence-transformers/all-MiniLM-L6-v2
```

Lands in `~/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2`
and needs `config.json`, `model.safetensors` and `tokenizer.json`. Override the
location with `--encoder DIR` or `BRAIN_MINILM_DIR`.

This is a 22M-parameter 6-layer BERT. It is the *only* pretrained weight in the
system; everything this sample trains sits on top of it.

### 1.2 The dataset (~53 MB)

```bash
python3 -m pip install pyarrow     # the one dependency, used only by this script
./fetch-dataset.sh                 # 20000 conversations, about a minute
./fetch-dataset.sh --rows 4000     # a quicker start
```

Writes `testdata/decide/salesconv/{train,test}.jsonl` (gitignored). Override
with `--data DIR` or `BRAIN_TESTDATA`.

**Why a script rather than `make fetch/testdata`.** The published dataset is a
single **7.2 GB CSV** whose 3072 of 3088 columns are Azure OpenAI embeddings
this sample does not use - it encodes the text itself, locally. The script
reads the Hugging Face parquet conversion instead and projects the 9 columns it
wants over HTTP range requests, because parquet is columnar: **53 MB instead of
7.2 GB**. `pyarrow` is needed for that and for nothing else in brain.

### 1.3 A GPU, or not

Runs on the default device. `BRAIN_DEVICE=cpu` works and is slow; training on
CPU is not recommended.

---

## 2. Run it

```bash
make samples/decision/salesagent/build
make samples/decision/salesagent/run
```

That is the whole end-to-end: it trains, evaluates, saves the head, replays two
held-out conversations turn by turn, and drops into a prompt. Roughly ten
minutes on one Tesla P40 at the defaults.

The program is one expression:

```rust
ConversionPipeline::from_pretrained(encoder)
    .train(spec)        // supervised warm start -> policy phase -> router calibration
    .evaluate()         // accuracy, AUC-ROC, Brier, per-turn agreement, routing bands
    .save(head)         // write the trained head
    .replay(held_out)   // watch the probability move, turn by turn
    .tui()              // then type your own conversation at it
    .report()           // what every stage did
    .finish()?          // the one error site in the program
```

`train`, `evaluate`, `save`, `tui`, `report` and `finish` are the stages **every**
brain pipeline has. `replay` is this architecture's own, added through the same
seam. A failed stage stops the chain and the rest skip, keeping the original
cause, so seven stages have one error site.

A shorter run, to see it work before committing ten minutes:

```bash
make samples/decision/salesagent/run ARGS="--train 500 --warmup 1500 --policy 300 --eval 100"
```

---

## 3. Reading the output

### 3.1 Training

```
  phase 1/3: supervised warm start, 12000 steps
  step      0  loss 0.1185
  step   3000  loss 0.0128
  phase 2/3: policy gradient, 2000 steps
  phase 3/3: fitting the router on 200 held-out conversations
    signal weights: semantic 0.12, convergence 0.09, learned 0.79  (over 200 conversations)
```

**The loss is a KL divergence, not a cross-entropy.** The labels are
probabilities, so a label of 0.5 costs `ln 2 = 0.693` even from a perfect
model, and a raw cross-entropy never approaches zero. The label's own entropy
is subtracted, so **0 means the model matched the label** and the number is
comparable across runs. Watch the trend, not single steps - each step is one
conversation.

The **signal weights** are fitted, not configured, and they are worth reading:
they say which of the three confidence signals actually predicted whether the
model was right on held-out data. A weight of 0.00 means that signal carried no
information on this dataset. That is a result, not a failure.

### 3.2 Replay

```
  conversation 1  (11 turns, Non-profit, ended: CLOSED)
     1 customer  0.22 [####................]  label 0.10  retrieve Hey, so I was at that webinar on data viz...
     5 customer  0.19 [####................]  label 0.50  retrieve Oh, okay, but like... how hard is it to set up
     7 customer  0.18 [####................]  label 0.50  retrieve Mmm, yeah, but... the cost? $2000/month...
    10 sales_rep 0.50 [##########..........]  label 0.75  retrieve Totally fair! Maybe we can look into a trial...
    11 customer  0.51 [##########..........]  label 0.80  retrieve Hmm, a trial sounds nice.. I'll need to talk...
```

One line per turn: the model's `P(closes)` **given only the turns up to and
including this one**, the dataset's own recorded probability at that turn
(`label`), and the routing decision. The truncation is the point - a model
shown the closing turn can read the outcome off it, so a trajectory built from
full transcripts would score well and predict nothing.

Two conversations are replayed by default, one that closed and one that did
not, so the two trajectories can be read against each other.

### 3.3 The prompt

```
customer/rep> customer: we looked at your pricing and it is well over our budget
  turn  1  P(closes) 0.31 [#######.................]
           confidence 0.62 -> retrieve (pull comparable conversations before acting)
           semantic 0.71  convergence 0.55  learned 0.58
```

Type turns as `customer: ...` or `rep: ...`; bare text is taken as the
customer, since that is who moves a conversation. `reset` starts a new
conversation. An empty line or `^D` ends the session.

The conversation **accumulates** - each line adds a turn and the model rescores
the whole thing, which is what makes the number a trajectory rather than a
sentiment score.

### 3.4 The final report

```
--- pipeline ---
  model: conversion model, 14000 training steps so far, router weights [0.00, 0.47, 0.53]
  train: 14000 steps, final loss -0.6049, 573.2s (41 ms/step)
  evaluate: accuracy 0.707 over 400 items, AUC-ROC 0.798, Brier 0.202, per-turn MAE 0.199,
            act n 8, act acc 0.750, retrieve n 368, retrieve acc 0.709,
            escalate n 24, escalate acc 0.667
  save: out/sales-head.safetensors
  replay: 2 conversations
```

The policy phase's loss is NEGATIVE because it is a reward being maximized, not
an error being minimized - `-0.60` means the proper scoring rule is paying about
0.60 per turn out of a maximum of 1.0. Only the warm-start loss is a divergence
that should approach zero.

A measured run of exactly this command is in [section 6.1](#61-a-measured-run).

---

## 4. Using a trained model

`--save` writes the trained **head** - not the encoder, which is imported from
the released checkpoint and costs nothing to re-import. Reload it and skip
training entirely:

```bash
# score one conversation interactively, no training
make samples/decision/salesagent/run ARGS="--head out/sales-head.safetensors"

# replay held-out conversations with a trained model, no prompt
make samples/decision/salesagent/run ARGS="--head out/sales-head.safetensors --replay 6 --batch"
```

With `--head` the chain skips `train`, `evaluate` and `save` and goes straight
to `replay` and `tui`. The router's projection and confidence network are
fitted during training, so a reloaded model falls back to the unweighted
signals - the probability is fully trained, the confidence is coarser.

From your own code the surface is three calls:

```rust
use brain::{ConversionPipeline, SalesMessage};

let mut pipe = ConversionPipeline::builder(encoder_dir)
    .head("out/sales-head.safetensors")
    .load()?;

let turns = vec![
    SalesMessage { speaker: "customer".into(), text: "this is over budget".into() },
    SalesMessage { speaker: "sales_rep".into(), text: "we have a smaller tier".into() },
];

let p = pipe.probability(&turns)?;          // just the number
let v = pipe.verdict(&turns)?;              // number + confidence + route
let traj = pipe.trajectory(&conversation)?; // P after every turn
```

`probability` is the cheap call. `verdict` additionally reads every layer's
hidden states back off the device to compute the confidence signals, so use it
when routing a decision, not inside a loop.

---

## 5. How it works

### 5.1 The shape of the model

```text
conversation text ──► WordPiece ──► windows ─┐
                                              ├─► encoder ─► head ─► score ─► sigmoid ─► P(closes)
proposition text  ──► WordPiece ──► one slot ─┘
```

* **The encoder** is the imported MiniLM, run over the conversation cut into
  overlapping 256-token windows. Windows make cost grow *linearly* with
  conversation length instead of quadratically, and let a conversation longer
  than the position table still be one state.
* **The head** is trained from scratch: the proposition attends over every
  state token (cross-attention), and a linear readout turns the result into one
  score.
* **The answer** is `sigmoid(score)`.

Both halves are fine-tuned, at different rates - the encoder arrives pretrained
and the head does not, so one rate would either leave the head too slow to
learn or move the encoder fast enough to forget what it was imported for.

### 5.2 Why ONE slot and a logistic, not two slots and a softmax

This is the single most important design decision here, and it was found the
hard way: the first implementation scored two options, `"...deal [SEP] the deal
closes"` against `"...deal [SEP] the deal is lost"`, and **evaluated at
chance - 0.502 accuracy, Brier 0.251, every turn pinned at 0.50.**

The head uses each option's slot as its attention **query**. Two slots that
differ by a word or two are nearly parallel - measured on the released
checkpoint, cosine **0.989**. Nearly-parallel queries attend to the state
almost identically, so both scores carry the same state information, and a
softmax sees only their *difference*:

```text
                     yes      no    yes-no   <- a softmax only sees this column
obvious close     0.5740  0.6161   -0.0421
obvious loss      0.6482  0.6969   -0.0487
undecided         0.3587  0.4008   -0.0421

absolute score spread across states: 0.2895   <- the state signal IS there
difference spread:                   0.0066   <- and the softmax cancels it
```

The encoder was never the problem - it separates those three conversations
cleanly (embedding cosine 0.37). The signal was present all along and the
output layer was subtracting it out as **common mode**. No amount of
learning-rate, step-budget or sampling tuning could have reached it.

So a `Noul` scores **one slot - the proposition - and reads a logistic off it**.
The absolute score survives. On an untrained head this takes the spread of
`P` over those three conversations from **0.0017 to 0.0679, forty-fold**.

This is also the *more* faithful reading of the paper, which describes "a policy
network that estimates conversion probability based on the current state" - a
scalar, not a two-way softmax over label text. And it costs nothing in
generality: the proposition is still supplied per request, so the model still
answers questions it was not trained on.

A `Choice` question is unaffected and keeps its softmax, because its options are
genuinely different text - which is what gives a softmax something to work with.
`samples/decision/triage` is that path, at 0.96 accuracy on BANKING77.

The measurement above is pinned by
`decide::policy::tests::a_softmax_over_two_slots_cancels_what_a_logistic_keeps`,
so it cannot silently regress.

### 5.3 SalesRLAgent: the sequential decision problem

The paper's formulation (§IV.A):

```text
state  s_t   the conversation up to turn t
action a_t   "conversion probability estimates" - the committed probability
reward r_t   the ACCURACY of that estimate
```

**The reward must be a proper scoring rule.** "Accuracy of the prediction"
admits two readings, and only one works. Read it as *pay 1 when the model calls
the outcome right*, sample a binary action and run a policy gradient: the
expected ascent direction is `(2q-1) * p * (1-p)` for an outcome rate `q`,
which has the sign of `q - 1/2` at **every** `p`. The objective is maximized by
the **mode**. A state whose conversations close 70% of the time trains to 1.0,
not 0.7 - accurate, and completely uncalibrated.

Read it instead as a proper scoring rule on the committed probability,
`r = 1 - (p - y)^2`, and the expected reward is maximized exactly at `p = q`.
That is what is implemented. `the_policy_optimum_is_the_conditional_rate` is
the test that separates the two readings, and it is the test that caught the
first one.

**What is left of PPO, honestly.** The conversation is *replayed* - the action
cannot change what the customer says next. There is no state distribution to
shift and every turn's return is its own reward, so the sequential objective
**reduces** to a trust-region-regularized proper scoring rule. That reduction
is a consequence of the environment not responding, not a simplification chosen
here. What survives is exactly the two properties the paper asks for:

* **Conservative policy updates** - the committed probability may not move more
  than `clip` (0.2) from the one that collected the batch. The region is
  one-sided, so a policy that has drifted can still come back.
* **Strong policy regularization** - an entropy term resisting collapse onto a
  confident answer before the conversation has said anything.

Plus one genuinely sequential piece: a **discount** `gamma^k` weighting a turn
`k` steps from the close, so a probability committed ten turns out is held to a
looser standard than the one committed at the close.

**Training runs in three phases**, following the paper's own recipe (§IV.D):

| phase | what | objective |
|---|---|---|
| 1. warm start | fit the per-turn probability the dataset recorded | binary cross-entropy against a **soft** target |
| 2. policy | the sequential objective above | clipped proper scoring rule + entropy + discount |
| 3. calibration | fit the router on held-out conversations | ridge regression + MSE |

Phase 1 fits a *soft* target because the label is a probability, not a class.
Rounding it to 0/1 would throw away exactly the calibration the trajectory
carries, and phase 2 would have to rediscover it from a scalar reward.

Two of the paper's six training measures change what the model *sees* rather
than how it is updated, and both are implemented:

* **Curriculum** - shortest conversations first, widening as the run
  progresses. A short conversation's turns are nearly all decisive; a 30-turn
  dialogue spends its first ten turns on small talk the outcome does not depend
  on, and starting there teaches the base rate first.
* **Outcome-balanced batches** - draws alternate closed and lost. A model
  rewarded on a skewed set collects most of its reward by learning the base
  rate, and the gradient that teaches it to *read* is the smaller part.

To which this sample adds one more, because it was needed: **turns are sampled
weighted toward the end** of the conversation (by the same `gamma`). A uniform
sampler spends almost every step on ambiguous middles - the mean label across
all turns is 0.43 against 0.55 over the last two - so the model learns to emit
the base rate and stops. That was a measured failure, not a precaution.

### 5.4 Confidence-Aware Routing

Three signals, combined and thresholded:

```text
C_sem      = cos(P(h_final), e_ref)                 (1)  does the decision
                                                         representation still
                                                         align with the text
C_conv     = Var(h_1..L/2) / (Var(h_L/2..L) + eps)  (2)  did the stack settle
C_learned  = phi(h_final)                           (3)  a fitted estimate
C_overall  = w1 C_sem + w2 C_conv + w3 C_learned    (4)

A(C) = act       if C >= 0.75      act on the number
       retrieve  if C >= 0.55      pull comparable conversations first
       escalate  if C >= 0.35      second opinion from a larger model
       human     otherwise         hand it to a person          (5)
```

The thresholds are the paper's own. The weights `w1..w3` are fitted on held-out
conversations by ridge regression against whether the model's decision turned
out right, then renormalized to sum to 1 - the paper's thresholds are stated
for a score that does.

**Calibration and evaluation use disjoint data.** The router is fitted on the
first half of the held-out split and `evaluate` scores the second. A confidence
estimator fitted to the split it is judged on is calibrated against the one
number it must not see.

This paper's reference embedding model is `all-MiniLM-L6-v2`, which is the
encoder this sample runs. So the roles shift down one level: the "internal
representation" is the head's decision-side output and the "reference" is the
encoder's own mean-pooled sentence embedding of the conversation. The question
(1) asks is unchanged - has the part that decides drifted away from the part
that reads.

### 5.5 Where the data comes from

[`DeepMostInnovations/saas-sales-conversations`][ds] - 100,000 synthetic B2B
SaaS sales dialogues, Apache-2.0, published alongside the SalesRLAgent paper.
Each carries a binary conversion outcome and a conversion probability recorded
at every turn. Conversations run 8 to 30 turns, median 12, and the outcomes are
close to balanced (about 50% closed).

**What the labels are.** Both the conversation and its trajectory were generated
together by GPT-4o. The trajectory is a self-consistent narration of the
dialogue it accompanies, **not a measurement of a real buyer**. A model that
matches it has learned to read these conversations the way the generator wrote
them - a real, checkable task, and not evidence about real-world conversion.
The two should never be quoted as if they were the same number.

[ds]: https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations
[p1]: https://arxiv.org/abs/2503.23303
[p2]: https://arxiv.org/abs/2510.01237

---

## 6. Interpreting the numbers

`evaluate` scores the **final turn** of each held-out conversation - the point
the whole conversation is about, and the only one whose right answer is not in
dispute.

### 6.1 A measured run

`--train 6000 --eval 600 --warmup 12000 --policy 2000`, one Tesla P40, 573 s of
training, scored on 400 held-out conversations. The baseline is a hashed
bag-of-words logistic regression fitted on the same 6000 training conversations
and scored on the same 400:

| | accuracy | AUC-ROC | Brier |
|---|---:|---:|---:|
| constant predictor | 0.500 | 0.500 | 0.250 |
| bag-of-words logistic | **0.730** | **0.805** | 0.257 |
| **salesagent** | 0.707 | 0.798 | **0.202** |

Read that honestly: **the model does not beat a bag-of-words baseline at picking
the winner.** It matches it at ranking (AUC 0.798 against 0.805) and loses
slightly on accuracy.

What it does clearly better is the thing it exists to do. Brier 0.202 against
0.257 is a large calibration gap: the lexical model ranks fine and then states
its answers far too confidently, while this one's outputs are usable *as
probabilities*. A Brier of 0.25 is what a constant 0.5 scores, so the baseline
is barely better than uninformative on that axis despite being accurate.

For reference, the first version of this sample - before the fix in
[5.2](#52-why-one-slot-and-a-logistic-not-two-slots-and-a-softmax) - scored
**accuracy 0.502, AUC 0.513, Brier 0.251**: a constant, dressed as a model.

The routing bands came out ordered the right way but with almost no spread:

| band | n | accuracy |
|---|---:|---:|
| act | 8 | 0.750 |
| retrieve | 368 | 0.709 |
| escalate | 24 | 0.667 |

Accuracy falls monotonically from `act` to `escalate`, which is the routing
paper's claim holding. But 92% of conversations land in one band, so the router
is not doing much work here - the confidence score is compressed into a narrow
range around 0.6. The ordering is real; the *coverage* is not yet useful, and
the `act`/`escalate` accuracies rest on 8 and 24 conversations respectively, so
they are indicative at best.

Per-turn MAE of 0.199 says the trajectory tracks the dataset's own curve to
about a fifth of the scale - visible movement in the right direction, but noisy
turn to turn, as the replay above shows.

### 6.2 What each metric means

| metric | what it means | reference point |
|---|---|---|
| **accuracy** | fraction where `P >= 0.5` matched the outcome | **0.50** is chance (the set is balanced). **0.72** is a bag-of-words logistic regression on the same split - the bar a real model must clear |
| **AUC-ROC** | ranking quality, threshold-free | 0.50 is chance. Moves independently of accuracy; a model can rank well and still be badly thresholded |
| **Brier** | mean squared error of the probability | **0.25** is a constant 0.5 - the value a model that has learned nothing returns. Lower is better; it penalizes overconfidence and underconfidence alike |
| **per-turn MAE** | mean absolute gap to the dataset's own recorded probability, sampled every third turn | this is the *trajectory* measure. Accuracy only ever asks the last turn |
| **act/retrieve/escalate/human n** | how many conversations landed in each routing band | coverage. A router that sends everything to one band is not routing |
| **act/retrieve/escalate/human acc** | accuracy **within** each band | **the number that says the confidence works.** It should fall monotonically from `act` to `human` - that is the whole claim of the routing paper. If the bands have equal accuracy, the confidence signal is not a confidence signal |

Two of those reference points are worth stating plainly because they are easy
to fool yourself with:

* **Brier 0.25 and accuracy 0.50 together mean the model is emitting a
  constant.** That is the failure mode section 5.2 describes, and it looks like
  "needs more training" until you check the spread.
* **Accuracy above chance is not the bar.** 0.72 from bag-of-words is, on this
  dataset, with the last two turns carrying essentially all of the signal.

---

## 7. Faithful, adapted, not done

| | |
|---|---|
| **Faithful** | the MDP formulation and its action space; supervised-then-RL training; conservative clipped updates with an entropy term; a discount over turns; curriculum by conversation length; outcome-balanced batches; turn-by-turn trajectory output; all three confidence signals; equation (5) and its published thresholds; calibration on held-out data |
| **Adapted** | **embeddings are local.** The paper uses Azure OpenAI's 3072-d model; this runs a 384-d MiniLM on the machine in front of you. The paper's own ablation puts 7.7 accuracy points on that choice, and it is the one deviation that cannot be closed without a cloud API |
| | **no value network.** With a deterministic action and a differentiable reward, the policy gradient IS the reward's gradient - there is no return to estimate and nothing for a critic to reduce the variance of |
| | **`P` is linear**, fitted by ridge regression, where the paper's is a deep network - trained, in their setup, on 72 examples. At that calibration size a closed-form fit is the honest choice and has no schedule to get wrong |
| | **`C_conv` is squashed** by `r/(1+r)`. Equation (2) is an unbounded variance ratio and (4) compares its sum against thresholds in `[0, 1]`; the paper does not say how it is bounded |
| | **`phi` is two layers** with the paper's stated MSE + L2, without the batch normalization and dropout it also lists |
| | **turn sampling is end-weighted** - an addition, not in the paper, and needed: see 5.3 |
| **Not done** | the 1.2M-conversation private training set (this uses the 100k public one); ensembles; adversarial counter-examples; the orchestration layer, vector search, CRM connectors and 8-bit quantization, which are product surface rather than method |
| **Known gap** | each window restarts its position ids at zero, so the head cannot tell window 1 from window 3 except by content. On a task where recency decides, that is a real handicap; conversations average ~318 tokens, so 2-3 windows |
| | the head gathers each slot's `[CLS]`, but this checkpoint is a sentence-transformer whose sentence representation is the **mean** - its `[CLS]` was never trained to be one. Measured but not yet changed |
| | **it does not beat bag-of-words on accuracy** (0.707 against 0.730). It wins on calibration, not on picking the winner |
| | **the router barely routes** - 92% of conversations land in one band. The band ordering is right, the spread is not useful yet |
| | the per-turn trajectory is noisy (MAE 0.199): it moves in the right direction over a conversation but jumps turn to turn |

---

## 8. Command reference

```text
salesagent [--encoder DIR] [--data DIR] [--head FILE] [--save FILE]
           [--train N] [--eval N] [--warmup N] [--policy N]
           [--replay N] [--batch]
```

| flag | default | meaning |
|---|---|---|
| `--encoder DIR` | `~/.local/share/brain/models/sentence-transformers/all-MiniLM-L6-v2` | encoder checkpoint directory. Also `BRAIN_MINILM_DIR` |
| `--data DIR` | `testdata/decide/salesconv` | dataset directory. Also `BRAIN_TESTDATA` |
| `--head FILE` | none | load trained head weights and **skip training** |
| `--save FILE` | `out/sales-head.safetensors` | where to write the trained head |
| `--train N` | 4000 | training conversations to use |
| `--eval N` | 400 | held-out conversations; half calibrate the router, half are scored |
| `--warmup N` | 3000 | supervised steps (phase 1) |
| `--policy N` | 1000 | policy-gradient steps (phase 2) |
| `--replay N` | 2 | conversations to replay turn by turn |
| `--batch` | off | skip the interactive stage, for scripted runs |

```bash
./fetch-dataset.sh [--rows N] [--force]
```

| flag | default | meaning |
|---|---|---|
| `--rows N` | 20000 | conversations to fetch. One in ten is held out |
| `--force` | off | refetch even if the output already looks complete |

## Cost

One brain dependency (`brain`, feature `decision`) and 26 crates in the
closure - the fewest of any sample here, tied with `decision/triage`, because it
names one surface and the SDK links only what that surface needs.
`make check/samples` enforces the budget.

---

Swedish Embedded AB builds conversation intelligence that runs on the customer's
own hardware - scoring a live dialogue, in milliseconds, without shipping it to a
third party. If your team needs judgment in the loop without handing control flow
to a text generator, you can procure our services by sending an email to
info@swedishembedded.com.
