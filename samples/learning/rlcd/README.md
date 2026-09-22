<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# rlcd - RLCD end to end: a world, a calibrated model, and an audited decision

RLCD here means Reinforcement Learning for Calibrated Decisions: a model
should emit a **calibrated probability distribution**, and the **action**
taken on it should be a separate, explicit function of that probability and
the actual cost of being wrong - never a policy trained directly on outcome
reward, which optimizes for the mode of a distribution rather than its rate.
This sample is the smallest complete demonstration of that pipeline, built
around one executable Bayesian world:

```rust
RlcdPipeline::from_pretrained(encoder)
    .train(spec)       // fit the EXACT oracle posterior, not a gold label
    .evaluate()         // calibration AND decision regret, not accuracy alone
    .save(head)
    .finish()?
```

Same stage chain every brain pipeline has. What makes this RLCD rather than
an ordinary classifier demo happens around that chain:

1. **The world is executable and exact.** A latent device fault at
   `P(fault) = 0.2`; a diagnostic test with `P(+|fault) = 0.8` and
   `P(+|healthy) = 0.1`. Its own numbers are checked BEFORE training starts,
   against `brain::check_information_refinement` - the identity
   `prior = E[posterior | evidence]` a Bayesian world's own probability model
   must satisfy. This is the oracle-honesty check `brain-rlcd`'s `atlas`
   module exists for, not a formality: an oracle that silently conditioned a
   posterior on hidden state can fail this even when each individual
   posterior looks plausible on its own.
2. **Training fits the exact posterior**, not a one-hot label:
   `P(fault | +) = 2/3`, `P(fault | -) = 1/19`, computed by closed-form Bayes'
   rule and never rounded to a class. Each evidence state is rendered as
   several distinct phrasings (see `world.rs`), so held-out evaluation tests
   generalization to unseen wording of the SAME evidence, not memorization of
   three fixed strings.
3. **Evaluation reports calibration AND decision regret** - ECE, NLL, Brier
   on the probability head, plus mean regret against TWO cost matrices the
   model never trained under (`symmetric`: false positives and false
   negatives cost the same; `safety-critical`: releasing a fault costs 10x
   blocking a healthy device). Regret is what tells a learned BELIEF apart
   from a learned POLICY: it is computed against the oracle's own exact
   posterior for each held-out example, never against the model's own belief,
   which would make every model appear to have zero regret by construction.
4. **A witness audit** (`brain::witness_search`) exhaustively checks the
   trained model against the oracle at the world's own three canonical
   evidence points, under a safety-critical cost matrix, and reports any case
   where the model's induced action disagrees with the Bayes-optimal one -
   the action, not the probability: two beliefs that differ numerically but
   agree on every action of interest are not a decision failure.

## Run it

```bash
brain pull sentence-transformers/all-MiniLM-L6-v2   # the encoder, once

make samples/learning/rlcd/build
make samples/learning/rlcd/run ARGS="--steps 400"
```

That checks the world, trains, evaluates, saves the head, audits for
witnesses, and drops into a prompt. Reuse trained weights without retraining:

```bash
make samples/learning/rlcd/run ARGS="--head out/rlcd-head.safetensors --ask 'The diagnostic test came back positive.'"
```

## What it shows

**Belief and action are two different questions, answered two different
ways.** The probability head is trained with a proper scoring rule
(`rlcd::scoring::decision_loss_soft`, focal + Brier). The action is a
deterministic, non-learned `argmin` over expected cost
(`rlcd::cost::bayes_action`) - no policy gradient anywhere in this sample,
because none is needed for a static probability head with known targets.

**Held-out evaluation is genuine generalization, not memorization.** The
eval split holds out DIFFERENT phrasings of each evidence class, never the
canonical training wording. Asked a phrasing it never trained on ("The
inspection turned up a positive fault indicator."), a real trained run in
this repo returned `P(healthy) = 0.374, P(faulty) = 0.626` against the exact
answer `[1/3, 2/3] = [0.333, 0.667]` - close, on wording the model had not
seen, which is the claim a held-out split is for.

**The encoder is frozen on purpose.** `RlcdSpec::freeze_encoder(true)` - this
world has dozens of training phrasings, not BANKING77's thousands, and a full
fine-tune of a 22.7M-parameter encoder on dozens of examples would memorize
rather than generalize. It is also the only way `Flow::save`'s head-only
checkpoint format is writable at all: `Decide::save_head` refuses to write
one over a moved encoder, since loading it back would silently reattach the
head to the wrong base.

## A measured run

400 steps, encoder frozen, one CPU-class run (`decide`'s MiniLM-L6 encoder,
6 layers, d=384):

```
rlcd: oracle self-consistency: OK (prior = E[posterior | evidence], exact)
rlcd: 18 training phrasings, 6 held out for evaluation (unseen wording, same evidence)
  step      0  loss 0.6387
  step    100  loss 0.4995
  step    200  loss 0.2507
  step    300  loss 0.1045

train:    400 steps, final loss 0.2136, 14.9s (37 ms/step)
evaluate: accuracy 0.833 over 6 items, ECE 0.554, NLL 0.382, Brier 0.235,
          regret[symmetric] 0.056, regret[safety-critical] 0.140

witness audit (safety-critical cost matrix: C_FP=1, C_FN=10):
  no decision failures found: the model's induced action agrees with the
  oracle at every canonical evidence point

ask: "Diagnostics: CLEAR."  (a TRAINING phrasing)
  healthy 0.948  faulty 0.052                    exact answer: 0.947 / 0.053

ask: "The inspection turned up a positive fault indicator."  (HELD OUT)
  healthy 0.374  faulty 0.626                    exact answer: 0.333 / 0.667
```

## What is not claimed

- **This is a toy world, not a real diagnostic model.** 18 training examples
  across 3 evidence classes is enough to demonstrate the pipeline, not enough
  to claim a usable fault classifier. ECE of 0.554 on a 6-item eval split is
  a small-sample number, reported honestly rather than smoothed over.
- **Synthetic correctness is conditional on the world model.** The oracle is
  exact WITHIN the stated Bayesian world (`P(fault)=0.2`, the two likelihoods)
  and says nothing about any real device.
- **No learned evidence-acquisition policy.** Deciding WHETHER to run the
  diagnostic before acting (`rlcd::cost::voi`, gated in `brain-rlcd`'s own
  test suite against this exact world) is closed-form here, not demonstrated
  as a trained policy in this sample - `rlcd::atlas`'s worked-example numbers
  cover that case, this sample's own scope stops at stages 1-2 (calibrated
  belief, then closed-form action).
- **The Laya decision backbone cannot train through this pipeline yet.**
  `RlcdPipeline` trains through `crates/decide` only; passing a Laya
  checkpoint's directory fails fast with a clear message rather than a
  confusing one three layers down.

## Cost

29 brain crates, matching the other `decision`-surface samples: it names one
surface and the SDK links only what that surface needs.

---

Swedish Embedded AB builds decision systems that report a calibrated
probability AND act on it according to the actual cost of being wrong, not
merely the most likely label - audited on calibration and decision regret,
not accuracy alone. If your team needs judgment under uncertainty that is
measured rather than assumed, you can procure our services by sending an
email to info@swedishembedded.com.
