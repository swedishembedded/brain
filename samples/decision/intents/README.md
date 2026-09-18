<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# intents - does a decision model actually READ its options?

This is the claim the whole decision surface rests on: **the options arrive at
run time, as text, and the model scores them by what they mean rather than by
where they sit.** An agent choosing among moves a game invents each tick, a
router choosing among services registered this morning, a triage step choosing
among categories somebody added yesterday - every one of them is worth exactly
as much as that claim is true.

It is also the claim that is easiest to pass without meaning to. Train a model
on a fixed list of 77 intents and it can reach a high number having learned a
77-way classifier in a costume: position alone identifies the answer, and
nothing ever forces it to look at the words.

So this run does three things a conventional intent benchmark does not.

1. **Every example sees a random SUBSET of the options**, always containing the
   right one, at a random position. There is no stable index to learn.
2. **Some intents are held back entirely.** The model trains on none of their
   examples and is then asked to pick them, by name, out of a list. A model that
   learned the option text can do this. A disguised classifier scores chance.
3. **A shuffled-state control.** The same held-out examples are re-scored with
   the input text replaced by a different example's. Accuracy must collapse
   toward chance - if it does not, the model is reading the option list alone,
   the answer is leaking from the question, and the headline number means
   nothing.

The third is the cheapest check here and the one nobody runs.

## Running it

```bash
make fetch/testdata                      # BANKING77 (PolyAI, CC-BY-4.0)
make samples/decision/intents/build
make samples/decision/intents/run ARGS="--encoder DIR --unseen 12 --steps 8000"
```

Needs a sentence encoder (`brain pull sentence-transformers/all-MiniLM-L6-v2`)
and nothing else. `--device`, `--encoder`, `--head`, `--save` and `--seed` come
from `brain::options`, shared with the `brain` binary and every other sample.

## The dataset

BANKING77: 13,083 customer-service utterances over 77 fine-grained banking
intents (Casanueva, Temcinas, Gerz, Henderson and Vulic, *Efficient Intent
Detection with Dual Sentence Encoders*, NLP4ConvAI @ ACL 2020, CC-BY-4.0).

The released intent names are the only option text there is - the dataset ships
no descriptions - so what the model reads for an option is `card_arrival`
humanized to `card arrival` and nothing more. That is a deliberately thin
signal, and it is the honest one: a real deployment names its actions, it does
not describe them.

## Results

MiniLM-L6-v2 encoder (frozen), 8,000 steps on 65 intents, 12 held back,
400 test utterances scored per row:

| | accuracy | chance | examples |
|---|---:|---:|---:|
| intents trained on | **92.5%** | 10.2% | 400 |
| intents **never trained on** | **25.0%** | 10.2% | 400 |
| same, state shuffled (control) | 6.0% | 10.2% | 400 |

**The claim holds.** The model picks intents it has never trained on at
2.45x chance, from nothing but a humanized name like `card arrival` in a list
- there is no index for it to have learned, because it never saw one. And
shuffling the input collapses it to 6.0%, below chance, so it is reading the
state rather than picking on some property of the option list.

The gap between 92.5% and 25.0% is the honest price of never having seen an
intent, and it is worth stating plainly rather than quoting the first number
alone. An untrained head scores 16.2% on the same held-out set - the pretrained
encoder already puts some of this in for free, which is another reason the
control matters.

## What this does NOT measure

- **Not a BANKING77 leaderboard number.** Published baselines (RoBERTa-base
  93.86%, ModernBERT-base 93.99%) always score against all 77 intents at once
  and are a different, easier task than a sampled subset with held-out intents.
  Quoting these numbers against those is meaningless in both directions.
- **Not calibration.** Whether the probabilities are *honest* - Brier score,
  ECE, failure-AUROC - is a separate question this run does not ask.
- **Not transfer to a different domain.** Held-out INTENTS are still banking
  intents. A held-out domain is a stronger claim and a different experiment.

---

Swedish Embedded AB builds the evaluation that tells a customer whether a model
does what its architecture claims, rather than whether it scores well on the
benchmark it was fitted to. If your team needs that, you can procure our
services by sending an email to info@swedishembedded.com.
