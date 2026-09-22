<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 133. A softmax over near-identical options cancels the signal as common mode

A decision model scoring a yes/no proposition evaluated at chance - accuracy
0.502 against 0.500, Brier 0.251 against the 0.250 a constant 0.5 scores. It
had trained for 14000 steps without complaint.

The head uses each option's slot representation as its attention QUERY. The two
slots were `"<instructions> [SEP] the deal closes"` and
`"<instructions> [SEP] the deal is lost"` - 12 of 14 tokens shared, and on the
released checkpoint their representations sit at **cosine 0.989**. Two
nearly-parallel queries attend to the state almost identically, so both scores
carry the same state information, and a softmax sees only their DIFFERENCE:

```text
                     yes      no    yes-no   <- a softmax only sees this
obvious close     0.5740  0.6161   -0.0421
obvious loss      0.6482  0.6969   -0.0487
undecided         0.3587  0.4008   -0.0421

absolute score spread across states: 0.2895   <- the signal IS there
difference spread:                   0.0066   <- and the softmax cancels it
```

The encoder was never at fault: it separated those three conversations cleanly
(embedding cosine 0.37). The signal was present the whole time and the OUTPUT
LAYER subtracted it out.

THE FIX. Score ONE slot - the proposition - and read a logistic off it. The
absolute score survives. On an untrained head that took the spread of `P` over
those three conversations from 0.0017 to 0.0679, forty-fold, before any
retraining at all.

WHAT MADE IT EXPENSIVE. Every symptom points somewhere else. Chance accuracy
with a falling loss reads as undertraining, so the first three attempts were a
larger step budget, an end-weighted turn sampler, and a longer schedule - each
defensible, each worth a training run, none able to touch it. The model could
also OVERFIT ten conversations to zero loss, which rules out a wiring bug and
argues for "needs more data" - the exact wrong conclusion.

WHAT FOUND IT. Not the loss curve. A probe that printed, for three deliberately
extreme inputs: the encoder's own embedding cosine between them, the two option
queries' cosine, and BOTH raw scores rather than the probability. The scores
are where it is visible; the probability is where it is hidden, because the
probability is the thing that has already done the subtracting.

THE RULE. A softmax is a statement that the options COMPETE. It is only
informative to the extent they differ - and for a fixed two-outcome question
(`yes/no`, `true/false`, `pass/fail`) they differ by a word, while the quantity
being asked about is common to both. Use a softmax where the options are
genuinely different text and a caller could supply more of them; use a scalar
and a logistic where the option set is fixed at two. A `Choice` over 8 distinct
intent names is the first case and works (0.96 accuracy on the same head); a
`Noul` is the second.

And when a model emits a near-constant, measure the spread of the RAW SCORES
before touching the training loop. If the scores move and the probability does
not, the defect is between them, and no amount of training will cross it.
