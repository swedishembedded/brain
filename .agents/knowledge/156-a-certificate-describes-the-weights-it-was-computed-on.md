<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 156. A certificate describes the weights it was computed on, and nothing else

`samples/learning/rlcd` ran its stages in this order:

```text
train belief -> evaluate -> SAVE -> witness audit -> train VOI policy -> answer questions
```

Every stage is individually right. The order is not. Stage 5 trains through
`Decide`, whose head scores (state, option-text) pairs, so a second question
asked of the same state goes through the same parameters as the first. The
policy-gradient run therefore moves the belief - and it ran after everything
that measured the belief.

So three different models wore one name: the one the report described, the
one written to disk, and the one that answered. Measured on the sample's own
400/600-step run, before and after 600 policy steps:

| | before stage 5 | after | delta |
|---|---|---|---|
| soft ECE | 0.0708 | 0.0940 | +33% |
| KL(oracle \|\| model) | 0.0552 | 0.0649 | +18% |
| regret[safety-critical] | 0.1404 | 0.2000 | **+42%** |

The witness audit is the part that stings. It reported "no decision failures
found" and that sentence was true - of weights that no longer existed by the
time anything used them. Pushed further, the certificate becomes actively
false: the policy trains ONLY on the "no evidence yet" phrasings, so those
are the states it rewrites hardest, and the belief there walks steadily away
from the oracle's 0.20.

| VOI steps | P(faulty \| no evidence) |
|---|---|
| 0 | 0.188 |
| 600 | 0.127 |
| 2400 | 0.106 |
| 6000 | 0.119 |
| 12000 | **0.040** |

The safety-critical block/release boundary is at `C_FP/(C_FP+C_FN)` = 1/11 =
0.0909. At 12k steps the model RELEASES a device whose true fault
probability is 0.20 - expected cost 2.0 against the optimal 0.8, a regret of
1.2 per decision - at exactly the evidence point a passing audit had named.

THE RULE. **Order the stages so that the last thing that writes weights
happens before the last thing that measures them.** Evaluate, audit, and
save AFTER every training stage, not after the one you were thinking about.
Where a later stage must run, re-measure and print the delta rather than
reordering the report: a cost you can see is a cost you can decide about,
and a shared-parameter multitask setup is a legitimate design whose price
simply has to be on the record.

The weaker tell, available without running anything: a struct field whose
doc says two things are separate while the code shows one object. This
pipeline's own `voi_question` was documented as "two different heads
answering two different questions", beside a `self.model.train_step_with`
that names the single head both go through. When a comment asserts a
separation, check that something enforces it.

Related: [[151-a-published-number-whose-command-was-never-recorded-cannot-be]]
- there the number could not be tied back to what produced it; here it can,
and what produced it no longer exists.
