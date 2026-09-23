<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 151. A published number whose command was never recorded cannot be defended or discarded

`samples/decision/rubiks/README.md` published a solve-rate table - 100/100/98.5/76/45/21
across scramble depths 1 to 6 - directly beneath the training command a reader
was told produced it. Both were real. They were not related.

Re-running the documented command and measuring the model it produced gave
96/78/45/16, and the run took an hour and a half rather than the twenty
minutes the card claimed. The published numbers had come from a sibling model
trained hours earlier with different settings that nobody wrote down. No log
survived, the checkpoint's `config.json` recorded architecture but nothing
about the fit, and shell history had rotated. The recipe was simply gone.

What makes this worse than an ordinary stale number is the **asymmetry it
creates for the next reader**. Confronted with 98.5% in the card and 96% on
their own machine, they cannot tell which of four things happened: their run
was unlucky, their setup differs, the card drifted, or the card was always
wrong. Every one of those has a different response, and the evidence needed to
choose has been destroyed. A number with no recipe cannot be defended, and it
cannot be discarded either - so it stays, and it keeps costing.

Note that nothing here was dishonest and no step was skipped. The numbers were
measured carefully, on the right model, with a tool that reports honestly. The
defect is entirely in what was *not* captured at the moment it was free to
capture.

THE RULE. **A number is publishable when the exact command that produces it is
published beside it and has been run to check.** Not "a command like this one".
The one that was run.

Two cheap defences, both of which this sample now has:

1. **Re-run the documented command before shipping the doc.** It is the only
   check that catches a recipe and a result parting company, and it catches
   it at the moment the recipe is still recoverable. Doing this found four
   further wrong claims in the same file - a throughput figure off by 3x, a
   counterfactual search time that was never measured at all, a crate budget
   that had drifted, and five example commands naming a binary that does not
   exist.
2. **Write the fit into the artifact.** A checkpoint that records the steps,
   batch, example count and seed that made it turns "which run was this?" from
   an archaeology problem into a field lookup. Architecture alone is not
   provenance: every model in this family has the same architecture.

Related: [[058-a-measured-number-in-a-doc-comment-outlives-the-tree-it-measured]]
is the same class one level down - a number whose *code version* was not
recorded. [[145-a-held-out-split-drawn-like-the-training-set-cannot-see-a]] is
the same sample's other measurement defect, and the two compound: a held-out
number that cannot see the skew, published without the recipe that produced
the skew.
