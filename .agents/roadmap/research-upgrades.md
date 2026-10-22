# research-to-stack - brain share

Scope: brain's share of the cross-project research upgrades.

**Status: planned. Nothing below has been started.** Update this file in the same commit that
finishes or changes an item. Evidence and citations for each item are in the cross-project
research note (items U3, U5, U6, U11, U13, U15); this file is the brain share. Brain provides
mechanism and arithmetic; it does not decide what is worth learning.

## Goal

Make the model side of a verified-task learning loop honest and cheap: evaluation arithmetic that
is correct at small n, training defaults that match what the literature measured for code
adapters, and the per-token controls that agent trajectories need.

## Left to do

- [ ] RS-B1 (S) Evaluation arithmetic for small exams in the `stats` crate.
  - The unbiased pass@k estimator from n samples, exact (Clopper-Pearson) and Wilson intervals
    for a proportion, a family-clustered interval, and a paired exact sign test power function
    (probability of rejection given N tasks, discordant fraction, win share).
  - Callers pass pairs and counts; the crate holds no task vocabulary.
  - Acceptance: golden tests with hand-computed values (n=20 discordant needs 15 wins at
    one-sided 0.05; N=60, discordance 0.4, win share 0.75 gives power 0.76 within 0.01); the
    estimator equals the closed form for n=k and for k=1; interval coverage simulation at
    n=10, 30, 100 within 2 points of nominal for Wilson and exact. `make test`.
- [ ] RS-B2 (S) LoRA defaults audit and a rank/target report.
  - Confirm which linears each LoRA-capable family adapts (attention and MLP projections) and
    print them in the adapter card; add a `finetune` check that refuses attention-only targets
    unless asked. Keep alpha default at twice the rank.
  - Do not change the default rank (8) in this item; the sweep in the research note (rank 16, 64,
    256 at a fixed learning-rate rule) decides it and is run by the caller.
  - Acceptance: a test that a freshly trained adapter's card lists the MLP projections; a test
    that an attention-only request is refused with a typed error. `make test`.
- [ ] RS-B3 (M) Per-token loss weights and masks for trajectories whose early part was
  truncated or compacted (extends the TokenObjectives work already planned).
  - A trajectory record may mark token spans as context-only (present in the sequence, zero
    weight). Segment-level selection by the caller is expressed the same way.
  - Acceptance: tiny-model CPU gradcheck that a masked span contributes zero gradient and the
    unmasked span's gradient equals training on that span alone with the masked prefix as context;
    red on HEAD. `make test/gradcheck`.
- [ ] RS-B4 (S) Verifier scoring call: given a prompt and a set of candidate answer tokens,
  return their log-probabilities at the answer position (for continuous verifier scores).
  - Acceptance: equals the log-softmax of the logits from a plain forward pass at tolerance on
    qwen3 and qwen35 tiny configs; works with an attached adapter. `make test`.
- [ ] RS-B5 (M) Batched multi-sequence decode for k independent samples of one prompt sharing
  the prefix (best-of-k and pass@k measurement).
  - Depends on the batched decode item already planned for qwen35; this item adds the shared
    prefix and per-sequence stop and seed control, and a throughput gate.
  - Acceptance: k=8 samples cost no more than 3x one sample in wall time on a short prompt;
    each sequence reproduces its single-sequence output for the same seed. `make test`.
