# rlcd - roadmap

RLCD (reinforcement learning for calibrated decisions): a model emits a
calibrated probability distribution, and the action taken on it is a
separate, explicit function of that probability and the actual cost of being
wrong, never a policy trained directly on outcome reward. `crates/rlcd` is a
new layer-3 leaf crate (added to `LEAF_CRATES` in
`scripts/gates/check-crate-layers.sh`), moved down out of `crates/decide` so
any decision-capable model can reach it without a model-to-model dependency:

- `rlcd::scoring` - proper scoring rules (focal + Brier) over a soft target
  distribution, not just a gold index. `decide::loss` re-exports it unchanged,
  so every pre-existing caller is untouched (verified against the exact
  formula it replaced, on a one-hot target, over 200 randomized cases).
- `rlcd::cost` - `CostMatrix`, expected cost, Bayes action/risk, decision
  regret, value of information. Gated against the device-diagnosis worked
  example by hand: posteriors 0.2/2/3/1/19, Bayes risk 0.8 -> 0.48, VOI 0.27,
  and the 1/11 -> 4/14 boundary flip that changes the optimal action without
  moving the probability.
- `rlcd::metrics` - ECE, AdaECE, classwise-ECE, NLL, Brier, reliability bins,
  coverage-vs-accuracy, failure-AUROC (AUGRC named but not implemented - see
  that module's own doc for why).
- `rlcd::atlas` - the `World` trait an executable probabilistic world
  implements to produce exact oracle targets, `DecisionContract`, and
  `check_information_refinement` (the `prior = E[posterior | evidence]`
  identity that catches an oracle conditioning on hidden state).
- `rlcd::witness` - exhaustive search over a `World`'s candidate points for
  cases where a `Learner`'s induced action disagrees with the oracle's,
  expanded into a corrective family (nearby correct point, resolving evidence
  reveal, a cost regime that flips the oracle's own action, a relabeled
  irrelevant variation).
- `brain::RlcdPipeline` (`crates/sdk/src/rlcd.rs`) trains a `Decide`-backed
  model against a soft-target distribution through `Decide::train_step_with`'s
  existing objective seam and evaluates it on calibration AND decision
  regret, not accuracy alone. `RlcdPipeline::train_voi_policy` trains a
  SECOND, genuinely-RL head - `{act_now, inspect}` through
  `decide::policy`'s clipped-choice PPO objective - gated against
  `rlcd::cost::voi`'s closed-form answer; only the evidence-gathering
  decision is learned, what "act now" means stays the closed-form Bayes
  action from stage 2.
- `samples/learning/rlcd` is the full pipeline end to end: an executable
  device-diagnosis world checked for oracle self-consistency before training,
  a real trained run (loss 0.64 -> 0.21 over 400 steps, encoder frozen), a
  witness audit against the trained model, a genuinely held-out phrasing
  predicting 0.374/0.626 against the exact 0.333/0.667, and a learned VOI
  policy that flips with the closed-form oracle across two tested cost
  regimes. See that sample's own README for the full measured run and its
  "what is not claimed" section.
- `crates/eval` re-exports `rlcd::metrics` as `eval::calibration`.
- The layer-3 leaf violation this crate's own closure inherited from
  `brain-data` (`brain-data -> brain-events -> brain-forecast`, a pre-existing
  regression unrelated to this crate, silently reintroduced 10 days after
  `check-crate-layers.sh` was added and not caught since the gate is not run
  on a regular cadence) is fixed: `base64` moved from `crates/events` into
  `crates/data`, `crates/events` re-exports it unchanged.
- `crates/modernbert/src/laya.rs`'s `LayaHead` gained an AdamW step
  (`adamw_step`/`adamw_step_scaled`, Laya M6), built on the same `ParamStore`
  + `optim::Optim` primitive `decide::model::Encoder` uses, verified on both
  backends via a real training-convergence check
  (`crates/gradcheck/tests/modernbert_adamw.rs`) alongside the existing
  finite-difference gate on its backward.

## Not yet done

- [ ] The Laya arm still cannot train through `RlcdPipeline`. `LayaHead` now
      has both a gradient-checked backward AND a working AdamW step, but no
      SDK-level path exists from `RlcdPipeline` (or `DecisionPipeline`) to
      it - `decision.rs`'s existing `LayaBackend` is inference-only and
      carries serving-calibration logic (per-option-count temperature
      scaling) that does not apply to a fresh training run, so reusing it
      directly is not the right shape; a trainable-Laya construction path
      (tokenizer + `ModernBert::new_train_on` + `LayaHead::new_train_on`,
      parallel to but not reusing `LayaBackend`) is substantial enough to be
      its own change, not attempted in this pass.
- [ ] No witness-family search over world PARAMETERS, only over a fixed
      world's observations and a caller-supplied bounded cost sweep. The
      research proposal this crate was built from also describes searching
      over generator parameters, observation masks and rendering choices; the
      `World` trait here takes one already-instantiated world, not a
      parameterized family. Extending it would need a second trait
      (`WorldFamily`, or similar) and is unscoped.
- [ ] The learned VOI policy was verified at two query costs (0.05, 1.0),
      not swept over a grid - it tracks the oracle's flip at both, but was
      not evaluated near the exact boundary itself, where a genuinely
      uncertain policy would be the correct answer, not a failure to
      diagnose. Per `.agents/roadmap/gauntlet.md`'s calibration-constraints
      section, a pass/fail GATE (rather than the plotted-curve evidence this
      already is) would need its own measured run at a `bench`-calibrated
      size before any threshold is set - not assumed in advance.
