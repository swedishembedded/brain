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
  regret, not accuracy alone.
- `samples/learning/rlcd` is the full pipeline end to end: an executable
  device-diagnosis world checked for oracle self-consistency before training,
  a real trained run (loss 0.64 -> 0.21 over 400 steps, encoder frozen), a
  witness audit against the trained model, and a genuinely held-out phrasing
  predicting 0.374/0.626 against the exact 0.333/0.667. See that sample's own
  README for the full measured run and its "what is not claimed" section.

## Not yet done

- [ ] The Laya arm cannot train through `RlcdPipeline` yet. `LayaHead` has a
      gradient-checked, objective-seeded `backward(d_logits, d_act_logits)`
      and `new_train_on`/`is_trainable`/`zero_grads`/`read_grad`, but no
      optimizer step - `Stages::supports_training` stays `false` on that arm.
      Not attempted in this pass: `crates/modernbert/src/laya.rs` was under
      active, unrelated development (Laya M1 through M5 landed during this
      same work) and adding an optimizer step to the same struct mid-flight
      risked a direct collision rather than a clean addition. Once an AdamW
      step exists on `LayaHead` (for whatever reason it gets added), wiring
      `rlcd::scoring`'s objective into it and flipping
      `supports_training` is the small remaining step - the seam
      (`d_logits` in, `dL/dlogits` out) is already exactly what
      `RlcdPipeline` produces.
- [ ] No learned evidence-acquisition policy. Deciding WHETHER to run a query
      before acting is closed-form today (`rlcd::cost::voi`, gated against
      the worked example in `rlcd::cost`'s own test suite) - training an
      actual inspect/act policy through `decide::policy`'s existing
      PPO-over-committed-probability objective, gated by regret against that
      closed-form VOI oracle, is designed but not built. Per
      `.agents/roadmap/gauntlet.md`'s calibration-constraints section, a gate
      for this needs its own measured run at a `bench`-calibrated model size
      before any threshold is set - not assumed in advance.
- [ ] No witness-family search over world PARAMETERS, only over a fixed
      world's observations and a caller-supplied bounded cost sweep. The
      research proposal this crate was built from also describes searching
      over generator parameters, observation masks and rendering choices; the
      `World` trait here takes one already-instantiated world, not a
      parameterized family. Extending it would need a second trait
      (`WorldFamily`, or similar) and is unscoped.
- [ ] `crates/eval` does not yet re-export `rlcd::metrics`. Placed at layer 3
      instead of inside `eval` (layer 5, which depends on
      `gpt2`/`lfm2`/`yolov8`/`audio`/`ecapatdnn`) specifically so a sample
      could use it without that closure; `eval` re-exporting it later is
      free and was not needed for anything built so far.
