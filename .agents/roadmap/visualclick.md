# visualclick - roadmap

A validation experiment, not a shipped capability: can a row `Decide`'s
encoder never produced - a rendered scene's color, projected by a small
trainable `Linear -> LayerNorm` host-side and spliced directly into the
head's cross-attention state - drive the model to a spatially grounded
answer? Built and verified in an isolated worktree
(`.claude/worktrees/visualclick`) before touching the working branch, per the
standing rule that an unproven new capability earns its way onto main only
after a real measured result, not before.

- `crates/decide::decide::Features` gained a public constructor
  (`from_parts`) and three getters (`state_rows`/`n_slots`/`hidden`). Its
  fields were private, so nothing outside the crate could build a `Features`
  except `Decide::score_keeping` itself - `from_parts` is the seam a caller
  edits specific rows through before handing a new one to
  `accumulate_kept`/`score_kept`. Gated by a round-trip test plus a
  row-edit-moves-the-score test proving the mechanism is real, not just
  well-typed.
- `crates/decide::model.rs`'s encoder `dx` buffers gained `COPY_SRC` - a real
  bug this work reproduced, not a design gap: nothing before this ever read
  the encoder's own seed buffer back to the host (it exists purely as
  GPU-side scratch the encoder's own reverse pass consumes), so it was
  allocated without the usage flag a host readback needs. First symptom was
  a wgpu validation panic, not a wrong number - the kind of bug that is loud
  rather than silently wrong.
- `crates/sdk::rlcd` widened its re-exports (`Decide` reachable via
  `RlcdPipeline::model_mut`, plus `Features`, `Answer`, `Opt`, `Question`,
  `decision_loss`, `decision_loss_soft`, `ece`) - required because
  `samples/README.md` rule 1 lets a sample name only the `brain` facade, and
  none of these were reachable through it before a caller needed to drive
  `Decide` below `RlcdPipeline`'s own `(state, target)` surface.
- `samples/learning/visualclick` is the full experiment: a synthetic
  rectangle-click world (`scene.rs`), a host-side projector with a
  gradient-checked LayerNorm backward (`patches.rs`), and three arms
  (`blind`/`text`/`pixels`) answering the same 16-way question. Measured,
  seed 11, `all-MiniLM-L6-v2` frozen: `blind` 0.053-0.072 (chance 0.0625,
  as designed), `text` 0.263 (3000 steps, the existing unmodified `Decide`
  path), `pixels` 0.273 (15000 steps, the new splice path) - matching
  `text` and decisively above chance. Both ablations (`--ablate noise`,
  `--ablate shuffle`) collapse back to chance (0.068, 0.062), which is the
  falsifiability check the result depends on. **Verdict: positive** - the
  splice mechanism works, at the scope this experiment tested it.
- Two design flaws were found by real measured runs and fixed, both
  recorded in `patches.rs`'s and `main.rs`'s own doc comments because either
  one alone is exactly the kind of thing that looks plausible and is
  silently wrong: (1) a shared `Question::instructions` string across every
  example gave the head's cross-attention no channel to condition on
  per-example content, fixed by building a fresh `Question` per example; (2)
  a placeholder text state left alongside the spliced image rows let the
  head spend its gradient budget on the familiar-but-useless text rather
  than the novel-but-informative image rows, fixed by dropping the
  placeholder entirely once its job (satisfying `pack_request`'s non-empty
  state requirement) was no longer load-bearing.
- A LayerNorm was added after the projector's linear stage (measured
  necessary: raw-RGB projector rows read RMS ~0.17-0.27 against real encoded
  rows' ~0.36-0.63), and the color feature was changed from continuous RGB
  to a discrete one-hot over the same six-color vocabulary `instruction()`
  draws from (a raw-RGB version, even with the LayerNorm fix, still trained
  flat at chance - a structural gap, not a scale one, per that module's own
  doc on why CLIP's contrastive pretraining is what a real vision tower
  would supply that raw pixels cannot).

## Not yet done

- [ ] Not rebased onto the working branch yet. This roadmap entry documents
      the worktree result; the plan's own gate is rebase only after a
      positive verdict, which this is - the rebase itself is a separate,
      explicit step.
- [ ] The color channel is a discrete one-hot over six known colors, not a
      real image embedding. No CLIP or other vision-tower checkpoint was
      available in the environment this was built in. `patches::Projector`'s
      interface (`[n_rows] -> Linear -> [n_rows, H]`) is the seam a real
      vision encoder's patch tokens would plug in behind, unattempted here.
- [ ] The splice point is `Decide::accumulate_kept`'s kept-features seam
      (post-encoder, into the head's cross-attention state), not the
      embedding stream itself (`crates/decide/src/model.rs`'s `x[0]`). A
      deeper splice was scoped in the original plan (`row_scatter` into
      `x[0]` before its LayerNorm, mirroring `crates/clip`'s
      `PatchSource::Tokens`) and red-teamed as architecturally sound but was
      not needed once the kept-features seam proved sufficient; it remains
      unbuilt.
- [ ] `pixels` needed roughly 5x the training steps `text` did (15000 vs
      3000) to reach a comparable held-out accuracy, and that ratio was
      measured at ONE task difficulty (3 rectangles, 4x4 grid, one frozen
      encoder). Whether it holds at a harder task, a different grid size, or
      a different encoder is untested.
- [ ] The projector is intentionally host-side CPU arithmetic, not a GPU
      kernel - correct for validating the mechanism, not for a production
      training loop at scale.
