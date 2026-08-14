# self-improve - roadmap

Continuous self-improvement for brain's production models, generic over any
`model::Model`-implementing architecture with thin per-model wiring, fed by
real coding-agent trajectories from `applications/sven` - not brain's own
toy tasks. Distilled from the Stanford CS329A "Self-Improving AI Agents"
lecture series (test-time compute scaling, verification, STaR/rejection
sampling, GRPO/DAPO train-time scaling) into what belongs specifically in
brain (the model-running engine) versus in sven (the coding agent). Full
context and the boundary rationale: see the planning session this roadmap
was extracted from; the phase numbering (P0–P6) is kept unchanged here for
continuity.

**Boundary** (holds for every phase below): sven owns generating real
trajectories, executing tools/tests, and stamping a reward/outcome signal
onto a concluded trajectory - brain owns everything from "a reward-stamped
trajectory" onward (ingestion, the weighted training objective, the LoRA
adapter it produces, and hot-swapping it into the serving model). sven
already treats brain purely as an OpenAI-compatible HTTP endpoint
(`brain serve --openai`) - this work does not add any other coupling.

## P0 - sven-side reward stamp

**Not implemented here - a prerequisite on sven's side, not brain's.**
sven's task machines (its HSM `machines`/`kernel` crates - sven has
undergone its own internal crate-rename churn during this work; check
sven's own `AGENTS.md` for current names before touching this) need to
stamp an outcome/reward signal (e.g. `{"reward": 1.0, "outcome":
"tests_passed"}`) into a concluded trajectory's `extra` field at the point
the task machine already knows whether it succeeded. Everything below
assumes trajectories arriving at brain already carry this.

## P1 - mirror `atif` into brain - DONE

sven's trajectory crate (ATIF v1.7 - `Trajectory`/`TraceStep`/
`sft_steps()`) was briefly named `crates/trace`, renamed to `crates/atif`
by sven mid-session; brain's mirror follows that rename. Landed at
`crates/atif` (package `brain-atif`, lib `atif`), copied verbatim from
`applications/sven/crates/atif`, kept manually in sync (not a Cargo path/
git dependency - brain stays a self-contained workspace). Wired into the
root `Cargo.toml` (workspace members, default-members, the new `chrono`/
`thiserror`/`tempfile`/`libc` workspace dependencies its `Cargo.toml`
needs - `libc` was previously only a literal per-crate dep in
`gpu-core`/`shutdown`; promoted to a workspace entry and both crates
migrated onto it instead of adding a third copy of the literal).

Verified: `cargo test -p brain-atif` - 37 tests, all green, matching
sven's own suite for the crate. `cargo check --workspace --all-targets`
- clean, nothing else in the workspace regressed.

## P2 - generic weighted-`Batch` contract - DONE (qwen3; other `Head`-using models not yet adopted)

Added `model::Batch::LmWeighted { tokens, targets, weights }` (per-POSITION
scalar weight on the CE gradient, `crates/model/src/lib.rs`) alongside the
existing `Batch::Lm` - fully additive, the two existing exhaustive-match
call sites (`model::parallel::clone_batch`, `model::shard::clone_batch`)
updated to re-borrow it, every other `Batch` consumer unaffected via their
existing wildcard arms.

`qwen3` is the first (and so far only) adopter, via:
- `Qwen::enable_weighted_loss(&mut self)` - an opt-in constructor-time
  toggle following the exact same pattern as `enable_mrope`/
  `enable_mm_splice` (allocate buffers, set a `Cell<bool>`, rebuild
  `bwd_steps`). **Ordinary (unweighted) training pays zero extra kernel
  dispatch** - the scale_row step and the extra buffers only exist on an
  instance that opted in.
- Backward: after `CE_GRAD_STATS` writes `d_logits`, an enabled instance
  routes it through `scale_row.wgsl` (already existed, already used
  elsewhere in `model::gdn`/`vit`/`sam1` - no new WGSL) into a separate
  `d_logits_weighted` buffer, per the field's own doc comment on why not
  in-place; every downstream consumer (`head` dw/dx) reads that buffer
  instead. `Batch::Lm` on an enabled instance implicitly weights every
  position `1.0` (reproducing the unweighted gradient exactly); only
  `Batch::LmWeighted` supplies real weights.
- `forward()` on an enabled instance returns the WEIGHTED loss (`Σ
  weights[i]·ce_loss[i] / count`), not the plain mean - required by the
  `Model::forward` contract ("the scalar loss `backward` differentiates")
  and by the gradcheck harness, which finite-differences whatever
  `forward()` returns.

**Gradchecked**: `gradcheck::check_qwen3_weighted` - deliberately
non-uniform weights including exact zeros (not degenerate all-ones),
`directional_check` against finite differences of the actual WGSL kernel
pipeline. `cargo test -p brain-gradcheck --lib qwen3_weighted` - green.

**Not yet done**: `qwen35moe`, `glmdsa`, and every other
`Head::TokenClassifier` model still only handle `Batch::Lm` (fall through
their existing wildcard arm on `Batch::LmWeighted` - a clean panic, not
silent wrong behavior, but no weighted training for them yet); no
`Head::Regression` model has adopted the parallel `mse_value_w`/
`mse_grad_w` recipe (those kernels exist, registered, currently used only
by `gradcheck/tests/glue.rs`). Adopting either is the same three-part
pattern qwen3 just established: opt-in buffers/flag, backward-step
routing, `forward()` weighted-sum branch.

## P3 - generic `crates/rl` weighted-training driver - DONE

New crate `crates/rl` (package `brain-rl`, lib `rl`), generic over `M:
model::Model`. `rl::fit_weighted` mirrors `model::train::fit`'s control flow
exactly (cosine LR, grad-accum, resumable checkpointing - reused via
`model::load_dataset`, not duplicated) but drives P2-shaped `Batch::
LmWeighted` batches and calls `Model::enable_weighted_loss` (a new trait
method, default `unimplemented!()`, overridden by `qwen3::Qwen` to delegate
to its inherent method) right after construction. Zero qwen3-specific code
in the driver itself - qwen3 is simply the first `M` it's instantiated
with; any other `Model` that overrides `enable_weighted_loss` works
unchanged.

**Dataset format**: an optional `train.weight.bin`/`val.weight.bin` (raw
`f32`, reusing the already-existing `data::binio::{read,write}_f32_bin` -
no new file format) parallel to `train.u32.bin`, one weight per token.
Absent means every position implicitly weights `1.0`
(`TokenDataset::get_batch_weighted`'s default, added alongside the existing
`get_batch` - new method, zero changes to `get_batch`'s 5 existing
callers), so an ordinary `model::train::fit` dataset directory also works
unchanged through `fit_weighted`.

**Verified**: `cargo test -p brain-data --lib loader` (5 new
`get_batch_weighted`/`with_weights` unit tests), `cargo test -p brain-rl
--lib` (3 tests for the weight-file attach/default/mismatch-error paths),
`cargo test -p brain-rl --test qwen3_fit_weighted` - a convergence
integration test (mirroring `toyseq2seq/tests/convergence.rs`'s
learnability-guard pattern) that trains a real tiny `qwen3::Qwen` on a
deterministic bigram through the FULL file-based `fit_weighted` driver and
asserts the loss actually converges - proving the plumbing end to end, not
just the gradient math (already proven by P2's gradcheck).

**Not yet done**: turning real ATIF trajectories into a weighted dataset
directory (that's P5) - `fit_weighted` today only consumes the on-disk
format above, however it was produced.

## P4 - unify device-adapter LoRA + generic residency hot-swap - DONE

**LoRA de-duplication**: `crates/model::lora::device_adapter` (new
submodule, alongside the pre-existing "host pair" family `flux2`/`s3dit`
already shared there) now holds the save/fold I/O and `fold_delta` math
that `qwen3`, `qwen35moe`, and `deepseek2`'s `lora.rs` files had each
carried as a near-verbatim copy (`qwen35moe`'s and `deepseek2`'s own doc
comments called theirs "a direct port"). Generic over `M: model::Model`
(`param_names`/`read_weight`, already on the trait - no new trait needed
for this half). Each of the three crates' `lora.rs` is now a thin wrapper
supplying its own `LoraCfg` type and family tag (`"qwen"`/`"qwen35"`/
`"deepseekv2"`) - same public signatures, so every existing caller
(finetune paths, CLI, tests) needed zero changes.

Verified: `cargo test -p brain-qwen35moe --lib save_and_fold`, `cargo test
-p brain-deepseek2 --lib save_and_fold`, `cargo test -p brain-qwen3 --test
lora_learning_gate` (the real train→save→fold→reload→predict integration
test) - all green, no behavior change.

**Residency hot-swap primitive**: `ResidencyManager::evict`/
`Executor::evict` - a new, PINNED-SAFE public single-device eviction path
(the pre-existing private `evict` was renamed `evict_entry`; it has no
pinned check because its only caller, the auto-eviction planner, already
pre-filters pinned entries via `Residents::lru_on`- confirmed by reading
that call site before reusing it, not assumed). Mirrors `evict_multi`'s
exact contract: refuses (`false`) while a job is actively running against
the key, so a hot-swap never interrupts an in-flight request; the swap
takes effect on the NEXT claim. Verified: `cargo test -p brain-residency
--lib` (78 tests, all green, including the new
`evict_frees_the_device_and_refuses_while_pinned`).

**The write side**: `QwenResident::set_adapter` (new) - the `adapter:
Option<String>` field became `adapter: RwLock<Option<String>>` (the only
interior-mutability change needed; `activate(&self, ..)` was already
read-only and now clones the path out from under a short-held read lock
before the slow fold/load work, so `set_adapter` never blocks on an
in-flight activation). Still `#[allow(dead_code)]` after P5 (below) - P5
produces adapter files but does not itself call `set_adapter`/`Executor::
evict`; see P5's own note on why, and what's still missing to close that
last gap.

## P5 - ATIF → weighted qwen3 examples + the continuous-loop cycle - DONE (the swap-trigger wiring is the one gap left)

Two new modules in `crates/rl`, both deliberately qwen3/chat-specific
(unlike P3's generic core) - see each module's own doc comment on why that
split is intentional, not an oversight.

**`rl::atif`** - walks `atif::Trajectory::sft_steps()`, converts each into
`data::chat::ChatMessage`s (reusing that existing chat/tool-call format,
not a second one), pulls the reward from `Trajectory.final_metrics.extra.
reward` (P0's contract), and writes a P3-shaped weighted dataset directory
(`train.{u32,mask,weight}.bin` + `meta.json`, plus empty `val.*` - `model::
load_dataset` requires them to exist; empty is its own deliberate
"skip eval" signal). v1 scope, stated in the module doc comment: text-only
step content and direct (non-subagent-delegated) tool calls/results only -
either is a hard error, not a silent drop. A trajectory with no reward
stamp is SKIPPED (not defaulted to weight 1.0) - training on an
unknown-outcome trajectory would be silently indistinguishable from a
known-good one. `ingest_dir` walks a whole directory of trajectory JSON
files at once, skipping (with a logged reason) any that fail to parse,
lack a reward stamp, or hit the v1 scope limits, so one bad file never
blocks the rest of a batch.

**`rl::continuous::run_cycle`** - one continuous-training cycle: `ingest_dir`
(returns `Ok(None)` - a quiet no-op, not an error - if nothing new is
waiting), then `fit_weighted::<Qwen>` (resumes the full base+adapter
training checkpoint if one exists, else starts fresh with a new `LoraCfg`),
then extracts and saves a NEW versioned adapter-only file (`adapter-
NNNNNN.safetensors`, via `qwen3::lora::save_adapter` - reusing P4's
de-duplicated save path) for serving. Returns the new adapter's path.

**Verified**: `cargo test -p brain-rl` - 9 tests total: `rl::atif`'s
reward-extraction/conversion/skip-unstamped unit tests, `run_cycle`'s
empty-no-op and a REAL train→save→fold round trip (produces a genuine
`adapter-000000.safetensors`, folds it into a copy of the base checkpoint,
asserts the fold actually changed weights - reusing `qwen3::lora::
fold_adapter_into` directly, not re-deriving the check), plus P3's own
convergence test, all green.

**What's still missing** (the one concrete gap left before P6 can run for
real): nothing yet calls `run_cycle` on a timer/watch loop, and nothing
yet pairs its returned adapter path with `QwenResident::set_adapter` +
`Executor::evict` to actually take effect on a live resident - `rl::
continuous` deliberately does not reach into `crates/residency`/
`crates/cli` itself (dependency direction stays model-crates ->
residency/cli, never the reverse), so that pairing is a small driver
living in `crates/cli` (or wherever the served process's own main loop
lives), not in `crates/rl`. `QwenResident::set_adapter` stays
`#[allow(dead_code)]` until that driver exists and calls it.

## P6 - the demonstrable proof - NOT STARTED

Run real sven sessions against `brain serve --openai` (qwen3), periodically
harvest reward-stamped trajectories, run P5's loop in the background, chart
a **held-out** real-task pass-rate (a fixed `sven-ci` headless suite,
disjoint from whatever tasks generated the training trajectories) over
wall-clock time across multiple continuous-loop cycles, plus a check that
LoRA hot-swap never drops or corrupts an in-flight request. This is the
concrete "brain trains continuously" deliverable.

## Why P5–P6 are not in this pass

P1–P4 are done and verified, each as its own gate (crate mirror, gradchecked
kernel composition, a convergence-tested generic driver, and a
de-duplication + tested concurrency-safe primitive). P5 (turning real ATIF
trajectories into weighted training data - needs P0's sven-side reward
stamp to have real, non-synthetic input to test against) and P6 (the actual
multi-session, two-repo end-to-end proof) are each substantial enough, and
depend on state outside this repo (P0), that building them in the same pass
without that dependency actually being available would mean testing against
fabricated stand-ins rather than the real thing - the opposite of this
repo's own "evaluate honestly" discipline. Picking this file back up: P5
first (it can start against hand-built weight-stamped ATIF fixtures even
before P0 lands on sven's side, same technique `crates/rl`'s own tests
already use for synthetic weight files), P6 once a real sven session can
produce real reward-stamped trajectories.
