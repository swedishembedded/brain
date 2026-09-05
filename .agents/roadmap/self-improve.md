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

**P7 onward (below)** extends this roadmap past P6's sven-traffic boundary:
a generic training-*regime* layer (DPO, GRPO, distillation, replay - not just
the P2/P3 weighted-SFT contract) usable by any `Model`, plus the
programmatic promote/reject machinery and lineage a real self-improvement
loop needs regardless of who supplies the trajectories. This does **not**
replace P6 - P6 is the real-sven-traffic proof and stays exactly as blocked
as the note at the end of this file says. P7+ is what makes the machinery
demonstrable on brain's own procedurally-generated, in-process-verifiable
environments (see `.agents/roadmap/gauntlet.md`) without fabricating a
stand-in for sven, which the same honesty discipline that deferred P6 would
otherwise forbid. **Standing invariant, unchanged by any of this: brain
never depends on sven.** `crates/atif` mirrors sven's trajectory format by
hand-copying it (see P1) precisely so that stays true; every environment
introduced from P11 on is self-contained inside brain with an in-process
oracle, and anything that needs an acting agent is an example that lives in
sven's own repo, talking to `brain serve --openai` - the one coupling this
whole roadmap has ever used.

**The keystone result P12–P14 rest on**, worth stating up front because nothing
below makes sense without it: DPO, GRPO (including clipping, off-policy
importance ratios and a k3 KL-to-reference term), and top-K distillation
**all reduce exactly - not approximately - to `Batch::LmWeighted` with
host-computed per-token weights.** `CE_GRAD_STATS` already writes
`d_logits[i] = (softmax(z_i) - e_{y_i})/C`, and `scale_row.wgsl` is a plain
per-row multiply (negative weights included), so `L_w = (1/C)Σ w_i·CE_i` is
differentiated exactly for any per-row scalar `w_i`. Concretely:
GRPO's clipped-surrogate gradient is reproduced by
`w_{g,t} = (C/N)·Â_g·r_{g,t}·1[unclipped]` (the importance ratio `r` **is**
the chain rule `∇r = r·∇log π`, not a stop-gradient surrogate), DPO's by
`w_t = ±β·σ(-u)·C` on chosen/rejected tokens (packed as two rows of one
batch - one forward, one backward per pair, not four), and top-K
distillation by `p - q_K = Σ_k q_k·(p - e_{v_k})` - literally K weighted-CE
passes, since `Σ_k q_k = 1`. Consequence: **no new `Batch` variant is added
anywhere in P7-P18** - the only two exhaustive matches on it
(`crates/model/src/parallel.rs`, `crates/model/src/shard.rs`) stay untouched
- no new WGSL kernel is needed, and any model that has adopted
`enable_weighted_loss` (P2) is automatically DPO-, GRPO-, and
distillation-capable. The one thing this reduction cannot express is an
entropy bonus (`∇_z H` is dense, not a scalar multiple of `p - e_y`) - v1
omits it and says so rather than silently approximating it.

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

## P6a - hot-swap cycle glue - DONE (still not wired to a live server or a timer)

`crates/cli/src/continuous_train.rs::hot_swap_cycle` is the driver P5's own
notes flagged as missing: `rl::continuous::run_cycle`, then (only if it
produced an adapter) `QwenResident::set_adapter` + `residency::Executor::
evict`. Lives in `crates/cli`, not `crates/rl` - `rl::continuous`
deliberately does not reach into `crates/residency`/`crates/cli` itself
(dependency direction stays model-crates -> residency/cli, never the
reverse). `QwenResident::set_adapter` lost its `#[allow(dead_code)]` - this
is its first real caller.

`evict`'s pinned-refusal contract means a refused evict (pinned, or simply
a key nothing has claimed yet) is not a lost swap: the adapter file and
the resident's pointer both already landed before the evict attempt, so
the very next successful evict of that key - from any cause - picks it up.

Verified: `cargo test -p brain-cli --bin brain hot_swap_cycle` - 2 tests
(nothing-to-train no-op; a real reward-stamped trajectory producing a
versioned adapter and pointing the resident at it, using the same tiny
synthetic fixtures as every other test in this roadmap - no real
checkpoint, no OOM risk), plus the existing `resident_llm` suite unaffected.

**What's still missing** before P6 can run for real: nothing yet calls
`hot_swap_cycle` on a timer/watch loop, and it is not wired into `brain
serve`'s startup path (`run_apis` in `crates/cli/src/run_cli.rs`) - that
wiring needs a real running server to verify end to end (this repo's own
gradual, independently-verified phases avoid doing that blind), and is
gated on the still-outstanding sven-side reward stamp (P0) having
something real to train on in the first place. When both exist, the
remaining work is: an opt-in flag/env var so default `brain serve`
behavior is unchanged, a background thread reusing the `brain_shutdown`
channel `run_apis` already wires up for clean process exit, and keeping a
concrete (not type-erased) `Arc<QwenResident>` handle alongside the
`Arc<dyn ResidentModel>` `run_apis` currently only keeps type-erased, since
`set_adapter` is inherent to `QwenResident`, not part of the
`ResidentModel` trait.

## P6 - the demonstrable proof - DEFERRED, blocked on sven (P0)

Run real sven sessions against `brain serve --openai` (qwen3), periodically
harvest reward-stamped trajectories, run P6a's cycle in the background,
chart a **held-out** real-task pass-rate (a fixed `sven-ci` headless suite,
disjoint from whatever tasks generated the training trajectories) over
wall-clock time across multiple continuous-loop cycles, plus a check that
LoRA hot-swap never drops or corrupts an in-flight request. This is the
concrete "brain trains continuously" deliverable, and the only piece of
this whole roadmap still not done.

Every piece on brain's side is built and independently tested (P1-P6a).
What's left is entirely outside this repo: sven's trajectories carry no
reward signal, so there is nothing real yet for the pipeline to train on
- the P0 boundary this roadmap set from the start. That's now written up
as a standalone task brief for whoever picks up sven next:
`applications/sven/.todo/self-improvement.md` (gitignored in sven, per
that repo's own convention for cross-session task notes - not something
this repo's history can point at by path, hence the full brief living
there rather than a citation here). Once a real sven session produces a
real reward-stamped trajectory, P6 is: point `hot_swap_cycle` (P6a) at
sven's real trajectory directory instead of a test fixture, wire it into
`brain serve`'s startup path behind an opt-in flag (deferred in P6a for
exactly this reason - no real server run to verify it against until now),
and let it run.

## Why P6 is not in this pass

P1-P6a are done and verified, each as its own gate (crate mirror,
gradchecked kernel composition, a convergence-tested generic driver, a
de-duplication + tested concurrency-safe primitive, and the cycle glue
tying them together). P6 is the actual multi-session, two-repo,
real-traffic proof - building it now would mean testing against fabricated
stand-ins rather than the real thing, the opposite of this repo's own
"evaluate honestly" discipline, and is why every phase up to here
deliberately used synthetic fixtures small enough to run safely (tiny
configs, no real checkpoints) rather than reaching for real weights or a
live server before there was something real to justify the risk.

## P7 - per-token logprobs - DONE

`crates/model/src/logprobs.rs`: `row_logprob(row, idx)` (stable
`log softmax(row)[idx]`) and `token_logprobs::<M>(m, tokens, targets)`, an
always-correct oracle over `Model::logits_all` - O(T·V) host work and a full
re-prefill, so it is the fallback/oracle, not the fast path. Migrated the
two hand-rolled log-softmax loops at `crates/qwen3/src/eval.rs:118,249` onto
one implementation.

`Model` gains two defaulted methods: `batch_token_logprobs(&self) ->
Option<Vec<f32>>` (default `None`) reads the per-position NLL a model's
CE kernel already wrote for the batch just forwarded and negates it -
`qwen3::Qwen` already reads exactly this buffer every forward
(`model.rs:1701`) and throws it away; `set_loss_weights(&self, &[f32])`
(default `unimplemented!()`, same pattern as `enable_weighted_loss`) is a
3-line delegate to the model's existing `write_weights`. Deliberately no
default written over `logits_all`: `Qwen::logits_all` calls `set_batch`
internally (`model.rs:2124`), so a default there would silently clobber
whatever batch a caller is mid-training-step on.

Gate: on tiny qwen3, `batch_token_logprobs()` immediately after `forward()`
must equal `logprobs::token_logprobs()` elementwise to 1e-5, on a batch that
includes IGNORE positions.

**Verified**: `qwen3::model::tests::batch_token_logprobs_matches_oracle_with_ignore`
- b=1/t=8 tiny Qwen, two IGNORE positions, fast and oracle paths agree to
1e-5 including exact `0.0` at both IGNORE positions. `cargo test -p
brain-model --lib logprobs` (2 unit tests) and `cargo test -p brain-qwen3
--lib` (124 passed; the only 3 failures are pre-existing wgpu
buffer-size-limit issues in `serve::tests`, unrelated - confirmed via `git
stash` against the same failures on the pre-change tree) both green.
`cargo check --workspace --all-targets` (excluding the pre-existing,
unrelated `qwen35`/`qwen35moe` `VisionConfig` test breakage, also confirmed
pre-existing via `git stash`) and `cargo clippy -p brain-model -p
brain-qwen3` clean on every touched file.

## P8 - the `Objective` seam - DONE (fit/fit_weighted; qwen3/qwen35/qwen3vl finetune deferred to P17)

The step loop is copy-pasted, not shared, in five places today:
`model::train::fit`, `rl::fit_weighted`, `qwen3::finetune::finetune`,
`qwen35::finetune::finetune`, `qwen3vl::finetune`. `gpt2::train::train` is
already a one-line delegate to `model::train::fit` - that is the target
shape for the other four.

New `crates/model/src/objective.rs`:

```rust
pub trait Objective<M: Model> {
    fn regime(&self) -> &'static str;
    fn prepare(&mut self, _model: &mut M) {}
    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32;
    fn eval(&mut self, _m: &M, _rng: &mut Rng, _batches: u32) -> Option<f32> { None }
    fn metrics(&self) -> Vec<(&'static str, f32)> { Vec::new() }
    fn itos(&self) -> Option<&[char]> { None }
}
```

The seam is deliberately at the level of "one micro-step" (set batch,
forward, decide weights, backward), not "produce a `Batch`" - a
pairwise/grouped objective's weights are a function of that forward's own
output (P7), so a `Batch`-producing seam cannot express DPO/GRPO. Everything
else - LR schedule, grad accumulation and its averaging scale, global-norm
clipping, AdamW, resume, wall-clock checkpointing, eval cadence, the final
save - stays in one loop, owned by `train::fit_with`, never by the
objective.

`crates/model/src/train.rs` gains `build_or_resume::<M>(cfg, opts, out,
dataset_vocab)` (the resume-vs-fresh-init block hoisted verbatim out of
`fit`) and `fit_with::<M, O: Objective<M>>(model, obj, opts, out)`. `fit`
and `fit_weighted` become delegates through `CausalLm`/`WeightedLm` with
**unchanged signatures** - zero call-site churn anywhere in the workspace.

This structurally fixes a real, previously-unnoticed bug: `fit_weighted`
calls `save()` where `fit` calls `save_with_itos()`, so every weighted
checkpoint has silently been losing its char vocab. With the loop owning
the save and asking the objective for `itos()`, no objective can forget to
carry it.

**Landed for `fit`/`fit_weighted` only.** `qwen3::finetune::finetune`,
`qwen35::finetune::finetune`, and `qwen3vl::finetune` were deliberately left
untouched in this pass - `qwen3`/`qwen35` finetune get their own resume fix
in P17, which is a better place to also migrate them onto `fit_with`, and
`qwen3vl::finetune` was out of this phase's scope. So the copy-paste count
this phase actually reduced is 5 -> 3, not 5 -> 1; the other three remain
open, tracked by P17 for two of them.

Gate: `cargo test -p brain-rl` (including the existing `qwen3_fit_weighted`
convergence test) and `-p brain-bench` green with **no behavior change**;
new test asserting a `fit_weighted` run now embeds `itos` in its output
checkpoint.

**Verified**: `cargo test -p brain-rl --all-targets` - 6 lib tests, 2
`continuous_cycle` integration tests, and both `qwen3_fit_weighted.rs`
tests (the pre-existing bigram convergence test, and the new
`fit_weighted_embeds_itos_in_its_output_checkpoint`, which fails against
the pre-P8 code and passes against this commit - the itos bug is real and
now fixed) all green. `cargo test -p brain-bench --all-targets` - every
integration test green including `brain_qa.rs`'s real from-scratch-Qwen
training run (96s) and the other GPU-bound gating tests (`qwen.rs`,
`scaling.rs`, `toolcall.rs`, `mqar.rs`, `parity.rs`), none touched by this
phase's changes. `cargo test -p brain-gpt2 --lib` - all 15 tests green,
including `trains_calculator_and_reduces_loss`, which now runs through
`fit_with`/`CausalLm`. `cargo check --workspace --all-targets` clean except
the pre-existing, unrelated `qwen35`/`qwen35moe` `VisionConfig` test
fixtures (fixed separately, see the top-level git history around this
commit) - confirmed via `git stash` before this phase touched anything.
`cargo clippy -p brain-model -p brain-rl --all-targets` clean on every
touched file (two `#[allow(clippy::type_complexity)]` added on the new
5-tuple-returning loader helpers, matching an existing repo convention for
that lint).

## P9 - hoist `WeightedCe`, adopt a second model - DONE

New `crates/model/src/lossw.rs::WeightedCe` hoists the three-part
buffers-plus-`SCALE_ROW`-step-plus-weighted-`forward` recipe currently
inlined in `qwen3::Qwen::enable_weighted_loss` (`model.rs:1765,2008`) so the
other `Head::TokenClassifier`-shaped models can adopt weighted loss (and
therefore DPO/GRPO/distillation, per the keystone result above) in ~10
lines instead of a 60-line copy. Migrate `qwen3` onto it first
(behavior-preserving), then adopt in **`gpt2`** - the second adopter is
what actually proves this is generic and not qwen3-shaped in disguise.
`gpt2` already carries `ce_buf`/`ce_stats`/`d_logits` in the identical
arrangement (`crates/gpt2/src/model.rs:704,773`).

Gate: `gradcheck::check_qwen3_weighted` stays green unchanged; new
`gradcheck::check_gpt2_weighted` following its exact pattern (non-uniform
weights including exact zeros, `directional_check` against finite
differences).

`WeightedCe` exposes exactly four operations - `new` (allocate the `[n]`
weight row + `[n·v]` scratch buffer), `hook` (append the `scale_row` step,
return the buffer downstream backward steps must read), `write` (upload
per-position weights), `loss` (the weighted-sum-over-count scalar `forward`
returns) - so a model owns a single `Option<WeightedCe>` field (`None` =
ordinary training, zero extra buffers/dispatch) in place of the three
separate fields (`Cell<bool>` flag + two raw buffers) `qwen3::Qwen` carried
before this phase. `qwen3::Qwen::enable_weighted_loss`/`write_weights`/
`forward`/`build_backward_steps` are now thin delegates to it, with no
change to what they compute - the pre-existing `check_qwen3_weighted`
gradcheck is the proof. `gpt2::Gpt` gained the identical trio
(`enable_weighted_loss`, `write_weights`, `batch_token_logprobs`) plus a
real `Batch::LmWeighted` arm in its `model::Model::set_batch` (previously a
wildcard panic) and its own `scale_row` kernel registration
(`crates/gpt2/src/model.rs`'s `PIPELINES`) - built in ~40 lines against
~60 for `qwen3`'s original inline version, confirming the hoist actually
shrinks a second adopter rather than just moving the copy.

**Verified**: `cargo test -p brain-gradcheck --lib weighted` - both
`qwen3_weighted_analytic_grads_match_finite_differences` (unchanged) and
the new `gpt2_weighted_analytic_grads_match_finite_differences` (confirmed
red first: it failed to compile against the pre-`enable_weighted_loss`
`gpt2::Gpt`) green. `cargo test -p brain-gpt2 --lib` - all 15 tests green,
including `pipelines_fully_costed` (confirms the newly-appended `scale_row`
kernel has a `gpu_core::cost` formula) and `backward_grads_finite`/
`forward_finite_and_deterministic` (unweighted path unchanged).
`cargo test -p brain-qwen3 --lib` - 124 passed, 2 ignored, 3 failed; the 3
failures (`embed_step_survives_a_vocab_table_that_exceeds_one_storage_binding`,
`head_matmul_over_binding_cap_does_not_panic`,
`head_matmul_tiled_matches_untiled_within_tolerance`) are this box's
integrated-GPU 2047 MiB `max_buffer_size` rejecting an oversized vocab-table
test fixture, unrelated to weighted loss - confirmed via `git stash` that
they fail identically on the pre-P9 tree. `cargo check --workspace
--all-targets --exclude brain-vulkan` clean. `cargo clippy -p brain-model -p
brain-qwen3 -p brain-gpt2 -p brain-gradcheck --all-targets` clean on every
touched file (pre-existing `doc_lazy_continuation`/`needless_range_loop`
warnings elsewhere in `brain-model`'s test fixtures and `qwen3::serve` left
untouched).

## P10 - generic rollout - DONE

New `crates/model/src/rollout.rs`: `Rollout` trait, `Completion { tokens,
logprobs, stop }` (the completion's own per-token logprobs under the
sampling policy - GRPO's `π_old` - captured free at sample time, never
re-derived later), `RolloutParams`, `ModelRollout` (the always-correct,
architecture-agnostic path over `Model::logits_all` - one sample at a time,
O(T²) re-prefill; works for every token-head `Model` with zero per-model
code, and doubles as the oracle `PagedRollout` is tested against), and
`PagedRollout` (the fast path over the already-existing
`model::serve::{PagedDecoder, Scheduler}` + `model::paged::{BlockTable::
fork, PrefixCache}`, which already shares one prompt's KV blocks across N
samples - three `PagedDecoder` impls exist today, all Qwen, any future
engine gets `PagedRollout` for free).

EOS and top-p do not need to be invented: `model::serve::SampleParams`
and `serve::sample_from_topk` already implement temperature / top-k /
nucleus / inverse-CDF sampling generically. `train::generate` becomes a
thin wrapper over `ModelRollout::sample_n(n=1)`, and its private
`sample_logits`/`argmax` are deleted - one sampler in the repo, not two.

Deliberately **not** built: a `Model -> PagedDecoder` bridge. `Rollout` is
the bridge at the level that matters; forcing every model to grow a paged
serving engine just to be rollout-capable would be the wrong shape.

Gate: seeded test asserting `generate`'s output is byte-identical to
today's for a fixed `(seed, temp, top_k)`; `ModelRollout` and `PagedRollout`
agree on greedy decoding for tiny qwen3; a `sample_n(n=8)` test asserting
the scheduler's own prefix-sharing stats show the prompt KV was actually
shared, not re-computed 8 times; EOS and top-p both honoured.

**Landed as spec'd, with one implementation deviation from `BlockTable::
fork`.** `sample_from_topk` grew a sibling, `sample_from_topk_with_logprob`
(factored out of the same shared `sample_from_topk_impl` so the two can
never drift on the actual math) - the one place a completion's per-token
`π_old` is computed, at sample time, off whatever candidate list was
already built to draw from. `PagedRollout::sample_n` prefills the SAME
prompt once per sample rather than forking one template `BlockTable` `n`
ways: `fork` shares a table's blocks byte-for-byte, including whatever
partially-filled tail block the prompt's own last, still-appendable block
is, and `BlockTable::append` only allocates a fresh block once the current
one is exactly full - so the very next token appended by any two forked
sequences would race to write the SAME physical slot. Privatizing that
tail first (`BlockTable::unshare_tail`) needs a device-side byte copy of
the block's live KV, which `PagedDecoder`'s trait surface has no operation
for. `PrefixCache`'s block-boundary-only sharing (already exercised by
`Engine::prefill`, unchanged) sidesteps the hazard entirely and is what
`PagedRollout` builds on instead - the gate's own literal wording
("prefix-sharing stats show the prompt KV was actually shared") points at
exactly this primitive. `BlockTable::fork` stays unused today; adopting it
safely later needs a `PagedDecoder::copy_block` (or similar) this phase did
not need to add.

The byte-identical gate is pinned at `temp = 0.0` (greedy), not a
real-sampling `(temp, top_k)` pair: `sample_from_topk`'s inverse-CDF walks
candidates in probability-SORTED order (required for its top-p nucleus
truncation), while the deleted `sample_logits` walked the raw vocab in
INDEX order - for the same `rng` draw those two summation orders pick
different tokens whenever more than one candidate survives filtering. Only
greedy decoding is order-independent (both the old `argmax` fold and the
new sorted-candidates-take-first agree on the lowest-index maximum), so it
is the one setting where "byte-identical" and "reuse `sample_from_topk`,
don't reinvent sampling" are simultaneously satisfiable.

**Verified**: `cargo test -p brain-model --lib` - 195 tests green (2 new in
`rollout::tests` for the candidate-sort tie-break and the EOS/`max_new`
accept helper, 2 new in `serve::tests` for `sample_from_topk_with_logprob`
against a manual softmax oracle and its greedy zero-logprob case).
`cargo test -p brain-qwen3 --lib` - 129 passed, 3 pre-existing failures
(`embed_step_survives_a_vocab_table_that_exceeds_one_storage_binding`,
`head_matmul_over_binding_cap_does_not_panic`,
`head_matmul_tiled_matches_untiled_within_tolerance`, all this box's GPU
`max_buffer_size` limit, exactly the 3 P7 already documented as unrelated -
confirmed again here via `git stash`), including all 5 new P10 gate tests:
`model::rollout_tests::generate_output_is_byte_identical_across_the_rollout_refactor`,
`serve::rollout_tests::model_rollout_and_paged_rollout_agree_on_greedy_decoding`,
`serve::rollout_tests::sample_n_shares_the_prompts_kv_through_the_prefix_cache`,
`serve::rollout_tests::eos_stops_a_completion_immediately_without_emitting_it`,
and `serve::rollout_tests::top_p_is_honoured_through_model_rollout`. `cargo
check --workspace --all-targets --exclude brain-vulkan` clean. `cargo
clippy -p brain-model -p brain-qwen3 --all-targets` clean on every touched
file (pre-existing warnings elsewhere in both crates, e.g. `crates/model/
tests/matmul_kq.rs`, `crates/model/src/int8.rs`, `crates/qwen3/src/
serve.rs:5229`, are untouched by this phase).

## P11 - `Environment` / `Verifier` - TODO

New `crates/rl/src/env.rs`: `Task { id, prompt: Vec<u32>, answer:
serde_json::Value }`, `Environment { name, tasks(seed) -> Vec<Task>,
step(...) -> StepOutcome, legal_actions(task, step) -> Option<&[u32]> }`
(single-turn is the default `step`; multi-step is supported, not bolted
on), `Reward { value, parts }`, `Verifier { verify(task, transcript,
completion) -> Reward }`.

The reward must be **programmatic and deterministic** - no model-as-judge -
so a run artifact is re-scorable and a promote/reject decision is
re-derivable from it later. `Task.answer` carries only what the verifier
needs to *recompute* the correct answer, never the answer itself - that is
the structural difference between a verifier and a label file, and it is
what later lets P18 assert no label ever touches training data.
`legal_actions` is on the trait, not an afterthought: an unconstrained
random policy's cold-start hit rate on even a small action space can be
too thin to bootstrap rejection sampling at all (worked example in
`.agents/roadmap/gauntlet.md`'s `AlienAPI` calibration); constraining
exploration to the environment's own declared legal actions restores the
honest chance rate, is applied identically to every arm of every
comparison, and the unconstrained score is still reported alongside it so
nothing is hidden.

`rl::atif::trajectory_reward` (P5) is refactored into a `Verifier` impl -
today's ATIF-ingestion path becomes a special case of this seam, not a
parallel system living next to it.

Gate: unit tests for a verifier over a small self-contained task family
(see `.agents/roadmap/gauntlet.md`) and for multi-step `step()`; existing
`rl::atif` tests stay green unchanged.

## P12 - `Grpo` objective (and RFT/STaR as its degenerate case) - TODO

`crates/rl/src/objective/grpo.rs`: group rollout via P10/P11, per-group
advantage `A_i = (r_i - mean_r)/(std_r + 1e-4)` with zero-variance groups
dropped (an all-right or all-wrong group contributes nothing - this is
what makes GRPO cheap here), the clipped importance-ratio weight and
optional k3 KL-to-reference term from the keystone derivation above, using
**precomputed** reference logprobs (no co-resident reference model needed
for a frozen reference - only its constant logprobs). Plain
rejection-sampling/STaR training is the same code with uniform weights and
one kept completion per prompt, deduplicated (over-weighting easy prompts
by keeping every correct sample is the classic STaR collapse mode).

Gate: a **local `CheckModel` harness** (12 existing precedents for
composite, non-`Model::forward` objectives, e.g. `crates/gradcheck/src/
clip.rs:112`) whose `loss()` recomputes the true clipped surrogate on the
host and whose `backward()` runs the weighted-CE backward, so finite
differences test the real objective, not a stand-in - the blanket `impl<M:
Model> CheckModel` does not apply here by design, since GRPO's `forward()`
!= what its `backward()` differentiates. Known, documented gotcha: the
clip indicator is piecewise-constant, so FD legitimately disagrees for any
token within some margin of the `1±ε` boundary; the harness asserts no
sampled token sits inside that margin rather than loosening the tolerance.

## P13 - `Dpo` objective - TODO

`crates/rl/src/objective/dpo.rs`: chosen/rejected packed as the two rows of
one `b=2` batch (one forward, one backward per pair - not four), weights
`w_t = ±β·σ(-u)·C` from the keystone derivation, reference logprobs for
both rows precomputed once and cached. Preference pairs are
**verifier-derived** (chosen = a verified-correct sample, rejected = a
verified-incorrect one from the same P12 group) - never a judge model.

Gate: `gradcheck::check_dpo_qwen3` (local `CheckModel`, FD against the host
`-log σ(u)`); an integration test on a synthetic pair set asserting the
chosen sequence's logprob rises and the rejected one's falls.

## P14 - `DistillTopK` objective - TODO

`crates/rl/src/objective/distill.rs`: K weighted-CE passes per the keystone
`p - q_K = Σ_k q_k·(p - e_{v_k})` identity - zero new kernels, zero new
gradient path beyond what P9 already gradchecked.

Gate: `gradcheck::check_distill_topk_qwen3` FD against the host KL; a
separate test asserting `K = vocab` reproduces exact `KL(q‖p)` to fp32
tolerance (proving the K-sparse path is a genuine special case of full KL,
not a different approximation that happens to also work).

## P15 - `Mixture` objective / anchor replay - TODO

`objective::Mixture<M>`: a probability-weighted draw over sub-objectives
per micro-step - zero new gradient math, it delegates. Paired with a fixed
anchor dataset (a frozen sample of the base model's own SFT data) for
anti-forgetting. Explicitly the weakest member of this family and said so
plainly: an L2-to-reference penalty would need a new argument threaded
through `Model::adamw_step` across all ~19 implementors, which is out of
scope here; LoRA's bounded-rank delta is the implicit regularizer this
whole self-improvement path already leans on.

Gate: deterministic mixture-ratio unit test (seeded draw counts match the
configured probabilities); an anchor-regression integration test.

## P16 - lineage + programmatic promote/reject - TODO

`checkpoint::ModelCard` gains an additive `training: Option<TrainingProvenance>`
(`code_revision` - the training CODE's git commit, `-dirty` when the tree
wasn't clean; `regime`; `seed`; `hyperparams`; `environment`; `gate`;
`trained_from`; `cycle`) - distinct from the existing `variant_of`, which is
an architecture-variant relation, not a training-lineage one.

New `crates/rl/src/gate.rs::gate(...)  -> GateReport` replaces "a human reads
a printed leaderboard" with a `Decision::{Promote, Reject(Cause)}`.
Promotion requires **all four**: an exact one-sided paired binomial sign
test `p <= 0.05` (conditioned on discordant pairs only - fixing a real
defect in the one existing precedent, see below); a minimum effect size
(so a significant-but-trivial win doesn't promote); the anchor suite not
regressing past a fixed budget; and a non-degeneracy check (completion
entropy over held-out tasks not collapsing relative to the incumbent) to
catch reward hacking / mode collapse rather than just missing it. Both
arms decode greedily, on the same held-out tasks in the same order, from
disk-reloaded checkpoints - not the freshly-trained in-memory instance
(the `lora_learning_gate` lesson: what's actually served must be what's
scored).

The sign test itself is hoisted, not re-derived: `crates/wan/tests/
finetune_ab.rs:439-454` already inlines one, and it has two real defects
this hoist fixes - it sums the binomial tail over *all* pairs instead of
conditioning on discordant ones (wrong-shaped for a binary 0/1 outcome,
though harmless for the continuous scores it was written for), and its
`binom_coeff`'s `n - k` underflows on `usize` when `k > n`. Moves to
`bench::metrics::sign_test` (`crates/bench` is already "how this repo
scores models"; `crates/stats` is telemetry, not statistics, so it is not
the right home). `wan`'s test migrates onto it.

Also fixes `rl::continuous::run_cycle`'s adapter versioning
(`adapter-{n:06}.safetensors` where `n = read_dir().count()` - deleting one
adapter silently overwrites the next one produced) to `max(existing) + 1`,
ideally sourced from `TrainingProvenance::cycle` once that exists.

Gate: `ModelCard` round-trips with `training` set and old cards without it
still deserialize; a `gate` unit test hitting every `Cause` variant on
synthetic paired scores; `wan`'s finetune A/B test green on the hoisted
`sign_test`.

## P17 - resumable LoRA adapters - TODO

`qwen3::finetune::finetune` always calls `init_weights` fresh
(`finetune.rs:31-116`), so a LoRA adapter can never be incrementally
continued across cycles - every call starts its adapter at zero-delta
init; `qwen35::finetune` mirrors it verbatim ("mirrors `qwen3::finetune`
exactly" per its own doc comment). No file-format change is needed to fix
this: `Qwen::save` already writes every `ps.params` entry including
`lora_a`/`lora_b`, and `cfg.to_json` already carries the LoRA config - the
defect is confined to the *load* path. Adds `finetune_from(base, dir,
opts, mode, out, resume: bool)`; when `resume && out.exists()`, loads `out`
(not `base`) for both config and weights and asserts the checkpoint's
`lora.rank`/`alpha` match the request, skipping the fresh-init overlay.
Once P8 lands, the better end state is `qwen3::finetune` collapsing onto
`fit_with(CausalLm)` and getting resume from `build_or_resume` for free.

Gate: train -> save -> resume -> train; loss keeps dropping and `‖B·A‖`
grows monotonically across the resume boundary, mirroring `crates/qwen3/
tests/lora_learning_gate.rs`'s own discipline of requiring the DATA to
explain the result, not just "the adapter changed."

## P18 - `rl::improve::cycle` - TODO

Generalizes P5/P6a's qwen3-only `rl::continuous::run_cycle` into: rollout
(P10) over an environment's explore split (P11) -> verify (P11) -> build a
weighted dataset via whichever objective is configured (P12-P15) -> train
(P8) -> gate against the incumbent (P16) -> stamp lineage (P16) -> version
the checkpoint. `rl::continuous` moves behind a `qwen3` cargo feature so
`crates/rl` stops pulling `brain-qwen3` into every generic consumer's
dependency graph by default.

Gate: an end-to-end tiny-model run producing a gated, lineage-stamped
adapter, where a **deliberately worse candidate is rejected** and the
incumbent retained (the gate proving it is a gate, not a rubber stamp),
plus two structural assertions that make "self-improvement" a checked
property rather than a claim: every completion span written into a
training set is a member of the multiset the policy actually sampled that
cycle (no label ever enters training), and every held-out/anchor task is
disjoint from the explore split (hashed and asserted, not assumed).

This is the machinery `.agents/roadmap/gauntlet.md`'s environments are
built to exercise - see that file for the environments themselves, the
retention-matrix and plasticity artifacts, and why they are a separate
crate from `crates/bench`.
