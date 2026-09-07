# continuous-learning

**Status: planned, not started. This is brain's half of a cross-repo
initiative - the entry point and the full cross-repo picture (MVP cut,
dependency DAG, deferred work with justification) live in
`whale/.agents/roadmap/continuous-learning.md`. sven's half is in
`sven/.agents/roadmap/continuous-learning.md`. Read the whale file first.**

## Goal, restated for this repo

Brain's job in the loop: turn a confirmed fact (or a batch of facts
extracted from a document) into training data, run a real study that trains
a LoRA adapter and gates promotion on a frozen, disjoint probe set, and
hot-swap a promoted adapter into a live serving process with zero downtime
and zero dropped in-flight requests. Nothing here is agent logic (that's
sven) and nothing here is scheduling/distribution (that's whale) - brain
does the math, per its own AGENTS.md.

## The load-bearing correction: there is no new training composition to build

The original design for this repo (called `B5` in early drafts) proposed a
new `rl::fact::edit_cycle` composing `rl::improve::cycle` with an
`Anchor`-mixed objective, for one fact at a time. **This is the wrong
primitive, and brain has already measured why.** `crates/rl/examples/
continual_learning.rs`'s own doc comment (lines 12-30) records a real run:
the `--regime grpo` composition (a per-cycle `improve::cycle`, exactly what
the rejected design describes) does **not** accumulate capability at this
model scale - ACC 0.271, *below* the untrained base's 0.354, 2 of 12 cycles
promoted. The regime that *was* measured to work is `--regime sft`
(`Regime::Sft(SftConfig::default())`, `crates/rl/src/continual.rs:99,120`,
teacher-forced, mixed 50/50 with a rehearsal pool): ACC 0.932 and 0.815
across two seeds, 12/12 and 10/12 cycles promoted, every pre-registered
target passing.

`rl::continual::Curriculum` (`continual.rs:159`), driven by `run_study`
(`:882`), already gives the retention matrix (`:596`), BWT (`:668`), a
fresh-adapter plasticity control (`:700`), and a pre-registered PASS/FAIL
block (`:616`) - everything the rejected `edit_cycle` design would have had
to reinvent, badly, on top of a composition already shown not to work. So:
**implement `Curriculum` for documents/facts; do not build a new cycle.**

## Milestones

### B1 - document/fact batch → masked chat dataset
Turn N confirmed `{fact, probe_question, expected_answer}` triples (sven
extracts these - see sven's file, `S7`) into `data::chat::ChatSample`s via
`data::chat::prepare_chat_samples` (`crates/data/src/chat.rs:415`), which
already writes the `train.mask.bin` companion file that `model::train::
load_dataset` prefers over char-offset masking (`crates/model/src/
train.rs:163`). This is where the JSONL-templating idea from early drafts
actually lands - not as a standalone helper, but as `Curriculum::
write_sft_dataset()` (`B5′` below).

**Test-first:** `a_document_fact_batch_writes_a_masked_chat_dataset_
load_dataset_accepts` - build N triples, run them through
`prepare_chat_samples`, assert `load_dataset` reads the result back with
every fact's answer span masked `train: true` and nothing else. Red: no
caller does this today for anything but hand-authored JSONL.
**Commit:** one.

### B2 - the document-learning `Environment`/`Verifier` + a pre-registered `GateConfig`
Implements `rl::continual`'s `Environment`/`Verifier` traits over the frozen
probe set. Verification is **programmatic exact-match against the probe's
`expected_answer`**, never model-as-judge - `env.rs` is explicit that this
is the discipline, and it's what makes `F1`'s number defensible rather than
vibes-based.

The probe split must be disjoint from the training span by construction,
mirroring brain's own shortcut-defeating checks: `explore_anchor_split`'s
id-hashed disjointness assert (`crates/rl/src/improve.rs:59`) and
`assert_trained_spans_were_sampled` (`:84`). A probe's expected answer must
never appear inside any row the model is trained on - sven's extraction
step (`S7`) freezes probes at extraction time for exactly this reason, and
this milestone is what actually checks it rather than trusting the caller.

**`GateConfig` for document-learning, pre-registered, not tuned after the
fact:**
- `alpha = 0.05` (default, `crates/rl/src/gate.rs:87`)
- `min_effect_size = 0.15` - **deliberately not the `0.02` default**
  (`gate.rs:86`). `effect_size = mean(candidate.held_out) -
  mean(incumbent.held_out)` over the *whole* scored set (`gate.rs:111`); at
  0.02, a 2% wobble on a large probe set would promote. `F1`'s claim is "it
  learned the document", not "it moved."
- `anchor_budget = 0.02`, `min_entropy_ratio = 0.5` (defaults)
- ≥48 held-out probes per cycle, matching the measured SFT regime's own
  `--eval-per-cycle 48` - this puts `bench::metrics::sign_test`'s
  discordant-pair floor (5 net wins needed for `p ≤ 0.05` - `k=n=5` gives
  `p=0.03125`) far out of the danger zone; with 20 facts × 3 probes = 60
  in-scope probes and, say, 40 flips, effect size is 0.40 with p far below
  α. Batching many facts makes the sign test *easier* to satisfy, not
  harder - the actual risk batching introduces is `B8`'s, not this
  milestone's.

**Test-first:**
- `the_probe_split_is_disjoint_from_the_training_split_by_task_id` (mirrors
  `explore_anchor_split`) - red until the disjointness check exists;
- `a_two_percent_wobble_does_not_promote_under_the_document_gate_config` -
  red against the *default* `GateConfig` (which would promote it), green
  once the document-specific config is wired in.
**Commit:** one.

### B2b - hoist the gate below the model layer (DONE)

`B3b` needs `B2`'s gate and probe contract from inside `crates/qwen3`, and
could not have them: `crates/qwen3` is layer 4, `rl::gate` computed its
significance bar with `bench::metrics::sign_test`, and `brain-bench` links
every model it benchmarks - so `brain-qwen3 -> brain-rl -> brain-bench ->
brain-qwen3` was a real Cargo cycle, not a style objection. It was found by
adding the dependency and reading `cargo tree`, and correctly refused rather
than routed around.

The fix is a **new leaf crate, `crates/promote`** (`brain-promote`), holding
the part of this machinery that never needed a model: the
`Environment`/`Verifier` seam (`promote::env`), the exact paired sign test
(`promote::stats`), the four-bar promote/reject decision (`promote::gate`),
and the frozen `{fact, probe_question, expected_answer}` contract with its
environment, verifier, `document_gate_config` and per-fact verdicts
(`promote::document`). Its whole dependency closure is `brain-data` and
below, so a model crate can depend on it.

Nothing was copied. `rl::env`/`rl::gate` are `pub use promote::{env, gate}`,
`rl::document` re-exports `promote::document::*` alongside the half that DOES
need a model (the `Curriculum` impl, `write_sft_dataset`, `run_document_
study`), and `bench::metrics::sign_test`/`SignTest` are re-exports too - so
every existing caller, `crates/wan`'s finetune A/B gate included, is
unchanged. `B2`'s and `bench`'s own named tests moved with the code and pass
verbatim; the only source changes are import paths, panic-message prefixes
that name the module they are in, and two items widened from private to
public (`DocumentEnv::row`, `document::task_id`) because the half left in
`rl` and any future gate action are now callers rather than neighbours.

The layering is machine-checked, not asserted in prose:
`scripts/gates/check-crate-layers.sh` (wired into `make check/scripts`) fails
if `brain-promote`'s `--all-features` closure ever reaches the model layer -
`--all-features` because `brain-rl` already hides `brain-qwen3` behind an
off-by-default feature, and a default-features check would wave the same
escape hatch through.

**Commit:** one.

### B3a - qwen3 `lora_train` as a `capability::Action`
Today qwen3 finetune is CLI-only (`crates/cli/src/qwen_cli.rs:559::
finetune_lora`) - not visible to whale's node-type generation at all.
flux2/s3dit/wan/qwen3vl already expose `lora_train` this way
(`crates/{flux2,s3dit,wan,qwen3vl}/src/caps.rs`); qwen3 needs the same
treatment, with one deliberate improvement over the precedent: **declare
`BlobSpec` I/O from day one** - dataset in as a `Media::Bytes` JSONL blob
(never a server filesystem path), adapter out as a `BlobSpec` matching
`flux2`'s existing output shape, base weights as a `ParamSpec::…
host_env("BRAIN_QWEN_WEIGHTS")` so `Manifest::for_serving` projects it out
of the marketplace-facing manifest entirely. `s3dit`'s own `caps.rs:68`
still uses filesystem path params for some inputs - that's precedent for
what *not* to copy, not what to follow.

**Test-first:**
- `manifest_lists_generate_and_lora_train` (mirrors `flux2/src/caps.rs:372`'s
  `assert_eq!(names, […])`) - red today, qwen3's manifest has one action;
- `lora_train_declares_the_adapter_as_a_retrievable_output_blob` (mirrors
  `flux2/src/caps.rs:393`);
- a real round-trip on the tiny-Qwen fixture: the returned adapter blob folds
  into the base checkpoint and measurably changes weights;
- **`the_served_manifest_carries_no_filesystem_path_param`** - a negative
  test, the concrete regression guard for the BlobSpec discipline above.
No CLI change needed - `brain qwen3 finetune --lora` is untouched, this
milestone only makes the existing training loop visible as a capability.
**Commit:** one.

### B3b - qwen3 `lora_gate` as a `capability::Action` (ON HOLD)
**On hold pending a real decision about whether whale scheduling is wanted
at all.** An attempt hit a hard Cargo cycle (`brain-qwen3 → brain-rl →
brain-bench → brain-qwen3`) - the gate machinery sits ABOVE qwen3 in the
layering, so reaching it from qwen3's own manifest needs crate surgery, not
a wiring fix. The product decision since taken is that the core learning
loop works with sven+brain alone and whale is an optional later scale-out,
so nothing downstream is blocked: `B9` gives the same study a plain CLI
entry point from `crates/cli`, which legally depends on both crates. Resume
this only when a whale graph is actually being built, and decide the crate
surgery then rather than forcing it now.

Wraps `B2`'s gate: candidate adapter blob + probe-set blob in, a
`GateReport` + promote/reject decision out. This is what makes the whale
graph two nodes (`lora_train` → `lora_gate`) rather than one - load-bearing
for whale's `W6′` testnet scenario, which depends on placement spreading a
*two*-node graph across two providers.

**Test-first:** the manifest name-list test, plus a round-trip on the tiny
fixture where a deliberately anchor-destroying candidate adapter returns
`reject: AnchorRegressed`.
**Commit:** one - deliberately separate from `B3a`; different purpose,
independently reviewable.

### B5′ - the document `Curriculum`, driven by `run_study` under `Regime::Sft`
Implements `rl::continual::Curriculum` for a document/fact batch:

| trait item | implementation |
|---|---|
| `env_for(k)` / `eval_env_for(k)` | cycle k's fact batch; explore/eval splits disjoint by construction (`B2`) |
| `verifier()` | `B2`'s programmatic exact-match verifier |
| `rehearsal_envs()` | the anchor suite - system-prompt adherence, refusal behaviour, tool-call format. This is the shortcut-defeating role `continual.rs:187-197` documents, and it doubles as sven's `S4` third layer of defence-in-depth against a poisoned fact |
| `write_sft_dataset()` | `B1` |
| `sft_mask_before()` | `None` - the token mask file (`B1`) supersedes character-offset masking |
| `shape()` | fixed `(prompt_len, completion_len)` across cycles - mandatory (`continual.rs:180-186`) and a real constraint on how sven's extraction (`S7`) must phrase probes |

Driven by `run_study(Regime::Sft(SftConfig::default()))` - the only regime
brain has *measured* to accumulate, per the correction above.

**Control arms, mandatory, not optional:** Arm 0 baseline (untrained model,
zero-shot column `run_study` already produces), Arm 1 gated (the real run),
Arm 2 null-gate (`GatePolicy::CoinFlip`, `continual.rs:369`) - brain's own
convention is that a learning claim without a control arm doesn't count; the
existing `--null-gate` diff (measured ACC 0.271 vs 0.056 in the GRPO case)
is what licenses the claim that a gate carries information at all, and
`F1`'s document-learning claim needs the same discipline. Arm 3 (joint
oracle, `continual.rs:1204`) is post-MVP.

**Test-first:**
- `a_document_curriculum_runs_a_full_study_and_emits_a_retention_matrix`;
- `no_probe_answer_appears_in_any_trained_span` (the disjointness guarantee,
  asserted end to end through the real curriculum, not just in `B2`'s unit
  test).
**Commit boundary: two** - (i) the `Curriculum` impl + `write_sft_dataset`,
(ii) the `run_study` wiring + the report surface (`B8`'s per-fact rows hang
off this).

### B7 - wire the existing hot-swap into a live `brain serve`
`QwenResident.adapter: RwLock<Option<String>>` + `set_adapter` +
`Executor::evict` (`crates/cli/src/resident_llm.rs:282,314-316,577-598`) is
already built, pinned-safe (refuses to evict mid-flight-request), and unit-
tested via `continuous_train.rs::hot_swap_cycle` (`:47-64`) - but "zero
cycles have ever run unattended… against a live serving process" (this
file's own prior entry, `self-improve.md:280-297`), and the function is
currently `#[allow(dead_code)]`, unreachable from `main`.

Wire an opt-in watcher into `run_apis` (`crates/cli/src/run_cli.rs:629`):
keep a **concrete** `Arc<QwenResident>` handle alongside the type-erased
`Arc<dyn ResidentModel>`, since `set_adapter` is inherent, not on the trait
(`resident_llm.rs:314`) - the type-erased handle alone cannot call it.

**Test-first:**
- flag off ⇒ `run_apis` returns no watcher handle (unit);
- `a_promoted_adapter_changes_a_live_serve_response_without_restart` -
  start `brain serve --openai` on the tiny fixture with the watcher pointed
  at a temp adapter directory, hold a request in flight, drop a real
  promoted adapter in mid-flight, assert the in-flight request completes
  uncorrupted **and** the next request reflects the new adapter. This is the
  literal closing of the "never run unattended against a live process" gap.
**Commit boundary: two** - (i) keep the concrete `QwenResident` handle
alongside the erased one, (ii) the opt-in watcher itself - drop the two
`#[allow(dead_code)]` in `continuous_train.rs:47` in this commit, where they
stop being true.

### B8 - per-fact promote/reject reporting
`StudyReport::matrix_table()` (`continual.rs:596`) reports per-*cycle*
rows. A batch of N facts trained and gated together produces one verdict for
the batch - "20 facts in, promote" can silently mean 15 landed and 5 didn't,
and neither the user nor sven's ledger (`S6′`) can tell which without this.
Add per-fact rows: for each fact in a promoted cycle, did *its own* probe
flip pass/fail independent of the aggregate decision.

**Test-first:** `a_batch_promote_still_names_every_fact_that_did_not_land` -
construct a batch where the aggregate gate promotes but one fact's probe
still fails post-training, assert the report names it.
**Commit:** one.

### B9 - `brain document-study`, the local entry point (DONE)
`B5′` made the study real; nothing shipped could run one. `rl::document::
run_document_study` was reachable only from `crates/rl/tests/
document_study.rs`, so the sven+brain loop had no way to train, gate and
publish a document adapter without a whale graph in front of it. This
milestone is that command, and nothing more:

```
brain document-study --arch <name> --weights BASE --dataset FILE.json
                     --adapter-dir DIR --report FILE.json [...]
```

- reads `{"cycles": [[{fact, probe_question, expected_answer}, …], …],
  "anchors": […]}` through `deny_unknown_fields` structs (the dataset is the
  one input brain did not produce, so serde is the structural validator, and
  it runs before anything touches the base);
- runs the gated arm and the null-gate control arm via
  `run_document_study`, unchanged;
- **always** writes the JSON report - per cycle: the facts trained, the
  incumbent's and the candidate's pass rate on that cycle's frozen probes,
  the gate's own `p_value`/`effect_size`/reject cause, the null-gate arm's
  parallel row, plus `promoted`/`decision` overall;
- on a promote only, publishes the adapter `rl::improve::cycle` already
  wrote into `--adapter-dir` as `adapter-{n:06}.safetensors`, one past the
  HIGHEST version already there - named through `rl::improve::
  latest_adapter`, the same function `brain serve --watch-adapters DIR`
  calls, so producer and consumer cannot drift.

**Top-level and `--arch`-driven, deliberately - not `brain qwen3
document-study`.** `run_study` is generic over `M: model::Model` and
`DocumentCurriculum` names no model type, so the study is architecture-
agnostic machinery; hanging its only entry point off one model's verb tree
would tie a general capability to one consumer and invite the next consumer
to grow a second copy (the workspace's most expensive recurring defect).
It follows `brain bench eval --arch <name>` instead: one top-level command,
a `brain_arch` id selecting which `Model` impl it monomorphises for, and a
registry a new architecture joins by adding one row. Everything
architecture-specific is a two-method `StudyArch` impl (how that
architecture spells a LoRA overlay, how to widen its context); the base is
read through `ModelConfig::from_json` and overlaid by the generic
`continual::overlay_adapter`.

**What bounds that registry is the TOKENIZER, not the model.**
`DocumentCurriculum`'s `Env`/`Ver` associated types name
`data::qwen_tokenizer::QwenBpe`, and `data::chat::prepare_chat_samples`
takes it, so an architecture qualifies iff it loads an HF `tokenizer.json`
BPE and its config can carry a LoRA overlay - today `qwen3`, `qwen35`,
`qwen35moe`. Widening it past the Qwen tokenizer family is a change to
`rl::document`, not to this command.

**Per-fact rows are NOT here** - see `B8`. A per-fact verdict needs the
candidate arm's own `(task id, score)` pairs, and `continual::CycleRecord`
collapses those to means before the study report is built; reconstructing
them in the CLI would mean a second decode pass, i.e. a different
measurement presented as the gate's. The report names each cycle's facts
instead. `B8` is the milestone that threads the per-task scores through.

**`B3b` is on hold, not superseded and not abandoned in place.** Exposing
the gate as a `capability::Action` on qwen3's own manifest hits a real Cargo
cycle (`brain-qwen3 → brain-rl → brain-bench → brain-qwen3`), and the
product decision is that the core learning loop works with sven+brain alone,
with whale as an optional later scale-out. `crates/cli` legally depends on
both `brain-qwen3` and `brain-rl` (it is the top of the layering), so this
command needs none of that resolved. Whether `B3b` is worth its
crate-surgery is a decision to make when whale scheduling is actually
wanted - not a prerequisite for the local loop.

**Test-first:**
- `a_document_study_writes_a_report_and_publishes_an_adapter_only_on_promote`
  - the real binary on the tiny CPU qwen3 fixture, asserting the report's
  shape both ways and that a promote leaves exactly the file
  `rl::improve::latest_adapter` (the watcher's own lookup) adopts;
- `a_malformed_dataset_is_refused_before_any_training_starts` - an unknown
  triple field fails by name, with no report written;
- `an_unregistered_architecture_is_refused_and_names_the_registered_ones`;
- unit: the publish naming/versioning contract, the dataset boundary, and
  that every registry id is a real `brain_arch` id.
**Commit:** one.

### F1 - the flagship autonomous document-learning benchmark
The MVP acceptance test. One user instruction to sven ("learn what you can
from this document"); fully autonomous thereafter through ingestion (sven),
extraction (sven, `S7`), baseline eval (this milestone, brain), batch
assimilation (`B5′`, scheduled via whale's `W7`), gate (`B2`/`B5′`), publish
(whale's `W8`), hot-swap (`B7`), and re-eval (this milestone again, brain).

**Topology:** the tiny CPU qwen3 fixture already used by `continuous_
train.rs`'s own tests (`QwenConfig::tiny()`), a 3-node `whale-testnet`
cluster, graph = `lora_train` → `lora_gate` asserted onto two distinct
providers (`W6′`).

**Honest risk, stated up front and not discovered mid-implementation:**
brain has already measured that the GRPO-style single-cycle composition does
*not* accumulate capability at this model scale. `F1` is built on the SFT
regime that *was* measured to accumulate - on a *synthetic* curriculum.
Whether it accumulates on real natural-language document facts is the open
question `F1` exists to answer, including negatively. **The MVP acceptance
criterion is that `F1` runs end to end and produces a defensible, controlled
number - not that the number is positive.** If run 1 comes back negative,
the decisive knobs are already documented in this file: rehearsal weight,
probe count, loss mask, step budget - don't discover this only after
shipping and being surprised.

Per brain's own convention, **the headline numbers live in the assertion of
a test, not in a report file** - a number nothing checks is a number that
silently goes stale.

**Test-first:** the whole benchmark, asserting the pre-registered targets -
scored honestly whichever way they land, red until every dependency above
lands.
**Commit:** one, once every dependency is green.

## Deferred, roadmapped, not lost

- **`B4` (real LoRA-edit wall-clock/cost measurement)** - informs whale's
  pricing, not correctness; whale's pricing params are all marked
  PLACEHOLDER already (`whale/.agents/roadmap/pricing.md`), so there's
  nothing downstream ready to consume the number yet. Do it once pricing
  is real.
- **`B6` (real per-token top-K logprobs threaded through the live serving
  path to the OpenAI-compat API - currently hardcoded `logprobs: Value::
  Null` at `crates/apiserve/src/openai.rs:633`)** - this is a genuine
  decoder-surface change (the default greedy serving path returns token-ids
  only, no logits; `sample_from_topk_with_logprob` at `crates/model/src/
  serve.rs:238` is only ever called from `rl/rollout.rs` today, never the
  live scheduler), the single most expensive non-security item either draft
  of this plan considered, and `F1` doesn't need it - a programmatic
  verifier's pass/fail is the signal, not the model's self-reported
  confidence. Deferred alongside sven's `S2` (below) as a matched pair; do
  not build one without the other.
- **`B10` (additive `max_runtime_s` on `capability::ActionSpec`, sourced
  authoritatively into whale's node types)** - a document-level override in
  whale's `NodeTypeSpec` (`W1a`) is sufficient while brain serves exactly
  one model through this pipeline. Matters once several models with
  different real training budgets coexist.
