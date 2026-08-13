# dtype - roadmap

Docs-honesty program: `docs/**/*.md` accumulates real, measured performance
numbers (wall-clock times, throughput, speedups) as optimization passes land.
Those numbers are true of the specific machine/driver/checkpoint they were
measured on and go stale - or get copy-pasted as if they were general
promises - the moment anything changes. The fix is mechanical, not just
editorial discipline: a gate that denies any bare number next to a
performance unit or claim unless a human has reviewed and deliberately
excepted it with an inline comment.

Phased: A1 builds the gate and proves it fires (RED) on the current tree. A2
(separate task) drives the violation count to zero - rephrasing each claim to
point at `brain perf run`/`brain flops` instead of a fixed figure, or marking
a genuinely-illustrative number as a reviewed exception.

## A1 - perf-number gate

**Gate**: `scripts/gates/check-no-perf-numbers.sh`, modelled on
`scripts/gates/check-env-docs.sh`'s house style (scan, list every violation
as `file:line: matched text`, exit non-zero on any hit, clear fix
instructions in the failure message).

Scans `docs/**/*.md` and denies a bare number adjacent to a performance unit
or claim: `ms`, bare `s`/`min` (durations), `fps`, `tok/s`/`tokens/s`,
`GB/s`/`MB/s`, `TFLOP`/`GFLOP`, `p50`/`p95`/`p99` followed by an actual
value - all unambiguous, checked without further context. `%` and
`Nx`/`N×` speedup patterns are also denied, but only when nearby lines
(a 4-line-back/2-line-forward window, since this repo's prose wraps a
paragraph's "measured"/"speedup" cue word across several physical lines)
carry measurement vocabulary - this is what keeps "a 16× convolutional token
compressor" (an architecture constant) and "30% token masking" (a training
hyperparameter) from being confused with "measured 2.5–3.6× the throughput"
(a real result). Escape hatch: `<!-- perf-number: <reason> -->` on the same
line or the line immediately before it.

**Makefile wiring**: added to `check/scripts` in the root `Makefile`,
alongside `check-scripts.sh`/`check-env-docs.sh`/`check-no-doc-citations.sh`
(same `bash scripts/gates/<name>.sh` invocation pattern); the `make help`
line for `check/scripts` updated to mention it.

**RED baseline on the current tree** (this is what A2 must drive to zero -
counts are real hits from a live run, not estimates):

| file | violations |
|---|---|
| `docs/performance/overview.md` | 209 |
| `docs/models/deepseek-ocr.md` | 36 |
| `docs/performance/hardware-notes.md` | 3 |
| `docs/scaling/data-parallel.md` | 2 |
| `docs/scaling/pipeline.md` | 1 |
| `docs/models/asr.md` | 1 |
| **total** | **252** |

`docs/models/asr.md`'s one hit ("default 30s" fixed transcription window) is
a capability limit rather than a measured result - left flagged rather than
gate-refined around, since a human still has to decide "rephrase" vs.
"escape-hatch" for it in A2; over-fitting the regex to one observed line
would just move the false-positive risk somewhere else.

`make check/scripts` currently fails **before** reaching this gate, on
`check-env-docs.sh` (`BRAIN_WGPU_SERIAL`/`BRAIN_WGPU_NO_SERIAL` undocumented)
- a pre-existing, unrelated failure this task did not introduce and does not
fix. Run the new gate directly to see it fire:
`bash scripts/gates/check-no-perf-numbers.sh`.

**Known imprecision**: the `%`/`Nx` context-gate is a heuristic (nearby
measurement vocabulary), not a parser. It is verified to correctly exclude
the false positives found during development (`16×` downsample factor, `1x`
render-quality option, `4x` checkpoint scale, `30%` token-masking
hyperparameter) and to correctly include every genuine measured claim in the
five files above. Any further false positive found in A2 belongs on the
escape-hatch list or as a targeted gate refinement - never a reason to weaken
the pattern wholesale.

**What's left (A2, separate task)**: rephrase or escape-hatch all 252
violations above so `check-no-perf-numbers.sh` - and therefore
`make check/scripts` - passes clean.

## A4 - device grammar unification

**Problem.** Two independent parsers for `--device`/`BRAIN_DEVICE` existed:

1. The **strong** parser - `DeviceSpec::parse` + `resolve`
   (`crates/gpu-core/src/devices.rs`): the full grammar (`gpu[N]`, `npu[N]`,
   `cpu[N|A-B]`, `vulkan`, `wgpu`, comma-unions). `crates/cli/src/main.rs`'s
   `select_backend` used this for the `--device` CLI flag.
2. The **weak** parser - `gpu_core::resolve_backend_name`'s bare
   `"cpu"`/`"vulkan"` string match, defaulting everything else to `"wgpu"`.
   Every NON-CLI caller took this for a bare `BRAIN_DEVICE` - every test
   binary, every `cargo test` invocation, any library caller that never went
   through `select_backend`.

So `BRAIN_DEVICE=gpu0 cargo test` silently meant "wgpu, whatever the ambient
card is" - NOT gpu0 specifically. Same for `npu`, `cpu0-7`, `wgpu`: all
silently mangled to the weak parser's `"wgpu"` fallback, with no ambient GPU
pin (`gpu_core::devices::set_ambient_gpu` was never called on that path).

**Fix.**

* `ComputeSet::apply` split into `apply_backend` (backend selection + ambient
  GPU pin - side-effect-light, safe from a library/test context) and `apply`
  (`apply_backend` + the CLI-only rayon pool sizing / `sched_setaffinity`
  core pinning). A library caller building a `Gpu` must never trigger
  process-wide thread-pool/affinity side effects just by resolving a device.
* New `gpu_core::ambient_compute_set() -> &'static ComputeSet`
  (`crates/gpu-core/src/devices.rs`), `OnceLock`-backed: returns whatever the
  CLI published via `publish_compute_set`, or - lazily, memoized for the
  process lifetime (same treatment this module already gives
  `BRAIN_GPU_INDEX`) - resolves `BRAIN_DEVICE` through the STRONG grammar,
  resolves against `Inventory::probe()`, and calls `apply_backend()` (never
  the CLI-only `apply()`). A malformed/unresolvable value (bad grammar, an
  out-of-range index, an NPU request with none present, …) prints ONE
  warning to stderr and falls back to the default all-devices set - never
  panics, never silently reinterprets the token.
* `resolve_backend_name`'s weak ladder deleted; it now reads
  `ambient_compute_set().backend` when no backend was explicitly selected via
  `set_default_backend`.
* `crates/cli/src/main.rs`'s `select_backend`: an explicit `--device` flag
  still takes precedence and is resolved (and hard-exits on a bad value, same
  as before) directly; with no flag, `BRAIN_DEVICE` is resolved by
  `gpu_core::ambient_compute_set()` itself (the CLI no longer re-reads the env
  var - ONE resolution, one reader). `set.apply()` (full, CLI-only semantics)
  still runs either way. The resolved set is published into gpu-core's shared
  `OnceLock` (`gpu_core::publish_compute_set`); `crates/cli`'s own
  `compute_set()` now just reads that back
  (`gpu_core::published_compute_set()`) instead of keeping a private,
  independently-populated static.
* `brain npu`'s own `--device` (an unrelated OpenVINO target-device string -
  `NPU`/`CPU`/`GPU`/`AUTO`) renamed to `--ov-device` to remove the flag-name
  collision with brain's own `--device` grammar. `--device` remains a
  deprecated alias for one release: `main.rs`'s `select_backend` translates
  `brain npu … --device X` to `--ov-device X` in place (printing a one-line
  deprecation note to stderr) before the generic `--device` loop runs, so an
  old invocation reaches `npu_cli`'s OpenVINO parsing rather than being
  silently swallowed as (or worse, misinterpreted as) a brain compute-device
  request. `npu_cli.rs`'s own `NpuOpts::parse_flag` also accepts bare
  `--device` directly (with the same deprecation note) for callers that
  bypass that translation. The `argv.get(1) == Some("npu")` bypass that used
  to skip `select_backend` entirely for `brain npu …` is deleted - the
  subcommand now goes through the same `--device`/`BRAIN_DEVICE` resolution
  path as everything else.

**Tests.** `crates/gpu-core/tests/device_grammar.rs`: a table test asserting
`--device <token>` and `BRAIN_DEVICE=<token>` (nothing published) resolve to
IDENTICAL `ComputeSet`s, adapted at run time to the real machine's GPU/CPU/NPU
counts (confirmed RED first, by temporarily reproducing the old weak-ladder
behaviour - mismatched for `gpu0`/`cpu`/others, as predicted); a test that an
indexed `BRAIN_DEVICE=gpu<N>` pins that specific ambient GPU, not just the
backend; a test that an out-of-range index falls back to all-devices with
exactly one stderr warning and no panic. Because `ambient_compute_set()`
memoizes per process, each distinct `BRAIN_DEVICE` case runs in its own
subprocess (the test binary re-execs itself against an `#[ignore]`d helper
test that prints the resolved state).

`scripts/gates/check-device-env-single-source.sh` (wired into
`make check/scripts`): greps that `BRAIN_DEVICE` is read via `std::env::var`
in exactly one file under `crates/gpu-core/src/` and nowhere else under any
`crates/*/src/`. One documented, accepted exception:
`crates/cli/src/qwen_cli.rs`'s `want_npu()` re-reads `BRAIN_DEVICE` as a
belt-and-suspenders check because the `NPU_REQUESTED` sidecar it also
consults isn't fully trusted yet - deleting that sidecar is a later phase
(referred to elsewhere as phase C3), not this one.

**Known follow-up risk (not fixed here, out of scope for A4).** A handful of
other crates' tests mutate `BRAIN_DEVICE` per-test via
`std::env::set_var("BRAIN_DEVICE", "cpu")` before building a `Gpu`
(`crates/{sam1,clip,deepseekocr,deepseekv2,qwenvl,fastvlm}`'s test/parity
code), relying on the old per-call-fresh env read. `ambient_compute_set()`'s
process-lifetime memoization (an intentional, documented design choice -
matching the precedent this module already set for `BRAIN_GPU_INDEX`) means
the FIRST resolution in a process now wins for the rest of that process. This
is only a real problem for a test binary that mixes expectations of two
DIFFERENT backends within one process; every sampled call site here sets the
same value (`"cpu"`) throughout its own file. Flagged here rather than
silently left for someone to rediscover.

## A2 - docs sweep

Drove `scripts/gates/check-no-perf-numbers.sh` from **252 violations to 0**
(`bash scripts/gates/check-no-perf-numbers.sh` now prints "no unreviewed perf
numbers found"). Full program from the approved plan, not just the gate:

**Cut/rewritten, per file:**

- **`docs/performance/overview.md`** (872 → 93 lines, 209 → 0 violations).
  Kept the number-free methodology sections (profiling/roofline probe,
  runtime kernel selector, INT8 inference, `brain flops`). Deleted the
  ~800-line DeepSeek-OCR case-study / "a real example" section entirely
  (verified first that `.agents/roadmap/deepseek-ocr.md`'s Phase 8 entries
  already carry the same content in full, including the retracted
  `llama.cpp` 1.8x comparison - git history is the only archive for the
  deleted prose, nothing was moved). Line ~55's citation to
  `docs/performance/flops.md` was dead (the file never existed, confirmed via
  `ls docs/performance/`); repointed to `brain flops --help` rather than
  writing a stub, since the CLI help is the thing that can't go stale. Added
  a closing "Where the numbers are" section pointing at `make perf`,
  `make perf/compare`, `results/*.json`, and the committed perf gates under
  `scripts/gates/` (`qwen-serving-perf-gate.sh`, `forecast-perf-gate.sh`,
  `wm-perf-gate.sh`), plus a pointer to `.agents/roadmap/<model>.md` for
  session-specific investigation logs. Folded `hardware-notes.md`'s one
  durable idea ("a number measured on one machine is not a promise for
  yours") in as its own methodology section.
- **`docs/performance/hardware-notes.md`** - deleted (42 lines, two
  hedged single-machine illustrations that would always go stale). All
  references fixed: `docs/manifest.txt`, `docs/readme.md`, `AGENTS.md`
  (table row split into a methodology row + a session-findings row pointing
  at `.agents/roadmap/<model>.md`), and the two scaling docs below.
  `AGENTS.md` is not in this task's formally scoped file list but item 4
  explicitly named it as a reference to fix, so it got the same two-line
  edit rather than leaving a dangling citation in the pandoc build.
- **`docs/models/deepseek-ocr.md`** (36 → 0 violations). Stripped every
  wall-clock/percentage/speedup number from the "Hardware and limits"
  section down to the structural facts (KV-cache exists, split
  vision/decoder backend, what was optimized) plus a pointer to
  `.agents/roadmap/deepseek-ocr.md` for the real numbers. Kept the ~22 GiB
  resident-memory figure (a sizing requirement, not a throughput claim) under
  `<!-- perf-number: hardware requirement, not a throughput claim -->` -
  verified the gate does not flag it either way (GiB-as-size was never
  matched by the gate's unit list) but the escape hatch documents the intent
  explicitly per the task brief. Deleted the retracted "~1.8x faster than
  llama.cpp" claim entirely (confirmed the retraction first by reading
  `overview.md`'s old §"corrected" section and
  `.agents/roadmap/deepseek-ocr.md`'s Phase 8 conclusion, both of which say
  the 33.4s figure could not be reproduced) - did not replace it with the
  "corrected" ~parity number either, since the whole point of this sweep is
  zero measured numbers in `docs/`, not more-honest measured numbers.
  **Flagging for a human**: this model is served (`brain do`, HTTP, D-Bus)
  but `docs/models/deepseek-ocr.md` is **not listed in `docs/manifest.txt`**
  - its docs never reach the compiled PDF. Not fixed here per the task's
  explicit instruction to flag rather than fix manifest structure.
- **`docs/models/asr.md`** (1 → 0 violations, not one of the 12 numbered
  items but part of the gate's worklist). The "default 30s" Qwen3-ASR window
  is a fixed config default (`BRAIN_QWEN_ASR_WINDOW`), not a measured result
  - marked `<!-- perf-number: fixed capability limit (config default), not a
  measured result -->` rather than rephrased, since the number itself
  (the actual default) is exactly the information a reader needs and won't
  go stale the way a timing claim would.
- **`docs/scaling/data-parallel.md`** / **`docs/scaling/pipeline.md`**
  (2 → 0, 1 → 0). Replaced the hedged "roughly 1.3x-1.6x" / "about 0.8x"
  single-machine speedup ranges with a qualitative explanation of what
  determines speed plus a pointer to `docs/performance/overview.md`'s
  profiling tools. Checked the Makefile first for a dedicated multi-GPU
  scaling-speedup target - `make bench/scaling` exists but measures a
  different thing entirely (the L(N)=E+A·N^-alpha loss-vs-model-size scaling
  law, `crates/bench`), and `data-parallel.md`'s own "Current scope" section
  already says there is no single CLI flag yet for launching a multi-GPU
  data-parallel run directly - so no Makefile target was cited; the fix
  points at profiling your own configuration instead of a specific command
  that doesn't exist yet.
- **`docs/introduction/hardware.md`** - rewrote the "Why one binary runs
  everywhere" section per the corrected thesis: portable fp32 baseline +
  capability-gated fast tiers (bf16/f16/int8/q4, declared per kernel) with
  automatic fallback, not "brain avoids these features." Verified against
  `crates/backend-api/src/lib.rs`'s `NumericSupport`/`DType`: the capability
  struct and the tier-selection *machinery* are real and in the tree today,
  but `DType::promote` currently always returns `F32` regardless of what a
  device supports - so the doc is explicit that bf16/f16/INT8/q4 *compute*
  paths are direction/in-progress (Group B), not shipped. Kept "no atomics,
  no subgroups" as accurate current facts, reframed as a chosen baseline
  rather than the whole precision story.
- **`docs/introduction/what-is-brain.md`** - rewrote "One engine, three
  backends" to "three eager backends, plus a separate NPU path": wgpu/
  CPU-JIT/native-Vulkan as the three eager backends sharing one WGSL kernel
  source (matches `AGENTS.md`'s own "Three backends, one build, one API"
  description), wasm/browser reframed as a wgpu build target rather than a
  fourth backend, and the Intel NPU described as a separate whole-graph
  OpenVINO compiler path for a model subset.
- **`README.md`** - split the "runs identically on your GPU, your CPU, an
  Intel NPU, or inside a web browser" sentence: "identically" now applies
  only to GPU/CPU/browser (same WGSL kernels, genuinely true), with the NPU
  described separately as a per-model ONNX/OpenVINO export path.
- **`docs/using/configuration.md`** - corrected the false "the residency
  scheduler does not yet place jobs on the NPU on its own" claim. Verified
  against `crates/residency/src/place.rs`'s `pick_device` (NPU tried before
  GPU/CPU whenever `MemCost.npu > 0`) and every `crates/cli/src/resident_*.rs`
  that calls `MemCost::with_npu`: depth (ZipDepth), both ASR models
  (Nemotron, Qwen3-ASR), and all three forecasters (chronos2, fincast,
  kronos). Note: kronos's own inclusion is grounded directly in
  `resident_forecast.rs`'s `KronosResident::estimate`/`activate` (declares
  `with_npu` and handles `Device::Npu`), even though that same file's own
  module-doc comment at the top says kronos's NPU rollout is still a
  "follow-up" - the code is more current than that comment; flagging the
  stale module doc for whoever touches `resident_forecast.rs` next, not
  fixed here (out of this task's scope). Rewrote `--device npu` as
  constraining/forcing the schedulable device set, not the sole trigger for
  reaching the NPU.
- **`docs/models/glm/npu.md`** - corrected "`infer --device npu` does not
  yet exist": `crates/cli/src/glm_cli.rs` dispatches it to
  `npu::glm_decode::generate` today (fp32 greedy decode, real). Also found
  `brain glm export --int8` is implemented and tested
  (`crates/npu/tests/glm_onnx.rs::glm_onnx_int8_runs`) but `infer --device
  npu` hardcodes `int8: false` and exposes no `--int8` flag - so INT8 is
  reachable via `export` today, not via `infer` end to end. Doc now states
  both precisely instead of a blanket "does not exist."
- **`crates/perf/src/lib.rs`** (code comment, not `docs/`) - "four backends"
  → "three backends (plus a separate whole-graph NPU compiler path)",
  matching `crates/perf/src/scenarios/placement.rs`'s existing correct
  wording.
- **`.agents/rules/kernels.md`** (the root-cause fix) - the "meta-rule"
  section used to instruct contributors to write cross-model findings
  *into* `docs/performance/overview.md`, which is exactly why the deleted
  800-line log accumulated there. Redirected: the generalizable
  finding/lesson goes in this file (or `.agents/rules/lessons.md` if
  cross-cutting); the session log with real numbers goes in
  `.agents/roadmap/<model>.md`; `docs/` gets zero measured numbers, ever.
  Also fixed the header's claim about what `docs/performance/overview.md`
  contains now that it's methodology-only.

**Judgment calls (item 12, the "fp32-only/no f16" sweep):**

- Grepped `docs/**/*.md` for `fp32 only`/`no f16`/`no bf16`/`no atomics`/`no
  subgroup`. Beyond `hardware.md` (fixed above), found two more hits, both
  left unchanged: `docs/models/moe.md:66` ("fp32 only" - a true, model-
  specific capability statement about the Sparse MoE Transformer today, not
  the "avoids as a virtue" framing item 1 was about) and three hits in
  `docs/reference/kernels.md` (per-kernel "no atomics" implementation notes
  in an apparently generated reference table, describing a specific kernel's
  scatter-vs-gather strategy, not a portability pitch). Neither is the
  pattern this sweep targets.
- `.agents/rules/kernels.md:118` ("No atomics, no subgroups, no f16") and
  `:133` ("fp32 only" in the storage-buffer/workgroup-size constraint list)
  - left unchanged. Both read as **current, accurate constraints for anyone
  writing a kernel today** (zero dispatch infrastructure exists yet for
  anything but fp32 compute), not the "sells a restriction as a virtue"
  framing problem item 1 fixed in `hardware.md`'s prose. Recommending a
  human decide whether these need the same "portable baseline + gated tiers"
  reframing once Group B's bf16/f16 kernel work actually lands - reframing
  them now, before any kernel can use anything but fp32, would overclaim in
  the other direction.

**Verification**: `bash scripts/gates/check-no-perf-numbers.sh` → 0
violations (252 → 0). `make docs` (pandoc + xelatex) builds clean:
`build/docs/brain-docs.{md,pdf}` written, 0 overfull hboxes, no dangling
citations. Grepped the whole repo for `hardware-notes` and
`performance/flops.md` - both clean of dangling references outside this
file's own historical A1 baseline table (left as-is, it documents what the
gate found at the time, not a live link).

## C1 - delete GraphBackend

**Deleted.** `backend_api::GraphBackend` (`crates/backend-api/src/lib.rs`) -
the whole-graph compile-then-run trait meant to abstract the OpenVINO NPU
path, parallel to the eager per-dispatch `Backend` trait wgpu/CPU-JIT/native-
Vulkan implement:

```rust
pub trait GraphBackend: Sized {
    type Config; type Output; type Error: std::error::Error;
    fn compile(onnx: &[u8], cfg: &Self::Config) -> Result<Self, Self::Error>;
    fn run(&mut self, input: &[f32], shape: [usize; 4]) -> Result<Self::Output, Self::Error>;
    fn device(&self) -> &str;
}
```

Along with its sole implementation, `impl backend_api::GraphBackend for
NpuSession` (`crates/npu/src/openvino/mod.rs`), and its sole call site
(`crates/npu/src/decode.rs`, `detect_weights_on_npu`).

**Why this is progress, not a regression.** A `git log` reader seeing NPU
code shrink and its dependency on `backend-api` disappear (the direct
`brain-backend-api` dependency edge in `crates/npu/Cargo.toml` is gone too,
since nothing under `crates/npu/src` referenced it once the trait impl and
its call site were gone) could mistake this for backing off the "make the
NPU a first-class backend" goal. The opposite is true:

- The trait was never object-safe (`Sized` + associated types forbid
  `Box<dyn GraphBackend>`), so it could never be used polymorphically the way
  `Box<dyn Backend>` is - it was fake integration, a trait that LOOKED like
  part of the backend abstraction but could not actually participate in it.
- It had exactly one implementation and exactly one call site - grepped the
  whole workspace (`grep -rn GraphBackend crates/`, excluding other agents'
  `.claude/worktrees/` checkouts) before deleting anything; confirmed no
  other site depended on it.
- Its `run(&[f32], [usize; 4])` signature is NCHW-vision-shaped and cannot
  express any LLM decode session's, ASR session's, or forecast session's real
  I/O (named multi-tensor inputs/outputs of varying rank), so it was never
  going to generalize past the one YOLO caller it had.
- The real generalized runner already exists one layer down and needed none
  of this: `NpuGraph` (`crates/npu/src/openvino/real.rs`) with
  `compile_bytes`/`compile_path`, `input_names()`/`output_names()`,
  `run(&[(&str, Feed)]) -> Vec<(String, Vec<usize>, Vec<f32>)>` and
  `Feed::{F32, I64}` - object-safe-shaped and already handling named,
  multi-tensor, variable-rank I/O. Every other NPU session type in the
  codebase (`DecoderSession`, `Chronos2Session`, `FincastSession`,
  `KronosS1Session`/`KronosS2Session`, `EmbedSession`, `LfmSession`,
  `CodecSession`, `KvSession`, `PrefillSession`, `BackStreamSession`,
  `FusedMtpSession`) already goes through `NpuGraph` or its own direct
  inherent methods, never through `GraphBackend` - `NpuSession`'s trait impl
  was genuinely isolated.

Deleting a trait that could not be used polymorphically, had one caller, and
was superseded by a real generalized runner INCREASES the NPU's actual
first-classness: what's left is code that either really is the eager
`Backend` contract (untouched) or really is the NPU's own concrete,
purpose-built session/graph API (`NpuSession`, `NpuGraph`, and the rest of
`openvino::real`), with no more trait that only pretended to be an
abstraction layer.

**Mechanical rewrite.** `decode.rs`'s `detect_weights_on_npu` called
`<NpuSession as backend_api::GraphBackend>::compile(&bytes, npu_cfg)?`; the
trait impl's own `compile` body was already just `NpuSession::load_bytes(onnx,
cfg)`, so the call site now calls `NpuSession::load_bytes(&bytes, npu_cfg)?`
directly - same inputs, same `Result<NpuSession, NpuError>`, one layer of
indirection removed. `crates/npu/Cargo.toml`'s now-unused
`brain-backend-api.workspace = true` dependency line was also removed
(confirmed via `grep -rn backend_api crates/npu/` returning nothing once the
impl and call site were gone).

**New structural test.** `crates/backend-api/src/lib.rs`,
`no_graph_concept::backend_api_names_no_graph_concept`: scans every `.rs`
file in the crate's own `src/` for a small banned-term list (the trait's
name, plus the two vendor/format identifiers its doc mentioned) and fails if
any file's source (case-insensitive) contains one. The banned-term array is
built with `concat!` fragments specifically so the array's own source text
never spells a banned word contiguously - otherwise the test would fail on
itself. TDD: written first against the pre-deletion tree (confirmed RED,
since the trait's own doc comment and body trip the scan) inside an isolated
scratch copy of the crate seeded from `git show HEAD:...` (the crate's real
`select.rs`/`lib.rs` in the working tree were mid-flight-broken by a
concurrent, unrelated DType-unification task at the time this test was
written, and `select.rs` is explicitly out of this phase's scope, so the
RED/GREEN cycle was proven in isolation rather than by touching or waiting
on that file); then the trait/impl were deleted and the same test went
GREEN. Re-ran for real once the concurrent task's `select.rs` became
buildable again: `cargo test -p brain-backend-api` ran 24 passed (0 failed),
including this test; `cargo build -p brain-npu` and `cargo test -p brain-npu
--lib` ran 19 passed, 1 pre-existing ignore; `cargo test -p brain-npu --test
npugraph` ran 1 passed (self-skips without real OpenVINO hardware, per its
own `npu_present()` guard).
