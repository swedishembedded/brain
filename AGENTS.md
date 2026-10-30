# AGENTS.md - brain

brain trains and serves neural networks on CPU, GPU and NPU from one engine:
pure Rust with WGSL and CUDA kernels, a Cargo workspace of small crates
under `crates/`, no Python in the build or test path. Backprop correctness is
gated by an in-repo finite-difference checker (`crates/gradcheck`), and every
imported model is parity-gated against its reference.

This file routes you to the file that owns a subject. It holds rules and
locations, not status: what is done, open or measured lives in the roadmap
ledgers (`.agents/roadmap/index.md`). `make check/agents-md` holds it to a line
budget and checks that every path it names exists.

## Repository scope

brain is the model engine.

- **In scope:** model architectures, kernels, inference, training algorithms
  (SFT/LoRA/GRPO/DPO/distillation), replay and continual-learning mechanisms,
  evaluation arithmetic, checkpoints and adapters, quantization, serving (CLI,
  HTTP, D-Bus), the public `brain` SDK and its samples.
- **Out of scope:** agent loops and tool execution, learning-campaign policy
  (what to learn next, which experiences qualify, release decisions for an
  application), fleet orchestration.

**Naming rule:** a tracked file names another project only if brain depends on
it. brain depends on no agent runtime or orchestration service, so describe a
caller generically ("an agent runtime", "an application", "an orchestrator").
The one exception is `README.md`'s introduction of brain's sibling projects.
Enforced by `make check/scope`; `CHANGELOG.md` is exempt.

## Where things are

| Path | What it is |
|---|---|
| `crates/` | the workspace. Layers, and what each crate is for: `.agents/rules/architecture.md` (its layer table is generated from `scripts/gates/crate-layers.map`) |
| `crates/arch` | the architecture registry (`ARCHS`): the one answer to "which architecture is this"; adding a model adds its row |
| `crates/kernels/wgsl/` | every WGSL kernel; catalogue in `docs/reference/kernels/index.md` |
| `crates/catalog` | every served model: manifest, provider and residency adapter in one list |
| `crates/sdk` | the public `brain` library; `crates/cli` is the `brain` binary over it |
| `docs/` | user-facing product documentation (`docs/index.md`); per-model pages in `docs/reference/models/index.md`; CLI in `docs/reference/cli.md`; configuration and environment in `docs/reference/configuration.md` |
| `.agents/rules/` | how to work here (table below) |
| `.agents/roadmap/` | status, open work and measured numbers, one ledger per model or workstream; start at `index.md` |
| `.agents/knowledge/` | defects already paid for, one finding per file, cited as `#NNN`; start at `index.md` |
| `samples/` | standalone Rust samples on the SDK or engine crates, plus Python and shell clients; contract in `samples/README.md` |
| `scripts/` | repo automation, run only by a make target or a bats test; `scripts/gates/` holds the quality gates |
| `tools/` | developer utilities run by hand (golden dumpers, converters) |
| `tests/` | bats suites: `tests/e2e` (real sockets and processes), `tests/gates` (gate specs) |
| `brain-py/` | Python client driving the `brain` binary as a subprocess |
| `papers/` | research papers, one scope each (`papers/README.md`); `make papers/<name>/pdf` builds one into `build/papers/` |

## Rules to read before you start

| Before you... | Read |
|---|---|
| design anything, or decide which crate owns a change | `.agents/rules/architecture.md` |
| port a model | `.agents/rules/porting.md`, then `serving-contract.md` and `sdk-design.md` |
| write, dispatch or optimize a kernel, or chase a slow path | `.agents/rules/kernels.md` (section F is the optimization loop) |
| write a test, a fixture or a gradient check | `.agents/rules/testing.md` |
| serve a model or add a capability | `.agents/rules/serving-contract.md` |
| add or change a public SDK type, CLI verb or Python surface | `.agents/rules/sdk-design.md` |
| touch `crates/apiserve` or `crates/dbus` | `.agents/rules/api-security.md` |
| bring in third-party code or weights | `.agents/rules/licensing.md` |
| write or change a paper, or change a finding one states | `.agents/rules/papers.md` |
| design a gate, or call a number an architecture's ceiling | `.agents/knowledge/index.md` |

## Commands

Build and test through the Makefile, never raw `cargo`: the targets share one
`./target` and the flags the project expects. `make help` lists every target.

```bash
make build/debug | build/release     # the brain binary
make test                            # suite; BRAIN_DEVICE=cpu runs it GPU-free
make test/full                       # suite + every gate (run before calling work done)
make test/gradcheck                  # backprop gate
make test/parity                     # cross-backend parity (CPU == Vulkan == NPU == CUDA)
make test/one P=<crate> T=<test> ARGS=...  # the edit-test loop: one target, seconds per rebuild
make check                           # lists the individual gates
make generate                        # lists the generators (kernel registry and catalogue, indexes)
make samples/<category>/<name>/run   # one standalone sample
```

`./target/release/brain <arch> <verb>` runs a model; `brain caps` lists every
capability. `--device` chooses where work runs (`cpu`, `gpu`, `gpu0,gpu1`,
`cpu0-7`, ...; omitted, every device): grammar in
`docs/guides/hardware/choosing-hardware.md`, parsed in
`crates/gpu-core/src/devices.rs`. `--trace-<family> <0-5>` turns on tracing
(`crates/trace`).

## Make targets

Every public target is `make <concept>/<subconcept>/.../<action>`: the area
first, the verb or variant last, lowercase `[a-z0-9-]` segments, one segment
per directory level for a tree-shaped set. Every prefix is itself a target (an
area runs its default set or only lists its children), a set that follows from
the tree is generated rather than hand-listed, and every target carries a
`## name - description` line under a `##@ Section`. A rename updates every
reference in the same commit, with no alias. `make check/make-targets` (also a
pre-commit hook) enforces it; the Makefile's header holds the full rule.

## Conventions & invariants

Each rule is one line here; the linked file holds the detail and the gate
enforces it.

- **Write down what you learned, in the same change.** A cross-cutting defect
  goes to `.agents/knowledge/<NNN>-<slug>.md` (`make generate/knowledge/index`),
  a kernel rule to `kernels.md`, a porting step to `porting.md`, a number about
  one model into the test that asserts it, open work into that model's roadmap
  ledger. A commit message is not a home for a finding.
- **`docs/` is user-facing product documentation.** It changes only when a
  public contract (command, flag, variable, supported capability) changes:
  never a status ledger, plan, measurement log or "known gaps" section there.
- **Status lives in `.agents/roadmap/`, nowhere else.** Guides and rules point
  at a ledger; they do not restate it.
- **Zero compile warnings,** including ones your change did not cause. Fix them
  properly; silence one only when the construct is intentional, with a comment
  saying why. `make check/clippy` gates the warning count.
- **One implementation.** Search before writing anything. Device math is a WGSL
  kernel, host math is `model::hostmath`, CPU parallelism is `backend_cpu::par`
  (no other crate depends on rayon), test filler is `data::rng::Lcg`, shared
  blocks are `model::block`/`model::vit`, a host `&[f32]` upload is
  `Gpu::write_f32`, image handling is `crates/imaging`. Never wrap a shared
  function in a local alias. Exceptions: a gradcheck oracle and a backend fast
  path validated against WGSL, each saying why in a comment.
- **One GPU device per process.** Share it (`Gpu::share`, `Gpu::new_like`,
  `share_or_new`); tests use `gpu_core::testgpu::dev`.
- **Host math does not run on the accelerator.** It ignores `--device`; a hot
  path needs a kernel.
- **WGSL is the source of truth and the reference provider.** Kernels live only
  in `crates/kernels/wgsl/`, declare their catalogue row in their own header,
  stay core-compute (fp32 arithmetic, at most 8 storage buffers, no atomics,
  subgroups or f16), and gate wider workgroups on queried `DeviceCaps`. Faster
  providers (tensor cores, CUDA, cooperative matrix) are selected through the
  `gpu_core::provider` seam and gated against WGSL. See `kernels.md`.
- **Four GPU backends, one API.** `gpu-core` exposes one `Gpu` surface over
  `backend-wgpu` (default), `backend-cpu` (WGSL JIT), `backend-vulkan` and
  `backend-cuda`, chosen at run time. There is no per-backend model code. The
  Intel NPU is a separate whole-graph export path (`crates/npu`), not a backend.
- **Weight-only quantization is group-wise** (`model::int8::GROUP`), never
  whole-channel. See `kernels.md`.
- **Profile with the driver's own tools.** On NVIDIA systems profiling or
  optimizing brain uses `nsys` (timeline) and `ncu` (kernels); elsewhere the
  vendor's equivalent. See `kernels.md` section F.0. Profile a kernel with
  `ncu` on its own test or bench binary at the production shape (seconds),
  never through a full model server (minutes per run); `nsys` on the server
  measures the before/after of the whole step.
- **Feedback is fast.** Iterate on one focused test (`make test/one`),
  run the broad gates once before committing, and move a test that takes
  minutes to the slow lane rather than paying for it on every run.
- **Fix selection, not copies.** Before writing a kernel, look for a faster
  sibling; before dispatching one, read its `Params`; before optimizing,
  profile per kernel kind. A finding from an audit or sweep is a hypothesis
  until checked against the source.
- **HARD RULE: brain runs at its roofline on every device, by selection, not by
  hand-tuning.** Running efficiently on any hardware is brain's main promise, so
  performance never depends on someone re-optimizing for a new card. Every hot
  op registers its implementations as candidates that declare the capabilities
  they need; brain measures the device's roofline (peak bandwidth and compute
  per precision/tensor-core path, by microbenchmark, not from a spec table),
  times the supported candidates per op, weight tier and shape class, checks
  each against the WGSL reference before it may be chosen, and caches the
  choice per device and build. Models only call the op: no device checks or
  hard-wired kernel choices in model code. Each chosen kernel's achieved
  fraction of its roofline bound is recorded and listed, and ops below
  threshold are reported as that device's gap list; kernel work is driven by
  that list and lands as a new candidate, never as a device-specific path. A
  gate enforces candidate coverage for every hot op and weight tier.
- **Backward plus a gradient check is the default for every model;** imported
  models are also parity-gated against goldens. See `testing.md`.
- **A model is done when it is served and usable as a library:** capability,
  residency adapter, real batching, D-Bus reachability and a sample
  (`serving-contract.md`), and one pipeline type per task through the SDK that
  the CLI, D-Bus and Python call too (`sdk-design.md`).
- **A capability is an `Action` in a `Provider`, never a new subcommand.**
- **A file location on the serving machine is the host's fact:** a
  `.host_resolved()` param resolved from the model store, never a plain one and
  never a new variable. Served models are named `<vendor>/<repo>[-<QUANT>]`
  (`crates/modelref`). See `serving-contract.md`.
- **The HTTP and D-Bus API stays watertight:** every change audits the whole API
  and adds automated security tests. See `api-security.md`.
- **Validate everything entering brain at the boundary,** structurally (typed
  `serde` structs with `deny_unknown_fields`, no `Value` indexing with
  defaults) and semantically (references resolve, sequences make sense). Fail
  loudly, naming the record and field; never coerce or drop.
- **No machine-specific absolute paths in tracked files.** Fixtures come from
  the gitignored `testdata/` (`make testdata/fetch`, `brain_testutil`), weights
  from a flag or the model store. `make check/paths`.
- **Settings are flags, not environment variables.** No new env reads
  (`make check/env-reads`) and no env writes in library code
  (`make check/env-writes`); both allowlists only shrink.
- **History stays linear:** self-contained commits, rebased, never merged
  (`scripts/gates/check-linear-history.sh`, also run before a push).
- **Every default seed is random** (`make check/seed-literals`).
- **Every backward takes the saved-activation policy** (`model::saved`:
  `set_saved_policy`, `BlockStack`, the block tape's `Activations`), or the
  crate is listed with its reason in `scripts/gates/saved-policy.allow`
  (`make check/saved-policy`).
- **Never skip a test silently:** `brain_testutil::skip` or `skip_unavailable`,
  never `eprintln!` plus `return` (`make check/test-skips`).
- **Gitignored:** `scratchpad/`, generated `data/`, `out/`, `build/`,
  `results/`, `testdata/`.

## Local task management

Tasks may be handed to you as Markdown files with frontmatter in a local,
gitignored task folder outside version control.

- **Pick up:** read the named file (or the oldest one matching the
  description), set `status: pending` to `status: in_progress`, and stay
  within its "Objective" and "In-Scope".
- **Complete:** when every item is built, tested and verified, commit the work
  as self-contained commits, then move the file into a `completed/` subfolder
  of that folder. Never delete a task file. A partly done task stays where it
  is, with done and remaining items recorded in it.
