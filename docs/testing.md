# brain — testing plan

The goal is to evaluate every architecture **reliably against the same input
data**, and to keep the from-scratch WGSL backprop provably correct without a
PyTorch oracle. Coverage is layered:

## 0. Test inputs: one PRNG, `data::rng::Lcg`

Tests never depend on `rand`. Deterministic filler comes from
**`data::rng::Lcg`** (`Lcg::new(seed)` → `signed()` `[-1,1)`, `unit()` `[0,1)`,
`scaled(a)` `[-a,a)`, plus the `vec*` bulk forms) — and from nowhere else.
`data::rng::Rng` (SplitMix64) is a *different* generator and belongs to the
on-disk dataset generators; its stream must not move.

The reason `Lcg` exists as a shared type rather than a per-file helper: the
copied helper was
`((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0`, and `u64 >> 33` keeps only
31 bits — so it returned `[-1, 0)` and **no test ever fed a positive value to
an activation kernel**. Three files had independently rediscovered and locally
patched that (`prelu_kernels.rs`, `convtr2d_kernels.rs`, `wm-core/tests/*`); ~40
had not. `Lcg::signed` shifts by 32 and straddles zero.

Consequence to remember when a fixture changes: goldens generated from this
stream move with it. `tools/goldens/vit_dump_gradcheck.py` replicates the Rust f32 op
order exactly and must be kept in step with `Lcg`.

## 1. Per-crate unit tests (`cargo test`)

| Crate | What's covered |
|---|---|
| `kernels` | all 53 WGSL kernels present + non-empty; `src()` lookup round-trips |
| `data` | RNG determinism/range; `meta.json` round-trip; char + **GPT-2 BPE** tokenizers (BPE pinned to exact vectors: `"hello world"`→`[31373,995]`, lossless round-trip); every generator (line ≤64 chars, one `=`, count, determinism; `number_to_words`); loader masking + line-aligned sampling; `prepare()` produces valid datasets |
| `gpu-core`/`paramstore`/`optim` | real GPU dispatch + readback; AdamW+clip vs hand computation |
| `moe`/`pid` | forward finite/deterministic; grads finite; (pid) param-list, config round-trip, inference timing |
| `gpt` | param shapes; config round-trip; forward deterministic & ≈ln(vocab); grads finite; **50-step overfit reduces loss**; end-to-end calculator training drops loss; cosine-LR schedule |
| `federated` | SHA-256 vectors; expert-id parsing; **split→assemble identity**; last-wins overlay |
| `gradcheck` | the correctness gate (below) |
| `eval` | perplexity definition sanity |

Run with `make test` (or `cargo test`); set `MOE_SKIP_GPU_TESTS=1` on a machine
with no GPU to skip the device-dependent tests.

### `testdata/` — fixture inputs and goldens, not a model store

Parity/import tests across ~20 crates (`fastvlm`, `moondream`, `qwenvl`, `nemotron`,
`qwen-asr`, `sam2`, `zimage`, `vae`, `clip`, `facenet`, `tts`, `codec`, `speaker`,
`vqgan`, `wm-genie`, `flux2`, `qwen`, `diffusion`, `npu`, `audio`) resolve their
fixtures through `brain_testutil::testdata(rel)` (one implementation, shared as a
dev-dependency — this used to be a byte-identical function copy-pasted into every
one of those crates). It resolves to `$BRAIN_TESTDATA` if set, else the gitignored
`<repo>/testdata/`.

**`testdata/` holds test inputs and goldens ONLY** — dumped-golden tensors a test
compares against and small input media (audio clips, images). It must never hold a
`.git` directory, runnable code (upstream source, notebooks, docs), or a model
checkpoint: `scripts/data/fetch-testdata.sh`, the one thing that populates it,
unconditionally strips `.git` and `.cache/huggingface` from everything it mirrors,
plus an extra exclusion list for trees whose mirror is a whole upstream checkout
(`vl_tree`'s `.py`/`.ipynb`/`.md`/`.pdf`/`.mp4`/`.pt` exclusion — none of
`fastvlm`/`moondream`/`qwenvl`'s tests read any of those).

**Real upstream checkpoints (`fastvlm`, `moondream`, `qwenvl`, `nemotron`,
`qwen-asr`'s parity/import tests) live in the model store, not `testdata/`** — the
same `<models-dir>/<vendor>/<repo>/` tree `brain fetch` writes and
`crates/modelstore` scans (see `docs/models/naming.md`). Tests resolve one with
`brain_testutil::model_dir("<vendor>/<repo>")`, which wraps
`brain_modelstore::default_root()` (`$BRAIN_MODELS_DIR`, else
`$XDG_DATA_HOME/brain/models`, else `$HOME/.local/share/brain/models`) — `None` when
unresolvable, which every call site turns into an empty path via `unwrap_or_default()`
so the existing `Path::new(&format!("{ckpt}/…")).exists()` skip check stays correct
either way.

Populate both with `make fetch/testdata` (hard-links from a local mirror —
`BRAIN_*_MIRROR` env vars, the ONE place a machine-specific path may appear in
this repo, per `AGENTS.md` — into `testdata/` for goldens/media, `$BRAIN_MODELS_DIR`
or its default for checkpoints). A test whose fixture is still absent **skips
itself** (`eprintln!` + early return, never `panic!`) — verify a change here by
removing `testdata/`, re-running `make fetch/testdata`, and re-running the
crates in the table above; a fixture that stopped resolving shows up as a new
skip, not a failure, which is itself the bug to look for.

## 2. Backprop correctness gate — numerical gradient check

`crates/gradcheck` replaces the dropped PyTorch oracle. For each parameter
tensor it compares the analytic WGSL directional derivative `⟨∇L, v⟩` to a
central finite difference `(L(w+εv) − L(w−εv))/2ε` (ε=5e-3), over several random
directions, with an `allclose`-style `|a−n| ≤ atol + rtol·max(|a|,|n|)` criterion
(atol=4e-3, rtol=8e-2 — fp32 on a software GPU). This validates the GPT's GELU
MLP, causal attention, LayerNorm, embeddings, and untied head across all 29
tensors. Run with `make gradcheck`. (Extending the `CheckModel` impl to MoE/PID
needs a `write_weight` on those models — a small follow-up.)

## 3. Federated round-trip & integrity

- `split → merge` reconstructs a checkpoint **tensor-for-tensor** (identity).
- Overlay assembly replaces exactly the targeted expert (last-wins), leaving
  others byte-identical.
- Manifests carry per-file SHA-256 + a base-config hash; `verify` rejects any
  tampering or config mismatch.
- `make federated-demo` exercises the whole pipeline on a real trained MoE.

## 4. Same-input model comparison (the "which is best" answer)

`crates/eval` provides one metric set applied identically across models:
- **Validation perplexity** = `exp(mean next-token CE)` on the val split — the
  architecture-agnostic number.
- **Task exact-match** for `LHS=RHS` datasets (calculator/reverser/wordcalc):
  greedily decode the RHS on a **held-out tail** and check string equality — the
  honest "did it learn the rule" metric (separate from perplexity, per README §3).

`make bench` trains + evaluates the GPT baseline on the shared char datasets with
a fixed seed/splits so runs are comparable.

## 5. Integration (Makefile, CI-time)

`make data/<name>` → `make train/gpt/<name>` → `make eval/gpt/<name>` runs the
full data→train→eval path; `make federated-demo` runs the federated round-trip;
`make gradcheck` runs the correctness gate. All complete on tiny configs quickly
enough for CI.

## 6. End-to-end (`bats`, real sockets/processes)

Layered above the cargo test suites: full processes, real sockets, real signals.
Nothing here needs GPU or trained weights unless a suite says so; each one is
independently runnable and skips honestly (never fails) when its prerequisite
tooling or weights are absent.

| Suite | `make` target | What it proves |
|---|---|---|
| `tests/e2e/api_conformance.bats` | `test/e2e/api-conformance` | The OpenAI/Anthropic/OpenRouter HTTP dialects, over a real socket, against the deterministic `BRAIN_MOCK=1` model — schema-validated against the vendored OpenAPI specs, plus the auth/DoS/error-hygiene security matrix. |
| `tests/e2e/shutdown.bats` | `test/e2e/shutdown` | `brain serve` actually exits on SIGINT/SIGTERM, for every surface combination (`--dbus` alone, `--dbus --openai` together, `--openai` alone) — each test starts and kills its own server. |
| `tests/e2e/examples.bats` | `test/e2e/examples` | Every example under `examples/` actually runs (against the mock, where the model it needs has a mock equivalent) or skips with a clear, honest reason — not silently. A completeness check fails the suite if a tracked example is missing from `tests/e2e/examples/manifest.tsv`, so a newly-added, never-wired example cannot rot the way every existing example did before this suite existed (see the git history around commit `38f384e`). |
| `tests/e2e/ready.bats` | `test/e2e/ready` | `brain serve --ready-file PATH` appears only after **every** requested surface (HTTP dialects + D-Bus) has bound its listener, and never at all when one fails — and therefore strictly after `--api-keys-out` is written, so a script can wait on one file and then read the keys with no retry. Covers a full bind, a failed bind, a partial bind, and D-Bus alone / D-Bus+HTTP together. |
| `tests/e2e/claude_code.bats` | `test/e2e/claude-code` | The real `claude` CLI working end-to-end against `brain serve --anthropic` (the deterministic `BRAIN_MOCK` model, so it needs no weights and never hangs on a cold fetch). Skips cleanly unless `claude`/`jq`/`timeout` and a brain binary are present. |
| `tests/e2e/scheduler.bats` | `test/e2e/scheduler` | Heavy, opt-in (`BRAIN_E2E=1`): residency scheduler batching/eviction, and the generate→detect→annotate pipeline, against **real** z-image + yolo weights and a GPU. Not part of the fast lane — `tests/e2e/examples.bats` runs the same generate/detect example against the mock instead. |

`make test/e2e` runs the four fast suites (api-conformance, shutdown, examples,
ready) in one shot; `make test/full` folds that into the release gate alongside the
cargo lanes. `claude-code` and `scheduler` stay separate targets — they need real
weights/a real `claude` install/a GPU, which the fast lane deliberately does not
require.

**Server lifecycle discipline**, followed by every suite above that starts a
process: record `$!` into a file immediately, poll readiness (never a fixed
sleep), and `teardown_file` kills **only** that recorded PID — never `pkill`. The
D-Bus suites additionally spin up a **private** `dbus-daemon` per run
(`dbus-daemon --session --fork --print-address --print-pid=3`) so nothing here
ever touches the real session/system bus.

## Known gaps (honestly tracked)

- **MoE / federated rows in `make bench`**: the MoE engine currently trains on
  its own 64-symbol rule task, not the char datasets, so the leaderboard covers
  GPT today. Training the MoE on the shared `data` crate datasets (and the
  frozen-backbone expert training) is the remaining federated-training
  integration.
- **Time-series model, DDP, YAML configs**: deferred from Phase 2b.
- Determinism is per-run with a fixed seed; cross-machine bit-identity is not
  guaranteed (fp32 GPU reductions).
