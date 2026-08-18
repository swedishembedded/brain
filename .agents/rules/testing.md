# brain - testing plan

The goal is to evaluate every architecture **reliably against the same input
data**, and to keep the from-scratch WGSL backprop provably correct without a
PyTorch oracle. Coverage is layered:

## 0. Test inputs: one PRNG, `data::rng::Lcg`

Tests never depend on `rand`. Deterministic filler comes from
**`data::rng::Lcg`** (`Lcg::new(seed)` → `signed()` `[-1,1)`, `unit()` `[0,1)`,
`scaled(a)` `[-a,a)`, plus the `vec*` bulk forms) - and from nowhere else.
`data::rng::Rng` (SplitMix64) is a *different* generator and belongs to the
on-disk dataset generators; its stream must not move.

The reason `Lcg` exists as a shared type rather than a per-file helper: a
copied helper elsewhere computed
`((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0`, and `u64 >> 33` keeps only
31 bits - so it returned `[-1, 0)` and no test that used it ever fed a
positive value to an activation kernel. Several files had independently
rediscovered and locally patched that; many more had not. `Lcg::signed`
shifts by 32 and straddles zero.

Consequence to remember when a fixture changes: goldens generated from this
stream move with it. `tools/goldens/vit_dump_gradcheck.py` replicates the Rust f32 op
order exactly and must be kept in step with `Lcg`.

## 1. Per-crate unit tests (`cargo test`)

| Crate | What's covered |
|---|---|
| `kernels` | every WGSL kernel present + non-empty; `src()` lookup round-trips |
| `data` | RNG determinism/range; `meta.json` round-trip; char + **GPT-2 BPE** tokenizers (BPE pinned to exact vectors: `"hello world"`→`[31373,995]`, lossless round-trip); every generator (line ≤64 chars, one `=`, count, determinism; `number_to_words`); loader masking + line-aligned sampling; `prepare()` produces valid datasets |
| `gpu-core`/`paramstore`/`optim` | real GPU dispatch + readback; AdamW+clip vs hand computation |
| `moe`/`pid` | forward finite/deterministic; grads finite; (pid) param-list, config round-trip, inference timing |
| `gpt` | param shapes; config round-trip; forward deterministic & ≈ln(vocab); grads finite; **50-step overfit reduces loss**; end-to-end calculator training drops loss; cosine-LR schedule |
| `federated` | SHA-256 vectors; expert-id parsing; **split→assemble identity**; last-wins overlay |
| `gradcheck` | the correctness gate (below) |
| `eval` | perplexity definition sanity |

Run with `make test` (or `cargo test`); set `MOE_SKIP_GPU_TESTS=1` on a machine
with no GPU to skip the device-dependent tests.

### `testdata/` - fixture inputs and goldens, not a model store

Parity/import tests across many crates (`fastvlm`, `moondream`, `qwenvl`, `nemotron`,
`qwen-asr`, `sam2`, `zimage`, `vae`, `clip`, `scrfd`, `arcface`, `tts`, `codec`, `speaker`,
`vqgan`, `wm-genie`, `flux2`, `qwen`, `diffusion`, `npu`, `audio`, `kronos`, `chronos2`,
`wan`, `s3dit`, `ltxv`, `gemma4`) resolve their
fixtures through `brain_testutil::testdata(rel)` (one implementation, shared as a
dev-dependency, rather than a byte-identical function copy-pasted into every one of
those crates). It resolves to `$BRAIN_TESTDATA` if set, else the gitignored
`<repo>/testdata/`.

**`testdata/` holds test inputs and goldens ONLY** - dumped-golden tensors a test
compares against and small input media (audio clips, images). It must never hold a
`.git` directory, runnable code (upstream source, notebooks, docs), or a model
checkpoint: `scripts/data/fetch-testdata.sh`, the one thing that populates it,
unconditionally strips `.git` and `.cache/huggingface` from everything it mirrors,
plus an extra exclusion list for trees whose mirror is a whole upstream checkout
(a `.py`/`.ipynb`/`.md`/`.pdf`/`.mp4`/`.pt` exclusion for the vision-language tree
- none of its consuming crates' tests read any of those).

**Real upstream checkpoints (`fastvlm`, `moondream`, `qwenvl`, `nemotron`,
`qwen-asr`'s parity/import tests) live in the model store, not `testdata/`** - the
same `<models-dir>/<vendor>/<repo>/` tree `brain serve`'s auto-fetch (and
`crates/modelstore::execute`) writes and scans (see `docs/models/naming.md`).
Tests resolve one with
`brain_testutil::model_dir("<vendor>/<repo>")`, which wraps
`brain_modelstore::default_root()` (`$BRAIN_MODELS_DIR`, else
`$XDG_DATA_HOME/brain/models`, else `$HOME/.local/share/brain/models`) - `None` when
unresolvable, which every call site turns into an empty path via `unwrap_or_default()`
so the existing `Path::new(&format!("{ckpt}/…")).exists()` skip check stays correct
either way.

Populate `testdata/` with `make fetch/testdata` (hard-links goldens and media from
a local mirror - `BRAIN_*_MIRROR` env vars, the ONE place a machine-specific path
may appear in this repo, per `AGENTS.md`). It does **not** download or copy the
store's checkpoints: it reports each one present or absent and names the
`brain fetch <vendor>/<repo>` that fetches it, because `$BRAIN_MODELS_DIR` is
usually on a different filesystem from any mirror, where the hard link fails and
a fallback copy would silently duplicate tens of gigabytes.
A test whose fixture is still absent **skips
itself**, and it does so through **`brain_testutil::skip(reason)`** - never a
bare `eprintln!` + early return, and never `panic!`. The helper exists because
cargo reports a skipped test as a PASS, so a skip that does not name itself is
indistinguishable from a comparison that ran; routing through it also lets
`BRAIN_REQUIRE_FIXTURES=1` (what `make parity/strict` sets) turn every
absent-fixture skip into a hard failure in a run whose purpose is to prove
parity. A skip for absent HARDWARE is the other helper,
**`brain_testutil::skip_unavailable(reason)`**, which no flag may turn fatal -
`BRAIN_REQUIRE_FIXTURES` asserts "the data is on this box" and has nothing to
say about a box with no discrete GPU, no NPU, no OpenVINO and no ffmpeg. Which
bucket a skip is in is a judgement a reviewer must be able to check by reading
the call, which is why there are two functions and not one flag.

Verify a change here by removing `testdata/`, re-running `make fetch/testdata`,
and re-running the crates in the table above; a fixture that stopped resolving
shows up as a new skip, not a failure, which is itself the bug to look for.

## 2. Backprop correctness gate - numerical gradient check

`crates/gradcheck` replaces the dropped PyTorch oracle. For each parameter
tensor it compares the analytic WGSL directional derivative `⟨∇L, v⟩` to a
central finite difference `(L(w+εv) − L(w−εv))/2ε` (ε=5e-3), over several random
directions, with an `allclose`-style `|a−n| ≤ atol + rtol·max(|a|,|n|)` criterion
(atol=4e-3, rtol=8e-2 - fp32 on a software GPU). This validates the GPT's GELU
MLP, causal attention, LayerNorm, embeddings, and untied head across all 29
tensors. Run with `make gradcheck`. (Extending the `CheckModel` impl to MoE/PID
needs a `write_weight` on those models - a small follow-up.)

## 3. Federated round-trip & integrity

- `split → merge` reconstructs a checkpoint **tensor-for-tensor** (identity).
- Overlay assembly replaces exactly the targeted expert (last-wins), leaving
  others byte-identical.
- Manifests carry per-file SHA-256 + a base-config hash; `verify` rejects any
  tampering or config mismatch.
- `make federated-demo` exercises the whole pipeline on a real trained MoE.

## 4. Same-input model comparison (the "which is best" answer)

`crates/eval` provides one metric set applied identically across models:
- **Validation perplexity** = `exp(mean next-token CE)` on the val split - the
  architecture-agnostic number.
- **Task exact-match** for `LHS=RHS` datasets (calculator/reverser/wordcalc):
  greedily decode the RHS on a **held-out tail** and check string equality - the
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
| `tests/e2e/api_conformance.bats` | `test/e2e/api-conformance` | The OpenAI/Anthropic/OpenRouter HTTP dialects, over a real socket, against the deterministic `BRAIN_MOCK=1` model - schema-validated against the vendored OpenAPI specs, plus the auth/DoS/error-hygiene security matrix. |
| `tests/e2e/shutdown.bats` | `test/e2e/shutdown` | `brain serve` actually exits on SIGINT/SIGTERM, for every surface combination (`--dbus` alone, `--dbus --openai` together, `--openai` alone) - each test starts and kills its own server. |
| `tests/e2e/examples.bats` | `test/e2e/examples` | Every example under `examples/` actually runs (against the mock, where the model it needs has a mock equivalent) or skips with a clear, honest reason - not silently. A completeness check fails the suite if a tracked example is missing from `tests/e2e/examples/manifest.tsv`, so a newly-added, never-wired example cannot rot the way examples used to before this suite existed. |
| `tests/e2e/ready.bats` | `test/e2e/ready` | `brain serve --ready-file PATH` appears only after **every** requested surface (HTTP dialects + D-Bus) has bound its listener, and never at all when one fails - and therefore strictly after `--api-keys-out` is written, so a script can wait on one file and then read the keys with no retry. Covers a full bind, a failed bind, a partial bind, and D-Bus alone / D-Bus+HTTP together. |
| `tests/e2e/claude_code.bats` | `test/e2e/claude-code` | The real `claude` CLI working end-to-end against `brain serve --anthropic` (the deterministic `BRAIN_MOCK` model, so it needs no weights and never hangs on a cold fetch). Skips cleanly unless `claude`/`jq`/`timeout` and a brain binary are present. |
| `tests/e2e/scheduler.bats` | `test/e2e/scheduler` | Heavy, opt-in (`BRAIN_E2E=1`): residency scheduler batching/eviction, and the generate→detect→annotate pipeline, against **real** z-image + yolo weights and a GPU. Not part of the fast lane - `tests/e2e/examples.bats` runs the same generate/detect example against the mock instead. |

`make test/e2e` runs the four fast suites (api-conformance, shutdown, examples,
ready) in one shot; `make test/full` folds that into the release gate alongside the
cargo lanes. `claude-code` and `scheduler` stay separate targets - they need real
weights/a real `claude` install/a GPU, which the fast lane deliberately does not
require.

`make test/full` also runs **`make parity/strict`**, and that is the one member
of the release gate that can fail for want of DATA rather than for want of
correct code. Everything else in the list is green on a box with no fixtures at
all, because cargo reports a skipped test as a PASS - so the cargo lanes cannot,
by construction, tell "every reference comparison matched" apart from "no
reference comparison ran". `parity/strict` re-runs the fixture-gated parity
suites with `BRAIN_REQUIRE_FIXTURES=1`, which turns each absent-fixture skip
into a hard failure, so a green run is evidence the comparisons happened. Its
default suite list is what `make fetch/testdata` provisions; where a suite
genuinely cannot be provisioned, narrow the list visibly
(`make test/full PARITY_STRICT_SUITES="..."`) rather than dropping the target,
because a narrowed list still says which comparisons were certified.

**Server lifecycle discipline**, followed by every suite above that starts a
process: record `$!` into a file immediately, poll readiness (never a fixed
sleep), and `teardown_file` kills **only** that recorded PID - never `pkill`. The
D-Bus suites additionally spin up a **private** `dbus-daemon` per run
(`dbus-daemon --session --fork --print-address --print-pid=3`) so nothing here
ever touches the real session/system bus.

## Test-only environment variables

These gate test fixtures, weight-required test paths, or benchmark knobs. None
of them are read by production serving code - they exist only so a test can
skip cleanly when its prerequisite isn't present, or so a benchmark can be
tuned from the environment instead of a recompile.

**Device / GPU selection for tests:**
- `BRAIN_DEV_GPU` - selects/forces a GPU device for a test run.
- `BRAIN_NPU_DEVICE` - selects the NPU device for npu-crate tests.
- `MOE_SKIP_GPU_TESTS` - skips device-dependent tests on a machine with no GPU.
- `SHARD_TEST_GPUS` - how many GPUs a pipeline/tensor-parallel sharding test should assume.

**Benchmark knobs:**
- `BRAIN_BENCH_REPS` - repetition count for a benchmark harness.

**Golden/fixture file paths:**
- `BRAIN_GGUF_TESTFILE` - path to a GGUF fixture for GGUF-reader tests. Also
  satisfied automatically by any `*.gguf` already in the model store, so the
  reader smoke does not need this set on a box that has fetched one.
- `BRAIN_INT8_TEST` - enables/points at an int8-specific test fixture.

**Model-weights-required test gates** (each enables a parity/import/training test
that needs a real checkpoint; unset means the test skips):
`BRAIN_EVA_CLIP`, `BRAIN_CONTROLNET`, `BRAIN_PULID`, `BRAIN_INSTANTID`,
`BRAIN_INSTANTID_CKPT`, `BRAIN_INSTANTID_CODE`, `BRAIN_T5ENCODER_XXL`, `BRAIN_ESRGAN`,
`BRAIN_FLUX1_FULL`, `BRAIN_FLUX2_TRANSFORMER`, `BRAIN_FLUX2_BATCH_LADDER`,
`BRAIN_FLUX2_BATCH_PRECISION`, `BRAIN_FLUX2_BATCH_REPS`, `BRAIN_S3DIT_I8`,
`BRAIN_LFM25_230M`, `BRAIN_LFM25_350M`, `BRAIN_QWEN3_4B`,
`BRAIN_QWEN35_SMOKE_GPUS`, `BRAIN_QWEN35_SMOKE_LAYERS`, `BRAIN_QWEN35_SMOKE_LR`,
`BRAIN_QWEN35_SMOKE_STEPS`, `BRAIN_QWEN35_SMOKE_T`, `BRAIN_QWEN3OMNIMOE_IMPORT_OUT`,
`BRAIN_MOONDREAM3_CKPT`, `BRAIN_QWEN3VL_CKPT`, `BRAIN_FASTVLM_CKPT`,
`BRAIN_FASTVLM_TEST_IMG`, `BRAIN_VL_PARITY_OUT`, `BRAIN_REF_RECT`,
`BRAIN_WAN_VAE`, `BRAIN_WAN_T5`, `BRAIN_WAN_TOKENIZER`, `BRAIN_WAN_GGUF` (a
released `city96/Wan2.1-*-gguf` file for `crates/wan`'s `gguf_import_real`
suite; falls back to whatever `*.gguf` the model store already holds for that
repo), `BRAIN_WAN_GGUF_OUT` (where that suite's `#[ignore]`d full conversion
writes its ~53 GiB checkpoint; a temp dir otherwise).

**`fetch-testdata` mirror paths** (local-mirror source for `make fetch/testdata`;
the one place a machine-specific path may appear in this repo). Two kinds:

- `BRAIN_MODEL_MIRROR` - a populated **model store** (`<vendor>/<repo>` layout) to
  hard-link out of, for the handful of checkpoints whose tests read them from
  `testdata/` rather than the store (the antelopev2 ONNX pair, the Qwen3-TTS
  checkpoint, SDXL's CLIP tokenizer). Defaults to `$BRAIN_MODELS_DIR` when set,
  since that IS the store on a configured box. It replaced the per-domain
  `BRAIN_SAM2_MIRROR` / `BRAIN_IDENTITY_MIRROR` / `BRAIN_UNET_MIRROR` roots, whose
  `<root>/<domain>/weights/…` layout predates the model store.
- `BRAIN_ASR_MIRROR`, `BRAIN_VL_MIRROR`, `BRAIN_TTS_MIRROR`, `BRAIN_GOLDEN_MIRROR` -
  dumped goldens and raw test media, which are not checkpoints, are not in the
  model store, and have no canonical address. Regenerated per box (the script
  names the `tools/goldens/*_dump_reference.py` for each), so a run that reports
  them absent is reporting the normal state, not a misconfiguration.

Plus `BRAIN_DIAMOND_REPO`, `BRAIN_GENIEREDUX_REPO`.

**Test/bench infrastructure:**
- `BRAIN_LOG_WEIGHTS` - verbose weight-loading logging in a test/bench run.
- `BRAIN_BIN` - path to the `brain` binary for e2e/bats suites.
- `BRAIN_TTS_SOCK` - socket path a TTS e2e test connects to.
- `BRAIN_TESTDATA` - overrides the `testdata/` resolution root (see §1).
- `BRAIN_REQUIRE_FIXTURES` - turns every `brain_testutil::skip` into a hard
  failure. Skipping an absent fixture is the right default, but cargo reports a
  skip as a PASS, so a green `cargo test -p <crate>` is on its own no evidence
  that a parity comparison happened at all. Set this in any run whose PURPOSE is
  to prove parity (`make wan/parity` is the worked example) and every comparison
  that did not really run turns the suite red. This was not hypothetical: the
  Wan suite reported `ok` while 7 of its 9 VAE stage comparisons and its real
  1.3B transformer comparison were all silently skipping, because their weights
  resolve from the environment.
- `BRAIN_REQUIRE_GOLDEN_SOURCE` - turns "this golden does not record which
  checkpoint produced it" into a hard failure. A golden dump is tensors plus a
  claim, and the claim only means anything together with the checkpoint that
  produced it; `tools/goldens/golden_source.py` writes that provenance and
  `brain_testutil::golden::Source` enforces it. A *mismatch* is always loud (it
  routes through `brain_testutil::skip`, so it is fatal under
  `BRAIN_REQUIRE_FIXTURES` too). This flag is about the weaker case: a dump from
  before the convention, which prints `UNVERIFIED GOLDEN SOURCE` and still runs.
  It is a ratchet, like the clippy one - switch it on per suite as each dumper
  is re-run, rather than taking every suite red at once.
- `BRAIN_MODELS_DIR` - overrides the model-store root tests resolve checkpoints under.
- `BRAIN_E2E` - enables the heavy, opt-in e2e suites (real weights + GPU).

**Production-namespaced but test-only in practice** (these look like serving
config but in this codebase only ever gate whole test suites):
`BRAIN_QWEN_TE_SHARD`, `BRAIN_VAE_DEVICE`, `BRAIN_VQGAN_DEVICE`,
`BRAIN_CODEFORMER_DEVICE`, `BRAIN_QWEN35_GGUF`, `BRAIN_SDXL`,
`BRAIN_SDXL_VAE_DEVICE` (the SDXL UNet has no CLI or serving surface at all
today - these two exist only for the dev-only `sdxl` binary).

## Internal engine-tuning environment variables

Real production reads, but kernel-selection A/B switches and debug/benchmark
knobs for contributors doing kernel or performance work - not part of the
user-facing configuration surface in `docs/using/configuration.md`. A user
should never need to set these; a contributor profiling or bisecting a kernel
regression will.

**Kernel-variant A/B switches** (force the naive/reference kernel instead of
the selector's normal choice, for isolating a regression to one variant):
`BRAIN_CONV_GEMM`, `BRAIN_CONV_GEMM_MIN`, `BRAIN_NAIVE_CONV`, `BRAIN_TILED_CONV`,
`BRAIN_WINOGRAD`, `BRAIN_GLMDSA_NAIVE_MM`, `BRAIN_GPT2_NAIVE_MM`, `BRAIN_GPT2_REG1`,
`BRAIN_LFM2_NAIVE_MM`, `BRAIN_QWEN_NAIVE_MM`, `BRAIN_NO_COOP_GRADNORM`,
`BRAIN_NO_COOP_LN`, `BRAIN_NO_FASTCONV`, `BRAIN_NO_KERNEL_UPGRADE`.

**Profiling / roofline internals:**
`BRAIN_NO_ROOF` (skip the roofline probe), `BRAIN_ROOF_BUDGET_S` (its time
budget), `BRAIN_TILE_BUDGET_WORDS` (override the tiled-matmul binding-size
budget used for kernel selection).

**Training-memory tradeoff:** `BRAIN_OFFLOAD_ADAM` (keep optimizer state in
host RAM instead of device memory).

**Per-model dev/debug knobs:** `BRAIN_VAE_COL_MIB`, `BRAIN_VAE_TAPS`,
`BRAIN_QWEN3OMNIMOE_DEBUG_LOGITS`, `BRAIN_OMNI_DEBUG_LOGITS` (top-3 logit dump
from the Qwen3-Omni int8 resident), `BRAIN_FLUX2_BENCH_BASELINE`,
`BRAIN_FLUX2_TIME_FORWARD`, `BRAIN_S3DIT_LAYERS` (truncates the Z-Image DiT
to N layers in the benchmark binary only), `BRAIN_WAN_VAE_TAPS` (record every
Wan-VAE block output for parity debugging), `BRAIN_LTXV_VAE_TAPS` (the same tap
recording for the LTX-Video VAE), `BRAIN_VAE3D_NOPOOL` (disable the
3D VAE builder's buffer pooling, which a tap would otherwise read after reuse),
`BRAIN_WAN_T5_FORCE_GPU` (run the umT5-XXL parity test on a GPU anyway - it
skips by default because 22.72 GB of fp32 weights exceed a 24 GB card).

**Backend internals:** `BRAIN_VK_ALLOC_DEBUG` (verbose Vulkan allocator
logging), `BRAIN_WGPU_SERIAL` / `BRAIN_WGPU_NO_SERIAL` (force / disable the
serialised-submit path the wgpu backend otherwise selects per adapter).

**Ad-hoc dev benchmarks (needs a Python + HuggingFace Transformers
environment, unlike `brain perf`):** `tools/bench/bench_qwen_inference.py`
runs brain's CPU/GPU/NPU Qwen3 inference head-to-head against HF Transformers
on the same prompt.
