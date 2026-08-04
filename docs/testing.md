# brain — testing plan

The goal is to evaluate every architecture **reliably against the same input
data**, and to keep the from-scratch WGSL backprop provably correct without a
PyTorch oracle. Coverage is layered:

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
| `tests/e2e/claude_code.bats` | `test/e2e/claude-code` | The real `claude` CLI working end-to-end against `brain serve --anthropic` with a real qwen checkpoint. Needs `claude` installed + `BRAIN_QWEN_WEIGHTS`/`BRAIN_QWEN_TOKENIZER`; skips cleanly without them. |
| `tests/e2e/scheduler.bats` | `test/e2e/scheduler` | Heavy, opt-in (`BRAIN_E2E=1`): residency scheduler batching/eviction, and the generate→detect→annotate pipeline, against **real** z-image + yolo weights and a GPU. Not part of the fast lane — `tests/e2e/examples.bats` runs the same generate/detect example against the mock instead. |

`make test/e2e` runs the three fast suites (api-conformance, shutdown, examples)
in one shot; `make test/full` folds that into the release gate alongside the
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
