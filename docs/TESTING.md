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

## Known gaps (honestly tracked)

- **MoE / federated rows in `make bench`**: the MoE engine currently trains on
  its own 64-symbol rule task, not the char datasets, so the leaderboard covers
  GPT today. Training the MoE on the shared `data` crate datasets (and the
  frozen-backbone expert training) is the remaining federated-training
  integration.
- **Time-series model, DDP, YAML configs**: deferred from Phase 2b.
- Determinism is per-run with a fixed seed; cross-machine bit-identity is not
  guaranteed (fp32 GPU reductions).
