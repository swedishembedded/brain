# Seq2seq encoder-decoder Transformer in brain (`crates/seq2seq`)

Encoder-decoder Transformer (ADR 0001 §5): a **bidirectional** encoder +
**causal** decoder with **cross-attention** to the encoder memory. Forward +
backprop as WGSL compute dispatches, sharing the engine seam (`gpu_core` /
`paramstore` / `optim` / `kernels`) with GPT/MoE/PID. Gradient-checked.

## Architecture

Pre-LN throughout, GELU MLPs, dropout disabled (`crates/seq2seq/src/model.rs`):

- **Encoder** (bidirectional self-attention): `e = tok_emb[src] + enc_pos[pos]`;
  per block `h=LN1(e); e += Wo·BidirAttn(h); h=LN2(e); e += proj·GELU(fc·h)`.
  The encoder memory is the **raw final residual** — no extra final encoder
  LayerNorm.
- **Decoder** (causal self-attn + cross-attn): `d = tok_emb[tgt] + dec_pos[pos]`;
  per block `h=LN1(d); d += Wo·CausalAttn(h); h=LN2(d); d += Wo·CrossAttn(q=h,
  kv=enc_mem); h=LN3(d); d += proj·GELU(fc·h)`. `logits = lm_head(LN_f(d))`.
- **Embeddings**: **shared src/tgt token embeddings** (`tok.weight`); **separate**
  `enc_pos` / `dec_pos` positional tables; **untied** `lm_head` (no bias).
- **Cross-attn layout**: Q contiguous `[B·T_dec, d]`; K/V fused per layer
  `[B·T_enc, 2d]` (K@0, V@d), produced by a fused `enc_mem @ Wkv`.
- **Loss**: masked cross-entropy over labels (`ignore_index = IGNORE`).

`Seq2SeqConfig`: `vocab`, `block_size` (decoder), `src_block_size` (encoder),
`n_enc`, `n_dec`, `d_model`, `n_heads`, `d_ff`. Init is GPT-2-style with
residual projections scaled by `0.02 / sqrt(2*(n_enc+n_dec))`.

## CLI

There is **no `brain seq2seq` subcommand**. The model is exercised through its
tests (the gradcheck unit test and the convergence integration test). The
generic `brain gradcheck` runs `check_gpt`; `check_seq2seq` runs as a unit test.

## What's implemented

- **Forward + backward** as WGSL `Step` lists (`forward_steps` /
  `build_backward_steps`), with an AdamW step.
- **Training** — forward+backward+AdamW are wired, but the generic
  `model::train::fit` does **not** drive it (`fit` feeds only `Batch::Lm`);
  the convergence test trains via a manual loop.

**Not implemented**: inference / sampling / serving (`Model::logits_all` returns
`None`; there is no decode or generate path), and there is no HF/safetensors
import — weights arrive only via `init_weights` or `checkpoint::load`. (A
`lib.rs` comment that "the generic trainer, sampler … cover it" is aspirational
for the sampler.)

## Parity / gradcheck

- `gradcheck::check_seq2seq` — finite-difference gradient check (central diff,
  `eps=5e-3`) over a tiny config that exercises `T_dec ≠ T_enc` and `IGNORE`
  masking; the `seq2seq_analytic_grads_match_finite_differences` test asserts
  no failures at `atol=4e-3, rtol=8e-2`.
- **Learnability** (`tests/convergence.rs::engine_learns_copy_via_cross_attention`)
  — COPY task (6-symbol alphabet, length 5), 400 steps: asserts `last < 0.5`
  vs the marginal floor ln6≈1.792.

## Kernel / block reuse

Encoder self-attention uses the shared `model::block` bidirectional builders
(`bidir_fwd` / `bidir_bwd`); LayerNorm uses the shared `model::block` LayerNorm
family (coalesced `_rows` variants). `PIPELINES` maps ~48 named kernels from
the shared `kernels` crate — seq2seq declares no local WGSL.

## Limitations

- No `brain seq2seq` CLI (tests only).
- No inference / sampling / serving path; not drivable by the generic
  `model::train::fit`.
- No weight-import path (only `init_weights` and checkpoint load/save).
- Dropout disabled; v1 design choices (shared embeddings, no final encoder LN,
  untied `lm_head`).

## See also

- `docs/architecture.md` — crate graph.
- `docs/testing.md` — the gradcheck gate.
- `AGENTS.md` → Models → Seq2seq.
