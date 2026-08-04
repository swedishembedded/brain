# Chronos-2 in brain — status & implementation plan

From-scratch Rust+WGSL reimplementation of Amazon's **Chronos-2** (encoder-only
T5-style patch transformer, 120M, Apache-2.0), imported exactly and parity-gated.
It plugs into the forecasting API as a [`forecast::ForecastModel`] with native
representation = **quantiles**.

Reference: `resources/time-series/repos/chronos-forecasting/src/chronos/chronos2/`.
Config verified from `amazon/chronos-2/config.json` (2026-07-20).

## Verified config (the published checkpoint)

`d_model=768, d_kv=64, d_ff=3072, num_layers=12, num_heads=12`, native **fp32**
(no bf16 path), 21 quantiles, `patch=16`, `context=8192`, `use_arcsinh=true`,
`use_reg_token=true`, `time_encoding_scale=8192`, `rope_theta=10000`.
`inner_dim = 12*64 = 768 = d_model` ⇒ attention projections are square 768×768.
Total params: **119,477,664** (≈120M). Download ~478 MB.

## Done (green)

- `config.rs` — `Chronos2Config`, `param_list()` in the reference's own
  `state_dict` names (170 tensors), `from_hf()`, `tiny()`, param-count self-check.
- `tests/t0_param_layout.rs` — T0 device-free layout gate; live gate env-gated on
  `CHRONOS2_CKPT` (diffs `param_list()` vs the real safetensors header).
- `preprocess.rs` — `instance_norm` (NaN-aware standardize + **arcsinh**) +
  `instance_norm_inverse` (sinh + affine); `context_features` (left-NaN-pad,
  unfold, mask, zero-missing, time-enc, 48-wide `[time,values,mask]`) +
  `future_features`. This is the numerically load-bearing input/output contract.

## Parity traps (do NOT get these wrong)

1. **Attention is UNSCALED** — no `1/sqrt(d_kv)`. The existing
   `attn_scores.wgsl` DOES scale and IS causal → need a new
   `attn_scores_full.wgsl`: bidirectional, additive mask, no scaling.
2. **RoPE is half-split / NeoX** — `rotate_half = cat([-x[d/2:], x[:d/2]])`,
   pairs `(j, j+d/2)`. The existing `rope.wgsl` is INTERLEAVED `(2j, 2j+1)` →
   need a new `rope_neox.wgsl`. Applied on **time attention only** (group
   attention has no positional encoding).
3. **RMSNorm** weight-only (no bias, no mean subtraction) — reuse `rmsnorm.wgsl`.
4. **ReLU** FFN (not gated) — reuse `matmul` + `relu_inplace`.
5. Patch embed = **ResidualBlock** over 48ch: `out(relu(hidden(x))) + residual(x)`,
   all three linears have bias. Same shared block embeds context AND future.
6. Quantile head = ResidualBlock `d_model→336`, then rearrange
   `b n (q p) -> b q (n p)` with **q OUTER, p inner** (i.e. index `q*16 + p`).
7. **REG token** (id 1) sits between context and future tokens; `shared.weight`
   row 1, extend the attention mask with a 1.
8. `TimeCrossAttention` in `layers.py` is **dead code** — never instantiated.

## Remaining work

### 1. New WGSL kernels — DONE, all isolation-tested (`tests/kernels.rs`, CPU backend)
The existing attention kernels are all **causal** (`for j in 0..=i`) and
`attn_apply` reads v from a **fused qkv** buffer — neither fits Chronos-2's
bidirectional, separate-q/k/v encoder. So the full attention set is four kernels:
- `attn_scores_full.wgsl` — `scores[b,h,i,j] = q·k + kmask[b,j]`, separate q,k
  buffers `[b,S,qk_stride]`, additive mask, **no scaling, no causal cut**.
- `rope_neox.wgsl` — half-split rotary, pairs `(j, j+d/2)`, `theta` as f32 param.
  (t=0 is the identity — test tokens at t≥1.)
- `attn_softmax_full.wgsl` — full-row softmax over all keys, padding-safe.
- `attn_apply_full.wgsl` — `out = sum_j probs·v` over all j, v from a **separate**
  `[b,S,v_stride]` buffer.
Reuse as-is: `matmul`, `matmul_dw`, `matmul_dx`, `rmsnorm`, `rms_inv`,
`rmsnorm_dw`, `rmsnorm_dx`, `relu_inplace`, `add`, `add_inplace`, `bias_add`.
Kernel dispatch contracts (from the headers): `matmul` out=x@Wᵀ, x[M,K] W[N,K]
out[M,N], threads=M*N; `rmsnorm` params (d_model, seq_len) bufs [x,weight,out];
`bias_add` out[m,n]+=bias[n] params (m,n); `add` dst+=src params (total).

### 2. `model.rs` forward — DONE (inference path), wiring-verified

`Chronos2::from_weights(cfg, HashMap<name,Vec<f32>>)` +
`forecast_quantiles(context, horizon) -> [num_quantiles, horizon]`. Composes
preprocessing + the kernels via immediate per-op submits (sidesteps
dynamic-length buffer-lifetime issues). Group attention uses the **B=1
degeneration**: `o_proj(v_proj(RMSNorm(h)))` per token — no attention kernel.
REG token spliced in host-side between context and future embeddings; head
rearrange (q-outer/p-inner) + denorm on host.

**Wiring test** (`zero_weights_forecast_the_series_mean`): with all weights zero,
the standardized-space output is zero, so denorm (`sinh(0)=0` → `*scale+loc`)
gives exactly the series **mean** for every quantile/step. Validates the whole
composition (embed → REG → L blocks → final-norm → head → rearrange → denorm)
end-to-end on the CPU backend with no real weights. Kernel math is covered by
`tests/kernels.rs`; preprocessing by `preprocess.rs`.

Remaining for training/gradcheck: `build_backward` + `impl model::Model` (the
forward here is inference-only, per-op-submit; a training path wants SSA buffers
and the backward of each op). Not required for inference/parity.

### 2b. `build_backward` + `impl model::Model` (for gradcheck/training)
Op sequence (univariate Phase 1, `group_ids = arange` ⇒ identity group mask):
```
patched = preprocess::context_features(instance_norm(x))      # host, [n_ctx, 48]
future  = preprocess::future_features(horizon)                # host, [n_out, 48]
emb_ctx = ResidualBlock_input(patched)                        # [n_ctx, D]
emb_fut = ResidualBlock_input(future)                         # [n_out, D]  (SHARED weights)
h = concat(emb_ctx, shared[REG], emb_fut)                     # [S, D], S=n_ctx+1+n_out
mask = concat(ctx_attn_mask, 1, ones(n_out))                  # [S]
for block in 0..12:
    # time self-attention (RoPE, unscaled, additive pad mask over S)
    n  = rmsnorm(h, block.layer.0.layer_norm)
    q,k,v = matmul(n, {q,k,v}); rope_neox(q); rope_neox(k)
    s  = attn_scores_full(q,k,mask); a = attn_softmax_masked(s); o = attn_apply(a,v)
    h  = h + matmul(o, o_proj)
    # group self-attention (identity group for univariate: each token attends
    # only to itself across the batch axis -> still runs the kernel, mask=I)
    n  = rmsnorm(h, block.layer.1.layer_norm)
    ... (same MHA, no RoPE, group mask) ; h = h + o
    # ReLU FFN
    n  = rmsnorm(h, block.layer.2.layer_norm)
    h  = h + matmul(relu(matmul(n, wi)), wo)
h = rmsnorm(h, encoder.final_layer_norm)
qp = ResidualBlock_output(h[-n_out:])                         # [n_out, 336]
qp = rearrange q-outer -> [21, n_out*16]
qp = instance_norm_inverse(qp)                                # sinh + affine
```
Implement `impl model::Model` (buys `model::fit`, sampling, and gradcheck).

### 3. `import.rs` — DONE
`load_hf(cfg, path)` (strict 1:1, native fp32), `config_from_dir`, `import(hf_dir,
out)` → brain `.safetensors` container; `Chronos2::load(path)` reads it back.
Roundtrip tested (zero weights → save → load → still forecasts the mean).
CLI: `brain forecast import --hf <dir> --out chronos2.safetensors`.

### 3b. `ForecastModel` adapter — DONE
`Chronos2Forecaster` (src/forecaster.rs) implements `forecast::ForecastModel`:
native = Quantiles (21 fixed levels, interpolated to the caller's requested
levels), Phase-1 univariate (covariates None). Tested through the seam with the
zero-weight model (returns series-mean quantiles + derived point). Registered in
the CLI: `brain forecast serve --chronos2 <weights>` loads it beside the
baselines, so it joins the runtime registry / server / Python client / harness.
Chronos-2 is now driveable end-to-end — the only remaining gap to *real*
forecasts is importing the actual checkpoint + the parity ladder below.

### 4. Parity ladder — **T0 + T5 PASS against the real `amazon/chronos-2`** ✅
Checkpoint downloaded to `resources/time-series/checkpoints/chronos-2/`
(model.safetensors 478 MB + config.json), imported to `chronos2.safetensors`.
- **T0** param layout (`tests/t0_param_layout.rs`, `CHRONOS2_CKPT=<file>`): brain's
  `param_list()` matches the checkpoint's **170 tensors name-for-name and
  shape-for-shape**, no missing/extra. PASS.
- **T5** end-to-end forward (`tests/t5_parity.rs`, `CHRONOS2_WEIGHTS=<.safetensors>`):
  brain's `forecast_quantiles` vs a golden dump from the official
  `Chronos2Pipeline` (`tools/goldens/chronos2_dump_reference.py`) on the identical
  context → **cosine = 1.000000, pearson = 1.000000, rel_max_abs = 0.0000**.
  Brain reproduces the reference to full fp32 precision. PASS.

Because T5 is exact end-to-end, T1–T4 (scaler / patch-embed / block / head) are
implicitly validated — no need to add per-stage goldens unless a future change
regresses T5. (`tools/goldens/chronos2_dump_reference.py` can be extended with module
hooks to dump T1–T4 if drill-down is ever needed.)

### 5. Register + gradcheck
`gradcheck::check_chronos2` (tiny config, float I/O — `check_autoencoder` is the
template; eps=5e-3 n_dirs=4 (atol,rtol)=(4e-3,8e-2)). Register in the fcbench
harness + server so it joins the comparison and negative-control gate.

## Deferred to Phase 2

- Real covariates: user `group_ids`, past/known-future masking, `group_time_mask`
  construction, categorical target-encoding (host-side, `preprocess.py`).
- Long-horizon (>1024) pipeline-level autoregressive quantile-path unrolling.
- NPU export (separate onnx+npu path).

## Multivariate / covariates (done — bit-exact parity)

Group attention (attention over the series axis at each patch position, unscaled,
no RoPE, masked by `group_ids`) is implemented for B>1. Past covariates enter as
extra series in one group with unknown futures; the target attends to them at every
patch position and its row is kept.

- `model.rs`: `block()` split into `time_attention` / `group_degenerate` / `ffn`;
  `forecast_quantiles_mv(series[], horizon)` runs per-block {time-attn per series →
  `group_attention_host` (pure-Rust B×B multi-head attention across series) → FFN
  per series}. B=1 matches the univariate path (self-check cosine>0.9999).
- `forecaster.rs`: `CovariateSupport::Full`; `forecast()` routes past-covariate
  variates through the mv path.
- Parity: `tools/goldens/chronos2_dump_mv_reference.py` + `tests/mv_parity.rs` →
  **cosine=1.000000 pearson=1.000000 rel_max=0.0** (target+covariate, group_ids=[0,0]).

Group attention is host-computed, so an N-series forecast is slow (~20–70 s for 8
series); a GPU group-attention kernel is the deferred speedup. Known-future covariate
*values* (future patches carrying covariate data) are not consumed yet — only the
past/multivariate path.
