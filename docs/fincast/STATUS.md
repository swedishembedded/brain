# FinCast in brain — status & extracted architecture

FinCast (`Vincent05R/FinCast`, `v1.pth`, 3.97 GB, **991.4M params**, native fp32,
Apache-2.0 *research/educational use only* — see `docs/licences.md`). A
**TimesFM-style patched decoder with a sparse top-2 MoE** and a
probabilistic-quantile (PQ) head. Reference: `resources/time-series/repos/FinCast`
(`src/ffm/pytorch_patched_decoder_MOE.py`, `src/st_moe_pytorch/st_moe_pytorch.py`).

## Extracted architecture (verified against the real header)

Config (`FincastConfig::default()`), all confirmed by the live T0 gate:

| field | value | note |
|---|---|---|
| num_layers | 50 | |
| hidden_size | 1280 | |
| num_heads / num_kv_heads | 16 / 16 | MHA (no GQA) |
| head_dim | 80 | `inner = 16*80 = 1280 = hidden` |
| qkv_proj out | 3840 | `(16 + 2*16)*80` (fused q,k,v) |
| patch_len | 32 | input feature = `[values, mask]` = 64 wide |
| horizon_len | 128 | one AR step emits 128 points |
| num_quantiles | 9 | levels `[0.1..0.9]`; head emits `mean + 9` |
| head_out | 1280 | `horizon_len*(1+9)` |
| **num_experts** | **4** | ⚠️ reference `FFMConfig` default says 3; the shipped weights carry `experts.0..3` + a `[4,1280]` gate |
| gating_top_n | 2 | |
| use_positional_embedding | false | model trained without it |

Forward (per `PatchedTimeSeriesDecoder_MOE`):
1. patch context into `patch_len` windows; masked per-instance standardize
   (`_masked_mean_std` on the first patch with >3 valid values); concat
   `[values, mask]` → `input_ff_layer` ResidualBlock → hidden.
2. `+ freq_emb[freq]` (3 buckets: high/med/low).
3. 50× decoder block: `input_layernorm` (RMSNorm) → **TimesFM attention** → +res;
   then **SparseMoEBlock** → +res.
4. `horizon_ff_layer` ResidualBlock → `[N, horizon_len, 1+9]`; reverse the
   standardization (× sigma + mu). AR-decode by feeding the mean forecast back.

TimesFM attention: fused `qkv_proj` (biased) → split → per-dim query scaling
`q *= (1.442695041/sqrt(head_dim)) * softplus(scaling[d])` (learned `scaling`
vector) → causal mask (+ padding mask) → softmax → `o_proj` (biased). **No RoPE.**

SparseMoEBlock (st_moe_pytorch, `add_ff_before/after=False`): `moe_prenorm`
(st-MoE RMSNorm = `x/‖x‖·√d·gamma`, i.e. RMSNorm w/ unit gamma init) → gate
`to_gates` (Linear d→num_experts, no bias) → softmax → top-2 → **each expert** =
`LayerNorm(eps 1e-6) → gate_proj → ReLU → down_proj`, **plus the expert's own
`+x` residual** → combine `w0·E[i0] + w1·E[i1]` (w renormalized to sum 1) → `+res`.

### Parity trap — stochastic eval routing
The reference gate is **stochastic even at eval**: it draws `uniform()` and routes
to the 2nd expert only if `u < gate2/threshold_eval`, and applies **capacity**
dropping. brain implements the deterministic top-2 expectation (always route
top-2, renormalized gates, no capacity drop). The parity dump neutralizes the
reference stochasticity (`threshold_eval→0`, `capacity_factor_eval→∞`) so the
golden is deterministic and comparable. Documented, not hidden.

## Progress

- [x] **P1 config** — `config.rs`, `param_list()` in reference names, `tiny()`, workspace wiring.
- [x] **P2 T0 gate** — golden `tests/golden/header.json` (from real `v1.pth`); `param_list()` matches golden AND the live 3.97 GB checkpoint name/shape (env `FINCAST_CKPT`).
- [x] **P3 import** — strict 1:1 over `checkpoint::safetensors` (dup/missing/extra checks).
- [x] **P4/P5 forward** — `model.rs` device forward (all GEMMs/norms/SiLU/ReLU via
  gpu-core kernels; runs on CPU + GPU) + host causal MHA (per-dim softplus q-scale)
  + deterministic top-2 MoE. Reused `matmul`/`bias_add`/`relu_inplace`/`silu`/`add`/
  `rmsnorm`/`layernorm` — **no new WGSL kernel needed**. Zero-weight self-check
  forecasts the first-patch mean.
- [x] **P6 parity** — `tools/fincast_dump_reference.py` runs the REAL reference
  (stochastic MoE neutralized) → committed golden. `tests/parity.rs` on the real
  991M weights: **cosine=1.000000, pearson=0.999980, rel_rms=0.000000**.
- [x] **P7 forecaster + CLI** — `FincastForecaster` (9 native quantiles +
  interpolation, freq buckets); `brain forecast import/serve/compare --fincast`.
- [x] **P8 NPU** — `fincast_topology.rs` (ONNX core: fused-qkv causal attention w/
  folded q-scale, in-graph top-2 MoE via TopK+GreaterOrEqual mask, SiLU head) +
  `fincast_export.rs` + `FincastSession` + `brain npu fincast`. Compiles and runs
  **on the real NPU** (`/dev/accel/accel0`): reproduces `core_forward` at
  **cosine 1.000000 on both CPU-OpenVINO (fp32) and the NPU (fp16)**.
- [x] **P9 training** — `train.rs` host fwd+backward of the full core (causal MHA
  w/ softplus-q-scale, top-2 MoE incl. softmax+renorm gate backward, LayerNorm/
  SiLU/RMSNorm) + PQ loss. Gradcheck (eps 5e-3) green + from-scratch learning test.
  Reuses `forecast::metrics::mean_pinball{,_grad}`. CPU + GPU (NPU inference-only).

### Parity trap found: opset-13 `ReduceSum` axes-as-input
The MoE weight normalization uses `ReduceSum` over the expert axis. At the
builder's default **opset 13**, `ReduceSum` takes `axes` as an **input tensor**,
not an attribute (whereas `ReduceMean` keeps the attribute until opset 18). The
first cut passed `axes` as an attribute, which opset-13 ignored → it reduced over
*all* axes, corrupting the per-token gate normalization → cosine ~0.92 on **both**
CPU-OpenVINO and NPU (so it was a graph bug, not an fp16 artifact — the fp16
hypothesis was disproved by the fp32 path also being 0.92). Passing `axes` as an
initializer input fixed it: cosine **1.000000** on CPU-OpenVINO and the NPU.

### Deferred / follow-ups
- Tiled/register-blocked GEMM fast path (currently naive `matmul` for CPU+GPU
  portability — the tiled kernel needs work-group barriers the CPU JIT rejects).
- Per-stage parity goldens (only end-to-end committed; end-to-end is exact).
- INT8/INT4 NPU QDQ calibration (the export supports the modes; not calibrated).
- LoRA fine-tune entry reusing Kronos's LoRA (host trainer + gradcheck are in place).

## Checkpoint

`tools/fincast_convert.py <v1.pth> <out.safetensors>` (torch pickle → flat fp32
safetensors, strips `_orig_mod.`/`module.` prefixes). Downloaded + converted to
`scratchpad/fincast/model.safetensors` (gitignored). Live gates read it via
`FINCAST_CKPT=<abs path>`.
