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
- [ ] P3 import — strict 1:1 over `checkpoint::safetensors`.
- [ ] P4/P5 kernels + model forward (reuse existing kernels; zero-weight self-check).
- [ ] P6 parity ladder (`tools/fincast_dump_reference.py`).
- [ ] P7 forecaster adapter + CLI.
- [ ] P8 NPU export + session.
- [ ] P9 training + gradcheck + fine-tune gate.

## Checkpoint

`tools/fincast_convert.py <v1.pth> <out.safetensors>` (torch pickle → flat fp32
safetensors, strips `_orig_mod.`/`module.` prefixes). Downloaded + converted to
`scratchpad/fincast/model.safetensors` (gitignored). Live gates read it via
`FINCAST_CKPT=<abs path>`.
