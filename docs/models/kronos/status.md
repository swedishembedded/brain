# Kronos in brain — status & implementation plan

From-scratch Rust+WGSL reimplementation of **Kronos** (financial K-line/candlestick
foundation model, arXiv 2508.02739, **MIT**), following the recipe that just took
Chronos-2 to cosine=1.0 parity. Two-stage: a **BSQ tokenizer** (OHLCV bar →
hierarchical discrete tokens) + an **autoregressive decoder** with a dual head.
Native representation for the `ForecastModel` adapter = **samples** (AR rollout).

Reference: `resources/time-series/repos/Kronos/model/{kronos.py,module.py}`.
Checkpoints (MIT): `NeoQuasar/Kronos-small` (24.7M) + tokenizer
`NeoQuasar/Kronos-Tokenizer-base` (~16MB) — Tier-1 ~425MB. Fetch config.json to
confirm the flagged ambiguities before T0.

## Resolved ambiguities (the doc/bundle got these WRONG)

1. **`max_context` counts BARS, not subtokens** — 512 = **512 bars** of history
   (double what `docs/kronos.md` feared). s1/s2 are two *parallel channels* fused
   at each position, not two sequence positions. Proof: `HierarchicalEmbedding`
   fuses `[s1_emb, s2_emb]` at one position; `pre_buffer`/`post_buffer` are two
   streams each indexed by bar (kronos.py:408, 254).
2. **BSQ order**: project (`quant_embed` Linear d_model→20) FIRST, then
   L2-normalize the 20-dim vector, then sign. The "k learnable hyperplanes" =
   `quant_embed.weight [20,256]`; the **BSQ module itself is parameter-free**.
3. Encoder/decoder block counts = `n_enc_layers-1` / `n_dec_layers-1` (off-by-one
   loop) — **count `encoder.{i}`/`decoder.{i}` keys in the real checkpoint**.
4. `learn_te` (config.json) changes `time_emb.*` key naming AND whether calendar
   tables are learned or frozen sinusoids — read it before T0.

## Concrete dims

**Kronos-small decoder**: d_model=512, n_layers=8, n_heads=8, ff_dim=1024,
s1_bits=s2_bits=10 → s1/s2 vocab=1024 each, head_dim=64, max_context=512 bars.
**Kronos-Tokenizer-base**: d_in=6 (OHLCVA), d_model=256, n_heads=4, ff_dim=512,
codebook_dim k=20, head_dim=64. Dropouts 0 at inference; loss hyperparams
(beta/gamma/zeta/group_size) have no learnable params — ignore.

## Architecture essentials (vs Chronos-2)

- **Attention: CAUSAL + SCALED** (1/√head_dim) — reuse `attn_scores.wgsl` (causal,
  scaled) + `attn_softmax_full` + `attn_apply_full`. (Chronos-2 was the opposite:
  non-causal unscaled.)
- **RoPE: half-split / NeoX**, base 10000, full head_dim → reuse **`rope_neox`**.
- **Norm: RMSNorm** eps 1e-5 weight-only → reuse `rmsnorm`.
- **FFN: SwiGLU** `w2(silu(w1(x))·w3(x))`, all bias-free → **NEW kernel**: SiLU
  (`x·sigmoid(x)`) + gated multiply.
- **BSQ quantize**: L2-normalize(20) → sign → ×(1/√20) → **NEW kernel** (tiny).
  STE is training-only. bits↔indices (LSB-first, 10 bits/stream) can be host-side.
- **HierarchicalEmbedding**: `fusion_proj(cat[emb_s1(s1)·√d, emb_s2(s2)·√d])`.
  NOTE the `×√d_model` scale is applied in the fusion path but **NOT** when
  `emb_s1` is reused raw as the dependency-layer sibling embedding — preserve the
  asymmetry.
- **DualHead + DependencyAwareLayer**: predict s1 (`head.proj_s1` d→1024), then
  sample s1 from `softmax(s1_logits.detach())` (exposure bias — the sample, not
  ground truth, at inference), embed it raw via `emb_s1`, cross-attend
  (query=sibling_embed, k/v=hidden) via `DependencyAwareLayer`
  (**n_heads=4** → head_dim 128, **non-causal at inference, scaled**, RoPE on q&k),
  then `head.proj_s2` (d→1024) → **NEW kernel**: scaled non-causal cross-attention
  scores (separate q vs k/v) — or add a `scale` uniform to `attn_scores_full`.
- **TemporalEmbedding**: 5 calendar tables (minute60/hour24/weekday7/day32/month13)
  summed, added to the fused embedding. Frozen sinusoidal if `learn_te=False`
  (recompute on host) else learned.
- **Sampling: host-side** (temperature/top-k/top-p/multinomial). Replicate the
  quirk: `top_k_top_p_filtering` returns inside the `top_k>0` branch, so if both
  are set only top_k applies. For deterministic parity, compare **logits** (argmax
  or seeded).

## param_list (state_dict names) — tokenizer + decoder

Tokenizer (`embed`, `quant_embed`, `post_quant_embed{,_pre}`, `head`,
`encoder.{i}.*`, `decoder.{i}.*` with `norm1/self_attn.{q,k,v,out}_proj/norm2/
ffn.{w1,w3,w2}`, `tokenizer.bsq.basis` buffer). Decoder (`embedding.{emb_s1,emb_s2,
fusion_proj}`, `time_emb.*_embed[.emb].weight`, `transformer.{i}.*`, `norm`,
`dep_layer.cross_attn.*`+`dep_layer.norm`, `head.{proj_s1,proj_s2}`). Full shapes
in the extracted spec (in the P2 task + agent transcript). PyTorch `[out,in]`.

## Parity — T0 + T1 + T2 + T4 PASS vs the real checkpoints ✅

Checkpoints downloaded to `resources/time-series/checkpoints/{kronos-small,
kronos-tokenizer-base}/`. Reference dump: `tools/goldens/kronos_dump_reference.py`
(deterministic rungs — avoids the stochastic rollout). Brain side:
`crates/kronos/tests/parity.rs` (env-gated `KRONOS_TOKENIZER_DIR` /
`KRONOS_DECODER_DIR`).
- **T0** layout: both nets' `param_list()` match the real checkpoints
  name/shape-for-shape (needed an I64-buffer fix in `checkpoint::safetensors`).
- **T1** tokenizer encode: `(s1, s2)` tokens **integer-exact** (120/120 each).
- **T2** tokenizer decode: reconstruction **cosine = pearson = 1.000000**.
- **T4** decoder `decode_s1`: s1 logits **cosine = pearson = 1.000000**.
Brain's Kronos (BSQ tokenizer + AR decoder) reproduces the reference to full fp32
precision. (The stochastic AR rollout / `decode_s2` sampling isn't a
deterministic rung, but decode_s2 reuses the T2/T4-validated primitives.)
Subtlety fixed: reference `decode_s1(stamp=None)` adds no temporal embedding —
brain skips it on an empty stamp.

## Parity ladder (T0–T5), cosine>0.99 / integer-exact where noted

- **T0** param layout — both nets vs real state_dicts (confirm block counts +
  learn_te variant).
- **T1** tokenizer encode — `quant_embed z` → normalize → zq(±1/√20) →
  `z_indices=[pre,post]` **integer-exact**.
- **T2** tokenizer decode — `indices_to_bits` → `post_quant_embed` → `head` recon
  (6-dim), cosine≈1; T1→T2 roundtrip reconstructs the bar.
- **T3** decoder one block — RMSNorm→RoPE-causal-scaled-attn→resid→RMSNorm→SwiGLU;
  isolate RoPE q/k.
- **T4** dual head — `decode_s1` → s1_logits(1024)+context; `decode_s2(context,
  s1_gt)` → dep_layer x2 + s2_logits(1024). Feed ground-truth s1 for determinism.
- **T4.5** exposure-bias sampling path (seeded).
- **T5** full generate — AR loop (argmax/seeded), decode, de-normalize; check
  per-step logits + final de-normalized bars; verify sample_count averaging +
  the top_k/top_p quirk. Reference dump via a `tools/goldens/kronos_dump_reference.py`
  (KronosPredictor), env-gated like Chronos-2.

## New kernels — DONE, isolation-tested (`crates/kronos/tests/kernels.rs`, CPU)

1. `bsq_quantize` — `sign(z)·(1/√k)` in place (the L2-normalize is sign-irrelevant;
   `z>0→+1 else −1`, matching `torch.where`).
2. `silu_gate` — SwiGLU `out = silu(a)·b`, `silu(x)=x·sigmoid(x)`.
3. `attn_scores_qk` — scores from **separate q,k** buffers with a `scale` uniform
   and an optional `causal` flag — covers BOTH Kronos modes: causal-scaled
   self-attention AND non-causal-scaled dependency-layer cross-attention. Reuse
   `attn_softmax_full` + `attn_apply_full` after it (causal −inf already baked in).

bit-pack/unpack (indices↔bits, LSB-first) and sampling: host-side (still TODO).

## Done so far (all green, mirrors the Chronos-2 progression)

- `config.rs` — both configs' `param_list()` in reference `state_dict` names
  (tokenizer ~4M, decoder ~24.7M); `from_hf`; `tiny()`. 6 tests.
- `tests/t0_param_layout.rs` — T0 gate; live gates env-gated on
  `KRONOS_TOKENIZER_CKPT` / `KRONOS_DECODER_CKPT`.
- 3 new WGSL kernels (`bsq_quantize`, `silu_gate`, `attn_scores_qk`), all
  isolation-tested (`tests/kernels.rs`).
- `preprocess.rs` — `normalize`/`denormalize` (per-feature z-score + clip±5),
  `quantized_to_indices`/`indices_to_bipolar` (LSB-first bit↔index packing). 5
  tests incl. full-range 10-bit roundtrips.

## Remaining (the two forwards + generate + parity)

## Build order (proven recipe)

config.rs+param_list (both nets) → T0 gate → tokenizer (encode/decode + BSQ) →
decoder (blocks + dual head + dep_layer + temporal) → import → sampling/generate →
T5 parity vs KronosPredictor → `ForecastModel` adapter (native=samples;
requires_variates=[open,high,low,close,volume]; needs a Panel→OHLCV bar mapping).
Per-series z-score normalize (clip ±5) on input, de-normalize on output.

## NPU deployment (done — parity closed)

Both AR-decoder graphs are exported to ONNX and run on the Intel NPU via
OpenVINO, verified against brain's own WGSL forward (chain
reference→WGSL→ONNX→NPU). The host keeps embedding (s1/s2 gather + `√d` +
`fusion_proj` + calendar sum), sampling, and the BSQ tokenizer — the same
host/device split the GLM/Qwen/Chronos-2 exports use.

- `crates/npu/src/kronos_topology.rs` — `build_kronos_decoder_graph` (decode_s1
  core: 8 causal+scaled biased-MHA blocks, NeoX RoPE, SwiGLU FFN, final norm,
  `proj_s1`; in `x:[1,T,D]` → out `ctx:[1,T,D]`, `s1_logits:[1,T,s1v]`) and
  `build_kronos_dep_graph` (decode_s2: non-causal scaled cross-attn with
  `dep_n_heads`, `norm(ctx+attn)`, `proj_s2`; in `ctx,sib:[1,T,D]` → out
  `s2_logits:[1,T,s2v]`). INT8/INT4 per-output-channel dequant path shared.
- `crates/npu/src/kronos_export.rs` — `export_onnx`/`export_dep_onnx` from the HF
  dir; env-gated tests dump brain's WGSL reference for a fixed input.
- `kronos::decoder::core_forward_s1` / `core_forward_s2` — the WGSL reference
  entries matching each graph's I/O contract (dep body factored into
  `dep_forward`, shared with `decode_s2`).
- `kronos::import::load_decoder` — decoder config+weights from the HF dir.

Parity (Kronos-small, T=16, fp32; probes in scratchpad
`npu_parity_kronos{,_dep}.py`):

| graph | output | CPU cosine | NPU cosine | NPU rel-max |
|---|---|---|---|---|
| decode_s1 | ctx | 0.999993 | 0.999989 | 0.0051 |
| decode_s1 | s1_logits | 0.999999 | 0.999998 | 0.0026 |
| decode_s2 | s2_logits | 1.000000 | 1.000000 | 0.0015 |

decode_s1's ~3e-3 rel-max is the ONNX/OpenVINO `Sigmoid` (SwiGLU gate) vs brain's
WGSL `silu_gate` across 8 layers; the s1_logits cosine (0.999999) means the
argmax token decisions are identical. Remaining for a full on-NPU forecast: an
AR-loop driver (`brain npu kronos`) that grows T over the two graphs + host BSQ
decode — mechanical, mirrors `qwen_decode`/`glm_decode`.

## Perf: KV-cache (done — exact, 6.9x)

The un-cached AR rollout re-runs `decode_s1` over the full window every step
(`O(T²)`/step). `crates/kronos/src/kvcache.rs` adds a pure-host KV-cached rollout:
prefill the context once, then advance one token at a time over per-layer RoPE'd
K/V caches — `O(T²)` prefill + `O(T)`/step. Correctness rests on **RoPE
shift-invariance** (attention scores depend only on `i−j`, so keys cached at
absolute positions stay valid as the window slides — we window attention to the
last `max_context`) and V carrying no RoPE. Host math reproduces the WGSL exactly
(rmsnorm eps 1e-6, NeoX RoPE θ=10000, attn scale 1/√head_dim).

- `KronosDecoder::host_weights` → `kvcache::HostW`; `KronosModel::forecast_cached`;
  `KronosForecaster` uses the cached path.
- Parity (`tests/kvcache_parity.rs`, real weights): **cosine=1.000000, rel_max=0.0**
  vs the un-cached decoder.
- Speed (release, 256-ctx / 20-horizon): un-cached **91 s → cached 13 s = 6.9×**.

Pure host, so it's device-agnostic (fast even with no GPU). Further multipliers:
thread the host matvecs (rayon), run the `O(T²)` prefill on the GPU, and
sample-batching (the kernels already support `bsz>1`; the stochastic adapter draws
`n_samples` cached rollouts sequentially).
