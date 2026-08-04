# VQGAN (`crates/vqgan`) — ledger

The `basicsr` VQ autoencoder behind **CodeFormer** blind face restoration
(`basicsr/archs/vqgan_arch.py`, `VQAutoEncoder`): Encoder (25-block flat
`nn.ModuleList`) → `VectorQuantizer` (nearest, 1024 × 256) → Generator
(25 blocks). Two checkpoints carry that exact module trio — `vqgan_code1024.pth`
(standalone) and `codeformer.pth` (same modules, stage-III finetuned encoder).

## Scope of this pass

**Goldens → import → FORWARD parity only.** Backward/gradcheck, the CodeFormer
transformer + controllable feature transformation + fidelity dial, and the
serving contract are deliberately deferred — see "Deferred" for what each needs.

## Deliverables

| Piece | Where |
|---|---|
| Reference dumper | `tools/goldens/codeformer_dump_reference.py` |
| Goldens (gitignored) | `testdata/restore/vqgan/{codeformer,vqgan_code1024}/{codebook,quantizer,stages_128,e2e_512_synth,e2e_512_face}.safetensors` + `manifest.json` |
| Weights (not in repo) | `$BRAIN_VQGAN_WEIGHTS/{codeformer,vqgan_code1024}.pth` |
| Shared block builder (**hoisted**) | `vae::blocks::{Builder, BlockNames, kernels_with, NEXT_SLOT}` |
| Config + block schedule | `crates/vqgan/src/config.rs` |
| Import (two-way coverage) | `crates/vqgan/src/import.rs` |
| Forward graphs | `crates/vqgan/src/model.rs` |
| Parity test | `crates/vqgan/tests/parity.rs` |

```bash
BRAIN_VQGAN_WEIGHTS=<dir with the two .pth> \
  CARGO_HOME=… cargo test --release -p brain-vqgan -- --nocapture
BRAIN_VQGAN_DEVICE=cpu   # same numbers on the CPU JIT
```

## Reuse: hoisted, not forked

`crates/vae`'s private `Builder` became the **public `vae::blocks::Builder`**,
parameterised by `BlockNames` (diffusers `conv_shortcut` / `to_q…to_out.0` /
`group_norm` vs VQGAN `conv_out` / `q…proj_out` / `norm`) — the two
architectures differ only in **schedule** and **leaf names**, not in the blocks.
`AutoencoderKL` was migrated onto it in the same change; `vae::decoder`
re-exports `{Tensors, KERNELS}` so `flux2_bench` is untouched.

`crates/vqgan` adds **no kernel and no block**: `vq_argmin` (via
`wm_core::vq::Vq`) for the assignment, `embed` for the lookup, and
`vae::blocks::kernels_with::<N>()` to *copy* — never restate — the shared kernel
slots.

## Forward parity, MEASURED (Tesla P40, release, GPU unless noted)

18 tests pass (9 unit + 9 parity). Cosine is printed to 9 dp; `1−cos` is given
because every stage prints `1.000000000`. **Relative L2** (`‖got−want‖/‖want‖`)
is reported alongside because cosine is scale-invariant and would not notice a
uniformly-wrong magnitude.

| run | worst stage | 1−cos | rel L2 | `output` max\|Δ\| | index mismatches |
|---|---|---|---|---|---|
| codeformer, `stages_128` (all 25+25 blocks + 11 sub-block taps) | `enc.21` | 1.84e-11 | 6.26e-6 | 7.48e-5 | 0/16 |
| vqgan_code1024, `stages_128` | `gen.19` | 1.08e-11 | 4.62e-6 | 2.37e-5 | 0/16 |
| vqgan_code1024, `stages_128`, **CPU JIT** | `gen.23` | 1.33e-11 | 5.20e-6 | 3.00e-5 | 0/16 |
| codeformer, `e2e_512_synth` | `vq.min_dist` | 4.37e-11 | 9.35e-6 | 6.40e-5 | 0/256 |
| codeformer, `e2e_512_face` | `vq.min_dist` | 1.62e-10 | 1.80e-5 | 1.96e-5 | 0/256 |
| vqgan_code1024, `e2e_512_face` | `vq.min_dist` | 1.63e-10 | 1.80e-5 | 1.69e-5 | 0/256 |
| both, quantizer unit (seeded `z`, no encoder) | `u4.min_dist` | 4.6e-14 | 3.07e-7 | `codebook_feat` **bit-exact** | 0/16 and 0/256 |

`generate(z_q)` — the generator-only suffix, i.e. the CodeFormer seam —
reproduces `decode` **bit-identically** (max\|Δ\| 0.0) at 512².

All 18 tests also pass on the **CPU JIT** (`BRAIN_VQGAN_DEVICE=cpu`) — but see
finding 9: at 512² the CPU backend is 24× less accurate than the GPU, which is
why the 512² gate is 3e-3 and the 128²/quantizer gates are 1e-4.

`vq.min_dist` is the widest gap everywhere and is expected: the reference
computes `|z|²+|e|²−2z·e`, the kernel `Σ(z−e)²`. The dumper measures the argmin
disagreement between the two forms and reports it; every parity run confirms
**0 index flips** at every site, which is the load-bearing property.

## Findings

1. **VQGAN `Upsample` is NOT a transposed convolution.** `vqgan_arch.py`'s
   `Upsample` is `F.interpolate(scale_factor=2, mode='nearest')` followed by
   `Conv2d(k3,s1,p1)` — the existing `upsample2` + conv path. `crates/vqgan`
   dispatches no `convtr2d`; `docs/imaging/plan.md` row 2 was corrected.

2. **The heads have no activation.** VQGAN is `GroupNorm → Conv2d`
   (`vqgan_arch.py:265, :313`); the diffusers VAE head is
   `GroupNorm → SiLU → conv_out`. Reusing the VAE head verbatim would have
   inserted a spurious SiLU.

3. **Attention is not mid-block-only.** At `attn_resolutions`, an `AttnBlock`
   follows *every* residual block, plus the mid triple — encoder 17/19/21,
   generator 2/5/7. `attn_resolutions` is resolved against `img_size` at
   **construction**, so the positions are frozen and do not move with the
   runtime input size (a 128² run still attends at blocks 17/19/21, where the
   spatial size is 4×4).

4. **The production graph was untested.** Every parity run builds with
   `taps = true`, which pins activations and therefore **disables**
   `vae::blocks::Builder`'s buffer pool. Real callers pass `taps = false`, where
   activations are aliased — a `free` issued one step early is invisible in the
   tapped build and silently corrupts the pooled one. `pooled_matches_tapped_*`
   now gates the pooled graph **bit-for-bit** against the tapped one
   (reconstruction, `latent()`, `quantized()`, and the `generate` suffix).
   Result: bit-identical on both checkpoints — the lifetimes are sound, but
   nothing had proved it.

5. **The parity gate was scale-blind.** `Report::finish` asserted on cosine
   only, which is scale-invariant: a dropped `1/√C`, a doubled residual or a
   twice-applied bias reports cosine 1.000000000. It now also gates relative L2
   at 1e-4 — measured worst 1.80e-5 (`vq.min_dist`, the formula difference) and
   ≤ 6.3e-6 on every network stage, so the gate has real headroom and real teeth.

6. **`embed` has no bounds check and brain compiles shaders
   `ShaderRuntimeChecks::unchecked()`.** `Vqgan::decode` is public and the
   CodeFormer follow-up feeds it transformer-predicted indices, so an
   out-of-range code would read past the codebook rather than trap.
   `decode` now validates the range, as `Codebook::lookup` already did.

7. **`vq_argmax_dot` is compiled but never dispatched.** `wm_core::vq::Vq` is a
   two-slot struct and `Gpu::new` compiles pipelines eagerly, so the cosine
   variant costs one shader module + pipeline per device. Left as-is; the right
   fix is per-kernel handles in `wm_core::vq`, not a hand-rolled dispatch here.

8. **Taps must carry the architecture's leaf name.** The hoisted `attn` tapped
   `.norm`/`.proj_out` unconditionally while `resnet` used the configured
   shortcut name — correct for VQGAN, wrong for a diffusers graph whose tensors
   are `.group_norm`/`.to_out.0`. Both now use `BlockNames`.

9. **The CPU JIT is 24× less accurate than the GPU at 512², and it is a
   GroupNorm summation-order effect, not a port defect.** `wgsl-cpu` refuses
   `gn_stats_wg` and `matmul_reg3` (both need more than one top-level
   `workgroupBarrier()`), so on CPU every GroupNorm falls back to `gn_stats`,
   which walks a group's up-to-16 M elements as one serial ascending sum instead
   of a 256-way tree. Measured worst relative L2 at 512²: **1.8e-5 GPU vs 4.3e-4
   CPU**; at 128² the two are indistinguishable (6.3e-6 vs 6.4e-6) because the
   groups are 64× smaller. Indices still match 0/256 and cosine is still
   0.9999999 on CPU. Consequence: **the CPU backend is not a parity oracle at
   512²**, and any future cross-backend gate must budget for this.

10. **The parity suite peaked at 90% of the card and flaked once.** Each 512²
    case builds a `taps = true` model, which pins every activation (the pool is
    off) — **6.9 GB measured on a P40** for one 512² graph. `cargo test` uses
    the core count for `--test-threads` (48 here), so all three ran together and
    the suite peaked at **22.2 GB of 24.5 GB**. One full-suite run was observed
    failing and did not reproduce in 11 retries — the signature of an allocation
    losing a race with a concurrent GPU user, which on this box is routine
    (sibling ports run their own parity suites). The 512² cases are now
    serialized behind one mutex: peak **22.2 GB → 9.8 GB** (2.3×) for ~3 s of
    wall time, and 6 consecutive full-suite runs are clean. Per-test peaks:
    512² 6.9 GB, `stages_128` 0.75 GB, `pooled_matches_tapped` (two 128²
    models) 1.2 GB.

## Import coverage

`vqgan::import::load(path, &cfg)` strips a uniform `params_ema.`/`params.`/
`state_dict.` wrapper, then validates **both directions**: every one of the 329
manifest tensors present exactly once with the exact shape and element count
(missing → error naming it), and every source tensor either consumed or matching
a **declared** `CODEFORMER_ONLY` prefix (`position_emb`, `feat_emb`, `ft_layers`,
`idx_pred_layer`, `fuse_convs_dict`), returned in `Import::skipped`. Anything
else is an error naming the tensor. **No zero-fill path exists.**

Measured: `vqgan_code1024.pth` → 329 imported / 0 skipped;
`codeformer.pth` → 329 imported / **186** skipped (515 − 329).

## Stubbed / not done

- Backward: nothing. Forward only.
- `perplexity`, `mean_distance`, `codebook_loss` are not computed (training
  statistics; `mean_distance` needs the full `[M,K]` matrix `vq_argmin`
  deliberately does not emit). The parity test asserts exact index equality
  instead, which subsumes perplexity.
- `vq.d` / `vq.d_direct` (the full distance matrices) are unchecked by design.
- q/k/v are not separately observable — the shared attention fuses them into one
  1×1 conv. `sub.attn17.{norm,proj_out}` bracket them and both match, which
  pins the fusion order.
- The graph emits the **raw gather**, not the straight-through
  `z + (z_q − z).detach()`; they agree at max\|Δ\| 6.0e-8 in the forward and
  differ only in the backward.
- No CLI subcommand, no `Model` impl, no INT8, no batching. Input is
  `[3,H,W]` fp32 NCHW from the caller (`crates/imaging` is not on this path
  yet); `H`/`W` must be multiples of the 32× downscale (asserted), and the
  graph is pre-recorded for one `(h,w)`.
- The encode and decode `Builder`s each own a 512 MiB im2col scratch
  (`BRAIN_VAE_COL_MIB`); they are not shared between the two graphs.
- `taps = true` is a whole-graph switch: there is no way to record *some*
  stages, so a 512² parity model costs 6.9 GB where the pooled production graph
  costs ~1 GB. A selective tap list would need `Builder::free` to know which
  buffers were tapped (buffer identity, which `DeviceBuffer` does not expose) —
  left for the backward workstream, which needs the same distinction.

## Deferred

**Backward / gradcheck**
- FD anchors already exist: the 67 taps in `stages_128.safetensors` at 128².
  Every stage is a fresh SSA buffer, so the forward cache is already the shape
  the backward wants — but the pool **aliases** buffers when `taps = false`, so
  a training build needs `taps`-like lifetimes or an explicit no-reuse mode.
- The VQ straight-through path needs `emb_bwd` (exists) plus the
  commitment/codebook MSE terms; `wm_core::vq` documents the composition.
- The trainable encoder needs the straight-through form (`add2` + a subtract) —
  two extra steps in `Vqgan::new`, no new kernel.
- `gradcheck::check_vqgan` needs a `Model` impl; there is none.

**Serving contract**
- One-shot image-in/image-out, so `capability::Action`'s `Progress` timeline
  collapses to a single latency (same treatment as `depth`/`yolo`).
- `run_batch`: every kernel takes an `N`, but `vae::blocks::Builder` hardcodes
  `N = 1`. True batching is a change to the **shared** builder, so it lands for
  `AutoencoderKL` at the same time — do it there, not in a vqgan fork.
- Residency: two graphs on one `Gpu`; ~72 M params fp32 ≈ 290 MB of weights plus
  ~4 GB of activations at 512² with taps off.
- D-Bus: image in / image out fits the existing fd-blob surface.

**CodeFormer (task #5, second half)**
- `Vqgan::generate(&z_q)` is the seam, verified bit-identical to `decode`.
- CodeFormer inference **bypasses** the argmin
  (`softmax(logits) → topk(1) → get_codebook_feat`); that path is
  `Codebook::lookup`, public and bit-exact against the golden `codebook_feat`.
- The 512² goldens already carry the CFT tap points (encoder
  `{2,5,8,11,14,18}` ↔ generator `{6,9,12,15,18,21}`, `codeformer_arch.py:204`),
  all at cosine 1.000000000 — no re-dump needed for the fuse inputs.
- Still to dump: the 9-layer `TransformerSALayer` I/O, `position_emb`,
  `feat_emb`, `idx_pred_layer` logits, `Fuse_sft_block` scale/shift, and a
  sweep over `w`. `import::CODEFORMER_ONLY` names exactly the prefixes those
  weights live under; the follow-up extends the manifest and removes them from
  that list.
