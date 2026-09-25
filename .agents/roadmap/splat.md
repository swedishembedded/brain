# splat — roadmap

Gaussian-splatting renderer/trainer (device sort/scan primitives, tiled
rasterizer, interactive viewer, and a differentiable backward pass), used for
brain's world-model image reconstruction pipeline and as a standalone
scene optimizer.

## Target: a professional reconstruction engine, not "3DGS + one trick"

Splat quality comes from eight cooperating systems, and a plain 2023 3DGS
trainer covers roughly the first two:

```text
capture preprocessing -> feature matching -> SfM / BA -> initialization
  -> differentiable renderer (projection, anti-aliasing, ordering)
  -> photometric camera model -> losses (RGB + SSIM + geometry + ISP reg)
  -> adaptive density (credit-assigned, budgeted) -> cleanup / meshing
```

Each row below is one capability, where it lives, and its state. "Paper" is
the specification it is built from - see **Provenance** for what that means.

### Renderer

| capability | where | state |
|---|---|---|
| EWA/Jacobian projection, tiled sort, per-tile compositing, atomic-free fwd+bwd | `renderer.rs`, `splat_*.wgsl` | done |
| Mip-Splatting 2D Mip filter + 3D smoothing floor (Yu et al., CVPR 2024) | `mip.rs`, `FitCfg::{antialiased,mip_scale}` | done |
| SH degree 0-3 view-dependent colour | `splat_sh.wgsl`, `FitCfg::sh_degree` | done |
| Progressive SH bands (SH0 -> SH3 over training) | `splat_sh.wgsl` held-out coefficient count, `FitCfg::sh_ramp` | done |
| Auxiliary render passes (per-gaussian features composited by the same weights: normals, depth moments, credit assignment) | `opt.rs::geometry_passes`, `density::credit_upstream` | done |
| SH in `render`/`view` (today they draw the DC colour only; a fitted scene's view dependence is exported to the PLY and used by other viewers) | CLI render path + `splat_sh.wgsl` | planned |
| Nonlinear cameras via the Unscented Transform (3DGUT, Wu et al., CVPR 2025): OpenCV radial/tangential, fisheye/Kannala-Brandt, equirectangular with seam duplication, rolling shutter | `camera.rs` (planned) | planned |
| StopThePop hierarchical per-tile re-sorting (Radl et al., SIGGRAPH 2024) | rasterizer | planned |
| `.splat` / `.spz` IO | `ply.rs` siblings | planned |
| Render-time optimization: per-stage profiling, radix chunk tuning, pipelined present | renderer | planned |

### Losses and photometric model

| capability | where | state |
|---|---|---|
| MSE (legacy default) | `loss.rs` | done |
| L1 + D-SSIM (λ=0.2, the 3DGS objective, Kerbl et al. 2023) | `loss.rs`, `FitCfg::loss` | done |
| Per-view exposure + white balance, per-sensor chromatic vignetting + response curve, gauge-fixed and identity-regularized, staged enable (PPISP-style decomposition, Deutsch et al. 2026) | `isp.rs`, `FitCfg::isp` | done |
| Scene-linear training: sRGB targets decoded, sRGB OETF inside the camera model, exposure brackets as separate measurements of one radiance, clipped-pixel down-weighting | `isp.rs` (`ColorSpace::SceneLinear`) | done |
| Low-capacity bilateral-grid residual for computational-photography captures (Wang et al., SIGGRAPH Asia 2024) | `isp.rs` | planned |
| Novel-view ISP controller (predict exposure/WB for unseen views) | `isp.rs` | planned (novel views render at the gauge-fixed mean camera today) |
| RGBA captures: random background per iteration + alpha loss | `opt.rs` | planned |
| Loss + ISP on the device (today the per-pixel loss runs on the host, one readback per view) | kernels | planned |

### Geometry

| capability | where | state |
|---|---|---|
| Expected-depth supervision with per-pixel confidence | `opt.rs`, `FitCfg::depth_weight` | done |
| Depth distortion (2DGS, Huang et al., SIGGRAPH 2024), squared form `2(A·M2 - M1²)` | `geometry.rs`, `FitCfg::distortion_weight` | done |
| Normal consistency against the rendered depth's own normals (2DGS) | `geometry.rs`, `FitCfg::normal_consistency_weight` | done |
| Normal prior supervision (`TargetView::with_normals`) | `geometry.rs`, `FitCfg::normal_prior_weight` | done |
| Monocular priors (MoGe-2 / Metric3D v2) with robust scale/shift alignment to SfM | model crates + `align.rs` | planned |
| Sky / background environment model at infinity | new `background.rs` | planned |
| Dynamic-object masks from SAM 2 (`crates/sam2` exists) wired into `TargetView::mask` | `recon` | planned |
| Mesh extraction: GOF opacity-field + marching tetrahedra, SuGaR-style Poisson, visibility Delaunay; texture baking | new crate | planned |

### Density control

| capability | where | state |
|---|---|---|
| Heuristic clone/split/prune (3DGS) with AbsGS homodirectional gradient (Ye et al., 2024) | `opt.rs::densify`, `splat_grad_reduce.wgsl` | done |
| 3DGS-MCMC relocation + SGLD noise + budget (Kheradmand et al., NeurIPS 2024) | `mcmc.rs` | done |
| **Hybrid credit-assigned controller**: per-gaussian contribution `U=Σ Tα`, residual responsibility `R=Σ Tα r / U`, edge-weighted responsibility, visibility count, AbsGS gradient; percentile-normalized scores; moment-preserving long-axis split (ImprovedGS-style, Deng et al. 2026); recovery-aware two-round pruning; budgeted growth schedule `N_target(t)` (Taming 3DGS, Mallick et al. 2024); relocation of reclaimed slots | `density.rs`, `Densify::Hybrid` | done |
| Geometry-residual gating (don't densify an appearance-only error: high RGB, low depth/normal residual) | `density.rs` | planned (needs a depth prior per view; the signal slot exists) |
| Spawn into uncovered high-residual pixels (unproject from the depth prior) | `density.rs` | planned |
| Low-texture coverage (large thin surface-aligned gaussians on walls) | `density.rs` | planned |

### Capture, SfM, calibration (upstream of `fit`)

| capability | where | state |
|---|---|---|
| Incremental SfM from photographs alone: SIFT (Lowe 2004) + RootSIFT, mutual ratio-test matching, 8-point essential in adaptive RANSAC, P3P registration, DLT triangulation, Schur-complement LM bundle adjustment with shared f/k1/k2, re-triangulation; focal length chosen by solving under candidates | `crates/sfm` | done |
| Photographs -> undistorted pinhole targets + masks + point-cloud init; `brain splat sfm`, `brain splat train` | `recon::photogrammetry`, `splat::init`, `cli/splat_cli.rs` | done |
| Frame selection (sharpness + near-duplicate) | `crates/recon/src/select.rs` | done (baseline; not yet wired into `train`) |
| Pose refinement inside the fit | `opt.rs`, `FitCfg::pose_lr` | done |
| Five-point essential solver (the 8-point one is weak on near-planar pairs) | `sfm::twoview` | planned |
| Per-image or mixed cameras (today one camera and one zoom for the whole capture), principal point refinement | `sfm` | planned |
| Device SIFT and matching (host today: ~35 s of a 51 s run on 16 photos is features + all-pairs matching) | kernels | planned |
| Staged calibration inside the fit (poses -> shared intrinsics -> distortion) with SfM priors; video pose smoothness | `opt.rs` | planned |
| ALIKED + LightGlue/LoMa matching, MAGSAC++, global SfM, 360° as a cubemap rig | `sfm` | planned |
| GPS Sim(3) (Umeyama inside RANSAC) + IMU gravity | `align.rs` has Umeyama | planned |

### Throughput (measured on a 2xP40 box, 16 photos at 1024x768)

The fit's cost is dominated by the backward, whose gradient records scale
with pixels x compositing depth: a scene grown from a sparse cloud is early on
a few thousand large blobs ~40 deep per pixel, 33M records per view on
average, 2-2.5 s per view. Host SSIM is ~150 ms per 1024x768 view.

- [x] View minibatches (`FitCfg::batch`) with epoch-level loss averaging and
      step-size backoff; geometry passes every Nth step (`geometry_every`).
- [ ] A gaussian-major backward (per-tile accumulation instead of per-pixel
      records sorted by gaussian) - the step change.
- [x] Coarse-to-fine resolution schedule for the sparse-start phase
      (`FitCfg::coarse`, `TargetView::half`).
- [x] Start opacity 0.5 rather than 3DGS's 0.1 (`--init-opacity`): 2.9x
      faster per iteration at 768x576 and lower in loss at iteration 100.
- [ ] Loss, SSIM and camera model on the device.

### Quality gates

`tests/s*.rs` gate each capability on small synthetic scenes on the CPU JIT
(`Gpu::new_cpu`). The CI benchmark set the design calls for (fabric/fence,
glossy, blank room, sky, 360, fisheye, exposure-changing video, HDR bracket,
moving people, thin geometry, close fly-through, orbit popping, sparse views)
does not exist yet; every row above that says "done" is gated by a
synthetic test that isolates its mechanism, not by a real-capture benchmark.

## Provenance: clean-room from papers, Apache-2.0 throughout

1. Implement from the **paper's equations**, never from a restrictively
   licensed reference implementation. The original 3DGS, Mip-Splatting,
   AbsGS, 3DGS-MCMC, StopThePop, ImprovedGS, 2DGS, GOF, SuGaR and
   WildGaussians releases inherit the Inria non-commercial license or other
   restrictions: read the paper, not the repository, and copy no constants,
   identifiers, kernel layouts or comments from them.
2. Permissive implementations (3DGUT/3DGRUT, PPISP and bilateral-grid
   processing are Apache-2.0; LightGlue Apache-2.0; ALIKED BSD-3; MAGSAC++
   BSD; COLMAP BSD; SAM 2 Apache-2.0; xatlas MIT) may be depended on or
   ported with their notices preserved - after a dependency review.
3. Model **weights** carry their own terms, separate from source licenses
   (SuperPoint is restrictive; Metric3D asks commercial users to contact the
   authors). Check them per release.
4. A clean-room implementation settles copyright provenance only; patents and
   dataset terms are a separate review.

Each module's header names the paper and the equation it implements, and
states where the implementation deliberately departs from it and why (e.g.
`mcmc::add_noise` samples `L ε` rather than the reference's `Σ ε`).

## Older open items

- [ ] Accumulate-mode attention backward for chunked (multi-view) fits.
- [ ] Backward pass computes per-view gradients un-chunked; chunking across
      views is not supported.

Finite differences are deliberately not used as the backward-pass oracle:
gaussian rasterization's 1/255 output truncation biases finite-difference
gradients in a way analytic (autograd-checked) gradients do not share.

## Serving contract - done

`render` (one-shot) and `fit` (streaming) are now [`capability::Provider`]
actions (`crates/splat/src/caps.rs`), registered in the residency scheduler
(`crates/cli/src/resident_splat.rs`) and the CLI catalog
(`crates/cli/src/catalog.rs`) under `brain/splat` - reachable over `brain do`,
D-Bus, and the event API with no splat-specific plumbing in any transport.
`view` (the interactive SDL fly-through) is deliberately NOT served: it has no
request/response shape. `fit`'s optimization loop
(`crates/splat/src/opt.rs::fit`) grew an `on_step(iter, mse) -> bool` hook,
polled once per completed iteration, that the CLI wires to a no-op-true
closure and the capability action wires to `Progress::step` + the
invocation's cancel token - so a served `fit` run reports live MSE and
aborts cleanly (`Err("cancelled")`) within a few iterations of being asked
to.

### Determinism finding (the step-0 sub-blocker)

`opt::fit`'s run-to-run bit-determinism was not previously established.
Probed by calling it twice on identical inputs and diffing the two returned
scenes' `f32::to_bits()` bitwise (`crates/splat/src/caps.rs`'s
`determinism_probe_fit_twice_on_identical_inputs` test). Measured result (Intel
Arc integrated GPU, Vulkan backend): **`0` differing bits** - `mse
0.000007867729` on both runs, identical to the ULP. The backward kernels
(`splat_grad_reduce`, `splat_bwd_emit`, `splat_bwd_count`, `splat_bwd_keys`,
`splat_project_bwd`) contain zero atomic operations - `splat_grad_reduce` is a
deterministic per-gaussian segmented reduction over id-sorted gradient records
- so bit-determinism was expected, and is now a measured fact rather than an
assumption. Following from this, the `fit` capability action's own
caps-vs-library test is gated at bit-identity for a fit run in isolation; the
tiny (sub-1e-6) deviations it separately measures between a `fit` action's PLY
output and the library call's raw `Splats` are the ALREADY-VERIFIED (item 3)
PLY round-trip's own quantization, not fit nondeterminism - the two are
distinguished by comparing bit-for-bit BEFORE any PLY round trip (this test)
and by tolerance AFTER one (the capability action test, whose wire format IS
PLY bytes).
