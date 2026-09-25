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
| SH in `render`/`view` (host shading per frame via `sh::shade`) | `cli/splat_cli.rs` | done |
| Ray-evaluated renderer (3DGUT-style, Wu et al., CVPR 2025): every pixel evaluates each gaussian at its maximum along the pixel's own ray through the real lens (pinhole, OpenCV Brown, Kannala-Brandt fisheye, equirectangular), rolling shutter, unscented tile bounds, 3D and 2D Mip filters with ray-marginal compensation; gradients of every gaussian parameter and of pose, shutter and calibration | `crates/camera`, `renderer.rs`, `splat_ray_*.wgsl`, `lib/splat_ray_pair.wgsl` | done |
| Tile-cooperative ray backward: one workgroup per tile, records reduced in workgroup memory at their emission slots (no record sort); 45.6 ms per view at 500k gaussians and 816x612 against ~480 ms for the slot grid it replaced (which stays as the per-pixel reference and CPU path) | `splat_ray_bwd_tile.wgsl`, `splat_gather_ids.wgsl` | done |
| Environment at infinity: real SH radiance by direction up to degree 8, composited behind the gaussians along each pixel's ray, fitted with the scene, baked into a distant shell for export | `env.rs`, `splat_env*.wgsl`, `FitCfg::environment` | done |
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
| Per-pixel loss on the device (`DeviceLoss`, `l1ssim_*`) | `loss.rs` | done |
| Camera model (ISP) forward/backward on the device (host today, one readback per view) | kernels | planned |

### Geometry

| capability | where | state |
|---|---|---|
| Range supervision along each pixel's ray: pseudo-Huber on log range, per-pixel confidence | `splat_geom_loss.wgsl`, `FitCfg::depth_weight` | done |
| Depth distortion (2DGS, Huang et al., SIGGRAPH 2024), the exact pairwise form composited in the rasterizer | `splat_ray_pair.wgsl`, `FitCfg::distortion_weight` | done |
| Normal consistency against the rendered range's own normals (2DGS), only where no normal prior exists | `splat_geom_loss.wgsl`, `FitCfg::normal_consistency_weight` | done |
| Normal prior supervision (`TargetView::with_normals`) | `geometry.rs`, `FitCfg::normal_prior_weight` | done |
| Monocular priors (MoGe-2 / Metric3D v2) with robust scale/shift alignment to SfM | model crates + `align.rs` | planned |
| Sky / background environment model at infinity | `env.rs` | done (see Renderer) |
| Transient masking by residual coherence (RobustNeRF, Sabour et al. 2023): large coherent unexplained regions per view stop being supervised; 4.6 dB better where a transient was in a synthetic test | `loss::transient_mask`, `FitCfg::transients` | done (opt-in) |
| Semantic dynamic-object masks from SAM 2 (`crates/sam2` exists) wired into `TargetView::mask` | `recon` | planned |
| Mesh extraction: GOF opacity-field + marching tetrahedra, SuGaR-style Poisson, visibility Delaunay; texture baking | new crate | planned |

### Density control

| capability | where | state |
|---|---|---|
| Heuristic clone/split/prune (3DGS) with AbsGS homodirectional gradient (Ye et al., 2024) | `opt.rs::densify`, `splat_grad_reduce.wgsl` | done |
| 3DGS-MCMC relocation + SGLD noise + budget (Kheradmand et al., NeurIPS 2024) | `mcmc.rs` | done |
| **Hybrid credit-assigned controller**: per-gaussian contribution `U=Σ Tα`, residual responsibility `R=Σ Tα r / U`, edge-weighted responsibility, visibility count, AbsGS gradient; percentile-normalized scores; moment-preserving long-axis split (ImprovedGS-style, Deng et al. 2026); recovery-aware pruning on a starvation record (exponential average, reclaimed at 0.6); budgeted growth schedule `N_target(t)` (Taming 3DGS, Mallick et al. 2024); relocation of reclaimed slots | `density.rs`, `Densify::Hybrid` | done |
| Geometric evidence: the credited residual includes `|ln(rendered / prior range)|` where a view has a range prior | `density::range_residual` | done |
| Spawn into residual pixels where the scene is empty or at the wrong depth, placed by the range prior (or the rendered range), budgeted by the residual share they carry | `density::Site`, `opt::residual_sites` | done |
| Children of flat gaussians displaced across their surface only | `density::in_surface` | done |
| Per-gaussian SH degree from its own views (count and angular spread) | `sh::supported_coefficients`, `splat_sh.wgsl` limit | done |
| Low-texture coverage (large thin surface-aligned gaussians on walls) | `density.rs` | planned |

### Capture, SfM, calibration (upstream of `fit`)

| capability | where | state |
|---|---|---|
| Incremental SfM from photographs alone: SIFT (Lowe 2004) + RootSIFT, mutual ratio-test matching, 8-point essential in adaptive RANSAC, P3P registration, DLT triangulation, Schur-complement LM bundle adjustment with shared f/k1/k2, re-triangulation; focal length chosen by solving under candidates | `crates/sfm` | done |
| Pair selection by retrieval: VLAD (intra-normalized, power-normalized) over the RootSIFT already computed, vocabulary by k-means on the capture itself (no weights), top-k per image plus capture-order neighbours; exhaustive up to 24 photographs | `sfm::retrieval`, `SfmCfg::pairs` | done |
| Global initializer (GLOMAP-style): relative rotations re-estimated through the swept focal length, robust rotation averaging (max spanning tree, then L1 and Geman-McClure IRLS on the Lie algebra, CG solve), global positioning of cameras and points from bearings (BATA objective, random start, exact alternation), triangulation and bundle adjustment under a tightening error bound; incremental kept as fallback, chosen by registered views then RMS | `sfm::{rotation, positioning}`, `incremental::Solver::global` | done |
| Metric scale + gravity: WGS84 -> local ENU, Umeyama Sim(3) in LO-RANSAC from camera centres to fixes (refused when the fixes' misfit exceeds 10% of their spread), gravity from the cameras' horizontal axes (robust), held exactly in the Sim(3) when known; `Reconstruction::{gauge, up}`; `splat::orient::upright_along` / `TrainingSet::upright` land the known up | `sfm::georef`, `recon::photogrammetry` | done |
| Photographs -> targets exactly as recorded through their own lens (masked where it has no ray), exact halvings, landed upright, dense stereo priors and start; one pipeline for `brain splat train`, the SDK and the sample, with automatic steps and budget | `recon::photogrammetry::reconstruct` | done |
| Frame selection (sharpness + near-duplicate) | `crates/recon/src/select.rs` | done (baseline; not yet wired into `train`) |
| Pose, rolling-shutter and calibration (focal, principal point, distortion) refinement inside the fit, gauge fixed | `opt.rs`, `FitCfg::camera` | done (off in the presets until measured to help) |
| Per-image or mixed cameras (today one camera and one zoom for the whole capture), principal point refinement | `sfm` | planned |
| Device SIFT and matching (host today: ~35 s of a 51 s run on 16 photos is features + all-pairs matching) | kernels | planned |
| Staged calibration inside the fit (poses -> shared intrinsics -> distortion) with SfM priors; video pose smoothness | `opt.rs` | planned |
| ALIKED + LightGlue/LoMa matching, MAGSAC++, 360° as a cubemap rig | `sfm` | planned |
| IMU gravity (per-photo accelerometer, where a container records it) | `sfm::georef` | planned |
| Feature extraction parallel across photographs as well as within one: 94-141 s -> 13.3 s for 16 x 8 MP, same reconstruction | `sfm::incremental` | done |
| Parallel pair matching (now the largest SfM cost, ~200 s for 120 pairs at 8000 features) | `sfm` | planned |
| Global SfM measured on a capture that breaks the incremental path (drift over a long walk); the chessboard registers fully either way | `sfm` | planned |

Measured on the 16-photo chessboard capture (3264x2448 ultrawide, network
GPS in EXIF), `crates/sfm/examples/sfm_folder.rs`, on a shared 48-core host
whose load varied between runs (so seconds compare within a run better than
across runs):

| run | pairs matched / verified | matching | solve | registered | points | RMS | lens |
|---|---|---|---|---|---|---|---|
| before (exhaustive, incremental) | 120 / 105 | 155 s | 28 s | 16/16 | 7104 | 0.645 px | fisheye f 1500.6 |
| default now (exhaustive at 16, global kept) | 120 / 105 | 187 s (higher load) | 19 s | 16/16 | 7104 | 0.645 px | fisheye f 1500.6 |
| `--pairs 5` (retrieval) | 53 / 47 | 85 s | 17 s | 16/16 | 6565 | 0.608 px | fisheye f 1499.2 |
| `--pairs 8` (retrieval) | 72 / 65 | 123 s | 33 s | 16/16 | 6805 | 0.626 px | fisheye f 1500.5 |

The global start alone (before the lens fit) reaches 16/16 at 0.726 px; rotation
averaging keeps 85 of 102 pair rotations (the rest miss by over 5 degrees
through the not-yet-fitted lens). Gravity from the cameras is 0.38 degrees
from the normal of the chessboard's plane (83% of the points); the network
GPS (arc-second cells, ~1 m misfit over ~1 m of camera spread) is refused,
as it should be. Features (94-141 s) are now the largest cost.

### Held-out quality on the chessboard capture

16 phone photographs (ultrawide, fisheye lens), every fourth held out (12
fitted), 816x612, 3000 steps, one P40 (`crates/recon/examples/photo_holdout.rs`):

| pipeline | held-out PSNR / SSIM | training PSNR | time |
|---|---|---|---|
| undistorted pinhole targets, sparse start (before) | 13.83 dB / - | 30.96 dB | - |
| native-lens targets, device optimizer, slot backward | 15.10 / 0.433 | 27.82 | 3421 s |
| same, tile backward, sparse start | 15.12 / 0.449 | 26.19 | 337 s |
| dense stereo start and priors | 16.89 / 0.574 | 24.71 | 969 s |
| + spawning, geometric evidence, per-gaussian SH | 16.93 / 0.577 | 25.44 | 923 s |
| priors weighted 0.3 / 0.1 instead of 0.1 / 0.05 | 16.72 / 0.574 | 24.62 | 1376 s |
| + environment (degree 6) | 16.88 / 0.591 | 24.89 | (shared GPU) |
| + pose refinement from 20%, calibration from 50% | 16.82 / 0.558 | 23.97 | 1008 s |

Refining the cameras loses here: structure from motion already fits these
photographs to 0.64 px, and the refinement trades scene quality for pose
changes the held-out views do not confirm. The presets leave it off.

What remains wrong in held-out views is where 12 photographs do not
constrain the scene: the frame borders (fence, deck seen by one or two
views), the glass table (transparent and reflective), and the thin chess
pieces. LPIPS is not reported: it needs pretrained network weights the
engine does not ship.

### Dense geometry (`crates/mvs`, upstream of `fit`)

GPU PatchMatch multi-view stereo through the real lens (no rectification):
per-view range/normal/confidence maps, fused points, and a surface-aligned
gaussian start (`mvs::to_splats`). Measured on the 16-photo chessboard
capture at 1632x1224 on one P40: 59 s of stereo (`mvs_pm` is 99.6 % of
device time), 62 % mean coverage, 3.1 M fused points in 6 s
(`crates/mvs/examples/mvs_folder.rs`).

| capability | where | state |
|---|---|---|
| Source view selection + per-view range bounds from SfM tracks | `mvs::select` | done |
| Red-black adaptive propagation, joint view selection, refinement, bilateral NCC through any lens | `mvs_pm.wgsl` | done |
| Coarse-to-fine pyramid with geometric consistency; consistency filter | `mvs::stereo`, `mvs_upsample.wgsl`, `mvs_filter.wgsl` | done |
| Fusion (gather-only ownership by finest footprint), surface-aligned splat init | `mvs::fuse`, `mvs::init` | done |
| Wired into `recon::photogrammetry` (`TrainingSet::densify`, `dense`) and the pipeline: every target's range, normal and confidence priors, the fused start; stereo on training photographs only when judging held-out views | `recon` | done |
| Truly textureless surfaces (planar priors, Xu & Tao AAAI 2020). Keeping zero-texture pixels on geometric consistency alone was measured wrong: a plane propagated into empty background is consistent in every view, 13.5 % of a synthetic view's measurements landed off the surface | `mvs` | planned |
| CPU backend: the `mvs_*` kernels use vector/struct locals and helper functions the CPU JIT does not lower yet | `wgsl-cpu` | blocked |
| Second P40: split a sweep's views across devices (each view then reads its sources' maps from the previous sweep rather than in order) | `mvs::stereo` | planned |

### Throughput (measured on a 2xP40 box, 16 photos at 1024x768)

The fit's cost is dominated by the backward, whose gradient records scale
with pixels x compositing depth: a scene grown from a sparse cloud is early on
a few thousand large blobs ~40 deep per pixel, 33M records per view on
average, 2-2.5 s per view. Host SSIM is ~150 ms per 1024x768 view.

- [x] View minibatches (`FitCfg::batch`) with epoch-level loss averaging and
      step-size backoff; geometry passes every Nth step (`geometry_every`).
- Measured end to end: 16 photos (2048x1536, no EXIF) -> 16/16 registered
  at 0.98 px -> 300k gaussians at 768x576 in 4000 iterations, ~60 min on one
  P40, loss 0.31 -> 0.060. Training views reproduce deck grain, fence and
  object; views on the capture orbit between photographs hold the object's
  shape but show floaters near thin parts and the thin dark sprinkler rose
  as a blur - the next quality work (below).
- [ ] Floater suppression beyond the surface terms (opacity reset or decay
      late in the fit, a visibility-count prune) and thin-structure
      coverage; judged on held-out views, not training views.
- [x] Backward restructured (measured on the trained 300k-gaussian scene at
      768x576, 2 views a step): per-(pixel, gaussian) records sorted by
      gaussian were 70% of the backward's device time and the one-thread-per-
      gaussian reduce 20%. Now each (tile, gaussian) instance is reduced over
      its 256 pixels in a fixed slot grid (`splat_bwd_slots`,
      `splat_bwd_tile_reduce`, one barrier) and only instances are sorted:
      backward 1178 -> 240 ms per step.
- [x] L1 + D-SSIM on the device (`l1ssim_*`): 447 -> 10 ms per step. Whole
      step 2419 -> 639 ms (`crates/splat/examples/fit_profile.rs`).
- [ ] Next: the geometry passes (~180 ms per step, host feature building and
      two extra forward+backward per view), the camera model's host round
      trip (~65 ms), and `splat_bwd_slots` itself (1.48M instances x 256
      pixels of slots per view).
- [ ] A gaussian-major backward (per-tile accumulation instead of per-pixel
      records sorted by gaussian) - the step change.
- [x] Coarse-to-fine resolution schedule for the sparse-start phase
      (`FitCfg::coarse`, `TargetView::half`).
- [x] Start opacity 0.5 rather than 3DGS's 0.1 (`--init-opacity`): 2.9x
      faster per iteration at 768x576 and lower in loss at iteration 100.
- [ ] Loss, SSIM and camera model on the device.

### Quality gates

End to end, the pipeline is judged on views the fit never saw:

- `crates/recon/examples/synthetic_e2e.rs` renders a known scene of flat
  gaussians (textured ground, sphere, box) from two camera rings and scores
  the reconstruction on cameras at other azimuths, heights and distances,
  against the truth: the trainer alone on the true cameras (A), and the whole
  pipeline from the images through structure from motion (B).
  `--bare-sphere` starts the sphere without a point, as structure from motion
  leaves a textureless object.
- `crates/recon/examples/photo_holdout.rs` holds every k-th photograph of a
  real capture out of the fit and scores against it.

At 512x384, 3000 iterations, 100k gaussians: A 44.2 dB on training views,
36.5 dB held out (views between the training rings 36-41 dB, views below them
31.5-35.5 dB); B 42.3 / 32.3 dB, the 4 dB between them being camera
precision (centres within 0.58% of the rig radius, focal within 0.17%). What
this found and fixed, each measured on held-out views:

- Geometry budgets stepped every gaussian at the MEDIAN one's rate, re-measured
  as density control subdivided the scene, so nothing could travel to a region
  structure from motion left empty: the bare sphere came back half missing
  (29.8 dB on its training views, 40.2 with each gaussian's own radius), and
  the real capture's textureless can came apart (18.6 -> 25.8 dB on training
  photographs).
- The camera model absorbed fit error on captures taken at one exposure
  (-2.5 dB on synthetic training views, -1.3 dB held out on a real capture);
  off by default, `--camera-model` turns it on.
- SH degree 3 on a few dozen views memorizes them (-2.4 dB held out at 24
  views); the degree now follows the view count.
- The Mip filter's compensation cannot be baked into a PLY; the preset fits
  under the dilation viewers render (+0.5 dB held out on synthetic).
- Structure from motion: an adaptive RANSAC bound that ended the search after
  one sample when the first model had no support, a SIFT pyramid that never
  doubled its input, and Lowe's contrast threshold - together 18/24 -> 24/24
  registered, 1.2% -> 0.58% worst camera centre, 3.9x the points.

Open: B's camera precision (photometric pose refinement gains 1.3 dB on
synthetic but loses 0.35 on the real capture); thin dark structure (a
sprinkler rose) stays soft; held-out views below every training camera.

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
(`splat_grad_reduce`, `splat_bwd_slots`, `splat_bwd_tile_reduce`,
`splat_ray_bwd_tile`, `splat_project_bwd`) contain zero atomic operations -
every reduction runs in a fixed order, and `splat_grad_reduce` sums each
gaussian's records from its contiguous emission slots - so bit-determinism was
expected, and is now a measured fact rather than an
assumption. Following from this, the `fit` capability action's own
caps-vs-library test is gated at bit-identity for a fit run in isolation; the
tiny (sub-1e-6) deviations it separately measures between a `fit` action's PLY
output and the library call's raw `Splats` are the ALREADY-VERIFIED (item 3)
PLY round-trip's own quantization, not fit nondeterminism - the two are
distinguished by comparing bit-for-bit BEFORE any PLY round trip (this test)
and by tolerance AFTER one (the capability action test, whose wire format IS
PLY bytes).
