# 3D Gaussian Splatting (splat)

brain's own from-scratch 3D Gaussian Splatting renderer and optimizer: load
a `.ply` scene, render still images or fly through it interactively, fit a
scene against a set of posed photos, or train one from a folder of ordinary
photographs with no camera information at all. Reach for it to view or inspect
any Gaussian-splat scene - including ones produced by [mirror](worldmirror2.md) -
or to optimize a scene of your own from a set of camera-posed images. The
same rendering kernels run on any GPU and on the CPU, so it works without
a discrete graphics card.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| CLI (`brain <arch> <action>`)       | [ ] (has its own `brain splat …` subcommand instead - see "Running it" below) |
| HTTP API               | [ ] `render`/`fit` both take a REQUIRED `scene` input blob, so neither is a text-to-image/chat action |
| D-Bus                  | [x] `render` (one-shot) and `fit` (streaming); `view` is human-in-the-loop and is not served |
| Batched serving        | [ ] |

## Getting the weights

There's no fetched model here - splat works directly on Inria-format `.ply`
scene files, whether produced by `brain worldmirror2 infer`, brain's own `fit`
(below), or another Gaussian Splatting tool.

## Running it

```bash
brain splat info   scene.ply
brain splat render scene.ply --out img.ppm [--eye x,y,z --target x,y,z]
brain splat view   scene.ply         # interactive fly-through (WASD + mouse)
brain splat fit     scene.ply --images photos/ --cameras cameras.json --out out/fitted.ply
brain splat sfm     --images photos/ --out out/sfm.ply          # cameras + sparse points only
brain splat train   --images photos/ --out out/scene.ply        # photographs -> finished scene
```

`view` opens an interactive window. With no `--eye`, the camera auto-frames
the scene from its bounds. Viewer controls:

| Key | Action |
|---|---|
| W A S D | move |
| Space / C | up / down |
| Shift | sprint |
| `m` | toggle mouse-look |
| Arrow keys | look |
| `[` / `]` | render quality (1x, 1/2, 1/4 resolution) |
| `v` | color / depth view |
| `p` | screenshot |
| Enter | reset camera |
| Esc | quit |

`fit` optimizes an existing `.ply` scene against a set of posed target
photos - the same rasterizer, run backward. `--cameras` takes the
`cameras.json` format `brain worldmirror2 infer` produces (a list of camera poses
and intrinsics); `--images` is a directory (or comma-separated list) of
photographs in the same order as the cameras - PPM, PNG, JPEG, BMP or TIFF.

Targets are resampled to their camera's size, so the SAME folder that produced
a reconstruction can be fitted against without a conversion step. A camera
recovered by `worldmirror2 infer` describes the model's own 518-px grid, never
the photograph's full resolution, so they essentially never match already. A
target whose ASPECT differs from its camera by more than 5% is refused instead
of stretched. This is how you turn a starting point cloud
(from `brain worldmirror2`, or your own SfM/COLMAP output converted to `.ply`)
into a scene that actually reproduces your photos.

## Options

| Flag | Command | Effect |
|---|---|---|
| `--out <path>` | `render`, `fit` | output file (`img.ppm` / `fitted.ply`) |
| `--width` / `--height` | `render`, `view` | output resolution |
| `--eye x,y,z --target x,y,z [--up x,y,z]` | `render`, `view` | explicit camera; omit both for auto-framing |
| `--cameras <path> --view I` | `render` | render camera `I` of a `cameras.json` (e.g. the one `train` writes), at its own size |
| `--fov D` | `render`, `view` | vertical field of view in degrees |
| `--depth` | `render` | render the depth view instead of color |
| `--bg r,g,b` | `render`, `view` | background color |
| `--aa` | `render` | opacity compensation for the dilation below |
| `--eps2d X` | `render` | anti-alias dilation in px^2 (default `0.3`). **This is a blur** - see below |
| `--naive` | `render` | use the reference (non-tiled) rasterizer instead of the tiled pipeline |
| `--bench N` | `render` | print steady-state per-frame timing over N warm re-renders |
| `--isect-cap N` | `render` | tile-instance budget (default: 8 per gaussian). A frame that exceeds it drops its depth-latest splats and says `CLAMPED` |
| `--frames N` | `view` | exit the viewer after N frames (scripted/headless runs) |
| `--cameras <path>` | `fit` | camera poses/intrinsics (default `out/mirror/cameras.json`) |
| `--images <dir\|list>` | `fit` | target photos, one per camera, in order |
| `--iters N` | `fit` | optimization steps (default `200`) |
| `--eps2d X` | `fit` | the dilation to optimize UNDER (default `0.3`); render the result with the same value |
| `--lr X` | `fit` | learning rate (default `5e-3`) |
| `--densify N` | `fit` | run density control every N iterations (default off) |
| `--densify-frac F` | `fit` | fraction of gaussians treated as under-reconstructed per step (default `0.05`) |
| `--max-gaussians N` | `fit` | refuse to grow past N |
| `--densify-strategy S` | `fit` | `heuristic` (the default), `mcmc` or `hybrid` (what `train` runs) - see below |
| `--loss L` | `fit` | `mse` (default) or `l1-ssim`, the objective 3DGS is defined with |
| `--camera-model` | `fit` | fit the photometric camera alongside the scene: per-photo exposure and white balance, the lens's vignetting, the sensor's colour matrix and response curve (default off) |
| `--no-camera-model` | `train` | do not fit the photometric camera, which `train` fits by default: it starts each photo's exposure from its EXIF, and fits linear, 16-bit or exposure-bracketed photographs in linear light |
| `--batch N` | `fit` | photographs per optimizer step (default all of them) |
| `--distortion W` / `--normal-consistency W` | `fit` | surface regularizers (default off) - see below |
| `--geometry-after F` | `fit` | fraction of the fit after which the surface regularizers start |
| `--images <dir\|list>` | `sfm`, `train` | the photographs of one static scene |
| `--halvings N` | `sfm` | exact 2x halvings of the photographs for the written training set (default `0`) |
| `--max-width N` | `train` | widest training image: the photographs are halved exactly until they fit (default `2048`) |
| `--focal-guess F` | `sfm`, `train` | starting focal length as a multiple of the long side (default `0.8`); it is re-estimated |
| `--iters N` | `train` | optimizer steps (default: about 500 visits per photograph, 3000 to 30000) |
| `--max-gaussians N` | `train` | the scene's gaussian budget (default: the dense start plus a quarter); below the stereo's point count the start is thinned uniformly to four fifths of it, each kept disc widened to cover the same surface |
| `--sparse` | `train` | start from structure from motion's points instead of multi-view stereo |
| `--transients` | `train` | the photographs hold people, traffic or other things not in all of them: stop supervising each photograph's large coherent regions the scene cannot explain (off by default; on a static capture it only withholds supervision from hard regions) |
| `--environment D` | `train` | fit the sky and distant scenery as radiance by direction (a degree-`D` spherical-harmonic environment, `D` up to 8) instead of leaving it to gaussians; the written PLY carries it as a distant shell of gaussians |
| `--init-opacity O` | `sfm` | opacity of the starting gaussians (default `0.1`) |
| `--cameras-out <path>` | `sfm`, `train` | where to write the recovered cameras (default next to `--out`) |

## From photographs: `sfm` and `train`

`train` needs nothing but the photographs: no camera poses, no EXIF, no
learned model. It runs `recon::photogrammetry::reconstruct`, the same
pipeline as the SDK's `brain::Reconstruction`, in four steps:

1. **Structure from motion** (`brain splat sfm` on its own): features in every
   photograph, matches between the pairs worth matching (every pair up to 24
   photographs; beyond that each photograph's most similar ones by a global
   image descriptor, plus its neighbours in capture order) checked against
   the geometry of two views, then every camera placed at once (rotation
   averaging, then global positioning of cameras and points) and refined
   with bundle adjustment - or, when that leaves photographs out, the
   incremental reconstruction that registers one photograph at a time, if it
   does better. Photographs carrying usable GPS come back in metres with up
   up; otherwise the direction of gravity is read off how the cameras were
   held, and the scene is landed with it. The
   camera is calibrated from the photographs themselves - its focal length
   is chosen by reconstructing under a range of candidates and keeping the
   one that registers the most photographs most accurately, and the
   principal point and lens distortion are estimated alongside, under
   whichever lens model explains the photographs best (pinhole,
   Brown-Conrady radial and tangential, or Kannala-Brandt fisheye, chosen by
   information criterion). It prints how many photographs it
   registered and the reprojection error; a photograph it could not place
   is listed and left out.
2. Every photograph becomes a target exactly as it was recorded - its own
   pixels through its own lens, halved exactly (a 2x2 box) until it is at
   most `--max-width` across - and the set is turned upright.
3. **Multi-view stereo** (`crates/mvs`) measures every photograph's range,
   surface normal and their confidence per pixel through the same lens.
   Those become each target's geometry priors, and the fused cloud becomes
   the starting scene: thin gaussians lying in the measured surfaces.
   `--sparse` skips this and starts from the structure-from-motion points.
4. The fit, rendering every pixel along its own ray through the lens: the
   L1 + D-SSIM objective with the stereo's range and normal priors,
   credit-assigned density control that refines where the error is and
   spawns where the scene has nothing to refine, each gaussian's
   view-dependent colour limited to what the photographs that see it
   support, and surface regularizers in the second half.
   View-dependent colour is a degree-`d` spherical-harmonic expansion,
   `(d+1)²` coefficients per channel per gaussian, and with about as many
   coefficients as photographs seeing a gaussian it memorizes the training
   photographs instead of describing the surface: `train` uses flat colour
   below 32 photographs and full degree 3 from 128. The photometric camera
   (per-photo exposure and white balance, vignetting, colour matrix,
   response) is fitted too: on a 16-photo capture shot at one exposure it
   changes held-out quality by +0.08 dB, and on one whose exposure or white
   balance changes it is what keeps the scene consistent.

```bash
brain splat train --images ~/captures/scene --out scene.ply
brain splat view scene.ply
```

Render a training camera next to its photograph to judge the result:

```bash
brain splat render scene.ply --cameras scene.ply.cameras.json --view 0 --out view0.png
```

Photograph the subject from all around with generous overlap between
neighbouring shots (every part of the scene in at least three photographs),
keep the zoom fixed, and avoid moving objects. A textured floor or table
under the subject helps registration a great deal.

### Judging a fit on photographs it never saw

`crates/recon/examples/photo_holdout.rs` keeps every k-th photograph out of
the fit and scores the scene from those photographs' cameras with PSNR, SSIM
and LPIPS (v0.1, AlexNet trunk: the perceptual distance novel-view synthesis
results are usually reported in; lower is closer, 0 is identical). LPIPS
runs on the same device as the render. Its weights are two small upstream
releases that are not auto-fetched; this puts both in the models directory,
checksum-verified, where brain finds them by content:

```bash
python tools/goldens/lpips_dump_reference.py      # needs torch, safetensors and lpips (make requirements)
```

Without them the example says LPIPS was not scored and reports the other two.
In your own code, `recon::eval::Viewer::with_lpips(lpips::Lpips::from_store(&gpu)?)`
adds the same column to every score.

## Density control: when the fit may ADD gaussians

Without it, a fit has exactly one way to cover a region it cannot represent:
make the gaussians it already has bigger. That is visible in a real
reconstruction's size distribution, before and after a 500-iteration fit, as
projected screen size:

| | median | p90 | p99 |
|---|---|---|---|
| feed-forward | 0.61 px | 1.34 px | 3.34 px |
| after fitting | 0.37 px | 2.69 px | 8.72 px |

Most gaussians get smaller while a tail grows into blobs - detail appearing
where it can, and bloat where it cannot.

`--densify N` turns on the reference behaviour: every N iterations, the
gaussians whose positional gradient is still largest are subdivided (the big
ones SPLIT into two at 1/1.6 the scale along their dominant axis, the small
ones CLONE), and anything below `prune_opacity` is dropped. On a scene
deliberately started eight times too coarse for its target, 160 iterations
took it from 64 to 395 gaussians and cut final MSE by 32% against the same fit
with density control off.

**It is off by default, and that is deliberate.** Density control fixes a
scene too SPARSE to hold its detail. A feed-forward reconstruction is the
opposite problem - it starts at one gaussian per source pixel per view, so six
photographs give 1.2M and a 48-frame video gives ~9.7M before anything is
added. Growing that pushes the backward past the device's gradient-record
ceiling and fails the run. On dense scenes reach for `--prune` instead.

Staging the run costs something on its own: Adam's momentum restarts at each
density-control boundary, measured at ~3.5% worse final loss than one unbroken
run. Density control has to earn that back before it pays at all, which is why
it is a flag rather than a default.

## Depth supervision, and per-view masks

The RGB loss cannot see one axis. A splat's image-plane gradient is orthogonal
to its own viewing ray, so a scene placed at the wrong DISTANCE and scaled to
subtend the same angle renders the **identical image** and has an exactly zero
RGB gradient. Measured against an analytic ground truth, a reconstruction put a
thin object 5.7% too far away, entirely systematically.

`splat::opt::FitCfg::depth_weight` (default `0.0`, so nothing changes unless
asked) adds a term on the renderer's per-pixel **expected depth**
`D = (Σ z·α·T) / A`, differentiated end to end. `TargetView::with_depth` supplies
the prior and, per pixel, how far to trust it; `TargetView::with_mask` weights
the whole loss per pixel and divides back out of the normalizer, so a masked
run's reported MSE stays comparable with an unmasked one's.

**The depth term is an ANCHOR, not a source of truth.** The prior a caller has
comes from the same model whose depth is 5.7% wrong; supervising against a
single view's prediction re-imposes exactly that error. Feed it a multi-view
FUSED depth with the fusion's own per-pixel agreement as the confidence
(cross-view disagreement measured 1.1% per view against 0.41% fused). What it
buys is geometry that stops drifting: an RGB-only fit measured over 400
iterations grew the longest gaussian axis nearly ninefold and took the flatness
ratio from 4.14 to 26.34 - large flat blades that look right from the training
cameras and render as fur at a grazing angle.

Two properties are worth knowing before turning it on, both measured in
`crates/splat/tests/s13_depth_supervision.rs`:

* **Confidence is a relative weight between pixels, not a volume knob.** AdamW's
  step is normalized per parameter, and along the viewing ray there is no RGB
  gradient competing for the direction, so scaling every depth residual by the
  same constant leaves the step identical. What confidence does control is where
  a gaussian settles when its pixels disagree: given a +10% prior on half its
  pixels and a -10% prior on the other half, equal trust lands it on the truth
  (measured +0.02%) and 3:1 trust lands it at the weighted mean (+4.96% against a
  predicted +5.00%). A confidence of zero is off.
* **Depth is not supervised where the frame is transparent** (accumulated alpha
  below `splat::renderer::MIN_DEPTH_ALPHA`): an expected depth there is the ratio
  of two near-zeros. The term holds geometry in place; it does not create it.

On the headline case - a scene slid 5.7% along every viewing ray, rendering the
correct image - an RGB-only fit leaves 6.21% of depth error after 150 iterations
and the same fit with `depth_weight: 1.0` leaves 0.08%.

### `--densify-strategy hybrid`: spend the budget where the image error is

The heuristic asks which gaussians' centres received a large gradient. The
hybrid controller asks which gaussians are RESPONSIBLE for the error that is
left: one extra backward pass per photograph attributes every pixel's
remaining error to the gaussians that drew it, by the same weights they drew
it with. Gaussians are ranked by that, by how much of it sits on an edge, and
by the gradient signal; the top of the ranking is refined - an elongated one
split along its long axis, a coarse blob in every direction, a small one
cloned - within a growth schedule that reaches `--max-gaussians` two thirds
of the way through density control. A gaussian that contributes nothing is
first dimmed and removed only if it is still contributing nothing at the next
round. On a scene started far too coarse, at an equal budget, it ends at half
the heuristic's error.

### Surface regularizers

An image loss is as happy with two half-transparent layers as with one
opaque surface, and with a disc whose flat side faces anywhere. `--distortion`
penalizes compositing weight spread along each ray (layers collapse onto one
surface) and `--normal-consistency` turns each gaussian's flat side onto the
surface the render's own depth describes. Both are extra render passes, so
they cost time; start them part-way through (`--geometry-after 0.4`), once
the scene has a shape.

### `--densify-strategy mcmc`: relocate instead of split and prune

The heuristic above decides where detail may appear by thresholding a
gradient statistic. 3DGS-MCMC (Kheradmand et al., NeurIPS 2024,
arXiv:2404.09591) removes the decision: the gaussians are samples from a
distribution, and the moves are to RELOCATE a sample that has gone
transparent onto one that has not, and to perturb positions with noise
proportional to the learning rate. Nothing is thresholded and nothing is
deleted.

A relocation is free because it is image-preserving. Placing N gaussians
where one was would composite to `1-(1-o)^N` and stack N tails, so the
opacity and the scale are corrected (the paper's Eq. 9) to keep the alpha
integral along a ray equal to what the single gaussian gave. Measured against
the gaussian they replace, 2 to 5 corrected copies render at 43.7 to 67.2 dB,
against 26.9 to 38.0 dB for the same copies made naively.

`--max-gaussians` stops being a safety limit and becomes the budget: the
scene grows geometrically into it over the first fifth of the density-control
rounds and then stops changing size, so the samples it spends have the rest of
the fit to settle on somewhere to be.

What MCMC has that the heuristic does not is somewhere to put a gaussian the
fit cannot use. The heuristic can only delete it, which hands the budget back
to a split rule that climbs to it again two children at a time. Measured at
64x64 over three views, 150 iterations, a 400-gaussian budget, from 36 live
gaussians plus 220 sitting behind the cameras where a bad depth prediction
leaves them:

| | final MSE | gaussians |
|---|---|---|
| no density control | 0.010884 | 256 |
| heuristic, `--densify-frac 0.05` (default) | 0.010820 | 55 |
| heuristic, `--densify-frac 0.30` | 0.008462 | 158 |
| heuristic, `--densify-frac 1.00` and above | 0.006916 | 400 |
| MCMC | 0.006626 | 400 |

About 4% at an equal budget, and it renders 22.4 dB against 22.1.

That margin was 43% when this was first measured, against a fit with no bound
on how FLAT a gaussian could become and none on how far it could be INFLATED.
Both bounds landed since, and they take away part of what relocation was
fixing: a heuristic that can no longer answer a badly placed gaussian by
stretching it into a blade is a much stronger baseline. The old number is not
reproducible and is not worth quoting.

It is not a free win everywhere either. On a scene with nothing wasted in it,
where every gradient is informative, the gradient-targeted heuristic is still
ahead (0.006774 against 0.008868 on the same budget from a uniform coarse
start). Relocation pays where budget is being wasted, and it is worth knowing
which of the two a scene is before choosing.

One interaction to know about: `--max-growth` bounds a gaussian against the
size it had at the start of a density-control stage, and relocation
deliberately SHRINKS what it moves - that opacity and scale correction is what
keeps a relocation from changing the rendered image. Leaving the bound on caps
how fast a relocated gaussian takes up its new place, and measured here it was
enough to reverse the comparison outright.

## Sharpness, and the anti-alias dilation

Every splat's screen-space covariance gets a constant added to its diagonal
before rasterizing - `eps2d`, in pixels squared, `0.3` by default, the
reference 3DGS value. It is there because splats smaller than a pixel alias
badly as the camera moves, and it is a low-pass filter: it costs detail, and
on a reconstruction it costs a lot of it.

Measured on a scene built to be a perfect representation of a test image (one
gaussian per pixel, 0.25 px across), rendered from the camera it was built
for, as a fraction of the source image's high-frequency content:

| `--eps2d` | fraction of the source's detail | PSNR |
|---|---|---|
| `0.3` (default) | 0.30 | 21.5 dB |
| `0.1` | 0.80 | 34.7 dB |
| `0` | 0.98 | 44.9 dB |

And on a real six-photograph reconstruction, rendered from a camera the model
recovered:

| `--eps2d` | fraction of the photograph's detail | PSNR |
|---|---|---|
| `0.3` (default) | 0.42 | 21.5 dB |
| `0.15` | 0.54 | 20.9 dB |
| `0.05` | 0.65 | 20.0 dB |
| `0` | 0.73 | 19.2 dB |

Note which way PSNR moves. Blurring a slightly-wrong scene IMPROVES its
mean-squared error while destroying its detail, so accuracy alone will tell
you the default is the best setting and it is not. That is what
`splat::quality::sharpness_ratio` exists to measure, and what the
`s6`/`s7` test gates hold.

### Which value to use

**The dilation is part of the forward model `fit` inverts, not a display
setting.** The optimizer folds compensation for it into the gaussians, so a
fitted scene has to be rendered at the value it was fitted under. `fit` prints
that value when it finishes, and both commands take `--eps2d`.

Measured on the same six-photograph reconstruction after a full-resolution
fit at the 0.3 default:

| rendered at | fraction of the photograph's detail | PSNR |
|---|---|---|
| `0.3` (**what the fit used**) | **0.92** | **29.1 dB** |
| `0.2` | 1.22 | 28.5 dB |
| `0.1` | 1.67 | 26.3 dB |
| `0.05` | 1.90 | 24.4 dB |

Above 1.0 is not extra sharpness, it is aliasing: detail the photograph does
not contain, produced by removing a blur the gaussians were shaped to cancel.
Both measures agree that matching is correct.

So:

* **fitted scene** - render at the `--eps2d` it was fitted under. Changing it
  is a way to make the scene wrong, not a quality dial.
* **raw feed-forward scene, never fitted** - nothing has compensated for
  anything, so lowering it does recover real detail (0.42 to 0.73 in the table
  above). `--eps2d 0.05` is a reasonable still-frame setting there.
* **moving camera** - keep the default. The aliasing it prevents is worse than
  the sharpness it costs.

## When a render says CLAMPED

The tiled rasterizer sorts one instance per (gaussian, tile) pair out of a
buffer sized 8 instances per gaussian. Big screen-space gaussians touch dozens
of tiles each, so a scene whose gaussians have GROWN - which is every scene
`fit` has optimized - can exceed that budget. The overflowing tail is dropped
rather than the frame failing, and because the tail is depth-latest, what you
see is a missing occluder: a bright flare where something should have been in
front of something else.

`--isect-cap N` raises the budget; the render line says how many instances the
frame actually wanted, so `--isect-cap` that number and re-render.

## Serving (D-Bus)

Model id: `brain/splat`. Always served - the scene arrives as request bytes,
so there is no checkpoint to configure.

```bash
brain caps brain/splat
```

`render` (one-shot): takes `scene` (Inria-layout binary PLY) plus the same
camera/size/depth/background params as `brain splat render` above, returns
`image`. `fit` (streaming): takes `scene` plus `video` (N target views,
concatenated interleaved-HWC f32 RGB frames - the same convention every other
video input uses) and a `views` param (the camera array as JSON, the shape
`cameras.json` uses), returns the optimized `scene` and reports the
per-iteration MSE as progress; cancellable mid-run like any other long-running
served action. `view` is deliberately not served - it is an interactive
WASD/mouse loop with no request/response shape. See
[`samples/python/vision/splat/splat_render.py`](../../samples/python/vision/splat/splat_render.py) and
[`splat_fit.py`](../../samples/python/vision/splat/splat_fit.py).

## Hardware and limits

Runs on any wgpu-supported GPU or on the CPU - no CUDA or vendor-specific
GPU is required. `render` and `view` shade spherical harmonics up to degree
3, so a scene's view-dependent colour shows. `fit` optimizes an existing set of
gaussians against posed photos; to start from photographs alone, use
`sfm` (cameras and points) or `train` (the whole pipeline). Structure from
motion assumes every photograph came from one camera at one zoom setting.
