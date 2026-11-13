# WorldMirror-2 (multi-view 3D reconstruction)

Feed it several photos of a scene or object from different angles and it
reconstructs a navigable 3D scene - a Gaussian Splatting point cloud, plus
per-frame depth, normal, and confidence maps and recovered camera poses.
Reach for it when you have a handful of photos and want a 3D reconstruction
without running a full structure-from-motion + optimization pipeline
yourself; for rendering or further optimizing the resulting scene, see the
[splat](splat.md) page.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| CLI (`brain <arch> <action>`)       | [x] `brain worldmirror2 reconstruct` / `import`, and the standalone [`sample-models-worldmirror2`](../../../samples/models/worldmirror2/README.md) |
| HTTP API               | [ ] `reconstruct` takes a REQUIRED `images` input blob, so it is not a text-to-image/chat action |
| D-Bus                  | [x] `reconstruct` (one-shot; `import` runs on files of the host and is not served) |
| Batched serving        | [ ] |

## Getting the weights

The checkpoint is found in the model store: a `.safetensors` holding the
released WorldMirror-2 tensors is recognised by its content, and
`--weights PATH` names one outright. To convert the released checkpoint (a
`.safetensors` file or a Hugging Face-style directory) once into brain's own
container, with the configuration embedded and every tensor shape-verified:

```bash
brain worldmirror2 import --source <model.safetensors|hf_dir> --dest out/mirror.safetensors
```

## Running it

```bash
# Reconstruct a scene from a folder of photos (or a video)
brain worldmirror2 reconstruct --images photos/ --maps \
    --out scene=scene.ply --out cameras=cameras.json --out depth=depth.mp4 --out normals=normals.mp4

# Fly through the result (WASD + mouse)
sample-models-splat view scene.ply   # make samples/models/splat/build
```

`sample-models-worldmirror2` is the same actions as a standalone program
(`make samples/models/worldmirror2/build`), with `export-npu` added.

## Turntable captures need a mask

This model assumes ONE RIGID SCENE and a moving camera. A turntable gives it
the opposite: a rotating object in front of a stationary background. Those are
not the same thing, and the model can only fit one of them - the static part
of the frame is evidence for a camera that never moved, working directly
against the rotation the object shows.

Mask the background away and the contradiction goes with it. What remains, an
object turning in front of nothing, is exactly equivalent to a camera orbiting
a still object, which is the case the model was trained on. Because the camera
does not move, ONE mask covers the whole capture.

The mask is any image: white where the object sweeps, black elsewhere. It is
sampled rather than interpolated when resized, so a mask at a different
resolution to the footage is fine.

A capture's own motion will draw the mask for you - the pixels that change
over the clip are the ones that rotate:

```bash
ffmpeg -i turntable.mp4 -vf "select='not(mod(n\,10))'" -vsync 0 f%03d.png
# then: mask = white where the per-pixel standard deviation across f*.png is high
brain worldmirror2 reconstruct --images turntable.mp4 \
    --mask mask.png --max-frames 24 --out scene=scene.ply
```

Shooting the camera around a stationary object avoids all of this and is
always the better capture when it is possible.

## What it produces

Four hand-held photographs of one object, with no poses, no calibration and no
structure-from-motion step, in; a Gaussian-splat scene and the camera each
photograph was taken from, out:

```console
$ brain worldmirror2 reconstruct --images views/ --maps --out scene=scene/scene.ply --out cameras=scene/cameras.json
reconstruct [3/3] 812224 gaussians
gaussians: 812224
frames: 4
width: 518
height: 392

$ brain splat render --scene scene/scene.ply --width 640 --height 480 --eye 0.53,-0.28,0.12 --target 0.02,-0.03,1.02 --out image=view.ppm
scene/scene.ply: 812224 gaussians -> view.ppm (640x480, tiled, 1496463 isects, 195 ms) <!-- perf-number: measured result; reproduced by the named model binary/test (`brain perf run` for timings) -->
```

One feed-forward pass: no per-scene optimisation, no iterative refinement. The
cost of that is visible and worth stating - the reconstruction is sharp where
the four cameras saw the object and smears where they did not, and the desk
plane degrades quickly as the viewpoint leaves the volume those four span. A
feed-forward model interpolates between the views it was given; it does not
invent the ones it was not. For a scene you want to hold up to arbitrary
viewpoints, optimise it afterwards with [`splat fit`](splat.md).

## Inputs

`--images` takes a directory of frames or a **video file**
(`.mp4/.mov/.mkv/.webm/.avi/.m4v`; AV1 is read in process, any other codec
through `ffmpeg` when it is on `PATH`). A directory is read in sorted order,
so a capture's own frame numbering is the order the model sees. PPM, PNG, JPEG,
BMP and TIFF all decode; any aspect ratio is fine, since frames are resized and
cropped to the model's patch grid automatically - as long as every frame shares
one size. A set of photographs of mixed sizes is first brought to one size.

This model was built for sequences, not for a handful of stills: the paper
evaluates it at 2-8 views (sparse) and 32-64 (dense), and reports the margin
over prior work GROWING with view count. A slow orbit video is the input it
wants. `reconstruct` returns the scene (`scene`, a PLY) and the recovered
camera of every frame (`cameras`, JSON in the shape `splat fit` takes); `--maps`
additionally returns the per-frame depth and normal maps as the `depth` and
`normals` clips, and `--frames-out` the frames exactly as the model saw them
(`frames`).

## Options

| Flag | Effect |
|---|---|
| `--images PATH` | the capture: a directory of frames or a video (required) |
| `--mask PATH` | black out everything the mask does not cover (white keeps), in every frame - for turntable captures, see above |
| `--poses PATH` | known cameras, one per selected frame, in the shape `cameras` has: condition the trunk and become the back-projection cameras |
| `--stride N` | use every Nth input frame |
| `--max-frames N` | use at most N frames, spread ACROSS the capture rather than taking a prefix (default 0: 48 for a video, a set of photographs is used whole) |
| `--fps X` | resample a video to X frames per second before selecting (default 0: the clip's own rate) |
| `--target-size N` | longest-edge cap the frames are resized to, floored to the 14px patch grid (default 952, the reference's inference default) |
| `--min-opacity X` | drop gaussians below this opacity (default `0.01`) |
| `--max-depth X` | drop gaussians beyond this depth, in scene units (default 0: no limit) |
| `--prune VOXEL` | voxel-merge duplicate gaussians across overlapping views (default `0.002`; 0 = off) |
| `--gs-mask-threshold X` | keep a pixel only where the Gaussian head's validity mask is above this (default `0.5`) |
| `--edge-depth-threshold X` | drop pixels on a depth discontinuity, a relative step above this between neighbours (default `0.03`) |
| `--fuse-depth X` | reconcile the per-view depth maps before they become geometry; pairs differing by more than this fraction are occlusions (default `0.05`) |
| `--min-support N` | keep a pixel only where at least N other frames agree a surface is there (default 0) |
| `--conf-percentile P` | drop this percentage of pixels, least confident first (default 0) |
| `--surface-align X` | lay each gaussian flat against its surface at this thickness ratio (default `4`; 0 = the model's own orientation) |
| `--position-from gsdepth\|points\|blend` | which head places the gaussians (default `gsdepth`) |
| `--normals-from geometry\|head` | where the normal behind `--surface-align` comes from (default `geometry`) |
| `--max-scale-quantile Q` | drop the gaussians whose scale is above this quantile (default `0.98`; 0 or 1 = off) |
| `--keep-camera-frame` | return the scene in frame 0's frame instead of re-framing it upright |
| `--maps` | also return the per-frame depth and normal maps |
| `--frames-out` | also return the frames exactly as the model saw them |

The trunk's global attention is quadratic in frame count, so a long capture
needs thinning before it reaches the model - hence the video default. The
spread-not-truncate rule matters: capping a 360-degree orbit to its first N
frames leaves a 60-degree arc, which reconstructs far worse than the same
budget spread over the whole path.

`sample-models-worldmirror2 export-npu` exports individual model stages
(`--stage dino|trunk|heads`) as ONNX for running on the Intel NPU or CPU via
OpenVINO - an advanced path for NPU deployment rather than everyday use.

## Serving (D-Bus)

Model id: `brain/worldmirror2`. The checkpoint is a machine-side setting
(the model store, or `BRAIN_WORLDMIRROR2_WEIGHTS` for the scheduled service),
not a per-request param: a remote caller cannot answer "where is the
checkpoint on THIS machine".

```bash
brain caps brain/worldmirror2
```

`reconstruct` (one-shot): takes `images` (N unposed frames, concatenated
interleaved-HWC f32 RGB - the same video convention every other video input in
this repo uses; every frame shares one `(w,h)`, and any size is resized to the
model's 14px patch grid), optional `mask` and `poses`, and every parameter of
the table above except the paths. Returns `scene` (the reconstructed Gaussian
scene, Inria-layout binary PLY), `cameras` (the per-frame cameras
WorldMirror-2 predicted, the same JSON shape `cameras.json` uses) and the
counts `gaussians`, `frames`, `width`, `height`; `depth`, `normals` and
`frames` clips when requested. See
[`samples/python/vision/worldmirror2-reconstruct/worldmirror2_reconstruct.py`](../../../samples/python/vision/worldmirror2-reconstruct/worldmirror2_reconstruct.py).

One resident instance serves every request shape: the model is
shape-adaptive internally (it rebuilds its own per-shape buffers on demand,
keeping the ~5GB checkpoint resident across a shape change), so the
scheduler keys the instance on the checkpoint alone, never on
frames/width/height. The one honest cost: only one shape's buffers are
cached at a time, so a workload that keeps ALTERNATING between two request
shapes on this one instance pays a full rebuild on every call - a latency
cost, not a correctness bug.

## Hardware and limits

This is reconstruction only - there's no training or fine-tuning action for
this model. Known camera poses can guide a run (`--poses`); depth priors are
not exposed. There's no built-in sky-segmentation filter - use `--min-opacity` /
`--max-depth` to prune sky and background artifacts from the reconstructed
scene. The released WorldMirror-2 weights are under Tencent's community
license, which excludes the EU - treat an imported checkpoint as
reference/research material rather than something to ship. brain's own
implementation is unencumbered; only the upstream weights carry that
restriction.
