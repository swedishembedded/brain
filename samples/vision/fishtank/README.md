<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: vision/fishtank

Head-coupled perspective ("fish tank VR"): the screen becomes a window onto a
scene that stays fixed in the room. Move your eye and the scene shows the
parallax a real window would: a rusty pipe reaching out of the glass toward
you and a steel ball floating in front of it move against you, the riveted
steel recess behind the glass turns, and the lake and mountains kilometres
away follow you.

```bash
make samples/vision/fishtank/build     # the binary, the face tracker and the OpenVINO runtime
make samples/vision/fishtank/run       # your webcam tracks your head
make samples/vision/fishtank/run ARGS="--pose-source manual"       # the mouse is your eye
make samples/vision/fishtank/run ARGS="--pose-source synthetic"    # a scripted orbit
make samples/vision/fishtank/run ARGS="--headless --eye -15,3,50 --output fishtank.png"
```

## Getting the illusion right

The scene only stays put if the sample knows the physical geometry:

1. Measure the visible area of your screen and pass `--display-cm W,H`.
2. Run fullscreen (`--fullscreen` or `F`): the glass is the window, and a
   window that is not where the sample thinks it is shifts the world.
3. Sit centred in front of the screen, facing it, and press `C`: the
   camera's aim (the lid's tilt, a sideways turn) is solved so the view is
   centred both ways at every distance.
4. Hold a ruler: sit at 50 cm (or `--calibration-distance-cm`) and press
   `Z`: depth is rescaled to your face.

`C` and `Z` save the result to `~/.config/brain/fishtank-profile.json` (or
the `--profile` file), which later runs load by themselves; the display and
camera flags apply only while there is no saved profile. If your webcam's
field of view is known, `--camera-hfov` makes depth right from the start.

## What it demonstrates

* `headpose::off_axis` - the generalized perspective projection through the
  physical glass from an eye position, recomputed every frame. The scene is
  never rotated to fake parallax; only the eye moves.
* The glass is the part of the display the window covers, found from the
  monitor's pixel size and the window's position on it, so moving the window
  looks through a different part of the same scene.
* The scene is ray traced in one full-screen pass on wgpu
  (`shaders/scene.wgsl`): every ray starts at the eye and passes through its
  pixel on the physical glass. All intersections are analytic (spheres, a
  hollow pipe, a box, planes), lighting is physically based (GGX, Fresnel, a
  low sun with exact shadows, sky and room light through the two openings by
  their form factors, ambient occlusion), and smooth steel and water trace
  one mirror bounce, so the pipe and the ball reflect the room behind you.
  Rust is steel under a rust mask with bump-mapped relief, from one
  material texture generated on the GPU at start-up (`shaders/bake.wgsl`,
  band-limited per mip level; nothing on disk).
* Frame time is measured on the GPU with timestamp queries, and the scene's
  resolution follows it so each frame fits the display's refresh: a
  throttled GPU loses pixels, never shading. `--headless --frames N` reports
  the median GPU time. Nothing is read back per frame on screen.

* Webcam head tracking, entirely on this machine: V4L2 frames carry their
  driver timestamp (`capture::Frame::stamp`); OpenSeeFace's face detector
  and landmark networks (`crates/openseeface`) find 66 landmarks - on the
  Intel NPU or GPU, compiled by OpenVINO from brain's own static-shape
  export of the network (`openseeface::export`), or on brain's own kernels; `headpose::fit_head` fits a rigid metric face to them
  through the camera model; the camera mount puts the eyes in the screen
  frame; `headpose::ViewerEstimator` smooths them and predicts them to the
  moment the frame is seen. Capture, tracking and rendering run on their own
  threads, and none waits for another.

The eye can also be a fixed point, the mouse or a scripted orbit, and the HUD
always says which. `--camera-image PHOTO` replays a still photo, panned
sideways, through the whole tracking chain - labelled as a replay - for a
machine without a camera.

## What it needs

A GPU wgpu can present to (Vulkan, Metal, DX12 or GL) and, for the window, a
desktop session. For webcam mode: a V4L2 camera offering YUYV (most laptop
webcams do at 640x480) that this process may open, and what `build` fetches
through `fetch-data.sh`: the face tracker (OpenSeeFace, BSD-2-Clause, 16 MB)
and the OpenVINO runtime (Apache-2.0, the libraries of the pinned official
wheel, about 160 MB, in `~/.local/share/brain/runtimes`). The NPU also needs
the host's NPU driver (`intel-level-zero-npu`). Without what it needs the
sample says what is missing and exits; it never falls back to a fake eye.

Where the face networks run is `--tracker-device`: by default the NPU, else
the GPU, both through OpenVINO, else brain's CPU backend; the HUD's `FACE`
line names the device actually used. On a Core Ultra 7 155H (Meteor Lake),
replaying a photo at 30 Hz through the whole chain on a heavily loaded
machine (2026-11-12), one frame of tracking took 11 ms on the NPU and 21 ms
on the GPU next to the renderer: the camera's rate either way. Brain's own
kernels run these small networks well below the GPU's capability today, so
`brain-gpu` is there for comparison, not speed. `--headless` needs no
window system.

## Options

| flag | default |
|---|---|
| `--pose-source webcam\|manual\|synthetic` | `webcam` |
| `--camera DEV` / `--camera-size WxH` | `/dev/video0` / `640x480` |
| `--camera-image PATH` | - (replay a photo instead of a camera) |
| `--camera-hfov DEG` | `60` |
| `--camera-gap-cm CM` | `0.8` (lens above the top edge of the visible screen) |
| `--ipd-mm MM` | `63` |
| `--landmarks fast\|best` | `fast` |
| `--tracker-device auto\|npu\|gpu\|cpu\|brain-cpu\|brain-gpu` | `auto` (NPU, else GPU, else brain CPU) |
| `--openvino-libs DIR` | where `fetch-data.sh` puts them |
| `--no-vsync` | off (present without waiting for the display refresh) |
| `--calibration-distance-cm CM` | `50` (what `Z` calibrates to) |
| `--eye X,Y,Z` | - (centimetres from the screen centre, +Z toward you) |
| `--display-cm W,H` | `34.4,19.4` (the visible area of a 15.6" 16:9 panel) |
| `--profile PATH` | `~/.config/brain/fishtank-profile.json` when it exists (a `headpose::calibration::Profile`; wins over the display and camera flags) |
| `--fullscreen` | off |
| `--headless --output PATH` | - |
| `--size WxH` | `1280x720` |
| `--time S` | `4` (scene time of a headless frame) |
| `--hud` | off (draw the diagnostics into a headless PNG) |
| `--frames N` | - (stop after N frames; with `--output`, a window run saves its last frame) |

## Keys

| key | |
|---|---|
| mouse / wheel | move the eye across / nearer or further (manual) |
| arrows, PageUp/PageDown | nudge the eye |
| `C` | (webcam) you sit centred, facing the screen: the view is centred horizontally and vertically by re-aiming the camera's pitch and yaw, saved to the profile |
| `Z` | (webcam) you sit at the calibration distance: fix depth, saved |
| `Space` | freeze the eye |
| `G` | grid on/off |
| `D` | diagnostics on/off |
| `P` | save `fishtank-NNN.png` |
| `F` | fullscreen |
| `Esc` | quit |
