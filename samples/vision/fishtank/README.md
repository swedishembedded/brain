<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: vision/fishtank

Head-coupled perspective ("fish tank VR"): the screen becomes a window onto a
small scene that stays fixed in the room. Move your eye and the scene shows
the parallax a real window would - the back of the tank follows you, objects
in front of the glass move against you, and pillars at different depths
uncover different rings.

```bash
samples/vision/fishtank/fetch-data.sh                              # once: the face tracker (16 MB)
make samples/vision/fishtank/build
make samples/vision/fishtank/run                                   # your webcam tracks your head
make samples/vision/fishtank/run ARGS="--pose-source manual"       # the mouse is your eye
make samples/vision/fishtank/run ARGS="--pose-source synthetic"    # a scripted orbit
make samples/vision/fishtank/run ARGS="--headless --eye -15,3,50 --output fishtank.png"
```

## Getting the illusion right

The scene only stays put if the sample knows the physical geometry:

1. Measure the visible area of your screen and pass `--display-cm W,H`.
2. Run fullscreen (`--fullscreen` or `F`): the glass is the window, and a
   window that is not where the sample thinks it is shifts the world.
3. Sit centred in front of the screen and press `C`: the camera's sideways
   mounting error is absorbed.
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
* A real GPU raster pipeline on wgpu, presented to the window surface: an HDR
  scene pass (procedural starfield at infinity, an analytic anti-aliased neon
  grid, instanced crystal, pillars, rings and orbs with Fresnel, iridescence
  and emission, exponential fog), a five-level bloom pyramid, and an ACES
  tonemap with vignette and dither. All geometry is procedural; there are no
  asset files. Nothing is read back from the GPU per frame - only for
  `--headless` output and the `P` screenshot.

* Webcam head tracking, entirely on this machine: V4L2 frames carry their
  driver timestamp (`capture::Frame::stamp`); OpenSeeFace's face detector
  and landmark networks (`crates/openseeface`, on brain's own kernels) find
  66 landmarks; `headpose::fit_head` fits a rigid metric face to them
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
webcams do at 640x480) that this process may open, and the face tracker from
`fetch-data.sh` (OpenSeeFace, BSD-2-Clause). Without them the sample says
what is missing and exits; it never falls back to a fake eye. The face
networks run on the CPU by default (`--tracker-device gpu` moves them),
leaving the GPU to the renderer. `--headless` needs no window system.

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
| `--tracker-device cpu\|gpu\|gpu<i>` | `cpu` |
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
| `C` | (webcam) you sit centred: recentre, saved to the profile |
| `Z` | (webcam) you sit at the calibration distance: fix depth, saved |
| `Space` | freeze the eye |
| `G` | grid on/off |
| `D` | diagnostics on/off |
| `P` | save `fishtank-NNN.png` |
| `F` | fullscreen |
| `Esc` | quit |
