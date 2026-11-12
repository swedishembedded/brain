<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: vision/fishtank

Head-coupled perspective ("fish tank VR"): the screen becomes a window onto a
small scene that stays fixed in the room. Move your eye and the scene shows
the parallax a real window would - the back of the tank follows you, objects
in front of the glass move against you, and pillars at different depths
uncover different rings.

```bash
make samples/vision/fishtank/build
make samples/vision/fishtank/run                                   # the mouse is your eye
make samples/vision/fishtank/run ARGS="--pose-source synthetic"    # a scripted orbit
make samples/vision/fishtank/run ARGS="--headless --eye -15,3,50 --output fishtank.png"
```

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

The eye here is a fixed point, the mouse or a scripted orbit, and the HUD says
which. Webcam tracking (an open-weight landmark model feeding
`headpose::fit_head` and `headpose::ViewerEstimator`) is the next step; its
status is in the fishtank roadmap ledger.

## What it needs

A GPU wgpu can present to (Vulkan, Metal, DX12 or GL) and, for the window, a
desktop session. No model weights. `--headless` needs no window system.

## Options

| flag | default |
|---|---|
| `--pose-source manual\|synthetic` | `manual` |
| `--eye X,Y,Z` | - (centimetres from the screen centre, +Z toward you) |
| `--display-cm W,H` | `34.4,19.4` (the visible area of a 15.6" 16:9 panel) |
| `--profile PATH` | - (a `headpose::calibration::Profile` JSON; its display size wins) |
| `--fullscreen` | off |
| `--headless --output PATH` | - |
| `--size WxH` | `1280x720` |
| `--time S` | `4` (scene time of a headless frame) |
| `--hud` | off (draw the diagnostics into a headless PNG) |
| `--frames N` | - (stop after N frames) |

The illusion is only as right as the display size: measure the visible area
of your screen and pass it with `--display-cm`, and run fullscreen (`F`) or
leave the window where it opened.

## Keys

| key | |
|---|---|
| mouse / wheel | move the eye across / nearer or further (manual) |
| arrows, PageUp/PageDown | nudge the eye |
| `Space` | freeze the eye |
| `G` | grid on/off |
| `D` | diagnostics on/off |
| `P` | save `fishtank-NNN.png` |
| `F` | fullscreen |
| `Esc` | quit |
