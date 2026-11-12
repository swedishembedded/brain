<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# fishtank - head-coupled perspective

Scope: Webcam head tracking to a calibrated eye position and the off-axis view through the display (`crates/headpose`, `crates/openseeface`, `samples/vision/fishtank`).

The goal: a laptop webcam, a locally run open-weight face landmark model, and
a shader-rendered scene that stays fixed behind (and in front of) the glass
while the viewer moves - motion parallax, not stereo.

## Decisions

- **Landmark model: OpenSeeFace** (code and weights BSD-2-Clause), ported
  natively from its released ONNX graphs (`mnv3_detection_opt.onnx`,
  `lm_model*_opt.onnx`). SCRFD is not the tracker: five landmarks, and its
  antelopev2 weights are non-commercial. MediaPipe's TFLite bundle is a
  possible second architecture; brain has no TFLite importer.
- **Crate split.** `crates/openseeface` is the model (arch id per
  `porting.md`; the id is the directory). `crates/headpose` is model-agnostic
  geometry: landmarks in pixels in, eyes in metres and matrices out. The
  landmark tracker owns its ROI state, never calibration.
- **Rendering lives in the sample**, as a real wgpu raster pipeline
  presented to a surface: `gpu-core` is a compute seam and has no surface,
  raster or depth attachments. Readback only for headless screenshots.
  Extract a shared GPU surface into `crates/viewport` once a second sample
  needs one.
- **The canonical face is anthropometric, not transcribed**: rigid points
  only (eye corners, nose), scaled to the viewer's IPD.

## Done

- [x] `crates/headpose`: generalized off-axis projection from display
      corners (corners map to the viewport from every eye, rays cross the
      glass where they should, parallax sign), camera mount and strict
      calibration profile with recentre / measured-distance corrections, the
      rigid fit on `sfm::pnp::refine` through `camera::Intrinsics` (exact
      noise-free, no false translation under +-40 degrees of yaw), and the One
      Euro estimator with speed-gated, lag-compensated prediction,
      confirmation of discontinuities and crossfaded reacquisition.
- [x] `capture::Frame` carries a `Stamp`: the driver's V4L2 timestamp when
      it declares `CLOCK_MONOTONIC` and is plausible, else the dequeue time,
      marked as such; `capture::clock::monotonic_ns` reads the same clock.
- [x] `samples/vision/fishtank` with a fixed, mouse or scripted eye: a wgpu
      raster pipeline presented to a winit surface (sky at infinity, neon
      tank grid, instanced crystal / pillars / rings / orbs, fog, bloom
      pyramid, ACES), the glass taken from the window's place on the
      monitor, HUD, `P` screenshots, `--headless --output` PNGs.

## Not yet done

- [ ] Sample: the window's glass assumes a top-left client origin and a
      monitor whose pixel grid spans the `--display-cm` area; a monitor's
      own physical size (EDID) would remove the flag. The HUD texture is
      re-created whenever its text width changes.
- [ ] `crates/openseeface`: arch row, import with two-way coverage,
      detector + landmark nets on `crates/vision`, reference goldens
      (`tools/goldens/`), parity ladder (preprocess, layer taps, output,
      landmarks; CPU and GPU), detect-track-recover ROI loop, capability,
      residency, compliance row and NOTICE attribution for the output decode.
- [ ] Model store: OpenSeeFace ships on GitHub only and the store's `Hub`
      is Hugging Face; until a pinned GitHub source exists the weights come
      from a sha256-checked fetch script into a local directory.
- [ ] SDK `FaceLandmarkPipeline` (detect + streaming track), its e2e test.
- [ ] Sample webcam mode: capture thread, tracking thread, persisted profile,
      calibration keys; a recorded-webcam replay for regression.
- [ ] Measure on real hardware: inference and render rate, capture-to-photon
      latency, static jitter; compare One Euro against a constant-velocity
      Kalman filter on recorded data.
