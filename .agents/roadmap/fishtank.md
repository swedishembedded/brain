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

- [x] MobileNetV3's pieces in the shared vision layer: `hardswish`,
      `hardsigmoid`, `relu6` kernel pairs (`Act::HardSwish`,
      `Act::HardSigmoid`, `Act::Relu6`; FD-checked in `crates/gradcheck`)
      and `vision::SqueezeExcite` with a selectable gate.
- [x] Sample webcam mode: capture, tracking and render threads, the
      tracker on the CPU backend by default, `C` recentre and `Z`
      measured-distance calibration saved to a profile that later runs
      load, `--camera-image` replaying a photo through the whole chain
      (verified here: fit residual 2.4 px on the portrait, render 57 fps on
      the Arc iGPU while tracking).

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
- [ ] Run against a live webcam: this development container may not open
      `/dev/video0` (EPERM), so live tracking, jitter and latency are
      unmeasured on a real camera.
- [ ] GPU inference cost: every `vision` block submits eagerly, so a face
      network is a few hundred small submissions; on the iGPU beside the
      renderer that measured 80-116 ms per forward (2026-11-12, loaded box)
      against 23-42 ms on the CPU backend. Record a forward as one
      submission (or a captured program) before making `gpu` the default.
- [ ] Intel NPU: `crates/npu` exports whole brain graphs to OpenVINO; the
      openseeface networks have no exporter there yet, and this box's
      container exposes no `/dev/accel` device to test one.
- [ ] Recorded-webcam replay (a frame sequence with timestamps) for
      regression; `--camera-image` replays a still photo only.
- [ ] `crates/fastvlm` and `crates/ecapatdnn` each keep a private
      squeeze-excite; move them onto `vision::SqueezeExcite` (needs their
      parity goldens to re-prove). The shared block has no backward yet, and
      the new activations take the unfused path (no `fused_code`).
- [ ] Measure on real hardware: inference and render rate, capture-to-photon
      latency, static jitter; compare One Euro against a constant-velocity
      Kalman filter on recorded data.
