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
- **Rendering lives in the sample**, as a full-screen ray-traced pass on
  wgpu presented to a surface: `gpu-core` is a compute seam and has no
  surface or raster pipeline. Readback only for headless screenshots. Ray
  tracing rather than raster because the scene is analytic and the off-axis
  ray from the eye through each pixel is the projection itself.
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
- [x] `samples/vision/fishtank` with a fixed, mouse or scripted eye: one
      ray-traced wgpu pass (analytic rusty pipe, ball, buoy and crate in a
      riveted steel recess; a lake, ridges and sky beyond; a room behind
      the viewer for reflections; GGX, exact sun shadows, form-factor sky
      light, occlusion, one mirror bounce, ACES), its material texture
      generated per mip level on the GPU, the scene's resolution following
      GPU timestamps to fit the refresh, the glass taken from the window's
      place on the monitor, HUD, `P` screenshots, `--headless --output`
      PNGs. Measured 2026-11-12 on the Arc iGPU (Meteor Lake, power limit
      raised to 45 W): GPU frame 9.0 ms at 1920x1200, 13.5 ms at 2880x1800
      headless; in a fullscreen window on the same machine under other
      load (GPU at 100-350 MHz) it held 60 fps by rendering the scene at
      43% resolution.

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
- [ ] Run against a live webcam: the maintainers' devcontainer profiles
      allow device majors 189/195/226/234/237/261 but not V4L2 (81), so
      `/dev/video0` is EPERM inside them; live tracking, jitter and latency
      are unmeasured on a real camera.
- [x] Intel NPU and GPU: `openseeface::export` (static-shape ONNX from
      brain's weights) compiled by OpenVINO (`crates/openvino-rt`, split
      out of `crates/npu` so a model can run on the NPU without linking
      every exporter); parity on CPU, iGPU and Meteor Lake NPU; the sample
      tracks at camera rate on the NPU (11 ms/frame) and iGPU (21 ms).
- [ ] brain's own kernels on the iGPU. Measured 2026-11-12 with the GPU's
      actual clock (`gt0/rps_act_freq_mhz`) and throttle reasons sampled
      beside every run: this box sat at its PL1 package power limit for
      100% of every run (other workloads drawing CPU power), holding the
      iGPU at 100-500 MHz of its 2250 MHz, so every absolute number here is
      several times too slow. At equal clock, brain's forward spends 2.5x
      (detector), 3.5x (lm0) and 7.7x (lm3) the GPU cycles of OpenVINO's,
      60-77% of it in `conv_bias_reg` (lm3's 960x160 pointwise conv at 7x7
      fills about 25 workgroups). im2col + `matmul_reg3_64` for biased 1x1
      convs computes the identical output in 25-35% fewer cycles - real,
      but not the gap, and landing it as one more hard-wired rule in
      `vision` would bypass per-device selection (the CPU backend has a
      native direct-conv path). Next: register 1x1/depthwise/GAP small-shape
      candidates through the `gpu_core::provider` seam and select by
      measurement (kernels.md section F), then fuse bias + activation into
      the GEMM epilogue. Measure only with the clock and PL1 logged, on an
      otherwise idle machine. `intel_gpu_top` needs CAP_PERFMON, absent in
      the devcontainer.
- [ ] Dependency isolation: the openseeface test binary links rav1e/rav1d,
      opus, image, imageproc, nalgebra, rustls and regex, almost all through
      `crates/imaging`, for one host resize; and `crates/perf`'s sysfs
      telemetry lives in a layer-6 crate. Move pure host pixel ops and the
      device telemetry into small leaf crates that the heavy ones re-export,
      so a consumer links what it calls.
- [ ] Recorded-webcam replay (a frame sequence with timestamps) for
      regression; `--camera-image` replays a still photo only.
- [ ] `crates/fastvlm` and `crates/ecapatdnn` each keep a private
      squeeze-excite; move them onto `vision::SqueezeExcite` (needs their
      parity goldens to re-prove). The shared block has no backward yet, and
      the new activations take the unfused path (no `fused_code`).
- [ ] Measure on real hardware: inference and render rate, capture-to-photon
      latency, static jitter; compare One Euro against a constant-velocity
      Kalman filter on recorded data.
