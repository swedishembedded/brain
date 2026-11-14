<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# book-debt

Scope: Defects, gaps and stale comments in brain's crates that writing the textbook surfaced, fixed in the crates so the chapters describe correct code.

Rule: the book does not record unfixed brain shortcomings. A chapter that finds one stops until the crate is fixed (test
first), then describes the corrected behaviour. Each item names the chapter that recorded it; when it is fixed, that
chapter's passage is rewritten in the same series of commits.

## Open: fixable in this tree

Numerics and data:
- [x] data rng: `gen_range_inclusive` modulo bias (use rejection), Box-Muller clamp at 1e-12 (redraw a zero `u1`), offset
      comment (v01 ch17).
- [x] eval: real tests for `gpt_val_perplexity`, `gpt_exact_match`, `greedy_until_newline`, `argmax`; drop the
      tautological one (v02 ch06).

Training kernels and graphs:
- [x] optim: `Optim` resolves the cooperative grad-norm pair itself (the `Gpu` appends it beside `gradnorm_sq`, `gpu_core::upgrade::TWINS`),
      so deepseekocr2, minicpmv, qwen35moe, florence2, solve, timesfm3, ricursive and every other model get it with no edit; the serial
      kernel stays for a device with no workgroup reduction (the CPU JIT) and as the bench reference (v03 ch08).
- [x] rmsnorm/layernorm: shared builder resolves the `_rows` twins by default (`Gpu::twin_of`); the "per-row loop is fine"
      comments are corrected (v03 ch10, v04 ch04).
- [x] model: `LayerTape`/`TapeEdge` are `SegmentedPass`/`SegmentEdge` (module `model::segmented`), `RevisitingTape` is `RevisitingPass` (v03 ch04).
- [x] toyautoencoder: delete the unused first-layer dX and its buffer (v03 ch04).
- [x] wgsl-cpu: bounds-checked dispatch that returns an error naming kernel and binding (v03 ch10); on in debug builds of the CPU backend
      and via `backend_api::set_checked_kernels`; the native fast paths of `backend-cpu` are not covered.
- [x] wgsl-cpu: typed compile error instead of matching the word "barrier" (v04 ch09).
- [x] backend-cuda / gpt2: derive golden tolerances from the round-off bound (v04 ch09); the CUDA comparisons skip without a card, the
      derived gpt2 bound is checked against CPU vs the default device.
- [x] gpt2: lm_head is tied to the token embedding (`check_gpt_tied`); a checkpoint with a separate `lm_head.weight` is refused on load (v05 ch03, ch10, ch15).
- [x] gpt2: the cached-decode bound is derived from the round-off bound and the greedy test asserts the top-2 gap (`tests/it/kv_decode.rs`, v05 ch15).
- [x] rope.wgsl: theta as a parameter (v05 ch08).
- [x] modernbert: banded local attention over a [rows, 2w+1] slab (v05 ch14); a span no longer than the band keeps the dense windowed form.
- [x] checkpoint: `read_model_dir` errors on an index key no shard returned (v06 ch02).
- [x] gradcheck: `check_moe` and `check_moe_balance` differentiate a host loss that includes the aux and z terms (v06 ch04).
- [x] modelcost: gemma4 and minimaxh3 registered, Recording wired in qwen35/qwen35moe, real-config pins, held/peak device bytes in `measure`; every kernel the registered pricers dispatch has a `gpu_core::cost` formula (coverage 1.0 gated on all seven); open: gemma4/minimaxh3/qwen35* have no `measure` (v06 ch10).
- [x] serve sampler: exact top-p/min-p and exact penalties over the full logits row on every serving engine and every sampling path (decode, first token, rollout, speculative verification, mixed iteration); `check/serve-conformance` requires the row readback (v05 ch11); the one top-k tie rule is done (qwen3vl keeps exactly k, lowest id first).

Vision and generation:
- [x] imaging letterbox: pixels and boxes share one integral pad (floor of half the slack, the upstream detectors' pad);
      `resize_nearest` mode 1 (half-pixel, integer form) and `pad2d` constant/edge fill; `Ctx::letterbox` on the device,
      `letterbox_rgb` its bitwise-equal host twin for host-memory callers, yolov5's board letterbox on the same tap;
      yolov5 golden regenerated (v07 ch01, ch04, ch12).
- [x] yolov8 `detect_batch` and yolov5 `detect` letterbox on the device (`Ctx::letterbox_hwc_into`), bitwise equal to the
      host path.
- [ ] yolov8 tests (p5_overfit, p6_infer, p7, p11) and the GPT convergence suite gate on `MOE_SKIP_GPU_TESTS` directly,
      which .agents/rules/testing.md routes through `brain_testutil` (found while moving the letterbox).
- [x] imaging: the owed call sites moved: s3dit latent mask (`mask::downsample`, `feather` with `FeatherEdge::Replicate`),
      outpaint canvas (two `Ctx::pad` on the VAE device), range maps via `pixels::{hwc_unit_to_chw_signed,
      chw_signed_to_hwc_unit}` in s3dit, flux1, flux2, sdxlunet, supir, controlnet (v07 ch01).
- [x] clip: DeepSeek-OCR vision tower runs quick-GELU on both import paths, as upstream and the tiny ladder; the tiny
      ladder runs again (trainable taps, frozen build gated at its outputs) (v07 ch08).
- [x] zipdepth: MiDaS SSI MSE + 4-scale gradient matching (closed-form fit, exact gradient, f64-oracle check), a paired
      RGB/16-bit depth dataset, `--data`/`--grad-weight` (v07 ch05).
- [x] instantid: per-site k and v computed once in `set_id` (v08 ch11).
- [x] rrdbnet: exact tiling, windows of tile + 2R inside the image with R derived from the graph (349 at x4plus);
      bitwise equal to the whole-image pass (v08 ch12).
- [x] vqgan: LPIPS (VGG16 and AlexNet trunks, input gradient), PatchGAN with hinge loss, adaptive weight, disc_start,
      device entropy term; llamagen_vq16 and codeformer presets read off the upstream files (v08 ch03, ch04).
- [x] mimi: HiFi-GAN multi-period and multi-scale discriminators, LSGAN and feature matching in the trainer, taking
      the saved-activation policy (v08 ch04, v09 ch03).

Audio, video, simulation:
- [x] mimi: the encoder applies its sliding window (unit spec: frames outside the window cannot reach the output) (v09 ch03).
- [x] mimi: multi-scale log-mel L1 with its backward (`audio::mel::log_mel_backward`, gradchecked), weight 45 in the trainer (v09 ch03).
- [x] qwen3omnimoe: generate samples, stops, scans tool calls and thinking through the shared machinery; audio+video in
      one prompt refused (positions frame-ordinal; the unread config fields removed); DeepStack injected; speak and
      converse cancellable (v09 ch08).
- [x] minimaxmusic3: joint vocoder GAN loop (multi-resolution STFT discriminators, LSGAN, feature matching) (v09 ch07).
- [x] videocodec / imaging: `decode_from` starts at the key frame before the first asked frame (container key flags,
      sequence header prepended when missing); `decode_frames_at` uses it (v10 ch01).
- [x] ltxv / wan: training batches use inference's positions, keyframe mask and per-token timesteps (conditioned
      tokens clean, timestep 0, outside the loss, as LTX-Video-Trainer); wan has one RoPE grid builder (v10 ch02). Wan image-to-video: user decision
      2026-11-13, `Task::I2v` stays and the book does not describe it.
- [x] genieredux: the reference MaskGIT sampler (Dynamics.sample) ported and pinned on a tiny model; real-checkpoint
      parity of more than one pass needs the checkpoint (v10 ch07).
- [x] genieredux: frame cache (bitwise equal to recompute), `WorldModel`, CLI entry. The chapter describes DIAMOND as
      the port it is (no reward head) (v10 ch07).
- [x] genieredux cache is device-resident (no per-frame re-upload, `Gpu::uploaded_bytes` proves it).
- [ ] flybody: muscle polarity pinned per muscle by a MuJoCo sweep (v10 ch08).

Forecasting and decisions:
- [x] forecast: weighted quantile loss normalised by the number of levels (v11 ch02).
- [x] forecast: fair CRPS ensemble estimator, n(n-1) (v11 ch02).
- [x] forecast convert: derived sample paths share one uniform per path, or say they are independent marginals (v11 ch02).
- [x] fcbench: paired uncertainty per cell; control gated on an interval (v11 ch04).
- [x] fcbench: seasonal MASE scale from the panel's period (v11 ch04).
- [x] fcbench: scorer reads `quantile_levels` (v11 ch04).
- [x] bench forecast_bench: candidate chosen from context only, or the header says the score is biased up (v11 ch04).
- [x] chronos2 / fincast: fine-tuning entry points on the shared contract; roadmaps corrected (v11 ch03).
- [x] rlcd: one documented probability floor (v11 ch06).

## Open: found while fixing the items above

- [x] model `it` on the Intel Arc device: `interleaved_rope_theta` (tolerance derived from the WGSL `sin`/`cos`/`pow`
      accuracy table, angles kept inside `[-pi, pi]`), `matmul_kq::dyn_matches_host_oracle_both_code_bits` (per-element
      `gamma_{ng+4} * S` bound from `testutil::rounding`) and `moe_compact_parity::compact_layer_submit_count_*`
      (`stats().submits` counted the Intel sliced-binding workaround's one-queue-submit-per-step; now one per batch,
      `queue_submits` is the physical count). Knowledge #451.
- [x] Flaky: moondream3 `geglu_shift_grads_match_finite_differences` (tolerance scaled by a cancelling derivative; now
      derived from the absolute sum of the terms), backend-wgpu `kernel_timing::dropping_a_shared_handle_does_not_wait_for_the_device`
      (`submit` drained the device through the unsynchronised-workgroup valve before the test looked; now one naive matmul).
      The other three checks in `crates/moondream3/src/kernels_check.rs` (rope_partial, adaptive_avgpool2d,
      tau_scale) had the same relative tolerance; they perturb ops linear in the perturbed argument and now assert
      the derived rounding bound of `assert_linear_directional`.
- [x] Binding-offset alignment: every listed `step_sliced` site proved aligned in a comment (`head_ce.rs`/`lossw.rs` chunk
      starts, `moe.rs` `route_rows` regions, `ops.rs` quant rows/embed tiles/two-level passes); the six `chunked_bidir_*`
      entry points refuse a window off the grain at the call (`block::assert_row_windows_bindable`).
      Open: CUDA's `step_native_sliced` and `step_library_gemm` (`crates/backend-cuda/src/backend.rs`) do not call
      `check_binding_offsets`, so a misaligned native slice is accepted there and refused everywhere else; adding it
      needs a CUDA device to confirm no provider passes an unaligned offset.
- [x] gemma4 and minimaxh3 `measure` landed (`modelcost`: held/peak bytes; gemma4 is the peak of one forward under its
      layer stream); `brain models profile --measure` prints both, or "not measured" (CLI; compiled in CI).
- [x] Commit 061f5d930 (gradcheck tied GPT-2 check) landed before f352eceb8 (the gpt2 tie) and does not build alone. Both
      are published, so history is not rewritten; every commit from f352eceb8 on builds, and no chapter cites 061f5d930.

- [x] mimi `recon::tests::decoder_weight_grad_is_taken_at_the_quantized_latent` checks with a relative `2e-2`: its
      weights (seed 42) and input are fixed, so the comparison is one deterministic number, not a random draw that can
      land on a cancellation; no change needed.

- [x] ltxv `av_concept_learning`: an unprinted random seed with ~1/30 negative margin; now 8 printed seeds, mean gain.

## Open: needs hardware, weights or a reference the box does not have

- [ ] cli: `brain forecast finetune --chronos2/--fincast` over the new host fine-tunes (needs a brain-cli build, which
      this workstation does not run; the forecast CLI's control-gate line is likewise compiled only in CI).
- [ ] vulkan coopmat: parity test against a CPU reference, gated on the capability (written here, run on an sm_75+ device) (v04 ch06).
- [ ] rknpu: SiLU LUT lowering, stem space-to-depth, max pool on the NPU, CMA load order (needs the board) (v04 ch13).
- [ ] qwen3vlmoe: import and gate the 30B-A3B checkpoint. User decision 2026-11-13: the crate stays and the book does
      not describe it (v07 ch09).
- [ ] moondream3 crop geometry, fastvlm preprocessing, pulid end to end: reference dumps (v07 ch06, ch09, v08 ch11).
- [ ] kaldi fbank: torchaudio reference dump (v09 ch05).
- [ ] s3tokenizer: speech_tokenizer_v3 for CosyVoice 3 (v09 ch06).
- [ ] minimaxmusic3: fixed-seed reference song (v09 ch07).
- [ ] ltxv: Gemma 3 text path (v10 ch02).
- [ ] qwen3 DeepSeek parity ladder (`crates/qwen3/tests/it/deepseek_parity.rs:40,105-110`) asserts only a row
  cosine (COS_FLOOR 0.9999), so a dropped or doubled scale passes; add a rel_l2 bound as `rrdbnet/tests/it/parity.rs`
  does (needs the checkpoint and its reference dump) (v06 ch02).

## Book text only

- [ ] The 16 passages that describe defects already fixed (stale text). Done: v03 ch10, ch11; v05 ch09, ch11; v06 ch10;
      v08 ch11; v09 ch08; v10 ch07, ch08 (two); v11 ch04, ch06 (reframed as found and fixed, with a crate test of the
      floor trade). Narrated as found-and-fixed history, kept: v03 ch12, v09 ch06, ch09. v07 ch12 (two) rewritten with the
      letterbox. Volume VII describes the corrected code throughout; VIII to X follow.
