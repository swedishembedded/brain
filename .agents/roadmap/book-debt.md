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
- [ ] imaging letterbox: pixels and boxes share one pad; half-pixel nearest resize and pad value on device; retire the
      host loop and the pinning test (v07 ch01, ch04, ch12).
- [ ] imaging: migrate the call sites the crate header lists as owed (v07 ch01).
- [ ] clip: DeepSeek-OCR vision tower activation agrees with upstream (v07 ch08).
- [ ] zipdepth: scale-and-shift-invariant loss with gradient term and a real loader (v07 ch05).
- [x] instantid: per-site k and v computed once in `set_id` (v08 ch11).
- [ ] rrdbnet: exact tiling or an asserted seam bound (v08 ch12).
- [ ] vqgan / mimi: perceptual (lpips) and adversarial terms wired into training (v08 ch03, ch04, v09 ch03).

Audio, video, simulation:
- [ ] mimi: apply the encoder sliding window (v09 ch03).
- [ ] mimi: mel reconstruction backward with gradcheck, used by the trainer (v09 ch03).
- [ ] qwen3omnimoe: time-aligned M-RoPE positions and chunked interleave, or reject what is parsed and ignored; reject
      ignored sampling parameters (v09 ch08).
- [ ] minimaxmusic3: wire or remove the uncalled discriminator backward; train.rs states only what exists (v09 ch07).
- [ ] videocodec / imaging: seek to the preceding key frame (v10 ch01).
- [ ] ltxv / wan: real positions and per-token timesteps in `make_flow_batch`; Wan image-to-video or drop `Task::I2v` (v10 ch02).
- [ ] genieredux: multi-pass MaskGIT sampler gated against a reference (v10 ch07).
- [ ] genieredux / diamond: what the volume promises (KV cache, `WorldModel`, reward head) or a corrected promise (v10 ch07).
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
      `queue_submits` is the physical count). Knowledge #437.
- [x] Flaky: moondream3 `geglu_shift_grads_match_finite_differences` (tolerance scaled by a cancelling derivative; now
      derived from the absolute sum of the terms), backend-wgpu `kernel_timing::dropping_a_shared_handle_does_not_wait_for_the_device`
      (`submit` drained the device through the unsynchronised-workgroup valve before the test looked; now one naive matmul).
      Latent, same shape, not changed: the `rel < 1e-2` against `max(|d|, 1e-3)` checks of the other tests in
      `crates/moondream3/src/kernels_check.rs`.
- [x] Binding-offset alignment: every listed `step_sliced` site proved aligned in a comment (`head_ce.rs`/`lossw.rs` chunk
      starts, `moe.rs` `route_rows` regions, `ops.rs` quant rows/embed tiles/two-level passes); the six `chunked_bidir_*`
      entry points refuse a window off the grain at the call (`block::assert_row_windows_bindable`).
      Open: CUDA's `step_native_sliced` and `step_library_gemm` (`crates/backend-cuda/src/backend.rs`) do not call
      `check_binding_offsets`, so a misaligned native slice is accepted there and refused everywhere else; adding it
      needs a CUDA device to confirm no provider passes an unaligned offset.
- [ ] gemma4 and minimaxh3 `measure` landed (`modelcost`: held/peak bytes; gemma4 is the peak of one forward under its
      layer stream); `brain models profile --measure` still does not print the new byte fields (CLI; needs a brain-cli build).
- [ ] Commit d9da9049b (gradcheck tied GPT-2 check) landed before 961487746 (the gpt2 tie) and does not build alone.

## Open: needs hardware, weights or a reference the box does not have

- [ ] cli: `brain forecast finetune --chronos2/--fincast` over the new host fine-tunes (needs a brain-cli build, which
      this workstation does not run; the forecast CLI's control-gate line is likewise compiled only in CI).
- [ ] vulkan coopmat: parity test against a CPU reference, gated on the capability (written here, run on an sm_75+ device) (v04 ch06).
- [ ] rknpu: SiLU LUT lowering, stem space-to-depth, max pool on the NPU, CMA load order (needs the board) (v04 ch13).
- [ ] qwen3vlmoe: import and gate the 30B-A3B checkpoint, or remove the crate (v07 ch09).
- [ ] moondream3 crop geometry, fastvlm preprocessing, pulid end to end: reference dumps (v07 ch06, ch09, v08 ch11).
- [ ] kaldi fbank: torchaudio reference dump (v09 ch05).
- [ ] s3tokenizer: speech_tokenizer_v3 for CosyVoice 3 (v09 ch06).
- [ ] minimaxmusic3: fixed-seed reference song (v09 ch07).
- [ ] ltxv: Gemma 3 text path (v10 ch02).

## Book text only

- [ ] The 16 passages that describe defects already fixed (stale text; v05 ch11 and v08 ch11 done), and the v11 ch06 sentence on an old commit message.
