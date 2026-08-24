# vlm - roadmap

Vision-language models: an image (and, for one variant, more) plus text in,
text out.

## qwen3vl - real-weight OOM - fixed

`brain qwen3vl generate` against the real `Qwen/Qwen3-VL-4B-Instruct`
checkpoint used to panic immediately with `wgpu error: Out of Memory` on a
24 GiB Tesla P40, reproducible on every attempt, with the GPU fully idle
immediately before and during the failing run. Root cause: `Qwen3Vl::new`
built its inner decoder via `Qwen::new`, the full BATCHED TRAINING
constructor - weight+grad+adam_m+adam_v (4x the checkpoint's weight bytes)
plus quadratic `[heads,seq_len,seq_len]` attention-score buffers sized for a
batched forward `Qwen3Vl::generate` never runs (it only ever drives the
decoder through its incremental KV-cache decode path). At the real 4B
checkpoint's scale that is a genuine multi-tens-of-GB accidental allocation
attempt, not a false-alarm binding-size cap. Fixed by switching to
`Qwen::from_tensors_decode`, the purpose-built decode-only constructor
(frozen weights only, linear KV-cache scratch instead of the quadratic
batched shape). Verified working end to end against the real checkpoint.

**Follow-on regression, also fixed**: `Qwen3Vl::new` is a SHARED
constructor - both the real-checkpoint `generate` path (via
`from_tensors`/`from_hf`) AND this crate's own `model::tests::
end_to_end_forward_is_finite` (which drives the batched-training
`forward()`/`backward()` path, not `generate`) go through it. The fix above
switched it to `Qwen::from_tensors_decode` unconditionally, which silently
broke `end_to_end_forward_is_finite`: a `decode_only` decoder never
allocates the batched `fwd_steps` graph's buffers, so `forward()`'s write
landed on whatever smaller buffer happened to be there instead -
`wgpu error: Validation Error ... Copy at offset 0 for 28 bytes would end up
overrunning the bounds of the Destination buffer of size 4`, followed by a
SIGSEGV on process exit (a real, silent correctness hazard, not just a
missed OOM) - 100% reproducible in isolation. Fixed by adding an explicit
`decode_only: bool` parameter to `Qwen3Vl::new`: `from_tensors`/`from_hf`
(the real-checkpoint path) pass `true` (preserving the original OOM fix);
`end_to_end_forward_is_finite` passes `false` (restoring the batched
`Qwen::new` constructor its `forward()` call needs);
`generate_is_deterministic_and_respects_eos`'s two construction sites pass
`true` (they only ever call `.generate()`). Verified: all 52
`brain-qwen3vl` lib tests pass, including both of the above.

## fastvlm - segfaults (sometimes hangs) on exit after correctly writing its output - same driver defect as the backend-vulkan/gpu-core teardown crash, root cause found, not fixable from this codebase

`brain fastvlm caption` against the real `apple/FastVLM-0.5B` checkpoint
intermittently segfaults (occasionally hangs instead, needing `SIGKILL`) on
process exit, but only after the caption has already been generated
correctly and written to the requested output file - the failure is in some
cleanup/Drop path, not in the computation itself.

**Root cause found**: this sandbox does NOT block `ptrace` (an earlier
write-up here was wrong) - it is YAMA-restricted to parent/child, and
running the crashing binary directly under `gdb -batch -ex run -ex bt`
(gdb spawns it as gdb's own child, satisfying the restriction) catches it
cleanly. The captured backtrace is frame-for-frame identical to a separate
crash documented in `.agents/roadmap/backend-vulkan.md`'s "intermittent
SIGSEGV at test-binary exit" entry, seen independently in
`brain-qwen3 --lib`'s test teardown: a thread
named `"[vkps] Update"`, owned by the NVIDIA driver itself (no symbols -
it's inside the closed-source driver blob), with a fully corrupted,
symbol-free stack. **This is the same bug as that entry, not a separate
one** - two different call paths (a real CLI command's device teardown here,
a test binary's pooled-device teardown there) hitting the same underlying
driver defect. See that entry for the full investigation: four independent
mitigations were tried against this exact failure (extra settle time before
device destruction, always-leak instead of destroy, forcing serial
execution, skipping libc's `atexit`/`dlclose` path via `libc::_exit()`) -
none helped, two made the measured crash rate worse, and one directly
falsified the pre-existing theory that concurrent GPU dispatch was the
trigger. The conclusion there applies here too: this reads as a genuine
NVIDIA driver defect (570.195.03), not something userspace Vulkan/wgpu API
sequencing can reliably avoid.

Worked around in `scripts/demo/quickstart.sh` (`|| true` plus a check that
the output file is actually non-empty) rather than blocking the quickstart
on it - the real bug is open, tracked, and (per the investigation above)
outside this codebase's ability to fix without a driver update.

## Moondream 3 - port completion

The earlier note here ("a capability manifest, a CLI path, a servable
pipeline") described the symptom, not the work. `caps.rs` was never the
blocker; the pieces a `caps.rs` would have to call did not exist.

### Done

- [x] **The composite can be resident at all.** `SiglipEncoder`, `Connector`,
      `MoondreamBlock`, `MoeFfn` and `MoondreamDecoder` each held a `&'g Gpu`,
      and a struct cannot own both a device and something borrowing it - so
      `MoondreamModel` stored host `Vec<f32>` weights and REBUILT the ViT, the
      connector and all 24 blocks inside every `forward`. At the preview config
      that re-uploads ~33 GB per request. All five now own their
      `DeviceBuffer`s and take `&Gpu` as an argument, the shape
      `sam1::SamEncoder`/`sam2::Sam2` already use and the reason THEY can be
      resident. Pinned by `a_second_forward_reuses_the_built_stack_and_agrees`.
- [x] **A production checkpoint loader.** `import::load` streams the shards
      through `checkpoint::WeightReader`, applies the name maps, splits the
      stacked MoE experts, and reports TWO-WAY coverage (refuses on an unmapped
      tensor AND on a missing required key). The only loader before this lived
      in a `#[cfg(test)]` module, so nothing user-facing could load the model.
      `brain-checkpoint` moved from a dev-dependency to a real one.
- [x] **Config from the checkpoint's own `config.json`**
      (`MoondreamConfig::from_json`/`from_dir`), which then REFUSES anything
      that is not the preview architecture, naming the field that differs.
- [x] **Greedy `generate`.** Exact despite the fixed-`seq_len` graph, because
      the mask is `(i<P && j<P) || (j<=i)` - causal past the image prefix, so
      padding cannot reach the row being read. `O(T²)` per token; no KV cache.

### Remaining, and what it is actually blocked on

**Memory. The model does not fit anything today**, and that is the reason the
serving surface is not worth writing yet rather than an independent task:

| | fp32, per-block scratch (today) | int8 + shared scratch |
|---|---|---|
| weights | 32.8 GiB | 8.2 GiB |
| activation scratch | 10.3 GiB | 0.6 GiB |
| **total** | **43.1 GiB** | **8.8 GiB** |

- [ ] **int8 expert weights** (32.8 -> 8.2 GiB). Every piece exists: pack with
      `model::int8::quantize_weight`, dispatch `moe_linear_gated_i8`, quantize
      the shared activation once per layer with `model::int8::quant_rows_steps`,
      and accumulate with `scale_add`. `model::moe::expert_fwd_i8` is the worked
      example and is ALMOST reusable - it composes gate/up -> `silu_mul` ->
      down, where Moondream needs w_h/w_g -> `geglu_shift` (`gelu(h)·(g+1)`) ->
      w_down. So this is an `MoeFfn8` beside the existing `MoeFfn`, not a
      change to it, plus an int8 branch in `MoondreamBlock::new` and
      quantization in `import::load`.
- [ ] **Share the inference scratch across blocks** (10.3 -> 0.6 GiB, 16.7x).
      Each block currently owns its full scratch set because the forward IS the
      backprop cache. 58% of it is `scores`+`probs` (`n_heads·t²` twice = 256
      MiB per block at t=1024). For inference nothing reads them again, so one
      shared set plus a per-block `out` suffices. Must be an inference-only
      path - sharing it under the existing backward would have every block
      differentiate against the last block's activations.
- [ ] **Pixel-space overlap multi-crop.** `preprocess.rs` implements the
      feature-space half (reconstruct -> adaptive-pool -> global‖local concat)
      and its own doc says the pixel-space `overlap_crop_image` is deferred
      because "brain still lacks a JPEG/PNG decoder" - that comment is STALE,
      `crates/imaging` has codecs now and `capability::blob::decode_image`
      hands over raw HWC f32 anyway. `imaging::tiling::moondream_select_tiling`
      already holds this model's tile-count policy.
- [ ] **The serving contract**, once the above make it runnable: `caps.rs`
      (one `caption` action, streaming, real `prompt_tokens`/
      `completion_tokens`/`finish_reason`), `crates/cli/src/
      resident_moondream3.rs`, a `catalog.rs` entry, D-Bus, and
      `examples/vision/moondream3_caption.py`. The `crates/arch` row also still
      lacks `default_ref` (`moondream/moondream3-preview`) and `weights_env`.
- [ ] A KV-cached decode path, to make `generate` `O(1)` per token rather than
      re-running 24 layers over a 730-row image prefix per token.

**None of this is verifiable at real scale on this box** (30 GiB RAM, one
integrated GPU, no checkpoint present). Gate it with tiny-config end-to-end
tests through the PRODUCTION path - `import::load`, not a test-local loader -
and leave the real-weight tests skip-if-absent, the arrangement
`crates/deepseek2ocr` uses.
