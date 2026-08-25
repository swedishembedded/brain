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

### Done: it fits, and it is served

**The memory wall was the real blocker**, not the missing `caps.rs`:

| | fp32, per-block scratch | int8 + shared scratch |
|---|---|---|
| weights | 32.8 GiB | 8.2 GiB |
| activation scratch | 10.3 GiB | 0.6 GiB |
| **total** | **43.1 GiB** | **8.8 GiB** |

- [x] **int8 expert weights** (32.8 -> 8.2 GiB). `MoeFfn8`: weights packed with
      `model::int8::quantize_weight`, dispatched through `moe_linear_gated_i8`,
      the layer input quantized ONCE and shared by all 64 experts, and unrouted
      rows skipped (87.5% of the expert-row work at top_k=8-of-64, which the
      fp32 tier still does densely before gating). A separate type rather than a
      flag on `MoeFfn`, because `MoeFfn` is the training path and a quantized
      weight has no gradient - a precision branch inside differentiated code for
      a tier that is never differentiated is how that code rots. The router
      stays fp32.
- [x] **Shared inference scratch** (10.3 -> 0.6 GiB, 16.7x). `BlockScratch` +
      `MoondreamBlock::forward_on` + `MoondreamDecoder::share_scratch`.
      Structurally inference-only: `without_scratch` drops the block's own set
      and `backward` asserts by name rather than differentiating against
      whatever the shared buffers hold.
- [x] **Pixel-space overlap multi-crop** (`preprocess::overlap_crop_image` +
      `plan_crops` + `patchify_crop`). The geometry is DERIVED from the
      already-ported feature-space `reconstruct_from_crops` - stride
      `grid - 2·margin` patches, crop `grid` patches, resized image
      `stride·tiles + 2·margin` - and a test round-trips the two so they cannot
      drift. The old "blocked on a JPEG/PNG decoder" note was stale.
- [x] **The serving contract.** `moondream3::caps` (one streaming `caption`
      action with real `prompt_tokens`/`completion_tokens`/`finish_reason`),
      `crates/cli/src/resident_moondream3.rs`, a `catalog.rs` entry (so
      `brain caps`, `brain moondream3 caption`, D-Bus and the HTTP surfaces all
      light up at once), and `examples/vision/moondream3_caption.py`. The
      `crates/arch` row gained `default_ref` and `weights_env`.

      The catalog id is `brain/moondream3`, NOT the upstream repo name.
      `crates/cli/tests/model_ids.rs` caught the first attempt at
      `moondream/moondream3-preview` and was right to: a catalog id names an
      upstream repo only when the weights are exactly that release or nothing
      (`deepseek-ai/DeepSeek-OCR`, one shipped GGUF pair). `BRAIN_MOONDREAM3_WEIGHTS`
      points at an arbitrary directory, so the id is a placeholder for
      "whatever is configured" and the upstream repo lives in the separate
      `default_ref` field.

      Precision is part of the INSTANCE KEY, so int8 and fp32 are two separately
      budgeted instances - an fp32 request on a machine without room fails
      placement instead of evicting a working int8 one.

### Remaining

- [x] A KV-cached decode path (`MoondreamModel::generate_kv`). The prompt pays
      ONE batched masked forward that also seeds every layer's `KvCache`; each
      token after that is `MoondreamDecoder::decode_step`, `O(pos)` rather than
      a full `O(T²)` recompute.

      Three things it had to get right, each of which runs and returns
      plausible ids when wrong:
      * **The prefill must stay masked.** Decode steps are causal by
        construction (`gqa_decode_step` reads cache rows `0..=pos` and no
        others), which is correct for generated rows and would silently drop
        the image prefix's BIDIRECTIONAL attention. So the prefix's K/V come
        from the batched pass under the full prefix-LM mask, once.
      * **A new kernel, `rope_partial_at`.** `rope_partial` takes its position
        from the ROW INDEX, so a one-row call is always position 0 - it cannot
        express a new token at position 137. `rope_at` is the existing
        explicit-position twin but rotates the FULL head_dim, where Moondream
        rotates 32 of 64. This is `rope_partial_at` standing to `rope_partial`
        exactly as `rope_at` stands to `rope_base`, and its header says so.
      * **The cache is seeded from the POST-RoPE, POST-tau `qkv`**, extracted
        per block DURING the forward - on the shared-scratch path every block
        writes the same buffer, so a single pass at the end would cache the
        last block's values for all 24 layers.

      Gated by `kv_decode_matches_the_recompute_path_token_for_token` (and a
      tau-off twin). The two paths share no code below `generate`, so agreement
      is evidence rather than tautology. A SIGSEGV during bring-up was the
      `embed` kernel's token buffer being written as f32 bits when it is
      `array<u32>` - a garbage gather index, not a logic error.

      `caps` now uses it.
- [ ] Real batching. Each request has its own image, so the ViT pass is
      per-request; the decoder has no batch axis wired. `run_batch` is the
      serial default and says why.
- [ ] Region/point/detect heads - recognized on import, not built.
- [x] A GPU placement. `MoondreamModel::new_on` / `Session::load_on` take a
      canonical device index and build under a SCOPED registry selection
      (`gpu_core::devices::with_gpu`), never an env write; `estimate` reports
      the footprint as `vram`, so `place::pick_device` prefers a card and falls
      back to the CPU pool on a GPU-less machine by its own rule for a
      weight-holding model.

      **What that rests on**: not a real-weight run - none exists anywhere for
      this model on an accelerator, and no checkpoint is on this box. It rests
      on the device PLUMBING being checked:
      `a_gpu_build_computes_the_same_function_as_the_cpu_build` builds a
      tiny-config model on a real card and on the CPU backend and requires the
      logits to agree (cosine > 0.9999; two backends, so not bit-equality). A
      scoped selection that fell through to the ambient device, or one tower on
      a different backend from its own buffers, both run and both fail it. An
      NPU assignment is refused by name.

**None of the above is verifiable at real scale on this box** (30 GiB RAM, one
integrated GPU, no checkpoint present). Gate it with tiny-config end-to-end
tests through the PRODUCTION path - `import::load`, not a test-local loader -
and leave the real-weight tests skip-if-absent, the arrangement
`crates/deepseek2ocr` uses.
