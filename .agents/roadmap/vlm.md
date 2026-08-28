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

### Known, unmeasured: two host syncs per layer per decoded token

`MoondreamBlock::decode_step` round-trips to the host twice per layer: once for
the tau scale (`tqr`/`tvr` -> `s3`, which the BATCHED path does too, so it is
inherent to how tau is implemented rather than new) and once to split the fused
`[1, 3d]` qkv row into the three separate buffers `model::block::gqa_decode_step`
binds. At 24 layers that is up to 48 pipeline syncs per generated token.

The split could be avoided with `Gpu::step_sliced` (sliced views of one buffer
at disjoint offsets are explicitly legal), but only by duplicating the shared
primitive's four dispatches locally or widening `model::block`'s decode API for
one caller. Neither is worth doing against an unmeasured cost on a machine that
cannot run the real weights. Profile per kernel kind first.

One round-trip in this loop WAS pure waste and is fixed: the hidden state used
to be read back and re-uploaded between every layer, on the mistaken belief that
a layer's `out` buffer needed carrying forward. Each layer has its own
`KvCache`, so layer i's `out` is consumed by layer i+1 immediately and nothing
else touches it - the state now stays on the device across all 24 layers.

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
- [ ] **Region/point/detect heads - BLOCKED on material that is not here, not
      on effort.** Everything this repo knows about them is the string
      `"model.region."`. There is no tensor manifest, no shape, no reference
      `region.py` (the golden dumper copies the modeling code out of the
      CHECKPOINT directory at runtime), no checkpoint on this machine, and no
      torch to run a dumper with. Writing the heads from memory would be an
      architecture invented against no golden, no reference and no weights -
      and the failure mode is a head that returns plausible coordinates that
      are wrong, which nothing here could detect.

      The first step is therefore discovery, and it is now free:
      `import::load`'s `Coverage::region_tensors` captures every
      `model.region.*` name and shape during the load that already streams
      every header (shape only, never the data - the heads are not built, so
      materialising them would be pure footprint). Whoever next has a checkpoint
      prints that instead of writing a throwaway script.

      After that, in order: dump per-stage goldens for the heads
      (`tools/goldens/`), port against them, then expose `point`, `detect` and
      `region_caption` as additional `capability::Action`s beside `caption` -
      the manifest already advertises one action, and adding more needs no new
      transport work.
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

## qwen3vl - where a caption's time went, and where it goes now

Captioning was measured at about six minutes per image on a box with two
idle 24 GiB Tesla P40s, with the run burning far more user CPU time than
wall time while both cards sat empty. The profiler built for this
(`crates/qwen3vl/src/bin/qwen3vl_bench.rs`, `caption` mode) attributes one
image per stage against the machine's own MEASURED roofline; every number
below is best-of-N with warm-up excluded, on one 2048x1536 photograph at
`--max-new 90`, machine otherwise idle, and the caption text is byte-for-byte
identical before and after.

| stage | before | after | fraction of measured roof | bound |
|---|---|---|---|---|
| image preprocess | 54 ms | 41 ms | 0.7% | bandwidth |
| vision tower | 133535 ms | 3587 ms | 20.4% | compute |
| projector / merger | 2257 ms | 220 ms | 14.8% | compute |
| prefill (1580 tokens) | 148662 ms | 104550 ms | 76.5% | bandwidth |
| decode + head (90 tokens) | 54891 ms | 10313 ms | 48.9% | bandwidth |
| **per image** | **339399 ms** | **118710 ms** | | |
| model build (once per process) | 8.5 s | 16.0 s | | |

2.9x end to end, and 37x on the vision half. The one-off model build is
noise next to the per-image cost, so there was never a loading problem to
find: the whole cost was marginal. (The build grew because the vision
weights now upload once there instead of once per image.)

### What was actually wrong

**The vision tower was pinned to the CPU JIT.** `Qwen3Vl::new` hard-coded
`Gpu::new_cpu(vision_pipelines())` for its vision half while the decoder
honoured the caller's placement, so the entire 24-block ViT and all four
PatchMergers ran on Cranelift-compiled WGSL no matter where the model was
placed. At the captioner's default pixel budget that is about 8 TFLOP per
image. The same line had been copied verbatim into `qwen35` and
`qwen35moe`, so it was a class rather than an incident; all three are fixed.

**Flipping the device alone would have crashed, not sped up - and this was
measured rather than assumed.** The tower attends the whole image as one
span and materialised a `[heads, n, n]` score slab; at 6400 patches that is
a 2621440000-byte binding against the card's 2147483644-byte
`max_storage_buffer_binding_size`, and the failure is a `create_bind_group`
validation error. `model::vit::attn_chunk_for` had existed for exactly this
and was never called from here, which is the most likely reason someone
pinned the tower to the CPU in the first place. Chunk first, prove the tower
runs on the card, then move it.

**Three fast siblings were registered and never dispatched**, which is this
repo's most reliable source of large wins:

- `matmul_reg3` instead of `matmul_rows` for the block linears, the patch
  embed and the projector: those dispatches went 39351 ms to 723 ms, the
  projector 2858 ms to 40 ms, and the tower 54.1 s to 14.8 s. `matmul_rows`
  is also the one member of the matmul family `backend-cpu` has no native
  AVX2 path for, so the same swap is worth 60.8 to 106.2 GFLOP/s on the CPU
  backend.
- `flash_attn_bidir_reg2` instead of the scores/softmax/apply trio: after
  the GEMM swap the trio was 94% of the tower's device time at about 2.8% of
  roof; fusing it took attention 13758 ms to 2086 ms and the tower to 24.0%
  of roof.
- `Qwen::decode_logits` instead of a host LM head. `generate` read the whole
  tied 151936x2560 table off the device once per caption (1.5 GB into RAM)
  and then swept it scalar and single-threaded per token: 45909 ms to
  2358 ms. That method's own doc already named this caller as the case it
  was written for.

**Two host round trips per token that nothing looked at.** Prefill drove
`step_mrope`/`step_embed_mrope`, each ending in a `[d_model]` readback the
loop discarded, 1580 times per image. `Qwen::prefill` already called that
readback "pure waste" in its own doc but could not carry an M-RoPE table or
a DeepStack row; `Qwen::prefill_mrope` is that step without the readback.

**An fp32 model was packing int8 activations.** `Ops::act` quantized every
activation unconditionally - a `max_abs_row` dispatch, a `quant_pack`
dispatch and a fresh `I8Scratch` allocation per activation group per layer
per token - for a buffer only the `I8`/`Q4` arm of `Ops::matmul` reads, and
there were no such weights. Two kernels at 4.4% of all device time, tens of
thousands of dead allocations per image, and dispatches per caption fell
from 310216 to 224968 when it stopped. `Ops::act_f32` is the opt-out and the
tier is read off the resident weights, never off a remembered request. This
one is not qwen3vl-specific: it applies to every fp32 `Qwen` decode.

**The tower was rebuilt inside every forward.** `VisionEncoder` and
`PatchMerger` borrowed a `&Gpu`, so a model could not hold one, so
`Qwen3Vl` kept host `Vec<f32>` weights and re-uploaded about 1.7 GB per
image. This is the same defect `moondream3` fixed for the same reason. It is
a FIXED cost per image, so it dominates small images: on a 600x600 photo the
tower was 4080 ms, most of it upload.

### What is left, and what it is worth

Prefill is now 105 s of the 119 s and sits at **76.5% of this card's
measured DRAM bandwidth** at the default budget and 89.3% at the smallest.
It is not a kernel problem any more: `qwen3vl_bench caption --profile` puts
`matmul_gemv_reg` at 83-87% of all device time and, dividing the weight
bytes it must stream by the time it takes, at essentially the card's full
measured 287.5 GB/s. There is nothing left in the GEMV. The decoder
is 4.02B fp32 parameters = 15.0 GiB read once per token at batch 1, which
bounds a batch-1 step at 17.8 tok/s on this card, so a 1580-token prompt
prefilled one token at a time cannot beat about 89 s however good the
kernels are. Only two things move it:

- **INT8 decoder weights.** The ceiling goes to 71.3 tok/s, so prefill would
  be about 28 s and the whole image about 50 s. `Qwen::new_shard_dt_decode`
  already takes a `Dtype` and the int8 GEMV kernels are already registered
  on the decode path, so the code change is small. It is LOSSY, so it must
  be opt-in and gated separately, never folded into a parity claim.
- **Batched prefill** - one weight sweep per N positions instead of per
  token, worth roughly an order of magnitude on the dominant stage. The
  decode tape is m=1 by construction (its activation buffers are sized for
  one row and `gqa_decode_step` attends one query against the cache), and
  `qwen3::serve`'s batched paged path has no M-RoPE, so this is a real piece
  of work rather than a patch. Note that `matmul_gemv`'s own header says it
  accepts `m <= 32`, so the kernel side of a small batch already exists.

### The pixel budget is the cheapest lever, and it is the caller's

Prefill dominates and its cost is linear in the visual-token count, so
`--max-pixels` buys more than any kernel left in this model. Measured on the
same 2048x1536 photograph at `--max-new 90`, after everything above:

| `--max-pixels` | visual tokens | per image | end-to-end tok/s |
|---|---|---|---|
| 512x512 | 234 | 22.8 s | 3.95 |
| 768x768 | 540 | 44.6 s | 2.02 |
| 1024x1024 | 972 | 76.9 s | 1.17 |
| 1280x1280 (the captioner default) | 1564 | 118.7 s | 0.76 |

At the smallest budget prefill reaches **89.3% of the card's measured DRAM
bandwidth**, which is about as close to the wall as this shape gets. 512x512
is **15x faster than the original run at the default budget**, and the
caption it produces is still a full descriptive paragraph - arguably better
organised prose than the largest budget, which spends its tokens on a
bulleted furniture inventory.

This is deliberately NOT a default change. Fewer visual tokens is less of
the image, and how much detail a caption needs is the caller's judgement,
not the engine's - so it stays a flag with a published cost curve.

### Doubt the meter first: three instrument bugs in one pass

Every one of them made the engine look better, or the analysis look sharper,
than it was, and none would have been caught by a test:

- `roof::utilisation_of` returns a PERCENT, and the profiler scaled it again
  - 1.6% of roof was printed as 162%.
- `roof::known` answers only from an in-process cache, so querying it before
  any device existed reported every stage as having no measured roof.
  `caps::device_roof` reaches the handle behind the resident instead.
- The prefill byte model charged the whole parameter count per token, but
  prefill applies no LM head and its embedding gather reads one row, not the
  table. About a tenth too much - enough to print prefill at 100.3% of a
  roof nothing can exceed.

Also: `BRAIN_PROFILE`'s per-kernel table prints when the device DROPS, and a
resident model's device never drops, so on this path it simply never
appeared. `qwen3vl_bench caption --profile` goes through `Gpu::dump_profile`,
which exists for exactly that.

### Cosine is not a gate. This is the fifth independent reproduction.

`crates/qwen3vl/tests/vision_tower_parity.rs` mutates one weight by a
relative 5e-4 and re-runs the tower. The mutant scores **cosine
0.999999998** against the reference - it passes a 0.9999999 cosine floor -
and is rejected only by rel_l2 at 5.7e-5 against a 1e-5 ceiling. The fused
flash path reproduced the same figure independently on the same fixture.

That is five separate components in this codebase where a real defect scored
0.99999+ on cosine, so it should be read as a property of these numerics
rather than as bad luck. **Assert cosine AND rel_l2, never cosine alone**,
and mutation-verify the pair in the test file itself: two gates found here
in one day could not fail at all, and a gate nobody has watched fail is a
hypothesis.

The same file caught a real out-of-bounds read while being written. A main
PatchMerger's `[in_dim]` LayerNorm handed to a DeepStack merger reads
`merge^2 - 1` rows past its own buffer, because the LayerNorm width comes
from the dispatch `Params` and not from the buffer; the result is NaN or
finite-looking garbage with nothing to say which weight was wrong.
`PatchMerger::new` now checks the four shapes that decide its dispatch.
