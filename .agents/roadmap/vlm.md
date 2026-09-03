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

- **INT8 decoder weights.** Done, opt-in, measured - see the next section.
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

### The int8 tier: what it costs in time, and what it costs in captions

`--precision int8` on `brain label images`, `precision` on the served action.
Opt-in, never a default, and gated in its own file
(`crates/qwen3vl/tests/int8_tier.rs`) which asserts nothing about caption
equality - the exact-output claim this crate makes is about the fp32 path and
must stay unqualified.

**Speed**, same 2048x1536 photograph, `--max-new 90`, Tesla P40:

| `--max-pixels` | fp32 | int8 | speedup |
|---|---|---|---|
| 512x512 (234 visual tokens) | 22.3 s | 10.7 s | 2.08x |
| 1280x1280 (1564 visual tokens) | 118.7 s | 63.6 s | 1.87x |

int8 at the small budget is **32x** the original 339 s run. The gain SHRINKS
with context, which is not a defect but a consequence: the per-step attention
is O(position) and the small per-step kernels are fixed, and both were hidden
under a 50 ms fp32 weight read that int8 cuts to 12.6 ms. At the default
budget int8 prefill sits at 39.7% of the DRAM roof where fp32 sits at 76.5%,
because there is simply less weight traffic left to be bound by.

**It only reached that speed after a kernel fix.** `matmul_gemv` had been
rescued from its workgroup-memory accumulator by `matmul_gemv_reg`, wired
through `gpu_core::upgrade` so every model inherits it - and the int8 twin
was never given the same treatment. `matmul_i8_gemv` still carried
`array<i32, 2048>` in workgroup memory, sized for the worst case, with a
read-modify-write per `(k, m)`: 149 GB/s against a 287.5 GB/s roof, where the
fp32 register kernel reaches essentially all of it. So int8's four-fold
smaller weights were returning a little over two-fold in time.
`matmul_i8_gemv_reg` plus its upgrade row took that kernel to 258 GB/s (90%
of roof) and a whole captioning run from 1.52x to 2.08x. Bit-identity there
is structural rather than careful - the accumulator is `i32`, and integer
addition is exact and associative - which let its gate check the raw bits AND
an exact `i64` oracle, so a bit-identical pair cannot be identically wrong.

**Quality: this is not a free 2x.** Over six real photographs, **0 of 6**
int8 captions matched fp32, mean word overlap 0.517. Greedy decoding is a
chain, so one flipped token rewrites everything after it and that count
overstates the QUALITY gap while accurately stating the determinism one. What
matters is the character of the differences, and reading them, they split
roughly evenly:

* **Structural, not factual.** Two images diverged at the first word into a
  different document shape - flowing prose against a bulleted room inventory.
  Both are good captions; neither is wrong.
* **Factual, and they cannot both be right.** The pendant lamp is "multiple
  concentric rings" at fp32 and "two concentric, circular tiers" at int8. A
  bedroom is "modern, rustic" against "modern, minimalist". A hallway has "a
  white wall-mounted unit holding a small vase with pink flowers" at fp32 and
  "a white pillar or partition" at int8 - different objects named in the same
  place.

For labelling training data that second class is the whole decision, and it
is why the tier is opt-in with a `compare` mode rather than a default with a
speed number. `qwen3vl_bench compare --dir <dir>` prints both captions in
full, and says in its own output that a similarity score cannot tell cosmetic
from substantive.

The bound underneath all of this is `qwen3`'s own
`int8_kv_decode_tracks_fp32`: relative L2 under 10% on the decode hidden
state. That is a generous bound, and a 10% perturbation of the hidden state
flips a greedy argmax readily - which is precisely why the captions differ,
and why nobody should be surprised by it.

### Batching across images: the same work as batched prefill, and cheaper than it looked

Asked whether a labelling run could batch N photographs at once. It can, and
the enabling work is the SAME piece as batched prefill - which changes the
cost calculus, so it is written down here rather than left implied.

**`qwen3::serve::Engine` already is the batched graph.** It has paged KV over
`max_batch` sequences, a `max_prefill`-row batched prefill forward,
`Input::Embeds` for raw embedding rows (which is exactly how an image's
merged visual tokens would enter), a device-side batched greedy head, and
`weights_int8`/`kv_int8` flags. It does not need the training constructor;
that was my earlier scoping and it was wrong.

**What it is missing is one thing: M-RoPE.** `serve` rotates through
`rope_paged`, which takes one SCALAR position per row and the analytic theta.
Qwen3-VL needs a 3-axis position with an interleaved section split, i.e. a
per-row cos/sin table. `rope2d` is already that shape (`rows, heads, half,
stride, offset, tmod`) and serve's `sc.q`/`sc.k` are contiguous `[b, hq]`
slabs, so the change is a table-driven RoPE variant plus a per-request table
upload. The genuinely new work beside it is DeepStack, which is wired into
`forward_steps` and `decode_steps` but not into `run_batched_steps`.

**Where the win actually is, which is not where it sounds like it is.**
Prefill within ONE image is already batchable: its 1580 positions are
independent given causal masking, and `max_prefill` processes a whole chunk
per weight sweep. That alone turns 1580 weight sweeps into single digits, and
it needs no second image. Across-image batching helps the DECODE phase, where
each image contributes exactly one row per step.

**Saturation.** Prefill stops being weight-bandwidth bound and becomes
compute bound at `R = 2F/B` rows per sweep - 73 rows at fp32's measured 10517
GFLOP/s and 287.5 GB/s. One image's prefill already supplies far more than
that, so batching MORE images adds nothing to prefill. Decode contributes one
row per image per step, so it scales close to linearly with batch size until
the same 73-row crossover, which is far beyond what memory allows.

**And memory is the binding constraint, but not the one expected.** Per
sequence at the default budget: KV cache 0.46 GiB, per-row activations
negligible - and the LM head's `[batch, vocab]` logits slab at **0.57 GiB per
sequence**. On a 24 GiB card, fp32 leaves about 7 GiB after weights and the
vision tower and int8 leaves about 17 GiB, so KV alone would allow roughly 15
and 37 sequences. The logits slab caps both at about **6**, because at 8
sequences it is 4.9 GB - past `max_buffer_size` (4094 MiB) entirely, and any
single binding of it is past `max_storage_buffer_binding_size` (2047 MiB)
from 4 sequences up.

That cap is removable and the mechanism already exists: `Qwen::head_steps`
vocab-tiles the head WEIGHT for exactly this reason. Tiling the logits too,
with a running argmax folded across tiles, would take int8 to roughly 16
sequences. Whoever picks this up should treat "tile the logits" as part of
the work rather than discovering the 2047 MiB limit the way the vision tower
did.

### Doubt the meter first: four instrument bugs in one pass

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
- The same byte model then priced INT8 weights at four bytes each and
  reported prefill at 164.5% of roof. Two of the four were caught by an
  impossible number rather than by a test, which is the argument for printing
  a fraction of a measured roof next to every stage instead of a bare
  millisecond count.

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

## qwen3vl - real sampling, and context as a real (checkpoint-derived) number instead of a fixed 4096

`Qwen3Vl::generate_timed`'s decode loop was hardwired to `argmax` and
`crates/qwen3vl/src/caps.rs` had a `const SEQ_LEN: u32 = 4096` gating every
request regardless of what the checkpoint actually declares - `1.56%` of a
real Qwen3-VL-4B-Instruct release's `262144`-token
`max_position_embeddings`, and both facts were stated in the served
manifest as permanent limitations rather than validation-tier defaults.

Fixed the two together, since both are "the served surface undersells what
the architecture can do":

- **Sampling**: `crates/qwen3vl/src/sample.rs` is this crate's own
  temperature/top-k/nucleus `sample_logits` (a small, deliberate duplicate
  of `qwen3::sample`'s algorithm, per this repo's existing per-model
  sampling-tail convention - see `qwen35moe::sample`'s own doc for the same
  reasoning). `Qwen3Vl::generate`/`generate_cb`/`generate_timed` take a new
  `SampleParams { temperature, top_k, top_p }` + `&mut Rng`;
  `SampleParams::greedy()` (temperature 0.0) reproduces the exact original
  argmax behaviour, and is still the served default. `caps::generate_spec`
  gained `temp`/`top_k`/`top_p`/`seed` params, same names/defaults/bounds as
  `qwen3::caps`, so the two served surfaces are consistent.
- **Context**: `SEQ_LEN` is gone. `caps::default_ctx_len()` reads
  `$BRAIN_QWEN3VL_CTX` (default 24576, mirroring `qwen3`'s own
  `BRAIN_QWEN_CTX`), and `resolved_ctx_len(cfg)` clamps it DOWN to the
  checkpoint's own `max_position_embeddings` (already parsed by
  `Qwen3VlConfig::from_hf`, previously unused) - never up, so a smaller
  checkpoint variant cannot over-allocate past what it was trained for. The
  resolved value is what the resident's decoder is actually built with
  (`Resident.seq_len`), and a request that overflows it is refused BY NAME,
  naming both the built capacity and the checkpoint's real ceiling.

**What this is not**: native-262144 or paged-KV serving. This decode path
still allocates a plain linear fp32 KV cache
(`Qwen::new_shard_dt_decode`) - at the 4B config's shape that is ~288 KiB
per token, so the real 262144-token ceiling would be ~77 GiB for one
request. Reaching that natively needs the same paged-KV engine
`qwen3::serve::Engine` already has, extended with an M-RoPE-aware
`rope_paged` (today scalar-position only) and DeepStack wired into
`run_batched_steps` (today only `forward_steps`/`decode_steps`) - real,
separately-scoped follow-on work, not attempted here. Multi-image, video,
resident D-Bus/HTTP serving, the MoE family variants, agent/tool calling and
LoRA fine-tuning are all likewise still open; sampling and a
checkpoint-derived (rather than hardcoded) context ceiling were the two
items that needed no new architecture, only wiring already-tested math
(`mrope::get_rope_index_multi`'s multi-image/video math is already ahead of
`caps.rs`'s single-image action surface, for the same reason).

## qwen3vlmoe - name recognition + shape-only splice, real import deliberately not attempted

Scope was to take Qwen3-VL-30B-A3B (the MoE sibling `docs/models/qwen3vl.md`
already named as refused-by-name) exactly as far as could be verified without
a real checkpoint, and stop there rather than guess. Landed:
`crates/arch`'s `qwen3vlmoe` row (its own real HF class,
`Qwen3VLMoeForConditionalGeneration` - a real, checked finding: it is a
DIFFERENT class from dense `qwen3vl`'s `Qwen3VLForConditionalGeneration`, so
the two were never going to resolve to the same importer even before this
row existed), `crates/qwen3vlmoe::import::GGUF_ARCHITECTURE` +
`crates/cli/src/gguf_import.rs`'s `Qwen3VlMoeImporter` (the `SupirImporter`
pattern - registered so a real file auto-dispatches the day one exists, `Err`
otherwise), and `crates/qwen3vlmoe::{config, model}` - a config type and a
composite `Qwen3VlMoe` struct proven to compose (vision tower -> merger ->
DeepStack -> MoE decoder -> M-RoPE) on synthetic tiny configs only.

**What was actually verified, and how - not assumed.** `curl`'d the real
`Qwen/Qwen3-VL-30B-A3B-Instruct/config.json` directly (raw file, not a
paraphrase - see `crates/qwen3vlmoe/src/config.rs`'s module doc for the exact
byte content quoted). Two things that finding settled: `vision_config` is
byte-identical to `qwen3vl::config::VisionConfig::qwen3_omni()` (depth 27,
hidden 1152, `deepstack_visual_indexes` [8,16,24], all of it) - no new vision
tower needed, reused as-is. And `text_config` is `mlp_only_layers: []` /
`decoder_sparse_step: 1` (every layer routed, no dense-then-MoE schedule) /
128 experts / top-8 / `norm_topk_prob: true` / no
`shared_expert_intermediate_size` key at all - i.e. a plain GQA+QK-norm+RoPE
decoder with a top-k sparse MoE FFN and NO shared expert, which is
byte-for-byte the shape `qwen3omnimoe::config::MoeTextConfig::
thinker_defaults` already models for Qwen3-Omni's Thinker (different
`hidden`/`vocab`/`rope_theta` numbers, identical structure). That match is
the real, checked reason `crates/qwen3vlmoe::model` reuses
`qwen3omnimoe::thinker`'s `layer_fwd`/`final_norm`/`lm_head_fwd` directly
rather than writing a second GQA+MoE decoder - and the real reason NOT to
copy `qwen35moe`'s decoder despite `qwen35moe::vl` being the closest splice
PATTERN in the repo: `qwen35moe` is Qwen3.5-35B-A3B, a hybrid
Gated-DeltaNet/GQA architecture, a different checkpoint family with a
different real shape from Qwen3-VL-30B-A3B - confirmed by the same fetch, not
inferred from the similar model name.

**What was inferred by convention, not confirmed from the fetch, and is
named as such in `config.rs`'s doc**: `use_qk_norm`. The real `text_config`
carries no such key (every Qwen3-family decoder in this workspace applies
per-head QK-norm unconditionally with no config flag naming it, and
`MoeTextConfig::thinker_defaults`'s own doc records the identical situation
for Qwen3-Omni's real checkpoint), so the default follows that established
convention rather than a byte this fetch could check.

**DeepStack had to be added back**, because `qwen3omnimoe::thinker` does not
carry it - Qwen3-Omni's own real served path skips DeepStack entirely
(`qwen3omnimoe::mm`'s own doc: "not needed for a plain (non-DeepStack) splice
path"). `crates/qwen3vlmoe::model` adds ONE extra kernel dispatch
(`kernels::SPLICE_ADD`, the exact WGSL kernel `qwen3::Qwen`'s own DeepStack
residual add already uses - reused, not reimplemented) after each of the
first `deepstack_indexes.len()` decoder layers, appended to
`thinker_pipelines()`'s table so every hard-coded kernel index
`qwen3omnimoe::thinker`'s functions dispatch by stays valid.

**Where this stopped, deliberately.** No real `Qwen3-VL-30B-A3B` checkpoint -
safetensors or GGUF - was available to import against in this sandboxed
environment. A GGUF MoE decoder packs its routed experts as 3D
`blk.N.ffn_*_exps.weight` tensors (llama.cpp's `LLM_TENSOR_FFN_*_EXPS`
convention), a genuinely different leaf vocabulary from the dense
`qwen3vl::gguf_import`'s flat 2D per-layer linears, so there was nothing to
verify a mapping against and guessing at expert-tensor names or a
safetensors leaf convention was refused rather than attempted - exactly the
"a finding is a hypothesis until checked" line this repo holds itself to.
`crates/qwen3vlmoe::model::Qwen3VlMoe::forward` is exercised ONLY on
synthetic tiny configs (`model.rs`'s own tests): finite output, correct
`[n, vocab]` shape, a `should_panic` on a token stream missing the image
run. This is wiring proof, not real-weight parity, and neither this file nor
`docs/models/qwen3vl.md` claims otherwise anywhere.

**Open, for whoever picks this up next**: (1) a real checkpoint (HF
safetensors or GGUF) to import against - without one, the tensor-name mapping
in either `crates/qwen3vlmoe::import` (GGUF) or a new safetensors importer is
unwritable without guessing; (2) `use_qk_norm`'s convention-based default,
worth a second confirmation once a real checkpoint's tensor names are in
hand (a real `q_norm`/`k_norm` weight per layer would settle it the way it
settled Thinker's); (3) no capability/`caps.rs`, no CLI verb, no residency
adapter - this architecture is not reachable through `brain` at all yet, by
design (nothing to serve without real weights); (4) no backward/gradcheck
entry point - forward-only, and unlike this repo's other forward-only
exceptions this one has no real-weight run to prioritize reaching yet, so it
is simply not attempted rather than deferred with a reason.

## qwen3vl - residency + D-Bus (serving-contract obligations 2 and 4)

Qwen3-VL's `capability::Provider` (`QwenVlProvider`/`GenerateAction` in
`crates/qwen3vl/src/caps.rs`) already satisfied obligation 1 - it had its own
internal one-request-at-a-time `Mutex<Option<Resident>>` and the full
`generate` action interface. What was missing was a real
`residency::ResidentModel` adapter (obligation 2: build once, own the
memory, get scheduled/budgeted) and the D-Bus reachability + example
(obligation 4) that depends on one existing.

**The split, mirroring `moondream3::caps::Session`.** `caps::Resident` (the
struct the static mutex already built) is now `pub`, with two new public
methods: `load_on(dir, max_pixels, precision, gpu: Option<u32>)` - a scoped
`gpu_core::devices::with_gpu` placement, never an env write, matching every
other server-lifetime resident in this repo - and `generate(inv, progress)`,
which is the body `GenerateAction::run` used to hold directly. The direct
provider's static-mutex path and `crates/cli/src/resident_qwen3vl.rs`'s
scheduled instance now run the exact same `generate` code; nothing about
preprocessing, prompt assembly or token accounting exists twice.
`Qwen::new_shard_dt_decode` (which `Qwen3Vl::new` calls) already documented
that it lands on "the ambient selection (`--device` / scoped `with_gpu`)",
so no device parameter needed to thread into `crates/qwen3vl/src/model.rs`
at all - scoping the call site in `caps.rs` was sufficient.

**The footprint is derived, not measured** - no real checkpoint has been run
through this resident on an accelerator on this machine. The arithmetic
(`resident_qwen3vl.rs`'s `FP32_BYTES`/`INT8_BYTES` doc comments): decoder
~4.02B params (36 layers x ~100.9M/layer + the tied 389M-param embedding) at
4 bytes = ~14.98 GiB, vision tower ~374M params (always fp32 - a small
fraction of the weights and none of the per-token bandwidth) at 4 bytes =
~1.39 GiB. Weights alone land at ~16.37 GiB, which agrees with
`Qwen3Vl::from_hf`'s own pre-existing doc comment ("the released 4B
checkpoint is ~16 GB in f32") - independent confirmation the arithmetic is
in the right ballpark. Add the KV cache at this resident's `SEQ_LEN` (4096)
and DeepStack/splice scratch, rounded up: **19 GiB fp32, 8 GiB int8** (the
decoder linears alone drop to one byte each; the vision tower and KV cache do
not shrink).

**D-Bus needed zero `crates/dbus` changes, verified by tracing the code, not
assumed.** `Manager::run`/`subscribe` in `crates/dbus/src/service.rs` are
already fully model-agnostic - they take a `model: String` and dispatch
through `self.executor.submit(Job::new(model, action, inv)...)`, with no
per-model branch. Registering `Qwen3VlResident` in
`crates/cli/src/catalog.rs`'s `resident_ctor_for` (which `resident.rs::
build_executor` folds into `Executor::start` via `catalog::residents()`) was
the entire D-Bus wiring.

**A side effect worth naming: HTTP came free too, and is now claimed.** The
`generate` action's shape (streaming, a `prompt`/`messages` param, a `Text`
output) was already documented in `caps.rs`'s own module doc as chosen
SPECIFICALLY to satisfy `apiserve::catalog::api_caps`'s chat classification
- but that classification reads from `Executor::manifests()`, which only
ever lists REGISTERED residents. Before this change qwen3vl had none, so it
was invisible to both `/v1/models` lists despite the manifest shape already
being correct. `resident_qwen3vl.rs`'s
`brain_qwen3vl_is_auto_exposed_on_openai_and_anthropic_model_lists` test
drives a real `axum` router over a real `Executor` (weights path
`/nonexistent`, no checkpoint needed - `GET /v1/models` never activates
anything) and confirms `brain/qwen3vl` now appears on both. `docs/models/
qwen3vl.md`'s support table is updated to match - HTTP API and D-Bus both
[x] - because this is now verified, not aspirational.

**Batching (obligation 3) stays the documented serial default,
deliberately.** Moondream 3's vision tower batches across requests because
`SiglipEncoder::encode` attends within each crop independently of every
other request's crops. Qwen3-VL's vision tower does not have that property
here: its output is spliced directly into the decoder's own incremental
KV-cache decode (`Qwen3Vl::generate_cb`), so every request's image
placement, prompt and KV cache are entangled with that one request from the
first token. There is no stage that is both shared across requests and
independent of per-request state, so `run_batch` is left at
`residency::model::Instance`'s default sequential loop - the same call this
repo's `sdxlunet`/`controlnet`/`flux1`/`pulid` residents make for their own
serial multi-step samplers, and for the same reason.

**What is still open.** No real Qwen3-VL-4B checkpoint has been run through
this resident (activate/generate/estimate) on a GPU on this machine, so the
footprint constants and the `gpu_core::devices::with_gpu` placement plumbing
are verified by code-path tracing and by the existing `served_generate_path_
runs_on_real_weights` skip-if-absent smoke in `qwen3vl::caps::tests` (which
exercises `GenerateAction`/the static-mutex path, not the residency
adapter's own `Resident::load_on` construction path) - not by a real-weight
run through `resident_qwen3vl.rs` itself. True continuous batching
(`model::serve::{Scheduler, PagedDecoder}`, the pattern `serving-contract.md`
names for autoregressive decoder LMs) is out of scope here, same as it is
for every other single-request VLM resident in this repo.

## qwen3vl - `generate` takes N images, not exactly one

The served `generate` action required exactly one `image` blob:
`generate_spec()` declared one required image input, `Prepared::build`
assembled a prompt with exactly one vision-start/`[IMG]*`/vision-end run, and
`Qwen3Vl::generate_timed`'s prefill loop assumed one contiguous image-token
run against one merged `visual` buffer. `mrope::get_rope_index_multi`
(already gated by `rope_index_two_adjacent_images` and
`multi_image_and_audio_interleave_independently`) already computed correct
3-axis M-RoPE positions for several interleaved image blocks in one stream -
the position math was never the gap, only the wiring above it.

Wire shape chosen: `crates/capability`'s `Invocation`/`Outcome` blob API is
keyed by one string name per call, with no array-blob convention anywhere in
the repo (an array blob would ripple into the D-Bus fd-map and every HTTP
transport, the same tradeoff `capability::blob::decode_video`'s own doc
weighs for a different input kind). Rather than invent that, `generate_spec`
now declares 8 numbered blob inputs - `image` (required) plus `image1`..
`image7` (optional) - read contiguous from `image` by a new
`caps::decode_images`; a request using only `image` is unchanged, byte for
byte, from before this session (regression-tested in `crate::model`'s
`generate_is_deterministic_and_respects_eos`, which now also pins the exact
pre-change output `[3, 3, 3, 3, 3]` for that tiny-config scenario).

Wiring: `Prepared::build_multi` resizes/packs each image independently and
concatenates N vision-start/`[IMG]*`/vision-end runs into one prompt, in
key order. `Qwen3Vl::generate_timed` now takes `images: &[ImageInput]` and
runs the vision tower once per image (`VisionEncoder::encode_with_taps` has
no batch axis), concatenating each image's merged visual rows - and each
DeepStack level's merged rows - into ONE flat buffer per level, in image
order. The prefill loop's existing `visual_row` counter already walked every
image-placeholder token in the stream regardless of which image it belonged
to, so it needed no change: the fix was upstream of it (assembling the right
buffers), not in it. `generate_timed` now calls `get_rope_index_multi`
directly (`get_rope_index` was always a wrapper around it, so this is a
naming change more than a math one) with the per-image `grids_llm` list built
in the same loop.

Capacity: `n_visual_capacity` (`caps::visual_capacity`) now bounds the SUM of
one request's images, not the largest single image - `8 x
per_image_visual_capacity(max_pixels)`. This is required, not merely safer:
`qwen3::Qwen::enable_deepstack` allocates one flat `[n_rows, d_model]` buffer
per level, and `decode_steps`'s `deepstack_row` addresses a row in it that
walks every image in the request in sequence, never a per-image sub-range. A
capacity sized for only the largest single image would let a second image's
rows land past the buffer's end.

Found while writing the "second image changes the sequence" test: an
untrained tiny-config decoder's near-identity residual stream echoes
whatever token immediately precedes the generated position almost
regardless of context, so a token stream ending on a real vocab id (as
`generate_is_deterministic_and_respects_eos`'s prompt does) makes greedy
argmax insensitive to the image by construction, at EVERY decoder-weight
seed tried (1..20) - not a wiring bug, a property of that specific tiny
random init. Ending the test prompt on an image row (a continuous embedding,
not any one vocab id's own row) removes the confound; with that fixed, every
seed sweep showed the second image changing the output, confirming the
wiring is real. Left open: no multi-image real-checkpoint smoke exists yet
(the real-weight test in `caps.rs` still exercises exactly one image) -
worth doing once a real Qwen3-VL checkpoint with genuinely distinguishable
photos is on hand.

## qwen3vl - a real video input path, and what "Text-Timestamp Alignment" actually turned out to mean

`mrope.rs`'s own doc used to say plainly: "Video timestamp handling is
deferred - images use `t = 1`; the temporal axis simply counts frames from
the anchor." That was true, and `Qwen3VlCaptioner` hard-refused any clip with
more than one frame (`max_frames: 1`), so there was no video input path at
all, only a placeholder shape (`get_rope_index_multi` already accepted a
`t > 1` grid and had a passing diagonal-case test for it -
`multi_video_run_with_no_spatial_extent_is_diagonal_like_audio` - but nothing
computed a real per-frame position from it).

**First finding, and it changes the honest scope of "Text-Timestamp
Alignment":** Qwen3-VL's own technical report (arXiv:2511.21631, "Video
Timestamp", §2.3) says it **replaced** exactly the mechanism this task set
out to build. Qwen2.5-VL ties the M-RoPE T-axis to absolute time
(`position_temporal[ti] = start + ti * tokens_per_second * second_per_grid_t`
- read directly from `transformers`' `modeling_qwen2_5_vl.py`,
`Qwen2_5_VLModel.get_vision_position_ids`). Qwen3-VL's report states this
produces "excessively large and sparse temporal position ids for long
videos" and switches instead to **explicit text tokens** interleaved with
the frames - each temporal patch prefixed with a formatted string like
`<3.0 seconds>` in the PROMPT, not a position-id trick at all. So "Text-
Timestamp Alignment" the architecture feature is a token-stream change, and
what this crate's `mrope.rs` comment was missing (a real-time-driven T-axis)
is the mechanism Qwen3-VL evolved *away from* (T-RoPE), not its final
design.

Given that, the scope actually built here is: **implement T-RoPE's real
mechanism correctly, generalized to genuinely non-uniform per-frame timing**
(`mrope::get_rope_index_video` - real elapsed seconds between frame groups
drive the T-axis delta, `t_pos[i] = anchor + round(tokens_per_second *
(frame_timestamps_s[i] - frame_timestamps_s[0]))`), which is a real, useful
fix for the "just count frames" bug and is bit-identical to the verified
upstream formula whenever the timestamps happen to be uniformly spaced. The
literal text-token timestamp interleaving Qwen3-VL itself ships is **not**
implemented - that is a prompt-assembly change (inserting formatted timecode
strings between frame groups' visual tokens), independent of this one, and
is left as an open item below.

**`tokens_per_second`'s default is cross-checked against a source already in
this repo, not guessed.** Multiple web sources disagreed (4, 25, 41, "2 in
the models" from a GitHub issue), so rather than pick one, `crates/
qwen3omnimoe/src/config.rs` already has a REAL, checkpoint-sourced
`VisionConfig::tokens_per_second: u32` field for the exact same Qwen-family
`vision_config.tokens_per_second` key, parsed from a real `config.json` with
fallback default `2`. `qwen3vl::config::VisionConfig` now carries the same
field with the same default, for consistency within this repo rather than an
independently guessed number. It is optional in `from_hf`/the GGUF importer
(falls back to `2`) since a real Qwen3-VL `config.json` may not even carry
this key any more, per the finding above.

**What was built**, TDD (mrope tests written and confirmed red before
`get_rope_index_video` existed):
- `mrope::get_rope_index_video` - the real-timestamp T-axis function, with
  four new unit tests (uniform spacing differs from frame-count, non-uniform
  spacing is honored, embedding in surrounding text, and the real-spatial-
  extent meshgrid case).
- `Qwen3Vl::generate_video_timed`/`generate_video_cb` - encodes each frame
  GROUP separately through the existing, already-tested single-frame
  `VisionEncoder::encode_with_taps` (the tower's own doc says "one image ->
  one whole-image span"; there is no native multi-frame attention path to
  wire into, and claiming one would be an unverified architectural change),
  concatenates the per-group merged visual rows and DeepStack taps T-major,
  then splices via a NEW shared `splice_prefill_and_decode` helper factored
  out of `generate_timed` so the image and video paths consume real
  positions/visual rows through exactly one code path, never two.
- `caps.rs`'s `generate` action: `image` is no longer `.required()`; a new
  optional `video` input (`capability::blob::decode_video`'s existing wire
  format - no container/codec decoding was added, matching
  `qwen3omnimoe::caps`'s own precedent for "optional image input, optional
  video input, caller hands us decoded frames") plus a required-with-video
  `fps` param drive `PreparedVideo::build`. Exactly one of image/video is
  enforced, and a video's frame count / missing `fps` are refused by name
  BEFORE the checkpoint directory is even read - mirrors the existing empty-
  prompt-before-touching-weights contract.
- `Qwen3VlCaptioner` no longer hard-refuses a multi-frame clip: `max_frames`
  is now `caps::MAX_VIDEO_FRAMES` (32 - a chosen, bounded scope, not a
  verified upstream limit), and `caption()` routes a non-still clip through
  the video path, reusing `captioner::Clip::fps` (the only per-clip timing
  that contract carries) as the single, uniform `fps` this task's scope
  settled on (see below).

**What is honestly still open:**
- Text-token timestamp interleaving (Qwen3-VL's actual "Text-Timestamp
  Alignment") - not implemented. The M-RoPE mechanism above is real and
  fixes a real bug, but it is the superseded upstream design, not this
  model's own.
- `Clip`/the caps.rs `fps` param is ONE constant per clip, matching
  upstream's own `second_per_grid_ts` (also one scalar per video) and
  `captioner::Clip`'s existing shape - genuinely per-frame irregular
  timestamps (e.g. a variable frame rate source) are NOT expressible at the
  `caps.rs`/`Captioner` surface today, only at the lower `mrope::
  get_rope_index_video`/`Qwen3Vl::generate_video_timed` level, which both
  already accept an arbitrary `&[f32]`.
- No container/codec (mp4 etc.) decoding anywhere in this crate or this
  change - confirmed absent before scoping this task, per the task's own
  instruction to check rather than assume. The video input surface is
  pre-decoded frames + fps, the same shape `capability::blob::decode_video`
  (already used by `qwen3omnimoe::caps`) and `sam2`'s video path both take.
  MAX_VIDEO_FRAMES = 32 keeps this to short clips; no streaming, no
  hours-long video.
- No training-time (`forward`) video path - only `generate_*` (serving/
  captioning). A `forward_video` for gradient-checked video training is not
  built.
- No real-checkpoint parity number for the video path is claimed anywhere -
  only tiny-config synthetic-weight plumbing tests
  (`generate_video_is_deterministic_and_runs_end_to_end`) prove the wiring
  runs end to end, stays in vocab, and is deterministic. There is no HF
  reference to check real Qwen3-VL video generation against in this
  workspace, matching this crate's existing image-path caveat (see
  `docs/models/qwen3vl.md`).

## qwen3vl - tool-calling request/response contract, reused from qwen3::chat

`qwen3vl::caps::generate_spec()` gained `tools`/`tool_choice` (same names and
help text as `qwen3::caps`'s own params) so a client driving both models
never has to special-case VLM tool-calling. Nothing about tool-choice
parsing, enforcement or tool-call scanning was reimplemented:
`qwen3::chat::{parse_tools, parse_tool_choice, ToolChoice}` parse and
validate the request exactly like the text-only path (a `Named` choice
naming a function absent from `tools` is rejected before any weights are
touched - proven with a nonexistent weights path in
`caps::tests::tool_choice_named_function_must_exist_in_tools_before_touching_weights`),
and `qwen3::chat::SeqState` now drives the whole per-token decode loop that
used to be hand-rolled here (`stream_delta` inlined by hand, no tool-call
scanning at all). `SeqState::finish` is what resolves
`prompt_tokens`/`completion_tokens`/`finish_reason`/`reasoning_content`/
`tool_calls` - the identical `Outcome` shape `qwen3::caps`'s `GenerateAction`
returns.

One seam genuinely could not be reused: `qwen3::chat::parse_request` renders
one whole `messages`+`tools` prompt as a single string via
`data::qwen_chat::render_for_generation`, and this crate needs to splice a
mid-prompt run of image-placeholder tokens (`<|vision_start|>[IMG]*n
<|vision_end|>`) into token-id space, which a string renderer has no seam
for. The tools *preamble* (the `<tools>...</tools>` system block) is instead
rendered standalone - `data::qwen_chat::render(&[], tools, TemplateOpts {
add_generation_prompt: false, .. })` - the SAME renderer the shared path
calls, on an empty message list. This is NOT byte-for-byte what `qwen3::caps
generate` renders by default, though: `qwen3::chat::parse_request` resolves
`reasoning_effort` to `Some("xhigh")` whenever `enable_thinking` is true (its
own default), and that injects an extra directive paragraph into the
preamble. This action has no `enable_thinking`/`reasoning_effort` param, so
that paragraph is always omitted here - the `<tools>` block and surrounding
structure match, that one paragraph does not. `tool_schema_names`
(named-tool validation) moved from
private to `pub` in `qwen3::chat` for exactly this cross-crate reuse - it was
the one piece of `parse_request`'s internal enforcement not already exposed.

**Scope boundary, stated plainly**: this is the request/response CONTRACT
only - declare tools, enforce `tool_choice`, parse the model's
`<tool_call>` output into a structured `tool_calls` field. Real tool
EXECUTION (running the named function, feeding its result back as a `tool`
turn, looping) is not implemented for either `qwen3vl` or `qwen3` and is a
separate, larger piece of work. Also unverified: whether a real Qwen3-VL
checkpoint actually EMITS a well-formed `<tool_call>` for a given image/tool
pair - the real-weight test
(`caps::tests::served_generate_with_tools_enforces_tool_choice_on_real_weights`)
asserts the contract fires (a `finish_reason` is always resolved, `tools:
"none"` never demands a call) without asserting which `finish_reason` a
specific image/tools/max_new combination produces, since that is a property
of the checkpoint's own weights, not of this plumbing.

## qwen3vl - LoRA fine-tuning wired: forward+backward+AdamW over a captioned-image dataset

Before this session, `DecoderBuild::Batched` (`Qwen::new`, the full
trainable-parameter graph) had exactly one caller -
`model::tests::end_to_end_forward_is_finite` - which forwards once and checks
the loss is finite. Nothing ever called `backward()`, nothing stepped an
optimizer, and `Qwen3VlConfig`'s `text: QwenConfig` never set `lora`. This
session closes that: `crate::finetune::run` builds the decoder via
`Qwen3Vl::from_tensors_train` (`DecoderBuild::Batched` + `cfg.text.lora =
Some(LoraCfg{..})`), loops `zero_grads -> forward -> backward -> adamw_step`
over `data::imageset::load_dir`'s captioned-image folder format (the one
`brain label` already writes), and saves an adapter-only checkpoint via
`Qwen3Vl::save_lora_adapter` (a one-line wrapper over `qwen3::lora::save_adapter`
- no new save/fold logic, reuses `model::lora::device_adapter` end to end).
Exposed as the `lora_train` capability::Action on `qwen3vl::caps::QwenVlProvider`
(catalog.rs already registers this provider with `resident: None`, matching
`generate`'s own pre-existing state - see the "capability-action vs CLI-only"
note below), returning the trained bytes as an output blob per
`.agents/rules/serving-contract.md`'s "training actions return their artifact
as a blob" rule (pattern: `zimage::caps`'s `lora_train`).

**Only the decoder trains.** `Qwen3Vl::new`'s vision tower + PatchMerger(s)
are built from plain weight maps with no gradient buffers regardless of
`DecoderBuild` - `backward()` reaches only whatever the decoder's own
`ParamStore` role assignment marks Trainable (LoRA: `.lora_a`/`.lora_b` only).
Adapting the vision tower too would need `DecoderBuild` (or a sibling)
extended with a trainable vision-encoder graph, which does not exist. This is
the same scope `qwen3::finetune::Mode::Lora` has on the text-only model,
carried over unchanged - not a new limitation introduced here.

**Training images share one fixed size per run**, unlike `caps.rs`'s
per-request `smart_resize`. The batched decoder graph is built once at a
fixed `seq_len` and a fixed image-token placement (`image_row0`/`n_visual`),
so every sample must produce the same visual-token count;
`data::imageset::load_dir`'s own center-crop-then-resize-to-`size` already
guarantees that geometry, so `crate::finetune` just derives the patch grid
from the same `size` rather than adding a second restriction. A caption that
overflows the fixed `seq_len` token budget is skipped and named in the
progress stream, never silently truncated.

**Capability-action, not a CLI-only verb - and why.** `qwen3vl::caps` already
exposes `generate` as a `capability::Action`; adding `lora_train` the same
way costs nothing new (the same `Provider`/`Invocation`/`Outcome` plumbing
`catalog.rs` already wires to `brain do qwen3vl …` and `brain caps`) and
gives it the D-Bus/HTTP path for free the moment this crate gets a residency
adapter, whereas a bespoke `brain qwen3vl finetune` subcommand would be a
second entry point to keep in sync and would need its own future migration
off the CLI. No concrete reason was found for a CLI-only route on this crate
specifically - `catalog.rs`'s existing `qwen3vl` entry already carries
`resident: None` with the comment "no residency adapter yet - `brain
caps`/`brain do` only, matching fastvlm's own state", a PRE-EXISTING,
already-documented gap (AGENTS.md's model ledger, `13d.`/`13c-bis.`) that this
change does not touch, worsen, or pretend to close. `lora_train` inherits
exactly that state: reachable via `brain do qwen3vl lora_train …`/`brain
caps`, not yet over D-Bus - honest, not new.

**Gradcheck scope, stated precisely.** No new device kernel and no new
differentiable math were added - the decoder's LoRA forward/backward is
exactly the code `gradcheck::check_qwen_lora` already finite-difference-checks
on a bare `qwen3::Qwen`, the M-RoPE table path is `check_qwen_mrope`'s, and the
image-embedding splice gradient is `check_vlm_splice`'s. What is new is the
COMPOSITION - wiring `cfg.text.lora` through `Qwen3Vl::new` and driving
`Qwen3Vl::forward`/`backward` together for the first time - so
`crate::model`'s new `lora_delta_gradient_matches_finite_difference` test
hand-rolls an elementwise central-difference check (`gradcheck::
elementwise_check`'s own recipe, on a handful of `lora_a` entries of one
projection) through the FULL composite forward, not just the isolated
decoder. A whole-model `gradcheck` entry point for `Qwen3Vl` itself (a
`CheckModel` impl over the composite, added to `crates/gradcheck`) was NOT
built - `Qwen3Vl::forward` takes `(tokens, targets, grid, pixels)`, not the
`CheckModel::loss(&self)` no-argument shape the blanket `model::Model` impl
needs, so wiring it in would mean either changing that trait's contract or a
bespoke adapter, and the per-piece coverage above already exercises every
math primitive this loop composes. This is a real, explicitly-scoped gap, not
an oversight.

**A real finding from building `crate::train_smoke`'s convergence test, worth
recording because it cost real debugging time and would recur:** a
FROM-SCRATCH random tiny transformer plus a FROM-SCRATCH random low-rank
LoRA adapter, trained on arbitrary random targets, can converge to a real
(gradient-confirmed, non-zero, shrinking-towards-zero) LOCAL MINIMUM within
~100 steps that is nowhere near the achievable floor - repeatably, across
rank 4/8/16, learning rates spanning 1e-2 to 1e-1, and with/without gradient
clipping, all landing on materially the same plateau. This is NOT the same
bug class as `check_qwen_lora`'s own doc note ("a few AdamW steps run first
so the zero-initialised `B` adapter... is non-trivial before the FD
comparison") - that note is about `A`'s gradient being exactly zero at
`B = 0` (true here too, confirmed by direct grad-norm dumps), not about a
stall AFTER `B` moves. The same architecture under FULL (non-LoRA) training
collapses the identical single-example loss to ~0 within ~100 steps -
confirmed directly while diagnosing this, not asserted by any checked-in
test, since it is not what `crate::finetune` ships - which rules out a
forward/backward wiring bug in `Qwen3Vl`/`crate::finetune` (the SAME composite
forward is exercised both ways) and points at LoRA's own `dA ∝ B` coupling
(state in `lora_fwd`/`proj_bwd`, `crates/qwen3/src/model.rs`) interacting
badly with a from-scratch random init and a frozen, TIED (`tie_embeddings:
true`) `d_model < vocab` embedding/head, which caps the reachable logit
subspace at rank `d_model` regardless of how much the projections' LoRA
adapts. `crate::train_smoke`'s convergence test therefore asserts a real but
modest bound (loss must fall below 90% of its start, cycling three distinct
image/caption pairs) rather than an overfit-to-near-zero claim like
`fastvlm::train_smoke`'s single-example test makes - that stronger claim
would be honest for a REAL pretrained checkpoint (where the base
representations are already well-organized, unlike random init) but is not
what this synthetic smoke test's own architecture can support, and asserting
it here would make the test seed-lucky rather than meaningful. Anyone porting
this convergence-smoke pattern to a different tiny synthetic LoRA test should
budget for the same investigation rather than assuming a stuck loss is
automatically a wiring bug.
