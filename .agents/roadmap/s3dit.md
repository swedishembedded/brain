# s3dit - roadmap

Z-Image (S³-DiT) text-to-image diffusion transformer, with fp32/int8/sharded
device engines, a VAE, a flow-matching scheduler, backward + LoRA fine-tuning,
and tiered weight-residency streaming so the checkpoint never has to be
loaded whole. Forward and backward parity against the reference are verified.

## Not yet done

- [ ] A true batched `run_batch` for the serving contract
- [ ] A runnable examples client over D-Bus
- [ ] Native lower-precision (bf16) device weight binding for the windowed
      fp32 path - fp32 inference currently streams full fp32 weight tiles
      per block, which is disk-bound rather than compute-bound
- [ ] Device-resident block chaining, so the reference path doesn't
      round-trip through host memory between blocks
- [ ] Unify the flow-matching dynamic-shift calculation with the shared
      implementation used elsewhere
- [ ] Parity coverage for short (padded) prompts against the reference

The checkpoint is far larger than fits in memory at once, so weights stream
from disk one tensor at a time rather than the model loading whole; until a
native lower-precision device format exists, this makes fp32 inference
disk-bound rather than compute-bound.

## Fixed: output corrupted at larger resolutions (int8 path, int8-quantized DiT)

`brain s3dit text2image` against the real `Tongyi-MAI/Z-Image-Turbo`
checkpoint (default `--precision int8`, the served default) used to produce
progressively corrupted output as `--width`/`--height` grew past 256px on
identical weights/prompt/seed/steps - a correct subject only near the center
of the frame at 512px, and pure noise everywhere by 768px, worsening outward
from the center as resolution grew.

**Root cause**: `crates/model/src/hostmath.rs`'s shared `timestep_embedding`
helper takes a `flip_sin_to_cos` parameter controlling `[cos‖sin]` vs.
`[sin‖cos]` half-ordering in the sinusoidal embedding; `crates/s3dit/src/
model.rs`'s wrapper was passing `false` when this model's real (diffusers)
reference requires `true`. Confirmed by a from-scratch real-checkpoint
diffusers parity dump at full 512×512/1088-token scale
(`tools/goldens/s3dit_real_512_dump_reference.py`,
`crates/s3dit/tests/real_parity.rs`'s `zimage_real_dit_matches_diffusers_at_512`)
- cosine similarity against the Python reference went from clearly-wrong to a
perfect `1.000000` once the flag flipped. Not a RoPE/axes-length bug (the
`axes_lens: [1536,512,512]` correction that accompanied this fix was a
separate, smaller correctness fix caught by the same parity harness, not the
root cause of the resolution corruption itself).

Verified end to end with real, clean 512×512 renders
(`docs/quickstart/img/seed.png`) - no center-only coherence, no speckle
noise, no visible block-grid artifact (the earlier "plausible per-token
int8-quantization" hypothesis for a block texture at 256px did not
independently reproduce once this fix landed; not investigated further since
there is nothing left to see).

**This class of bug can recur.** `timestep_embedding`'s doc comment now
carries an explicit warning that `flip_sin_to_cos`/`downscale_freq_shift`
must be checked against each model's own real reference rather than copied
from a sibling call site - the two diffusion models in this codebase
(`s3dit`, and any future one built on the same shared helper) do not
necessarily agree on this convention just because they share the function.

## Fixed: fp32 ("hifi") text2image OOM building the VAE decoder

`brain s3dit text2image --precision fp32` used to panic with `wgpu error: Out
of Memory` while building the VAE decoder, 100% reproducible, immediately
after the fp32 DiT finished sharding across both GPUs. Two real, distinct
bugs, both fixed together: (1) `--device gpu0`/`--device gpu1` was not
restricting `hifi_needs_window`'s GPU-count decision, so a single-GPU request
still tried to shard the DiT across both cards and never took the
narrower-footprint windowed path meant for exactly this case; (2) the VAE
decoder was unconditionally placed on-GPU on top of whatever the DiT shard
already used, rather than falling back to a CPU-JIT decode when the fp32
shard already fills the card. Fixed: `--device` now actually restricts the
GPU count `hifi_needs_window` sees, and the VAE decoder runs on CPU when the
fp32 shard leaves no room for it.

## Fixed: inpaint - no visible edit at demo settings

`brain s3dit inpaint` used to complete cleanly and write a real,
byte-different PNG, but the masked region was visually unchanged - the
original content (e.g. an apple) survived instead of being replaced by the
prompt's requested edit. Root cause: the masked region was being seeded from
a blend that still leaked the original latent's signal into the denoising
trajectory, instead of pure noise, so the "edit" mostly re-derived the
existing content. Fixed and verified with a real, visibly-correct edit
(`docs/quickstart/img/inpainted.png` - the apple in `seed.png` is genuinely
replaced by a slice of chocolate cake, everything else held fixed).

## First-dispatch GPU wait bound too short on a cold Pascal card

`brain s3dit text2image` against the real `Tongyi-MAI/Z-Image-Turbo`
checkpoint panicked on its first attempt at `sampling [2/10]`: `buffer read:
GPU submit did not complete within 30s (BRAIN_GPU_WAIT_S) -- device likely
wedged` (the documented `BRAIN_GPU_WAIT_S` panic from `.agents/rules/
lessons.md` #38, working as designed - it turned what used to be a silent
hang into a clear, attributable error). `nvidia-smi` showed the card fully
idle and healthy immediately after, so nothing was actually wedged: the
first-ever heavy dispatch against a freshly-built 13 GB int8 DiT on a Tesla
P40 (an old Pascal card, JIT/pipeline warm-up plus whatever one-time driver
state the wgpu backend sets up on its first large submit) genuinely took
longer than 30s. Retried with `BRAIN_GPU_WAIT_S=300` and it completed
cleanly end to end (real image, `docs/quickstart/img/seed.png`). Worth a
default review for this card generation - 30s as the FIRST-submit bound (as
opposed to steady-state) is tight for a decade-old GPU with a large model.

## Q8_0 GGUF import - implemented (DiT only), not yet wired into auto-fetch

`s3dit::import::import_gguf` (called from `brain import <gguf-file>`, registered
alongside `Qwen35MoeImporter` in `crates/cli/src/gguf_import.rs`) converts
`unsloth/Z-Image-Turbo-GGUF`'s `z-image-turbo-Q8_0.gguf` into a brain-native
single-file safetensors checkpoint `BRAIN_S3DIT_DIT` can point at directly -
eager dequant of every tensor (no streaming; the whole DiT fits comfortably in
host RAM), through the exact same `import_comfy` remap the safetensors path
already uses, since the GGUF's tensor names are unchanged from the original/
Comfy layout. Guarded against `general.architecture = "lumina2"` being shared
with real Lumina2 releases (see [below](#the-lumina2-discriminator)) by
requiring a Z-Image-only tensor (`cap_embedder.0.weight`) before converting
anything.

**Not wired into `brain s3dit text2image`'s own auto-fetch.** The GGUF release
is DiT-only (7.2 GB vs the safetensors checkpoint's ~25 GB for that one
component) - the VAE and Qwen-4B text encoder still have to come from
`Tongyi-MAI/Z-Image-Turbo` itself, a **different upstream repo**. brain's
fetch plan (`brain_modelstore::plan`) is built around ONE `ModelRef` → one
repo's file listing → one `Plan`; blending a small GGUF from one repo with
safetensors components from another inside that same machinery is a real
design decision (a second repo reference per role? a recipe that itself
issues a second HTTP fetch?), not a follow-on line. Auto-fetch (`default_ref`
+ `weights_env`) still pulls the full ~33 GB safetensors checkpoint from one
repo - a genuine one-liner today, just a larger download than the GGUF path
would be. Wiring the smaller path in is tracked here, not done.

### The lumina2 discriminator

Both of brain's GGUF importer registries handle a shared/ambiguous
architecture value with a **discriminator** in the general case
(`crates/gguf/src/registry.rs`'s `ArchEntry.discriminator`, built for the
DeepSeek-OCR vision GGUF sharing `clip` with every other mtmd projector, told
apart by `clip.projector_type`) - but `crates/cli/src/gguf_import.rs`'s
`GgufArchitectureImporter`/`IMPORTERS` (the simpler registry `Qwen35MoeImporter`
and now `S3ditImporter` share) has no discriminator field on the trait itself.
Rather than extend that trait for one architecture, `S3ditImporter::import`
carries its own guard inline: it refuses to proceed unless the GGUF has
`cap_embedder.0.weight` (Z-Image-only, absent from a real Lumina2 checkpoint),
failing with a clear message rather than silently misconverting. A real
`GgufArchitectureImporter`-level discriminator (matching `crates/gguf`'s
pattern) would be the more general fix if a second `lumina2`-tagged
architecture ever needs importing.

## Original investigation notes

`unsloth/Z-Image-Turbo-GGUF`'s `z-image-turbo-Q8_0.gguf` (7.2 GB vs the
diffusers safetensors checkpoint's 33 GB) was checked for feasibility:
- Its tensor names are **byte-identical** to the "Comfy/original" layout
  `s3dit::import::import_comfy` already remaps (`layers.N.attention.qkv.weight`,
  `context_refiner.N.*`, `noise_refiner.N.*`, `t_embedder.mlp.*`,
  `x_embedder.*`, `cap_embedder.*`, `final_layer.*`) - so the remap itself is
  a same-day job, not a port, reusing `import_comfy` as-is.
- It is **DiT-only** (390 `layers.*` + refiners/embedders, no `vae.*`/
  `text_encoder.*` prefixes) - the VAE and the Qwen3-4B text encoder still
  come from elsewhere; brain's existing dequant-to-f32 GGUF reader
  (`checkpoint::gguf`, Q8_0 support already in) covers the tensor math.
- Two tensors have no counterpart in brain's current graph: `cap_pad_token`
  and `x_pad_token` ([3840,1] f32) - brain masks padding via the attention
  mask rather than a learned pad embedding, so these would be imported and
  unused, not a blocker.

**What stopped it landing this pass**: the file's `general.architecture` KV
is `"lumina2"` - Z-Image is architecturally Lumina2-adjacent and unsloth
reused that tag, but it is not a Z-Image-specific spelling. Both of brain's
GGUF importer registries handle a shared/ambiguous architecture value with a
**discriminator** (`crates/gguf/src/registry.rs`'s `ArchEntry.discriminator`,
built for exactly this - the DeepSeek-OCR vision GGUF shares `clip` with
every other mtmd projector and is told apart by `clip.projector_type`), but
`crates/cli/src/gguf_import.rs`'s `GgufArchitectureImporter`/`IMPORTERS` (the
registry `Qwen35MoeImporter` uses, and the one an `s3dit` importer would
naturally join) has **no discriminator mechanism at all** - registering under
bare `architecture() == "lumina2"` would silently misroute a real Lumina2
GGUF into Z-Image's tensor remap, and the mtmd/DeepSeek-OCR precedent in this
same codebase says not to do that. Adding a discriminator to that trait is
the right fix but is its own small design decision (does it key on a KV pair
the way `crates/gguf` does, or on a tensor-name signature since `lumina2` has
no natural discriminating KV field?) rather than "one line in `IMPORTERS`".

Deferred rather than rushed. The README's Quick start currently fetches
Z-Image via the existing `Tongyi-MAI/Z-Image-Turbo` safetensors `default_ref`
(33 GB) instead of the Q8 GGUF every other demo model uses - noted there as
the one exception, not silently presented as uniform.
