# MiniMax-H3: audit of the pipeline glue around the DiT

## Why this pass happened

A real generation (128x128 canvas, 124 frames, 20 real denoising steps, real
weights throughout, via `crates/minimaxh3/examples/generate_t2va.rs`) produced
frames showing clean, regular, high-contrast diagonal-stripe / checkerboard
patterns - not noise, not a blurry scene, but confident-looking structured
garbage.

`model::tests::dit_matches_the_real_reference_numerically_layer_by_layer`
already proved the DiT's own block math against the real checkpoint (worst
cosine 0.9999998613), but it *hand-builds* its packed sequence. Everything
that ASSEMBLES that sequence was therefore invisible to it. This pass audited
exactly that surrounding glue.

The working theory going in was that 128x128 is simply too coarse to ever show
a scene, so there might be no bug at all. That theory turns out to be **right
about the symptom and wrong about being the whole story**: there is a real,
independent bug on the generation path, and there is also a concrete mechanism
by which 128x128 produces geometric patterns rather than blur.

## The answer

**Both.** Two real defects were found, neither of which is the resolution
theory, plus a quantified mechanism that explains the specific artifact.

1. **Real bug, fixed:** the generation path fed the VAEs placeholder latent
   normalization (`mean=0`/`std=1`) instead of the checkpoint's real
   per-channel statistics.
2. **Real gap, already documented, made loud:** the video VAE's spatial tiling
   is not implemented. It costs nothing at or below 256 pixels - so it is *not*
   the cause of the 128x128 symptom - but it makes any larger canvas
   structurally wrong, and it used to overflow a `u32` silently.
3. **Not a bug:** at 128x128 the DiT's spatial rotary grid is ~8x coarser than
   anything the model was trained on, and 2 of its 16 spatial rotary
   frequencies alias between adjacent tokens. That produces regular geometric
   interference, which is what was observed.

## 1. Real bug, fixed: placeholder VAE latent normalization

`caps::VaeWeights::load` set

```rust
video_latents_mean: vec![0.0; latent_channels],
video_latents_std:  vec![1.0; latent_channels],
```

and the same for audio, with a doc comment claiming the real per-channel values
"are not available from this port's current checkpoint download". That claim
was stale: they are in the checkpoint, in `vae/config.json` and
`audio_vae/config.json`.

This is on the real generation path - `generate_t2va.rs` builds its checkpoint
through `caps::LoadedWeights::load`.

Why it matters more than a stale comment suggests: `mean=0`/`std=1` is the
**identity** of the affine transform, so this did not approximate the real
normalization, it deleted it. The reference normalizes on the encode side
(`encoders.py:136`, `(latents - mean) / std`) and denormalizes on the decode
side (`decoders.py:185`, `latents * std + mean`), so the diffusion model works
in a whitened latent space and the VAE decoder never sees anything but the raw
one. With the placeholders the decoder was handed a whitened latent directly.

The real spread is nowhere near the identity, on both sides:

| | channels | `latents_mean` range | `latents_std` range |
|---|---|---|---|
| `vae/config.json` (video) | 24 | -1.368 .. 1.066 | 0.450 .. 3.276 |
| `audio_vae/config.json` | 32 | -0.370 .. 0.591 | 1.492 .. 3.299 |

Note the audio side was affected too, and arguably worse in one respect: every
audio `latents_std` is greater than 1.49, so with `std = 1` the vocoder was fed
latents uniformly 1.5x to 3.3x too small in amplitude on every channel.

### Measured effect

A probe (`h3_denorm_probe.py`, scratch) encoded a real image through the REAL
`AutoencoderKLMiniMaxH3` at 128x128 / 22 frames, normalized the latent exactly
as the pipeline does, then decoded it both ways:

| | correct (`z*std+mean`) | placeholder (`z`) |
|---|---|---|
| reconstruction | near-perfect: the disc, both gradients, correct colour | washed-out colour, visible 16px patch blockiness |
| luma std | 0.052 | 0.046 |
| mean abs d/dx | 0.0023 | 0.0050 (2.2x) |
| high-frequency spectral share | 67.5% | 76.4% |

So it is a genuine, measurable degradation - but the output is still
scene-shaped. **The missing denormalization alone does not produce diagonal
stripes.** It is a real bug that was making every frame worse; it is not the
whole explanation of the reported artifact.

That same probe is also the strongest single piece of evidence against a
VAE-level resolution floor: at 128x128 the real VAE reconstructs a clearly
recognizable scene from an 8x8 latent. The decoder is not the bottleneck.

### The fix

`caps::read_latent_stats` reads `latents_mean`/`latents_std` from each VAE
component's own `config.json`, validates the length against the latent channel
count and that every std is finite and positive. **A missing or malformed
config is a hard error, not a fallback** - a fallback here is precisely the
failure mode that caused this, since nothing fails when the identity is wrong.

Regression test:
`caps::tests::vae_latent_normalization_is_read_from_the_checkpoint_and_is_not_the_identity`
asserts both halves of the bug - that the values are read, and that they are
not the identity that made reading them look unnecessary.

## 2. Real gap: video VAE spatial tiling (not the 128x128 cause)

The reference ships `use_tiling = True` (`autoencoder_kl_minimax_h3.py:607`)
with 256x256-pixel tiles. `_decode_clip` lays tiles over the canvas, maps each
to a **16x16 latent** window, runs the ViT decoder once per tile and blends the
overlaps - so the reference's ViT decoder never sees anything but a 16x16
spatial token grid. The port implements only the `use_tiling=False` path.

This was already an honestly recorded scope cut in `video_vae.rs`'s module
doc. What was *not* recorded is how it fails, which is not gracefully:

- The decoder's rotary coordinates are normalized to `[-1, 1)` across whatever
  grid it is handed. At the released 768x1344 canvas the untiled grid is 48x84
  instead of 16x16, so every positional relationship the 36 attention layers
  were trained on is stretched by 3x in h and 5.25x in w, while the per-token
  `proj_out` still paints a hard 16x16 pixel block. That is a regular patch
  grid with long-wavelength beating in both axes - i.e. it *is* shaped like the
  reported artifact, just not at the reported resolution.
- The sequence grows from 1797 to 28229 tokens.

**Crucially this does not explain the 128x128 run.** `_split_tiles` begins

```python
if tile_size >= length:
    return [0], [length], []
```

so at any canvas of 256 pixels or less there is exactly one full-size tile and
the tiled and untiled paths are the same computation. The gap costs nothing at
128x128 - and nothing at 256x256 either.

### What was changed

Not implemented (it is a substantial piece of work and belongs in its own
pass), but the silent failure was removed. `video_vae::attention` computed
`heads * seq_len * seq_len` in `u32` before casting to `u64`, which wraps past
`seq_len` 11585 at the real config's 32 heads - the untiled 768x1344 decode
gives 28229, so `32 * 28229^2 = 2.55e10` wrapped to 4.03e9 and produced a
wrong-sized allocation and a wrong dispatch count. It is now computed in `u64`
and asserted, with a message naming the tiling gap as the cause. The module doc
now states the above consequence concretely.

## 3. Not a bug: the 128x128 resolution floor, with a mechanism

The task asked for a mechanism by which a well-converged rectified-flow model
at this few tokens produces *regular geometric patterns* rather than a smooth
blur, or an explicit statement that no such mechanism was found. There is one,
and it is quantitative.

MiniMax-H3's spatial rotary grid is aspect-normalized and scaled by a fixed
`ROPE_SPATIAL_SCALE = 32`, so for a square canvas the rotary distance between
two spatially adjacent DiT tokens is exactly `32 * patch / (W / vae_ratio)` =
**1024 / W**. The scale is fixed, so the token grid gets rotationally *coarser*
as the canvas shrinks:

| canvas | tokens/frame | rotary step between adjacent tokens | vs released canvas |
|---|---|---|---|
| 768x1344 (released) | 1008 | 1.008 | 1.00x |
| 1024x1024 | 1024 | 1.000 | 0.99x |
| 512x512 | 256 | 2.000 | 1.98x |
| 256x256 | 64 | 4.000 | 3.97x |
| **128x128** | **16** | **8.000** | **7.94x** |

With `rope_freq_dim = 16` and `rope_theta = 10000`, the per-frequency phase
advance between two adjacent tokens is `step * theta^(-k/16)`. A relative
position code carries information only while that advance stays below `pi`;
past `pi` it wraps and two different offsets become indistinguishable.

| | frequencies aliasing between adjacent tokens |
|---|---|
| released 768x1344 canvas | **0 of 16** |
| 128x128 | **2 of 16** |

At 128x128 the two fastest spatial frequencies advance by 8.00 and 4.50 radians
per token - i.e. **more than a full turn between neighbours**, a spatial period
of 0.79 and 1.40 tokens across a frame that is only 4 tokens wide.

That is the mechanism. Sub-token-period rotary phases beat against the integer
token lattice, and because h and w are stretched by the identical factor their
beat patterns superimpose - which is a diagonal interference pattern, not a
blur. The model is not under-resolved into vagueness; it is being asked to read
positional codes that alias, and it confidently reconstructs the aliasing.

This is **not a port defect, and that is proven rather than argued**. The new
parity golden is dumped at exactly the failing geometry (8x8 latents = a
128x128 canvas), and `position_ids` matched the reference bit-exactly there -
cosine 1.0000000000, max_abs 0.0, over all 3129 values. So the port handed the
DiT precisely the rotary coordinates the reference would have handed it. The
aliasing above is what the reference itself would do at this canvas; it is
inherent to running the model ~8x below the spatial token density it was
trained at, not to this implementation of it. The released pipeline defaults to
a 768 short edge for exactly this reason.

Practical consequence: a square canvas needs to be near **1024 px** for the
spatial rotary density to match training. 128x128 cannot produce a recognizable
scene no matter how correct the port is.

## What was verified, item by item

New numeric gate:
`pipeline::tests::pipeline_layout_matches_the_real_reference_numerically`,
fed by `tools/minimaxh3_layout_dump_reference.py`. It replays the REAL installed
`diffusers==0.40.0` `build_packed_sequence` / `build_row_timesteps` /
`MiniMaxH3Scheduler` / `patchify_video_latents` outputs, dumped at the geometry
an actual generation runs (8x8 latents, 37 latent frames, 207 audio latents,
20 steps - not a toy: 37 frames wraps the `(1,4,4,4,4)` rotary spacing seven
times), for both the `t2va` and the `first`+`last`-anchored `fl2va` layouts.
Result: **11 taps, worst cosine 1.0000000000.**

| Item | Verdict |
|---|---|
| Audio position-id placement (`_fill_audio_positions`) | **MATCHED**, now numerically. Channel-major tiling, `t = num_text_tokens + arange`, `w` pinned to `width_grid[0]` for channel 0 and `width_grid[-1]` for the rest, no `h` coordinate. |
| `build_packed_sequence` full row ordering / index building | **MATCHED**, now numerically at real geometry. `position_ids` bit-exact (max_abs 0.0, n=3129 t2va / 3225 fl2va); `token_tags`, all three index arrays and both condition-row counts exactly equal. |
| `schedule.rs` dual rectified-flow Euler step | **MATCHED**. Sigma grid, shift, `t = 1 - sigma`, the data-ward `+` velocity sign, and the deliberate two-source sigma split (grid for the Euler ratio, `1 - timestep` for the `x0` recovery) all match; sigmas/timesteps/Euler step at float32 noise (max_abs <= 2.4e-7). `shift = 12.0` / `3.0` confirmed against the checkpoint's own `scheduler/` and `audio_scheduler/` configs. Note the `unique_consecutive` dedup is unreachable in practice at both shifts - the float32 grid stays strictly decreasing at every step count up to 5000 - so it is ported but never fires. |
| Per-step row-timestep plan (`build_row_timesteps`) | **MATCHED**. All 19 steps, both layouts, exact per-row indices. Includes the detail that step 0 has only ONE distinct timestep (both shifts fix sigma=1, so video and audio coincide at t=0) where every later step has two. |
| DiT-level patchify / unpatchify | **MATCHED**, bit-exact at real shape (n=56832), and `unpatchify` checked against the reference's own reshape/permute rather than only against its own forward direction. |
| VAE latent normalization | **MISMATCHED - FIXED.** See section 1. Sign and order were correct; the values were placeholders. |
| Video VAE ViT decoder token->pixel reassembly | **MATCHED.** Tokenize order, register + zero cls token placement, stripping after `proj_out`, and the `(oc, dt, dh, dw)` feature nesting all verified element-for-element. |
| Video VAE decoder RoPE | **MATCHED.** `inv_freq` ladder, axis concatenation order, `tile(2)` half-split pairing at distance 24, the `2*pi` factor, and the unrotated tail channels (`rope_dim_ratio = 0.75`). |
| Video VAE temporal chunking (`clip_length=17`, `token_drop=3`) | **MATCHED.** `num_chunks` formula, sub-chunk split, 5-frame cross-fade, `pad_frames` tail trim - traced through a real 124-frame / 37-latent-frame decode. |
| Video VAE spatial tiling | **MISMATCHED - NOT the 128x128 cause.** See section 2. Not implemented; overflow made loud, doc corrected. |
| Video VAE GroupNorm / eps / reflect + causal padding | **MATCHED.** (Also: none of it is on the ViT decode path.) |
| `attention()` `u32` score-count arithmetic | **MISMATCHED - FIXED** (widened to `u64` + assert). Only reachable via the untiled path at a large canvas. |
| Row scatter/gather glue (`index_copy` order, `adaln_indices`) | **MATCHED** by source read; `text`, `video`, `audio` scatter order matches, and `position_ids` is cast to f32 on both sides before RoPE. |

## Not reached

- **`ref2va`** layout (`build_ref2va_packed_sequence`) - deliberately not
  implemented in this port, so nothing to compare.
- **Text conditioning** (`get_qwen3vl_prompt_embeds`) - built by the caller;
  the Qwen3-VL splice gap is separately documented and was out of scope here.
- **Audio VAE / vocoder decode internals** - only the latent normalization and
  the channel-major transpose feeding it were checked (both match).
- **A real-weight numeric comparison of this port's OWN video VAE decode
  against the reference's.** The reference decoder was run at 128x128 (it
  reconstructs a recognizable scene from an 8x8 latent - see section 1), and
  this port's decoder math is covered by the existing tiny-config golden, but
  the two were not compared at real weights and real geometry. That gap is
  worth closing along with the tiling work in section 2, since the tiny golden
  cannot see either.
- **An end-to-end real-weight generation at a corrected canvas.** The
  conclusions above are from component-level real-weight numerics plus the
  quantified rotary-density argument. A ~1024x1024 run with the latent-stats
  fix would be the natural confirmation, and needs the spatial tiling from
  section 2 first, since 1024 > 256.

## Recommended next steps, in order

1. Implement the video VAE's spatial tiling (`_split_tiles` / `_blend` /
   `_stitch_tiles`, tiled `_decode_clip` / `_encode_clip`). It is the blocker
   for generating at any useful canvas, and the current tiny-config golden
   cannot catch it - that golden was dumped with `use_tiling=False`, and
   `tiny()`'s `decoder_attention_head_dim: 8` gives one RoPE frequency per
   axis, where `theta^0 = 1` hides any frequency-ladder error too.
2. Re-run the real generation at 768x1344 (or at least 1024x1024) with the
   latent-stats fix and tiling in place. Do not judge port correctness from a
   128x128 result - section 3 shows that resolution cannot work.
3. Consider refusing, or at least warning on, a canvas whose spatial rotary
   step is far from the trained ~1.0. A silent 8x-off positional grid is a
   large foot-gun that costs a full generation run to discover.
