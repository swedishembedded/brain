# wan - roadmap

Wan2.1/2.2 video diffusion: a DiT denoising a 3D `(frame, height, width)` latent
volume under flow matching, with per-block text cross-attention from umT5-XXL and
a causal 3D VAE at (4, 8, 8) stride. `wan` names the family, not the release --
2.1 and 2.2 share one architecture, one HF class and one GGUF tag, so the release
is a `WanConfig` variant.

The port is following `.agents/rules/porting.md` in order. Reference material
(official repo = math authority, diffusers = tensor naming, ComfyUI = third
opinion, ComfyUI-GGUF = arch detection) is cloned under
`scratchpad/reference/wan/`, with the 16 background papers in
`scratchpad/wan-papers/` and settled convention questions written up in
`scratchpad/wan-notes/`.

## Not yet done

- [x] **Causal `conv3d` kernels** -- `conv3d`, `conv3d_dx`, `conv3d_dw` landed as
      direct kernels rather than an `im2col3d` + `matmul_reg3` lowering: a 3D
      im2col operand is the classic way to blow past
      `max_storage_buffer_binding_size` (2047 MiB on the P40 this was written
      on), and the direct form binds only the natural tensors, so the fallback
      split is over N. Per-axis kernel extent, per-axis stride, symmetric
      spatial pad, one-sided temporal low-pad `pt` (already doubled, as in
      `dwconv3d`), bias, and groups. Gate: `crates/gradcheck/tests/conv3d_kernels.rs`,
      including two structural causality probes -- an output frame may not move
      when a future input frame does, and `_dx` may not push gradient forward in
      time. Both were mutation-verified against a symmetric-pad kernel.
- [x] **Wan-VAE** as a sibling 3D builder in `crates/vae/src/blocks3d.rs`, not a
      widening of `blocks.rs` -- widening every `(prefix, c, h, w, x)` signature
      would destabilise five existing consumers (AutoencoderKL, VQGAN,
      CodeFormer, RRDBNet, SDXL-UNet) to no benefit. The Wan *schedule* (encoder,
      decoder, tensor names, chunked driver) sits in `crates/wan/src/vae3d.rs`,
      the same split `crates/vae`'s own `decoder.rs` has against `blocks.rs`.
      **Zero new kernels**: every op is an existing one at 3D `Params`, including
      the three that looked like they needed new ones --
      * a per-frame 2D conv, a `(3,1,1)` temporal conv and a `(1,1,1)`
        projection are all `conv3d` at different extents, so the whole model
        dispatches exactly ONE convolution kernel;
      * time-axis slice / place / concat are the channel-axis kernels
        (`concat_split`, `chan_place`, `concat2`) on the `[N=C, T, H, W]` view;
      * `upsample3d`'s channel-to-time fold (`[2C,T] -> [C,2T]` interleaved) is
        two `concat_split`s plus a `concat2` over the `[N=C*T, 1, H, W]` view.
- [x] **`feat_cache`**, the causal VAE's cross-chunk state (`CACHE_T = 2`). All
      three `upsample3d` states reproduced (`None` -> no time_conv and mark the
      slot; `'Rep'` -> time_conv against an all-zero history, with the zero frame
      a REAL operand of the next chunk; a cached tensor -> the ordinary path).
      Because the whole clip is recorded as one graph before a single submit,
      the cache is an SSA buffer flowing between chunk sub-graphs -- no device
      state, no readback, one submit per clip.
- [x] **The chunked-vs-unchunked gate, and what it caught.** Two independent
      forms of it, in the order they were built:
      * `tools/goldens/wan_vae_dump_reference.py` carries a whole-clip
        formulation of a model upstream only ever runs chunked (with
        `feat_cache=None` the `Resample` blocks silently skip their temporal conv
        entirely, so there IS no upstream whole-clip mode), and asserts the two
        agree before writing a byte. Derivation and both special cases are in
        that file's header. They agree to within 3e-6 relative across every
        stage tap at both clip lengths (fp32 reassociation, nothing else).
      * **Encode is chunk-size invariant; decode is not.** The encoder's
        `downsample3d` keeps one frame of history and consumes stride-2 windows
        at even positions, so (1,4,4) and (1,8) are the same computation, per
        output element in the same summation order -- brain asserts it
        **bit-exactly**, weight-free, at toy dims, as the first test in the file.
        The decoder's `'Rep'` state breaks the property (a 2-frame first chunk
        zero-fills two history slots where two 1-frame chunks fill one), so
        `WanVaeDecoder` hardcodes upstream's one-latent-frame chunking instead of
        offering a knob that would be quietly wrong.
      That bit-exact test found a real defect on its first run: the attention
      block's `nchw_nlc`/`nlc_nchw` permutes were given `H*W` where the operand
      is `[C, T, H, W]` and the argument means "everything below the channel
      axis", i.e. `T*H*W`. **The two are identical at `T == 1`, and every chunk
      of upstream's own encode and decode reaches the middle attention with
      exactly one frame** -- so the golden could not see it at any clip length,
      and neither could a real generation. Pre-fix the invariance test read
      max_abs 1.1e-2 / cosine 0.99962; post-fix, exactly 0.0.

      Parity reached (`crates/wan/tests/vae_parity.rs`, 9 and 17 frames at
      64x64, Vulkan on a P40 and the CPU JIT, same numbers on both): every
      boundary at **cosine 1.000000**, rel_l2 6e-7..1.4e-6, max_abs 4.8e-7 on
      `z_denorm`, 3.5e-6 on the reconstruction, 3.2e-5 on the deepest encoder
      tap. Encode, decode against the chunked reference, decode against the
      independent unchunked reference, and the composed round trip all agree.
      Perf is deliberately NOT addressed: every conv is the direct `conv3d`
      kernel, and the per-frame spatial convs that could take `blocks.rs`'s
      `im2col_at` + `matmul_reg3` lowering are a minority of the FLOPs, so the
      lowering is a later change with its own measurement.
- [ ] **umT5-XXL** in `crates/t5encoder`: vocab 32128 -> 256384, per-block
      relative position bias (see below), attention masking with a zero-pad to
      `text_len = 512`. The crate's own size analysis understates umT5 by about
      3.7 GB because it assumes the T5 v1.1 vocabulary.
- [ ] **SentencePiece unigram tokenizer** in `crates/data` -- brain has GPT-2,
      Qwen3 and CLIP BPE, but no unigram implementation at all, and umT5's
      256k-entry `spiece.model` needs one.
- [ ] **Goldens** (`tools/goldens/wan_dump_reference.py`) before any DiT Rust,
      per porting.md section 1.
- [x] **UniPC multistep in the flow-matching parameterisation**, plus the sigma
      shift `s' = shift*s / (1 + (shift-1)*s)`. Landed as
      `crates/diffusion/src/flowsolvers.rs` (a third scheduler family beside
      `scheduler.rs` and `discrete.rs`): brain's `DpmSolverPlusPlusScheduler` is
      built on `alphas_cumprod` and cannot be pointed at flow-match sigmas, so
      nothing was bent to fit. `flow_shift` sits next to
      `time_shift_exponential` in `scheduler.rs` with the contrast written down,
      because the two shifts are easy to mistake for each other and picking the
      wrong one changes every sigma silently. **The DPM++ flow variant landed
      too** rather than being deferred: once the flow `(alpha, sigma)` pair and
      the schedule plumbing existed it was ~70 lines, and it is the only way to
      prove that the two solvers do *not* share a schedule (see below).
      Gate: `crates/diffusion/tests/wan_schedule_parity.rs` against
      `tools/goldens/wan_schedule_dump_reference.py` (no weights needed - it
      imports the two scheduler classes only). Sigmas and timesteps are
      **bit-exact** over 16 (solver, shift, steps) combinations; the `step()`
      trajectory agrees to 5.2e-6 max abs over 50 steps, which is f32 rounding
      accumulating (the reference does its scalar coefficients in f32, brain in
      f64), mutation-checked against a wrong order (2.8e-1), a skipped
      corrector (7.5e-2) and an off-by-one in the corrector history (inf).
      Two facts worth keeping: the reference is constructed with `shift=1` and
      given the real shift at `set_timesteps` (applying it in both places
      squares it), and **the two solvers start at different sigmas** - UniPC at
      the training grid's top `1 - 1/1000 = 0.999` (first timestep 999), dpm++
      at exactly 1.0 (first timestep 1000), because `get_sampling_sigmas`
      builds its own `linspace(1, 0, N+1)`.
- [ ] **`crates/wan`** proper -- the DiT itself. `config.rs`, `vae3d.rs` and
      `import.rs` (VAE, both name spaces, two-way validated) are in; `model.rs`,
      the pipeline and the CLI wiring are not.
- [ ] **`capability::Media::Video`**. There is no video media type, and
      `.agents/rules/serving-contract.md` section 4 requires extending `Media`
      and the D-Bus frame handling rather than adding a side channel. This is a
      breaking enum change across `capability`, `dbus`, `apiserve`, `cli` and
      `server`, so it wants its own commit ahead of `caps.rs`.
- [ ] **Video encoding.** `imaging::video` decodes via an `ffmpeg` subprocess but
      cannot encode; nothing in the workspace writes mp4, webm, gif or y4m.
      The mirror-image `encode_frames` belongs next to `decode_frames`, gated on
      the existing `ffmpeg_available()` with a numbered-PPM fallback.
- [ ] **Training + LoRA** (`grad.rs`, `modelgrad.rs`, `devgrad.rs`, `train.rs`,
      `lora.rs`, `finetune.rs`) and `gradcheck::check_wan`.
- [ ] **I2V branch**: 36-channel input (16 latent + 4 mask + 16 conditioning
      frame) and the CLIP ViT-H/14 vision tower's 257 tokens through `img_emb`.
      Only `clip.visual(...)` is used -- the checkpoint's XLM-RoBERTa text side
      is dead weight for our purposes.
- [ ] **FLF2V and VACE** are out of scope for the first landing. Both use
      `shift = 16`, which is far enough from 5.0 to look like a bug if
      encountered without warning.

## The bar for the CLI

One command, in one shell, produces a playable file. The reference point is
LTX's distilled pipeline, which is worth beating rather than matching:

```
python -m ltx_pipelines.distilled \
  --transformer-path   models/.../transformer-bf16.safetensors \
  --text-encoder-path  models/.../gemma4-12b-with-proj-bf16.safetensors \
  --video-vae-path     models/.../video-vae-bf16.safetensors \
  --prompt "A belgian malinois running on a paved highway, cinematic lighting" \
  --seed 42 \
  --output-path output_distilled.mp4
```

Everything the run needs is a flag on the one command: each weight by path, the
prompt, the seed, the output file. No environment variables to export first, no
config to write, no second step to turn frames into a video.

Two things follow for `brain wan`:

- **Every `BRAIN_WAN_*` variable needs a flag twin** (`--dit`, `--vae`, `--t5`,
  `--tokenizer`, `--clip`), and the flag wins. Env vars are fine as the
  serving-path default, but a user who has just downloaded four files should
  never have to learn them to try the model once.
- **`--out video=out.mp4` is worse than `--output-path out.mp4`** for the common
  case. The `name=path` form comes from `run_do` deriving its parser from the
  capability action's blob schema, which is the right machinery for actions with
  several outputs. A single-output action should still accept the plain form.

With auto-fetch this can beat the reference outright, since the paths become
optional:

```
brain wan t2v --prompt "..." --seed 42 --output-path out.mp4
```

That is the target. If the demo needs a paragraph of setup to explain, it is
not done.

A structural constraint worth stating early: brain's fetch plan is one
`ModelRef` to one repo listing to one `Plan`. Blending a GGUF DiT from `city96/`
with a VAE and text encoder from `Wan-AI/` cannot be expressed, which is the
same limitation `.agents/roadmap/s3dit.md` records. Choosing the *native*
`Wan-AI/Wan2.1-T2V-1.3B` repo as `default_ref` sidesteps it entirely for the
default path, because that one repo carries all four roles.

## Scope that collapsed once the reference was read

Planning assumed four new kernel families. Reading
`wan/modules/vae.py` removed three of them, and the write-up is in
`scratchpad/wan-notes/01-vae-and-t5-conventions.md`. Recording it here because
the reasoning generalises: **video models are not automatically 3D everywhere**,
and assuming they are buys kernels nobody needs.

- `upsample3` -- **not needed.** The VAE's spatial upsample is a *per-frame* 2D
  `nearest-exact` at scale 2, applied under
  `rearrange('b c t h w -> (b t) c h w')`. For an exact integer 2x,
  `nearest-exact` and `nearest` are provably identical
  (`floor(d/2 + 0.25) == floor(d/2)` for integer `d`), so `upsample2.wgsl` is
  bit-correct as-is. Only a non-integer scale would break the equivalence, so
  the 2x is worth asserting rather than assuming.
- General strided `conv3d` for resampling -- **not needed.** Spatial resampling
  is `nn.Conv2d`, again per-frame with time folded into the batch. The existing
  `conv2d_gd` path covers it, and the asymmetric `nn.ZeroPad2d((0,1,0,1))` is
  the same trick `vae::blocks::conv_down` already implements.
- `pad3d` -- **not needed** as a separate kernel. `CausalConv3d` pads
  symmetrically in space and `2*pad_t` on the low side of time only, which is
  exactly the semantics `dwconv3d.wgsl`'s `pt` parameter was already written
  for ("temporal pad (2,0) with K=3: pt=2" is in that kernel's own header).
- Temporal resampling is `CausalConv3d(c, ..., (3,1,1))`, a kernel that touches
  only the time axis -- i.e. a 1D conv, reachable through the existing
  `conv1d` / `conv1d_dx` / `conv1d_dw` under a `[b*h*w, c, t]` view.

What survived: `ResidualBlock`'s `CausalConv3d(c_in, c_out, 3, padding=1)` is a
genuine (3,3,3) convolution, and that is the one real gap.

## Convention questions settled from source, not experiment

- **umT5 uses per-block relative position bias.** `wan/modules/t5.py:456-466`:
  `umt5_xxl` explicitly passes `shared_pos=False`, overriding a class default of
  `True`. brain's `crates/t5encoder` computes the bias once in block 0 and
  shares it, which is correct for T5 v1.1 and wrong for umT5 -- 24 independent
  `[num_buckets, num_heads]` tables are needed instead. **This class of bug is
  silent**: the wrong bias produces plausible-looking embeddings and subtly
  wrong video, with nothing to catch it short of stage parity against a golden.
- **The Wan-VAE norm is `RMS_norm`, not GroupNorm** (`vae.py:39-54`), and it
  normalises over the *channel* axis: `F.normalize(x, dim=1) * sqrt(dim) * gamma`.
  The exact brain match is `l2norm_scale.wgsl` (plus `l2norm_scale_dx` and
  `l2norm_scale_dg` for training) -- **not** `rmsnorm*`, which is
  `x / sqrt(mean(x^2) + eps)` over the last axis, and not `gn_*`.
- **Sampling defaults live in `generate.py`'s argument defaults**, not in any
  config the checkpoint ships. Guidance is 5.0 for every task; shift is 5.0
  everywhere except I2V at 480p, which is 3.0; steps are 50 for T2V and 40 for
  I2V. Planning had all three of these wrong -- see
  `scratchpad/wan-notes/02-sampling-defaults.md`. A port reading only
  `config.json` would silently invent its own schedule.
- **T2V-1.3B is 480p-only** upstream (`wan/configs/__init__.py` `SUPPORTED_SIZES`).
  The 75,600-token 720p case therefore only arises on the 14B tier.

## Pre-existing drift found while surveying

`docs/reference/kernels.md` claimed 401 kernels and was missing
`flash_attn_causal_gqa`, which is present at
`crates/kernels/wgsl/flash_attn_causal_gqa.wgsl` and registered in
`crates/kernels/src/lib.rs`, so `make kernels-table/check` was already failing
independent of this port. Regenerated alongside the `conv3d` kernels (401 -> 405:
one stale omission plus the three new ones) so a real drift signal is not buried
under a stale one. The kernel itself was never missing; only the catalogue was.
