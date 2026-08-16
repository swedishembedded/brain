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

- [ ] **Causal `conv3d` kernels** (`conv3d`, `conv3d_dx`, `conv3d_dw`, or an
      `im2col3d` + `matmul_reg3` lowering). This is the only genuinely new WGSL
      family the port needs -- see "scope that collapsed" below for why the other
      four planned kernels turned out to be unnecessary.
- [ ] **Wan-VAE** as a sibling 3D builder in `crates/vae/src/blocks3d.rs`, not a
      widening of `blocks.rs` -- widening every `(prefix, c, h, w, x)` signature
      would destabilise five existing consumers (AutoencoderKL, VQGAN,
      CodeFormer, RRDBNet, SDXL-UNet) to no benefit.
- [ ] **`feat_cache`**, the causal VAE's cross-chunk state (`CACHE_T = 2`). Three
      sentinel states exist upstream for `upsample3d` -- `None`, the string
      `'Rep'`, and a real cached tensor -- selecting between "no time_conv",
      "time_conv against a zero cache" and "time_conv against the real cache".
      All three have to be reproduced, and the chunked-vs-unchunked equality
      test has to exist before any of the rest of the VAE is trusted. This is
      the single most likely source of a "correct at 8 frames, wrong at 81" bug.
- [ ] **umT5-XXL** in `crates/t5encoder`: vocab 32128 -> 256384, per-block
      relative position bias (see below), attention masking with a zero-pad to
      `text_len = 512`. The crate's own size analysis understates umT5 by about
      3.7 GB because it assumes the T5 v1.1 vocabulary.
- [ ] **SentencePiece unigram tokenizer** in `crates/data` -- brain has GPT-2,
      Qwen3 and CLIP BPE, but no unigram implementation at all, and umT5's
      256k-entry `spiece.model` needs one.
- [ ] **Goldens** (`tools/goldens/wan_dump_reference.py`) before any DiT Rust,
      per porting.md section 1.
- [ ] **UniPC multistep in the flow-matching parameterisation**, plus the sigma
      shift `s' = shift*s / (1 + (shift-1)*s)`. Neither exists: brain's
      `DpmSolverPlusPlusScheduler` is built on `alphas_cumprod` and cannot be
      pointed at flow-match sigmas, and there is no UniPC anywhere in the repo.
- [ ] **`crates/wan`** proper -- currently only `config.rs`.
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

`docs/reference/kernels.md` claims 401 kernels and is missing
`flash_attn_causal_gqa`, which is present at
`crates/kernels/wgsl/flash_attn_causal_gqa.wgsl` and registered in
`crates/kernels/src/lib.rs`. `make kernels-table/check` should therefore be
failing already, independent of this port. Worth regenerating alongside the
`conv3d` kernels so a real drift signal is not buried under a stale one.
