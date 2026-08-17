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
- [x] **umT5-XXL** in `crates/t5encoder` as a second `T5Config` variant
      (`umt5_xxl()`), not a second crate: same block topology, three deltas.
      * **vocab 32128 -> 256384.** The whole 918 M parameter difference is the
        embedding table, i.e. **+3.67 GB** in fp32 before a block is allocated
        (4.762 B -> 5.681 B, 19.05 -> 22.72 GB). The crate's size analysis was
        understating it by exactly that and now states both.
      * **Per-block relative position bias** (`shared_pos=False`): the manifest
        gains `blocks.<l>.rel_bias.weight` and loses the shared one (171 -> 194
        tensors), and the gather+permute pair moves inside the block loop. The
        bias slabs stay per block (67 MB each at T=512) rather than sharing one
        scratch, so `read_block_bias` can gate block 0 AND block 23 - with one
        scratch a shared-bias regression would only be visible at the last
        block. The bucket math is UNCHANGED between the two variants, so
        `hostbias` is reused as-is.
      * **Key padding**, one new kernel: `attn_keypad_mask` (405 -> 406), the
        bidirectional twin of `attn_prefix_mask`, added into the score slab
        between the bias and the softmax. `attn_scores_bidir_bias`'s bias is
        `[H,T,T]` with no batch axis, so folding the mask into it would have
        been correct only at B=1. An unmasked config records no mask step at
        all, so FLUX's certified graph is unchanged byte for byte.
      * **The 512 pad is applied AFTER the encoder, as hard zeros**
        (`read_context`), because `T5EncoderModel.__call__` trims to `seq_len`
        and `WanModel.forward` re-pads with `new_zeros`. The encoder's own
        output at those positions peaks at 0.87 and is discarded.
      Training is deliberately NOT extended: `T5Trainer` folds one shared
      `rel_bias` gradient across the block stack and attends over every key, so
      it asserts against both flags instead of returning a wrong gradient.
- [x] **SentencePiece unigram tokenizer** -- `crates/data/src/unigram.rs`, the
      first non-BPE tokenization model in the workspace (Viterbi over the piece
      lattice, `fuse_unk`, Metaspace pre-tokenizer, `TemplateProcessing`).
      Built from **`tokenizer.json`, not `spiece.model`**: the JSON is the
      artifact `AutoTokenizer.from_pretrained` actually loads (which is what
      `wan/modules/tokenizers.py` wraps), it needs no protobuf decoder, and it
      needs neither sentencepiece's `precompiled_charsmap` normalizer nor its
      piece-type table - umT5's normalizer is one `" {2,}" -> " "` rule. Exact
      ids on all 9 golden prompts including the unknown-piece path.
- [x] **umT5 goldens** (`tools/goldens/wan_t5_dump_reference.py`), five
      self-validations before a byte is written, including an independent
      Viterbi that doubles as the spec for the Rust tokenizer. The DiT's own
      goldens are still to come.
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
- [x] **The DiT itself** -- `rope.rs`, `block.rs`, `model.rs` (host-orchestrated
      reference), `dev.rs` (device-resident engine, weights uploaded once, the
      whole 30-block stack as ONE recorded graph) and the DiT half of
      `import.rs`. **Zero new kernels**: the whole transformer is existing
      kernels at Wan's shapes.
      * **Goldens first** (`tools/goldens/wan_dit_dump_reference.py`), with two
        independent paths asserted before a byte is written: the official
        `wan/modules/model.py` and diffusers' `WanTransformer3DModel`. The
        1.3B weights ship in the diffusers name space, so the dumper converts
        them to the reference names -- the same mapping `import.rs` implements,
        which means a mapping mistake fails in Python instead of surfacing as a
        cosine deficit thirty blocks deep. The two agree at cosine
        1.0000000000, rel 4.98e-6.
      * **One shim, recorded in the manifest**: `flash_attention` asserts
        `q.device.type == 'cuda'`, and its own fallback would run SDPA in
        **bfloat16** with the key-padding mask dropped. Replaced with an fp32
        SDPA that honours `k_lens`; the diffusers cross-check is what says the
        replacement invents nothing.
      * **The `seq_len` pad is provably irrelevant**, settled by experiment
        rather than argument: `text2video.py` computes `seq_len` as exactly the
        token count at `sp_size = 1`, and a forward at `tokens + 37` leaves the
        output at cosine 1.0000000000 (3.4e-6 relative, fp32 reassociation in
        SDPA's key blocking). brain therefore computes content rows only and
        carries no token mask. The text encoding's own pad rows are real and
        are reported as a separate population.
      * **The modulation fold** (`.agents/rules/porting.md` section 7):
        `e0 = time_projection(time_embedding(t))` is `[1, 6, dim]`, a function
        of the timestep alone, so `LN_noaffine(x)*(1+scale)+shift` is exactly
        `LayerNorm(gamma = 1+scale, beta = shift)`. Six vectors become two
        `(gamma, beta)` pairs plus two `gate_row` gates per block, computed once
        per forward on the host. Wan 2.2's TI2V passes a **per-token** `temb`,
        which breaks the token-independence; `ModBufs::upload` takes one
        `[6·dim]` vector so that variant cannot be fed to it by accident.
      * **QK-norm is across ALL heads, not per head.** `WanRMSNorm(dim)` runs
        before the `view(b, s, n, d)`; diffusers spells it
        `RMSNorm(dim_head * heads)` under the config name
        `"rms_norm_across_heads"`. Per-head would divide by a different scalar
        per head and still produce plausible video.
      * **diffusers' `norm2` and `norm3` are SWAPPED against upstream's.**
        diffusers `norm2` is the cross-attention norm (upstream `norm3`);
        diffusers `norm3` is the FFN pre-norm (upstream `norm2`, affine-free and
        therefore absent from the checkpoint). A pass-through mapping imports
        cleanly, validates cleanly, and normalises with the wrong learned affine.
      * **Two patch orderings, and they differ.** `patch_embedding` is a
        `Conv3d` whose weight row flattens `[c][kt][kh][kw]`, so its token
        vector is channel-OUTERMOST; the head's row is `view(*patch_size, c)`,
        channel-INNERMOST. One ordering for both gives a shuffled latent that
        still looks like video.
      * **Attention**: 32,760 tokens at 480p makes a materialised score matrix
        51 GB across 12 heads against the P40's 2047 MiB per-binding ceiling, so
        self-attention is `flash_attn_bidir{,_split}` on any device with
        workgroup reductions and query-chunked `[heads, chunk, t]` slabs
        otherwise (the CPU JIT cannot run the flash barriers). Cross-attention
        is query-chunked against the 512 text keys. A weight-free test builds
        and submits a real 32,760-token graph at toy widths.
      Parity (`crates/wan/tests/dit_parity.rs`): tiny 3-block model at 320
      tokens **cosine 1.000000000** on both backends (flash on Vulkan, chunked
      on the CPU JIT), rel_l2 2.4e-7 at the output; the real 1.3B weights at
      **4,680 tokens** (latent 3x60x104, i.e. 480p) at **cosine 1.000000000**,
      rel_l2 3.755e-6, max_abs 1.814e-4 against the reference and 3.635e-6 /
      1.566e-4 against diffusers, with every fourth block tapped and the
      host-orchestrated and device-resident forwards agreeing to the last digit.
      Perf is deliberately NOT addressed: the chunked fallback's naive cross
      trio is far too slow to be a GPU path at 4,680 tokens (it needs a raised
      `BRAIN_GPU_WAIT_S` to finish at all), and `model::block::gemm_bidir_fwd`
      is the measured answer when that ladder starts.
- [x] **The pipeline and the CLI wiring** on top of the parity-proven forward:
      `crates/wan/src/pipeline.rs` (tokenize -> umT5 -> 512-pad -> seeded
      latent -> UniPC/DPM++ with CFG -> VAE decode -> RGB frames) and
      `crates/cli/src/wan_cli.rs` (`brain wan t2v`, one `ARCH_HANDLERS` row).
      * **Three models, never resident together.** The staging is a design
        constraint, not an optimisation: umT5-XXL is 22.72 GB in fp32 and
        provably does not fit the 24 GB card, so `encode_text` is a *function*
        whose return drops the encoder before the DiT allocates anything, and
        the DiT is dropped before the VAE builds. Placement is
        `--t5-device` / `BRAIN_WAN_T5_DEVICE`, defaulting to CPU - the
        `BRAIN_FLUX2_TE_DEVICE` precedent.
      * **Both prompts ride ONE `B = 2` text forward**, and both are embedded
        through `text_embedding` ONCE (`WanDitDev::set_context_embed`, added
        here). The MLP is ~9 GFLOP of host work per call at 1.3B widths and the
        answer is fixed for the whole run; re-deriving it per CFG branch per
        step would have put minutes of host math inside the loop.
      * **A step's two forwards must bracket their own context upload.** The
        engine has one context buffer, so hoisting either `set_context_embed`
        out of the loop silently conditions every step on whichever prompt was
        last uploaded - a defect that still produces plausible video.
      * `--seed` is `data::rng::Rng` + Box-Muller, deliberately NOT torch's
        Philox: no golden here asks for bit-identical noise, and claiming it
        would be unbacked.
      * **`BRAIN_GPU_WAIT_S`**: one forward is the whole 30-block stack in a
        single submit, which at 480p is far past the backend's 30 s deadlock
        guard - the first real 480p run died as "device likely wedged". The CLI
        raises it (announced) unless the caller already set it.
- [ ] **Perf.** First real end-to-end run: 33 frames at 832x480, 25 UniPC
      steps with CFG (50 forwards of 14,040 tokens), P40 + Vulkan, **57.5 min
      wall clock** - text 246 s, DiT load 20 s, denoise 2308 s (46 s/forward
      under a contended CPU; 37 s on an idle box), VAE decode 876 s. Three
      things that ladder points at:
      * **the VAE decode is now a quarter of the run** and is pure `conv3d` at
        every layer - the `im2col_at` + `matmul_reg3` lowering the VAE section
        already names is the first measurement to take;
      * **the text encode is a fixed ~4 min tax on every generation**, because
        umT5-XXL runs on the CPU. INT8 (`t5encoder::model::int8`, already the
        crate's own stated answer) is what would put it on the card;
      * 81 frames at 480p is 32,760 tokens, 5.4x the attention work of the
        measured point, so the flagship configuration is hours today.
- [ ] **`capability::Media::Video`**. There is no video media type, and
      `.agents/rules/serving-contract.md` section 4 requires extending `Media`
      and the D-Bus frame handling rather than adding a side channel. This is a
      breaking enum change across `capability`, `dbus`, `apiserve`, `cli` and
      `server`, so it wants its own commit ahead of `caps.rs`.
- [x] **Video encoding.** `imaging::video::encode_frames`, the mirror image of
      `decode_frames`: numbered PPMs into a temp directory, one `ffmpeg`
      invocation, `-pix_fmt yuv420p` forced so the file plays outside the tool
      that wrote it. Three things worth keeping:
      * the no-ffmpeg fallback is a **separate public function**
        (`write_frame_dir`), because on a machine that HAS ffmpeg a test
        driving only `encode_frames` never reaches the fallback - the path that
        exists precisely for machines the test never runs on;
      * it returns `Encoded::{Video, Frames}` rather than an error, so a
        generation that took an hour is never thrown away for want of an
        encoder, and the `Frames` arm carries the exact command that finishes
        the job;
      * **odd dimensions are padded, loudly.** 4:2:0 cannot represent them;
        libx264 rejects the stream and other encoders quietly drop a row.
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
