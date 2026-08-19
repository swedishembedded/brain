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
        self-attention is the `flash_attn_bidir{,_split,_reg,_reg2}` family on any device with
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
- [ ] **Perf. Three of the four named defects are FIXED; self-attention is
      not.** The baseline below is rung 2 of `.agents/rules/porting.md` section
      10; the "after" tables at the end of this entry are rungs 3-5 for the VAE
      convolution, the cross-attention scores and the host stages. Everything
      here is a measurement, not a plan, and every "after" number was taken with
      the parity gates green at their existing bars (DiT cosine 1.000000000 at
      every block tap against the real 1.3B checkpoint, VAE cosine 1.000000 at
      every stage tap against the real VAE checkpoint).

      **End to end**, `brain wan t2v` on one Tesla P40 (24 GB, Vulkan), umT5-XXL
      on the CPU, fp32 throughout:

      | Request | text | DiT load | denoise | VAE decode | total |
      |---|---|---|---|---|---|
      | 33 frames 832x480, 25 steps | 246 s | 20 s | 2308 s (46 s/fwd) | 876 s | 57.5 min |
      | 9 frames 416x240, 20 steps | 241 s | 19 s | 81 s (2.0 s/fwd) | 41 s | 6.4 min |
      | 9 frames 256x256, 4 steps | 240 s | 19 s | 12 s (1.5 s/fwd) | 28 s | 5.0 min |

      The 46 s/forward row was taken under a contended CPU; an idle box gives
      37 s. The three rows exist to separate the FIXED cost from the
      size-dependent one: **the text encode does not shrink**, so at any size
      worth calling a smoke test it is most of the wall clock.

      **Per kernel kind** - `crates/wan/src/bin/wan_bench.rs`, which builds the
      REAL graphs (`WanDitDev::build` / `WanVaeDecoder::build`) over a
      zero-filled source of the checkpoint's own manifest shapes, so it needs no
      weights on disk and cannot drift from what a generation submits. Each kind
      is timed by submitting only its own steps; the per-dispatch floor is
      0.074 ms of queue round-trip (`wan_bench floor`), so the isolation costs
      about 1 ms per table.

      **DiT, one forward at 14,040 tokens** (33 frames at 832x480), 1140
      dispatches, 34.26 s in one submit, 72,915 GFLOP -> 2128 GFLOP/s = **18.1%
      of the P40's 11.76 TFLOP/s fp32 peak**:

      | kernel | disp | ms | % | GFLOP/s | % peak |
      |---|---|---|---|---|---|
      | `flash_attn_bidir_split` (self-attn) | 30 | 19029 | 53.2% | 1909 | 16.2% |
      | `matmul_reg3` (every projection + FFN) | 300 | 7330 | 20.5% | 4810 | 40.9% |
      | `attn_scores_cross` | 30 | 7062 | 19.7% | 94 | 0.80% |
      | `attn_apply_cross` | 30 | 1557 | 4.4% | 426 | 3.6% |
      | everything else (9 kinds) | 750 | 787 | 2.2% | | |

      Sum of kinds 35.76 s against 34.26 s in one submit, i.e. the
      instrumentation is worth about 4%.

      **The surprise is cross-attention, and it is the kind of surprise section
      10 rung 3 exists for.** `attn_scores_cross` and `attn_apply_cross` do
      *identical* arithmetic - 662 GFLOP each over the whole stack, `q·kᵀ` and
      `p·v` against the same 512 text keys - and scores takes **4.5x** as long.
      Nobody would have guessed that the 512-key cross-attention costs as much
      as all 300 GEMMs put together; the GEMMs are the healthiest thing in the
      table at 40.9% of peak. Two independent reasons this is a defect rather
      than a ceiling: the two halves of the same computation disagree by 4.5x,
      and 0.80% of peak is two orders of magnitude under the roofline.

      Self-attention is the majority and is a genuine `O(t²)` cost, but 16.2% of
      peak against `matmul_reg3`'s 40.9% says the flash kernel is not close to
      what the same silicon does on a plain GEMM either.

      **VAE decode, 9 frames at 832x480** (latent `[16,3,60,104]`), 762
      dispatches, 199.6 s in one submit:

      | kernel | disp | ms | % |
      |---|---|---|---|
      | `conv3d` | 113 | 204687 | **96.5%** |
      | `attn_scores_bidir` | 3 | 3132 | 1.5% |
      | `l2norm_scale` | 90 | 1369 | 0.6% |
      | `nchw_nlc` / `nlc_nchw` | 186 | 1904 | 0.9% |
      | everything else (10 kinds) | 370 | 1030 | 0.5% |

      The decode is ONE kernel. The two heaviest shapes are
      `[cin 192, (6,240,416)] -> [cout 192, (4,240,416)]` and
      `[cin 96, (6,480,832)] -> [cout 96, (4,480,832)]`, both 3x3x3: **795 GFLOP
      per dispatch, six dispatches each, 35.6 s and 35.5 s** - i.e.
      **134 GFLOP/s, 1.14% of peak**. Minimum traffic for one of those is about
      1.53 GB in 5.9 s = 259 MB/s against the card's 346 GB/s, so it is not
      bandwidth-bound either. Under 10% of BOTH rooflines is section 10 rung 5's
      definition of a defect, and rung 5 also names the fix: this is the
      structural case for an algorithmic change (`im2col_at` + `matmul_reg3`),
      not for tuning - the same `matmul_reg3` reaches 40.9% of peak in the table
      above, a 36x gap.

      Three things follow, in the order the shares put them:
      * **`conv3d` is the single biggest win available** and it is one kernel
        against one already-fast alternative. The VAE section above already
        named the lowering; this is the measurement that says to take it.
        **Done** - see "VAE decode, after" below, 10.8x.
      * **`attn_scores_cross` is the cheapest win available** - 20% of the DiT
        for arithmetic its own twin does 4.5x faster. **Done** - see
        "cross-attention, after" below, 5.96x.
      * **the text encode is a fixed tax** and no kernel change touches it:
        umT5-XXL runs on the CPU because 22.72 GB of fp32 does not fit the card.
        INT8 (`t5encoder::model::int8`, already the crate's own stated answer)
        is what would put it on the card, and section 10 rung 6 is the warning
        that comes with it - quantization buys residency first and speed only
        if the profile says arithmetic is the limiter. **Not done.**

      ## What 20x would require, and why fp32 cannot give it

      Settle this before reading the "after" tables, because it bounds them. One
      DiT forward is 72,915 GFLOP. At the P40's **entire** 11.76 TFLOP/s fp32
      peak that is 6.20 s, so the 50 forwards of a 25-step CFG request cannot go
      below **310 s in fp32 no matter what any kernel does** - and that is 1.8x
      the 173 s that a 20x speedup of the 3454 s baseline would allow for the
      whole pipeline, text encode and VAE included. Self-attention alone is
      36,333 GFLOP a forward (49.8% of the total) and floors at 3.09 s.

      So 20x is not a hard optimization target here, it is an arithmetically
      unreachable one at fp32, and no amount of kernel work reaches it. The
      honest ceiling in fp32 is bounded by that 310 s denoise floor; INT8 (rung
      6) is the only lever that moves the floor itself, because the P40's DP4A
      rate is ~4x its fp32 rate.

      ## After

      Same machine, same bench, same shapes. Hardware named exactly: one Tesla
      P40 (24 GB, Vulkan, sm_61), 48-core Xeon E5-2690 v3 host.

      **End to end, the baseline request re-run** (33 frames 832x480, 25 steps,
      seed 42, the same prompt), `brain wan t2v`:

      | phase | before | after | |
      |---|---|---|---|
      | text encode (umT5-XXL, CPU) | 245.8 s | 236.8 s | 1.04x |
      | DiT load | 20.1 s | 11.5 s | 1.75x |
      | denoise (50 forwards) | 2308.3 s (46.2 s/fwd) | 1763.7 s (35.3 s/fwd) | 1.31x |
      | VAE decode | 876.1 s | **101.6 s** | **8.62x** |
      | **total** | **3454 s (57.5 min)** | **2115.6 s (35.3 min)** | **1.63x** | <!-- perf-number: a real end-to-end run of the shipped CLI on one named card, the same request the baseline row above reports -->

      The clip is visually identical to the pre-optimization reference at the
      same seed (same subject, framing, lighting and motion at frames 0/10/20/32;
      374,655 vs 374,946 bytes encoded).

      **The card thermally throttles, and it matters to every number here.**
      Under sustained load GPU 0 sits at 89-90 C and drops to **999 MHz of its
      1531 MHz** clock (`clocks_event_reasons.active = 0x20`, SW thermal
      slowdown), recovering to ~1300 MHz intermittently. So:
      * the per-step cost *rises* through a run - the cumulative average went
        57.4 s/step at step 2 to 70.5 s/step at step 26;
      * `wan_bench`'s best-of-N numbers are taken in short bursts at a higher
        clock and are therefore OPTIMISTIC against a 30-minute request. The
        bench predicts 31.8 s/forward; the sustained run measured 35.3 s;
      * **the honest sustained fp32 peak is ~7.67 TFLOP/s, not 11.76**, which
        raises the 50-forward denoise floor from 310 s to about **475 s**. Every
        roofline percentage in the tables above is quoted against the boost
        figure and is thus a lower bound on true utilisation.

      **Host stages, per forward** (`wan_bench host`, 33 frames at 832x480).
      `wan::model::linear` was a single-threaded scalar triple loop; it is now
      row-parallel through `backend_cpu::par` (the workspace's only rayon seam).
      The split is over output rows with each row's dot product untouched, so it
      is **bit-identical**, not merely close - which is what lets a
      parity-gated path take it unconditionally.

      | stage | before | after | |
      |---|---|---|---|
      | `embed_tokens` (patchify + proj) | 872.0 ms | 43.3 ms | 20.1x |
      | `postprocess` (head + unpatchify) | 1285.5 ms | 154.4 ms | 8.3x |
      | RoPE tables | 8.2 ms | **0** (hoisted to build) | - |
      | `timestep_cond` | 14.6 ms | 19.3 ms | ~1x |
      | **per-forward total** | **2.188 s** | **0.225 s** | **9.7x** |
      | `text_embed` (once a prompt) | 3881.2 ms | 118.2 ms | 32.8x |

      The RoPE tables are a pure function of the (f, h, w) patch grid and the
      grid is fixed for the life of an engine, so they were being recomputed and
      re-uploaded (~1.8 M sin/cos pairs, ~14 MB) on every one of the 50 forwards
      for nothing. They are built once in `WanDitDev::build` now.

      This is also why the baseline's 46 s/forward and an idle box's 37 s
      disagreed: a single-threaded 2.2 s host leg is exactly the part that
      degrades under CPU contention. At 0.225 s the pipeline is far less
      sensitive to what else the box is doing.

      **Cross-attention, after.** `attn_scores_cross` parallelises over the KEY
      index and reduces over `head_dim`, so against the natural `[text_len,
      dim]` K every lane of a warp landed on its own cache line. `attn_apply_cross`
      moves the *same* bytes of the *same* slab with `d` as its thread index -
      contiguous - which is the whole of the 4.5x. Transposing K once a block
      (`kv_k_headt`, a 512x1536 shuffle, 1.4 ms across all 30 blocks) and
      reading it key-minor (`attn_scores_cross_kt`) buys that coalescing:

      | kernel | before | after | |
      |---|---|---|---|
      | `attn_scores_cross` -> `_kt` | 7303.4 ms | 1224.7 ms | **5.96x** |
      | `kv_k_headt` (new, once a block) | - | 1.4 ms | |

      It lands at 1225 ms against `attn_apply_cross`'s 1333 ms - i.e. exactly
      the rate its twin already got for identical traffic, which is the
      confirmation that coalescing was the entire defect. Both kernels are
      additive: `attn_scores_cross` is unchanged and still serves the chunked
      self-attention fallback and every other model that dispatches it.

      **DiT, one forward at 14,040 tokens, after** - 1170 dispatches, 28.106 s
      in one submit, 72,915 GFLOP -> 2594 GFLOP/s = **22.1% of peak** (was
      34.147 s / 2135 GFLOP/s / 18.2%):

      | kernel | disp | ms | % | note |
      |---|---|---|---|---|
      | `flash_attn_bidir_split` | 30 | 17960 | **63.6%** | unchanged - the wall |
      | `matmul_reg3` | 300 | 6927 | 24.5% | healthy, ~41% of peak |
      | `attn_apply_cross` | 30 | 1333 | 4.7% | |
      | `attn_scores_cross_kt` | 30 | 1225 | 4.3% | was 7303 |
      | everything else (10 kinds) | 780 | 661 | 2.3% | |

      A full forward including host stages, uploads and the readback is
      **37.82 s -> 31.8 s**; the 3.67 s gap between the graph and the whole
      forward is upload/readback, and is NOT host math (that is the 0.225 s
      above). Nobody has attacked it.

      **VAE decode, after** (9 frames at 416x240, the size the before-table was
      re-taken at). `conv3d` measured a **flat 134-138 GFLOP/s, 1.14-1.17% of
      peak, across four very different shapes** - a rate that does not move with
      shape is structural, which is rung 5's definition of "needs an algorithmic
      change, not tuning". The change is the one the 2D builder already made and
      documented (`im2col_at` + `matmul_reg3` + `nlc_bias_nchw`), lifted to the
      time axis as `im2col3d_at`; `conv3d`'s weight index is already
      `(((co*Cin + cl)*KT + kt)*KH + kh)*KW + kw`, so the same weight tensor is
      the GEMM's B operand with no repacking.

      | | before | after | |
      |---|---|---|---|
      | whole decode graph | 46.016 s | **4.268 s** | **10.8x** |
      | `conv3d` | 113 disp, 49515 ms, 98.1% | 4 disp, 66 ms, 1.5% | |
      | `matmul_reg3` | - | 250 disp, 1965 ms, 46.1% | |
      | `im2col3d_at` | - | 250 disp, 1343 ms, 31.5% | |

      The four surviving `conv3d` dispatches are the low-channel ones the
      `GEMM_CONV3D_MIN_COUT` guard correctly keeps direct. `im2col3d_at` at
      31.5% is now the obvious next VAE target - it is pure data movement.

      ## Hypotheses killed, with numbers

      Negative results, so the next person does not re-run them:
      * **INT8 is not the DiT's answer, at least not first.** The GEMMs are the
        *healthiest* thing in the profile at ~41% of peak and only 24.5% of the
        graph; attention is 68% and no GEMM precision touches the flash kernel.
        A perfect 4x on every `matmul_reg3` would take 28.1 s to 22.9 s - 1.23x.
        Rung 6 in one line.
      * **Batching the two CFG forwards to B=2 is not "close to a straight
        2x".** It does not reduce FLOPs at all; it only amortises per-forward
        fixed cost, and at 14,040 tokens the card is already saturated
        (`matmul_reg3` at 41% of peak, a 6.2 s/forward fp32 floor). The engine's
        attention slab *would* take it - `model::block::flash_bidir_step`
        documents `bsz > 1` as sample-major and bit-identical - so the ceiling
        is the 3.67 s/forward of upload/readback, not 2x.
      * **The GEMMs are already on the register-tiled kernel.** `matmul_reg3` is
        what all 300 projection/FFN dispatches land on, at 4300-4836 GFLOP/s
        across the four shapes Wan uses. There is no naive-`matmul` bug to find.
      * **`flash_attn_bidir` vs `_split` is not a live choice on this card.**
        `flash_bidir_variant` takes the split kernel whenever
        `max_workgroup_size >= 256`, which the P40 satisfies; there is no shape
        knob to sweep here without writing a third variant. **Two were written -
        see below. The shape knob was then swept anyway and there is still no
        crossover: the faster kernel wins at every T from 256 to 14040.**

      ## What is left, in the order the shares now put it

      * **`flash_attn_bidir_split` is THE wall.** 17.96 s of a 28.11 s forward
        (63.6%), ~900 s of the 2116 s request, and untouched. It runs at 2023
        GFLOP/s = 17.2% of the boost-clock peak while `matmul_reg3` on the same
        silicon reaches 41%. That gap is worth ~10 s a forward (~500 s a
        request) and it is the only remaining item big enough to change the
        headline. It is not a defect by rung 5's test (17.2% is over the 10%
        line), so it needs a better kernel, not a bug fix. **Done - see "Self-
        attention: the wall, taken down" below; the kernel is 1.98x and the
        forward is no longer attention-dominated.**
      * **The text encode is untouched and is now 11% of the request.** 236.8 s
        of fixed CPU tax. bf16 weight storage (`crates/model/src/half.rs`, the
        `@dtype` kernel headers) would fit umT5-XXL's 22.72 GB in 11.4 GB and
        put it on the card; INT8 is the crate's own stated answer.
      * **3.67 s a forward of upload/readback** - the gap between the 28.11 s
        graph and the 31.8 s full forward, and NOT host math (that is 0.225 s).
        ~185 s a request in `write_f32`/`read` of the 86 MB token slab either
        side of each submit. Nobody has looked at it.
      * **`im2col3d_at` is 31.5% of the new VAE decode**, pure data movement,
        and is the obvious next VAE step now that `conv3d` is gone.
      * **`attn_scores_cross` itself is still slow for every OTHER model that
        dispatches it** - `sam1`, `sam2`, `clip`, `pulid`, `instantid`,
        `fastvlm`, `deepseekocr`, `vit`. Wan now routes around it via the
        additive `attn_scores_cross_kt`; the shared kernel was deliberately not
        changed, because its callers were not measured here. The same transpose
        would help all of them and is the single highest-leverage follow-up
        outside this model.

      Not measured, and worth saying so: 81 frames at 480p is 32,760 tokens,
      5.4x the attention work of the profiled point, and no per-kernel table was
      taken there; the flagship configuration is extrapolated, not measured.

      ## Self-attention: the wall, taken down

      The section above closed with "`flash_attn_bidir_split` is THE wall" -
      17.96 s of a 28.11 s forward, 63.6%, at 2023 GFLOP/s. It is now
      `flash_attn_bidir_reg2` at 9.06 s, and the DiT graph is 19.30 s.

      **What the profile actually said.** The kernel was not bandwidth-bound and
      not occupancy-limited, and the online-softmax bookkeeping was not the
      cost. Both inner loops issued exactly ONE shared load per fused
      multiply-add:

          s   = s   + q[c]  * ksh[ko + c*LANES]
          acc = acc + pj[j] * vsh[j*HD + vo]

      A Pascal SM issues an FFMA warp-instruction every clock but retires a
      shared load-store warp-instruction only every fourth (32 LD/ST units
      against 128 fp32 lanes), so a 1:1 mix cannot exceed a QUARTER of the
      card's fp32 rate however well the loads are laid out. That is the number
      that was measured: 2023 GFLOP/s against the device's own measured
      10542 GFLOP/s roof = 19.2%, i.e. sitting on a 25% structural ceiling. The
      tile reads were ALREADY broadcast across the 8 query rows of a warp and
      already hit 16 distinct banks, so shared memory itself was nowhere near
      saturated - there was no bank-conflict bug to find, and rung 5's
      "<10% of both rooflines" defect test correctly did not fire.

      **The fix, in the four steps that were each measured separately** (Tesla
      P40, T=14040, 12 heads, head_dim 128, `wan_bench flash`, min-of-N, graded
      against the MEASURED roof, not the datasheet):

      | kernel | ms | GFLOP/s | % of measured roof | vs split |
      |---|---|---|---|---|
      | `flash_attn_bidir` | 16494 | 73 | 0.7% | 0.04x |
      | `flash_attn_bidir_split` | 599 | 2022 | 19.2% | 1.00x |
      | `flash_attn_bidir_reg` - vec4 tiles, mix 1:4 | 483 | 2508 | 23.8% | 1.24x |
      | + 2 query rows per thread, mix ~1:7 | 346 | 3486 | 33.1% | 1.73x |
      | + tile depth 8 -> 16, half the barriers per key | 330 | 3669 | 34.8% | 1.82x |
      | + vec4 cross-lane partials | 327 | 3702 | 35.1% | 1.83x |
      | `flash_attn_bidir_reg2` - + prefetch, 3 barriers -> 2 per tile | **302** | **4003** | **38.0%** | **1.98x** |

      Every row agrees with `flash_attn_bidir` to **cosine 1.000000000**
      (max_abs 4.8e-6 over the [14040, 1536] context slab).

      Tile depth 4 was measured too and is WORSE (440 ms): below 8 the barrier
      and per-key bookkeeping dominate. Depth 32 does not fit - `reg2` already
      uses 49152 B of workgroup memory, exactly the limit this card reports.

      **Where it sits against the roof now.** 38.0% of the card's own measured
      fp32 roof, against `matmul_reg3`'s 41% on the same silicon - the best
      kernel class this engine has. The mix is now ~7:1 FMA:LDS so the LD/ST
      ceiling is no longer binding; what remains is the two-query-row block's
      register pressure holding the kernel to ONE workgroup per SM (8 warps to
      hide latency with) and the two barriers per tile. Getting to one barrier
      means removing the cross-lane partial reduction, and with no subgroups and
      128 channels per query row there is no way to do that inside a
      255-register budget. **This is close to the wall, not short of it.**

      **The card does NOT throttle on this kernel, though it does on the
      request.** The section above records 999 MHz of 1531 at 89-90 C under
      sustained load and warns that short benches overstate. Re-measured with a
      155-second continuous attention load and a one-second clock sample:
      **1531 MHz throughout, 83 C, 169 W** of a 250 W board, and the 40-rep and
      3-rep numbers agree within 1%. Attention alone does not draw enough power
      to trip the thermal slowdown. During a real request the card WAS observed
      at 1088-1227 MHz and 89 C - so the throttling is driven by the GEMM-heavy
      phases around attention, not by attention. Consequence: the kernel table
      above is a sustained-regime measurement, but the per-request cost of the
      same kernel is higher because the GEMMs beside it heat the card.

      **A shape sweep, because BR changed and BR is a grid parameter.** `reg2`
      owns 128 query rows where every other member owns 64, which halves its
      global K/V traffic (38 GB -> 19 GB per attention call at Wan's shape) but
      also halves the workgroup count, and an underfilled grid is a real way to
      lose. Swept at 12 heads, head_dim 128, ms:

      | T | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 14040 |
      |---|---|---|---|---|---|---|---|
      | `split` | 0.6 | 1.2 | 4.0 | 13.6 | 51.0 | 206.0 | 599.0 |
      | `reg` | 0.6 | 1.1 | 3.5 | 11.4 | 41.5 | 162.8 | 483.0 |
      | `reg2` | **0.4** | **0.8** | **2.4** | **7.6** | **26.8** | **104.8** | **302.5** |

      `reg2` wins at every point over a 55x range of T, including T=256 where it
      launches only 24 workgroups for 30 SMs. **There is no crossover to encode**
      - the selection rule is device caps only, and deliberately not shape.

      **The fix is in the SELECTOR, not the call sites.**
      `model::block::FlashIds` grew `reg` and `reg2`, and `flash_bidir_variant`
      now returns `(kernel, workgroup size, BR)` - BR is no longer a family
      constant, so a caller must take it from the selector. Gating is on queried
      `DeviceCaps` only: `reg2` needs `max_workgroup_size >= 256` AND
      `workgroup_mem_bytes >= 49152` (four times the 16 KiB a Vulkan
      implementation is only *required* to offer), with `reg` as the 16 KiB rung
      below and the old pair below that. The two new fields are REQUIRED rather
      than defaulted, so the change broke all six consumers at compile time -
      `flux1`, `flux2`, `lfm2`, `sdxlunet`, `s3dit`, `wan` - and each was
      adopted deliberately instead of being silently missed, which is the
      failure mode a `..Default::default()` would have invited.

      **Gates.** DiT cosine 1.000000000 at every block tap (0, 4, 8, 12, 16, 20,
      24, 28, 29) and at the output against both the reference transformer and
      diffusers, on the real 1.3B checkpoint at 4680 tokens; VAE unchanged. The
      gate was mutation-verified: swapping the second query row's `q1` for `q0`
      inside the register block drops block.0 to cosine 0.999183774 and fails
      five tests, so the suite does exercise the two-row path rather than
      passing on the first row alone. `lfm2`'s `pipelines_fully_costed` caught a
      genuine omission - the new kernels had no `gpu_core::cost` formula and
      their dispatches would have been reported UNCOVERED - which is exactly
      what that gate exists for.

      **Pre-existing bug found and fixed.** `wan_bench` never raised
      `BRAIN_GPU_WAIT_S` from its 30-second default, which is sized for a
      token-at-a-time decoder. At the bench's OWN documented default shape a DiT
      forward was 28 s of device time, so `wan_bench dit` printed its whole
      table and then panicked with "device likely wedged" on the readback that
      follows. `brain wan t2v` already raised the limit to 1200 s for the same
      reason; the bench now does too. The failure was silent in the worst way -
      the table above the panic looked correct.

      **What is left after this.** The forward is no longer
      attention-dominated: `flash_attn_bidir_reg2` is 46.7% of a 19.30 s graph
      and `matmul_reg3` is 35.7%, so the GEMMs are now the largest single
      addressable block and they are already at ~41% of the roof. The next
      items by share are unchanged and none of them is the kernel:
      `attn_apply_cross` (7.2%), `attn_scores_cross_kt` (6.3%), the 3.6 s a
      forward of upload/readback, and the CPU text encoder.

      **DiT forward, one submit at 14,040 tokens**, 1170 dispatches:

      | kernel | disp | before ms | after ms | |
      |---|---|---|---|---|
      | self-attention | 30 | 17959 (63.4%) | **9064 (46.7%)** | **1.98x** |
      | `matmul_reg3` | 300 | 6950 | 6922 | untouched |
      | `attn_apply_cross` | 30 | 1390 | 1395 | untouched |
      | `attn_scores_cross_kt` | 30 | 1229 | 1232 | untouched |
      | everything else (10 kinds) | 780 | 690 | 685 | untouched |
      | **whole graph** | 1170 | **28.119 s** | **19.298 s** | **1.46x** |

      Graph utilisation went 2593 -> 3778 GFLOP/s on 72,915 GFLOP of work.

      **End to end, the baseline request re-run** (33 frames 832x480, 25 steps,
      seed 42, same prompt), `brain wan t2v`, twice from the SAME binary. Both
      runs produced a **bit-identical** MP4 (md5 757d6cf0..., 301,813 bytes), so
      the spread between them is thermal, not algorithmic:

      | phase | baseline | run A (card already hot) | run B (card cold) |
      |---|---|---|---|
      | text encode (umT5-XXL, CPU) | 236.8 s | 395.7 s | 396.4 s |
      | DiT load | 11.5 s | 39.3 s | 41.6 s |
      | **denoise (50 forwards)** | **1763.7 s (35.3 s/fwd)** | **1343.7 s (26.9 s/fwd)** | **982.4 s (19.6 s/fwd)** |
      | VAE decode | 101.6 s | 91.4 s | 84.2 s |
      | total | 2115.6 s (35.3 min) | 1870.7 s (31.2 min) | 1505.4 s (25.1 min) |

      **Read this table carefully; two things in it are not the kernel.**

      1. **The thermal spread is large and it is the dominant uncertainty.** Run
         A began on a card heat-soaked by the kernel benchmarks and ran the
         denoise at 1088-1227 MHz and 89 C. Run B began cool and held **1531 MHz
         at 67 C for the entire denoise** - it never throttled. Same binary, same
         seed, bit-identical output, and 1.37x between them. The baseline was
         itself recorded on a throttling card, so run A is the like-for-like
         comparison (**1.31x on the denoise**) and run B is what the change gives
         when the card is not thermally limited (**1.80x**). A plausible
         second-order effect worth testing deliberately: a forward that is 1.46x
         shorter deposits less heat, so the request is now more likely to stay
         off the thermal limit - but one cold run is not evidence of that, and
         it is NOT claimed here.

      2. **The CPU phases regressed, and NOT from this change.** Text encode
         measured 395.7 and 396.4 s against the baseline's 236.8 s, and DiT load
         39.3 and 41.6 s against 11.5 s - consistently across both runs, so it is
         deterministic rather than contention. Nothing in this change can reach
         either phase: the flash kernels are dispatched only by the DiT
         self-attention inside the denoise graph, and the umT5 encoder runs on
         the CPU backend, which does not register them at all. The ~190 s is a
         pre-existing drift in the working tree (or the box) relative to when the
         baseline row was taken, and it means the TOTAL column understates the
         change. Normalising text+load back to the baseline's 248.3 s puts run A
         at 28.1 min and run B at **21.9 min**. **This wants its own
         investigation and its own re-baseline before the headline table in
         `docs/models/wan.md` is rewritten** - which is why that table was left
         alone here.

      ## Profiler rewrite and a re-baseline under real host load

      `crates/wan/src/bin/wan_bench.rs`'s `dit`/`vae` modes host-bracketed
      per-kernel-KIND groups with a hardcoded `P40_FP32_TFLOPS = 11.76` and a
      local `dit_flop()` formula - the exact anti-pattern `.agents/rules/lessons.md`
      warns about. Rewritten onto `gpu_core::profile::profile` +
      `gpu_core::roof::ensure` + `gpu_core::cost::kernel_cost`, matching
      `vqgan_bench`'s `report()` shape; a `train` mode was added for the host
      trainer (below).

      **The old ranking, checked against the new one, was NOT inverted for
      these two graphs** - both agree within 2 points on every kernel's share
      (e.g. `flash_attn_bidir_reg2` 46.8% old / 45.8% new on the DiT,
      `matmul_reg3` 35.2% old / 35.1% new on the VAE). Grouping by kind vs by
      contiguous submit-run happens to coincide here because each kernel kind
      runs in one contiguous burst per block. What the rewrite actually
      changes: the assumed peak was 11.76 TFLOP/s; the MEASURED roof on this
      card is **10.542 TFLOP/s** - about 10% lower - so every old %-of-peak
      number was overstated by roughly that much, and the new tool reports
      against the measured figure. It also adds bound classification
      (compute/memory) and a DEFECT flag for a kernel underperforming its OWN
      roof by more than a floor, which the old tool had no concept of.

      **Two new DEFECT findings, neither previously recorded:**
      - DiT forward: `attn_apply_cross` (cross-attention's apply step) reaches
        only **4.7% of its own compute roof** (floor 30%) while costing 6.6% of
        the graph - underperforming worse than either GEMM class beside it.
      - VAE decode: `attn_scores_bidir` reaches only **0.3% of its own compute
        roof** (floor 30%) while costing **11.7%** of the whole decode - the
        single worst-utilised kernel measured anywhere in this file, at 9 calls
        totalling 9.4 s. Neither is fixed here; both are recorded as the next
        addressable items ahead of `im2col3d_at`'s 34.0% (pure data movement,
        no defect flag - it is not underperforming a roof, it is simply large).

      **`wan_bench train`** (new): times the HOST trainer
      (`grad::block_forward`/`block_backward`) with min-of-N wall clock, since
      it is CPU rayon code with no device `Step` graph to profile and no CPU
      `Roofs` to grade against. At the full T=14040 shape it did not complete
      in a practical time budget; at T=768 (tiny-cfg scale):
      `block_forward` 4416.7 ms, `block_backward` 9548.8 ms,
      **backward/forward = 2.16x**.

      **Upload/readback, measured directly for the first time**: 0.453 s of a
      20.394 s DiT forward (2.2%) and 0.135 s of an 81.184 s VAE decode
      (0.2%). A 3.6 s/forward figure carried into this validation effort's own
      plan going in was an unmeasured assumption and was wrong by roughly 8x -
      this line item is not worth pursuing.

      **A third and fourth end-to-end run, same shape as the table above** (33
      frames 832x480, 25 steps, seed 42), on the same card:

      | phase | run 1 | run 2 (see caveat below) |
      |---|---|---|
      | text encode | 451.7 s | 535.7 s |
      | DiT load | 16.8 s | 94.4 s |
      | denoise (50 forwards) | 982.4 s (19.6 s/fwd) | 980.4 s (19.6 s/fwd) |
      | VAE decode | 95.3 s | 86.0 s |
      | total | 1547.0 s (25.8 min) | 1697.4 s (28.3 min) |

      **Denoise reproduces to 0.2%** and matches run B above almost exactly
      (982.4 s here vs 982.4 s there) - the strongest evidence yet that the
      GPU-side number is stable and the earlier "1.98x on attention, 1.46x on
      the graph" result is not a measurement artifact.

      **Run 2 was NOT an isolated measurement, and that is the finding.** It
      ran while a second process (`crates/wan/tests/lora_train.rs`'s new
      held-out-loss gate, this same validation effort's own finetune-validation
      work) was doing real host CPU training at 2000-2800% CPU across the
      box's 48 threads. Text encode and DiT load are BOTH CPU/host-bound
      stages (umT5-XXL on the CPU backend; weight deserialize+upload), and
      both are the worst of any run recorded in this file: text 535.7 s (vs
      run 1's 451.7 s, vs the "baseline" row's 236.8 s above), load 94.4 s (vs
      run 1's 16.8 s, vs run A/B's ~40 s above). The GPU-only denoise phase is
      untouched (980.4 s vs 982.4 s) because it does not compete for CPU
      threads with the other process.

      This complicates the earlier claim that the ~190 s regression "is
      deterministic rather than contention" (that claim rested on run A and
      run B agreeing with EACH OTHER, not on either being confirmed
      contention-free). Run 1's own text-encode number (451.7 s) is already
      elevated without a KNOWN concurrent contender, so host contention is not
      shown to be the SOLE cause of the original drift either - it is shown
      to be A real, measurable contributor on top of whatever the original
      drift was. **The isolated, nothing-else-running re-measurement this file
      already asked for is still outstanding** and should be taken before
      `docs/models/wan.md`'s headline table is touched.
- [x] **`capability::Media::Video`**, landed ahead of `caps.rs` as its own unit
      of work, exactly as `.agents/rules/serving-contract.md` section 4 asks --
      extending `Media` rather than adding a side channel. Three things it
      turned out to be:
      * the wire format already existed (`blob::video_blob` /
        `decode_video_hwc`, written for Qwen3-Omni's video INPUT) and was
        tagged `Media::Bytes` with a comment saying "there is no
        `Media::Video`". So this is a retag plus a rename to `decode_video`
        (the `decode_image` parallel), not a new codec. `decode_video` still
        accepts an untyped `Bytes` payload, the same leniency `decode_hwc`
        gives images, so a client that sends no media tag is unaffected.
      * **the D-Bus layer needed no change at all**: it carries the media kind
        as its `Media::parse`-able string, so a new variant rides the existing
        frames. That is the design working -- it is also why the change had to
        be `Media`, not a bespoke channel.
      * the CLI DID need real work: `caps_cli`'s `load_blob`/`save_blob` had a
        `_ => raw bytes` catch-all, so `--out video=out.mp4` would have written
        f32 frames into a file no player opens (the exact bug the audio arm
        exists to fix). Both arms now go through `imaging::video`.
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
- [x] **The serving contract** (`.agents/rules/serving-contract.md`), all five
      obligations: `crates/wan/src/caps.rs` (a weights-free manifest with one
      `t2v` action whose defaults ARE `WanConfig`'s), `crates/cli/src/
      resident_wan.rs` (residency), the D-Bus surface (unchanged - `t2v` is a
      `Subscribe` job and a `Cancel`able one) with `examples/videogen/`, and a
      catalog row so `brain caps` / `brain do brain/wan t2v` reach the same
      manifest the scheduler serves. Four things worth keeping:
      * **Cancellation is the obligation that bites here.** The measured run is
        57.5 minutes; `pipeline::denoise` now polls `inv.cancel` once per step
        (not per forward - a forward is one submit of the whole block stack and
        is not interruptible from the host), so the worst-case abort latency is
        one step, ~1.5 min at 480p.
      * **The hot cache is the DiT alone** (`pipeline::HotDit`), keyed on
        `(variant, latent extent, device)` - the only things that fix the built
        graphs. The umT5 encoder cannot be resident (22.72 GB) and the VAE is
        508 MB against the DiT's 5.7 GB, so caching either would trade the
        "three models are never resident together" staging for very little.
        `generate` (one-shot, the CLI) still frees each stage before the next
        allocates; `generate_hot` (served) holds the DiT across the VAE decode
        and `resident_wan::estimate` budgets both together.
      * **`run_batch` is sequential, and says so.** What the batch shares is the
        expensive thing - one resident DiT for every job at the key. What it
        cannot share is the forward: the engine records one graph for one latent
        volume and holds ONE context buffer, which the CFG loop already
        re-uploads between its own two forwards. A real batched forward means a
        batch axis through the engine, RoPE and the flash slabs; at 46 s per
        forward that is a measurement-led change, not a wiring one.
      * **`brain/wan` needs no `ARCH_HANDLER_CATALOG_ID_OVERRIDES` row**,
        unlike flux2: the catalog id is `brain/<arch id>`, the pattern
        `caps_cli` already assumes.
- [x] **GGUF import**: one `GgufArchitectureImporter` impl plus one line in
      `crates/cli/src/gguf_import.rs`'s `IMPORTERS`, delegating to
      `wan::import::import_gguf`. DiT only, like s3dit's - a Wan GGUF is the
      transformer alone, so it replaces one of four roles rather than standing
      on its own. Two things it does differently:
      * **No `AMBIGUOUS_TAG_EXCEPTIONS` row.** s3dit needs one because its tag
        (`lumina2`) is shared with real Lumina2 releases and `brain_arch`
        therefore refuses to claim it; `wan`'s id IS its GGUF spelling, so
        `by_gguf("wan")` resolves and the registry's own drift test passes
        without an exception.
      * **The variant is read off the tensor shapes**
        (`dit_config_from_shapes`), not off a KV field: a repacked GGUF can
        carry any metadata, but it cannot carry the wrong
        `patch_embedding.weight` and still be loadable. `(dim, num_layers)`
        separates the T2V tiers; a 36-channel patch embedding is an I2V
        checkpoint and is refused by name rather than imported into a pipeline
        that does not exist. Both name spaces resolve, since the diffusers
        export renames every leaf but that one tensor.
      **Covered on real bytes**: `crates/wan/tests/gguf_import_real.rs`
      (`BRAIN_WAN_GGUF`, `brain_testutil::skip` when absent) runs the whole
      dequantize -> `import_dit` -> safetensors path against the released
      `city96/Wan2.1-T2V-14B-gguf` Q3_K_S file - 1095 tensors, set-equal to
      `dit_manifest(t2v_14b)` in both directions, the 5-D `patch_embedding`
      rank GGUF stores above `GGML_MAX_DIMS`, and Q3_K values pinned bit-exact
      against an independent reimplementation of `dequantize_row_q3_K`. The
      synthetic tests (dispatch by tag, variant derivation over the full
      manifest in both spellings, the refusals) stay as the fast lane.
- [x] **Training + LoRA**, HOST ONLY -- `grad.rs` (one block, fwd + analytic
      bwd), `modelgrad.rs` (the whole DiT under flow-matching velocity MSE),
      `lora.rs`, `finetune.rs`, plus `gradcheck::check_wan` and
      `check_wan_conditioning`. One implementation generic over the float type:
      the f64 instantiation is the FD oracle, the f32 instantiation is the
      trainer, so oracle and trainer cannot drift.
      * **All four gates, with numbers.** Block FD (`tests/block_grad.rs`, 27
        weight tensors **plus the three input adjoints** `dx`, `dctx`, `de0`):
        worst error **1.8e-9** against a 1e-4 gate. Whole-model FD
        (`check_wan`, all **69** tensors of the checkpoint manifest): worst
        **1.7e-8** against a 1e-3 gate. Host f32 forward vs the device forward
        (`tests/host_forward_parity.rs`): **cosine 1.000000000** on the P40's
        flash path AND on the CPU JIT's chunked path, rel_l2 1.7e-7. LoRA
        (`tests/lora_train.rs`): exact no-op at init (bit-identical weights),
        loss 0.2373 -> 0.1415 over 40 rank-4 steps, and fold-vs-apply
        **bit-equal**. Overfit (`tests/overfit.rs`): 0.549 -> 3.9e-6 in 400
        Adam steps over every parameter.
      * **A directional check alone would have been a false certificate here.**
        This model has three folded/shared conditioning sites -- `e0` summed
        across the whole block stack, `e` feeding BOTH the head site and
        `time_projection`, and one `ctx` slab read by every block's
        cross-attention -- and a contraction onto one random direction can be
        small while a *share* of such a gradient is missing.
        `check_wan_conditioning` therefore does per-ENTRY differences on the
        four tensors that sit exactly at those folds
        (`time_projection.1.bias` = `d e0`, `time_embedding.2.bias` = `d e`,
        `text_embedding.2.bias` = summed `dctx`, and every
        `blocks.{l}.modulation`): 528 entries, worst 4.7e-11.
      * **The block backward differentiates the UNFOLDED modulation**, and the
        six-vector grad it returns is ONE vector used twice -- `d(modulation)`
        and the block's contribution to `d e0` -- because the fold's operand is
        their sum. `block_grad.rs` asserts that identity directly (perturbing
        `e0` and perturbing `modulation` move the loss identically).
      * **Wan fuses nothing**, so the "one adapter pair per fused slice" rule
        collapses: `self_attn.{q,k,v,o}`, `cross_attn.{q,k,v,o}`, `ffn.0` and
        `ffn.2` are ten independently-named tensors, each pair covers a whole
        tensor at offset 0, and that is why fold-vs-apply can be asserted
        bit-equal rather than close.
      * **The dataset is `data::episode`, not a new format.** A video clip and
        a recorded episode are the same object: a run of frames that must never
        be sampled across. `finetune::ClipSet` is an episode dataset (one
        episode per clip) plus one `captions.json`; the windowing is
        `sample_window`/`iter_windows` verbatim.
      * **No device trainer.** The host f32 path is the trainer, as `flux2` is
        -- practical for short adapter runs at small latent extents, not a path
        to a 1.3B training run. `s3dit`'s `devgrad.rs`/`train.rs`/`shard.rs` is
        the precedent to follow when that ladder starts, and it is a separate
        measurement-led change.
      * Not wired to a `lora_train` capability action yet (flux2 and s3dit both
        expose theirs that way); `finetune::run` is the entry point.
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

**Auto-fetch reached that bar** with `modelstore::recipe::WanRecipe` + a
`convert_wan` finish step + a `wan` arm in `model_dir::resident_for_compound`.
The failure it fixes is worth recording, because the diagnosis was not the
obvious one: the flagless command reported
`Wan-AI/Wan2.1-T2V-1.3B: unsupported architecture "t2v"`, which reads like a
missing `brain_arch` row. It was not. That repo DOES ship a root `config.json`
- 249 bytes, `{"_class_name": "WanModel", "model_type": "t2v", ...}` with no
`architectures` key - so `TransformersRecipe` (the catch-all) claimed it,
`declared_architecture` fell back to `model_type`, and the gate rejected
`"t2v"`. The fix is a recipe with first refusal, not a wider architecture
table: adding `"t2v"` to `Arch::hf` would have made the catch-all fetch the
curated transformers file set, which for this repo is `config.json` and
nothing else.

Two shapes it handles, confirmed against the live HF listing rather than the
local checkout (which is a deliberately partial `allow_patterns` download and
therefore evidence of nothing): the 1.3B tier's single
`diffusion_pytorch_model.safetensors`, and the 14B tiers' six-way shard set
plus index - where the `dit` role becomes the repo directory itself, since
`checkpoint::safetensors::read_model_dir` follows the index there and reads
only the shards, never the two `.pth` siblings.

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
