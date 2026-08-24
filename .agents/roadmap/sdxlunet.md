# sdxlunet - roadmap

Port of the SDXL UNet2DConditionModel diffusion backbone, plus the shared
discrete-step diffusion schedulers (DDIM, Euler, Euler-ancestral,
DPM-Solver++ x {eps, v-pred}) in `diffusion::discrete`.

Forward parity is verified against the reference implementation at 165
comparisons / 0 failed, worst cosine 0.9999999999, `out.sample` cosine
1.0000000000 and rel_l2 3.258e-6 (both are asserted - cosine alone is
scale-invariant). The schedulers are gated at 66 checks / 0 failed. The
end-to-end pipeline exists (`pipeline::Sdxl::generate` - dual CLIP
conditioning, a discrete Euler step, CFG, VAE decode), as does the serving
contract: `sdxlunet::caps` (`text2image`), `resident_sdxl::SdxlResident`, a
`catalog.rs` entry, D-Bus `Run`, `examples/imagegen/sdxl_generate.py`.
ControlNet integration is wired via `Unet::new_controlled` /
`Unet::run_with_control` (`model.rs:809`) and consumed by `crates/controlnet`.

## Not yet done

- [x] Backward pass / gradient check for the UNet graph (`check_unet`).
      `crates/sdxlunet/src/train.rs` (`UnetTrainer`) + `gradcheck::check_unet`
      and `check_unet_conditioning_elementwise`.

      **The "cheapest unmet backward" framing was half right and dangerously
      half wrong.** Right that no kernel was needed - `matmul_dx`/`matmul_dw`,
      `bias_grad`, `gelu_erf_bwd`, `layernorm_dx`/`_dgamma`/`_dbeta`,
      `add_chan_bcast_dv`, `concat_split` and the `attn_bwd_*_cross` quartet all
      already existed for the decoder LMs. Wrong that the transformer half was
      merely un-checked: `Rec` emitted ALL of it with
      `vae::blocks::Builder::push_step`, which records nothing on the reverse
      tape, and a pushed step mid-chain gives every parameter upstream of it a
      silent ZERO gradient (that method's own doc says so). So this was latent
      breakage waiting for the first `.backward()`, not a missing test.

      The fix is in the SHARED builder, not this crate: `vae::blocks` grew real
      recorders (`linear`, `layernorm`, `gelu_erf`, `mul`, `add_chan`,
      `self_attn`, `cross_attn`, plus a tape entry on the existing `concat`),
      seven new `Op` variants and their adjoints, and an `XformerIds` seam for
      the caller's own forward slots (`vae::blocks` cannot register LayerNorm/
      GEMM/cross-attention itself - the caller already does, and a second
      registration of one kernel name is what the CPU JIT rejects). `Rec` now
      routes through them. Purely additive: `vqgan`, `codeformer`, `rrdbnet`
      and `AutoencoderKL` record none of the new variants and are unchanged.

      Two forward changes train mode requires, both of which run fine and are
      wrong (see `.agents/rules/lessons.md` #55): flash attention never
      materialises the softmax its adjoint binds, so a recording builder takes
      the materialised path regardless of device; and each attention SITE needs
      its own `probs` slab, where the eval graph shared one.

      The elementwise half runs at a NARROWED conditioning chain
      (`time_embed_dim = 16`). At the full tiny config `elementwise_check`'s
      `2*numel` forwards is ~100k full UNet passes - a gate nobody would run,
      which this repo counts as no gate. 16 catches the same defect class: all
      17 resnets still consume `silu(emb)`.

- [ ] `check_controlnet` - now unblocked. ControlNet's trainable copy IS the
      UNet's blocks (recorded by `sdxlunet::model::Rec`), so its backward
      composes the same tape; what it additionally needs is an adjoint for the
      residual injection points and the conditioning-image embedder.
- [ ] Batch > 1 support - needed for classifier-free guidance as a single
      batched forward instead of two passes. Today every request is its own
      multi-step sample, so `run_batch` is the serial default (documented in
      `resident_sdxl.rs`)
- [ ] INT8 quantization
- [ ] Performance: 2198 dispatches and 4.06 s per forward at the native
      128x128 latent, unoptimized and not yet profiled per kernel kind

**2 567 463 684 params = 10.27 GB fp32**, so it fits one 24 GB P40.

Two GroupNorm epsilons live in one graph (1e-5 in the resnets, 1e-6 inside
every transformer), which is why `vae::blocks::Builder` gained `set_eps` -
a single-epsilon assumption here is silently wrong, not a shape error.
