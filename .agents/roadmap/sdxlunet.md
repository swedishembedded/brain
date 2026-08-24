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

- [ ] Backward pass / gradient check for the UNet graph (`check_unet`).
      **The cheapest unmet backward in the repo** - the forward is built
      entirely from existing conv/transformer blocks, so this composes
      existing adjoints rather than needing new kernel work. It also
      unblocks `check_controlnet`.
      Use `gradcheck::elementwise_check`, not `directional_check` alone, for
      the timestep embedding: it is added into every resnet, so it is a
      shared parameter accumulated over many contributors - exactly the
      folded-parameter class where best-of-n directional checks pass while
      the gradient is partially wrong (see `.agents/rules/lessons.md`).
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
