# controlnet - roadmap

Backbone-agnostic `ControlAdapter` seam (`adapter.rs`: named `InjectionPoint`s
matched by name and element count, so a permutation type-checks rather than
silently producing a plausible image) plus the SDXL `ControlNetModel` that is
its first producer. The trainable copy *is* the UNet's blocks, recorded by
`sdxlunet::model::Rec`; adds no kernel beyond `scale_chan`.

Residual-parity-gated against a hooked diffusers `ControlNetModel` at 140
comparisons / 0 failed, worst 1−cos 1.914e-11, on both a P40 and
`BRAIN_DEVICE=cpu`. The serving contract is met: `controlnet::caps`
(`text2image`, its own sampler loop over `Unet::new_controlled` +
`Unet::run_with_control`), `resident_controlnet::ControlnetResident`, a
`catalog.rs` entry, D-Bus `Run`, `examples/imagegen/controlnet_generate.py`.

## Not yet done

- [ ] Backward / gradient check (`check_controlnet`) - no training path
      exists at all. **`check_unet` is now closed, so the prerequisite is
      met**: the trainable copy is the same `sdxlunet::model::Rec`-recorded
      blocks, and those now record onto `vae::blocks`' reverse tape rather
      than being `push_step`ed past it, so `Trace::backward` already
      differentiates them. What is left is specific to this crate: an adjoint
      for the residual injection points (each is an `Op::Add2` on the UNet
      side, so the gradient reaching a residual is just `d` of that buffer -
      `Reverse::d()` exposes it) and for the conditioning-image embedder's
      own conv stack (already `vae::blocks`, so also already on the tape).
      Model it on `crates/sdxlunet/src/train.rs`.
      Re-assessed while closing `crates/supir`'s own training path
      (`supir::train::SupirTrainer` + `gradcheck::check_supir`, the same
      class of gate this bullet describes, over the same `Rec`/`Trace`
      machinery): still genuinely open, not a small mirror of
      `SupirTrainer`, for two concrete reasons found while building that
      trainer. First, `ControlNet::new` records via `Rec::new` (eval mode)
      only - there is no `Rec::new_train` wiring at all yet. Second,
      `scale_buf` (this crate's `model.rs`) applies `conditioning_scale` via
      `Builder::push_step`, which records nothing on the reverse tape by
      design (the same silent-zero-gradient trap this crate's own module doc
      already names as the reason `controlnet::model::scale_buf`'s idiom was
      disqualified for SUPIR's `control_scale` lerp) - so the residual
      outputs, as recorded today, are NOT on the differentiable path at all;
      closing this needs either a real tape op for the scale (SUPIR's
      `Op::Mix`/`edm_mix` a=1 case would do it, since `y = 1*x` is the
      pass-through a training-time build wants anyway) or a train-mode build
      that skips the scale multiply outright (`conditioning_scale` is a
      runtime-only knob, always 1.0 during training upstream). Third, the
      loss head SUPIR's/UNet's own `mse_value`/`mse_grad` pair assumes ONE
      output buffer; `ControlNet::run` returns a `Residuals` map of several
      DIFFERENT-shaped buffers (one per injection point plus mid), so the
      loss wiring is not the same one-line MSE-over-one-buffer `UnetTrainer`/
      `SupirTrainer` share - a real second trainer shape, not a mirror.
- [ ] A fused on-device path - today residuals round-trip through the host
      between the ControlNet and UNet graphs even though both already run on
      one device with one kernel set
- [ ] Parity at SDXL's native 128x128 latent. `tests/parity.rs` gates a
      32x32 latent and a deliberately non-square 24x16 one (the non-square
      case is not redundant: at a square latent an H/W transposition is
      invisible). The resolution SDXL actually generates at is untested
- [ ] `guess_mode` (per-injection-point scale ramp) and
      `global_pool_conditions`
- [ ] Depth-conditioned ControlNet wiring from `crates/zipdepth`'s own depth
      predictor (the adapter function exists; needs a depth-conditioned
      checkpoint to validate against)
- [ ] Batch > 1 and INT8 - every request is its own multi-step sample, so
      `run_batch` is the serial default (documented in-file)

`caps` is its own sampler loop rather than a composition on top of
`sdxlunet::pipeline::Sdxl`, because that pipeline has no seam for a per-step
residual - see `caps.rs`'s module docs.
