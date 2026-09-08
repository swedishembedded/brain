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

- [x] Backward / gradient check (`check_controlnet`) - **closed**.
      `crates/controlnet/src/train.rs` adds `ControlNetTrainer`: the
      trainable copy (`Rec::new_train`, `"controlnet."`-prefixed) + the
      frozen backbone, with the residual `Op::Add2` injected directly onto
      the SAME tape at every skip site and the mid block - the FULL closed
      loop, not the isolated-trainer fallback the item's brief pre-approved
      as a fallback. `crates/gradcheck/src/controlnet.rs` adds
      `check_controlnet`/`check_controlnet_elementwise`, gated by
      `controlnet::tests::{controlnet_gradients_match_finite_differences,
      controlnet_conditioning_gradients_match_per_entry_finite_differences}`.
      Measured at `eps = 2.5e-4` (`check_unet`'s own starting value, needed
      no adjustment): `check_controlnet: 150 tensors, max_rel = 1.395e-1`
      (directional, `ControlNetConfig::tiny()`, `(atol,rtol)=(4e-3,8e-2)`,
      `bad.is_empty()`); `check_controlnet_elementwise: 128 entries, max_rel
      = 8.205e-1` (`controlnet.time_embedding.linear_2.weight` +
      `controlnet.add_embedding.linear_2.weight`, `time_embed_dim` narrowed
      to 8 for cost, same tolerance, `bad.is_empty()`) - the high elementwise
      `max_rel` is every outlier sitting at fp32 noise near zero (e.g.
      `add_embedding.linear_2.weight[51]`: analytic `-1.32e-4`, numeric `0`,
      well under the `4e-3` absolute floor), not a missing contribution.
      A tape-coverage floor (`checks.len() > 100`, against a measured 150)
      guards the "fully discarded tape passes vacuously" failure mode the
      item called out as the single most likely regression.

      **Why "`check_controlnet` follows directly from `check_unet`"
      undercounted the real cost** (the original roadmap entry above,
      written after `check_unet` closed, before `check_supir` existed):
      `ControlNet::new` recorded via `Rec::new` (eval mode) with no
      `Rec::new_train` wiring at all; `scale_buf` applied
      `conditioning_scale` via `Builder::push_step` (a discarded-tape bug,
      fixed in the same commit as item 7's `rrdbnet` fix - see below);
      `ControlNet::run` returns a `Residuals` MAP of several
      different-shaped buffers, not the one output buffer `UnetTrainer`/
      `SupirTrainer`'s `mse_value`/`mse_grad` pair assumes; and
      `sdxlunet::model::Unet::record_into`'s own `control: bool` branch -
      the obvious place to look for the injection - turned out to be the
      wrong seam for training: it allocates its OWN `control_in` storage
      buffers and expects a HOST `write_f32` (`Unet::run_with_control`'s own
      contract), which severs the tape between the two models. Closing this
      needed a genuine second trainer (`ControlNetTrainer`), not a mirror of
      `SupirTrainer` with different tensor names - specifically a
      hand-rolled copy of `record_into`'s up-path tail (~40 lines) wired
      against ControlNet's own zero-conv outputs instead of `record_into`'s
      internal buffers, because there is no `pub fn` seam to hand
      `record_into` externally-produced, still-on-the-tape buffers.

      **The tensor-name collision.** `ControlNetConfig::tensor_manifest` is
      a FILTER of the backbone's own manifest (`is_controlnet_half`), not a
      re-derivation with a prefix, so `controlnet::init::init_weights`
      returns names (`"time_embedding.linear_2.weight"`, ...)
      BYTE-IDENTICAL to the frozen backbone's own. `crates/supir` merges a
      frozen backbone with its own delta the same way and does not hit this,
      only because SUPIR's OWN trunk tensors already carry a
      `"control_model."` prefix at `supir::init::init_weights` time. A naive
      merge here would silently collide on every shared name (a `HashMap`
      overwrite) and train the wrong or only one copy. Fixed in
      `controlnet::train::tensors_for`: ControlNet's half is inserted under
      an explicit `"controlnet."` prefix at merge time, `Rec::set_prefix`
      handles it for every `Rec`-level convenience call
      (`conditioning`/`down_path`/`mid_block`), and the raw
      `vae::blocks::Builder` calls (the conditioning-image embedder, the
      zero-convs) get the prefix baked into their literal name strings.
      `trainable_names()` in the gradcheck filters on that same
      `"controlnet."` prefix, mirroring `check_supir`'s own
      `"control_model."`/`"project_modules."` filter. A regression test
      (`train::tests::tensors_for_has_no_lost_entries_from_the_merge`) pins
      the merged tensor count to `backbone_manifest.len() +
      controlnet_manifest.len()`, so a future edit that drops the prefix
      fails loudly on a count mismatch instead of silently training the
      wrong tensor.

      **Injection order.** `crate::adapter`'s own doc warns that a
      same-shaped permutation of injection points (`ControlNetConfig::tiny()`
      at an 8x8 latent has two such same-shaped pairs: index 0/1 both
      `(32,8,8)`, index 3/4 both `(64,4,4)`) type-checks and runs wrong.
      `ControlNetTrainer` does not re-derive injection order from
      `injection_points()`/`Residuals` at all - both the trainable copy's
      residual list and the frozen backbone's own skip list come from the
      literal SAME `Rec::down_path` call (once prefixed, once not) against
      the same backbone config, so index `k` means "the `k`-th skip
      `down_path`'s own walk pushed" identically on both sides by
      construction. There is no name lookup or reordering step for a swap
      to hide in, so `ControlNetConfig::tiny()` did not need a
      shape-distinguishing tweak.

      **The deliberately dropped `scale_chan_dg` (dscale) adjoint.**
      `conditioning_scale` is a per-request `ParamSpec` (`crate::caps`), a
      one-element device buffer (`model.rs`), NOT a manifest tensor -
      `ControlNetConfig::tensor_manifest` never lists it and
      `controlnet::init::init_weights` never emits it, so a
      `read_weight`/`read_grad` lookup on it would simply fail. Only the
      `dL/dx = dy · scale` PASS-THROUGH direction was implemented
      (`vae::blocks::Op::ScaleChan`, `Builder::scale_chan`, `grad.rs`'s
      matching reverse arm, reusing the ALREADY-REGISTERED `scale_chan`
      backward slot `Op::Gn`'s own `dyg = dy · gamma` dispatches). No
      `dscale` kernel exists because nothing in this tree would ever read
      it - implementing one would be dead code with no consumer, which is
      exactly what the item's brief called out in advance.

      `crate::model::ControlNet::new`'s own `scale_buf` now calls
      `Builder::scale_chan` instead of `Builder::push_step` too, so
      inference-mode residuals are unaffected but the tape is no longer
      silently discarded if anyone later records a `ControlNet` in train
      mode through that path directly.
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
