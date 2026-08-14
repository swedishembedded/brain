# controlnet - roadmap

Backbone-agnostic `ControlAdapter` seam plus an SDXL `ControlNetModel` whose
per-injection-point residuals reproduce the reference implementation; a brain
UNet can consume its residuals. Forward parity is verified against the
reference implementation.

## Not yet done

- [ ] Backward / gradient check (no training path exists at all)
- [ ] The full serving contract: capability provider, residency adapter,
      `run_batch`, D-Bus surface, example client
- [ ] A fused on-device path - today residuals round-trip through the host
      between the ControlNet and UNet graphs even though both already run on
      one device with one kernel set
- [ ] Parity/validation at SDXL's native resolution (only verified at small
      latent sizes so far)
- [ ] `guess_mode` (per-injection-point scale ramp) and
      `global_pool_conditions`
- [ ] Depth-conditioned ControlNet wiring from `crates/zipdepth`'s own depth
      predictor (the adapter function exists; needs a depth-conditioned
      checkpoint to validate against)
