# flux1 — roadmap

FLUX.1 / Kontext (`crates/flux1`): one transformer evaluation. There is no
sampler loop, no VAE glue, no text-encoder call, no CLI subcommand and no
serving surface — "FLUX.1 works" is not a claim this crate supports yet.
Forward parity is verified against the reference implementation at reduced
depth (fp32) and full depth (int8, single GPU).

## Not yet done

- [ ] Backward / gradient check (`check_flux1`)
- [ ] Full-depth fp32 parity — the full model does not fit on one GPU at
      fp32; needs more memory or sharding
- [ ] The non-fast (materialized) attention path has no permanent numeric
      gate — it is only reachable on backends without workgroup reductions
      and is checked by hand, not by an automated test
- [ ] The entire serving contract: capability provider, residency adapter,
      `run_batch`, D-Bus surface, example, CLI
- [ ] Performance profiling (no speed claim is made yet)
