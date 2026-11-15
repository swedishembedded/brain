# flux1 - roadmap

Scope: FLUX.1 / Kontext text-to-image (`crates/flux1`).

FLUX.1 / Kontext (`crates/flux1`): the 12B MMDiT (19 double-stream +
38 single-stream blocks, per-block modulation, 3-axis interleaved RoPE),
its sampler loop and VAE/text-encoder glue (`pipeline::Flux1::generate` -
T5-XXL context + CLIP-L pooled conditioning, BFL's own linear
`calculate_shift` schedule, a 16-channel VAE decode), and the full serving
contract: `flux1::caps` (`text2image`), `resident_flux1::Flux1Resident`
(`BRAIN_FLUX1_DIR`), a `catalog.rs` entry, D-Bus `Run`, and
`examples/imagegen/flux1_generate.py`.

Forward parity is verified against the reference implementation at reduced
depth (fp32) and full depth (int8, single GPU).

## Placement

`pipeline::part_needs` prices the DiT and T5-XXL for the production placement
policy. Measured on a Tesla P40 (`tests/it/placement_real.rs`, driver figure
less the context): int8 DiT at 256 px 13,948 MiB against a 12,978 MiB
estimate (the difference is the driver's block rounding, inside the
headroom's allocator share); T5-XXL at 512 tokens, built output-only
(`t5encoder::model::Taps::Output`), 18,600 MiB against 18,600 MiB. On two
idle 24 GiB cards the plan puts each part on its own card; on one card the
DiT keeps it and T5-XXL takes the host tier. Knowledge #458 records why the
mid-run cancellation proof used to need a `te_device` pin.

## Not yet done

- [ ] Backward / gradient check (`check_flux1`)
- [ ] Full-depth fp32 parity - the full model does not fit on one GPU at
      fp32; needs more memory or sharding
- [ ] The non-fast (materialized) attention path has no permanent numeric
      gate - selection turns on `gpu.caps().workgroup_reductions`
      (`model.rs:319`), so the fallback is only reachable on backends
      without workgroup reductions and is checked by hand, not by an
      automated test
- [ ] Kontext image *editing*, img2img and LoRA - text-to-image only today
- [ ] Batch > 1 - every request is its own multi-step sample, so
      `run_batch` is the serial default (documented in-file, as the
      serving contract requires)
- [ ] Performance profiling (no speed claim is made yet)

The pipeline glue has **no end-to-end fixture in this workspace** to verify
it against - see `pipeline.rs`'s module docs for the honest scope of what is
and is not checked. Parity covers the transformer, not a full generation.

There is deliberately no `brain flux1` CLI module: the model is reached
generically as `brain flux1 text2image` through the capability dispatch,
which is what the serving contract asks for.
