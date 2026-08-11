# pulid — roadmap

PuLID-FLUX identity conditioning: turns a face embedding and image-tower features into ID tokens that are cross-attended into the FLUX.1 diffusion backbone at fixed points, to preserve a person's identity in generated images.

## Not yet done

- [ ] End-to-end image generation — the FLUX.1 backbone this depends on has no sampler loop or VAE glue yet, so no image can be produced or an identity-fidelity number measured.
- [ ] `id_weight` / `start_step` sweep — depends on the sampler loop above.
- [ ] Full-depth conditioning run across all injection sites — only a reduced-depth run has been exercised; an int8 full-depth run is possible in principle but has not been done.
- [ ] Image preprocessing for the identity condition: the face alignment/parsing step needed to build the vision-tower input from a raw photo does not exist in this codebase yet.
- [ ] Wiring the face-embedding computation (photo in, embedding out) into this crate — the underlying component exists elsewhere but isn't connected here.
- [ ] Multi-image identity conditioning — only a single reference embedding per identity is supported.
- [ ] Backward pass / gradient check for the adapter.
- [ ] Serving contract — no capability manifest, no residency adapter, no batched request handling, no D-Bus surface, no CLI.

The current forward pass reuses buffers across layers in an inference-shaped way, so a backward pass needs a distinct training-mode forward with per-layer buffer allocation rather than a flag on the existing one.
