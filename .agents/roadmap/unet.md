# unet — roadmap

Port of the SDXL UNet2DConditionModel diffusion backbone, plus the shared
discrete-step diffusion schedulers (DDIM, Euler, Euler-ancestral,
DPM-Solver++). Forward-pass parity against the reference implementation is
verified; the surrounding stack is not yet built.

## Not yet done

- [ ] Backward pass / gradient check for the UNet graph
- [ ] Full serving contract: a capability provider, a residency adapter, a
      real batched `run_batch`, D-Bus exposure, and an examples client
- [ ] Batch > 1 support (needed for classifier-free guidance as a single
      batched forward)
- [ ] ControlNet integration at the down-block residual and mid-block
      injection points
- [ ] End-to-end pipeline wiring: the VAE, the text encoders, a tokenizer
      caller, and a sampler loop tying them together
- [ ] INT8 quantization

The forward graph is built entirely from existing convolutional and
transformer building blocks rather than new kernels, so the backward pass is
expected to compose existing adjoints for those pieces rather than needing
new kernel work.
