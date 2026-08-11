# zimage — roadmap

Z-Image (S³-DiT) text-to-image diffusion transformer, with fp32/int8/sharded
device engines, a VAE, a flow-matching scheduler, backward + LoRA fine-tuning,
and tiered weight-residency streaming so the checkpoint never has to be
loaded whole. Forward and backward parity against the reference are verified.

## Not yet done

- [ ] A true batched `run_batch` for the serving contract
- [ ] A runnable examples client over D-Bus
- [ ] Native lower-precision (bf16) device weight binding for the windowed
      fp32 path — fp32 inference currently streams full fp32 weight tiles
      per block, which is disk-bound rather than compute-bound
- [ ] Device-resident block chaining, so the reference path doesn't
      round-trip through host memory between blocks
- [ ] Unify the flow-matching dynamic-shift calculation with the shared
      implementation used elsewhere
- [ ] Parity coverage for short (padded) prompts against the reference

The checkpoint is far larger than fits in memory at once, so weights stream
from disk one tensor at a time rather than the model loading whole; until a
native lower-precision device format exists, this makes fp32 inference
disk-bound rather than compute-bound.
