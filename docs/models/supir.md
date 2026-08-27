# SUPIR

SUPIR ("Scaling Up to Excellence", CVPR 2024) is a photo-realistic blind image
restoration model: give it a low-quality photo (compression artefacts,
downscaling, noise, blur) and it produces a plausible high-quality
reconstruction, driven by a frozen SDXL 1.0 base UNet plus a 1.24B control
trunk and 12 adaptor modules trained specifically for restoration.

## Status

**Not yet implemented.** The architecture id is registered
(`crates/arch`) and the crate exists as a placeholder; the port itself has not
started. See `.agents/roadmap/supir.md` for the full architecture spec and the
staged implementation plan.

## Support

| Capability | Supported |
|---|---|
| Inference             | [ ] |
| LoRA fine-tune         | [ ] |
| Full backbone fine-tune | [ ] |
| INT8                   | [ ] |
| CLI (`brain <arch> <action>`)       | [ ] |
| HTTP API               | [ ] |
| D-Bus                  | [ ] |
| Batched serving        | [ ] |

## Licence - read before fetching weights

The SUPIR **weights** are released under the SUPIR Software License Agreement
(© 2024 SupPixel Pty Ltd): **non-commercial only**. The licence's definition
of commercial use is broad and expressly includes SaaS deployment, selling
processed images, product integration, and using SUPIR's output as training
data for another model. There is no official HuggingFace repo; the mirrors
that exist are unofficial. Derivative works of the weights are prohibited
without written permission from the licensor.

This is a constraint on the **released checkpoints**, not on brain's own code
in this crate (Apache 2.0, same as the rest of the workspace). Point brain at
weights you have obtained yourself and cleared for your own use - there is no
`default_ref` auto-fetch for this architecture, deliberately.
