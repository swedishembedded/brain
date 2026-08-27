# LLaVA-1.5-13B

LLaVA-1.5-13B pairs a CLIP-L/14@336 vision tower with a Vicuna-1.5 (LLaMA-2
13B) decoder to answer questions about, or produce a detailed caption for, an
image. It is brought into this workspace as [SUPIR](supir.md)'s optional
captioner - SUPIR's restoration prompt can come from LLaVA describing the
low-quality input, or from a prompt supplied directly.

## Status

**Not yet implemented.** The architecture id is registered
(`crates/arch`) and the crate exists as a placeholder; the port itself has not
started. See `.agents/roadmap/llava.md` for the staged implementation plan.

## Support

| Capability | Supported |
|---|---|
| Inference (`caption`) | [ ] |
| INT8                   | [ ] |
| CLI (`brain <arch> <action>`)       | [ ] |
| HTTP API               | [ ] |
| D-Bus                  | [ ] |
| Batched serving        | [ ] |
