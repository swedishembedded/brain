# Gemma-4 unified text tower (op sequence only, not yet a usable encoder)

The text encoder [LTX-2.5](ltxv.md) conditions its diffusion transformer on.
The released checkpoint is a *unified* text, vision and audio model, but LTX-2.5
only ever calls its text side, so this crate implements only the text-only
forward path through the decoder-layer stack. There is no vision tower, no
audio tower, and no image, video or audio token handling.

## Support

| Capability | Supported |
|---|---|
| Op-sequence parity at tiny dims | [x] |
| Real-checkpoint import | [ ] |
| Inference on real weights | [ ] |
| INT8 | [ ] |
| CLI | [ ] |
| HTTP API | [ ] |
| D-Bus | [ ] |
| Training | [ ] |

**Read that table before relying on this.** What exists is the architecture,
proven correct in shape and order against the reference at small dimensions.
What does not exist is the ability to load the real weights and encode text.
The registry row is a name reservation, which is what a registry row is; it is
not a claim that the model runs.

## What is actually proven

The forward is ported from `transformers.models.gemma4_unified` and gated
against goldens dumped from that reference at the same tiny dimensions, with
every flag that changes the op sequence set to its real LTX-2.5 value. So the
things this establishes are structural:

- the **5:1 alternation** of sliding-window and full attention across the layer
  stack,
- the **two RoPE bases** - one construction for sliding layers, another for
  full layers - and which existing kernel each reuses,
- the `attention_k_eq_v` variant that global layers take,
- the 49-hidden-state **aggregate-embed projection** LTX-2.5 reads.

This is the same approach `crates/ltxv`'s own tiny-config DiT milestone used,
and it is deliberate: an op sequence can be verified exactly at dimensions that
fit anywhere, and getting it wrong is the failure mode that survives a
plausible-looking output.

## What is not proven, and why

Real-weight import is out of scope rather than merely unfinished. The
checkpoint is `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`: 12 B parameters,
26 GB in bf16, and gated upstream. Nothing here has seen it. So the port
establishes the operation sequence, not fidelity to the released weights, and
the two are genuinely different claims - a tiny-dimension parity run cannot
catch a tensor-naming error in an importer that does not exist yet.

The architecture id is brain-defined rather than taken from llama.cpp's
vocabulary. `transformers.models.gemma4_unified` carries a 2026 date in its own
license header, and this repo has no local llama.cpp checkout to range-check
against, so the naming rule's "no entry there yet" branch applies. Re-verify
that against a real llama.cpp checkout before the assumption carries weight
anywhere.

## Getting the weights

There is no auto-fetch and no `BRAIN_GEMMA4_*` variable, because there is no
import path for them to point at. When one lands, the checkpoint is the
`gemma4-12b-with-proj` file shipped alongside LTX-2.5's other components.

Package: `brain-gemma4`.
