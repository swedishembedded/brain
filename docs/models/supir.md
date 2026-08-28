# SUPIR

SUPIR ("Scaling Up to Excellence", CVPR 2024) is a photo-realistic blind image
restoration model: give it a low-quality photo (compression artefacts,
downscaling, noise, blur) and it produces a plausible high-quality
reconstruction, driven by a frozen SDXL 1.0 base UNet plus a 1.24B control
trunk (`GLVControl`) and 12 adaptor modules (10 `ZeroSFT` + 2 `ZeroCrossAttn`)
trained specifically for restoration. Optional captioning of the low-quality
input is [LLaVA-1.5-13B](llava.md), dispatched through a `capability::Registry`
rather than a direct dependency.

## Status

The forward pass (trunk + adaptors + frozen backbone, one graph), the
restoration pipeline (dual encode, dual-CLIP conditioning, `RestoreEDMSampler`,
wavelet/AdaIN colour fix), INT8 host-memory quantization, training (gradcheck,
LoRA, adaptor-only/full-backbone finetune), the served `restore` action, the
CLI/D-Bus/`imgpipe` wiring and a `ZeroCrossAttn` NPU export are implemented and
weight-free-gated. **Not yet exercised end to end against real checkpoint
bytes on this port's own development machine** - the combined graph (trunk +
adaptors + frozen backbone) is measured to exceed that machine's single
integrated GPU's device memory even quantized (INT8 in this codebase reduces
HOST memory only, not device memory - see `.agents/roadmap/supir.md`'s
"Deferred" section for the real, measured numbers). Implementation and wiring
are complete; a real restoration run needs either more device memory or a
genuine on-device INT8 GEMM path this port did not close.

## Support

| Capability | Supported |
|---|---|
| Inference (`restore`)  | [x] (untested end to end against real weights - see Status) |
| LoRA fine-tune          | [x] |
| Full backbone fine-tune | [x] (`adaptor_only`, `full_backbone`) |
| INT8                    | [x] (host memory only - see Status) |
| CLI (`brain <arch> <action>`) | [x] |
| HTTP API                | [ ] |
| D-Bus                   | [x] |
| Batched serving         | [ ] (every request is its own multi-step sample) |
| `imgpipe` stage          | [x] (`supir_restore`, a size-changing tail like `upscale`) |
| NPU export               | [x] partial - `ZeroCrossAttn` only, see Hardware and limits |
| GGUF import              | [ ] (registered, no real file has ever been observed - see Options) |

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

## Getting the weights

Model id: `brain/supir`. Two checkpoints:

- `BRAIN_SDXL_DIR` - the frozen backbone: a released `stable-diffusion-xl-base-1.0`
  diffusers checkpoint root (`unet/`, `vae/`, `text_encoder/`, `text_encoder_2/`,
  `tokenizer/`, `tokenizer_2/`) - the same directory [SDXL](sdxlunet.md) and
  [ControlNet](controlnet.md) already load.
- `BRAIN_SUPIR_DIR` - SUPIR's own delta: a directory holding the released
  `SUPIR-v0Q_fp32.safetensors` (or that file directly).

## Running it

```bash
BRAIN_SDXL_DIR=/path/to/stable-diffusion-xl-base-1.0 \
BRAIN_SUPIR_DIR=/path/to/SUPIR-v0Q_fp32.safetensors \
  brain supir restore --caption "a weathered brick building" \
    --in image=degraded.ppm --out image=restored.ppm --json
```

Over D-Bus - see
[`examples/restore/supir_restore.py`](../../examples/restore/supir_restore.py)
and that directory's own README for the full worked example, mirroring
[`examples/restore/restore_face.py`](../../examples/restore/restore_face.py)'s
shape.

## Options

- `caption` - the image caption; empty auto-captions through a registered
  `brain/llava` when one is available (`BRAIN_LLAVA_WEIGHTS`), else stays
  empty - upstream's own `--no_llava` path, not an error.
- `positive_suffix` / `negative_prompt` - upstream's own default prompt
  suffix/negative text, overridable per call.
- `steps` (`edm_steps`, default `50`), `cfg_scale` (`s_cfg`, default `4.0`),
  `spt_linear_cfg` (default `1.0`), `control_scale` (`s_stage2`, default
  `1.0`), `s_churn` (default `5.0`), `s_noise` (default `1.01`), `restore_cfg`
  (`s_stage1`, default `-1.0` - **restoration guidance is OFF by default**,
  matching upstream's own CLI despite its shipped YAML's `4.0`), `seed`.
- `image` input - raw HWC f32 pixels in `[0,1]`.
- No official GGUF release of SUPIR exists; `supir::import::GGUF_ARCHITECTURE`
  (`"sdxl"`, a borrowed spelling - see that constant's own doc) is registered
  with `brain import-gguf` so a future release auto-dispatches with no CLI
  change, though nothing can be converted against it today.

## Hardware and limits

Does not batch concurrent requests - every `restore` call is its own
50-step (by default) sample, same as [SDXL](sdxlunet.md)/
[ControlNet](controlnet.md). `linear_s_stage2` (upstream's optional per-step
control-scale ramp) is not implemented: `control_scale` is baked into the
recorded graph as a constant, the same design choice
`crates/supir/src/model.rs` makes for the forward itself, so a per-step ramp
would mean rebuilding the whole trunk+adaptors+backbone graph every step.
Tiled VAE/tiled diffusion (needed for large images on constrained hardware)
are not wired into the serving pipeline. NPU export covers `ZeroCrossAttn`
only (`crates/npu/src/supir_topology.rs`) - the `ZeroSFT` adaptors and the
1.24B `GLVControl` trunk have no export path yet, and the trunk realistically
exceeds what an NPU can hold on the hardware this port was written on even if
one existed. See `.agents/roadmap/supir.md` for the full, itemised list of
what remains.
