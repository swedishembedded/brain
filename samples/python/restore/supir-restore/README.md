# sample: restore/supir-restore (D-Bus)

`supir_restore.py` drives `brain/supir`'s `restore` action over D-Bus:
one degraded image in, a full photo-realistic reconstruction out, through
a frozen SDXL 1.0 base UNet, a 1.24B `GLVControl` trunk and 12
`ZeroSFT`/`ZeroCrossAttn` adaptors, driven by `RestoreEDMSampler`. Unlike
CodeFormer's aligned-face crop, SUPIR takes *any* degraded image (photo
compression, downscaling, noise, blur) and regenerates the whole frame - a
real multi-step (50 by default) diffusion sample, so this call is
seconds-to-minutes, not sub-second.

```bash
BRAIN_SDXL_DIR=/path/to/stable-diffusion-xl-base-1.0 \
BRAIN_SUPIR_DIR=/path/to/SUPIR-v0Q_fp32.safetensors \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/restore/supir-restore/supir_restore.py --image degraded.ppm
```

## What it demonstrates

* The output size is SUPIR's own resize/snap rule (short side >= 1024, both
  axes snapped to a 64px multiple) - `supir_restore.py` reads it back from
  the result rather than assuming it.
* Set `BRAIN_LLAVA_WEIGHTS` too (leave `--caption` unset) to auto-caption
  the degraded image through LLaVA-1.5-13B before restoring it - SUPIR's
  own upstream behaviour when `--no_llava` is not passed. `crates/supir`
  links no VLM itself; the auto-caption call goes through a
  `capability::Registry` build, the same as any other cross-model call.

## What it needs

* `BRAIN_SDXL_DIR` (the frozen SDXL backbone) + `BRAIN_SUPIR_DIR` (SUPIR's
  own delta checkpoint). `BRAIN_LLAVA_WEIGHTS` is optional, for
  auto-captioning.
* **Licence**: SUPIR's weights carry a non-commercial licence (SUPIR
  Software License Agreement, (c) 2024 SupPixel Pty Ltd) - commercial use,
  including SaaS deployment and using the output as training data for
  another model, needs written permission from the licensor. Read that
  licence before using output commercially.
* **Device memory**: the combined trunk+adaptors+backbone graph is large
  even quantized (INT8 reduces host memory only in this codebase, not
  device memory) - this port's own development machine has never completed
  a real end-to-end run for want of a big enough GPU. The wiring is
  complete and weight-free tested regardless.

## Options

| flag | default |
|---|---|
| `--image PATH` | *required* - binary PPM (P6) of the degraded image |
| `--caption TEXT` | empty (auto-captions via `brain/llava` when served, else stays empty) |
| `--steps N` | `50` (`edm_steps`) |
| `--cfg-scale F` | `4.0` (`s_cfg`) |
| `--control-scale F` | `1.0` (`s_stage2`) |
| `--seed N` | `0` |
| `--out PATH` | `/tmp/restored.ppm` |
