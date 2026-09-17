# sample: imagegen/controlnet (D-Bus)

`controlnet_generate.py` adds a conditioning image (edge map, depth map,
pose, ...) to the same SDXL loop `../sdxl/sdxl_generate.py` drives:
`crates/controlnet/src/caps.rs` builds the backbone with `Unet::new_controlled`
instead of `Unet::new` and runs the `ControlNet` once per denoising step,
threading its residuals in via `Unet::run_with_control`
(`crates/controlnet/src/adapter.rs`'s `ControlAdapter`/`ControlSource` seam -
backbone-agnostic by design, so a FLUX ControlNet would plug into the same
seam without touching it).

```bash
BRAIN_SDXL_DIR=/path/to/stable-diffusion-xl-base-1.0 \
BRAIN_CONTROLNET_DIR=/path/to/controlnet-canny-sdxl-1.0 \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/imagegen/controlnet/controlnet_generate.py \
    --prompt "a red fox in the snow" --control canny_edges.ppm
```

## What it demonstrates

- A second input blob (`control_image`) riding alongside the ordinary
  `text2image` params, over the same `Run` (non-streaming) call as plain
  SDXL.
- The conditioning image is resized on the device to the output size, so it
  need not be pre-sized to match `--width`/`--height`.

## What it needs

`BRAIN_SDXL_DIR` (see `../sdxl/README.md`) plus `BRAIN_CONTROLNET_DIR`, a
diffusers ControlNet checkpoint directory (e.g. `controlnet-canny-sdxl-1.0`).

`pip install -e brain-py` (jeepney with fd passing).

## Options

| flag | default |
|---|---|
| `--prompt TEXT` | required |
| `--control PATH` | required - binary PPM (P6) conditioning image |
| `--negative TEXT` | `""` |
| `--out PATH` | `controlnet.ppm` |
| `--width N` | `1024` (multiple of 8) |
| `--height N` | `1024` (multiple of 8) |
| `--steps N` | `30` |
| `--guidance F` | `5.0` |
| `--conditioning-scale F` | `1.0` |
| `--seed N` | `0` |
