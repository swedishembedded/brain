# sample: imagegen/sdxl (D-Bus)

`sdxl_generate.py` drives the SDXL `text2image` action - dual CLIP-L/OpenCLIP-bigG
conditioning, a discrete Euler scheduler, and classifier-free guidance
(`crates/sdxlunet/src/pipeline.rs`). Unlike FLUX.2 Klein's action, this one is
a plain `Run` (no per-step progress hook yet), so it is one blocking call.

```bash
BRAIN_SDXL_DIR=/path/to/stable-diffusion-xl-base-1.0 \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/imagegen/sdxl/sdxl_generate.py --prompt "a red fox in the snow"
```

## What it demonstrates

- The `Run` (non-streaming) D-Bus call path, as opposed to `flux2-klein`'s
  `Subscribe` - one blocking request in, one finished image out.
- `brain_py.dbus.BrainDBus.run()` and reading a plain image blob back
  (`out.blobs["image"]` + `out.meta`).

## What it needs

`BRAIN_SDXL_DIR` is a released diffusers SDXL checkpoint root (`unet/`,
`vae/`, `text_encoder/`, `text_encoder_2/`, `tokenizer/`, `tokenizer_2/`). No
batching: every request runs its own denoising loop (`resident_sdxl.rs`'s
module docs explain why grouping would not help).

`pip install -e brain-py` (jeepney with fd passing).

## Options

| flag | default |
|---|---|
| `--prompt TEXT` | required |
| `--negative TEXT` | `""` (only used when `--guidance > 1.0`) |
| `--out PATH` | `sdxl.ppm` |
| `--width N` | `1024` (multiple of 8) |
| `--height N` | `1024` (multiple of 8) |
| `--steps N` | `30` |
| `--guidance F` | `5.0` |
| `--seed N` | `0` |
