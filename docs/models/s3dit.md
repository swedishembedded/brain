# Z-Image Turbo

Z-Image Turbo is a fast text-to-image diffusion transformer: give it a prompt
and it generates an image in a handful of steps, trading a little quality
headroom against the slower base Z-Image model for much lower latency. Reach
for it when you want quick image generation from a text prompt, plus
image-to-image, inpainting, and outpainting against an existing image - and
it can be fine-tuned with LoRA on your own images.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [x] |
| INT8                   | [x] |
| CLI (`brain <arch> <action>`)       | [x] |
| HTTP API               | [x] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

Model id: `Tongyi-MAI/Z-Image-Turbo`. Weights auto-fetch (⤓) on first use - no
env var or manual download needed. The first request downloads and converts
the checkpoint into the local model store; every request after is instant.

The slower, non-distilled `Tongyi-MAI/Z-Image` base model auto-fetches the
same way when requested by name (use it with a higher `guidance` and more
`steps`).

### From a GGUF

`brain import-gguf` also accepts a quantized Z-Image DiT (unsloth publishes
`Z-Image-GGUF` and `Z-Image-Turbo-GGUF`), converting it to a brain-native
safetensors checkpoint that `BRAIN_S3DIT_DIT` can point at. These files declare
`general.architecture = "lumina2"`, which real Lumina2 releases also use, so the
importer refuses anything without Z-Image's own `cap_embedder.0.weight` rather
than guessing, and reads the variant off the tensor shapes.

DiT only: the VAE and the Qwen3-4B text encoder are not in the GGUF and still
come from the repository above. The conversion is fp32, so plan for disk, not
RAM: the 6.15 G-parameter DiT (3.4 GB at Q2_K) becomes a ~24 GB safetensors
checkpoint. Host memory stays bounded at a few hundred MB regardless, because
the importer streams tensor by tensor rather than materialising the whole model.

## Running it

Over HTTP:

```bash
brain serve --openai 8788
curl localhost:8788/v1/images/generations -H "Authorization: Bearer $KEY" \
     -d '{"model":"Tongyi-MAI/Z-Image-Turbo","prompt":"a red apple on a wooden table"}'
```

There is no dedicated `brain zimage` CLI subcommand - Z-Image is served
through the generalized capability interface:

```bash
brain caps brain/s3dit                        # discovery, no weights needed
brain s3dit text2image --prompt "a red apple on a wooden table" \
    --out image=apple.ppm --json
```

The same interface covers `image2image`, `inpaint`, `outpaint`, and
`lora_train` (`brain s3dit image2image ...` etc.) and is reachable over
D-Bus as well.

## Options

Shared generation parameters: `steps` (default 8), `guidance` (default 0.0),
`seed`, `width` / `height` (default 1024), `precision` - `int8` (default) or
`fp32`.

Actions: `text2image`, `image2image`, `inpaint`, `outpaint`, `lora_train`.

## Hardware and limits

The default `int8` precision keeps the resident weight footprint small enough
to fit a single mid-range GPU. There is no dedicated `brain zimage` CLI
subcommand - use `brain caps` / `brain do` (or D-Bus / HTTP). Requests are
served one image at a time; there is no batched serving yet.
