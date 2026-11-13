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

Model id: `Tongyi-MAI/Z-Image-Turbo`. Weights auto-fetch (⤓, opt-in `--autofetch`) on first use - no
env var or manual download needed. The first request downloads and converts
the checkpoint into the local model store; every request after is instant.

The slower, non-distilled `Tongyi-MAI/Z-Image` base model auto-fetches the
same way when requested by name (also opt-in) (use it with more `steps`; classifier-free
guidance is not implemented, so the base model samples unguided).

`brain serve` resolves the four components from the model store with the same
`s3dit::spec::S3ditSpec` scan/candidate rules a one-shot command uses.
`brain serve --pin s3dit.dit=FILE` (and `vae`, `text_encoder`, `tokenizer`)
pins one component outright, overriding whatever the store would otherwise
pick; a one-shot command takes the same roles as `--dit FILE` and so on.

### From a GGUF

`brain import` also accepts a quantized Z-Image DiT (unsloth publishes
`Z-Image-GGUF` and `Z-Image-Turbo-GGUF`), converting it to a brain-native
safetensors checkpoint for the `dit` role. These files declare
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
D-Bus as well. `adapter`, and `lora_train`'s `data` and `save`, name files on
the serving machine, so only `brain do` on that machine takes them; D-Bus
and HTTP neither list nor accept them.

`lora_train` takes an optional `activations` (`keep` or `recompute`): what the
backward keeps of the forward. The trainers rebuild every block from its input,
so `recompute` is what they do and `keep` is refused by name before the
dataset is read.

## Options

Shared generation parameters: `steps` (default 8), `seed` (random when
omitted; the outcome reports the one used, and passing it back reproduces the
image), `width` / `height` (default 1024), `precision` -
`int8` (default) or `fp32`.

Actions: `text2image`, `image2image`, `inpaint`, `outpaint`, `lora_train`.

Placement and kernel parameters (unset is the automatic behaviour; each one
that changes what is built keys the resident instance, so a changed value
rebuilds it):

| Parameter | Actions | Meaning |
| --- | --- | --- |
| `attention` | all generating actions | `auto` (flash once the scores matrix would reach the device's binding limit), `flash` or `materialized` to force one |
| `window_blocks` | all generating actions | main-stack blocks the fp32 single-GPU decode window keeps resident (default 2) |
| `encoder_gpu` | `text2image`, `lora_train` | the card that hosts the Qwen-4B encoder; for `text2image` unset lets the placement plan choose, for `lora_train` the default is card 1 |
| `encoder_resident` | `text2image` | keep the int8 encoder resident even when it shares the DiT's card |
| `encoder_fp32_split`, `encoder_cut` | `text2image` | the opt-in fp32 encoder split across two cards, and the layer it cuts at (default about two thirds) |
| `retain_int8_cache` | `text2image` | served residency only: keep the quantised int8 DiT on the host across demote/promote, trading multi-GB host RAM for a faster reactivation |

## Hardware and limits

The default `int8` precision keeps the resident weight footprint small enough
to fit a single mid-range GPU. There is no dedicated `brain zimage` CLI
subcommand - use `brain caps` / `brain do` (or D-Bus / HTTP). Requests are
served one image at a time; there is no batched serving yet.
