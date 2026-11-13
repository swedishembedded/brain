# sample: imagegen/pulid (D-Bus)

`pulid_generate.py` adds a face photo to the `../flux1/` loop: ArcFace (raw
embedding) + EVA-CLIP-L/336 (CLS + 5 tapped hidden states) compose into
`id_cond`, `crate::model::IdFormer` projects 32 ID tokens, and
`crate::adapter::PulidAdapter` cross-attends them into the DiT at 20 points
through `flux1::pipeline::Flux1::generate_injected` -
`crates/pulid/src/caps.rs`'s module docs are the full account of what
composes what, including the one real preprocessing gap (a plain resize
where the reference uses face-parsing alignment brain does not have).

```bash
# Each role resolves from the model store; these name them outright.
ROLES="--flux1 /path/to/FLUX.1-dev --pulid /path/to/pulid_flux_v0.9.1.safetensors \
  --arcface /path/to/antelopev2 --clip /path/to/eva-clip-dir --bisenet /path/to/bisenet-dir"

# directly over the CLI (any image format brain decodes, not just PPM):
brain pulid text2image $ROLES --prompt "a photo of a person hiking in the mountains" \
  --in face_image=portrait.jpg --out image=out.png --precision int8

# or over D-Bus, the same action (pins are the server-side spelling):
tools/dbus-session.sh --serve "--dbus --pin pulid.flux1=/path/to/FLUX.1-dev \
    --pin pulid.pulid=/path/to/pulid_flux_v0.9.1.safetensors \
    --pin pulid.arcface=/path/to/antelopev2 --pin pulid.clip=/path/to/eva-clip-dir \
    --pin pulid.bisenet=/path/to/bisenet-dir" -- \
  python3 samples/python/imagegen/pulid/pulid_generate.py \
    --prompt "a photo of a person hiking in the mountains" --face portrait.ppm
```

## What it demonstrates

- Identity conditioning riding in as a `face_image` input blob, one field
  added on top of `../flux1/flux1_generate.py`'s params.
- The same `brain pulid text2image` action is reachable directly over the
  CLI (no server) or over D-Bus (this script) - pick whichever fits the
  workflow.
- `--id-weight` as the one identity-specific dial beyond FLUX.1's own
  params.

## What it needs

Five roles, each from the model store or named outright: `flux1` (the FLUX.1
root), `pulid` (the PuLID adapter `.safetensors`), `arcface` (the antelopev2
directory), `bisenet` (a directory holding `parsing_bisenet.safetensors`) and
`clip` (only needs to hold the EVA-CLIP-L/336 file,
`EVA02_CLIP_L_336_psz14_s6B.pt`, at its root - the same convention
`clip::caps` uses, not the CLIP-L/OpenCLIP-bigG text towers).

`--precision int8` (the CLI default) is what fits FLUX.1-dev's ~48 GiB fp32
DiT on a single <=24 GiB card; `--precision fp32` is the parity reference and
needs a card that can hold it.

`pip install -e brain-py` (jeepney with fd passing).

## Options

| flag | default |
|---|---|
| `--prompt TEXT` | required |
| `--face PATH` | required - binary PPM (P6) photo of the identity |
| `--variant` | `dev` (`dev`\|`kontext-dev`\|`schnell`; only `dev` is validated against a PuLID reference) |
| `--out PATH` | `pulid.ppm` |
| `--width N` | `1024` (multiple of 16) |
| `--height N` | `1024` (multiple of 16) |
| `--steps N` | `0` (variant default) |
| `--guidance F` | `3.5` |
| `--id-weight F` | `0.8` - identity conditioning strength |
| `--max-len N` | `512` |
| `--seed N` | random: drawn and printed, echoed in the result |

Only `dev` is validated against a PuLID reference (the reference is built on
FLUX.1-dev, not Kontext or schnell). Same scope/verification caveats as plain
FLUX.1 (`../flux1/README.md`), doubled: this also has no end-to-end fixture
for the ID-conditioning wiring itself.
