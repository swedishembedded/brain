<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: shell/videogen/character_replace

Replace a person in a video with the character from a character sheet,
keeping their path, their timing and the whole background - or run one step
of that on its own. Fill in the inputs, copy a command, collect the output.

```bash
samples/shell/videogen/character_replace/character_replace.sh run \
  --input street.mp4 --character sheet.png --point 405,195 \
  --prompt "a woman with short copper-red hair in a mustard raincoat walks toward the camera, smiling"
```

writes `out/character_replace/street-replaced.mp4`, with every intermediate
beside it in `out/character_replace/street-replaced.mp4.work/`.

## What it does

`run` is `brain ltxv replace`:

1. **Track** - SAM 2.1 follows the clicked person through the clip.
2. **Paint stills** - FLUX.2 repaints the person's silhouette with the
   character from the sheet on frame 0 and on guide frames (every 40th and
   the last), keeping the rest of each frame. An edit whose figure does not
   overlap the person enough is retried with another seed; a guide that
   never does is dropped and reported.
3. **Regenerate** - LTX-2.5 masked video-to-video redraws only the person's
   region, grown to whole latent tokens, with frame 0's figure held and the
   other stills as keyframe guides at half strength.
4. **Composite** - every pixel outside the region is the source's own.

| mode | runs | for |
|---|---|---|
| `run` | the whole process | the result |
| `track` | `brain sam2 track` | checking the clicks pick the whole person; the mask sequence can go back into `run --actor-mask` |
| `video` | `brain ltxv v2v --mask ... --keyframe ...` | redoing the video pass alone over a region and stills you already have (a previous run's `input.mp4`, `region/` and `still_*.png`) |

## Inputs

| input | accepted |
|---|---|
| `--input` | any video brain decodes: `.mp4`, `.webm`, `.mkv`, `.mov`, any length and size. LTX-2.5 takes 1 + 8k frames with sides a multiple of 32, so `run` trims the end and centre-crops onto that grid, moves the clicks with the crop, and prints what it did (`--no-conform` refuses instead). `--fps` gives the rate when the container does not |
| `--character` | `.png` or `.jpg`: a sheet, a portrait, any picture of the character |
| `--point` / `--exclude` | clicks in source pixels on frame `--prompt-frame` (default 0), repeatable: a few over head, coat and legs pick the whole person |
| `--actor-mask` | instead of clicks: a `brain/sam2-maskseq/1` directory (what `track` writes) |
| `--soundtrack` | a `.wav` carried into the result, cut to the clip's length. brain reads no sound out of a video container, so without it the result is silent, and the run says so |

## Build

From brain's checkout:

```bash
make build/release
```

The sample is a script over the `brain` binary; that binary is the only
thing to build. `--brain <path>` names another one.

## Models

Every model resolves from brain's model store (`~/.local/share/brain/models`,
or `--brain-data-dir`). Gated repositories need a Hugging Face account that
has accepted the model's terms, and `hf auth login` (or `HF_TOKEN`) before
pulling.

**LTX-2.5** (video pass, gated): the video VAE, the text encoder and the
distilled DiT, which brain runs from a GGUF it quantizes itself:

```bash
brain pull https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors
brain pull https://huggingface.co/Lightricks/LTX-2.5/blob/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
brain pull https://huggingface.co/Lightricks/LTX-2.5/blob/main/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
STORE=~/.local/share/brain/models/Lightricks/LTX-2.5
brain quantize "$(find $STORE -name ltx-2.5-22b-distilled-transformer-bf16.safetensors)" \
  --out $STORE/ltx-2.5-22b-distilled-transformer-Q8_0.gguf --arch ltxv
```

`--tier Q4_K` instead of the default Q8_0 halves the DiT's memory.

**FLUX.2** (still edits). klein-4B is open; klein-9B is gated and
non-commercial:

```bash
brain pull black-forest-labs/FLUX.2-klein-4B           # DiT, VAE, text encoder, tokenizer
brain pull unsloth/FLUX.2-klein-4B-GGUF-Q8_0            # optional: a smaller DiT
brain pull black-forest-labs/FLUX.2-klein-9B            # optional: the larger editor
```

**SAM 2.1** (tracking):

```bash
brain pull facebook/sam2.1-hiera-tiny                   # the default
brain pull facebook/sam2.1-hiera-large                  # optional: the larger tracker
```

**Adapters** - any LoRA in ComfyUI/PEFT `.safetensors` form. For the still
edits, a FLUX.2-klein-4B LoRA (`--edit-adapter`); for the video pass, an
LTX-2.5 LoRA (`--adapter`), such as one trained on your character, or a
published one (Lightricks' are gated):

```bash
brain pull https://huggingface.co/fal/flux-2-klein-4B-outpaint-lora/blob/main/flux-outpaint-lora.safetensors
brain pull https://huggingface.co/Lightricks/LTX-2.5-22b-LoRA-Cinemagraph/blob/main/ltx-2.5-22b-lora-cinemagraph-0.9.safetensors
```

When the store holds more than one candidate for a role, brain lists them
and stops; name the one to use with the model flags below.

## Examples

Set the store path once, so the commands below can be pasted as they are:

```bash
M=~/.local/share/brain/models
S=samples/shell/videogen/character_replace/character_replace.sh
```

The whole process with the models named outright (FLUX.2-klein-4B GGUF
editor, the tiny tracker):

```bash
$S run --input street.mp4 --character sheet.png \
  --point 405,100 --point 405,195 --point 405,368 \
  --prompt "a woman with short copper-red hair in a mustard raincoat walks toward the camera, smiling" \
  --video-dit $M/Lightricks/LTX-2.5/ltx-2.5-22b-distilled-transformer-Q8_0.gguf \
  --editor-variant klein-4b --editor-dit $M/unsloth/FLUX.2-klein-4B-GGUF/Q8_0.gguf \
  --editor-vae $M/black-forest-labs/FLUX.2-klein-4B/vae/diffusion_pytorch_model.safetensors \
  --editor-text-encoder $M/black-forest-labs/FLUX.2-klein-4B/text_encoder \
  --editor-tokenizer $M/black-forest-labs/FLUX.2-klein-4B/tokenizer/tokenizer.json \
  --tracker-weights $M/facebook/sam2.1-hiera-tiny/sam2.1_hiera_tiny.pt
```

The larger editor and tracker:

```bash
$S run --input street.mp4 --character sheet.png --point 405,195 --prompt "..." \
  --editor-variant klein-9b --editor-dit $M/black-forest-labs/FLUX.2-klein-9B/flux-2-klein-9b.safetensors \
  --tracker-variant large --tracker-weights $M/facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt
```

Adapters on both passes, each with its strength:

```bash
$S run --input street.mp4 --character sheet.png --point 405,195 --prompt "..." \
  --edit-adapter $M/fal/flux-2-klein-4B-outpaint-lora/flux-outpaint-lora.safetensors --edit-lora-scale 0.6 \
  --adapter my-character-ltx-lora.safetensors --lora-scale 0.8
```

Check the clicks first, then hand the track to `run`:

```bash
$S track --input street.mp4 --point 405,100 --point 405,195 --point 405,368
$S run --input street.mp4 --character sheet.png --actor-mask out/character_replace/street-track --prompt "..."
```

Redo only the video pass of a previous run, with other settings - over the
clip that pass saw (conformed, with the painted frame 0). The result is the
pass itself, not composited over the source:

```bash
W=out/character_replace/street-replaced.mp4.work
$S video --input $W/input.mp4 --mask $W/region --prompt "..." \
  --keyframe 40:$W/still_40.png --keyframe 80:$W/still_80.png --keyframe 120:$W/still_120.png \
  --keyframe-strength 0.5 --out out/character_replace/street-rerun.mp4
```

A clip with sound, any size, more guides, held harder:

```bash
$S run --input phone-clip.mov --soundtrack phone-clip.wav --character sheet.png --point 520,300 \
  --prompt "..." --guide-every 24 --guide-strength 0.7
```

`--dry-run` prints the brain command instead of running it.

## Options

The script's own flags are in `character_replace.sh --help`. Every other flag
goes to the brain command unchanged, so everything the commands take is a
parameter here. For `run`, from `brain ltxv replace --help`:

| flag | default | what it sets |
|---|---|---|
| `--guide-frames <list>` / `--guide-every <n>` | every 40th and the last | the frames painted as guides (`none` for none) |
| `--guide-strength <s>` | 0.5 | how hard the video holds each guide; at 1.0 it cross-fades between stills instead of moving |
| `--edit-attempts <n>` | 3 | seeds an edit gets to cover the person |
| `--min-overlap <f>` | 0.5 | the overlap (IoU) of figure and person a still needs |
| `--edit-size <px>` | 768 | the long side the stills are painted at |
| `--edit-prompt <text>` | put the sheet's character in the person's place | the instruction for the still edits |
| `--repaint-grow <f>` | 0.2 | how far past the person's outline an edit paints, as a fraction of their width |
| `--body-ratio <f>` | 4 | a standing person's height over their silhouette's median width: how much body a still paints below the frame where the frame cuts the person off |
| `--paste-grow <px>` | 6 | the margin a still is pasted back in |
| `--region-grow <px>` | 12 | the margin around person and figure the video pass regenerates |
| `--frame0-inset <px>` | 16 | how far inside the figure frame 0 is redrawn |
| `--feather <px>` | 3 | the feather across every composite edge |
| `--seed`, `--steps`, `--dit-config`, `--device` | | the run's seed, the video's steps, DiT and placement |
| `--work-dir <dir>` | `<out>.work` | where the intermediates go |

## Outputs

| path | what |
|---|---|
| `<out>` | the result |
| `<out>.json` | the run record: every input, setting and seed |
| `<out>.work/actor/` | the person's track (a mask sequence) |
| `<out>.work/still_<k>.png` | the painted stills; `edit_<k>_<seed>.png` every attempt, `canvas_<k>.png` / `repaint_<k>.png` what each edit was given |
| `<out>.work/region/` | the regenerated region per frame (a mask sequence) |
| `<out>.work/input.mp4` | the clip the video pass sees: conformed, with the painted frame 0 |
| `<out>.work/generated.mp4` | the video pass before compositing |

## What it needs

A GPU that holds the LTX-2.5 DiT (the Q8_0 GGUF runs on one 24 GB card at
768x448, 121 frames). A run of 121 frames there takes about 23 minutes
with the tiny tracker and 40 with the large one: tracking, four stills and
the video pass.
