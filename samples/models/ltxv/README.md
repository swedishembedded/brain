<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/ltxv

LTX-2.5 as a standalone program: video (and its sound) from a prompt, from
stills or from a soundtrack; masked editing of an existing clip; upscaling
and extending a clip; replacing a person in a clip with a character from a
sheet; and chaining a folder of stills into one clip.

Every action is LTX-2.5's served one, with a command line built from its
manifest, so everything `brain serve` exposes for the model is here with the
same parameters and defaults. `chain` is the one workflow written in this
sample: it runs `t2v` once per pair of stills.

## Build

```bash
make samples/models/ltxv/build
L=target/release/sample-models-ltxv
```

## Weights

The VAE, the Gemma-4 text encoder and the distilled DiT, which brain runs
from a GGUF it quantizes itself (LTX-2.5 is gated: accept its terms on
Hugging Face and `hf auth login` first):

```bash
brain pull https://huggingface.co/Lightricks/LTX-2.5/blob/main/vae/ltx-2.5-video-vae-conv-bf16.safetensors
brain pull https://huggingface.co/Lightricks/LTX-2.5/blob/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
brain pull https://huggingface.co/Lightricks/LTX-2.5/blob/main/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
S=~/.local/share/brain/models/Lightricks/LTX-2.5
brain quantize "$(find $S -name ltx-2.5-22b-distilled-transformer-bf16.safetensors)" \
  --out $S/ltx-2.5-22b-distilled-transformer-Q8_0.gguf --arch ltxv
```

Every role resolves from the model store; `--dit`, `--vae`, `--text-encoder`,
`--spatial-upsampler`, ... name one outright. `replace` also runs SAM 2.1,
FLUX.2 and (optionally) ArcFace, from the same store:

```bash
brain pull facebook/sam2.1-hiera-tiny
brain pull black-forest-labs/FLUX.2-klein-4B
brain pull https://huggingface.co/DIAMONIK7777/antelopev2/blob/main/glintr100.onnx
brain pull https://huggingface.co/DIAMONIK7777/antelopev2/blob/main/scrfd_10g_bnkps.onnx
```

## Video from text, stills or sound

```bash
$L t2v --prompt "a boat sailing at sunset" --out video=boat.mp4
$L t2v --prompt "a cat walking on a beach" --start-frame cat.png --out video=cat.mp4
$L t2v --prompt "..." --audio --out video=talk.mp4              # the clip's own sound, generated with it
$L t2v --prompt "..." --audio --audio-source song.wav --out video=dance.mp4
$L t2v --scenes "97:a lighthouse at dawn|97:the same lighthouse in a storm" --out video=two.mp4
$L t2a --prompt "rain on a tin roof" --out audio=rain.wav
```

## Chaining stills into one clip

A folder holding `prompt.txt` and `image-00.png`, `image-01.png`, ... (zero
padded, so name order is clip order). Each pair of consecutive stills
becomes one segment, start frame on the first and end frame on the second;
the shared still appears once in the result. `--mid` uses triples (start,
mid, end) for a segment that has to stay on course through its middle; it
needs an odd number of stills.

```bash
$L chain shots/ --out video=story.mp4 --width 1280 --height 704 --steps 8
```

Every `t2v` flag applies to every segment; the segments are kept in
`story.segments/`.

## Editing an existing clip

```bash
$L v2v --video clip.mp4 --start 2 --end 4 --prompt "..." --out video=retake.mp4   # regenerate seconds 2-4
$L v2v --video clip.mp4 --mask masks/ --prompt "..." --out video=swap.mp4         # regenerate a tracked region
$L extend --video clip.mp4 --seconds 3 --prompt "..." --out video=longer.mp4
$L upscale --video clip.mp4 --prompt "..." --out video=sharp.mp4
```

`masks/` is a mask sequence, as `brain sam2 track --out masks=masks/`
writes it.

## Replacing a person with a character

```bash
$L replace --video street.mp4 --character sheet.png --points "405,100;405,195;405,368" \
  --prompt "a woman with short copper-red hair in a mustard raincoat walks toward the camera, smiling" \
  --soundtrack street.wav --work-dir street.work --out video=street-replaced.mp4
```

What it does:

1. **Track** - SAM 2.1 follows the clicked person through the clip.
2. **Paint stills** - FLUX.2 repaints the person's silhouette with the
   character from the sheet on frame 0 and on guide frames (every 40th and
   the last), keeping the rest of each frame. Every guide is painted against
   the sheet and frame 0's accepted still, so the character keeps the same
   face, hair and clothes to the last frame. An edit whose figure does not
   overlap the person enough is retried with another seed; with ArcFace in
   the store the edit whose face is closest to the sheet's is kept.
3. **Regenerate** - LTX-2.5 masked video-to-video redraws only the person's
   region, with frame 0's figure held and the other stills as guides at half
   strength.
4. **Composite** - every pixel outside the region is the source's own.

Any clip size and length is accepted: it is trimmed to 1 + 8k frames and
centre-cropped to multiples of 32 (`--conform false` refuses instead), and
the run's `notes` say what was done. `--actor-mask masks/` gives the
person's track instead of clicks; `--components flux2.dit=<file>,sam2.weights=<file>`
names another model's checkpoint outright.

The run is resumable: repeat the same command after a failure (out of memory
in the video pass, say) and the track and the stills are loaded from the work
directory; changing `--prompt`, `--steps` or `--seed` redoes just the video
pass, a new `--character` or `--edit-prompt` repaints the stills, `--fresh`
redoes everything. The work directory holds:

| path | what |
|---|---|
| `actor/` | the person's track (a mask sequence) |
| `still_<k>.png` | the painted stills; `edit_<k>_<seed>.png` every attempt, `canvas_<k>.png` / `repaint_<k>.png` what each edit was given |
| `appearance.png` | frame 0's character, cut to the head, hair and clothes |
| `region/` | the regenerated region per frame |
| `input.mp4` | the clip the video pass sees: conformed, with the painted frame 0 |
| `generated.mp4` | the video pass before compositing |

Redo only the video pass, over the clip that pass saw:

```bash
W=street.work
$L v2v --video $W/input.mp4 --mask $W/region --prompt "..." \
  --keyframes "40:$W/still_40.png,80:$W/still_80.png" --keyframe-strength 0.5 --out video=rerun.mp4
```

The dials (`--guide-every`, `--guide-strength`, `--edit-attempts`,
`--min-overlap`, `--edit-size`, `--repaint-grow`, `--body-ratio`,
`--paste-grow`, `--region-grow`, `--frame0-inset`, `--feather`), the video
and edit adapters (`--adapter`, `--edit-adapter`, each a `path@strength`
list) and everything else: `sample-models-ltxv replace --help`.

## What it needs

A GPU that holds the LTX-2.5 DiT: the Q8_0 GGUF runs on one 24 GB card at
768x448, 121 frames. `--device gpu0` places the run.
