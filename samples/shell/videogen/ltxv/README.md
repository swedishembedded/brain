# sample: videogen/ltxv (LTX-2.5, shell)

Four shell wrappers around `brain ltxv t2v`/`v2v` - LTX-2.5's native
audio-visual video generation, driven straight from the CLI with no server
and no D-Bus in the loop. All four share one weight-resolution helper so
`BRAIN_LTXV_*` only has to be worked out once per machine.

```bash
samples/shell/videogen/ltxv/text_to_video.sh "a boat sailing at sunset" clip.mp4 5
```

## The four scripts

### `text_to_video.sh` - prompt in, clip out

The simplest path: one prompt, one playable clip with sound.

```bash
samples/shell/videogen/ltxv/text_to_video.sh "a boat at sunset" out.mp4 5
```

LTX-2.5 is natively audio-visual - `--audio` runs the model's audio half
from the same forwards as the picture and muxes it in, but only produces a
real stereo image if the prompt describes where sounds sit and how they
move. A clip longer than one denoising window is generated as several
windows with a rolling latent context; each window re-reads the prompt with
weaker anchoring, so one or two scene beats hold up far better than many.

### `images_to_video.sh` - one still in, one clip out (per image)

Point it at a single image, or a folder of them, and get one *independent*
clip per image back, named after its source (`photo.png` -> `photo.mp4`).

```bash
samples/shell/videogen/ltxv/images_to_video.sh photo.png "a cat walking on a beach at sunset"

mkdir shots/ && echo "a cat walking on a beach at sunset" > shots/prompt.txt
cp photo1.png photo2.png shots/
samples/shell/videogen/ltxv/images_to_video.sh shots/
```

Each image conditions only its own clip's opening frame
(`brain ltxv t2v --start-frame`) - N images make N unrelated clips, never
one clip that passes through several of them. The prompt is, in order: the
second CLI argument if given; else `prompt.txt` inside the folder; else a
generic motion placeholder (a wiring convenience, not a quality claim - a
real prompt is always better). For a single continuous clip that opens on
one still and closes on another, see `chain_images_to_video.sh` instead.

### `chain_images_to_video.sh` - a numbered sequence in, one long clip out

For more than two stills: point it at `image-00.*, image-01.*, image-02.*,
...` (zero-padded so a plain `sort` gives the right order) plus one shared
`prompt.txt`, and it generates one clip per consecutive PAIR - image-00 to
image-01, then image-01 to image-02 - and concatenates them with a
no-re-encode `ffmpeg -c copy` (the clips already agree at every seam,
because each clip's end still is the next clip's start still).

```bash
mkdir story/
echo "a boat sailing across the ocean at sunset" > story/prompt.txt
cp frame0.png story/image-00.png
cp frame1.png story/image-01.png
cp frame2.png story/image-02.png

samples/shell/videogen/ltxv/chain_images_to_video.sh story/ out.mp4 5
```

`MID=1` uses the mid-frame conditioning slot too, grouping stills into
non-overlapping triples (00/01/02, then 02/03/04, ...) instead of pairs -
needs an odd still count, and is the way to keep a longer or moving-camera
segment on course through its own middle, not only at its ends. Segment
`N` is kept at `<out>.segments/clip-N.mp4` for inspection.

### `character_swap.sh` - replace one character, keep everything else

Takes a clip, a mask sequence, and a prompt describing the *whole* frame,
and replaces only what the mask marks as non-conditioning:

```bash
brain sam2 track --video stunt.mp4 --point 640,300 --out masks/
samples/shell/videogen/ltxv/character_swap.sh stunt.mp4 masks/ "a woman in a red coat" out.mp4
```

The mechanism is masked conditioning - LTX-2.5's `VideoConditionByMask`,
ported and parity-gated in `ltxv::maskcond`. Every latent position the mask
marks as conditioning is handed the source clip's own latent and excluded
from denoising, so the set, the camera move and the lighting come out of
the sampler bit-exactly unchanged; everything else is renoised from the
prompt. This needs no adapter (Lightricks has published exactly one
IC-LoRA for LTX-2.5, and it is a pixel spatial upscaler), which is why
masked conditioning is the route here rather than a reference-image path.

**This is not an identity swap.** Nothing in this path takes a face crop or
a per-subject embedding - who the new character is comes from the prompt
alone. `brain sam2 track` writes the mask sequence with SAM 2's native
polarity (white = tracked subject) and records it in `masks.json`; the
reader honours that field and refuses to run if it is missing, rather than
guessing backwards and preserving the character while regenerating
everything else. The script's last line of output is the check that the
swap actually happened: mean `|delta|` inside the replaced region versus
the preserved one - a preserved value near zero against a much larger
replaced value is the conditioning working.

## What it needs

**Weights.** None of the four asks for `BRAIN_LTXV_{DIT,VAE,TEXT_ENCODER}`
by hand - `_resolve_ltxv_weights.sh` (sourced, not run directly; internal
plumbing shared by `images_to_video.sh`, `chain_images_to_video.sh` and
`character_swap.sh`) finds them under `$BRAIN_MODELS_DIR/Lightricks/LTX-2.5`
from the official [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)
filenames. Put the files there **flat** (no `vae/`/`text_encoders/`/
`diffusion_models/` subfolders - that repo ships them nested, this expects
them moved up one level); `LTX_MODEL_DIR` points elsewhere if you keep them
somewhere else. Whatever isn't found is asked for once, interactively (and
errors immediately rather than hangs if stdin isn't a terminal).

**The DiT is the one exception.** Lightricks publishes the 22B transformer
only as `.safetensors`; brain's loader reads a GGUF quantization of it
(`ltx-2.5-22b-distilled-transformer-{Q8_0,Q4_K_M}.gguf`), which has to come
from a community conversion or from running `brain quantize` yourself - it
will never just appear in that directory from the official repo alone.
`LTX_TINY=1` skips all of this and runs the tiny random-weight DiT instead:
a real wiring test (the stills genuinely condition the noise), not a
quality claim. `character_swap.sh` additionally requires
`BRAIN_LTXV_VAE` unconditionally - the conditioning is defined on that
latent and there is no stand-in for it.

`text_to_video.sh` is the one script that does NOT source the resolver - it
takes `BRAIN_LTXV_{DIT,TEXT_ENCODER,VAE,AUDIO_VAE}` directly.

## Options

| env var | scripts | meaning |
|---|---|---|
| `WIDTH`, `HEIGHT` | all | output resolution (default 1280x704) |
| `STEPS`, `SEED`, `FPS` | all | denoise steps / RNG seed / output frame rate |
| `LTX_TINY=1` | all but `text_to_video.sh` | run the tiny random-weight DiT instead of resolving real weights |
| `LTX_AUDIO=0` | `images_to_video.sh`, `chain_images_to_video.sh` | disable the audio half (cheaper, no `BRAIN_LTXV_AUDIO_VAE` needed) |
| `LTX_TRACE=<0-5>` | `images_to_video.sh`, `chain_images_to_video.sh` | `brain --trace-ltxv` verbosity (default 4; 5 adds per-block GPU timings) |
| `LTX_MODEL_DIR` | scripts sourcing the resolver | folder holding LTX-2.5's files flat (default `$BRAIN_MODELS_DIR/Lightricks/LTX-2.5`) |
| `MID=1` | `chain_images_to_video.sh` | group stills in triples (start/mid/end) instead of pairs; needs an odd still count |
| `STRENGTH`, `GUIDANCE` | `character_swap.sh` | conditioning strength (1.0 = pin exactly) / CFG scale |
| `BRAIN_DEVICE`, `BRAIN` | all | device selector / path to the `brain` binary (default `./target/release/brain`) |
