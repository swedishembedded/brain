# sample: videogen/ltxv (LTX-2.5, shell)

Four shell wrappers around `brain ltxv t2v`/`v2v` - LTX-2.5's native
audio-visual video generation, driven straight from the CLI with no server
and no D-Bus in the loop. Weights come from brain's model store with no
configuration.

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

**Weights.** Nothing to configure: `brain ltxv` resolves every role from
the model store (default root `~/.local/share/brain/models`, repo directory
`Lightricks/LTX-2.5`), and stops with a message naming what is missing when
the store cannot supply a role. To use files outside the store, put brain's
role flags after a literal `--` and they are passed to `brain` unchanged:

```bash
samples/shell/videogen/ltxv/text_to_video.sh "a boat at sunset" out.mp4 5 -- \
  --dit /w/ltx-2.5-22b-distilled-transformer-Q8_0.gguf --vae /w/ltx-2.5-video-vae-conv-bf16.safetensors
```

The flags are `--dit`, `--vae`, `--audio-vae`, `--text-encoder`,
`--tokenizer`, `--spatial-upsampler` and `--temporal-upsampler`; `brain ltxv
<command> --help` lists the ones each command reads.

**The DiT is the one exception.** Lightricks publishes the 22B transformer
only as `.safetensors`; brain's loader reads a GGUF quantization of it
(`ltx-2.5-22b-distilled-transformer-{Q8_0,Q4_K_M}.gguf`), which has to come
from a community conversion or from running `brain quantize` yourself.
`LTX_TINY=1` runs the tiny random-weight DiT instead: a real wiring test
(the stills genuinely condition the noise), not a quality claim.
`character_swap.sh` needs the video VAE unconditionally - the conditioning
is defined on that latent and there is no stand-in for it - and, with no real
DiT available, brain runs the tiny random-weight DiT by itself.

## Options

| env var | scripts | meaning |
|---|---|---|
| `WIDTH`, `HEIGHT` | all | output resolution (default 1280x704) |
| `STEPS`, `SEED`, `FPS` | all | denoise steps / RNG seed (a random one is drawn and printed when unset) / output frame rate |
| `LTX_TINY=1` | `images_to_video.sh`, `chain_images_to_video.sh` | run the tiny random-weight DiT instead of the real one |
| `LTX_AUDIO=0` | `images_to_video.sh`, `chain_images_to_video.sh` | disable the audio half (cheaper, no audio VAE needed) |
| `LTX_TRACE=<0-5>` | `images_to_video.sh`, `chain_images_to_video.sh` | `brain --trace-ltxv` verbosity (default 4; 5 adds per-block GPU timings) |
| `MID=1` | `chain_images_to_video.sh` | group stills in triples (start/mid/end) instead of pairs; needs an odd still count |
| `STRENGTH`, `GUIDANCE` | `character_swap.sh` | conditioning strength (1.0 = pin exactly) / CFG scale |
| `BRAIN_DEVICE`, `BRAIN` | all | device selector / path to the `brain` binary (default `./target/release/brain`) |
