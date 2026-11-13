<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: models/flux2

FLUX.2 Klein as a standalone program: generate an image from a prompt, edit
one against reference images, inpaint a masked region, paint a large canvas
in windows, or train a LoRA on a folder of captioned images.

The program is FLUX.2's served actions with a command line built from their
manifest, so everything `brain serve` exposes for the model is here with the
same parameters and defaults.

## Build

```bash
make samples/models/flux2/build
```

## Weights

```bash
brain pull black-forest-labs/FLUX.2-klein-4B        # DiT, VAE, text encoder, tokenizer
brain pull unsloth/FLUX.2-klein-4B-GGUF-Q8_0         # optional: a smaller DiT
```

The four roles resolve from the model store; `--dit`, `--vae`,
`--text-encoder` and `--tokenizer` name one outright. The 9B variants are
under the FLUX.2 Non-Commercial license and run only with
`--allow-non-commercial` on each request.

## Run

```bash
F=target/release/sample-models-flux2

# text to image
$F text2image --prompt "a red apple on a wooden table" --seed $RANDOM --out image=apple.png

# inpaint: white in the mask is redrawn, the canvas takes the image's size
$F edit --image apple.png --mask mask.png --prompt "a green pear on a wooden table" --seed $RANDOM --out image=pear.png

# edit against several references (up to eight), each bounded to 512 px unless --ref-size says otherwise
$F edit --image scene.png --image0 person.png --prompt "the person from image 2 sitting in the scene" --seed $RANDOM --out image=out.png

# a large canvas in 512 px windows
$F text2image --prompt "a harbour at dusk" --width 2048 --height 1152 --tile-size 512 --seed $RANDOM --out image=harbour.png

# two adapters, each at its own strength
$F text2image --prompt "..." --adapter "style.safetensors@0.6,subject.brain" --seed $RANDOM --out image=a.png

# a LoRA from a folder of images and captions
$F lora_train --data my-set/ --save my.brain --steps 500 --seed $RANDOM
```

`sample-models-flux2 --help` lists the actions, `sample-models-flux2 <action>
--help` every parameter with its default. `--device gpu0` places the run.

## What it shows

How a program embeds one brain model through its served actions:
`invoke::Model` names the manifest, how the weights are resolved and how the
provider is built, and `invoke::main` does the rest.
