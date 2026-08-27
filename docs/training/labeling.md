<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
-->

# Labeling a dataset

`brain label` captions a folder of images with a vision-language model and
writes the caption file the image trainers already read. It is a **top-level
verb over a model seam**, not an action on one model: the workflow is the same
whichever model does the describing, so adding a captioning model must not add
a labeling command.

```bash
brain label images ./my-dataset --model qwen3vl --trigger "bohemian style"
```

That writes `./my-dataset/captions.yaml`, which is exactly what
`brain flux2 finetune ./my-dataset --out adapter.safetensors` consumes.

## Why the captions matter more than the hyper-parameters

The captions *are* the training signal. On a small dataset each one carries a
large share of it, and a vague caption teaches an adapter almost nothing the
base model did not already know. Prefer concrete, visible detail over the name
of a style:

> A low rattan bed with a woven headboard, layered with a rust-orange linen
> throw and three kilim cushions in ochre and deep red.

teaches far more than "boho bedding". The default `--prompt` asks for exactly
that: room type, furniture and materials, textiles and patterns, colour
palette, lighting direction and quality, camera viewpoint, architectural
features, and mood, as one paragraph. Override it with `--prompt` for a
non-interior dataset.

## The trigger phrase

`--trigger` weaves a phrase into every caption so an adapter trained on the
folder binds the concept to it. There is a real trade-off in what you pick:

- **A rare invented token** (`sks`, `TOK`) carries no prior, so the adapter
  binds it cleanly to the new concept and the effect is easy to isolate. It is
  also unreadable, composes badly with the rest of a prompt, and collides with
  any other adapter that chose the same token.
- **A natural phrase** (`bohemian style`) already means something to the base
  model, so the adapter *shifts* an existing concept rather than creating one.
  It reads naturally and composes well, but it leaks: every generation
  mentioning the phrase drifts toward your dataset, and an A/B against the base
  model is harder to read because the base already responds to the words.

Neither is wrong. Decide which you want, then check that it actually bound by
generating with and without the phrase at a fixed seed and comparing.

## Resumable and idempotent

Captions are written after **every** image, and a re-run only fills in what is
missing:

```bash
brain label images ./my-dataset          # captions 50 images, dies at 40
brain label images ./my-dataset          # captions the remaining 10
```

An image that already has a caption is left exactly as it is, so you can
correct captions by hand and a later run will not overwrite the correction.
`--overwrite` is the explicit opt-out. Re-running a finished folder is a no-op
that does not even load the model.

## The caption file

`captions.yaml` is a flat `filename: caption` mapping, and a caption may be a
YAML block scalar spanning as many lines as it needs:

```yaml
room-01.jpg: |-
  A wide-angle photograph of a living room in bohemian style. The low rattan
  sofa is layered with a rust-orange linen throw and three kilim cushions in
  ochre and deep red; a jute rug covers the oak floor.

  Warm low-angle afternoon light enters from a tall window on the left.
room-02.jpg: a single line still works
```

That is what makes a long caption editable: the text sits on its own lines with
no quoting and no escaping, and `#` and `:` inside it are just characters.
Single-line forms - bare, `"quoted"`, `'quoted'`, with `#` comments - parse
exactly as they always did. `captions.jsonl` still overrides `captions.yaml`
entry by entry. See `data::imageset`'s module documentation for the full
grammar.

## Choosing a model

| `--model` | Weights env | Notes |
|---|---|---|
| `qwen3vl` (default) | `BRAIN_QWEN3VL_WEIGHTS` | aspect-preserving smart resize, per-image token count (`--max-pixels` caps it); the more detailed describer, and much the more expensive |
| `fastvlm` | `BRAIN_FASTVLM_WEIGHTS` | small and fast, fixed 1024 px square input; `fp32` or `int8` decoder |

Both implement `captioner::Captioner`, and the labeler drives them through it
without knowing anything about either. A model crate joins by adding one file:
see `crates/qwen3vl/src/captioner.rs`.

## Video

The seam's unit is a **clip** - an ordered run of frames plus an optional frame
rate - of which a still image is the one-frame case. A video captioner joins by
reporting `max_frames > 1`, reading all of the frames rather than the first,
and writing `data::videoset`'s `captions.json` (the format
`wan::finetune::ClipSet` already reads) instead of `captions.yaml`. Only the
image path is built today; `Captioner::validate` refuses a multi-frame clip to
a still-image model by name rather than silently describing frame 0.

## See also

- [LoRA training](lora.md)
- [The CLI](../using/cli.md)
