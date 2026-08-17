# Wan (video generation)

Wan turns a text prompt - or a single still image - into a short video: about
five seconds, 81 frames at 16 fps. <!-- perf-number: the output frame rate is a property of the video the model generates, not a throughput claim -->
It is the first video model in brain, and the smallest variant is deliberately
modest: Wan2.1-T2V-1.3B generates 480p video in roughly 8 GB of VRAM, which puts
it on consumer hardware rather than on a datacentre card.

Reach for it when you want motion. For a single still frame, the image models
(`s3dit`, `flux2`) are far cheaper - Wan spends most of its work on making
frames agree with each other over time, which is wasted if you only need one.

`wan` names the whole upstream family rather than one release, the same way
`qwen3` covers every Qwen3 size. Wan2.1 and Wan2.2 share one architecture, one
Hugging Face class and one GGUF architecture tag (`wan`); which release you have
is a matter of configuration, not a different model.

## Support

| Capability | Supported |
|---|---|
| Inference (text to video) | [x] |
| Inference (image to video) | [ ] |
| LoRA fine-tune | [x] (library, no command yet) |
| INT8 | [ ] |
| CLI (`brain wan t2v`) | [x] |
| Capability (`brain caps`, manifest + action schema) | [x] |
| HTTP API | [ ] |
| D-Bus | [x] |
| Batched serving | [ ] |

**The port is partly done.** Text-to-video runs end to end from one command and
is served like every other model - a weights-free manifest `brain caps
brain/wan` prints, a residency adapter the scheduler budgets and places, and a
cancellable `Subscribe` job over D-Bus. Image-to-video does not exist yet.
`.agents/roadmap/wan.md` tracks what remains and carries the bug ledger.

LoRA fine-tuning exists as a gradient-checked library path (`wan::finetune`)
rather than a command: the transformer's forward and backward, the adapter, and
the captioned-clip dataset are all there and gated, but nothing yet exposes them
as `brain wan ...` or as a capability action, so training today means calling
`wan::finetune::run` from Rust. The trainer runs on the host, which is fine for
short adapter runs at small latent extents and is not a path to training the
full 1.3B model.

There is no HTTP surface because there is no OpenAI/Anthropic-shaped endpoint
for video generation to fit; the D-Bus and capability paths are the served
ones. Batched serving is listed as unsupported deliberately rather than
silently: concurrent requests at the same size share one resident transformer,
but each denoises in turn - the engine records one graph for one latent volume
and holds a single text-context buffer, so a genuinely batched forward is a
change to the engine, not to the adapter.

## Generating a video

```
brain wan t2v --prompt "A belgian malinois running on a paved highway" \
              --seed 42 --output-path out.mp4
```

Everything else has a default taken from the variant's own configuration, and
every weight path is both a flag (`--dit`, `--vae`, `--t5`, `--tokenizer`) and
an environment variable, with the flag winning. `--frames` must be `1 + 4k`,
and `--width`/`--height` must be multiples of 16.

Writing the file needs the `ffmpeg` CLI. Without it the frames are still
written, as numbered PPMs in `<output-path>.frames/`, and the exact `ffmpeg`
command that assembles them is printed - a generation is never thrown away for
want of an encoder.

`--steps` and `--frames` are the two knobs that decide how long a run takes.
Halving the frame count more than halves the transformer cost, because
attention is quadratic in the token count and the token count is linear in
frames. Start small (`--frames 9 --width 256 --height 256 --steps 8`) to check
a set of weights before committing to a long run.

## Serving it

Wan declares one action, `t2v`, through the generalized capability interface -
the same schema `brain caps` prints and the D-Bus and event surfaces dispatch:

```
brain caps brain/wan
```

Only `--prompt` is required; every other parameter defaults from the action's
own schema, which is the variant's own configuration. The output is a clip
blob (N RGB frames plus the frame rate), which a client writes to a real
container.

Over D-Bus, `t2v` is a streaming `Subscribe` job: progress arrives per denoise
step and the clip comes back as a file descriptor. It is **cancellable** -
`Cancel(job)` flips the job's token and the denoise loop aborts at its next
step boundary, which matters for a model whose default run occupies a card for
a long time. `examples/videogen/` is the runnable client.

The action is not reachable as a CLI verb of its own: `wan` has a dedicated
CLI module, and the resolver gives those precedence over generic capability
dispatch, so `brain wan t2v` runs that module rather than the action. Both
drive the same pipeline, so this is a routing difference, not a behavioural
one - the same state `flux2-klein` is in.

Served, the transformer stays resident between requests, keyed on the variant
and the latent extent - the only things that fix its graphs. Two requests that
differ only in prompt, seed, steps, guidance or solver reuse the same build;
changing the frame count or the size rebuilds it. The text encoder and the VAE
are still built and dropped per request, for the memory reason below.

## Variants

| Variant | Parameters | Sizes | Steps | Shift |
|---|---|---|---|---|
| `t2v-1.3B` | 1.3 B | 832x480, 480x832 | 50 | 5.0 |
| `t2v-14B` | 14 B | + 1280x720, 720x1280 | 50 | 5.0 |
| `i2v-14B` (480p) | 14 B | 832x480, 480x832 | 40 | 3.0 |
| `i2v-14B` (720p) | 14 B | + 1280x720, 720x1280 | 40 | 5.0 |

Note that 720p exists only on the 14 B tier; the 1.3 B variant is 480p-only
upstream. The shift value is a task-and-size rule rather than a resolution rule:
only image-to-video at 480p departs from 5.0.

## Getting the weights

The default checkpoint is `Wan-AI/Wan2.1-T2V-1.3B` (17.6 GB), which is
self-contained - transformer, VAE, umT5-XXL text encoder and tokenizer all live
in the one repository. The `-Diffusers` variant of the same model is 28.9 GB
because it stores the text encoder in fp32 rather than bf16; brain reads the
native repository by default and uses the diffusers naming only for importing.

You do not have to fetch it by hand. A command that names none of the four
paths - and has none of the four variables exported - downloads that repository
into the model store first and points itself at what landed:

```
brain wan t2v --prompt "..." --seed 42 --output-path out.mp4
```

Naming all four paths on the command line, or exporting all four variables,
skips the fetch entirely; a partial set still fetches, because a fetch that
resolved only the missing roles would mix two checkpoints.

```
export BRAIN_WAN_DIT=.../diffusion_pytorch_model.safetensors
export BRAIN_WAN_VAE=.../Wan2.1_VAE.pth
export BRAIN_WAN_T5=.../models_t5_umt5-xxl-enc-bf16.pth
export BRAIN_WAN_TOKENIZER=.../google/umt5-xxl
```

A Wan GGUF (the transformer alone - `general.architecture = "wan"`) converts
with `brain import <file>`, which reads the variant off the checkpoint's own
tensor shapes. The VAE, text encoder and tokenizer still come from the native
repository above, so an imported GGUF replaces one of the four roles rather
than standing on its own.

The tokenizer variable names the *directory*, not a file: brain reads that
directory's `tokenizer.json` rather than the `spiece.model` protobuf beside it.
Both describe the same 256k pieces, but the JSON is what
`AutoTokenizer.from_pretrained` loads and therefore what upstream Wan actually
tokenizes with, and it spells out its own normalizer and pre-tokenizer instead
of leaving them to sentencepiece's defaults. See [t5encoder](t5encoder.md).

Image-to-video additionally needs a CLIP ViT-H/14 vision tower in
`BRAIN_WAN_CLIP`. That one is not auto-fetched, because it ships with the I2V
checkpoints rather than the text-to-video one.

## Hardware and limits

81 frames at 480p is 32,760 transformer tokens; at 720p it is 75,600. A dense
attention score matrix at those lengths would be 4.3 GB and 22.9 GB per head
respectively, so Wan always runs through chunked or flash attention - that is a
requirement here, not a tuning option.

The three models are never resident at once. The text encoder runs first and is
dropped before the transformer loads, because umT5-XXL is 22.72 GB in fp32 and
does not fit a 24 GB card at all; it therefore runs on the CPU by default
(`--t5-device`). The transformer is dropped in turn before the VAE decodes.

One transformer forward is the whole 30-block stack in a single submit, which
is minutes at 480p rather than seconds, so `brain wan t2v` raises the backend's
<!-- perf-number: BRAIN_GPU_WAIT_S is a configured timeout the command sets, not a measurement -->
`BRAIN_GPU_WAIT_S` deadlock guard to 1200 s unless the caller has already set
it. Two forwards run per step whenever guidance is above 1, since that is what
classifier-free guidance means.

<!-- perf-number: 4x/8x are the VAE's fixed compression strides, an architectural constant, not a speedup -->
The video autoencoder compresses time 4x and space 8x, and gives the first frame
a latent frame of its own. Frame counts therefore have to take the form
`1 + 4k`: 81 frames become 21 latent frames. Asking for 80 is rejected rather
than quietly rounded.
