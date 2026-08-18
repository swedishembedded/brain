# Wan (video generation)

Wan turns a text prompt into a short video: give it a sentence and it produces a
clip of about five seconds, 81 frames at 16 fps. <!-- perf-number: the output frame rate is a property of the video the model generates, not a throughput claim -->
It is the first video model in brain, and the smallest variant is deliberately
modest: Wan2.1-T2V-1.3B is a consumer-card model rather than a datacentre one.
Upstream quotes about 8 GB of VRAM for it at bf16; brain runs it in fp32, and a
33-frame 832x480 request was observed holding about 14.9 GB on the card while
the transformer was resident.

Reach for it when you want motion. For a single still frame the image models
([s3dit](s3dit.md), [flux2](flux2.md)) are far cheaper - Wan spends most of its
work making frames agree with each other over time, which is wasted if you only
need one. Reach for it knowing it is slow: a full-size clip occupies a card for
half an hour or more, so start small.

`wan` names the whole upstream family rather than one release, the same way
`qwen3` covers every Qwen3 size. Wan2.1 and Wan2.2 share one architecture, one
Hugging Face class and one GGUF architecture tag (`wan`); which release you have
is a matter of configuration, not a different model.

## Support

| Capability | Supported |
|---|---|
| Inference (text to video) | [x] |
| Inference (image to video) | [ ] |
| LoRA fine-tune         | [x] (library only, no command) |
| INT8                   | [ ] |
| CLI (`brain wan t2v`)  | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

Text-to-video runs end to end from one command and is served like every other
model: a weights-free manifest `brain caps brain/wan` prints, a residency
adapter the scheduler budgets and places, and a cancellable streaming job over
D-Bus. Image-to-video does not exist yet.

LoRA fine-tuning is a gradient-checked library path (`wan::finetune`) rather
than a command. The transformer's forward and backward, the adapter and the
captioned-clip dataset are all there and gated, but nothing exposes them as
`brain wan ...` or as a capability action, so training today means calling
`wan::finetune::run` from Rust. That trainer runs on the host, which is
practical for short adapter runs at small latent extents and is not a path to
training the full 1.3B model.

There is no HTTP surface because there is no OpenAI-shaped or Anthropic-shaped
endpoint for video generation to fit into; D-Bus and the capability interface
are the served paths. Batched serving is listed as unsupported deliberately
rather than silently: concurrent requests at the same size share one resident
transformer, but each denoises in turn - the engine records one graph for one
latent volume and holds a single text-context buffer, so a genuinely batched
forward is a change to the engine, not to the adapter.

## Getting the weights

Model id: `brain/wan` on the capability surface; the checkpoint it fetches is
`Wan-AI/Wan2.1-T2V-1.3B` (17.6 GB). Weights auto-fetch on first use - no env var
or manual download needed. That one repository is self-contained:
transformer, VAE, umT5-XXL text encoder and tokenizer all live in it, which is
why it is the default rather than the `-Diffusers` export (28.9 GB, because it
stores the text encoder in fp32 rather than bf16; brain reads the native
repository and uses the diffusers naming only for importing).

A command that names none of the four paths, and has none of the four variables
exported, downloads that repository into the model store first and points
itself at what landed. Naming all four paths, or exporting all four variables,
skips the fetch entirely; a partial set still fetches, because a fetch that
resolved only the missing roles would mix two checkpoints.

```bash
export BRAIN_WAN_DIT=…/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
export BRAIN_WAN_VAE=…/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
export BRAIN_WAN_T5=…/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
export BRAIN_WAN_TOKENIZER=…/Wan2.1-T2V-1.3B/google/umt5-xxl
```

The tokenizer variable names the *directory*, not a file: brain reads that
directory's `tokenizer.json` rather than the `spiece.model` protobuf beside it.
Both describe the same 256k pieces, but the JSON is what
`AutoTokenizer.from_pretrained` loads and therefore what upstream Wan actually
tokenizes with, and it spells out its own normalizer and pre-tokenizer instead
of leaving them to sentencepiece's defaults. See [t5encoder](t5encoder.md).

A Wan GGUF (the transformer alone, `general.architecture = "wan"`) converts with
`brain import <file>`, which reads the variant off the checkpoint's own tensor
shapes. The VAE, text encoder and tokenizer still come from the repository
above, so an imported GGUF replaces one of the four roles rather than standing
on its own.

The conversion is fp32, so plan for disk, not RAM: the released 14B
(`city96/Wan2.1-T2V-14B-gguf`, 7.0 GB at Q3_K_S) becomes a ~53 GiB
safetensors checkpoint. Host memory stays bounded at a few GiB regardless of
model size, because the importer streams tensor by tensor rather than
materialising the whole model.

## Running it

```bash
brain wan t2v --prompt "A belgian malinois running on a paved highway" \
              --seed 42 --output-path out.mp4

# start small to check a set of weights before committing to a long run
brain wan t2v --prompt "a golden retriever running along a sandy beach at sunset" \
              --frames 9 --width 416 --height 240 --steps 20 --output-path dog.mp4
```

Everything else has a default taken from the variant's own configuration, and
every weight path is both a flag (`--dit`, `--vae`, `--t5`, `--tokenizer`) and
an environment variable, with the flag winning. `make wan/t2v` runs the small
command above.

Writing the container needs the `ffmpeg` CLI. Without it the frames are still
written, as numbered PPMs in `<output-path>.frames/`, and the exact `ffmpeg`
command that assembles them is printed - a generation is never thrown away for
want of an encoder.

Wan also declares one action, `t2v`, through the generalized capability
interface - the same schema `brain caps` prints and the D-Bus and event
surfaces dispatch:

```bash
brain caps brain/wan                     # discovery, no weights needed
brain serve --dbus                       # then drive it from examples/videogen/
```

Only `prompt` is required; every other parameter defaults from the action's own
schema, which is the variant's own configuration. The output is a clip blob (N
RGB frames plus the frame rate), which a client writes to a real container.
Over D-Bus, `t2v` is a streaming subscription: progress arrives per denoise step
and the clip comes back as a file descriptor. It is **cancellable** - a cancel
flips the job's token and the denoise loop aborts at its next step boundary,
which matters for a model whose default run occupies a card for a long time.
`examples/videogen/` is the runnable client for both.

The action is not reachable as a CLI verb of its own: `wan` has a dedicated CLI
module and the resolver gives those precedence over generic capability
dispatch, so `brain wan t2v` runs that module rather than the action. Both drive
the same pipeline, so this is a routing difference, not a behavioural one - the
same state `flux2-klein` is in.

Served, the transformer stays resident between requests, keyed on the variant
and the latent extent - the only things that fix its graphs. Two requests that
differ only in prompt, seed, steps, guidance or solver reuse the same build;
changing the frame count or the size rebuilds it. The text encoder and the VAE
are still built and dropped per request, for the memory reason below.

## Options

- `--frames <N>` - must be `1 + 4k` (81 by default). The causal video
  autoencoder gives the first frame a latent frame of its own, so 81 frames
  become 21 latent frames; asking for 80 is rejected rather than quietly
  rounded.
- `--width` / `--height` - multiples of 16 (832x480 by default).
- `--steps` (50), `--seed`, `--fps` (16).
- `--guidance` (5.0) - classifier-free guidance. Above 1.0 a step is TWO
  transformer forwards; at or below 1.0 it is one.
- `--shift` (5.0) - the flow-matching sigma shift. It is a task-and-size rule
  upstream rather than a resolution rule: only image-to-video at 480p departs
  from 5.0.
- `--solver unipc|dpm++` - the multistep solver. The two do not share a
  schedule: UniPC starts at the training grid's top sigma, dpm++ at exactly 1.0.
- `--variant t2v-1.3B|t2v-14B`, `--negative-prompt`, `--output-path`.
- `--device cpu|gpu` for the transformer and VAE, `--t5-device cpu|gpu` for the
  text encoder (see below for why they are separate).

| Variant | Parameters | Sizes | Steps | Shift |
|---|---|---|---|---|
| `t2v-1.3B` | 1.3 B | 832x480, 480x832 | 50 | 5.0 |
| `t2v-14B` | 14 B | + 1280x720, 720x1280 | 50 | 5.0 |
| `i2v-14B` (480p) | 14 B | 832x480, 480x832 | 40 | 3.0 |
| `i2v-14B` (720p) | 14 B | + 1280x720, 720x1280 | 40 | 5.0 |

720p exists only on the 14 B tier; the 1.3 B variant is 480p-only upstream. The
two image-to-video rows are the configuration surface, not a supported path -
see the support table.

## Hardware and limits

Measured on one Tesla P40 (24 GB, Vulkan) with the umT5-XXL text encoder on the
CPU, at fp32 throughout:

| Request | Text encode | Transformer load | Denoise | VAE decode | Total |
|---|---|---|---|---|---|
| 33 frames, 832x480, 25 steps | 237 s | 12 s | 1764 s | 102 s | 35.3 min | <!-- perf-number: a real end-to-end run of the shipped CLI on one named card, which is what this section exists to report -->

The same request measured 57.5 min before the VAE convolution was lowered to a  <!-- perf-number: the before figure for the same named run in the table above; the whole point of the sentence is the delta -->
GEMM, the cross-attention scores were made coalescing and the host stages were
parallelised - 1.63x, at unchanged output (cosine 1.000000000 against the  <!-- perf-number: the measured ratio of the two runs above, stated so the parity claim beside it is anchored to a specific change -->
reference transformer at every block, 1.000000 against the reference VAE at
every stage). The two smaller rows this table used to carry were measured
before those changes and have not been re-taken.

Note the card throttles: under sustained load a P40 drops from 1531 MHz to
about 999 MHz at 90 C, so a long request costs perceptibly more per step at the
end than at the start, and short benchmarks overstate what a full run achieves.

Three things that table is saying:

- **The text encode is a fixed tax on every generation**, and at small sizes it
  is most of the run. umT5-XXL is 22.72 GB in fp32 and provably does not fit a
  24 GB card - a single 256384x4096 embedding table is already past this
  backend's 4094 MiB per-buffer ceiling - so it runs on the CPU and there is no
  flag that makes that cheap today. INT8 is the crate's own stated answer and
  is not implemented.
- **The VAE decode is pure 3D convolution at every layer**, lowered to
  `im2col` + a tiled GEMM rather than run as a direct convolution. That is worth
  8.6x on this phase and is why the decode is no longer a large minority of a  <!-- perf-number: measured phase speedup from the im2col lowering, cited as the reason the decode stopped dominating a short clip -->
  short clip; a handful of low-channel convolutions still take the direct
  kernel, which is faster for them.
- **Cost in the transformer is superlinear in size.** 81 frames at 480p is
  32,760 tokens per forward and 720p is 75,600; attention is quadratic in that
  count and the count is linear in frames. Halving the frame count more than
  halves the transformer cost.

The three models are never resident at once, and that staging is a design
constraint rather than an optimisation. The text encoder runs first and is
dropped before the transformer loads; the transformer is dropped in turn before
the VAE decodes. `--t5-device` exists because of it.

A dense attention score matrix at Wan's sequence lengths would be 4.3 GB per
head at 480p and 22.9 GB at 720p, so self-attention always runs through flash or
query-chunked attention. That is a correctness requirement here, not a tuning
option.

One transformer forward is the whole 30-block stack in a single submit, which at
any real size is minutes rather than seconds, so `brain wan t2v` raises the
<!-- perf-number: BRAIN_GPU_WAIT_S is a configured timeout the command sets, not a measurement -->
backend's `BRAIN_GPU_WAIT_S` deadlock guard to 1200 s unless the caller has
already set it. `brain serve` does not raise it, because the same process may
be serving models for which that guard is doing its job - export it yourself
when serving Wan.

Image-to-video additionally needs a CLIP ViT-H/14 vision tower in
`BRAIN_WAN_CLIP`. That one is not auto-fetched, because it ships with the I2V
checkpoints rather than the text-to-video one, and the I2V path is not
implemented.
