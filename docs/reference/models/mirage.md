# Visual perception (Mirage)

A 51M-parameter vision model that reads an image and produces structured
output: a scene description, object boxes and masks, text lines read from the
image, regions, relations, events and the answer to a form. It is small enough
to run in real time on a phone-class CPU. Both its weights and its published
code are Apache-2.0.

brain runs the released float32 weights on the CPU backend and on a GPU,
batched over images, behind the same capability, scheduler, SDK, CLI, HTTP and
D-Bus surfaces as every other model.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [ ] |
| CLI (`brain mirage <action>`) | [x] |
| HTTP API              | [x] |
| D-Bus                 | [x] |
| Batched serving       | [x] |
| SDK (`PerceptionPipeline`) | [x] |
| brain-py (`describe`, `extract`) | [x] |

## Getting the weights

Model id: `Cactus-Compute/mirage`.

```bash
brain pull Cactus-Compute/mirage
```

fetches the three published files into the models directory and serves them
exactly as published - there is no conversion step: `config.json`,
`mirage.safetensors` (float32 weights) and `mirage.cact` (the deployment
container, which also holds the tokenizer and the reader's character set, so it
is part of the model and not an optional extra). A repository placed by hand
under `Cactus-Compute/mirage/` is found the same way, by what its `config.json`
declares (`model_type: "mirage"`), with no variable to set.

## Running it

```bash
brain caps Cactus-Compute/mirage

brain mirage describe --image photo.jpg --json
brain mirage extract  --image photo.jpg --json \
  --form '{"animal": {"enum": ["cat", "dog", "bird"]}, "count": {"type": "integer", "minimum": 0, "maximum": 9}}'
```

The same two actions are reached through `POST /v1/run`, the D-Bus `Run`
method, `brain_py` and the SDK. They all run one implementation
(`mirage::caps::MirageSession`), so for the same picture and options every
surface returns the same JSON.

```rust
let pipe = brain::PerceptionPipeline::from_pretrained("Cactus-Compute/mirage")?;
let photo = brain::Image::open("photo.jpg")?;
let described = pipe.describe(&photo)?;          // described.json(), .objects(), .text_lines(), .usage()
let answered = pipe.extract(&photo, &serde_json::json!({"animal": {"enum": ["cat", "dog"]}}))?;
```

```python
from brain_py import Brain
brain = Brain()
described = brain.describe(pil_image, sections=["objects", "text"], mask_format="bitmap")
answered = brain.extract(pil_image, {"animal": {"enum": ["cat", "dog"]}})
described.usage            # images, batch, decoder_tokens, vision_ms, decode_ms, ...
```

Samples: [`samples/vision/perception`](../../../samples/vision/perception/README.md)
(Rust, through the SDK) and
[`samples/python/vision/mirage-perception`](../../../samples/python/vision/mirage-perception/README.md)
(Python over D-Bus, with `--concurrent N`).

## Actions

Both take the `image` input: raw interleaved HWC `f32` in `[0, 1]` (the CLI and
the SDK encode a PNG, JPEG or PPM for you). An image may declare at most 36
million pixels and its payload must be exactly `w x h x 3` floats.

### `describe`

| Param | Meaning |
|---|---|
| `sections` | comma-separated sections to return and decode, of `scene objects regions text relations events reasoning` (default all). Asking for fewer also shortens decoding. |
| `fields` | comma-separated per-item fields, of `bbox score mask position description count` (default `bbox,score,mask,position,description`) |
| `conf`, `text_conf` | minimum object and text-line confidence, 0 to 1 (default 0.3) |
| `max_objects`, `max_text_lines` | most objects (1 to 64) and text lines (1 to 32) considered |
| `max_tokens` | decoding budget, 1 to 1536 (default 512) |
| `mask_threshold` | bitmap and polygon threshold, 0 to 1 (default 0.5) |
| `mask_format` | `largest` (default: the largest polygon as a flat `[x, y, ...]` list), `merged`, `polygons`, or `bitmap` (column-major run-length at the source size) |
| `language` | the language the model writes in: `en es fr pt de it nl pl ru ar zh` (default `en`) |
| `precision` | `fp32` (the safetensors weights, the default whenever they exist), `cq` (the packed deployment archive `mirage.cact`) or `cq-a8` (the archive with int8 activations, as the engine runs it); see [Precision](#precision). `extract` takes it too. |

The outcome has one output per section that was requested and produced -
`scene` (attributes), `objects` (`id`, `name`, `bbox` in 0..1000 coordinates of
the source image, `score`, `mask`, `position`, `description`), `regions`,
`text` (`string`, `bbox`, `score`), `relations` (`subject`, `predicate`,
`object`), `events` and `reasoning` - plus `usage`.

### `extract`

`form` (required) is a JSON object of fields, each one of

- `{"enum": [...]}` - choose one value (strings or numbers, at most 256),
- `{"type": "integer", "minimum": a, "maximum": b}` - a count (at most 256 values),
- `{"type": "string", "maxLength": n}` - free text (at most 256 characters, 60 by default),
- `{"type": "string", "format": "box", "x-source": "<choice field>"}` - the object the choice names: after
  the committed answer of the source field the model writes the value itself, like free text. `x-source` must name
  a choice field of the form; a box is placed right after its source whatever its position in the JSON.

The other params are `conf`, `text_conf`, `max_objects`, `max_text_lines`,
`max_tokens` and `language` as above. The outcome is
`{"form": {"<field>": {"value": ..., "probabilities": {"<candidate>": p, ..., "-": p}}}}`:
the probability of every candidate with abstention (`-`) last, and `value` is
`null` when the model abstains. A text or box field has no candidate table: it answers
`{"value": "<generated text>", "probability": p}` with the product of the probabilities of the generated tokens
(`-` is the model's abstention). A form is refused above 64 KiB, 32 fields or
1024 candidate values in total: every candidate is scored by a decoder pass.

### Usage accounting

Every outcome carries `usage`: `images`, `batch` (images in the device batch the
request ran in), `decoder_tokens` (0 for `extract`), and the latency split of
that batch in milliseconds - `vision_ms` (letterbox, upload, encoder, dense
head), `memory_ms` (proposals and decoder memory), `decode_ms`, `post_ms` (masks,
reader, JSON). `extract` adds `memory_reused`.

## Serving and batching

`brain serve` loads the model once per device and serves many callers:

- **Batching.** All requests share one instance with a batch limit of 64
  images. Whatever is queued when the instance frees up is one group; requests in
  a group with equal options are one `describe_batch` (a batched encoder over
  every image and one lockstep decoder loop), requests with different options
  are separate calls of the same group. A request is answered with what it would
  have got alone: bit-identical on the CPU backend; on an accelerator the GEMM
  selector may pick another kernel for another row count, so scores and mask
  pixels at the threshold can move in the last place.
- **Cancellation.** A request cancelled before its group starts takes no slot in
  it; one cancelled while the group decodes stops costing decoder steps once
  every request of the group has stopped. Either way it is answered `cancelled`.
- **Admission.** A request that can never run - an option out of range, an
  unknown section, a form past its bounds, an image past the pixel bound - is
  refused at the door with the offending field named, before a device is
  claimed; the transports bound the queue and the wait to be admitted.
- **Reuse between actions.** The decoder memory of an image (what the decoder
  reads of the vision half) depends only on the pixels and the proposal options
  (`conf`, `text_conf`, `max_objects`, `max_text_lines`). A `describe` keeps it
  and a later `extract` of the same image under the same options skips the
  vision half. The cache is keyed by a hash of the image content, the
  preprocessing version, the weights revision and those options; it lives in the
  served instance, holds at most 256 MiB, evicts the least recently used entry and
  is dropped with the instance.
- **Unload.** Dropping the instance (the scheduler evicts it, or the server
  stops) frees the weights, the batch buffers and the cache.

`brain serve` reports `mirage_batches`, `mirage_images`, `mirage_widest_batch`,
`mirage_memory_hits` and the cache size among the instance metrics.

## What is different from the reference engine

- **The weights are float32.** The vendor's engine runs the 2-bit `.cact`; brain
  runs `mirage.safetensors`. Detections, boxes, text and structure agree on the
  vendor's corpus, but the wording of a description, the reader's spelling and
  every probability can differ from the engine's 2-bit outputs.
- **Polygon vertices are region-equivalent, not vertex-identical.** Masks are
  traced at the model's field resolution and simplified with the engine's
  tolerance; the polygons cover the same region (rasterised IoU 0.97 to 1.00
  over the vendor's corpus) but their vertex lists differ. `bitmap` masks are
  exact.
- **Quantised execution is opt-in.** `precision: cq` serves the archive's 2-bit
  matrices packed on the device (see [Precision](#precision)); the engine rounds
  activations to int8, brain keeps them in float32 unless `precision: cq-a8` is
  asked for, so exact tokens are not expected to match the engine's either.

## Precision

```bash
brain mirage describe --image photo.jpg --precision cq --json
brain mirage describe --image photo.jpg --precision cq-a8 --json
brain mirage quantize --dest ./mirage-cq [--weights DIR]
```

`fp32` (default) keeps every weight as float32: about 206 MB resident (the 65-tap
HConvGLU bands and the Hadamard rotations run as structured kernels, not as dense
`n x n` matrices). `cq` keeps the archive's projections, embedding and engram
tables packed (2 to 4 bits per weight): about 30 MB resident, a few points of
agreement with the float32 answers (top-1 of the decoder logits 216/237 on the
reference photograph) and similar throughput. `cq-a8` serves the same packed
weights and runs the products over them the way the engine does (W2A8): after the
128-lane Walsh rotation each activation group is rounded to int8 with one scale
(`absmax / 127`, half away from zero), the codebook is reduced to an int8 table and
a group's dot product accumulates in integers. It is lower precision than `cq`
(measured against the float32 logits: the same top-1 agreement and a mean KL within
0.0003 nats of `cq`'s), never a default, and covers the products over packed matrices
only (the convolutions and the attention keys and values keep float32). One
precision is resident at a time; asking for another replaces it.

`brain mirage quantize` writes `mirage.cact` (plus a copy of `config.json`, so the
output directory loads as a model directory) from the float32 checkpoint, folding an
adapter in first. The quantisation plan (which tensors are packed at which width,
the codebooks, the tokenizer) is read from the `mirage.cact` already in the weights
directory. The file is written under a temporary name and renamed.

## Fine-tunes

Mirage fine-tunes through the Rust crate (`mirage::train`): full, head-only
(reader, dense head, decoder) or LoRA on the dense projections, with AdamW, a
global gradient clip, exact resume and per-term loss weights. The host trainer
(`train::trainer::Trainer`) and the device trainer
(`device::train::trainer::DevTrainer`, selected by handing it a `Device` on any
backend: CPU JIT, Vulkan, CUDA) take the same settings and write the same files,
so a run can be resumed on either. A training run writes a model
directory: `mirage.safetensors` (the trained weights in checkpoint layout, or the
unchanged base for a LoRA run) and, for LoRA, `adapter.safetensors`. After
`train::prepare::make_servable` (which copies `config.json` and the deployment
file beside them) every surface that opens a model directory serves it
unchanged; an `adapter.safetensors` next to the weights is folded into them when
the directory is opened. Both trainers take the shared saved-activation policy
(`set_saved_policy`, `model::saved::SavedPolicy`): `Keep` holds every intermediate
of the forward for the backward, `Recompute` holds only the output of each
recompute block (stem stage, body block, top layer, dense head, mask branch,
reader stage and mixer, engram site, decoder layer) and rebuilds one block at a
time inside the backward. The gradients, and so the optimiser step, are the same
either way (bit for bit on the CPU backend); a narrow boundary format is refused.
Attention never stores its `tq x hq x tk` probabilities: the backward rebuilds
them per row from the queries, keys and one saved log-sum-exp per row. The
training objective is brain's own: the vendor
published no recipe. Training records are JSONL with task-presence semantics
(`mirage::train::data`): a missing label family is unknown, never a negative.

## Hardware and limits

Weights are 216 MB on disk; a served instance holds the folded mixer matrices
and the activations of a full batch on the device. Throughput depends on the
device and the batch (see the ledger in `.agents/roadmap/mirage.md` for measured
curves).
