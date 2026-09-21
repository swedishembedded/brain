# The Rust SDK

The `brain` crate (package `brain`, not `brain-sdk` - see its own doc
comment for why) is a small, embeddable library over brain's model backends:
no CLI process, no capability-dispatch server, nothing to run alongside your
application. Add it to `Cargo.toml` and call it directly.

```toml
[dependencies]
brain = { version = "1", features = ["image"] }
```

## Text-to-image

```rust
let pipe = brain::ImagePipeline::from_pretrained("black-forest-labs/FLUX.2-klein-9B")?;
pipe.generate("a whale submarine")?.save("out.png")?;
```

`from_pretrained` resolves the model id the same way `brain flux2 generate`
does - a local model directory or a `<vendor>/<repo>` hub reference, fetching
only what's missing - and returns a pipeline that's ready to call: no second
initialization step. The returned `Image` is one normalized type regardless
of which backend actually generated it, with an obvious `.save()` that picks
the file format from the extension (`.png`, `.jpg`/`.jpeg`, `.ppm`).

Common options layer on as a builder, without changing the simple call above:

```rust
let image = pipe
    .generate_with("a whale submarine", brain::ImageGenerationOptions::new().steps(30).seed(7))?;
```

A caller that needs progress reporting or the ability to cancel a
long-running generation from another thread uses the full-control entry
point both calls above delegate to:

```rust
let cancel = capability::CancelToken::default();
let image = pipe.generate_with_progress(
    "a whale submarine",
    brain::ImageGenerationOptions::new(),
    &cancel,
    &mut |step, total, message| println!("{step}/{total}: {message}"),
)?;
```

## Embodied simulation

`brain::Creature` runs a connectome driving a physics body - a different
shape from `ImagePipeline` on purpose, since it has state that survives a
call and is stepped in a loop rather than generated once:

```rust
let mut fly = brain::Creature::fruit_fly()
    .connectome("resources/connectome")
    .body("flybody/floor.xml")
    .build()?;
fly.drive(1.5);
for _ in 0..500 {
    fly.step()?;
}
println!("{:?} after one second", fly.position());
```

`brain::View` opens a window onto a running `Creature` for interactive use;
see `samples/fly/interactive` in the repository for a complete, runnable
example with keyboard control.

## Time-series forecasting

```rust
let pipe = brain::ForecastPipeline::from_pretrained("NeoQuasar/Kronos-base")?;
let series: Vec<f32> = vec![100.0, 101.2, 99.8, 102.5];
let forecast = pipe.forecast(&series, 24)?;
println!("{} steps ahead, {} target(s)", forecast.horizon, forecast.targets.len());
```

Covers kronos and timesfm3 today; `pipe.capabilities()` reports what the
resolved model actually supports (context/horizon limits, native
representation, covariate handling), and `pipe.forecast_with(&panel, &spec)`
is the full-control entry point for a multi-item, multi-variate `Panel` or
for requesting samples/a distribution instead of quantiles.

## Text generation

```rust
let pipe = brain::TextGenerationPipeline::from_pretrained("/models/qwen3-4b-q8_0.gguf")?;
let out = pipe.generate("Explain DMA in one sentence.")?;
println!("{}", out.text);
```

Loads from a local checkpoint path rather than a hub id today (qwen3 has no
model-store resolver registered yet). A `.gguf` checkpoint carries its own
tokenizer; a brain-format `.safetensors` checkpoint needs one named
explicitly:

```rust
let pipe = brain::TextGenerationPipeline::builder("/models/qwen3-4b.safetensors")
    .tokenizer("/models/qwen3-4b/tokenizer.json")
    .load()?;
```

`pipe.generate_with(prompt, brain::TextGenerationOptions::new().max_new_tokens(256).temperature(0.7))`
layers on the common knobs; the result's `prompt_tokens`/`completion_tokens`/
`finish_reason` mirror what the served `/v1/chat/completions` endpoint
reports, since both run through the same chat-templating and sampling code.

## Text embedding

```rust
let pipe = brain::EmbeddingPipeline::from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")?;
let v = pipe.embed("a whale submarine")?;
println!("{} dims", v.len());
```

One type, two backbones, dispatched from `model_id`: CLIP's text towers
(CLIP-L by default; `.builder(id).tower("openclip_bigg")` for the larger
one, behind the `vision` Cargo feature), and, for a real 32768-token
context, the Qwen3 decoder used the way Qwen3-Embedding is meant to be -
last-token pooled, L2-normalized (behind the `text` Cargo feature). Neither
feature is named `embedding` - both backbones already have a registered
architecture domain in this workspace (CLIP's `Vision`, Qwen3's `Text`).

A literal local checkpoint path always resolves to the Qwen3 backbone
(CLIP's own resolution reads a released directory, never a bare file):

```rust
let pipe = brain::EmbeddingPipeline::builder("/models/qwen3-embedding-0.6b.safetensors")
    .tokenizer("/models/qwen3-embedding-0.6b/tokenizer.json")
    .capacity(32768)
    .load()?;
let query = pipe.embed_with(
    "what does the report say about Q3 revenue",
    brain::EmbeddingOptions::new().instruction("Given a query, retrieve relevant passages"),
)?;
let passage = pipe.embed("Q3 revenue rose 12% year over year...")?;
println!("{:.3}", query.cosine_similarity(&passage));
```

`pipe.embed_batch(&["a", "b", "c"])` runs one batched forward on the CLIP
backbone; the Qwen3 backbone has no batched forward on its decode-only
build, so it loops one prefill per string instead - see
`EmbeddingPipeline::embed_batch_with`'s own doc. `EmbeddingOptions::dimensions(n)`
truncates AND renormalizes by default, deliberately differing from the
`/v1/embeddings` HTTP endpoint, which does not re-project after truncating.

### Contrastive fine-tuning over frozen embeddings

`brain::EmbeddingTrainer` (also `text`) trains a small linear refinement on
top of embeddings a pipeline already produced - a symmetric (CLIP-style)
InfoNCE objective over `(anchor, positive)` pairs, entirely on the host
(the batch is a few dozen vectors, nowhere near where GPU dispatch pays for
itself). It does NOT fine-tune the backbone itself - that needs a seeded
backward pass the engine does not have yet - so cache the pipeline's
embeddings once and train against the cache:

```rust
let anchors: Vec<brain::Embedding> = pipe.embed_batch(&queries)?;
let positives: Vec<brain::Embedding> = pipe.embed_batch(&matching_passages)?;
let mut trainer = brain::EmbeddingTrainer::new(anchors[0].dim(), 42);
for _ in 0..steps {
    let loss = trainer.step(&anchors, &positives, 0.05);
}
let refined = trainer.project(&pipe.embed("a new query")?);
```

## Speech-to-text

```rust
let pipe = brain::TranscribePipeline::from_pretrained("Qwen/Qwen3-ASR-1.7B")?;
let out = pipe.transcribe_wav(&std::fs::read("clip.wav")?)?;
println!("{}", out.text);
```

`transcribe_wav` accepts any WAV file (channel count/sample rate handled
automatically); `transcribe(samples)` takes already-16 kHz mono f32 PCM
directly. `out.truncated` is `Some((available_secs, window_secs))` when the
clip exceeded this pipeline's fixed decode window (`.builder(id).window_secs(60.0)`
to widen it) - audio past the window is dropped, and this field says so
rather than returning a silently partial transcript.

## Features

Name the surfaces you use and you get their dependencies and nothing else:

| feature | adds |
| --- | --- |
| `image` | `ImagePipeline`, `Image` - text-to-image and image editing |
| `creature` | `Creature`, `View` - a connectome running a body, and a window onto it |
| `forecast` | `ForecastPipeline` - time-series forecasting |
| `text` | `TextGenerationPipeline` - text generation, from a local checkpoint path; also the Qwen3 backbone of `EmbeddingPipeline` (32768-token context) and `EmbeddingTrainer` (contrastive fine-tuning over frozen embeddings) |
| `vision` | `EmbeddingPipeline` - CLIP text embedding (named for CLIP's registered domain, not the capability) |
| `audio` | `TranscribePipeline` - speech-to-text (qwen3-asr, offline) |
| `full` | every surface; this is the default |

## Errors

Every fallible call returns `brain::Error`, one enum covering every failure
this crate can produce - an unresolvable model id, an ambiguous or missing
role, a license gate, a missing required builder argument, and a catch-all
for anything else the underlying backend reports. Match on the specific
variants your application needs to react to differently; format the rest
with `{err}` for a human-readable message.

## Resource safety

This is a library, not a server: it has no request queue, admission control,
or concurrency limit of its own. A built pipeline holds real, multi-gigabyte
GPU/host memory for as long as it lives. An application that accepts
requests from an untrusted network and drives this SDK underneath needs its
own admission and backpressure layer in front of it - the same way brain's
own HTTP and D-Bus surfaces provide one in front of everything else in this
workspace.

Swedish Embedded AB implements client-embeddable model inference for its
clients - turning an internal research pipeline into a small, stable library
surface a product can link directly. If your team needs an SDK facade over
its own model stack, you can procure our services at
**info@swedishembedded.com**.
