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

## Features

Name the surfaces you use and you get their dependencies and nothing else:

| feature | adds |
| --- | --- |
| `image` | `ImagePipeline`, `Image` - text-to-image and image editing |
| `creature` | `Creature`, `View` - a connectome running a body, and a window onto it |
| `forecast` | `ForecastPipeline` - time-series forecasting |
| `text` | `TextGenerationPipeline` - text generation, from a local checkpoint path |
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
