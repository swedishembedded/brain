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

Takes a local checkpoint path or a `<vendor>/<repo>` hub id through the same
call - a string naming a real file or directory on disk is always the local
path. Any Qwen3, Qwen2 or Llama checkpoint loads as downloaded: a Hugging Face
directory, a GGUF or a brain `.safetensors` file. A directory reads the
`tokenizer.json`, chat template and `generation_config.json` beside its
weights, and a `.gguf` carries its own tokenizer; a brain-format
`.safetensors` file needs one named explicitly:

```rust
let pipe = brain::TextGenerationPipeline::builder("/models/qwen3-4b.safetensors")
    .tokenizer("/models/qwen3-4b/tokenizer.json")
    .load()?;
```

Generation stops on the checkpoint's own end tokens, and chat requests render
through its own template. A checkpoint of 6B parameters or more loads with
int8 linears (its fp32 weights do not fit one 24 GB card);
`.precision("fp32")` or `.precision("int8")` chooses explicitly, and
`pipe.precision()` reports what was built.

`pipe.generate_with(prompt, brain::TextGenerationOptions::new().max_new_tokens(256).temperature(0.7))`
layers on the common knobs; the result's `prompt_tokens`/`completion_tokens`/
`finish_reason` mirror what the served `/v1/chat/completions` endpoint
reports, since both run through the same chat-templating and sampling code.

### Chat with tools

`brain::ChatPipeline` is the multi-turn half of the same surface: a
conversation, tool schemas and sampling knobs in; the visible answer, the
reasoning, typed tool calls, a finish reason and token counts out.

```rust
use brain::{ChatMessage, ChatPipeline, ChatRequest};

let chat = ChatPipeline::from_pretrained("unsloth/Qwen3-4B-GGUF")?;
let reply = chat.generate(&ChatRequest::new(vec![ChatMessage::user("Explain DMA in one sentence.")]))?;
println!("{}", reply.text);
```

Every loader knob - tokenizer, a LoRA adapter, context capacity, device - is
`TextGenerationPipeline::builder`'s; a chat pipeline is the model that
builder loaded. Tools, streaming and cancellation:

```rust
use brain::chat::{ChatDelta, ToolChoice, ToolSchema};
use brain::{CancelToken, ChatMessage, ChatPipeline, ChatRequest, TextGenerationPipeline};

let chat = ChatPipeline::from(
    TextGenerationPipeline::builder("/models/qwen3-4b.safetensors")
        .tokenizer("/models/qwen3-4b/tokenizer.json")
        .adapter("/adapters/support.safetensors")
        .capacity(16384)
        .load()?,
);
let request = ChatRequest::new(vec![ChatMessage::user("Weather in Paris?")])
    .tools(vec![ToolSchema::new("get_weather", "Current weather", serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}))])
    .tool_choice(ToolChoice::Auto)
    .thinking(false)
    .max_tokens(512);
let cancel = CancelToken::armed();
let reply = chat.generate_stream(&request, &cancel, |delta| {
    if let ChatDelta::Text(text) = delta {
        print!("{text}");
    }
})?;
for call in &reply.tool_calls {
    println!("{} {} {}", call.id, call.name, call.arguments);
}
```

- A tool's result goes back as `ChatMessage::tool(call_id, content)`, after
  the assistant turn that made the call
  (`ChatMessage::assistant("").with_tool_calls(calls)`).
- `reply.finish_reason` is `Stop`, `StopSequence`, `Length`, `ToolCalls`,
  `ToolChoiceUnmet` (a `Required`/`Named` tool choice the model did not
  honour) or `Cancelled` (the token fired; the reply is partial).
- `reply.usage.prompt_tokens`/`completion_tokens` are `Option<u32>`: a count
  that was not measured is `None`, never `0`.
- `max_tokens` is an upper bound, and so is the context: a budget bigger than
  what the prompt leaves generates until the context is full and reports
  `Length`. A prompt that fills the context alone is an error.
- Cancellation is noticed between prefill chunks (512 prompt tokens by
  default, `ChatPipeline::prefill_chunk`) and after every token.
- `chat.identity()` names what was loaded by content: the base weights file
  and, when one is attached, the adapter - each with its card id and a
  `sha256:` digest of the file.
- The builder's `.adapter(path)` folds the adapter into an fp32 base at
  load, which is exact and leaves decode as fast as the base. On an int8
  base, where a fold would round the adapter away, the adapter runs beside
  the base instead. `chat.attach_adapter(path)` switches to another adapter
  from the next turn on, beside the base, and `chat.detach_adapter()`
  returns to exactly the base. Neither reloads the model; taking out a
  folded adapter re-reads only the linears it changed. A file that is not a
  LoRA adapter for this base is refused, and the pipeline keeps what it had.
  `TextGenerationPipeline` has the same two methods.
- `request.render_prompt()` and `request.parse_reply(raw)` need no model:
  the prompt the request renders to, and a recorded completion parsed the way
  a generation's own is.
- `brain::ChatTokenizer` turns that prompt into token ids and ids back into
  text, for a loop that trains or scores on ids and must feed the model the
  tokens serving would: `ChatTokenizer::from_model_dir(dir)?.prompt_ids(&request)?`
  is the rendered prompt tokenized by the checkpoint's own tokenizer, and
  `.decode(&ids)` reads a generation back. There is no second template to drift
  from the first.

A pipeline runs one generation at a time (it is `Send`, not `Sync`): give it
its own thread, or share it behind a `Mutex`.

### Fine-tuning a chat model

`brain::ChatFineTune` (the `study` feature) trains a LoRA adapter on a chat
dataset - `generic-messages-v2` JSONL, one conversation per line, `train`
per message marking what is supervised - and exports the adapter plus a
training record. `brain::score_chat` is the held-out measurement: mean
per-token cross-entropy over the supervised positions of held-out records,
for the base alone or with an adapter folded in exactly as serving folds it.

```rust
let outcome = brain::ChatFineTune::from_pretrained("/models/qwen3-0.6b/model.brain.safetensors")
    .dataset("train.jsonl")
    .held_out("held_out.jsonl")
    .replay("earlier.jsonl")
    .out_dir("runs/support-1")
    .rank(8)
    .steps(200)
    .run()?;
println!("{:?} -> {:?}", outcome.base_score, outcome.tuned_score);
let chat = brain::ChatPipeline::from(
    brain::TextGenerationPipeline::builder("/models/qwen3-0.6b/model.brain.safetensors")
        .tokenizer("/models/qwen3-0.6b/tokenizer.json")
        .adapter(outcome.adapter.as_ref().unwrap().to_str().unwrap())
        .load()?,
);
```

- The tokenizer (`tokenizer.json`) and chat template are read from the base
  checkpoint's directory, and every dataset is checked against them
  (`validate_chat_dataset_for`) before a device is claimed.
- `.replay(path)` mixes every record of another dataset into training; call
  it once per file. The mix is exactly the union of the files.
- `.continue_from(adapter)` trains an existing adapter further at its own
  rank and alpha; its digest is recorded as the new adapter's parent
  (`trained_from` on the outcome and on the adapter card's training
  provenance).
- `.rank`, `.alpha`, `.steps`, `.lr`, `.seed`, `.max_block` (the longest
  training row; a longer record is refused, not truncated) and `.device`
  are the knobs.
- `.monitor(path)` is a set of records scored during training, every
  `.eval_every(n)` steps (one deterministic pass over every record each
  time), never trained on; without one the held-out set is monitored. The
  evaluations are the run's curve (`outcome.curve`: step, the mean training
  loss since the previous evaluation, the monitoring loss). With
  `.patience(k)` the run stops once the monitoring loss has not improved for
  `k` evaluations, and with either `.patience(k)` or `.keep_best(true)` the
  adapter exported is the evaluation's with the lowest monitoring loss, not
  the last step's; `outcome.selected_step` and `outcome.selection` say which
  and why, as do the training record and the adapter card's hyperparameters.
  A selection made on the held-out set biases the held-out score it is then
  measured by, so a caller that decides anything on that score gives the run
  a monitoring set of its own.
- `run_with(&cancel, |progress| ..)` reports every optimizer step and stops
  at the next step boundary once the `CancelToken` fires. A cancelled run
  exports nothing and leaves its training state in the out directory
  (`outcome.resume_state`); running the same fine-tune again continues it
  and ends at the same adapter, bit for bit, an uninterrupted run produces -
  its monitoring curve and best adapter so far included. `.checkpoint_every(n)`
  also saves that state every `n` steps. A state from a run with different
  data, options or starting point is refused.
- The outcome's losses and scores are `Option`s: a value that was not
  measured (no held-out set, a cancelled run) is `None`.
- The adapter card records the digest of the base it was trained against
  (`base_digest` in its training provenance), so `brain serve --adapter`
  refuses to fold it into any other base. `outcome.base_digest` is that
  same digest, so a caller binding the adapter to its base need not hash the
  base again; `outcome.adapter_digest` is the digest `brain serve` prints
  for the adapter it serves.
- `samples/study/chat` is the worked example: train, score, resume a
  cancelled run, chat with the adapter.

### Preference (DPO) fine-tuning a chat model

`brain::PreferenceFineTune` (also `study`) is the DPO counterpart of
`ChatFineTune`: given pairs of a preferred and a dispreferred answer to the
same prompt, it trains a LoRA adapter so the model prefers the `chosen`
answer over the `rejected` one relative to a frozen reference.

```rust
let outcome = brain::PreferenceFineTune::from_pretrained("/models/qwen3-0.6b/model.brain.safetensors")
    .dataset("pairs.jsonl")
    .held_out("held_out_pairs.jsonl")
    .out_dir("runs/prefs-1")
    .beta(0.1)
    .steps(200)
    .run()?;
println!("{:?}", outcome.held_out_score);
```

The dataset is **`generic-preference-v1`** JSONL, one pair per line:

```json
{"prompt": [{"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "What is the capital of France?"}],
 "chosen": {"role": "assistant", "content": "Paris."},
 "rejected": {"role": "assistant", "content": "I am not sure."},
 "tools": [],
 "metadata": {"source": "review-queue"}}
```

- `prompt` (required, non-empty) is a conversation in the
  `generic-messages-v2` message shape - `role`, `content`, optional
  `tool_calls` and `tool_call_id` - **without `train`**: supervision follows
  from position, so the prompt is never supervised and the candidates always
  are. It must not end in an assistant turn.
- `chosen` and `rejected` (required) are single assistant messages of the
  same shape, and may carry `tool_calls` exactly as `generic-messages-v2`
  writes them (`function.arguments` a JSON-encoded string).
- `tools` (optional) is the tool schema array the chat template's preamble
  renders; `metadata` (optional) is an object carried for the producer and
  never read.
- Refused, by line and field: an unknown or mistyped field, a candidate that
  is not an assistant turn, `chosen` identical to `rejected`, a tool result
  answering no earlier call, and arguments that are not valid JSON.
  `brain::validate_preference_dataset_for(path, model_dir, max_block)` also
  renders both candidates through the base's own tokenizer and chat template
  and refuses a pair that does not render, whose candidates render to the
  same tokens, or that is longer than `max_block`; the fine-tune runs the
  same check before a device is claimed. `validate_preference_dataset(path)`
  is the parse-only check.

The objective is standard DPO: per pair, `-log sigmoid(beta * ((log
pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))))`,
each log-probability summed over that candidate's assistant-turn tokens
only. The reference is the model the run starts from - the base, or the base
plus the `.continue_from(adapter)` adapter - frozen: its log-probabilities are
computed once per pair before the first step and cached, so no second model
copy is held while training, and the first step's margin is exactly zero
(the initial loss is `ln 2`). `beta` defaults to `brain::DEFAULT_DPO_BETA`
(0.1). Each optimizer step trains one pair, both candidates in one forward.

- `.continue_from`, `.rank`, `.alpha`, `.steps`, `.lr`, `.seed`,
  `.max_block`, `.checkpoint_every`, `.cycle`, `.device` and
  `run_with(&cancel, |progress| ..)` behave as they do on `ChatFineTune`,
  including exact resume: a cancelled run leaves its state, and running the
  same fine-tune again ends at the same adapter, byte for byte. A state from
  a different `beta` is refused like any other different run.
- The adapter card's training provenance has regime `"dpo"`, the base's
  digest, and in its hyperparameters `beta` and the reference
  (`reference.base_digest`, `reference.adapter_digest` - the continued
  adapter, or null). `training.json` beside it records the same, plus the
  scores. `outcome.base_digest` is the base's digest, as on
  `ChatFineTuneOutcome`.
- `brain::score_preference(base, adapter, pairs)` scores base plus `adapter`
  against the base alone: `mean_margin` is the mean of `(log pi(chosen) -
  log ref(chosen)) - (log pi(rejected) - log ref(rejected))` in nats (without
  `beta`), and `accuracy` the fraction of pairs where it is positive. The
  outcome's `train_score` and `held_out_score` are the same measurement
  against the run's own reference. Unmeasured values are `None`.

## Text embedding

```rust
let pipe = brain::EmbeddingPipeline::from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")?;
let v = pipe.embed("a whale submarine")?;
println!("{} dims", v.len());
```

One type, three backbones, dispatched from `model_id`: CLIP's text towers
(CLIP-L by default; `.builder(id).tower("openclip_bigg")` for the larger
one, behind the `vision` Cargo feature), the Qwen3 decoder used the way
Qwen3-Embedding is meant to be - last-token pooled, L2-normalized, a real
32768-token context (behind the `text` Cargo feature) - and LFM2.5-Encoder,
a bidirectional encoder with its own long-context YaRN scaling (also
`text`). Neither feature is named `embedding` - every backbone already has a
registered architecture domain in this workspace (CLIP's `Vision`, Qwen3's
and LFM2's `Text`).

A literal local checkpoint path resolves to Qwen3 or LFM2 by the
checkpoint's own `ModelCard.family` (CLIP's own resolution reads a released
directory, never a bare file, so it is never in this decision at all); a
hub id tries Qwen3's resolver first (the pre-existing default), falling
back to LFM2's only when Qwen3 reports the model genuinely missing:

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
backbone; Qwen3's decode-only build and LFM2's exact-length bidirectional
build both have no batched forward, so each loops one call per string
instead - see `EmbeddingPipeline::embed_batch_with`'s own doc.
`EmbeddingOptions::dimensions(n)` truncates AND renormalizes by default,
deliberately differing from the `/v1/embeddings` HTTP endpoint, which does
not re-project after truncating. `EmbeddingOptions::instruction` is a
Qwen3-Embedding-only option, refused (not silently ignored) on the CLIP or
LFM2 backbones.

LFM2 is bidirectional: its graph is rebuilt at the EXACT request length
whenever that length changes (unmasked padding corrupts bidirectional
attention), so `.capacity(n)` bounds the longest request it will build for
rather than reserving a fixed KV cache the way Qwen3's does. Quality at a
real long context is unvalidated extrapolation past LFM2.5's native
8192-token training extent - see `.agents/roadmap/lfm2.md`.

### Contrastive fine-tuning over frozen embeddings

`brain::EmbeddingTrainer` (also `text`) trains a small linear refinement on
top of embeddings a pipeline already produced - a symmetric (CLIP-style)
InfoNCE objective over `(anchor, positive)` pairs, entirely on the host
(the batch is a few dozen vectors, nowhere near where GPU dispatch pays for
itself). It does NOT fine-tune the backbone itself - cache the pipeline's
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

### Full-encoder contrastive fine-tuning (LFM2.5-Encoder only)

`brain::EncoderFineTuner` (also `text`) trains the SAME InfoNCE objective,
but re-runs a live LFM2.5-Encoder's own forward and backward every step
instead of a frozen-embedding cache - it is training the checkpoint's own
weights, driving `lfm2::model::Lfm`'s seeded backward pass. Qwen3 has no
seeded backward pass built, so this surface is LFM2-only. Because LFM2's
bidirectional attention has no padding mask, every text in a training batch
must tokenize to at least a fixed `seq_len` (shorter is refused, not padded):

```rust
let mut tuner = brain::EncoderFineTuner::open(
    "/models/lfm2-encoder.safetensors", "/models/lfm2-encoder/tokenizer.json",
    queries.len(), 32,
)?;
for _ in 0..steps {
    let loss = tuner.step(&queries, &matching_passages, 3e-5)?;
}
tuner.save("/models/lfm2-encoder-finetuned.safetensors");
```

`samples/text/encoder-finetune` runs this end to end, reporting recall@1
before and after against `EmbeddingPipeline`.

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
| `text` | `TextGenerationPipeline` - text generation, from a local checkpoint path or a hub id; `ChatPipeline` - multi-turn chat with tool calling, streaming and cancellation; also the Qwen3/LFM2.5-Encoder backbones of `EmbeddingPipeline` (32768-token context), `EmbeddingTrainer` (contrastive fine-tuning over frozen embeddings), and `EncoderFineTuner` (full-encoder contrastive fine-tuning, LFM2 only) |
| `study` | `ChatFineTune`, `score_chat` - LoRA fine-tuning of a Qwen3 chat model and its held-out score; `PreferenceFineTune`, `score_preference` - its DPO counterpart; `DocumentStudy` - teaching a model documents behind a gate; `promote` - the promotion gate and its paired sign test (`promote::stats::sign_test`) for deciding whether a trained model replaces the one before |
| `vision` | `EmbeddingPipeline` - CLIP text embedding (named for CLIP's registered domain, not the capability) |
| `audio` | `TranscribePipeline` - speech-to-text (qwen3-asr, offline; or nemotron-asr, one-shot): the checkpoint asked for selects the model |
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
