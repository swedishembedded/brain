# Text-to-Speech (Qwen3-TTS)

Turns text into a 24 kHz speech waveform. Give it a few seconds of reference
audio and it clones that voice for the sentence you ask it to speak; without
a reference it falls back to a default, speaker-free voice. Reach for it when
you need narration, a cloned voice for a demo, or a spoken-output leg for an
assistant pipeline - all running locally, no external TTS service.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [x] |
| CLI (`brain <arch> <action>`)       | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched/streaming serving | [x] |

## Getting the weights

Model id: `brain/qwen3tts`. `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (Base only, the
variant with a speaker encoder for voice cloning) auto-fetches (⤓, opt-in `--autofetch`) on first
CLI use - no env var, no manual `import` step. The checkpoint (including its
nested `speech_tokenizer/` codec) is served exactly as downloaded: the
Talker, MTP, codec and speaker encoder are read from its safetensors, bf16
converted to f32 in memory as they load, and nothing is written beside it.

A CustomVoice or VoiceDesign checkpoint, or one you already have locally, is
resolved from the model store the same way; the standard role flags name one
outright: `--weights-dir` (the checkpoint, or a directory of brain-format
files) and `--ckpt` (the Hugging Face checkpoint directory, for `config.json`
and the tokenizer). The checkpoint is the provider's, never a per-call param.
Voice cloning needs the speaker encoder, which only the Base checkpoints
ship.

`--weights-dir` may instead name a directory of brain-format files, which the
`import` action writes from a checkpoint:

```bash
brain qwen3tts import --source /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  [--codec-source <dir>] [--speaker-source <dir>] \
  --dest out/tts
```

That writes `talker.safetensors`, `mtp.safetensors`, `codec.safetensors` and
(for a Base checkpoint) `speaker.safetensors`, the same tensors the
checkpoint is read as; a release with no speaker encoder (CustomVoice,
VoiceDesign) skips it and says so. A directory holding `talker.safetensors`
is read as these files; any other is read as the checkpoint itself.

## Running it

`brain qwen3tts <action> ...` runs a served action, and so does the
standalone `sample-models-qwen3tts` program (`samples/models/qwen3tts`), which
adds the NPU route. Every flag below is the action's served param; the reference
recording and the output are blobs: `--ref voice.wav` names an input,
`--out audio=out.wav` an output.

Speaker-free synthesis:

```bash
brain qwen3tts synth --text "Hello from brain." --out audio=out.wav --seed $RANDOM
```

Voice cloning from a reference clip (with `--ref-text`, the in-context path;
without, x-vector-only):

```bash
brain qwen3tts clone \
  --text "This sentence is spoken in the reference speaker's voice." \
  --ref voice.wav --ref-text "transcript of voice.wav" \
  --out audio=demo.wav --seed $RANDOM
```

Instructed voice design / preset speakers (CustomVoice/VoiceDesign
checkpoints only):

```bash
brain qwen3tts design --text "..." --instruct "a calm, low voice" \
  [--speaker NAME] --out audio=out.wav
```

`batch` decodes several texts interleaved in one call (`--requests` takes a
JSON array; request `i` is the output `audio_<i>`), and `sim --a A.wav --b
B.wav` prints the speaker-embedding cosine similarity of two utterances (the
timbre-preservation metric, e.g. a quantized Talker's clone against its
reference). The actions are in the capability manifest too (`brain caps
brain/qwen3tts`), reachable over D-Bus/HTTP.

Resident serving for repeated calls without reloading weights each time:

```bash
brain serve --dbus
```

`brain serve` serves the checkpoint the model store resolves. Its `synth`,
`clone` and `design` actions run on a hot engine and are streaming: each
decoded codec chunk (`stream_chunk` codec frames, 16 by default)
arrives as an `audio` chunk of raw mono f32 little-endian PCM at 24 kHz while
the rest is still generating, and the terminal `audio` blob is the whole clip
as a WAV. A reference voice's conditioning is computed once per recording and
kept, so repeat clones of one timbre skip the speaker encoder and the codec
encode. The other actions (`batch`, `import`, `finetune`, `sim`) are the
provider's own, run on the device the scheduler assigned. This replaces the
old `speak` action and the `BRAIN_QWEN3TTS_REF` environment variable: clone
is its own action with the reference as an input.

The event-driven stdio controller (`brain serve --stdio`, the JSONL protocol)
can also synthesize, answering a `user_synth_request` with a stream of
`audio_chunk` events. That path is **off in the default binary**: the TTS stack
is far heavier than the rest of the controller's dependencies, so it lives
behind a build feature.

```bash
cargo build --release -p brain-cli --features qwen3tts-synth
brain serve --stdio --pin qwen3tts.weights_dir=out/tts \
  --pin qwen3tts.ckpt=[path/to/Qwen3-TTS-12Hz-0.6B-Base]
```

It serves the Qwen3-TTS checkpoint the model store resolves; the pins above
name one outright. A request's own language field picks the language
(English when it names none). Without the feature, or with no checkpoint, a
`user_synth_request` is still answered - with an empty terminal
`audio_chunk`.

### On the NPU

`sample-models-qwen3tts` compiles the Talker and the codec to OpenVINO graphs
(the Intel NPU by default; the first run exports and compiles them, later
runs reuse the cache):

```bash
L=target/release/sample-models-qwen3tts
$L npu-synth --text "Hello from the NPU." --out audio=out.wav
$L npu-clone --text "..." --ref voice.wav --ref-text "..." --out audio=demo.wav
$L npu-design --text "..." --instruct "a calm, low voice" --out audio=out.wav
```

They take every flag of `synth`/`clone`/`design`, plus `--ov-device
npu|cpu|gpu|auto` (`cpu` validates the fp32 graph, which the NPU runs in
fp16), `--turbo` (the NPU's turbo clock), `--talker` (`cpu`, `npu`,
`npu-int8`, `npu-kv`, `npu-kv-fp32`, `npu-kv-int4`; unset follows the
checkpoint's size: the INT8 KV-cache graph for the large model, fp32 for the
small), `--mtp` (`cpu`, `npu`, `fused`) and `--parity` (report the Talker
prefix's difference to the CPU decoder). The NPU loops do not poll for
cancellation.

`npu-serve` is the low-latency load-once form: it keeps the compiled graphs
resident and streams synthesized audio back over a line-delimited JSON
protocol on a Unix socket (documented in `qwen3tts::socket`):

```bash
$L npu-serve --socket /tmp/brain-tts.sock --talker-quant int4 \
  --clone-ref voice.wav --clone-ref-text voice.txt
```

Its engines are `synth` (always), `clone` (with `--clone-ref`), and
`design`/`customvoice` (with `--design-weights`/`--design-ckpt`,
`--cv-weights`/`--cv-ckpt`); `--enable` names exactly which to host. An
engine without its own checkpoint flags uses the one the store resolves.
`--cap`, `--talker-quant int8|int4|fp32`, `--mtp`, `--codec
npu-stream|windowed|cpu-stream`, `--stream-chunk`, `--stream-window`,
`--npu-device`, `--turbo` and `--profile` shape the engines.
`samples/python/tts/voice-server/` holds example clients.

LoRA fine-tuning (single-speaker) of the Talker on a `text -> codes` dataset:

```bash
brain qwen3tts finetune --data data/tts --save out/tts/talker_lora.safetensors \
  [--full] [--steps N --lr X --rank R --alpha A --batch B --block T --seed S --activations keep|recompute]
```

This freezes the resolved checkpoint's Talker and trains attention adapters
only; `--full` trains every Talker weight instead (Qwen's own documented
single-speaker fine-tuning workflow). The adapters are attention-only, so the
shared LoRA parameter set's `targets` and `base_format` do not apply and are
not declared. The Talker rebuilds each layer in the backward whatever is
asked, so `--activations recompute` is accepted and `--activations keep` is
refused by name before the first step. Cancelling stops at a step boundary.

## Options

- `--lang` - synthesis language (default `english`).
- `--max-frames` - max codec frames, i.e. an upper bound on clip length
  (default `256`).
- `--temp`, `--top-k`, `--top-p`, `--seed` - sampling controls for the Talker.
- `--repetition-penalty` - codebook-0 repetition penalty. Leave it alone.
  Sampling alone does not keep the Talker out of a repetition loop: once
  codebook-0 repeats a token its next-token top-1 probability climbs toward
  1.0, and a run that locks in decodes to silence for the rest of the clip.
  `1.0` disables the penalty and re-opens that failure mode.
- `--residual-temp`, `--residual-top-k`, `--residual-top-p` - the same three
  controls for the MTP's residual codebooks (1..15), the reference's separate
  `subtalker_*` knobs. Like the codebook-0 flags they have no hardcoded
  default and resolve from the checkpoint (`subtalker_dosample=true,
  subtalker_temperature=0.9, subtalker_top_k=50, subtalker_top_p=1.0` on the
  12 Hz Base checkpoint), so **the residual codebooks are sampled by
  default**, matching the reference. `--residual-temp 0` pins them back to a
  greedy argmax, which is quieter and flatter: these 15 codebooks carry most
  of the acoustic detail.
- `--ref-codes` (`clone`) - an input: an external `[T,16]` codec-codes file
  (8-byte little-endian count header + u32 data) for the in-context (ICL)
  cloning path, used instead of encoding the reference in-tree when you
  already have codes for the reference clip.
- `--profile` - print the resolved sampling plan and each stage's wall time on
  stderr.
- On the NPU: the placement flags of the `npu-*` actions above.

Output is always mono 24 kHz f32 PCM WAV.

### Where the sampling defaults come from

`--temp`, `--top-k`, `--top-p`, `--repetition-penalty` and the three
`--residual-*` flags have no hardcoded default. A flag you do not pass is
resolved, once per generation call, as:

1. the value you passed (a CLI flag, a D-Bus param, a `GenOpts`
   field you set) - always wins, including a deliberate `--repetition-penalty
   1.0`;
2. the checkpoint's own `generation_config.json`, read from the checkpoint (`--ckpt`). The
   12 Hz 0.6B Base checkpoint ships `do_sample=true, temperature=0.9,
   top_k=50, top_p=1.0, repetition_penalty=1.05`;
3. the reference implementation's hard defaults, which for this model are the
   same numbers. Used when the checkpoint has no `generation_config.json`, or
   the file is unreadable or malformed - a bad config never fails a synth.

This exists because a value transcribed into Rust is a value that can drift
from the checkpoint that owns it, which is exactly how `repetition_penalty`
shipped at `1.0` (disabled) while the checkpoint said `1.05`, and how a
default `synth` could decode to silence.

Pass `--profile` to print the resolved plan for a run:

```
qwen3tts: resolved plan: sample=true temp=0.9 top_k=50 top_p=1 rep_penalty=1.05
  (source: generation_config.json) | length: max_frames=256 applied,
  max_new_tokens=8192 reported-only | subtalker (resolved, not yet wired): ...
```

`max_new_tokens` from the checkpoint is reported but **not applied**:
`--max-frames` stays the cap, because in brain it also sizes the Talker KV
cache and the compiled NPU graph, so adopting the reference's 8192-frame
ceiling would grow every run's allocation ~32x for a limit a healthy clip
never reaches.

The same run also prints a one-line warning if codebook-0 ever repeats a
single token for more than 20 consecutive frames at a post-filter probability
above 0.99 - the signature of the silent-collapse failure. It is a
diagnostic: nothing is reseeded or retuned, so the clip you get is still
exactly the clip the resolved plan produced.

## Hardware and limits

- CPU and NPU (OpenVINO) are the supported inference paths for real-time use;
  GPU (Vulkan) forward passes exist for correctness checks but are not the
  path used for practical synthesis speed. The NPU path uses a resident,
  KV-cached decode graph and a streaming codec, which is substantially faster
  than a cold, cache-free run - use the `npu-*` actions or `npu-serve` of
  `sample-models-qwen3tts` (`--device npu` places the served actions
  themselves on the NPU only where the scheduler has an NPU budget).
- The in-context (ICL) cloning path takes externally-supplied reference codes
  (`--ref-codes`) for some flows - brain's own codec encoder also produces
  them in-tree when you pass `--ref-text` without `--ref-codes`.
- No HTTP endpoint for the speech actions' raw PCM stream beyond the generic
  capability surface: TTS is reachable from the CLI and D-Bus/`brain do`, not
  from the OpenAI/Anthropic-compatible chat APIs.
- LoRA fine-tuning covers the Talker only, for single-speaker adaptation; it
  does not retrain the codec or the speaker encoder.
