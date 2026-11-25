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
| CLI (`sample-models-qwen3tts <action>`) | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched/streaming serving | [x] |

## Getting the weights

Model id: `brain/qwen3tts` for `brain do` and `brain caps`. `brain serve`
serves every Qwen3-TTS checkpoint in the model store as a model of its own,
under its `<vendor>/<repo>` id (`Qwen/Qwen3-TTS-12Hz-0.6B-Base`,
`local/<name>` for a trained one), each with its own languages and voices;
checkpoint roles named outright are served as `brain/qwen3tts`.
`Qwen/Qwen3-TTS-12Hz-0.6B-Base` (Base only, the
variant with a speaker encoder for voice cloning) auto-fetches (⤓, opt-in `--autofetch`) on first
CLI use - no env var, no manual `import` step. The checkpoint (including its
nested `speech_tokenizer/` codec) is served exactly as downloaded: the
Talker, MTP, codec and speaker encoder are read from its safetensors, bf16
converted to f32 in memory as they load, and nothing is written beside it.
`brain pull` recognises any Qwen3-TTS repo by its files (the
`speech_tokenizer/` codec beside the tokenizer), whatever the repo is called:
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` and a trained release pull as `qwen3tts` too.

A CustomVoice or VoiceDesign checkpoint, or one you already have locally, is
resolved from the model store the same way; the standard role flags name one
outright: `--weights-dir` (the checkpoint, or a directory of brain-format
files) and `--ckpt` (the Hugging Face checkpoint directory, for `config.json`
and the tokenizer). The checkpoint is the provider's, never a per-call param.
Voice cloning needs the speaker encoder, which only the Base checkpoints
ship.

`--weights-dir` may instead name a directory of brain-format files, which the
`import` action writes from a checkpoint. The actions run from the
`sample-models-qwen3tts` program (`samples/models/qwen3tts`); build it once:

```bash
make samples/models/qwen3tts/build
Q=target/release/sample-models-qwen3tts
$Q import --source /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  [--codec-source <dir>] [--speaker-source <dir>] \
  --dest out/tts
```

That writes `talker.safetensors`, `mtp.safetensors`, `codec.safetensors` and
(for a Base checkpoint) `speaker.safetensors`, the same tensors the
checkpoint is read as; a release with no speaker encoder (CustomVoice,
VoiceDesign) skips it and says so. A directory holding `talker.safetensors`
is read as these files; any other is read as the checkpoint itself.

## Running it

`$Q <action> ...` (the `sample-models-qwen3tts` program built above) runs a
served action on the host, and adds the NPU route; the `brain` binary itself
has no `qwen3tts` verb and names this program when asked for one. Every flag below is the action's served param; the reference
recording and the output are blobs: `--ref voice.wav` names an input,
`--out audio=out.wav` an output.

Speaker-free synthesis:

```bash
$Q synth --text "Hello from brain." --out audio=out.wav --seed $RANDOM
```

Voice cloning from a reference clip (with `--ref-text`, the in-context path;
without, x-vector-only):

```bash
$Q clone \
  --text "This sentence is spoken in the reference speaker's voice." \
  --ref voice.wav --ref-text "transcript of voice.wav" \
  --out audio=demo.wav --seed $RANDOM
```

Instructed voice design / preset speakers. Each released checkpoint is
trained for one way of choosing the voice, read from its `config.json`
(`tts_model_type`): `clone` runs only on a Base checkpoint, `design` with an
instruction alone only on VoiceDesign, `design` with `--speaker` only on
CustomVoice, and an instruction beside a speaker only on the 1.7B CustomVoice
(the 0.6B one was not trained to follow one). Any other combination is
refused naming the checkpoint kind that would serve it, never spoken as if it
had been followed:

```bash
$Q design --text "..." --instruct "a calm, low voice" \
  [--speaker NAME] --out audio=out.wav
```

`batch` decodes several texts interleaved in one call (`--requests` takes a
JSON array; request `i` is the output `audio_<i>`), and `sim --a A.wav --b
B.wav` prints the speaker-embedding cosine similarity of two utterances (the
timbre-preservation metric, e.g. a quantized Talker's clone against its
reference). The actions are in the capability manifest too (`brain caps
brain/qwen3tts`), reachable over D-Bus/HTTP.

`encode` turns a recording into what a training shard records for it: input
blob `audio` (a WAV at any sample rate, at most 128 MiB and 600 s; resampled to
the codec's 24 kHz), optional param `text` (its transcript). Outputs: blob
`codes` (u16 little-endian, `[frames, 16]` row-major, the codec's codes), blob
`xvector` (f32 little-endian, the speaker encoder's x-vector), with `text` blob
`text_ids` (u32 little-endian: the checkpoint tokenizer's ids of
`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`, exactly what
generation tokenises the same text into), and the values `frames`,
`codebooks`, `frame_rate`, `xvector_dim`, `sample_rate_in` (the given WAV's
rate) and `seconds` (its duration). It needs a Base checkpoint (the speaker
encoder). Served, it keeps its encoders loaded and spreads concurrent requests
over several of them. From Python, `encode_audio(wav_bytes, text)` on any
`brain_py` client returns the decoded arrays.

Resident serving for repeated calls without reloading weights each time:

```bash
brain serve --dbus
```

Over HTTP, `POST /v1/audio/speech` reaches the same actions: a request with
a reference recording in `input_references` runs `clone` (in context when it
carries the transcript too), one with `instructions` or a preset `voice` runs
`design`, and a plain one `synth`; `seed` picks the take and comes back in
`X-Seed` (see the HTTP API reference).

`brain serve` serves the checkpoint the model store resolves. Its `synth`,
`clone` and `design` actions run on a hot engine and are streaming: each
decoded codec chunk (`stream_chunk` codec frames, 16 by default)
arrives as an `audio` chunk of raw mono f32 little-endian PCM at 24 kHz while
the rest is still generating, and the terminal `audio` blob is the whole clip
as a WAV. A reference voice's conditioning is computed once per recording and
kept, so repeat clones of one timbre skip the speaker encoder and the codec
encode. The other actions (`batch`, `import`, `finetune`, `sim`, `encode`) are the
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
$Q finetune --data data/tts --save out/tts/talker_lora.safetensors \
  [--full] [--steps N --lr X --rank R --alpha A --batch B --block T --seed S --activations keep|recompute]
```

This freezes the resolved checkpoint's Talker and trains attention adapters
only; `--full` trains every Talker weight instead (Qwen's own documented
single-speaker fine-tuning workflow). The adapters are attention-only, so the
shared LoRA parameter set's `targets` and `base_format` do not apply and are
not declared. The Talker rebuilds each layer in the backward whatever is
asked, so `--activations recompute` is accepted and `--activations keep` is
refused by name before the first step. Cancelling stops at a step boundary.

## Trained checkpoints

The served training job `speech_sft` of `brain/train` trains one (see
[Training as a served job](../../guides/training/served-training.md#speech-fine-tuning)). A checkpoint trained in brain is exported in the upstream layout of the base it
was trained from (`qwen3tts::export`), so it loads unchanged in brain and in
the upstream `qwen-tts` package:

- `model.safetensors` holds every tensor of the base under its upstream name
  and dtype (BF16 for the released checkpoints, rounded to nearest even); the
  tensors training did not change are the base's bytes.
- `config.json` is the base's, with the trained languages added to
  `talker_config.codec_language_id` and the named voices to `spk_id` (each
  with `spk_is_dialect: false`). Names are lowercase `[a-z0-9_-]`.
- A checkpoint with named voices is a `custom_voice` checkpoint: it speaks as
  one of its voices (`voice` = a voice name) and, like upstream's own
  fine-tunes, carries no speaker encoder, so it does not clone a reference
  recording.
- `generation_config.json`, the tokenizer files and `speech_tokenizer/` are
  the base's, and `brain.manifest.json` registers it as `qwen3tts`.

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

### Text layout

The text reaches the Talker in one of two layouts, as in the reference's
`non_streaming_mode`: `streaming` adds one text token to each generated
frame's input, `non_streaming` puts the whole text before the first frame.
The speech actions take `text_layout` (`streaming` or `non_streaming`);
omitted, a preset speaker or a designed voice (`design`) speaks
non-streaming, as the reference's CustomVoice and VoiceDesign entry points
default to, and `synth` and `clone` speak streaming. A checkpoint speaks best
in the layout it was trained on. An in-context clone (`clone` with
`ref_text`) builds the streaming layout only and refuses `non_streaming`.

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
