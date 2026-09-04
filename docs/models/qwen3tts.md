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
CLI use - no env var, no manual `import` step. That default run:
downloads the checkpoint (including its nested `speech_tokenizer/` codec),
converts it exactly the way `brain qwen3tts import` does by hand (below),
and points `--weights-dir`/`--ckpt` at the result.

For a CustomVoice or VoiceDesign checkpoint, or one you already have
locally, convert it yourself:

1. Convert the upstream HF checkpoint into brain's own checkpoint format:

   ```bash
   brain qwen3tts import --ckpt /path/to/Qwen3-TTS-12Hz-0.6B-Base \
     [--codec-ckpt <dir>] [--speaker-ckpt <dir>] \
     --out-dir out/tts
   ```

   This writes four files into `--out-dir`: `talker.safetensors`,
   `mtp.safetensors`, `codec.safetensors`, `speaker.safetensors`. (Voice
   cloning needs `speaker.safetensors`; CustomVoice/VoiceDesign checkpoints
   don't ship one, so `import` skips it with a warning for those.)

2. Point the CLI or the server at the imported directory and the original HF
   checkpoint (for its tokenizer/config):

   - `--weights-dir` (CLI flag, defaults to `$BRAIN_QWEN3TTS_WEIGHTS` else
     `out/tts`) or `BRAIN_QWEN3TTS_WEIGHTS` (D-Bus serving) - the directory
     from step 1.
   - `--ckpt` (CLI flag) or `BRAIN_QWEN3TTS_CKPT` - the original HF checkpoint
     directory.

## Running it

Speaker-free synthesis:

```bash
brain qwen3tts synth --text "Hello from brain." --out out.wav \
  --weights-dir out/tts --ckpt /path/to/Qwen3-TTS-12Hz-0.6B-Base
```

Voice cloning from a reference clip:

```bash
brain qwen3tts clone \
  --text "This sentence is spoken in the reference speaker's voice." \
  --ref voice.wav --ref-text "transcript of voice.wav" \
  --out demo.wav \
  --weights-dir out/tts --ckpt /path/to/Qwen3-TTS-12Hz-0.6B-Base
```

Instructed voice design / preset speakers (CustomVoice/VoiceDesign
checkpoints only):

```bash
brain qwen3tts design --text "..." --instruct "a calm, low voice" \
  [--speaker NAME] --out out.wav
```

The same `synth` action is also in the capability manifest (`brain caps
brain/qwen3tts` lists it, and it's reachable over D-Bus/HTTP) - the `brain
qwen3tts synth` form above is what reaches it from the CLI specifically,
same as every other architecture with its own dedicated CLI module.

Resident server for repeated calls without reloading weights each time:

```bash
BRAIN_QWEN3TTS_WEIGHTS=out/tts BRAIN_QWEN3TTS_CKPT=/path/to/Qwen3-TTS-12Hz-0.6B-Base \
  brain serve --dbus
```

The resident exposes one action, `speak` (text in, 24 kHz PCM out); set
`BRAIN_QWEN3TTS_REF` (+ `BRAIN_QWEN3TTS_REF_TEXT`) to have every `speak` call clone that
reference voice instead of running speaker-free.

The event-driven stdio controller (`brain serve --stdio`, the JSONL protocol)
can also synthesize, answering a `user_synth_request` with a stream of
`audio_chunk` events. That path is **off in the default binary**: the TTS stack
is far heavier than the rest of the controller's dependencies, so it lives
behind a build feature.

```bash
cargo build --release -p brain-cli --features qwen3tts-synth
BRAIN_QWEN3TTS_WEIGHTS=out/tts BRAIN_QWEN3TTS_CKPT=[path/to/Qwen3-TTS-12Hz-0.6B-Base] \
  brain serve --stdio
```

It reads the same `BRAIN_QWEN3TTS_*` variables as the resident above (including
`BRAIN_QWEN3TTS_REF`/`_REF_TEXT` for cloning). Without the feature, or with
`BRAIN_QWEN3TTS_WEIGHTS` unset, a `user_synth_request` is still answered - with
an empty terminal `audio_chunk`.

There is also a dedicated low-latency server, `brain qwen3tts serve`, which keeps
compiled NPU graphs resident and streams synthesized audio back over a
line-delimited JSON protocol on a Unix socket - see `brain qwen3tts serve --help`
for its engine/socket flags. `scripts/tts/voice-clone.py` and
`scripts/tts/voice-design.py` are example clients that speak to it and play
the result.

LoRA fine-tuning (single-speaker) of the Talker on a `text -> codes` dataset:

```bash
brain qwen3tts finetune --base out/tts/talker.safetensors --data data/tts \
  --out out/tts/talker_lora.safetensors \
  [--steps N --lr X --rank R --alpha A --batch B --block T --seed S]
```

This freezes the base Talker and trains attention adapters only.

## Options

- `--lang` / `--language` - synthesis language (default `english`).
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
- `--ref-codes` - an external `[T,16]` codec-codes file (8-byte little-endian
  count header + u32 data) for the in-context (ICL) cloning path, used
  instead of the default x-vector-only cloning when you already have codes
  for the reference clip.
- `--device npu` - run on an NPU (OpenVINO) if present; `BRAIN_QWEN3TTS_TALKER=cpu`
  falls the Talker back to CPU while keeping the codec on NPU.
- `BRAIN_QWEN3TTS_LANG`, `BRAIN_QWEN3TTS_REF`, `BRAIN_QWEN3TTS_REF_TEXT` - resident-server
  defaults for language and voice-clone reference.

Output is always mono 24 kHz f32 PCM WAV.

### Where the sampling defaults come from

`--temp`, `--top-k`, `--top-p`, `--repetition-penalty` and the three
`--residual-*` flags have no hardcoded default. A flag you do not pass is
resolved, once per generation call, as:

1. the value you passed (a CLI flag, a `brain do` / D-Bus param, a `GenOpts`
   field you set) - always wins, including a deliberate `--repetition-penalty
   1.0`;
2. the checkpoint's own `generation_config.json`, read from `--ckpt`. The
   12 Hz 0.6B Base checkpoint ships `do_sample=true, temperature=0.9,
   top_k=50, top_p=1.0, repetition_penalty=1.05`;
3. the reference implementation's hard defaults, which for this model are the
   same numbers. Used when the checkpoint has no `generation_config.json`, or
   the file is unreadable or malformed - a bad config never fails a synth.

This exists because a value transcribed into Rust is a value that can drift
from the checkpoint that owns it, which is exactly how `repetition_penalty`
shipped at `1.0` (disabled) while the checkpoint said `1.05`, and how a
default `synth` could decode to silence.

Set `TTS_PLAN=1` (or `TTS_PROFILE=1`) to print the resolved plan for a run:

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
  than a cold, cache-free run - pass `--device npu` to use it.
- The in-context (ICL) cloning path needs externally-supplied reference codes
  (`--ref-codes`) for some flows - brain's own codec encoder can also produce
  them in-tree when you pass `--ref-text` without `--ref-codes`.
- No HTTP endpoint: TTS is reachable from the CLI and D-Bus/`brain do`, not
  from the OpenAI/Anthropic-compatible chat APIs.
- LoRA fine-tuning covers the Talker only, for single-speaker adaptation; it
  does not retrain the codec or the speaker encoder.
