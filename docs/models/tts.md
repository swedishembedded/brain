# Text-to-Speech (Qwen3-TTS)

Turns text into a 24 kHz speech waveform. Give it a few seconds of reference
audio and it clones that voice for the sentence you ask it to speak; without
a reference it falls back to a default, speaker-free voice. Reach for it when
you need narration, a cloned voice for a demo, or a spoken-output leg for an
assistant pipeline — all running locally, no external TTS service.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [x] |
| CLI (`brain do`)       | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched/streaming serving | [x] |

## Getting the weights

Model id: `brain/tts`. Reserved vendor `brain/` — never auto-fetched; you
need a local Qwen3-TTS checkpoint (Base, CustomVoice, or VoiceDesign).

1. Convert the upstream HF checkpoint into brain's own checkpoint format:

   ```bash
   brain tts import --ckpt /path/to/Qwen3-TTS-12Hz-0.6B-Base \
     [--codec-ckpt <dir>] [--speaker-ckpt <dir>] \
     --out-dir out/tts
   ```

   This writes four files into `--out-dir`: `talker.safetensors`,
   `mtp.safetensors`, `codec.safetensors`, `speaker.safetensors`. (Voice
   cloning needs `speaker.safetensors`; CustomVoice/VoiceDesign checkpoints
   don't ship one, so `import` skips it with a warning for those.)

2. Point the CLI or the server at the imported directory and the original HF
   checkpoint (for its tokenizer/config):

   - `--weights-dir` (CLI flag, defaults to `out/tts`) or `BRAIN_TTS_WEIGHTS`
     (D-Bus serving) — the directory from step 1.
   - `--ckpt` (CLI flag) or `BRAIN_TTS_CKPT` — the original HF checkpoint
     directory.

## Running it

Speaker-free synthesis:

```bash
brain tts synth --text "Hello from brain." --out out.wav \
  --weights-dir out/tts --ckpt /path/to/Qwen3-TTS-12Hz-0.6B-Base
```

Voice cloning from a reference clip:

```bash
brain tts clone \
  --text "This sentence is spoken in the reference speaker's voice." \
  --ref voice.wav --ref-text "transcript of voice.wav" \
  --out demo.wav \
  --weights-dir out/tts --ckpt /path/to/Qwen3-TTS-12Hz-0.6B-Base
```

Instructed voice design / preset speakers (CustomVoice/VoiceDesign
checkpoints only):

```bash
brain tts design --text "..." --instruct "a calm, low voice" \
  [--speaker NAME] --out out.wav
```

Generic capability CLI (same synthesis, uniform invocation):

```bash
brain do brain/tts synth --text "Hello from brain." \
  --weights_dir out/tts --ckpt /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --out audio=out.wav
```

Resident server for repeated calls without reloading weights each time:

```bash
BRAIN_TTS_WEIGHTS=out/tts BRAIN_TTS_CKPT=/path/to/Qwen3-TTS-12Hz-0.6B-Base \
  brain serve --dbus
```

The resident exposes one action, `speak` (text in, 24 kHz PCM out); set
`BRAIN_TTS_REF` (+ `BRAIN_TTS_REF_TEXT`) to have every `speak` call clone that
reference voice instead of running speaker-free.

There is also a dedicated low-latency server, `brain tts serve`, which keeps
compiled NPU graphs resident and streams synthesized audio back over a
line-delimited JSON protocol on a Unix socket — see `brain tts serve --help`
for its engine/socket flags. `scripts/tts/voice-clone.py` and
`scripts/tts/voice-design.py` are example clients that speak to it and play
the result.

LoRA fine-tuning (single-speaker) of the Talker on a `text -> codes` dataset:

```bash
brain tts finetune --base out/tts/talker.safetensors --data data/tts \
  --out out/tts/talker_lora.safetensors \
  [--steps N --lr X --rank R --alpha A --batch B --block T --seed S]
```

This freezes the base Talker and trains attention adapters only.

## Options

- `--lang` / `--language` — synthesis language (default `english`).
- `--max-frames` — max codec frames, i.e. an upper bound on clip length
  (default `256`).
- `--temp`, `--top-k`, `--seed` — sampling controls for the Talker.
- `--ref-codes` — an external `[T,16]` codec-codes file (8-byte little-endian
  count header + u32 data) for the in-context (ICL) cloning path, used
  instead of the default x-vector-only cloning when you already have codes
  for the reference clip.
- `--device npu` — run on an NPU (OpenVINO) if present; `BRAIN_TTS_TALKER=cpu`
  falls the Talker back to CPU while keeping the codec on NPU.
- `BRAIN_TTS_LANG`, `BRAIN_TTS_REF`, `BRAIN_TTS_REF_TEXT` — resident-server
  defaults for language and voice-clone reference.

Output is always mono 24 kHz f32 PCM WAV.

## Hardware and limits

- CPU and NPU (OpenVINO) are the supported inference paths for real-time use;
  GPU (Vulkan) forward passes exist for correctness checks but are not the
  path used for practical synthesis speed. The NPU path uses a resident,
  KV-cached decode graph and a streaming codec, which is substantially faster
  than a cold, cache-free run — pass `--device npu` to use it.
- The in-context (ICL) cloning path needs externally-supplied reference codes
  (`--ref-codes`) for some flows — brain's own codec encoder can also produce
  them in-tree when you pass `--ref-text` without `--ref-codes`.
- No HTTP endpoint: TTS is reachable from the CLI and D-Bus/`brain do`, not
  from the OpenAI/Anthropic-compatible chat APIs.
- LoRA fine-tuning covers the Talker only, for single-speaker adaptation; it
  does not retrain the codec or the speaker encoder.
