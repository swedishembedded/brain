# Speech-to-Text (Nemotron ASR + Qwen3-ASR)

Two speech-to-text models behind the same audio-in/text-out contract.
**Nemotron 3.5 ASR Streaming** transcribes audio as it arrives — reach for it
for live/interactive transcription (a mic feed, a call, anything where you
want text as the speaker talks). **Qwen3-ASR** transcribes a complete audio
clip in one pass — reach for it for offline batch transcription where you
have the whole file up front and want the more accurate offline pass.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [ ] |
| CLI (`brain do`)       | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched/streaming serving | [x] (streaming + real batching: Nemotron only; Qwen3-ASR is offline, single-request) |

## Getting the weights

- **Id:** `brain/nemotron` (streaming), `brain/qwen-asr` (offline). Reserved
  vendor `brain/` — never auto-fetched.
- **Weights:**
  - `BRAIN_NEMOTRON` — Nemotron 3.5 ASR checkpoint directory (HF layout).
  - `BRAIN_QWEN_ASR` — Qwen3-ASR checkpoint directory (HF layout).

## Running it

Pass a WAV file directly - `--in audio=` decodes a RIFF/WAVE file, downmixes it
to mono and resamples it to 16 kHz for you:

```bash
BRAIN_NEMOTRON=/path/to/nemotron/hf \
  brain do brain/nemotron transcribe --in audio=clip.wav --out text=out.txt

BRAIN_QWEN_ASR=/path/to/qwen3-asr/hf \
  brain do brain/qwen-asr transcribe --in audio=clip.wav --out text=out.txt
```

The blob wire format itself is **raw mono f32 little-endian PCM at 16 kHz**
(what the D-Bus fd transport and the Python examples send). A headerless file
already in that format is still accepted as-is - `--in audio=clip.pcm` passes
its bytes through untouched; only a RIFF/WAVE header triggers decoding.

Resident server (D-Bus), both models:

```bash
BRAIN_NEMOTRON=/path/to/nemotron/hf BRAIN_QWEN_ASR=/path/to/qwen3-asr/hf \
  brain serve --dbus
```

Nemotron additionally exposes `transcribe_stream` for a live session: send
successive windows of a mic feed under the same `stream` session id, and a
final `eos`-only call to flush and close it. See
[`examples/asr/README.md`](../../examples/asr/README.md) and the reference
client [`examples/asr/transcribe_mic.py`](../../examples/asr/transcribe_mic.py)
(`--model brain/nemotron|brain/qwen-asr`, `--wav FILE` or live mic capture)
for the full protocol.

```bash
BRAIN_NEMOTRON=/path/to/nemotron/hf dbus-run-session -- bash -c '
  brain serve --dbus & sleep 2
  python3 examples/asr/transcribe_mic.py --model brain/nemotron --wav clip.wav'
```

## Options

- `prompt_id` — language-prompt id (default `0` = English).
- `sample_rate` — input PCM sample rate; must be `16000`.
- `BRAIN_QWEN_ASR_WINDOW` — Qwen3-ASR's audio window in seconds (default
  `30`); audio longer than the window is truncated to its first window.
- `BRAIN_QWEN_ASR_MAXNEW` — Qwen3-ASR max generated tokens (default `200`).
- `stream`, `eos` — Nemotron-only `transcribe_stream` params: session id and
  a flag to flush/close it.

## Hardware and limits

- No LoRA/fine-tuning or from-scratch training command for either model
  today.
<!-- perf-number: fixed capability limit (config default), not a measured result -->
- Qwen3-ASR transcribes a fixed window per call (default 30s, `BRAIN_QWEN_ASR_WINDOW`);
  longer clips are truncated rather than chunked - use Nemotron's streaming
  path for clips longer than the window.
- Qwen3-ASR processes one request at a time; Nemotron batches concurrent
  streaming windows that share a language prompt into a single forward pass.
- No HTTP endpoint for either model — CLI (`brain do`) and D-Bus only.
