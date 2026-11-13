# Qwen3-ASR (speech-to-text)

Transcribes a complete audio clip in one pass - reach for it for offline
batch transcription where you have the whole file up front and want the
more accurate offline pass. Whisper-style audio encoder spliced into a
Qwen3-1.7B decoder. For live/streaming transcription instead, see
[Nemotron 3.5 ASR Streaming](nemotronasr.md); both are compared on the
[speech-to-text overview](asr.md).

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [ ] |
| CLI                    | [x] |
| HTTP API               | [x] (`/v1/audio/transcriptions`) |
| D-Bus                  | [x] |
| Batched/streaming serving | [ ] (offline, single-request) |

## Getting the weights

Model id: `brain/qwen3asr` (the served id itself is never fetched, per this
project's naming grammar) - but `Qwen/Qwen3-ASR-1.7B` auto-fetches (⤓, opt-in `--autofetch`) on
first CLI use, no env var needed. To use a different HF-layout checkpoint
directory, put it in the model store, or name it with `--weights DIR`
(one-shot) or `brain serve --pin qwen3asr.weights=DIR`.

## Running it

Pass a WAV file directly - `--in audio=` decodes a RIFF/WAVE file, downmixes
it to mono and resamples it to 16 kHz for you:

```bash
brain qwen3asr transcribe --in audio=clip.wav --out text=out.txt   # --weights /path/to/qwen3-asr/hf names another
```

The blob wire format itself is **raw mono f32 little-endian PCM at 16 kHz**.
A headerless file already in that format is still accepted as-is -
`--in audio=clip.pcm` passes its bytes through untouched; only a RIFF/WAVE
header triggers decoding.

Resident server (D-Bus):

```bash
brain serve --dbus                        # --pin qwen3asr.weights=/path/to/qwen3-asr/hf names another
```

Reference client:
[`samples/python/asr/transcribe-mic/transcribe_mic.py`](../../../samples/python/asr/transcribe-mic/transcribe_mic.py)
(`--model brain/qwen3asr`, `--wav FILE`) - see
[`samples/python/asr/transcribe-mic/README.md`](../../../samples/python/asr/transcribe-mic/README.md) for the full
protocol.

## Options

- `prompt_id` - language-prompt id (default `0` = English).
- `sample_rate` - input PCM sample rate; must be `16000`.
- `window_secs` - audio window in seconds (default `30`, host-side); audio
  longer than the window is truncated to its first window and reported
  `truncated`, not chunked - use [Nemotron](nemotronasr.md)'s streaming path
  for clips longer than the window. An engine is built per window, and kept.
- `max_new` - max generated tokens (default `200`, host-side).

## Hardware and limits

No LoRA/fine-tuning or from-scratch training command today. Processes one
request at a time. Served over HTTP on `POST /v1/audio/transcriptions`
(see [the HTTP reference](../api/http.md)), the CLI and D-Bus.
