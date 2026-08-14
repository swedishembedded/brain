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
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched/streaming serving | [ ] (offline, single-request) |

## Getting the weights

Model id: `brain/qwen-asr` (reserved vendor - never auto-fetched).
`BRAIN_QWEN3ASR` - checkpoint directory (HF layout).

## Running it

Pass a WAV file directly - `--in audio=` decodes a RIFF/WAVE file, downmixes
it to mono and resamples it to 16 kHz for you:

```bash
BRAIN_QWEN3ASR=/path/to/qwen3-asr/hf \
  brain qwen3asr transcribe --in audio=clip.wav --out text=out.txt
```

The blob wire format itself is **raw mono f32 little-endian PCM at 16 kHz**.
A headerless file already in that format is still accepted as-is -
`--in audio=clip.pcm` passes its bytes through untouched; only a RIFF/WAVE
header triggers decoding.

Resident server (D-Bus):

```bash
BRAIN_QWEN3ASR=/path/to/qwen3-asr/hf brain serve --dbus
```

Reference client:
[`examples/asr/transcribe_mic.py`](../../examples/asr/transcribe_mic.py)
(`--model brain/qwen-asr`, `--wav FILE`) - see
[`examples/asr/README.md`](../../examples/asr/README.md) for the full
protocol.

## Options

- `prompt_id` - language-prompt id (default `0` = English).
- `sample_rate` - input PCM sample rate; must be `16000`.
- `BRAIN_QWEN3ASR_WINDOW` - audio window in seconds (default `30`); audio
  longer than the window is truncated to its first window, not chunked -
  use [Nemotron](nemotronasr.md)'s streaming path for clips longer than
  the window.
- `BRAIN_QWEN3ASR_MAXNEW` - max generated tokens (default `200`).

## Hardware and limits

No LoRA/fine-tuning or from-scratch training command today. Processes one
request at a time. No HTTP endpoint - CLI and D-Bus only.
