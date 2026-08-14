# Nemotron 3.5 ASR Streaming (speech-to-text)

Transcribes audio as it arrives - reach for it for live/interactive
transcription (a mic feed, a call, anything where you want text as the
speaker talks). FastConformer encoder (depthwise-sep causal subsampling,
macaron FFs, Transformer-XL rel-pos attention, GLU conv module) + RNN-T
transducer. For offline batch transcription of a complete clip instead, see
[Qwen3-ASR](qwen3asr.md); both are compared on the
[speech-to-text overview](asr.md).

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| LoRA fine-tune         | [ ] |
| CLI                    | [x] |
| HTTP API               | [ ] |
| D-Bus                  | [x] |
| Batched/streaming serving | [x] (real streaming + batching: concurrent windows sharing a language prompt fold into one forward pass) |

## Getting the weights

Model id: `brain/nemotronasr` (reserved vendor - never auto-fetched).
`BRAIN_NEMOTRONASR` - checkpoint directory (HF layout).

## Running it

Pass a WAV file directly - `--in audio=` decodes a RIFF/WAVE file, downmixes
it to mono and resamples it to 16 kHz for you:

```bash
BRAIN_NEMOTRONASR=/path/to/nemotron/hf \
  brain nemotronasr transcribe --in audio=clip.wav --out text=out.txt
```

The blob wire format itself is **raw mono f32 little-endian PCM at 16 kHz**
(what the D-Bus fd transport and the Python examples send). A headerless
file already in that format is still accepted as-is - `--in audio=clip.pcm`
passes its bytes through untouched; only a RIFF/WAVE header triggers
decoding.

Resident server (D-Bus):

```bash
BRAIN_NEMOTRONASR=/path/to/nemotron/hf brain serve --dbus
```

Additionally exposes `transcribe_stream` for a live session: send successive
windows of a mic feed under the same `stream` session id, and a final
`eos`-only call to flush and close it. See
[`examples/asr/README.md`](../../examples/asr/README.md) and the reference
client [`examples/asr/transcribe_mic.py`](../../examples/asr/transcribe_mic.py)
(`--model brain/nemotronasr`, `--wav FILE` or live mic capture) for the full
protocol.

```bash
BRAIN_NEMOTRONASR=/path/to/nemotron/hf dbus-run-session -- bash -c '
  brain serve --dbus & sleep 2
  python3 examples/asr/transcribe_mic.py --model brain/nemotronasr --wav clip.wav'
```

## Options

- `prompt_id` - language-prompt id (default `0` = English).
- `sample_rate` - input PCM sample rate; must be `16000`.
- `stream`, `eos` - `transcribe_stream` params: session id and a flag to
  flush/close it.

## Hardware and limits

No LoRA/fine-tuning or from-scratch training command today. No HTTP
endpoint - CLI and D-Bus only.
