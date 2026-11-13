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
| HTTP API               | [x] (`/v1/audio/transcriptions`) |
| D-Bus                  | [x] |
| Batched/streaming serving | [x] (real streaming + batching: concurrent windows sharing a language prompt fold into one forward pass) |

## Getting the weights

Model id: `brain/nemotronasr` (the served id itself is never fetched, per
this project's naming grammar) - but `nvidia/nemotron-3.5-asr-streaming-0.6b`
auto-fetches (⤓, opt-in `--autofetch`) on first CLI use, no env var needed. To use a different HF-layout checkpoint
directory, put it in the model store, or name it with `--weights DIR`
(one-shot) or `brain serve --pin nemotronasr.weights=DIR`.

## Running it

Pass a WAV file directly - `--in audio=` decodes a RIFF/WAVE file, downmixes
it to mono and resamples it to 16 kHz for you:

```bash
brain nemotronasr transcribe --in audio=clip.wav --out text=out.txt   # --weights /path/to/nemotron/hf names another
```

The blob wire format itself is **raw mono f32 little-endian PCM at 16 kHz**
(what the D-Bus fd transport and the Python examples send). A headerless
file already in that format is still accepted as-is - `--in audio=clip.pcm`
passes its bytes through untouched; only a RIFF/WAVE header triggers
decoding.

Resident server (D-Bus):

```bash
brain serve --dbus                        # --pin nemotronasr.weights=/path/to/nemotron/hf names another
```

Additionally exposes `transcribe_stream` for a live session: send successive
windows of a mic feed under the same `stream` session id, and a final
`eos`-only call to flush and close it. See
[`samples/python/asr/transcribe-mic/README.md`](../../../samples/python/asr/transcribe-mic/README.md) and the reference
client [`samples/python/asr/transcribe-mic/transcribe_mic.py`](../../../samples/python/asr/transcribe-mic/transcribe_mic.py)
(`--model brain/nemotronasr`, `--wav FILE` or live mic capture) for the full
protocol.

```bash
dbus-run-session -- bash -c '
  brain serve --dbus & sleep 2
  python3 samples/python/asr/transcribe-mic/transcribe_mic.py --model brain/nemotronasr --wav clip.wav'
```

## Options

- `prompt_id` - language-prompt id (default `0` = English).
- `sample_rate` - input PCM sample rate; must be `16000`.
- `stream`, `eos` - `transcribe_stream` params: session id and a flag to
  flush/close it.

## Hardware and limits

No LoRA/fine-tuning or from-scratch training command today. Offline
`transcribe` is served over HTTP on `POST /v1/audio/transcriptions` (see
[the HTTP reference](../api/http.md)); the live `transcribe_stream` session is
D-Bus only.
