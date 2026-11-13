# sample: tts/voice-server

Two clients for `sample-models-qwen3tts npu-serve`, the dedicated low-latency TTS server:
a Unix-socket, line-delimited-JSON protocol (not D-Bus) that keeps compiled
NPU graphs resident and streams synthesized 24 kHz f32 PCM back as it's
generated, playing it live via `sounddevice` (or writing a WAV with `--out`).
`brain_tts_client.py` is the shared socket client both scripts import - not an
independent entry point.

```bash
L=target/release/sample-models-qwen3tts
$L npu-serve --clone-ref voice.wav --clone-ref-text voice.txt \
  --design-weights out/tts-vd --design-ckpt /path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign &

python3 samples/python/tts/voice-server/voice-clone.py "hi, this is my voice clone"
python3 samples/python/tts/voice-server/voice-design.py \
  --instruct "a deep cinematic narrator" --text "in a world..."
```

## `voice-clone.py` - speak in the cloned reference voice

```bash
python3 samples/python/tts/voice-server/voice-clone.py "..." --out clone.wav
```

Uses the `clone` engine: the reference voice the server was started with
(`--clone-ref WAV`, and `--clone-ref-text FILE` for its transcript). No
`--speaker`/`--instruct` - one script, one voice, the one the server was
started with.

## `voice-design.py` - describe a voice, or pick a preset

```bash
python3 samples/python/tts/voice-server/voice-design.py \
  --instruct "a deep cinematic narrator" --text "in a world..."
python3 samples/python/tts/voice-server/voice-design.py \
  --engine customvoice --speaker serena --text "hi there"
```

`--engine design` (default) synthesizes a voice from a natural-language
description (`--instruct`); `--engine customvoice` picks a preset speaker
(`--speaker serena|ryan|vivian|eric|...`) instead.

## What it needs

`npu-serve` running and reachable at `--socket` (default: the OS
temp dir's `brain-tts.sock`, override with `BRAIN_TTS_SOCK` or `--socket` on
either script - `npu-serve --help` lists the server's own flags, including
`--socket`). `pip install sounddevice soundfile numpy` for live playback; `--out
FILE.wav` works without `sounddevice` (no speakers needed).

## Options

| flag | scripts | default |
|---|---|---|
| `--socket` | both | OS temp dir's `brain-tts.sock` (`$BRAIN_TTS_SOCK`) |
| `--out FILE.wav` | both | none - play to speakers |
| `--lang` | both | `english` |
| `--temp`, `--top-k`, `--seed`, `--max-frames` | both | `0.9`, `50`, random (drawn and printed), `256` |
| `--instruct TEXT` | `voice-design.py` | none |
| `--engine design\|customvoice` | `voice-design.py` | `design` |
| `--speaker NAME` | `voice-design.py` | none (required for `customvoice`) |
