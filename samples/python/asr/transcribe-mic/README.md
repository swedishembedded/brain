# sample: asr/transcribe-mic

Live microphone -> brain (over D-Bus) -> streaming transcription.
`transcribe_mic.py` opens the microphone (or a `--wav` file), streams raw
16 kHz mono f32 PCM to a running `brain serve --dbus` over the
`StreamTranscribe` D-Bus method through one pipe fd, and prints the
transcription **segments** as each ~1 s window decodes.

```bash
BRAIN_NEMOTRONASR=$BRAIN_TESTDATA/asr/nemotron/hf \
  tools/dbus-session.sh --serve "--dbus --device cpu" -- \
  python3 samples/python/asr/transcribe-mic/transcribe_mic.py --model brain/nemotronasr --seconds 15
```

Transcribe a WAV file instead of the mic (a good smoke test - no audio
hardware needed):

```bash
python3 samples/python/asr/transcribe-mic/transcribe_mic.py \
  --wav $BRAIN_TESTDATA/asr/audio/librispeech_mr_quilter.wav
```

Add `--fast` with `--wav` to feed as fast as possible (throughput), or drop
it to pace at real time (latency, as a live mic would).

### No microphone? Make a test clip with brain's own TTS

```bash
brain qwen3tts synth --text "the quick brown fox jumps over the lazy dog" --out /tmp/tts.wav
sox /tmp/tts.wav -r 16000 -c 1 /tmp/tts16k.wav          # 24 kHz -> 16 kHz mono
python3 samples/python/asr/transcribe-mic/transcribe_mic.py --wav /tmp/tts16k.wav
```

## What it demonstrates

* Two served models, one wire: `nemotron` (FastConformer + RNN-T) streams as
  a **frame-synchronous session** with ~0.32 s algorithmic latency; `qwen-asr`
  (Whisper-style encoder + Qwen3 decoder) falls back to independent per-window
  jobs.
* The exact same `StreamTranscribe` fd-pipe protocol
  `samples/python/asr/bench-streams/` load-tests at concurrency.
* `--window-ms` controls near-real-time latency: roughly one window plus the
  model's compute (RTF << 1 on the FastConformer encoder).

## What it needs

- `BRAIN_NEMOTRONASR` and/or `BRAIN_QWENASR` pointed at checkpoint weights.
- `jeepney` (D-Bus with fd passing, always required) - `pip install -e brain-py`.
- `sounddevice` + `numpy` only for live mic capture (`pip install sounddevice
  numpy`); not needed for `--wav`.

## Options

| flag | default |
|---|---|
| `--model ID` | `brain/nemotronasr` (or `brain/qwen3asr`) |
| `--window-ms N` | `1000` - server-side transcription window |
| `--seconds N` | `0` - mic capture duration (0 = until Ctrl-C) |
| `--wav PATH` | unset - stream this file instead of the mic |
| `--fast` | off - with `--wav`, feed as fast as possible instead of real time |
