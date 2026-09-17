# sample: asr/bench-streams

End-to-end ASR benchmark: N concurrent streams through brain over D-Bus.
`bench_streams.py` drives `--streams` concurrent `StreamTranscribe` sessions
of the same wav clip against a running `brain serve --dbus`, feeding each as
fast as possible, and reports per-model throughput, real-time factor (RTF),
first-/final-segment latency, and the scheduler's batch counters - proof
that concurrent windows actually batched.

```bash
BRAIN_NEMOTRONASR=$BRAIN_TESTDATA/asr/nemotron/hf \
  tools/dbus-session.sh --serve "--dbus --device cpu" -- \
  python3 samples/python/asr/bench-streams/bench_streams.py --model brain/nemotronasr \
    --wav $BRAIN_TESTDATA/asr/audio/librispeech_mr_quilter.wav \
    --streams 1,2,4
```

Reports one row per concurrency level so you can see batching scale
throughput.

## What it demonstrates

* `nemotron` (NVIDIA Nemotron 3.5 ASR Streaming 0.6B): a **frame-synchronous
  session** - stateful across windows, concurrent sessions batch through one
  encoder pass on the shared residency Executor.
* `qwen-asr` (Qwen3-ASR 1.7B): offline, each window falls back to an
  independent `transcribe` job - no session state across windows.
* Scale from 1 to N concurrent streams and watch RTF and the scheduler's
  batch counters move together - the same protocol path
  `samples/python/asr/transcribe-mic/` uses, but adversarial on concurrency
  rather than live audio.

## What it needs

- `BRAIN_NEMOTRONASR` and/or `BRAIN_QWENASR` pointed at checkpoint weights
  before `brain serve --dbus`.
- A 16 kHz mono 16-bit PCM `--wav` clip to replay.
- `pip install -e brain-py` (jeepney with fd passing) - the only dependency;
  no `sounddevice` needed here.

## Options

| flag | default |
|---|---|
| `--model ID` | `brain/nemotronasr` |
| `--wav PATH` | *required* |
| `--streams LIST` | `1,2,4` (comma-separated concurrency levels) |
| `--window-ms N` | `1000` |
