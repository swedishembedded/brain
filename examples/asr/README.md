# Streaming speech-to-text over D-Bus

Live microphone → `brain serve --dbus` → transcription events, end to end, using the
`StreamTranscribe` method on `com.swedishembedded.Brain1`. The client streams raw
**16 kHz mono f32 little-endian PCM** through one pipe fd; the server reads it and
streams back `segment` frames, then a terminal `done` with the full transcript.

For a model that advertises the `transcribe_stream` capability (nemotron), the
whole stream is **one live session**: every window is a frame-synchronous step of
a stateful encoder/decoder (cached attention left-context, no per-window
re-encode), each `segment` is the *newly emitted* text, and concurrent sessions
batch through one encoder pass on the shared residency **Executor**. For an
offline model (qwen-asr) each window falls back to an independent `transcribe`
job.

Two models are served:

| model | what | streaming |
|---|---|---|
| `nemotron` | NVIDIA Nemotron 3.5 ASR Streaming 0.6B (FastConformer + RNN-T) | **frame-synchronous session** — stateful across windows, ~0.32 s algorithmic latency, batched across concurrent sessions |
| `qwen-asr` | Qwen3-ASR 1.7B (Whisper-style encoder + Qwen3 decoder) | offline, fixed audio window (independent per-window jobs) |

## Run it

Serve on a private session bus (no system config needed) with the checkpoint(s)
pointed at your `testdata` tree, then run the client:

```bash
# Nemotron only (streaming demo), CPU backend:
BRAIN_NEMOTRON=$BRAIN_TESTDATA/asr/nemotron/hf \
  dbus-run-session -- bash -c '
    brain serve --dbus --device cpu & sleep 2
    python3 examples/asr/transcribe_mic.py --model nemotron --seconds 15
  '
```

Transcribe a WAV file instead of the mic (a good smoke test — no audio hardware):

```bash
python3 examples/asr/transcribe_mic.py \
  --wav $BRAIN_TESTDATA/asr/audio/librispeech_mr_quilter.wav
```

Add `--fast` with `--wav` to feed as fast as possible (throughput), or drop it to
pace at real time (latency, as a live mic would).

### No microphone? Make a test clip with brain's own TTS

```bash
brain tts synth --text "the quick brown fox jumps over the lazy dog" --out /tmp/tts.wav
sox /tmp/tts.wav -r 16000 -c 1 /tmp/tts16k.wav          # 24 kHz -> 16 kHz mono
python3 examples/asr/transcribe_mic.py --wav /tmp/tts16k.wav
```

## Dependencies

- `jeepney` — D-Bus with fd passing (always required).
- `sounddevice` + `numpy` — only for live mic capture (`pip install sounddevice numpy`).
  Not needed for `--wav`.

## How it works

```
mic / wav ──f32 PCM──▶ os.pipe() write end
                          │  (client)
                          ▼
      StreamTranscribe(model, params, pcm_read_fd) ─┐  D-Bus, fd passed via SCM_RIGHTS
                                                     ▼  (server, crates/dbus)
              reader thread: accumulate PCM → 1 s windows
                          │
                          ▼  one transcribe Job per window
                 residency::Executor  ── batches concurrent streams,
                          │                schedules across CPU/GPU/NPU
                          ▼
              segment frames  ◀──SEQPACKET event fd──  client prints live
              ... done{text}
```

The window size is `--window-ms` (default 1000). Near-real-time latency ≈ one window
plus the model's compute (RTF ≪ 1 on the FastConformer encoder). The protocol frames
(`segment` / `done` / `error`) are the same SEQPACKET transport `Subscribe` uses; see
`crates/dbus/src/stream.rs`.

## Benchmark

`examples/asr/bench_streams.py` drives N concurrent streams through the same path and
reports per-model RTF, first-segment latency, throughput and the scheduler's batch
counters — see its `--help`.
