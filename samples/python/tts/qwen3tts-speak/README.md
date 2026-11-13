# sample: tts/qwen3tts-speak (D-Bus)

`qwen3tts_speak.py` drives `brain/qwen3tts`'s resident `synth`/`clone`/`design`
actions - all `.streaming()`: the server vocodes in chunks and sends each
one as an out-of-band blob frame on the `Subscribe` stream while
generation is still running, then sends the complete clip (a WAV) as the
terminal blob.

```
qwen3tts_speak.py                        brain-py/brain_py/dbus.py
        | jeepney (session bus)
        v
com.swedishembedded.Brain1  --Subscribe-->  (job, event fd: SEQPACKET)
        |                                      | blob frame  <- audio chunk 1
        |                                      | blob frame  <- audio chunk 2
        v                                      |   ... (on_chunk fires live)
residency executor --instance key--> resident qwen3tts (LOAD-ONCE engine)
                                               | blob frame  <- the whole clip
                                               | done frame
```

```bash
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/tts/qwen3tts-speak/qwen3tts_speak.py \
      --text "Hello from a resident text to speech model." \
      --max-frames 40 --stream-chunk 4 --out speak.wav
```

Real output from that command on the 0.6B Base checkpoint:

```
synth 'Hello from a resident text to speech model.' (streaming=True):
  chunk 1: 7680 samples at 11.5s
  chunk 2: 7680 samples at 13.3s
  ...
  chunk 8: 5760 samples at 23.6s
  time to first audio: 11.5s of 23.6s total
  streamed 59520 samples - matches the terminal clip's 59520
wrote speak.wav (59520 samples, 2.48s, 24000 Hz) in 23.6s
```

## What it demonstrates

* **Resident.** The Talker (2.8 GiB), MTP, codec and tokenizer load once
  when the scheduler activates the instance, not per request -
  back-to-back requests reuse them.
* **Progressive.** `on_chunk` fires the moment each segment lands, so the
  example reports a real time-to-first-audio - the number that decides
  whether playback can start before generation finishes. The chunks carry
  the same samples as the terminal clip, which the script checks by count.
* The mid-run chunks are **raw mono f32 little-endian PCM at 24 kHz** (their
  metadata says so); the terminal `audio` blob is the whole clip as a WAV,
  which the script writes as it is.

The voice is chosen per call: `synth` is speaker-free, `clone` takes a
reference recording as an input blob (`--ref`, in-context when `--ref-text`
also gives its transcript) and `design` takes `--instruct`/`--speaker` per
call, and needs a CustomVoice or VoiceDesign checkpoint.

## What it needs

A Qwen3-TTS checkpoint in the model store (`brain pull
Qwen/Qwen3-TTS-12Hz-0.6B-Base`). `--stream-chunk` is the chunk size in CODEC
frames; it defaults to 16 - smaller
chunks mean earlier first audio and more frames on the wire.

## Options

| flag | default |
|---|---|
| `--text TEXT` | *required* |
| `--action` | `synth` (`clone` needs `--ref`; `design` needs `--instruct`/`--speaker` and a CustomVoice/VoiceDesign checkpoint) |
| `--ref WAV`, `--ref-text TEXT` | `clone` only: the reference recording and its transcript |
| `--lang` | `english` |
| `--stream-chunk N` | the engine's default |
| `--max-frames N` | action default (codec frame cap) |
| `--seed N` | random: drawn and printed, echoed in the result |
| `--out PATH` | `qwen3tts_speak.wav` |

## Batching, for comparison

`batch` synthesizes N texts in one interleaved, ragged call - every request
advances one frame per round and finishes at its own EOS, so a short clip
is not stuck behind a long one:

```bash
brain qwen3tts batch \
    --requests '[{"text":"First.","max_frames":40},{"text":"Second.","max_frames":24}]' \
    --out audio_0=a.wav --out audio_1=b.wav
```
