# sample: tts/qwen3tts-speak (D-Bus)

`qwen3tts_speak.py` drives `brain/qwen3tts`'s resident `speak`/`design`
actions - both `.streaming()`: the server vocodes in chunks and sends each
one as an out-of-band blob frame on the `Subscribe` stream while
generation is still running, then sends the complete waveform as the
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
BRAIN_QWEN3TTS_WEIGHTS=out/tts-base06 \
BRAIN_QWEN3TTS_CKPT=/path/to/Qwen3-TTS-12Hz-0.6B-Base \
BRAIN_QWEN3TTS_STREAM_CHUNK=4 \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/tts/qwen3tts-speak/qwen3tts_speak.py \
      --text "Hello from a resident text to speech model." \
      --max-frames 40 --seed 7 --out speak.wav
```

Real output from that command on the 0.6B Base checkpoint:

```
speak 'Hello from a resident text to speech model.' (streaming=True):
  chunk 1: 7680 samples at 11.5s
  chunk 2: 7680 samples at 13.3s
  ...
  chunk 8: 5760 samples at 23.6s
  time to first audio: 11.5s of 23.6s total
  streamed 59520 samples - matches the terminal blob
wrote speak.wav (59520 samples, 2.48s, 24000 Hz) in 23.6s
```

## What it demonstrates

* **Resident.** The Talker (2.8 GiB), MTP, codec and tokenizer load once
  when the scheduler activates the instance, not per request -
  back-to-back requests reuse them.
* **Progressive.** `on_chunk` fires the moment each segment lands, so the
  example reports a real time-to-first-audio - the number that decides
  whether playback can start before generation finishes. The chunks
  concatenate byte-for-byte into the terminal blob, which the script
  checks.
* Unlike CosyVoice's, this action's `audio` blob is **raw mono f32
  little-endian PCM at 24 kHz** (exactly as `speak_spec` declares), so the
  script wraps it in a WAV container itself with the stdlib `wave` module.

The served voice is configured by the ENVIRONMENT, not per call: with
`BRAIN_QWEN3TTS_REF` set to a reference clip, `speak` voice-clones that
timbre (in-context when `BRAIN_QWEN3TTS_REF_TEXT` also gives its
transcript); without it, `speak` is speaker-free synthesis. `--action
design` takes its `--instruct`/`--speaker` per call instead, and needs a
CustomVoice or VoiceDesign checkpoint.

## What it needs

`BRAIN_QWEN3TTS_WEIGHTS` (the imported brain checkpoints) and
`BRAIN_QWEN3TTS_CKPT` (the HF checkpoint dir, for `config.json` + the
tokenizer). `BRAIN_QWEN3TTS_STREAM_CHUNK` is the chunk size in CODEC
frames (12.5 Hz); it defaults to 16 (~1.3 s of audio per chunk) - smaller
chunks mean earlier first audio and more frames on the wire.

## Options

| flag | default |
|---|---|
| `--text TEXT` | *required* |
| `--action` | `speak` (`design` needs `--instruct`/`--speaker` and a CustomVoice/VoiceDesign checkpoint) |
| `--lang` | server default |
| `--max-frames N` | action default (codec frame cap) |
| `--seed N` | action default |
| `--out PATH` | `qwen3tts_speak.wav` |

## Batching, for comparison

`batch` synthesizes N texts in one interleaved, ragged call - every request
advances one frame per round and finishes at its own EOS, so a short clip
is not stuck behind a long one:

```bash
brain qwen3tts batch --weights_dir out/tts-base06 --ckpt /path/to/hf-ckpt \
    --requests '[{"text":"First.","max_frames":40},{"text":"Second.","max_frames":24}]' \
    --out audio_0=a.wav --out audio_1=b.wav
```
