# Speech synthesis over D-Bus

Two clients, both against brain's generic D-Bus interface
(`com.swedishembedded.Brain1`, protocol in `examples/dbus/README.md`):

| script | model | what it shows |
| --- | --- | --- |
| `qwen3tts_speak.py` | `brain/qwen3tts` | a **resident** model, with audio arriving in chunks WHILE it generates |
| `cosyvoice_synth.py` | `brain/cosyvoice` | zero-shot voice cloning from a reference clip (load-per-call) |

---

# Streaming synthesis from a resident model (Qwen3-TTS)

`qwen3tts_speak.py` drives the `brain/qwen3tts` resident's `speak` (or
`design`) action. Both are `.streaming()`: the server vocodes in chunks and
sends each one as an out-of-band blob frame on the `Subscribe` stream while
generation is still running, then sends the complete waveform as the terminal
blob.

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

The two things that matter here, and the reason this action exists on the
D-Bus surface at all:

* **Resident.** The Talker (2.8 GiB), MTP, codec and tokenizer are loaded once
  when the scheduler activates the instance, not per request. Back-to-back
  requests reuse them.
* **Progressive.** `brain_py`'s `subscribe(..., on_chunk=...)` fires the
  moment each segment lands, so the example can report a real
  time-to-first-audio - the number that decides whether playback can start
  before generation finishes. The chunks concatenate byte-for-byte into the
  terminal blob, which the example checks.

Unlike CosyVoice's, this action's `audio` blob is **raw mono f32
little-endian PCM at 24 kHz** (exactly as `speak_spec` declares), so the
example wraps it in a WAV container itself with the stdlib `wave` module.

The served voice is configured by the ENVIRONMENT, not per call: with
`BRAIN_QWEN3TTS_REF` set to a reference clip, `speak` voice-clones that timbre
(in-context when `BRAIN_QWEN3TTS_REF_TEXT` also gives its transcript);
without it, `speak` is speaker-free synthesis. `--action design` takes its
`--instruct`/`--speaker` per call instead, and needs a CustomVoice or
VoiceDesign checkpoint.

## Run

```bash
# deps: pip install -e brain-py   (jeepney with fd passing)
dbus-run-session -- bash -c '
  BRAIN_QWEN3TTS_WEIGHTS=out/tts-base06 \
  BRAIN_QWEN3TTS_CKPT=/path/to/Qwen3-TTS-12Hz-0.6B-Base \
  BRAIN_QWEN3TTS_STREAM_CHUNK=4 \
  brain serve --dbus & sleep 4

  python3 examples/tts/qwen3tts_speak.py \
      --text "Hello from a resident text to speech model." \
      --max-frames 40 --seed 7 --out speak.wav'
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

`BRAIN_QWEN3TTS_STREAM_CHUNK` is the chunk size in CODEC frames (12.5 Hz);
it defaults to 16 (~1.3 s of audio per chunk). Smaller chunks mean earlier
first audio and more frames on the wire.

## Batching, for comparison

`batch` synthesizes N texts in one interleaved, ragged call - every request
advances one frame per round and finishes at its own EOS, so a short clip is
not stuck behind a long one:

```bash
brain qwen3tts batch --weights_dir out/tts-base06 --ckpt /path/to/hf-ckpt \
    --requests '[{"text":"First.","max_frames":40},{"text":"Second.","max_frames":24}]' \
    --out audio_0=a.wav --out audio_1=b.wav
```

---

# Zero-shot voice cloning over D-Bus (CosyVoice)

`cosyvoice_synth.py` drives brain's generic D-Bus interface
(`com.swedishembedded.Brain1`, protocol in `examples/dbus/README.md`)
against the `brain/cosyvoice` model: streaming zero-shot voice cloning -
target text + a reference audio clip and its transcript in, a real 24 kHz
WAV out.

```
cosyvoice_synth.py                       brain-py/brain_py/dbus.py
        | jeepney (session bus)
        v
com.swedishembedded.Brain1  --Subscribe-->  (job, event fd: SEQPACKET)
        |                                      | progress frames
        v                                      | blob frame (the WAV)
residency executor --instance key--> resident cosyvoice (load-per-call)
```

Like `minimaxmusic3 generate` and `qwen3tts synth`, the `audio` output blob
here is already a **complete WAV byte stream** (`meta.format == "wav"`), so
the example writes it to disk directly with no client-side encoding step.

The reference clip goes the other direction as an **input** blob
(`ref_audio`): the example sends the raw WAV file bytes, header included,
rather than decoding to PCM first. `cosyvoice::caps::decode_ref_audio`
(server-side) parses a WAV container's own sample rate directly, so this
preserves the clip's full native rate. That is deliberately different from
`brain do cosyvoice synth --in ref_audio=clip.wav` on the CLI, which
downsamples any input clip to a fixed 16 kHz before the action ever sees
it (`crates/cli/src/caps_cli.rs`'s generic audio-blob loader, shared by
every model with an audio input) - going through D-Bus directly, as this
script does, avoids that cap.

## Run

```bash
# deps: pip install -e brain-py   (jeepney with fd passing)
dbus-run-session -- bash -c '
  BRAIN_COSYVOICE_LLM=/path/to/cosyvoice2 \
  BRAIN_COSYVOICE_FLOW=/path/to/cosyvoice2 \
  BRAIN_COSYVOICE_HIFT=/path/to/cosyvoice2 \
  BRAIN_S3TOKENIZER_V2=/path/to/speech_tokenizer_v2.onnx-dir \
  BRAIN_CAMPPLUS_DIR=/path/to/campplus.onnx-dir \
  BRAIN_COSYVOICE_TOKENIZER=/path/to/cosyvoice2/CosyVoice-BlankEN \
  brain serve --dbus & sleep 2

  python3 examples/tts/cosyvoice_synth.py \
      --text "Hello, this is a cloned voice." \
      --ref-audio reference.wav \
      --ref-text "the reference clip'\''s own transcript" \
      --out clone.wav'
```

Every `BRAIN_COSYVOICE_*` variable can instead point at ONE directory
holding all three CosyVoice 2 checkpoints (`llm.pt`/`flow.pt`/`hift.pt`) -
the released layout `cosyvoice::pipeline`'s own module doc describes.

Only `variant=cosyvoice2` (the default) actually runs - CosyVoice 3's
pipeline is accepted as a value but not implemented yet; passing
`--variant cosyvoice3` reaches the server and gets a clear, typed error
back, never a silent fallback to CosyVoice 2's weights.

## What this has and has not been validated against

Every piece of this pipeline - CAM++/S3Tokenizer reference-clip analysis,
the speech-token LM's autoregressive sampling, the flow decoder's Euler
CFM loop, the HiFT vocoder - is real, real-weight code, and
`cosyvoice::pipeline::generate` (the exact function this action serves)
has itself been run end to end on real checkpoints, producing a real
playable WAV, on the machine this milestone was built on. The `synth`
action wraps that same function; only the D-Bus round trip in the diagram
above (rather than a direct CLI invocation) is unexercised by this repo's
own CI, since that leg needs a session bus and a running `brain serve`
process, not different pipeline code.

## The CLI, for comparison

The same generation without a server, one command, one playable file:

```bash
BRAIN_COSYVOICE_LLM=… BRAIN_COSYVOICE_FLOW=… BRAIN_COSYVOICE_HIFT=… \
BRAIN_S3TOKENIZER_V2=… BRAIN_CAMPPLUS_DIR=… BRAIN_COSYVOICE_TOKENIZER=… \
brain cosyvoice synth \
    --text "..." --ref_text "..." \
    --in ref_audio=reference.wav --out audio=clone.wav
```

---

## Who builds brain

brain is built by **[Swedish Embedded AB](https://swedishembedded.com)** - we
put AI on hardware that ships.

Swedish Embedded AB implements speech synthesis and voice cloning for products
that need a voice of their own, generated locally. If your team needs
expertise in text-to-speech, neural vocoders, or streaming audio generation,
you can procure our services by sending an email to
**info@swedishembedded.com**.

More about what we build: <https://swedishembedded.com>.
