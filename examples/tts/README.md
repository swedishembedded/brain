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
