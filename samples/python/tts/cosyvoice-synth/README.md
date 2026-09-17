# sample: tts/cosyvoice-synth (D-Bus)

`cosyvoice_synth.py` drives `brain/cosyvoice`'s streaming `synth` action:
target text plus a reference clip and its transcript in, a real 24 kHz WAV
out - zero-shot voice cloning.

```
cosyvoice_synth.py                       brain-py/brain_py/dbus.py
        | jeepney (session bus)
        v
com.swedishembedded.Brain1  --Subscribe-->  (job, event fd: SEQPACKET)
        |                                      | progress frames
        v                                      | blob frame (the WAV)
residency executor --instance key--> resident cosyvoice (load-per-call)
```

```bash
BRAIN_COSYVOICE_LLM=/path/to/cosyvoice2 \
BRAIN_COSYVOICE_FLOW=/path/to/cosyvoice2 \
BRAIN_COSYVOICE_HIFT=/path/to/cosyvoice2 \
BRAIN_S3TOKENIZER_V2=/path/to/speech_tokenizer_v2.onnx-dir \
BRAIN_CAMPPLUS_DIR=/path/to/campplus.onnx-dir \
BRAIN_COSYVOICE_TOKENIZER=/path/to/cosyvoice2/CosyVoice-BlankEN \
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/tts/cosyvoice-synth/cosyvoice_synth.py \
      --text "Hello, this is a cloned voice." \
      --ref-audio reference.wav \
      --ref-text "the reference clip's own transcript" \
      --out clone.wav
```

## What it demonstrates

* Like `minimaxmusic3 generate` and `qwen3tts synth`, the `audio` output
  blob is already a **complete WAV byte stream** (`meta.format == "wav"`),
  so the script writes it to disk with no client-side encoding step.
* The reference clip goes the OTHER direction as an **input** blob
  (`ref_audio`): raw WAV file bytes, header included, rather than decoded
  PCM. `cosyvoice::caps::decode_ref_audio` (server-side) parses the WAV
  container's own sample rate directly, so this preserves the clip's full
  native rate - deliberately different from
  `brain do cosyvoice synth --in ref_audio=clip.wav` on the CLI, which
  downsamples any input clip to a fixed 16 kHz before the action ever sees
  it. Going through D-Bus directly, as this script does, avoids that cap.

## What it needs

Six env vars, all weights-related: `BRAIN_COSYVOICE_{LLM,FLOW,HIFT}`, the
speech tokenizer (`BRAIN_S3TOKENIZER_V2`), the speaker encoder
(`BRAIN_CAMPPLUS_DIR`), and the text BPE tokenizer identity
(`BRAIN_COSYVOICE_TOKENIZER`). Every `BRAIN_COSYVOICE_*` variable can
instead point at ONE directory holding all three CosyVoice 2 checkpoints
(`llm.pt`/`flow.pt`/`hift.pt`).

Only `--variant cosyvoice2` (the default) actually runs today -
`--variant cosyvoice3` reaches the server and gets a clear, typed error
back, never a silent fallback to CosyVoice 2's weights.

## Options

| flag | default |
|---|---|
| `--text TEXT` | *required* |
| `--ref-audio WAV` | *required* - any sample rate |
| `--ref-text TEXT` | *required* - the reference clip's own transcript |
| `--variant` | `cosyvoice2` (`cosyvoice3` errors cleanly, not implemented yet) |
| `--n-timesteps N` | model default (Euler steps the flow decoder's CFM solver takes) |
| `--seed N` | `0` |
| `--out PATH` | `cosyvoice_synth.wav` |

## The CLI, for comparison

```bash
BRAIN_COSYVOICE_LLM=… BRAIN_COSYVOICE_FLOW=… BRAIN_COSYVOICE_HIFT=… \
BRAIN_S3TOKENIZER_V2=… BRAIN_CAMPPLUS_DIR=… BRAIN_COSYVOICE_TOKENIZER=… \
brain cosyvoice synth \
    --text "..." --ref_text "..." \
    --in ref_audio=reference.wav --out audio=clone.wav
```
