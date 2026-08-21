# Music generation over D-Bus (MiniMax Music 3)

`generate_song.py` drives brain's generic D-Bus interface
(`com.swedishembedded.Brain1`, protocol in `examples/dbus/README.md`)
against the `brain/minimaxmusic3` model: streaming lyrics+caption-conditioned
song generation.

```
generate_song.py                         brain-py/brain_py/dbus.py
        | jeepney (session bus)
        v
com.swedishembedded.Brain1  --Subscribe-->  (job, event fd: SEQPACKET)
        |                                      | progress frames
        v                                      | blob frame (the WAV)
residency executor --instance key--> resident minimaxmusic3 (load-per-call)
```

Unlike the image/video blob conventions, the `audio` output blob here is
already a **complete WAV byte stream** (`meta.format == "wav"` - the same
convention `qwen3tts synth` uses), so the example writes it to disk
directly with no client-side encoding step.

## Run

```bash
# deps: pip install -e brain-py   (jeepney with fd passing)
dbus-run-session -- bash -c '
  BRAIN_MINIMAXMUSIC3_LM=/path/to/language_model \
  BRAIN_MINIMAXMUSIC3_DEPTH=/path/to/rvq_depth_decoder \
  BRAIN_MINIMAXMUSIC3_CONDITION=/path/to/condition_encoder \
  BRAIN_MINIMAXMUSIC3_DIT=/path/to/transformer \
  BRAIN_MINIMAXMUSIC3_VOCODER=/path/to/vocoder \
  BRAIN_MINIMAXMUSIC3_TOKENIZER=/path/to/qwen3-8B-tokenizer-music \
  BRAIN_DEVICE=cpu \
  brain serve --dbus & sleep 2

  python3 examples/musicgen/generate_song.py \
      --caption "warm acoustic ballad, gentle piano, soft vocals, 80 BPM" \
      --lyrics "$(printf "[verse]\nquiet morning light\n[chorus]\nhold on to this feeling\n")" \
      --out song.wav'
```

`BRAIN_DEVICE=cpu` matters on a machine whose GPU cannot hold the Global
LLM's ~3.28 GB embedding/`lm_head` tensors as single buffers (an Intel
integrated GPU, for instance).

## What this has and has not been validated against

Every piece of this pipeline - prompt assembly, the CFG-guided AR sampling
loop, chunked DiT denoising, vocoder crop-and-stitch - is real, real-weight
code, unit and structurally tested against the reference algorithm. What
this repo's own CI/dev environment could **not** do is run a real
end-to-end generation through the served path above: whole-Global-LLM
residency (~8B parameters) does not fit in RAM on the machine this port
was built on, on either of its backends, a measured and diagnosed gap
recorded in this repo's own roadmap ledger and in
`crates/minimaxmusic3/src/global_llm.rs`'s own `import` doc. This example
is real, ready code for a machine with more RAM, a real int8-capable CPU
compute path, or a discrete GPU - not a demonstration this repo has itself
run to completion.

## The CLI, for comparison

The same generation without a server, one command, one playable file:

```bash
BRAIN_MINIMAXMUSIC3_LM=… BRAIN_MINIMAXMUSIC3_DEPTH=… BRAIN_MINIMAXMUSIC3_CONDITION=… \
BRAIN_MINIMAXMUSIC3_DIT=… BRAIN_MINIMAXMUSIC3_VOCODER=… BRAIN_MINIMAXMUSIC3_TOKENIZER=… \
BRAIN_DEVICE=cpu brain minimaxmusic3 generate \
    --lyrics "..." --caption "..." --duration_seconds 10 \
    --out audio=song.wav
```
