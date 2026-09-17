# sample: musicgen/generate-song

Music generation over D-Bus (MiniMax Music 3). `generate_song.py` drives
brain's generic D-Bus interface (protocol in
`samples/python/dbus/brain-dbus/README.md`) against the
`brain/minimaxmusic3` model: streaming lyrics+caption-conditioned song
generation.

```
generate_song.py                         brain-py/brain_py/dbus.py
        | jeepney (session bus)
        v
com.swedishembedded.Brain1  --Subscribe-->  (job, event fd: SEQPACKET)
        |                                      | progress frames
        v                                      | blob frame (the WAV)
residency executor --instance key--> resident minimaxmusic3 (load-per-call)
```

```bash
BRAIN_MINIMAXMUSIC3_LM=/path/to/language_model \
BRAIN_MINIMAXMUSIC3_DEPTH=/path/to/rvq_depth_decoder \
BRAIN_MINIMAXMUSIC3_CONDITION=/path/to/condition_encoder \
BRAIN_MINIMAXMUSIC3_DIT=/path/to/transformer \
BRAIN_MINIMAXMUSIC3_VOCODER=/path/to/vocoder \
BRAIN_MINIMAXMUSIC3_TOKENIZER=/path/to/qwen3-8B-tokenizer-music \
BRAIN_DEVICE=cpu \
  tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/musicgen/generate-song/generate_song.py \
      --caption "warm acoustic ballad, gentle piano, soft vocals, 80 BPM" \
      --lyrics "$(printf "[verse]\nquiet morning light\n[chorus]\nhold on to this feeling\n")" \
      --out song.wav
```

`BRAIN_DEVICE=cpu` matters on a machine whose GPU cannot hold the Global
LLM's ~3.28 GB embedding/`lm_head` tensors as single buffers (an Intel
integrated GPU, for instance).

## What it demonstrates

* Subscribes to the streaming `generate` action and writes the returned
  `audio` blob straight to disk: unlike the video/image blob conventions,
  this action already packs a **complete WAV byte stream** (`meta.format ==
  "wav"`, the same convention `qwen3tts synth` uses), so no client-side
  encoding step is needed.
* This sample's own default duration (10 s) is deliberately short - a full
  song can run several minutes to generate. Pass `--duration 240` for
  something closer to a real song.

## What it needs

Six weight directories, one per component (Global LLM, RVQ depth decoder,
condition encoder, flow-matching DiT, vocoder, tokenizer - see the flags
above), plus `jeepney` (`pip install -e brain-py`).

**What this has and hasn't been validated against**: every piece of this
pipeline - prompt assembly, the CFG-guided AR sampling loop, chunked DiT
denoising, vocoder crop-and-stitch - is real, real-weight code, unit and
structurally tested against the reference algorithm. Whole-Global-LLM
residency (~8B parameters) does not fit in RAM on the machine this port was
built on, on either of its backends - a measured and diagnosed gap recorded
in this repo's own roadmap ledger and in
`crates/minimaxmusic3/src/global_llm.rs`'s own `import` doc. This is real,
ready code for a machine with more RAM, a real int8-capable CPU compute
path, or a discrete GPU - not a demonstration this repo has itself run to
completion.

## Options

| flag | default |
|---|---|
| `--caption TEXT` | *required* - genre, BPM, vocal timbre, instrumentation |
| `--lyrics TEXT` | *required* - `[verse]`/`[chorus]`/etc structural tags |
| `--model MODEL` | `brain/minimaxmusic3` |
| `--out PATH` | `song.wav` |
| `--duration N` | `-1` - target seconds (`-1` = the model's own default, 10s) |
| `--steps N` | `-1` - Euler steps per denoise chunk (`-1` = model default) |
| `--seed N` | `0` |

## The CLI, for comparison

The same generation without a server, one command, one playable file:

```bash
BRAIN_MINIMAXMUSIC3_LM=... BRAIN_MINIMAXMUSIC3_DEPTH=... BRAIN_MINIMAXMUSIC3_CONDITION=... \
BRAIN_MINIMAXMUSIC3_DIT=... BRAIN_MINIMAXMUSIC3_VOCODER=... BRAIN_MINIMAXMUSIC3_TOKENIZER=... \
BRAIN_DEVICE=cpu brain minimaxmusic3 generate \
    --lyrics "..." --caption "..." --duration_seconds 10 \
    --out audio=song.wav
```
