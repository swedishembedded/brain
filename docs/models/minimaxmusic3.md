# MiniMax Music 3 (lyrics + caption-conditioned music generation)

Give it lyrics and a structured music description (genre, BPM, vocal
timbre, instrumentation, arrangement) and it generates a full song - up to
five minutes, 44.1 kHz stereo, with vocals, evolving arrangement and a
coherent structure (intro/verse/chorus/bridge/outro). Reach for it when you
need a first-draft song from a text brief, not for editing or remixing
existing audio (there is no audio-conditioned path).

**Status: under construction, component by component.** Four of five
components - condition encoder, vocoder, RVQ depth decoder, and the
flow-matching DiT - have import + device forward, each verified at real
weights (cosine 1.0 vs the reference). The vocoder, depth decoder, and DiT
are also fully trainable (full fine-tune with gradcheck, LoRA); the DiT
additionally has an INT8 storage tier and `model::Shardable` pipeline
sharding (single-device validated only - no discrete GPU on the machine
this was built on). The Global LLM (the fifth component, a real Qwen3-8B)
is not yet wired; nothing end-to-end exists yet - no CLI verb, no actual
song generation. See
[`.agents/roadmap/minimaxmusic3.md`](../../.agents/roadmap/minimaxmusic3.md)
for the live milestone ledger.

## Support

| Capability | Supported |
|---|---|
| Inference              | [ ] |
| Training from scratch  | [ ] |
| LoRA fine-tune         | [ ] |
| INT8                   | [ ] |
| CLI (`brain minimaxmusic3 <action>`) | [ ] |
| HTTP API               | [ ] |
| D-Bus                  | [ ] |
| Batched serving        | [ ] |
| Multi-device sharding  | [ ] |
| NPU                    | [ ] |

## Getting the weights

Model id: `brain/minimaxmusic3`. `MiniMaxAI/MiniMax-Music3` auto-fetches on
first CLI use once import lands. Five components, one role each:

| Env var | Role |
|---|---|
| `BRAIN_MINIMAXMUSIC3_LM` | Global LLM (Qwen3-8B architecture) |
| `BRAIN_MINIMAXMUSIC3_DEPTH` | RVQ depth decoder |
| `BRAIN_MINIMAXMUSIC3_CONDITION` | Condition encoder |
| `BRAIN_MINIMAXMUSIC3_DIT` | Flow-matching DiT |
| `BRAIN_MINIMAXMUSIC3_VOCODER` | Vocoder |
| `BRAIN_MINIMAXMUSIC3_TOKENIZER` | Tokenizer (Qwen2Tokenizer-compatible) |

## Running it

Not wired yet - no `brain minimaxmusic3` verb exists.

## Options

Not wired yet.

## Hardware and limits

License: MiniMax-Music3 Community License (attribution + a UI credit
requirement for commercial use, a >$20M/yr revenue registration clause; no
non-commercial restriction). See the checkpoint's own `LICENSE` file.
