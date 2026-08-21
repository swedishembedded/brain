# MiniMax Music 3 (lyrics + caption-conditioned music generation)

Give it lyrics and a structured music description (genre, BPM, vocal
timbre, instrumentation, arrangement) and it generates a full song - up to
five minutes, 44.1 kHz stereo, with vocals, evolving arrangement and a
coherent structure (intro/verse/chorus/bridge/outro). Reach for it when you
need a first-draft song from a text brief, not for editing or remixing
existing audio (there is no audio-conditioned path).

**Status: under construction, component by component.** All five
components - condition encoder, vocoder, RVQ depth decoder, the
flow-matching DiT, and the Global LLM (a real Qwen3-8B, reused verbatim
from `crates/qwen3`) - have import + forward, each verified at real
weights (cosine 1.0 vs the reference; the Global LLM via a single
streamed real decoder layer, not whole-model residency - see below). The
vocoder, depth decoder, and DiT are also fully trainable (full fine-tune
with gradcheck, LoRA); the DiT additionally has an INT8 storage tier and
`model::Shardable` pipeline sharding (single-device validated only - no
discrete GPU on the machine this was built on). The pipeline glue -
prompt assembly, the CFG-guided AR sampling loop, chunked DiT denoising,
vocoder crop-and-stitch - is implemented and unit/structurally tested,
but a real, real-checkpoint short end-to-end generation could **not** be
validated on this development machine: whole-8B-model residency exceeds
what either of this machine's backends can hold (see "Hardware and
limits" below) - a measured, diagnosed gap, not an unwritten feature.
Nothing is wired to the CLI yet. See
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

Whole-8B-model residency (the Global LLM) does not fit on the machine
this port was built on, on either of its backends: the CPU-JIT backend's
`int8` request silently promotes to fp32 (no backend in this workspace
executes real int8 compute yet - `backend-cpu`'s own
`caps().numeric.int8_dot` is `false`), and this machine's actual GPU (an
Intel integrated Vulkan device, not a discrete card) caps single buffers
at 2047 MiB, below the ~3.28 GB embedding/`lm_head` tensors regardless of
dtype. Measured directly (a single Global LLM instance OOM-kills on this
machine's ~26 GB available RAM). Neither is a defect in this port - both
are pre-existing framework limits this port's own testing was the first
to exercise at whole-8B-model scale; see
[`.agents/roadmap/minimaxmusic3.md`](../../.agents/roadmap/minimaxmusic3.md)'s
Phase 10 for the full diagnosis. A machine with more RAM, a real
int8-capable CPU path, or a discrete GPU can run
`crates/minimaxmusic3/tests/e2e_short_generation.rs` as-is.

License: MiniMax-Music3 Community License (attribution + a UI credit
requirement for commercial use, a >$20M/yr revenue registration clause; no
non-commercial restriction). See the checkpoint's own `LICENSE` file.
