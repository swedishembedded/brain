# Seq2seq

A general encoder-decoder Transformer: a bidirectional encoder over the
source sequence, and a causal decoder that attends both to its own prior
tokens and to the encoder's output via cross-attention. This is brain's
architecture for sequence-to-sequence tasks shaped like translation — mapping
one token sequence to another, rather than continuing a single stream the
way GPT/Qwen3/GLM do.

## Support

| Capability | Supported |
|---|---|
| Inference             | [ ] |
| Training from scratch | [x] |

## Getting the weights

There's no published checkpoint, model id, or HuggingFace import path.
Weights come only from fresh random initialization or loading a raw
checkpoint file directly.

## Running it

There is no `brain seq2seq` subcommand yet — this architecture doesn't have
a command-line workflow of its own. Forward, backward, and an AdamW step are
all wired and exercised by its own gradient-check and convergence tests
(it's proven, for example, to learn a copy task through cross-attention
alone), but training currently has to be driven by a manual loop rather than
brain's shared generic trainer, and there's no decode/generate path at all
yet — so there's nothing to run for inference today.

## Options

The encoder and decoder context lengths are configured independently
(source and target sequences can differ in length), along with encoder
depth, decoder depth, hidden width, attention head count, and MLP width —
but only via direct configuration, since there's no CLI flag surface yet.

## Hardware and limits

No inference, sampling, or serving path — training only. No `brain seq2seq`
CLI. No HuggingFace or safetensors import; weights arrive only via fresh
initialization or a raw checkpoint load. Dropout is disabled.
