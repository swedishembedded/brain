# GPT

A dense, decoder-only Transformer built to nanoGPT parity: token embeddings
plus learned positional embeddings, causal multi-head self-attention, and a
GELU-activated MLP, repeated per layer with pre-norm LayerNorm. It's brain's
simplest and most direct reference decoder — reach for it when you want a
straightforward from-scratch train/evaluate/generate loop on your own
character-level or BPE dataset, without the extra machinery (grouped-query
attention, mixture-of-experts, quantization, paged serving) the larger models
carry.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| INT8                   | [ ] |
| CLI (`brain do`)       | [x] |
| HTTP API               | [x] |
| D-Bus                  | [x] |
| Batched serving        | [ ] |

## Getting the weights

`brain/gpt` is a reserved built-in id with no upstream checkpoint — there's
nothing to fetch. Train your own with `brain gpt train`, or point
`BRAIN_GPT2_WEIGHTS` at an existing brain-format checkpoint to serve one you
already have.

## Running it

```bash
brain gpt train <data_dir> --out gpt.safetensors --steps 2000
brain gpt eval  --weights gpt.safetensors --data <data_dir>
brain gpt gen   --weights gpt.safetensors --prompt "..." --max-new 200
```

`gen` / `sample` / `generate` are all accepted as aliases for the same
inference path. Generation uses an incremental KV-cache, so long completions
don't recompute the whole prefix each step.

> Bare `brain train` / `brain eval` / `brain generate` (no `gpt`) run brain's
> separate toy Sparse MoE model, not this one — always say `brain gpt …` to
> reach this decoder.

To serve it over HTTP or D-Bus:

```bash
BRAIN_GPT2_WEIGHTS=gpt.safetensors brain serve --dbus --openai
```

Once serving, it's reachable like any other resident model over `brain do`,
D-Bus, or the HTTP APIs — one request decodes at a time (no concurrent
continuous-batching engine, unlike Qwen3).

## Options

- `--steps`, `--batch`, `--block` (context length), `--lr`, `--layers`,
  `--d-model`, `--heads`, `--warmup`, `--grad-accum`, `--seed`, `--mask`,
  `--align` — training.
- `--batches`, `--samples`, `--seed` — eval.
- `--prompt`, `--max-new`, `--temp`, `--top-k`, `--seed` — generation
  (greedy argmax when `--temp <= 0`).
- `--device cpu|gpu|gpuN` (or `BRAIN_DEVICE`) — backend selection, global.

## Hardware and limits

Runs on the cpu and gpu backends. The output head is untied from the token
embedding (unlike classic nanoGPT's weight tying), and GELU uses the tanh
approximation rather than exact erf. There's no HuggingFace import, no INT8
quantization, and no paged-KV concurrent serving engine — those are Qwen3's.
Context length is whatever `--block` you train with; there's no fixed cap
beyond available memory.
