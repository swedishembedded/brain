# Quickstart

Two runnable paths: train a small model from scratch, then run a real LLM
without touching a checkpoint file yourself. Both assume you've already run
`make release` (see [Install](install.md)).

## Train something in 5 minutes

This trains brain's dense nanoGPT-parity baseline on a synthetic arithmetic
task, end to end:

```bash
make data/calculator                  # generate a toy addition dataset
make train/gpt/calculator             # train -> out/gpt-calculator.safetensors
make eval/gpt/calculator              # validation perplexity + task exact-match
```

`make data/calculator` writes a `train.bin`/`val.bin` dataset of arithmetic
problems; `make train/gpt/calculator` trains a GPT decoder on it and writes
`out/gpt-calculator.safetensors`; `make eval/gpt/calculator` loads that
checkpoint back and prints its validation perplexity and exact-match accuracy
on held-out problems, so you can see it actually learned the task.

## Run a real LLM locally

This runs `Qwen/Qwen3-0.6B`, a real instruct/chat model, over brain's
OpenAI-compatible HTTP API — no manual download or conversion step:

```bash
brain serve --openai 8788 &
# brain prints a line like: APIKEY openai sk-brain-...  -- copy that key

curl http://localhost:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <key from brain serve>' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

The first request downloads and converts `Qwen/Qwen3-0.6B` from Hugging Face
automatically (a one-time cost), then answers it; every request after that
is served from the already-converted, resident checkpoint. See
[Models & weights](../using/models-and-weights.md) for how model ids and
auto-fetch work, and [Serving](../using/serving.md) for the full `brain
serve` reference.

## Where to go next

- [The CLI](../using/cli.md) — every `brain <cmd>`, and the uniform
  `brain caps`/`brain do` entry points.
- [Configuration](../using/configuration.md) — the `BRAIN_*` environment
  variables that control devices, weights, and tuning.
- [Hardware](hardware.md) — running on CPU, GPU, or the Intel NPU.
