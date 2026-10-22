# Quickstart

Two runnable paths: train a small model from scratch, then run a real LLM
without touching a checkpoint file yourself. Both assume you've already run
`make build/release` (see [Install](install.md)).

## Train something in 5 minutes

This trains brain's dense nanoGPT-parity baseline on a synthetic arithmetic
task, end to end:

```bash
make gpt/calculator/data                  # generate a toy addition dataset
make gpt/calculator/train             # train -> out/gpt-calculator.safetensors
make gpt/calculator/eval              # validation perplexity + task exact-match
```

`make gpt/calculator/data` writes a `train.bin`/`val.bin` dataset of arithmetic
problems; `make gpt/calculator/train` trains a GPT decoder on it and writes
`out/gpt-calculator.safetensors`; `make gpt/calculator/eval` loads that
checkpoint back and prints its validation perplexity and exact-match accuracy
on held-out problems, so you can see it actually learned the task.

## Check that this machine can serve a model

Before the first run, ask brain whether it can serve the model you have in mind.
It reads the model store, the free disk and each GPU's free memory; it loads no
weights and opens no network:

```bash
brain doctor --model Qwen/Qwen3-0.6B
```

```
ok   store: ~/.local/share/brain/models holds 35 servable model(s)
ok   disk: 159.5 GiB free
ok   model: Qwen/Qwen3-0.6B: 2.2 GiB of weights on disk
ok   tokenizer: ~/.local/share/brain/models/Qwen/Qwen3-0.6B/tokenizer.json
ok   gpu: the weights (2.2 GiB) fit on gpu1 (gpu0 3.9 GiB free of 24.0 GiB, gpu1 5.8 GiB free of 24.0 GiB)
ready: brain can serve here
```

(Paths are shortened here; brain prints them in full.) Each failed check is
printed as `FAIL` with the command or flag that fixes it - a model that is not
pulled names `brain pull <vendor>/<repo>`, a missing store names `--models-dir`
- and the exit status is 1, so a script can stop on it. A `warn` does not stop
anything: it says, for example, that no GPU has room for the weights right now,
in which case brain would run the model on the CPU backend, far slower. `--json`
prints the same report for tools.

## Run a real LLM locally

This runs `Qwen/Qwen3-0.6B`, a real instruct/chat model, over brain's
OpenAI-compatible HTTP API — no manual download or conversion step:

```bash
brain serve --autofetch &   # the OpenAI API on 127.0.0.1:8788; --autofetch: allow the one-time fetch below
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
automatically (a one-time cost; fetching is opt-in, hence the `--autofetch`
above - without it the request errors naming the missing weights), then
answers it; every request after that is served from the already-converted,
resident checkpoint. See
[Models & weights](../guides/inference/models-and-weights.md) for how model ids and
auto-fetch work, and [Serving](../guides/serving/serving.md) for the full `brain
serve` reference.

## Where to go next

- [The CLI](../reference/cli.md) - every `brain <cmd>`, and the uniform
  `brain caps`/`brain do` entry points.
- [Configuration](../reference/configuration.md) - the `BRAIN_*` environment
  variables that control devices, weights, and tuning.
- [Hardware](../guides/hardware/choosing-hardware.md) - running on CPU, GPU, or the Intel NPU.
- [Model catalog](../reference/models/index.md) - every other architecture brain
  supports (larger LLMs, vision, audio, image/video generation, and more),
  each with its own getting-started page.
