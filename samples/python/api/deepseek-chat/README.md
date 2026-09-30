# sample: api/deepseek-chat

Chat with a DeepSeek-R1 distill over brain's OpenAI-compatible HTTP surface,
with the model's reasoning returned apart from its answer.

```bash
brain pull deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
python3 samples/python/api/deepseek-chat/deepseek_chat.py
```

That launches (and stops) its own `brain serve --openai`. Point at a server
you already launched instead:

```bash
brain serve --openai 8788 --api-keys-out /tmp/keys.json &
python3 samples/python/api/deepseek-chat/deepseek_chat.py --base-url http://127.0.0.1:8788 --keys-file /tmp/keys.json
```

## What it demonstrates

* The checkpoint's own chat template: an R1 distill's template opens the
  `<think>` block itself, and brain renders it as shipped.
* `message.reasoning_content` apart from `message.content`, and the same two
  channels as separate streamed deltas.
* Generation ends on the checkpoint's own end token (`finish_reason: "stop"`),
  not on the budget. `max_tokens` covers the reasoning too, so it is generous
  by default.

Any `deepseek-ai/DeepSeek-R1-Distill-*` checkpoint works through `--model`; a
7-8B one is served with int8 linears on a single 24 GB card. `--model
brain/mock` runs it offline against the weight-free mock.

## Options

| flag | default |
|---|---|
| `--model ID` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `--question TEXT` | `What is 17 + 25? Reply with just the number.` |
| `--max-tokens N` | `2048` |
| `--base-url URL` | unset (self-launches a server instead) |
| `--api-key KEY` / `--keys-file PATH` | required with `--base-url` |
| `--brain PATH` | `./target/release/brain` (self-launch mode only) |
| `--port N` | `8788` (self-launch mode only) |
