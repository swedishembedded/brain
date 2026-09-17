# sample: api/openai-client

Every D-Bus sample in this tree drives brain over `com.swedishembedded.Brain1`.
This is the odd transport out - plain HTTP, stdlib only (`urllib`/`http.client`,
no `requests`) - exercising all three OpenAI route families of brain's
OpenAI-compatible surface (`crates/apiserve`) in one pass: chat completions
(non-streaming, then streaming SSE), embeddings, and image generation.

```bash
BRAIN_MOCK=1 python3 samples/python/api/openai-client/openai_client.py --model brain/mock
```

That launches (and stops) its own `brain serve --openai` on the weight-free
mock model - offline and CI-safe. Point at a real model instead (auto-fetched
on first use):

```bash
python3 samples/python/api/openai-client/openai_client.py --model Qwen/Qwen3-0.6B
```

Or point at a server you already launched:

```bash
brain serve --openai 8788 &
python3 samples/python/api/openai-client/openai_client.py --base-url http://127.0.0.1:8788 \
    --api-key "$(...)" --model brain/mock
```

## What it demonstrates

* `POST /v1/chat/completions` - non-streaming, then streaming (SSE deltas),
  including the separate `reasoning_content` delta channel a reasoning model
  (Qwen3's whole family) uses.
* `POST /v1/embeddings` and `POST /v1/images/generations`.
* brain's auth contract: a fresh, per-launch API key printed as `APIKEY openai
  <key>` before the listener binds - a client either scrapes that line or
  reads it back from `--api-keys-out FILE`'s JSON (`--keys-file`).
* A model that doesn't serve a given route (e.g. a chat-only model hit
  against `/v1/embeddings`) is a per-demo skip, not a fatal error - every demo
  after it still runs.

## What it needs

Nothing extra to try it: `--model` defaults to `brain/mock` (deterministic,
weight-free), so this runs offline. Point `--model` at any served id to
exercise a real model - a `<vendor>/<repo>`-style reference auto-fetches on
first use.

## Options

| flag | default |
|---|---|
| `--model ID` | `brain/mock` |
| `--base-url URL` | unset (self-launches a server instead) |
| `--api-key KEY` | required with `--base-url`, unless `--keys-file` |
| `--keys-file PATH` | read the `openai` key from a `--api-keys-out` JSON file |
| `--brain PATH` | `./target/release/brain` (self-launch mode only) |
| `--port N` | `8788` (self-launch mode only) |
| `--out PATH` | `/tmp/openai_client_image.png` |
