<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: llm/edgerock

EdgeRock question answering over a long document, over any of brain's three
transports. `edgerock.py` sends a long document that states one fact at its
start, asks about it, then asks a second question about the same document.

EdgeRock is the Qwen3.5 hybrid decoder with a sliding window and a bounded
anchor memory, so a conversation costs the same fixed state at any length, and
`brain serve` keeps a copy of that state for each recent prompt. The second
request is resumed from the copy and reads in only its new tokens, which the
printed times show.

## What it needs

The model in the store and a running server:

```bash
brain pull rockmatrix-ai/EdgeRock
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/llm/edgerock/edgerock.py --dbus
```

Over OpenAI-compatible HTTP (the server prints `APIKEY openai <key>` at
startup):

```bash
brain serve --openai 8788 &
python3 samples/python/llm/edgerock/edgerock.py --openai localhost:8788 --api-key sk-brain-...
```

Anthropic-compatible HTTP is the same with `brain serve --anthropic 8787` and
`--anthropic localhost:8787`. Ad hoc, with no server, `brain do edgerock
generate --prompt "..."` runs the same model in-process.

Without the model the script prints what to run and exits cleanly. A
deps-free wire-contract check against the mock resident:

```bash
BRAIN_MOCK=1 tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/llm/edgerock/edgerock.py --dbus --model brain/mock
```

## Options

| flag | default |
|---|---|
| `--dbus` / `--openai URL` / `--anthropic URL` | exactly one is required |
| `--api-key KEY` | for the HTTP transports |
| `--model NAME` | `rockmatrix-ai/EdgeRock` |
| `--paragraphs N` | `60` |
| `--max-new N` | `24` |
