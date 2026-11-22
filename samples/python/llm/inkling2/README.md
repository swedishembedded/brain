<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# sample: llm/inkling2

Inkling 2 reasoning at two efforts, over any of brain's three transports.
`inkling2.py` asks one question with `reasoning_effort` low and then high and
prints how much each reply reasoned before it answered.

Inkling 2 replies in typed messages: a thinking message, the answer, then the
end of the reply. The served reply carries the reasoning in
`reasoning_content` and the answer as the text; a generation runs through the
thinking message to the reply's end. `reasoning_effort` is the checkpoint's own
template variable: `low`, `medium`, `high` or a number from 0 to 1; anything
else is refused by name.

## What it needs

The model in the store and a running server:

```bash
brain pull thinkingmachines/Inkling-2-Mini
tools/dbus-session.sh --serve "--dbus" -- \
  python3 samples/python/llm/inkling2/inkling2.py --dbus
```

Over OpenAI-compatible HTTP (the server prints `APIKEY openai <key>` at
startup):

```bash
brain serve --openai 8788 &
python3 samples/python/llm/inkling2/inkling2.py --openai localhost:8788 --api-key sk-brain-...
```

Anthropic-compatible HTTP is the same with `brain serve --anthropic 8787` and
`--anthropic localhost:8787`. Ad hoc, with no server, `brain do inkling2
generate --prompt "..."` runs the same model in-process.

Without the model the script prints what to run and exits cleanly. A
deps-free wire-contract check against the mock resident:

```bash
tools/dbus-session.sh --serve "--dbus --mock" -- \
  python3 samples/python/llm/inkling2/inkling2.py --dbus --model brain/mock
```

## Options

| flag | default |
|---|---|
| `--dbus` / `--openai URL` / `--anthropic URL` | exactly one is required |
| `--api-key KEY` | for the HTTP transports |
| `--model NAME` | `thinkingmachines/Inkling-2-Mini` |
| `--max-new N` | `512` |
