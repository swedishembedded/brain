# Claude Code against a local brain server

`claude-with-brain.sh` runs the real `claude` CLI against `brain serve --anthropic`
instead of the hosted Anthropic API — brain's Anthropic-compatible HTTP surface
(`crates/apiserve`) speaks the Messages API, so Claude Code needs no code change,
only different `ANTHROPIC_*` environment variables.

## What it does

1. Preflight: `brain` binary present, `claude` on `PATH`, a brain-native qwen3
   checkpoint + tokenizer on disk.
2. Launches `brain serve --anthropic $PORT` in the background, waits for it to bind,
   and reads the freshly-generated per-launch API key from its log
   (`APIKEY anthropic <key>`, printed once at startup — see
   `crates/apiserve/src/surface.rs`).
3. Exports `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_KEY` and points every model alias
   (including the haiku-class background model) at the local `qwen` resident, so
   nothing reaches the hosted API for the duration of the session.
4. `exec`s `claude "$@"` — fully interactive. The brain server is stopped
   automatically (`trap cleanup EXIT INT TERM`) when Claude Code exits.

## Run it

```bash
make release
./target/release/brain qwen import --hf /path/to/Qwen3-0.6B --out qwen3.safetensors
# (a HuggingFace Qwen3 dir: config.json + model.safetensors + tokenizer.json)

examples/api/claude-with-brain.sh                 # interactive claude on your local qwen3
examples/api/claude-with-brain.sh -p "hi"          # or pass any claude flags through
```

Override the checkpoint location with `BRAIN_QWEN_WEIGHTS`/`BRAIN_QWEN_TOKENIZER`,
or the port with `PORT`.

## Why this needs real weights (today)

Unlike every other example here, this one cannot run against `BRAIN_MOCK` in the
automated harness (`tests/e2e/examples.bats`) — Claude Code itself makes real,
multi-turn tool-calling requests that need an actual language model behind them,
not a canned echo. It is exercised there only in `mode=check` (server up, key
captured, one authenticated request answered — see the harness manifest), and it
stays fully interactive when run by hand as above. The end-to-end interactive path
against a real qwen model is `tests/e2e/claude_code.bats`
(`make test/e2e/claude-code`), which itself skips cleanly unless `claude` is
installed and `BRAIN_QWEN_WEIGHTS`/`BRAIN_QWEN_TOKENIZER` are set.
