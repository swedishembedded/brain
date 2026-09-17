# sample: api/claude-with-brain

Run the real `claude` CLI against `brain serve --anthropic` instead of the
hosted Anthropic API - brain's Anthropic-compatible HTTP surface
(`crates/apiserve`) speaks the Messages API, so Claude Code needs no code
change, only different `ANTHROPIC_*` environment variables.

```bash
make build/release
samples/shell/api/claude-with-brain/claude-with-brain.sh                # interactive claude on Qwen/Qwen3-0.6B
samples/shell/api/claude-with-brain/claude-with-brain.sh -p "hi"        # or pass any claude flags through
```

## What it demonstrates

1. Preflight: `brain` binary present, `claude` on `PATH`. Nothing else -
   `MODEL` (a fully-qualified `<vendor>/<repo>` reference, `Qwen/Qwen3-0.6B`
   by default) does not need to be fetched or converted ahead of time.
2. Launches `brain serve --anthropic $PORT --ready-file PATH` in the
   background, waits for `PATH` to appear (touched only once the listener is
   actually bound; see `brain_shutdown::ready::Gate`), and reads the
   freshly-generated per-launch API key from its log (`APIKEY anthropic
   <key>`, printed once at startup, strictly before the ready file - see
   `crates/apiserve/src/surface.rs`).
3. Exports `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_KEY` and points every model
   alias (including the haiku-class background model) at `MODEL`, so nothing
   reaches the hosted API for the duration of the session.
4. `exec`s `claude "$@"` - fully interactive. The brain server is stopped
   automatically (`trap cleanup EXIT INT TERM`) when Claude Code exits.

That's it - no import step. The first message sent in `claude` is the first
request that names `MODEL`; brain's transparent auto-fetch downloads and
converts it right then, streaming progress to Claude Code while it does, and
every request after is instant.

## What it needs

- `make build/release` (a `brain` binary).
- The `claude` CLI on `PATH`.
- Nothing pre-fetched: point `MODEL` at any `<vendor>/<repo>[-<QUANT>]` your
  build of brain can serve to use a different model. Set `BRAIN_AUTO_FETCH=0`
  to require `MODEL` already be resident instead (the pre-auto-fetch
  behavior) and fail fast rather than fetching.

## Options

| env var | default |
|---|---|
| `MODEL` | `Qwen/Qwen3-0.6B` |
| `PORT` | `8787` |
| `BRAIN` | `./target/release/brain` |
| `BRAIN_MOCK=1` | serve the weight-free mock model instead (also skips the `claude`-installed check under `--check`) |

`--check` runs the same preflight -> launch -> key-capture sequence, makes
ONE authenticated `GET /v1/models`, prints `OK`, and exits without exec'ing
`claude` - useful as a CI-safe liveness check that needs no interactive
terminal.

## Why this needs a real model (today)

Unlike the other API sample, this one cannot run its interactive path
against `BRAIN_MOCK` in an automated harness - Claude Code makes real,
multi-turn tool-calling requests that need an actual language model behind
them, not a canned echo. `--check` is what stays fast and offline-safe
(a discovery route, which never triggers a fetch). The full end-to-end
interactive path is `tests/e2e/claude_code.bats` (`make test/e2e/claude-code`),
which runs against the deterministic `BRAIN_MOCK` model instead (so it needs
no network and never hangs on a cold fetch) and skips cleanly unless
`claude`/`jq`/`timeout` and a brain binary are present.
