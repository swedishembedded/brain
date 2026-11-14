# sample: api/claude-with-brain

Run the real `claude` CLI against `brain serve --anthropic` instead of the
hosted Anthropic API - brain's Anthropic-compatible HTTP surface
(`crates/apiserve`) speaks the Messages API, so Claude Code needs no code
change, only different `ANTHROPIC_*` environment variables.

```bash
make build/release
samples/shell/api/claude-with-brain/claude-with-brain.sh                # interactive claude on Qwen/Qwen3-0.6B
samples/shell/api/claude-with-brain/claude-with-brain.sh -p "hi"        # or pass any claude flags through
samples/shell/api/claude-with-brain/claude-with-brain.sh --model Qwen/Qwen3-8B -- -p "hi"
```

## What it demonstrates

1. Preflight: `brain` binary present, `claude` on `PATH`. Nothing else -
   the model (`--model`, a fully-qualified `<vendor>/<repo>` reference,
   `Qwen/Qwen3-0.6B` by default) does not need to be fetched or converted ahead of time.
2. Launches `brain serve --anthropic PORT --ready-file PATH` in the
   background, waits for `PATH` to appear (touched only once the listener is
   actually bound; see `brain_shutdown::ready::Gate`), and reads the
   freshly-generated per-launch API key from the `--api-keys-out` file (0600,
   written strictly before the ready file - see `crates/apiserve/src/surface.rs`).
3. Exports `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_KEY` and points every model
   alias (including the haiku-class background model) at the model, so nothing
   reaches the hosted API for the duration of the session.
4. `exec`s `claude "$@"` - fully interactive. The brain server is stopped
   automatically (`trap cleanup EXIT INT TERM`) when Claude Code exits.

That's it - no import step. The first message sent in `claude` is the first
request that names the model; brain's transparent auto-fetch downloads and
converts it right then, streaming progress to Claude Code while it does, and
every request after is instant.

## What it needs

- `make build/release` (a `brain` binary).
- The `claude` CLI on `PATH`.
- Nothing pre-fetched: point `--model` at any `<vendor>/<repo>[-<QUANT>]` your
  build of brain can serve to use a different model. The script serves with
  `--autofetch`; pass `--no-autofetch` to require the model already be pulled
  (`brain pull REF`) and fail fast rather than fetching.

## Options

Options come before any `claude` flags; `--` ends them.

| option | default |
|---|---|
| `--model REF` | `Qwen/Qwen3-0.6B` |
| `--port N` | `8787` |
| `--brain PATH` | `./target/release/brain` |
| `--models-dir DIR` | the model store `brain serve` resolves (`--brain-data-dir`, else the user's data directory) |
| `--max-output-tokens N` | `8192`: the reply budget Claude Code asks for. Unset, it asks for 32000 on top of a prompt of about 20000 tokens, which a request must fit in the local model's context or it is refused |
| `--ready-timeout SECONDS` | `600`: how long to wait for the server to come up (its startup scans the whole model store) |
| `--no-autofetch` | do not download the model on demand; it must already be in the model store |
| `--mock` | serve the weight-free mock model instead (also skips the `claude`-installed check under `--check`) |

`--check` runs the same preflight -> launch -> key-capture sequence, makes
ONE authenticated `GET /v1/models`, prints `OK`, and exits without exec'ing
`claude` - useful as a CI-safe liveness check that needs no interactive
terminal.

## Why this needs a real model

Claude Code makes multi-turn tool-calling requests (its tools reach the model
as function tools; the model's calls come back as `tool_use` blocks), and only
a language model that calls tools answers them usefully; the mock echoes.
`--check` is what stays fast and offline-safe (a discovery route, which never
triggers a fetch). The full end-to-end
interactive path is `tests/e2e/claude_code.bats` (`make test/e2e/claude-code`),
which runs against the deterministic mock model (`brain serve --mock`) instead (so it needs
no network and never hangs on a cold fetch) and skips cleanly unless
`claude`/`jq`/`timeout` and a brain binary are present.
