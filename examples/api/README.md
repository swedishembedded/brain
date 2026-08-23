# Claude Code against a local brain server

`claude-with-brain.sh` runs the real `claude` CLI against `brain serve --anthropic`
instead of the hosted Anthropic API — brain's Anthropic-compatible HTTP surface
(`crates/apiserve`) speaks the Messages API, so Claude Code needs no code change,
only different `ANTHROPIC_*` environment variables.

## What it does

1. Preflight: `brain` binary present, `claude` on `PATH`. Nothing else — `MODEL`
   (a fully-qualified `<vendor>/<repo>` reference, `Qwen/Qwen3-0.6B` by default)
   does not need to be fetched or converted ahead of time.
2. Launches `brain serve --anthropic $PORT --ready-file PATH` in the background,
   waits for `PATH` to appear (touched only once the listener is actually bound;
   see `brain_shutdown::ready::Gate`), and reads the freshly-generated per-launch
   API key from its log (`APIKEY anthropic <key>`, printed once at startup,
   strictly before the ready file — see `crates/apiserve/src/surface.rs`).
3. Exports `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_KEY` and points every model alias
   (including the haiku-class background model) at `MODEL`, so nothing reaches the
   hosted API for the duration of the session.
4. `exec`s `claude "$@"` — fully interactive. The brain server is stopped
   automatically (`trap cleanup EXIT INT TERM`) when Claude Code exits.

## Run it

```bash
make release
examples/api/claude-with-brain.sh                 # interactive claude on Qwen/Qwen3-0.6B
examples/api/claude-with-brain.sh -p "hi"          # or pass any claude flags through
```

That's it — no import step. The first message you send in `claude` is the first
request that names `MODEL`; brain's transparent auto-fetch
downloads and converts it right then, streaming progress to Claude Code while it
does, and every request after is instant. Point `MODEL` at any
`<vendor>/<repo>[-<QUANT>]` your build of brain can serve to use a different model,
or the port with `PORT`. Set `BRAIN_AUTO_FETCH=0` to require `MODEL` already be
resident instead (the pre-auto-fetch behavior) and fail fast rather than fetching.

## Why this needs a real model (today)

Unlike every other example here, this one cannot run against `BRAIN_MOCK` in the
automated harness (`tests/e2e/examples.bats`) for its INTERACTIVE path — Claude Code
itself makes real, multi-turn tool-calling requests that need an actual language
model behind them, not a canned echo. It is exercised there only in `mode=check`
(server up, key captured, one authenticated `GET /v1/models` answered — a discovery
route, which never triggers a fetch, so `--check` stays fast even against the real,
not-yet-resident `MODEL` — see the harness manifest). The full end-to-end interactive
path is `tests/e2e/claude_code.bats` (`make test/e2e/claude-code`), which runs against
the deterministic `BRAIN_MOCK` model instead (so it needs no network and never hangs
on a cold fetch) and skips cleanly unless `claude`/`jq`/`timeout` and a brain binary
are present.

---

## Who builds brain

brain is built by **[Swedish Embedded AB](https://swedishembedded.com)** - we
put AI on hardware that ships.

Swedish Embedded AB implements self-hosted inference back-ends for agent
tooling, so a coding agent or an internal assistant runs against hardware you
control instead of a third-party API. If your team needs expertise in pointing
existing agent tooling at your own infrastructure, you can procure our
services by sending an email to **info@swedishembedded.com**.

More about what we build: <https://swedishembedded.com>.
