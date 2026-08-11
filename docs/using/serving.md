# Serving — `brain serve`

`brain serve` loads whichever models are configured (see
[`docs/using/configuration.md`](configuration.md)), makes them resident under
one scheduler, and exposes them over one or more transports at once.

```
brain serve [--openai[:PORT]] [--anthropic[:PORT]] [--openrouter[:PORT]] \
            [--dbus] [--models-dir DIR] [--api-keys-out FILE] \
            [--ready-file PATH]
```

- `--openai[:PORT]` — serve an OpenAI-compatible HTTP API (default port if
  omitted). Every route is served both with and without the `/v1` prefix —
  either works as a client's `base_url`.
- `--anthropic[:PORT]` — serve an Anthropic-compatible HTTP API (`/v1` only).
- `--openrouter[:PORT]` — serve an OpenRouter-compatible HTTP API (with and
  without `/v1`, same as OpenAI).
- `--dbus` — serve the D-Bus control surface (`com.swedishembedded.Brain1`).
- `--models-dir DIR` — model directory to scan at startup (overrides
  `BRAIN_MODELS_DIR`).
- `--api-keys-out FILE` — also write the generated API keys to `FILE`, not
  just stdout.
- `--ready-file PATH` — see **Readiness** below.

At least one surface flag is meaningful for a useful server; an unknown flag
is a hard error (exit 2), not a warning. Run `brain serve --help` for the
authoritative flag list.

## Access control is always on

There is no way to run `brain serve` without authentication. Every requested
surface gets its own freshly generated API key at startup: printed to
stdout as `APIKEY` lines, and — if `--api-keys-out FILE` is given — also
written to `FILE`. Both the keys and the file are written before any socket
binds, so a client can never race the server up.

## Admission and backpressure

A request that can't be admitted to run promptly gets an HTTP 429 rather than
queuing indefinitely. The deadline is `BRAIN_ADMIT_DEADLINE_MS`, with one
exception: a model's first-ever activation ("cold start" — building the
model on device for the first time) gets a longer grace window,
`BRAIN_COLD_BUILD_ADMIT_DEADLINE_MS`, because that one activation is
inherently slower than every request after it. Both are documented with
their defaults in [`docs/using/configuration.md`](configuration.md#serving--admission).

Genuine overload — the server has more concurrent work than it can ever get
to, not just a momentary queue — sheds requests as HTTP 503.

## One model, every transport

A served model is uniformly discoverable and callable no matter how you
reach it:

- `brain caps` lists every model currently loaded and what it can do.
- `brain do <model> <action>` invokes any action, from the CLI, HTTP, or
  D-Bus — all three paths go through the same scheduler, the same batching,
  and the same cancellation semantics. There is no separate code path or
  separate behavior per transport.

Long-running actions — multi-minute generation, training — are cancellable
mid-flight from any transport that supports it (D-Bus `Cancel`, or closing
the client connection over HTTP).

## Readiness for scripted startup

`--ready-file PATH` touches an empty marker file once **every** requested
surface has actually bound — every HTTP dialect plus D-Bus — and never at
all if any of them fails to come up. Because the API keys are written before
any socket binds, the marker appearing implies both that the keys are on
disk and that the socket(s) are accepting connections, so a launcher script
can simply poll for the file's existence instead of polling a port or
grepping logs.

There is deliberately no `/healthz`-style unauthenticated route — readiness
here is a process-level property that also has to cover D-Bus, which has no
HTTP route to answer for it. Use `--ready-file` for scripted startup instead.

## Per-protocol detail

This page covers the `brain serve` command itself. For the wire-level detail
of each transport, see:

- [`docs/using/http-api.md`](http-api.md) — the OpenAI/Anthropic/OpenRouter
  HTTP surfaces.
- [`docs/using/dbus-api.md`](dbus-api.md) — the D-Bus control surface.
