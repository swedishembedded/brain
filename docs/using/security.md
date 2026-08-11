# Security — before you expose this on a network

What to know before you bind an HTTP or D-Bus surface somewhere other than your own
machine.

- **Every route requires a key, including error responses.** Every HTTP route on every
  provider — Anthropic, OpenAI, OpenRouter — is authenticated, and that includes the
  404 fallback for a route that doesn't exist. An unauthenticated caller can't even
  enumerate which routes are live.
- **Servers bind localhost by default.** `127.0.0.1` is the default for every surface;
  binding a public interface is something you do explicitly by choosing a different
  address, not something that happens on its own.
- **Keys are generated fresh per server start.** Nothing is reused across runs, and a
  key is never written into a log line or echoed back in an error body — if you lose
  it, restart the server (or read it back from `--api-keys-out FILE` if you passed
  one) rather than looking for it in output.
- **Malformed or oversized input gets a clean rejection, not a crash.** Request bodies
  and inputs are size- and depth-bounded, so an overly large or maliciously nested
  request is turned away with a 4xx rather than exhausting memory or hanging the
  server.
- **What brain talks to on the network:** with auto-fetch enabled (the default —
  `BRAIN_AUTO_FETCH`, see [`docs/using/configuration.md`](configuration.md#serving--admission)),
  a request for a model brain doesn't have locally can trigger an outbound download.
  The only destination that ever reaches is `huggingface.co` — nothing else. Set
  `BRAIN_AUTO_FETCH=0` if you want brain to only ever serve models already present on
  disk and make no outbound connections at all.

For the wire-level detail of each surface, see
[`docs/using/http-api.md`](http-api.md) and [`docs/using/dbus-api.md`](dbus-api.md).
