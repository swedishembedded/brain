# API security audit (mandatory on every API-surface change)

**Invariant.** Any change to a brain **API surface** — the HTTP providers
(`crates/apiserve`: Anthropic / OpenAI / OpenRouter) **or** the D-Bus surface
(`crates/dbus`, `com.swedishembedded.Brain1`) — requires auditing the **whole** API
against this checklist before the change is considered done, **not** just the handler
you touched. A change to one route can weaken an invariant another route relied on.

Run the pass with the repo `security-review` skill over `crates/apiserve` +
`crates/dbus` (+ the CLI wiring in `crates/cli/src/run_cli.rs`), fix every finding, and
note the audit in the change. These surfaces are internet-reachable when the operator
binds them, so treat all request input as hostile.

## Checklist

### 1. Authentication & authorization
- [ ] Every route requires the per-provider key — including the 404 fallback — so an
      unauthenticated caller cannot even enumerate which routes exist. (apiserve auth is
      a layer wrapping the whole router; verify no route is added *outside* it.)
- [ ] Key comparison is **constant-time** (no early-return on first mismatched byte).
- [ ] The key is never logged, echoed in an error body, or written to a world-readable
      file. `--api-keys-out` writes with restrictive permissions; the startup `APIKEY`
      line goes to stderr only.
- [ ] Anthropic uses `x-api-key`; OpenAI/OpenRouter use `Authorization: Bearer`. A
      missing/blank/malformed header → 401 with a provider-shaped body, no stack/detail.
- [ ] D-Bus: the bus name and (system-bus) method access policy are intentional; no
      method exposes more than the operator intends.

### 2. Input handling / DoS
- [ ] Request bodies are size-limited (a max content length) so a huge body can't OOM.
- [ ] JSON parsing is depth/size-bounded (no unbounded nesting / billion-laughs).
- [ ] Numeric params are range-checked before use: `max_tokens`/`max_new`,
      `n`, `top_k`, `dimensions`, batch/`input` array length, image `size`. Reject
      absurd values (400) rather than allocating on them.
- [ ] The request `model` string is treated as an opaque catalog id — it must NOT be
      interpolated into a filesystem path, a URL, a shell command, or a D-Bus path.
      Resolve it only against `exec.manifests()`.

### 3. Resource safety & backpressure
- [ ] Admission is bounded: a request that can't start within the admit deadline → 429
      (not an unbounded queue); the edge concurrency limit + load-shed → 503. Neither
      lets a client pin all lanes or grow memory without bound.
- [ ] Cancel-on-disconnect actually frees compute: a dropped SSE/response cancels the
      job (the `CancelGuard`/`CancelToken` reaches the running action, which polls it).
      Verify a disconnected client's generation stops within ~one step.
- [ ] Per-connection / per-stream limits exist where relevant; no route holds a lane
      while doing unbounded host allocation (see the streaming/mmap OOM invariant).

### 4. SSRF / egress
- [ ] No handler fetches an attacker-controlled URL (e.g. image/document `url` inputs,
      OpenRouter passthrough fields). Either resolve locally or reject; never let a
      request cause brain to make an outbound request to an arbitrary host.

### 5. Error hygiene
- [ ] Error bodies carry a provider-shaped `{type/message/code}` only — never an
      internal error string, file path, panic message, or stack trace.
- [ ] A panicking handler is isolated (one connection/request), never taking down the
      server or leaking state across requests.

### 6. Output / data exposure
- [ ] `/models` (and any listing) exposes only the operator's served models and the
      capability-appropriate subset per provider — no internal paths, budgets, or keys.
- [ ] Stats/observability surfaces (`crates/stats`, D-Bus stats stream) never include
      keys or request contents.

### 7. Transport
- [ ] Servers bind `127.0.0.1` by default (localhost only); binding a public interface
      is an explicit operator choice, documented, and gated behind auth (always on).
- [ ] CORS is intentional (the permissive default is acceptable only because every
      route is key-gated; re-confirm if that changes).

## When to run
- Adding/removing/altering any route, handler, auth path, error shape, admission
  policy, or D-Bus method.
- Adopting a new upstream spec field via the spec-sync command (`api-sync`) that
  changes accepted input.
- Before any release that changes the API surface.
