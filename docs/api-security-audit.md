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
- [ ] The request `model` string is treated as an opaque catalog id for DISPATCH — it
      must NOT be interpolated into a shell command. It IS deliberately parsed as a
      `<vendor>/<repo>[-<QUANT>]` model reference for **auto-fetch** classification
      (`residency::ModelSupplier::classify`, `crates/cli/src/supply.rs`'s
      `StoreSupplier`) when it doesn't already resolve against `exec.manifests()` — that
      parse (`brain_modelref::ModelRef::parse`) is itself a security boundary and must
      reject: a segment containing `/` beyond the one separator, a `.`/`..` segment
      (path traversal — `Store::repo_dir` joins vendor/repo onto the store root with no
      other check), and anything under a reserved vendor (`brain`/`local`/`test`) unless
      it's already on disk. `classify()` must do this with ZERO network or filesystem
      I/O for every reject case — only a name that both parses AND isn't reserved may
      trigger `ensure()`'s fetch. See item 4 for the resulting egress and item 5 for why
      a failed/refused fetch must still collapse to the exact same generic 404/"no
      model" a genuinely-unknown model gets.
- [ ] Tool-calling input (`crates/apiserve/src/openai.rs`'s `validate_tools`/
      `validate_tool_choice`, called from `to_invocation` BEFORE anything reaches the
      resident model / prompt renderer): `tools` must be an array of at most 128
      entries, at most 256 KiB serialized; every element's `function.name` must be a
      non-empty string of at most 64 characters; `tool_choice` must be one of
      `"auto"`/`"none"`/`"required"` or `{"type":"function","function":{"name":...}}`;
      any `role:"tool"` message must carry a non-empty `tool_call_id` (the linkage a
      malformed/adversarial client could otherwise omit to desync a multi-turn
      tool-calling conversation). Each is a 400, not a truncation/best-effort pass-through
      — see the bounds cases in `crates/apiserve/tests/api.rs`
      (`openai_chat_tools_bounds_are_400_not_panics_or_500s`).

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
- [ ] **Auto-fetch is the one sanctioned exception** — a dispatch request (chat/
      embeddings/images on HTTP; `Run`/`Subscribe`/`StreamTranscribe` on D-Bus) for a
      model that classifies `Fetchable` DOES cause an outbound request, to
      `huggingface.co` only (`crates/modelstore/src/hub.rs`'s `HfHub`, whose redirect
      handling is host-allowlisted — verify `hub.rs`'s allowlist test still covers this).
      The HOST is fixed by `HfHub`, never by request input — only the PATH (the
      vendor/repo/file) is client-influenced, which this checklist's own precedent
      treats as not a distinct SSRF concern by itself. Confirm: (a) `GET /models` and
      `GET /models/{id}` (pure discovery/read routes) never reach `ensure_and_recheck`/
      `ensure_resident` — only a dispatch route may fetch; (b) `BRAIN_AUTO_FETCH=0`
      (operator env var, not request-controlled) fully disables the supplier before a
      process ever constructs one; (c) HTTP surfaces stay loopback-only (item 7) — an
      operator who binds non-loopback is opting a wider set of callers into this
      already-sanctioned egress path, same as any other dispatch route.
- [ ] Live fetch-progress rendering (`bridge::stream_with_autofetch`'s SSE COMMENT
      lines, `Manager::subscribe`'s `phase:"fetching"` frames) never carries content
      that could break the wire format: the `name` field of a download step comes
      from the remote repo's OWN file names (chosen by whoever owns the HF repo, not
      by the requesting client) and is untrusted — `axum::sse::Event::comment` PANICS
      on an embedded newline/CR, so it must be stripped before rendering (see the
      regression test `a_newline_in_the_fetch_progress_name_does_not_panic_the_sse_
      stream`). The message is display-only in both wire shapes; no code path
      interpolates it into a path, command, or further request.

### 5. Error hygiene
- [ ] Error bodies carry a provider-shaped `{type/message/code}` only — never an
      internal error string, file path, panic message, or stack trace.
- [ ] A panicking handler is isolated (one connection/request), never taking down the
      server or leaking state across requests.
- [ ] Auto-fetch specifically: `ensure()`'s failure reason (a hub URL, HTTP status, or
      on-disk path from `crates/modelstore`'s `plan`/`execute`/the `qwen`/`glm`/`lfm`
      importers' `import_as`) is logged server-side (`eprintln!`) only —
      `bridge::ensure_and_recheck` (HTTP) and `Manager::ensure_resident` (D-Bus) both
      collapse EVERY non-success outcome (no supplier, `Unknown`, fetch `Err`, or a
      panicked `spawn_blocking` task) to the exact same generic "model not found" the
      client would see for a model that was simply never going to exist — a failed
      fetch must be indistinguishable from an unknown model, from the response alone.

### 6. Output / data exposure
- [ ] `/models` (and any listing) exposes only the operator's served models and the
      capability-appropriate subset per provider — no internal paths, budgets, or keys.
- [ ] Stats/observability surfaces (`crates/stats`, D-Bus stats stream) never include
      keys or request contents.
- [ ] Tool-call `arguments` (`message.tool_calls[].function.arguments`, both the
      non-streaming body and the streamed `delta.tool_calls[].function.arguments`
      fragments) are the MODEL's raw generated JSON text, re-serialized verbatim by
      `crates/apiserve/src/openai.rs::openai_tool_calls` — the server never parses
      and executes a tool call itself, and no server-side state (file paths, other
      requests, prior sessions) is ever echoed into an `arguments` string. The
      resident layer (`crates/cli/src/resident_llm.rs::QwenInstance::run`) guarantees
      raw `<think>`/`<tool_call>` markup never leaks into `message.content`/
      `delta.content` — only `ChatEvent::Content` ever feeds those fields (see
      `bridge::StreamMsg`'s doc comment and `openai.rs::event_delta`, which is the
      ONLY path that builds `reasoning_content`/`tool_calls` deltas).

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
