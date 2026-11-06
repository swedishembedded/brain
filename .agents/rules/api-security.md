# API security audit (mandatory on every API-surface change)

**Invariant.** Any change to a brain **API surface** — the HTTP providers
(`crates/apiserve`: Anthropic / OpenAI / OpenRouter) **or** the D-Bus surface
(`crates/dbus`, `com.swedishembedded.Brain1`) — requires auditing the **whole** API
against this checklist before the change is considered done, **not** just the handler
you touched. A change to one route can weaken an invariant another route relied on.

Run the pass with the repo `security-review` skill over `crates/apiserve` +
`crates/dbus` (+ the CLI wiring in `crates/cli/src/run_cli.rs`), fix every finding, and
note the audit in the change. Write automated tests for whatever the audit finds —
a checklist item that is only ever verified by eye will regress silently the next
time a route changes. The tests live in two places, and a new route or field adds
its assertions to both: the socket-level `tests/e2e/api_conformance.bats` (against
the `BRAIN_MOCK` model: auth matrix including no enumeration, no key leak, 413,
error hygiene, input-bound 400s, admission 429) and the in-process
`crates/apiserve/tests/it/api.rs`. These surfaces are internet-reachable when the operator binds
them, so treat all request input as hostile.

## Checklist

### 1. Authentication & authorization
- [ ] Every route requires the per-provider key — including the 404 fallback — so an
      unauthenticated caller cannot even enumerate which routes exist. (apiserve auth is
      a layer wrapping the whole router; verify no route is added *outside* it.)
- [ ] Key comparison is **constant-time** (no early-return on first mismatched byte).
- [ ] The key is never logged, echoed in an error body, or written to a world-readable
      file. `--api-keys-out` writes with restrictive permissions; the startup `APIKEY`
      line carries the key only when no `--api-keys-out` file holds it.
- [ ] Anthropic uses `x-api-key`; OpenAI/OpenRouter use `Authorization: Bearer`. A
      missing/blank/malformed header → 401 with a provider-shaped body, no stack/detail.
- [ ] D-Bus: the bus name and (system-bus) method access policy are intentional; no
      method exposes more than the operator intends. A vetted `system.d` policy
      ships with the deb (`scripts/build/com.swedishembedded.Brain1.conf`):
      calls restricted to root + the `brain` group, so `--dbus-system` never
      depends on an operator improvising an allow-everyone policy.

### 2. Input handling / DoS
- [ ] Request bodies are size-limited (a max content length) so a huge body can't OOM.
- [ ] JSON parsing is depth/size-bounded (no unbounded nesting / billion-laughs).
- [ ] Numeric params are range-checked before use: `max_tokens`/`max_new`,
      `n`, `top_k`, `dimensions`, batch/`input` array length, image `size`. Reject
      absurd values (400) rather than allocating on them.
- [ ] No caller-settable param names a file on the serving machine. A path param
      (`weights`, `data`, `save`, `adapter`, `start_frame`, ...) is declared
      `.host_env(..)`/`.host_resolved()`; `/v1/run`, `/v1/jobs` and D-Bus
      `Run`/`Subscribe`/`Plan` validate through `ActionSpec::validate_served`,
      which refuses it as unknown, and `/v1/capabilities` and D-Bus `Manifests`
      list `for_serving` manifests. `crates/catalog/tests/it/served_paths.rs` pins
      this over every catalog manifest.
- [ ] Sampling parameters go through `crates/apiserve/src/sampling.rs` on every text
      surface (chat and `/completions`): an omitted `seed`/`top_k` is NOT filled in (the action's own default
      applies; an unseeded request must not decode a fixed sequence), `top_k` outside
      0..=1000 or a non-integer `seed` is a 400, and an OpenAI parameter brain cannot
      honour (penalties, `logit_bias`, logprobs, non-text `response_format`) is a
      400 naming it rather than silently ignored.
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
      tool-calling conversation). Each is a 400, not a truncation/best-effort pass-through.

### 3. Resource safety & backpressure
- [ ] Media a client sends is decoded by brain's own code only, within a budget
      checked as it decodes: a clip goes through `imaging::video::clip_from_untrusted_file`
      (never `clip_from_file`, whose ffmpeg fallback lets the file choose demuxers and
      protocols), every container count is bounded by the file's own length and every
      size sum is checked (`videocodec::mp4` tests), and a decoded frame is capped at
      `videocodec::MAX_FRAME_PIXELS` inside the AV1 decoder, before it is allocated.
- [ ] No request decodes media on an async worker: images, audio and clips go through
      `crate::preprocess::off_runtime` (blocking pool, `MAX_CONCURRENT_DECODES` at once,
      503 past `DECODE_QUEUE_DEADLINE`), and an image is refused from its header against
      what is left of `MAX_REQUEST_PIXELS`, never after decoding it.
- [ ] No peer, keyed or not, can hold a connection without sending: every surface
      is served by `crates/apiserve/src/listener.rs` (never `axum::serve`, which
      runs no header timer), with a header deadline that also ends idle kept-alive
      connections, a whole-body deadline (not a per-frame one, which a byte-per-
      interval trickle defeats), a connection cap closing the excess at accept, and
      HTTP/1.1 only (HTTP/2 has no header deadline). `crates/apiserve/tests/it/connections.rs`
      pins each.
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
      completions/embeddings/images on HTTP; `Run`/`Subscribe`/`StreamTranscribe` on D-Bus) for a
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
      on an embedded newline/CR, so it must be stripped before rendering. The message
      is display-only in both wire shapes; no code path interpolates it into a path,
      command, or further request.

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
      resident layer (`crates/catalog/src/resident_llm.rs::QwenInstance::run`) guarantees
      raw `<think>`/`<tool_call>` markup never leaks into `message.content`/
      `delta.content` — only `ChatEvent::Content` ever feeds those fields (see
      `bridge::StreamMsg`'s doc comment and `openai.rs::event_delta`, which is the
      ONLY path that builds `reasoning_content`/`tool_calls` deltas). The Anthropic
      surface's `thinking` blocks and `thinking_delta`s are built only from the same
      `reasoning` events and the outcome's `reasoning_content`
      (`anthropic.rs::Blocks`), never from the text stream.
- [ ] `/completions` is the one exception, by design: its `text` is the raw
      generation (the resident's `ChatScanner::raw`), markup included, because the
      client sent a raw prompt and asked for its continuation. It is still only
      model output; `suffix` reaches the model only as the text after a
      fill-in-the-middle hole, and a vocabulary with no FIM tokens is a 400 with a
      fixed message (`bridge::map_reply_err`), never the resident's error text.

### 7. Transport
- [ ] Servers bind `127.0.0.1` by default (localhost only); binding a public interface
      is an explicit operator choice, documented, and gated behind auth (always on).
- [ ] Nothing but loopback is served in plaintext: `apiserve::tls::check_exposure` runs
      before any model loads, so a non-loopback address needs `--tls-cert`/`--tls-key`
      (rustls on the ring provider, HTTP/1.1 by ALPN, handshake inside the header
      deadline) or the operator's explicit `--insecure-plaintext`. A private network
      is still a network: the key in a request header is readable to every host on it.
      `crates/apiserve/tests/it/tls.rs` and the unit tests in `tls.rs` pin it.
- [ ] CORS is intentional (the permissive default is acceptable only because every
      route is key-gated; re-confirm if that changes).

### 8. Model output that is executed (training verifiers)
- [ ] brain never executes model output itself. The one path that runs
      generated code is `rl::verify::ExternalVerifier`: an OPERATOR-supplied
      command (the sandbox) that receives the completion as JSON data on stdin -
      never in its argv, never through a shell brain starts - and prints a
      reward. Nothing constructs one by default, and no serving surface (HTTP,
      D-Bus, `brain do`) can reach it; it exists only for training runs an
      operator configures explicitly.
- [ ] The command runs in its own process group, killed whole (a stray child
      included) at its timeout or when it exits; its stdout is read up to 1 MiB;
      a timeout, non-zero exit or malformed reward is a zero reward, never a
      parse of partial output. `crates/rl/src/verify.rs`'s tests pin each.
- [ ] Isolation (no network, a read-only or scratch filesystem, CPU/memory
      limits) is the operator's command's job, and the docs of any flag that
      takes such a command must say so. Threat: a policy trained against a
      verifier learns to emit whatever the verifier rewards, including code
      that escapes a weak sandbox - treat every completion as hostile code.

## The training job surface (`brain/train`, `--training`)

Training is a model of the same server, so every item above applies to it; these are the parts specific to it.
`crates/trainjob/tests/it/{door,http,dbus}.rs` and `tests/e2e/api_conformance.bats` pin each.

- [ ] A caller names no path. The base is a `<vendor>/<repo>` parsed by `ModelRef::parse` (separators, dot
      segments and adapter suffixes refused) and resolved in the model store; a dataset is a blob of the call or a
      digest, which parses from exactly 64 lowercase hex digits and so cannot spell a path.
- [ ] The artifact store is the one place a client puts bytes the server reads. A file is opened by digest with
      `O_NOFOLLOW | O_NONBLOCK`, must be a regular file, is size-bounded, and is copied into the job's private
      directory while it is hashed; one that does not hash to its name is removed and refused. What a remote
      caller is told never names a path of this machine.
- [ ] Every scalar is bounded by the action's spec (steps, rank, row length, accumulation...), the `data` document
      is at most 64 KiB, parsed with unknown fields refused, at most 8 replay sets, and the evaluations of a run
      are bounded so the report fits one datagram.
- [ ] A request that could never run (unknown base, digest not in the store, objective-specific settings on the
      wrong objective, a run larger than the largest card) is refused at the door (`ResidentModel::check_job`, a 400
      or a bus error) before a job exists or a card is claimed; a run never evicts a serving instance to make room
      for something that then fails.
- [ ] The event stream is a per-caller view of the job like its status and result (another caller gets 404), is
      bounded (4096 frames kept, 8 followers a job) and carries the run's statistics and refusals, not data.
- [ ] Anyone holding the key can occupy a card with a run: training is off unless the operator passes
      `--training`, and its admission obeys the surface's deadline (429), but there is no per-caller quota.

## When to run
- Adding/removing/altering any route, handler, auth path, error shape, admission
  policy, or D-Bus method.
- Adopting a new upstream spec field via the spec-sync command (`api-sync`) that
  changes accepted input. brain keeps at most two sources of truth for an API:
  its code and the vendored upstream OpenAPI specs in
  `crates/apiserve/tests/specs/`, validated against each other by the
  conformance tests. There is no separate hand-maintained brain spec.
- Before any release that changes the API surface.
