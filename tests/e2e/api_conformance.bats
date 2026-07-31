#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# HTTP conformance harness for brain's inference API surface, driven end-to-end over a
# REAL socket against a single `brain serve` process backed by the built-in
# deterministic mock model (BRAIN_MOCK=1). No real weights, no GPU, no `claude` — the
# mock is instant, so the whole suite runs in seconds and is fully deterministic.
#
# It validates every provider dialect (OpenAI, Anthropic, OpenRouter) against the
# vendored OpenAPI specs in crates/apiserve/tests/specs via scripts/validate_spec.py
# (the same sanitize→Draft-2020-12 path the Rust api.rs harness uses). If the Python
# `jsonschema` package is missing the schema checks degrade to structural jq checks
# (noted once at startup); the shape/behavior assertions always run.
#
# Run: make test/e2e/api-conformance   (or: BRAIN_BIN=./target/debug/brain bats tests/e2e/api_conformance.bats)
#
# SAFETY: teardown kills ONLY the recorded `brain serve` PID (kill -9 $SERVER_PID).
# It NEVER uses pkill / killall, which could match this harness or an unrelated
# process.

setup_file() {
  command -v jq >/dev/null 2>&1 || skip "jq not installed"
  command -v curl >/dev/null 2>&1 || skip "curl not installed"

  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no brain binary (build with: CARGO_BUILD_JOBS=6 make build, or set BRAIN_BIN)"
  export BRAIN

  export SPECS="$REPO/crates/apiserve/tests/specs"
  export VALIDATE="$REPO/scripts/validate_spec.py"

  # Is Python jsonschema available? If not, schema validation degrades to jq shape
  # checks (behavior assertions are unaffected).
  if python3 -c "import jsonschema" >/dev/null 2>&1; then
    export HAVE_SCHEMA=1
  else
    export HAVE_SCHEMA=0
    echo "# validate_spec.py: jsonschema unavailable — degrading to structural jq checks" >&3
  fi

  export CONF_DIR="$(mktemp -d)"
  export KEYS="$CONF_DIR/keys.json"
  export LOG="$CONF_DIR/serve.log"
  export OPENAI_PORT="${OPENAI_PORT:-8896}"
  export ANTHROPIC_PORT="${ANTHROPIC_PORT:-8897}"
  export OPENROUTER_PORT="${OPENROUTER_PORT:-8898}"

  # ONE server, all three surfaces, CPU-only, backed by the mock. Record its PID.
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve \
    --openai "$OPENAI_PORT" \
    --anthropic "$ANTHROPIC_PORT" \
    --openrouter "$OPENROUTER_PORT" \
    --api-keys-out "$KEYS" \
    >"$LOG" 2>&1 &
  export SERVER_PID=$!
  echo "$SERVER_PID" > "$CONF_DIR/pid"

  # Poll until all three /models return 200 with the per-provider key (bounded ≤ 30s).
  local ready=0
  for _ in $(seq 1 60); do
    if [ -f "$KEYS" ]; then
      local ok=$(jq -r .openai "$KEYS" 2>/dev/null)
      local ak=$(jq -r .anthropic "$KEYS" 2>/dev/null)
      local rk=$(jq -r .openrouter "$KEYS" 2>/dev/null)
      if [ -n "$ok" ] && [ "$ok" != "null" ]; then
        if curl -fsS --max-time 5 -o /dev/null -H "Authorization: Bearer $ok" "http://127.0.0.1:$OPENAI_PORT/v1/models" 2>/dev/null \
           && curl -fsS --max-time 5 -o /dev/null -H "x-api-key: $ak" "http://127.0.0.1:$ANTHROPIC_PORT/v1/models" 2>/dev/null \
           && curl -fsS --max-time 5 -o /dev/null -H "Authorization: Bearer $rk" "http://127.0.0.1:$OPENROUTER_PORT/models" 2>/dev/null; then
          ready=1
          break
        fi
      fi
    fi
    # Bail early if the server died.
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 0.5
  done
  if [ "$ready" != 1 ]; then
    echo "--- brain serve log ---" >&3
    cat "$LOG" >&3 || true
    skip "brain serve did not become ready on all three surfaces"
  fi

  export OPENAI_KEY="$(jq -r .openai "$KEYS")"
  export ANTHROPIC_KEY="$(jq -r .anthropic "$KEYS")"
  export OPENROUTER_KEY="$(jq -r .openrouter "$KEYS")"
}

teardown_file() {
  # Kill ONLY recorded brain serve PIDs — never pkill.
  if [ -n "${SERVER_PID:-}" ]; then
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  # The admission-429 test starts a SECOND short-deadline server and records its PID
  # here; kill it too even if that test aborted mid-way.
  if [ -n "${CONF_DIR:-}" ] && [ -f "$CONF_DIR/pid2" ]; then
    kill -9 "$(cat "$CONF_DIR/pid2")" 2>/dev/null || true
  fi
  [ -n "${CONF_DIR:-}" ] && rm -rf "$CONF_DIR"
}

# ------------------------------------------------------------------ helpers

# base url for a provider tag.
base() {
  case "$1" in
    openai) echo "http://127.0.0.1:$OPENAI_PORT" ;;
    anthropic) echo "http://127.0.0.1:$ANTHROPIC_PORT" ;;
    openrouter) echo "http://127.0.0.1:$OPENROUTER_PORT" ;;
  esac
}

# the auth header for a provider tag (with its key).
auth_hdr() {
  case "$1" in
    openai) echo "Authorization: Bearer $OPENAI_KEY" ;;
    anthropic) echo "x-api-key: $ANTHROPIC_KEY" ;;
    openrouter) echo "Authorization: Bearer $OPENROUTER_KEY" ;;
  esac
}

# the vendored spec file for a provider tag.
specfile() {
  case "$1" in
    openai) echo "openai.json" ;;
    anthropic) echo "anthropic.json" ;;
    openrouter) echo "openrouter.json" ;;
  esac
}

# validate <specfile> <SchemaName> <bodyfile>  — no-op if jsonschema is unavailable.
validate() {
  [ "$HAVE_SCHEMA" = 1 ] || return 0
  python3 "$VALIDATE" "$SPECS/$1" "$2" "$3"
}

# POST helper: post_json <provider> <path> <json-body> — writes status to $STATUS,
# response JSON to $CONF_DIR/resp.json (echoed as the last line for `run`).
post_json() {
  local prov="$1" path="$2" body="$3"
  local out="$CONF_DIR/resp.json"
  STATUS=$(curl -sS --max-time 15 -o "$out" -w '%{http_code}' \
    -H "$(auth_hdr "$prov")" -H 'content-type: application/json' \
    -X POST "$(base "$prov")$path" -d "$body")
  RESP="$out"
}

# GET helper: get <provider> <path>
get_url() {
  local prov="$1" path="$2"
  local out="$CONF_DIR/resp.json"
  STATUS=$(curl -sS --max-time 15 -o "$out" -w '%{http_code}' \
    -H "$(auth_hdr "$prov")" "$(base "$prov")$path")
  RESP="$out"
}

# ------------------------------------------------------------------ /models

@test "openai /models: 200, jq shape, lists mock, validates ListModelsResponse" {
  get_url openai /v1/models
  [ "$STATUS" -eq 200 ]
  [ "$(jq -r '.object' "$RESP")" = "list" ]
  [ "$(jq -r '.data | map(.id) | index("mock")' "$RESP")" != "null" ]
  validate openai.json ListModelsResponse "$RESP"
}

@test "openrouter /models: 200, lists mock, validates ModelsListResponse" {
  get_url openrouter /models
  [ "$STATUS" -eq 200 ]
  [ "$(jq -r '.data | map(.id) | index("mock")' "$RESP")" != "null" ]
  validate openrouter.json ModelsListResponse "$RESP"
}

@test "anthropic /models: 200, lists mock (chat surface)" {
  get_url anthropic /v1/models
  [ "$STATUS" -eq 200 ]
  [ "$(jq -r '.data | map(.id) | index("mock")' "$RESP")" != "null" ]
  [ "$(jq -r '.data[0].type' "$RESP")" = "model" ]
}

# ------------------------------------------------------ chat: non-streaming

@test "openai chat non-stream: 200, usage + finish_reason, validates CreateChatCompletionResponse" {
  post_json openai /v1/chat/completions \
    '{"model":"mock","messages":[{"role":"user","content":"hello there"}]}'
  [ "$STATUS" -eq 200 ]
  [ "$(jq -r '.object' "$RESP")" = "chat.completion" ]
  [ "$(jq -r '.choices[0].message.content' "$RESP")" = "You said: hello there" ]
  [ "$(jq -r '.choices[0].finish_reason' "$RESP")" = "stop" ]
  [ "$(jq -r '.usage.completion_tokens' "$RESP")" -eq 4 ]
  [ "$(jq -r '.usage.total_tokens' "$RESP")" -eq "$(jq -r '.usage.prompt_tokens + .usage.completion_tokens' "$RESP")" ]
  validate openai.json CreateChatCompletionResponse "$RESP"
}

@test "anthropic chat non-stream: 200, maps stop_reason, validates Message" {
  post_json anthropic /v1/messages \
    '{"model":"mock","max_tokens":64,"messages":[{"role":"user","content":"hello there"}]}'
  [ "$STATUS" -eq 200 ]
  [ "$(jq -r '.type' "$RESP")" = "message" ]
  [ "$(jq -r '.content[0].text' "$RESP")" = "You said: hello there" ]
  [ "$(jq -r '.stop_reason' "$RESP")" = "end_turn" ]
  [ "$(jq -r '.usage.output_tokens' "$RESP")" -eq 4 ]
  validate anthropic.json Message "$RESP"
}

@test "openrouter chat non-stream: 200, native_finish_reason, validates ChatResult" {
  post_json openrouter /chat/completions \
    '{"model":"mock","messages":[{"role":"user","content":"hello there"}]}'
  [ "$STATUS" -eq 200 ]
  [ "$(jq -r '.choices[0].message.content' "$RESP")" = "You said: hello there" ]
  [ "$(jq -r '.choices[0].finish_reason' "$RESP")" = "stop" ]
  [ "$(jq -r '.choices[0].native_finish_reason' "$RESP")" = "stop" ]
  validate openrouter.json ChatResult "$RESP"
}

# ------------------------------------------------------------- chat: SSE

@test "openai chat SSE: ordered chunks, [DONE] terminal, deltas concat to text, validates stream schema" {
  local sse="$CONF_DIR/openai_sse.txt"
  curl -sS -N --max-time 20 \
    -H "$(auth_hdr openai)" -H 'content-type: application/json' \
    -X POST "$(base openai)/v1/chat/completions" \
    -d '{"model":"mock","messages":[{"role":"user","content":"hello there"}],"stream":true,"stream_options":{"include_usage":true}}' \
    >"$sse"

  # Terminal marker is the last data payload.
  [ "$(grep '^data: ' "$sse" | tail -n1)" = "data: [DONE]" ]

  # First chunk announces the assistant role.
  local first="$(grep '^data: ' "$sse" | head -n1 | sed 's/^data: //')"
  [ "$(echo "$first" | jq -r '.choices[0].delta.role')" = "assistant" ]

  # Concatenate deltas + validate each JSON payload against the stream schema.
  local content=""
  while IFS= read -r line; do
    local payload="${line#data: }"
    [ "$payload" = "[DONE]" ] && continue
    echo "$payload" > "$CONF_DIR/chunk.json"
    validate openai.json CreateChatCompletionStreamResponse "$CONF_DIR/chunk.json"
    local piece="$(echo "$payload" | jq -r '.choices[0].delta.content // empty')"
    content="$content$piece"
  done < <(grep '^data: ' "$sse")
  [ "$content" = "You said: hello there" ]

  # include_usage emitted a usage chunk.
  grep '^data: ' "$sse" | grep -q '"usage"'
}

@test "anthropic messages SSE: message_start..message_stop, deltas concat, validates each event" {
  local sse="$CONF_DIR/anthropic_sse.txt"
  curl -sS -N --max-time 20 \
    -H "$(auth_hdr anthropic)" -H 'content-type: application/json' \
    -X POST "$(base anthropic)/v1/messages" \
    -d '{"model":"mock","max_tokens":64,"messages":[{"role":"user","content":"hello there"}],"stream":true}' \
    >"$sse"

  [ "$(grep '^event: ' "$sse" | head -n1)" = "event: message_start" ]
  [ "$(grep '^event: ' "$sse" | tail -n1)" = "event: message_stop" ]

  # Walk (event,data) pairs: validate each data against its event's schema and
  # concatenate the text_delta pieces.
  local ev="" content=""
  while IFS= read -r line; do
    case "$line" in
      "event: "*) ev="${line#event: }" ;;
      "data: "*)
        local payload="${line#data: }"
        echo "$payload" > "$CONF_DIR/evt.json"
        case "$ev" in
          message_start) validate anthropic.json MessageStartEvent "$CONF_DIR/evt.json" ;;
          content_block_start) validate anthropic.json ContentBlockStartEvent "$CONF_DIR/evt.json" ;;
          content_block_delta)
            validate anthropic.json ContentBlockDeltaEvent "$CONF_DIR/evt.json"
            content="$content$(echo "$payload" | jq -r '.delta.text // empty')" ;;
          content_block_stop) validate anthropic.json ContentBlockStopEvent "$CONF_DIR/evt.json" ;;
          message_delta) validate anthropic.json MessageDeltaEvent "$CONF_DIR/evt.json" ;;
          message_stop) validate anthropic.json MessageStopEvent "$CONF_DIR/evt.json" ;;
          ping) : ;;
        esac ;;
    esac
  done < "$sse"
  [ "$content" = "You said: hello there" ]
}

@test "openrouter chat SSE: [DONE] terminal, native_finish_reason, validates ChatStreamChunk" {
  local sse="$CONF_DIR/openrouter_sse.txt"
  curl -sS -N --max-time 20 \
    -H "$(auth_hdr openrouter)" -H 'content-type: application/json' \
    -X POST "$(base openrouter)/chat/completions" \
    -d '{"model":"mock","messages":[{"role":"user","content":"hello there"}],"stream":true}' \
    >"$sse"

  [ "$(grep '^data: ' "$sse" | tail -n1)" = "data: [DONE]" ]
  local saw_native=0 content=""
  while IFS= read -r line; do
    local payload="${line#data: }"
    [ "$payload" = "[DONE]" ] && continue
    echo "$payload" > "$CONF_DIR/chunk.json"
    validate openrouter.json ChatStreamChunk "$CONF_DIR/chunk.json"
    content="$content$(echo "$payload" | jq -r '.choices[0].delta.content // empty')"
    if [ "$(echo "$payload" | jq -r '.choices[0].finish_reason // empty')" = "stop" ]; then
      [ "$(echo "$payload" | jq -r '.choices[0].native_finish_reason')" = "stop" ]
      saw_native=1
    fi
  done < <(grep '^data: ' "$sse")
  [ "$content" = "You said: hello there" ]
  [ "$saw_native" -eq 1 ]
}

# ------------------------------------------------------------- embeddings

@test "openai /embeddings: 200, validates CreateEmbeddingResponse, float dim 8" {
  post_json openai /v1/embeddings '{"model":"mock","input":"hello world"}'
  [ "$STATUS" -eq 200 ]
  [ "$(jq -r '.object' "$RESP")" = "list" ]
  [ "$(jq -r '.data[0].embedding | length' "$RESP")" -eq 8 ]
  [ "$(jq -r '.usage.prompt_tokens' "$RESP")" -gt 0 ]
  validate openai.json CreateEmbeddingResponse "$RESP"
}

@test "openai /embeddings base64: decodes to 32 bytes (8 f32)" {
  # The base64 form is a string, which the spec's array-typed `embedding` does not
  # admit (OpenAI's own schema only models the float form), so — like the Rust api.rs
  # harness — we assert the decode, not the schema.
  post_json openai /v1/embeddings '{"model":"mock","input":"hello","encoding_format":"base64"}'
  [ "$STATUS" -eq 200 ]
  [ "$(jq -r '.data[0].embedding | type' "$RESP")" = "string" ]
  local n=$(jq -r '.data[0].embedding' "$RESP" | base64 -d | wc -c)
  [ "$n" -eq 32 ]
}

@test "openrouter /embeddings: 200, validates CreateEmbeddingResponse" {
  post_json openrouter /embeddings '{"model":"mock","input":["a","b"]}'
  [ "$STATUS" -eq 200 ]
  [ "$(jq -r '.data | length' "$RESP")" -eq 2 ]
  validate openai.json CreateEmbeddingResponse "$RESP"
}

# --------------------------------------------------------- image generation

@test "openai /images/generations: 200, validates ImagesResponse, b64_json decodes to PNG" {
  post_json openai /v1/images/generations '{"model":"mock","prompt":"a red cat","size":"256x256"}'
  [ "$STATUS" -eq 200 ]
  [ "$(jq -r '.created' "$RESP")" != "null" ]
  validate openai.json ImagesResponse "$RESP"
  local sig=$(jq -r '.data[0].b64_json' "$RESP" | base64 -d | od -An -tx1 -N8 | tr -d ' \n')
  [ "$sig" = "89504e470d0a1a0a" ]
}

@test "openrouter /images/generations: 200, b64_json decodes to PNG" {
  post_json openrouter /images/generations '{"model":"mock","prompt":"a dog","size":"256x256"}'
  [ "$STATUS" -eq 200 ]
  validate openai.json ImagesResponse "$RESP"
  local sig=$(jq -r '.data[0].b64_json' "$RESP" | base64 -d | od -An -tx1 -N8 | tr -d ' \n')
  [ "$sig" = "89504e470d0a1a0a" ]
}

# ----------------------------------------------------- anthropic count_tokens

@test "anthropic /v1/messages/count_tokens: 200, {input_tokens} > 0" {
  post_json anthropic /v1/messages/count_tokens \
    '{"model":"mock","messages":[{"role":"user","content":"count these tokens please"}]}'
  [ "$STATUS" -eq 200 ]
  [ "$(jq -r '.input_tokens' "$RESP")" -gt 0 ]
  validate anthropic.json BetaCountMessageTokensResponse "$RESP"
}

# --------------------------------------------------------------- auth / errors

@test "missing key -> 401 on each provider" {
  local st
  st=$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' "$(base openai)/v1/models")
  [ "$st" -eq 401 ]
  st=$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' "$(base anthropic)/v1/models")
  [ "$st" -eq 401 ]
  st=$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' "$(base openrouter)/models")
  [ "$st" -eq 401 ]
}

@test "malformed JSON body -> 400" {
  local st
  st=$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' \
    -H "$(auth_hdr openai)" -H 'content-type: application/json' \
    -X POST "$(base openai)/v1/chat/completions" -d '{ not json')
  [ "$st" -eq 400 ]
}

@test "unknown model -> 404 on each provider" {
  post_json openai /v1/chat/completions '{"model":"nope","messages":[{"role":"user","content":"hi"}]}'
  [ "$STATUS" -eq 404 ]
  validate openai.json ErrorResponse "$RESP"

  post_json anthropic /v1/messages '{"model":"nope","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}'
  [ "$STATUS" -eq 404 ]
  validate anthropic.json ErrorResponse "$RESP"

  post_json openrouter /chat/completions '{"model":"nope","messages":[{"role":"user","content":"hi"}]}'
  [ "$STATUS" -eq 404 ]
  validate openrouter.json InternalServerResponse "$RESP"
}

# ============================================================ security audit
# These drive the docs/api-security-audit.md invariants over the REAL socket, so a
# regression that reopens a hole fails CI. Everything below reuses the ONE mock server
# from setup_file except the admission-429 test, which needs a short admit deadline +
# a slow lane and so starts (and kills by PID) its own second server.

# --- 2. Input handling / DoS: an oversized body is rejected (413) before buffering.

@test "security: request body over the 8 MiB limit -> 413 (not buffered/OOM)" {
  # A ~9 MiB body: over MAX_BODY_BYTES (8 MiB), so DefaultBodyLimit rejects it with a
  # 413 before any handler buffers it. Built on disk with head/printf (no giant argv).
  local big="$CONF_DIR/big.json"
  { printf '{"model":"mock","messages":[{"role":"user","content":"'
    head -c 9000000 /dev/zero | tr '\0' 'a'
    printf '"}]}'
  } > "$big"
  local st
  st=$(curl -sS --max-time 20 -o /dev/null -w '%{http_code}' \
    -H "$(auth_hdr openai)" -H 'content-type: application/json' \
    -X POST "$(base openai)/v1/chat/completions" --data-binary @"$big")
  [ "$st" -eq 413 ]
}

# --- 1. Auth: auth wraps the fallback, so an unauthenticated caller cannot enumerate
# routes — an unknown path is a 401 (auth), never a 404 (route probe).

@test "security: unauthenticated unknown path -> 401 (no route enumeration)" {
  local st
  st=$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' "$(base openai)/v1/nope/does/not/exist")
  [ "$st" -eq 401 ]
  st=$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' "$(base anthropic)/v1/secret/admin")
  [ "$st" -eq 401 ]
  # WITH a valid key the same unknown path is a 404 — proving auth (not routing) is
  # what produced the 401 above.
  st=$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' -H "$(auth_hdr openai)" "$(base openai)/v1/nope/does/not/exist")
  [ "$st" -eq 404 ]
}

# --- 1. Auth: the key is never echoed in ANY response body (200 / 401 / 400 / 404).

@test "security: the API key never appears in any response body" {
  # 200 chat.
  post_json openai /v1/chat/completions '{"model":"mock","messages":[{"role":"user","content":"hello there"}]}'
  [ "$STATUS" -eq 200 ]
  ! grep -qF "$OPENAI_KEY" "$RESP"

  # 401 (no auth header).
  local b401="$CONF_DIR/b401.json"
  curl -sS --max-time 10 -o "$b401" "$(base openai)/v1/models"
  ! grep -qF "$OPENAI_KEY" "$b401"

  # 400 (malformed JSON).
  local b400="$CONF_DIR/b400.json"
  curl -sS --max-time 10 -o "$b400" -H "$(auth_hdr openai)" -H 'content-type: application/json' \
    -X POST "$(base openai)/v1/chat/completions" -d '{ not json' >/dev/null
  ! grep -qF "$OPENAI_KEY" "$b400"

  # 404 (unknown model).
  post_json openai /v1/chat/completions '{"model":"nope","messages":[{"role":"user","content":"hi"}]}'
  [ "$STATUS" -eq 404 ]
  ! grep -qF "$OPENAI_KEY" "$RESP"
}

# --- 2. Input handling: numeric/shape bounds are range-checked -> 400, never 500/panic.

@test "security: out-of-bounds inputs -> 400 (not 500/panic)" {
  # OpenAI chat n:5 (multiple choices unsupported).
  post_json openai /v1/chat/completions \
    '{"model":"mock","messages":[{"role":"user","content":"hi"}],"n":5}'
  [ "$STATUS" -eq 400 ]
  validate openai.json ErrorResponse "$RESP"

  # embeddings dimensions oversized (far beyond the model's vector length).
  post_json openai /v1/embeddings '{"model":"mock","input":"hello","dimensions":100000}'
  [ "$STATUS" -eq 400 ]
  validate openai.json ErrorResponse "$RESP"

  # embeddings token-array input (documented unsupported — brain has no tokenizer).
  post_json openai /v1/embeddings '{"model":"mock","input":[1,2,3,4]}'
  [ "$STATUS" -eq 400 ]
  validate openai.json ErrorResponse "$RESP"
}

# --- 5. Error hygiene: an internal action failure is scrubbed — the client gets only a
# generic, spec-shaped error, never the mock's internal path/panic detail.

@test "security: internal failure is scrubbed (no path/panic leak)" {
  post_json openai /v1/chat/completions \
    '{"model":"mock","messages":[{"role":"user","content":"please __mock_fail__ this"}]}'
  # A provider-shaped error (the handler maps the runtime failure to a 400).
  [ "$STATUS" -eq 400 ]
  validate openai.json ErrorResponse "$RESP"
  # None of the mock's fabricated internal detail may reach the client.
  ! grep -qiE 'secret|model\.gguf|/home/|panic' "$RESP"
  # ...only the generic message is present.
  [ "$(jq -r '.error.message' "$RESP")" = "the model failed to process the request" ]
}

# --- 3. Resource safety: a pinned single lane sheds a concurrent request with 429
# within the admit deadline (never an unbounded hang), carrying Retry-After.

@test "security: saturated lane sheds with 429 + Retry-After (own short-deadline server)" {
  local port="${SEC_PORT:-8899}"
  local keys2="$CONF_DIR/keys2.json"
  local log2="$CONF_DIR/serve2.log"

  # A SECOND server: one CPU lane, a 3s mock delay (pins the lane), a 500ms admit
  # deadline (fast shed). OpenAI surface only. Record its PID for teardown.
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu BRAIN_MOCK_DELAY_MS=3000 BRAIN_ADMIT_DEADLINE_MS=500 \
    "$BRAIN" serve --openai "$port" --api-keys-out "$keys2" >"$log2" 2>&1 &
  local pid2=$!
  echo "$pid2" > "$CONF_DIR/pid2"

  # Wait until /models answers (a route that does NOT touch the lane).
  local key2="" ready=0
  for _ in $(seq 1 60); do
    if [ -f "$keys2" ]; then
      key2=$(jq -r .openai "$keys2" 2>/dev/null)
      if [ -n "$key2" ] && [ "$key2" != "null" ] \
         && curl -fsS --max-time 5 -o /dev/null -H "Authorization: Bearer $key2" "http://127.0.0.1:$port/v1/models" 2>/dev/null; then
        ready=1
        break
      fi
    fi
    kill -0 "$pid2" 2>/dev/null || break
    sleep 0.5
  done
  if [ "$ready" != 1 ]; then
    cat "$log2" >&3 || true
    kill -9 "$pid2" 2>/dev/null || true
    skip "second (short-deadline) brain serve did not become ready"
  fi

  # Request 1 pins the single CPU lane for ~3s. Fire it in the background and let it
  # get ADMITTED (into its delay) before the probe.
  curl -sS --max-time 15 -o /dev/null \
    -H "Authorization: Bearer $key2" -H 'content-type: application/json' \
    -X POST "http://127.0.0.1:$port/v1/chat/completions" \
    -d '{"model":"mock","messages":[{"role":"user","content":"pin the lane"}]}' &
  local job1=$!
  sleep 0.8

  # Request 2: the lane is pinned, so it cannot be admitted within 500ms -> 429,
  # returning near the deadline (NOT waiting ~3s for the lane to free).
  local out="$CONF_DIR/sec_resp.json" hdrs="$CONF_DIR/sec_hdrs.txt"
  local t0 t1 ms st
  t0=$(date +%s%N)
  st=$(curl -sS --max-time 10 -D "$hdrs" -o "$out" -w '%{http_code}' \
    -H "Authorization: Bearer $key2" -H 'content-type: application/json' \
    -X POST "http://127.0.0.1:$port/v1/chat/completions" \
    -d '{"model":"mock","messages":[{"role":"user","content":"shed me"}]}')
  t1=$(date +%s%N)
  ms=$(( (t1 - t0) / 1000000 ))

  # Clean up: stop the pinning curl and the second server (by recorded PID only).
  kill "$job1" 2>/dev/null || true
  kill -9 "$pid2" 2>/dev/null || true
  rm -f "$CONF_DIR/pid2"

  [ "$st" -eq 429 ]
  [ "$ms" -lt 2000 ]                                   # shed near 500ms, did not hang ~3s
  grep -qi '^retry-after:[[:space:]]*[0-9]' "$hdrs"    # a numeric back-off hint
  validate openai.json ErrorResponse "$out"
}
