#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for putting one inference endpoint
# in front of every card of a cluster for its clients. If your team needs
# expertise in multi-node LLM serving then you can procure our services by
# sending an email to info@swedishembedded.com.

# End-to-end: `brain serve --upstream` routing over two real `brain serve --mock`
# processes on loopback. The router holds the upstream key, clients hold the
# router's; the listing is merged, a request is answered by an upstream, an
# upstream that stops is taken out of rotation and the other keeps serving, a
# drained upstream takes nothing new, and misconfiguration is refused at start.
#
# SAFETY: every test kills ONLY the PIDs it recorded itself.

setup_file() {
  command -v jq >/dev/null 2>&1 || skip "jq not installed"
  command -v curl >/dev/null 2>&1 || skip "curl not installed"
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no brain binary (build with: make build/debug, or set BRAIN_BIN)"
  export BRAIN
}

setup() {
  TDIR="$(mktemp -d)"
  chmod 700 "$TDIR"
  export TDIR
  mkdir -p "$TDIR/models"
  echo "sk-upstream-$RANDOM$RANDOM" >"$TDIR/upkey"
  chmod 600 "$TDIR/upkey"
  PIDS=()
}

teardown() {
  for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
  rm -rf "$TDIR"
}

# start_upstream N PORT: a mock server behind the upstream key, its own runtime dir.
start_upstream() {
  mkdir -p -m 700 "$TDIR/rt$1"
  "$BRAIN" serve --mock --device cpu --models-dir "$TDIR/models" --runtime-dir "$TDIR/rt$1" \
    --openai "127.0.0.1:$2" --api-key-file "$TDIR/upkey" --ready-file "$TDIR/ready$1" >"$TDIR/up$1.log" 2>&1 &
  PIDS+=($!)
  eval "UP$1_PID=$!"
}

wait_file() {
  for _ in $(seq 1 300); do [ -e "$1" ] && return 0; sleep 0.1; done
  return 1
}

start_router() {
  mkdir -p -m 700 "$TDIR/rtr"
  "$BRAIN" serve --runtime-dir "$TDIR/rtr" --openai "127.0.0.1:$1" \
    --upstream "http://127.0.0.1:$2,http://127.0.0.1:$3" --upstream-key-file "$TDIR/upkey" \
    --upstream-health-ms 200 --api-keys-out "$TDIR/keys.json" --ready-file "$TDIR/ready-router" >"$TDIR/router.log" 2>&1 &
  PIDS+=($!)
  ROUTER_PID=$!
  wait_file "$TDIR/ready-router" || { cat "$TDIR/router.log"; return 1; }
  KEY="$(jq -r .openai "$TDIR/keys.json")"
  ROUTER="http://127.0.0.1:$1"
}

# healthy_count: the upstreams the router has in rotation.
healthy_count() {
  curl -s -H "Authorization: Bearer $KEY" "$ROUTER/v1/health" | jq '[.upstreams[] | select(.healthy and (.draining|not))] | length'
}

wait_healthy() {
  for _ in $(seq 1 100); do [ "$(healthy_count)" = "$1" ] && return 0; sleep 0.1; done
  echo "router has $(healthy_count) upstreams in rotation, wanted $1"; return 1
}

chat() {
  curl -s -o "$TDIR/answer" -w '%{http_code}' -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
    "$ROUTER/v1/chat/completions" -d '{"model":"brain/mock","messages":[{"role":"user","content":"hello"}],"max_tokens":8}'
}

@test "one router key serves every upstream's models, and an upstream that stops is routed around" {
  start_upstream 1 18911
  start_upstream 2 18912
  wait_file "$TDIR/ready1" && wait_file "$TDIR/ready2"
  start_router 18910 18911 18912
  wait_healthy 2

  # Every route needs the router's key; the upstream key is not it.
  [ "$(curl -s -o /dev/null -w '%{http_code}' "$ROUTER/v1/models")" = 401 ]
  [ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(cat "$TDIR/upkey")" "$ROUTER/v1/models")" = 401 ]

  # The listing names each model once although both upstreams serve it.
  run bash -c "curl -s -H 'Authorization: Bearer $KEY' $ROUTER/v1/models | jq '[.data[].id | select(. == \"brain/mock\")] | length'"
  [ "$output" = 1 ]

  [ "$(chat)" = 200 ]
  jq -e '.choices[0].message.content | test("hello")' "$TDIR/answer"

  # Streamed through as server-sent events.
  run curl -sN -H "Authorization: Bearer $KEY" -H 'content-type: application/json' "$ROUTER/v1/chat/completions" \
    -d '{"model":"brain/mock","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"stream":true}'
  [[ "$output" == *"data: [DONE]"* ]]

  # Drained: out of rotation for new requests, back on resume.
  [ "$(curl -s -o /dev/null -w '%{http_code}' -X POST -H "Authorization: Bearer $KEY" "$ROUTER/v1/upstreams/0/drain")" = 200 ]
  wait_healthy 1
  [ "$(curl -s -o /dev/null -w '%{http_code}' -X POST -H "Authorization: Bearer $KEY" "$ROUTER/v1/upstreams/0/resume")" = 200 ]
  wait_healthy 2

  # One upstream stops: the router takes it out and the other answers.
  kill -TERM "$UP1_PID"
  wait "$UP1_PID" || true
  for _ in 1 2 3 4; do [ "$(chat)" = 200 ]; done
  wait_healthy 1

  # Both gone: refused at once with Retry-After, never queued.
  kill -TERM "$UP2_PID"
  wait "$UP2_PID" || true
  wait_healthy 0
  run curl -s -D - -o /dev/null -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
    "$ROUTER/v1/chat/completions" -d '{"model":"brain/mock","messages":[]}'
  [[ "$output" == *" 503"* ]]
  [[ "${output,,}" == *"retry-after:"* ]]

  kill -TERM "$ROUTER_PID"
  wait "$ROUTER_PID"
}

@test "a router refuses a plaintext upstream off this machine and model flags it would ignore" {
  run "$BRAIN" serve --runtime-dir "$TDIR" --openai 127.0.0.1:18913 --upstream http://192.0.2.1:8000
  [ "$status" -eq 2 ]
  [[ "$output" == *"--insecure-plaintext"* ]]
  run "$BRAIN" serve --runtime-dir "$TDIR" --openai 127.0.0.1:18913 --upstream http://127.0.0.1:18914 --mock
  [ "$status" -eq 2 ]
  [[ "$output" == *"--mock"* ]]
  run "$BRAIN" serve --runtime-dir "$TDIR" --openai 127.0.0.1:18913 --upstream http://127.0.0.1:18914/v1
  [ "$status" -eq 2 ]
  run "$BRAIN" serve --device cpu --runtime-dir "$TDIR" --openai 127.0.0.1:18913 --upstream-key-file "$TDIR/upkey"
  [ "$status" -eq 2 ]
}
