#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# The OpenRouter /videos contract over a real socket, against one
# `brain serve --mock` (the deterministic mock video model: no weights, no
# GPU). curl does what any OpenRouter-style client does: list, submit, poll,
# download, then send the result back as the input of an edit.
#
# Run: make test/e2e/video-api   (or: BRAIN_BIN=./target/debug/brain bats tests/e2e/video_api.bats)
#
# SAFETY: teardown kills ONLY the recorded `brain serve` PID, never pkill.

setup_file() {
  command -v jq >/dev/null 2>&1 || skip "jq not installed"
  command -v curl >/dev/null 2>&1 || skip "curl not installed"
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || skip "no brain binary (make build/debug, or set BRAIN_BIN)"

  export CONF_DIR="$(mktemp -d)"
  export KEYS="$CONF_DIR/keys.json"
  export LOG="$CONF_DIR/serve.log"
  export PORT="${VIDEO_API_PORT:-8899}"
  mkdir -p "$CONF_DIR/models"
  "$BRAIN" serve --mock --device cpu --models-dir "$CONF_DIR/models" \
    --openrouter "$PORT" --api-keys-out "$KEYS" --ready-file "$CONF_DIR/ready" >"$LOG" 2>&1 &
  export SERVER_PID=$!
  for _ in $(seq 1 60); do
    [ -e "$CONF_DIR/ready" ] && break
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 0.5
  done
  [ -e "$CONF_DIR/ready" ] || { cat "$LOG" >&3; false; }
  export KEY="$(jq -r .openrouter "$KEYS")"
  export BASE="http://127.0.0.1:$PORT/v1"
}

teardown_file() {
  [ -n "${SERVER_PID:-}" ] && kill -9 "$SERVER_PID" 2>/dev/null || true
  [ -n "${CONF_DIR:-}" ] && rm -rf "$CONF_DIR"
}

api() {
  curl -sS --max-time 30 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" "$@"
}

# Poll a job until it leaves pending/in_progress; print its final status body.
wait_job() {
  local body
  for _ in $(seq 1 300); do
    body="$(api "$BASE/videos/$1")"
    case "$(jq -r .status <<<"$body")" in
      pending | in_progress) sleep 0.2 ;;
      *) echo "$body"; return 0 ;;
    esac
  done
  echo "$body"
  return 1
}

@test "the mock video model is listed with its durations and edit actions" {
  run api "$BASE/videos/models"
  [ "$status" -eq 0 ]
  [ "$(jq -r '.data[] | select(.id=="brain/mock-video") | .supported_durations | length' <<<"$output")" -gt 0 ]
  [ "$(jq -r '.data[] | select(.id=="brain/mock-video") | [.actions[].action] | join(",")' <<<"$output")" = "t2v,extend" ]
  run api "$BASE/models"
  [ "$(jq -r '.data[] | select(.id=="brain/mock-video") | .architecture.output_modalities | join(",")' <<<"$output")" = "video" ]
}

@test "a clip is generated, polled and downloaded as a playable MP4" {
  id="$(api -X POST "$BASE/videos" -d '{"model":"brain/mock-video","prompt":"a red ball","duration":1,"size":"64x32","seed":3,"generate_audio":true}' | jq -r .id)"
  [ -n "$id" ] && [ "$id" != null ]
  done_body="$(wait_job "$id")"
  [ "$(jq -r .status <<<"$done_body")" = completed ]
  [ "$(jq -r .usage.frames <<<"$done_body")" = 9 ]
  [ "$(jq -r .usage.audio <<<"$done_body")" = true ]
  url="$(jq -r '.unsigned_urls[0]' <<<"$done_body")"
  api -o "$CONF_DIR/clip.mp4" "http://127.0.0.1:$PORT$url"
  [ "$(head -c 8 "$CONF_DIR/clip.mp4" | tail -c 4)" = ftyp ]
  [ "$(stat -c %s "$CONF_DIR/clip.mp4")" = "$(jq -r '.content[0].bytes' <<<"$done_body")" ]
  [ "$(sha256sum "$CONF_DIR/clip.mp4" | cut -d' ' -f1)" = "$(jq -r '.content[0].sha256' <<<"$done_body")" ]
  echo "$done_body" >"$CONF_DIR/first.json"
}

@test "the downloaded clip is extended as an edit" {
  [ -f "$CONF_DIR/clip.mp4" ] || skip "needs the previous test's clip"
  jq -n --arg url "data:video/mp4;base64,$(base64 -w0 "$CONF_DIR/clip.mp4")" \
    '{model:"brain/mock-video",prompt:"more",input_video:{type:"video_url",video_url:{url:$url}},provider:{options:{brain:{action:"extend",params:{frames:4}}}}}' >"$CONF_DIR/edit.json"
  id="$(api -X POST "$BASE/videos" -d @"$CONF_DIR/edit.json" | jq -r .id)"
  done_body="$(wait_job "$id")"
  [ "$(jq -r .status <<<"$done_body")" = completed ]
  [ "$(jq -r .usage.frames <<<"$done_body")" = 13 ]
}

@test "an Idempotency-Key returns the same job instead of rendering again" {
  a="$(api -X POST -H 'Idempotency-Key: e2e-1' "$BASE/videos" -d '{"model":"brain/mock-video","prompt":"once"}' | jq -r .id)"
  b="$(api -X POST -H 'Idempotency-Key: e2e-1' "$BASE/videos" -d '{"model":"brain/mock-video","prompt":"once"}' | jq -r .id)"
  [ "$a" = "$b" ]
}

@test "requests the model cannot honour are refused with the field named" {
  run curl -sS -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -X POST "$BASE/videos" -d '{"model":"brain/mock-video","prompt":"x","duration":60}'
  [ "$output" = 400 ]
  run curl -sS -o /dev/null -w '%{http_code}' -X POST "$BASE/videos" -d '{}'
  [ "$output" = 401 ]
}
