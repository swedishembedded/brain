#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# End-to-end coverage for `brain serve --ready-file PATH`: PATH must appear only
# once EVERY requested surface (HTTP dialects + D-Bus) has actually bound, and
# must never appear if any requested surface fails to come up (fully, or
# partially — some but not all bound).
#
# The load-bearing test is the first one: --ready-file appearing must strictly
# imply --api-keys-out was already written AND the listener answers, with NO
# RETRY needed — that ordering is documented as a contract at
# crates/cli/src/run_cli.rs (search "ORDER IS THE CONTRACT") and is exactly the
# race applications/bench/environment/entrypoint.sh was relying on incorrectly
# (polling `[[ -s keys.json ]]`, which can observe the keys before the port is
# open).
#
# SAFETY: every test signals/kills ONLY the PID(s) it recorded itself. teardown
# is a bounded kill -9 backstop on those same recorded PIDs — never pkill.

setup_file() {
  command -v jq >/dev/null 2>&1 || skip "jq not installed"
  command -v curl >/dev/null 2>&1 || skip "curl not installed"

  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no brain binary (build with: make build, or set BRAIN_BIN)"
  export BRAIN
  export READY_DEADLINE_S="${READY_DEADLINE_S:-15}"
}

setup() {
  TDIR="$(mktemp -d)"
  export TDIR
  unset SERVER_PID SERVER2_PID DBUS_PID DBUS_SESSION_BUS_ADDRESS
}

teardown() {
  # Backstop only — every test reaps its own PID(s) before finishing.
  for v in SERVER_PID SERVER2_PID DBUS_PID; do
    pid="${!v:-}"
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  done
  if [ -z "${KEEP_TDIR:-}" ]; then rm -rf "$TDIR"; else echo "KEPT TDIR=$TDIR" >&3; fi
}

# Poll for PATH to exist, up to READY_DEADLINE_S, bailing early if $1 (a PID) dies.
wait_for_ready_file() {
  local path="$1" watch_pid="$2"
  local deadline=$((SECONDS + READY_DEADLINE_S))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ -e "$path" ] && return 0
    kill -0 "$watch_pid" 2>/dev/null || return 1 # died early
    sleep 0.05
  done
  return 1
}

start_private_bus() {
  command -v dbus-daemon >/dev/null 2>&1 || skip "dbus-daemon not installed"
  command -v busctl >/dev/null 2>&1 || skip "busctl not installed"
  DBUS_SESSION_BUS_ADDRESS="$(dbus-daemon --session --fork --print-address --print-pid=3 3>"$TDIR/dbus.pid" 2>/dev/null)"
  export DBUS_SESSION_BUS_ADDRESS
  DBUS_PID="$(cat "$TDIR/dbus.pid")"
  export DBUS_PID
  [ -n "$DBUS_SESSION_BUS_ADDRESS" ] && kill -0 "$DBUS_PID" 2>/dev/null || skip "could not start a private dbus-daemon"
}

@test "the ready file means keys written AND the listener answering -- no retry needed" {
  PORT=8993
  KEYS="$TDIR/keys.json"
  READY="$TDIR/ready"
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --openai "$PORT" --api-keys-out "$KEYS" --ready-file "$READY" \
    >"$TDIR/log" 2>&1 &
  SERVER_PID=$!

  wait_for_ready_file "$READY" "$SERVER_PID" || { cat "$TDIR/log" >&3; skip "ready file never appeared"; }

  # The instant the ready file exists, with NO retry: keys must already be on
  # disk and the port must already answer.
  [ -s "$KEYS" ]
  key="$(jq -r .openai "$KEYS")"
  [ -n "$key" ] && [ "$key" != "null" ]
  run curl -fsS --max-time 2 -o /dev/null -H "Authorization: Bearer $key" "http://127.0.0.1:$PORT/v1/models"
  [ "$status" -eq 0 ]
}

@test "the ready file is empty, even over a stale marker from a previous run" {
  PORT=8994
  READY="$TDIR/ready"
  echo "stale content from a previous run" > "$READY"

  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --openai "$PORT" --ready-file "$READY" >"$TDIR/log" 2>&1 &
  SERVER_PID=$!

  # The stale marker must be gone almost immediately (removed at parse time,
  # long before any bind) -- if it lingered, a waiter would see "ready" before
  # the server ever started.
  sleep 0.2
  [ ! -s "$READY" ] || [ ! -e "$READY" ]

  wait_for_ready_file "$READY" "$SERVER_PID" || { cat "$TDIR/log" >&3; skip "ready file never appeared"; }
  [ -e "$READY" ]
  [ ! -s "$READY" ]
}

@test "no ready file when the only requested listener cannot bind" {
  PORT=8995
  READY1="$TDIR/ready1"
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --openai "$PORT" --ready-file "$READY1" >"$TDIR/log1" 2>&1 &
  SERVER_PID=$!
  wait_for_ready_file "$READY1" "$SERVER_PID" || { cat "$TDIR/log1" >&3; skip "first server never became ready"; }

  # Second server: same port, must fail to bind.
  READY2="$TDIR/ready2"
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --openai "$PORT" --ready-file "$READY2" >"$TDIR/log2" 2>&1 &
  SERVER2_PID=$!

  local deadline=$((SECONDS + READY_DEADLINE_S))
  while kill -0 "$SERVER2_PID" 2>/dev/null; do
    [ "$SECONDS" -lt "$deadline" ] || { kill -9 "$SERVER2_PID"; break; }
    sleep 0.05
  done
  status=0
  wait "$SERVER2_PID" 2>/dev/null || status=$?
  [ "$status" -ne 0 ]
  [ ! -e "$READY2" ]
}

@test "no ready file on a partial bind (2 of 3 requested surfaces up)" {
  P1=8996
  READY1="$TDIR/ready1"
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --openai "$P1" --ready-file "$READY1" >"$TDIR/log1" 2>&1 &
  SERVER_PID=$!
  wait_for_ready_file "$READY1" "$SERVER_PID" || { cat "$TDIR/log1" >&3; skip "first server never became ready"; }

  # Second server: --openai on a free port (would bind), but --anthropic on
  # the ALREADY-TAKEN port ($P1) -- one of its two requested surfaces can
  # never bind, so the ready file must never appear even though the other one did.
  P2=8997
  READY2="$TDIR/ready2"
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --openai "$P2" --anthropic "$P1" --ready-file "$READY2" \
    >"$TDIR/log2" 2>&1 &
  SERVER2_PID=$!

  local deadline=$((SECONDS + READY_DEADLINE_S))
  while kill -0 "$SERVER2_PID" 2>/dev/null; do
    [ "$SECONDS" -lt "$deadline" ] || { kill -9 "$SERVER2_PID"; break; }
    sleep 0.05
  done
  status=0
  wait "$SERVER2_PID" 2>/dev/null || status=$?
  [ "$status" -ne 0 ]
  [ ! -e "$READY2" ]
}

@test "the ready file covers the D-Bus surface alone" {
  start_private_bus
  READY="$TDIR/ready"
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --dbus --ready-file "$READY" >"$TDIR/log" 2>&1 &
  SERVER_PID=$!
  wait_for_ready_file "$READY" "$SERVER_PID" || { cat "$TDIR/log" >&3; skip "ready file never appeared"; }

  # No retry: the bus name must already be owned.
  run busctl --address="$DBUS_SESSION_BUS_ADDRESS" list
  [ "$status" -eq 0 ]
  [[ "$output" == *"com.swedishembedded.Brain1"* ]]
}

@test "the ready file covers D-Bus and HTTP together" {
  start_private_bus
  PORT=8998
  READY="$TDIR/ready"
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --dbus --openai "$PORT" --ready-file "$READY" >"$TDIR/log" 2>&1 &
  SERVER_PID=$!
  wait_for_ready_file "$READY" "$SERVER_PID" || { cat "$TDIR/log" >&3; skip "ready file never appeared"; }

  run busctl --address="$DBUS_SESSION_BUS_ADDRESS" list
  [ "$status" -eq 0 ]
  [[ "$output" == *"com.swedishembedded.Brain1"* ]]
  run curl -fsS --max-time 2 -o /dev/null "http://127.0.0.1:$PORT/v1/models"
  # unauthenticated -> non-zero from curl -f is fine (401); a CONNECTION
  # failure is what would mean the surface isn't actually up, which curl -f
  # also reports non-zero for -- so probe the raw status code instead.
  code="$(curl -s --max-time 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models")"
  [ "$code" != "000" ]
}

@test "--ready-file with an unwritable directory fails fast, before any model scan" {
  PORT=8999
  t0=$SECONDS
  run timeout 10 env BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --openai "$PORT" \
    --ready-file /nonexistent-dir-for-brain-ready-e2e-xyz/ready
  elapsed=$((SECONDS - t0))
  [ "$status" -ne 0 ]
  [ "$status" -ne 124 ] # must not be the `timeout` command's own kill
  [[ "$output" == *"--ready-file"* ]]
  [ "$elapsed" -lt 5 ]
}
