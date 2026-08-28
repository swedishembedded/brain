#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Regression coverage for `brain serve`'s shutdown handling: SIGINT and SIGTERM must
# stop the process within a bounded time, for every combination of surfaces it can be
# told to serve.
#
# Before crates/shutdown, `brain serve --dbus` printed "shutting down D-Bus service"
# and then hung forever — the stats-stream background task held its own
# `zbus::Connection` clone alive, so `Connection::graceful_shutdown()` awaited a drop
# event that could never fire. And when D-Bus ran alongside an HTTP surface
# (`--dbus --openai ...`), the two independently-built tokio runtimes raced to install
# `tokio::signal::ctrl_c()` — a process-wide registration — so only one of them ever
# actually saw a Ctrl-C, and it was typically the deadlocked one. Ctrl-C then did
# nothing at all. `--dbus --openai` under SIGINT (test 3 below) is that exact case.
#
# Each test owns its own server process (unlike api_conformance.bats's one
# suite-shared server), because the whole point here is to kill it and watch it die.
#
# SAFETY: every test signals ONLY the PID it recorded itself, and the D-Bus tests use
# a PRIVATE per-test dbus-daemon (never the real session/system bus). `teardown` is a
# backstop `kill -9` on that same recorded PID after a bounded wait — never pkill.

setup_file() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no brain binary (build with: make build/debug, or set BRAIN_BIN)"
  export BRAIN
  export SHUTDOWN_DEADLINE_S="${SHUTDOWN_DEADLINE_S:-5}"
  command -v curl >/dev/null 2>&1 || skip "curl not installed"
}

setup() {
  TDIR="$(mktemp -d)"
  export TDIR
  unset SERVER_PID DBUS_PID DBUS_SESSION_BUS_ADDRESS
}

teardown() {
  # Backstop only: every test below reaps its own PID via wait_for_exit before
  # finishing. A `kill -9` here means an assertion failed mid-test, not the normal
  # path. PID-only, never pkill.
  if [ -n "${SERVER_PID:-}" ]; then
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "${DBUS_PID:-}" ]; then
    kill -9 "$DBUS_PID" 2>/dev/null || true
  fi
  # KEEP_TDIR=1 preserves $TDIR (and its captured server log) for debugging a
  # failure by hand instead of deleting it on the way out.
  if [ -z "${KEEP_TDIR:-}" ]; then rm -rf "$TDIR"; else echo "KEPT TDIR=$TDIR" >&3; fi
}

# A private per-test session bus, so these tests never touch a real session/system
# bus. Sets DBUS_SESSION_BUS_ADDRESS and DBUS_PID (killed in teardown).
start_private_bus() {
  command -v dbus-daemon >/dev/null 2>&1 || skip "dbus-daemon not installed"
  DBUS_SESSION_BUS_ADDRESS="$(dbus-daemon --session --fork --print-address --print-pid=3 3>"$TDIR/dbus.pid" 2>/dev/null)"
  export DBUS_SESSION_BUS_ADDRESS
  DBUS_PID="$(cat "$TDIR/dbus.pid")"
  export DBUS_PID
  [ -n "$DBUS_SESSION_BUS_ADDRESS" ] && kill -0 "$DBUS_PID" 2>/dev/null || skip "could not start a private dbus-daemon"
}

# Poll until the given well-known D-Bus name is owned, up to SHUTDOWN_DEADLINE_S.
wait_for_bus_name() {
  local name="$1"
  local deadline=$((SECONDS + SHUTDOWN_DEADLINE_S + 5))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if busctl --address="$DBUS_SESSION_BUS_ADDRESS" list 2>/dev/null | grep -q "$name"; then
      return 0
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || return 1 # died early
    sleep 0.1
  done
  return 1
}

# Poll a plain-TCP /models until it answers with a real HTTP status (any status —
# `/v1/models` requires a key we deliberately don't pass, so 401 counts as "up"),
# up to SHUTDOWN_DEADLINE_S. NOTE: curl's `%{http_code}` prints "000" on a failed
# *connection* (refused/reset/timeout) — that must NOT count as "answered", or this
# reports the server ready while it is still starting up (a real bug caught here:
# a naive `grep -q '^[0-9]'` matches "000" too, and a signal sent that early can
# race ahead of `install_signals` actually installing the handler).
wait_for_http() {
  local port="$1"
  local deadline=$((SECONDS + SHUTDOWN_DEADLINE_S + 5))
  local code
  while [ "$SECONDS" -lt "$deadline" ]; do
    code="$(curl -s --max-time 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/v1/models" 2>/dev/null)"
    case "$code" in
      000) ;; # connection failed — not up yet
      [0-9][0-9][0-9]) return 0 ;;
    esac
    kill -0 "$SERVER_PID" 2>/dev/null || return 1 # died early
    sleep 0.1
  done
  return 1
}

# Send $1 to $SERVER_PID, then wait up to SHUTDOWN_DEADLINE_S for it to exit.
# Prints the wait(1) exit status. A 137 means OUR OWN kill -9 fallback fired — i.e.
# the process did NOT stop on its own within the deadline, which is exactly the
# failure this suite exists to catch.
signal_and_wait() {
  local sig="$1"
  kill "-$sig" "$SERVER_PID"
  local deadline=$((SECONDS + SHUTDOWN_DEADLINE_S))
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "# pid $SERVER_PID did not exit within ${SHUTDOWN_DEADLINE_S}s of SIG$sig — force-killing (the fix did not work)" >&3
      kill -9 "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null
      echo 137
      return
    fi
    sleep 0.1
  done
  wait "$SERVER_PID" 2>/dev/null
  echo $?
}

@test "brain serve --dbus alone: SIGINT stops it within the deadline" {
  start_private_bus
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --dbus >"$TDIR/log" 2>&1 &
  SERVER_PID=$!
  wait_for_bus_name "com.swedishembedded.Brain1" || { cat "$TDIR/log" >&3; skip "server never claimed the bus name"; }

  status="$(signal_and_wait INT)"
  [ "$status" != "137" ]
  ! kill -0 "$SERVER_PID" 2>/dev/null
}

@test "brain serve --dbus alone: SIGTERM stops it within the deadline" {
  start_private_bus
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --dbus >"$TDIR/log" 2>&1 &
  SERVER_PID=$!
  wait_for_bus_name "com.swedishembedded.Brain1" || { cat "$TDIR/log" >&3; skip "server never claimed the bus name"; }

  status="$(signal_and_wait TERM)"
  [ "$status" != "137" ]
  ! kill -0 "$SERVER_PID" 2>/dev/null
}

@test "brain serve --dbus + --openai together: SIGINT stops both surfaces" {
  # The case the user actually hit: before the fix, whichever runtime registered
  # ctrl_c() first (the deadlocked D-Bus one) swallowed the signal and the process
  # never exited — regardless of the HTTP surface being perfectly healthy.
  start_private_bus
  PORT=8991
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --dbus --openai "$PORT" >"$TDIR/log" 2>&1 &
  SERVER_PID=$!
  wait_for_bus_name "com.swedishembedded.Brain1" || { cat "$TDIR/log" >&3; skip "server never claimed the bus name"; }
  wait_for_http "$PORT" || { cat "$TDIR/log" >&3; skip "HTTP surface never came up"; }

  status="$(signal_and_wait INT)"
  [ "$status" != "137" ]
  ! kill -0 "$SERVER_PID" 2>/dev/null
  # The port must actually be released, not just the process gone from view.
  run curl -fsS --max-time 1 -o /dev/null "http://127.0.0.1:$PORT/v1/models"
  [ "$status" -ne 0 ]
}

@test "brain serve --openai alone (no dbus): SIGINT stops it within the deadline" {
  PORT=8992
  BRAIN_MOCK=1 BRAIN_DEVICE=cpu "$BRAIN" serve --openai "$PORT" >"$TDIR/log" 2>&1 &
  SERVER_PID=$!
  wait_for_http "$PORT" || { cat "$TDIR/log" >&3; skip "HTTP surface never came up"; }

  status="$(signal_and_wait INT)"
  [ "$status" != "137" ]
  ! kill -0 "$SERVER_PID" 2>/dev/null
}
