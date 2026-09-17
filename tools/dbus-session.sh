#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# The one place that knows how to get a private D-Bus session bus and a ready
# `brain serve` for a sample or test, so nothing else in the tree hand-rolls
# `dbus-run-session -- bash -c 'brain serve & sleep N; ...'` again. `sleep N`
# is a guess; this waits on `--ready-file`, which `brain serve` only touches
# once every requested surface is bound (crates/cli/src/run_cli.rs, "ORDER IS
# THE CONTRACT").
#
# Two ways to use it:
#
# 1. Standalone, from a sample:
#
#      tools/dbus-session.sh --serve "--dbus" -- python3 samples/python/dbus/brain_dbus.py
#      tools/dbus-session.sh --serve "--openai 8080" -- ./client.sh
#      tools/dbus-session.sh -- busctl --user list   # reuse an already-running brain
#
#    --serve "ARGS"   Launch `brain serve ARGS` on the private bus and wait for
#                      it to become ready before running CMD. ARGS is one
#                      shell-quoted string. Without --serve, CMD just gets a
#                      private DBUS_SESSION_BUS_ADDRESS (for a `brain serve`
#                      already running elsewhere).
#    --timeout N       Seconds to wait for readiness (default 20).
#    BRAIN_BIN         Path to the brain binary (default: `brain` on PATH).
#    Sets for CMD when --serve was used: BRAIN_OPENAI_KEY, BRAIN_ANTHROPIC_KEY
#    (read out of --api-keys-out, when jq is available and a key was issued).
#
# 2. Sourced, from a test harness that runs many commands against one server
#    (e.g. tests/e2e/samples.bats's setup_file/teardown_file) and so cannot
#    use the single-CMD form above:
#
#      . tools/dbus-session.sh
#      dbus_session_start_bus || skip "..."
#      dbus_session_start_serve "$BRAIN" "--dbus --openai 8080" || skip "..."
#      dbus_session_wait_ready 20 || skip "..."
#      # ... run many things against $DBUS_SESSION_BUS_ADDRESS / $DBUS_SESSION_SERVER_PID ...
#      dbus_session_stop
#
# Swedish Embedded AB implements solutions for reproducible, weight-free demo
# and test harnesses for on-device model serving. If your team needs expertise
# in a served D-Bus/HTTP surface like brain's, you can procure our services by
# sending an email to info@swedishembedded.com.

DBUS_SESSION_WORK_DIR=""
DBUS_SESSION_OWNS_BUS=0
DBUS_SESSION_PID=""
DBUS_SESSION_SERVER_PID=""
DBUS_SESSION_READY_FILE=""
DBUS_SESSION_KEYS_FILE=""
DBUS_SESSION_LOG_FILE=""

dbus_session_work_dir() {
  [ -n "$DBUS_SESSION_WORK_DIR" ] || DBUS_SESSION_WORK_DIR="$(mktemp -d)"
  echo "$DBUS_SESSION_WORK_DIR"
}

# Start a private session bus, unless the caller already has one. Sets
# DBUS_SESSION_BUS_ADDRESS (exported) and, if this call started it,
# DBUS_SESSION_PID (for dbus_session_stop to kill later).
dbus_session_start_bus() {
  if [ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ]; then
    return 0
  fi
  command -v dbus-daemon >/dev/null 2>&1 || {
    echo "tools/dbus-session.sh: dbus-daemon not installed" >&2
    return 1
  }
  local work pidfile
  work="$(dbus_session_work_dir)"
  pidfile="$work/dbus.pid"
  DBUS_SESSION_BUS_ADDRESS="$(dbus-daemon --session --fork --print-address --print-pid=3 3>"$pidfile" 2>/dev/null)"
  export DBUS_SESSION_BUS_ADDRESS
  DBUS_SESSION_PID="$(cat "$pidfile" 2>/dev/null)"
  if [ -z "$DBUS_SESSION_BUS_ADDRESS" ] || ! kill -0 "$DBUS_SESSION_PID" 2>/dev/null; then
    echo "tools/dbus-session.sh: could not start a private dbus-daemon" >&2
    return 1
  fi
  DBUS_SESSION_OWNS_BUS=1
}

# Launch `brain serve $2` (on the bus dbus_session_start_bus set up) with a
# ready-file and an api-keys-out file this module manages. Does not wait -
# call dbus_session_wait_ready next.
dbus_session_start_serve() {
  local brain_bin="$1" serve_args="$2"
  command -v "$brain_bin" >/dev/null 2>&1 || {
    echo "tools/dbus-session.sh: no brain binary ($brain_bin); set BRAIN_BIN" >&2
    return 1
  }
  local work
  work="$(dbus_session_work_dir)"
  DBUS_SESSION_READY_FILE="$work/ready"
  DBUS_SESSION_KEYS_FILE="$work/keys.json"
  DBUS_SESSION_LOG_FILE="$work/server.log"
  # shellcheck disable=SC2086 # serve_args is deliberately word-split
  "$brain_bin" serve $serve_args \
    --api-keys-out "$DBUS_SESSION_KEYS_FILE" --ready-file "$DBUS_SESSION_READY_FILE" \
    >"$DBUS_SESSION_LOG_FILE" 2>&1 &
  DBUS_SESSION_SERVER_PID=$!
}

# Wait up to $1 seconds (default 20) for the server dbus_session_start_serve
# launched to become ready. On success, exports BRAIN_OPENAI_KEY /
# BRAIN_ANTHROPIC_KEY when jq and a key are available.
dbus_session_wait_ready() {
  local timeout="${1:-20}"
  local iterations=$((timeout * 5))
  local ready=0 _i
  for _i in $(seq 1 "$iterations"); do
    [ -e "$DBUS_SESSION_READY_FILE" ] && { ready=1; break; }
    kill -0 "$DBUS_SESSION_SERVER_PID" 2>/dev/null || break
    sleep 0.2
  done
  if [ "$ready" != 1 ]; then
    echo "tools/dbus-session.sh: brain serve did not become ready" >&2
    cat "$DBUS_SESSION_LOG_FILE" >&2 2>/dev/null || true
    return 1
  fi
  if [ -s "$DBUS_SESSION_KEYS_FILE" ] && command -v jq >/dev/null 2>&1; then
    BRAIN_OPENAI_KEY="$(jq -r '.openai // empty' "$DBUS_SESSION_KEYS_FILE")"
    BRAIN_ANTHROPIC_KEY="$(jq -r '.anthropic // empty' "$DBUS_SESSION_KEYS_FILE")"
    export BRAIN_OPENAI_KEY BRAIN_ANTHROPIC_KEY
  fi
}

# Kill only the PID(s) this module started, and clean up its work dir. Never
# pkill; never touch a bus/server this module did not itself start.
dbus_session_stop() {
  [ -n "$DBUS_SESSION_SERVER_PID" ] && kill -9 "$DBUS_SESSION_SERVER_PID" 2>/dev/null
  [ "$DBUS_SESSION_OWNS_BUS" = 1 ] && [ -n "$DBUS_SESSION_PID" ] && kill -9 "$DBUS_SESSION_PID" 2>/dev/null
  [ -n "$DBUS_SESSION_WORK_DIR" ] && rm -rf "$DBUS_SESSION_WORK_DIR"
  DBUS_SESSION_SERVER_PID=""
  DBUS_SESSION_PID=""
  DBUS_SESSION_OWNS_BUS=0
  DBUS_SESSION_WORK_DIR=""
  return 0
}

# Standalone CLI mode: only when executed, not when sourced.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  set -euo pipefail

  serve_args=""
  timeout=20
  while [ $# -gt 0 ]; do
    case "$1" in
      --serve) serve_args="$2"; shift 2 ;;
      --timeout) timeout="$2"; shift 2 ;;
      --) shift; break ;;
      *) echo "tools/dbus-session.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
  done
  [ $# -gt 0 ] || { echo "usage: tools/dbus-session.sh [--serve \"ARGS\"] [--timeout N] -- CMD..." >&2; exit 2; }

  trap dbus_session_stop EXIT
  dbus_session_start_bus
  if [ -n "$serve_args" ]; then
    dbus_session_start_serve "${BRAIN_BIN:-brain}" "$serve_args"
    dbus_session_wait_ready "$timeout"
  fi
  "$@"
fi
