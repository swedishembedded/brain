#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Keep exactly one development brain server running, on a private session bus,
# and restart it when the binary it is running has been rebuilt.
#
# This is a development-loop helper, deliberately NOT a feature of the server:
# a daemon that watches its own binary and re-execs has to decide what happens
# to in-flight jobs, resident weights and GPU allocations mid-flight. That
# complexity does not belong in production code to serve a build loop.
#
#   eval "$(scripts/dev/brain-dev-serve.sh env)"   # ensure running, export the bus
#   scripts/dev/brain-dev-serve.sh status
#   scripts/dev/brain-dev-serve.sh stop
#
# `up` (the default) is idempotent and cheap: if the recorded pid is alive, the
# bus is alive, and the binary's mtime still matches the one that pid was
# started from, it does nothing at all. Otherwise it restarts. That is the
# whole "reload on rebuild" mechanism.

set -u

cd "$(dirname "$0")/../.."

BIN="${BRAIN_BIN:-./target/release/brain}"
RUN_DIR="${BRAIN_DEV_RUN_DIR:-${XDG_RUNTIME_DIR:-/tmp}/brain-dev}"
READY_TIMEOUT="${BRAIN_DEV_READY_TIMEOUT:-180}"

BUS_ADDR_FILE="$RUN_DIR/bus.address"
BUS_PID_FILE="$RUN_DIR/bus.pid"
PID_FILE="$RUN_DIR/brain.pid"
STAMP_FILE="$RUN_DIR/brain.stamp"
READY_FILE="$RUN_DIR/brain.ready"
LOG_FILE="$RUN_DIR/brain.log"

mkdir -p "$RUN_DIR"

alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }
read_file_or_empty() { [ -f "$1" ] && cat "$1" || true; }

# The identity of the running build. mtime alone is enough here and costs
# nothing; a content hash would be more precise but every rebuild touches
# mtime anyway, and this runs on every test invocation.
binary_stamp() {
    [ -f "$BIN" ] || return 1
    stat -c %Y "$BIN" 2>/dev/null || stat -f %m "$BIN" 2>/dev/null
}

bus_pid()   { read_file_or_empty "$BUS_PID_FILE"; }
brain_pid() { read_file_or_empty "$PID_FILE"; }

start_bus() {
    alive "$(bus_pid)" && return 0
    # --print-address on fd 1, daemonised, so the address outlives this shell.
    local out
    out=$(dbus-daemon --session --fork --print-address=1 --print-pid=1) || return 1
    printf '%s\n' "$out" | sed -n '1p' > "$BUS_ADDR_FILE"
    printf '%s\n' "$out" | sed -n '2p' > "$BUS_PID_FILE"
}

stop_brain() {
    local pid; pid=$(brain_pid)
    if alive "$pid"; then
        kill "$pid" 2>/dev/null
        for _ in $(seq 1 50); do alive "$pid" || break; sleep 0.1; done
        alive "$pid" && kill -9 "$pid" 2>/dev/null
    fi
    rm -f "$PID_FILE" "$STAMP_FILE"
}

# Refuse to be the second brain on this machine. Only ever reports a server we
# did not start; our own is matched by pid, never by pattern, so a restart
# cannot take out somebody else's session.
foreign_server() {
    local mine; mine=$(brain_pid)
    for p in $(pgrep -f 'brain serve' 2>/dev/null); do
        [ "$p" = "${mine:-0}" ] || { echo "$p"; return 0; }
    done
    return 1
}

up() {
    local stamp; stamp=$(binary_stamp) || { echo "no brain binary at $BIN (cargo build --release --bin brain)" >&2; exit 1; }

    if alive "$(brain_pid)" && alive "$(bus_pid)" && [ "$(read_file_or_empty "$STAMP_FILE")" = "$stamp" ]; then
        return 0
    fi

    local foreign; foreign=$(foreign_server) && {
        echo "another 'brain serve' is already running (pid $foreign); stop it first" >&2
        exit 1
    }

    stop_brain
    start_bus || { echo "could not start a session bus" >&2; exit 1; }

    rm -f "$READY_FILE"
    DBUS_SESSION_BUS_ADDRESS="$(cat "$BUS_ADDR_FILE")" \
        nohup "$BIN" serve --dbus --ready-file "$READY_FILE" >"$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"

    # --ready-file appears only once every requested surface is listening, so
    # this waits on one file instead of polling the bus. It is never created if
    # a surface fails, hence the liveness check in the same loop.
    local waited=0
    while [ ! -f "$READY_FILE" ]; do
        alive "$pid" || { echo "brain exited during startup; see $LOG_FILE" >&2; tail -n 20 "$LOG_FILE" >&2; exit 1; }
        sleep 0.2
        waited=$((waited + 1))
        [ "$waited" -gt $((READY_TIMEOUT * 5)) ] && { echo "brain did not become ready in ${READY_TIMEOUT}s; see $LOG_FILE" >&2; exit 1; }
    done

    echo "$stamp" > "$STAMP_FILE"
}

case "${1:-up}" in
    up)     up ;;
    env)    up; echo "export DBUS_SESSION_BUS_ADDRESS='$(cat "$BUS_ADDR_FILE")'" ;;
    status)
        if alive "$(brain_pid)"; then
            echo "brain running (pid $(brain_pid), bus $(read_file_or_empty "$BUS_ADDR_FILE"))"
            [ "$(read_file_or_empty "$STAMP_FILE")" = "$(binary_stamp || echo none)" ] || echo "  binary has been rebuilt since; next 'up' restarts it"
        else
            echo "brain not running"
        fi
        ;;
    stop)
        stop_brain
        local_bus=$(bus_pid); alive "$local_bus" && kill "$local_bus" 2>/dev/null
        rm -f "$BUS_PID_FILE" "$BUS_ADDR_FILE"
        ;;
    *) echo "usage: $0 [up|env|status|stop]" >&2; exit 2 ;;
esac
