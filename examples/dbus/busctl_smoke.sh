#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Smoke-test brain's D-Bus control surface with systemd's `busctl`.
#
# Run it inside a private session bus so it needs no system config:
#
#     dbus-run-session -- bash examples/dbus/busctl_smoke.sh [path/to/brain]
#
# It starts `brain serve --dbus`, then introspects the interface, reads properties,
# lists models, fetches the manifests, and calls Run on the always-available `demo`
# model (which returns its result as a file descriptor). `busctl` is ideal for
# validating the surface + reply signatures; a real client (see brain_dbus.py) is
# needed to actually consume returned fds.
#
# Set BRAIN_DBUS_EXTERNAL=1 to reuse an ALREADY-RUNNING `brain serve --dbus` on
# this bus instead of starting (and later killing) one of our own — a single
# well-known bus name can only have one owner, so this is what lets
# tests/e2e/examples.bats run this script against its own shared server rather
# than racing a second `brain serve --dbus` for ownership of the name.
set -euo pipefail

BIN="${1:-target/debug/brain}"
BUS=com.swedishembedded.Brain1
OBJ=/com/swedishembedded/Brain1
IFACE=com.swedishembedded.Brain1.Manager

if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
  echo "no session bus — run me under:  dbus-run-session -- bash $0" >&2
  exit 1
fi

if [[ "${BRAIN_DBUS_EXTERNAL:-0}" == "1" ]]; then
  echo "== BRAIN_DBUS_EXTERNAL=1: reusing the already-running server =="
else
  echo "== starting: $BIN serve --dbus =="
  "$BIN" serve --dbus &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null || true' EXIT
fi

# Wait for the well-known name to appear on the bus.
for _ in $(seq 1 50); do
  if busctl --user list | grep -q "$BUS"; then break; fi
  sleep 0.2
done

echo; echo "== introspect =="
busctl --user introspect "$BUS" "$OBJ" | { grep -E "Manager|method|property" || true; }

echo; echo "== properties =="
echo -n "Version:    "; busctl --user get-property "$BUS" "$OBJ" "$IFACE" Version
echo -n "ActiveJobs: "; busctl --user get-property "$BUS" "$OBJ" "$IFACE" ActiveJobs
echo -n "Models:     "; busctl --user get-property "$BUS" "$OBJ" "$IFACE" Models

echo; echo "== ListModels =="
busctl --user call "$BUS" "$OBJ" "$IFACE" ListModels

echo; echo "== Manifests (truncated) =="
MANIFESTS=$(busctl --user call "$BUS" "$OBJ" "$IFACE" Manifests)
echo "${MANIFESTS:0:360} ..."

echo; echo "== Run demo.echo (result returned as an fd) =="
# Run(model s, action s, params s, in_fds a{sh}, in_meta s, transport s)
#      -> (result s, out_fds a{sh}, out_meta s)
busctl --user call "$BUS" "$OBJ" "$IFACE" Run sssa{sh}ss \
  demo echo '{"text":"brain-over-dbus ","times":3,"mode":"upper"}' 0 '' memfd

echo; echo "== Cancel on a bogus job id (expect: b false) =="
CANCELLED=$(busctl --user call "$BUS" "$OBJ" "$IFACE" Cancel t 999999999)
echo "$CANCELLED"
[[ "$CANCELLED" == "b false" ]] || { echo "Cancel(bogus) returned '$CANCELLED', want 'b false'" >&2; exit 1; }

echo; echo "== OK: surface + FD-returning Run + Cancel validated =="
