#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

# Regression harness for everything under examples/: nothing ran any of them
# before this file existed, which is exactly why they all silently rotted after
# the P19 brain-py API rewrite (see the git history around commit 38f384e).
#
# ONE shared server for the whole suite (BRAIN_MOCK=1, D-Bus + Anthropic HTTP, on
# a private per-run dbus-daemon — never the real session/system bus), matching
# tests/e2e/api_conformance.bats's pattern. Each example that CAN run against the
# weight-free mock model does so for real; the rest skip (exit 77, mapped to a
# bats `skip` — see brain_py.base.skip) with the reason printed. The final test
# below is what keeps this harness honest: every tracked example must appear in
# examples/manifest.tsv and vice versa, so a new, unwired example fails the suite
# instead of quietly rotting the way these all did.
#
# SAFETY: the server is started once in setup_file and its PID recorded; teardown_file
# kills ONLY that recorded PID (and the private dbus-daemon's) — never pkill.

setup_file() {
  command -v dbus-daemon >/dev/null 2>&1 || skip "dbus-daemon not installed"
  command -v curl >/dev/null 2>&1 || skip "curl not installed"
  command -v jq >/dev/null 2>&1 || skip "jq not installed"
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || BRAIN="$REPO/target/release/brain"
  [ -x "$BRAIN" ] || skip "no brain binary (build with: make build, or set BRAIN_BIN)"
  export BRAIN
  PY="${EXAMPLES_PY:-python3}"
  export PY
  "$PY" -c "import jeepney" >/dev/null 2>&1 || skip "python jeepney not installed (pip install -e brain-py)"
  export PYTHONPATH="$REPO/brain-py"

  CONF_DIR="$(mktemp -d)"
  export CONF_DIR
  OUT="$CONF_DIR/out"
  mkdir -p "$OUT"
  export OUT

  DBUS_SESSION_BUS_ADDRESS="$(dbus-daemon --session --fork --print-address --print-pid=3 3>"$CONF_DIR/dbus.pid" 2>/dev/null)"
  export DBUS_SESSION_BUS_ADDRESS
  DBUS_PID="$(cat "$CONF_DIR/dbus.pid")"
  export DBUS_PID
  [ -n "$DBUS_SESSION_BUS_ADDRESS" ] && kill -0 "$DBUS_PID" 2>/dev/null || skip "could not start a private dbus-daemon"

  ANTHROPIC_PORT="${ANTHROPIC_PORT:-8991}"
  export ANTHROPIC_PORT
  OPENAI_PORT="${OPENAI_PORT:-8992}"
  export OPENAI_PORT
  KEYS="$CONF_DIR/keys.json"
  export KEYS
  READY="$CONF_DIR/ready"
  export READY
  # BRAIN_MOCK_DELAY_MS is a SERVER-side knob (crates/cli/src/resident_mock.rs
  # reads it once per request from ITS OWN environment) — it must be set on this
  # launch, not on a client invocation later, or the cancellation test below
  # races every step to completion before Cancel can land. 300ms split across
  # text2image's 4 steps is unnoticeable for every other test here.
  BRAIN_MOCK=1 BRAIN_MOCK_DELAY_MS=300 BRAIN_DEVICE=cpu "$BRAIN" serve \
    --dbus --anthropic "$ANTHROPIC_PORT" --openai "$OPENAI_PORT" \
    --api-keys-out "$KEYS" --ready-file "$READY" \
    >"$CONF_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  export SERVER_PID
  echo "$SERVER_PID" > "$CONF_DIR/pid"

  # Wait on the ready file rather than polling D-Bus directly: run_cli.rs's
  # "ORDER IS THE CONTRACT" guarantee means the ready file is touched only
  # once every requested surface (D-Bus included) is bound AND strictly after
  # --api-keys-out is written, so a single wait covers all three -- no retry
  # loop needed for the key read below.
  local ready=0
  for _ in $(seq 1 60); do
    [ -e "$READY" ] && { ready=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 0.3
  done
  if [ "$ready" != 1 ]; then
    echo "--- brain serve log ---" >&3
    cat "$CONF_DIR/server.log" >&3 || true
    skip "brain serve did not become ready"
  fi

  OPENAI_KEY="$(jq -r .openai "$KEYS" 2>/dev/null)"
  export OPENAI_KEY
  ANTHROPIC_KEY="$(jq -r .anthropic "$KEYS" 2>/dev/null)"
  export ANTHROPIC_KEY
}

teardown_file() {
  # Kill ONLY the recorded PIDs — never pkill.
  if [ -n "${SERVER_PID:-}" ]; then
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  if [ -n "${DBUS_PID:-}" ]; then
    kill -9 "$DBUS_PID" 2>/dev/null || true
  fi
  [ -n "${CONF_DIR:-}" ] && rm -rf "$CONF_DIR"
}

# Run a Python example; exit 77 becomes a bats skip (with the printed reason),
# any other non-zero is a real failure. `skip` inside this helper still ends the
# enclosing @test — bats' skip() exits the whole test body, not just this call.
run_example() {
  run "$PY" "$@"
  if [ "$status" -eq 77 ]; then
    skip "$(echo "$output" | grep '^SKIP:' | tail -1)"
  fi
  if [ "$status" -ne 0 ]; then
    echo "$output" >&3
  fi
  [ "$status" -eq 0 ]
}

# ------------------------------------------------------------- dbus/

@test "examples/dbus/brain_dbus.py runs against the mock (imageops path)" {
  run_example "$REPO/examples/dbus/brain_dbus.py"
  [[ "$output" == *"imageops.gradient"* ]]
}

@test "examples/dbus/busctl_smoke.sh runs against the shared harness server" {
  run env BRAIN_DBUS_EXTERNAL=1 bash "$REPO/examples/dbus/busctl_smoke.sh" "$BRAIN"
  if [ "$status" -ne 0 ]; then
    echo "$output" >&3
  fi
  [ "$status" -eq 0 ]
  [[ "$output" == *"OK: surface + FD-returning Run + Cancel validated"* ]]
}

@test "examples/dbus/detect_pipeline.py skips cleanly without z-image+yolo weights" {
  run_example "$REPO/examples/dbus/detect_pipeline.py"
}

# ------------------------------------------------------------- embedding/

@test "examples/embedding/embed_document.py runs against the mock" {
  run_example "$REPO/examples/embedding/embed_document.py" --input "$REPO/README.md" --model brain/mock
  [[ "$output" == *"tokens x 8 dim"* ]]
}

# ------------------------------------------------------------- forecast/

@test "examples/forecast/forecast_client.py runs against the mock" {
  run_example "$REPO/examples/forecast/forecast_client.py" --model brain/mock --horizon 8
  [[ "$output" == *"kind=quantiles"* ]]
}

# ------------------------------------------------------------- omni/

@test "examples/omni/omni.py runs against the mock over D-Bus" {
  run_example "$REPO/examples/omni/omni.py" --dbus --model brain/mock --in-text "hi" --out-stdio
  [[ "$output" == *"You said: hi"* ]]
}

@test "examples/omni/omni.py skips cleanly on unimplemented input/output flags" {
  run_example "$REPO/examples/omni/omni.py" --dbus --model brain/mock --in-image foo.png --in-text ignored --out-stdio
}

@test "examples/omni/omni.py runs against the mock over OpenAI-compatible HTTP" {
  # brain.models() is real on this transport, so the served-model precheck
  # (omni.py:144-153) runs before the generate call -- brain/mock is served,
  # so this exercises the full precheck-then-generate path, not just generate.
  run_example "$REPO/examples/omni/omni.py" \
    --openai "127.0.0.1:$OPENAI_PORT" --api-key "$OPENAI_KEY" \
    --model brain/mock --in-text "hi" --out-stdio
  [[ "$output" == *"You said: hi"* ]]
}

@test "examples/omni/omni.py runs against the mock over Anthropic-compatible HTTP" {
  # BrainAnthropic.manifests() raises NotImplementedError (Anthropic's API has
  # no /v1/models equivalent), so omni.py:148-151 catches it and skips the
  # served-model precheck entirely on this transport -- an unserved model
  # would fail at generate time, not as a skip(). brain/mock IS served here,
  # so this is still the real generate path, just without the precheck step
  # the OpenAI-transport test above exercises.
  run_example "$REPO/examples/omni/omni.py" \
    --anthropic "127.0.0.1:$ANTHROPIC_PORT" --api-key "$ANTHROPIC_KEY" \
    --model brain/mock --in-text "hi" --out-stdio
  [[ "$output" == *"You said: hi"* ]]
}

# ------------------------------------------------------------- imagegen/

@test "examples/imagegen/generate.py runs against the mock" {
  run_example "$REPO/examples/imagegen/generate.py" --model brain/mock --prompt test --width 8 --height 8 --out "$OUT/mock.ppm"
  [ -f "$OUT/mock.ppm" ]
}

@test "examples/imagegen/edit_image.py skips cleanly without FLUX.2 weights (mock has no edit action)" {
  run_example "$REPO/examples/imagegen/edit_image.py" --image "$OUT/mock.ppm" --prompt test --model brain/flux2-klein
}

@test "examples/imagegen/lora_finetune.py skips cleanly without FLUX.2 weights" {
  run_example "$REPO/examples/imagegen/lora_finetune.py" --data /nonexistent --save /nonexistent/out.lora --model brain/flux2-klein
}

@test "examples/imagegen/cancel_generation.py actually cancels a mock job" {
  # The shared server was started with BRAIN_MOCK_DELAY_MS=300 (see setup_file)
  # so there is real time to call Cancel between the first two progress frames.
  run_example "$REPO/examples/imagegen/cancel_generation.py" --model brain/mock
  [[ "$output" == *"'cancelled' (expected)"* ]]
}

# ------------------------------------------------------------- asr/

@test "examples/asr/bench_streams.py skips cleanly without real ASR weights" {
  run env "$PY" "$REPO/examples/asr/bench_streams.py" --model brain/nemotron --wav /dev/null --streams 1
  # argparse's own --wav validation may fire before the model check; either a
  # clean skip (77) or a clean argument error is acceptable — a hang or a Python
  # traceback is not.
  [ "$status" -eq 77 ] || [ "$status" -eq 1 ] || [ "$status" -eq 2 ]
}

@test "examples/asr/transcribe_mic.py skips cleanly without real ASR weights" {
  run "$PY" "$REPO/examples/asr/transcribe_mic.py" --model brain/nemotron --wav /dev/null
  [ "$status" -eq 77 ]
}

# ------------------------------------------------------------- api/

@test "examples/api/claude-with-brain.sh --check runs against the mock" {
  run env BRAIN="$BRAIN" BRAIN_MOCK=1 PORT=8993 bash "$REPO/examples/api/claude-with-brain.sh" --check
  if [ "$status" -ne 0 ]; then
    echo "$output" >&3
  fi
  [ "$status" -eq 0 ]
  [[ "$output" == *"OK: authenticated GET /v1/models succeeded"* ]]
}

@test "examples/api/openai_client.py runs against the mock's --openai surface" {
  [ -n "$OPENAI_KEY" ]
  run_example "$REPO/examples/api/openai_client.py" --base-url "http://127.0.0.1:$OPENAI_PORT" --api-key "$OPENAI_KEY" --model brain/mock --out "$OUT/openai_client.png"
  [[ "$output" == *"images/generations"* ]]
  [ -f "$OUT/openai_client.png" ]
}

# ------------------------------------------------------------- completeness

@test "every tracked example is accounted for in examples/manifest.tsv" {
  cd "$REPO"
  local manifest="tests/e2e/examples/manifest.tsv"
  local tracked listed
  tracked="$(git ls-files 'examples/*.py' 'examples/*.sh' | sort)"
  listed="$(tail -n +2 "$manifest" | cut -f1 | sort)"

  local missing extra
  missing="$(comm -23 <(echo "$tracked") <(echo "$listed"))"
  extra="$(comm -13 <(echo "$tracked") <(echo "$listed"))"

  if [ -n "$missing" ]; then
    echo "examples not listed in $manifest:" >&3
    echo "$missing" >&3
  fi
  if [ -n "$extra" ]; then
    echo "$manifest lists paths that no longer exist / aren't tracked:" >&3
    echo "$extra" >&3
  fi
  [ -z "$missing" ]
  [ -z "$extra" ]
}
