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
  [ -x "$BRAIN" ] || skip "no brain binary (build with: make build/debug, or set BRAIN_BIN)"
  export BRAIN
  PY="${EXAMPLES_PY:-python3}"
  export PY
  "$PY" -c "import jeepney" >/dev/null 2>&1 || skip "python jeepney not installed (pip install -e brain-py)"
  export PYTHONPATH="$REPO/brain-py"

  # shellcheck source=../../tools/dbus-session.sh
  . "$REPO/tools/dbus-session.sh"

  CONF_DIR="$(dbus_session_work_dir)"
  export CONF_DIR
  OUT="$CONF_DIR/out"
  mkdir -p "$OUT"
  export OUT

  dbus_session_start_bus || skip "could not start a private dbus-daemon"
  DBUS_PID="$DBUS_SESSION_PID"
  export DBUS_PID

  ANTHROPIC_PORT="${ANTHROPIC_PORT:-8991}"
  export ANTHROPIC_PORT
  OPENAI_PORT="${OPENAI_PORT:-8992}"
  export OPENAI_PORT

  # BRAIN_MOCK_DELAY_MS is a SERVER-side knob (crates/cli/src/resident_mock.rs
  # reads it once per request from ITS OWN environment) — it must be set on this
  # launch, not on a client invocation later, or the cancellation test below
  # races every step to completion before Cancel can land. 300ms split across
  # text2image's 4 steps is unnoticeable for every other test here.
  BRAIN_MOCK=1 BRAIN_MOCK_DELAY_MS=300 BRAIN_DEVICE=cpu \
    dbus_session_start_serve "$BRAIN" "--dbus --anthropic $ANTHROPIC_PORT --openai $OPENAI_PORT"
  SERVER_PID="$DBUS_SESSION_SERVER_PID"
  export SERVER_PID

  dbus_session_wait_ready 20 || {
    echo "--- brain serve log ---" >&3
    cat "$DBUS_SESSION_LOG_FILE" >&3 2>/dev/null || true
    skip "brain serve did not become ready"
  }

  OPENAI_KEY="${BRAIN_OPENAI_KEY:-}"
  export OPENAI_KEY
  ANTHROPIC_KEY="${BRAIN_ANTHROPIC_KEY:-}"
  export ANTHROPIC_KEY
}

teardown_file() {
  dbus_session_stop
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

# ------------------------------------------------------------- llm/

@test "examples/llm/qwen35moe.py runs against the mock over D-Bus" {
  run_example "$REPO/examples/llm/qwen35moe.py" --dbus --model brain/mock --in-text "hi" --out-stdio
  [[ "$output" == *"You said: hi"* ]]
}

@test "examples/llm/qwen35moe.py runs against the mock over OpenAI-compatible HTTP" {
  run_example "$REPO/examples/llm/qwen35moe.py" \
    --openai "127.0.0.1:$OPENAI_PORT" --api-key "$OPENAI_KEY" \
    --model brain/mock --in-text "hi" --out-stdio
  [[ "$output" == *"You said: hi"* ]]
}

@test "examples/llm/qwen35moe.py runs against the mock over Anthropic-compatible HTTP" {
  run_example "$REPO/examples/llm/qwen35moe.py" \
    --anthropic "127.0.0.1:$ANTHROPIC_PORT" --api-key "$ANTHROPIC_KEY" \
    --model brain/mock --in-text "hi" --out-stdio
  [[ "$output" == *"You said: hi"* ]]
}

# ------------------------------------------------------------- qwen3omnimoe/

@test "examples/qwen3omnimoe/omni.py runs against the mock over D-Bus" {
  run_example "$REPO/examples/qwen3omnimoe/omni.py" --dbus --model brain/mock --in-text "hi" --out-stdio
  [[ "$output" == *"You said: hi"* ]]
}

@test "examples/qwen3omnimoe/omni.py skips cleanly on unimplemented input/output flags" {
  run_example "$REPO/examples/qwen3omnimoe/omni.py" --dbus --model brain/mock --in-image foo.png --in-text ignored --out-stdio
}

@test "examples/qwen3omnimoe/omni.py runs against the mock over OpenAI-compatible HTTP" {
  # brain.models() is real on this transport, so the served-model precheck
  # (omni.py:144-153) runs before the generate call -- brain/mock is served,
  # so this exercises the full precheck-then-generate path, not just generate.
  run_example "$REPO/examples/qwen3omnimoe/omni.py" \
    --openai "127.0.0.1:$OPENAI_PORT" --api-key "$OPENAI_KEY" \
    --model brain/mock --in-text "hi" --out-stdio
  [[ "$output" == *"You said: hi"* ]]
}

@test "examples/qwen3omnimoe/omni.py runs against the mock over Anthropic-compatible HTTP" {
  # BrainAnthropic.manifests() raises NotImplementedError (Anthropic's API has
  # no /v1/models equivalent), so omni.py:148-151 catches it and skips the
  # served-model precheck entirely on this transport -- an unserved model
  # would fail at generate time, not as a skip(). brain/mock IS served here,
  # so this is still the real generate path, just without the precheck step
  # the OpenAI-transport test above exercises.
  run_example "$REPO/examples/qwen3omnimoe/omni.py" \
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

# ------------------------------------------------------------- videogen/

@test "examples/videogen/generate_video.py skips cleanly without Wan weights" {
  # The mock model serves no `t2v` action, so this is a skip by design -- what
  # it proves is that the client reaches `models()` and reports the missing
  # model cleanly instead of tracebacking. A real run needs BRAIN_WAN_* and is
  # minutes even at the script's own reduced default size, most of it the
  # umT5-XXL text encode on the CPU.
  run_example "$REPO/examples/videogen/generate_video.py" --prompt test --model brain/wan --out "$OUT/wan.mp4"
}

@test "examples/videogen/cancel_generation.py skips cleanly without Wan weights" {
  run_example "$REPO/examples/videogen/cancel_generation.py" --model brain/wan
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
