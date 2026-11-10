#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Mirage served end to end from the checkpoint as published: `brain serve` over a
# models directory holding only that repo lists it, and the CLI, the HTTP
# `/v1/run` route, the D-Bus `Run` method and brain-py all return the same JSON
# for the same picture. Concurrent D-Bus requests are merged into device batches
# and every answer equals the single-request answer.
#
# Needs the checkpoint under $BRAIN_MODELS_DIR (brain pull Cactus-Compute/mirage)
# and a picture in $MIRAGE_IMAGE (any JPEG or PNG); skips what is missing.
# MIRAGE_FLAGS carries brain's global device flags (for example "--device gpu1 --backend cuda"),
# given to the server and to every command the suite runs; unset means the default device.
# Run: MIRAGE_IMAGE=cats.jpg BRAIN_BIN=./target/fast/brain bats tests/e2e/mirage.bats

MODEL="Cactus-Compute/mirage"
FORM='{"animal": {"enum": ["cat", "dog", "bird"]}, "count": {"type": "integer", "minimum": 0, "maximum": 6}}'

setup_file() {
  command -v dbus-daemon >/dev/null 2>&1 || skip "dbus-daemon not installed"
  command -v jq >/dev/null 2>&1 || skip "jq not installed"
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  export REPO
  BRAIN="${BRAIN_BIN:-$REPO/target/debug/brain}"
  [ -x "$BRAIN" ] || skip "no brain binary (set BRAIN_BIN)"
  export BRAIN
  PY="${EXAMPLES_PY:-python3}"
  export PY
  "$PY" -c "import jeepney, PIL" >/dev/null 2>&1 || skip "python jeepney and pillow are needed"
  export PYTHONPATH="$REPO/brain-py"
  local store="${BRAIN_MODELS_DIR:-$HOME/.local/share/brain/models}"
  [ -f "$store/$MODEL/mirage.safetensors" ] || skip "$MODEL is not downloaded (brain pull $MODEL)"
  [ -f "${MIRAGE_IMAGE:-}" ] || skip "set MIRAGE_IMAGE to a picture"

  # shellcheck source=../../tools/dbus-session.sh
  . "$REPO/tools/dbus-session.sh"
  CONF_DIR="$(dbus_session_work_dir)"
  export CONF_DIR
  mkdir -p "$CONF_DIR/models/Cactus-Compute"
  ln -s "$store/$MODEL" "$CONF_DIR/models/$MODEL"
  export BRAIN_MODELS_DIR="$CONF_DIR/models"
  "$PY" -c "
from PIL import Image
im = Image.open('$MIRAGE_IMAGE').convert('RGB')
open('$CONF_DIR/photo.ppm', 'wb').write(b'P6 %d %d 255\n' % im.size + im.tobytes())
"
  # The suite always owns its bus: an inherited address may name a socket nobody listens on.
  unset DBUS_SESSION_BUS_ADDRESS
  dbus_session_start_bus || skip "could not start a private dbus-daemon"
  export PORT="${MIRAGE_PORT:-8907}"
  # shellcheck disable=SC2086 # MIRAGE_FLAGS is deliberately word-split
  dbus_session_start_serve "$BRAIN" "--dbus --openai $PORT --models-dir $BRAIN_MODELS_DIR ${MIRAGE_FLAGS:-}"
  dbus_session_wait_ready 120 || { cat "$DBUS_SESSION_LOG_FILE" >&3; skip "brain serve did not become ready"; }
  export KEY="${BRAIN_OPENAI_KEY:-}"
  export FLAGS="${MIRAGE_FLAGS:-}"
}

teardown_file() {
  dbus_session_stop
}

# The JSON a run printed, without the accounting, key-sorted.
printed() { jq -cS 'del(.usage)' <<<"$1"; }

@test "brain caps lists describe and extract" {
  run "$BRAIN" caps "$MODEL"
  [ "$status" -eq 0 ]
  [[ "$output" == *describe* && "$output" == *extract* ]]
}

@test "the server lists the model with both actions" {
  run "$PY" - <<'PY'
from brain_py.dbus import BrainDBus
with BrainDBus() as b:
    assert b.actions("Cactus-Compute/mirage") == ["describe", "extract"], b.actions("Cactus-Compute/mirage")
PY
  [ "$status" -eq 0 ]
}

@test "describe: CLI, HTTP, D-Bus and brain-py return the same JSON" {
  cli="$("$BRAIN" $FLAGS mirage describe --image "$CONF_DIR/photo.ppm" --mask_format bitmap --max_tokens 600 --json 2>/dev/null | tail -1)"
  http="$("$PY" "$REPO/tests/e2e/mirage/surfaces.py" http "$PORT" "$KEY" describe "$CONF_DIR/photo.ppm")"
  bus="$("$PY" "$REPO/tests/e2e/mirage/surfaces.py" dbus describe "$CONF_DIR/photo.ppm")"
  py="$("$PY" - <<PY
import json
from brain_py.dbus import BrainDBus
from brain_py.image import load_ppm
d, w, h = load_ppm("$CONF_DIR/photo.ppm")
with BrainDBus() as b:
    print(json.dumps(dict(b.describe((d, w, h), max_tokens=600, mask_format="bitmap")), sort_keys=True))
PY
)"
  [ "$(printed "$cli")" = "$(printed "$http")" ]
  [ "$(printed "$http")" = "$(printed "$bus")" ]
  [ "$(printed "$bus")" = "$(printed "$py")" ]
  [ "$(jq '.objects | length' <<<"$cli")" -ge 0 ]
}

@test "extract: CLI, HTTP and D-Bus return the same JSON" {
  cli="$("$BRAIN" $FLAGS mirage extract --image "$CONF_DIR/photo.ppm" --form "$FORM" --json 2>/dev/null | tail -1)"
  http="$("$PY" "$REPO/tests/e2e/mirage/surfaces.py" http "$PORT" "$KEY" extract "$CONF_DIR/photo.ppm" "$FORM")"
  bus="$("$PY" "$REPO/tests/e2e/mirage/surfaces.py" dbus extract "$CONF_DIR/photo.ppm" "$FORM")"
  [ "$(printed "$cli")" = "$(printed "$http")" ]
  [ "$(printed "$http")" = "$(printed "$bus")" ]
  [ "$(jq '.form | keys | length' <<<"$cli")" -eq 2 ]
}

@test "D-Bus and HTTP refuse path-like params, malformed forms and oversized images before any work" {
  run "$PY" - <<PY
import base64, json, urllib.error, urllib.request
from brain_py import BrainError
from brain_py.base import MIRAGE_MODEL
from brain_py.dbus import BrainDBus
from brain_py.image import load_ppm

data, w, h = load_ppm("$CONF_DIR/photo.ppm")
meta = {"image": {"media": "image", "w": w, "h": h, "c": 3}}
with BrainDBus() as b:
    for params, needle in (({"weights": "/etc/passwd"}, "weights"), ({"path": "/etc/passwd"}, "path"), ({"sections": "objects,bogus"}, "bogus")):
        try:
            b.run(MIRAGE_MODEL, "describe", params, blobs={"image": data}, meta=meta)
        except BrainError as e:
            assert needle in str(e), (params, str(e))
        else:
            raise AssertionError(f"{params} was accepted")
    try:
        b.run(MIRAGE_MODEL, "extract", {"form": "{not json"}, blobs={"image": data}, meta=meta)
    except BrainError as e:
        assert "form" in str(e), str(e)
    else:
        raise AssertionError("a malformed form was accepted")
    huge = {"image": {"media": "image", "w": 20000, "h": 20000, "c": 3}}
    try:
        b.run(MIRAGE_MODEL, "describe", {}, blobs={"image": data}, meta=huge)
    except BrainError as e:
        assert "pixels" in str(e), str(e)
    else:
        raise AssertionError("an oversized image was accepted")

def post(body):
    req = urllib.request.Request("http://127.0.0.1:$PORT/v1/run", data=json.dumps(body).encode(), headers={"Authorization": "Bearer $KEY", "Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        return e.code
    return 200

blob = {"media": "image", "data": base64.b64encode(data).decode(), "meta": {"w": w, "h": h, "c": 3}}
assert post({"model": MIRAGE_MODEL, "action": "describe", "params": {"weights": "/etc/passwd"}, "blobs": {"image": blob}}) == 400
assert post({"model": MIRAGE_MODEL, "action": "extract", "params": {"form": "[]"}, "blobs": {"image": blob}}) == 400
PY
  echo "$output" >&3
  [ "$status" -eq 0 ]
}

@test "16 concurrent requests are batched and each equals the single answer" {
  run "$PY" "$REPO/samples/python/vision/mirage-perception/mirage_perception.py" --image "$CONF_DIR/photo.ppm" --max-tokens 600 --concurrent 16
  echo "$output" >&3
  [ "$status" -eq 0 ]
  [[ "$output" == *"identical to the single answer: True"* ]]
  # The device batch sizes the 16 requests ran in: at least one batch held more than one.
  sizes="$(grep -o 'device batch sizes: \[[0-9, ]*\]' <<<"$output" | tr -dc '0-9,')"
  [ "$(tr ',' '\n' <<<"$sizes" | sort -n | tail -1)" -gt 1 ]
}
