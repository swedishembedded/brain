#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for keeping large Rust inference and
# training stacks layered and buildable for its clients. If your team needs
# expertise in architectural dependency enforcement then you can procure our
# services by sending an email to info@swedishembedded.com.

# Spec for scripts/gates/crate-layers.py, the full layer-map gate: every
# package has a layer, a normal dependency on a higher-numbered layer fails
# unless allowlisted with a reason, a stale allowlist entry fails, and the
# crate table in the architecture rules must match the map. Each case runs
# against a synthetic workspace (cargo metadata JSON), never the real tree.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  W="$BATS_TEST_TMPDIR"
  printf 'brain-low\t1\nbrain-mid\t3\nbrain-top\t4\n' >"$W/map"
  : >"$W/allow"
  printf 'x\n<!-- crate-layers:begin -->\n<!-- crate-layers:end -->\n' >"$W/arch.md"
  meta '' '"brain-low"' '"brain-mid"'
  python3 "$REPO/scripts/gates/crate-layers.py" --metadata "$W/meta.json" --map "$W/map" --allow "$W/allow" --arch-doc "$W/arch.md" --write-doc >/dev/null
}

# meta <deps of low> <deps of mid> <deps of top>: write the synthetic workspace.
meta() {
  python3 - "$W/meta.json" "$1" "$2" "$3" <<'PY'
import json, sys
out, low, mid, top = sys.argv[1:5]
def deps(spec, kind=None):
    return [{"name": n.strip().strip('"'), "kind": kind} for n in spec.split(",") if n.strip()]
pk = [
    {"name": "brain-low", "dependencies": deps(low)},
    {"name": "brain-mid", "dependencies": deps(mid)},
    {"name": "brain-top", "dependencies": deps(top)},
]
json.dump({"packages": pk}, open(out, "w"))
PY
}

gate() {
  run python3 "$REPO/scripts/gates/crate-layers.py" --metadata "$W/meta.json" --map "$W/map" --allow "$W/allow" --arch-doc "$W/arch.md"
}

@test "a workspace that only depends downward passes" {
  gate
  [ "$status" -eq 0 ]
  [[ "$output" == *"OK"* ]]
}

@test "a lower layer depending on a higher one fails and names both" {
  meta '"brain-top"' '' ''
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"UPWARD DEPENDENCY"*"brain-low"*"brain-top"* ]]
}

@test "the same upward edge passes when allowlisted with a reason" {
  meta '"brain-top"' '' ''
  printf 'brain-low\tbrain-top\ta reason\n' >"$W/allow"
  gate
  [ "$status" -eq 0 ]
}

@test "an allowlist entry whose edge is gone fails (the list only shrinks)" {
  printf 'brain-low\tbrain-top\ta reason\n' >"$W/allow"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE ALLOWLIST ENTRY"* ]]
}

@test "a package missing from the map fails" {
  printf 'brain-low\t1\nbrain-mid\t3\n' >"$W/map"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"UNASSIGNED"*"brain-top"* ]]
}

@test "a map line naming no package fails" {
  printf 'brain-ghost\t2\n' >>"$W/map"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE MAP LINE"*"brain-ghost"* ]]
}

@test "a dev-dependency does not count as a dependency" {
  python3 - "$W/meta.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
m["packages"][0]["dependencies"] = [{"name": "brain-top", "kind": "dev"}]
json.dump(m, open(sys.argv[1], "w"))
PY
  gate
  [ "$status" -eq 0 ]
}

@test "a crate table that disagrees with the map fails and --write-doc repairs it" {
  printf 'brain-low\t2\nbrain-mid\t3\nbrain-top\t4\n' >"$W/map"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"DOC DRIFT"* ]]
  run python3 "$REPO/scripts/gates/crate-layers.py" --metadata "$W/meta.json" --map "$W/map" --allow "$W/allow" --arch-doc "$W/arch.md" --write-doc
  [ "$status" -eq 0 ]
  gate
  [ "$status" -eq 0 ]
}
