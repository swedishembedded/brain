#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements release pipelines whose green result means the
# tests really ran, for its clients. If your team needs expertise in CI gates
# for machine-learning and embedded software, you can procure our services by
# sending an email to info@swedishembedded.com.

# Spec for scripts/ci/suite-report.py: a required suite that ran nothing, failed or skipped is not green,
# and the matrix names every required suite that has no result as not-run.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  OUT="$BATS_TEST_TMPDIR/results"
  REPORT="$REPO/scripts/ci/suite-report.py"
}

check() { # name format [flags...] < output
  local name="$1" format="$2"; shift 2
  run python3 "$REPORT" check --name "$name" --format "$format" --input - --out "$OUT" "$@"
}

@test "a cargo run whose tests all passed is a pass, counting every test binary" {
  check unit cargo <<'EOF'
running 3 tests
test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.01s
running 2 tests
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
EOF
  [ "$status" -eq 0 ]
  [ "$(python3 -c "import json;r=json.load(open('$OUT/unit.json'));print(r['status'],r['passed'],r['ignored'])")" = "pass 5 1" ]
}

@test "a cargo run that reports a SKIP line is a skip, not a pass" {
  check parity cargo <<'EOF'
SKIP: needs the checkpoint
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
EOF
  [ "$status" -eq 1 ]
  grep -q '"status": "skip"' "$OUT/parity.json"
}

@test "an allowed number of skips still passes" {
  check parity cargo --max-skips 1 <<'EOF'
SKIP: needs the checkpoint
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
EOF
  [ "$status" -eq 0 ]
}

@test "a cargo run with no tests, or with a failure, is not a pass" {
  check empty cargo <<'EOF'
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
EOF
  [ "$status" -eq 1 ]
  grep -q '"status": "empty"' "$OUT/empty.json"
  check failing cargo <<'EOF'
test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
EOF
  [ "$status" -eq 1 ]
  grep -q '"status": "fail"' "$OUT/failing.json"
}

@test "ignored tests fail a suite only when asked" {
  local run_output='test result: ok. 3 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.01s'
  check lenient cargo <<<"$run_output"
  [ "$status" -eq 0 ]
  check strict cargo --fail-on-ignored <<<"$run_output"
  [ "$status" -eq 1 ]
}

@test "a bats suite in which every test skips exits 0 for bats and is a skip here" {
  check e2e tap <<'EOF'
1..2
ok 1 first # skip no dbus-daemon
ok 2 second # skip no dbus-daemon
EOF
  [ "$status" -eq 1 ]
  grep -q '"skipped": 2' "$OUT/e2e.json"
  grep -q '"status": "skip"' "$OUT/e2e.json"
}

@test "a bats suite that passed is a pass, and one whose plan is not met is a failure" {
  check good tap <<'EOF'
1..2
ok 1 first
ok 2 second
EOF
  [ "$status" -eq 0 ]
  check short tap <<'EOF'
1..3
ok 1 first
ok 2 second
EOF
  [ "$status" -eq 1 ]
  grep -q 'the plan announced 3' "$OUT/short.json"
}

@test "the matrix fails on a required suite that has no result and lists it as not-run" {
  check unit cargo <<<'test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s'
  run python3 "$REPORT" matrix --results "$OUT" --require unit,gpu --summary "$BATS_TEST_TMPDIR/summary.md"
  [ "$status" -eq 1 ]
  [[ "$output" == *"| gpu | not-run |"* ]]
  [[ "$output" == *"| unit | pass | 3 |"* ]]
  grep -q "not-run" "$BATS_TEST_TMPDIR/summary.md"
}

@test "the matrix passes when every required suite passed" {
  check unit cargo <<<'test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s'
  run python3 "$REPORT" matrix --results "$OUT" --require unit
  [ "$status" -eq 0 ]
}
