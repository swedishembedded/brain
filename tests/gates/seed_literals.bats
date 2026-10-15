#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible, auditable
# randomness in ML pipelines for its clients. If your team needs expertise in
# seeding discipline for training and generation then you can procure our
# services by sending an email to info@swedishembedded.com.

# Spec for scripts/gates/check-seed-literals.sh, the seed-literal ratchet: a
# literal seed in production code must be allowlisted with a reason, a stale
# entry fails, a `todo:` entry passes but is counted, and a literal seed in a
# test needs a `// seed:` reason when test mode is enforced. Every case runs the
# gate against a copy of the fixture under scripts/gates/testdata/seed-literals,
# never the real tree.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  FIXTURE="$REPO/scripts/gates/testdata/seed-literals"
  TREE="$BATS_TEST_TMPDIR/tree"
  ALLOW="$BATS_TEST_TMPDIR/allow.d"
  cp -r "$FIXTURE/tree" "$TREE"
  cp -r "$FIXTURE/allow.d" "$ALLOW"
}

gate() {
  run bash "$REPO/scripts/gates/check-seed-literals.sh" --root "$TREE" --allow-dir "$ALLOW" "$@"
}

@test "an allowlisted salt, a justified test seed and look-alikes in comments and strings pass" {
  gate
  [ "$status" -eq 0 ]
  [[ "$output" == *"OK"* ]]
}

@test "a new production literal fails and says the default must be random" {
  cat >>"$TREE/crates/a/src/lib.rs" <<'RS'
pub struct Opts {
    pub seed: u64,
}
impl Default for Opts {
    fn default() -> Self {
        Opts { seed: 1337 }
    }
}
RS
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"NEW"*"Opts::default:seed=1337"* ]]
  [[ "$output" == *"random"* ]]
}

@test "every literal form is caught" {
  cat >>"$TREE/crates/a/src/lib.rs" <<'RS'
pub fn forms(a: &mut Args) {
    let mut seed: u64 = 11;
    let _ = a.u64_or("--seed", 12);
    let _ = Builder::new().seed(13);
    let _ = Lcg::new(14);
    let _ = a.get("seed").unwrap_or(15);
    let _ = Opts { seed: Some(16), ..Default::default() };
}
RS
  gate
  [ "$status" -ne 0 ]
  for lit in "seed=11" "--seed=12" ".seed(13)" "Lcg::new(14)" "unwrap_or=15" "seed=Some(16)"; do
    [[ "$output" == *"$lit"* ]] || { echo "missing $lit"; echo "$output"; return 1; }
  done
}

@test "a python seed default and a Makefile SEED default fail" {
  printf 'ap.add_argument("--seed", type=int, default=0)\n' >>"$TREE/tools/gen.py"
  printf 'SEED ?= 1337\n' >>"$TREE/Makefile"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"tools/gen.py"*"--seed=0"* ]]
  [[ "$output" == *"Makefile"*"SEED=1337"* ]]
}

@test "a stale allowlist entry fails (the ratchet only shrinks)" {
  printf 'crates/a/src/lib.rs\t-:GONE=1\tconst\tsalt: removed\n' >>"$ALLOW/a.allow"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"STALE"*"GONE=1"* ]]
}

@test "an entry without a recognised reason prefix fails" {
  sed -i 's/salt: mixed/because: mixed/' "$ALLOW/a.allow"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"reason"* ]]
}

@test "an entry in another crate's allowlist file fails" {
  mv "$ALLOW/a.allow" "$ALLOW/b.allow"
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"b.allow"* ]]
}

@test "a todo entry passes, is counted, and --no-todo refuses it" {
  sed -i 's/salt: mixed/todo: mixed/' "$ALLOW/a.allow"
  gate
  [ "$status" -eq 0 ]
  [[ "$output" == *"1 todo"* ]]
  gate --no-todo
  [ "$status" -ne 0 ]
}

@test "a test literal without a seed reason is reported, and fails only in test mode" {
  sed -i '/seed: the expected value/d' "$TREE/crates/a/src/lib.rs"
  gate
  [ "$status" -eq 0 ]
  [[ "$output" == *"1 test"* ]]
  gate --tests
  [ "$status" -ne 0 ]
  [[ "$output" == *"Rng::new(7)"* ]]
  [[ "$output" == *"// seed:"* ]]
}

@test "python module seed constants and torch, numpy and random seeding calls are caught" {
  cat >>"$TREE/tools/gen.py" <<'PY'
SEED = 21
NOISE_SEED: int = 22


def dump():
    torch.manual_seed(23)
    g = torch.Generator().manual_seed(24)
    np.random.seed(25)
    random.seed(26)
    torch.manual_seed(args.seed)
PY
  gate
  [ "$status" -ne 0 ]
  for lit in "SEED=21" "NOISE_SEED=22" "dump:manual_seed=23" "dump:manual_seed=24" "dump:np.random.seed=25" "dump:random.seed=26"; do
    [[ "$output" == *"$lit"* ]] || { echo "missing $lit"; echo "$output"; return 1; }
  done
  [[ "$output" != *"args.seed"* ]]
}

@test "a python seed literal in a test needs a seed reason" {
  mkdir -p "$TREE/tools/tests"
  printf 'SEED = 5\n' >"$TREE/tools/tests/test_gen.py"
  gate --tests
  [ "$status" -ne 0 ]
  [[ "$output" == *"test_gen.py"*"SEED=5"* ]]
  printf '# seed: the fixture is defined by this value\nSEED = 5\n' >"$TREE/tools/tests/test_gen.py"
  gate --tests
  [ "$status" -eq 0 ]
}
