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

@test "constructors of every RNG family are caught" {
  cat >>"$TREE/crates/a/src/lib.rs" <<'RS'
pub fn families() {
    let _ = Mt19937::new(31);
    let _ = StdRng::seed_from_u64(32);
    let _ = ChaCha8Rng::seed_from_u64(33);
    let _ = Philox4x32::new(34);
}
RS
  gate
  [ "$status" -ne 0 ]
  for lit in "Mt19937::new(31)" "StdRng::seed_from_u64(32)" "ChaCha8Rng::seed_from_u64(33)" "Philox4x32::new(34)"; do
    [[ "$output" == *"$lit"* ]] || { echo "missing $lit"; echo "$output"; return 1; }
  done
}

@test "a seed hidden behind a zero-argument function or a serde default is caught" {
  cat >>"$TREE/crates/a/src/lib.rs" <<'RS'
fn unrecorded_seed() -> Seed {
    Seed::new(41)
}
fn legacy() -> u64 {
    42
}
fn draw() -> u64 {
    random_seed()
}
pub struct Cfg {
    #[serde(default = "legacy")]
    pub seed: u64,
}
RS
  gate
  [ "$status" -ne 0 ]
  for lit in "fn unrecorded_seed=41" "fn legacy=42"; do
    [[ "$output" == *"$lit"* ]] || { echo "missing $lit"; echo "$output"; return 1; }
  done
  [[ "$output" != *"fn draw"* ]]
}

@test "a literal or a seedless constant passed where a callee takes a seed is caught" {
  cat >>"$TREE/crates/a/src/lib.rs" <<'RS'
fn init(n: usize, seed: u64) -> usize { n + seed as usize }
fn caller(s: u64) {
    let _ = init(3, 51);
    let _ = init(3, s);
    let _ = init(3, SFT_BASE);
    let _ = init(3, STREAM_SEED_SALT);
}
pub struct M;
impl M {
    pub fn fit(&self, steps: usize, seed: u64) {}
    pub fn run(&self) {
        self.fit(2, 52);
    }
}
RS
  gate
  [ "$status" -ne 0 ]
  for lit in "caller:init#1=51" "caller:init#1=SFT_BASE" "M::run:fit#1=52"; do
    [[ "$output" == *"$lit"* ]] || { echo "missing $lit"; echo "$output"; return 1; }
  done
  [[ "$output" != *"init#1=s"* ]]
  [[ "$output" != *"init#1=STREAM_SEED_SALT"* ]]
}

@test "a positional literal in a test needs no entry and is reported only on request" {
  cat >>"$TREE/crates/a/src/lib.rs" <<'RS'
fn init(n: usize, seed: u64) -> usize { n + seed as usize }
#[cfg(test)]
mod more {
    #[test]
    fn t() { let _ = super::init(3, 53); }
}
RS
  gate --tests
  [ "$status" -eq 0 ]
}

@test "a loop or a range that enumerates literal seeds is caught" {
  cat >>"$TREE/crates/a/src/lib.rs" <<'RS'
pub fn sweep(members: u64, data: &Data) {
    for seed in 1..=members {
        train(seed);
    }
    let _ = (1..=3).map(|seed| cv_runs(data, seed)).count();
    for member_seed in 0..4 {
        train(member_seed);
    }
    for i in 0..4 {
        train(i);
    }
}
RS
  gate
  [ "$status" -ne 0 ]
  for lit in "sweep:for seed in 1.." "sweep:|seed| in (1..)" "sweep:for member_seed in 0.."; do
    [[ "$output" == *"$lit"* ]] || { echo "missing $lit"; echo "$output"; return 1; }
  done
  [[ "$output" != *"for i in"* ]]
}

@test "python defaults, keywords, dicts, environment fallbacks and RNG constructors are caught" {
  cat >>"$TREE/tools/gen.py" <<'PY'


def forecast(series, seed: int = 61):
    return post({"seed": 62, "series": series})


@dataclass
class Spec:
    seed: int = 63


def run(args):
    rng = np.random.default_rng(64)
    gen = random.Random(65)
    state = np.random.RandomState(66)
    call(seed=67)
    s = os.environ.get("BRAIN_SEED", "68")
    chosen = args.seed if args.seed is not None else 69
    other = args.seed or 70
    ok = args.seed if args.seed is not None else draw()
    call(seed=args.seed)
PY
  gate
  [ "$status" -ne 0 ]
  for lit in "forecast:seed=61" "forecast:\"seed\"=62" "Spec:seed=63" "run:default_rng=64" "run:random.Random=65" "run:RandomState=66" "run:seed=67" "run:environ.BRAIN_SEED=68" "run:seed-fallback=69" "run:seed-fallback=70"; do
    [[ "$output" == *"$lit"* ]] || { echo "missing $lit"; echo "$output"; return 1; }
  done
  [[ "$output" != *"=args"* ]]
}

@test "shell, bats and Makefile recipe seeds are caught" {
  mkdir -p "$TREE/scripts"
  cat >"$TREE/scripts/run.sh" <<'SH'
#!/usr/bin/env bash
SEED=${SEED:-71}
tool --seed 72
tool --noise-seed=73
for s in $(seq 1 3); do tool --seed "$s"; done
tool --seed "$SEED"
SH
  printf 'demo:\n\ttool --seed 74\n' >>"$TREE/Makefile"
  gate
  [ "$status" -ne 0 ]
  for lit in "scripts/run.sh"*"SEED:-71" "--seed=72" "--noise-seed=73" "seq 1 3" "Makefile"*"demo:--seed=74"; do
    [[ "$output" == *$lit* ]] || { echo "missing $lit"; echo "$output"; return 1; }
  done
  [[ "$output" != *'--seed "$'* ]]
}

@test "a documented command line with a seed needs the word reproduce beside it" {
  mkdir -p "$TREE/docs"
  cat >"$TREE/docs/run.md" <<'MD'
# Run

```sh
tool --seed 81
```

To reproduce the figures:

```sh
tool --seed 82
```
MD
  gate
  [ "$status" -ne 0 ]
  [[ "$output" == *"docs/run.md"*"--seed=81"* ]]
  [[ "$output" != *"--seed=82"* ]]
}

@test "a test reason that says any value is no reason" {
  cat >>"$TREE/crates/a/src/lib.rs" <<'RS'
#[cfg(test)]
mod anyvalue {
    #[test]
    fn t() {
        // seed: any value works here
        let _ = Rng::new(91);
    }
}
RS
  gate --tests
  [ "$status" -ne 0 ]
  [[ "$output" == *"Rng::new(91)"* ]]
}
