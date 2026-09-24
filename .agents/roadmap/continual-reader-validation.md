# continual-reader: the validation run

Paste the block below into a fresh session on a machine that has real
weights and a network. It is the whole starting context.

Everything it refers to is built and unit-tested. What has NOT happened is a
run against a real model, so no continual-learning claim has been measured
and none should be repeated until this produces one.

---

## THE BRIEF (paste from here)

You are validating whether brain's continual reader can actually learn.

Repository: `applications/edgeai/brain`, branch `main`. Read
`.agents/roadmap/continual-reader.md` first: it is the design, and Part 3.3
is the pre-registered acceptance block you are checking against. Do not edit
that block. If you find yourself wanting to relax a threshold after seeing a
number, that is the thing it exists to stop.

### What is already true

`brain-audit` (91 tests), `brain-residency` (110), `brain-promote` (35), the
sample (8) and the SDK record surfaces (4) all pass. The instrument is
built. `RunFacts` has only ever been filled in by hand.

### Step 0, before anything runs: pre-register

Write your thresholds into a file and commit it BEFORE the first run:

- the battery delta you will call a success
- the largest seed-repeat delta you will accept as noise
- the battery regression budget
- how many episodes constitutes the run

A number chosen after seeing the result is not a threshold.

### Step 1: build and prove the machinery still holds

    make build/release
    cargo test --release --offline -p brain-audit --lib
    cargo test --release --offline -p brain-residency --lib
    cargo build --release -p sample-learning-reader

### Step 2: the corpus and the frozen baseline

    ./target/release/sample-learning-reader generate --corpus corpus/ --seed 1
    ./target/release/sample-learning-reader battery --model Qwen/Qwen3-0.6B \
        --run-dir run-a/ --seed 1 | tee battery-before.txt

The tool the corpus describes did not exist until that seed was drawn, so
the baseline should be at or near zero. **If it is not near zero, stop and
find out why before reading anything.** A model that already scores well has
either been given the answers or is guessing a format, and every later
number is then meaningless.

### Step 3: read

    ./target/release/sample-learning-reader read --corpus corpus/ \
        --run-dir run-a/ --model Qwen/Qwen3-0.6B --seed 1 | tee read.txt
    ./target/release/sample-learning-reader report --run-dir run-a/ | tee report.txt

### Step 4: the result

    ./target/release/sample-learning-reader battery --model Qwen/Qwen3-0.6B \
        --run-dir run-a/ --seed 1 | tee battery-after.txt

The difference between before and after, on tasks frozen before any reading
and never trained on, is the only thing here that is a result. The promote
rate is plumbing.

### Step 5: the controls, which decide whether step 4 means anything

    ./target/release/sample-learning-reader selftest --corpus corpus/ --run-dir run-a/

Then, for the arms the sample can already run:

- **seed repeat**: run steps 2 to 4 again into `run-a2/` at the SAME seed.
  The adapters should be byte-identical and the battery delta zero. Any
  effect smaller than what this produces is not an effect.
- **multi-seed**: repeat at seeds 2, 3, 4 into their own run directories.
  The spread across them is the instrument's noise floor
  (`audit::arms::seed_spread`).
- **order permutation**: the same corpus at a different `--seed` reorders
  the documents. Compare the retention diagonals, not the promote rates.
- **injections**: `selftest` already checks these. It must exit zero.

### Step 6: what you will have to build to finish the block

Three clauses cannot be filled by running the sample as it stands, and this
is stated plainly so you do not report them as passed:

1. **The null-gate arm** is not a run mode. `audit::triage` decides with
   `promote::gate`; a null arm means running the same stream with
   `GatePolicy::CoinFlip` and counting promotions. Clause 1 needs it.
2. **BWT and the retention matrix** are computed inside `audit::reader` but
   not surfaced through `ReadOutcome`. Clause 3 needs them exposed.
3. **The shuffled-labels arm** is not a run mode. It means training each
   episode against another episode's frozen probes.

`audit::arms` already has the scoring functions for all three
(`shuffled_labels`, `order_permutation`, `seed_spread`, `injections`) and
`audit::acceptance::Acceptance::evaluate` assembles the verdict. What is
missing is the plumbing that produces their inputs from a real run.

### Step 7: report

Fill `audit::acceptance::RunFacts` and print
`Acceptance::evaluate(&facts, &arms).table()`. Every clause, passed or not.

**The acceptance criterion is a defensible number, not a positive one.** A
reader that learned little and says so, with its controls intact, has
passed. A large battery delta with one inconclusive control has not. If the
run shows no learning, that is a result: report it, with the controls that
make it credible, and do not tune anything to improve it.

### Traps that would make a positive result fake

- a baseline that was not near zero
- an effect smaller than the seed-repeat delta
- `selftest` exiting non-zero and being ignored
- the `shortcut/` lane promoting, which means the model trained on battery
  answers
- the `format/` lane promoting while the paraphrase probes stay flat, which
  is template learning
- any clause reported as passed that was actually unanswered

## END OF BRIEF
