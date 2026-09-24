# continual-reader: the validation run

Paste the block below into a fresh session on a machine that has real
weights and a network. It is the whole starting context.

Everything it refers to is built and unit-tested. What has NOT happened is a
run against a real model, so no continual-learning claim has been measured
and none should be repeated until this produces one.

---

## THE BRIEF (paste from here)

You are validating whether brain's continual reader can actually learn.

Repository: `applications/edgeai/brain`, on the working branch (`latest`).
Read `.agents/roadmap/continual-reader.md` first: it is the design, and Part
3.3 is the pre-registered acceptance block you are checking against. Do not
edit that block. If you find yourself wanting to relax a threshold after
seeing a number, that is the thing it exists to stop.

**Run everything from a scratch directory outside the repository.** Every
command below writes relative paths - a corpus, run directories, transcripts
- and those are run artefacts, not source. Only the pre-registration of Step
0 is committed.

    mkdir -p /tmp/reader-validation && cd /tmp/reader-validation
    BRAIN=<path to>/applications/edgeai/brain
    READER=$BRAIN/target/release/sample-learning-reader

### What is already true

`brain-audit` (100 tests), `brain-residency` (110), `brain-promote` (35), the
sample (11), the SDK record surfaces (4) and `rl`'s `reader_learner` (4,
`--features qwen3`, needs a device) all pass. The instrument is built.
`RunFacts` has only ever been filled in by hand.

### Step 0, before anything runs: pre-register

Write your thresholds into a file and commit it BEFORE the first run:

- the battery delta you will call a success
- the largest seed-repeat delta you will accept as noise
- the battery regression budget
- how many episodes constitutes the run

A number chosen after seeing the result is not a threshold.

**What the episode count can be.** `generate` writes ten documents across
nine lanes, two of them in the `learn` lane. Ten is therefore the whole
stream, and `--until` can only make it shorter. A pre-registered N above ten
needs a larger corpus first; do not pre-register one this corpus cannot
produce.

### Step 1: build and prove the machinery still holds

    make build/release
    cargo test --release --offline -p brain-audit --lib
    cargo test --release --offline -p brain-residency --lib
    cargo build --release -p sample-learning-reader

The last one is not optional and is not implied by the first: samples are
workspace members but never default members, so `make build/release` does
not build them.

### Step 2: the corpus and the frozen baseline

    $READER generate --corpus corpus/ --seed 1
    $READER battery --model Qwen/Qwen3-0.6B --run-dir run-a/ --seed 1 \
        | tee battery-before.txt

**One seed selects the tool, and the tool selects the battery.** `generate
--seed S` invents the tools; `battery --seed S` derives its held-out tasks
from the same S. Scoring a battery at a seed the corpus was not generated at
asks about a tool that was never written down - and the batteries are not
even the same size between seeds. Keep `generate` and `battery` on the same
seed for the whole run.

The tool the corpus describes did not exist until that seed was drawn, so
the baseline should be at or near zero. **If it is not near zero, stop and
find out why before reading anything.** A model that already scores well has
either been given the answers or is guessing a format, and every later
number is then meaningless.

### Step 3: read

    $READER read --corpus corpus/ --run-dir run-a/ \
        --model Qwen/Qwen3-0.6B --seed 1 | tee read.txt
    $READER report --run-dir run-a/ | tee report.txt

`read` reports backward transfer over the retention matrix when the audit
schedule has revisited anything, and prints UNMEASURED when it has not. Ten
episodes may well not be enough for it to come round; that is a fact about
the run's length, and it is reported rather than defaulted to zero.

### Step 4: the result

    $READER battery --model Qwen/Qwen3-0.6B --run-dir run-a/ --seed 1 \
        | tee battery-after.txt

The difference between before and after, on tasks frozen before any reading
and never trained on, is the only thing here that is a result. The promote
rate is plumbing.

`battery` scores what the run currently serves, which is what the gate last
promoted into `run-a/work/incumbent.safetensors`. A run directory is bound
to the model reference it was created with and refuses to reopen under
another one, so the before and after are about the same base by
construction.

### Step 5: the controls, which decide whether step 4 means anything

    $READER selftest --corpus corpus/ --run-dir run-a/

`selftest` reads the corpus's own `labels.json` and checks each episode's
verdict against its lane's label. It takes no seed and writes nothing.

Then, for the arms the sample can already run:

- **seed repeat**: run steps 2 to 4 again into `run-a2/` with every seed
  unchanged. The promoted adapters should be byte-identical and the battery
  delta zero. Any effect smaller than what this produces is not an effect.
  Training is reproducible from its seed and a saved adapter is a function
  of its content, so a difference here is a real finding, not serialisation
  noise.
- **order permutation**: the same corpus read in a different order. Vary
  ONLY `read --seed` - leave `generate` and `battery` at the seed the corpus
  was written with, so the battery stays the frozen one. Compare the
  retention diagonals, not the promote rates.
- **multi-seed**: for a noise floor over the instrument, repeat the order
  permutation at several `read` seeds and use the spread of their battery
  scores (`audit::arms::seed_spread`). Regenerating the corpus at seeds 2, 3
  and 4 instead would measure how hard three DIFFERENT invented tools are,
  which is not a noise floor for this one.
- **null gate**: `read --null-gate S` into its own run directory. A coin
  decides what carries forward while the real gate still runs and is still
  recorded, so the two arms' promote rates are directly comparable. Compare
  the ledger's `carried` column, not its `promoted` one - `promoted` is what
  the gate said under both arms, which is the point.
- **shuffled labels**: `read --shuffled-labels S` into its own run
  directory. Each episode is trained on its own rows and gated against
  another episode's frozen probes, so it should promote at chance. A rate
  near the real arm's means the gate is responding to the training run
  happening rather than to what it taught.
- **injections**: `selftest` already checks these. It must exit zero.

### Step 6: the one clause still unbuilt

Three of the four things that used to be missing are run modes now, and are
in Step 5 above. What remains:

**Serving while reading.** The block's preamble asks for N episodes read "in
one process that also served requests throughout". Nothing connects the
reader to `brain-residency`; a read serves nothing. Report the run as not
having met the preamble, and say so rather than letting a run that never
served read as one that did.

`audit::arms` has the scoring (`shuffled_labels`, `order_permutation`,
`seed_spread`, `injections`) and `audit::acceptance::Acceptance::evaluate`
assembles the verdict from `RunFacts`. Fill `RunFacts` from what the run
actually produced: `bwt` and `worst_block_drop` from `ReadOutcome`,
`null_gate_promoted` from the null arm's own ledger (its `carried` column,
not its `promoted` one), the battery numbers from Steps 2 and 4.

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
- a battery scored at a seed the corpus was not generated at
- any clause reported as passed that was actually unanswered

## END OF BRIEF
