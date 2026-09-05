# gauntlet - roadmap

**Status: designed, not implemented.** Nothing in this file is code today -
see `.agents/roadmap/self-improve.md` for the training-regime substrate
(P7-P18) this depends on, all of which is itself still TODO as of this
writing.

The Weight-Learning Gauntlet answers a different question than
`.agents/roadmap/self-improve.md`'s continuous-training loop, and a
different one than `crates/bench`. `bench` asks "does this *architecture*
learn?" (a fixed task, a fixed training recipe, one number). The gauntlet
asks: **can this *system* discover that it lacks a capability, acquire that
capability through parameter updates, verify the update actually helped,
retain previously-learned capabilities, remain capable of further learning,
and improve its own learning process over repeated cycles?** That is a
different, harder claim than "loss goes down" or "benchmark score goes up",
and it needs its own artifacts (a retention matrix, a plasticity curve), not
a leaderboard row.

## Why a separate crate (`crates/gauntlet`, not `crates/bench`)

1. **The immutable-evaluator boundary must be structural, not a convention.**
   A loop that can see or influence its own test will eventually learn to
   satisfy the test instead of the task - an experimentally documented
   failure mode for tool-using agents, not a hypothetical one. Putting the
   generator (which holds a secret seed) and the verifier in a crate the
   training loop only ever calls through `rl::env::{Environment, Verifier}`
   - never one it can introspect or that shares mutable state with it - is
   what makes that boundary real.
2. **No dependency cycle.** `crates/bench` does not depend on `crates/rl`
   today. A `bench`-registered self-improvement benchmark would need `rl`
   (for the training loop) while `rl` already depends on `bench`-adjacent
   pieces for reuse - a cycle. `crates/gauntlet` depends on `brain-rl`,
   `brain-bench`, `brain-model`; nothing depends back on it. **Written down
   so it isn't rediscovered the hard way**: if a self-improvement benchmark
   is ever wanted inside `bench`'s own registry, it belongs in the CLI
   layer instead, not in either crate.
3. **Self-containment.** Per the standing invariant in `self-improve.md`
   ("brain never depends on sven"), every environment here is procedurally
   generated in-process with an exact, executable oracle - no network, no
   external service, no downloaded checkpoint. That is what lets the
   self-improvement examples ship inside brain and run from a clean
   checkout, and it is also what makes rule-holdout and generator-holdout
   evaluation (below) possible at all: there is no external state to leak
   through.

## Making "weight learning" experimentally identifiable

For a given training episode, the question "did the WEIGHTS get better, or
did some other kind of state?" only has a clean answer if the system has
no other kind of state to confound it. Brain doesn't: no skill library, no
vector DB, no scratchpad, no conversation memory. So brain's honest
contribution to the standard four-configuration ablation is exactly two of
the four cells, and it gets them for free rather than having to engineer
them:

| Configuration | Weights | External learned state | Who measures it |
|---|---|---|---|
| `BASE`    | old | wiped    | brain - the natural starting state |
| `WEIGHTS` | new | wiped    | brain - the natural state after training, since there IS no external state to wipe |
| `FULL`    | new | retained | sven - needs a real skill library / episodic memory to retain |
| `CONTROL` | old | retained | sven - same reason |

`WEIGHTS - BASE` is therefore the weight contribution, measurable in brain
alone, with no sven dependency and nothing blocked on sven's side. Building
a fabricated "external learned state" stand-in inside brain just to fill in
`FULL`/`CONTROL` would violate the same "evaluate honestly, don't fabricate
a stand-in" discipline that deferred P6 - so those two cells are explicitly
left to whoever wires this into a real agent harness (sven), not simulated
here.

## The five procedural environments

Every environment: seed-generated per run, an exact in-process executable
oracle (so any reward is programmatic and re-scorable, never a judge model),
and an effectively unbounded space of unseen instances so a static
benchmark cannot be memorized once and reused forever.

| Env | The policy must learn | Primary diagnostic |
|---|---|---|
| `AlienLang`  | a randomly generated vocabulary + grammar, including novel compositions never demonstrated | basic weight acquisition (P1-2 in the ladder below) |
| `AlienAPI`   | a synthetic tool/object API's argument rules and ordering constraints | procedural learning, implicit state machines |
| `AlienAlgo`  | a hidden deterministic transformation from input/output pairs alone | rule induction vs. memorization |
| `AlienWorld` | planning over a grid/state environment | long-horizon credit assignment |
| `AlienGame`  | a symmetric two-player game with a cheaply computable winner | self-play bootstrapping without cyclic false progress |

`AlienAPI` is not started from scratch: it extends `crates/bench/src/
toolcall.rs`, which already has a calibrated, measured-learnable task in
this exact shape (verb-to-tool routing plus name-routed argument copying
past distractors, 2 layers / d_model 64, exact-match 1.00 across three
seeds in minutes on CPU). It needs one small additive change -
`gen_sequence`/`Layout` are private today; a `pub fn task(&self, rng)` is
the only new surface `bench` needs to expose.

## Sixteen diagnostics, and what each one catches

Numbered as originally specified, grouped by what they test:

**Acquisition (1-5)** - alien vocabulary, an unknown mathematical
transformation, a novel DSL (grammar + semantics from a manual and
execution feedback), an unknown API (synthetic tool semantics), a stateful
protocol (correctness depends on command *sequence*, not just individual
commands - tests whether the learned representation contains a state
machine rather than isolated command associations).

**Continual learning (6-11)** - compositional transfer (skills A, B, C
taught separately; does A+B+C succeed zero-shot?); interfering twin skills
(two skill sets that actively conflict, e.g. `foo` means increment in one
context and decrement in another - the sharpest catastrophic-forgetting
probe available, sharper than accuracy averaged over unrelated benchmarks);
rule revision (a protocol changes version; correct behavior is *replacing*
the old rule conditionally, not accumulating both - retention without the
ability to revise is its own failure mode); a long sequential skill stream
(the retention matrix, below); the plasticity test (does a skill from the
*same* difficulty distribution as skill 1, introduced after N update
cycles, take more gradient steps to reach a fixed accuracy? - a different
failure from forgetting, and the literature is explicit that it needs
separate instrumentation); sparse old-skill retention (a skill taught once
early and never rehearsed - catches a replay/consolidation mechanism that
accidentally makes retention proportional to recency of exercise).

**Autonomous improvement (12-15)** - oracle-only discovery (the agent gets
an environment, a verifier, and a training ability, but no dataset - it
must generate its own exercises, attempt, verify, filter, and construct
training data unsupervised; the loop-mechanics gate in P18 already asserts
"no label enters training", which is exactly the property this diagnostic
is stress-testing at scale); self-play with an **expanding league of
historical checkpoints**, not just the immediately-previous generation -
without the league, `A beats B, B beats C, C beats A` reads as monotonic
improvement when it is actually cyclic specialization; poisoned self-data
(5-20% of self-generated training examples are subtly wrong - does true
held-out performance diverge from training accuracy, and does the system
react?); a deliberately hackable evaluator (a proxy reward with a cheap
exploit available alongside the real solution - this is exactly why P16's
gate includes a non-degeneracy/entropy check and why the evaluator lives
outside the crate the training loop can introspect).

**Meta-improvement (16)** - learning-to-learn: tasks drawn from a common
meta-distribution, measuring `C_i` = compute/examples needed to reach a
fixed accuracy on the *i*-th task in the stream. `C_100 ~ C_1` is a healthy
lifelong learner; `C_100 > C_1` is plasticity loss (already covered by
diagnostic 10); **`C_100 << C_1` is the interesting one** - it means the
system has learned something about *how to learn* (representations, a
better training-data-selection strategy, better hyperparameter choices)
that transfers to acquiring the next, unrelated skill faster. This is the
earliest experimentally clean signature worth calling proto-RSI, and it is
a curve to plot, not a threshold to pass/fail.

## Artifacts

A single scalar score would erase exactly the information this gauntlet
exists to surface, so it doesn't produce one:

- **Retention matrix** `M[i,j]` = performance on skill *j* immediately
  after training on skill *i*, for a stream of *N* skills. A healthy system
  renders as a filled lower-triangle (everything learned stays learned or
  improves); a forgetting system renders as a diagonal stripe (only the
  most-recently-trained skill scores well). This is the primary
  continual-learning artifact - visual, not a single averaged number that
  can hide a stripe inside a mediocre-looking average.
- **Plasticity curve** `C_i` vs. cycle index (diagnostics 10 and 16).
- **Metric vector**, not a weighted sum: acquisition, final retention, max
  forgetting, backward transfer (does learning skill *i* improve
  performance on an earlier related skill *j*? - a stronger property than
  merely not-forgetting), forward transfer, sample efficiency, compute
  efficiency, plasticity, compositionality, generalization, weight
  contribution (`WEIGHTS - BASE`), interference, self-generated-data
  quality, verifier integrity (honest-success rate vs. exploit-success
  rate under diagnostic 15), autonomy (how much external supervision an
  improvement cycle needed), meta-learning rate (`dC_i/di`). Shaped to fit
  `bench::EvalReport`'s existing axis-scores map rather than inventing a
  second report format.

## Three holdout tiers

Applied per environment, per skill:

1. **Instance holdout** - same generated rules, unseen input instances.
   Catches plain memorization; the weakest tier, always required as a
   floor.
2. **Rule holdout** - a freshly generated rule set from the same task
   family (a new vocabulary, a new transformation, a new API surface).
   Catches failure to abstract the underlying pattern rather than
   memorizing this run's specific instantiation of it.
3. **Generator holdout** - a second, independently written generator for
   the same task family, producing structurally equivalent but
   differently-distributed instances. Catches exploitation of incidental
   artifacts of one specific generator implementation (token-frequency
   quirks, a particular random-seed structure, etc.) - the hardest tier,
   and the one most directly relevant to a self-improvement loop, since a
   loop that has been training against its own generator's quirks for many
   cycles is exactly what this tier is built to expose.

## The L0-L8 ladder

A way to say precisely how far up the self-improvement hierarchy a given
result actually reaches, so "the benchmark score went up" and "this system
is self-improving" don't get conflated:

| Level | Capability |
|---|---|
| L0 | Static inference |
| L1 | Humans create training data and fine-tune |
| L2 | Agent gathers data; humans initiate training |
| L3 | Agent identifies a weakness and trains itself |
| L4 | Agent manages forgetting and rollback itself |
| L5 | Agent invents curricula and training strategies |
| L6 | Agent improves its own ability to learn future skills |
| L7 | Agent improves the mechanism responsible for L6 |
| L8 | Sustained recursive improvement under a fixed external compute/data budget |

**L6 is where an RSI claim starts being worth taking seriously** - going
`55 -> 65 -> 75` on a benchmark through repeated fine-tuning is ordinary
self-training and stops at L3-L4. Landing `.agents/roadmap/self-improve.md`
P7-P18 (the programmatic gate, lineage, rollback via reject, and the
mixture/replay anti-forgetting mechanism) gets brain to a credible L3-L4.
The gauntlet's diagnostics 10 and 16 are what would produce L6-shaped
evidence, if it exists.

## Calibration constraints - why gates run on small full-parameter models, not tiny LoRAs

Measured, in-repo, and binding on every threshold this gauntlet sets:

| Source | Finding |
|---|---|
| `crates/qwen3/tests/lora_learning_gate.rs` (doc comment) | a tiny qwen3 LoRA **plateaued at ~40% *training-set* accuracy** on an algorithmic rule across 800-3000 steps, d_model 16-64, and every learning rate tried, with predictions collapsing onto a handful of attractor classes. **Do not build a gate on a tiny LoRA learning an algorithmic rule - it's already been tried and it doesn't work in a test budget.** |
| `crates/bench/src/toolcall.rs` | 2-layer / d_model-64 full-parameter GPT, 800 steps: exact-match **1.00** across seeds {1337, 7, 42}, chance 0.0017, minutes on CPU |
| `crates/bench/src/dyck.rs` | same size, 600 steps: **~0.99**, chance 0.333 |
| `crates/bench/src/parity.rs` | same size, 500 steps: **1.00**, chance 0.5 |
| `crates/bench/src/mod_add.rs` | marked `informational()` by its own author - held-out accuracy is a grokking phase transition that has scored **below chance on one seed** at the same budget others reached ~0.7. **Disqualified as a gate anywhere in this file.** |

Consequence: every gauntlet diagnostic that needs a pass/fail gate (not a
plotted curve) runs on small full-parameter architectures at
`bench`-calibrated sizes, and each module gets a measured `## Calibration
(CPU / Cranelift JIT backend)` doc-comment block - seeds, measured curve,
wall clock - written from real runs during implementation, the same
convention every `bench` module already follows. Thresholds are set from
that measurement, never assumed in advance.

**Named so they don't get built:** `mod_add` as any kind of gate (see
table); any cold-start environment whose correct answer exceeds ~3 tokens
(exploration probability is `V^-L` - rejection sampling returns literally
zero positive examples long before that, which rules out sorting,
countdown, and expression-to-target as cold-start RL environments, though
they remain fine as pure-SFT tasks with labels); asserting
`candidate != incumbent` as a pass criterion (`lora_learning_gate`'s own
words: "a statistic a broken result also satisfies is not a check"); a
per-cycle threshold in place of an end-to-end delta plus the gate's own
statistical decision (per-cycle thresholds are how a demo becomes flaky
under normal training-run variance).

## Sequencing

Not started. Depends on `.agents/roadmap/self-improve.md` P10 (rollout),
P11 (`Environment`/`Verifier`), P12 (GRPO/RFT), and P16 (the gate) landing
first - the gauntlet is a set of `Environment`/`Verifier` implementations
plus a curriculum runner and the retention-matrix/plasticity artifact
collection over that substrate, not a parallel training mechanism. When
picked up, `AlienAPI` (extending `toolcall`) is the natural first
environment - it is the only one of the five with an already-calibrated,
measured-learnable base task to extend rather than calibrate from zero.
