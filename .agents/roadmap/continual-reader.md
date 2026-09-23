# continual-reader

**Status: designed. R0 landed 2026-09-23; R1 onwards not started.** A
self-driving continual learner as a brain sample: point it at a directory or a
stream, leave it running unattended, and it decides what to learn, what to
refuse, and what it can still do.

Distinct from `.agents/roadmap/continuous-learning.md`, which is brain's half
of a cross-repo loop where fact extraction and scheduling live outside this
repo. This one has nothing else in the loop. Its whole point is that the
learner audits itself.

---

## Part 1 - what brain has today, and the seven things that stop it

brain's continual-learning machinery is strong on measurement and weak on
mechanism. `crates/rl/src/continual.rs` already produces a retention matrix, a
BWT number, a fresh-adapter plasticity control, a joint-training capacity
oracle and a null-gate arm, all behind a pre-registered PASS/FAIL block. The
`Regime::Sft` composition was measured to accumulate (ACC 0.932 and 0.815
across two seeds, 12/12 and 10/12 cycles promoted) where the per-cycle GRPO
composition was measured not to (ACC 0.271, below the untrained base's 0.354).
`crates/promote` holds a four-bar gate with an exact paired sign test.

None of that is in question. What stops it being a continual learner:

**L1. It is a study, not a process.** `run_study` runs a fixed N-cycle
curriculum and terminates. There is no unbounded stream, no resumption, no
answer to "what happens at cycle 10,000".

**L2. Evaluation cost is quadratic in history.** The anchor suite is exactly
every earlier cycle's probes (`continual.rs:1016`), decoded for BOTH arms at
every cycle. At `eval_per_cycle = 48`, cycle 1,000 decodes 96,000 anchor
completions to make one promote decision, and cumulative eval cost is
`O(N^2)`. This is correct for a 12-cycle study and fatal for a stream. It is
the single hardest problem in turning the study into a process, and neither
this repo nor any approach it has been compared against bounds it or states a
detection latency instead.

**L3. The per-block retention bar was computed and thrown away.**
`promote::document::document_gate_config()` inherited `max_block_drop:
f64::INFINITY` (OFF) while `continual::run_study` was already handing the
gate one block per earlier cycle (`continual.rs:1042` sets
`anchor_block_len: eval_per_cycle`; `improve.rs:366` builds the pairs). So
the pooled `anchor_budget` was the only retention bar, and **a pooled mean
goes blind as history grows**: with `k` earlier cycles a block dropping `d`
moves the pooled mean by `d/k`, so the drop is invisible once
`k >= d / anchor_budget`. At the default budget of 0.02 that is 15 cycles to
hide losing 30% of one earlier cycle, and **50 cycles to destroy an entire
earlier cycle with the pooled bar never firing**. The protection weakens
exactly as the run gets longer, which is precisely backwards for a continual
learner. FIXED (R0, 2026-09-23): `document_gate_config()` now arms
`MAX_BLOCK_DROP = 0.20`, derived from the sampling error of a
48-probe block mean rather than picked.

**L3b. The entropy bar being off is NOT the same kind of finding, and an
earlier draft of this file got that wrong.** `min_entropy_ratio: 0.0` in
`crates/rl/examples/continual_learning.rs:157` and
`crates/rl/tests/continual_study.rs:135` is a documented, measured decision
with a correct reason: the position-copy family has exactly one correct
completion per prompt, so a policy that has SOLVED a rule decodes it greedily
with near-zero entropy and is indistinguishable by entropy from a collapsed
one. A candidate scoring a perfect 1.000 was rejected as `Degenerate` at
entropy ratio 0.078 before this was found. The check is not dropped, it is
REPLACED by a sharper one: `PREREG_MIN_DISTINCT_FRAC = 0.50` over the number
of distinct greedy completions across the probes. A collapsed policy emits
one completion for every prompt; a correct one emits a different completion
per prompt.

**This has a direct consequence for the design below**, which is why it is
recorded rather than just corrected: this sample's battery is also
deterministic-verifiable (a command-line invocation has one canonical form),
so the entropy ratio is the WRONG degeneracy instrument here for the same
reason. V14 uses distinct completions, not entropy. See Part 3.1.

**L4. Capacity is one adapter at a fixed rank.** `overlay_adapter` produces a
single lineage. Everything ever learned must fit in one low-rank subspace, so
saturation is inevitable. `joint_oracle` DETECTS capacity exhaustion and
nothing FIXES it: the only knobs are rank, rehearsal weight and learning rate,
all of which slide along a retention/plasticity trade curve instead of moving
it.

**L5. The regime that works is the one whose anti-shortcut check is vacuous.**
`continual.rs`'s own header says it: `Regime::Sft` trains on the environment's
known-correct completions, so `improve::assert_trained_spans_were_sampled` is
vacuous by construction under it. The structural defence against learning a
cue-independent shortcut does not apply to the measured regime.

**L6. The curriculum is synthetic and difficulty-invariant by construction**,
and the header is explicit that "that control was bought by removing exactly
the properties that break real systems". Real documents have drifting
difficulty, repeated content, contradictions and adversarial spans. Every
control that depends on difficulty invariance has to be rebuilt differently.

**L7. No live operation, no order independence, no seed robustness.** Stated
in the header as things a passing run does not prove. A learner that runs
unattended for weeks needs all three to be measured, not disclaimed.

---

## Part 2 - the mechanism changes

Seven, each aimed at a numbered limit. They are ordered by ratio of effect to
cost, not by dependency.

### M1. Arm the per-block retention bar (DONE, R0, 2026-09-23)

Fixes L3. `promote::document::document_gate_config()` now sets
`max_block_drop: MAX_BLOCK_DROP` (0.20), derived from the sampling standard
error of a `MIN_HELD_OUT_PROBES`-probe block mean (`0.5/sqrt(48) = 0.072`, so
the bar sits ~2.8 SE out) rather than picked. Two paired tests in
`crates/promote/src/document.rs`: one where a single earlier cycle collapses
behind a healthy pooled mean and must now be rejected as `BlockRegressed`
naming the block, and one where ordinary per-block wobble inside one standard
error must still promote. The first test also asserts that the POOLED bar
alone promotes the same input, which is what makes the per-block bar
load-bearing rather than redundant.

Callers with no block structure are unaffected by construction: `qwen3`'s
`lora_gate` passes an empty `anchor_blocks` and the bar cannot fire on blocks
that were never supplied.

**Not done, deliberately: the two recorded studies keep their configs.**
Arming a bar inside `crates/rl/examples/continual_learning.rs` or
`crates/rl/tests/continual_study.rs` would change what those runs promote,
and a recorded result must reproduce rather than silently change. Re-running
them with the bar armed is a NEW measurement with its own before/after, not a
fix, and it is not what R0 was for. `min_entropy_ratio` stays at 0.0 in both
for the reason recorded in L3b.

Under M2 a block becomes a SAMPLED mean rather than a complete one, so its
noise grows and `MAX_BLOCK_DROP` has to be derived from the realised sample
size instead of the floor. That is R5's work, not R0's.

### M2. Bound the retention audit, and report a detection latency instead of a
### guarantee

Fixes L2. A fixed decode budget `B` per cycle, split:

- **Canary half**: a fixed-size stratified sample of the probe bank, refreshed
  every `R` cycles. Always evaluated. Catches a broad regression immediately.
- **Rotating half**: a deterministic schedule that covers the entire bank
  every `ceil(2N/B)` cycles, so every past episode is re-checked within a
  bounded interval.

The honest claim this buys is not "nothing is forgotten". It is **"any
regression larger than `d` is detected within `ceil(2N/B)` episodes"**, with
`d` derived from the block sample size. That number is reported as a
first-class output beside the retention matrix. A learner that claims perfect
retention over an unbounded stream without bounding its own audit is claiming
something it did not measure.

Priority weighting on top of the rotation: episodes whose last score sat
nearest the pass/fail boundary, and episodes that regressed and recovered, are
resampled sooner. Deterministic given the seed, so M7 still holds.

### M3. Capacity grows, and the joint oracle is what triggers it

Fixes L4, and replaces heuristic growth brakes with evidence. When the promote
rate drops, the question "is this interference or is this saturation" has two
opposite fixes, and `joint_oracle` already answers it:

| joint oracle on episodes `1..k` | sequential run | diagnosis | action |
|---|---|---|---|
| passes | fails | interference | raise rehearsal, lower rate |
| fails | fails | capacity | add an adapter |
| passes | passes | healthy | nothing |

So growth is triggered by a measurement rather than by tuned constants. The
oracle is expensive, so it runs on a schedule and on promote-rate drop, not
every cycle; its cost comes out of the same budget `B` and is reported.

Capacity itself is a **pool of LoRA adapters over a frozen base**.
`model::dispatch::LoraW` already documents that a stack of adapters over one
linear is a single rank-`sum(r_i)` correction
(`sum_i s_i B_i A_i = [s1B1|s2B2|...].[A1;A2;...]`), so selecting a working set
is selecting which pairs concatenate. No new kernel. Selection happens at
segment granularity, not per token.

The base stays frozen, so forgetting in the base is zero by construction
rather than by a tuned constant. **The trade, stated not hidden:** a frozen
base cannot acquire new low-level representations - a new script, new byte
statistics, a new modality. A design that keeps the base plastic can. For a
reader over documents in languages the base already models the trade is
strongly favourable; for "a model trained from scratch that is entirely yours"
it is not, and this sample should not claim that ground. brain can train from
scratch on its own kernels (`gpt2`, `toymoe`) if that becomes the goal.

Retirement is **archive, never delete**. A capability asked for once a quarter
is not a dead capability, and deletion is the one action a self-driving learner
cannot undo. Liveness is decided by staleness (has anything addressed it), not
by a contribution score: a unit that is selected constantly and contributes
little per use reads as dead on a contribution test, and a unit nothing has
wanted for months reads as alive.

### M4. An anti-shortcut check that is not vacuous under teacher forcing

Fixes L5. Since SFT trains on known-correct completions, span-sampling checks
cannot help. The replacement is a **counterfactual probe**: for every probe,
a minimally edited variant whose correct answer differs. Learning the mapping
passes both, each with its own answer. Memorising the surface passes the
original and answers the counterfactual with the original's answer. That is
programmatic, exact-match, cheap, and works precisely under the regime the
old check could not reach.

### M5. Every increment is attributable and individually revertible

Each promoted adapter carries `improve::AdapterMeta` provenance: episode id,
source content hash, the frozen probes, the gate report that admitted it. "Undo
what it learned from that document" is a file operation against a named
adapter, not a rollback to a checkpoint that also discards everything since.

### M6. Rehearsal is a bounded reservoir over promoted episodes

brain measured that rehearsal is what makes accumulation work at this scale,
so the reservoir is not optional. It is reservoir-sampled over PROMOTED
episodes only, deduplicated by content hash, hard-capped, and mixed per
`Regime::Sft`. Bounded is the point. Disabling it is an arm, not a dead code
path.

### M7. Determinism is the instrument

Same seed and same bytes in, bit-identical adapter out. brain's kernels are
deterministic and this is already gated repo-wide. Every claim the sample makes
is reported beside its seed-repeat delta, so an effect is never confused with
dispatch noise. A learner whose run-to-run variance exceeds its claimed effect
has not measured the effect.

---

## Part 3 - validating it end to end

This is the part that matters. A self-driving learner that grades its own
homework is worthless, so every claim below is paired with a control that can
produce the opposite answer.

### 3.1 The failure-mode catalogue

Each row: what goes wrong, what it looks like from inside (which is why it is
not caught by accident), the control that discriminates it, and the assertion.

| # | Failure mode | Looks like | Discriminating control | Assertion |
|---|---|---|---|---|
| **V1** | **Probe leakage.** The probe's answer appeared in a trained span. | A clean, large, real-looking gain. | n-gram containment check of every frozen probe answer against every trained row, not id-hash disjointness (real text repeats where synthetic ids do not). | `no_probe_answer_ngram_appears_in_any_trained_row` |
| **V2** | **Format learning.** It learned the probe template, not the content. | Probe pass-rate rises; nothing else does. | **Paraphrase probes**, generated at ingest from a different template, never trained on, scored separately. | paraphrase pass-rate must move with template pass-rate within a stated tolerance |
| **V3** | **Surface memorisation under teacher forcing.** | Every probe passes. | **Counterfactual probes** (M4). | passing the original while answering the counterfactual with the original's answer is a FAIL, reported per episode |
| **V4** | **The gate is decorative.** Promotions carry no information. | A healthy promote rate. | **Null-gate arm** (`GatePolicy::CoinFlip`), already implemented. | gated arm separated from null arm beyond seed noise, else the run does not count |
| **V5** | **The gate is inverted.** It promotes noise. | Same as V4. | **Shuffled-label arm**: episodes trained against probes randomly reassigned from other episodes. | promote rate on the shuffled arm must be indistinguishable from chance |
| **V6** | **Silent forgetting outside the anchor suite.** | Retention matrix is clean. | **Independent battery** from `crates/bench` (`mod_add`, `mqar`, `toolcall`, `mad_recall`), never trained on, never gated on, never tuned against. | battery score must not regress past its own budget across the whole run |
| **V7** | **Anchor rot.** After N promotions every adapter has been implicitly selected against the anchors, so the anchors have become a training set. | Anchors look perfect; everything else drifts. | **Reserve anchors**: a fraction held out of every gate, swapped in on a schedule. | reserve and gated anchors must agree within tolerance; divergence means the gated set is overfit |
| **V8** | **Loss of plasticity.** Cycle 10,000 cannot learn any more. | Promote rate falls; looks like "nothing new to learn". | **Fresh-adapter control** (`xfer_vs_fresh_control`) at every audit point: a brand-new adapter on the same episode at the same budget. | incumbent-continued must not score materially below fresh |
| **V9** | **Capacity saturation read as forgetting.** | Identical symptoms to V8, opposite fix. | **Joint oracle** (M3). | the diagnosis table in M3 is asserted, not inferred |
| **V10** | **Difficulty drift.** "Episode 900 scored lower" vs "episode 900 was harder". Synthetic curricula engineer this away; real documents cannot. | A downward trend that looks like decay. | **Per-episode zero-shot column**: every episode's probes scored by the UNTRAINED base at ingest, frozen forever. Every later number is a delta over that episode's own baseline. | no absolute score is ever compared across episodes |
| **V11** | **Order dependence.** The result is an artefact of the stream order. | A good run that does not reproduce. | **Order-permutation arm**: the same episode multiset, different order, same seed. | final retention diagonals agree within seed noise |
| **V12** | **Seed noise mistaken for effect.** | A small consistent-looking win. | **Seed-repeat** (M7) plus a multi-seed arm for anything stochastic. | every reported effect exceeds the measured seed-repeat delta |
| **V13** | **Probe-selection gaming.** Ingest picks easy spans, so everything passes. | Uniformly high pass rates. | Probe spans selected by a pure function of the content hash, plus a **blind fraction** the selector never sees. | blind and selected pass-rates agree within tolerance |
| **V14** | **Degeneration.** Output collapses to one repeated answer. | Probes still pass; generation is unusable. | **Distinct greedy completions** across the battery, NOT the entropy ratio - on a deterministic-verifiable task a solved model and a collapsed one are both low-entropy, and L3b records a perfect candidate rejected at ratio 0.078 for succeeding. Plus a repeated-n-gram tripwire. | distinct-completion fraction checked at every promote, not sampled |
| **V15** | **Poisoned or junk episode absorbed.** | One quiet promote. | Adversarial injections: random bytes, and a **contradiction episode** asserting the negation of an already-promoted fact. | random bytes must reject; a contradiction must reject or flag, never silently flip an earlier probe |
| **V16** | **Audit starvation.** Eval grows until the loop does nothing else. | Throughput quietly collapses. | Eval compute reported as a fraction of total compute, with the bound from M2. | the fraction stays under its budget and the detection latency is reported |
| **V17** | **Rollback that is not.** A failed live validation leaves the process changed. | No error is raised. | Byte-equality of live responses before staging and after rollback, not absence of an exception. | `a_rolled_back_stage_leaves_the_live_response_byte_identical` |

### 3.2 How they are scheduled, so this is affordable

Running seventeen controls every episode is not possible and not necessary.
Four tiers, and the tier is part of the claim:

- **T0, every episode, structural, near-free.** V1 (containment), V10
  (zero-shot delta), V13 (blind fraction), V14 (entropy + n-gram), V16
  (accounting). These are asserts and counters, not decode passes.
- **T1, every episode, alongside the real arm.** V3 (counterfactual), V4
  (null gate). These come out of the decode budget `B` and are the reason `B`
  exists.
- **T2, periodic audit, every `R` episodes.** V2 (paraphrase), V6 (independent
  battery), V7 (reserve anchors), V8 (fresh adapter), V9 (joint oracle).
  Expensive, scheduled, and the schedule is reported.
- **T3, pre-registered adversarial suite, run at fixed checkpoints.** V5
  (shuffled labels), V11 (order permutation), V12 (multi-seed), V15
  (injections), V17 (rollback). These are experiments with their own arms, not
  monitors.

A run that skipped a tier reports which one. A number produced without its
tier's controls is not a result.

### 3.3 The end-to-end acceptance run

One pre-registered assertion, written before the run, asserted in a test
rather than printed to a report, per this repo's convention that a number
nothing checks goes stale silently:

> From a cold start, N episodes of real documents read unattended in one
> process that also served requests throughout.
>
> 1. Every rejection names its cause; the promote rate is reported with the
>    null-gate arm beside it and the two are separated beyond seed noise.
> 2. At episode N, every promoted episode's frozen probes, its paraphrase
>    probes and its counterfactual probes are scored, and the retention matrix
>    is reported as deltas over each episode's own zero-shot baseline.
> 3. BWT >= -epsilon, with the per-block bar armed so no single earlier
>    episode collapsed behind a healthy pooled mean.
> 4. The independent battery, never trained on and never gated on, has not
>    regressed past its budget.
> 5. Retention audit coverage is stated as a detection latency, not as a
>    guarantee.
> 6. The shuffled-label arm promoted at chance; the order-permutation arm
>    reproduced the diagonal; random-byte and contradiction injections were
>    rejected.
> 7. Eval compute stayed under its share of total compute.
> 8. The whole run reproduces bit-identically from its seed, and the reported
>    seed-repeat delta is smaller than every effect claimed.

Any one of these failing is a result, reported as such. The acceptance
criterion is that the run produces a defensible controlled number, not that the
number is positive.

---

## Part 4 - milestones

Test-first throughout. One commit each unless stated.

**R0 - arm the per-block retention bar. DONE 2026-09-23.** M1 above. Landed
as `promote::document::MAX_BLOCK_DROP` plus the paired fire/stay-silent tests,
verified red before green (the behavioural red was `expected
Reject(BlockRegressed), got Promote`). The `Degenerate` half of this milestone
was DROPPED after reading why the entropy arm is off: see L3b. It is replaced
by the distinct-completions check in V14.

**R1 - the episode stream. DONE 2026-09-23.** Landed as the new leaf crate
`crates/audit` (`brain-audit`) and its `stream` module, registered in
`scripts/gates/check-crate-layers.sh` so the leaf property is machine-checked
rather than intended. Eight tests, red before green.

Four properties are decided there and nowhere else, each with its reason in
the module doc: a document is read start to end and an episode never spans
two; document order is a seeded permutation of the SORTED paths (`read_dir`
order is not stable, so sorting first is what makes the seed the only source
of order, and the order-permutation control arm is then "same corpus,
different seed" rather than a second code path); text is decided by content
and never by extension; and an episode's identity is the digest of its own
bytes rather than of `(path, ordinal)`, because the questions identity
answers later are "have I read this content before" and "is this already in
the reservoir".

Two of the eight tests are the fire/stay-silent pair this file asks for
everywhere: a resume under the same corpus and config continues the
uninterrupted sequence exactly, and a resume after `episode_chars` changed or
the corpus grew is refused as `CursorMismatch` rather than silently reading a
different stream. The binary test also asserts both halves - a text file named
`.bin` is read, a binary file named `.txt` is skipped - since either half
alone is equally consistent with an extension test.

**Recorded, not built: the corpus is fixed for the life of a stream.** A
`Cursor` is a document index, so a growing directory shifts what an index
means. Following a growing corpus needs an append-only document identity
rather than an index; that is a different mechanism and R1 refuses the unsafe
case by name instead of pretending to handle it.

**R2 - probes, and the three probe families. DONE 2026-09-23.**
`crates/audit`'s `bank` module. A probe is held-out LINE CONTINUATION: some
lines are withheld from training and become the questions, with the lines
before them as the prompt, so verification is an exact match and never a
judge. Eight tests, red before green.

**Only two of the three families can be built without a model, and the crate
says so instead of pretending.** `Literal` and `Counterfactual` are built
here; `Paraphrase` needs to know what the text MEANS, which is either a
model or a corpus that generated itself and kept its own semantics.
`ProbeSet::coverage` reports zero paraphrase coverage rather than letting a
run quietly claim a control it never had. The sample's generated tool is
exactly the corpus that CAN supply them, which is a further reason the
battery lives there and not here.

**Correction to this file's own V13.** A pure-hash selection does not on its
own answer probe-selection gaming: it is the selection RULE that can be
biased, not merely its reproducibility, and a rule that prefers easy lines
stays perfectly deterministic while making everything pass. So a
`SpanSelector` exists to be audited rather than trusted - a second draw is
taken uniformly from the lines it REJECTED, both are withheld from training,
and `selection_bias` reports selected-minus-blind pass rate. Tested as a
pair.

Containment is substring, not equality, because real text restates itself in
ways a synthetic curriculum never does. A property worth knowing before it
looks like flakiness: a refusal is about THIS DRAW, not the episode in the
abstract, so the same file can build cleanly at one seed and be refused at
another. Recorded in the module doc.

V1, V3, V10 and V13 each have their fire and their stay-silent test.

**R3 - the adapter pool, its working set, and its archive. Commit 1 of 2
DONE 2026-09-23.**

**Corrected placement.** This file put the pool in `crates/model` on the
assumption it needed delta arithmetic. It does not: `model::lora::
RuntimeDelta::absorb` already IS the rank concatenation
(`s1*B1*A1 + s2*B2*A2 = [s1B1|s2B2].[A1;A2]`), and adapter persistence with
`ModelCard` provenance already exists in `model::lora::device_adapter`. What
did not exist is the bookkeeping: which adapters there are, which are active,
which have gone unaddressed, and where a retired one went. All of that is
model-free, so it landed in `crates/audit`'s `pool` module with fast tests and
no device. Only the MATERIALISATION of a working set into a runtime
correction needs `crates/model`, and that is commit 2.

Seven tests. The three rules each have the pair that proves the rule is doing
work: a resident keeps its slot when demand merely reorders (and loads
nothing); a candidate displaces only when it is clear of `margin`, and does
not when it is inside it; a newcomer is protected for its `dwell` however
wanted the challenger, and displaceable immediately after. Retirement
archives by rename and never deletes, an archived adapter resurrects byte for
byte, and nothing currently resident can be retired at all.

**Found while building it, and it changes commit 2: an adapter's Adam moments
do not survive a round trip through its file today.** `model::lora::Pair`
owns `ma/va/mb/vb`, but `device_adapter::save_adapter` writes only the
`.lora_a`/`.lora_b` tensors and `Pair::from_ab` leaves the moments empty by
design - correct for the one-shot finetune that path exists for, where a
model trains, saves and folds once. In a POOL eviction is routine, so an
adapter evicted and re-admitted would restart its optimiser state every time.
That is the same failure this design named from the start, reached from the
other direction. Commit 2 closes it by having the pool's own file carry the
moments alongside the weights, as a superset of the existing format so
existing readers are unaffected.

**Commit 2 of 2 DONE 2026-09-23**, and smaller than this file expected,
because two thirds of it already existed.

*Materialising a working set needed no new code.* `RuntimeLora::new` already
merges every delta covering the same rectangle through `absorb`, so a working
set is `RuntimeLora::new(selected.flat_map(|a| a.deltas()))` and nothing
more. What was missing was a TEST: the identity
`s1*B1*A1 + s2*B2*A2 = [s1B1|s2B2].[A1;A2]` is what makes a working set cost
one correction instead of one per adapter, and it was documented and never
measured. Three adapters of ranks 2, 4 and 3 now fold to exactly the sum of
what each contributes alone, at merged rank 9, with the stay-silent half
asserting that adapters over DIFFERENT rectangles stay separate so "merge"
cannot quietly mean "merge everything". Stated honestly: those two are
characterisation tests of correct existing code, not red-then-green.

*The moments gap was real and is closed.* `lora::resumable::save`/`load`
persist `ma/va/mb/vb` alongside the weights, and `Pair` gains `has_moments`
and `restart_moments` so "this can resume" and "this would restart" are
distinguishable rather than implied. The format is a superset: readers that
select on the `.lora_a` suffix, which is what `fold_adapter_into` does, do
not match `.lora_a.m` and are unaffected, so a pool adapter can still be
folded by the ordinary path. The test is the property rather than the
plumbing - after a save and a load the NEXT optimiser step lands
bit-identically where it would have without the round trip - with the
contrast that makes it worth asserting: the same weights without their
moments take a visibly different step.

**R4 - triage and the per-episode gate. DONE 2026-09-23.**
`crates/audit`'s `triage` module: the four filters of Part 8, each naming the
stage it stopped at, so the share of the stream reaching each is countable. A
triage that passes everything to the gate is one that is not working, and the
`stage()` accessor is what makes that measurable rather than asserted.

`reader_gate_config()` is deliberately NOT a new set of numbers - it is
`document_gate_config()`, per-block bar armed, because a reader is a
continual learner in exactly the sense that config was pre-registered for. A
test asserts the bar is in force HERE, not merely available, by collapsing
one of twenty earlier episodes behind a healthy pooled mean.

**A fourth filter this file did not have: an episode too small to be
evidence.** An exact paired sign test reaches `p <= 0.05` only from five
clean wins upward (`2^-5 = 0.031`), so an episode yielding fewer probes than
that cannot produce a significant result however well the model did.
Promoting on it would be promoting on a number that was never capable of
being evidence. `MIN_EPISODE_PROBES = 12` leaves room to lose a few rather
than requiring a perfect sweep of the minimum, and such an episode is
ACCUMULATED for a later attempt rather than rejected.

**Correction to Part 8's T-A.** That section said to reject text a cheap
compressor cannot compress. Measured, that is wrong: deflate takes any text
over a restricted alphabet towards `log2(symbols)/8` whether or not there is
a pattern in it, so random base64 lands near 0.75 and random hex near 0.5
while containing nothing to learn. The screen measures the deflated size
against the text's OWN order-0 entropy instead, leaving how much compression
found that the symbol frequencies alone did not. A test pins the separation
AND asserts that a raw ratio would not have produced it, so the extra term
cannot quietly become unnecessary without something failing.

Also corrected: degenerate repetition (one line five hundred times) scores
HIGH on structure and is correctly not a filter-A concern. It is caught at
filter B, where a model that predicts it trivially reports a low loss. Each
filter answers one question.

**R5 - the bounded retention audit. DONE 2026-09-23.**
`crates/audit`'s `schedule` module. Nine tests, red before green.

**Design change made while building it, and it matters.** M2 said the
rotating half should carry a priority weighting. It must not. Anything that
can reorder rotation can DELAY an episode, which turns an exact coverage
guarantee into a hope, and the guarantee is the entire product here. So the
two halves have one job each and neither touches the other's: rotation is
strict round-robin and yields `L = ceil(N / rotating)` exactly, while the
canary is where priority is spent and makes no coverage claim at all. Keeping
them apart is what lets the reader be opportunistic without weakening the
only thing it promises. Two tests hold that line: rotation covers the bank
within the reported latency, and one tick FEWER does not, so `L` is tight
rather than a comfortable over-estimate.

**`MAX_BLOCK_DROP` re-derived, as R3 flagged.** `block_drop_bar(m)` returns
the pre-registered 0.20 or two standard errors of the realised sample
(`1/sqrt(m)`), whichever is looser. It can only loosen, never tighten: a
thinner sample buys a shorter latency and pays for it in the smallest
regression it can still see. Holding 0.20 against a 12 probe block would
have rejected candidates for sampling noise, which is the classic way a
tightened audit makes a system look worse than it is.

**One real bug the tests caught**: the first implementation bounded rotation
by the budget but not the canary, so an oversized `canary_blocks` overspent
by 5x. An oversized canary now costs the canary, never the guarantee and
never the budget.

**R6 - the bounded rehearsal reservoir. DONE 2026-09-23.**
`crates/audit`'s `reservoir` module. Six tests, red before green.

Vitter's Algorithm R over PROMOTED episodes only, capped, deduplicated by
content. The admission rule lives in the reservoir rather than at each call
site, so there is one place it can be got right: rehearsing something the
gate refused would let a rejected episode influence every later cycle
through the back door.

**The test that earns its keep is the distribution one.** A cap and a dedup
check pass equally well against the two easy wrong implementations, keep the
first `cap` and keep the last `cap`, and both are actively harmful: a
reservoir biased to the recent rehearses precisely the material least at
risk of being forgotten, and one biased to the oldest never rehearses
anything learned since. So uniformity is asserted on the mean held position
over 2000 offers and eight seeds, with the failure message naming what each
wrong answer would have produced.

A duplicate deliberately does NOT advance the seen counter. That counter is
the denominator the retention probability is drawn against, so counting a
duplicate would lower every later episode's chance of being held, for an
episode that was never a candidate.

`draw` takes its own seed rather than advancing the reservoir's, so asking
the reservoir a question cannot alter the experiment: the same cycle asked
twice gets the same mix, and drawing leaves what is retained untouched.

**R7 - growth, triggered by the oracle. DONE 2026-09-23.**
`crates/audit`'s `growth` module. Nine tests, red before green.

**Scoping correction: one commit, not two.** The planned second commit, "the
pool growth it drives", is a single `Pool::admit` call in a reader loop that
does not exist yet. Writing it now would be a call site with no caller, so
it folds into the loop milestone instead. The decision machinery is the part
that can stand on its own, and it is model-free.

**M3's table gained a fourth row while being implemented, and then a fifth.**
The oracle is supposed to bound the sequential run from above, so an oracle
scoring BELOW it means the measurement is untrustworthy: an under-trained
oracle, a mismatched probe set, a seed with too much variance. Both remedies
are wrong under a broken measurement, so the action is `Hold` with a stated
reason rather than being folded silently into "healthy". The fifth case is
subtler and was caught only by noticing an untested branch: when BOTH arms
fail and the oracle is still underneath, a plain "both failed" reading calls
it saturation and grows the pool, but the ordering says the measurement is
broken. Ordering wins, and a test now pins all three of that row's
outcomes.

**The cooldown governs growth only.** Raising rehearsal costs no memory, so
putting the cheap remedy behind the expensive one's timer would leave a
diagnosable problem untreated for the length of a cooldown it has nothing to
do with.

The promote rate is measured over a WINDOW, not the run. A reader that
promoted well for a thousand episodes and has promoted nothing for fifty has
a problem now, and a lifetime average would hide it for a long time. A
part-window returns `None` rather than a rate, so the oracle's cost is never
paid on the strength of the first few episodes of a run.

**The reader loop. DONE 2026-09-23**, and not a milestone this file had,
because the plan treated it as plumbing that would fall out of the parts.

`crates/audit`'s `reader` module ties all six together: screen, one forward
pass, freeze the probes, the evidence precondition, then train and gate.
Eight tests against a fake learner, red before green.

**It is generic over a `Learner` seam rather than over a model**, and that
is the decision that matters. Everything model-touching is four methods: a
forward pass, a training run, a decode per arm, and the oracle. Behind a
trait, the ORCHESTRATION - which is where the mistakes are - stays in a
crate with no device and no weights and a suite that runs in a sixth of a
second. It is the seam this workspace already uses for a scheduler generic
over a decoder and a study generic over an environment; the binding to a
real model is a thin implementation above this crate.

**A precondition on the oracle that neither M3 nor R7 had.** The oracle
bounds what a schedule could have achieved over what the reader has LEARNED,
so a run that has promoted NOTHING has nothing for it to bound. Such a run
looks exactly like a stalled one by promote rate, and asking would spend a
full training run to be told nothing useful. Found because the obvious test
(a reader that promotes nothing must ask why) failed: the guard was right
and the test modelled a case that cannot arise. A real stall is a run that
promoted for a while and then stopped, and both are now tested.

The headline property is asserted end to end: over sixty episodes the audit
cost of a step never exceeds the budget while the bank grows past forty and
the reported detection latency grows with it. That is the whole claim of
this design in one test.

**R8 - serve while learning, staged.** The forcing function for the
`stage`/`validate`/`commit`/`rollback` API that `continuous-learning.md` B7
recorded as missing from `crates/residency`. Tests: a promoted adapter changes
a live response without restart; V17. **Two commits.**

**R9 - the run directory.** One directory is the model: manifest, base
reference, adapter pool, archive, reservoir, probe bank, gate ledger,
retention matrix, audit schedule. Atomic write-then-rename throughout, best
state rather than latest. Tests: an interrupted write leaves a loadable
directory; a resumed run continues the same stream at the same episode.

**R10 - the adversarial and reproducibility suite.** T3 of 3.2: shuffled
labels, order permutation, multi-seed, injections. Tests are the arms.

**R11 - the acceptance run.** 3.3, asserted.

---

## Part 5 - honest scope

R0 is hours and improves a shipped result today. R1-R4 is the smallest cut
that produces a reader which actually refuses a bad document, and is worth
doing before anything else in the list. R5 and R7 are the two genuinely novel
pieces and the two most likely to need a second design pass; R5 in particular
turns a guarantee nobody can make into a latency anyone can check, and that
reframing is the main contribution of this file. R8 depends on a
`crates/residency` API that does not exist yet and should not be started until
R4 is green.

The claim this design is built to support is narrow and checkable: not that
the model never forgets, but that **every increment it keeps is one it can
justify, attribute and undo, and every capability it claims to retain is one
it re-checks within a stated interval.**

---

## Part 6 - the final shape

What exists when this is done, before any of it is built.

### 6.1 Where the code lives

`samples/README.md` rule 1: a sample depends on the `brain` SDK and nothing
else, and "a sample is not a test and not a fixture". So the machinery is
crates, the sample is a thin application, and the assertions are crate tests.

| | what | why there |
|---|---|---|
| **`crates/audit`** (new leaf, `brain-audit`) | the probe bank, the bounded audit schedule, the detection-latency computation, the control-arm registry, the run ledger | Model-free. Closure is `brain-data` and below, the same carve-out discipline that produced `crates/promote`, so a model crate can depend on it and `scripts/gates/check-crate-layers.sh` keeps it honest. Every control is unit-testable with no device and no weights. |
| **`crates/model`** | the adapter pool: on-disk set, working-set selection by rank concatenation, archive and resurrection, optimiser moments owned by the adapter | `lora.rs` and `dispatch.rs::LoraW` already live here and the pool is a property of LoRA, not of one architecture. |
| **`crates/rl`** | `reader.rs`: the stream process. Episode to candidate to gate to promote/reject to bank update to audit tick, plus the oracle-triggered growth decision | Needs a `Model`. Reuses `promote::gate` and `continual`'s matrix, BWT, fresh-adapter control and null-gate arm unchanged. |
| **`crates/sdk`** | `brain::ContinualReader`, feature `reader` | The primary surface, per the B9 correction: SDK first, no CLI verb. |
| **`crates/residency`** | `stage`/`validate`/`commit`/`rollback` | R8 only. Does not exist yet; nothing before R8 depends on it. |
| **`samples/learning/reader`** | package `sample-learning-reader` | The application. |

### 6.2 The sample

```
samples/learning/reader/
  Cargo.toml       brain = { workspace = true, features = ["reader"] }
                   [package.metadata.brain] max-brain-crates = <measured>
  README.md        what it demonstrates, how to run, what it needs
  fetch-data.sh    generates the demo corpus. No committed fixtures.
  src/main.rs      flag parsing, then one builder call
  src/corpus.rs    the labelled corpus generator
```

Five verbs:

```
sample-learning-reader battery  --model REF --run-dir DIR
sample-learning-reader read     <DIR> --run-dir DIR [--serve PORT] [--budget N] [--until N]
sample-learning-reader audit    --run-dir DIR [--tier t2|t3]
sample-learning-reader report   --run-dir DIR
sample-learning-reader selftest --run-dir DIR
```

`battery` is the headline and is covered in Part 7. `read` is the product:
unattended, resumable, serving while it learns. `audit` runs the periodic and
adversarial tiers on demand. `report` prints the retention matrix, the
detection latency and which tiers ran. `selftest` is the pre-registered
acceptance run of Part 3.3 over the generated corpus, exiting non-zero on any
FAIL. Self-audit is a capability of a self-driving learner, not a test harness,
which is why it is a verb.

**The model is named by `--model REF` and resolved through `crates/loader`**,
the same resolver `brain do` and every other sample uses
(`samples/imagegen/generate/src/main.rs` is the pattern:
`X::from_pretrained(&a.model)` under the `resolve` tier). So a reference that
is not on disk is fetched by brain's own model handler, and `brain pull` works
against it unchanged. There is no `--weights` flag and no path parameter.

Per samples rule 5 it runs with **no model downloaded** by default, against the
tiny CPU-runnable Qwen3 fixture that `crates/sdk/tests/document_study.rs`
already uses. In that mode it proves the machinery: every control fires when it
should and stays silent when it should, and every verdict matches its lane
label. It does NOT produce a learning claim, and the report says so the way the
document study's `preregistered: false` already does. A real `--model` is what
turns the battery number into a result.

### 6.3 The corpus: a tool the model has never seen

`fetch-data.sh` generates, from a seed, a **fictional command-line tool**:
subcommands, flags with types, required and optional, mutually exclusive
groups, and a man page per subcommand. Those man pages are the corpus. The
sample ships the generated grammar's **parser** as the verifier.

This is what makes the sample self-contained in the way `samples/README.md`
requires: no network, no committed fixture, no external tool, no compile step,
and verification is an in-process parse taking microseconds. It also makes the
zero-shot baseline **provably** zero rather than argued: the tool did not exist
before the seed was drawn.

Two tools are generated, `A` and `B`, and read in sequence. Whether learning
`B` destroys `A` is catastrophic forgetting in its most legible possible form,
and it is the question the retention matrix answers.

Every episode carries a ground-truth sidecar, so every verdict is checkable
against what should have happened.

| lane | content | expected verdict | proves |
|---|---|---|---|
| `learn/` | real man pages for tool A, then tool B | PROMOTE | the loop learns, and B does not destroy A |
| `repeat/` | an earlier page reworded | `AlreadyKnown` at triage, no gradient spent | duplicates cost one forward pass, not a train-and-gate |
| `noise/` | random bytes | `Unstructured` at triage, no model touched | junk is refused for free |
| `contradict/` | a page claiming a flag takes an argument when it does not | REJECT or promote-with-conflict-flagged | never silently breaks an already-learned flag |
| `degenerate/` | one line repeated five hundred times | REJECT on the distinct-completion floor | V14. NOT the entropy ratio: on a deterministic-verifiable battery a solved model and a collapsed one are both low-entropy (L3b) |
| `shortcut/` | a page containing a battery task's expected invocation verbatim | REFUSED AT INGEST | V1 containment fires before any training |
| `format/` | correct man-page shape describing nothing real | may promote, but paraphrase probes must NOT move | V2 separates format from capability |
| `counterfact/` | near-identical flags with opposite meaning (`--since` / `--until`) | both answered on their own terms | V3 catches surface memorisation |
| `rare/` | one subcommand documented in episode 3 and never again | still invokable at episode N | archive-not-delete, and the rotating audit reached it |

### 6.4 The run directory, which is the model

```
run/
  manifest.json    base ref, seed, schema version, the PRE-REGISTERED bars
  adapters/        promoted adapters + meta.json provenance per lineage
  archive/         retired adapters, resurrectable, never deleted
  bank/            per-episode probes: literal, paraphrase, counterfactual,
                   blind fraction, each with its frozen zero-shot baseline
  reservoir/       the bounded rehearsal sample
  ledger.jsonl     one row per episode: verdict, cause, every bar's number,
                   ground truth where known
  retention.json   matrix, BWT, detection latency, coverage, tiers run
  audit/           T2 and T3 arm outputs
```

Atomic write-then-rename throughout. Best state, not latest. A resumed run
continues the same stream at the same episode.

### 6.5 The tests, and the symmetry that makes them worth having

**Machinery** (`crates/audit` units, `crates/rl/tests/continual_reader.rs`):
every one of the seventeen controls gets **two** tests, one that makes it fire
and one that makes it stay silent. A control that can only ever pass is not a
control, and that symmetry is the single most important property of this suite.

Success cases:
- `a_learnable_episode_promotes_and_its_probes_still_pass_at_the_end_of_the_run`
- `a_rarely_referenced_capability_survives_to_the_end_of_the_run`
- `an_archived_adapter_resurrects_and_reproduces_its_outputs_bit_identically`
- `the_rotating_audit_covers_the_whole_bank_within_the_computed_latency`
- `a_healthy_run_never_triggers_growth`

Failure-handling cases:
- `an_episode_of_random_bytes_is_rejected`
- `a_probe_whose_answer_is_in_a_trained_span_is_refused_at_ingest`
- `a_contradiction_episode_never_silently_flips_an_earlier_probe`
- `a_format_only_episode_does_not_move_its_paraphrase_probes`
- `surface_memorisation_answers_the_counterfactual_with_the_originals_answer_and_fails`
- `a_degenerate_candidate_is_rejected_as_Degenerate`
- `an_earlier_episode_collapsing_behind_a_healthy_pooled_mean_is_rejected_as_BlockRegressed`
- `an_interference_shaped_failure_raises_rehearsal_and_does_not_grow`
- `a_capacity_shaped_failure_grows_instead_of_raising_rehearsal`
- `a_rolled_back_stage_leaves_the_live_response_byte_identical`

Control-integrity cases, the ones that check the checker:
- `the_null_gate_arm_is_separated_beyond_seed_noise`
- `the_shuffled_label_arm_promotes_at_chance`
- `an_order_permutation_reproduces_the_retention_diagonal`
- `a_seed_repeat_produces_a_bit_identical_adapter`
- `reserve_anchors_and_gated_anchors_diverge_when_the_gated_set_is_overfit`

**Surface** (`crates/sdk/tests/continual_reader.rs`, mirroring
`document_study.rs`): a report is always written; an adapter is published only
on a promote, under the name `improve::latest_adapter` returns; a malformed
corpus is refused before any training; an unregistered architecture is refused
by name; an interrupted write leaves a loadable run directory; a resumed run
continues at the same episode.

**Acceptance**: one test drives the sample's own `selftest` over the generated
corpus and asserts the Part 3.3 block.

### 6.6 What a run prints

```
episode 412  learn/rfc-8446.txt          PROMOTE   effect +0.31  p 0.004  adapter 7
episode 413  noise/rand-0009.bin         REJECT    NotSignificant p 0.51
episode 414  contradict/tls-version.txt  REJECT    BlockRegressed block 118 drop 0.22
episode 415  format/template-0031.txt    PROMOTE   effect +0.19  p 0.011  adapter 7
             paraphrase +0.01 WARN format-only, flagged not failed

retention   N=415 promoted 271 (65.3%)   null-gate arm 48 (11.6%)
            BWT -0.004   per-block worst -0.03 on episode 118
            bank coverage 100% within 62 episodes  detection latency 62
            independent battery 0.71 -> 0.71 (budget 0.02)
            eval compute 18.4% of total (budget 25%)
            seed-repeat delta 0.000 (bit-identical)
```

### 6.7 Decisions taken, so they are not rediscovered mid-build

- **Tiny fixture by default, a real `--model` opt-in.** Satisfies samples
  rule 5 and keeps the demo runnable on any machine; a learning claim requires
  a real `--model` and the report says which mode it came from. The reference
  is resolved by `crates/loader`, so brain's own model handler fetches it.
- **The adapter pool lives in `crates/model`**, beside `lora.rs`, not in a new
  crate. It is a property of LoRA, and `LoraW` is already there.
- **`crates/audit` is a new leaf crate, not more surface on `crates/promote`.**
  `promote` is the promote/reject decision; this is the evidence-gathering
  around it, it is larger than `promote` will be, and the layering gate treats
  them the same way.
- **Four verbs, not one.** `read` alone cannot express "run the adversarial
  tier now" or "tell me what you know", and folding those into flags on `read`
  makes the unattended path carry options it never uses.

---

## Part 7 - the capability contract: what the user gains

The probes of Part 3 answer "did this episode land, and what did it cost". They
are drawn from the episode, so on their own they are circular: they can only
ever show that the model learned the thing extracted from the thing it read.
They are the GATE's instrument, and they are not a capability claim.

The user's instrument is a separate one.

| | **probes** | **capability battery** |
|---|---|---|
| question | did this episode land, at what cost? | what can the user do now that they could not before? |
| drawn from | the episode itself | written independently of the corpus |
| frozen | at ingest of that episode | **before any reading happens at all** |
| read by | the gate | the user |
| cadence | every episode, fast | before the run, and periodically |
| trained on | never | never, and never gated on either |

The battery for this sample is **40 held-out natural-language tasks** over the
generated tool ("archive everything in /var/log older than a week, compressed")
whose answer is a command-line invocation. The verifier is the generated tool's
own parser plus a comparison against the canonical invocation: does it parse,
do the flags exist, are the required ones present, do the mutually exclusive
groups hold, does it mean what was asked. Exact, programmatic, in-process,
never a model judging a model.

The whole story is three commands:

```
sample-learning-reader battery --model qwen3:0.6b --run-dir run/
    capability 2/40   baseline frozen

sample-learning-reader read corpus/ --run-dir run/
    ... 400 episodes, unattended ...

sample-learning-reader battery --run-dir run/
    capability 34/40  (+32 over a frozen baseline of 2/40)
```

**Before, the tool rejects what the model writes. After, it accepts it.** That
is the ability the user gains, it is checked by a parser rather than asserted,
and it is not reachable by looking anything up at inference time because
nothing was retrieved.

"Promote rate 65%" is plumbing. The battery delta is the result, and a run that
improves its promote rate while the battery stays flat has failed, loudly.

---

## Part 8 - triage: what is worth learning, and what is garbage

Training on every episode in an unbounded stream is not affordable and not
desirable. Three filters at increasing cost, so most garbage never reaches a
gradient.

**T-A. Well-formedness. Free, no model.** Encoding valid, not binary by
content, and **compressibility**: if a cheap compressor cannot compress the
episode there is no structure in it to learn. Random bytes die here having cost
nothing. `rejected: Unstructured`.

**T-B. Novelty and reach. One forward pass, no training.** Score the episode
under the CURRENT model.

| | | |
|---|---|---|
| loss below `lo` | the model already predicts this | `skipped: AlreadyKnown` |
| loss above `hi` | the model cannot model this at all | `deferred: OutOfReach` |
| between | **the learnable band** | goes to T-C |

This is a learning-progress criterion and it is the principled answer to "what
is worth learning". Note what makes it more than a heuristic: the band is a
property of the model's CURRENT state, so the same document can be out of reach
at episode 10 and learnable at episode 800. That is why `OutOfReach` parks the
episode in a retry queue rather than discarding it, and why the retry queue is
part of the design rather than a convenience.

An optional sharper form, for episodes sitting ambiguously in the band: split
the episode, take k steps on one half, measure the loss change on the other.
That is **measured** learning progress rather than inferred. It costs a short
training run, so it is not the default.

**T-C. The four-bar gate. Train and score, expensive.** Only band survivors
reach it.

The fraction of the stream reaching each stage is a reported number. It is the
throughput story, and a triage that passes everything through to T-C is a
triage that does not work.

### What triage deliberately does NOT decide

**Learnable is not true.** A document can be well formed, novel, inside the
band, and wrong. The harness does two things instead of adjudicating:

- **Trust is an input, never inferred.** The user marks a source trusted per
  directory or per stream. Content-inferred trustworthiness is precisely how a
  learner ends up confidently absorbing nonsense that was written confidently.
- **Conflict is surfaced, not resolved.** A candidate that flips an
  already-promoted probe is reported with both sources named and the flipped
  probe quoted. The learner says "these disagree, here"; it does not pick a
  winner.

Conflating learnable with true is the failure mode that turns a continual
learner into a confabulator. It is a stated non-goal of this design, not an
oversight in it.
