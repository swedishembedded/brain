<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Success criteria for the arena sample - FROZEN (second freeze)

Written before the experiments that judge it, and **not to be edited to match a
result**. If a number below turns out to be unreachable, the honest report is
"criterion X was not met", not a smaller X.

## Why there is a second freeze

The first freeze was applied to a **different game**, and the game was broken:
seven hand-written policies were measured on it and the one that ignored the
observation entirely won. Its ceiling was ~64%, which was also what the naive
policy scored, so the primary criterion (>71%) asked the learner to exceed a
bar no policy could reach.

Freezing criteria is right. Freezing an **unvalidated** criterion is not - it
locks in a number nobody can hit and then reads the failure as a property of
the method. The missing step is now step 0 below.

The first freeze's verdict stands as reported, on the game it was written for.
It is not carried over.

## Step 0: the environment must be shown capable of distinguishing policies

Before any threshold is set:

* hand-written policies are measured on a ladder (`policy_ladder`), and
* **at least one policy that reads the observation must beat one that ignores
  it, by a wide margin.**

`the_policy_ladder_shows_the_headroom` asserts this, so a broken testbed fails
loudly instead of producing uninterpretable results.

Measured, 300 episodes:

| policy | win rate | return |
|---|---:|---:|
| random | 11.0% | −2.84 |
| always shoot (naive) | 41.7% | −0.28 |
| demon first | 20.0% | −1.98 |
| nearest first | 58.7% | +1.00 |
| imps first | 72.0% | +1.98 |
| **imps first, then close before firing** | **77.3%** | **+2.45** |

A 35-point spread, and everything it depends on - which monster, at what range -
is in the option text.

## The measurement, fixed

* **Metric**: win rate over `control::EVAL_SEEDS` = `1_000_000..1_000_200`,
  **200 episodes**, actions taken **greedily**.
* **Secondary**: mean episode return over the same 200.
* Rollout seeds count up from 0, so training never touches an eval episode.
* Standard error at n=200 is at most 3.5 points. **"Beats" means +7 points.**

## The reference band, fixed (measured on the evaluation seeds)

| | win rate | return |
|---|---:|---:|
| random, sampled | 11.0% | −2.86 |
| **untrained head, greedy - MEDIAN over 8 init seeds** | **~0%** | - |
| **teacher (always shoot), = what cloning starts from** | **39.0%** | −0.50 |
| best hand-written | 78.0% | +2.49 |

**The untrained bar is a median, and it has to be.** A greedy untrained head is
deterministic - it prefers one kind of option forever - so it is usually
catastrophic, but occasionally lucky. Measured across 8 initialisations: seed 0
scores **74%**, every other seed scores **0-5%**. Quoting one seed would have
made a near-optimal policy look like the starting point. The sample's default
head seed is 1, a typical one, and the outlier is on the record.

## Criteria

| # | criterion | threshold | judged by |
|---|---|---|---|
| **A1** | the action set genuinely varies | test | `the_legal_action_set_changes_with_the_situation` |
| **A2** | options are read as text, not position | test | `options_describe_themselves` |
| **A3** | a decision stays inside a realtime budget | **< 20 ms** | measured, reported |
| **A4** | the environment can distinguish policies | 35-point ladder spread | `the_policy_ladder_shows_the_headroom` |
| **L1** | **primary**: reinforcement learning beats the teacher it cloned | **win rate > 46%** | 200 eval episodes |
| **L2** | floor: clearly above an untrained head and random play | **win rate > 20%** | 200 eval episodes |
| **L3** | beats the teacher on return as well as on wins | **return > −0.50** | 200 eval episodes |
| **L4** | no collapse: the run does not end below its own best point | curve | reported |
| **L5** | stretch: at least halfway from teacher to ceiling | **win rate > 58%** | 200 eval episodes |

L1 is the learning claim and it is deliberately the one that cannot be passed
by cloning: behaviour cloning reproduces the teacher (measured: 38.5% against
the teacher's 39.0%), so **anything above 46% is the policy gradient's doing**.

## What does NOT count

* Rollout win rate during training - measured under exploration, on training
  seeds, at n=48 where the standard error is 7 points.
* A single initialisation seed, for any bar. See the untrained row.
* The PPO surrogate loss.
* Any number from a run with fewer than ~50,000 environment steps. Published
  PPO needs ~80,000 to solve CartPole, which is easier than this along every
  axis. An under-powered verdict is "not tested", not "failed".

## Erratum, recorded after freezing - do not rewrite the table above

The band above records behaviour cloning at **38.5%**, matching the teacher's
39.0%, and L1's threshold of 46% was justified by that: seven points clear of
what cloning could reach, so anything above it was the policy gradient's doing.

**That BC measurement was wrong.** It was taken with a warm start that did not
converge - switching the update to minibatches cut cloning from 600 optimizer
steps to nine, and nobody noticed because the number it produced was close to
the teacher's for one initialisation and 1% for another. With cloning run to
convergence (12 passes), BC alone scores **44.0%** (return −0.132).

Consequences, stated rather than patched:

* **L1 (>46%) no longer separates the policy gradient from cloning.** Passing
  it would be +2 over BC, inside the noise. It stays in the table because the
  bar is not being moved to suit a result, but on its own it now proves
  nothing.
* **L5 (>58%) is the criterion that does the work.** It is 14 points clear of
  converged BC, which is four standard errors, and it was frozen before any of
  this was known.
* A run that lands between 46% and 58% must be reported as **"L1 met, but not
  distinguishable from cloning; L5 not met"** - not as a success.

The lesson is the same one that caused the second freeze: a threshold is only
as good as the baseline it was set against, and a baseline is a measurement
that can be wrong.

## Verdict

Two runs, head seeds 1 and 2, ~60,000 environment steps each, 200 held-out
episodes, greedy:

| | win rate | return |
|---|---:|---:|
| seed 1 | 79.0% | +2.47 |
| seed 2 | 78.0% | +2.50 |

| # | threshold | result |
|---|---|---|
| A1, A2 | tests | **MET** |
| A3 | < 20 ms per decision | **MET** - 8.2 ms |
| A4 | 35-point ladder spread | **MET** |
| L1 | > 46% | **MET** - but see the erratum; this no longer separates PPO from cloning |
| L2 | > 20% | **MET** |
| L3 | return > −0.50 | **MET** |
| L4 | no collapse | **MET** - rollout wins climb 21-23/48 to 38-41/48 and hold |
| **L5** | **> 58%** | **MET** |

L5 is the one that carries the claim: 14 points clear of converged behaviour
cloning, four standard errors, frozen before the baseline that invalidated L1
was known to be wrong. Policy gradient added **+35 points over cloning** and
matched the best hand-written policy.
