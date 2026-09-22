# doom - roadmap: search, then compression, to UV-Max

`samples/decision/doom` plays DOOM from the engine's own structured state
through `brain::ControlPipeline`. The target this file plans toward is
**UV-Max on every level of episode 1**: Ultra-Violence, 100% kills, 100%
secrets, level exited, minimum time.

## Where it actually stands (measured 2026-09-22, this box)

| | |
|---|---|
| scripted teacher, E1M1, skill 3, 3000 decisions | **dies at decision 1463, 6 of 29 kills**, 0 health |
| best generation ever measured, summed progress over 9 levels | 1.18 (the teacher); the trained generations scored 0.73 |
| levels ever finished, by anything, at any skill | **zero** |
| engine replay determinism | **bit-identical** across separate process launches (same seed, 400 decisions: identical return, kills, health, route numbers) |
| decision cost, no model in the loop | ~19 ms (scripted, 1463 decisions in 27.8 s) |
| snapshot slots the engine offers | 4096 |
| `leveltime` in tics | already in `/api/state` and already parsed (`obs::Level::tic`) |

So the gap to UV-Max is not a tuning gap. Nothing here has ever finished a
level, and the search that is supposed to find how has never been asked to.

## The three things that block it, named

1. **Search is not a thing you can run.** It is `ControlPipeline::explore`, a
   private method reached only as `--explore N` rounds *inside* a training
   iteration, which means every search also pays for an encoder, a head and a
   policy gradient. Discovery and learning are one program.

2. **Nothing in the loop optimises what UV-Max scores.** `report::Score`
   weights items (which Max does not require), has **no time term at all**, and
   its unfinished branch saturates on the first subgoal. A search maximising it
   is not searching for a speedrun.

3. **The archive's cell cannot see the endgame.** `Env::cell` buckets kills,
   items and secrets into *tenths*, so 27/29 and 28/29 kills are the same
   niche. UV-Max is won or lost in the last three monsters and the last secret,
   which is exactly the resolution that bucketing throws away.

## Design

### The split the whole plan rests on

> **Search may use snapshots. The artifact may not.**

Discovery restores engine snapshots freely - that is what makes it affordable.
What it *produces* is an action list replayed from the level's own start, and a
candidate is not in the solution set until that replay has been run and has
reproduced the same kills, secrets, exit and tic count. Determinism (measured
above) is what makes that a real gate rather than a hope, and it is what makes
the final recording honest: the recorded run is a policy playing the level
start to finish, with no oracle, no restore and no teacher underneath it.

### Phase 1 - DISCOVERY (no model at all)

New **leaf** crate `crates/search` (`brain-search`), model-agnostic, no GPU, no
engine, unit-testable on its own:

- **A quality-diversity archive, not a leaderboard.** One elite per niche.
  The niche is *what kind of situation this is*; the quality within it is
  *how fast it was reached*. That single choice makes the archive a speedrun
  optimiser for free: the elite of the niche "everything killed, every secret
  found, standing at the exit" **is** the run we are trying to produce.
- **An operator set with a bandit allocating budget between them**, scored on
  measured archive gain per second - the only honest way to answer "where
  should evaluation n+1 be spent":
  1. `resume+repeat` - today's Go-Explore random walk with action repetition
  2. `resume+teacher` - the scripted player as the proposal distribution
  3. `resume+policy` - the trained policy (phase 3 only)
  4. `hunt` - route to the nearest **unkilled monster** or **unfound secret**.
     Goal-directed, and the one operator a random walk can never substitute
     for: three secrets on E1M6 will not be found by wandering.
  5. `refine` - take an elite, perturb one decision, keep it if the outcome is
     no worse in fewer tics. Trajectory superoptimisation.
  6. `splice` - two elites that pass through one niche: prefix of one, suffix
     of the other.
- **An evaluator cascade**, cheapest rung first: is this fragment better than
  the cell it resumed from (free) -> is its niche new or its elite beaten
  (free) -> **replay from the level start and verify** (a whole episode, run
  only for a candidate claiming a level solution or a new best time).

`ControlPipeline::explore` becomes an adapter over this crate. It does not get
a second implementation - one archive, per the workspace's own rule.

### Phase 2 - COMPRESSION

Behaviour-clone the **verified** trajectories and the archive's elite
fragments. Two arms, measured head to head rather than chosen:

| arm | what it is | the honest expectation |
|---|---|---|
| `decide` / MiniLM-L6 | 22M, what the sample uses today, ~28 ms a decision | fast enough to train in the loop |
| `laya` / ModernBERT-large | 395M, reads option *text* far better zero-shot (71% on unseen intents vs 9.5% chance) | ~18x the parameters; its real-weight training gate is currently **red** (`.agents/roadmap/laya.md` M8). Viable as a frozen proposal policy and as a final-run policy, questionable inside a loop that needs thousands of episodes |

### Phase 3 - POST-TRAINING

The model's own rollouts become search operator 3, discoveries re-enter the
archive, and the policy is re-fitted. Selection moves off `--gauge 3`, which
the sample's own trace shows is choosing noise (0.081, 0.039, 0.134, 0.078,
0.061 with no trend against a per-episode spread of 0.0 to 0.5).

### Phase 4 - RECORD

One continuous policy-only episode per level to
`~/Downloads/doom-record/E1M<n>.mp4`: no snapshot, no `--full-map`, no teacher
fallback, no search in the loop.

## What this plan does NOT claim

**Beating the human UV-Max world records is very unlikely, and the plan should
not be read as promising it.** Those runs are built out of frame-level movement
technique - SR50 straferunning, wallrunning, tic-precise turns - and this
agent's action space is held-button macros of 4 to 8 tics (`walk forward, 320
units`). The tech that makes a record is not expressible in it, and no amount
of search finds a trajectory the action space cannot represent. Widening the
action space toward tic-level control is possible and is a separate decision
with its own cost: it multiplies the search horizon by roughly the macro
length.

What IS reachable, and what nothing in this sample has ever done, is **100%
kills + 100% secrets + exit, on every level of episode 1, at Ultra-Violence** -
and then the minimum time this action space admits. The time term is in the
design from the start so that the number improves for as long as the search
runs, rather than being retrofitted onto a score that cannot see it.

## Milestones

- [x] **M1 - `crates/search`** (`96192c4f3`): archive keyed by
      `Niche`, elite-on-cost within a niche, UCB1 allocator over gain per
      second, evaluator cascade. Leaf, gated by `check-crate-layers.sh`, 26
      spec tests, no engine and no GPU.
- [x] **M2 - UV-Max is a scorable thing** (`79859abeb`): `Mission::UvMax` on
      `report::Bar::UvMax`. **No time term in the score**, against the
      original plan - a penalty that grows while the episode runs makes dying
      the cheapest way to stop it, so time went to the archive's tiebreaker
      instead. The exit is worth LESS than clearing (0.6 against 1.3), because
      stepping on a DOOM exit ends the level and a bar that ranks
      "sprinted out" above "cleared but still inside" teaches the search to
      discard the trajectories worth keeping.
- [x] **M3 - `doom search`** (`79859abeb`): discovery as its own subcommand,
      no encoder loaded and no device opened. Top cascade rung replays a
      claimed solution from the level's own start.
### Measured so far, E1M1 at Ultra-Violence, 180 s, seed 1

Every row is the same level, the same seed and the same budget. Best score is
out of the 2.0 a completed category scores.

| campaign | best | cells | x-cover | monsters-left buckets | steps |
|---|---|---|---|---|---|
| 60-step operators, niche without capability | 0.166 | 120 | 15 | 2 | 6 720 |
| long teacher operators (`commit`/`chase`, 400 steps) | 0.193 | 73 | 12 | 2 | 12 988 |
| ...plus weapons and health in the niche | **0.497** | 284 | **30** | **3** | 11 318 |

0.497 is about 18 of E1M1's 29 monsters, against the scripted teacher's 6.
Nothing has yet completed the category, so there is still no verified
solution and no recording.

Two things were learned rather than guessed, and both are recorded where a
later reader will find them:

- **The niche must hold capability, not the score's own terms**
  (`.agents/knowledge/` #146). Dropping items from the niche because the
  category does not count them also dropped the shotgun, and the archive
  then deleted every armed state in favour of the faster unarmed one.
- **Operator rates are only informative once the niche measures something
  that matters.** With the position-heavy niche the four operators scored
  1.88 to 2.75 gain/s and the allocator had nothing to choose between; with
  capability axes they separated to 4.74 (`wander`) against 8.33 (`commit`),
  and `commit` - 400 decisions of real play resumed from an archived cell -
  is now clearly the one worth the budget.

- [x] **M4 - hunting and frisking** (`afd7e4bcc`). Not the operator the plan
      described: routing to an unfound SECRET would be an oracle, and so
      would routing to a monster nobody has seen. Both are fair play
      instead - `Tag::Hunt` goes back for a monster the agent actually saw
      (a coarse room-scale ledger, `memory::Haunt`), and `frisk` presses on
      walls, which is how anybody finds a DOOM secret without being told
      where one is. The engine needed nothing new: `API_RouteTo` already
      existed.
- [ ] **M5 - first verified UV-Max on one level**, replayed from the start.
- [ ] **M6 - `refine`/`splice`**: minimise tics on a solved level.
- [ ] **M7 - compression**: clone the verified set; both arms measured.
- [ ] **M8 - post-training**: policy rollouts as a search operator.
- [ ] **M9 - the nine recordings.**

Each milestone lands with its measurement. A milestone with no number beside it
is not done.
