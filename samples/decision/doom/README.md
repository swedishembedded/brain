<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# doom - a decision model learning to play DOOM from the game's own state

The 1993 engine, modified to host an HTTP API inside its game loop, runs as a
subprocess in **lockstep**: it advances only when the agent says so. Every
decision reads a JSON observation - health, what is in sight and where, how far
there is to walk in each direction, where the exit is - and picks from a list of
options **rebuilt from that state at every step**.

No pixels reach the model. The frame in the window is for you.

![a decision](docs/decision.png)

*The whole system while it plays: the game, the text the model actually reads,
every option with the probability the policy gave it, and the reward each
decision earned. Drawn into one canvas by [`src/view.rs`](src/view.rs), so a
headless run writes exactly this image to a PNG - which is where the ones in
this README came from.*

The probabilities are the point. A flat row of bars is a policy that has not
made up its mind; a single bar at 100% is one that has collapsed onto one
option. Both are visible at a glance and neither is visible in a return.

With `--window` on a machine that has a display, the same canvas is a window:

![the window](docs/window.png)

*Captured from a real X server rather than described - `Xvfb` plus `import`, so
"it opens a window" is a measurement.*

## What this demonstrates

**A decision model, not a policy network.** The usual RL policy ends in a layer
whose *width* is the action space, so the actions must be known when the weights
are created and be the same at every step. Here they arrive with the
observation, as text. `attack the former human sergeant 180 units away, 12
degrees to your left` exists only while that sergeant does; one step later the
same slot holds something else. The model reads what an option **means** instead
of looking it up by index, which is why an option nobody trained on still works.

**The orders are part of the input.** An episode runs under a mission, and the
mission text is prepended to every option:

| mission | what it is told | what it is paid for |
|---|---|---|
| `clear` | kill everything, the exit can wait | kills weighted 1.5, exit 3 |
| `speedrun` | reach the exit, fight only what blocks you | kills 0.2, exit 15 |
| `survive` | stay alive, avoid damage | damage costs 4x, exit 5 |

Train with `--mix` and the mission is sampled per episode, so one policy has to
read what it was asked to do. The instruction is not decoration: the reward
weights move with it, and a policy that ignores it is scored against whichever
one it was actually given.

**Realtime.** A decision costs about 16 ms end to end on one Tesla P40 - about
10 ms of model and 6 ms of game - so the agent decides 5-9 times per second of
game time, which is roughly the rate a human plays at. See
[measured cost](#what-a-decision-costs).

## Running it

```bash
./fetch-data.sh                      # engine, WAD and encoder; prints the flags

make samples/decision/doom/build
make samples/decision/doom/run ARGS="probe --doom-bin … --wad … --encoder …"
```

Five commands, and every one of them works headless:

| | |
|---|---|
| `probe` | one episode of the scripted player. **Run this first**: it exercises the whole path - process, socket, observation, action, reward, frame - with no trained policy and no encoder quality needed. |
| `train` | clone the best of the scripted player, then improve it with PPO, then score both. |
| `eval` | score a policy and the scripted player over the **same** episodes. |
| `play` | run episodes showing every decision. `--window` if you have a display. |
| `bench` | what a decision costs, against state length and option count. |

Add `--window` to watch, `--frames DIR` to write every decision as a PNG, and
`--transcript FILE` to record every request and reply as JSON lines - which is
the artifact to read when the agent does something inexplicable, because it is
exactly what the model saw, in order:

```bash
jq -c 'select(.req.path=="/api/step") | .resp | {tic, hp:.player.health, outcome}' transcript.jsonl
```

**Nothing is read from the environment and no path is baked in.** Where the
engine, the WAD and the encoder live is what you typed on the command line, so
a run is reproducible from its own invocation. `--device`, the training knobs
and the window flags come from `brain::options`, shared with the `brain` binary
itself, so they mean the same thing here as everywhere else.

## How it is put together

```
restful-doom  --apilockstep        the game, frozen between steps
     |  HTTP/1.1 keep-alive, one request per decision
     v
src/doom.rs      process lifetime, transport, episodes
src/obs.rs       JSON -> typed state -> the text the model reads
src/action.rs    the state -> what can be done right now
src/env.rs       reward, missions, the scripted player
src/view.rs      the inspector: frame + state + decision + reward
     |  brain::Env
     v
brain::ControlPipeline            encoder + decision head, PPO
```

The engine side needed real work, and it is in the DOOM repository rather than
here: an HTTP layer that survives a request per tic, lockstep stepping, an
observation in one round trip, episodes that restart reproducibly, a derived
event log, and a framebuffer endpoint. See that repository's history.

### What the model is given

Everything, as prose, because that is what a decision model reads:

```
health 101 armor 0, pistol with 50 rounds. Killed 2 of 29 enemies, 1 of 37 items, 0 of 3 secrets.
In sight: former human sergeant close at 180 units 12 degrees left, coming for you.
An explosive barrel sits 279 units 78 degrees right.
Room to move: 320 ahead, 44 left, 320 right, 320 behind.
The exit switch is 2217 units 53 degrees left, with 0 units of clear floor that way.
You have been through here 3 times. You have not actually moved for 4 decisions. 22 patches of this level explored.
Just now: took 15 damage.
```

Four details in there are load-bearing, and each was wrong once:

- **Bearings, not map angles.** "12 degrees left" is actionable; "at 143
  degrees" needs the reader to know its own facing and do the subtraction.
- **Threats are monsters only.** Barrels are shootable too, and reporting one as
  an enemy meant "kill everything" was scored against a target that does not
  count toward the level's kill total.
- **Clearance is walkability, measured by tracing the level's lines** - not
  "could I stand exactly there", which was the first implementation. Doom slides
  a player along whatever they brush, so a decorative pillar beside the path
  read as a wall, and the option list hid "walk forward" at a spot the player
  then crossed 83 units of.
- **The last line is the agent's own history**, which the game does not report
  and the policy cannot remember - every decision is an independent forward
  pass. Leaving it out is what made the first trained policy lose; see
  [run 1](#run-1---it-lost-to-the-scripted-player).

## Learning to navigate rather than learning to walk into walls

This is the part worth reading, because the obvious design is wrong and it is
wrong in a way that produces plausible numbers.

The exit is hundreds of decisions away and pays once. Something has to fill the
gap. The obvious filler is to reward getting closer to it. That is
potential-based shaping ([Ng, Harada & Russell
1999](https://www.andrewng.org/publications/policy-invariance-under-reward-transformations-theory-and-application-to-reward-shaping/)),
so it provably leaves the optimal policy alone - but the guarantee is about the
optimum, not about what gets learned on the way there, and **in a building the
straight line goes through walls**.

Measured, before this was changed: a trajectory spent 90 of its 120 decisions
shuttling between two spots - walking at the wall the exit is behind, backing
off, walking at it again - and was paid for every one of them. That is a reward
function teaching an agent to walk into walls, and no amount of training fixes a
reward that is wrong.

What fills the gap instead is a **count-based exploration bonus**: the first
visit to a 128-unit patch of floor in an episode pays, and the *n*-th visit pays
`1/sqrt(n)` of it. It is the cheap, network-free member of the
intrinsic-motivation family that [ICM](https://www.alphaxiv.org/abs/1705.05363)
and RND belong to - the family that solved exactly this problem on ViZDoom's
sparse navigation tasks - and it is the one that fits here, because it needs no
second model and no gradient of its own.

Walking into a wall discovers nothing and earns nothing. Finding a corridor
earns. **The agent is not told where to go and it is not told that walls are
bad; it is paid for finding out.** The exit stays in the observation, because a
player can see the level and so should the agent. What has gone is being paid
for pointing at it.

The same concern applies to the demonstrations. A scripted teacher is not
uniformly good - it is good in the situations it was written for and arbitrary
everywhere else, and a heuristic navigator's bad episodes are bad in a
specific, *learnable* way. So the warm start is **filtered behaviour cloning**:
keep the best `--warmup-keep` of the teacher's episodes by return and throw the
rest away, rather than teaching the policy something the policy gradient then
has to spend its samples unlearning. A real run:

```
warm start: kept 7 of 12 scripted episodes (best +11.06, dropped down to +2.87)
```

### The player it has to beat

A baseline that is merely broken makes a learned number unreadable - beating it
would prove nothing - so the scripted player is a real one: fight what is in
front of you, take what is under your nose, otherwise go wherever there is most
room, preferring the way the exit lies, and when the last sixteen decisions have
gone nowhere in aggregate, commit to one direction for eight steps to break out
of the cycle. It is still deliberately crude: greedy, no map, never retreats
from a fight it is losing, never prioritises the enemy actually shooting at it,
and does exactly the same thing whatever the orders say. That last one is the
headroom the learned policy is supposed to take.

## Progress

Measurements, in order. Every row is a real run on this repository's own
hardware (two Tesla P40s, one used); nothing here is projected.

### What a decision costs

`doom bench`, which times one decision against the two things its cost could
scale with:

| state (words) | options | ms/call |
|---:|---:|---:|
| 4 | 1 | 6.8 |
| 4 | 8 | 9.9 |
| 4 | 24 | 17.0 |
| 40 | 8 | 10.7 |
| 200 | 8 | 24.7 |

Which decomposes into **6.8 ms fixed, 0.09 ms per state word, 0.44 ms per
option**. So at this sample's own shape - about 75 words and 7 options - a
decision is ~16 ms, of which nearly half is the fixed cost of running a
six-layer encoder at all rather than anything about the text. That fixed floor
is engine overhead (per-dispatch cost and two device syncs per call), not
something this sample can edit its prose out of; the numbers are here so the
next person does not have to rediscover which lever is which.

The game itself is ~6 ms per decision at 6 tics, and lockstep decouples it from
the 35 Hz clock entirely: measured with no model in the loop, **2 966 tics/s,
85x realtime**.

### Does it learn?

Runs in order, each one a real run whose log is kept. Charts come from a run's
own stdout (`./plot-run.py out/train-run1.log docs/`).

#### Run 1 - it lost to the scripted player

10 iterations x 12 episodes, 140 decisions each, E1M1 at Ultra-Violence, the
mission sampled per episode. 39 162 optimizer steps, 19 minutes.

| | return | kills | items | exits | deaths |
|---|---:|---:|---:|---:|---:|
| scripted | **+6.39** | 0.2 | 1.1 | 0 | 0 |
| policy | +3.20 | 0.0 | 1.0 | 0 | 0 |

![training](docs/training.png)

The rollout return climbed (+6.30 to +7.19) and the critic fitted, and the
policy still lost by 3.19. **The diagnostic is the gap between those two
numbers**: rollouts are SAMPLED and averaged +7.19, while the greedy evaluation
of the same weights scored +3.20. A policy whose sampling beats its own argmax
is not a weak policy, it is a policy in a loop - greedy is deterministic, so if
the best action in a state returns it to that state, it takes the same action
forever, and sampling is the only thing that was breaking out.

Which is a bug in the **observation**, not in the training. The policy is
memoryless: each decision is an independent forward pass with no recurrence. It
was not told how long it had been stuck or whether it had stood here before -
but the scripted teacher it was cloned from uses exactly those two facts, so
cloning taught it a mapping that does not exist, the same observation with the
teacher choosing differently on history the policy could not see.

#### Run 2 - the same run, with the agent's own history in the observation

`This is new ground.` / `You keep coming back here - 6 times now.` /
`You have not actually moved for 4 decisions.` / `22 patches explored.`
Everything else identical, so the comparison isolates one change.

| | return | kills | items |
|---|---:|---:|---:|
| scripted | **+6.39** | 0.2 | 1.1 |
| policy (run 1) | +3.20 | 0.0 | 1.0 |
| policy (run 2) | +5.87 | **0.3** | 0.7 |

**The gap closes from -3.19 to -0.52 on one change to the observation**, and on
kills the policy passes the scripted player. Nothing about the training changed;
the policy was simply being asked to act on state it could not see.

#### Run 3 - one mission, a longer episode, a longer warm start

140 decisions was not enough for either player to reach much combat, so: a
single mission, 250 decisions, 16 warm-start episodes over 12 passes.

| | return | game score | kills | items |
|---|---:|---:|---:|---:|
| scripted | **+7.41** | -0.59 | 0.0 | 1.0 |
| policy | +4.27 | **-0.29** | 0.0 | 1.0 |

![run 3](docs/training-run3.png)

The rollout return climbs from +9.26 to +13.23 and the policy ends **+0.30
ahead on the game's own score** and 3.13 behind on total return - i.e. it plays
DOOM slightly better and explores less.

And **neither player killed anything in twelve episodes**. That is the finding
that matters, and it is not about the policy at all: at 250 decisions from the
level start, an agent spends the episode getting out of the spawn area. A
scripted probe on another seed reaches two kills in the same budget, so combat
is rare and high-variance rather than absent - which makes the game score a
statistic with almost no events in it, and no training run can learn from a
signal that mostly is not there.

#### Where that leaves it

**The policy does not yet beat the scripted player on total return.** It is
level with or slightly ahead on the game's own score in both runs 2 and 3, and
behind on exploration. The honest summary is that the system learns - the
return climbs, the diagnosis of run 1 was confirmed by run 2's single change -
and that the EXPERIMENT is still mis-shaped for what is being asked of it.

The next change is not a hyperparameter. It is the initial state: an episode
that begins in contact with the problem rather than a few hundred decisions
away from it. That is what
[ViZDoom's scenario set](https://github.com/Farama-Foundation/ViZDoom) exists
for - `defend_the_center`, `deadly_corridor`, `health_gathering` are all hand-made
starting positions, for exactly this reason - and this API can already do it:
`POST /api/world/objects` spawns a monster at a distance, and `PATCH
/api/world/objects/{id}` moves the player. A randomised start in a room that
has something in it turns combat from a rare event into every episode.

### What would move this next

In the order the measurements point at, not in the order they are interesting:

1. **A longer episode.** At 140 decisions on E1M1 neither player reaches enough
   combat for the game's own score to differentiate them - measured, +0.09
   against -0.01, both of them noise around zero. The scripted player needs
   about 250 decisions before it has killed anything. Until the horizon is past
   that, the comparison is almost entirely a comparison of who covered more
   floor.
2. **A warm start that converges.** Cloning stops at a cross-entropy of about
   0.5 over roughly eight options, which is a policy agreeing with the teacher
   maybe 60% of the time. The pipeline's own documentation is explicit that a
   clone which has not converged leaves the policy gradient starting from
   something that is neither the teacher nor random.
3. **One mission at a time first.** `--mix` splits an already small budget
   three ways and asks the policy to learn instruction-following on top of
   playing. Beat the baseline on `clear` alone, then re-introduce the mix and
   measure per-mission.
4. **Batching decisions across parallel games.** The profile says 6.8 ms of
   every decision is fixed cost paid per CALL - two device syncs and the
   dispatch overhead of a six-layer encoder. Eight games stepping together
   would amortise it eight ways, and the engine already packs one state and all
   its options into a single batch; what it cannot yet do is pack several
   states. That is the one change with a multiple in it rather than a
   percentage.

## What is not claimed

- **It does not finish E1M1.** Reaching an exit switch across a level is a long
  exploration problem and this is a first result, not a solved one.
- The sentence encoder is frozen by default (`--train-encoder` to change it):
  a few hundred high-variance policy gradients per iteration are not enough to
  move 22M pretrained parameters anywhere useful, only enough to damage the
  language understanding that made the option text readable.
- `--mix` trains one policy over three missions; whether it has learned to
  *read* the instruction rather than average over them is measured by scoring
  it per mission, which is what `eval --mission` does.
- Only E1M1 is exercised. `--map` accepts the rest of the shareware episode,
  and a policy that only works on one map has memorised it.

---

Swedish Embedded AB builds realtime decision systems that run on the customer's
own hardware - reading a machine's real state, choosing among actions that
machine defines at run time, in milliseconds, with no text generator in the
loop. If your team needs judgment inside a control loop, you can procure our
services by sending an email to info@swedishembedded.com.
