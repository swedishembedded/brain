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

## What you need, and how to get it

Three things this sample cannot ship, and one script that fetches all of them:

```bash
./fetch-data.sh                 # into ./.data (gitignored), prints the flags
```

| | what | why it is not committed |
|---|---|---|
| the engine | [`mkschreder/restful-doom`](https://github.com/mkschreder/restful-doom) | a C program, built for your machine |
| the game data | `doom1.wad`, the DOOM shareware episode | id Software's, not ours to redistribute |
| the encoder | `sentence-transformers/all-MiniLM-L6-v2` | `brain pull` already manages model weights |

The engine is Chocolate Doom with an HTTP API inside its game loop. This
sample needs the fork above, which adds the agent surface (`/api/state`,
`/api/step`, `/api/episode`, `/api/frame`), lockstep stepping, a derived event
log, and the geometry queries the observation is built from.

The two sides are VERSIONED TOGETHER by the observation schema: this sample
parses the engine's JSON into typed structs with `deny_unknown_fields` and no
optional stand-ins for required data, so an engine that is missing a field -
or has grown one - fails to parse with a message naming it, rather than
training on a plausible default. An engine older than the sample will say so
on the first step. Building it needs
`gcc make automake autoconf pkg-config` and the SDL2, SDL2_mixer and SDL2_net
development packages.

For the encoder, any BERT-shaped checkpoint directory works
(`config.json`, `model.safetensors`, `tokenizer.json`):

```bash
brain pull sentence-transformers/all-MiniLM-L6-v2
```

**Nothing is read from the environment and no path is baked in.** Where the
engine, the WAD and the encoder live is what you type on the command line, so
a run is reproducible from its own invocation. `fetch-data.sh` ends by printing
the exact flags for what it just fetched.

## Running it

```bash
make samples/decision/doom/build
make samples/decision/doom/run ARGS="probe --doom-bin … --wad … --encoder …"
```

Below, `$D` stands for those three flags. Every command works headless.

### 1. Check the plumbing

```bash
doom probe $D --arena 3
```

One episode of the scripted player. **Run this first on a new machine**: it
exercises process, socket, observation, action, reward and frame with no
trained policy and no encoder quality needed, and `--frames DIR` /
`--transcript FILE` leave artifacts to look at when something is wrong.

### 2. Train from scratch

```bash
# Learn to FIGHT: three monsters around the player every episode.
doom train $D --arena 3 --skill 2 --mission clear --max-steps 120 \
  --iterations 10 --episodes 10 --warmup 14 --warmup-keep 0.6 \
  --save out/doom-arena.safetensors

# Learn to FINISH THE LEVEL from the level's own spawn.
doom train $D --mission speedrun --max-steps 260 \
  --iterations 12 --episodes 8 --warmup 12 --warmup-keep 0.7 \
  --save out/doom-spawn.safetensors

# The same, from a reverse curriculum: start AT the exit and walk the start
# back as the policy keeps finishing. Worth it when the teacher cannot
# finish the level on its own, which on a new map it may not.
doom train $D --curriculum --mission speedrun --max-steps 100 \
  --iterations 14 --episodes 10 --warmup 12 --warmup-keep 0.7 \
  --save out/doom-curriculum.safetensors
```

A flag nothing recognises stops the run rather than warning: `--warmup-episodes`
for `--warmup` is otherwise a training run whose output says nothing about
having ignored it.

Each ends by scoring the trained policy AND the scripted player over the same
episodes, which is the only comparison that means anything.

### 3. Watch it play, and record it

```bash
doom play $D --head out/doom-curriculum.safetensors --curriculum \
  --play 4 --fps 35 --record out/run.mp4
```

`--record` encodes every decision straight into an MP4 as it is drawn: raw
frames piped to `ffmpeg`, so the memory cost is one frame however long the run,
and there are no intermediate images to clean up. It captures one frame per
game TIC rather than per decision, so the motion is real-time rather than a
slideshow. Add `--window` on a machine with a display to watch live.

### 4. Score a policy

```bash
doom eval $D --head out/doom-curriculum.safetensors --curriculum --eval-episodes 24
```

### 5. What a decision costs

```bash
doom bench $D
```

## How the training works

Four mechanisms, in the order they run. Each is there because something
measurably did not work without it.

**1. A scripted teacher, cloned - but only its good episodes.** Reinforcement
learning from a random start over a text action space is slow enough to look
like a plateau, so the run begins by cloning a scripted player. It is
deliberately crude: fight what is in front of you, take what is under your
nose, otherwise go where there is most room, and commit to one direction for
eight decisions when the last sixteen went nowhere.

Only the best `--warmup-keep` of its episodes are cloned. A heuristic teacher
is good in the situations it was written for and arbitrary everywhere else, and
its bad episodes are bad in a specific, *learnable* way - so cloning them all
teaches the policy something the policy gradient then has to spend its samples
unlearning. This is filtered behaviour cloning.

**2. PPO over the text action space.** The options arrive with the observation
and change every step, so the policy scores option TEXT rather than indexing a
fixed head. The action decides which state the next decision is made from, so
the policy shifts its own data distribution and the trust region is load-bearing.

**3. A count-based exploration bonus, NOT distance to the exit.** See
[the section below](#learning-to-navigate-rather-than-learning-to-walk-into-walls)
- this is the one that produced a policy trained to walk into walls.

**4. A reverse curriculum, when the goal is the exit.** `--curriculum` starts
the episode AT the exit, random-walks it a few decisions away, and moves the
start further back once the policy finishes 6 of its last 8 episodes.

The problem it solves is not difficulty, it is *silence*: reaching the exit of
E1M1 from the level's own start pays once, several hundred decisions later, and
across four training runs it never happened even once. A reward that never
fires is not a hard reward, it is an absent one, and no amount of training or
tuning addresses that. Starting where the reward is makes the signal exist;
the curriculum then walks the start backwards as fast as the policy can follow.

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

## Where the route comes from

This is worth being precise about, because "the model does pathfinding" would
be a false claim and "the engine tells it where to go" would be a misleading
one.

**The geometry is solved in the engine, in a standard format.** The level's
walkable floor is sampled onto a **uniform 32-unit grid** - half the player's
width - and the engine computes, once per level:

- `walk[]`, whether a 32-unit body fits in each cell;
- `edges[]`, which of the eight steps out of each cell a player can take;
- a **breadth-first distance field** over those steps, flooded from a standing
  spot at the exit.

The agent is then given two numbers from it: **how far the exit is along ground
it can walk** and **which way to set off**, the latter by descending the field
a few cells and aiming at the furthest point it could walk straight at. That is
a waypoint, string-pulled - the classic grid-plus-funnel arrangement, not
anything learned.

**What is not in the engine is the decision.** The route is one option among
the ones rebuilt every step, competing with attacking, taking a pickup, backing
off, sidestepping, and opening a door. Nothing makes the agent follow it. That
is the dilemma you actually face in a game: *the exit is 3400 units that way,
there is a sergeant at 180 units, and a medikit 90 units off the path* - and
that trade-off is the decision the model is trained to make. A route solves
"which way is the exit", which is geometry and has a right answer. It does not
solve "is the exit what I should be doing", which does not.

A sector graph was tried first and was wrong: sector adjacency says two sectors
share a line, not that a body can get between them, and a follower oscillated
for 400 decisions inside one convex room. The grid replaced it.

Four bugs in the grid are worth knowing about, because each produced an agent
that looked broken while every probe said the floor was clear:

- **A sight ray is not a body.** Every crossing test started as a ray, and a
  ray is a point: it threads gaps a 32-unit body wedges in. Testing one point
  across the gap is not enough either - a doorway exactly the player's width
  fails whenever its middle lands on a cell boundary, which is an accident of
  where the level sits on the grid.
- **Steps have to exist in both directions.** A traverse between two points is
  not bit-for-bit the same run in reverse, so asking from each end in turn gave
  128 one-way steps on E1M1. A breadth-first field over a graph like that has
  **local minima** - a cell one step further from the exit than its neighbour,
  with no step to it - and the descent stops dead while the field itself looks
  perfectly sensible.
- **Diagonals cut corners.** Both cells either side of a corner being roomy
  says nothing about getting between them. A diagonal is only allowed where one
  of the two L-shaped ways round it is allowed, as steps.
- **Seeding at the exit line's midpoint puts the seed inside a wall**, and any
  walkable spot can be a sealed pocket on the far side of the switch. The seed
  is chosen from the spots the player can actually reach.

### Doors

The route runs **through** shut doors, on purpose, because a player opens them.
So "the way out starts 7 degrees to your right" while the player cannot walk
there is the normal state of affairs at every door in the game, and an agent
told only a bearing can do nothing but press into it - which is exactly what it
did, at the first door 1400 units from E1M1's spawn, for the rest of every
episode.

The observation now names what is in the way and whether pressing use will open
it, and there is an option aimed at the door. Aimed matters: `use` reaches 64
units in the direction the player is **facing**, so a generic "push on the wall
in front of you" is useless against a door off to one side. Facing it and
opening it are separate decisions, and the option holds for thirty tics because
that is what a door needs to rise - pressing again while it moves sends it back
down.

### Slime

Nukage and lava end a run as surely as a monster does, and the straight way
across E1M1's big room is through the slime: health drained from 106 to 0 over
five hundred decisions with almost none of it from monsters. Three things
handle it. The distance field **excludes damaging sectors**, falling back to
allowing them only when that leaves the exit unreachable. The observation says
`standingInDamage` so the agent knows why it is losing health. And the scripted
teacher treats standing in damage as overriding everything else.

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

#### Run 4 - an episode that starts in contact with the problem

`--arena 3` places three monsters around the player at the start of every
episode, on open floor and in sight, drawn from the episode's own seed so both
players meet the identical arena.

This is scenario design, not a cheat, and it is the same move
[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) makes -
`defend_the_center`, `deadly_corridor` and `health_gathering` are all hand-made
starting positions, for exactly this reason. The game, the actions, the reward
and the opponent are unchanged. What changes is that every episode now contains
the thing being scored.

The effect on the experiment is visible before any training. The scripted
player kills three monsters inside the first twenty decisions, and the spread
of its episodes narrows from +2.87..+11.06 to +9.23..+11.31 - it is no longer
mostly measuring whether a run happened to stumble into a corridor.

With that, over 16 episodes at 120 decisions:

| | return | game score | kills | items |
|---|---:|---:|---:|---:|
| scripted | +10.35 | **+5.01** | 3.0 | 1.2 |
| policy | **+11.12** | +4.89 | 3.0 | 0.6 |

**The policy beats the scripted player on return, +0.78 per episode**, killing
the same three monsters and covering more ground. On DOOM's own score the two
are a statistical tie (+4.89 against +5.01); the policy gives up the difference
in items, picking up 0.6 against 1.2.

So: the first win on the headline metric, and a tie on the game's own. Worth
being precise about what that is and is not. It is a measured improvement over
a real baseline on a fair comparison - same seeds, same arenas, same horizon.
It is not "the policy plays DOOM better than a scripted bot" in any general
sense: this is one map, one skill, one scenario, three monsters, and the
scripted player still collects more of what is lying around.

#### Watching it

```bash
doom play --head out/doom-policy-arena.safetensors --arena 3 --record out/run.mp4
```

`--record` encodes every decision straight into an MP4 as it is drawn - piped
to `ffmpeg` as raw frames, so the memory cost is one frame and there are no
intermediate images to clean up. It works headless, which is the point: the
video from a training server is the same video a window would have shown.

#### Run 5 - the reverse curriculum: finishing from NEAR the exit, which is not the same thing

`--curriculum --mission speedrun`. The start begins AT the exit and walks back
as the policy keeps finishing.

The warm start tells you immediately that the task is now learnable: cloning
converges to a cross-entropy of **0.0009**, against roughly 0.5 in every
earlier run. The teacher is no longer arbitrary, because near the exit there is
an obviously right thing to do.

```
curriculum - 8/8 finished, moving the start back to 2 decisions
iter   1  return +15.20  wins  8/10
curriculum - 8/8 finished, moving the start back to 4 decisions
iter   4  return +16.49  wins  9/10
curriculum - 8/8 finished, moving the start back to 12 decisions
iter   5  return +15.90  wins  9/10
curriculum - 8/8 finished, moving the start back to 14 decisions
```

**Read this for exactly what it is.** The policy reaches the exit in 9 of 10
episodes *from a start that is a bounded random walk away from that exit* -
at 36 decisions of walk-back, usually the same room or the one next to it.
It is NOT completing the level. Calling it that, which an earlier draft of
this file did, is the kind of claim this README exists to avoid.

What is settled: the exit reward now fires, which it never did before, and the
policy learns from it - beating the scripted player 11 exits to 4 over the same
episodes. What is not settled is the thing the sample is actually for, which is
getting there from the level's own spawn.

The curriculum did not solve navigation. It removed the need for it, by making
the path short enough that local information suffices. That distinction is the
whole of the remaining work.

#### Run 6 - from the level's own spawn, which is the only run that counts

Runs 4 and 5 both start the episode somewhere convenient: run 4 places monsters
around the player, run 5 places the player near the exit. Neither is E1M1.
Starting at the level's own spawn, the scripted player had **never once**
reached the exit, and a warm start cannot clone a demonstration that does not
exist.

It was not the reward, the horizon, the curriculum or the model. It was four
bugs in the route and one in the option list, and each of them produced an
agent that looked broken while every probe reported open floor:

| what was wrong | what it looked like |
|---|---|
| steps existed one way and not the other | the field had local minima; the route said "no route at all" 400 units from the spawn |
| diagonals cut corners the straight steps forbid | turned into a wall, slid along it, was sent back, forever |
| the crossing test was a sight ray | a route pointing confidently at a wall 20 units ahead |
| `use` is aimed, and the route runs through shut doors | pressed forward against E1M1's first door for the rest of every episode |
| a fixed stride past a 21-unit waypoint | bounced between two cells 45 units apart, bearing swinging 133 degrees |

With those fixed, from E1M1's own spawn, under `speedrun` orders:

```
step  100  hp 107  kills 3/6  items 17  return +15.18  alive
step  160  hp 107  kills 5/6  items 17  return +23.14  alive
step  192  hp 107  kills 5/6  items 17  return +41.25  exited
```

**192 decisions, five of six monsters dead, 107 health, level ended.** Three
doors opened on the way. That is the scripted player - the bar the policy has
to clear - and it is the first time it has existed at all.

#### Where that leaves it

Five runs, and the shape of the story is that **every one of the problems was in
the experiment rather than in the model**:

| run | change | result |
|---|---|---|
| 1 | baseline | -3.19 return; diagnosed as a greedy loop |
| 2 | agent's history in the observation | -0.52, and ahead on kills |
| 3 | one mission, 250 decisions | game score ahead, zero kills for EITHER player |
| 4 | arena start | **+0.78 return** over the scripted player |
| 5 | reverse curriculum | **completes the level, 10 of 10 episodes** |
| 6 | the route the teacher follows is fixed | **the scripted player finishes E1M1 from its own spawn** |

Not one of those six changes was a hyperparameter. They were: a reward that
paid for walking into walls, an observation missing the history the teacher
decided on, an episode that never reached combat, a goal reward that never
fired, and a route with dead ends in it. The model and the training loop were
the same throughout.

### What would move this next

In the order the measurements point at, not in the order they are interesting:

1. **Walk the curriculum all the way back.** The start is currently a bounded
   random walk from the exit. "The policy completes E1M1" in the unqualified
   sense means the start reaching the level's own spawn, and the honest
   question is how far back the win rate holds before it collapses. That is a
   long run, not a new mechanism.
2. **Both skills at once.** `--arena` teaches fighting and `--curriculum`
   teaches finishing, and nothing yet trains one policy to do both. A level is
   finished by a player that can also survive what is in the way.
3. **One mission at a time, then the mix.** `--mix` splits an already small
   budget three ways and asks the policy to learn instruction-following on top
   of playing. Beat the baseline on each mission alone, then re-introduce the
   mix and measure per-mission.
4. **Batching decisions across parallel games.** The profile says 6.8 ms of
   every decision is fixed cost paid per CALL - two device syncs and the
   dispatch overhead of a six-layer encoder. Eight games stepping together
   would amortise it eight ways, and the engine already packs one state and all
   its options into a single batch; what it cannot yet do is pack several
   states. That is the one change with a multiple in it rather than a
   percentage.

## What is not claimed

- **The run from E1M1's own spawn is the SCRIPTED player, not the policy.** It
  starts where the level starts, kills five of six monsters, opens three doors
  and ends the level in 192 decisions, in one continuous episode. That is the
  bar, and it is what the recorded video shows. The trained policy finishes
  from a curriculum start, and training one from the spawn now that a
  demonstration of finishing exists is the next run, not a result.
- **E1M2 and E1M3 cannot be routed at all**, and the engine says why: the
  exit's own side of the level is 216 cells on one and 48 on the other, walled
  off from everything the player can reach. A distance field over geometry
  cannot cross a teleporter or a wall that a switch lowers, and that is what
  those two exits are behind. Transfer to an unseen map is a fair question and
  this is not yet a fair test of it.
- **The fighting policy and the finishing policy are different runs.** Nothing
  here yet trains one policy that does both.
- The sentence encoder is frozen by default (`--train-encoder` to change it):
  a few hundred high-variance policy gradients per iteration are not enough to
  move 22M pretrained parameters anywhere useful, only enough to damage the
  language understanding that made the option text readable.
- `--mix` trains one policy over three missions; whether it has learned to
  *read* the instruction rather than average over them is measured by scoring
  it per mission, which is what `eval --mission` does.
- Only E1M1 is exercised end to end. `--map` accepts the rest of the shareware
  episode and the scripted player explores them, but see the routing note
  above: on E1M2 and E1M3 it is exploring, not heading anywhere.

---

Swedish Embedded AB builds realtime decision systems that run on the customer's
own hardware - reading a machine's real state, choosing among actions that
machine defines at run time, in milliseconds, with no text generator in the
loop. If your team needs judgment inside a control loop, you can procure our
services by sending an email to info@swedishembedded.com.
