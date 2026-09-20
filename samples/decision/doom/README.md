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
headless run writes exactly this image to a PNG.*

The probabilities are the point. A flat row of bars is a policy that has not
made up its mind; a single bar at 100% is one that has collapsed onto one
option. Neither is visible in a return.

With `--window` on a machine that has a display, the same canvas is a window:

![the window](docs/window.png)

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
read what it was asked to do.

**Realtime.** Measured on one Tesla P40, a decision is 28-33 ms end to end -
about 28 ms of model and 5 ms of game - so the agent decides ~30 times a second,
well above the rate a human plays at. See [what a decision
costs](#what-a-decision-costs).

## What you need

Three things this sample cannot ship, and one script that fetches all of them:

```bash
./fetch-data.sh                 # into ./.data (gitignored), prints the flags
```

| | what | why it is not committed |
|---|---|---|
| the engine | [`mkschreder/restful-doom`](https://github.com/mkschreder/restful-doom) | a C program, built for your machine |
| the game data | `doom1.wad`, the DOOM shareware episode | id Software's, not ours to redistribute |
| the encoder | `sentence-transformers/all-MiniLM-L6-v2` | `brain pull` already manages model weights |

The engine is Chocolate Doom with an HTTP API inside its game loop. This sample
needs the fork above, which adds the agent surface (`/api/state`, `/api/step`,
`/api/episode`, `/api/snapshot`, `/api/frame`), lockstep stepping, a derived
event log, and the geometry queries the observation is built from. Building it
needs `gcc make automake autoconf pkg-config` and the SDL2, SDL2_mixer and
SDL2_net development packages.

The two sides are **versioned together by the observation schema**: this sample
parses the engine's JSON into typed structs with `deny_unknown_fields` and no
optional stand-ins for required data, so an engine that is missing a field - or
has grown one - fails to parse with a message naming it rather than training on
a plausible default.

For the encoder, any BERT-shaped checkpoint directory works (`config.json`,
`model.safetensors`, `tokenizer.json`).

**Nothing is read from the environment and no path is baked in.** Where the
engine, the WAD and the encoder live is what you type on the command line, so a
run is reproducible from its own invocation.

## Running it

```bash
make samples/decision/doom/build
make samples/decision/doom/run ARGS="probe --doom-bin … --wad … --encoder …"
```

Below, `$D` stands for those three flags. Every command works headless.

| command | what it does |
|---|---|
| `probe` | one scripted episode. Run this first on a new machine: it exercises process, socket, observation, action, reward and frame with no trained policy |
| `train` | warm-start on the scripted player, optionally DAgger, then PPO |
| `fit` | fit the head to the teacher and report how often it agrees - no reward, no critic |
| `whatif` | go back to decisions and take a different one, to see what it was worth |
| `eval` | score a policy and the scripted player on the **same** episodes |
| `play` | run episodes showing every decision, optionally to an MP4 |
| `bench` | time one decision against state length and option count |

### Train on generated problems, keep the real levels for the exam

Three fixed levels are not three worlds' worth of practice. DOOM is
deterministic: the same level at a different seed is the same level, so a
rollout over three maps presents three situations however many episodes it runs.

`--scenario` plays levels the engine **builds** instead of the game's own, one
drawn per episode, each generated fresh from that episode's seed. There is no
map to memorise. The game's own levels then stop being training data and become
the exam.

```bash
doom train $D --scenario my-way-home,health-gathering,deadly-corridor,defend-the-line,take-cover \
  --reward gauge --mission speedrun --max-steps 400 \
  --iterations 30 --episodes 8 --warmup 24 --warmup-keep 0.5 \
  --save out/doom-scenarios.safetensors

doom eval  $D --maps 1,2,3,4,5 --mission speedrun --max-steps 900 \
  --eval-episodes 10 --head out/doom-scenarios.safetensors
```

The nine scenarios are ViZDoom's, rebuilt on this engine. The designs are
theirs; none of the files are - theirs are UDMF geometry driven by compiled ACS
and this engine reads neither, and porting the observation onto ViZDoom's engine
would mean losing most of it. Its state gives line endpoints and a blocking
flag, with no sector special, no linedef special and no automap flag, so burning
floor, doors, switches, the exit and the fair-play seen set are all underivable
there. Training and scoring have to produce the **same text** or nothing
transfers, which makes one engine a correctness requirement rather than a
convenience.

```
basic                     one monster: see, face, shoot
deadly-corridor           advance under fire
defend-the-center         a ring closing in
defend-the-line           the same, from a wall
health-gathering          a burning floor and medkits
health-gathering-supreme  the same, in a maze
my-way-home               find the armour in a new maze
predict-position          a walking target, a slow rocket
take-cover                dodge, and keep dodging
```

A generated level is published under a lump of its own rather than into an
`ExMy` slot, and the episode and map the engine believes it is playing never
move off 1 and 1. Nothing of the game's own is displaced, the same process can
score on E1M1 straight afterwards, and the sky, music and intermission art stay
the ones the IWAD ships.

A training run's own stdout is the chart: `./plot-run.py out/train.log docs/`
draws return, wins and progress per iteration from the lines it prints.

### Prove it generalizes, which is the only claim that matters

Train on some levels and score on one that was not among them.

```bash
doom train $D --maps 1,2,3 --mission speedrun --max-steps 400 \
  --iterations 10 --episodes 6 --warmup 12 --warmup-keep 0.5 \
  --save out/doom-explore.safetensors

doom eval  $D --maps 4,5 --mission speedrun --max-steps 900 --eval-episodes 8 \
  --head out/doom-explore.safetensors
```

The scored table has an `exits` column and a `burned` one. The first is whether
it finished; the second is health lost to burning floor, which is the single
most reliable way to die on a level nobody has walked before.

### Watch it play, and record it

```bash
doom play $D --head out/doom-policy.safetensors --play 4 --fps 35 --record out/run.mp4
```

`--record` encodes every decision straight into an MP4 as it is drawn: raw
frames piped to `ffmpeg`, so the memory cost is one frame however long the run.
It captures one frame per game **tic** rather than per decision, so the motion
is real-time rather than a slideshow.

## How it is put together

```
restful-doom  --apilockstep        the game, frozen between steps
     |  HTTP/1.1 keep-alive, one request per decision
     v
src/doom.rs      process lifetime, transport, episodes, snapshots
src/obs.rs       JSON -> typed state -> the text the model reads
src/action.rs    the state -> what can be done right now
src/memory.rs    what was seen a moment ago, placed from where you stand now
src/report.rs    how far an episode got, on a scale two runs can be compared on
src/env.rs       reward, missions, the scripted player
src/view.rs      the inspector: frame + state + decision + reward
     |  brain::Env
     v
brain::ControlPipeline            encoder + decision head, cloning/DAgger/PPO
```

The engine side is in the DOOM repository rather than here: an HTTP layer that
survives a request per tic, lockstep stepping, an observation in one round trip,
reproducible episodes, a derived event log, a framebuffer endpoint, state
snapshots, and the route below.

### What the model is given

Everything, as prose, because that is what a decision model reads:

```
health 101 armor 0, pistol with 50 rounds. Killed 2 of 29 enemies, 1 of 37 items, 0 of 3 secrets.
In sight: former human sergeant close at 180 units 12 degrees left, coming for you, the one you have been hitting.
Last seen: medikit 220 units behind you; imp 340 units to your right, which moves.
An explosive barrel sits 279 units 78 degrees right.
Room to move: 320 ahead, 44 left, 320 right, 320 behind.
The exit switch is 2217 units 53 degrees left, with 0 units of clear floor that way.
You have been through here 3 times. You have not actually moved for 4 decisions. 22 patches of this level explored.
Just now: took 15 damage.
```

Four details are load-bearing, and each was wrong once:

- **Bearings, not map angles.** "12 degrees left" is actionable; "at 143
  degrees" needs the reader to know its own facing and do the subtraction.
- **Threats are monsters only.** Barrels are shootable too, and reporting one as
  an enemy meant "kill everything" was scored against a target that does not
  count toward the level's kill total.
- **Clearance is walkability, measured by tracing the level's lines** - not
  "could I stand exactly there". Doom slides a player along whatever they brush,
  so a decorative pillar beside the path read as a wall, and the option list hid
  "walk forward" at a spot the player then crossed 83 units of.
- **The last line is the agent's own history**, which the game does not report
  and the policy cannot remember - every decision is an independent forward
  pass. Leaving it out was worth 2.67 of return on its own: the first trained
  policy lost to the teacher by 3.19 and by 0.52 once it was added, with nothing
  else changed. Cloning a teacher that decides on history the policy cannot see
  teaches a mapping that does not exist.

### What the agent is NOT told, and why that matters more

An agent handed the answer is not solving the problem, and the way that shows up
is subtle: it plays well and learns nothing transferable. The route used to be a
distance field over the **whole level**, flooded from the exit at level load -
through doors that had not been opened, past a key whose existence was not yet
known. A policy trained against that follows a line painted on the floor, and
takes the same path every time, because the path was never its decision.

So the observation was audited against what a player at the controls can have:

| Was in it | Why a player cannot have it |
|---|---|
| The exit's position, from the first tic | You cannot know where a level's exit is before walking a step of it |
| A route to it over unexplored ground | A solved map of rooms nobody has entered |
| Which key the exit needs | Before ever seeing the locked door |
| Which switch opens the way | Found by flooding the exit's own island |
| Monsters behind walls, with bearings | A census; a player has sound |
| `Killed 5 of 6` | DOOM shows the totals after the level |

The engine already knew what had been seen, because the renderer marks every
wall it draws with `ML_MAPPED` for the automap. That is the honest map. The
distance field now floods **only over sectors with a wall the player has looked
at**, and when the exit has not been found the route leads to the **frontier** -
the nearest cell with a step into somewhere unseen - and says so. The exit
becomes the goal the moment it has been seen. `--full-map` restores the old
behaviour as a control.

What that costs, with the scripted player:

| | with the whole map | seeing only what it has looked at |
|---|---|---|
| E1M1 | finishes, 156 decisions | **finishes, 247 decisions** |
| E1M2 | finishes, 366 decisions | stalls two cells from the frontier |
| E1M3 | finishes, 688 decisions | dies in the hellslime it explores into |

### What it IS told: what it just saw

Fair play cuts the other way. If the observation is only what is in line of
sight right now, then turning away from a medikit deletes it, and the only way
anything is picked up is by walking into it. That is not honesty, it is amnesia.

So what has been seen is kept for a while. Nothing enters that memory which was
not in an observation the agent was already shown. Two details make it a memory
rather than a map:

- **Stored as a position, reported as a bearing.** Walking away from a
  remembered medikit turns it into "220 units behind you", which is true rather
  than stale. Map coordinates never leave the module, because a policy told
  where things are in map units would be memorising a map - the one thing the
  generated levels exist to make worthless.
- **A monster goes stale and an item does not.** A monster is forgotten after
  ten decisions out of sight, because by then it could be anywhere. An item is
  forgotten only on evidence: the player reached the spot and found nothing,
  which is also what picking it up looks like from here.

The same memory answers a second question. Two identical sergeants at mirrored
bearings produce two option sentences differing only in "left" or "right", and a
memoryless policy flipping between them is responding correctly to a state in
which they are indistinguishable - which, watching a recording, is the thing
that least resembles someone playing. The memory holds the most health each
thing was ever seen with, so anything below its own best is one this player has
been shooting, and the attack option says so.

### Where the route comes from

Worth being precise about, because "the model does pathfinding" would be false
and "the engine tells it where to go" would be misleading.

**The geometry is solved in the engine.** The walkable floor is sampled onto a
uniform 32-unit grid - half the player's width - and the engine computes whether
a body fits in each cell, which of the eight steps out of it are possible, and a
breadth-first distance field over those steps flooded from the goal. The agent
gets two numbers: how far the goal is **along ground it can walk**, and which
way to set off - a waypoint, string-pulled. Grid plus funnel, nothing learned.

**What is not in the engine is the decision.** The route is one option among
those rebuilt every step, competing with attacking, taking a pickup, backing
off, sidestepping and opening a door. Nothing makes the agent follow it. That is
the dilemma you actually face: *the exit is 3400 units that way, there is a
sergeant at 180 units, and a medikit 90 units off the path*. A route solves
"which way is the exit", which is geometry and has a right answer. It does not
solve "is the exit what I should be doing", which does not.

When the way out is shut the route leads to **whatever opens it**. A locked door
is a wall until its key is held, so the field floods from the key and the
observation says the way out is locked and which colour it needs; picking the
key up changes which doors are walls and the route goes back to the exit with
nothing else having to notice. A sector only a **switch** opens is also a wall,
and the route leads to a standing spot in front of the switch that operates that
sector's tag - found by flooding the exit's own island and sampling just off
each of that sector's lines to see which island it borders.

The route runs **through** shut doors on purpose, because a player opens them.
So "the way out starts 7 degrees to your right" while the player cannot walk
there is the normal state of affairs at every door in the game. The observation
names what is in the way and whether pressing use will open it, and there is an
option aimed at the door - aimed matters, because `use` reaches 64 units in the
direction the player is **facing**.

Nukage and lava end a run as surely as a monster does. The distance field
**excludes damaging sectors**, falling back to allowing them only when that
leaves the goal unreachable; the observation says the floor is burning and which
way its edge lies; and the scripted teacher treats standing in damage as
overriding everything else - but only when there is somewhere dry to go, since
on `health-gathering` the whole floor burns and an emergency that never ends is
not an emergency.

### What a decision is paid for

`--reward gauge` pays a decision exactly what it moved the score the run is
finally kept or discarded on, so an episode's undiscounted return **is** that
score:

```text
sum_t [ M(h_t+1) - M(h_t) ]  =  M(h_T) - M(h_0)  =  M(h_T)
```

That identity is algebra, not tuning. It holds because the score can be computed
on a **prefix** - how far along the route the run has ever got, how much health
it has now, how many decisions it has survived - so it exists after every
decision. The check is visible in any run's log: under `--reward gauge` an
episode's `return` and its `progress` print the same number.

The default, `--reward shaped`, is the older scheme: separately chosen weights
for kills, items, damage taken, floor newly walked and route closed. It has no
such guarantee, and measured, about 92% of a shaped episode's return was the
exploration bonus - which the score does not read at all. Every training run
before the gauge existed was *trained* on one number and *kept* on another.

Two things about the shaped scheme are worth keeping even so, because both were
learned the hard way:

- **Closing on the exit is not the dense reward.** Paying for a straight line to
  the exit is potential-based shaping, so it provably leaves the optimum alone -
  but in a building the straight line goes through walls. Measured: a trajectory
  spent 90 of its 120 decisions shuttling between two spots, walking at the wall
  the exit is behind, and was paid for every one of them. The `--approach` term
  that replaced it pays for cells of **route** closed, which is zero for
  standing still, zero for going out and coming back, and zero for walking into
  a wall. Ablated, it was worth +0.023 ± 0.040 - i.e. nothing measurable.
- **A count-based exploration bonus fills the gap instead.** The first visit to
  a patch of floor pays and the *n*-th pays `1/sqrt(n)`. Walking into a wall
  discovers nothing and earns nothing. The agent is not told that walls are bad;
  it is paid for finding out.

### The player it has to beat

A baseline that is merely broken makes a learned number unreadable, so the
scripted player is a real one: fight what is in front of you, take what is under
your nose, otherwise go wherever there is most room, preferring the way the exit
lies, and when the last sixteen decisions have gone nowhere, commit to one
direction for eight steps to break the cycle. It is still deliberately crude:
greedy, no map, never retreats from a fight it is losing, never prioritises the
enemy actually shooting at it, and does the same thing whatever the orders say.
That last one is the headroom the learned policy is supposed to take.

Only its **best** episodes are cloned (`--warmup-keep`). A heuristic teacher is
good in the situations it was written for and arbitrary everywhere else, and its
bad episodes are bad in a specific, *learnable* way - so cloning them all
teaches the policy something the gradient then spends its samples unlearning.

### Reading a run that went wrong

Every scored episode ends with one line saying which of three things happened -
it finished, it died, or it stopped making progress - and the numbers that
distinguish them:

```
doom: finished in 366 decisions; 21 of 41 kills, 100 health; closed on the red
      key from 2112 units to 32 at decision 76; closed on the switch that opens
      the way from 5728 units to 0 at decision 290; closed on the exit from
      1312 units to 32 at decision 365; damage 14 to FORMER HUMAN

doom: died at decision 533 to IMP, at (234, -1552); 42 of 74 kills, 0 health;
      closed on the blue key from 4576 units to 64 at decision 195; damage 105 to IMP

doom: stalled: the last 60 of 1400 decisions covered 3 patches of floor around
      (-1420, 2059); ... a wall in the way
```

Beside each is a **progress** number, because the line is prose and two runs
need comparing. Return cannot do it: almost all of it is the single payment at
the exit, which neither of two failing runs collected. Progress is what a person
reading the two runs would say instead - finishing beats everything, and short
of that it is how far along the way the run got by **route** distance over
walkable ground, discounted by how close it came to dying:

```
died at decision 368 on E1M3, a quarter of the way        0.23
stalled at a switch on E1M3, a third of the way, 44 hp    0.36
finished E1M1 with 101 health                             1.25
```

Two things it refuses to count. Unexplored ground is not a goal, because the
frontier moves every time it is reached. And where there is nothing to walk
toward - the whole of `health-gathering` - lasting IS the task, scaled so it can
never outrank a run that actually went somewhere.

Three pieces of machinery make those lines possible. **The engine attributes
damage**, so `hurt` and `death` carry the inflictor's name - or, when there is
no inflictor, which only a damaging floor has, the sector special by name.
"Died at decision 276" is not a diagnosis; "died to an IMP having lost most of
its health to nukage" is. **Each goal is measured on its own**, since the route
leads to a key, then a switch, then the exit, and one number for all three reads
as though the part that went right was the failure. And **stalling is a distinct
outcome, and the commonest one** - an episode that ends that way asks the engine
what the route made of the spot it stopped in, there and then, while the level
is still in the state that produced it.

## What was measured

Every number below is a real run on this repository's own hardware (two Tesla
P40s, one used). Nothing is projected.

### What a decision costs

`doom bench` times one decision against the two things its cost could scale
with:

| state (words) | options | ms/call |
|---:|---:|---:|
| 4 | 1 | 6.8 |
| 4 | 8 | 9.9 |
| 4 | 24 | 17.0 |
| 40 | 8 | 10.7 |
| 200 | 8 | 24.7 |

Which decomposes into **6.8 ms fixed, 0.09 ms per state word, 0.44 ms per
option**. Nearly half of a short decision is the fixed cost of running a
six-layer encoder at all rather than anything about the text - per-dispatch cost
and two device syncs, not something this sample can edit its prose out of. The
game itself is ~5 ms per decision, and lockstep decouples it from the 35 Hz
clock entirely: with no model in the loop, **2 966 tics/s, 85x realtime**.

### Where the learning actually stands

Four measurements, in the order they were taken. Together they close the
question of why six separate interventions on the policy gradient produced two
real bug fixes and no improvement.

**1. Can the model express the decision?** Yes, comfortably. `doom fit` fits the
head to the teacher by plain supervised learning - no reward, no critic - then
asks how often its own best action IS the teacher's:

| | agreement | floor |
|---|---:|---:|
| unfitted head, unseen episodes | 23.0% | 24.9% |
| fitted head, the fitted episodes | 84.8% | 21.6% |
| fitted head, **unseen** episodes | **88.8%** | 24.9% |

The floor is the better of guessing uniformly and always naming the same
position in the option list. The second matters: the teacher does not choose
uniformly and the options do not arrive in a random order, so a head that reads
nothing useful still beats `1/n`. The unfitted head sits exactly at that floor,
which is the control that makes the other two readable.

So the frozen 384-number representation carries the decision, 444,673 parameters
express it, and 96 distinct worlds are enough to generalize from. Three live
suspicions, all three closed.

**2. Does that survive the policy acting on its own?** No, and this is the
largest single effect anywhere in this sample:

```text
88.8%  agreement on states the TEACHER reaches
  34%  agreement on states the STUDENT reaches
```

Cloning only ever produces labels on the teacher's trajectory. The student's
first mistake takes it somewhere that trajectory is silent about, so it makes a
second; the errors compound, and the classic bound on them grows with the
**square** of the episode's length rather than linearly.

**3. Does closing that gap help?** It closes, and it does not help. `--dagger N`
runs the student, asks the teacher at every state the student reached what it
would have done there, records the answer *without executing it*, aggregates
with everything collected so far and refits:

| round | agreement on its own states | fixed block |
| ---: | ---: | ---: |
| 1 | 34% | 0.670 |
| 2 | 65% | 0.646 |
| 3 | 78% | 0.615 |
| 4 | 80% | 0.634 |
| 5 | 76% | 0.596 |

Agreement more than doubles and heads for the 88.8% ceiling. The score does not
follow it up, and scored afterwards against the scripted player over 16 shared
episodes: **0.73 to its 0.76, six exits each**.

Which is the answer rather than another null. Every imitation method has the
teacher as its ceiling, and this teacher scores 0.76. Agreeing with it more
often cannot carry the student past it - only *to* it, which is where the
student already was. Cloning, DAgger and PPO over both now land in the same
place for a measured reason.

**4. Is there room above the teacher?** Yes, and it is concentrated. `doom
whatif` goes back to decisions the policy faced, takes something other than what
the teacher chose, lets the teacher play the rest out, and scores the whole
trajectory - prefix included - with the same gauge the run is judged on. Two
runs of 20 episodes, three candidates at each of ~160 decisions, differing only
in how long the departure from the teacher is held before handing back:

| held for | the options all led to the same place | some alternative beat the teacher | by | gain a decision |
|---|---:|---:|---:|---:|
| 1 decision | - | 13% | 0.064 | 0.008 |
| 30 decisions | 71% of the time | **17%** | **0.072** | **0.012** |

Read the first column first. At **71% of decisions every candidate leads to the
same place**, so no method can improve them and no method should be judged on
them. Of the 29% where the choice has a consequence at all, the teacher fails to
pick the best available at roughly **three in five** - worth 0.072 of a score
whose full range is 1.25.

Holding the deviation longer raises every number, which is the point of the
dial: a teacher good at recovering undoes whatever one decision did, so a
single-action deviation understates by construction. It is also why an earlier
22-decision sample of this measurement said 0.003 / 9% / 0.001 and had to be
discarded - a warning about reading small samples, not a second result.

This is the only signal measured in this sample that is **not bounded above by
the teacher**, and it is what a ranker trained on measured outcomes rather than
on the teacher's choice would be learning from.

`--wide` draws the alternatives at random instead of from what the policy ranks
highest. The default is the right set for deciding whether to train on this,
since it is what an update would move toward; it is the wrong set for asking
whether room exists, because a policy fitted to the teacher ranks the teacher's
near-duplicates highest and so asks about the actions least likely to lead
anywhere different.

**Going back is a real snapshot, not a replay of the actions that led there.**
That distinction is load-bearing: the observation reads `ML_MAPPED` to decide
what the player has seen, `ML_MAPPED` is set by the renderer, and rendering is
not part of the deterministic simulation. Replaying a prefix reached the same
player and monsters with a *different set of options* three times in ten - and
the replays that survived were the short prefixes, so what was left was biased
as well as smaller. The engine's `/api/snapshot` writes the vanilla savegame
path to memory instead of a slot, and that format archives line flags.
`Env::hold` and `Env::resume` are the SDK's side; `DoomEnv` restores what lives
on the client too, because half a run restored is worse than no restore - it
looks like an answer. The measurement checks itself: after every restore the
options offered must be the options offered before, or the decision is discarded
and counted. Across both runs above, 966 restores, it never fired.

### Two things the policy demonstrably learned

**Slime is bad.** Eight PPO iterations of six episodes on E1M3, warm-started
from a teacher that is naive about slime, dies in it every episode, and is
deliberately left that way:

```
              return     game    kills    items    exits   deaths   burned
scripted       -4.79    -7.64     15.0      3.0        0        4     95.0
policy         +1.93    -1.88     17.8      3.8        0        1     66.5
```

Deaths 4 of 4 down to 1 of 4, health lost to the floor down 30%. It has not
stopped wading and neither player finishes E1M3, but the signal is there and the
gradient follows it. Two things had to be true first, and neither was about the
policy: the feature has to be **in the state** (the observation used to say only
that the floor already underfoot was burning, so "learn that slime is bad" was
unlearnable rather than hard), and the reward has to **agree** (nukage does 5
points every 32 tics and a decision is 4 to 6 of them, so at the old weight
wading cost 0.018 while exploring paid 0.02 - wading was profitable, and the
agent that kept doing it was correctly learning what it had been told).

**On unfamiliar ground it survives longer and fights better.** Trained on
E1M1-E1M3 under fair play, scored on **E1M4**, which neither player has seen:

```
              return     game    kills    items    exits   deaths   burned
scripted       -4.70    -7.99      5.3      0.0        0        4      0.0
policy         -3.74    -8.75      9.7      0.3        0        5     29.3
```

Neither finishes. The policy lasts 689 decisions against 461 and kills 9.7
against 5.3, where the teacher spends its last 60 decisions circling two patches
of floor and dies to the same imp at the same spot five times out of six. It
burned 0.0 on the training levels and 29.3 here, so what it learned in run 9 was
closer to "this pool" than to "burning floor".

## What is not claimed

- **Reinforcement learning has not beaten behaviour cloning here.** PPO over
  generated scenarios ends at the teacher's score. Six interventions on the
  learner were measured and none improved it. What the sample demonstrates today
  is an environment, an observation, and a policy that **imitates** a
  hand-written teacher - not one that improves on it. Part 4 above measures the
  room that exists to improve on it - the teacher is beatable at about one
  decision in six, worth 0.072 - and nothing here has taken it yet.
- **Generalization is not proven.** A policy trained on E1M1-E1M3 does not
  finish E1M4, and neither does the teacher.
- **The policy finishes E1M1 from the spawn but does not beat the script.** Six
  of twelve scored episodes end the level, which is what the scripted player
  manages on the same twelve. Finishing is a result; winning is not one yet.
- **"In sight" means line of sight, not field of view.** The engine reports a
  thing when `P_CheckSight` can draw an unobstructed line to it, and that test
  has no cone in it: a monster directly behind the player is reported exactly as
  one in front. A player at the controls sees about ninety degrees. This is the
  one place the observation gives MORE than a player has, it is inconsistent
  with the burning-floor scan beside it - which does use a proper ninety-degree
  fan - and it is why the memory matters less than it should: turning away from
  a medikit does not currently lose it, only a wall does. Narrowing it would
  make every measurement above incomparable, so it has not been done.
- **The route cannot be pointed at a remembered thing.** It floods from its own
  goal, so "walk to where that medikit was" has to be a straight line, and the
  option is withheld when there is no floor that way rather than routed round
  the corner. In an open room this costs nothing; in a maze it is the difference
  between a memory that can be acted on and one that can only be read.
- **Only E1M1 is finished under fair play.** With the whole map handed to it the
  scripted player finishes all three; seeing only what it has looked at, it
  finishes E1M1 and not E1M2 or E1M3.
- **E1M3 is not finished.** The scripted player crosses it - blue key at
  decision 195, within 480 units of the exit at 531 - and then loses a fight.
  That is a combat failure, not a routing one.
- **The fighting policy and the finishing policy are different runs.**
  `--arena` teaches fighting and `--curriculum` teaches finishing, and nothing
  here yet trains one policy that does both.
- **Teleporters are not in the route.** A teleport linedef moves the player
  instantly and the distance field is over geometry, so the two cells either
  side of one are as far apart as the level is wide. Nothing breaks when the
  player steps on one - the field re-plans from where they land, which is what
  happens to a player who walks onto a pad without knowing what it is - but the
  route will never *aim* at a teleporter as a way of getting somewhere. Not
  exercised end to end: the shareware episode puts the first ones in E1M5.
- **`--record` does not reproduce a run exactly.** Recording steps the game one
  tic at a time so every rendered frame can be kept. A recorded run now tracks
  an unrecorded one closely and ends the same way - 228 decisions against 226,
  same kills, same health, both finishing - but it is not bit-identical, so a
  recording is a close illustration of a run and not the run itself.
- The sentence encoder is frozen by default (`--train-encoder` to change it): a
  few hundred high-variance policy gradients per iteration are not enough to
  move 22M pretrained parameters anywhere useful, only enough to damage the
  language understanding that made the option text readable.

---

Swedish Embedded AB builds realtime decision systems that run on the customer's
own hardware - reading a machine's real state, choosing among actions that
machine defines at run time, in milliseconds, with no text generator in the
loop. If your team needs judgment inside a control loop, you can procure our
services by sending an email to info@swedishembedded.com.
