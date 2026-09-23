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
| `uvmax` | every monster, every secret, then the exit | the category itself - see below |

Train with `--mix` and the mission is sampled per episode, so one policy has to
read what it was asked to do.

**Search, then compression.** Imitating a hand-written teacher cannot exceed
that teacher, and no teacher worth writing by hand plays DOOM to 100%. So the
loop has two halves: a search that keeps an archive of places reached and
returns to them to explore further, and a cloning phase that compresses what
the search found into weights. The archive is the durable artifact; the weights
are a lossy compression of it. See [the search half](#the-search-half).

**Realtime.** Measured on one Tesla P40, a decision is 28-33 ms end to end -
about 28 ms of model and 5 ms of game - so the agent decides ~30 times a second,
well above the rate a human plays at.

## Requirements

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

## Quick start

```bash
make samples/decision/doom/build
make samples/decision/doom/run ARGS="probe --doom-bin … --wad … --encoder …"
```

`probe` plays one scripted episode and exercises process, socket, observation,
action, reward and frame with no trained policy. Run it first on a new machine:
if it works, everything else is a matter of flags.

Below, `$D` stands for those three flags. Every command works headless.

## Commands

| command | what it does |
|---|---|
| `search` | **discovery.** Finds solutions; with no `--head` it loads no encoder and opens no device, and with one the policy joins the operators as a proposal distribution |
| `learn` | fit a head to the decisions a search kept. Needs no engine and no WAD |
| `gate` | play two heads over the **same** episodes and decide, by paired sign test, whether the candidate may replace the one in service |
| `probe` | one scripted episode, end to end, no policy |
| `train` | warm-start on the scripted player, then search, self-imitation, DAgger or PPO |
| `fit` | fit the head to the teacher and report how often it agrees - no reward, no critic |
| `whatif` | go back to decisions and take a different one, to see what it was worth |
| `eval` | score a policy and the scripted player on the **same** episodes |
| `play` | run episodes showing every decision, optionally to an MP4 |
| `bench` | time one decision against state length and option count |

`doom --help` prints every flag, including the shared training flags from
`brain`'s control pipeline.

## Recipes

### Score a policy on every level

```bash
DOOM_BIN=… WAD=… ENC=… DBIN=… ./scoreboard.sh out/scores.csv 0            # teacher
DOOM_BIN=… WAD=… ENC=… DBIN=… ./scoreboard.sh out/scores.csv 1 head.safetensors
```

One row per level rather than one averaged number, because an average hides
which levels moved. `STEPS` defaults to 3000: success here is every monster,
every item, every secret and the way out, and at six hundred decisions the
episode ends long before any of that is settled, so the number rewards whatever
pays fastest and punishes anything that invests. `SKILL=3` is Ultra-Violence.

`./plot-progress.py out/scores.csv out/plot/` draws progress per level per
generation.

### Search and compress

```bash
doom train $D --maps 1,2,3,4,5,6,7,8,9 --skill 3 --mission clear --reward gauge \
  --max-steps 3000 --warmup 36 --warmup-keep 0.5 \
  --explore 12 --explore-steps 60 --archive out/archive.json \
  --self-imitate 6 --gauge 3 --save out/doom-policy.safetensors
```

`--explore` is the search half and `--self-imitate` is the compression half.
`--archive` is what makes a run a *generation* rather than a fresh start:
without it every run searches from nothing and a level solved once can be
quietly lost again.

### The loop that improves itself

```bash
DOOM_BIN=… WAD=… ENC=… DBIN=… ./improve.sh 10 1800
```

`SEARCH -> COMPRESS -> SELECT -> better SEARCH`, ten generations of it, across
every level of the episode. It is three commands in a loop and each one can be
run on its own:

```bash
doom search $D --maps 1,2,3,4,5,6,7,8,9 --skill 3 --mission uvmax --reward gauge \
  --max-steps 3000 --search-budget 1800 \
  --archive out/arc --lessons out/lessons.jsonl \
  --head out/serving.safetensors --encoder $ENC   # omit both to search alone
doom learn --encoder $ENC --lessons out/lessons.jsonl --save out/gen4.safetensors
doom gate  $D --maps 1,2,3,4,5,6,7,8,9 --skill 3 --mission uvmax --encoder $ENC \
  --head out/gen4.safetensors --incumbent out/serving.safetensors
```

See [Running the loop](#running-the-loop) for what compounds between
generations and what does not.

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

### Prove it generalizes

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
is real-time rather than a slideshow. Recording steps the game one tic at a time
so every frame can be kept, which is not bit-identical to an unrecorded run -
measured, 228 decisions against 226, same kills, same health, both finishing -
so a recording is a close illustration of a run rather than the run itself.

## How it works

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
brain::ControlPipeline            encoder + decision head, search and cloning
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

### What the agent is not told, and why that matters more

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

### What it is told: what it just saw

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

Steps the grid cannot walk but a player can cross are modelled rather than
excluded. A **teleport** linedef is a portal edge: the two cells either side of
one are geometrically as far apart as the level is wide, so the flood carries an
explicit edge from the pad to its destination and the route can aim at a
teleporter as a way of getting somewhere. A step too high to climb **onto or off
a sector that moves** is a lift ride rather than a wall, since a player steps on
at the bottom and off at the top, and the option says which it is.

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
on a **prefix** - how far along the route the run has ever got, how many
monsters it has killed, items taken and secrets found, how much health it has
now, how many decisions it has survived - every one of those either a running
maximum or a monotone counter, so the score exists after every decision. The
check is visible in any run's log: under `--reward gauge` an episode's `return`
and its `progress` print the same number.

**The score is DOOM's own definition of finishing a level**, which is what the
intermission screen reports:

```text
finished                       1.0
  + full clear            up to 1.0     kills .5, items .25, secrets .25
  + walked out healthy    up to 0.25
not finished              under 1.0
```

So above 1.0 means it got out, and 2.0 means it got out having taken the level
apart. That is deliberate: a loop can only climb the number it is given, so the
number has to be the goal. Leaving kills, items and secrets out of it - which
this scored for most of its life - ranks a run that sprinted past everything
level with one that cleared the map.

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

### UV-Max, and searching for it

`--mission uvmax` scores DOOM speedrunning's **Max** category at Ultra-
Violence: every monster, every secret, then the exit. One number in `0..=2`,
and `2.0` is exactly a completed category - which is what lets the search
recognise a solution without knowing anything about DOOM.

Items are not in it. The category does not require them, and scoring them
ranks a run that swept up forty health bonuses above one that found the last
secret.

**The exit is worth less than clearing the level** (0.6 against 1.3), which is
the opposite of every other mission here. Stepping on a DOOM exit ENDS the
level: a run that leaves early has not taken a shortcut, it has destroyed its
own episode with the work undone. A score that ranks "sprinted out" above
"cleared but still inside" teaches the search to throw away exactly the
trajectories worth keeping - measured, under the obvious weighting a full
clear scored 1.00 against a sprint-to-exit's 1.10.

**There is no time term in it**, and a speedrun is scored on time. A penalty
that grows while the episode runs makes dying the cheapest way to stop it
growing, and an agent paid that way learns to end its own run. Time is the
ARCHIVE's tiebreaker instead: achievement decides which solutions are kept,
tics decide between two that achieved the same thing, and the policy inherits
the speed by being cloned from elites that are already time-minimal. You can
watch it work in a campaign's own log - the best cell on E1M1 held its score
of 0.956 while its time fell from 10 659 tics to 9 071.

### Search as its own command

```bash
doom search $D --map 1 --skill 3 --mission uvmax --reward gauge   --search-budget 1800 --archive out/arc --solutions out/solutions.json
```

`search` is the SEARCH half of `SEARCH -> VERIFY -> SELECT -> COMPRESS`, split
out of the training pipeline it used to be a phase inside. It loads no
encoder, opens no device and runs at engine speed - about 19 ms a decision
against roughly 100 ms with a network in the loop, which is most of why a
search budget buys anything.

It hosts the workspace's `brain-search` crate: a quality-diversity archive, a
bandit allocating budget between search operators on measured gain per second,
and an evaluator cascade.

**The contract that makes the result honest:**

> Search may use snapshots. The artifact may not.

Returning to a promising position by restoring an engine snapshot is what
makes the search affordable. But a trajectory assembled out of restores is not
a run anybody could play, so what a campaign PRODUCES is an action list from
the level's own start, and it is not a solution until that list has been
replayed from the start and has reproduced the same kills, secrets, exit and
tic count. The engine is deterministic - two separate process launches on one
seed agree bit for bit - which is what makes that a gate rather than a hope. A
claim that does not replay is reported as a defect, never quietly dropped.

**The operators**, and which one is worth the budget is measured, not chosen:

```
wander   60 decisions, uniformly random      the only one that can produce an
                                             action no teacher would ever pick
probe    60 decisions, mostly the teacher    plausible, then wandering off it
commit   400 decisions of real play          long enough to finish a firefight
chase    the same, from the best cell        pushes the front of the search
frisk    200 decisions, half of them a push  the only way a secret is ever
                                             found by someone not told where
```

**The niche** - what counts as "a different kind of situation" - is where the
run stands, which keys and **weapons** it holds, its health band, how many
monsters are left and how many secrets it has found. Weapons and health are
load-bearing and were nearly left out: items do not count toward the category,
so they were dropped wholesale, which also dropped the shotgun. Without a
weapons axis "in the courtyard with a shotgun" and "in the courtyard with a
pistol" are one cell, the archive keeps whichever got there in fewer tics -
systematically the run that sprinted past the shotgun - and the search then
sets off from the weaker one and loses the fight. Adding those two axes was
worth 2.6x on an otherwise identical campaign.

### Hunting what you saw

A level is not cleared by what walks into you. Every way of fighting here
reacted to what was in front of the player, so the last monsters of a level -
the ones glimpsed once from a doorway - could never be gone back for.

The memory made that impossible on purpose: a monster's exact position goes
stale in ten decisions, because it has moved, and the way to where it was is
the way to where it is not. That is right for aiming and wrong for hunting - a
monster in a room is still in that room. So there is a second, coarser ledger:
the REGION a live monster was seen in, forgotten on the same evidence an item
is (the player went there and looked) rather than on a timer.

Fair play holds. Nothing in it is a monster the agent was not already shown,
and it is a room rather than a position precisely because claiming to know
where the monster is NOW would be the lie the item memory is careful not to
tell.

Measured on E1M1 at Ultra-Violence, one seed, ~600 s: 0.607 without it against
0.801 with, and the arm that hunted found the first secret any campaign had
found - going back for a monster seen once takes the player into the corners
and side rooms it otherwise walks past, which is where secrets are.

### Searching for secrets instead of hoping for them

A DOOM secret is a wall that opens when pushed and looks very nearly like a
wall that does not. Nothing in the observation says which, and saying which
would be handing the agent the answer - so the only fair mechanism is the one
a player uses: go somewhere you have not tried, and push on things.

Pushing on things was already here and it was not a search. The operator
pressed use wherever the player happened to be standing, so most of its budget
went on re-testing wall it had already tested; measured, campaigns totalling
well over an hour on E1M1 found one secret of three. The archive then deleted
what searching it had done, because a trajectory that had pushed on forty
walls and one that had pushed on none were the same cell and the archive keeps
whichever arrived in fewer tics - which is always the one that did not search.

Three changes, and all three are about MEMORY rather than about DOOM:

- **A ledger of walls pushed on** (`memory::Sweep`), at spot-and-facing
  resolution, carried in the engine snapshot like everything else the agent
  remembers. A wall already pushed teaches nothing by being pushed again, so
  the sweep slides along to the next piece of wall instead.
- **Rooms not yet searched are a frontier**, filed the way monsters seen and
  never gone back for are. A room the player has STOOD IN and never pushed on
  is a destination derived from the run's own history - the option is "go and
  search the walls of a room you have not searched yet", routed by the engine
  like any other walk. A room retires when its walls have been tested, not
  when the player walks through it: looking at a wall is no evidence about
  what is behind it, which is exactly the rule a monster sighting DOES retire
  on, and why this is a separate ledger.
- **How much has been searched is part of the archive's name for a
  situation.** Two runs standing in the same room with the same monsters dead
  are not in the same situation when one of them is part-way through testing
  its walls. Without that axis the archive throws the searching away at the
  next admission.

The scripted player ranks the search after fighting and hunting and before
leaving, and only while the level still owes a secret - the counter is on the
player's own status bar, so "0 of 3" is not being told anything.

**A push only counts when it reached something.** DOOM's use range is 64
units, so pressing use in the middle of a room touches no wall - and the
ledger was counting those, which retired rooms as searched on the strength of
the agent having walked about in them pressing air. Half the operator's
decisions went on it. Measured on E1M1 at Ultra-Violence, 420 s, one seed:

| | counting every push | counting only pushes that reached a wall |
|---|---|---|
| best score, of 2.0 | 0.441 at 420 s | **1.039 at 117 s** |
| secrets found | 0 | 1 |
| verified solutions | 0 | **2** |
| "walls tested" | 174 | 19 |

The walls number collapsing is the result, not a cost: the old 174 were
pushes at nothing. Giving those decisions back to the walk is most of the
gain.

Nineteen real wall tests in seven minutes is honest and close to inert,
though, and the reason is structural: every other option this agent has
prefers SPACE, because space is where it is safe to walk, where the route
goes and where a fight can be had. Left alone it rarely stands nose to wall.
Steering the sweep at the nearest wall was tried and is WORSE - 0.331 by 93
seconds against 1.039, with the count of walls actually tested flat after the
first thirty - because walking at the least room also selects the ways
backward, so the walk oscillates into a corner instead of coming alongside
anything. Getting a searcher up against walls is still open.

Measured earlier, on the ledger itself, at 600 s: **0.690 before, 0.817
with the ledger**, and 133 distinct walls tested against effectively none. The
bandit moved with it - `frisk` went from the third-best operator to the best,
and took a third of all the draws.

One thing cost more than it looks like it should have. Asking the engine to
route to the new frontier was a THIRD round trip per decision, and a round
trip is most of what a decision costs with no model in the loop: it took the
search from about 13 ms a decision to about 41 and threw away more than the
frontier was worth. All three ledgers now ask in one call. A frontier costs
what it costs to ask about, which is a separate decision from whether it is
the right frontier.

### What a recorded action has to say

> **Search may use snapshots. The artifact may not.**

A trajectory is only an artifact if it replays from the level's own start.
Replaying means finding, in the option list the agent is offered, the action
that was written down - so what is written down has to name exactly one of
them.

It did not. An action was recorded as what it SENDS to the game, `tics` and
the command list, and on E1M1 that named more than one option on nearly every
decision: "walk toward the shotgun" and "advance" both send `forward 8` held
for six tics. So does "head for the exit" down a corridor. The replay took
whichever came first.

The symptom pointed exactly away from the cause:

```
at decision 144 the replay stands exactly where the search stood -
(169, -3315) facing 1 on tic 1601 - and READS something different there.
```

Position, angle and tic identical - the simulation agrees completely, because
it was sent identical bytes and cannot tell the two acts apart. The AGENT
disagrees, because its own bookkeeping is keyed on which act it took: what it
counts as having tried, what it has committed to following for the next few
decisions. All of that is in the observation it reads next.

Three rounds of engine work went into determinism before this was found -
snapshot fidelity, held keys, turn acceleration, the event ring - and every
one of them was a real bug and none of them was this one. Bit-exact
simulation is necessary and it is not sufficient: an agent whose state
depends on the MEANING of its action has extended the state past what the
simulator holds.

An action is now recorded as `tag|tics|commands`, which is exactly the three
fields `DoomEnv::apply` reads off an option -
`every_field_of_an_option_either_replays_or_cannot_change_the_run`
destructures `Option_` with no `..`, so adding a fourth fails to compile
until somebody says which side of the line it is on.

A second cause sat behind that one and was not about replay at all.
`Memory::clear` - "a new episode is a new world" - cleared only what was in
sight, so every episode after the first **in a process** began holding the
last one's monsters, rooms and pushed-on walls. The visible half of that is
an agent that remembers a level it has not played. The expensive half is that
a trajectory replayed after another one is offered different options and
diverges, for a reason nothing in the trajectory can reveal. Both `Option_`
and `Memory` are destructured exhaustively now, so adding a field fails to
compile until somebody says which side of the line it is on.

**Measured on a 374-cell E1M1 archive, six trails sampled by length: every
one replays from the level's own start, the longest 1466 decisions.** The
audit had never passed before; it was failing four of six.

That is also why generations never compounded. Rebuilding a carried-in cell's
snapshot means replaying its trail, and an ambiguous replay landed somewhere
else - so the search explored one place and filed the result under another.
That replay is checked against the trail's own witnesses now, and a cell that
does not reproduce is dropped instead of used.

The recording format change is breaking and cannot be repaired: every trail
on disk is a list of indices into a vocabulary written the old way. Such an
archive is refused on load and says so.

### What a player can see, and what would be cheating

The agent reads structured state, not pixels. That is mostly a help, and on
one thing it was a crippling handicap: **the cue a human uses to find a DOOM
secret is visual.** Players find them by noticing a wall that looks unlike
the ones beside it - a different texture, or the same texture visibly out of
alignment - and pushing on it. Our agent had none of that. It could only push
on everything, and a level has several hundred wall faces.

So the engine now reports two things it always knew and never said. Both are
things a player perceives; the line between them and the answer is worth
being exact about, because it is the line the whole exercise rests on.

| Reported - a human perceives it | Not reported - it IS the answer |
|---|---|
| the texture names and alignment offset of the wall face in front | the linedef's special |
| what a push came to: `solid`, `worked`, `nothing there` | whether a line is usable |
| that the world changed after a push | where the secrets are |

**The wall face.** `facingWall` carries what is drawn and where it is drawn,
and nothing about what it is for. The offset is load-bearing rather than a
detail: a secret door is very often the SAME texture as its neighbours with
its alignment out of step, so a signal built on the name alone would miss a
whole class of them.

From that the agent works out for itself whether a wall looks out of place -
it tallies the faces it has looked at, and calls one odd when it is far rarer
than the commonest (`memory::Memory::odd_wall`). A level is built from a
handful of textures repeated everywhere, so the ones a run keeps seeing are
the ordinary ones. Measured over one episode: `STARTAN3` seventy-three times,
blank twenty-one, and `DOOR3` once.

It is a guess and it is allowed to be wrong. Plenty of odd-looking walls are
just walls. What it buys is an ORDER to search in, not an answer, and the
sweep still tests everything it can reach.

The cue goes into the option's SENTENCE - "it does not look like the others
around here" - because the sentence is what a policy reads. A ledger the
search consults is no use to a cloned model: it chooses among words and
nothing else, so a cue it cannot see is a cue it cannot learn.

**What a push came to.** `P_UseSpecialLine` already returns whether the push
fired, and `p_map.c` threw the answer away. A player does not: a solid wall
answers with `sfx_noway` and a door that opens is seen and heard to open. It
arrives as an event on the decision that DID it, which is the point - the
secret counter does not move until the player walks into the sector several
decisions later, by which time the credit lands on the wrong choice.

Two things read it. The ledger takes the engine's verdict instead of guessing
from clearance - measured, of 56 pushes in one short campaign **21 reached
nothing at all**, and every one of those was being filed as a wall tested.
And "things this run has opened" is an axis of the archive's niche, on the
same argument that puts keys and weapons there: a run that has opened a door
can reach ground a run that has not cannot, so the two are not in the same
situation standing in the same place.

### Preferring the edge of what has been reached

Selection weight gained Go-Explore's frontier term. A cell with neighbours on
every side is in the middle of ground already covered; one with none is on
the edge of it, and the middle of a swept room is the least likely place for
anything new to be.

`Archive::exploring(&[1, 2], EDGE)` names WHICH axes of the niche count -
`crates/search` cannot know what an axis means, and "one square of floor
further on" is a direction you can walk in where "one more key held" is not.
Naming none leaves selection exactly as it was.

### Closing the loop: the policy as a search operator

`search --head PATH` puts the trained policy in among the search operators, as
a proposal distribution drawn from rather than an argmax taken. That is the
half of `SEARCH -> COMPRESS -> better SEARCH` that makes it a loop. Without it
a campaign is exactly as good as the scripted player it was written with and
gets no better when the model does, so every generation starts from where the
last one started.

Sampled, not greedy, on purpose: a deterministic policy resumed from the same
archived cell walks the same way every time, so as a search operator it would
be worth exactly one evaluation per cell.

The operator is only offered when a policy was actually supplied. An arm that
silently degrades into a different operator still reports its gain under its
own name, and the bandit then spends real budget comparing an operator with a
copy of another one.

### Deciding whether the new policy is better

```bash
doom gate --head out/gen4.safetensors --incumbent out/serving.safetensors \
          --maps 1,2,3 --skill 3 --mission uvmax --eval-episodes 12
```

Both arms play the SAME episodes in the same order, and the decision is
`brain::promote::gate` - a paired sign test plus four bars, shared with every
other thing in this workspace that promotes a checkpoint. Two means cannot
decide it: a candidate that wins hugely on one seed and loses everywhere else
is indistinguishable from one that wins consistently, and this sample has
already watched four verdicts reverse under a three-seed block.

One block per level, so a candidate that wins on average by learning E1M1 and
forgetting E1M3 is refused. It exits 3 on a reject, which is what lets a loop
branch on it without parsing anything.

Without a gate the loop has no ratchet: a generation that produced a worse
policy is adopted exactly as readily as one that produced a better one, and
nothing notices, because the only number anybody looks at is the newest one.

### Running the loop

```bash
DOOM_BIN=… WAD=… ENC=… DBIN=… ./improve.sh 10 1800
```

Ten generations, thirty minutes of search per level per generation. Each one
searches (resuming from the archive the last generation left, and proposing
with the policy it trained), fits a head to every decision any campaign has
ever kept, and then has to WIN a paired comparison before going into service.

Two files per level compound across generations: the archive, which is where
the search has been, and the lessons, which is what it learnt on the way.
Lessons are APPENDED rather than rewritten - with an archive carried in, a
campaign only files the cells it newly reached, so rewriting would fit each
generation to a shrinking slice of the archive that produced it. A rejected
generation still leaves both files better than it found them; only the weights
are refused.

### The search half

Imitation cannot exceed its teacher, and this teacher does not finish a level at
Ultra-Violence. Sampling from the policy does not fix that either: it only ever
finds what is near what the policy already does. Reaching a strategy nobody
demonstrated needs a search, and the one here follows Go-Explore.

An **archive** holds the best trajectory found to each *cell*, where a cell is
not a place but a coordinate of progress: where the player is, plus what has
been achieved there. Two visits to the same doorway, one holding the blue key
and one not, are different cells, because they are different positions in the
problem. An exploring episode picks a cell, **returns to it by restoring the
engine snapshot** rather than replaying the actions that led there, and then
explores randomly from it, repeating each action with high probability so the
walk covers ground instead of jittering.

Four properties make the difference between a search and a random walk:

- **Return is a restore, not a replay.** DOOM is deterministic given the same
  inputs, but a replay of three hundred actions to reach a promising spot costs
  three hundred steps every time and breaks the moment anything upstream
  changes. A snapshot is one request.
- **The archive outlives the level.** A snapshot carries its own episode, map
  and skill, so it reloads its own level whatever is loaded. There is one
  archive per level and one shared pool of slots, and a campaign that rotates
  nine maps therefore compounds on all nine at once rather than restarting each
  time the map changes.
- **Selection is weighted, not uniform.** A cell is drawn with weight
  `worth / sqrt(times_chosen + 1)`, so a cell that has been returned to often
  loses priority and a cell that led somewhere good keeps it. Uniform selection
  spends the budget on the hundreds of cells in the opening corridor.
- **A fragment is scored on what it added.** The obvious mistake is to score a
  fragment by the absolute progress where it ended, almost all of which was
  inherited from the cell it resumed at - which teaches the policy that
  wandering from a good position is good. Measured, fixing this moved a
  generation from 0.46 to 0.73 and cut the decisions cloned per round from 2291
  to 901.

The cloning phase then compresses the archive into weights, imitating only the
trajectories that beat what the policy already achieves. The weights are
disposable; the archive is not.

### The player it has to beat

A baseline that is merely broken makes a learned number unreadable, so the
scripted player is a real one: fight what is in front of you, take what is under
your nose, otherwise go wherever there is most room, preferring the way the exit
lies, circle-strafe rather than stand still in a firefight, switch to the best
weapon you are carrying, and when the last sixteen decisions have gone nowhere,
commit to one direction for eight steps to break the cycle. It is still
deliberately crude: greedy, no map, never retreats from a fight it is losing,
never prioritises the enemy actually shooting at it, and does the same thing
whatever the orders say. That last one is the headroom the learned policy is
supposed to take.

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

## Results

Every level of the shareware episode, at **Ultra-Violence**, 3000 decisions,
mission `clear`, scored with the gauge. Generation 0 is the scripted teacher;
each later generation searches with the archive it inherits and clones what the
search found. Progress is per level, averaged over seed blocks.

| generation | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | summed | kills | exits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 - scripted teacher | 0.27 | 0.09 | 0.01 | 0.10 | 0.01 | 0.15 | 0.00 | 0.54 | 0.02 | **1.18** | 58.8 | 0 |
| 1 - searched, cloned | 0.36 | 0.10 | 0.01 | 0.06 | 0.01 | 0.02 | 0.01 | 0.14 | 0.02 | 0.73 | 50.4 | 0 |
| 2 | 0.36 | 0.10 | 0.01 | 0.06 | 0.01 | 0.02 | 0.01 | 0.14 | 0.02 | 0.73 | 50.0 | 0 |

Read honestly, that says three things.

**The loop runs end to end and the search half works.** An archive grows from 16
to 175 entries over a campaign, cells reached per level rise across generations
(E1M2: 29, 42, 53, 61 as the maps rotate), and the routing is correct on all
nine levels for the first time - two lead to the exit, six to the key the exit
actually needs, and E1M8 to the switch that opens its sealed room.

**On one level a learned policy beat the teacher.** E1M1, 0.27 to 0.36. That is
the first time any policy here has done so, and it is one level.

**Overall it did not, and nothing finishes a level.** Summed progress fell from
1.18 to 0.73, and there are zero exits anywhere. Generation 2 is identical to
generation 1 because no round beat what it was handed, so the selection step
correctly returned its input unchanged. Neither difficulty nor budget is the
wall: at skill 0 with 4000 decisions E1M1 still stalls at the same coordinate.

What survives from every earlier measurement is not numeric. It is the set of
things that turned out to be true about the method, each of which cost a
measurement to find:

- **Imitation cannot exceed its teacher.** Cloning and DAgger produce a better
  and a more robust copy; neither produces a mechanism for beating it. This is
  why the search half exists.
- **An option nothing can ever take is worth auditing for.** Two of the
  teacher's options - change weapon, circle-strafe - were constructible and
  never once selected. Making them reachable roughly doubled the teacher's
  kills, from 23.5 to 57.2 across nine levels; circle-strafe alone was +22%.
- **A phase that selects its best round must include the policy it was handed**
  in that comparison, or a phase whose every round made things worse still
  adopts one of them.
- **An outcome-fitted phase has to aggregate** across rounds and roll out with
  a mixture of the learner and the teacher. Fitting only the newest round is
  Follow-the-Last-Leader, which has linear regret; rolling out with the learner
  alone is full reinforcement learning, which is the problem that phase exists
  to avoid.
- **A Monte Carlo rollout is a high-variance way to value an action.** Re-run
  one candidate and its own score moves; if it moves by more than the gap
  between candidates, no learner can recover a ranking from it. Measuring that
  noise is cheap and belongs beside any claim about how much room there is.
- **One seed is not a measurement.** Four verdicts in this sample's history
  reversed under a three-seed block. Nothing here is quoted from a single run.

## Roadmap for the future

In rough order of how much each is currently costing.

1. **Selection runs on too few episodes to be a selection.** `--gauge 3` keeps
   the best of a generation on three episodes while per-episode spread is 0.0 to
   0.5. The trace of selected scores across a fixed block oscillated 0.081,
   0.039, 0.134, 0.078, 0.061 with no trend - that is choosing noise, and it is
   the likeliest reason a search that demonstrably finds new ground produces a
   policy that does not improve. The gauge budget needs raising substantially,
   or the comparison needs to become a paired one on identical worlds.

2. **The score saturates on the first subgoal.** Progress takes a running
   maximum across *changing* subgoals, so an easy first key fills the measure
   and crossing the rest of the level afterwards earns nothing. A means-goal
   should be worth a bounded share and the exit the rest. Changing it invalidates
   every generation measured so far, so it needs a full re-baseline with it.

3. **Nothing finishes a level at Ultra-Violence.** The route reaches the right
   goal on all nine; what is missing is surviving the walk there. The teacher
   never retreats from a fight it is losing and never prioritises whatever is
   actually shooting at it, and both of those were tried and measured *worse*
   as written - so the fix is a real one, not the obvious one.

4. **"In sight" means line of sight, not field of view.** The engine reports a
   thing when `P_CheckSight` can draw an unobstructed line to it, and that test
   has no cone in it: a monster directly behind the player is reported exactly
   as one in front. A player at the controls sees about ninety degrees. This is
   the one place the observation gives *more* than a player has, it is
   inconsistent with the burning-floor scan beside it - which does use a proper
   ninety-degree fan - and it is why the memory matters less than it should.
   It should be narrowed to a cone.

5. **The route cannot be pointed at a remembered thing.** It floods from its own
   goal, so "walk to where that medikit was" is a straight line and the option is
   withheld when there is no floor that way, rather than routed round the corner.
   In an open room this costs nothing; in a maze it is the difference between a
   memory that can be acted on and one that can only be read.

6. **The navigation grid has no executable witness per edge.** A step is
   admitted when a body fits at both ends and the height difference is
   crossable, which is a necessary condition and not a sufficient one: it does
   not prove `P_TryMove` actually executes the move along the body's real
   trajectory rather than the centre ray, and it does not accumulate stair
   height across a run of cells. Relaxing the collision checks to paper over the
   difference is the wrong direction; each admitted edge should be provable.

7. **One policy that both fights and finishes.** `--arena` teaches fighting and
   `--curriculum` teaches finishing, and nothing here yet trains one policy that
   does both.

8. **Generalization is unproven.** The train-on-some, score-on-others recipe
   exists and runs; no current number bears on whether what it learns transfers.

The sentence encoder stays frozen by default (`--train-encoder` to change it),
and that is a decision rather than a gap: a few hundred high-variance policy
gradients per iteration are not enough to move 22M pretrained parameters
anywhere useful, only enough to damage the language understanding that made the
option text readable.

---

Swedish Embedded AB builds realtime decision systems that run on the customer's
own hardware - reading a machine's real state, choosing among actions that
machine defines at run time, in milliseconds, with no text generator in the
loop. If your team needs judgment inside a control loop, you can procure our
services by sending an email to info@swedishembedded.com.
