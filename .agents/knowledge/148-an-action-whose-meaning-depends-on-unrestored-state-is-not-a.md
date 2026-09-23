<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 148. An action whose meaning depends on unrestored state is not a recordable action

A search that returns to promising positions by restoring snapshots produces
trajectories. For a trajectory to be an ARTIFACT - replayable, and therefore
checkable by anybody - every recorded action has to mean the same thing when
it is played back.

In `samples/decision/doom` it did not, and the failure was invisible for a
long time because from inside the search it looks exactly like success.

## What was measured

Every archived trajectory longer than about sixty decisions failed to replay
from the level's own start, all of them at the SAME decision, with the same
symptom:

```text
at decision 16 the replay stands at (722, -3377) facing 89 on tic 117,
where the search stood at (722, -3379) facing 89 on tic 117 - 2 units apart
```

Same tic, same facing, two map units of position. From there the two runs
drift until the trail names an option that is no longer offered.

The engine was exonerated by direct measurement: restoring a snapshot and
stepping on reproduces stepping on without one, bit for bit, turns included;
and resetting an episode is reproducible across resets and across processes.

## The cause

`go to ground nobody has looked at yet, 352 units of walking ahead` is
computed from the route's FRONTIER, which reads a process-global `visited`
grid inside the engine's route module. That grid is not part of a snapshot.

And the option's TEXT does not determine its COMMAND. The text carries the
route's total distance; the command carries the next waypoint's step, a
different number that appears nowhere in the sentence. Two states with
different route state therefore produce the same sentence and different
inputs - and the trajectory records the sentence.

## The rule

> A recorded action must be either a controller INPUT, or a sentence whose
> meaning is a pure function of state the snapshot restores. Anything else is
> a trajectory that cannot be replayed, and the search will not notice,
> because the search never replays anything.

Two repairs, and they answer different questions rather than being
alternatives:

- **record the input.** A speedrun artifact is a sequence of controller
  inputs - that is what a DOOM demo is - and it is immune to any amount of
  derived state changing underneath it.
- **restore the derived state.** Needed anyway for search QUALITY, separately
  from replay: after a restore the agent is told what has and has not been
  explored on the strength of other walks' history, so it is being pointed at
  a frontier that is not its own.

## What it turned out to be, in the end

The sentence-versus-input problem above is real and was fixed. It was not the
whole of it: the engine's snapshot restored the WORLD and not the things
derived from it, and each missing piece was found by fixing the last one and
watching the first divergence march later - decision 1, then 13, then 16,
then 83.

Five, and the shape of every one is the same:

| missing | what it broke |
|---|---|
| the route's `visited` grid | "the nearest ground nobody has looked at" answered from where OTHER attempts had walked |
| the API's `keys_down` / `target_angle` | a key held for a countdown of tics outlives a step by design, so it is part of the state a decision plays out from |
| the game's own `gamekeydown` | the controller believed it held a key the game believed was up: a turn continued in one run and did not happen at all in the other |
| Doom's `turnheld` | a turn accelerates the longer its key is held, so re-pressing made the first turn after a return a slow one |
| the event-derivation baseline | events are a DIFF against the previous tic, and a restore moves the world to another moment - so it reported "took 1 damage" at full health |

Measured before and after, on raw fixed-point state: 347 001 units and nine
degrees apart, against identical in every field.

## The measuring instrument mattered as much as the fixes

None of this was visible through the observation API, which reports whole map
units. A sixteenth of a unit of momentum is invisible there and is two units
of position twenty tics later, so a divergence that has ALREADY HAPPENED
reads as agreement until it is too large to diagnose - and by then it
presents as "an option was not offered", three hundred decisions downstream.

What made it tractable was a conformance endpoint exposing raw simulation
state, and witnesses recorded in the trajectory itself:

- position, facing and the clock localise the divergence to a few decisions;
- a digest of the OBSERVATION distinguishes "the world differs" from "the
  world agrees and what the agent makes of it does not", which are different
  faults with different fixes;
- keeping the observation TEXT for the first stretch turns the second into a
  one-line diff, which is how "took 1 damage" at full health was found.

## How it was found, which is the transferable part

An **always-on audit** at archive load: replay a sample of stored trails,
spread across their lengths, and report whether each arrives where the
archive claims. Nothing else was going to find it. The search restores
snapshots and carries happily on; the damage appears only at the far end, in
an artifact nobody had tried to use yet. A gate that must be asked for would
not have been asked for (see #1).

The audit only became a DIAGNOSIS rather than a complaint once trails carried
**witnesses** - position, facing and the level clock, written down every few
decisions. Before that, the failure read as "an option was not offered" three
hundred decisions in, which says that two runs disagree and nothing whatever
about where they began to.
