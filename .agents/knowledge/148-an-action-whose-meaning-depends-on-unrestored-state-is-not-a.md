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
