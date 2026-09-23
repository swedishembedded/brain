<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 150. What an action SENDS is not what an action MEANS

A trajectory is only an artifact if it replays. Replaying means finding, in
the option list the agent is offered, the action that was written down - so
what is written down has to name exactly one of them. Write down only what
the action sends to the outside world and it will name more than one
whenever two options send the same thing, and the replay will take whichever
comes first.

Measured on `samples/decision/doom`. Actions were recorded as
`tics|commands` - the engine's whole input - and that was ambiguous on
**nearly every decision of E1M1**: "walk toward the shotgun" (`Tag::Grab`)
and "advance" (`Tag::Advance`) both send `forward 8` held for six tics. So
does "head for the exit" in a corridor, and "fall back the way you came".

The symptom is what makes this worth its own entry, because it points
exactly away from the cause:

```
at decision 144 the replay stands exactly where the search stood -
(169, -3315) facing 1 on tic 1601 - and READS something different there.
```

Position identical, angle identical, tic identical. The simulation agrees
completely, because the simulation was sent identical bytes and could not
tell the two acts apart. The AGENT disagrees, because its own bookkeeping
is keyed on which act it took: what it counts as having tried, what it has
committed to following for the next few decisions, which of its options it
has exhausted. All of that is in the observation it reads next, so the
replay reads a different sentence while standing in exactly the right place.

Three campaigns' worth of debugging went into the engine's determinism
before this was found - snapshot fidelity, held keys, turn acceleration,
event-ring rebaselining - and all of it was real and none of it was the
cause. Bit-exact simulation is necessary and it is not sufficient: an agent
whose state depends on the MEANING of its action has extended the state
beyond what the simulator holds.

The rule, and it generalises past DOOM to any agent with memory,
commitments, or an internal mode:

> A recorded action must name every field the step function reads. Not the
> fields the environment reads - the fields the STEP reads, including the
> ones the agent consults about itself.

Enforce it structurally rather than by care. An exhaustive destructure of
the option type, with no `..`, fails to compile when a field is added until
somebody classifies it:

```rust
fn every_field_of_an_option_either_replays_or_cannot_change_the_run(o: &Option_) {
    let Option_ { tag: _, tics: _, commands: _, text: _, room: _ } = o;
}
```

And note the format change is breaking: every trail already on disk is a
list of indices into a vocabulary written the old way, so those archives
replay as a different run and cannot be repaired. They have to be refused
loudly on load. An archive that loads and quietly produces unverifiable
trajectories is worse than one that will not load, for the same reason a
gate that never runs is worse than no gate (#1).

## The second cause behind it, which was not about replay at all

Fixing the recording moved the failure and did not remove it. The rest was
`Memory::clear`, documented "a new episode is a new world" and clearing only
what was in sight - so every episode after the first IN A PROCESS began
holding the last one's remembered monsters, rooms and pushed-on walls.

That is a bug in its own right and an obvious one once stated: an agent that
remembers a level it has not played. What makes it worth recording beside
the first is that **its symptom is indistinguishable**. A trajectory
replayed after another trajectory starts with the other's memory, so the
same actions are offered different options, and the replay diverges for a
reason nothing in the trajectory can reveal. Verification runs replays
back to back, so verification is exactly where it bites.

Both fixes are the same shape, and the shape is the transferable part:

```rust
// No `..`. Adding a field fails to compile until somebody says which side
// of the line it is on.
let Memory { held, haunts, swept, pressed, took } = self;
```

Measured, E1M1, a 374-cell archive, six trails sampled across their lengths:
four of six failed before, all six replay after, the longest 1466 decisions.
The audit had never once passed.

And the same defect silently corrupts the search itself, not only its
artifacts: returning to a carried-in cell means replaying its trail, so an
ambiguous or memory-polluted replay lands somewhere other than where the
archive says. The search then explores one place and files the result under
another - which is why generations never compounded. Check a rebuilt walk
against the trail's own witnesses and drop the cell when it does not
reproduce.

See also #148 - an action whose meaning depends on unrestored state is not
a recordable action - of which this is the mirror image: there the meaning
was not restored, here it was never written down.
