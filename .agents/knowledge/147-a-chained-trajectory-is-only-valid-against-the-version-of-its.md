<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 147. A chained trajectory is only valid against the VERSION of its parent

A Go-Explore-style archive stores how each cell was reached. Storing a whole
trajectory per cell is hundreds of megabytes of almost entirely duplicated
prefix, so the natural representation is a chain: "resume at cell P, then do
these things", and the full trajectory is rebuilt by walking back to the
start.

The chain has a silent failure mode that nothing about it announces.

An archive **replaces** a cell's contents whenever a better way of reaching it
turns up - that is the whole mechanism by which a quality-diversity archive
improves. But every trajectory hanging off P was recorded after resuming the
P that was there **at the time**, and those steps only continue THAT state.
Improve P and each of its children now reads as `new-prefix-to-P` plus
`steps-that-continued-the-old-P` - a sequence that describes nothing that ever
happened, and looks exactly like one that does.

What it costs depends on what the chain is used for:

- **Verification** catches it, which is what verification is for - but only
  after paying for a whole episode against the real environment to find out.
- **Rehydration** (rebuilding a snapshot by replaying a trail) burns one of
  its bounded attempts on a trail that cannot work.
- **Cloning** is where it is genuinely dangerous: a training set built from
  reconstructed trajectories has no replay gate in front of it, so the
  invalid ones are simply fitted, and the policy is taught a continuation
  that the state it is conditioned on never leads to.

## The fix

Give each cell a **generation** counter, bumped when its contents are
replaced and NOT when an offer is refused (a refused offer changed nothing,
and invalidating every chain on it would be a self-inflicted wound). Each
child records the generation of the parent it was recorded against, and
rebuilding a trajectory refuses when the two disagree.

That turns a silently-wrong reconstruction into a detected stale one, which
is the whole of the difference.

## The general rule

> Any structure that stores "do X relative to Y" while Y is mutable needs a
> version on Y, checked at use.

The same shape appears in delta-encoded caches, patch chains against a moving
base commit, incremental checkpoints, and any plan expressed as a diff from a
state some other process may improve.
