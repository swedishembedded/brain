<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 139. A cap bounds only the path that was written to check it

**Lesson:** A resource limit belongs to the resource, not to the first caller that spends it. When a
second spender is added, the limit has to move to a place both must pass through.

## What happened
`splat::opt::densify` takes `max_gaussians`. A fit handed a hard limit of 256 came back with 258.

## Why
It computed the gradient-chosen set as `min(n * densify_frac, cap - n)` and stopped there, so the
count was bounded for the one path that consulted it. The EXPLORATION path, added later, splits a
share of the large gaussians whether or not the gradient asked, and nothing subtracted those from
the same allowance.

There were two ways for that to be wrong, and both were:

- The limit is about the FINAL count, so the thing to track is a running allowance that every
  emitting path spends from, not a set size computed once up front from a rule that only one path
  obeys.
- The function returned early when already at the cap, which also skipped the PRUNE in the same
  pass. A scene that started at its budget kept every transparent gaussian it had, forever, and
  density control on it was a no-op rather than a redistribution.

## What to do
A resource limit belongs to the resource, not to the first caller that spends it. When a second
spender is added, the limit has to move to a place both of them have to pass through. "No room to
grow" is never by itself a reason to skip reclaiming.

## Evidence
Running the prune anyway took that scene's final MSE from 0.009217 to 0.007836.
