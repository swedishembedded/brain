<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 146. A quality-diversity niche must hold CAPABILITY, not just the score's own terms

A quality-diversity archive keeps one elite per behavioural niche, so the
niche is the definition of "a different kind of solution". The mistake that
costs the most is deriving that definition from the SCORE - keeping the
axes the objective happens to read and dropping the rest.

Measured on `samples/decision/doom`, searching E1M1 at Ultra-Violence for
UV-Max (every monster, every secret, then the exit). The category does not
count items, so items were dropped from the niche. That is correct for the
score and wrong for the niche, and the difference was 2.6x:

| niche axes | best score, of 2.0 | cells | x-coverage | monsters killed |
|---|---|---|---|---|
| map, x, y, keys, monsters-left, secrets | 0.193 | 73 | 12 | ~7 of 29 |
| ...plus weapons carried, health band | **0.497** | 284 | **30** | **~18 of 29** |

Identical seed, identical 180 s budget, identical operators.

## Why dropping it was so expensive

A weapon is a CAPABILITY, exactly like a key. Without a weapons axis,
"standing in the courtyard holding a shotgun" and "standing in the courtyard
holding a pistol" are one cell. The archive keeps whichever reached it at
lower cost, and lower cost is systematically the run that sprinted PAST the
shotgun - so the archive actively deletes the armed state, then repeatedly
sets off from the unarmed one and loses the fight it was sent to win.

Health is the same argument in a different direction. The bar does not score
health and should not - a run is not better for ending healthy. But a search
that cannot tell a 100-health state from a 9-health one at the same spot
keeps the faster, which is again the one that skipped the fights, and spends
its budget on trajectories that were dead on arrival.

## The rule

> The score answers "how good is this run?". The niche answers "is this a
> different kind of situation?". They are different questions and the second
> one is not derivable from the first.

An axis belongs in the niche when holding it changes what the agent CAN DO
next - a key, a weapon, a vehicle, an unlocked tool, a credential, a budget
remaining - whether or not the objective reads it. An axis belongs in the
score only when it is part of the goal.

The diagnostic that found this is worth keeping too: per-axis coverage
(`search::Archive::coverage`), printed every progress line. A cell count
cannot distinguish a search opening new regions from one stuck in a room
producing a new square of floor every few steps. The coverage line said
`1/12/10/1/2/1` - twelve columns and ten rows of floor, but only TWO distinct
kill counts and not one key or secret - which named the defect directly:
the search was exploring geography and barely touching achievement.
