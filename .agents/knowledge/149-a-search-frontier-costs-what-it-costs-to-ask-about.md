<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 149. A search frontier costs what it costs to ASK about

Giving a search a new kind of frontier - somewhere it has not been, something
it has not tried - is usually reasoned about as a change to what the search
CAN reach. It is also, always, a change to what every single decision costs,
because a frontier that cannot be routed to is not a destination, and routing
is a question somebody has to answer.

Measured on `samples/decision/doom`, E1M1 at Ultra-Violence, 600 s, one seed.
The new frontier was "rooms this run has stood in and never pushed on the
walls of", which is how a DOOM secret is found by anybody who has not been
told where one is. It was asked about in its own call to the engine's router,
the way the two ledgers before it already were:

| route calls per decision | steps in 600 s | ms/decision | best score, of 2.0 |
|---|---|---|---|
| 2 (before the frontier existed) | 44 113 | ~13 | 0.690 |
| 3 (the frontier, asked separately) | 14 731 | ~41 | 0.497 |
| **1 (all three ledgers, one call)** | **15 343** | ~39 | **0.817** |

The third row is the point, and it is not the one the table looks like it is
making. Batching did not restore the throughput - a decision still costs what
a decision costs, and this campaign's steps are slower because its
trajectories are longer and its archive is deeper. What batching bought was
the round trip back, and what the frontier bought was **0.690 -> 0.817 on the
same budget**: an 18% better result out of a search doing a third of the
steps, because the steps are spent on ground the old search could not aim at.

Three things generalise:

1. **Count the round trips before counting the benefit.** The frontier was
   right and the throughput loss was avoidable; they were separate decisions
   that looked like one. A frontier that triples per-decision latency has to
   be three times better just to break even, and the version that shares a
   call with the frontiers already being asked about is free.
2. **Separate ledgers are not separate QUESTIONS.** A remembered item, a
   monster seen and never gone back for, and an unsearched room have three
   different forget rules and three different lifetimes - which is why they
   are three ledgers - but "how do I get there from here" is one question, and
   the transport does not care whose list the coordinates came from.
3. **An unbounded frontier is an unbounded cost.** The first version routed
   to every unsearched room, and that set grows with every room entered and
   shrinks only by being searched. Only the nearest few can ever be offered as
   an option, so only the nearest few are worth asking about.

Steps per second is the measurement that catches all three, and it is not the
one a search reports by default: cell counts and best scores both went the
wrong way here too, but neither of them says why.
