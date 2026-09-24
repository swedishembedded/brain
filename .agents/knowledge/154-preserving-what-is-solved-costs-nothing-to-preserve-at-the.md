<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 154. Preserving what is solved costs the most where there is nothing to preserve

`samples/decision/rubiks` solves any cube by repeatedly playing a macro that
strictly decreases a monotone measure. The library was built from conjugated
commutators: sequences that move three or four cubies and leave the other
sixteen exactly as they found them. That property is what lets a macro be
AIMED - at three specific pieces, late in a solve, without undoing the rest.

It is also what made the solver cost 213 moves, against 80 to 120 for the
beginner method a child is taught.

The waste was not in the macros. Each is a reasonable price for what it does.
The waste was playing them AT THE START, where the cube is fully scrambled,
nothing is solved, and therefore nothing needs protecting. A conjugated
commutator spends two setup moves, an eight move algorithm and two moves
undoing the setup to place one cubie. A single face turn, early on, often
brings three cubies home for one move - and cannot "break" anything, because
nothing was intact.

Adding every one, two and three turn sequence to the library took a solve
from 213 moves to 156, a 27% cut, with the guarantee untouched: the measure
still decides what is admissible, so cheap options can only give it more ways
to be satisfied. Only 0.1% of the candidates offered at each step had been
shorter than eight moves; the library had no cheap options to offer at all.

THE RULE. **An invariant-preserving method pays for the invariant everywhere,
including where the invariant is vacuous.** When a procedure is built from
tools that protect accumulated work, check what those tools cost in the phase
where there is no accumulated work - which is usually the phase that consumes
most of the budget, because it is where the state is furthest from the goal.

How to see it without knowing the answer: **look at the distribution of
option COSTS, not just the decision quality**. A step whose candidates are
all expensive is one where the method has no cheap way to be right, and no
amount of choosing better among them fixes it. Here the tell was that the
cheapest option available was almost never short - a fact visible from the
candidate list at any single step, and not visible at all from the solve
length or the success rate, both of which looked fine.

The general shape is that human methods STAGE their tools - unconstrained and
cheap while nothing is solved, surgical and expensive at the end - and a
uniform method gives that up by construction. What remains, and is not fixed
by cheap macros, is that a strictly monotone measure forbids the temporary
regressions efficient solving needs: it buys a termination proof and pays for
it in optimality. Related:
[[153-a-value-that-fits-its-own-trajectory-cannot-rank-the]].
