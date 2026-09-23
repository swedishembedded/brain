<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 152. Making a signal accurate can break what relied on it being TOTAL

A measurement said a ledger was wrong, the fix was correct, and it made the
system markedly worse. The reason generalises past the case.

In `samples/decision/doom` a search records which walls it has pushed on.
It decided a push had touched something from a clearance reading - a
raycast from the player's centre along their facing - which is a guess:
`P_UseLines` traces its own line and can miss where the probe hit. The
engine knows the real answer and was reporting it, so the ledger was changed
to use it. Of 56 pushes in one campaign, 21 had reached nothing at all and
every one had been filed as a wall tested.

The change was right and the result was a clear loss on a paired run:

| E1M1, 420 s, one seed | before | after |
|---|---|---|
| best score, of 2.0 | 1.011 | 0.552 |
| secrets found | 1 | **0** |
| verified solutions | 5 | **0** |
| walls tested | 38 | 7 |

## Why

The ledger answered two questions with one field:

1. *Have I tried here?* - which stops the sweep pushing the same spot again.
2. *Have I searched this room?* - which retires a room from the frontier.

The old guess was WRONG but it was TOTAL: every push recorded something, so
question 1 was always answered. The accurate version records only pushes
that met a line, so a push that reached nothing now answers question 1 with
"no" - and the sweep pushes there again, forever. Forty percent of the
budget went into that loop.

The fix is two fields, and the questions make the names: `pressed` records
spots pushed AT, `walls` counts walls pushed ON. Tried either way; tested
only on contact.

## The transferable part

**When a signal becomes more accurate, it usually also becomes more
partial - and a consumer that relied on totality will break silently.**
Accuracy and coverage are different properties and improving one can cost
the other. Before replacing an approximation, list what reads it and ask of
each: does this need the right answer, or does it need an answer?

Two smaller things, both of which cost real time here:

- **A mechanism argument is not a measurement.** "21 of 56 pushes were
  miscounted" was true, verifiable, and insufficient. It described the
  defect and said nothing about the consequence of repairing it.
- **The unit tests all passed**, and they could not have caught this: each
  asserted the ledger's answer to one question, and the bug was in the
  relationship between two. Only an end-to-end paired run found it - and
  only because the report tracked SECRETS FOUND and VERIFIED RUNS rather
  than the search's own score, which is a measure of how much searching
  happened rather than of whether anything was found.
