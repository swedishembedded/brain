<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# book

Scope: The textbook under `docs/book` (*From First Principles to Intelligent Machines*): its specification, tooling, chapters and their verification.

Rules: `.agents/rules/book.md`. Specification: `docs/book/book.yaml`
(16 volumes, 165 chapters: 150 teaching chapters and 15 projects). A chapter is **written** when its file exists and
passes the book gates, and **verified** when an independent reader has
reproduced its worked examples from the text alone in another language.

## Done

- [x] Glossary generated from every chapter's Key terms
      (`make generate/book/glossary`).

- [x] Tooling: specification and concept-graph gate (`make check/book`),
      generated figures, worked examples, chronology and navigation
      (`make generate/book/*`, `make check/generated/book`), the book PDF
      (`make book/pdf`) and the Book part of the published docs.
- [x] Evidence inventory: `scripts/book/history.py` extracts every commit on
      every ref (6576 commits, 5830 on main at the time of writing) with
      author and commit dates, touched crates and added knowledge entries.
- [x] Front matter: index, preface, how to read, notation, pseudocode
      conventions; chronology appendix generated from history.

## Chapters

- [x] Volume I, Thinking in Mathematics: all 17 chapters and the project
      written. Verified by two independent C re-implementations from the
      text alone (`scratchpad/reader/v01`, `scratchpad/reader/v01-recheck`):
      the first found 2 failing and 14 guessed exercises plus missing
      inputs, seeds and pseudocode operations; after fixes the recheck
      passed 227 items and its remaining 9 failures, 19 guesses and 13 text
      defects were fixed in the following commit. A third pass over the
      final text has not been run.
- [x] Volume II, What Does It Mean to Learn?: all 8 chapters and the
      project written. Verified by two independent C re-implementations
      (`scratchpad/reader/v02`, `scratchpad/reader/v02-recheck`): the first
      passed 169 of 181 items (11 after a guess, 1 failure); after fixes the
      recheck passed all 297 items. Nine wording nits it noted (none
      blocking reproduction) are open below.
- [x] Volume III, Building Neural Networks from Nothing: all 13 chapters
      and the project written. Verified by two independent C
      re-implementations (`scratchpad/reader/v03`,
      `scratchpad/reader/v03-recheck`): the recheck passed 224 of 246 items
      (8 after a guess, 14 failures, mostly chaotic cells printed with more
      digits than survive a 1e-12 nudge). All were fixed in the following
      commit with a shared five-run stable-digits rule and exact captions;
      a third pass has not been run.
- [x] Volume IV, Computation and Hardware: all 13 chapters and the project
      written. Verified by an independent C re-implementation from the text
      alone (`scratchpad/reader/v04`): 1124 of 1170 items passed, 32 after a
      guess and 14 failed (unstated literals, seeds and orders, stale prose
      numbers, and one simulator bug in the ragged tiled-matmul count). All
      were fixed in the following commits; a recheck has not been run.
- [x] Volume V, Language, Sequences and Transformers: all 14 chapters and
      the project written. Verified by an independent C re-implementation
      from the text alone (`scratchpad/reader/v05`): 1956 of 2297 items
      passed, 221 after a guess, 113 failed (29 of them Unicode database
      cells, now marked as such; the rest unstated initialisation recipes,
      draw and summation orders, chaotic late training cells, a baseline
      bug in a worked module and stale prose). All were fixed in the
      following commits; the project reproduced completely. A recheck has
      not been run.
- [x] Volume VI, Advanced Language Model Architectures: all 10 chapters and
      the project written. Verified by an independent C re-implementation
      from the text alone (`scratchpad/reader/v06`): 1024 of 1235
      items passed, 188 after a guess, 23 failed (unstated metadata, stage
      counts and schedules, chaotic training cells claimed stable, totals
      without shapes, and an offload ratio the prose got wrong). All were
      fixed in the following commits; a recheck has not been run.
- [x] Volume VII, Vision and Understanding Images: all 12 chapters written.
      Independent readers reproduced 1166 items in C from the page alone:
      1007 passed, 131 after a guess, 28 failed (toy heads, initialisations
      and sub-blocks given only as shapes, constants printed nowhere, and a
      pseudocode routine behind its fixed prose). All were fixed in the
      following commits; a recheck has not been run.
- [x] Volume VIII, Generating Images: all 13 chapters written.
      Independent readers reproduced 1259 items in C from the page alone:
      987 passed, 254 after a guess, 18 failed (draw and shuffle orders
      stated loosely, seeds named but not printed, toy parts given by role
      rather than formula, architecture counts without their conventions).
      All were fixed in the following commits; a recheck has not been run.
- [x] Volume IX, Audio, Speech and Music: all 9 chapters written.
      Independent readers reproduced 948 items in C from the page alone:
      727 passed, 177 after a guess, 44 failed (architecture counts
      without their layer layouts, a pseudocode shuffle that disagreed
      with the module, conventions stated only in words, and three real
      table slips). All were fixed in the following commits; a recheck
      has not been run.
- [x] Volume X, Video, Space and World Models: all 9 chapters written.
      Independent readers reproduced 834 items in C from the page alone:
      613 passed, 95 after a guess, 114 failed and 12 were blocked (inputs
      that lived only in the worked modules, unprinted layer inventories,
      a pseudocode order that disagreed with its module, three real slips).
      All were fixed in the following commits; a recheck has not been run.
- [ ] Volume XI, Forecasting, Decisions and Uncertainty: chapters 1 to 5 written.

## Open

- [ ] Every chapter in `docs/book/book.yaml`, in reading order, each with its
      volume introduction, worked examples, figures and exercises.
- [ ] Independent-reader verification of each chapter's [Implement]
      exercises.
- [ ] Volume II wording nits from `scratchpad/reader/v02-recheck/REPORT.md`
      (a stale symbol in the project chapter, a midpoint rule in chapter 1,
      Key terms for chapter 4's helper terms, and similar).
