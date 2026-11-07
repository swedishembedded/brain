<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# book

Scope: The textbook under `docs/book` (*From First Principles to Intelligent Machines*): its specification, tooling, chapters and their verification.

Rules: `.agents/rules/book.md`. Specification: `docs/book/book.yaml`
(16 volumes, 165 chapters: 150 teaching chapters and 15 projects). A chapter is **written** when its file exists and
passes the book gates, and **verified** when an independent reader has
reproduced its worked examples from the text alone in another language.

## Done

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
- [ ] Volume II, What Does It Mean to Learn?: chapters 1 to 4 written.
- [x] Volume III: chapter 4 (backpropagation, the pilot) written.

## Open

- [ ] Every chapter in `docs/book/book.yaml`, in reading order, each with its
      volume introduction, worked examples, figures and exercises.
- [ ] Independent-reader verification of each chapter's [Implement]
      exercises.
- [ ] A glossary generated from the concept graph.
