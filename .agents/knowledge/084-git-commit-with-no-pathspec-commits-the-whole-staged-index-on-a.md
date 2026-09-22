<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 84. `git commit` with no pathspec commits the WHOLE staged index on a shared box, not just what you `git add`ed this turn

M5.3 isolated two files (`crates/vae/src/blocks.rs`, `crates/vae/src/
blocks/grad.rs`) out of a live tree several other sessions were
concurrently editing, per #83's own procedure - `git add` those two files,
then a plain `git commit -m ...` with no pathspec on the commit itself.
The resulting commit's own `git show --stat` named a THIRD file nobody
intended: another session's already-staged, unrelated, in-progress work
(`crates/kernels/src/lib.rs`'s Q4 kernel registrations) had ridden along.
`git add` only ADDS to the index; it never narrows what a later bare `git
commit` includes, and on a box where several agents share one working
tree, the index at any given instant can already hold staged hunks from a
session that has never itself run `git commit` yet. The fix attempt made
the SAME mistake a second time - the follow-up revert commit, run the same
way, swept in a second unrelated file (`crates/gpu-core/src/cost.rs`'s
staged cost-model additions) - because the index had gained more staged
content in the few seconds between the two commits. Caught both times from
the commit's own `git show --stat` output, not from a later gate failing.

**Fix applied here**: two further commits, each using the PATHSPEC form of
`git commit` (`git commit -m ... -- <path>`, which stages+commits only
that path's current working-tree content and touches nothing else in the
index) to revert exactly one swept-in file back to its pre-accident
content, then restoring that session's actual current progress on disk
afterward (uncommitted, exactly where it was). Verified by diffing the
whole multi-commit range against its starting point and confirming only
the four files this milestone actually touched had moved net of the
reverts.

**Rule going forward**: on a shared working tree, a bare `git commit -m
"..."` with no pathspec is never safe, no matter how carefully `git add`
was scoped beforehand - the index can hold another session's staged file
at commit time regardless of what you personally staged. Always commit
with an explicit pathspec (`git commit -m "..." -- <exact files>`), and
check `git diff --cached --stat` immediately before every commit as a
second, independent confirmation - not just `git status` right after `git
add`, since the index can change in the gap between those two commands.
