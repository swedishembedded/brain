<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 83. `git status` shows FILES in flight, not HUNKS - a shared working tree can hand you another session's uncommitted work interleaved inside a file you also need to edit

M4.1 needed to check whether a CPU-backend `serve::` test failure was a
regression from its own change. The fast way to check "does this fail on
plain HEAD" is `git stash push -- <file>`; doing that here reverted
`crates/qwen3/src/serve.rs` to a state missing a whole test
(`decode_step_submits_are_not_one_per_metadata_write`) that had been
PASSING moments earlier in this same session's own full-suite run - not
because the stash was buggy, but because another session's uncommitted,
unrelated milestone (a decode/prefill submit-batching change, its own
`M3.1`/`M3.3` labels visible in its doc comments) was ALREADY sitting,
uncommitted, in the exact same file before this milestone's own edits
began. `git status` only says the FILE is modified; it says nothing about
how many independent, uncommitted changes are textually interleaved
inside it on a box where sessions share one working tree instead of one
worktree per agent. `git stash push -- <file>` (or any revert of the whole
file) discards ALL of them together - popping it back only works if
nothing else has touched that file (including a `git commit`) in between,
which is exactly the operation this situation needed.

**Fix applied here**: diffed the file against `HEAD` (`git diff -U2`),
classified every `@@` hunk by content as "mine" or "not mine" (the doc
comments name their own milestone, which is the tell), extracted only
"mine" into a standalone patch, applied it to a scratch copy of `git show
HEAD:<file>` (never the live working tree, which already had both sets of
edits applied and so rejects a patch built to be applied to `HEAD`),
verified byte-for-byte that `scratch_head_plus_mine` diffed against the
live file reproduced exactly the "not mine" hunks with nothing missing and
nothing of mine leaked in, built and tested that scratch reconstruction
standalone, then swapped it into the real path, staged, committed, and
restored the original live (both-sets) content on disk afterward so the
other session's WIP was never at risk of being lost or half-committed.

**Rule going forward**: on a box where `git status` can show a file
modified by more than one concurrent agent, never `git stash`/`checkout`/
`reset` a shared file to answer a "does this reproduce on HEAD" question -
that command's job is reverting YOUR OWN changes, and it cannot distinguish
yours from anyone else's inside one file. Diff against `HEAD`, read every
hunk, and attribute it before touching the working tree; when in doubt
that a hunk is yours, treat the file as co-owned and separate hunks by
content (context lines, doc comments, milestone labels) rather than by
line-number ranges, which shift under concurrent edits.
