<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 85. Pathspec-scoped commits are not immune: a manually-reconstructed shared file can still revert a concurrent session's already-committed change if it was rebuilt from a stale `HEAD` snapshot

M5.2 hit the failure mode #84's own fix does not cover. `crates/kernels/
src/lib.rs` and `crates/gpu-core/src/cost.rs` are mechanically-generated/
generated-adjacent files another M5.1 session was actively appending to
(`rmsnorm_dx_rows`'s registration and cost-model arm) in the same few
minutes. Following #83/#84's procedure correctly - `git show HEAD:<file>`
to get a clean base, hand-insert only this milestone's own hunks, `git
commit -m ... -- <path>` - still landed a commit that silently DELETED the
other session's `rmsnorm_dx_rows` cost-model arm, because the `HEAD`
snapshot the reconstruction started from was taken several tool calls
earlier, and that session's own commit landed in the gap between the
snapshot and this session's `git commit`. The pathspec form protects
against the index holding someone else's STAGED content (#84); it does
nothing against the reconstruction ITSELF being built from a `HEAD` that
has since moved - a bare `git commit -- <path>` commits whatever is
currently on disk at that path, and disk had the stale reconstruction, not
the newer one.

A second, worse failure mode showed up trying to commit this very lesson:
on a box with several concurrent `git commit` invocations, `pre-commit`'s
own "stash unstaged changes to a patch, run hooks, restore the patch"
cycle is not safe to run from more than one process at once - it has no
locking of its own beyond git's index lock, which it releases between its
stash and restore steps. Two overlapping invocations can each compute
their "unstaged diff" against a working tree the OTHER one has already
partially cleared, and the loser's edit silently vanishes with no error
from `git commit` itself (it simply reports "no changes added to commit"
after the hook exits 0) - this happened to this lesson's own first two
commit attempts, each retried from scratch after finding the addition
gone from disk.

**Fix applied here**: a same-turn follow-up commit re-added exactly the
deleted `rmsnorm_dx_rows` cost-model hunk (diffed the losing commit
against its parent to get it back verbatim); this lesson's own text was
re-typed and re-committed each time it disappeared until one attempt
landed with no concurrent commit racing it.

**Rule going forward**: on a shared, fast-moving box, `git show HEAD:
<file>` is only safe to build a reconstruction from at the INSTANT it is
taken - re-run it (and re-diff) immediately before the `git add`/`git
commit` that consumes it, not once at the start of a multi-step
reconstruction. Prefer small structural insertions (append a `match` arm,
a registry row) over reconstructing a whole file's content by hand, so
there is less to go stale. And after ANY commit attempt on a shared
working tree - success, failure, or a hook that ran - re-check the actual
file content on disk before trusting the edit survived; a `git commit`
that reports success only promises the COMMIT it made is what it says,
never that your working-tree edit was still there to be picked up.
