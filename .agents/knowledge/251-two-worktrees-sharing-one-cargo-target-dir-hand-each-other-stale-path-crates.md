<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 251. Two git worktrees that share one cargo target dir hand each other stale path-crate artifacts

Symptom. `cargo test -p brain-model --test X` printed `Compiling brain-model`
and then failed with `unresolved import model::new_module` for a module that
sits in the working tree, or ran a test binary that lacked an edit, while the
other worktree was building the same crate into the same target dir.

Cause. Cargo's `-C metadata` hash of a workspace member is the same in both
worktrees (it uses the package id relative to the workspace root), so both
write `libmodel-<hash>.rlib` and one fingerprint file. The fingerprint is
judged by source mtimes against the artifact's mtime, so a build made from the
other tree after your edit makes your tree look up to date. A build serialised
by the target-dir lock is still wrong, because the staleness is a decision, not
a race.

Fix that costs one recompile. Give the crates you edit a private hash with a
per-package profile override, which changes only those packages and their
dependents (not the 300 upstream crates):

    cargo test --config 'profile.dev.package.brain-model.debug=1' ...

Use a different value (or package list) per worktree. A copy of the target dir
works too but costs the whole tree's disk.
