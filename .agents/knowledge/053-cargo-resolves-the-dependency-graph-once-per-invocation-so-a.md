<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 53. Cargo resolves the dependency graph once per invocation, so a `Cargo.toml` edited during a long build produces an "unresolved import" for a dependency that IS there

A full `make test/nextest` (a release build of every test binary, ~1 h on this
box) was launched, and ~46 minutes into it a new crate dependency was added:
`brain-capability` into `crates/glmdsa/Cargo.toml`, alongside a new
`caps.rs` that does `use capability::{...}`.

The run failed with:

```
error[E0432]: unresolved import `capability`
  --> crates/glmdsa/src/caps.rs:45:5
   = help: if you wanted to use a crate named `capability`,
           use `cargo add capability` to add it to your `Cargo.toml`
```

The dependency was in `Cargo.toml`. It was in `Cargo.lock`. Both were
committed. `make build`, `cargo test -p brain-glmdsa`, and a targeted
`cargo test -p brain-glmdsa --release --offline` all succeeded, before and
after.

**Cargo resolves the dependency graph once, at the start of an invocation, and
then compiles from the working tree as it stands when each crate's turn
arrives.** A build long enough to overlap an edit therefore compiles NEW source
against an OLD graph. Source edits alone are harmless-ish (you get a stale
result); a manifest edit is what produces this, because the graph is the one
thing the invocation will not re-read.

What makes it expensive is that the error message is actively misleading: it
names a dependency that is present and correct, and its `help` line tells you
to add something you already added. The instinct is to go looking for a
feature-gate, a `[lib] name` mismatch, or a workspace-inheritance problem -
none of which are wrong in any way you can find, because nothing is wrong.

Two rules:

1. **Do not edit any `Cargo.toml` while a long build or test run is in
   flight.** Source-only edits are recoverable by re-running; a manifest edit
   invalidates the run AND misdiagnoses itself.
2. **Before believing a dependency-resolution error from a long-running build,
   reproduce it in a fresh, short invocation** (`cargo test -p <crate>
   --release --offline --no-run` is enough). If the fresh run is green, the
   failure was the overlap, not the code - re-run the long build rather than
   "fixing" a manifest that was already right.

The same shape applies to any tool that snapshots configuration at startup and
reads inputs lazily afterwards; cargo is just the one that costs an hour.
