<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 80. A fetch trigger with two entry points inherits NONE of the other one's gates - every condition must be re-stated at each of them

Adding `--model` to `brain flux2 generate` surfaced a hang that had nothing
to do with the new flag: `brain flux2 generate --help` blocked on the
network. The default-weights fetch has TWO entry points in
`crates/cli/src/resolve.rs` - `dispatch_arch` calls `ensure_env_weights`
(fill the `BRAIN_*` variables) and then `maybe_inject_default_weights`
(fetch and append `--weights`). The help gate existed only at the FIRST:
the comment above `ensure_env_weights` even says "help text must never
block on a network fetch", while the second entry point walked straight
into `ensure_default_weights`, which consults neither `BRAIN_AUTO_FETCH`
nor anything else about the invocation. `canon_verb("generate")` is
"infer", so a usage request was inference-shaped enough to fetch a 4B
checkpoint. The same gap ate `--model` (the invocation names its weights;
the injection would only have added a flag the parser rejects) and would
eat the next such condition too.

**Fix**: `maybe_inject_default_weights` now short-circuits on the same
conditions the first entry point gates on - help, `--weights`, `--model` -
and `weights_already_named` treats `--model` as naming the primary role
(`weights_env[0]`), so pointing a command at its own DiT does not
re-download the default ref's DiT it is about to override.

**Rule going forward**: when a side effect (fetch, write, spawn) is
reachable through more than one dispatch path, the list of conditions that
suppress it is part of the side effect's contract, not of one call site's
comment. Either funnel both entry points through one predicate, or give
each of them a test that names the full condition list - a gate that lives
in only one of two entry points is a hang or a redundant download waiting
for the invocation shape nobody tried yet (today that shape was `--help`).
