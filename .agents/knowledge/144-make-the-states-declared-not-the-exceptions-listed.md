<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 144. Make the states declared, not the exceptions listed

**Lesson:** When you find a drift between two tables, ask what the full set of valid states is
before writing the assertion. "Every item is in a declared state" is a property of the design, and
only it keeps holding when someone adds a third table.

## What happened
Lesson #136 fixed a two-list drift by asserting the lists agree. That was the right fix for the bug
and the wrong shape for the problem: it says "these two tables must match", which catches a drift
between exactly those two tables and nothing else. Three more defects were sitting one table over.

What the narrow test missed: `crates/catalog/src/lib.rs::resolver_spec_for` is a THIRD registry,
read by every served surface (D-Bus, HTTP, `build_executor`) but not by the CLI, which resolves its
own `Assembly` and passes it in. A model whose `ModelEntry.provider` reads a role while its id is
absent from that registry works perfectly from the command line and fails when served, with
"assembly 'local/…' has no <role> role". `brain/flux2-klein` and `brain/wan` were both in that state
already.

Three more were found by the reshaped form:

- `brain/splat` was classified as a model with missing weights when it has none at all (its scene
  is an `--in scene=` PLY blob, not a checkpoint).
- `brain/ltxv` declares no weights param on its served manifest at all, so an off-machine caller
  cannot name a DiT.
- The converse gate, "an architecture with a usable `ArchSpec` that nobody switched on", caught
  `llava`, `cosyvoice` and `minimaxmusic3`.

## Why
The pairwise assertion describes an implementation (two lists agree), not the design (what states a
model may legitimately be in). A third table is invisible to it.

The converse gate needed the same discipline. Its first form would have forced `llava` to be
declared migrated. Since its provider ignores the `Assembly` and its action already declares
`.host_env`, that would have suppressed auto-fetch while changing nothing else: exactly the
half-wired state #136 is about. A gate that creates the defect it was written to prevent is worse
than no gate.

## What to do
Instead of pairwise agreement, assert that every model is in exactly ONE of the states that
actually exist, and detect the state from DATA rather than from a hand-maintained list:

1. resolver-backed: provider reads a role, `resolver_spec_for` supplies one
2. weights-as-param: some action declares a `host_env` param
3. env-only: listed BY NAME, with the reason it cannot classify

A new model that is in none of them fails the gate and has to pick. The exception list still
exists, but it is now the small explicit thing rather than the silent default.

The converse gate now skips an architecture already in state 2, by READING the manifest for a
`host_env` param rather than by listing llava.
