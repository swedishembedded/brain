<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# samples - standalone applications built on the brain SDK

A **sample** is a complete, standalone application that links the public
`brain` SDK the way a downstream product would, declares exactly which parts of
brain it needs, and is built and run on its own:

```bash
make samples/imagegen/generate/build
make samples/imagegen/generate/run ARGS="--prompt 'a whale submarine'"
make samples/list
```

This is the Zephyr `samples/` idea: the engine is one thing, and the
applications that demonstrate what you can build with it are another. A sample
is not a test, not a fixture, and not part of the engine build.

## samples/ vs examples/

| | `examples/` | `samples/` |
|---|---|---|
| what it is | a **client** script (Python/shell) | a **Rust application** |
| how it reaches brain | over D-Bus / HTTP / the `brain` CLI | by linking the `brain` SDK crate |
| what it proves | the served surface works off-process | the SDK is a real, usable library |
| built by | nothing - run against a running brain | `make samples/<path>/build` |

Both stay. They demonstrate the two genuinely different ways to consume brain,
and a capability is only fully demonstrated when both work.

## Rules

These are not style preferences. Each exists to keep a sample's dependency
closure honest or the engine build unaffected.

1. **A sample depends on `brain` (the SDK crate), not on `brain-<short>` engine
   crates.** The SDK is where the *feature vocabulary* lives - it is this
   workspace's Kconfig symbol table. If a sample needs something the SDK does
   not expose, the fix is to widen the SDK; that is what samples are *for*. A
   sample may depend freely on third-party crates.

2. **A sample must NAME the SDK surfaces it uses, and only those.**

   ```toml
   brain = { workspace = true, features = ["image"] }
   ```

   It may name only *surface* features (the `brain_arch::Domain` names) - never
   the `device`/`resolve` infrastructure tiers, never `full`, never a
   dependency's own feature. The workspace baseline sets
   `default-features = false` for `brain`, so a sample that names nothing gets
   an SDK with no public types and **fails to compile** rather than silently
   linking every model crate. `make check/samples` verifies the declaration
   against the real `cargo tree`.

3. **Samples are workspace members but never `default-members`.** Membership is
   what makes them share `./target`, one lockfile and one registry; exclusion
   from `default-members` is what keeps `make build` and `make test` from ever
   building an application.

4. **The package name is derived from the path**: `samples/<a>/<b>` is the
   package `sample-<a>-<b>`. The Makefile relies on it; the gate enforces it.

5. **A sample must run without weights, or say what it needs and exit cleanly.**
   Printing "set `$BRAIN_...`" or "run `brain pull <model>`" and returning
   non-zero beats a panic or a multi-gigabyte surprise download.

6. **Features select code; they never carry values.** No `steps-64`-style
   feature name. Cargo features are booleans that *union* across a build, so a
   value encoded as a feature multiplies the build cache by the value space and
   still cannot stop another consumer enabling a different value. Values are
   builder arguments first, then a `Config` struct, then `BRAIN_*` env - in
   that order, and a sample should almost never reach past the first.

7. **No mutually-exclusive features, ever.** Features cannot be turned off, so
   a build with two conflicting ones on must still compile and do something
   sensible. Pick-one belongs at runtime, the way `Device` already works.

8. **A sample declares its closure budget:**

   ```toml
   [package.metadata.brain]
   max-brain-crates = 40
   ```

   The gate fails if the real closure exceeds it. A budget that has to be
   raised in the same commit as a new dependency is the point: closure growth
   becomes a decision instead of a drift.

## Why a sample is built with `-p <sample>` alone

A sample **cannot** be built alongside the engine's package selection. Cargo's
v2 resolver unifies features across everything an invocation *selects*, and a
selected workspace member is built with its own default features. `crates/sdk`
is a default member, so a selection containing both it and a sample unions the
sample's `features = ["image"]` with the SDK's own `default = ["full"]` - and
the sample's declaration becomes a silent no-op that links every model crate
anyway. There is no per-package feature flag, and features cannot be
subtracted. Narrow selection is the only mechanism that honours the
declaration, which is what `make samples/<path>/build` does.

`make` is still the right entry point, for two smaller reasons: it pins
`CARGO_HOME` (cargo records the absolute source path of every registry crate in
its fingerprints, so building under a different one shares no artifacts with
what `make` produced) and it pins the profile.

## Cost model

Measured with `cargo tree`, so these are reproducible on any machine rather
than a timing on this one:

| what a consumer names | brain crates linked |
|---|---|
| nothing (bare core: `Error` only) | 9 |
| `device` tier | 18 |
| `resolve` tier | 27 |
| `image` surface | 38 |
| *(before surfaces existed: everything, always)* | *69* |

Cargo compiles one brain subtree per distinct feature combination, so samples
that name the same surfaces share artifacts and samples that genuinely need
different capabilities genuinely compile different code. The incremental loop
is unaffected in every configuration: **editing a sample and rebuilding
compiles exactly 1 crate, 0 of them brain crates**, and that is what
`make check/samples` asserts by touching a sample's sources and measuring.

If you ever see a sample build recompiling `brain-*` crates, it is being built
with a selection other than its own.

## Adding a sample

    samples/<category>/<name>/
        Cargo.toml       package `sample-<category>-<name>`, the surfaces it
                         uses, and a max-brain-crates budget
        README.md        what it demonstrates, what it needs, how to run it
        src/main.rs      SPDX header, Copyright (c) 2026 Martin Schröder

Nothing else to register: the workspace picks it up through the `samples/*/*`
glob, and `make samples/<category>/<name>/{build,run}` works immediately.
