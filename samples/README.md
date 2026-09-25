<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# samples - standalone applications and client scripts built on brain

A **sample** is a complete, standalone demonstration of one brain capability,
built and run on its own - never part of the engine build. This is the Zephyr
`samples/` idea: the engine is one thing, and the applications that demonstrate
what you can build with it are another. A sample is not a test and not a
fixture.

There are three kinds, one per way of reaching brain:

| | `samples/<category>/<name>/` | `samples/python/<pipeline>/<name>/` | `samples/shell/<pipeline>/<name>/` |
|---|---|---|---|
| what it is | a **Rust application** | a **Python** client script | a **shell** client/CLI script |
| how it reaches brain | by linking the `brain` SDK crate | over D-Bus / HTTP (`brain_py`) | over D-Bus (via `tools/dbus-session.sh`) or the `brain` CLI directly |
| what it proves | the SDK is a real, usable library | the served surface works off-process | the CLI/served surface works off-process, no Python needed |
| built by | `make samples/<path>/build` | nothing - run against a running brain | nothing - run against a running brain |

A capability is only fully demonstrated when the ways that apply to it work:
every served model gets at least a Python or shell client; a capability the
SDK exposes as a library gets a Rust sample too when linking it standalone is
a meaningfully different story from driving it over the bus.

## Every sample is a directory

Regardless of kind, a sample is `samples/.../<name>/`, always containing:

- **An entry point** - `src/main.rs` for a Rust sample, or one or more
  `.py`/`.sh` scripts for a Python/shell sample. A sample that legitimately
  needs several entry points around one workflow (train step + generate step,
  or several actions on one served model) keeps them together in one
  directory with one README, rather than splitting one story across several
  samples.
- **`README.md`** - what it demonstrates, what it needs, how to run it, and
  what it needs to build (for a Rust sample). This compiles into the docs
  manual (`make docs`) - see `scripts/build/gen-samples-manifest.py`.
- Optionally, **`docs/`** - images the README displays. An image a README
  points at must be committed: the author is the one person who cannot see a
  broken one, because the untracked file is sitting in their working tree. The
  converse holds too - an image nothing points at is weight every clone pays
  for, which is what a generated chart written into a tracked directory becomes,
  so point plotting scripts at an ignored output directory instead.
  `make check/samples` measures both directions.
- Optionally, **`fetch-data.sh`** - a script that pulls or generates the
  sample's own input data on demand. No sample commits a fixture; if it needs
  a CSV, an image, or a checkpoint conversion, it fetches or generates it
  itself, usually by wrapping a `tools/` utility.
- A shell/Python sample may keep its own private helper scripts beside its
  entry point (e.g. a Python script a shell sample shells out to) - those are
  implementation details of that one sample, not independent samples in their
  own right, and are not expected to run standalone.

Python samples reach brain via `brain_py` (`pip install -e brain-py`) and
`tools/dbus-session.sh` for the launch-and-wait boilerplate (a private session
bus, `brain serve` with a real readiness wait, teardown on exit) - see that
script for its usage. Shell samples either do the same, or shell out to the
`brain` CLI directly when no bus is needed.

## Rules for Rust samples

These are not style preferences. Each exists to keep a sample's dependency
closure honest or the engine build unaffected. (Shell/Python samples have no
Cargo closure to police; their contract is the directory shape above, plus
`make check/samples`' structural checks - README present, entry script
executable and SPDX-headed, no stray committed fixtures.)

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
| `image` surface | 42 |
| *(before surfaces existed: everything, always)* | *69* |

Cargo compiles one brain subtree per distinct feature combination, so samples
that name the same surfaces share artifacts and samples that genuinely need
different capabilities genuinely compile different code. The incremental loop
is unaffected in every configuration: **editing a sample and rebuilding
compiles exactly 1 crate, 0 of them brain crates**, and that is what
`make check/samples` asserts by touching a sample's sources and measuring.

If you ever see a sample build recompiling `brain-*` crates, it is being built
with a selection other than its own.

## Adding a Rust sample

    samples/<category>/<name>/
        Cargo.toml       package `sample-<category>-<name>`, the surfaces it
                         uses, and a max-brain-crates budget
        README.md        what it demonstrates, what it needs, how to run it
        src/main.rs      SPDX header, Copyright (c) 2026 Martin Schröder

Nothing else to register: the workspace picks it up through the `samples/*/*`
glob, and `make samples/<category>/<name>/{build,run}` works immediately.

## Adding a Python or shell sample

    samples/python/<pipeline>/<name>/       or samples/shell/<pipeline>/<name>/
        README.md        what it demonstrates, what it needs, how to run it
        <name>.py (.sh)  SPDX header, Copyright (c) 2026 Martin Schröder,
                         executable (chmod +x)
        fetch-data.sh    optional: pulls/generates this sample's own input

`<pipeline>` is the existing taxonomy (`api asr dbus embedding forecast
imagegen imaging llm musicgen qwen3omnimoe restore tts videogen vision`) -
reuse an existing one or add a new one only when the sample genuinely opens a
new pipeline. These trees are **excluded from the Cargo workspace**
(root `Cargo.toml`'s `exclude`) - they hold no `Cargo.toml` and are never
built. `make check/samples` still validates their directory shape; `make
samples/list` lists them alongside the Rust samples.
