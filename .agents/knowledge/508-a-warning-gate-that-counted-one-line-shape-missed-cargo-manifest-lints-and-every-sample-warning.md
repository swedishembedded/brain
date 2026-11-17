<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 508. A warning gate that counted one line shape missed cargo's manifest lints and every sample warning

**Lesson:** a zero-warnings gate must count every shape of warning the toolchain prints, and every target it lints
must have its warnings counted, not only its exit code. Check the gate against a log that holds each shape.

## What happened

`scripts/gates/clippy-gate.sh` counted lines matching `: warning:`, which is the shape of a compiler or clippy
diagnostic in `--message-format=short` (`path:line:col: warning: ...`). Cargo 1.100's manifest lints print a
different shape, with no location: `warning: unused dependency ...`, `warning: unused workspace dependency ...` and
`warning: unused field ... in workspace.package`. The gate reported 0 warnings (baseline 0) while
`cargo check --workspace --all-targets` printed 22 of them:
- 14 unused dependencies in 13 crates;
- 4 unused workspace dependencies;
- 3 workspace metadata fields no member inherited, although the comment above them said every member did;
- a dropped `ExitCode` in a sample's test.

The samples are linted separately by `scripts/gates/check-samples.sh`. It ran `cargo clippy --all-targets` per
sample, discarded the output and failed only on a non-zero exit, so warnings could never fail it. Five samples
carried 16 clippy warnings.

## Fix

- `scripts/gates/cargo-warnings.sh` prints every warning in a cargo log, in both shapes. It leaves out a build
  script's informational line (`warning: <pkg>@<version>: ...`) and the `generated N warnings` summaries.
- Both gates count through it. The samples gate fails on any sample warning, test targets included.
- The unused dependencies were removed or moved:
  - A dependency used only by tests or examples went to `[dev-dependencies]`.
  - One used only by the x86_64 OpenVINO backend became a target-specific dependency.
- Members now inherit `repository`, `authors` and `homepage`, as the workspace manifest says they do.
- The sample warnings were fixed at the code.

A dependency the lint calls unused can still be used under a non-default feature or a `cfg`. Check each one in
`src/`, `tests/`, `examples/` and `build.rs` before removing it. In this round every hit inside `src/` was a doc
comment or a local module of the same name.
