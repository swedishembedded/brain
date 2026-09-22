<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 88. Two sessions fixing the SAME missing struct field merge cleanly into a duplicate-field compile error - and a compile error makes the clippy gate hide every warning scheduled behind it

`crates/qwen35/tests/vl.rs` and `crates/qwen35moe/tests/vl.rs` each carried
`tokens_per_second: 2,` TWICE inside one `VisionConfig` literal (E0062,
"field specified more than once"). Both test binaries failed to compile, on
`HEAD`, in the committed tree.

The provenance is the interesting part. Two commits added that field
independently, for the same reason - `61ed68b16` ("fix tokens_per_second in
three more test fixtures") and `fa2425f94` ("fix pre-existing compile/clippy
breakage"), each written by a session that found the fixture missing a field
`VisionConfig` had gained. They inserted it at DIFFERENT lines of the same
struct literal, and two insertions at different offsets in one file are a
textually clean merge: neither git's merge nor a rebase had anything to
report. The defect only exists at type-check time, and only after both
halves are present. Duplicate struct fields, duplicate match arms and
duplicate imports are all this same shape - the "add the one missing thing"
fix is precisely the fix two parallel sessions produce independently.

The second half is worse than the first. `scripts/gates/clippy-gate.sh` did
its job and said so loudly - "clippy exited 101, so the lint pass ABORTED,
and any warning count from this run is meaningless" - but the abort landed
early in the crate graph, so twenty-one OTHER pre-existing warnings, in six
crates the error had nothing to do with (`gguf`, `glmdsa`, `qwen3`,
`qwen3tts`, `qwen35`, plus a test target), were never linted at all. Against
a baseline of 0, those warnings had been invisible for exactly as long as the
compile error had been present. Nothing else covered the gap either: `make
test`'s first line is `cargo test ... --no-run`, which fails on the same
error, so the fast lane had not been run on this tree since the duplicate
landed.

**Rules going forward**:
- After ANY merge, rebase or fast-forward on a shared box, type-check before
  believing the tree is sound. `make clippy` is sufficient on its own for
  this - it is a full `--workspace --all-targets` compile - and it is also
  the gate that will refuse to tell you anything useful if the tree does not
  build.
- When the clippy gate reports a non-zero EXIT rather than a warning count,
  treat that run's count as UNKNOWN, never as zero, and re-run the gate after
  fixing the compile error. What sits behind an abort is not bounded by what
  sat in front of it; here it was 21 warnings in crates the error never
  touched.
