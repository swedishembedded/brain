<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 200. Code gated to a target nobody built on is a wish, not a feature

The first native build on Linux aarch64 (NVIDIA Grace) failed four ways in code
that no x86 host had ever compiled, and one more in code that compiled but could
not be reached:

- `capture` declared libc's variadic `open(path, flags, ...)` without the
  ellipsis. Current rustc denies redefining a symbol std itself links, and on
  aarch64 variadic arguments are passed differently from fixed ones.
- `vulkan` typed extension-name pointers `*const i8`; `c_char` is unsigned on
  aarch64, so ash's `*const c_char` did not match.
- `backend-cpu` had a NEON `SDOT` int8 kernel behind `#[cfg(target_arch =
  "aarch64")]` whose own docs said it had never been compiled. It had never been
  selected either: the tier enum had no NEON variant, and once it was compiled
  for the first time the dispatch `match` was not exhaustive.
- x86-only helpers (a tier enum, a constant, two tests) became dead code or
  unresolved names on aarch64.

The NEON kernel, run for the first time, passed its sign-corner test against
the scalar oracle; it was correct, and unused. The lesson is about the gating:
a `cfg`-gated implementation with no build that exercises it accumulates
defects at a steady rate that no review sees, and "documented as unvalidated"
is not a state a codebase can stay in. Build, lint and test every supported
target in CI-equivalent form, and wire a target's fast path through the same
dispatch as the others so a missing arm is a compile error.
