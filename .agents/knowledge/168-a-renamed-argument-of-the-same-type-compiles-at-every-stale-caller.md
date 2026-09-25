<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 168. A renamed argument of the same type compiles at every stale caller

`recon::photogrammetry::training_set(photos, width, ...)` became
`training_set(photos, halvings, ...)` when targets stopped being resampled
to a width. Both are `u32`, so every caller still compiled - and the SDK's
`Reconstruction::run` went on passing 1024, which now meant 1024 exact 2x
halvings of each photograph. Nothing ran that path between the change and
its discovery by reading the code; the unit tests used the new callers.

**Rule:** when a parameter's MEANING changes, change its type or its name
at the call boundary (a `Halvings(u32)` newtype, a config struct field, a
renamed function), so the compiler lists every caller that still means the
old thing. Where that is not done, grep every caller before committing and
run at least one end-to-end path per surface (SDK, CLI, sample).
