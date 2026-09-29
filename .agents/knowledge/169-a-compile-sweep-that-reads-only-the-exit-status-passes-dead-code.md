<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 169. A compile sweep that reads only the exit status passes dead code

`check/sdk-features` builds the `brain` SDK once per surface, alone, so a
feature nobody builds on its own cannot silently stop compiling. It judged
each build by `cargo check`'s exit status - and `cargo check` exits 0 on a
warning. When it was first made to read the warnings, 7 of 15 builds
failed: a `study`-only build carried `device::resolve` as dead code (the
feature selected the `device` tier without using it), `forecast` carried
the single-spec resolver it never calls, `text`/`video`/`vision`/
`multimodal` carried the two-spec one, and an unused import sat in every
build including the default. Extending the sweep to the test targets
then found six integration tests with no feature gate at all, so `cargo
test -p brain --no-default-features --features <any surface>` did not
compile, and a unit-test module whose every test needs `decision` was dead
under every other surface. An application that selects one surface -
`study` alone, for training - saw warnings the workspace build never
showed, because in the full build some other surface used each item.

**Rule:** a per-feature sweep is a warning gate, not only a compile gate:
fail on any warning located in the crate under test, and build its test
targets too (`--lib --tests`). An item only some
surfaces call carries a `cfg` naming exactly those surfaces, and a feature
selects a tier only if its code uses it.
