<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 110. `include_str!` proves registry -> file, never file -> registry

A source-only kernel registry (`kernels`, `kernels-cuda`) embeds each kernel
with `include_str!`, which makes a registry entry naming a missing file a
compile error. The opposite direction has no checker at all: a `.cu`/`.wgsl`
file that no entry embeds compiles, tests and ships as a file nothing
dispatches, nothing validates and nothing can delete safely, while looking
exactly like working functionality to the next reader.

Nothing in Rust can see it - the crate's own tests only ever walk the table.
The check has to be a script that walks the DIRECTORY and intersects it with
the table, which is one more reason a kernel catalogue gate reads the source
tree rather than trusting the registry it renders.
