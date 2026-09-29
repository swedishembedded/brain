<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 171. A params list shorter than the uniform reads zero

A dispatch hands its `Params` words as a plain `&[u32]`, and nothing
compared that list's length with the struct the kernel binds as
`var<uniform>`. The wgpu backend zero-pads the uniform buffer to its 16-byte
binding size, so a list that stops early is not an error anywhere: every
missing trailing field reads `0`.

That is how an `eps` field added to a kernel becomes `eps = 0.0` at every
call site nobody updated, and how a benchmark silently measures a different
computation than the model runs. Found this way: `flux2_bench`'s RMSNorm
variant comparison dispatched `rmsnorm_rows` (`{d, rows, eps}`) with
`[d, rows]`, so the cooperative arm normalized at eps 0 against a reference
at 1e-6.

`gpu_core::parse_params_words` now reads the bound uniform struct's size
from each kernel's own source at registration (flat scalar blocks only;
vector/nested layouts carry alignment padding and are declined, not
guessed), and `Gpu::step`/`dispatch`/`*_sliced` refuse a list shorter than
it, naming the kernel. Longer is still legal: shared call sites hand one
list to variants whose blocks differ in length.

**Rule:** a kernel's `Params` block is its ABI; a dispatch that passes fewer
words than the block declares is a bug even when every value it does pass
is right.
