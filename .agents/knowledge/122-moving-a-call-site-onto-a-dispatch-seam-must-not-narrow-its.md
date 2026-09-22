<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 122. Moving a call site onto a dispatch seam must not narrow its bindings

Four façade methods moved from `Gpu::step` (which binds every buffer whole)
onto a seam whose operands carry an explicit `(offset, len)` range. Writing
the logical extent - `m * n` words for an output - reads like an improvement
and is a behaviour change: callers legitimately hand these methods buffers
larger than the tensor (a scratch arena, a slab shared with the next stage),
and a binding sized to the tensor starts refusing those. The range convention
already has a spelling for "whole buffer", and a pure move uses it.

The general shape: when a migration introduces a field the old call site did
not have, the faithful value is the one that reproduces the old behaviour,
not the most informative one available. Tightening it is a separate change
with its own reason and its own test.
