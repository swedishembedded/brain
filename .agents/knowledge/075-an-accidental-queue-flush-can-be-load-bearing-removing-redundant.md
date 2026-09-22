<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 75. An accidental queue flush can be load-bearing: removing redundant work made a decode pass 1.45x SLOWER

The qwen35 decode tape rebuilt an M-RoPE cos/sin row on the host and uploaded
it twice for every GQA layer, at every token. The row is a pure function of
the position, so at `full_attention_interval = 4` those were 16 identical
recomputes and 32 identical uploads per token, of which 1 and 2 were needed.
Textbook lesson #F.2b dedup: key on the position, upload once, reuse.

The dedup was correct, strictly reduced work, and cost **1.45x on the whole
pass** - 7.19 tok/s down to 4.96.

The reason is not in the M-RoPE code at all. `Gpu::submit` on the wgpu backend
does **not** submit: it appends to a pending list that is flushed at the next
readback, so a whole decode step's ~1250 dispatches are recorded on the host,
in one command buffer, and only then handed to the card. `Gpu::write*`,
however, calls `flush_inner()` first, so that a host write can never race
ahead of dispatches recorded before it. Those 32 M-RoPE uploads were therefore
32 queue submissions per token, chopping the pass into pieces the device could
start on while the host was still building the next one. The model had been
getting host/device overlap **by accident, as a side effect of a correctness
guard in an unrelated function**, and nothing in either function said so.

Flushing on purpose - one `Gpu::flush()` per decoder layer, which is the
primitive that exists for exactly this ("start recorded work without
waiting") - recovered all of it and then some: 7.44 tok/s WITH the dedup.
Cadence turned out to matter and to have a real optimum: two flushes per layer
measured 7.31 and every second layer 7.44, against 7.47 for one per layer.

**Rules that follow.** First, before deleting a redundant host write in a
dispatch-heavy loop, find out whether that write is the only thing submitting
the queue - `grep` the backend's `write`/`write_at` for a `flush` before
assuming an upload is only bytes. Second, a pass whose submission is lazy has
a *submission cadence* as a real tuning parameter, and leaving it implicit
means it is set by whichever unrelated call happens to flush; make it
explicit and measure it, because the accident is unlikely to have picked the
optimum and is guaranteed not to survive the next refactor of the function it
is hiding in. Third, this is another entry for §E's table of confident
hypotheses the profile killed - "remove redundant work" is normally free and
here it was the most expensive change of the session, and only a whole-pass
measurement could tell, since the per-kernel device table barely moved
(103.08 -> 104.61 ms/token on the dominant GEMM) while wall clock moved 45%.
