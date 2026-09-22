<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 58. A measured number in a doc comment outlives the tree it measured, and the next reader will read it as current

`ltxv::dit::ada_layer_norm_single` carried, in an ordinary `//` comment, "the
bulk of the ~76 s this stage cost per forward call (measured: the table GEMM
alone is ~14 s of it)". A later profiling pass measured the same stage, on the
same command, same shape, same box, at **10.2 s** - table GEMM 8.9 s, embedder
1.4 s. That reads as a 7.5x contradiction on the total and a ~45x one on the
embedder, and it cost a full round trip to resolve.

Both numbers were right. The comment described the tree *before* the two
changes made in the same commit that wrote it (batching the timestep embedder,
and `backend_cpu::host_gemm`'s blocked GEMM); the profile described the tree
after. The comment's own tense said "was", which is exactly as much hedging as
a reader gets - and it is not enough, because nothing beside it says *which*
version of the code the number belongs to.

Three things this actually shows:

* **A number in a comment has no version.** Code is diffed, comments are not
  re-derived. A figure that was a *before* number the day it was written
  becomes an *unmarked* number the day after, and the next profile is compared
  against it as though it were a claim about today.
* **The disagreement points the wrong way.** The reflex on a 7.5x gap is to
  suspect the new measurement - the instrumentation, the scope, the box. Here
  the instrumentation was correct, the scope was correct, and the box was
  quiet; the stale figure was the only wrong input, and it was the one nobody
  thought to date.
* **The workspace already has the right home for it.** Per AGENTS.md, a
  measured number about one model belongs in that model's
  `.agents/roadmap/<model>.md` (dated, phase by phase, with the command) or in
  the assertion of a test that will go red when it stops being true. Both of
  those places have a version. A comment does not.

So: **a comment may say what the shape is and why it costs what it costs; it
must not carry the number.** Where a figure is genuinely load-bearing for
understanding the code, name the harness that reproduces it
(`ltxv_bench streamed <layers> <t> <ctx> 1 1 <distinct_timesteps>`) instead of
the answer it printed once. And when you do have to leave a historical figure
in place, say which commit's behaviour it describes, in the same sentence.

**The nesting corollary, from the same investigation.**
`gpu_core::profile::stage_time` prints one line per call and has no idea that
one bracket encloses another. `forward_q_streamed` brackets the whole adaLN
stage AND its two halves, so the three lines are `parent = child + child`
(measured residual: 5 ms in 10,300). Summing the printed lines double-counts,
and the profile looks like the stage costs twice what it does. The fix is
free and belongs in the label: the enclosing bracket says it is a TOTAL and
names what it nests.
