<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 205. A decode token is host-bound until it is one replayed graph

Qwen3.8-27B int8 on a GH200, single stream: 81 ms/token of wall against 25 ms
of device time (nsys: the card was busy a quarter of the time), 2500 kernel
launches a token, and a graph path that "worked" in every unit test.

Four separate defects, each hiding the next:

1. **The graph never engaged.** The decode submits per layer, every layer names
   different weights, and a capture fired only when the SAME submission shape
   arrived twice in a row - which a sixty-four-layer step never does. One
   `cuGraphInstantiate` in a whole run. A trigger has to be a set of recently
   seen shapes (a token is two shapes in alternation: the layer stack, then the
   head after its readback), and the unit of capture has to be the token, not
   the layer (`begin_pass`/`end_pass` hold the layer submissions and issue them
   as one).
2. **2500 `cuMemAlloc`/`cuMemFree` pairs a token.** Every temporary was a fresh
   allocation, and `cuMemFree` waits for the whole device, so the host and the
   card ran in lockstep - and every free bumps the allocator epoch, which
   discards every captured graph. The scratch arena removes both, but only if
   EVERY buffer the step names lives in it: the per-token inputs (token, M-RoPE
   rows) were `storage_init`, the logits head used the same arena as the layer
   stack and evicted it slot for slot, and the linears' activation scratch is
   allocated through a second device handle (`Ops` runs on `gpu.share()`), which
   has an arena of its own. The attention scratch stride tracked the position,
   so it changed size every token; it is rounded up to a bucket.
3. **A copy node per step.** Once the graph replayed it was SLOWER than launching
   each kernel alone: a captured host-to-device parameter copy is a node the
   device executes between two kernels at about 12 us, and a token has 2500
   steps. One graph-private parameter block, one copy node, every step's uniform
   pointing into its slice.
4. **The card idled while the host built the token.** A pass is issued when it
   ends, so ~2.7 ms of host work (recording ~860 steps, resolving and signing
   them) sat in front of ~13 ms of device work. `flush` inside a pass issues what
   is held once 256 steps are, so the first chunk is running while the rest is
   built; the cut depends only on how many steps have been held, so a repeating
   token is cut identically and each chunk replays.

How to see it: `qwen35_decode_profile` prints the per-token spread and the host
calls per token (`0 cuMemAlloc, 0 individual launches, 5 graph replays` is the
steady state), and `nsys profile --trace=cuda --cuda-graph-trace=node` plus the
sqlite kernel table gives device-busy against the token's span. Note that
`--cuda-graph-trace=node` inflates the per-node gaps to ~15 us; trust it for
busy time, never for gaps. A graph of 865 dependent trivial kernels costs about
0.75-1.0 us a node (`graphgap` microbench), which is the floor.

Rule: measure the replayed pattern the caller actually uses, with the readback
between submissions, and count driver calls per token - a green graph unit test
on a repeated identical submission says nothing about whether the real tape
repeats.
