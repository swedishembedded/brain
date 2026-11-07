<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 414. An optimisation proven on the default KV tier left the tier the reference servers default to on the slow kernel, and only a matched-settings comparison showed it

**Lesson:** An optimisation proven on one storage tier of the cache is not proven on the others.
Check the profile of every tier a dispatch serves, and compare engines at identical settings, never
defaults against defaults.

## What happened

The staged warpgroup prefill attention (widen the round's keys once, then bulk-copy fp16 tiles) was
built and measured over the int8 and FP8 caches, brain's defaults. A bf16 cache, which is what the
reference servers run by default and what an engine comparison at identical settings must use, kept
version 2: 180.9 ms of attention per 8K prompt against 30 ms (nsys, lone 8K prompt, FP8 weights, f32
GDN state).

The first matched-settings smoke cell (8K x 1, bf16 KV, f32 state, no speculation) put brain at 67.5
tok/s | 725 ms TTFT behind both references (76.0 | 559, 72.9 | 599), and the profile diff of the two
brain runs named the one kernel.

## Why

Every brain-only scoreboard was run at brain's defaults and never exercised that path.

## What to do

`flash_prefill_f16_hd256_wgs_widen_bf16`: the staging pass for a bf16 cache. A bf16 value is the top
half of an f32, so the pass widens to f32 and rounds to fp16 with the same `brain_pack_f16x2`
version 2 uses, writes unit row scales (K scale 1/sqrt(d) log2 e, V scale 1, zero for keys past the
longest sequence) into the same records, and the unchanged attention kernel reads them.

`paged_flash_prefill_wg_steps` takes the cache as `WgKv` (bf16, or 8-bit with row scales); there is
no unstaged warpgroup form for bf16, so a short round declines and keeps version 2.

A kernel family proven on one storage tier is not proven on the others: list the tiers a dispatch
serves and check the profile of each, at least the one every comparison will use. Engine comparisons
run at identical weight, KV, state and speculation settings, never defaults against defaults; the
defaults-against-defaults scoreboard counted wins that came from a smaller KV cache and speculation,
and hid this loss.

## Evidence

Served, gpu1, FP8 weights, bf16 KV, f32 GDN state, no speculation, two interleaved rounds, tok/s |
median TTFT ms, base -> new: 8K x 1 66.7 | 783 -> 70.5 | 595, 32K x 1 34.0 | 4303 -> 43.9 | 2636, 8K
x 16 203.3 | 3752 -> 221.0 | 3322. Agreement with the portable fp32 kernel is at least as good as
the int8 staged path's (rel-L2 8.8e-4 at score scale 16x).
