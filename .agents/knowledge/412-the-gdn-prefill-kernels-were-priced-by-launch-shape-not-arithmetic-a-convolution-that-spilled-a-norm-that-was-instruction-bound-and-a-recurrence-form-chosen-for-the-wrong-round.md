<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 412. The GDN prefill kernels were priced by launch shape, not arithmetic: a convolution that spilled, a norm that was instruction-bound, and a recurrence form chosen for the wrong round

**Lesson:** Kernels priced by their launch shape instead of their arithmetic hide their real bound:
a launch-bounds cap made a convolution spill, a norm was limited by instructions rather than bytes,
and the recurrence form was chosen for rounds the server does not run. Benchmark the rounds that are
actually served, and check each kernel's registers, instructions and chain length before accepting
its cost.

## What happened
Question (the brief of 2026-11-01): the four Gated DeltaNet prefill kernels cost ~1.85 s of an 8K x
16 served cell against ~1.21 s for the reference server's fused delta-rule kernel + convolution +
post-convolution. Which of them are bound by bytes, which by the sequential chain, and what is left?

Where the time was (gpu1, one 4096-row segment, per layer, us; nsys of the served cell for the
shares):

| kernel | bytes at the 3.4 TB/s roof | before | after | what bound it |
|---|---|---|---|---|
| `gdn_conv_silu_seg_vec` | 168 MB -> 49 us | 104.4 | 71.1 | 85 registers (6 blocks an SM): the eleven-row window and the weights spilled to local memory |
| `gdn_gated_norm` | 126 MB -> 37 us | 65.4 | 42.4 | issue slots, 87 % busy at 57 % of the byte roof |
| `gdn_chunk_prep_seg_bf16` | ~210 MB -> 62 us | 125.9 | 125.9 | block latency (3 blocks an SM, 7.8 waves), L1/TEX 77 %; not changed |
| recurrence | 210 MB -> 62 us | 121 (1 segment), 133-213 (2-4 segments) | 117 (1 segment), 93-123 | the chain of the longest segment, then the copies |

ncu on the test binaries (`gdn_prefill_bench`, `the_serving_prefill_layer_in_a_4096_row_round`, 1 /
2 / 3 / 4 / 16 segments and seven uneven rounds).

## Why
### Wrong turns and traps

- Everything in the bench ran at boost clock; the served kernels ran 1.3-1.4 x longer (prep 126 ->
  150 us, recurrence 117 -> 156 us a launch). Ratios hold, absolute numbers do not.
- **A native kernel's launch configuration is fixed by the first registration of its name.**
  - The two-warpgroup folded form first shared `gdn_chunk_loop_seg_wg` with the unfolded form, which
    serving still launches for small rounds with 201 KB of dynamic shared memory against the folded
    form's 169 KB.
  - In the served 8K x 16 cell the later launch ran on the smaller reservation: an illegal memory
    access (`cuLaunchKernel failed`, Xid 13/43; 16 -> 596 Xid lines in `dmesg` from the repro runs)
    that no test saw, because each test process launched one form.
  - `CUDA_LAUNCH_BLOCKING=1` named the launch. The folded form is its own kernel,
    `gdn_chunk_loop_seg_wg_full`.
  - compute-sanitizer cannot run these kernels at all (the arch-specific cubins are not offered
    under it).
- Tried and dropped: 2 blocks an SM for the prep (125.9 -> 125.9 us; its 1.1 M local requests are
  not what binds it); dropping a 16 KiB tile from the recurrence's copies one at a time (-6 us
  each).

## What to do
### What moved it

- **Convolution: the launch bound was the defect.** ncu reported local loads and stores and 38 M
  issued instructions for 168 MB. `__launch_bounds__(128, 6)` capped the kernel at 85 registers;
  four blocks an SM (128 registers) removed the spills: 104 -> 71 us (three blocks 72, five 80). The
  roof is 49 us; the window re-reads three rows of eleven, so ~70 % of the roof is where this layout
  ends.
- **Gated norm: instructions, not bytes.**
  - 192 warp instructions a head row: library `expf`, IEEE division and `sqrtf`, 4 values a lane
    with a reduction per row.
  - A half-warp a row (8 values a lane, 16-byte loads), a warp walking four rows with all loads
    first, `__expf` / `__fdividef` / `rsqrtf` for the SiLU and the RMS: 65.4 -> 42.4 us (88 % of the
    roof). The E4M3 conversion helper was already minimal.
  - A test against the f64 definition (both input forms, ragged row counts, first row past zero,
    NaN-poisoned outputs) replaces the segment-round gates as the check of the bytes: they only
    compare at bf16 resolution.
- **Recurrence: the served rounds were not the benched rounds.**
  - The nsys of the 8K x 16 cell shows 2160 of its 3312 recurrence launches as the single-stage
    three-blocks-an-SM form (grid 192, i.e. two segments) at 270 us on average, because the rule was
    "more blocks than multiprocessors".
  - Those rounds are one long segment (~3900 rows) and one short one: the chain of the long one sets
    the time, and the single-stage form costs 3.3 us a chunk against 1.8 for the form with a copy
    ring.
  - Measured per form (us, 4096 rows):

  | round | Half (3-stage) | Full (two warpgroups) | Lean (1-stage) |
  |---|---|---|---|
  | 3900 + 196 | 115 | 155 | 198 |
  | 3500 + 300 + 296 | 106 | 144 | 209 |
  | 3000 + 1096 | 115 | 129 | 165 |
  | 2600 + 1496 | 127 | 115 | 150 |
  | 2048 + 2048 | 123 | 92 | 133 |
  | 3584 + 256 + 128 + 128 | 107 | 146 | 213 |
  | 4 x 1024 | 103 | 104 | 96 |
  | 16 x 256 | 147 | | 126 |

  `gdn_loop_form` now chooses from the chunk counts: Half by default, Full for two segments with the
shorter at least half the longer, Lean only for more than 132 blocks with every chain at most 24
chunks.

  Independently, the copy ring went from two to three stages (209 KB; `ci % STAGES`,
`cp.async.wait_group 1` while the chunk after next is in flight): one segment 121 -> 117 us.

  With the staging removed altogether the one-segment chain is 83 us: the 34 us left is the
`cp.async` copies themselves (about 6 us per 16 KiB tile, not their latency), which a TMA copy of
the tiles would take off the issue slots. Not done.

### Open

- Prep + recurrence per token per layer is 0.084 us against the reference's fused 0.066 (served
  exclusive time over the 8K x 16 cell, 254K prompt tokens, 48 layers): the convolution (0.026
  against 0.027) and the norm (0.0139 against 0.0139, with the reference's post-convolution kernel)
  are at parity.
- What is left: the recurrence's copies (TMA; the two halves of a head and the three heads of a key
  group fetch the same q / k tiles), and the prep's L1/TEX load (77 %, the `cp.async` destination
  writes show 10.5 wavefronts an instruction where 4 is the ideal, 1.1 M local requests), neither of
  which has a measured fix.
- The v channels' convolution could ride the prep (-41 us of the convolution's 71, the old history
  is read by the prep's first chunk, so the pool update must move behind it).
- The gated norm could ride the Full form's epilogue (both halves of a head in one block, the E4M3
  rows written directly; ~30 us net, two-segment rounds only).

## Evidence
### Quality

The prep and the recurrence arithmetic are unchanged; the SiLU / RMS approximations in the norm are
a few f32 ulp, far below the E4M3 step.

- The warpgroup-against-`mma.sync` round gate pins each launch form over rounds of 1 to 12 segments
  (output 1.7e-3 and state 2.1e-3 relative rms, limit 7.8e-3); training backward + gradcheck suites
  unchanged and green.
- `qwen35_fp8_parity` (8 x 1500 tokens, 32 steps, int8 KV, gpu1): base and now both 99.22 % greedy
  agreement, mean KL 1.765e-3, worst rel-L2 1.232e-1, 0 flips at margin > 0.5 (identical).
- Back-to-back stress (`the_segment_kernels_survive_back_to_back_launches`, ignored): the four
  kernels over a two-segment round with bf16 pools, eight layers a batch, outputs poisoned and pools
  reset between batches, each form pinned for 60 s and Half / Full alternating: 155 batches a form,
  results identical to the first batch, no new Xid.

### Served (gpu1, same client)

Exclusive kernel time of the 8K x 16 cell (nsys, 254K prompt tokens, 48 layers, gpu1), seconds: prep
0.523 -> 0.521, recurrence 0.749 -> 0.505, convolution 0.410 -> 0.319, gated norm 0.278 -> 0.170;
the four 1.96 -> 1.52 s; per token per layer 0.161 -> 0.124 us against the reference's fused delta
rule 0.066 + convolution 0.027 + post-convolution 0.014 = 0.107.

`matrix.sh` (SPEC unset, `NPROMPTS=8` for x1), base = origin/main binary of 6928ccd61, interleaved
twice (base, new, base, new), mean of the two rounds, base -> now:

- 8K x 1 mean TTFT 562 -> 562 ms (562.9 / 561.4 -> 567.4 / 557.3; the single stream is one-segment
  rounds, where only the convolution and norm gains apply: ~5 ms of 560 at the measured per-layer
  numbers, below the 1 % run-to-run).
- 8K x 16 output tok/s 311.5 -> 316.5 (313.1 / 309.9 -> 316.3 / 316.7), mean TTFT 3273 -> 3193 ms.
- 32K x 4 tok/s 73.2 -> 74.4 (73.7 / 72.8 -> 74.3 / 74.5), mean TTFT 4789 -> 4699 ms.
