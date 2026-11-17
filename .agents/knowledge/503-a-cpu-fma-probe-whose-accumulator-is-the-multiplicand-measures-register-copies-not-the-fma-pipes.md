<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 503. A CPU FMA probe whose accumulator is the multiplicand measures register copies, not the FMA pipes

**Lesson:** on aarch64 `fmla` (and x86's three-operand FMA forms) overwrite their ADDEND. A probe chain written
`acc = acc * c + d` (the GPU `roof_fma` shape, fixed point 1.0) puts the accumulator in the multiplicand, so the
compiler must copy `d` into a fresh register before every FMA and move the result back: the loop issues three
instructions per FMA and the probe reports the copy rate. Write CPU chains as `acc += x * y`, the accumulator as the
addend, and read the disassembly once.

## What happened

`brain perf roofline cpu` used to report the CPU's fp32 roof as brain's own conv2d mix (124 GFLOP/s on a 288-core
GH200 node, under half a percent of the cores' peak). Its replacement, sixteen NEON chains per pinned thread
(`backend_cpu::roofline::measure_fma`), first measured 22 GFLOP/s on one Neoverse V2 core and 5.4 TFLOP/s on 288:
a fifth of the derived peak (cores x clock x 32 FLOP a cycle). The loop body held 16 `fmla`, 16 `movi` and 16 `mov`.
Rewritten as `acc = vfmaq_f32(acc, x, y)` (one `fmla` each, nothing else), the same core measured 60 GFLOP/s alone and
101 GFLOP/s a core with 72 threads (94 % of the derived peak; the single-thread figure is the clock before it ramps),
and 288 threads 22-24 TFLOP/s on the shared node.

## What to do

- A CPU throughput probe keeps its accumulator as the instruction's addend; the multiplier operands are loop
  invariants the compiler cannot fold (`black_box` one of them).
- Check the loop body in the disassembly (`objdump -d` on the test binary) before trusting a probe's number: named
  locals versus an array made no difference here, the operand order did.
- Grade brain's CPU kernels against the probe, not against each other: the conv mix row sits beside it in the report.
