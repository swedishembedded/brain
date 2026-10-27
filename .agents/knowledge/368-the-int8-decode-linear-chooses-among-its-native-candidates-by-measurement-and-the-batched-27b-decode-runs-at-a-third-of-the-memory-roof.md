<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 368. The int8 decode linear chooses among its native candidates by measurement, and a batched 27B decode step streams its weights at a third of the memory roof

Question: why does Qwen3.8-27B int8 decode reach only 480-830 tokens/s aggregate at 12-32 streams on a GH200 when
the weights are 27 GiB and the card reads 3.7 TB/s, and how do we stop answering that per card by hand?

Mechanism (`gpu_core::native_tune`, rows in `gpu_core::native_upgrade`): a kernel slot may list several native
candidates; each states the capability it needs through its registry floor and the shapes it serves. On the first
handle of a device the int8 decode linear is timed on a grid of shape classes (row bucket x K class x weight-bytes
class), each candidate is checked against an f64 oracle (a disagreeing one is rejected for that class), the device's
read roof and launch floor are measured with a streaming probe, and the table (choice, microseconds, GB/s, fraction of
the bound `max(bytes / roof, launch floor)`) is cached per device, build and candidate source. `brain roofline gpu
--kernels` lists it and the classes under 60 % of the bound; `--kernel slot=candidate` pins one, `--no-autotune`
keeps the defaults. Rows 1-4 are served by the dp4a block alone (the exact-decode contract keeps those bit-identical
to a single row); a candidate that copies 16 bytes at a time declares its alignment and a sliced binding off it keeps
a candidate that can run (without that check the tiled GEMM raised `misaligned address` on a window at word offset 1).
A slot with several rows that the tuner has no problem generator for (the K-quant narrow and wide blocks) keeps its
row-count rule: the tuner must only launch what it can build inputs and an oracle for.

Result on the GH200 (idle card, 4 candidates x 56 classes, tuned in about 20 s once): the choice is the dp4a GEMV up
to 4 rows, the tiled tensor-core GEMM at 9-16 rows for the large matrices and the 32-row GEMM above, with the new
wide tensor-core GEMV winning the mid-size classes at 9-24 rows. Decode step `qwen35_decode_profile --batches
12,20,32 --kv int8 --context 2048`, 6 steps, best of two alternating runs, baseline and tuned build: 487 / 592 / 825
tok/s against 490 / 601 / 829, i.e. no change end to end: the tuner confirmed the existing hand choice and no
candidate beats it. The honest content of the table is the gap: 53 of 56 classes are under 60 % of the roof; the
27B step's weight phase streams 1.0-1.2 TB/s (batch 12: 1171 GB/s of 3680), the 17408-wide down projection runs at
18-41 % at 9-32 rows, and the single-row GEMV reaches 75 % only on the largest matrices.

What did not move it (idle card, device-timed, weights past L2): the wide GEMV's staged loads per thread, the
blocks-per-SM cap and the register budget are flat within 3 %; replacing every MMA by an XOR (no arithmetic) still
reads only 2.1 TB/s at 12 rows, so the limit is the load structure (8-byte weight loads, a thread's x and scale
loads per group), not the tensor-core work. Loading the scales as 16-byte vectors was 30 % slower (a runtime-selected
fast path costs more than it saves). Next candidate: stage weight tiles through shared memory with cp.async/TMA and
feed `ldmatrix`, as the prefill GEMM does, with the activation tile resident; it lands as another row of the slot
and the table says at once whether it wins.

Trap (fixed): `brain roofline` used to open every GPU of the box whatever `--device` said (it probed the second card
of a shared node once); it now opens only the cards `--device` selects (`devices::schedulable_gpus`).
