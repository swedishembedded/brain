<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 469. A per-kv-head row scale shares its sector with the other heads, so an 8-bit decode attention must run the kv heads of a key range together

**Lesson:** the 8-bit KV pools keep one f32 scale per (token slot, kv head), the heads of a slot adjacent. A decode
block reads the scales of ITS head only, so each 4-byte scale costs a whole 32-byte sector unless the blocks of the
other heads of the same key range run at the same time and find that sector in L2. Order the grid kv head fastest;
the bytes a kernel must read are the bytes the profile counts only when every sector it touches is shared that way.

## What happened

The tensor-core 8-bit decode (`paged_flash_decode_gqa_kv8.cu`), generalised to head width 128 for the dense Qwen3
family (8 kv heads of 128), plateaued at 82 % of the 3.4 TB/s roof at 64 sequences x 32K keys while the same kernel
at head width 256 (4 kv heads) reached 95 %. `ncu` on the kernel's own test binary (`decode_attention_matrix_cell_report`)
showed `dram__bytes_read.sum` 5.37 GB against 4.43 GB of keys, values and scales, with DRAM 85 % busy: a fifth of the
traffic was over-fetch. At 8 heads a slot's eight scales fill one sector, and a 16-key chunk's 16 scales of one head
touch 16 sectors (512 bytes for 64 used) for 4 KiB of elements, a quarter more bytes; at 4 heads of 256 it is 6 %.
The grid ran the sequence fastest, the split next and the kv head slowest, so the heads of one key range were a whole
grid apart and every head fetched the sectors again.

## What to do

- In a kernel that reads per-(slot, head) metadata from a pool of interleaved heads, make the kv head the
  fastest-varying part of the block index (then the row, so a verify round's rows of one sequence still run together).
  The 8-bit and bf16 decode kernels do; the partial-record layout does not depend on the order.
- Count the bytes DRAM actually read (`dram__bytes_read.sum`) against the ideal before tuning occupancy or
  instruction count: a kernel at 85 % DRAM busy with a fifth of its bytes wasted is a layout problem, not a pipeline one.

## Evidence

GH200 gpu3, `decode_attention_matrix_cell_report`, int8 pool, 32 query heads over 8 kv heads of 128: the reorder took
64 x 32K from 1571 to 1332 us (82 to 97 % of 3.4 TB/s) and 16 x 32K from 392 to 352 us (81 to 90 %); the 27B's 4 kv
heads of 256 gained too (64 x 32K 1351 to 1285 us). Hoisting the copies' address arithmetic out of the chunk loop and
loading the next chunk's block-table entry one chunk ahead moved 16 x 32K from 414 to 401 us before the reorder.
