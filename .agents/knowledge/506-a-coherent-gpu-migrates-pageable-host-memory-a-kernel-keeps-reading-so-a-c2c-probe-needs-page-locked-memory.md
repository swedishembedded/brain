<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 506. A coherent GPU migrates pageable host memory a kernel keeps reading, so a C2C probe needs page-locked memory

**Lesson:** on a GH200 a kernel may read ordinary (`AllocPolicy::System`) host memory directly, and the driver moves
pages it reads repeatedly into HBM. The read probe over 1 GiB of such memory, written first by a core next to the
card, measured 1767 GB/s: above the 450 GB/s one direction of NVLink-C2C can carry, so it was reading device memory.
Over the same buffer page-locked (`AllocPolicy::Pinned`) it measured 415 GB/s, 92 % of the link.

## What to do

- Measure or budget the host link with page-locked memory; a rate past the link's published peak is migration, not
  a fast link (`device_rates`' `c2c-read` row, checked by the CUDA test `device_rates` against the published peak).
- The same migration is why weights left in pageable host memory can look free to read after the first passes and
  then cost device memory: measure placement, do not infer it from the allocation policy.
