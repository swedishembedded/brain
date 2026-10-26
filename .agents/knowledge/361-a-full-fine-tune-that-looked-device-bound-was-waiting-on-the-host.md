<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 361. A full fine-tune that looked device-bound was waiting on the host

Symptom: a 4B full fine-tune on a GH200 (forward plus backward 0.2-0.4 s per
step) took 25-30 s per step in fp32 and 12 s per step resident, started after
8 minutes (resident) and spent a minute after the last step. The card showed
0% utilization.

Causes, each found by a wall-clock stage timer (`BRAIN_PROFILE=1`) and none by
a kernel profile:

- A tile reader re-decoded the whole tensor for every window it was asked for
  (`with_tensor_chunks` has no range), so the embedding was decoded ~90 times.
  Windowed readers use `with_tensor_range`.
- Dtype decode and encode ran element at a time on one thread.
- A device readback into a fresh `Vec` is page-faulted in while the copy engine
  waits (5 GB/s); into a warm buffer it runs at the link's speed (170 GB/s
  measured here, pageable memory included, through ATS). Read into reused
  buffers (`Gpu::read_into`).
- The host update forked and joined the CPU pool ~10 times per tile, each costing
  about a millisecond on a 144-thread pool, and used every allocated core on a
  four-module node. Restricting the work to the cores of the card's own module
  (`local_cpulist` of its PCI device) and its memory made the same code 3-4x
  faster. A bigger tile and one in-place pass per tile (decode, rule, encode
  without scratch) removed the rest.
- The save gathered the model as `Vec<f32>`, converted it to a byte buffer,
  serialized it into a third and copied it into a fourth: 64 GB of host memory and
  ten single-threaded seconds for a 16 GB file. Tensors are now produced one at a
  time and written by several threads.
- The portable SHA-256 of `sha2` runs at a few hundred MB/s on aarch64; the `asm`
  feature selects the hardware instructions (1.6 GB/s here).

Rules: on a multi-module node, measure host work with the process pinned to the
card's module before concluding anything about the host; a loop that is cheap
per element is not cheap per fork-join; never read device data into a fresh
allocation per tile; a whole-model buffer in a save path is a host-memory bug
whatever the model size.
