<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 62. A mapped buffer's pointer is only as aligned as the allocator happened to place it, and `cast_slice` on one holds by luck

`WgpuBackend`'s per-kernel timestamp readback did
`bytemuck::cast_slice::<u8, u64>(&slice.get_mapped_range())`, and the readback
path did the same to `f32`. Both had worked for the life of the file. Then an
unrelated change - keeping ONE staging buffer alive between reads instead of
allocating a new one per read - moved `gpu-allocator`'s packing, the
timestamp-resolve buffer landed at a suballocation offset that was 4-mod-8,
and a real 22B run died mid-forward with
`cast_slice>TargetAlignmentGreaterAndInputNotAligned` after every stage of the
first forward had already printed.

The general shape:

* **A mapped pointer's alignment is an allocator artifact.** It is
  `memory_block.mapped_ptr + suballocation.offset`, and the offset is chosen
  from the buffer's own `memoryRequirements.alignment` (4 for a plain
  host-visible transfer buffer here) and from whatever else happens to be live.
  Nothing promises it is aligned for the type you want to read.
* **Widen the DESTINATION instead of casting the source.** `&mut [T]` viewed as
  bytes (`cast_slice_mut::<T, u8>`) is always byte-aligned, so
  `copy_from_slice` from an arbitrarily aligned `&[u8]` is well defined, the
  `memcpy` is the same one `to_vec` was doing, and the precondition disappears
  rather than being documented. One helper
  (`backend_wgpu::copy_pod_from_bytes`) now covers all three sites.
* **The bug class is testable even though the trigger is not.** No test can
  force the allocator to hand back a 4-mod-8 offset, and a gate that has never
  been seen fail is a hypothesis. What IS deterministic is the property: copy
  the same payload out of a byte slice at every source offset in `0..8`. That
  gate is red under `cast_slice` with the exact panic the real run produced,
  and it needs no GPU.
* **Watch for this whenever an allocation-lifetime change lands.** Pooling,
  recycling and arenas all move packing, and anything downstream that assumed
  an incidental alignment fails far away from the change - here, in a profiler,
  in a different subsystem, several minutes into a run.

A second, unrelated hazard was confirmed while chasing this and is worth
recording separately: `crates/backend-wgpu/tests/upload_flush.rs` **hangs at
100% CPU** when its two tests run concurrently (`--test-threads` > 1), because
each builds its own `WgpuBackend` and two Vulkan devices in one process is the
documented deadlock this driver has. Reproduced on unmodified `main`, so it is
pre-existing, and the default `TEST_THREADS=8` reaches it.
