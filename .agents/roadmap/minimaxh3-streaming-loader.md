# MiniMax-H3 streaming loader: a slowdown hunt that ended in three dead
# hypotheses, one retracted finding, and a measurement harness

Kept out of `minimaxh3.md` deliberately - that file was being written
concurrently while this ran, and a separate file is one merge instead of one
guaranteed conflict. Fold it in whenever convenient.

**Read the retraction section before reusing any number from an earlier
draft of this investigation.** The headline finding it started with was
wrong, and the way it was wrong is the most reusable thing here.

## The question

`H3Transformer::forward_streaming_with_taps` at real dimensions
(hidden 5376, ffn 14336, 56x128 heads, 50 blocks) was reported taking far
longer on a Tesla P40 via wgpu than the same forward on the CPU backend,
with the GPU idle for most of the wall-clock while one host core stayed
pinned - the signature of a host/driver-side cost, not a compute one.

## What the premise turned out to be

**On this branch the pathology does not reproduce.** With the real
checkpoint, the real golden, and `BRAIN_MINIMAXH3_TEST_DEVICE=vulkan`, the
50-block streaming forward completes in a time comparable to (in fact below)
the CPU backend's, with VRAM bounded across the whole run and the reference
cosine unchanged. The very large figure that motivated the hunt was measured
against a working state that is not committed here, so whatever produced it
is not in this branch's `block.rs`/`model.rs`. Anyone picking this up should
re-measure on their own working tree **first** - the harness below exists to
make that a single command.

## Hypotheses killed, with the probe that killed each

`crates/gpu-core/tests/bench_upload_cadence.rs` - per-CALL fixed costs of
the upload primitives on a real P40. Every timed region ends in a real
readback, never a bare `write`/`submit` return (see `vram_overhead.rs`'s own
history of probes that measured nothing because `write_buffer` had not been
submitted yet).

1. **`poll_wait` has a large fixed per-call cost, so calling it per tensor
   rather than per block is a regression.** No. A bare `poll_wait` with
   nothing in flight is on the order of a microsecond; per-tensor against
   per-block cadence differs by single-digit milliseconds per block. Note
   the sign as well: `write_at` followed by `poll_wait` measured *faster*
   than `write_at` left unfenced, because polling lets staging buffers be
   recycled instead of accumulating. Polling more often was helping.

2. **`UPLOAD_CHUNK_WORDS` (4 MiB) is too conservative, and the per-`write_at`
   overhead dominates a 308 MB tensor.** No. Warm, such a tensor uploads at
   low-single-digit GB/s and the spread across 4 MiB / 16 MiB / 64 MiB /
   unchunked is tens of milliseconds. 16 MiB is mildly the best warm - not
   worth changing a shared constant that four other crates depend on.

3. **The forked wgpu's `StagingBufferPool` misses every block, re-pinning
   host pages each iteration.** No - it works as documented. The cold cost
   its own doc describes is plainly visible on the first touch of a size
   class and is gone by the next repetition. Reuse engages. **No wgpu-fork
   change was needed or made.**

Consequence: a whole block's worth of tensors uploads in well under a second
warm, so the `write_at` path was never where a large missing time could hide.

## The retraction

A first version of `crates/minimaxh3/tests/bench_block_load.rs` compared
whole-tensor decode against bounded chunked decode and reported whole-tensor
several times slower, which pointed convincingly at
`TensorSource::with_tensor` materialization as the root cause (every DiT
tensor in the checkpoint is BF16, so `raw_words` can never lend a zero-copy
view and something must decode). **That result was an artifact of the probe,
twice over:**

* The whole-tensor arm was written
  `with_tensor(&name, &mut |d| whole = d.to_vec())`. But `with_tensor`
  *already* allocates the whole `Vec<f32>` and lends it - the `.to_vec()`
  charged that arm a second allocation plus a full extra copy that the real
  `block::load_dev` (which reads the borrowed slice in place via
  `ctx.upload`) never pays.
* The first read of any block also paid a cold page cache on a 66 GB
  checkpoint, and that was silently attributed to decode cost.

Corrected - both arms reduced to a checksum, cache warm - **the result
reverses**: whole-tensor decode is roughly two-thirds the cost of the
chunked decode, consistently, on every large tensor. Host decode is a modest
share of a forward, not the bulk of it, and chunking it is a pessimization.

This was then confirmed end-to-end rather than left as a probe result: a
streaming-loader change (routing `load_dev`/`load_fc1` through
`paramstore::upload::Uploader`) measured as a **regression** on the CPU
backend and as within run-to-run noise on the GPU. It was therefore
**not shipped**. On the CPU backend a "device" buffer is host memory and
`storage_init` merely reads the borrowed slice, so there is no upload to
avoid and all the chunked path adds is per-chunk overhead; on wgpu the
`create_buffer_init` doubling that would have justified it was already fixed
upstream in the fork (`vram_overhead.rs`: "wgpu now measures no overhead on
any probe here").

**The lesson, which the rules already state and this is now a second witness
for: the measuring apparatus is under test too.** A probe whose two arms do
different amounts of work will confidently report a difference that is
entirely its own, and a first-touch run on a large mmap measures the disk,
not the code.

## What was kept

* `crates/gpu-core/tests/bench_upload_cadence.rs` - the three probes above.
* `crates/minimaxh3/tests/bench_block_load.rs` - the corrected host-decode
  comparison, with the `.to_vec()` trap documented in place so the next
  person does not re-dig it.
* `crates/minimaxh3/src/model.rs`:
  - `BRAIN_MINIMAXH3_TEST_DEVICE` (default `cpu`) on the real-weight
    layer-by-layer test. Without it there was **no way to regression-check
    the streaming forward on an actual card** - which is why a
    device-specific claim went unchecked for as long as it did. This is the
    single most useful artifact of the exercise.
  - one `poll_wait()` per block iteration after an explicit `drop(w)`.
    A memory-safety bound, not a speed change: buffers dropped host-side are
    not freed on the card until the commands referencing them retire, so a
    submit-only loop accumulates every iteration's block. Measured to cost
    effectively nothing at this cadence.

Correctness is unchanged and identical on both devices: worst cosine
0.9999998613 at `output_audio` against the real diffusers reference.

## Where to look next, if the slowdown resurfaces

1. Re-measure on your own tree with `BRAIN_MINIMAXH3_TEST_DEVICE=vulkan`
   before theorising - the premise did not hold here.
2. `adaln_proj.linear.weight` is 260 M parameters per block, read in full
   every block and never uploaded, so no upload-side change can touch it.
   `precompute_adaln.rs` already exists to fold it into a small gathered
   table (its own doc: 260M params/block x 50 blocks = 13.0B of the 33B
   total). That is the one structurally large win left in this loop, and it
   is a checkpoint-transform decision rather than a loader fix.
3. If a host-side cost is suspected again, time it with **no device open**
   (as `bench_block_load.rs` does) before attributing anything to the
   driver, and warm the cache before believing the first number.
