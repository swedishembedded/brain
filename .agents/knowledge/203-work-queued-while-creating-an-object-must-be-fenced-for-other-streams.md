<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 203. Work queued while creating an object must be fenced for the other streams

After the out-of-bounds reads of #202 were fixed, `qwen35`'s CUDA tests still
flaked under load - bit-exact comparisons (`two_shard_q4_decode_matches_the_
whole_shard_model`) failed on some runs and passed on others, with no sanitizer
report: compute-sanitizer serialises kernels, so it hides exactly this.

Tracing every dispatch (sync after each launch, download its bound buffers and
uniform block, hash them, diff a passing run against a failing one) put the
first divergence at one kernel whose uniform block read `[0, 0]` on the device
although the host had uploaded `[1, 96]` into it and the pinned source still
held `[1, 96]` afterwards. Replacing the zeroing memset of that block with a
`0xAA` memset made the failing block read `0xAAAAAAAA`: the memset, not any
other writer, ran after the upload. Printing the stream of the handle that
created the block and of the handle that used it gave 2 and 1.

A model holds two handles on one card (its own, and the one its ops run on, from
`Gpu::share`), and a recorded `Step` carries its buffers and its uniform block
to whichever handle submits it. `backend-cuda` fenced a buffer after a kernel or
a write, but not after the work it queues when creating one: `storage()`,
`buffer()` and `uniform_dynamic()` zero the new allocation, and recording a step
allocates and zeroes its uniform block, all on the creating handle's stream. The
submitting handle's stream was not ordered after that zero, so it landed after
the other handle's upload of the parameters (or after its writes into the
buffer) whenever the creating stream ran late - which a loaded host makes
common.

Everything a handle creates and initialises is now fenced as the creating
stream's, a submission fences the uniform blocks it uploads into as well as the
buffers (so the next handle's upload cannot overtake a kernel still reading the
previous parameters), and
`backend::tests::what_a_handle_creates_carries_a_fence_for_other_handles` pins
it.

Rule: an asynchronous initialisation is a write like any other. A fence per
buffer is only as good as the places that set it, so every place that queues work
on an object another stream may touch - including its creation - sets it.
