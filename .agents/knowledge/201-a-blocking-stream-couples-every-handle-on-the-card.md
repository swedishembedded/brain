<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 201. A blocking stream couples every CUDA handle on the card

`backend-cuda` made one stream per handle with `cuStreamCreate(.., 0)` (a
blocking stream), synchronised with `cuCtxSynchronize`, and moved bytes with the
synchronous `cuMemcpyHtoD`/`cuMemcpyDtoH`/`cuMemsetD8` calls. The reasoning, in
the code comments, was sound for one handle: those synchronous calls run on the
legacy default stream, and only a blocking stream is ordered against it.

With two handles on the same card it is wrong in three ways, all of which the
driver reports as an error rather than as slowness:

- `cuCtxSynchronize` waits on the whole shared primary context, so a
  synchronise on one handle failed with "operation not permitted when stream is
  capturing" whenever any other handle was mid-capture.
- A blocking stream joins the legacy stream's implicit synchronisation domain,
  so a legacy-stream copy or clear from another thread failed with "operation
  would make the legacy stream depend on a capturing blocking stream".
- The first of these is what made `cuda_graphs.rs` fail intermittently, but
  only on a box where the tests ran in parallel and the timing lined up, which
  is why it looked like noise. It skipped green on every box without a GPU.

Now each handle owns a non-blocking stream, waits with `cuStreamSynchronize`,
and every transfer and clear is enqueued on that stream
(`cuMemcpyHtoDAsync`/`cuMemcpyDtoHAsync`/`cuMemsetD8Async`), which keeps host
writes ordered against the dispatches that read them without the legacy stream.
`a_handle_is_not_blocked_by_another_handle_capturing_a_graph` captures on one
handle while another thread transfers, clears and synchronises on another; it
fails against the old code with the second message above.

The same change removed a race from `the_binding_ceiling_is_the_portable_one...`,
which compared two `cuMemGetInfo` readings taken at different times on a card
that other tests were allocating on.

Rule: a resource shared by every handle in a process (a primary context, the
legacy default stream) must not be what a handle waits on or orders against.
