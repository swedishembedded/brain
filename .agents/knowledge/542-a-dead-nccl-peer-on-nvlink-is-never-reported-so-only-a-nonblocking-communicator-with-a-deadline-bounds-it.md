<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 542. A dead NCCL peer on NVLink is never reported, so only a nonblocking communicator with a deadline bounds it

**Lesson:** when one rank of a one-rank-per-process NCCL communicator exits after the group formed, the surviving
rank's next collective does not fail: over NVLink (peer-to-peer, no socket to close) its kernel keeps waiting for the
peer's data and `ncclCommGetAsyncError` keeps reporting nothing, so a blocking communicator, a blocking
`ncclCommInitRank`, or a `cuStreamSynchronize` after the collective hangs forever. Measured 2026-11-21 on one GH200
node with NCCL 2.32.3 (`crates/model/tests/it/nccl_job.rs`, `a_rank_that_dies_is_an_error_within_the_op_timeout`):
the survivor saw an error only from its own deadline, 5.5 s for a 5 s op timeout, and `ncclCommAbort` then stopped the
kernel and released the stream.

**How to apply:** create every cross-process communicator nonblocking (`ncclConfig_t.blocking = 0`), poll
`ncclCommGetAsyncError` with a deadline after init, split and every enqueue (`ncclInProgress` is normal there), wait for
a collective by polling the stream and the async error against the job's op timeout instead of synchronizing, and
abort the communicator when the deadline passes; `backend_cuda::nccl::NcclRank` does this. Never rely on NCCL to
notice a lost peer by itself.
