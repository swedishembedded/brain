<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 548. A nonblocking NCCL communicator puts a thread and a poll under every small collective

**Lesson:** brain's per-process NCCL communicator was nonblocking (`blocking = 0`) so a deadline could bound every
call. NCCL then launches each collective from a thread of its own and the caller polls `ncclCommGetAsyncError` until
the launch finished. On two GH200 ranks every operation from 8 B to 1 MiB took about 126 us p50; nsys on rank 0
showed one `pthread_create` per operation and the 50 us poll sleeping about 104 us per call. The pre-multiplied-sum
reduction op and the buffer fences, the first suspects, cost nothing measurable (the op is made only for a scale other
than 1; the fences return at once with one CUDA context per process). Joining nonblocking and running the collectives
on a blocking `ncclCommSplit` child, with a watchdog thread that calls `ncclCommAbort` on a call past its deadline,
brought small operations to 5-7 us (1 MiB all-reduce 29.5 us) with large-message bandwidth unchanged and the NCCL
fault test passing. Measured 2026-11-21 on n467 with `brain bench collectives`.

**How to apply:** keep NCCL calls on a blocking communicator and bound them with a second thread's abort, not with a
nonblocking one and polling. Only the join (where a peer may never arrive) needs the nonblocking mode. A split from a
nonblocking parent may return `ncclInProgress` with the child not yet written: wait on the parent, then read the child.
Before blaming per-op setup, profile with the OS runtime trace (`nsys -t osrt`): thread creation and sleeps show there.
