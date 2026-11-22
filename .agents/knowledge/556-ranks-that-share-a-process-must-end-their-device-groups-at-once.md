<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 556. Ranks that share a process must end their device groups at once

**Lesson:** ending an NCCL communicator is a collective. `NcclRank`'s drop calls `ncclCommFinalize`, which waits in
NCCL's intra-node bootstrap barrier until every rank on the node finalizes its own communicator of the group. Ranks
that are threads of one process therefore have to drop their collectives concurrently, each on its own thread. An
object that holds every rank's collective and outlives the rank threads (a shared registry, a `Vec` of the ranks kept
by the runner) drops them one after another on one thread: the first finalize waits for ranks whose finalize can only
start after it returns, and the process hangs after a run that completed. A registry of the ranks holds them weakly
(`brain::DataParallel`'s local group keeps `Weak` members), and a runner moves each rank to the thread that trains it
and keeps only a `GroupBreaker` to abort it. An aborted communicator is not finalized, so failure paths never showed
it.

## Evidence (2026-11-22)

- A served job of 3 ranks over two servers on n442 (two ranks in one process, one in another) trained its steps,
  then hung: gdb showed the runner thread in `ncclCommFinalize -> commDestroySync -> bootstrapIntraNodeBarrier`
  from `drop(LocalGroup)` in `trainjob::job::train_chat_ranks`, the other process's rank in the same barrier.
- Regression test: `brain` `data_parallel::tests::a_ranks_collective_ends_with_the_rank_not_with_its_peers`.
