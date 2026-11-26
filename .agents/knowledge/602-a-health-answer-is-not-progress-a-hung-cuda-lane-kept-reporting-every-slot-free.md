<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 602. A health answer is not progress: a hung CUDA lane kept reporting every slot free, and the router kept feeding it

**Lesson:** a server whose `/health` answers can still be doing nothing. Judge a serving process by whether the work it
holds moves (bytes streamed, generations finished), never by whether its health route answers or what its last
dispatcher pass counted; and bound every device wait on every backend, or a wedged kernel turns into a process that
looks idle and accepts work forever.

## What happened

On 2026-11-26, 16 `brain serve --openai` servers (one per GH200, Qwen3.8-27B-FP8 and Qwen3-4B resident,
`--max-batch 64 --training`, `--speculative` at its default `auto`) ran agent traffic behind one router per node
(`brain serve --upstreams-file ...`). After several hours, 7 of the 16 stopped answering:

- GPU utilisation at 100%; every request hung, while `/v1/health` reported every decode slot free and `queue_depth` 0.
- A gdb dump of one (`wedged-8810.bt`, operator's evidence directory): the qwen35 intake threads idle, the dispatcher
  and workload threads waiting on their channels, and the main GPU lane thread (`brain-lane-Gpu(`) blocked inside
  libcuda in a semaphore wait (`sem_wait` with no timeout, under a stream or event synchronisation): a kernel on the
  device never completed.
- The servers' last log lines showed speculation switching on and off at 1 to 2 running sequences.

All 16 were restarted (with `--speculative off` as a mitigation; whether the hang recurs without speculation is being
watched). The routers then showed every upstream healthy with all slots free and nothing in flight, yet upstreams
0, 6 and 9 reported `queue_depth` 5, 10 and 2, and every chat request through the router hung while the same request
sent directly to an upstream answered in about 10 ms. A freshly started router worked at once.

## Why

The hang itself is not explained yet. What is known:

- The CUDA backend's waits are unbounded. `Context::sync` and `Event::synchronize` (`crates/backend-cuda/src/exec.rs`)
  call `cuStreamSynchronize` and `cuEventSynchronize` directly; nothing applies `--gpu-wait-s`, which the wgpu and
  Vulkan backends honour (`backend_api::hardware::wait_timeout`). So the device-failure contract
  (`docs/concepts/execution/device-failure.md`: a wait past the ceiling fails the job, frees the lane and quarantines
  the card) never fires on CUDA; the lane thread blocks for good and the lane is never freed.
- `/health` cannot see it. `status` is always `ok`, `free_slots` is `decode_slots` minus `running_sequences` as of the
  dispatcher's last pass, and `queue_depth` is a counter the stuck lane never updates; none of them says whether work
  is moving.

Open hypotheses for the kernel that never completed, none confirmed:

1. A speculative verify step (draft/verify kernels or the mixed verify path) at 1 to 2 running sequences, where `auto`
   switches speculation on and off between steps: the last log lines put the hang there.
2. Any kernel: a launch whose grid or a device-side loop bound came from a stale or out-of-range value (a sequence
   count or KV page index changing between steps), spinning on the device; the 100% utilisation fits a spinning
   kernel better than a lost launch.
3. Not ruled out: a driver or hardware fault on those cards; the 7 hung servers have not been mapped to cards and
   nodes yet.

The router's half is explained (`crates/apiserve/src/route/mod.rs` before the fix):

- A session (`prompt_cache_key`) returned to its upstream while that one had a free slot by the router's own count;
  a restart dropped the upstream's prefix record but not its sessions, and nothing else checked whether the upstream
  ran what it was sent.
- Prefix affinity weighed the upstream's reported queue at one eighth of a prompt per job (`balance_slack` for 64
  slots), so a cached conversation still chose an upstream reporting 5 jobs waiting.
- Nothing bounded a request already sent: `--upstream-timeout-ms` (31 min by default) bounds only the wait for a status
  line, longer than any agent's own timeout, and a stream that has started is not bounded at all. Clients gave up
  after 30 to 60 s, which released the router's slot, so `in_flight` read 0 while every new request of those
  conversations went back into the same queue. A fresh router had no sessions and no prefix record, and the queue
  weight alone sent its requests elsewhere.

## What to do

- Landed in the router: an upstream that reports jobs waiting takes no new request while another with room reports
  none, whatever prefix or session it holds; an upstream holding work that answers, streams and finishes nothing for
  `--upstream-stall-ms` (default 10 min) is stalled: what was sent to it is failed (504, or an error ending a started
  stream), it leaves rotation although its health answers, and it returns when it restarts, finishes something or is
  resumed; a restart drops its sessions with its prefix record
  (`crates/apiserve/tests/it/route.rs`: `an_upstream_that_queues_work_without_running_it_gets_none_while_another_has_room`,
  `an_upstream_that_stops_making_progress_fails_what_it_holds_and_leaves_rotation_until_it_restarts`).
- Open, in the server (`.agents/roadmap/gpu-supervisor.md`, "device watchdog"): a CUDA lane whose synchronisation
  passes `--gpu-wait-s` must fail its jobs, quarantine the card and say so in `/health`, so a router or a load balancer
  takes the server out instead of reading every slot free.
- To find the kernel: on the next hang, before restarting, take `cuda-gdb` or `compute-sanitizer` (or `nsys` with
  NVTX) on the wedged process, or at least record which kernel the lane last launched; a wait bounded by `--gpu-wait-s`
  should name the step it was waiting on.

## Consequence: killing a hung server leaks its device memory

The seven hung servers ignored SIGTERM (the lane thread is inside libcuda, waiting on a kernel that never ends) and
were killed with `kill -9`. On one node three cards then kept 5.4 to 6.4 GB in use with no process attached:
`nvidia-smi` counts the memory, `--query-compute-apps` lists no owner and no process holds `/dev/nvidia*` for it. The
driver cannot tear a context down while one of its kernels is still running on the device, so the context outlives
its process; only a GPU reset (`nvidia-smi -r`, root, with nothing else on the card) frees it. Nothing in brain can
release it, and a device watchdog would not either: a bounded wait fails the job and quarantines the card, but the
kernel keeps running. What the watchdog does buy is a server that says it is wedged instead of one an operator ends
up killing blind.

What follows from it:

- Every later server on such a card sees that much less free memory. A plan sized from the card's total, or one
  whose estimate charges more than its sizing did, is then refused at startup ("UNSERVABLE ... withdrawn before
  ready", knowledge #605); a plan sized from what is free shrinks and serves.
- Operationally: before restarting servers on a node, compare each card's `memory.used` with what its processes
  hold; a card with memory used and no owner needs a reset by an administrator, or its servers must be planned for
  the smaller share (they are, by default, since the budgets are free bytes). Taking `cuda-gdb`/`nsys` evidence of
  the hung kernel before the kill (above) is the only chance to learn which kernel it was.
