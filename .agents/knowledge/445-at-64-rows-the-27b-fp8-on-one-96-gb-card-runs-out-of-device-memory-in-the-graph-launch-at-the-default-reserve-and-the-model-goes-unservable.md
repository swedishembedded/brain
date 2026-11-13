<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 445. At 64 rows the 27B FP8 on one 96 GB card runs out of device memory in the graph launch at the default reserve, and the model goes unservable

**Lesson:** The scheduler's answer to a refused allocation (requeue, halve the rows in flight) works for the allocations
it sees; an out-of-memory from `cuGraphLaunch` is not one of them and ends in 500s and an `UNSERVABLE` model until the
server restarts.

## What happened

`brain serve --max-batch 64 --ctx 1024` of Qwen/Qwen3.8-27B-FP8 on one GH200 (resident 83 GB, default `--reserve-gb 2`),
64 concurrent requests: the scheduler logged `the device ran out of memory (100663296 bytes refused); 61 sequence(s) go
back to the queue and at most 30 run at once`, halved again to 15 and 7, and then every request failed with `launching
graph 3 of a frozen program failed after the first was queued: cuGraphLaunch failed: out of memory`; the residency manager
then reported the model `UNSERVABLE ... does not fit the memory this server can use right now`, and every later request
was a 500 until the process was restarted. With `--reserve-gb 16` the same load (64 rows, base and adapters) ran.

## What to do

A graph launch that cannot get its memory should be the same typed refusal the allocator's is, so the scheduler backs off
instead of failing the request and condemning the model; and a server asked for `--max-batch 64` should size the activation
reserve for it instead of leaving it at 2 GB. Neither is done here (the typed device out-of-memory is the numerics lane's).

Also seen on the way: `--serve-adapters` offered its adapters on the watcher's first tick, after the ready file was
written, so a client that started on readiness got a 404 for an adapter on disk; the first pass now runs before any
surface binds.
