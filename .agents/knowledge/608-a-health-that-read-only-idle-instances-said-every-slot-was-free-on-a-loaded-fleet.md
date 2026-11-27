<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 608. A health that read only idle instances said every slot was free on a loaded fleet

**Lesson:** a metric read with `try_lock` from an object that is locked while it works is a metric of the idle
object only. Observed 2026-11-27 on 16 GH200 servers behind four routers: every `/health` sample on every upstream
(30 samples each, 14 min) read `free_slots` 3466 (the nominal sum of every model's batch) and `kv_free_tokens` null,
with `queue_depth` up to 148 and 244 requests running on one card. The router's KV penalty and its idle test
(`free_slots >= decode_slots`) were therefore dead, and its view of each card was "empty".

## Why

`ResidencyManager::all_metrics` asked each instance for its metrics with `try_lock` so the dispatcher never blocks on
a lane, and skipped any instance it could not lock. A continuous-batching group (`Instance::run_open`) holds its
instance for as long as anything runs on it, which under load is always, so the instances that were decoding were
exactly the ones never read. `health.rs` then summed `running_sequences` over nothing (`unwrap_or(0)`), and the KV
sums were `None`.

## Fix

A running group publishes its metrics through its feed (`JobFeed::publish`) after a scheduler pass, at most every
`METRICS_PUBLISH_INTERVAL` (250 ms): `run_open_scheduled` does it for every scheduler-backed text model, and a
supervised worker forwards it as a `metrics` frame (worker protocol 6). `all_metrics` keeps each instance's last
reading and, for an instance it cannot lock, overlays what its running group last published. Spec:
`crates/apiserve/tests/it/api.rs::health_counts_the_sequences_a_busy_model_decodes` (red before: free_slots 8 of 8
and kv null with three sequences running), `crates/residency/tests/hosted/linux.rs::an_open_group_takes_late_joiners_and_finishes_each_job_through_the_feed`.

**How to apply:** any reading taken "when the thing is free" needs a path for when it is not; ask of every
health field what it says under full load, and test it under load.
