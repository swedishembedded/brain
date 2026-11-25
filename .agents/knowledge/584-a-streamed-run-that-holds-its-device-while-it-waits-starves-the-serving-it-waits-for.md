<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 584. A streamed run that holds its device while it waits starves the serving it waits for

**Lesson:** a workload (`ResidentModel::transient`) held its device lane for its whole run. A streaming
policy-gradient job is a workload whose input is drawn from the policy the same server serves: on a server
whose policy and job share a device, the job waited in `Objective::supply` for records that could only be
produced by requests queued behind it on that same lane. Nothing failed; every rollout request simply went
unanswered until its deadline, so the orchestrator scored empty rollouts (no tokens, no trace) and the run never
trained. The fix is generic, not the stream's: a running workload lends its device while it waits
(`capability::lease::DeviceLease::lent`, armed by the executor on every workload job). Each device has a serving
lane and a workload lane; a lent device is not `busy`, so serving is admitted there on the serving lane; when the
workload asks for it back, nothing new starts there and running serving groups there take no late jobs, and the
device returns once they finish. One group still computes on a device at a time. The stream objective lends at
every step boundary (`RolloutFeed`), so a request waiting for the device gets its turn even while records are
queued.

## Evidence (2026-11-25, lane agrpo-e2e2)

On the real stack on the CPU (`brain serve --openai --training`, a toy qwen3 base, a `train.online` stage), a chat
request to `local/toy` sent while the stream job ran got no answer within 60 s, and every rollout record came back
with 0 tokens, `exact: false`, span [0, 0]. `residency` `it::admission::a_workload_that_lends_its_device_while_it_waits_lets_serving_run_and_takes_it_back_after`
(one CPU device: a request served while the workload waits; a request running when it asks back keeps the device
until it ends) failed before the change (the job ran with its lease unarmed, so it could not lend; the test asserts
an armed lease first). `trainjob` `it::stream::a_stream_job_trains_groups_as_they_arrive_and_serves_each_version_it_publishes`
now asks the published adapter for a reply before the first group (version 1) and after the first step (version
2); with the lease left unarmed the test did not finish within 400 s.

Not exercised: a GPU. A serving group and a workload never compute on one device at once, but they now run on two
threads, so a backend with per-thread device state would see the workload's instance only on the workload lane and
serving instances only on the serving lane.
