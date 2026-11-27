<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 612. A stall bound longer than the client's patience saves nobody

**Lesson:** a failure detector's bound is set by who is waiting for its verdict, not only by the longest legitimate
silence. Observed 2026-11-27: the router's default stall bound was 600 s and the agent runtime in front of it gave up
on a model call after 480 s. Requests held by a card that made no progress (41 on a training card at 0 tok/s, 64 on
a card rebuilding its model) ended with the client's timeout, all 105 of one adapter batch among them, and one of
three routers had still not stalled the card ten minutes on. A detected stall fails its requests with 504 while
their clients can still send them elsewhere; one detected after the clients left only takes the card out of
rotation.

**How to apply:** set a stall bound below the shortest client timeout you serve, with time left to retry
(`route::DEFAULT_STALL_TIMEOUT` is now 300 s: with one probe it leaves a 480 s step a third of its time), and above
the longest silence a healthy upstream shows (a lone unstreamed generation; any finished or streamed request counts
as progress). Spec: `crates/apiserve/tests/it/route.rs::the_default_stall_bound_fires_while_an_agent_step_still_waits`.
