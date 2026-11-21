<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 544. A router's listener cap below its fleet plus its queue turns overload into resets

**Lesson:** the HTTP listener closes a connection over its cap at accept, before reading a byte (the right defence for
one server). A router fronting a fleet holds a client connection for every request its upstreams run and every
request it queues, so the serving default of 1024 connections sits below what four 4B servers alone run (4 x 263
decode slots). Measured 2026-11-21: 3000 streamed requests at 1500 in flight through `brain serve --upstream` over
four Qwen3-4B servers ended with 1976 client connections reset (broken pipe), not one 503, while the fleet had room
queued behind it; client round-robin straight to the servers completed all 3000.

**How to apply:** size a router's connection cap from the fleet, not from the single-server default:
`apiserve::route::serve` raises it to twice the upstreams' decode slots plus the queue bound, so overload reaches the
router's admission and is answered with 503 and `Retry-After`. After the fix the same run completed all 3000 with the
router holding 1052 in flight and about 450 queued. Any proxy in front of a pool needs its accept limit above the
pool's concurrency plus its own queue.
