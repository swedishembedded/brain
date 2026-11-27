<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 543. A cached prefix every request shares pins every request to one wide replica

**Lesson:** "the longest cached prefix among replicas with a free slot" is only a balance rule while slots are few.
A served 4B model advertises hundreds of decode slots, so its replica is never full, and a system prompt every request
shares is a useful prefix on whichever replica served it first: every later request followed it there. Measured
2026-11-21 through `brain serve --upstream` over four Qwen3-4B servers (one GH200 each, 263 decode slots each), 512
single-turn requests with one shared ~1400-token system prompt at 256 in flight: 944 completion tok/s through the
router against 4787 for client round-robin, one card doing all the work.

**How to apply:** a prefix hit may cost balance only up to a bound. `residency::affinity::choose_replica` takes the
hit only among free replicas within `balance_slack(slots)` requests (an eighth of the slots, at least 8) of the
least-loaded share; past it load decides, and the next replica records the prefix too. With it the same workload ran
at 4907 tok/s (round-robin 4978) while multi-turn conversations kept their 0.82 hit rate. Any affinity policy keyed
on cache state needs a load bound that scales with the replica's width, not a "full" test.

Superseded in its bound by #607: a bound in slots fails when the slots are an admission figure, not a decode
rate; the bound now follows the lightest replica's load.
