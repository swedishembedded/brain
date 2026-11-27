<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 607. A balance bound in advertised slots let cached prompts hold a batch on four cards while eight idled

**Lesson:** a server's `decode_slots` is what it admits, not what its card decodes at speed. #543 bounded the load a
cached prompt may cost at an eighth of the slots; on servers advertising 3466 slots that is 433 requests. Observed
2026-11-27 through the router over 12 GH200 servers, every model resident everywhere, agent conversations with
`prompt_cache_key` sessions and long shared prefixes (hit rate 0.95): in flight [60, 83, 85, 86, 0 x 8], the four
loaded cards at 98% and eight at 0%. The session rule "back to its upstream while it has a free slot" was the same
trap: those slots never fill.

**How to apply:** `residency::affinity::balance_slack` is now a function of the load, not the slots: a quarter of
the lightest active replica's load (as a share of slots, so unequal widths still balance), at least
`MIN_BALANCE_SLACK` = 8 requests. Against an idle card a cached prompt is worth 8 requests of load (it saves one
prefill; the idle card serves every request after it); under an even load a quarter's imbalance is the most a hit
buys, so balanced conversations stay put and a moved one does not swing back. The router weighs a session's upstream
as holding the whole prompt and lets `choose_replica` decide, so sessions follow the same bound. Specs:
`crates/residency/src/affinity.rs::a_prefix_every_request_shares_does_not_pin_a_wide_replica_while_others_idle`,
`crates/apiserve/tests/it/route.rs::a_batch_on_one_upstream_spreads_onto_idle_ones_whatever_prefix_or_session_it_holds`.
Any load bound read from an advertised capacity needs the same check: is that number what the device does at speed?
