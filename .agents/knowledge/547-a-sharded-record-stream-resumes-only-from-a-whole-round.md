<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 547. A sharded record stream resumes exactly only if every draw ends on a whole round of `world` records

**Lesson:** data-parallel ranks share one record stream and rank `r` keeps record `n` when `n % world == r`. A rank
that stopped drawing as soon as it had its own record stood at a different stream position than its peers between
steps (rank 0 after records 0..4, rank 1 after 0..5 at world 2, batch 1, three steps), and the resumed run restarted
the record counter at 0, so the ranks' shard assignment shifted. A two-rank GPT run stopped after 3 of 6 steps and
resumed ended 0.22 (absolute) away from the uninterrupted run in one tensor; one rank resumed bit for bit. Found
2026-11-21 by `model::conformance_zero::check_sharded_training`, whose replicated case is the reference the sharded
cases are held to.

**How to apply:** a sharded draw consumes the rest of its round (`drawn_records` a multiple of `world` after every
draw), and `Objective::begin` positions a sharded stream at `world` times the rank's own draws. Any new objective that
shards a stream keeps both rules; test resume with more than one rank, never only with one.
