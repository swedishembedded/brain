<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 333. A streamed response ended with the longest request of its group, and one checkpoint was served by two residents

Symptom (the CE4 audit server, 8B, adapter in the bank, 10 streaming clients): requests needing 2-40 completion tokens all
finished at the same instant as one 3937-token adapter decode (818-824 s), and an 8-token request got no response headers
for 120 s. The agent clients' 600 s first-byte bound fired. The first reading was "no continuous batching"; the scheduler
was innocent (`model::serve::Scheduler` admits every step; the generic conformance suite
`model::serve::conformance` pins that). Three causes, found in this order with a 1.7B probe and then the 8B with its adapter
(`runs/b8/lockstep.py`, `lc.py`):

1. A streamed response ends when its channel closes, and the channel's sender is the job's progress sink. The open group
   (`residency` `GroupFeed`) kept every job's sink until the group ended, so every stream, finished or not, ended only when
   the longest request of the group did. Non-streamed requests were unaffected, which is why the non-stream replays never
   showed it. Fix: `GroupFeed::finish` drops the sink.
2. `brain serve --qwen-weights <dir> --adapter ...` registered the same checkpoint as two residents: `brain/qwen3` (what the
   adapter flags attach to) and the model-dir entry (`Qwen/Qwen3-8B`, what clients name). Two residents are two instances: a
   second copy of the weights (+38 GB, so a server that reserved 55 GB held 76 GB), a 13.8 s build at the first adapter request,
   and base and adapter requests that never share a batch; a running group also stops taking late jobs while a job of another
   instance waits, so a short base request queued behind a long adapter decode. Fix: the model-dir resident keeps the checkpoint,
   `brain/qwen3` is a `residency::Alias` of it, and the adapter flags attach to it. Check: after the fix the card holds 38.3 GB
   from load to the end and no `promote ... (building)` line appears.
3. The KV pool was two contexts (65 k tokens at ctx 32768), so more than 4 long conversations could not stay cached: above 4
   clients every agent turn re-prefilled its whole history (1 % hits at 32 clients). The default pool is now `max_batch`
   half-context histories, halved until it fits the card (`--qwen-kv-tokens` overrides). A model-dir resident had no card budget
   to size it against (only the env resident did); it has now.

Measured effect on the 8B agentic replay (streaming, 6 turns, 6000+1500 token histories), default flags: 8 / 16 / 32 clients
wall 45 / 124 / 255 s to 25 / 42 / 78 s, request p90 14 / 34 / 68 s to 6 / 11 / 23 s, cache hits 42 / 8 / 1 % to 77 % (the ideal
for the replay is 75 %). Table: `runs/b8/profile-prefix-cache.md`.

Traps: (a) the `request done` log line prints the cumulative cache counters only when the instance lock is free (metrics refresh
`try_lock`s the busy instance), so read them from a request sent after the load; (b) a plain LRU prefix cache evicts the head of
a chain first (stranding the rest) and, with as many conversations as the pool holds, evicts exactly the conversation whose next
turn is nearest; eviction is now leaf-first and admission spares the chain it is about to adopt; (c) the card helper's turn lock
is held by a queued job while it waits for shares, so one stale reservation (a server still draining after SIGTERM) idles the
card behind it: `brain serve` now exits within 30 s of the first signal even if its surfaces do not drain.
