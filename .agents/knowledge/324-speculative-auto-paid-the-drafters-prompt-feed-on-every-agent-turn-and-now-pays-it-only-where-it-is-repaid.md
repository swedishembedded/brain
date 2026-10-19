<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 324. `--speculative auto` paid the drafter's prompt feed on every agent turn; it now pays it only where speculation repays it

Question: why does the default (`--speculative auto`, MTP head loaded) lose a third of the wall time on
a multi-turn agent workload against `--speculative off`, and can the default be made right instead of
asking operators to switch it off? All figures **VERIFIED-BY-RUN**, one 27B Q8_0 server on one GH200
(A2.6 flags: ctx 32768, int8 KV, max batch 16, 16 prefix slots), replayed with
`agentic_replay.py` (16 or 48 conversations of 6 turns: 4,000 tokens of source text, then each reply
plus 1,200 new tokens, at most 48 new tokens, greedy, concurrency 8, about 600k counted prompt
tokens per 16 conversations). The card was shared with other lanes' short jobs, which moves a single
run by 10-20 percent; both settings were run back to back.

Cause: with a drafter loaded, every prompt is fed to the MTP head in 128-row rounds, each round with a
device sync and host work (`mtp_after_prefill`), and the hidden rows of every prefill slice are read
back for it. On an agent turn (long prompt, a reply of tens of tokens) speculation saves a few
milliseconds of decode and the feed costs seconds.

| setting | 16 conversations | 48 conversations |
|---|---:|---:|
| off | 110.4 s, 112.1 s | 361.1 s |
| auto, before (feed on every prompt) | 175.7 s | not run |
| auto, with the gate | 152.0 s | 404.6 s |

The gate (`model::serve::speculate::PrefillGate`): the decoder reports the measured per-sequence feed
time (`spec_prefill_cost`), the scheduler weighs it against the cost gate's measured per-token saving of
speculating times the running mean reply length, and prompts for which speculation cannot repay the feed
are not fed (`set_spec_prefill(false)`); one prompt in sixteen is fed anyway so the saving stays
measured. A skipped prompt's prefix-cache entry is marked `mtp_valid = false`, so a prompt that resumes
from it does not copy head rows that were never filled. The 33 percent penalty of auto against off
becomes 12 percent; the remainder was not attributed (the learning phase feeds the first wave of long
prompts; per-step MTP bookkeeping and speculative probes remain).

Not measured: the decode-heavy case where auto should win (the replay's replies are one sentence, so
`--max-tokens 400` changes nothing: 4,598 completion tokens at 400 and 4,160 at 48), so the claim that
auto keeps its decode gain at low concurrency rests on the unit test (`serve_speculative.rs`, a long
reply with a cheap feed keeps the feed) and on knowledge 318's decode numbers.
