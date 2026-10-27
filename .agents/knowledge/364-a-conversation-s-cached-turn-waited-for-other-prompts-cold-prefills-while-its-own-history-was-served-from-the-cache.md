<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 364. A conversation's cached turn waited for other prompts' cold prefills while its own history was served from the cache

Symptom: agents driving the served 27B over D-Bus saw time-to-first-token of ~27 s for a 14k-token prompt and
`cached_tokens` of 0 on every request, and concluded that each turn re-prefilled its whole history.

Cause, measured: the cache was working and nothing reported it. No outcome carried the cached count (the
client's 0 was an absent field), and the delay was queueing. Ten conversations of 4.8k-61k tokens, three turns
each, thinking on, one card, `--max-batch 32`: turn 2 and 3 requests were served from the cache
(`cached_tokens` = prompt minus the 16-token tail and the 64-token chunk alignment, 3520 of 3578 up to 61632 of
61689), yet turn 2 had a median time-to-first-token of 13 s and a maximum of 50 s. A prompt is read in whole,
oldest first, by one background task that cannot be interrupted, so a request that needs 60 tokens of prefill
waits for the cold prompts admitted before it (the same 10 cold prompts took 85 s together, ~2.2k prompt
tok/s). Turn 3, with no cold prompt left in front, took 0.2-1.2 s.

Tried and removed: starting the prompt that needs the least computation first (admission order by the prompt
less its cached prefix, with a bound on how often a prompt may be passed). It cut the median of turn 1 from 70 s
to 37 s and of turn 2 from 13 s to 4 s but moved the large cold prompts behind the later turns, so turn 3 of
other conversations then waited 60-70 s behind them: the mean time-to-first-token over the 30 requests was about
the same (28 s against 29 s). Ordering cannot beat a prefill that runs 25 s without a break.

Fix: the background path reads a prompt in slices of at most 4096 tokens (a multiple of the prefill round and of the
recurrent chunk, so the rounds are the unsliced prefill's: greedy tokens of a 4.8k cold prompt identical sliced and
unsliced) and, between slices, starts the oldest prompt that fits one slice before the oldest long one (a long one
gets a slice after 8 short ones in a row). The slice's recurrent state is already stored in the sequence's pool row
when the slice ends; a later slice reloads it into the working copy. Same ten-conversation run: turn 2 median
13.2 s to 1.4 s (maximum 49.8 s to 4.4 s), turn 3 maximum 1.2 s to 2.5 s; the cold turn 1 finished at 94 s against
85 s (the large cold prompts now share the card with the turns that arrive meanwhile).

Reproducibility, same finding: a seeded request is sampled from its own RNG (`Running::rng`), but its logits
depend on the co-batch (knowledge #358), so a seed reproduces a decode only with `--exact-decode`.
