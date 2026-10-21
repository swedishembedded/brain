<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 347. The served 27B prefix cache never hit across turns of a thinking conversation, so every turn prefilled its whole history

Symptom: the teacher server (27B Q8_0 GGUF, int8 KV, batch 8) delivered 58-70 completion tok/s for 8 concurrent agents
against 290 tok/s for a synthetic batch of 8, with 1,580 counted prompt tok/s (63 percent of its ~2,500 tok/s prefill
capacity) and 14.7k-token prompts. In a 20 s nsys window of a replay of recorded agent trajectories, prefill kernels were 54
percent of device time and decode 41; single-stream decode (14 ms per token, 90 percent device busy inside CUDA graphs) was
not the limit.

Cause: a cache entry was the whole previous prompt, and a Gated-DeltaNet layer resumes only from the state after exactly the
cached tokens. The chat template ends a prompt with the generation prompt (`<think>` and a newline with thinking on) and
re-renders that position in the next turn as the finished assistant message (`<think>` and a blank line ...), so the last
tokens differ and the exact-token match failed. Probe (4.8k tokens, turn 2 = turn 1 + reply + new user message): thinking
off, time to first token 0.16 s (hit); thinking on, 2.09 s, the same as turn 1, on the old binary, on main, and with
`--speculative auto`. Only an exact repeat hit.

Fix: the prefill stops `CHECKPOINT_TAIL` (16) tokens short of the prompt, files the state there, and runs the rest; the entry
is the checkpoint, so the next turn resumes from it and recomputes the tail and its new tokens. Entries of prompts resumed
from a checkpoint of their own length are kept as they are. The checkpoint's MTP head rows are not filled, so a prompt
resuming from it runs without them (speculation is off on the teacher; with `--speculative auto` this should be revisited).
Tests: `a_prompt_diverging_in_its_last_tokens_resumes_from_the_cached_prompts_checkpoint` (hit between 48 and 68 of a
72-token prompt, output equal to a cold prefill's); the paged-block and recorded-token tests now name the 56-token checkpoint.

Measured (card 1, same binary except the fix, recorded trajectories from the CE8 dataset, concurrency 8, thinking on, 24
records x 10 turns, 788k counted prompt tokens): wall 549 s to 312 s, completion 102 to 182 tok/s, time to first token
median 2.0 to 0.5 s, per-stream decode 15 to 26 tok/s. The 4.8k probe: turn 2 2.09 s to 0.58 s.

Not causes (measured): host launch rate in decode (5 graph replays and ~8 plain launches per token), sampling (temperature
0 against 0.6: 111 against 107 tok/s), the 7.8 GB neighbour process. Still open (profile in runs, ranked): decode attention at
~5 percent of HBM bandwidth (122 us per layer at 12.8k context), the int8 prefill GEMM at ~14 percent of tensor peak
(mma.sync, 283 TOPS), 23 percent device idle in the c8 window.
