<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 435. A full card failed admitted requests with a 500 because activation scratch lives outside the KV pool, and the scheduler refused them instead of sending the sequences back

Question (production report, one GH200, `--max-batch 48 --speculative off`, ~100 concurrent agent conversations of 3k-60k tokens, prefix cache on, binary 0c6503dcf): 90 of ~1500 requests failed with HTTP 500 `server error (backend_failure): qwen35: the device ran out of memory (139890688 bytes refused) while the prompt was read in`, beside 37 x 504 and stalls. Why does a server that admits by free KV blocks run out of memory, and why is that a 500?

## Finding

- **The refused allocation is not KV.** 139890688 bytes is exactly 8036 x 17408, and 17408 is the 27B's `intermediate_size`: one feed-forward intermediate of a round of 4018 bf16 rows (8036 x 1 byte for an 8-bit tier). That is activation scratch of a prefill or mixed round, taken from the stream-ordered device allocator, not a block of the pool (`ServingShape::stage_workspace`, `qwen35/src/int8_gguf_resident.rs:467`, is what the planner keeps free for it). The pool's block count cannot see it. Which of the planner's margins was short on the card (the 5/2 live-layers factor of `prefill_workspace_bytes`, fragmentation of the block cache, the decode scratch running beside the round on another stream) was not measured here: no GPU was available. The fix does not depend on it.
- **KV was already covered.** `Scheduler::admit_one` (`model/src/serve.rs`) charges prompt blocks + first token + state row + everything in flight is owed, evicts cached prefix blocks before it waits, and `ensure_decode_room` preempts (demote to host, else recompute) when decode outgrows the pool. Reserving `prompt + max_new` for every sequence was rejected there on purpose (a ceiling is not a forecast).
- **The 500 was the scheduler's answer to the failure it could not foresee.** `catch_device_oom` caught the typed `DeviceOutOfMemory` (#363) and `refuse_prefilling` / the decode and mixed arms refused every sequence the call carried with `RejectReason::DeviceOutOfMemory`, which `llmserve::Frontend::rejection` classed `backend-failure` (500, not retryable). One failed round therefore killed every conversation in it, although each of them would have fitted on a card carrying fewer sequences. With 48 conversations in flight, each failure took up to 48 requests.
- **The 504s and stalls are the same load seen from outside**: requests beyond `--max-batch` wait in the executor inbox under `--admit-deadline-ms` (429 when it passes, #bridge `wait_for_admission`); once a job is taken into the scheduler there is no deadline, and the client's own timeout ends the wait. Not changed: the wait of an admitted request is bounded by the progress of the sequences ahead of it, which recompute preemption keeps moving oldest first.

## Fix (generic, `model::serve::Scheduler`)

- A device allocation failure sends the sequences it carried back to the head of the queue (decoding ones by recompute with the tokens they produced and their sampler stream, prefilling ones from their prompt), oldest first; the decoder drops its prefill work (`abandon_prefill`). Applies to every path: sequential, batched and background prefill, plain decode, speculative verify, mixed rounds.
- `MemoryPressure`: the failure halves the number of sequences admitted at once (never below one); every 32 iterations with work in flight and no failure give one sequence back. A growth the card cannot hold costs one recompute per 32 iterations at most. Admission (`admit_one`) enforces the limit, so queued requests wait for the sequences ahead to finish and complete once memory frees.
- A request is refused only when it is the only sequence in the scheduler and has failed three times; the prefix cache is given back (`reclaim_prefix(u32::MAX)`) before the last attempt. That refusal is `FailureClass::Unavailable` (`unavailable`, HTTP 503 with `Retry-After`, Anthropic `overloaded_error`), no longer a 500.
- Every failure writes one warning line naming the bytes refused and the new limit, so the card's real capacity shows in the log.

## Tests

`crates/model/tests/it/serve_device_oom.rs`: a burst that overfills the card completes with the tokens a roomy card gives; a decode step that fails loses no sequence (tokens equal the uninterrupted ones); a prompt that cannot fit alone is refused after the cache was given back, in at most four failures, and does not take its neighbours down; the batch grows back; 100 long requests over a pool of about eight prompts and a card of five complete with no refusal and at most 40 failures. `llmserve` `rejection` and `apiserve` `bridge` assert the class and the 503 with `Retry-After`.

## Rules

- Admission by pool blocks is a model of the card, not the card. Anything an engine allocates outside the pool (scratch, state, graphs) can fail, so a typed device failure must be a scheduling signal (retry with less in flight), not a request outcome.
- A refusal that a client can fix by waiting is never a 500: give it a retryable class at the layer that knows (`Unavailable`).
- Open, needs the card: measure which planner margin was short at 8036 rows with 48 sequences decoding (nsys of the allocator, `stage_workspace` against the observed peak) and whether a mixed round's rows can be capped under pressure instead of only the batch.
