<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 527. An exhausted KV pool panicked the lane instead of queueing the sequences it could not back

**Lesson:** Capacity is an admission and back-pressure decision, never a panic. A resource the scheduler's accounting
can get wrong (a pool block count, a prefix it counted on) must fail through the typed path the scheduler already
recovers from, not through `expect`. A panic in an engine call takes the lane, and every request in flight, with it.

## What happened

Qwen3-4B bf16 with a LoRA adapter, `--max-batch 64`, 24 to 72 concurrent agent sequences with contexts to about 30k
tokens: `table.reserve(cc, &mut self.alloc).expect("KV pool exhausted")` in the prefill chunk loop of
`qwen3::serve::Engine` panicked the lane thread (117 times on one card and 174 on another in about three hours; the
default `--max-batch 16` panicked too). Every request in flight then failed with HTTP 500 and the instance was
rebuilt. Dev-eval arms lost most of their trials to it.

The scheduler already had the answer for a full device: a typed `DeviceOutOfMemory` unwind is caught around prefill and
decode, the sequences the step carried go back to the queue, fewer run at once until the card has held more, and only
a request that fails alone is refused with a retryable reason (knowledge 435). A KV pool that cannot back a block was
simply not on that path: six `expect`s in the Qwen3 engine, and the same in the Qwen3.8 and MoE engines.

## What fixed it

`BlockTable::append_or_raise` and `reserve_or_raise` unwind with the typed payload (`what: "the paged KV pool"`, the
allocator's own account as the detail). The Qwen3, Qwen3.8, Qwen3.8 MoE and Qwen3.8 GGUF engines use them. The
regression test is `a_kv_pool_that_cannot_back_an_admitted_prefill_queues_the_request_instead_of_panicking`
(`crates/model/tests/it/serve_device_oom.rs`): admission is told more blocks are free than there are, and every
request still completes with the tokens a roomy pool gives, none refused.

## Not established

Why admission let a prefill in that the pool could not back is not measured. A candidate: a request is charged for
the cached prefix blocks it will adopt (`cached_prefix_blocks`), but the adoption happens at its first prefill chunk,
and another admission's `reclaim_prefix_keeping` can evict that chain in between, so the debt undercounts. Pinning an
admitted request's chain at admission would remove the gap; it needs a decoder-level pin and a card measurement.

An explicit `--max-batch 64` plans 91.9 GB of the card for the 4B instance, and a served training job then leaves it
unservable because an explicit plan is never re-planned (only a defaulted one is, by design). Making the KV pool yield
to a co-tenant job is open.
