<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 321. A request the engine would refuse waited for a free slot to be told so, and a streamed refusal was a 200

Symptom (a CE1 solve, 27B GGUF server, max batch 16): a request whose prompt plus `max_tokens`
exceeded the engine's context (`qwen35: needs 33805 tokens, engine capacity is 32768`) was logged
by the server but its client never saw response headers and gave up at its own 600 s timeout.

Two defects, both in how a refusal is delivered, not in the refusal itself:

1. **The refusal was decided at admission, and admission needs a slot.** A running group takes
   waiting jobs only while it has free slots (`JobFeed::take(capacity - running)`), and the
   scheduler tested capacity, vocabulary and adapter only for the request it was about to admit.
   With every slot busy on long generations, an impossible request sat in the dispatcher queue
   (no response headers: the streaming path answers only after admission) until a slot freed,
   however long that took. Refusing a request needs no slot.
2. **A streamed refusal arrived as `200 text/event-stream` with one `error` event**, because the
   headers are committed before the first event. A client that reads `data:` frames and waits for
   `[DONE]` (or for a status) does not recognise that as a failure.

Fix: a running group now holds up to 32 waiting jobs beyond its free slots in its inbox and asks
the instance about each once (`JobFeed::screen`). `llmserve`'s group asks the scheduler
(`Scheduler::refusal`: the same test admission applies) plus the frontend's route and parse (an
unknown adapter, a template or tokenizer error) and answers a request that can never run at once,
with its admission signal fired. A streamed request is given 300 ms to fail before its headers go
out (`EventStream::settle_start`): a refusal within that window is the HTTP status an unstreamed
request would get (400 `context_length_exceeded` for length), anything else streams as before.

Evidence: `crates/residency` `a_job_the_full_group_cannot_serve_is_refused_without_waiting_for_a_slot`
(a 600 ms running job, the refusal answered within 300 ms), `crates/catalog/tests/scheduled_batch.rs`
`a_request_that_cannot_run_is_refused_while_the_group_is_full` (real tiny engine: the refusal is
the first event, before the running request finishes), `crates/apiserve/tests/api.rs`
`an_engine_refusal_is_an_immediate_http_error_streamed_or_not` (four engine refusal wordings,
streamed and not: HTTP 400, OpenAI error body, no internal prefix; red with the settle step removed:
the streamed body is not JSON) and the socket-level `tests/e2e/api_conformance.bats` over-context case.

Still true: a request that fits waits for its slot (queue wait is bounded by `--admit-deadline-ms`),
and a failure after the first token is an `error` event in a 200 stream; `max_tokens` is not clamped
to the remaining context (a client that wants the maximum reads the model's `context_length`).
