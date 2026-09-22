<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 24. A benchmark that measures a path no request takes is a healthy number about the wrong thing

Every serving benchmark before `perf::targets::HttpTarget` drove
`qwen3::serve::Scheduler`/`residency::Executor` directly - real kernels, real
batching, genuinely fast. Meanwhile `crates/cli/src/resident_llm.rs`, the
ONLY code an actual `/v1/chat/completions` request ever reaches, called a
single-sequence decode loop that touched none of it: no paged KV, no
scheduler, no batching. The benchmark suite was green and fast while a real
agentic client saw 600+ seconds, because "the engine is fast" and "the
request reaches the engine" are two different claims, and only the second
one is what a user experiences. Nothing forced the benchmark to prove the
second claim - it was structurally impossible for it to be wrong about the
first while being catastrophically wrong about the second, and it stayed
that way for as long as no target actually drove the transport layer.

The fix generalizes past this one bug: a target that measures a scheduler,
an engine, or a codec directly is answering "is the fast path fast", never
"does a request reach the fast path" - those need a DIFFERENT harness that
goes in through the same door a client does (here, `apiserve::router()` via
`tower::Service::oneshot`, no socket, but every layer a real HTTP request
passes through: auth, admission, JSON parsing, chat-template rendering).
Keep both kinds of target - the direct one is cheaper and still useful for
kernel-level regressions - but never let the direct one stand in for "is the
served path fast," because the gap between them is exactly where a
serving-path regression like this one hides.
