<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 39. A `write_*` call with no matching read on that code path is a silent no-op, not a bug the type system catches

`qwenvl::Qwen3Vl::generate()` (KV-cache generation) called
`self.decoder.write_deepstack(level, &data)` during prefill - a real
function, on a real buffer, compiling cleanly, doing exactly what its name
says. The bug: `qwen3::Qwen`'s incremental decode path
(`decode_steps`/`step`/`step_mrope`) never READS `deepstack_bufs` - the one
dispatch that adds them into the residual (`SPLICE_ADD`) lives only inside
`forward_steps()`, the BATCHED training-graph builder, which `generate()`
never calls. So the write happened, the data sat in a real GPU buffer,
correctly, and was never consulted by anything - for any checkpoint enabling
DeepStack (`VisionConfig::deepstack_indexes` non-empty, which a real
production config is), every generated token was silently missing a real
architectural contribution. No panic, no wrong-shape error, no type error:
`write_deepstack` and `decode_steps` are both individually correct functions
that simply never talk to each other on this path. The existing test
(`generate_is_deterministic_and_respects_eos`) passed throughout, because it
only checks determinism and EOS-stopping, never a numerical value that would
reveal a missing contribution - the same shape of gap this file's own
"gates that lie" pattern describes, just for a hand-written test instead of
an automated gate.

Found by asking a narrow, specific question while wiring an UNRELATED
follow-up: "does the fast (incremental) path apply the same architectural
pieces as the slow (batched) reference path?" - not by a test failing. A
targeted grep across the file in question is what actually answered it
(every reference was setup or the one batched-only consumption site). **The
general shape**: when a model has TWO forward implementations for the same
architecture (a batched training/reference path and an incremental/decode
fast path - extremely common for any KV-cache-capable model), a feature added
to one must be independently verified to exist in the other, not assumed
from "the setter function got called." A parity test between the two paths
(`deepstack_step_matches_full_recompute`) is the gate that would have caught
this before it shipped, not after.
