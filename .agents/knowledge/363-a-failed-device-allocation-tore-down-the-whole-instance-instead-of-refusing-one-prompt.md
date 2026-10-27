<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 363. A failed device allocation tore down the whole instance instead of refusing one prompt

Symptom: one long prompt that ran a card out of memory (`cuMemAllocFromPoolAsync
failed: out of memory`) panicked the lane. The executor caught the panic, but it
answered every request the instance held with an unclassed
`panicked while running` error and rebuilt the model from disk (tens of seconds
for a 27B), so sequences that were decoding fine were lost with it.

Cause: the CUDA backend reported a full card as a formatted panic string, which no
layer could tell from a bug, so the only recovery point was the lane.

Fix: the backend unwinds with a typed payload (`gpu_core::DeviceOutOfMemory`,
no panic message). The generic scheduler catches it around the calls that read a
prompt in, asks the decoder to drop its prefill work (`PagedDecoder::
abandon_prefill`), returns the prefilling sequences' blocks and refuses them with
`RejectReason::DeviceOutOfMemory`, which a frontend delivers as a
`backend-failure` classed error. Decoding sequences, the prefix cache and the
instance stay. `crates/model/tests/serve_device_oom.rs` is the contract.

Limits: an allocation failing inside a decode or verify step is still a panic
(the step's tensors are half written, and every sequence in it is affected); the
planner's decode scratch charge ([#362]) is what keeps that unreachable at the
defaults. Any other panic keeps unwinding to the lane.

Rule: a resource-exhaustion failure belongs to the request that caused it; give it
a type at the point it is detected so the layer that owns the request can refuse
it, and do not let it travel as a string.
