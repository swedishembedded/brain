<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 82. "Two flags request the same capability" is a claim about KERNELS, and it is only true if BOTH kernels actually call the intrinsic

A call-site map claimed `qwen3::serve::Engine`'s `kv_int8` was `weights_
int8`/`w8_on`'s ungated twin: both flags pick a packed-int8 tier, `w8_on`
correctly gates on `caps.numeric.int8_dot` before dispatching a packed
GEMM, `kv_int8` gated on nothing before dispatching `paged_decode_scores_
i8_batched`/`paged_decode_apply_i8_batched` - so, by the pattern, it looked
ungated and wrong. It was implemented that way (`kv8_on = kv_int8 &&
caps.numeric.int8_dot`) before anyone opened the three kernels' actual
WGSL source. None of them - `paged_decode_scores_i8_batched`,
`paged_decode_apply_i8_batched`, `paged_kv_append_i8_clipped_batched` -
call `dot4I8Packed` anywhere; all three dequantize/pack a byte with plain
scalar shift-and-mask WGSL and are `@cpu yes, @gpu yes` in the catalogue,
exactly as portable as an ordinary fp32 kernel. `weights_int8`/`w8_on`
gates because `matmul_i8_dyn`/`matmul_i8_gemv*` genuinely DO call
`dot4I8Packed`. The two flags share a name ("packed int8") and a shape
("one gated, one not"), but not the physical requirement the shape implies
- `kv_int8`'s original, ungated code was already correct.

The "fix" was a real regression, not a no-op: on `backend-cpu` (`int8_dot:
false`), a `kv_int8: true` request now silently degraded to fp32 KV even
though the int8 KV kernels run correctly there. It was caught immediately
by re-running the existing `BRAIN_DEVICE=cpu … --lib serve::` suite - 5
tests failed - which is exactly the gate this class of change should
always be checked against BEFORE trusting a pattern-matched diagnosis, not
after.

**Rule going forward**: "flag A is gated on capability X, flag B looks
structurally identical and requests the same tier, therefore B needs the
same gate" is a hypothesis, not a finding - `kernels.md` §B's "read the
contract before dispatching" applies just as much to READING an existing,
working dispatch as to writing a new one. Before gating (or ungating) a
flag to match a sibling, grep the KERNELS each one actually dispatches for
the specific intrinsic/feature in question (`dot4I8Packed`, a subgroup op,
an atomic) and check their `@cpu`/`@gpu` header - two kernels performing
"the same operation" on "the same dtype" can still have completely
different physical requirements, and only the kernel source settles which
one a given flag actually needs.
