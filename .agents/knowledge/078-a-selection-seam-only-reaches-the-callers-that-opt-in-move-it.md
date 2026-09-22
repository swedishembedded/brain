<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 78. A selection seam only reaches the callers that opt in - move it INTO the shared builder and one line per model reaches all of them, including the norms the model cannot see

Lesson #76 audited the whole tree and found fourteen crates dispatching the
per-element `rmsnorm` by hardcoded index while `rmsnorm_rows`,
`backend_api::select` and `block::rms_variant` all already existed. It also
named the reason the list was that long, and this entry is the follow-through
on it.

`block::rmsnorm_fwd` hardcoded `k.rmsnorm`. That builder is called from inside
`block::gqa_attn_qkv`, `gqa_mixer_fwd` and `gdn_mixer_fwd`, so **a model
cannot fix those norms from its own call sites at all** - which is why a
per-model audit keeps reporting "this crate is fine" about call sites while
the builder underneath is not. For qwen3omnimoe that was three of the four
norms per layer.

So the selection moved into `rmsnorm_fwd`. `KernelIds` grew one slot,
`rmsnorm_rows`, holding `UNREGISTERED` until a model registers the pipeline.
The change to the shared builder is behaviour-neutral by construction: every
existing literal keeps the sentinel, so nothing computes anything different
until a model opts in - and then ONE line switches every RMSNorm that model
composes, the shared ones included.

**Where the leverage actually was.** Five per-token decoders adopted it:
qwen35moe, qwen3omnimoe (thinker and talker), qwen3tts (Talker and MTP),
deepseek2, glmdsa. Four needed one pipeline entry and one slot. Only glmdsa
needed real code, because it has no `KernelIds` - but it has ONE `norm_fwd`
funnel, so it was still a single edit. Measured on a P40, one generated
token's worth of RMSNorm, reference vs coalesced: **8.7x-23.5x**, worst for
glmdsa at 186 ms -> 7.9 ms per token (78 layers x four one-row norms). The
whole-pass evidence remains qwen35's 3.94 -> 7.44 tok/s on real weights; none
of these five checkpoints is on this box and three of them do not fit on it,
so the honest per-model number here is the norms alone.

Three things generalise.

**A behaviour-neutral shared-seam commit is the cheap half.** Registering the
slot per model is then a one-line, individually-revertible change with its own
gate. Splitting it that way meant the risky part (a params-layout change on a
builder twenty crates call) landed proven-inert, and the numerically-visible
part landed per model.

**The measuring apparatus is under test too.** The first run of the new
`rmsnorm_decode_cost` table reported 0.1x-1.6x - including *slowdowns* - and
it was completely wrong: `Gpu::submit` only QUEUES, so it was timing enqueue
cost, which is identical for both kernels. One readback per rep to drain the
queue turned the same code into 8.7x-23.5x. A performance result that
disagrees with a mechanism you can state should be treated as a bug in the
harness until proven otherwise.

**A bit-identity oracle that builds its own ids is a second call site.**
`qwen3omnimoe`'s two `*_decode` tests hand-build a `KernelIds` to compute the
expected value. Registering the coalesced kernel in the crate broke them
instantly - correctly, because their oracle was now on a different kernel than
the code under test. They resolve the slot BY NAME now. Any test that
reconstructs a production id set is coupled to it and will go red for reasons
that have nothing to do with what it tests.

**Still open, deliberately.** Nine of #76's fourteen are untouched, and two of
them are mislabelled in that entry: **lfm2 is not a per-token decoder** - it is
a bidirectional encoder whose only `rows` is the token count, minimum 8 on the
QK-norms, so there is no `m = 1` shape there at all - and **gemma4** is a
one-shot text encoder for LTX-2.5 in the same position (`rows = 1` is reachable
only as the empty-prompt fallback). Both would still gain, since the
cooperative kernel wins at every row width, but neither is the archetype and
both carry cosine-1.000000 parity gates that a 3e-6 reduction-order change has
to be re-checked against, so neither is a drive-by. The remaining seven
(toymoe, chronos2, fincast, mimi, kronos, s3dit, ltxv's `na_decoder.rs`) are
training or diffusion tapes at `m > 1`; four of them reach `rmsnorm_fwd` and
so are now one slot away.
