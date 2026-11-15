<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 464. A CUDA source is compiled whole, so one entry's instruction floor became every entry's, and a Pascal card lost the exact GDN decode

**Lesson:** the backend compiles a registry entry's whole `src` for the device, not just its entry point. One kernel
that issues an instruction a part lacks makes the module fail to build there, and every other entry of the same
source disappears with it, whatever floor it declares. Declaring the newest instruction's floor on every entry hides
the failure but keeps the loss.

## What happened

`gdn_decode.cu` held the exact Gated DeltaNet decode bodies (plain f32 arithmetic, offered from compute 5.0) when the
split pooled recurrence added `cp.async` (8.0) to the same file, and later the streaming kernels added
`cp.async.bulk` and transaction-counted barriers (9.0). The registry check then rejected the baseline floors, and all
eighteen entries were given the 8.0 floor. On a compute 6.1 card that removed the native GDN step, the exact verify
round and the pooled decode: `qwen35`'s chunk rounds fell back to the chunk tape, so `chunk_tape_decode_parity` failed
bit identity (bisected to the commit that added the first `cp.async`), and the decode tape ran the nineteen-kernel
chain. The streaming entries kept an 8.0 floor while needing 9.0.

## What to do

- A source that mixes instruction generations compiles each newer section only under its capability
  (`#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800`, `>= 900`), so the module builds everywhere and each entry
  declares the floor of its own body. A helper the older entries share gets a portable branch, not a raised floor.
- A raised floor is a capability withdrawn from a device class: say which entries actually need it, and run that
  device's tests (`gdn_decode_native` and `chunk_tape_decode_parity` on a pre-8.0 card).

## Evidence

At the change: `kernels-cuda`'s registry test accepts the per-entry floors (with a `cp.async.bulk` rule beside the
`cp.async` one); on a compute 6.1 card `gdn_decode_native`, `gdn_verify_native` and `chunk_tape_decode_parity` pass
bit for bit, the streaming tests skip with their 9.0 reason, and `decode_fusion`'s one-launch GDN step engages.
