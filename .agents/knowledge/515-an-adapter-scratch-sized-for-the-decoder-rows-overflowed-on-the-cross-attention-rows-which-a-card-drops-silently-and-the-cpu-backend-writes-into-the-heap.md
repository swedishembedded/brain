<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 515. An adapter scratch sized for the decoder's rows overflowed on the cross-attention's rows, which a card drops silently and the CPU backend writes into the heap

**Lesson:** A scratch buffer sized by "the rows this module runs" is wrong for a module that runs products of two
lengths: an encoder-decoder's cross-attention projects the encoder's rows. A card's robust buffer access drops the
writes past the end, so the result is silently wrong and every GPU test stays green; only the CPU backend, whose
buffers are heap vectors, shows it, as an allocator abort far from the write. Size such scratch for every product
the module issues, and have the owner of the scratch refuse a product it was not sized for.

## What happened

`florence2::text::decoder::Decoder::new_train` built its `AdapterRuntime` for `max_t` rows, while the decoder's
cross-attention runs the k/v projections (LoRA targets) over `t_enc` encoder rows. The recompute test
(`crates/florence2/tests/it/recompute.rs`, 6 prompt tokens, 5 decoder positions) aborted the test binary on the CPU
backend with `free(): invalid next size (normal)` while dropping the scratch; on a card the k/v adapter gradients
were computed from a truncated intermediate. It surfaced only once the CPU backend accounted for its allocations: the
test had panicked on `memory_usage` first.

## Fix

The decoder sizes the scratch for `max(max_t, t_enc)`, and `AdapterRuntime::forward`/`backward` assert that a
product's rows and outputs fit what the runtime was built for, naming the linear.
