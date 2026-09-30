<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 182. A length prefix is untrusted input

A safetensors file starts with an 8-byte little-endian header length, and
every reader in the workspace trusted it. `st::read_metadata`,
`st::read_card`, `st::param_count_from_header` and
`st::declared_data_extent` did `vec![0u8; len]` before reading the header.
A corrupt checkpoint whose first 8 bytes happened to decode to a size the
allocator could not satisfy aborted the process. An abort is not a panic,
so no caller can catch it, and the requested size in the abort message was
just the file's first 8 bytes read as a `u64`. The mapped readers
(`MmapSafetensors::open`, `safetensors::read`/`parse`) computed
`8 + len` unchecked. That overflows for a length near `u64::MAX`, which
panics in debug builds and wraps in release to a slice that panics.

`checkpoint::safetensors::validate_header_len` is now the one check every
reader calls before it sizes anything by the prefix. A length is refused
with `HeaderLenError` (file, claimed length, file size) if it is longer than
what follows the prefix in the file, or longer than `MAX_HEADER_BYTES`
(100 MB, the cap the format's reference implementation enforces).
`crates/checkpoint/tests/corrupt_header.rs` drives every reader with
`u64::MAX`, `file_size + 1` and `cap + 1`.

The same audit found that `toymoe`'s inference loader parsed a pre-safetensors
container (`{"config", "tensors": [{offset, numel}]}`) that nothing had
written since `checkpoint::save` switched to safetensors. Every checkpoint
the trainer wrote failed to load with `missing config.vocab_size`, and no
test loaded one. It now reads through `st::load_safetensors`, and a unit
test loads a `checkpoint::save` output.
