<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 437. Requantizing a GGUF without carried KVs produces a file the scan can never serve

**Lesson:** A GGUF is a container plus the metadata that makes its tensors a model: a
recontainerizer that keeps only tensor data and the KVs it writes itself silently produces a
file that parses but can never be served. A GGUF→GGUF conversion must carry the source's config
KVs and embedded tokenizer through, and must not confuse "the KVs I write myself" with "the only
KVs that exist".

## What happened

`brain quantize old.gguf --out new.gguf --tier Q4_K` (GGUF→GGUF) wrote only its four/five generic
KVs (`general.architecture`, `general.name`, `general.quantization_version`,
`general.source.name`, `general.file_type`). Every architecture-specific KV - the `qwen3.*` block
counts a GGUF resident reads beside the tensors - and the embedded tokenizer were dropped. The
result imported fine as raw tensors and then failed the model-dir scan, which dispatches on
`general.architecture` and hands the config KVs to the resident; the conversion "worked" and
yielded an unservable file. The quantize output was also renamed to the source file's stem
instead of keeping the model's own `general.name`, so a file quantized from `Qwen3 0.6B Instruct`
GGUF changed its served identity.

## Why

- **The scan's contract is container + KVs.** The model-dir scanner reads `general.architecture`
  to pick the importer and the resident reads the config KVs beside the tensor data; a GGUF
  without them is a directory of numbers, and the scanner can only decline it.
- **The generic KVs the CLI writes are overrides, not the universe.** Treating them as a complete
  KV set conflated "the header this command owns" with "the header the output needs". The
  architecture KVs are not derivable from the tensors - the importer's expectations (block count,
  head count, embedding width) are knowledge the source file already carries.
- **The name is the served id.** A GGUF's `general.name` is the model's name, not a filename
  stem; falling back to the stem reidentified the model on every quantize round.

## What to do

`Source::carried_kvs` copies every source KV except the five generics the command writes itself
(the output's own tier, quantization version and source name win); `Source::general_name` keeps
the model's name when it declares one, with the stem as the last fallback. A safetensors source
carries nothing - there is no KV to carry. Gated by
`a_gguf_source_carries_its_kvs_and_the_output_overrides_the_generics`
(`crates/cli/src/quantize_cli.rs`), which writes a real GGUF with `qwen3.*` config KVs and
asserts both the ride-along and the overrides.

## Evidence

`brain quantize` GGUF→GGUF round trip: before, the output held 5 KVs and the scan declined it;
after, the output carries the source's config KVs and tokenizer and is served from the model-dir
scan like any downloaded GGUF.
