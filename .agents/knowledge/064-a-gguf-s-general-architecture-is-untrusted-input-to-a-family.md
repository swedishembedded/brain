<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 64. A GGUF's `general.architecture` is UNTRUSTED input to a family-name match - dispatch it in the same `match` as native checkpoints and any collision routes a file to a loader that cannot read it

`crates/cli/src/model_dir.rs` synthesizes a `ModelCard` for a `.gguf` with
`family = general.architecture` **verbatim** - a string the file itself
supplies - and then fed it to the SAME `resident_for` match that dispatches
brain-native checkpoint families. Two namespaces, one match: the moment an
architecture string spells a family name, the raw GGUF is handed to that
family's resident. `qwen35`/`qwen35moe` are the live examples - their serving
`Engine` reads `checkpoint::load`, which is safetensors-only, so a raw
`qwen35moe.gguf` (a real file, produced by llama.cpp, with a registered
importer sitting right there) was routed into a loader that cannot open GGUF
bytes at all and died with a low-level parse error, instead of the one-line
`run \`brain import-gguf ...\`` advice the fallback arm already knew how to
print.

The fallback arm DID have the GGUF advice, and two arms above it had comments
about a previous collision between the two qwen35 crates. That is the tell: a
defect being handled per-arm, with each new arm expected to remember, is a
defect in the dispatch shape. The fix is a gate BEFORE the match - a `.gguf`
reaches a family arm only if that family opted in (`family_reads_gguf`:
`gpt`/`glm`/`qwen`, which load through `checkpoint::weightio::WeightReader`
and so read either container) - and it is an ALLOWLIST, not a list of
exclusions, so the default for the next family somebody adds, and for the next
architecture string llama.cpp invents, is the helpful advice rather than a
mis-route.

The general shape: when one `match` keys on a string that arrives from two
sources - one you control (what your own writer stamps) and one you do not
(what a third-party file declares) - the collision is not hypothetical, it is
scheduled. Separate the two before the match, not inside it.
