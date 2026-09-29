<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 172. A gate red for everything hides the failures it was written for

`scripts/gates/check-arch-names.sh`'s "every model page is published" check
looked for `models/<page>.md` in `docs/manifest.txt`, which lists pages as
`docs/models/<page>.md`. Once the manifest took that form every one of the 67
model pages was reported unpublished, the gate was red on every run, and a red
that never changes is read as noise. Underneath it, four real violations had
accumulated unseen: four infra verbs (`models`, `plan`, `gguf`, `roofline`)
missing from the dispatch allowlist, a registered architecture (`qwen3vlmoe`)
with no docs page at all, and two legitimate pages for things that are not
registry architectures (`fly`, `optionhead`) with no category to live in.

**Rule:** a gate that fails for every input has stopped checking anything;
treat "always red" as a defect in the gate, fix the gate first, and then
read what it was hiding.
