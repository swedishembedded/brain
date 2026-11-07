<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 136. Two lists that must agree, and nothing checking that they do

**Lesson:** When a capability is gated on N tables agreeing, the assertion that they agree is part
of the feature, not follow-up work. A migration is only observable by running the command, not by
reading the lists.

## What happened
Declaring an architecture "resolver-migrated" (so it finds its weights by scanning the model store
instead of reading a `BRAIN_*_DIR` variable) takes THREE edits in three files:

    resolve.rs        RESOLVER_MIGRATED_ARCHS   suppresses the env auto-fetch
    resolver_cli.rs   with_arch_spec            supplies the ArchSpec instead
    catalog/lib.rs    ModelEntry.provider       builds from the Assembly

FOUR ARCHITECTURES WERE ALREADY IN THAT STATE: `kronos`, `qwen3vl`, `fastvlm` and `deepseek2ocr`.
They had been since they were listed. Nothing failed, because no test related the two lists and
every worked example in the docs passes `--weights` explicitly. They were found by a test written
while migrating two more, not by anything noticing on its own.

## Why
Making only the first edit is worse than making none of them. `RESOLVER_MIGRATED_ARCHS` turns the
env path OFF, and `with_arch_spec` is what turns the replacement ON. An architecture in the first
list but missing from the second gets neither.

In that case `run_generic_migrated` returns `None`, dispatch falls through to the env path it was
supposedly migrated off, and auto-fetch is already disabled. The failure mode is a command
demanding a `BRAIN_*` variable for a checkpoint sitting in the model store, which is exactly the
symptom migrating was meant to remove.

## What to do
When a capability is gated on N tables agreeing, the assertion that they agree is part of the
feature, not follow-up work. The test here is four lines
(`every_generically_dispatched_migrated_arch_has_a_spec`). It constrains only the architectures
that actually reach the generic path: an `ARCH_HANDLERS` architecture returns from its own branch
first, so requiring a spec of it would be a false positive.

The second half is not optional either. With both lists agreeing, `scrfd` still demanded
`BRAIN_SCRFD_DIR`, because the catalog's `ModelEntry.provider` was still `from_env!`. A migration
is only observable once the provider is built from the `Assembly`, so verify the migration by
RUNNING the command with the variable unset, not by reading the lists.

Also worth knowing: a spec's role resolves to the FILE the resolver classified, while a provider
that loads several released files by name wants the DIRECTORY holding them. `catalog::role_dir` is
the one conversion between the two, so a role never has to be declared twice to mean both.
