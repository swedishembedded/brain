<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 577. A repository is not a loose file: a dotted repo name lost its tail as an extension

**Lesson:** `modelstore::resolve::model_id_from_path` names a per-instance model `<vendor>/<repo>` from its position in
the store, and for a path two components deep it took the second component's *file stem*, the rule for a loose file
(`unsloth/flux-2-klein-9b-Q8_0.gguf` -> `unsloth/flux-2-klein-9b-Q8_0`). A repository *directory* two components deep
went through the same rule, so `Qwen/Qwen3-TTS-12Hz-0.6B-Base` was served as `Qwen/Qwen3-TTS-12Hz-0` and
`Qwen/Qwen3-ASR-1.7B` would be `Qwen/Qwen3-ASR-1`: a client naming the real repo was told no such model exists. Every
instance whose instance role is the checkpoint directory itself (Qwen3-TTS's `ckpt`) hit it; FLUX.2's `dit`, a file
inside its repo, never did, which is why its tests passed.

A directory now keeps its whole name (`model_id_from_path_keeps_a_repository_directorys_whole_name`).

**Second half, same change:** `resolve_all` resolved an instance's other roles across the whole store, so two
Qwen3-TTS releases (each a `ckpt` and a `weights_dir` from one manifest) gave each instance two `weights_dir` candidates
and served neither. A role the instance's own repository holds exactly one candidate of is now that one; only a role
its repository lacks is chosen store-wide (`resolve_all_takes_an_instances_other_roles_from_its_own_repository`).
