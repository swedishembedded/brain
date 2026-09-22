<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 135. A cache that REPLACES a description must reproduce every part of it

`brain pull` writes a `brain.manifest.json` beside what it fetched, and
`inventory::compound_records` reads that manifest INSTEAD of walking the
directory - "the manifest is a complete description of what lives here on its
own", so it replaces the generic per-file walk rather than supplementing it.
It emitted one record per declared role, tagged `ArtifactKind::Compound`.

No `ArchSpec::classify` in the workspace accepts `Compound`. Every one of them
gates on `Torch`/`Safetensors`/`Gguf`/`HfDir` before reading a checkpoint's own
tensor shapes, because that is how it decides whether the file is even its
architecture's. So a role naming a single weight FILE produced exactly one
record, of a kind nothing classifies, and the checkpoint was invisible to the
architecture that had just fetched it:

    $ brain pull schwgHao/RealESRGAN_x4plus
    ... fetched 63.9 MiB -> .../schwgHao/RealESRGAN_x4plus/RealESRGAN_x4plus.pth
    $ brain rrdbnet upscale --in image=x.png --out image=y.png
    rrdbnet: weights: no artifact classifies as weights for arch rrdbnet

WHY IT SURVIVED. `brain models list` reads the store, not the resolver, and
happily reported the same checkpoint as `local` with its size and a "gpu0,gpu1
fits" verdict - so the two commands disagreed about whether the model existed,
and only the one nobody scripts against was right. Passing `--weights <path>`
also worked, which is what every test and every worked example in the docs
does, because they predate auto-fetch.

THE HALF-FIX THAT HID IT. This exact failure had already been found and fixed
ONCE, for the directory case: a pulled HF checkpoint now yields BOTH a
`Compound` record and an `HfDir` one
(`a_pulled_hf_checkpoint_is_both_compound_and_hfdir`, whose own doc comment
names `brain nemotronasr transcribe` and `brain fastvlm caption` failing this
way). The single-file twin was never covered, so `sam2`, `rrdbnet` and every
other recipe whose `roles` name a file kept failing - with a green suite.

THE RULE. When a description replaces rather than supplements a generic one, it
owes the consumer every property the generic one provided. `compound_records`
now pushes the role's extension-derived record alongside the `Compound` one, so
a manifest ADDS the manifest's knowledge instead of subtracting the file's.
Fixed in one place for every architecture rather than by teaching 30 `classify`
implementations a new kind - the same "fix it in the selector" rule kernels
follow.

AND: when a fix is written for one shape of an input, enumerate the other
shapes of that same input before closing it. The directory fix and the file fix
are four lines apart in the same function.
