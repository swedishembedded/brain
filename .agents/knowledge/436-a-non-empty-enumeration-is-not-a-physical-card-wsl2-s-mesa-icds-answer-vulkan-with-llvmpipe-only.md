<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 436. A non-empty enumeration is not a physical card: WSL2's Mesa ICDs answer Vulkan with llvmpipe only, so the CUDA fallback never fired

**Lesson:** A registry fallback keyed on "the higher-precedence source returned nothing" strands a
real card behind a software rasteriser whenever the high-precedence source answers with junk. A
source counts only when it yields at least one *physical* card; on WSL2, Mesa's Vulkan ICDs answer
with llvmpipe alone, so "non-empty" and "has a card" disagree exactly there.

## What happened

`brain devices` on this machine (WSL2, Ubuntu, NVIDIA Quadro M2000M with a working kernel driver)
printed "no physical GPUs" and every GPU request ran on the CPU, while `nvidia-smi` listed the card.
`--device cuda` was equally dead: the canonical registry was empty, so no CUDA ordinal could be
resolved. The vulkan and wgpu paths and the CUDA path are gated by `BRAIN_ALLOW_CUDA`, which the
driver export sets to 1 in the command environment (`/proc/sys/kernel/cap_last_cap` in the control
environment is irrelevant here).

## Why

- **The fallback fired on raw emptiness, not on "no physical card".** `registry()` accepted the
  Vulkan enumeration whenever it returned `Ok(v) if !v.is_empty()`. WSL2 mounts Mesa's ICDs
  (`/usr/lib/wsl/lib/libvulkan_dzn...` etc.) but ships no NVIDIA Vulkan ICD, so Vulkan reports
  exactly one adapter: llvmpipe, `DeviceClass::Cpu` - a non-empty list that holds no physical card.
  The CUDA source was then never probed, and `from_identities` dropped the rasteriser from the
  canonical registry, leaving it empty.
- **`from_identities` already knew what a card is** (`class != DeviceClass::Cpu`); the fallback
  simply never asked the same question of its sources. A source that answers with a rasteriser is
  not a source that answered.
- **`dlopen` is the reason the fallback exists; it is not a reason to skip it.** CUDA is last
  because probing it loads the driver into the process - but the load only happens when the first
  two sources yield no physical card, so a box whose Vulkan already answers never pays it.

## What to do

`resolve_registry_sources` walks the three sources declaratively and accepts a source only when
`has_physical_card` says so (any identity whose class is not `Cpu`). The CUDA fallback now fires
when the first two yield no *physical* card, never on raw emptiness. CUDA is probed at most once,
and only where the first two did not answer - a card visible to Vulkan alone never loads the
driver. Each source that is passed over records its reason for the trace, so `RUST_LOG=debug`
shows why the winner won. Gated by `a_software_rasteriser_alone_leaves_the_cuda_fallback_open`
and `a_real_vulkan_card_beats_the_cuda_fallback_and_never_probes_it`
(`crates/gpu-core/src/devices.rs`).

## Evidence

Quadro M2000M on WSL2 (`/usr/lib/wsl/lib` on the driver side, Mesa ICDs mounted): before, `brain
devices` reported "no physical GPUs"; after, the registry holds the card under source `cuda`,
`--device cuda` and `--device gpu` schedule onto it, and `brain devices` names it.
