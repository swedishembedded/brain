<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 521. A card hidden from CUDA made the whole machine portable, so `CUDA_VISIBLE_DEVICES=0` ran the 27B on wgpu

**Lesson:** Vulkan and wgpu enumerate every card on the machine, whatever `CUDA_VISIBLE_DEVICES` says. The CUDA
driver enumerates only the cards it was given. A rule that asks "does CUDA run on every card" reads a card the
operator hid from CUDA as one CUDA cannot drive. Treat the hidden cards as the operator's statement that they belong
to another process, not as a reason to leave CUDA.

## What happened

On a four-GH200 node, `CUDA_VISIBLE_DEVICES=0 brain serve ...` built the canonical registry from Vulkan with all four
cards. `Hardware::detected` matched only card 0 to a CUDA device, so cards 1-3 had no compute capability, and
`Hardware::cuda_everywhere` was false. The `BACKEND` rule then chose wgpu for the whole process, with no warning:
`cuda_fallback_notice` speaks only when CUDA cannot load, and it could.

That had two effects:

- The 27B ran on wgpu over card 0 and crashed with a wgpu validation error. Driving the other three cards would
  also have touched cards that belonged to other processes.
- With the backend forced to CUDA, every default ruled on `cuda_everywhere` was still the portable one:
  `--gdn-state f32` ("the bf16 pool kernels are CUDA's") and `--speculative off` ("speculation verifies with CUDA
  kernels only") on a run that served on CUDA.

## Fix

- `Hardware::cuda_cards` and `Hardware::hidden_from_cuda` report, as card masks, which cards CUDA drives and which
  NVIDIA cards a working CUDA driver does not list.
- `Hardware::own_cards` leaves the hidden cards out. `cuda_everywhere`, `min_compute_capability` and the Qwen3
  prefill planner range over the cards this process owns.
- `Inventory` carries both masks. `DeviceSpec::resolve` makes "every card" (no `--device`, or `gpu`) mean the cards
  CUDA drives, on CUDA.
- A run that names no backend is CUDA's whenever CUDA drives every card it names.
- A named card hidden from CUDA is refused, and the refusal names the card, the cards CUDA does drive, and the
  `--backend wgpu|vulkan` way to drive it anyway. An explicit portable backend still drives what it names.

Tests:

- `gpu_core` `a_card_hidden_from_cuda_is_told_apart_from_one_cuda_cannot_drive` and
  `a_card_hidden_from_cuda_never_turns_the_run_portable`, on modelled four-card inventories.
- The existing mixed-vendor and no-driver rules are unchanged.

Checked on the node with `CUDA_VISIBLE_DEVICES=0 brain serve` and no device flags: the result was `compute gpu[0]`,
replies with `backend: cuda`, and the 27B served. The CUDA-only defaults now apply as they do with all four cards
visible.
