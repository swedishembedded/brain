<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 44. Destroying the last Vulkan instance can permanently blind a process to its own GPUs

`brain ltxv t2v` with real weights ran all 8 denoise steps, then died the
instant VAE decode opened its device:

```
brain: wgpu enumerated 0 adapters while looking for "Tesla P40" (pci ...);
       falling back to wgpu's own adapter request
limits: max_buffer_size 2047 MiB, max_storage_buffer_binding_size 128 MiB
wgpu error: Validation Error
  Buffer binding 2 range 452984832 exceeds `max_*_buffer_binding_size` limit 134217728
```

Every plausible reading of that is wrong. It is not VRAM exhaustion (the card
was under 2 GB of 24 GB throughout), not driver corruption (`nvidia-smi` and
`brain devices` enumerate perfectly the moment the process exits), and not a
placement bug in the identity matching. The process had simply stopped being
able to see hardware that never went anywhere - and then silently continued on
a software rasteriser, whose 128 MiB binding limit is what actually raised the
error, several layers away from the cause.

**The cause: instance churn, not device churn.** Creating a `wgpu::Instance`
(or an `ash` one) makes the Vulkan loader `dlopen` every installed ICD;
destroying the last one makes it `dlclose` them. The NVIDIA ICD does not
survive that cycle. With `VK_LOADER_DEBUG=error` the tenth round says exactly
what happened:

```
loader_scanned_icd_add: Could not get 'vkCreateInstance' via
    'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
```

The library still loads; it just no longer has a working entry point. From
then on the process enumerates zero physical devices, forever. brain hit this
because "open a fresh device per forward call" is a deliberate and measured
design in several models, and each open built its own instance - nine were
enough.

**Rules that follow.** A Vulkan instance is process-scoped state: create one,
never destroy it. Logical devices are still created and destroyed per use;
only the instance is permanent, and it owns no VRAM. And an adapter-enumeration
fallback must be *loud*, because a software adapter computes correct answers
slowly right up until an allocation exceeds its much smaller limits - which is
why the churn tests that already existed
(`crates/gpu-core/tests/device_churn.rs`) passed all the way through this bug:
they asserted the arithmetic was right and never asserted **which card ran it**.
A device-lifecycle test has to gate placement, not just results.

Two earlier investigations recorded confident, wrong conclusions from this same
symptom - "whatever wedges the enumeration is not brain's device lifecycle",
and a "slow-reclaim driver-side handle pool" theory measured to a deterministic
count. A deterministic failure count is a strong hint to go find the mechanism
(here: one `VK_LOADER_DEBUG=error` run), not a phenomenon to characterise
further.
