<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 108. A CUDA ordinal is not a device identity, and `CUDA_VISIBLE_DEVICES` is why

brain's canonical `gpu<i>` is physical cards sorted by PCI bus id. CUDA
enumerates by its own policy, and `CUDA_VISIBLE_DEVICES` both filters and
renumbers that enumeration *per process*. So "CUDA ordinal 0 is gpu0" is a
statement about one process's environment, not about the machine - it is
usually true, which is exactly what makes assuming it dangerous.

The sound key is `cuDeviceGetUuid`, which returns the same 16 bytes Vulkan
reports as `VkPhysicalDeviceIDProperties::deviceUUID` and NVML reports as the
GPU UUID. Verified on a 2-card box: all three agree byte for byte, which is
what lets a CUDA ordinal be resolved to a canonical index through the identity
the registry already stores, with no second enumeration to keep in sync.

Two related facts worth keeping: the CUDA Driver API reports the PCI *slot*
(domain/bus/device) but no PCI *function* and no chip PCI device id, so a
CUDA-sourced identity must leave `GpuIdentity::device_id` at 0 rather than
inventing one - fabricating it would corrupt `same_device`'s weakest key for
everyone else. And the device registry's Vulkan and wgpu sources are BOTH
empty on a headless driver-only box, where CUDA is the only source that sees
the cards at all - hence CUDA as a third source, last in precedence.
