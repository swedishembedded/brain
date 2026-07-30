# The canonical device registry

`gpu_core::devices` owns the ONE enumeration of physical GPUs in a brain
process. Everything that names a card — `--device gpu<i>`, `Shard.gpu_index`,
`residency::Device::Gpu(i)`, residency budgets — resolves through it, so an
index provably means the same physical card everywhere.

## Canonical index

The registry enumerates once (lazily, cached for the process), keeps physical
cards only (a software rasteriser is not a card; when any discrete GPU exists,
only discrete cards are indexed), and sorts by **PCI bus id** — stable across
boots and driver updates. The resulting position is the canonical index:
`gpu0` is the lowest PCI bus id. `brain devices` prints the table:

```
$ brain devices
canonical device registry (source: vulkan enumeration, PCI-bus order)
index  name        pci bus        uuid      vram      backends
gpu0   Tesla P40   0000:04:00.0   0c223d47  24.0 GiB  vulkan+wgpu
gpu1   Tesla P40   0000:82:00.0   ce637d48  24.0 GiB  vulkan+wgpu
ambient selection: none pinned — Gpu::new lands on gpu0 (Tesla P40)
```

nvidia-smi (NVML) order is **not assumed** to match: the residency budgets in
`run_cli::query_gpu_mem` key nvidia-smi's per-card capacities by
`pci.bus_id` back onto canonical indices.

## Identity keys, strongest first

1. **PCI bus id** — `VK_EXT_pci_bus_info` in the ash enumeration; also what
   `nvidia-smi --query-gpu=pci.bus_id` reports, giving the NVML↔registry map.
2. **Vulkan `deviceUUID`** (`VkPhysicalDeviceIDProperties`, 1.1 core) — equals
   the NVML GPU UUID on NVIDIA.
3. **Fallback**: `(vendor:device, ordinal)`, where the ordinal counts devices
   with the same vendor:device pair in `vkEnumeratePhysicalDevices` order —
   the tiebreaker for identical twins when neither key above exists.

The canonical enumeration is ash-based (`backend_vulkan::enumerate_physical_gpus`).
Where no Vulkan loader exists, `backend_wgpu::enumerate_gpus` fills in; its
identity is read through the `wgpu::Adapter::as_hal::<Vulkan>` escape hatch
(wgpu's `AdapterInfo` exposes neither PCI nor UUID) — wgpu-hal and brain pin the
same `ash 0.38`, so the raw-handle property queries are shared code shape. Both
lists are the same ICD order, which is what makes the fallback ordinal key
consistent between them.

## Selecting a card

* **Explicit**: `Gpu::new_on(&DeviceId, kernels)` / `Gpu::new_on_index(i, …)`.
  Backends match the identity against their own fresh enumeration
  (`WgpuBackend::new_on`, `VulkanBackend::try_new_on` → `VkContext::new_select`)
  — never by position across independent enumerations, which was observed to
  reorder between calls on the 2×P40 box.
* **Scoped**: `devices::with_gpu(i, || …)` — a thread-local selection every
  `Gpu::new` under the closure resolves to. Race-free under the residency
  executor's parallel activation lanes (the reason the old
  `set_var("BRAIN_GPU_INDEX")` dance had to go).
* **Ambient** (`Gpu::new` with no scope): the `--device gpu<i>` pin recorded by
  `ComputeSet::apply`, else `BRAIN_GPU_INDEX`, else canonical card 0. On a
  GPU-less box the wgpu backend's own default (software rasteriser) applies.

Under `--device cpu` / `BRAIN_DEVICE=cpu`, `new_on`/`with_gpu` still build the
CPU backend — placement is moot there, so device-plumbed call sites run
unchanged on CPU-only boxes and in the CPU test suite.

## Env vars are user input, not an API

`BRAIN_GPU_INDEX`, `BRAIN_FLUX2_TE_DEVICE`, `BRAIN_ZIMAGE_ENCODER_GPU` remain
supported as **user input**, parsed to canonical indices once at the consuming
edge (registry init, flux2/z-image pipeline build). No brain code mutates
process env to place a device anymore: `Shard.gpu_index` (with
`Shard::ANY_GPU` = "ambient"), `DeviceId` plumbing and scoped selections carry
placement instead. Grep gate: `grep -rn 'set_var("BRAIN_GPU_INDEX"' crates/`
must stay empty.

Legacy note: `BRAIN_VK_DEVICE` (a raw `vkEnumeratePhysicalDevices` index into
the native-Vulkan backend) is only consulted when the registry has no cards to
resolve — on GPU boxes the canonical selection governs the Vulkan backend too,
so one index vocabulary covers every backend.
