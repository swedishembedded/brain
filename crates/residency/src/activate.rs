// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for building models on exactly the
// accelerator a scheduler chose, for its clients. If your team needs
// expertise in race-free device placement for concurrent model loading then
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Building an instance on the device the executor assigned.

use crate::Device;

/// Run `f` placed on the residency-assigned device: a GPU assignment becomes a
/// scoped (thread-local) selection in the canonical device registry, so every
/// `Gpu::new` inside `f` binds that physical card, race-free across the
/// executor's concurrent activation lanes. The one way a resident's
/// `activate` (or a run that builds device state) binds its device, in the
/// serving process and in a worker hosting the model alike.
///
/// `f` always builds a wgpu (GPU/CPU backend) engine, so a `Device::Npu`
/// assignment must never reach here silently: a resident that wants NPU
/// placement (`MemCost::with_npu`) branches on `Device::Npu` in its own
/// `activate` before calling this.
///
/// `Device::Cpu` scopes with [`gpu_core::devices::with_host_tier`], not a
/// bare `f()`: an unscoped `f()` would leave every `Gpu::new` inside bound to
/// whatever card is ambient (the memoized `auto_home()` answer from startup),
/// so a residency fallback to the host tier would double-book the GPU it was
/// falling back from instead of relieving it
/// (`crates/gpu-core/tests/it/host_tier.rs`).
pub fn on_device<R>(device: Device, f: impl FnOnce() -> R) -> Result<R, String> {
    match device {
        Device::Gpu(i) => gpu_core::devices::with_gpu(i, f),
        Device::Cpu => Ok(gpu_core::devices::with_host_tier(f)),
        Device::Npu(i) => Err(format!(
            "on_device: got Device::Npu({i}) but this resident has no NPU activation path; \
             a resident declaring MemCost::with_npu must branch on Device::Npu in its own \
             activate() before calling on_device, not fall through to the wgpu build"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Device::Cpu` builds `f` inside the host tier, a GPU assignment never
    /// does, and an NPU assignment is refused rather than built as wgpu.
    #[test]
    fn on_device_cpu_scopes_to_the_real_host_tier_not_an_unscoped_call() {
        assert!(!gpu_core::devices::on_host_tier(), "no scope must be active before this test starts");
        let was_host_tier = on_device(Device::Cpu, gpu_core::devices::on_host_tier).unwrap();
        assert!(was_host_tier, "Device::Cpu must build f() inside with_host_tier, not bare");
        assert!(!gpu_core::devices::on_host_tier(), "the scope must not outlive the call");

        let was_host_tier = on_device(Device::Gpu(0), gpu_core::devices::on_host_tier).unwrap();
        assert!(!was_host_tier, "Device::Gpu must not be dragged onto the host tier");

        assert!(on_device(Device::Npu(0), || ()).is_err(), "Device::Npu must be refused here, never silently built as wgpu");
    }
}
