// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **What a checkpoint tensor costs once it is on a device, and what the
//! device will actually accept** - the two byte questions every placement
//! decision and every upload has to answer, in one dtype-agnostic place.
//!
//! # Why this is not per-model, and not per-dtype
//!
//! Deciding which card a tensor lands on is a question about *bytes and
//! devices*. It must not care whether those bytes are stored as `F32`, `BF16`,
//! `F16`, brain's packed-int8 `U32`, or - the case this module exists to keep
//! open - a checkpoint whose layers are quantized at DIFFERENT levels
//! (attention kept high-precision, MoE experts squeezed harder), which is
//! ordinary in real GGUF releases. Every one of those decodes to f32 on the
//! way to the device through the same bounded reader
//! ([`checkpoint::TensorSource::with_tensor_chunks`]); the only thing that
//! varies is how many f32 elements a stored element expands to, which is
//! exactly [`device_f32_words`] and nothing else.
//!
//! So a cost model built on this is per-TENSOR by construction: ask each
//! tensor its own declared dtype and shape, never the checkpoint "its" dtype.
//! A mixed-quant checkpoint plugs into the same placement planner and the same
//! uploader as a uniform one, with no second implementation.
//!
//! # The device's own ceilings
//!
//! [`DeviceLimits`] carries the two limits a load can trip over, both READ
//! from the live device (`gpu_core::Gpu`), never assumed:
//!
//! * `max_buffer_bytes` - the largest single allocation the driver accepts.
//!   On a Vulkan backend this comes from the driver's `maxMemoryAllocationSize`.
//! * `max_binding_bytes` - the largest single storage-buffer BINDING a shader
//!   may see. Usually the smaller of the two (on wgpu it is additionally
//!   clamped to `i32::MAX`, independently of what the driver reports), and a
//!   buffer larger than it is still legal - the caller binds sub-ranges
//!   (`Gpu::step_sliced`).
//!
//! Hardcoding either is a bug: they differ per card, per driver and per
//! backend, and the numbers do not even agree with each other on one device.

use gpu_core::Gpu;

/// f32 words tensor data of `dtype` with `stored_numel` STORED elements
/// occupies once decoded onto a device.
///
/// One entry differs from the identity, and it is a brain convention rather
/// than a dtype: a safetensors `U32` tensor is brain's packed-int8 storage
/// (`model::int8::quantize_weight`'s 4-lanes-per-word layout, whose scale
/// sibling is `[n, k/32]`), so its shape
/// says `[n, k/4]` while the f32 it decodes to is `[n, k]` - four times as
/// many elements. Charging it its stored count is a 4x under-report of what a
/// loader actually places on the card.
///
/// Every other dtype brain can read - `F32`, `F16`, `BF16`, the integer types,
/// and each GGUF k-quant block type - decodes one f32 per stored element,
/// whatever its on-DISK byte width was. That on-disk width is deliberately not
/// modelled here: it bounds how much is read, never how much lands on the
/// device, and it is placement's answer that this function gives.
pub fn device_f32_words(dtype: Option<&str>, stored_numel: u64) -> u64 {
    match dtype {
        Some("U32") => stored_numel.saturating_mul(4),
        _ => stored_numel,
    }
}

/// [`device_f32_words`] in bytes - what a placement planner charges a device.
pub fn device_bytes(dtype: Option<&str>, stored_numel: u64) -> u64 {
    device_f32_words(dtype, stored_numel).saturating_mul(4)
}

/// One tensor's device cost, read straight off a checkpoint header: its own
/// declared dtype and its own shape, per tensor. `None` when the checkpoint
/// does not have the tensor at all - never a panic and never a silent 0, since
/// a missing tensor means the caller's shape model and the checkpoint disagree
/// and a plan built on a 0 would overrun the card it was placed on.
pub fn tensor_device_bytes(reader: &checkpoint::weightio::WeightReader, name: &str) -> Option<u64> {
    let numel: u64 = reader.shape(name)?.iter().product();
    Some(device_bytes(reader.dtype(name), numel))
}

/// The ceilings one device imposes on a single buffer, read from that device.
/// See the module doc for why the two numbers are separate and why neither may
/// be a constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLimits {
    /// Largest single allocation the driver accepts, in bytes.
    pub max_buffer_bytes: u64,
    /// Largest single storage-buffer binding a shader may see, in bytes.
    pub max_binding_bytes: u64,
}

impl DeviceLimits {
    /// Query the live device. The only supported way to obtain these.
    pub fn of(gpu: &Gpu) -> DeviceLimits {
        DeviceLimits { max_buffer_bytes: gpu.max_buffer_bytes(), max_binding_bytes: gpu.max_storage_binding_bytes() }
    }

    /// `Err` naming BOTH the request and the queried ceiling when a `numel`-word
    /// f32 buffer cannot be allocated on this device.
    ///
    /// Only the allocation ceiling is enforced: a buffer over
    /// `max_binding_bytes` is legal and merely constrains how a kernel may bind
    /// it (sub-ranges via `Gpu::step_sliced`), so refusing it here would reject
    /// loads that work.
    pub fn check_alloc(&self, name: &str, numel: usize) -> Result<(), String> {
        let want = 4u64.saturating_mul(numel as u64);
        if want > self.max_buffer_bytes {
            return Err(format!(
                "upload '{name}': needs a single {want}-byte buffer but this device's queried max_buffer_size is {} bytes",
                self.max_buffer_bytes
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spec: the cost model is per-tensor and dtype-agnostic - a mixed-dtype
    /// checkpoint (the real GGUF case: attention at higher precision, experts
    /// quantized harder) costs each tensor by its OWN declared dtype, and every
    /// float/quant dtype lands as one f32 per element regardless of its on-disk
    /// width.
    #[test]
    fn every_dtype_costs_one_f32_per_element_except_packed_int8() {
        for d in ["F32", "BF16", "F16", "I8", "U8", "Q4_K", "Q6_K", "Q8_0"] {
            assert_eq!(device_bytes(Some(d), 1000), 4000, "{d} must cost one f32 per element on the device");
        }
        // brain's packed-int8 convention: [n, k/4] stored, [n, k] consumed.
        assert_eq!(device_bytes(Some("U32"), 1000), 16000);
        // An unknown/absent dtype is treated as plain, never as packed -
        // over-charging by 4x would make a fitting model look unplaceable.
        assert_eq!(device_bytes(None, 1000), 4000);
    }

    /// Spec: a mixed-quant layer's total is the SUM of per-tensor costs, so no
    /// caller can accidentally derive it from a single "the checkpoint's dtype".
    #[test]
    fn a_mixed_quant_layer_sums_its_tensors_own_dtypes() {
        let tensors: [(Option<&str>, u64); 3] = [(Some("BF16"), 100), (Some("Q4_K"), 200), (Some("U32"), 50)];
        let total: u64 = tensors.iter().map(|&(d, n)| device_bytes(d, n)).sum();
        assert_eq!(total, 400 + 800 + 800);
    }

    /// Spec: the allocation guard compares against the CARRIED (queried) limit
    /// and names both numbers, so an operator can tell a real driver ceiling
    /// from a mis-sized request. No constant appears in the check itself.
    #[test]
    fn the_alloc_guard_uses_the_carried_limit_not_a_constant() {
        let small = DeviceLimits { max_buffer_bytes: 4096, max_binding_bytes: 2048 };
        let err = small.check_alloc("w", 2048).expect_err("8192 bytes must not fit a 4096-byte ceiling");
        assert!(err.contains("8192") && err.contains("4096"), "{err}");
        small.check_alloc("w", 1024).expect("4096 bytes must fit exactly");
        // A buffer larger than the BINDING ceiling is still allocatable.
        let big = DeviceLimits { max_buffer_bytes: 1 << 40, max_binding_bytes: 1024 };
        big.check_alloc("w", 1 << 20).expect("a buffer over the binding ceiling is legal");
    }
}
