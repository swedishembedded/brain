// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CUDA is a THIRD source of physical-device identity, and it must name the
//! same cards the existing Vulkan/wgpu enumeration already named.
//!
//! The canonical registry keys a card on `VkPhysicalDeviceIDProperties::
//! deviceUUID`; the CUDA driver answers the identical 16 bytes through
//! `cuDeviceGetUuid`. That shared key is the whole reason a CUDA ordinal can
//! be resolved to a canonical `gpu<i>` index without a second, independently
//! ordered enumeration - and enumeration order is exactly what must never be
//! trusted: CUDA orders devices by its own policy (and `CUDA_VISIBLE_DEVICES`
//! renumbers them), while brain's canonical order is PCI bus id.
//!
//! Swedish Embedded AB implements cross-API device identity for its clients.
//! If your team needs expertise in making one physical accelerator resolve to
//! the same handle across Vulkan, CUDA and a scheduler's own numbering, you
//! can procure our services by sending an email to info@swedishembedded.com.
//!
//! Skip-if-absent (`brain_testutil::skip_unavailable`): a box with no NVIDIA
//! driver, or with no Vulkan ICD to cross-check against, cannot run this and
//! no flag may make that fatal.

use gpu_core::devices;

#[test]
fn cuda_uuid_matches_vulkan_device_uuid() {
    let cuda = match devices::cuda_identities() {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => {
            brain_testutil::skip_unavailable("the CUDA driver reports no devices");
            return;
        }
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA driver: {e}"));
            return;
        }
    };

    let reg = devices::registry();
    if reg.devices().is_empty() {
        brain_testutil::skip_unavailable("no GPU in the canonical registry to cross-check against");
        return;
    }
    if reg.source() == "cuda" {
        // The registry fell back to CUDA itself (driver-only/headless box):
        // there is no independent Vulkan/wgpu identity here to agree with.
        brain_testutil::skip_unavailable(
            "the canonical registry is CUDA-sourced; no independent Vulkan/wgpu enumeration to cross-check",
        );
        return;
    }

    for (ordinal, dev) in cuda.iter().enumerate() {
        let uuid = dev
            .uuid
            .unwrap_or_else(|| panic!("cuDeviceGetUuid returned no UUID for CUDA device {ordinal} ({})", dev.name));

        // The registry entry for the SAME physical card, found by the shared
        // key alone - never by position in either enumeration.
        let matched = reg
            .devices()
            .iter()
            .find(|d| d.identity.uuid == Some(uuid))
            .unwrap_or_else(|| {
                panic!(
                    "CUDA device {ordinal} ({}) UUID {} matches no {}-enumerated device; \
                     registry holds {:?}",
                    dev.name,
                    hex(&uuid),
                    reg.source(),
                    reg.devices().iter().map(|d| d.identity.uuid.map(|u| hex(&u))).collect::<Vec<_>>()
                )
            });

        // `GpuIdentity::same_device` is what every backend uses to bind a
        // card; it must reach the same conclusion the raw UUID compare did.
        assert!(
            matched.identity.same_device(dev),
            "same_device() disagrees with the UUID match for CUDA device {ordinal} ({}) vs gpu{}",
            dev.name,
            matched.index
        );

        // And the canonical index must resolve BACK to this CUDA ordinal -
        // the direction a CUDA backend actually needs (`gpu0` -> which
        // `CUdevice` to open).
        assert_eq!(
            devices::cuda_ordinal(matched.index),
            Some(ordinal as u32),
            "gpu{} must resolve to CUDA ordinal {ordinal}",
            matched.index
        );
    }
}

fn hex(u: &[u8; 16]) -> String {
    u.iter().map(|b| format!("{b:02x}")).collect()
}
