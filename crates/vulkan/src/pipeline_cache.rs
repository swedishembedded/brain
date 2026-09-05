// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Persisted `VkPipelineCache` blob, keyed by physical-device identity, so a
//! later process's `vkCreateComputePipelines` becomes a driver cache hit
//! instead of a full shader recompile (M6.5). Mirrors `backend-wgpu`'s own
//! `PlCache` - same env var, same atomic-rename persistence, same "never
//! trust a mismatched blob" discipline - for the native-Vulkan (ash) path,
//! which reaches the driver directly and so cannot share `wgpu::PipelineCache`
//! even when both backends open the same physical card.
//!
//! Spec header layout (`VkPipelineCacheHeaderVersionOne`, Vulkan 1.3 §10.2),
//! all fields little-endian, 32 bytes total:
//! `[headerSize:u32][headerVersion:u32][vendorID:u32][deviceID:u32][pipelineCacheUUID:16]`.
//! `vkCreatePipelineCache` already discards a mismatched blob safely per the
//! spec, but this module checks the header itself before ever offering bytes
//! to the driver, rather than trusting that behaviour blindly.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

const HEADER_LEN: usize = 32;
const HEADER_VERSION_ONE: u32 = 1;

/// Test-only override of the persist directory, exactly mirroring
/// `gpu_core::roof`'s own `DIR_OVERRIDE` (this crate cannot depend on
/// `gpu-core`, which sits above it in the dependency graph, so the pattern is
/// duplicated rather than shared). A test that sets this and a concurrently
/// running test that builds its own `VkContext` while the override is live
/// pick up the same directory - the identical, already-accepted hazard
/// `gpu_core::roof`'s tests carry; callers of this override are expected to
/// clear it (`None`) when done, same as that precedent.
static DIR_OVERRIDE: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Point every `VkContext` built in this process at `dir` instead of the
/// real `BRAIN_PIPELINE_CACHE_DIR`/`~/.cache/brain` - or clear the override
/// with `None`. Test-only.
pub fn set_dir_override(dir: Option<PathBuf>) {
    *DIR_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) = dir;
}

fn dir() -> Option<PathBuf> {
    if let Some(d) = DIR_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()).clone() {
        return Some(d);
    }
    if let Ok(d) = std::env::var("BRAIN_PIPELINE_CACHE_DIR") {
        return Some(d.into());
    }
    if let Ok(d) = std::env::var("XDG_CACHE_HOME") {
        return Some(Path::new(&d).join("brain"));
    }
    std::env::var("HOME").ok().map(|h| Path::new(&h).join(".cache/brain"))
}

/// Where this device's cache blob lives, or `None` if there is nowhere to
/// persist. Keyed by `(vendorID, deviceID, pipelineCacheUUID)` - the triple
/// the Vulkan spec guarantees changes whenever compiled-pipeline
/// compatibility changes (a driver update rotates `pipelineCacheUUID` even on
/// the same silicon), which is exactly the granularity a cache file should
/// invalidate at. `vk-` prefixed so it never collides with `backend-wgpu`'s
/// own `PlCache` file in the same directory.
pub fn path(vendor_id: u32, device_id: u32, uuid: [u8; 16]) -> Option<PathBuf> {
    let hex: String = uuid.iter().map(|b| format!("{b:02x}")).collect();
    Some(dir()?.join(format!("vk-pipeline-cache-{vendor_id:08x}-{device_id:08x}-{hex}.bin")))
}

/// Whether `data`'s own embedded spec header names this exact device -
/// checked independently of the driver's own (spec-mandated) validation.
fn header_matches(data: &[u8], vendor_id: u32, device_id: u32, uuid: [u8; 16]) -> bool {
    if data.len() < HEADER_LEN {
        return false;
    }
    let u32_at = |off: usize| u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    u32_at(4) == HEADER_VERSION_ONE && u32_at(8) == vendor_id && u32_at(12) == device_id && data[16..32] == uuid
}

/// Load the persisted blob at `path`, or `None` if it does not exist, cannot
/// be read, or its own header does not name this exact
/// `(vendorID, deviceID, pipelineCacheUUID)` - a stale file from a different
/// card or driver build is never even offered to `vkCreatePipelineCache`.
pub fn load(path: &Path, vendor_id: u32, device_id: u32, uuid: [u8; 16]) -> Option<Vec<u8>> {
    let data = std::fs::read(path).ok()?;
    header_matches(&data, vendor_id, device_id, uuid).then_some(data)
}

/// Persist `data` at `path` via write-then-rename (atomic on the same
/// filesystem), so a concurrent reader never observes a torn write.
/// Best-effort: a read-only filesystem just loses the warm start, same
/// contract as `backend-wgpu`'s `PlCache::persist`.
pub fn persist(path: &Path, data: &[u8]) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, data).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header(vendor_id: u32, device_id: u32, uuid: [u8; 16]) -> Vec<u8> {
        let mut h = vec![0u8; HEADER_LEN];
        h[0..4].copy_from_slice(&32u32.to_le_bytes());
        h[4..8].copy_from_slice(&HEADER_VERSION_ONE.to_le_bytes());
        h[8..12].copy_from_slice(&vendor_id.to_le_bytes());
        h[12..16].copy_from_slice(&device_id.to_le_bytes());
        h[16..32].copy_from_slice(&uuid);
        h
    }

    #[test]
    fn header_matches_rejects_wrong_identity_and_truncated_data() {
        let uuid = [7u8; 16];
        let good = make_header(0x8086, 0x7d55, uuid);

        assert!(header_matches(&good, 0x8086, 0x7d55, uuid));
        assert!(!header_matches(&good, 0x10de, 0x7d55, uuid), "wrong vendor must not match");
        assert!(!header_matches(&good, 0x8086, 0x1234, uuid), "wrong device must not match");
        assert!(!header_matches(&good, 0x8086, 0x7d55, [9u8; 16]), "wrong uuid must not match");
        assert!(!header_matches(&good[..16], 0x8086, 0x7d55, uuid), "truncated header must not match");
    }

    #[test]
    fn load_round_trips_a_matching_blob_and_rejects_a_foreign_one() {
        let dir = std::env::temp_dir().join(format!(
            "brain-vk-plcache-unit-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let uuid = [3u8; 16];
        let mut blob = make_header(0x8086, 0x7d55, uuid);
        blob.extend_from_slice(b"fake pipeline payload");
        let p = dir.join("test.bin");
        persist(&p, &blob);

        assert_eq!(load(&p, 0x8086, 0x7d55, uuid), Some(blob), "a matching header must round-trip the whole blob");
        assert!(
            load(&p, 0x10de, 0x7d55, uuid).is_none(),
            "a blob whose header names a different vendor must be ignored"
        );
        assert!(load(&dir.join("missing.bin"), 0x8086, 0x7d55, uuid).is_none(), "a missing file must load as None");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn path_differs_by_device_identity_and_never_collides_with_wgpus_own_file_name() {
        let uuid_a = [1u8; 16];
        let uuid_b = [2u8; 16];
        let dir = std::env::temp_dir().join(format!(
            "brain-vk-plcache-path-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        set_dir_override(Some(dir));
        let pa = path(0x8086, 0x7d55, uuid_a).unwrap();
        let pb = path(0x8086, 0x7d55, uuid_b).unwrap();
        assert_ne!(pa, pb, "distinct pipelineCacheUUIDs must resolve to distinct files");
        assert!(
            pa.file_name().unwrap().to_string_lossy().starts_with("vk-pipeline-cache-"),
            "must be namespaced away from backend-wgpu's own cache-file naming"
        );
        set_dir_override(None);
    }
}
