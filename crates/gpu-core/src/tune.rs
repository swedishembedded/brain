// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! File-backed persistence for measured kernel choices (S5).
//!
//! Keyed by adapter identity plus a caller-supplied fingerprint of the kernel
//! sources being chosen between — edit a candidate kernel and the old winners
//! stop applying by *filename*, rather than by trusting stale measurements.
//! A missing or unparseable file is ignored, never trusted; writes are atomic
//! (tmp + rename) and best-effort — a read-only filesystem just loses
//! persistence, not correctness.

use backend_api::select::TuneStore;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

pub struct FileTuneStore {
    path: PathBuf,
    map: Mutex<HashMap<String, String>>,
}

/// FNV-1a over kernel sources — the fingerprint half of the store's key.
pub fn source_fingerprint(sources: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for s in sources {
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// The one cache-directory resolution (`BRAIN_PIPELINE_CACHE_DIR` >
/// `XDG_CACHE_HOME/brain` > `~/.cache/brain`) — shared with `roof`'s persist
/// layer, which used to carry a verbatim copy.
pub(crate) fn cache_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("BRAIN_PIPELINE_CACHE_DIR") {
        return Some(d.into());
    }
    if let Ok(d) = std::env::var("XDG_CACHE_HOME") {
        return Some(std::path::Path::new(&d).join("brain"));
    }
    std::env::var("HOME").ok().map(|h| std::path::Path::new(&h).join(".cache/brain"))
}

impl FileTuneStore {
    /// The store for the adapter this process selected. `None` when no wgpu
    /// adapter was recorded (pure CPU / native-Vulkan runs then tune per
    /// process, memo-only) or there is nowhere to persist.
    pub fn for_adapter(source_hash: u64) -> Option<FileTuneStore> {
        let (desc, _) = crate::adapter_info()?;
        let dir = cache_dir()?;
        let slug: String =
            desc.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        let path = dir.join(format!("tune-{slug}-{source_hash:016x}.txt"));
        let mut map = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    map.insert(k.to_string(), v.to_string());
                }
            }
        }
        Some(FileTuneStore { path, map: Mutex::new(map) })
    }
}

impl TuneStore for FileTuneStore {
    fn load(&self, key: &str) -> Option<String> {
        self.map.lock().unwrap_or_else(|e| e.into_inner()).get(key).cloned()
    }
    /// Merge-on-save, I/O OUTSIDE the lock: the old shape held the mutex
    /// across create_dir_all + write + rename (an autotune sweep like
    /// `qwen3::serve::tune_i8` saves up to ~48 times, each a whole-file
    /// rewrite under the lock), and wrote only THIS process's map — two
    /// stores in one process (or two processes) silently dropped each
    /// other's entries. Now: update the in-memory map under the lock, snap a
    /// copy, then merge with whatever is on disk before the atomic rename —
    /// last-writer-wins per KEY instead of per FILE. (`unwrap_or_else
    /// (into_inner)` is the crate-wide poison policy; bare `unwrap` here was
    /// the one divergence.)
    fn save(&self, key: &str, value: &str) {
        let snapshot: HashMap<String, String> = {
            let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
            map.insert(key.to_string(), value.to_string());
            map.clone()
        };
        // Merge with concurrent writers' on-disk entries (keys we don't hold).
        let mut merged: HashMap<String, String> = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&self.path) {
            for line in text.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    merged.insert(k.to_string(), v.to_string());
                }
            }
        }
        merged.extend(snapshot);
        let body: String = merged.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = self.path.with_extension("tmp");
        if std::fs::write(&tmp, body).is_ok() {
            let _ = std::fs::rename(&tmp, &self.path);
        }
    }
}
