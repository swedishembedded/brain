// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The artifact inventory: every weight-shaped file or directory under a
//! models root, however it got there - canonical `<vendor>/<repo>/...`,
//! a bare vendor-flat file (`<vendor>/foo.gguf`, no repo directory at all),
//! or a legacy root-flat file (`<root>/foo.gguf`) - plus whether each one
//! is actually complete.
//!
//! [`Store`](crate::Store) answers "what can I load right now" for the one
//! canonical layout it knows; [`scan`] answers "what weight-shaped bytes
//! exist on this disk at all", which a resolver needs in order to notice a
//! checkpoint a user copied in by hand rather than fetched through this
//! crate. The two are deliberately independent views over the same
//! directory, not a replacement of one by the other.
//!
//! Swedish Embedded AB implements model-store discovery layers like this one
//! for clients running mixed fleets of hand-placed and fetched checkpoints.
//! If your team needs a real inventory of what is actually loadable on a
//! machine, you can procure our services by emailing info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The on-disk shape one artifact record describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactKind {
    Gguf,
    Safetensors,
    /// A directory holding a foreign (non-brain) HF checkpoint: `config.json`
    /// plus one or more `model*.safetensors` files, or a
    /// `model.safetensors.index.json` shard set - the loader takes the
    /// directory, not its individual shard files, so this is ONE record
    /// regardless of how many shards it collapses.
    HfDir,
    TokenizerJson,
    /// A bare `model_index.json` (diffusers pipeline manifest) - evidence
    /// that a pipeline lives nearby, never itself weights.
    PipelineIndex,
    Opaque,
}

/// Whether an artifact's bytes are actually all there.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Completeness {
    Complete,
    /// A `.part` sibling exists (or, for a directory record, one of its
    /// members does) - `final_path` is the name the download would have
    /// renamed to on success.
    Partial { final_path: PathBuf },
    /// The header parses, but declares more bytes than the file(s) actually
    /// hold on disk - a hand-copied file that was cut short, with no `.part`
    /// sibling to give it away.
    Truncated { declared: u64, actual: u64 },
    /// The header itself could not be read (bad magic, corrupt JSON, I/O
    /// error, …) - the message is diagnostic, not matched on.
    Unreadable(String),
}

/// One artifact found on disk.
#[derive(Clone, Debug, PartialEq)]
pub struct ArtifactRecord {
    /// Absolute path to the artifact itself - a file, or (for
    /// [`ArtifactKind::HfDir`]) the directory.
    pub path: PathBuf,
    pub size: u64,
    pub mtime_ns: u64,
    pub kind: ArtifactKind,
    pub completeness: Completeness,
}

impl ArtifactRecord {
    /// A record a resolver may actually build a model from.
    pub fn usable(&self) -> bool {
        matches!(self.completeness, Completeness::Complete)
    }
}

// ===================== the on-disk cache =====================

const CACHE_FILE: &str = ".brain-inventory.json";
const CACHE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct CacheFile {
    version: u32,
    root: PathBuf,
    entries: Vec<CacheEntry>,
}

#[derive(Clone, Serialize, Deserialize)]
struct CacheEntry {
    /// Relative to `root`, so the cache survives the store being moved to a
    /// different absolute path (only the file CONTENTS would need
    /// re-probing then, and mtimes would have changed anyway).
    rel: PathBuf,
    size: u64,
    mtime_ns: u64,
    kind: ArtifactKind,
    completeness: Completeness,
}

fn load_cache(root: &Path) -> BTreeMap<PathBuf, CacheEntry> {
    let bytes = match std::fs::read(root.join(CACHE_FILE)) {
        Ok(b) => b,
        Err(_) => return BTreeMap::new(),
    };
    let Ok(cache) = serde_json::from_slice::<CacheFile>(&bytes) else {
        return BTreeMap::new();
    };
    if cache.version != CACHE_VERSION || cache.root != root {
        return BTreeMap::new();
    }
    cache.entries.into_iter().map(|e| (e.rel.clone(), e)).collect()
}

/// Best-effort: a read-only store (or a full disk) just means every future
/// scan re-probes from scratch, not a hard failure.
fn save_cache(root: &Path, entries: &[CacheEntry]) {
    let cache = CacheFile { version: CACHE_VERSION, root: root.to_path_buf(), entries: entries.to_vec() };
    let Ok(bytes) = serde_json::to_vec_pretty(&cache) else { return };
    let tmp = root.join(format!("{CACHE_FILE}.tmp"));
    if std::fs::write(&tmp, &bytes).is_err() {
        return;
    }
    std::fs::rename(&tmp, root.join(CACHE_FILE)).ok();
}

// ===================== probing (the completeness check) =====================

/// `(size, mtime_ns)` for `path` - the whole staleness signal the cache
/// trusts. Never a content hash: this crate's stores run 50+ GB, and hashing
/// them on every CLI invocation would be a serious, pointless regression.
fn stat(path: &Path) -> std::io::Result<(u64, u64)> {
    let m = std::fs::symlink_metadata(path)?;
    let mtime_ns = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as u64).unwrap_or(0);
    Ok((m.len(), mtime_ns))
}

/// Probe one `.gguf`/`.safetensors` file that is NOT a `.part` - reads only
/// its header (never tensor bytes) to decide [`Completeness`]: a streamed
/// download can be cut short partway through its tensor data with no `.part`
/// sibling to give it away, so the header's own declared extent is checked
/// against what actually landed on disk.
fn probe_streamed(path: &Path, kind: ArtifactKind) -> Completeness {
    let actual = match std::fs::metadata(path) {
        Ok(m) => m.len(),
        Err(e) => return Completeness::Unreadable(e.to_string()),
    };
    let declared = match kind {
        ArtifactKind::Gguf => checkpoint::gguf::declared_data_extent(path.to_string_lossy().as_ref()),
        ArtifactKind::Safetensors => checkpoint::st::declared_data_extent(path.to_string_lossy().as_ref()).map_err(|e| e.to_string()),
        _ => unreachable!("probe_streamed only ever called for Gguf/Safetensors"),
    };
    match declared {
        Ok(declared) if actual < declared => Completeness::Truncated { declared, actual },
        Ok(_) => Completeness::Complete,
        Err(e) => Completeness::Unreadable(e),
    }
}

/// [`Completeness`] for a file that is NOT a `.part` and not a streamed
/// checkpoint (`Gguf`/`Safetensors`) - a `tokenizer.json`/`model_index.json`
/// manifest is read whole rather than streamed, so it carries no "declared
/// extent" a truncated download could fall short of; a stat that succeeds at
/// all means the file is there to read, and whether its bytes actually parse
/// is [`crate::resolve::ArchSpec::classify`]'s job, not the inventory's.
fn probe_whole_file(path: &Path) -> Completeness {
    match std::fs::metadata(path) {
        Ok(_) => Completeness::Complete,
        Err(e) => Completeness::Unreadable(e.to_string()),
    }
}

/// Build one record for `path`, reusing `cache` when `(size, mtime_ns)`
/// hasn't changed. `rel` is `path` relative to the scan root (the cache
/// key). `probed` is called exactly once per file this function actually
/// re-reads the header of - tests use it to prove the cache is honored.
fn record_for(path: &Path, rel: &Path, kind: ArtifactKind, cache: &BTreeMap<PathBuf, CacheEntry>, probed: &mut dyn FnMut(&Path)) -> Option<ArtifactRecord> {
    let (size, mtime_ns) = stat(path).ok()?;
    if let Some(cached) = cache.get(rel) {
        if cached.size == size && cached.mtime_ns == mtime_ns && cached.kind == kind {
            return Some(ArtifactRecord { path: path.to_path_buf(), size, mtime_ns, kind, completeness: cached.completeness.clone() });
        }
    }
    probed(path);
    let completeness = if is_part_file(path) {
        Completeness::Partial { final_path: final_path_of(path) }
    } else if matches!(kind, ArtifactKind::Gguf | ArtifactKind::Safetensors) {
        probe_streamed(path, kind)
    } else {
        probe_whole_file(path)
    };
    Some(ArtifactRecord { path: path.to_path_buf(), size, mtime_ns, kind, completeness })
}

fn is_part_file(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "part")
}

/// The path a `.part` file would have been renamed to on a successful
/// download - `fetch::stream_to_file`'s own naming, mirrored here rather
/// than imported (this crate already owns both sides of that convention).
fn final_path_of(part_path: &Path) -> PathBuf {
    part_path.with_extension("")
}

fn kind_of_extension(fname: &str) -> Option<ArtifactKind> {
    if fname.ends_with(".gguf") || fname.ends_with(".gguf.part") {
        Some(ArtifactKind::Gguf)
    } else if fname.ends_with(".safetensors") || fname.ends_with(".safetensors.part") {
        Some(ArtifactKind::Safetensors)
    } else {
        None
    }
}

// ===================== the walk =====================

/// Every artifact found under `root`: canonical `<vendor>/<repo>/...`,
/// vendor-flat (`<vendor>/*.gguf`/`*.safetensors`), and legacy root-flat
/// (`<root>/*.gguf`/`*.safetensors`) files, in one flat list. Reuses
/// `<root>/.brain-inventory.json` when present and matching this root,
/// re-probing only entries whose `(size, mtime_ns)` changed, are new, or
/// whose path no longer exists (dropped). Never follows symlinks.
pub fn scan(root: &Path) -> Vec<ArtifactRecord> {
    scan_with_probe_hook(root, &mut |_| {})
}

fn scan_with_probe_hook(root: &Path, probed: &mut dyn FnMut(&Path)) -> Vec<ArtifactRecord> {
    let cache = load_cache(root);
    let mut out = Vec::new();
    walk_root(root, root, &cache, probed, &mut out);

    let entries: Vec<CacheEntry> = out
        .iter()
        .filter_map(|r| {
            let rel = r.path.strip_prefix(root).ok()?.to_path_buf();
            Some(CacheEntry { rel, size: r.size, mtime_ns: r.mtime_ns, kind: r.kind, completeness: r.completeness.clone() })
        })
        .collect();
    save_cache(root, &entries);

    out
}

fn is_symlink(entry: &std::fs::DirEntry) -> bool {
    entry.file_type().map(|t| t.is_symlink()).unwrap_or(false)
}

/// Top level: each entry under `root` is either a vendor directory
/// (canonical + vendor-flat layouts live under it) or a legacy root-flat
/// weight file sitting directly in `root`.
fn walk_root(root: &Path, scan_root: &Path, cache: &BTreeMap<PathBuf, CacheEntry>, probed: &mut dyn FnMut(&Path), out: &mut Vec<ArtifactRecord>) {
    let Ok(entries) = std::fs::read_dir(root) else { return };
    for entry in entries.flatten() {
        if is_symlink(&entry) {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            walk_vendor_dir(&path, scan_root, cache, probed, out);
        } else if let Some(fname) = path.file_name().and_then(|s| s.to_str()) {
            // Root-flat legacy layout: a bare weight file directly under root.
            if let Some(kind) = kind_of_extension(fname) {
                push_file_record(&path, scan_root, kind, cache, probed, out);
            }
        }
    }
}

/// One `<vendor>/` directory: vendor-flat loose weight files directly inside
/// it, and repo subdirectories (the canonical layout).
fn walk_vendor_dir(vendor_dir: &Path, scan_root: &Path, cache: &BTreeMap<PathBuf, CacheEntry>, probed: &mut dyn FnMut(&Path), out: &mut Vec<ArtifactRecord>) {
    let Ok(entries) = std::fs::read_dir(vendor_dir) else { return };
    for entry in entries.flatten() {
        if is_symlink(&entry) {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            walk_repo_dir(&path, scan_root, cache, probed, out, 0);
        } else if let Some(fname) = path.file_name().and_then(|s| s.to_str()) {
            if let Some(kind) = kind_of_extension(fname) {
                push_file_record(&path, scan_root, kind, cache, probed, out);
            }
        }
    }
}

/// Deepest directory level the walk will recurse into below a repo
/// directory (`<repo>/<component-subdir>/<shard files>` - files are leaves,
/// not a further level, so `1` covers it; kept generous rather than exact).
const MAX_COMPONENT_DEPTH: u32 = 3;

/// A repo directory (or, recursively, a component subdirectory inside one):
/// collapses to a single [`ArtifactKind::HfDir`] record if it qualifies,
/// otherwise walks its files and recurses into subdirectories up to
/// [`MAX_COMPONENT_DEPTH`].
fn walk_repo_dir(dir: &Path, scan_root: &Path, cache: &BTreeMap<PathBuf, CacheEntry>, probed: &mut dyn FnMut(&Path), out: &mut Vec<ArtifactRecord>, depth: u32) {
    if let Some(record) = hfdir_record(dir, scan_root, cache, probed) {
        out.push(record);
        // The directory collapsed to one record for its WEIGHTS, but a
        // `tokenizer.json` sitting right beside `config.json` in that same
        // directory is a role of its own (an architecture's text-encoder
        // role and its tokenizer role are ordinarily two different roles of
        // the SAME checkpoint) - it must still be independently
        // discoverable rather than disappearing into the one HfDir record.
        let tok = dir.join("tokenizer.json");
        if tok.is_file() {
            push_file_record(&tok, scan_root, ArtifactKind::TokenizerJson, cache, probed, out);
        }
        return;
    }
    if depth > MAX_COMPONENT_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        if is_symlink(&entry) {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            walk_repo_dir(&path, scan_root, cache, probed, out, depth + 1);
            continue;
        }
        let Some(fname) = path.file_name().and_then(|s| s.to_str()) else { continue };
        if let Some(kind) = kind_of_extension(fname) {
            push_file_record(&path, scan_root, kind, cache, probed, out);
        } else if fname == "tokenizer.json" {
            push_file_record(&path, scan_root, ArtifactKind::TokenizerJson, cache, probed, out);
        } else if fname == "model_index.json" {
            push_file_record(&path, scan_root, ArtifactKind::PipelineIndex, cache, probed, out);
        }
    }
}

fn push_file_record(path: &Path, scan_root: &Path, kind: ArtifactKind, cache: &BTreeMap<PathBuf, CacheEntry>, probed: &mut dyn FnMut(&Path), out: &mut Vec<ArtifactRecord>) {
    let Ok(rel) = path.strip_prefix(scan_root) else { return };
    if let Some(r) = record_for(path, rel, kind, cache, probed) {
        out.push(r);
    }
}

/// The shard filenames a directory declares, from `model.safetensors.index.json`
/// if present (its `weight_map` values, deduped), else every loose
/// `model*.safetensors` file sitting directly in it.
fn shard_filenames(dir: &Path) -> Vec<String> {
    let index = dir.join("model.safetensors.index.json");
    if let Ok(bytes) = std::fs::read(&index) {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            if let Some(map) = v.get("weight_map").and_then(|m| m.as_object()) {
                let mut files: Vec<String> = map.values().filter_map(|v| v.as_str().map(str::to_string)).collect();
                files.sort();
                files.dedup();
                return files;
            }
        }
    }
    let mut files: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|n| n.starts_with("model") && n.ends_with(".safetensors"))
        .collect();
    files.sort();
    files
}

/// `Some` (whether or not it resolves cleanly) when `dir` qualifies as a
/// foreign HF checkpoint directory: `config.json` plus a shard set. `None`
/// when `dir` doesn't have that shape at all, so the caller falls through to
/// walking it file-by-file instead.
fn hfdir_record(dir: &Path, scan_root: &Path, cache: &BTreeMap<PathBuf, CacheEntry>, probed: &mut dyn FnMut(&Path)) -> Option<ArtifactRecord> {
    if !dir.join("config.json").is_file() {
        return None;
    }
    let has_index = dir.join("model.safetensors.index.json").is_file();
    let shards = shard_filenames(dir);
    if !has_index && shards.is_empty() {
        return None;
    }

    let rel = dir.strip_prefix(scan_root).ok()?.to_path_buf();
    let (size, mtime_ns) = stat(dir).ok()?;
    // A directory record's own (size, mtime_ns) is the directory entry's,
    // which changes whenever a member is added/removed/renamed but NOT when
    // a member's own bytes change in place -- acceptable here since a shard
    // file is only ever written once, atomically, by `save_safetensors`.
    if let Some(cached) = cache.get(&rel) {
        if cached.size == size && cached.mtime_ns == mtime_ns && cached.kind == ArtifactKind::HfDir {
            return Some(ArtifactRecord { path: dir.to_path_buf(), size, mtime_ns, kind: ArtifactKind::HfDir, completeness: cached.completeness.clone() });
        }
    }
    probed(dir);
    Some(ArtifactRecord { path: dir.to_path_buf(), size, mtime_ns, kind: ArtifactKind::HfDir, completeness: hfdir_completeness(dir, &shards) })
}

fn hfdir_completeness(dir: &Path, shards: &[String]) -> Completeness {
    let mut declared_total = 0u64;
    let mut actual_total = 0u64;
    for name in shards {
        let path = dir.join(name);
        if !path.is_file() {
            // A declared shard missing its final name is exactly what an
            // interrupted fetch leaves behind, whether or not a `.part`
            // sibling happens to still be sitting there.
            return Completeness::Partial { final_path: path };
        }
        match checkpoint::st::declared_data_extent(path.to_string_lossy().as_ref()) {
            Ok(declared) => {
                let actual = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                declared_total += declared;
                actual_total += actual;
            }
            Err(e) => return Completeness::Unreadable(format!("{}: {e}", path.display())),
        }
    }
    if actual_total < declared_total {
        Completeness::Truncated { declared: declared_total, actual: actual_total }
    } else {
        Completeness::Complete
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-inventory-test-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_gguf(path: &Path) {
        let raw: Vec<u8> = (0..16i32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        checkpoint::gguf_write::write(
            path.to_str().unwrap(),
            &[("general.architecture".to_string(), checkpoint::gguf::GgufValue::String("toy".to_string()))],
            &[checkpoint::gguf_write::TensorOut { name: "w".to_string(), shape: vec![16], ty: checkpoint::gguf::GgmlType::F32.id(), data: raw }],
            32,
        )
        .unwrap();
    }

    fn tiny_safetensors(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &[("w".to_string(), vec![4], vec![1.0, 2.0, 3.0, 4.0])], &serde_json::json!({}), None).unwrap();
    }

    /// The scenario that hides a real, complete checkpoint from today's
    /// `Store::scan`: a bare `.gguf` sitting directly under `<root>/<vendor>/`,
    /// no repo subdirectory at all (`Store::scan` only ever looks inside a
    /// `<vendor>/<repo>/` directory, so it never even lists this file).
    #[test]
    fn vendor_flat_loose_gguf_is_inventoried() {
        let root = scratch_root("vendor-flat");
        tiny_gguf(&root.join("unsloth").join("foo.gguf"));

        let found = scan(&root);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].kind, ArtifactKind::Gguf);
        assert_eq!(found[0].completeness, Completeness::Complete);
        assert_eq!(found[0].path, root.join("unsloth").join("foo.gguf"));
    }

    /// A real `tokenizer.json`/`model_index.json` sitting inside a repo
    /// directory - `walk_repo_dir` recognizes both filenames explicitly, but
    /// `probe_file` only ever knew how to compute a declared-vs-actual byte
    /// extent for `Gguf`/`Safetensors`. Neither kind streams: a manifest this
    /// small is read whole, so "complete" is just "readable" (an interrupted
    /// download is still caught by its `.part` sibling, same as any other
    /// kind - see `a_part_file_is_partial_and_absent_from_a_usable_filter`).
    #[test]
    fn a_real_tokenizer_json_and_model_index_json_scan_without_panicking() {
        let root = scratch_root("tokenizer-and-index");
        let repo = root.join("black-forest-labs").join("FLUX.2-klein-4B");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::write(repo.join("model_index.json"), br#"{"_class_name": "Flux2KleinPipeline"}"#).unwrap();
        let tok_dir = repo.join("tokenizer");
        std::fs::create_dir_all(&tok_dir).unwrap();
        std::fs::write(tok_dir.join("tokenizer.json"), br#"{"version": "1.0"}"#).unwrap();

        let found = scan(&root);
        let by_kind = |k: ArtifactKind| found.iter().filter(|r| r.kind == k).count();
        assert_eq!(by_kind(ArtifactKind::PipelineIndex), 1, "{found:?}");
        assert_eq!(by_kind(ArtifactKind::TokenizerJson), 1, "{found:?}");
        for r in &found {
            assert_eq!(r.completeness, Completeness::Complete, "{r:?}");
            assert!(r.usable());
        }
    }

    #[test]
    fn a_part_file_is_partial_and_absent_from_a_usable_filter() {
        let root = scratch_root("part-file");
        let part = root.join("Tongyi-MAI").join("Z-Image-Turbo").join("transformer").join("x.safetensors.part");
        std::fs::create_dir_all(part.parent().unwrap()).unwrap();
        std::fs::write(&part, b"partial bytes").unwrap();

        let found = scan(&root);
        assert_eq!(found.len(), 1);
        let r = &found[0];
        assert_eq!(r.kind, ArtifactKind::Safetensors);
        match &r.completeness {
            Completeness::Partial { final_path } => assert_eq!(final_path, &part.parent().unwrap().join("x.safetensors")),
            other => panic!("expected Partial, got {other:?}"),
        }
        assert!(!r.usable());
    }

    #[test]
    fn a_truncated_gguf_is_detected_from_its_header_without_hashing() {
        let root = scratch_root("truncated-gguf");
        let path = root.join("unsloth").join("foo.gguf");
        tiny_gguf(&path);

        let declared = checkpoint::gguf::declared_data_extent(path.to_str().unwrap()).unwrap();
        let tensor_bytes = 16u64 * 4; // 16 F32 elements
        let data_start = declared - tensor_bytes;
        // Truncate to EXACTLY the header/data boundary: zero tensor bytes
        // remain on disk, so a correct Truncated{declared, actual} here is
        // only possible without ever reading past the header.
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(data_start).unwrap();
        drop(f);

        let found = scan(&root);
        assert_eq!(found.len(), 1);
        match &found[0].completeness {
            Completeness::Truncated { declared: d, actual } => {
                assert_eq!(*d, declared);
                assert_eq!(*actual, data_start);
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
        assert!(!found[0].usable());
    }

    #[test]
    fn an_hf_shard_set_collapses_to_one_hfdir_record_not_n_shards() {
        let root = scratch_root("hfdir-shards");
        let dir = root.join("Qwen").join("Qwen3-8B");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        tiny_safetensors(&dir.join("model-00001-of-00002.safetensors"));
        tiny_safetensors(&dir.join("model-00002-of-00002.safetensors"));
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let found = scan(&root);
        assert_eq!(found.len(), 1, "expected one HfDir record, got {found:?}");
        assert_eq!(found[0].kind, ArtifactKind::HfDir);
        assert_eq!(found[0].path, dir);
        assert_eq!(found[0].completeness, Completeness::Complete);
    }

    /// An HF checkpoint directory that collapses to one `HfDir` record
    /// returned immediately, before ever walking its own files -- so a
    /// `tokenizer.json` sitting right next to `config.json` in that same
    /// directory never became a record of its own, and a role that wants
    /// "this checkpoint's weights" and a role that wants "this checkpoint's
    /// tokenizer" (two different roles of the same architecture, the
    /// ordinary case) could never both resolve to the one real repo.
    #[test]
    fn a_tokenizer_co_located_with_a_collapsed_hfdir_is_still_its_own_record() {
        let root = scratch_root("hfdir-with-tokenizer");
        let dir = root.join("Qwen").join("Qwen3-8B");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        tiny_safetensors(&dir.join("model.safetensors"));
        std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();

        let found = scan(&root);
        assert_eq!(found.iter().filter(|r| r.kind == ArtifactKind::HfDir).count(), 1, "{found:?}");
        let tok = found.iter().find(|r| r.kind == ArtifactKind::TokenizerJson).unwrap_or_else(|| panic!("no TokenizerJson record, got {found:?}"));
        assert_eq!(tok.path, dir.join("tokenizer.json"));
    }

    #[test]
    fn the_index_is_reused_when_nothing_moved_and_reprobed_per_file_on_change() {
        let root = scratch_root("cache-reuse");
        let untouched = root.join("unsloth").join("untouched.gguf");
        let touched = root.join("unsloth").join("touched.gguf");
        tiny_gguf(&untouched);
        tiny_gguf(&touched);

        let mut probes = Vec::new();
        scan_with_probe_hook(&root, &mut |p| probes.push(p.to_path_buf()));
        assert_eq!(probes.len(), 2, "first scan must probe both files: {probes:?}");

        // Bump `touched`'s size AND mtime (forced forward, so this can't
        // land in the same coarse-grained tick the filesystem already
        // recorded) without touching `untouched` at all.
        let mut bytes = std::fs::read(&touched).unwrap();
        bytes.push(0);
        std::fs::write(&touched, &bytes).unwrap();
        let future = std::time::SystemTime::now() + std::time::Duration::from_secs(5);
        std::fs::File::open(&touched).unwrap().set_modified(future).unwrap();

        let mut probes2 = Vec::new();
        let found = scan_with_probe_hook(&root, &mut |p| probes2.push(p.to_path_buf()));
        assert_eq!(probes2, vec![touched.clone()], "only the changed file should be re-probed, got {probes2:?}");
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn an_index_from_a_different_root_is_discarded_whole() {
        let root = scratch_root("stale-root");
        tiny_gguf(&root.join("unsloth").join("foo.gguf"));

        // Write a cache file whose `root` field points somewhere else.
        let bogus = CacheFile { version: CACHE_VERSION, root: PathBuf::from("/nowhere/at/all"), entries: vec![] };
        std::fs::write(root.join(CACHE_FILE), serde_json::to_vec(&bogus).unwrap()).unwrap();

        // Must not error, and must still find the real file via a fresh walk.
        let found = scan(&root);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].kind, ArtifactKind::Gguf);
    }
}
