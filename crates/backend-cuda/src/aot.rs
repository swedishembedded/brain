// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Ahead-of-time cubins: build them once, load them with the driver alone.
//!
//! Swedish Embedded AB implements deployment pipelines for GPU software for its
//! clients, including shipping compiled kernels to machines that carry a
//! driver and no toolchain. If your team needs expertise in making a CUDA
//! application run on a locked-down box without a compiler, you can procure
//! our services by sending an email to info@swedishembedded.com.
//!
//! # What this is for
//!
//! NVRTC ships with the CUDA toolkit, and a machine that can run CUDA does not
//! necessarily have a toolkit. The generated tier and the native kernels are
//! therefore compiled offline (`make cuda/aot`) into a directory of cubins plus
//! a manifest, and [`crate::exec::Context::compile_with`] consults it before
//! it ever asks NVRTC for anything.
//!
//! # The manifest
//!
//! A tab-separated text file (`manifest.tsv`), one row per image:
//!
//! ```text
//! brain-cuda-aot 1
//! abi 1
//! kernel  target  kind  toolchain  key  image_sha256  file
//! ```
//!
//! `kernel` and `toolchain` are for people; the loader matches on `key`, a hash
//! of everything that can change the machine code short of the compiler
//! version (source, entry point, target, flags, specialization, launch ABI -
//! see [`aot_key`]). So a stale cubin cannot be served for edited source: the
//! edited source has a different key and misses. `image_sha256` is checked
//! before an image reaches the driver, so a truncated or corrupted file is a
//! named error and not a driver fault.
//!
//! # What is loaded, in what order
//!
//! For one kernel on one device the loader tries, in order: the cubin for the
//! device's own architecture (with the arch-specific suffix first, where the
//! kernel asks for it), then the cubins for lower minors of the same major
//! (binary compatible), then - in `Context` - the disk cache and NVRTC, and
//! finally the PTX the build kept, which the driver compiles for whatever it is
//! running on. Every miss is reported by name when nothing works.

use crate::nvrtc::{self, Cc, Target};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The manifest's file name inside an AOT directory.
pub const MANIFEST_FILE: &str = "manifest.tsv";
/// Names the AOT directory, overriding the default under the cache directory.
pub const ENV_DIR: &str = "BRAIN_CUDA_AOT_DIR";
/// `0` switches the AOT lookup off, so a run exercises NVRTC regardless of
/// what a directory holds.
pub const ENV_ENABLE: &str = "BRAIN_CUDA_AOT";
const FORMAT_LINE: &str = "brain-cuda-aot 1";

/// What an image is: machine code for one real architecture, or portable PTX
/// the driver compiles at load time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Cubin,
    Ptx,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Kind::Cubin => "cubin",
            Kind::Ptx => "ptx",
        }
    }

    fn parse(s: &str) -> Option<Kind> {
        match s {
            "cubin" => Some(Kind::Cubin),
            "ptx" => Some(Kind::Ptx),
            _ => None,
        }
    }
}

/// One built image.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Entry {
    /// The kernel's registry name, for people.
    pub kernel: String,
    /// `sm_90`, `sm_90a`, or `compute_61` for PTX.
    pub target: String,
    pub kind: Kind,
    /// What built it, e.g. `nvrtc 12.9`, for people and for diagnosing a
    /// rejected image.
    pub toolchain: String,
    /// The lookup key; see [`aot_key`].
    pub key: String,
    /// Lowercase hex sha256 of the image file's bytes.
    pub image_sha256: String,
    /// The image's file name, relative to the manifest's directory.
    pub file: String,
}

/// The parsed manifest.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Manifest {
    pub entries: Vec<Entry>,
}

impl Manifest {
    /// Parse a manifest. Every failure names the line it is about: a manifest
    /// is read on every start of a process that finds one, and "could not
    /// parse" with no location would be impossible to act on.
    pub fn parse(text: &str) -> Result<Manifest, String> {
        let mut lines = text.lines().enumerate();
        match lines.next() {
            Some((_, l)) if l == FORMAT_LINE => {}
            other => {
                return Err(format!(
                    "line 1: expected `{FORMAT_LINE}`, found `{}` (a manifest from a different format version; rebuild with `make cuda/aot`)",
                    other.map(|(_, l)| l).unwrap_or("")
                ))
            }
        }
        match lines.next() {
            Some((_, l)) if l == format!("abi {}", nvrtc::CUBIN_ABI_VERSION) => {}
            Some((_, l)) => {
                return Err(format!(
                    "line 2: `{l}` but this build launches kernels with `abi {}`; the images were built for a different launch contract, rebuild with `make cuda/aot`",
                    nvrtc::CUBIN_ABI_VERSION
                ))
            }
            None => return Err("line 2: missing `abi` line".to_string()),
        }
        let mut entries = Vec::new();
        for (i, line) in lines {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let [kernel, target, kind, toolchain, key, image_sha256, file] = f[..] else {
                return Err(format!("line {}: expected 7 tab-separated fields, found {}", i + 1, f.len()));
            };
            let kind = Kind::parse(kind).ok_or_else(|| format!("line {}: unknown image kind `{kind}`", i + 1))?;
            if file.contains('/') || file.contains("..") || file.is_empty() {
                return Err(format!("line {}: image file `{file}` must be a plain file name inside the manifest's directory", i + 1));
            }
            entries.push(Entry {
                kernel: kernel.into(),
                target: target.into(),
                kind,
                toolchain: toolchain.into(),
                key: key.into(),
                image_sha256: image_sha256.into(),
                file: file.into(),
            });
        }
        Ok(Manifest { entries })
    }

    /// The text [`Self::parse`] reads back.
    pub fn render(&self) -> String {
        let mut out = format!("{FORMAT_LINE}\nabi {}\n# kernel\ttarget\tkind\ttoolchain\tkey\timage_sha256\tfile\n", nvrtc::CUBIN_ABI_VERSION);
        for e in &self.entries {
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                e.kernel,
                e.target,
                e.kind.as_str(),
                e.toolchain,
                e.key,
                e.image_sha256,
                e.file
            ));
        }
        out
    }

    /// Add `entry`, replacing what it supersedes: an earlier image of the same
    /// kernel for the same target and kind. The key is what changes when the
    /// source does, so a rebuild after an edit replaces the stale row instead
    /// of leaving a second one that can never match. Returns the file names
    /// that are no longer referenced.
    pub fn upsert(&mut self, entry: Entry) -> Vec<String> {
        let mut dropped = Vec::new();
        self.entries.retain(|e| {
            let supersede = e.kernel == entry.kernel && e.target == entry.target && e.kind == entry.kind;
            if supersede && e.file != entry.file {
                dropped.push(e.file.clone());
            }
            !supersede
        });
        self.entries.push(entry);
        dropped
    }
}

/// The manifest and the directory it describes.
#[derive(Debug)]
pub struct Store {
    dir: PathBuf,
    manifest: Manifest,
    by_key: HashMap<String, usize>,
}

impl Store {
    /// Read `dir`'s manifest. `Ok(None)` when there is none, which is the
    /// ordinary state of a machine nobody ran `make cuda/aot` on; `Err` when
    /// there is one and it cannot be used, because that is something a person
    /// should be told rather than something to silently route around.
    pub fn open(dir: &Path) -> Result<Option<Store>, String> {
        let path = dir.join(MANIFEST_FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("reading {}: {e}", path.display())),
        };
        let manifest = Manifest::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let by_key = manifest.entries.iter().enumerate().map(|(i, e)| (e.key.clone(), i)).collect();
        Ok(Some(Store { dir: dir.to_path_buf(), manifest, by_key }))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn entries(&self) -> &[Entry] {
        &self.manifest.entries
    }

    /// The entry built for exactly this key.
    pub fn find(&self, key: &str) -> Option<&Entry> {
        self.by_key.get(key).map(|&i| &self.manifest.entries[i])
    }

    /// The image `entry` names, verified against the manifest's checksum.
    pub fn image(&self, entry: &Entry) -> Result<Vec<u8>, String> {
        let path = self.dir.join(&entry.file);
        let bytes = std::fs::read(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        let got = sha256_hex(&bytes);
        if got != entry.image_sha256 {
            return Err(format!(
                "{} does not match its manifest checksum (manifest {}, file {got}); the image is truncated or was modified, rebuild with `make cuda/aot`",
                path.display(),
                entry.image_sha256
            ));
        }
        Ok(bytes)
    }

    /// The distinct targets this store holds an image for, for diagnostics.
    pub fn targets(&self) -> Vec<String> {
        let mut t: Vec<String> = self.manifest.entries.iter().map(|e| e.target.clone()).collect();
        t.sort();
        t.dedup();
        t
    }
}

/// Where the AOT images live: [`ENV_DIR`], else `cuda-aot` under the cache
/// directory every backend shares.
pub fn default_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var(ENV_DIR) {
        if !d.is_empty() {
            return Some(d.into());
        }
    }
    backend_api::cache_dir().map(|d| d.join("cuda-aot"))
}

/// The process-wide store, read once. `Ok(None)` means there is nothing to
/// consult (no manifest, or the lookup is switched off); `Err` carries why a
/// manifest that exists could not be used, for the error a failed lookup
/// ends up reporting.
pub fn global() -> Result<Option<&'static Store>, &'static str> {
    static STORE: std::sync::OnceLock<Result<Option<Store>, String>> = std::sync::OnceLock::new();
    let r = STORE.get_or_init(|| {
        if std::env::var(ENV_ENABLE).is_ok_and(|v| v == "0") {
            return Ok(None);
        }
        let Some(dir) = default_dir() else { return Ok(None) };
        let r = Store::open(&dir);
        if let Err(e) = &r {
            tracing::warn!("backend-cuda: ignoring the AOT manifest: {e}");
        }
        r
    });
    match r {
        Ok(s) => Ok(s.as_ref()),
        Err(e) => Err(e.as_str()),
    }
}

/// The lookup key for one image: hex sha256 over the source, entry point,
/// architecture name, compile flags, specialization macros and launch ABI
/// version.
///
/// This is [`nvrtc::cache_key`] minus the compiler version, which an ahead-of-
/// time consumer has no way to know (it is the point of this module that there
/// is no compiler on the box). The toolchain that built an image is recorded in
/// the manifest instead.
pub fn aot_key(src: &str, entry: &str, arch: &str, defines: &[(&str, &str)]) -> String {
    let mut sorted = defines.to_vec();
    sorted.sort_unstable();
    let mut h = Sha256::new();
    for field in [
        src.as_bytes(),
        entry.as_bytes(),
        arch.as_bytes(),
        nvrtc::flags_for_arch(arch, &sorted).join(" ").as_bytes(),
        format!("abi{}", nvrtc::CUBIN_ABI_VERSION).as_bytes(),
    ] {
        h.update((field.len() as u64).to_le_bytes());
        h.update(field);
    }
    format!("{:x}", h.finalize())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The cubin targets that can run on a device of capability `cc`, best first.
///
/// The arch-specific build comes first where the kernel wants it and one
/// exists. After it come the plain builds for `cc` and every lower minor of
/// the same major: a cubin is binary compatible upward within a major. Whether
/// the DRIVER agrees (it does not for every generation) is its call, which is
/// why the loader tries each candidate rather than trusting this list.
/// A kernel that requires the suffix has no other candidate.
pub fn cubin_targets(cc: Cc, features: nvrtc::ArchFeatures) -> Vec<Target> {
    use nvrtc::ArchFeatures::*;
    let mut out = Vec::new();
    if features != Portable && nvrtc::arch_specific_since(cc).is_some() {
        out.push(Target { cc, arch_specific: true });
    }
    if features != Required {
        out.extend((0..=cc.1).rev().map(|minor| Target::plain((cc.0, minor))));
    }
    out
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

/// One kernel to build.
#[derive(Clone, Debug)]
pub struct Job {
    /// The registry name, recorded in the manifest. A kernel is replaced by
    /// name on a rebuild, so two jobs that differ only in their specialization
    /// macros or entry point need different names.
    pub kernel: String,
    pub src: String,
    pub entry: String,
    pub defines: Vec<(String, String)>,
    /// The lowest capability the source is valid on; targets below it are
    /// skipped rather than failed.
    pub min_cc: Cc,
}

/// What to build each job for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub cubins: Vec<Target>,
    /// The virtual architecture of the portable PTX fallback, if any.
    pub ptx: Option<Cc>,
}

impl Plan {
    /// The targets a toolkit lane ships by default. CUDA 12 is the lane that
    /// still compiles for Pascal and Volta; CUDA 13 dropped them and starts at
    /// Hopper. The PTX fallback is for the oldest cubin in the lane, so every
    /// newer device can be served by the driver's own compiler.
    pub fn for_toolkit(nvrtc: (u32, u32)) -> Plan {
        if nvrtc.0 >= 13 {
            Plan { cubins: [(9, 0), (10, 0), (12, 0)].map(Target::plain).to_vec(), ptx: Some((9, 0)) }
        } else {
            Plan { cubins: [(6, 1), (7, 0), (8, 0), (8, 6), (8, 9), (9, 0)].map(Target::plain).to_vec(), ptx: Some((6, 1)) }
        }
    }
}

/// What a build did.
#[derive(Debug, Default)]
pub struct Report {
    pub built: usize,
    /// `kernel target: why`, for targets below a kernel's floor.
    pub skipped: Vec<String>,
}

/// Compile every job for every target of `plan` into `dir` and write the
/// manifest, merging into one that is already there (so the CUDA 12 and
/// CUDA 13 lanes can fill one directory in turn).
///
/// Jobs compile in parallel; NVRTC compilations are independent. Any compile
/// failure aborts the build with every failure listed - a kernel that does not
/// compile for a target the plan names is a defect to fix, not a gap to ship.
pub fn build(dir: &Path, jobs: &[Job], plan: &Plan) -> Result<Report, String> {
    let toolkit = nvrtc::version()?;
    let toolchain = format!("nvrtc {}.{}", toolkit.0, toolkit.1);
    std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;

    // Refuse a target the toolkit cannot emit before compiling hundreds of
    // kernels into the same error.
    for t in &plan.cubins {
        nvrtc::compile("extern \"C\" __global__ void brain_aot_probe() {}", t, &[])
            .map_err(|e| format!("toolkit {toolchain} cannot build for {}: {e}", t.name()))?;
    }

    // (job, arch, kind) work items.
    struct Item<'a> {
        job: &'a Job,
        arch: String,
        target: Option<Target>,
        ptx_cc: Option<Cc>,
    }
    let mut items = Vec::new();
    let mut report = Report::default();
    for job in jobs {
        for t in &plan.cubins {
            if t.cc < job.min_cc {
                report.skipped.push(format!("{} {}: below the kernel's floor {}.{}", job.kernel, t.name(), job.min_cc.0, job.min_cc.1));
            } else {
                items.push(Item { job, arch: t.name(), target: Some(*t), ptx_cc: None });
            }
        }
        if let Some(cc) = plan.ptx {
            let cc = cc.max(job.min_cc);
            items.push(Item { job, arch: format!("compute_{}{}", cc.0, cc.1), target: None, ptx_cc: Some(cc) });
        }
    }

    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::<(usize, Result<(Entry, Vec<u8>), String>)>::new());
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(items.len().max(1));
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(item) = items.get(i) else { break };
                let defines: Vec<(&str, &str)> = item.job.defines.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                let image = match (item.target, item.ptx_cc) {
                    (Some(t), _) => nvrtc::compile(&item.job.src, &t, &defines).map(|b| (b, Kind::Cubin)),
                    (None, Some(cc)) => nvrtc::compile_ptx(&item.job.src, cc, &defines).map(|b| (b, Kind::Ptx)),
                    (None, None) => unreachable!("an item is a cubin or a PTX build"),
                };
                let r = image.map(|(bytes, kind)| {
                    let key = aot_key(&item.job.src, &item.job.entry, &item.arch, &defines);
                    let ext = if kind == Kind::Ptx { "ptx" } else { "cubin" };
                    let file = format!("{}.{}.{}.{ext}", file_stem(&item.job.kernel), item.arch, &key[..12]);
                    let entry = Entry {
                        kernel: item.job.kernel.clone(),
                        target: item.arch.clone(),
                        kind,
                        toolchain: toolchain.clone(),
                        key,
                        image_sha256: sha256_hex(&bytes),
                        file,
                    };
                    (entry, bytes)
                });
                results.lock().unwrap_or_else(|e| e.into_inner()).push((i, r));
            });
        }
    });

    let mut results = results.into_inner().unwrap_or_else(|e| e.into_inner());
    results.sort_by_key(|(i, _)| *i);
    let failures: Vec<String> = results
        .iter()
        .filter_map(|(i, r)| r.as_ref().err().map(|e| format!("{} for {}:\n{e}", items[*i].job.kernel, items[*i].arch)))
        .collect();
    if !failures.is_empty() {
        return Err(format!("{} of {} compilation(s) failed:\n{}", failures.len(), items.len(), failures.join("\n")));
    }

    let manifest_path = dir.join(MANIFEST_FILE);
    let mut manifest = match Store::open(dir) {
        Ok(Some(s)) => s.manifest,
        Ok(None) => Manifest::default(),
        // A manifest that cannot be read (older ABI, hand-edited) is replaced:
        // its images were built for a contract that no longer exists.
        Err(e) => {
            tracing::warn!("backend-cuda: replacing an unusable AOT manifest: {e}");
            Manifest::default()
        }
    };
    for (_, r) in results {
        let (entry, bytes) = r.expect("failures returned above");
        // Image before manifest row: a reader that sees the row finds the file.
        write_atomic(&dir.join(&entry.file), &bytes)?;
        report.built += 1;
        for stale in manifest.upsert(entry) {
            let _ = std::fs::remove_file(dir.join(stale));
        }
    }
    write_atomic(&manifest_path, manifest.render().as_bytes())?;
    Ok(report)
}

fn file_stem(kernel: &str) -> String {
    kernel.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, bytes).map_err(|e| format!("writing {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("publishing {}: {e}", path.display())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvrtc::ArchFeatures;

    fn entry(kernel: &str, target: &str, key: &str, file: &str) -> Entry {
        Entry {
            kernel: kernel.into(),
            target: target.into(),
            kind: Kind::Cubin,
            toolchain: "nvrtc 12.9".into(),
            key: key.into(),
            image_sha256: "00".into(),
            file: file.into(),
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("brain-aot-unit-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_manifest_survives_a_round_trip() {
        let mut m = Manifest::default();
        m.upsert(entry("add", "sm_90", "k1", "add.sm_90.cubin"));
        m.upsert(entry("add", "sm_80", "k2", "add.sm_80.cubin"));
        assert_eq!(Manifest::parse(&m.render()).unwrap(), m);
    }

    /// A rebuild after a source edit has a new key; the row it replaces can
    /// never match again, so it must not linger (and its file must go).
    #[test]
    fn a_rebuild_replaces_the_stale_row_for_the_same_kernel_and_target() {
        let mut m = Manifest::default();
        m.upsert(entry("add", "sm_90", "old", "add.old.cubin"));
        m.upsert(entry("add", "sm_80", "other", "add.sm_80.cubin"));
        let dropped = m.upsert(entry("add", "sm_90", "new", "add.new.cubin"));
        assert_eq!(dropped, ["add.old.cubin"]);
        assert_eq!(m.entries.len(), 2);
        assert!(m.entries.iter().any(|e| e.key == "new") && m.entries.iter().all(|e| e.key != "old"));
    }

    #[test]
    fn a_manifest_that_cannot_be_trusted_is_refused_with_the_line_and_the_remedy() {
        let e = Manifest::parse("something else\n").unwrap_err();
        assert!(e.contains("line 1") && e.contains("make cuda/aot"), "{e}");
        let e = Manifest::parse("brain-cuda-aot 1\nabi 999\n").unwrap_err();
        assert!(e.contains("abi 999") && e.contains("make cuda/aot"), "{e}");
        let e = Manifest::parse(&format!("brain-cuda-aot 1\nabi {}\nonly\tthree\tfields\n", nvrtc::CUBIN_ABI_VERSION)).unwrap_err();
        assert!(e.contains("line 3") && e.contains("7"), "{e}");
        let bad_path = format!("brain-cuda-aot 1\nabi {}\nk\tsm_90\tcubin\tt\tkey\tsum\t../escape\n", nvrtc::CUBIN_ABI_VERSION);
        assert!(Manifest::parse(&bad_path).unwrap_err().contains("plain file name"));
    }

    #[test]
    fn the_key_covers_source_entry_target_specialization_and_is_order_independent() {
        let base = aot_key("s", "e", "sm_90", &[]);
        assert_ne!(base, aot_key("t", "e", "sm_90", &[]));
        assert_ne!(base, aot_key("s", "f", "sm_90", &[]));
        assert_ne!(base, aot_key("s", "e", "sm_90a", &[]));
        assert_ne!(base, aot_key("s", "e", "compute_90", &[]));
        assert_ne!(base, aot_key("s", "e", "sm_90", &[("N", "1")]));
        assert_eq!(aot_key("s", "e", "sm_90", &[("A", "1"), ("B", "2")]), aot_key("s", "e", "sm_90", &[("B", "2"), ("A", "1")]));
    }

    /// Best first: the device's own architecture, then lower minors of the same
    /// major, never another major and never a higher minor.
    #[test]
    fn cubin_candidates_are_the_device_then_lower_minors_of_its_major() {
        let names = |cc, f| cubin_targets(cc, f).iter().map(Target::name).collect::<Vec<_>>();
        assert_eq!(names((8, 9), ArchFeatures::Portable), ["sm_89", "sm_88", "sm_87", "sm_86", "sm_85", "sm_84", "sm_83", "sm_82", "sm_81", "sm_80"]);
        assert_eq!(names((9, 0), ArchFeatures::Portable), ["sm_90"]);
        assert_eq!(names((9, 0), ArchFeatures::Preferred), ["sm_90a", "sm_90"]);
        assert_eq!(names((9, 0), ArchFeatures::Required), ["sm_90a"]);
        assert!(names((8, 6), ArchFeatures::Required).is_empty(), "no arch-specific variant exists for 8.6");
    }

    #[test]
    fn each_toolkit_lane_ships_its_own_targets() {
        let names = |p: &Plan| p.cubins.iter().map(Target::name).collect::<Vec<_>>();
        let p12 = Plan::for_toolkit((12, 9));
        assert_eq!(names(&p12), ["sm_61", "sm_70", "sm_80", "sm_86", "sm_89", "sm_90"]);
        assert_eq!(p12.ptx, Some((6, 1)));
        let p13 = Plan::for_toolkit((13, 0));
        assert_eq!(names(&p13), ["sm_90", "sm_100", "sm_120"]);
        assert_eq!(p13.ptx, Some((9, 0)));
    }

    #[test]
    fn no_manifest_is_the_ordinary_state_and_not_an_error() {
        let d = scratch("none");
        assert!(Store::open(&d).unwrap().is_none());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_corrupted_image_is_a_named_error_and_never_reaches_the_driver() {
        let d = scratch("corrupt");
        let bytes = b"image bytes";
        std::fs::write(d.join("k.cubin"), bytes).unwrap();
        let mut m = Manifest::default();
        m.upsert(Entry { image_sha256: sha256_hex(bytes), ..entry("k", "sm_90", "key", "k.cubin") });
        std::fs::write(d.join(MANIFEST_FILE), m.render()).unwrap();
        let store = Store::open(&d).unwrap().expect("manifest present");
        let e = store.find("key").expect("indexed by key").clone();
        assert_eq!(store.image(&e).unwrap(), bytes);
        std::fs::write(d.join("k.cubin"), b"image byte").unwrap();
        let err = store.image(&e).unwrap_err();
        assert!(err.contains("checksum") && err.contains("make cuda/aot"), "{err}");
        std::fs::remove_dir_all(d).unwrap();
    }
}
