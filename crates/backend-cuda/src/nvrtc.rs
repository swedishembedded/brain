// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! NVRTC - compile CUDA C++ text to a cubin at run time, and cache the result.
//!
//! Swedish Embedded AB implements runtime kernel compilation pipelines for its
//! clients, cache keys included. If your team needs expertise in shipping a
//! JIT-compiled compute path that stays reproducible across driver and toolkit
//! upgrades, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this is optional, and loaded at run time
//!
//! `libnvrtc` ships with the CUDA **toolkit**, not with the driver, so a box
//! that can run CUDA cannot necessarily compile it. The normal deployment path
//! for the catalogue is therefore an ahead-of-time cubin; NVRTC is the
//! development path, and its absence is an ordinary `Err` naming the missing
//! library rather than a link failure at process start.
//!
//! # The cache key
//!
//! Everything that can change the emitted machine code takes part in the key:
//! the source text, the target it is compiled FOR (compute capability and the
//! arch-specific suffix), the NVRTC version doing the compiling, the exact
//! compile flags, the specialization macros, the launch ABI version and the
//! entry point name. Miss any of those and a stale cubin is served for a different
//! question - which, for a compute kernel, means wrong numbers rather than a
//! crash. The file is published with `rename(2)` so a concurrent reader sees
//! either the old entry or the complete new one, never a half-written cubin.

use std::ffi::{c_char, c_int, c_void, CString};
use std::path::PathBuf;

/// `nvrtcResult`; 0 is `NVRTC_SUCCESS`.
type NvrtcResult = c_int;
/// `nvrtcProgram`, an opaque handle.
type NvrtcProgram = *mut c_void;

const NVRTC_SUCCESS: NvrtcResult = 0;

/// The SONAMEs tried, in order. A major version is part of the SONAME, not a
/// statement about hardware: NVRTC's ABI is versioned with the toolkit, and a
/// box may have any one of them installed. `libnvrtc.so` (the unversioned
/// development symlink) comes last because it belongs to the `-dev` package
/// and may be absent on a machine that has the runtime.
const SONAMES: &[&str] = &["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so"];

/// Names an exact NVRTC library to load instead of searching (a path or a
/// SONAME): the way to pin one toolkit when several are installed.
const ENV_NVRTC: &str = "BRAIN_NVRTC";
/// The toolkit root whose `lib64`/`lib` hold NVRTC, the same variable the CUDA
/// toolchain itself uses.
const ENV_CUDA_PATH: &str = "CUDA_PATH";

/// Every library name to try loading, in order: the explicit override alone if
/// there is one; otherwise the toolkit root's `lib64` then `lib`, then the bare
/// SONAMEs for the system loader. A toolkit the user installed without root is
/// found by its root, not by whatever the loader's cache happens to hold.
fn candidate_libraries(explicit: Option<&str>, cuda_path: Option<&str>) -> Vec<String> {
    if let Some(lib) = explicit.filter(|l| !l.is_empty()) {
        return vec![lib.to_string()];
    }
    let mut out = Vec::new();
    if let Some(root) = cuda_path.filter(|r| !r.is_empty()) {
        for dir in ["lib64", "lib"] {
            out.extend(SONAMES.iter().map(|s| format!("{root}/{dir}/{s}")));
        }
    }
    out.extend(SONAMES.iter().map(|s| s.to_string()));
    out
}

struct Nvrtc {
    _lib: libloading::Library,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvrtcResult,
    create_program: unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    destroy_program: unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult,
    compile_program: unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult,
    get_program_log_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    get_program_log: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    get_cubin_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    get_cubin: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
}

// A mapped library plus bare `extern "C"` pointers; NVRTC compilations are
// independent and the library is immutable after load.
unsafe impl Send for Nvrtc {}
unsafe impl Sync for Nvrtc {}

static NVRTC: std::sync::OnceLock<Result<Nvrtc, String>> = std::sync::OnceLock::new();

fn nvrtc() -> Result<&'static Nvrtc, &'static str> {
    match NVRTC.get_or_init(load) {
        Ok(n) => Ok(n),
        Err(e) => Err(e.as_str()),
    }
}

fn load() -> Result<Nvrtc, String> {
    let mut tried = Vec::new();
    let explicit = std::env::var(ENV_NVRTC).ok();
    let cuda_path = std::env::var(ENV_CUDA_PATH).ok();
    for soname in &candidate_libraries(explicit.as_deref(), cuda_path.as_deref()) {
        // SAFETY: loading a shared object runs its initialisers; NVRTC's are
        // the ordinary CUDA toolkit ones.
        let lib = match unsafe { libloading::Library::new(soname) } {
            Ok(l) => l,
            Err(e) => {
                tried.push(format!("{soname}: {e}"));
                continue;
            }
        };
        // SAFETY: every name below is resolved at the signature `nvrtc.h`
        // declares for it, and the pointers live as long as `lib`, which the
        // returned struct owns.
        let built = unsafe {
            (|| -> Result<Nvrtc, String> {
                Ok(Nvrtc {
                    version: sym(&lib, b"nvrtcVersion\0")?,
                    create_program: sym(&lib, b"nvrtcCreateProgram\0")?,
                    destroy_program: sym(&lib, b"nvrtcDestroyProgram\0")?,
                    compile_program: sym(&lib, b"nvrtcCompileProgram\0")?,
                    get_program_log_size: sym(&lib, b"nvrtcGetProgramLogSize\0")?,
                    get_program_log: sym(&lib, b"nvrtcGetProgramLog\0")?,
                    // Cubin rather than PTX: a cubin is final machine code for
                    // the capability it names, so the driver does no second
                    // compilation at module load and cannot disagree with what
                    // was cached.
                    get_cubin_size: sym(&lib, b"nvrtcGetCUBINSize\0")?,
                    get_cubin: sym(&lib, b"nvrtcGetCUBIN\0")?,
                    _lib: lib,
                })
            })()
        };
        match built {
            Ok(n) => {
                tracing::debug!("{soname} loaded");
                return Ok(n);
            }
            Err(e) => tried.push(format!("{soname}: {e}")),
        }
    }
    Err(format!("no usable NVRTC ({})", tried.join("; ")))
}

/// # Safety
/// `T` must be the exact ABI signature `nvrtc.h` declares for `name`.
unsafe fn sym<T: Copy>(lib: &libloading::Library, name: &[u8]) -> Result<T, String> {
    lib.get::<T>(name)
        .map(|s| *s)
        .map_err(|e| format!("no {}: {e}", String::from_utf8_lossy(&name[..name.len() - 1])))
}

/// `(major, minor)` of the loaded NVRTC, part of every cache key.
pub fn version() -> Result<(u32, u32), String> {
    let n = nvrtc().map_err(str::to_string)?;
    let (mut ma, mut mi) = (0, 0);
    // SAFETY: both are valid out-parameters for the duration of the call.
    let rc = unsafe { (n.version)(&mut ma, &mut mi) };
    if rc != NVRTC_SUCCESS {
        return Err(format!("nvrtcVersion failed with {rc}"));
    }
    Ok((ma.max(0) as u32, mi.max(0) as u32))
}

/// A compute capability as `(major, minor)`.
pub type Cc = (u32, u32);

/// The version of the contract between a compiled kernel and the code that
/// launches it: argument order and widths, how bound lengths and the uniform
/// block follow the pointers. A cubin compiled under one contract and launched
/// under another reads its arguments from the wrong slots, so the version is
/// part of every cache key and every ahead-of-time manifest. Bump it whenever
/// that contract changes.
pub const CUBIN_ABI_VERSION: u32 = 1;

/// What a kernel says about the architecture-specific instruction set
/// (`sm_90a`: `wgmma`, `setmaxnreg`, TMA multicast and friends).
///
/// Arch-specific code runs on exactly one compute capability and on none
/// other, so it is a property a kernel DECLARES and a device either has or
/// lacks - never something to switch on everywhere.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ArchFeatures {
    /// The source is valid on every target from its floor upward; compile for
    /// the plain `sm_XY`. The generated tier is always this.
    #[default]
    Portable,
    /// The source has an arch-specific fast path guarded by the feature macros
    /// (`__CUDA_ARCH_FEAT_SM90_ALL`) and a portable body for the rest: use the
    /// suffix where it exists and the plain target elsewhere.
    Preferred,
    /// The source is only valid with the suffix. Where the device or toolkit
    /// cannot provide it the kernel is refused, not compiled into an error.
    Required,
}

/// The machine-code target of one compilation: a real architecture (`sm_`),
/// with or without the arch-specific suffix.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Target {
    pub cc: Cc,
    /// Compile for `sm_<cc>a`.
    pub arch_specific: bool,
}

impl Target {
    /// The plain, forward-compatible-within-a-major target for `cc`.
    pub const fn plain(cc: Cc) -> Target {
        Target { cc, arch_specific: false }
    }

    /// The target for a device of capability `cc` and a kernel declaring
    /// `features`, under NVRTC `nvrtc`.
    pub fn resolve(cc: Cc, features: ArchFeatures, nvrtc: (u32, u32)) -> Result<Target, String> {
        let available = arch_specific_available(cc, nvrtc);
        match features {
            ArchFeatures::Portable => Ok(Target::plain(cc)),
            ArchFeatures::Preferred => Ok(Target { cc, arch_specific: available }),
            ArchFeatures::Required if available => Ok(Target { cc, arch_specific: true }),
            ArchFeatures::Required => Err(format!(
                "the kernel needs arch-specific code (sm_{}{}a) and compute capability {}.{} with NVRTC {}.{} cannot provide it",
                cc.0, cc.1, cc.0, cc.1, nvrtc.0, nvrtc.1
            )),
        }
    }

    /// The architecture name NVRTC and the manifest use: `sm_90`, `sm_90a`.
    pub fn name(&self) -> String {
        format!("sm_{}{}{}", self.cc.0, self.cc.1, if self.arch_specific { "a" } else { "" })
    }
}

/// Whether `sm_<cc>a` exists, and whether NVRTC `nvrtc` can emit it.
///
/// The suffix is defined per architecture by NVIDIA (Hopper from CUDA 12.0,
/// Blackwell from 12.8, its later parts from 12.9), so this is a table of
/// when each was introduced, not a rule about the numbers. A capability not
/// listed has no arch-specific variant as far as this crate knows, which makes
/// it fall back to the plain target - the safe direction.
pub fn arch_specific_available(cc: Cc, nvrtc: (u32, u32)) -> bool {
    const INTRODUCED: &[(Cc, (u32, u32))] =
        &[((9, 0), (12, 0)), ((10, 0), (12, 8)), ((12, 0), (12, 8)), ((10, 1), (12, 8)), ((10, 3), (12, 9)), ((12, 1), (12, 9))];
    INTRODUCED.iter().any(|(c, since)| *c == cc && nvrtc >= *since)
}

/// The compile flags the generated tier is built with, for a queried compute
/// capability.
///
/// `--fmad=false` is not a tuning choice: NVRTC contracts `a*b + c` into a
/// single fused multiply-add by default, which rounds ONCE where the portable
/// reference rounds twice. The cross-backend agreement this project asserts is
/// absolute (maxabs), so a "better" answer is still a failure, and a
/// contracted accumulation drifts further the longer the reduction.
pub fn flags(target: &Target, defines: &[(&str, &str)]) -> Vec<String> {
    let mut f = vec![
        // Real architecture (`sm_`), not virtual (`compute_`): the output is a
        // cubin for exactly the capability the device reported, which is read
        // back at run time and never assumed.
        format!("--gpu-architecture={}", target.name()),
        "--fmad=false".to_string(),
    ];
    f.extend(defines.iter().map(|(k, v)| format!("-D{k}={v}")));
    f
}

/// Compile `src` to a cubin for `target`, with `defines` as `-DNAME=VALUE`.
///
/// The NVRTC log is included in the error, because a generated kernel's
/// compile failure is a defect in the generator and the line it names is the
/// only way back to the IR that produced it.
pub fn compile(src: &str, target: &Target, defines: &[(&str, &str)]) -> Result<Vec<u8>, String> {
    let n = nvrtc().map_err(str::to_string)?;
    let csrc = CString::new(src).map_err(|_| "source contains a NUL byte".to_string())?;
    let name = CString::new("brain-generated.cu").unwrap();

    let mut prog: NvrtcProgram = std::ptr::null_mut();
    // SAFETY: `csrc`/`name` outlive the call; no headers are supplied, which
    // is why the emitted source may not contain an `#include`.
    let rc = unsafe {
        (n.create_program)(&mut prog, csrc.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null())
    };
    if rc != NVRTC_SUCCESS {
        return Err(format!("nvrtcCreateProgram failed with {rc}"));
    }
    // From here on every exit must destroy the program.
    let result = compile_loaded(n, prog, target, defines);
    // SAFETY: `prog` was created above and is destroyed exactly once.
    unsafe {
        (n.destroy_program)(&mut prog);
    }
    result
}

fn compile_loaded(n: &Nvrtc, prog: NvrtcProgram, target: &Target, defines: &[(&str, &str)]) -> Result<Vec<u8>, String> {
    let opts: Vec<CString> = flags(target, defines).into_iter().map(|f| CString::new(f).unwrap()).collect();
    let ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
    // SAFETY: `ptrs` names `opts.len()` valid NUL-terminated strings that
    // outlive the call.
    let rc = unsafe { (n.compile_program)(prog, ptrs.len() as c_int, ptrs.as_ptr()) };

    let log = program_log(n, prog);
    if rc != NVRTC_SUCCESS {
        return Err(format!("nvrtcCompileProgram failed with {rc}:\n{log}"));
    }
    if !log.trim().is_empty() {
        tracing::warn!("NVRTC diagnostics for a generated kernel:\n{log}");
    }

    let mut size = 0usize;
    // SAFETY: `size` is a valid out-parameter.
    let rc = unsafe { (n.get_cubin_size)(prog, &mut size) };
    if rc != NVRTC_SUCCESS {
        return Err(format!("nvrtcGetCUBINSize failed with {rc}"));
    }
    let mut cubin = vec![0u8; size];
    // SAFETY: `cubin` is writable for exactly `size` bytes, which is what
    // NVRTC just reported it needs.
    let rc = unsafe { (n.get_cubin)(prog, cubin.as_mut_ptr() as *mut c_char) };
    if rc != NVRTC_SUCCESS {
        return Err(format!("nvrtcGetCUBIN failed with {rc}"));
    }
    Ok(cubin)
}

fn program_log(n: &Nvrtc, prog: NvrtcProgram) -> String {
    let mut size = 0usize;
    // SAFETY: valid out-parameter; a program that failed to compile still has
    // a log.
    if unsafe { (n.get_program_log_size)(prog, &mut size) } != NVRTC_SUCCESS || size <= 1 {
        return String::new();
    }
    let mut buf = vec![0u8; size];
    // SAFETY: `buf` is writable for the size NVRTC just asked for.
    if unsafe { (n.get_program_log)(prog, buf.as_mut_ptr() as *mut c_char) } != NVRTC_SUCCESS {
        return String::new();
    }
    while buf.last() == Some(&0) {
        buf.pop();
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// The cache key for one compilation: hex sha256 over every input that can
/// change the output.
pub fn cache_key(src: &str, entry: &str, target: &Target, nvrtc_version: (u32, u32), defines: &[(&str, &str)]) -> String {
    cache_key_with_abi(src, entry, target, nvrtc_version, defines, CUBIN_ABI_VERSION)
}

/// [`cache_key`] under an explicit launch ABI version - the seam that lets a
/// test show the version is part of the key.
pub fn cache_key_with_abi(
    src: &str,
    entry: &str,
    target: &Target,
    nvrtc_version: (u32, u32),
    defines: &[(&str, &str)],
    abi: u32,
) -> String {
    use sha2::{Digest, Sha256};
    // The set of macros is the question, not the order they were listed in.
    let mut sorted = defines.to_vec();
    sorted.sort_unstable();
    let mut h = Sha256::new();
    // Each field is length-prefixed so no two different tuples can hash the
    // same byte stream by concatenating differently.
    for field in [
        src.as_bytes(),
        entry.as_bytes(),
        target.name().as_bytes(),
        format!("nvrtc{}.{}", nvrtc_version.0, nvrtc_version.1).as_bytes(),
        flags(target, &sorted).join(" ").as_bytes(),
        format!("abi{abi}").as_bytes(),
    ] {
        h.update((field.len() as u64).to_le_bytes());
        h.update(field);
    }
    format!("{:x}", h.finalize())
}

/// Where a cubin for `key` lives, or `None` when there is nowhere to persist.
pub fn cache_path(key: &str) -> Option<PathBuf> {
    backend_api::cache_dir().map(|d| d.join("cuda-cubin").join(format!("{key}.cubin")))
}

/// Read a cached cubin, if one was published for this key.
pub fn cache_load(key: &str) -> Option<Vec<u8>> {
    let p = cache_path(key)?;
    let bytes = std::fs::read(p).ok()?;
    // A zero-length file is not a cubin. It would mean a writer was
    // interrupted in a way the rename below is supposed to make impossible, so
    // treat it as a miss rather than handing an empty module to the driver.
    (!bytes.is_empty()).then_some(bytes)
}

/// Publish a cubin for `key`. Best effort: a read-only or full filesystem
/// costs a recompile, never a failure.
pub fn cache_store(key: &str, cubin: &[u8]) {
    let Some(path) = cache_path(key) else { return };
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    // Unique temp name: two processes compiling the same kernel concurrently
    // must not write the same file, or a reader could observe a torn one. The
    // rename then publishes atomically, and whichever lands last wins with
    // byte-identical content.
    let tmp = dir.join(format!(".{key}.{}.tmp", std::process::id()));
    if std::fs::write(&tmp, cubin).is_ok() && std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(src: &str, entry: &str, cc: Cc, nvrtc: (u32, u32)) -> String {
        cache_key(src, entry, &Target::plain(cc), nvrtc, &[])
    }

    /// Every input that can change the compiled output must change the key.
    /// A key that ignores one of them serves machine code compiled for a
    /// different question, which for a kernel means wrong numbers.
    #[test]
    fn the_cache_key_separates_every_input_that_changes_the_output() {
        let base = key("a", "k", (6, 1), (12, 2));
        assert_ne!(base, key("b", "k", (6, 1), (12, 2)), "source");
        assert_ne!(base, key("a", "j", (6, 1), (12, 2)), "entry name");
        assert_ne!(base, key("a", "k", (7, 0), (12, 2)), "compute capability");
        assert_ne!(base, key("a", "k", (6, 1), (12, 3)), "NVRTC version");
        assert_eq!(base, key("a", "k", (6, 1), (12, 2)), "and is stable");
    }

    /// `sm_90` and `sm_90a` are different machine code for the same device:
    /// the arch-specific one carries instructions the plain one cannot encode.
    #[test]
    fn the_cache_key_separates_the_arch_suffix_the_specialization_and_the_abi() {
        let t90 = Target::plain((9, 0));
        let t90a = Target { cc: (9, 0), arch_specific: true };
        let base = cache_key("a", "k", &t90, (12, 9), &[]);
        assert_ne!(base, cache_key("a", "k", &t90a, (12, 9), &[]), "arch suffix");
        assert_ne!(base, cache_key("a", "k", &t90, (12, 9), &[("TILE", "64")]), "specialization");
        assert_ne!(
            cache_key("a", "k", &t90, (12, 9), &[("TILE", "64")]),
            cache_key("a", "k", &t90, (12, 9), &[("TILE", "128")]),
            "specialization value"
        );
        assert_eq!(
            cache_key("a", "k", &t90, (12, 9), &[("A", "1"), ("B", "2")]),
            cache_key("a", "k", &t90, (12, 9), &[("B", "2"), ("A", "1")]),
            "the order specializations are listed in is not part of the question"
        );
        assert_eq!(base, cache_key_with_abi("a", "k", &t90, (12, 9), &[], CUBIN_ABI_VERSION), "the live ABI is the default");
        assert_ne!(base, cache_key_with_abi("a", "k", &t90, (12, 9), &[], CUBIN_ABI_VERSION + 1), "launch ABI version");
    }

    /// Length-prefixing, not concatenation: `("ab","c")` and `("a","bc")` are
    /// different compilations and must be different keys.
    #[test]
    fn the_cache_key_cannot_be_confused_by_a_shifted_field_boundary() {
        assert_ne!(key("ab", "c", (6, 1), (12, 2)), key("a", "bc", (6, 1), (12, 2)));
    }

    /// The arch-specific target is chosen from the queried capability and the
    /// toolkit, never assumed: Hopper gets `sm_90a` when asked and able, every
    /// other device the plain name, and a kernel that REQUIRES the suffix is
    /// refused where it cannot be had instead of being compiled into an error.
    #[test]
    fn the_arch_specific_target_is_chosen_from_capability_and_toolkit() {
        let name = |cc, f, v| Target::resolve(cc, f, v).map(|t| t.name());
        assert_eq!(name((9, 0), ArchFeatures::Portable, (12, 9)).unwrap(), "sm_90");
        assert_eq!(name((9, 0), ArchFeatures::Preferred, (12, 9)).unwrap(), "sm_90a");
        assert_eq!(name((9, 0), ArchFeatures::Required, (13, 0)).unwrap(), "sm_90a");
        // Preferred degrades to the plain name where the suffix does not exist.
        assert_eq!(name((8, 0), ArchFeatures::Preferred, (12, 9)).unwrap(), "sm_80");
        assert_eq!(name((8, 9), ArchFeatures::Preferred, (13, 0)).unwrap(), "sm_89");
        // A toolkit that predates the suffix cannot emit it.
        assert_eq!(name((9, 0), ArchFeatures::Preferred, (11, 8)).unwrap(), "sm_90");
        // Required is a refusal, with the reason, never a silent downgrade.
        let e = name((8, 6), ArchFeatures::Required, (12, 9)).unwrap_err();
        assert!(e.contains("8.6") && e.contains("arch-specific"), "{e}");
        let e = name((9, 0), ArchFeatures::Required, (11, 8)).unwrap_err();
        assert!(e.contains("11.8"), "{e}");
    }

    /// Arch-specific code runs on exactly the capability it names, so `sm_90a`
    /// is not offered for a 9.x device that is not 9.0, and the Blackwell
    /// suffixes need a toolkit that knows them.
    #[test]
    fn the_arch_suffix_exists_only_where_the_toolkit_and_architecture_define_it() {
        assert!(arch_specific_available((9, 0), (12, 0)));
        assert!(!arch_specific_available((9, 1), (13, 0)));
        assert!(!arch_specific_available((9, 0), (11, 8)));
        assert!(arch_specific_available((10, 0), (12, 8)));
        assert!(!arch_specific_available((10, 0), (12, 4)));
        assert!(arch_specific_available((12, 0), (13, 0)));
        assert!(!arch_specific_available((7, 5), (13, 0)));
    }

    /// A CUDA 13 toolkit installs `libnvrtc.so.13`, and nothing links the
    /// unversioned development symlink for it, so the major version must be a
    /// candidate in its own right.
    #[test]
    fn every_supported_toolkit_major_is_a_candidate_soname() {
        let c = candidate_libraries(None, None);
        for major in ["13", "12", "11"] {
            assert!(c.iter().any(|l| l == &format!("libnvrtc.so.{major}")), "{major}: {c:?}");
        }
    }

    /// An explicit override is the only candidate: asking for one library and
    /// silently getting another would make a toolkit pin meaningless.
    #[test]
    fn an_explicit_override_is_the_only_candidate() {
        assert_eq!(candidate_libraries(Some("pinned/libnvrtc.so.12"), Some("toolkit")), ["pinned/libnvrtc.so.12"]);
    }

    /// A toolkit root is searched before the system loader, `lib64` before
    /// `lib`, so a user-owned toolkit wins over whatever the loader finds first.
    #[test]
    fn a_toolkit_root_is_searched_before_the_system_loader() {
        let c = candidate_libraries(None, Some("toolkit"));
        let at = |needle: &str| c.iter().position(|l| l == needle).unwrap_or_else(|| panic!("{needle} missing from {c:?}"));
        assert!(at("toolkit/lib64/libnvrtc.so.13") < at("toolkit/lib/libnvrtc.so.13"));
        assert!(at("toolkit/lib/libnvrtc.so") < at("libnvrtc.so.13"));
        assert_eq!(c.last().map(String::as_str), Some("libnvrtc.so"));
    }

    /// The flags are part of the contract, not an implementation detail: the
    /// tier's numerical agreement depends on contraction being off.
    #[test]
    fn contraction_is_disabled_and_the_architecture_is_the_queried_one() {
        let f = flags(&Target::plain((7, 5)), &[]);
        assert!(f.iter().any(|o| o == "--fmad=false"), "{f:?}");
        assert!(f.iter().any(|o| o == "--gpu-architecture=sm_75"), "{f:?}");
        let f = flags(&Target { cc: (9, 0), arch_specific: true }, &[("TILE", "64")]);
        assert!(f.iter().any(|o| o == "--gpu-architecture=sm_90a"), "{f:?}");
        assert!(f.iter().any(|o| o == "-DTILE=64"), "{f:?}");
    }
}
