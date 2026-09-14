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
//! the source text, the compute capability it is compiled FOR, the NVRTC
//! version doing the compiling, the exact compile flags, and the entry point
//! name. Miss any of those and a stale cubin is served for a different
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
const SONAMES: &[&str] = &["libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so"];

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
    for soname in SONAMES {
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

/// The compile flags the generated tier is built with, for a queried compute
/// capability.
///
/// `--fmad=false` is not a tuning choice: NVRTC contracts `a*b + c` into a
/// single fused multiply-add by default, which rounds ONCE where the portable
/// reference rounds twice. The cross-backend agreement this project asserts is
/// absolute (maxabs), so a "better" answer is still a failure, and a
/// contracted accumulation drifts further the longer the reduction.
pub fn flags(cc: (u32, u32)) -> Vec<String> {
    vec![
        // Real architecture (`sm_`), not virtual (`compute_`): the output is a
        // cubin for exactly the capability the device reported, which is read
        // back at run time and never assumed.
        format!("--gpu-architecture=sm_{}{}", cc.0, cc.1),
        "--fmad=false".to_string(),
    ]
}

/// Compile `src` to a cubin for compute capability `cc`.
///
/// The NVRTC log is included in the error, because a generated kernel's
/// compile failure is a defect in the generator and the line it names is the
/// only way back to the IR that produced it.
pub fn compile(src: &str, cc: (u32, u32)) -> Result<Vec<u8>, String> {
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
    let result = compile_loaded(n, prog, cc);
    // SAFETY: `prog` was created above and is destroyed exactly once.
    unsafe {
        (n.destroy_program)(&mut prog);
    }
    result
}

fn compile_loaded(n: &Nvrtc, prog: NvrtcProgram, cc: (u32, u32)) -> Result<Vec<u8>, String> {
    let opts: Vec<CString> = flags(cc).into_iter().map(|f| CString::new(f).unwrap()).collect();
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
pub fn cache_key(src: &str, entry: &str, cc: (u32, u32), nvrtc_version: (u32, u32)) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    // Each field is length-prefixed so no two different tuples can hash the
    // same byte stream by concatenating differently.
    for field in [
        src.as_bytes(),
        entry.as_bytes(),
        format!("cc{}.{}", cc.0, cc.1).as_bytes(),
        format!("nvrtc{}.{}", nvrtc_version.0, nvrtc_version.1).as_bytes(),
        flags(cc).join(" ").as_bytes(),
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

    /// Every input that can change the compiled output must change the key.
    /// A key that ignores one of them serves machine code compiled for a
    /// different question, which for a kernel means wrong numbers.
    #[test]
    fn the_cache_key_separates_every_input_that_changes_the_output() {
        let base = cache_key("a", "k", (6, 1), (12, 2));
        assert_ne!(base, cache_key("b", "k", (6, 1), (12, 2)), "source");
        assert_ne!(base, cache_key("a", "j", (6, 1), (12, 2)), "entry name");
        assert_ne!(base, cache_key("a", "k", (7, 0), (12, 2)), "compute capability");
        assert_ne!(base, cache_key("a", "k", (6, 1), (12, 3)), "NVRTC version");
        assert_eq!(base, cache_key("a", "k", (6, 1), (12, 2)), "and is stable");
    }

    /// Length-prefixing, not concatenation: `("ab","c")` and `("a","bc")` are
    /// different compilations and must be different keys.
    #[test]
    fn the_cache_key_cannot_be_confused_by_a_shifted_field_boundary() {
        assert_ne!(cache_key("ab", "c", (6, 1), (12, 2)), cache_key("a", "bc", (6, 1), (12, 2)));
    }

    /// The flags are part of the contract, not an implementation detail: the
    /// tier's numerical agreement depends on contraction being off.
    #[test]
    fn contraction_is_disabled_and_the_architecture_is_the_queried_one() {
        let f = flags((7, 5));
        assert!(f.iter().any(|o| o == "--fmad=false"), "{f:?}");
        assert!(f.iter().any(|o| o == "--gpu-architecture=sm_75"), "{f:?}");
    }
}
