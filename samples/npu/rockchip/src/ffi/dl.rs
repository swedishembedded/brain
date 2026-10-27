// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Run-time loading of the board's vendor libraries.
//!
//! Nothing is linked at build time and nothing is redistributed: the sample
//! opens the board's own `/oem/usr/lib/*.so` with `dlopen` and resolves the
//! few entry points it uses. [`dl_api!`] declares such a table once, as a
//! struct of typed function pointers, so a missing symbol is one clear error
//! at start-up instead of a crash in the middle of a stream.

use crate::error::{err, Result};
use std::ffi::{c_char, c_int, c_void, CStr, CString};

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

fn last_dl_error() -> String {
    // SAFETY: dlerror returns null or a NUL-terminated string valid until the
    // next dl* call on this thread, and it is copied immediately.
    unsafe {
        let p = dlerror();
        if p.is_null() {
            "unknown dlopen error".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// Directory holding the board's vendor libraries and their dependencies.
pub const VENDOR_LIB_DIR: &str = "/oem/usr/lib";

/// Makes sure the dynamic loader will look in [`VENDOR_LIB_DIR`] for the
/// dependencies of the vendor libraries (`librockit` needs `librga` and
/// `librockchip_mpp`, which are not on the board's default search path).
/// The loader reads `LD_LIBRARY_PATH` once, at process start, so when it is
/// not set yet this re-executes the program with it set; it returns only if no
/// re-execution was needed.
pub fn ensure_vendor_search_path() -> Result<()> {
    use std::os::unix::process::CommandExt;
    let current = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    if current.split(':').any(|d| d == VENDOR_LIB_DIR) {
        return Ok(());
    }
    let path = if current.is_empty() { VENDOR_LIB_DIR.to_string() } else { format!("{VENDOR_LIB_DIR}:{current}") };
    let exe = std::env::current_exe().map_err(|e| err!("locating this program to restart it: {e}"))?;
    let error = std::process::Command::new(exe).args(std::env::args_os().skip(1)).env("LD_LIBRARY_PATH", path).exec();
    Err(err!("restarting with LD_LIBRARY_PATH={VENDOR_LIB_DIR}: {error}"))
}

/// A shared object opened with `dlopen`. It is never closed: the vendor
/// libraries start their own threads, and unloading them under those threads
/// is worse than keeping the mapping until the process exits.
pub struct Library {
    handle: *mut c_void,
    name: &'static str,
}

// SAFETY: the handle is an opaque token that dlsym may use from any thread.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    pub fn open(path: &'static str) -> Result<Self> {
        let c = CString::new(path).map_err(|_| err!("library path {path:?} contains a NUL"))?;
        // SAFETY: `c` is a valid NUL-terminated string.
        let handle = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if handle.is_null() {
            return Err(err!("loading {path}: {}", last_dl_error()));
        }
        Ok(Library { handle, name: path })
    }

    /// Resolves `symbol` as a `T`, which must be a function-pointer type.
    ///
    /// # Safety
    /// `T` must be the exact `extern "C"` signature the library exports.
    pub unsafe fn sym<T: Copy>(&self, symbol: &str) -> Result<T> {
        assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
        let c = CString::new(symbol).map_err(|_| err!("symbol {symbol:?} contains a NUL"))?;
        let p = dlsym(self.handle, c.as_ptr());
        if p.is_null() {
            return Err(err!("{} does not export {symbol}: {}", self.name, last_dl_error()));
        }
        Ok(std::mem::transmute_copy::<*mut c_void, T>(&p))
    }
}

/// Declares a table of vendor functions loaded from one library.
///
/// ```ignore
/// dl_api! {
///     pub struct Rockit from "/oem/usr/lib/librockit.so" {
///         fn RK_MPI_SYS_Init() -> i32;
///     }
/// }
/// let api = Rockit::load()?;
/// ```
macro_rules! dl_api {
    ($vis:vis struct $name:ident from $lib:literal {
        $( fn $f:ident ( $($a:ident : $t:ty),* $(,)? ) $(-> $r:ty)? ; )*
    }) => {
        #[allow(non_snake_case, dead_code)]
        $vis struct $name {
            _lib: $crate::ffi::dl::Library,
            $( pub $f: unsafe extern "C" fn($($t),*) $(-> $r)?, )*
        }

        impl $name {
            pub fn load() -> $crate::error::Result<Self> {
                let lib = $crate::ffi::dl::Library::open($lib)?;
                // SAFETY: each pointer type is the signature declared for it
                // in the vendor header this table was derived from.
                unsafe {
                    Ok(Self {
                        $( $f: lib.sym::<unsafe extern "C" fn($($t),*) $(-> $r)?>(stringify!($f))?, )*
                        _lib: lib,
                    })
                }
            }
        }
    };
}
pub(crate) use dl_api;
