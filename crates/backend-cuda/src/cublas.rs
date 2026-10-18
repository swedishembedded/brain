// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The vendor dense-GEMM library (`libcublas`), reached by `dlopen` exactly as
//! `libcuda` and NVRTC are: never linked, so a box without the toolkit simply
//! has no library GEMM and every product runs on brain's own kernels.
//!
//! Swedish Embedded AB implements tensor-core training and inference stacks
//! that fall back cleanly on hardware without them. If your team needs
//! expertise in making a mixed-precision GPU path optional rather than a hard
//! dependency, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! One handle belongs to one [`crate::exec::Context`] and is bound to that
//! context's stream, so a product is ordered with the kernels around it by the
//! stream itself. Operands are bf16 with binary32 accumulation
//! (`CUBLAS_COMPUTE_32F`), the mixed-precision recipe of bf16 LoRA/QLoRA
//! training; the output is binary32.

use std::ffi::{c_int, c_void};

use backend_api::{LibraryDtype, LibraryGemm};

use crate::driver::{CuDevicePtr, CuStream};

type Status = c_int;
type Handle = *mut c_void;

const SUCCESS: Status = 0;
const OP_N: c_int = 0;
const OP_T: c_int = 1;
/// `cudaDataType` values of `library_types.h`.
const R_32F: c_int = 0;
const R_16BF: c_int = 14;
/// `CUBLAS_COMPUTE_32F`.
const COMPUTE_32F: c_int = 68;
/// `CUBLAS_GEMM_DEFAULT`: the library's own heuristic picks the algorithm.
const GEMM_DEFAULT: c_int = -1;

/// The SONAMEs tried, newest first; the major version is part of the SONAME.
const SONAMES: &[&str] = &["libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so"];

struct Lib {
    _lib: libloading::Library,
    create: unsafe extern "C" fn(*mut Handle) -> Status,
    destroy: unsafe extern "C" fn(Handle) -> Status,
    set_stream: unsafe extern "C" fn(Handle, CuStream) -> Status,
    gemm_strided_batched_ex: unsafe extern "C" fn(
        Handle,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        *const c_void,
        *const c_void,
        c_int,
        c_int,
        i64,
        *const c_void,
        c_int,
        c_int,
        i64,
        *const c_void,
        *mut c_void,
        c_int,
        c_int,
        i64,
        c_int,
        c_int,
        c_int,
    ) -> Status,
    gemm_ex: unsafe extern "C" fn(
        Handle,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        *const c_void,
        *const c_void,
        c_int,
        c_int,
        *const c_void,
        c_int,
        c_int,
        *const c_void,
        *mut c_void,
        c_int,
        c_int,
        c_int,
        c_int,
    ) -> Status,
}

// A mapped library plus bare function pointers, immutable after load.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

static LIB: std::sync::OnceLock<Result<Lib, String>> = std::sync::OnceLock::new();

fn lib() -> Result<&'static Lib, &'static str> {
    LIB.get_or_init(load).as_ref().map_err(String::as_str)
}

fn load() -> Result<Lib, String> {
    let mut tried = Vec::new();
    for soname in crate::nvrtc::toolkit_library_candidates(SONAMES) {
        // SAFETY: loading a shared object runs its initialisers; the CUDA
        // toolkit libraries' are ordinary.
        let lib = match unsafe { libloading::Library::new(&soname) } {
            Ok(l) => l,
            Err(e) => {
                tried.push(format!("{soname}: {e}"));
                continue;
            }
        };
        // SAFETY: each name is resolved at the signature `cublas_api.h`
        // declares for it; the pointers live as long as `lib`, which the
        // returned struct owns.
        let built = unsafe {
            (|| -> Result<Lib, String> {
                Ok(Lib {
                    create: sym(&lib, b"cublasCreate_v2\0")?,
                    destroy: sym(&lib, b"cublasDestroy_v2\0")?,
                    set_stream: sym(&lib, b"cublasSetStream_v2\0")?,
                    gemm_ex: sym(&lib, b"cublasGemmEx\0")?,
                    gemm_strided_batched_ex: sym(&lib, b"cublasGemmStridedBatchedEx\0")?,
                    _lib: lib,
                })
            })()
        };
        match built {
            Ok(l) => return Ok(l),
            Err(e) => tried.push(format!("{soname}: {e}")),
        }
    }
    Err(format!("no usable cuBLAS ({})", tried.join("; ")))
}

/// # Safety
/// `T` must be the exact ABI signature the library declares for `name`.
unsafe fn sym<T: Copy>(lib: &libloading::Library, name: &[u8]) -> Result<T, String> {
    lib.get::<T>(name).map(|s| *s).map_err(|e| format!("no {}: {e}", String::from_utf8_lossy(&name[..name.len() - 1])))
}

/// A cuBLAS handle bound to one stream; destroyed when dropped.
pub struct Blas {
    lib: &'static Lib,
    handle: Handle,
    /// A handle is not safe for concurrent calls; threads sharing a context
    /// take turns here.
    busy: std::sync::Mutex<()>,
}

// The handle is used under the owning context's `cuCtxSetCurrent` and, per the
// library's contract, by one thread at a time: callers serialise on the
// context's stream-issue path.
unsafe impl Send for Blas {}
unsafe impl Sync for Blas {}

impl Blas {
    /// Load the library and bind a new handle to `stream`. The caller made the
    /// owning context current.
    ///
    /// # Safety
    ///
    /// `stream` must be a live stream of the current context and must outlive
    /// the returned handle: the vendor library keeps it and issues work on it.
    pub unsafe fn new(stream: CuStream) -> Result<Blas, String> {
        let lib = lib().map_err(str::to_string)?;
        let mut handle: Handle = std::ptr::null_mut();
        // SAFETY: `handle` is a valid out-parameter.
        check(unsafe { (lib.create)(&mut handle) }, "cublasCreate")?;
        let blas = Blas { lib, handle, busy: std::sync::Mutex::new(()) };
        // SAFETY: the handle is live and `stream` outlives it (the context
        // owns both and drops the handle first).
        check(unsafe { (lib.set_stream)(handle, stream) }, "cublasSetStream")?;
        Ok(blas)
    }

    /// Enqueue `gemm` on the handle's stream over device addresses `a`, `b`, `c`.
    pub fn gemm(&self, gemm: &LibraryGemm, a: CuDevicePtr, b: CuDevicePtr, c: CuDevicePtr) -> Result<(), String> {
        let ty = |d: LibraryDtype| match d {
            LibraryDtype::F32 => R_32F,
            LibraryDtype::Bf16 => R_16BF,
        };
        let op = |t: bool| if t { OP_T } else { OP_N };
        let dim = |v: u32, what: &str| c_int::try_from(v).map_err(|_| format!("cublasGemmEx: {what} {v} exceeds the library's 32-bit range"));
        let _turn = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        let (m, n, k) = (dim(gemm.m, "m")?, dim(gemm.n, "n")?, dim(gemm.k, "k")?);
        let (lda, ldb, ldc) = (dim(gemm.lda, "lda")?, dim(gemm.ldb, "ldb")?, dim(gemm.ldc, "ldc")?);
        let alpha = &gemm.alpha as *const f32 as *const c_void;
        let beta = &gemm.beta as *const f32 as *const c_void;
        // SAFETY: the pointers are device addresses the caller bound for this
        // product with the leading dimensions, strides and extents stated in
        // `gemm`; alpha/beta are host scalars read during the call (host
        // pointer mode).
        let rc = unsafe {
            if gemm.batch <= 1 {
                (self.lib.gemm_ex)(
                    self.handle,
                    op(gemm.trans_a),
                    op(gemm.trans_b),
                    m,
                    n,
                    k,
                    alpha,
                    a as *const c_void,
                    ty(gemm.a),
                    lda,
                    b as *const c_void,
                    ty(gemm.b),
                    ldb,
                    beta,
                    c as *mut c_void,
                    ty(gemm.c),
                    ldc,
                    COMPUTE_32F,
                    GEMM_DEFAULT,
                )
            } else {
                (self.lib.gemm_strided_batched_ex)(
                    self.handle,
                    op(gemm.trans_a),
                    op(gemm.trans_b),
                    m,
                    n,
                    k,
                    alpha,
                    a as *const c_void,
                    ty(gemm.a),
                    lda,
                    gemm.stride_a,
                    b as *const c_void,
                    ty(gemm.b),
                    ldb,
                    gemm.stride_b,
                    beta,
                    c as *mut c_void,
                    ty(gemm.c),
                    ldc,
                    gemm.stride_c,
                    dim(gemm.batch, "batch")?,
                    COMPUTE_32F,
                    GEMM_DEFAULT,
                )
            }
        };
        check(rc, "cublasGemmEx")
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `cublasCreate` and is destroyed once.
        unsafe { (self.lib.destroy)(self.handle) };
    }
}

fn check(rc: Status, what: &str) -> Result<(), String> {
    if rc == SUCCESS {
        Ok(())
    } else {
        Err(format!("{what} failed with cublasStatus_t {rc}"))
    }
}

/// Whether the library can be loaded at all, with the reason when it cannot.
pub fn availability() -> Result<(), String> {
    lib().map(|_| ()).map_err(str::to_string)
}
