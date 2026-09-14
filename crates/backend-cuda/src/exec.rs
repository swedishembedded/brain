// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Compile a kernel and run it on a device: context, allocations, modules,
//! launches.
//!
//! Swedish Embedded AB implements accelerator runtimes for its clients - the
//! context, memory and launch plumbing under a compute backend. If your team
//! needs expertise in driving a GPU from Rust without a vendor SDK in the
//! build, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Scope
//!
//! This is the execution substrate the generated (T0) tier is validated
//! against: enough to compile one kernel, move a few tiny buffers and launch a
//! grid. It is deliberately NOT [`backend_api::Backend`] - there is no
//! allocator, no stream, no graph and no step cache here, and a partial
//! `Backend` impl would be a backend that claims to run models and cannot.
//!
//! # The context is per device and shared
//!
//! `cuDevicePrimaryCtxRetain` rather than `cuCtxCreate`: the primary context is
//! what every other CUDA library on the process (and the runtime API, if
//! anything links it) uses, so retaining it keeps one context per device
//! instead of two competing ones. It is made current on the calling thread
//! before every operation, because a CUDA context is thread-local state and
//! this type is used from whichever test thread got it.
//!
//! # Nothing about the hardware is assumed
//!
//! The compute capability a kernel is compiled for is read back from the
//! device that will run it. There is no default and no fallback capability: a
//! cubin compiled for a capability the device does not have is either rejected
//! at module load or, worse, silently mis-scheduled.

use crate::driver::{CuContext, CuDevicePtr, CuFunction, CuModule, Driver, ExecFns};
use std::ffi::{c_void, CString};

/// A compiled cubin, and whether it came back from the on-disk cache.
pub struct Cubin {
    pub cubin: Vec<u8>,
    /// The content-addressed cache key - see `nvrtc::cache_key`.
    pub key: String,
    /// True when this compilation was served from disk rather than run.
    pub cached: bool,
}

/// One device with its primary context retained.
pub struct Context {
    d: &'static Driver,
    fns: &'static ExecFns,
    dev: std::ffi::c_int,
    ctx: CuContext,
    cc: (u32, u32),
    name: String,
}

// The context handle is used under `cuCtxSetCurrent` before every call, so it
// is sound to move between threads; the driver's own entry points are
// thread-safe.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Open the device at CUDA `ordinal`.
    ///
    /// An ordinal is a handle within THIS process, not an identity:
    /// `CUDA_VISIBLE_DEVICES` both filters and renumbers it. Callers that mean
    /// a particular physical card resolve it through the device registry's
    /// UUID first.
    pub fn open(ordinal: u32) -> Result<Context, String> {
        let d = crate::driver::driver().map_err(str::to_string)?;
        let fns = d.exec().map_err(str::to_string)?;
        let devices = d.devices()?;
        let info = devices
            .get(ordinal as usize)
            .ok_or_else(|| format!("CUDA ordinal {ordinal} does not exist ({} devices)", devices.len()))?;
        let dev = d.device_handle(ordinal)?;
        let mut ctx: CuContext = std::ptr::null_mut();
        // SAFETY: `ctx` is a valid out-parameter and `dev` came from
        // `cuDeviceGet`. The retain is released in `Drop`.
        d.check(unsafe { (fns.primary_ctx_retain)(&mut ctx, dev) }, "cuDevicePrimaryCtxRetain")?;
        let c = Context {
            d,
            fns,
            dev,
            ctx,
            cc: (info.cc_major, info.cc_minor),
            name: info.name.clone(),
        };
        c.make_current()?;
        Ok(c)
    }

    /// Compute capability of THIS device, as the driver reported it. Every
    /// compilation and every capability decision keys on this value; none may
    /// be written down anywhere.
    pub fn compute_capability(&self) -> (u32, u32) {
        self.cc
    }

    /// The device's product name, for error messages.
    pub fn name(&self) -> &str {
        &self.name
    }

    fn make_current(&self) -> Result<(), String> {
        // SAFETY: `self.ctx` is retained for the lifetime of `self`.
        self.d.check(unsafe { (self.fns.ctx_set_current)(self.ctx) }, "cuCtxSetCurrent")
    }

    /// Allocate `bytes` of device memory.
    pub fn alloc(&self, bytes: usize) -> Result<DeviceMem, String> {
        self.make_current()?;
        let mut ptr: CuDevicePtr = 0;
        // SAFETY: `ptr` is a valid out-parameter; the `_v2` entry point takes
        // a 64-bit size.
        self.d.check(unsafe { (self.fns.mem_alloc)(&mut ptr, bytes.max(1)) }, "cuMemAlloc")?;
        Ok(DeviceMem { d: self.d, fns: self.fns, ctx: self.ctx, ptr, len: bytes })
    }

    /// Copy host bytes into a device allocation.
    pub fn upload(&self, mem: &DeviceMem, src: &[u8]) -> Result<(), String> {
        if src.len() > mem.len {
            return Err(format!("upload of {} bytes into a {}-byte allocation", src.len(), mem.len));
        }
        self.make_current()?;
        // SAFETY: `src` is valid for `src.len()` bytes and the destination was
        // allocated with at least that many.
        self.d.check(
            unsafe { (self.fns.memcpy_htod)(mem.ptr, src.as_ptr() as *const c_void, src.len()) },
            "cuMemcpyHtoD",
        )
    }

    /// Copy device bytes back to the host.
    pub fn download(&self, mem: &DeviceMem, dst: &mut [u8]) -> Result<(), String> {
        if dst.len() > mem.len {
            return Err(format!("download of {} bytes from a {}-byte allocation", dst.len(), mem.len));
        }
        self.make_current()?;
        // SAFETY: `dst` is writable for `dst.len()` bytes and the source holds
        // at least that many.
        self.d.check(
            unsafe { (self.fns.memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, mem.ptr, dst.len()) },
            "cuMemcpyDtoH",
        )
    }

    /// Compile `src` for THIS device's capability, through the on-disk cubin
    /// cache.
    pub fn cubin(&self, src: &str, entry: &str) -> Result<Cubin, String> {
        let version = crate::nvrtc::version()?;
        let key = crate::nvrtc::cache_key(src, entry, self.cc, version);
        if let Some(cubin) = crate::nvrtc::cache_load(&key) {
            return Ok(Cubin { cubin, key, cached: true });
        }
        let cubin = crate::nvrtc::compile(src, self.cc)?;
        crate::nvrtc::cache_store(&key, &cubin);
        Ok(Cubin { cubin, key, cached: false })
    }

    /// Compile and load `src`, returning the module its entry points live in.
    pub fn compile(&self, src: &str, entry: &str) -> Result<Module, String> {
        let c = self.cubin(src, entry)?;
        self.load(&c.cubin)
    }

    /// Load an already-compiled cubin.
    pub fn load(&self, cubin: &[u8]) -> Result<Module, String> {
        self.make_current()?;
        let mut m: CuModule = std::ptr::null_mut();
        // SAFETY: `cubin` is a complete image for this device's capability and
        // outlives the call; the driver copies what it needs.
        self.d.check(
            unsafe { (self.fns.module_load_data)(&mut m, cubin.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        Ok(Module { d: self.d, fns: self.fns, ctx: self.ctx, m })
    }

    /// Launch `f` over `grid` blocks of `block` threads with one pointer
    /// argument per entry in `args`, in order.
    ///
    /// `cuLaunchKernel` takes an array of pointers TO the argument values, so
    /// a pointer argument is passed as a pointer to the `CUdeviceptr` - not as
    /// the device address itself. Getting that one indirection wrong reads a
    /// host stack address as a device pointer, which faults rather than
    /// miscomputing, but only once a kernel dereferences it.
    pub fn launch(
        &self,
        f: &Function,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[&DeviceMem],
    ) -> Result<(), String> {
        self.make_current()?;
        let mut values: Vec<CuDevicePtr> = args.iter().map(|a| a.ptr).collect();
        let mut params: Vec<*mut c_void> =
            values.iter_mut().map(|v| v as *mut CuDevicePtr as *mut c_void).collect();
        // SAFETY: `params` names `args.len()` valid pointers to device
        // addresses that outlive the call, the function belongs to a module
        // still loaded in this context, and no dynamic shared memory is
        // requested (the generated kernels declare theirs statically).
        self.d.check(
            unsafe {
                (self.fns.launch_kernel)(
                    f.f,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    /// Block until every launch on this context has completed. A launch is
    /// asynchronous and reports only the errors it can see BEFORE running, so
    /// this is where a faulting kernel is actually reported.
    pub fn sync(&self) -> Result<(), String> {
        self.make_current()?;
        // SAFETY: the context is current on this thread.
        self.d.check(unsafe { (self.fns.ctx_synchronize)() }, "cuCtxSynchronize")
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: balances the retain in `open`. The primary context is
        // reference-counted, so this does not tear down a context another
        // holder is still using.
        unsafe {
            (self.fns.primary_ctx_release)(self.dev);
        }
    }
}

/// One device allocation, freed when dropped.
pub struct DeviceMem {
    d: &'static Driver,
    fns: &'static ExecFns,
    ctx: CuContext,
    ptr: CuDevicePtr,
    len: usize,
}

impl DeviceMem {
    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        // The context must be current for the free to reach the right device.
        // SAFETY: `ctx` is kept alive by the `Context` that made this
        // allocation, and `ptr` came from `cuMemAlloc` on it.
        unsafe {
            if (self.fns.ctx_set_current)(self.ctx) == 0 {
                let rc = (self.fns.mem_free)(self.ptr);
                if rc != 0 {
                    tracing::warn!("cuMemFree failed: {}", self.d.error_text(rc));
                }
            }
        }
    }
}

// Same reasoning as `Context`: a device address is not thread-affine, the
// context is made current before it is used.
unsafe impl Send for DeviceMem {}
unsafe impl Sync for DeviceMem {}

/// A loaded module, unloaded when dropped.
pub struct Module {
    d: &'static Driver,
    fns: &'static ExecFns,
    ctx: CuContext,
    m: CuModule,
}

impl Module {
    /// Resolve an entry point by name. The generated kernels are `extern "C"`,
    /// so the name in the source is the name in the symbol table.
    pub fn function(&self, name: &str) -> Result<Function<'_>, String> {
        let cname = CString::new(name).map_err(|_| "entry name contains a NUL byte".to_string())?;
        let mut f: CuFunction = std::ptr::null_mut();
        // SAFETY: `self.m` is loaded in `self.ctx`, and `cname` outlives the
        // call.
        self.d.check(
            unsafe { (self.fns.module_get_function)(&mut f, self.m, cname.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Function { f, _module: self })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: the module was loaded in this context and is unloaded once.
        unsafe {
            if (self.fns.ctx_set_current)(self.ctx) == 0 {
                (self.fns.module_unload)(self.m);
            }
        }
    }
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// An entry point. Borrows its module, because a `CUfunction` is only valid
/// while the module that defines it is loaded.
pub struct Function<'a> {
    f: CuFunction,
    _module: &'a Module,
}
