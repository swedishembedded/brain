// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The CUDA **Driver API**, reached by `dlopen`ing `libcuda.so.1` at run time.
//!
//! Nothing here is linked at build time. `libcuda.so.1` ships with the NVIDIA
//! *kernel driver*, not with the CUDA toolkit, so a build host may have the
//! toolkit and no driver, a driver and no toolkit, or neither - and brain must
//! build and test green in all four cases. Runtime loading is the only way to
//! express that: an absent library is an ordinary `Err`, never a link failure
//! and never a missing-symbol abort at process start.
//!
//! Swedish Embedded AB implements runtime-loaded accelerator drivers for its
//! clients. If your team needs expertise in binding a vendor driver without
//! making it a build-time dependency of your firmware or application, you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! # Nothing about the hardware is assumed
//!
//! Device count, compute capability, VRAM, SM count and integrated-vs-discrete
//! are all read back per device through `cuDeviceGetCount` /
//! `cuDeviceGetAttribute` / `cuDeviceTotalMem`. No constant in this file
//! describes a particular card, and none may be added: the only fixed value is
//! NVIDIA's PCI vendor id, which is a property of the vendor whose driver this
//! library *is*, not of any card plugged into it.
//!
//! # Symbol names are the ABI-versioned ones
//!
//! `cuda.h` `#define`s several entry points onto a `_v2` suffix
//! (`cuDeviceTotalMem` -> `cuDeviceTotalMem_v2`, whose `bytes` out-parameter is
//! a 64-bit `size_t` rather than v1's 32-bit `unsigned int`). A header does
//! that rewriting for a C compiler; `dlsym` does not. Loading the unsuffixed
//! name here would silently bind the v1 entry point and write 4 bytes where 8
//! are expected - so the suffixed name is what this file asks for.

use std::ffi::{c_char, c_int, c_void, CStr};

/// `CUresult`. `CUDA_SUCCESS` is 0; every other value is an error whose text
/// comes from the driver itself via `cuGetErrorString`.
type CuResult = c_int;
/// `CUcontext`/`CUmodule`/`CUfunction`/`CUstream` are all opaque handles.
pub type CuContext = *mut c_void;
pub type CuModule = *mut c_void;
pub type CuFunction = *mut c_void;
/// `CUdeviceptr` - an integer device address, NOT a host pointer. It is 64-bit
/// on every 64-bit platform the driver supports, and passing it to
/// `cuLaunchKernel` means passing a pointer TO this integer, never the integer
/// itself.
pub type CuDevicePtr = u64;
/// `CUdevice` - an opaque device handle, not an index. It happens to be an
/// `int`, but it is obtained from `cuDeviceGet(ordinal)` and never formed by
/// casting an ordinal.
type CuDevice = c_int;

const CUDA_SUCCESS: CuResult = 0;

/// PCI vendor id of the vendor whose driver this library is. Not a statement
/// about any particular device.
const VENDOR_NVIDIA: u32 = 0x10de;

// `CUdevice_attribute` discriminants, transcribed from `cuda.h`. These are a
// stable public ABI enumeration, not hardware facts - each one names a
// QUESTION asked of whatever card is present.
const ATTR_INTEGRATED: c_int = 18;
const ATTR_PCI_BUS_ID: c_int = 33;
const ATTR_PCI_DEVICE_ID: c_int = 34;
const ATTR_PCI_DOMAIN_ID: c_int = 50;
const ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const ATTR_MULTIPROCESSOR_COUNT: c_int = 16;

/// The loaded driver: the `libloading::Library` plus the entry points resolved
/// out of it once, with `cuInit` already called.
pub struct Driver {
    /// Keeps the shared object mapped for as long as any function pointer
    /// below is callable. Dropping it would unmap the code they point at.
    _lib: libloading::Library,
    get_error_string: Option<unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult>,
    device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    device_get_uuid: unsafe extern "C" fn(*mut u8, CuDevice) -> CuResult,
    device_total_mem: unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    /// The entry points needed to RUN something, resolved separately and
    /// allowed to fail on their own. Device identity must keep working on a
    /// driver that is missing one of them: a failure here is "this box cannot
    /// execute", not "this box has no CUDA".
    exec: Result<ExecFns, String>,
}

/// The context / memory / module / launch half of the Driver API.
///
/// Note which names carry `_v2`. `cuda.h` `#define`s those onto the versioned
/// entry point for a C compiler; `dlsym` does no such rewriting, so asking for
/// the bare name here would bind a different ABI - for `cuMemAlloc` the v1
/// entry takes a 32-bit size, which silently truncates every allocation above
/// 4 GiB instead of failing.
pub struct ExecFns {
    pub(crate) primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    pub(crate) primary_ctx_release: unsafe extern "C" fn(CuDevice) -> CuResult,
    pub(crate) ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    pub(crate) ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    pub(crate) mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    pub(crate) mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    pub(crate) memcpy_htod: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult,
    pub(crate) memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    pub(crate) module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    pub(crate) module_unload: unsafe extern "C" fn(CuModule) -> CuResult,
    pub(crate) module_get_function:
        unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    #[allow(clippy::type_complexity)]
    pub(crate) launch_kernel: unsafe extern "C" fn(
        CuFunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
}

// Every field is either a mapped `Library` (which `libloading` already
// documents as `Send + Sync`) or a bare `extern "C"` function pointer. The
// driver's own entry points used here are thread-safe reads.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

/// The process-wide load attempt: performed at most once, and its OUTCOME is
/// cached, failure included. A box with no NVIDIA driver must not pay a
/// `dlopen` per query, and `cuInit` must not be called twice.
static DRIVER: std::sync::OnceLock<Result<Driver, String>> = std::sync::OnceLock::new();

/// The loaded driver, or why it is unavailable.
///
/// The error text is what a skip-if-absent test prints and what an explicit
/// `--backend cuda` fails with, so it names the concrete reason (no such
/// library / no symbol / `cuInit` status) rather than a generic "no CUDA".
pub fn driver() -> Result<&'static Driver, &'static str> {
    match DRIVER.get_or_init(load) {
        Ok(d) => Ok(d),
        Err(e) => Err(e.as_str()),
    }
}

fn load() -> Result<Driver, String> {
    // SONAME, not the `libcuda.so` development symlink: that symlink belongs
    // to the toolkit/stub package and may be absent (or a non-functional stub
    // that returns CUDA_ERROR_NOT_INITIALIZED) on a box that nevertheless has
    // a working driver.
    let lib = unsafe { libloading::Library::new("libcuda.so.1") }
        .map_err(|e| format!("libcuda.so.1 could not be loaded: {e}"))?;

    // SAFETY: each name below is resolved at the signature `cuda.h` declares
    // for it (see the module doc on `_v2`), and the resulting pointers are
    // only ever called while `lib` is still mapped - `Driver` owns it.
    unsafe {
        let init: unsafe extern "C" fn(u32) -> CuResult = sym(&lib, b"cuInit\0")?;
        // Optional: a driver old enough to lack it still works, it just cannot
        // explain its own error codes.
        let get_error_string = sym(&lib, b"cuGetErrorString\0").ok();

        let exec = load_exec(&lib);
        let d = Driver {
            exec,
            device_get_count: sym(&lib, b"cuDeviceGetCount\0")?,
            device_get: sym(&lib, b"cuDeviceGet\0")?,
            device_get_name: sym(&lib, b"cuDeviceGetName\0")?,
            device_get_uuid: sym(&lib, b"cuDeviceGetUuid\0")?,
            device_total_mem: sym(&lib, b"cuDeviceTotalMem_v2\0")?,
            device_get_attribute: sym(&lib, b"cuDeviceGetAttribute\0")?,
            get_error_string,
            _lib: lib,
        };

        // `cuInit(0)` is the precondition for every other Driver API call and
        // is documented as safe to call more than once - but this `OnceLock`
        // means it happens exactly once per process anyway.
        let rc = init(0);
        if rc != CUDA_SUCCESS {
            return Err(format!("cuInit failed: {}", d.error_text(rc)));
        }
        tracing::debug!("libcuda.so.1 loaded and cuInit succeeded");
        Ok(d)
    }
}

/// Resolve the execution half. Separate from `load` so a missing symbol here
/// disables only execution.
///
/// # Safety
/// Each name is resolved at the exact signature `cuda.h` declares for it, and
/// the pointers stay valid for as long as `lib` (owned by `Driver`) is mapped.
unsafe fn load_exec(lib: &libloading::Library) -> Result<ExecFns, String> {
    Ok(ExecFns {
        primary_ctx_retain: sym(lib, b"cuDevicePrimaryCtxRetain\0")?,
        primary_ctx_release: sym(lib, b"cuDevicePrimaryCtxRelease_v2\0")?,
        ctx_set_current: sym(lib, b"cuCtxSetCurrent\0")?,
        ctx_synchronize: sym(lib, b"cuCtxSynchronize\0")?,
        mem_alloc: sym(lib, b"cuMemAlloc_v2\0")?,
        mem_free: sym(lib, b"cuMemFree_v2\0")?,
        memcpy_htod: sym(lib, b"cuMemcpyHtoD_v2\0")?,
        memcpy_dtoh: sym(lib, b"cuMemcpyDtoH_v2\0")?,
        module_load_data: sym(lib, b"cuModuleLoadData\0")?,
        module_unload: sym(lib, b"cuModuleUnload\0")?,
        module_get_function: sym(lib, b"cuModuleGetFunction\0")?,
        launch_kernel: sym(lib, b"cuLaunchKernel\0")?,
    })
}

/// Resolve one NUL-terminated symbol name AT the type the caller expects,
/// yielding the bare function pointer.
///
/// Typed rather than `transmute`d from `*mut c_void`: the signature each entry
/// point is bound at is then written once, in the `Driver` field it fills, and
/// a mismatch is a type error instead of a silent ABI corruption.
///
/// # Safety
/// `T` must be the exact ABI signature `cuda.h` declares for `name`, and the
/// returned pointer is only valid while `lib` stays mapped.
unsafe fn sym<T: Copy>(lib: &libloading::Library, name: &[u8]) -> Result<T, String> {
    lib.get::<T>(name).map(|s| *s).map_err(|e| {
        format!("libcuda.so.1 has no {}: {e}", String::from_utf8_lossy(&name[..name.len() - 1]))
    })
}

impl Driver {
    /// The execution entry points, or why this driver cannot run anything.
    pub fn exec(&self) -> Result<&ExecFns, &str> {
        self.exec.as_ref().map_err(|e| e.as_str())
    }

    /// A `CUdevice` handle for an ordinal, for callers that go on to open a
    /// context on it. Deliberately the only way out of this module to a device
    /// handle: an ordinal is not an identity and must be resolved each time.
    pub fn device_handle(&self, ordinal: u32) -> Result<c_int, String> {
        self.device(ordinal)
    }

    /// The driver's own text for a `CUresult`, or the bare code when this
    /// driver is too old to expose `cuGetErrorString`.
    pub(crate) fn error_text(&self, rc: CuResult) -> String {
        if let Some(f) = self.get_error_string {
            let mut p: *const c_char = std::ptr::null();
            // SAFETY: `p` is a valid out-parameter; the driver returns a
            // pointer to its own static string, which is not freed.
            unsafe {
                if f(rc, &mut p) == CUDA_SUCCESS && !p.is_null() {
                    return CStr::from_ptr(p).to_string_lossy().into_owned();
                }
            }
        }
        format!("CUresult {rc}")
    }

    pub(crate) fn check(&self, rc: CuResult, what: &str) -> Result<(), String> {
        if rc == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{what} failed: {}", self.error_text(rc)))
        }
    }

    /// `cuDeviceGetCount` - how many devices this driver exposes *to this
    /// process*. Not a machine property: `CUDA_VISIBLE_DEVICES` filters and
    /// renumbers it, which is precisely why an ordinal is never an identity.
    pub fn device_count(&self) -> Result<u32, String> {
        let mut n: c_int = 0;
        // SAFETY: `n` is a valid out-parameter for the duration of the call.
        self.check(unsafe { (self.device_get_count)(&mut n) }, "cuDeviceGetCount")?;
        Ok(n.max(0) as u32)
    }

    fn device(&self, ordinal: u32) -> Result<CuDevice, String> {
        let mut dev: CuDevice = 0;
        // SAFETY: `dev` is a valid out-parameter for the duration of the call.
        self.check(unsafe { (self.device_get)(&mut dev, ordinal as c_int) }, "cuDeviceGet")?;
        Ok(dev)
    }

    fn attribute(&self, dev: CuDevice, attr: c_int) -> Result<i32, String> {
        let mut v: c_int = 0;
        // SAFETY: `v` is a valid out-parameter; `attr` is a `cuda.h`
        // `CUdevice_attribute` discriminant.
        self.check(unsafe { (self.device_get_attribute)(&mut v, attr, dev) }, "cuDeviceGetAttribute")?;
        Ok(v)
    }

    /// Everything this crate asks the driver about one device, in one place -
    /// so no caller has to remember which questions make a device identity.
    fn describe(&self, ordinal: u32) -> Result<CudaDevice, String> {
        let dev = self.device(ordinal)?;

        // `cuDeviceGetName` truncates silently rather than reporting the size
        // it needs, so the buffer is generously larger than any product name
        // the driver has ever returned, and the result is read as a C string.
        let mut buf = [0i8; 256];
        // SAFETY: `buf` is writable for `buf.len()` bytes; the driver
        // NUL-terminates within that length.
        self.check(
            unsafe { (self.device_get_name)(buf.as_mut_ptr() as *mut c_char, buf.len() as c_int, dev) },
            "cuDeviceGetName",
        )?;
        // SAFETY: the call above succeeded, so `buf` holds a NUL-terminated
        // string no longer than `buf.len()`.
        let name = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) }.to_string_lossy().into_owned();

        // `CUuuid` is 16 raw bytes - and on NVIDIA they are the SAME 16 bytes
        // Vulkan reports as `VkPhysicalDeviceIDProperties::deviceUUID` and
        // NVML reports as the GPU UUID. That equality is the entire basis for
        // resolving a CUDA ordinal to brain's canonical `gpu<i>` index.
        let mut uuid = [0u8; 16];
        // SAFETY: `uuid` is writable for exactly the 16 bytes `CUuuid` holds.
        self.check(unsafe { (self.device_get_uuid)(uuid.as_mut_ptr(), dev) }, "cuDeviceGetUuid")?;

        let mut total_mem: usize = 0;
        // SAFETY: the `_v2` entry point writes a `size_t`, which `usize` is.
        self.check(unsafe { (self.device_total_mem)(&mut total_mem, dev) }, "cuDeviceTotalMem")?;

        Ok(CudaDevice {
            ordinal,
            name,
            uuid,
            pci_domain: self.attribute(dev, ATTR_PCI_DOMAIN_ID)? as u32,
            pci_bus: self.attribute(dev, ATTR_PCI_BUS_ID)? as u32,
            pci_device: self.attribute(dev, ATTR_PCI_DEVICE_ID)? as u32,
            total_mem: total_mem as u64,
            cc_major: self.attribute(dev, ATTR_COMPUTE_CAPABILITY_MAJOR)? as u32,
            cc_minor: self.attribute(dev, ATTR_COMPUTE_CAPABILITY_MINOR)? as u32,
            integrated: self.attribute(dev, ATTR_INTEGRATED)? != 0,
            multiprocessors: self.attribute(dev, ATTR_MULTIPROCESSOR_COUNT)?.max(0) as u32,
        })
    }

    /// Describe every device this driver exposes, in CUDA ordinal order.
    pub fn devices(&self) -> Result<Vec<CudaDevice>, String> {
        (0..self.device_count()?).map(|i| self.describe(i)).collect()
    }
}

/// One CUDA device, as the driver describes it. Every field is a runtime
/// answer; none is a compiled-in assumption about a card.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDevice {
    /// Position in `cuDeviceGetCount`'s enumeration - a handle to re-open this
    /// device *within this process*, and nothing more. It is not stable across
    /// processes (`CUDA_VISIBLE_DEVICES`) and must never be used as identity.
    pub ordinal: u32,
    pub name: String,
    /// `cuDeviceGetUuid` - equal to Vulkan's `deviceUUID` for the same card.
    pub uuid: [u8; 16],
    pub pci_domain: u32,
    pub pci_bus: u32,
    pub pci_device: u32,
    /// `cuDeviceTotalMem`, bytes.
    pub total_mem: u64,
    /// Compute capability, queried. Kernel selection keys on this pair; no
    /// code, comment or table may hardcode a value for it.
    pub cc_major: u32,
    pub cc_minor: u32,
    /// `CU_DEVICE_ATTRIBUTE_INTEGRATED` - shares memory with the host.
    pub integrated: bool,
    /// SM count, for occupancy/grid sizing decisions that must be queried.
    pub multiprocessors: u32,
}

impl CudaDevice {
    /// `"domain:bus:device.function"`, the spelling
    /// `backend_vulkan`'s `VK_EXT_pci_bus_info` path produces, so the two
    /// strings are directly comparable.
    ///
    /// The CUDA Driver API exposes no PCI *function*; NVIDIA's compute
    /// function is 0 on every device it enumerates, and this is a fallback key
    /// anyway - the UUID above is what identity is actually resolved on.
    pub fn pci_bus_id(&self) -> String {
        format!("{:04x}:{:02x}:{:02x}.0", self.pci_domain, self.pci_bus, self.pci_device)
    }

    /// This device as brain's backend-neutral [`backend_api::GpuIdentity`].
    ///
    /// `device_id` is 0: the Driver API reports the PCI *slot* (bus/device)
    /// but not the chip's PCI device id, and inventing one would corrupt
    /// `GpuIdentity::same_device`'s weakest key. It is never consulted here,
    /// because both stronger keys - UUID, then PCI bus id - are always
    /// present on this path.
    pub fn identity(&self) -> backend_api::GpuIdentity {
        backend_api::GpuIdentity {
            name: self.name.clone(),
            vendor_id: VENDOR_NVIDIA,
            device_id: 0,
            uuid: Some(self.uuid),
            pci_bus: Some(self.pci_bus_id()),
            ordinal: self.ordinal as usize,
            vram_bytes: self.total_mem,
            class: if self.integrated {
                backend_api::DeviceClass::IntegratedGpu
            } else {
                backend_api::DeviceClass::DiscreteGpu
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The PCI string must be byte-identical to the one `backend_vulkan`
    /// formats, or the fallback identity key silently stops matching on a box
    /// whose driver reports no UUID.
    #[test]
    fn pci_bus_id_matches_the_vulkan_spelling() {
        let d = CudaDevice {
            ordinal: 0,
            name: "x".into(),
            uuid: [0; 16],
            pci_domain: 0,
            pci_bus: 0x82,
            pci_device: 0,
            total_mem: 0,
            cc_major: 0,
            cc_minor: 0,
            integrated: false,
            multiprocessors: 0,
        };
        assert_eq!(d.pci_bus_id(), "0000:82:00.0");
    }
}
