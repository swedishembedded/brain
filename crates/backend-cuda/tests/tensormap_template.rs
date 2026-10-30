// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The 128-byte `CUtensorMap` template the FP8 prefill GEMM carries (`kernels-cuda/cu/matmul_fp8_wgmma_tma.cu`).
//!
//! Swedish Embedded AB implements solutions for Hopper tensor-memory-accelerator pipelines for its clients. If your team
//! needs expertise in feeding warpgroup tensor cores through TMA without a host-side descriptor, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The kernel is launched with plain device pointers, so it builds its tensor maps on the device: it starts from this
//! template (everything the shape does not change: 2-D, one byte an element, a 128-byte box of 128 rows, the 128-byte
//! swizzle, zero fill) and replaces only the global address, the dimensions and the row stride. The test asks the driver
//! to encode the same map and holds the template to it, so a driver that encodes a field differently fails here, not in
//! a GEMM.

use std::ffi::c_void;

const TEMPLATE_ADDRESS: u64 = 0x10000;
const TEMPLATE_ROWS: u64 = 256;
const TEMPLATE_COLS: u64 = 256;

/// The ordinal of the device whose context the encoding borrows: the one the suite is pointed at (`BRAIN_DEVICE=gpu1`), else the first.
fn device_ordinal() -> i32 {
    std::env::var("BRAIN_DEVICE").ok().and_then(|d| d.strip_prefix("gpu").and_then(|n| n.parse().ok())).unwrap_or(0)
}

/// Encode the template with the driver's own `cuTensorMapEncodeTiled`.
fn encode() -> Option<[u64; 16]> {
    #[repr(C, align(64))]
    struct Map([u64; 16]);
    type Encode = unsafe extern "C" fn(*mut Map, u32, u32, *mut c_void, *const u64, *const u64, *const u32, *const u32, u32, u32, u32, u32) -> i32;
    // SAFETY: libcuda.so.1 is the driver; the symbol has this signature in cuda.h, and the call touches no device state.
    unsafe {
        let lib = libloading::Library::new("libcuda.so.1").map_err(|e| eprintln!("dlopen: {e}")).ok()?;
        let init: libloading::Symbol<unsafe extern "C" fn(u32) -> i32> = lib.get(b"cuInit\0").ok()?;
        init(0);
        // The driver wants a current context even for this host-side encoding; the first device's primary context does.
        let device_get: libloading::Symbol<unsafe extern "C" fn(*mut i32, i32) -> i32> = lib.get(b"cuDeviceGet\0").ok()?;
        let retain: libloading::Symbol<unsafe extern "C" fn(*mut *mut c_void, i32) -> i32> = lib.get(b"cuDevicePrimaryCtxRetain\0").ok()?;
        let set_current: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> i32> = lib.get(b"cuCtxSetCurrent\0").ok()?;
        let mut dev = 0;
        let mut ctx: *mut c_void = std::ptr::null_mut();
        if device_get(&mut dev, device_ordinal()) != 0 || retain(&mut ctx, dev) != 0 || set_current(ctx) != 0 {
            return None;
        }
        let encode: libloading::Symbol<Encode> = lib.get(b"cuTensorMapEncodeTiled\0").map_err(|e| eprintln!("sym: {e}")).ok()?;
        let mut map = Map([0; 16]);
        let dims = [TEMPLATE_COLS, TEMPLATE_ROWS];
        let strides = [TEMPLATE_COLS];
        let boxd = [128u32, 128];
        let elem = [1u32, 1];
        // CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0, interleave none = 0, swizzle 128B = 3, L2 promotion 256B = 3, OOB fill none (zeros) = 0.
        let rc = encode(&mut map, 0, 2, TEMPLATE_ADDRESS as *mut c_void, dims.as_ptr(), strides.as_ptr(), boxd.as_ptr(), elem.as_ptr(), 0, 3, 3, 0);
        eprintln!("rc {rc}");
        (rc == 0).then_some(map.0)
    }
}

/// The template the kernel source carries: the words of its `BRAIN_FWT_TEMPLATE` define.
fn shipped() -> Vec<u64> {
    let src = include_str!("../../kernels-cuda/cu/matmul_fp8_wgmma_tma.cu");
    let line = src.lines().find_map(|l| l.strip_prefix("#define BRAIN_FWT_TEMPLATE ")).expect("the kernel defines BRAIN_FWT_TEMPLATE");
    line.split(',').map(|w| u64::from_str_radix(w.trim().trim_start_matches("0x").trim_end_matches("ull"), 16).expect("a hexadecimal word")).collect()
}

#[test]
fn the_kernel_template_is_what_the_driver_encodes() {
    let Some(driver) = encode() else {
        brain_testutil::skip_unavailable("a CUDA driver with a device whose context the encoder can borrow");
        return;
    };
    assert_eq!(shipped(), driver.to_vec(), "regenerate BRAIN_FWT_TEMPLATE from the driver's encoding (print it with `encode()`)");
}
