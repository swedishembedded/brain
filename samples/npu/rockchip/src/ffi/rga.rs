// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Declarations for Rockchip's 2D raster accelerator library (`librga.so`),
//! used to scale and colour-convert camera frames in hardware.
//!
//! Provenance: derived from `drmrga.h` and `RgaApi.h`, licensed Apache-2.0
//! (Copyright Rockchip Electronics Co. LTD). The legacy C entry point
//! `c_RkRgaBlit` is used because it takes pointers; the newer `improcess`
//! passes large structs by value, whose calling convention is not worth
//! guessing at across a dlopen boundary.

use super::dl::dl_api;

/// `RK_FORMAT_YCbCr_420_SP`: NV12.
pub const RK_FORMAT_YCBCR_420_SP: i32 = 0xa << 8;
/// `RK_FORMAT_RGB_888`: three bytes per pixel, R first.
pub const RK_FORMAT_RGB_888: i32 = 0x2 << 8;

/// `rga_rect_t`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RgaRect {
    pub xoffset: i32,
    pub yoffset: i32,
    pub width: i32,
    pub height: i32,
    pub wstride: i32,
    pub hstride: i32,
    pub format: i32,
    pub size: i32,
}

/// `rga_info_t`: the leading fields are mirrored, the many feature flags that
/// follow stay zero (their C default) in an opaque tail of the measured size.
#[repr(C)]
pub struct RgaInfo {
    pub fd: i32,
    pub vir_addr: *mut u8,
    pub phy_addr: *mut u8,
    pub hnd: u32,
    pub format: i32,
    pub rect: RgaRect,
    pub blend: u32,
    pub buffer_size: i32,
    pub rotation: i32,
    pub color: i32,
    pub test_log: i32,
    pub mmu_flag: i32,
    tail: [u8; 612],
}

impl RgaInfo {
    /// A buffer described by a dma-buf `fd`, viewed as a `width` x `height`
    /// image of `format` with the given strides.
    pub fn from_fd(fd: i32, rect: RgaRect) -> Self {
        // SAFETY: integers and raw pointers; all-zero is the C default.
        let mut info: RgaInfo = unsafe { std::mem::zeroed() };
        info.fd = fd;
        info.format = rect.format;
        info.rect = rect;
        // 1 asks the driver to map the dma-buf through the RGA's IOMMU, which
        // is how the legacy API addresses an fd-backed buffer.
        info.mmu_flag = 1;
        info
    }
}

dl_api! {
    pub struct Rga from "/oem/usr/lib/librga.so" {
        fn c_RkRgaInit() -> i32;
        fn c_RkRgaDeInit();
        fn c_RkRgaBlit(src: *mut RgaInfo, dst: *mut RgaInfo, src1: *mut RgaInfo) -> i32;
    }
}

#[cfg(target_pointer_width = "32")]
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<RgaRect>() == 32);
    assert!(size_of::<RgaInfo>() == 688);
    assert!(offset_of!(RgaInfo, rect) == 20);
    assert!(offset_of!(RgaInfo, blend) == 52);
    assert!(offset_of!(RgaInfo, mmu_flag) == 72);
};
