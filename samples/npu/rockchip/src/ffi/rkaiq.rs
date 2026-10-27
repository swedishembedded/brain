// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Declarations for Rockchip's ISP 3A library (`librkaiq.so`): auto exposure,
//! white balance and the tuning (IQ) files that make the sensor image usable.
//!
//! Provenance: derived from `rk_aiq_user_api2_sysctl.h` and
//! `common/rk_aiq_types.h`, licensed Apache-2.0 (Copyright Rockchip
//! Electronics Co. LTD). Only the system-control entry points are needed.

use super::dl::dl_api;
use std::ffi::{c_char, c_void};

/// `rk_aiq_working_mode_t`: `RK_AIQ_WORKING_MODE_NORMAL`.
pub const AIQ_WORKING_MODE_NORMAL: i32 = 0;

/// `sizeof(rk_aiq_static_info_t)` measured with the vendor headers (304).
/// The only field this sample reads, `sensor_info.sensor_name`, is a C string
/// at offset 0, so the struct is handed over as a zeroed byte buffer.
pub const STATIC_INFO_SIZE: usize = 304;

dl_api! {
    pub struct Rkaiq from "/oem/usr/lib/librkaiq.so" {
        fn rk_aiq_uapi2_sysctl_enumStaticMetas(index: i32, info: *mut u8) -> i32;
        fn rk_aiq_uapi2_sysctl_init(
            sensor_entity: *const c_char,
            iq_dir: *const c_char,
            err_cb: *const c_void,
            metas_cb: *const c_void,
        ) -> *mut c_void;
        fn rk_aiq_uapi2_sysctl_prepare(ctx: *mut c_void, width: u32, height: u32, mode: i32) -> i32;
        fn rk_aiq_uapi2_sysctl_start(ctx: *mut c_void) -> i32;
        fn rk_aiq_uapi2_sysctl_stop(ctx: *mut c_void, keep_ext_hw_state: bool) -> i32;
        fn rk_aiq_uapi2_sysctl_deinit(ctx: *mut c_void);
    }
}
