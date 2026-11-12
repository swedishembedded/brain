// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The clock frame stamps are in.
//!
//! On Linux that is `CLOCK_MONOTONIC`, the clock V4L2 drivers stamp buffers
//! with, read directly so a consumer can subtract a frame's [`crate::Stamp`]
//! from "now" without converting between clock domains. `std::time::Instant`
//! uses the same clock on Linux but exposes no way to build one from a
//! driver's timestamp, which is why this exists.

/// Nanoseconds of the monotonic clock.
#[cfg(target_os = "linux")]
pub fn monotonic_ns() -> u64 {
    #[repr(C)]
    struct Timespec {
        tv_sec: std::os::raw::c_long,
        tv_nsec: std::os::raw::c_long,
    }
    extern "C" {
        fn clock_gettime(clk: std::os::raw::c_int, tp: *mut Timespec) -> std::os::raw::c_int;
    }
    const CLOCK_MONOTONIC: std::os::raw::c_int = 1;
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `ts` is a valid, writable timespec for the call's duration.
    let rc = unsafe { clock_gettime(CLOCK_MONOTONIC, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_MONOTONIC) cannot fail on Linux");
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Nanoseconds since the first call: monotonic, in no domain any driver
/// shares, which is all a platform without V4L2 needs.
#[cfg(not(target_os = "linux"))]
pub fn monotonic_ns() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}
