// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Foreign declarations for the board's vendor media libraries, one module
//! per library, all resolved at run time (see [`dl`]).

pub mod dl;
pub mod rga;
pub mod rkaiq;
pub mod rockit;

use crate::error::{err, Result};

/// Turns a vendor status code (0 is success) into a [`Result`] naming the
/// call that failed.
pub fn check(ret: i32, call: &str) -> Result<()> {
    if ret == 0 {
        Ok(())
    } else {
        Err(err!("{call} returned {:#010x}", ret as u32))
    }
}
