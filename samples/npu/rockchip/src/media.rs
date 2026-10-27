// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The rockit media system: loaded once, initialised once, and shared by
//! everything that touches a vendor buffer or channel.

use crate::error::{Context, Result};
use crate::ffi::check;
use crate::ffi::rockit::Rockit;

/// An initialised rockit system. `RK_MPI_SYS_Exit` runs when the last owner
/// (camera, encoder, frame tap) lets go, so no channel outlives the system.
pub struct Media {
    pub api: Rockit,
}

impl Media {
    pub fn init() -> Result<Self> {
        let api = Rockit::load().context("loading the rockit media library")?;
        // SAFETY: no arguments; the library is loaded.
        check(unsafe { (api.RK_MPI_SYS_Init)() }, "RK_MPI_SYS_Init")?;
        Ok(Media { api })
    }
}

impl Drop for Media {
    fn drop(&mut self) {
        // SAFETY: balances the successful RK_MPI_SYS_Init in `init`.
        unsafe { (self.api.RK_MPI_SYS_Exit)() };
    }
}
