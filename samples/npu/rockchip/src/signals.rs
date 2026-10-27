// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Ctrl-C / SIGTERM handling: the handler only sets a flag, and the pipeline
//! notices it and unwinds through its destructors, which release the camera,
//! the ISP and the encoder. A vendor channel left open by a killed process
//! would make the next run fail, so exiting by signal must still clean up.

use crate::error::{err, Result};
use std::sync::atomic::{AtomicBool, Ordering};

static REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    // Async-signal-safe: a single atomic store.
    REQUESTED.store(true, Ordering::Release);
}

/// Routes SIGINT and SIGTERM to [`requested`].
pub fn install() -> Result<()> {
    for sig in [libc::SIGINT, libc::SIGTERM] {
        // SAFETY: the handler is async-signal-safe and the sigaction struct is
        // zero-initialised, which is a valid empty mask and no flags.
        let rc = unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_signal as extern "C" fn(libc::c_int) as usize;
            libc::sigaction(sig, &action, std::ptr::null_mut())
        };
        if rc != 0 {
            return Err(err!("installing the handler for signal {sig}: {}", std::io::Error::last_os_error()));
        }
    }
    Ok(())
}

/// True once SIGINT or SIGTERM has arrived.
pub fn requested() -> bool {
    REQUESTED.load(Ordering::Acquire)
}
