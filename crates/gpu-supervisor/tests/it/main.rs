// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary. The worker tests
//! spawn real child processes (a POSIX shell standing in for a device driver
//! that hangs or aborts), so they need no GPU and are deterministic.

mod breaker;
#[cfg(target_os = "linux")]
mod worker;
