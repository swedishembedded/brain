// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Persistence helpers: atomic whole-document writes, a cheap header-only
//! reader, and NDJSON step streaming.

mod atomic;
mod header;
mod ndjson;

pub use atomic::{
    read_trajectory_with_fingerprint, remove_trajectory, write_trajectory, write_trajectory_atomic,
    FileFingerprint, PersistError,
};
pub use header::{
    read_trajectory_header, read_trajectory_header_fast, HeaderReadError, TrajectoryHeader,
};
pub use ndjson::{read_steps_ndjson, write_steps_ndjson, NdjsonError};
