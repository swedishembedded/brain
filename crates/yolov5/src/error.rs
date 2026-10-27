// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The crate's error type: an explicit enum (the SDK convention) so a caller can
//! act on the variant, and every message names the field, tensor or size at
//! fault.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The `brain.config` header is missing, malformed or unsupported.
    Config(String),
    /// The checkpoint file could not be opened or read.
    Io(String),
    /// The checkpoint's tensors do not match the graph its config implies.
    /// Carries every problem found, not just the first.
    Checkpoint(Vec<String>),
    /// A caller-supplied size, buffer or threshold is unusable.
    Input(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(m) => write!(f, "yolov5 config: {m}"),
            Error::Io(m) => write!(f, "yolov5 checkpoint: {m}"),
            Error::Checkpoint(problems) => {
                write!(f, "yolov5 checkpoint does not match its config ({} problem(s)): {}", problems.len(), problems.join("; "))
            }
            Error::Input(m) => write!(f, "yolov5 input: {m}"),
        }
    }
}

impl std::error::Error for Error {}
