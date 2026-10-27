// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A small error type that carries a chain of context, so a failure deep in a
//! vendor library call reads as "starting the camera: enabling VI channel:
//! RK_MPI_VI_EnableChn returned 0xa0018005" at the top.

use std::fmt;

/// An error message built from the innermost cause outwards.
#[derive(Debug)]
pub struct Error(String);

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(msg: impl Into<String>) -> Self {
        Error(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

/// Prefixes a failure with what the caller was doing.
pub trait Context<T> {
    fn context(self, what: impl fmt::Display) -> Result<T>;
}

impl<T, E: fmt::Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: impl fmt::Display) -> Result<T> {
        self.map_err(|e| Error(format!("{what}: {e}")))
    }
}

/// Builds an [`Error`] from a format string.
macro_rules! err {
    ($($arg:tt)*) => { $crate::error::Error::new(format!($($arg)*)) };
}
pub(crate) use err;
