// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! brain's own compiler and runtime for the Rockchip RV1103/RV1106 NPU.
//!
//! No vendor SDK is involved: brain lowers an int8 network to the register
//! command programs the NPU's program counter executes, and submits them
//! through the vendor kernel driver's `/dev/rknpu` ioctls.
//!
//! - [`driver`]: the device, NPU memory and job submission.
//! - [`regs`]: the task register profile and program encoding.
//! - [`layout`]: the NPU's 16-lane feature-map layout.
//! - [`quant`]: the int8 arithmetic the hardware performs, exactly.
//! - [`conv`]: dense convolution tasks, with height tiling.
//! - [`job`]: a linked chain of tasks and the arena it addresses.
//!
//! Every task kind also has a host reference that computes the identical
//! integers, so a board run is checked byte for byte, not within a tolerance.
//!
//! The register knowledge is ported from open-rknpu (MIT), with names from
//! Mesa's Rocket driver (MIT); see `/NOTICE.md`.

pub mod conv;
pub mod driver;
pub mod job;
pub mod layout;
pub mod quant;
pub mod regs;

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// A driver call failed; `op` names it.
    Driver { op: &'static str, source: std::io::Error },
    /// The layer or graph is outside what this NPU (or this compiler) supports.
    Unsupported(String),
    /// Float parameters cannot be represented in the NPU's integers.
    Quantization(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Driver { op, source } => write!(f, "rknpu: {op} failed: {source}"),
            Error::Unsupported(why) => write!(f, "rknpu: unsupported: {why}"),
            Error::Quantization(why) => write!(f, "rknpu: quantization: {why}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Driver { source, .. } => Some(source),
            _ => None,
        }
    }
}
