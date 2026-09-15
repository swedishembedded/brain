// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain` -- the public, embeddable SDK facade over brain's model backends.
//!
//! This first slice covers [`ImagePipeline`]: resolve a model through
//! `crates/loader` (the same resolver `brain flux2 generate` uses, never a
//! parallel path), build a real `flux2::pipeline::Pipeline`, and generate +
//! save images -- with no CLI binary, no capability-dispatch machinery, and
//! no environment variable required in the loop.
//!
//! ```no_run
//! let mut pipe = brain::ImagePipeline::from_pretrained("black-forest-labs/FLUX.2-klein-9B")?;
//! pipe.generate("a whale submarine")?.save("out.png")?;
//! # Ok::<(), brain::Error>(())
//! ```
//!
//! `use brain::ImagePipeline;` is the whole point of this crate's name --
//! see `Cargo.toml` for why `brain` (not `brain-sdk`) is this workspace's
//! one deliberate exception to its `brain-<short>` package-naming
//! convention.
//!
//! Swedish Embedded AB implements client-embeddable model inference for its
//! clients -- turning an internal research pipeline into a small, stable
//! library surface a product can link directly, with no CLI process and no
//! capability-dispatch server in the loop. If your team needs an SDK facade
//! over its own model stack, you can procure our services by sending an
//! email to info@swedishembedded.com.

mod error;
mod image;
mod pipeline;

pub use error::Error;
pub use image::Image;
pub use pipeline::{Device, DType, ImageGenerationOptions, ImagePipeline, ImagePipelineBuilder};

/// This crate's one `Result` alias -- every fallible public entry point
/// returns it.
pub type Result<T> = std::result::Result<T, Error>;
