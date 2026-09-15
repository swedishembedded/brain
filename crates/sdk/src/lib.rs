// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain` -- the public, embeddable SDK facade over brain's model backends.
//!
//! This first slice covers [`ImagePipeline`]: resolve a model through
//! `crates/loader` (the same resolver `brain flux2 generate`/`brain do
//! z-image text2image` uses, never a parallel path), build a real,
//! resident flux2- or s3dit-backed pipeline, and generate + save images --
//! with no CLI binary, no capability-dispatch machinery, and no environment
//! variable required in the loop.
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
//! ## Resource safety
//!
//! This crate is an in-process library, not a server: unlike
//! `crates/apiserve`/`crates/dbus`, it has no admission control, request
//! queue, or concurrency limit of its own -- the "resource safety &
//! backpressure" discipline those network-facing surfaces are held to does
//! not apply the same way to an in-process call. Each
//! `ImagePipeline` you build holds real, multi-gigabyte GPU/host memory for
//! as long as it lives; building several concurrently (or handing untrusted
//! callers direct control over how many get built, or over
//! [`ImageGenerationOptions`]'s `steps`/size, which are NOT range-checked
//! here -- see that type's own doc) is entirely this crate's caller's
//! responsibility to bound, exactly as it is a caller's responsibility to
//! bound any other in-process allocation. A service built ON TOP of this
//! crate that accepts requests from an untrusted network needs its own
//! admission/backpressure layer in front of it -- this crate does not, and
//! is not meant to, provide one.
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
