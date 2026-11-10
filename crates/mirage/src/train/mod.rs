// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gradient-checked training stacks for compact perception models for its clients.
// If your team needs expertise in fine-tuning detection, segmentation and text-recognition models then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Training of Mirage: a generic-precision reverse-mode graph with hand-written backward primitives.
//!
//! The graph runs in `f64` as the finite-difference oracle and in `f32` as the trainer (see [`real`]). Every
//! primitive in [`ops`], [`spatial`] and [`seq`] has its backward checked against finite differences.

pub mod decoder;
pub mod net;
pub mod ops;
pub mod real;
pub mod seq;
pub mod spatial;
pub mod tape;
pub mod vision;
