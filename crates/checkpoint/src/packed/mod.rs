// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Quantized tensors as first-class citizens: descriptors, codecs, bounded
//! tiled reads and loads. See [`encoding`] for the vocabulary.

pub mod codec;
pub mod encoding;
#[cfg(test)]
mod testfix;

pub use codec::{Codec, CodecRegistry, Tile, TileBuf};
pub use encoding::{BlockGeometry, Codebook, Element, FloatFormat, Packing, QuantEncoding, ScaleGranularity, ScaleLevel, TensorDescriptor};
