// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Quantized tensors as first-class citizens: descriptors, codecs, bounded
//! tiled reads and loads. See [`encoding`] for the vocabulary.

pub mod budget;
pub mod codec;
mod concat;
pub mod encoding;
pub mod file_sink;
pub mod load;
pub mod reader;
pub mod sink;
pub mod source;
#[cfg(test)]
mod testfix;

pub use budget::{BoundedWorkspace, BudgetError, Lease, MemoryBudget, Pool, PoolStats};
pub use codec::{Codec, CodecRegistry, EncodeCtx, Tile, TileBuf};
pub use encoding::{BlockGeometry, Codebook, Element, FloatFormat, Packing, QuantEncoding, ScaleGranularity, ScaleLevel, TensorDescriptor, INT8_SCALE_SUFFIX};
pub use load::{LoadConfig, LoadOp, LoadPlan, LoadPolicy, LoadReport, LoadRequest, LoadStep, Loader};
pub use reader::{PackedReader, PackedTensorReader};
pub use file_sink::SafetensorsSink;
pub use sink::{MemorySink, TensorSink};
pub use source::{stored_source, Binding, PackedSource};
