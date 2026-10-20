// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Where loaded tiles go. A [`TensorSink`] is the destination half of a load:
//! a device upload, a parameter store, a file writer. The executor drives it
//! tile by tile and never hands it more than one tile at a time.

use std::ops::Range;

use super::codec::Tile;
use super::encoding::TensorDescriptor;

/// Destination of a tiled load.
pub trait TensorSink {
    /// A load of `dst` starts. Allocate the destination here (the sink knows
    /// the physical byte sizes from the descriptor).
    fn begin(&mut self, dst: &TensorDescriptor) -> Result<(), String>;

    /// Tile `tile.elems` of `dst`, in `dst`'s encoding. Tiles arrive in
    /// ascending order and cover the tensor exactly once; the borrowed bytes
    /// are valid only for the call. Scale slices of consecutive tiles may
    /// overlap (shared 2-D scale bands) and then carry identical values.
    fn write_tile(&mut self, dst: &TensorDescriptor, tile: &Tile<'_>) -> Result<(), String>;

    /// Every tile was written.
    fn finish(&mut self, dst: &TensorDescriptor) -> Result<(), String>;

    /// The load failed after `begin`: discard the partial destination.
    fn abort(&mut self, dst: &TensorDescriptor);
}

/// Collects a load in host memory at full size. For tests and small tensors:
/// it holds the whole destination, so it is exactly what a bounded load must
/// not be used with for large tensors.
#[derive(Debug, Default)]
pub struct MemorySink {
    pub payload: Vec<u8>,
    pub scales: Vec<Vec<u8>>,
    pub tiles: usize,
    pub finished: bool,
    pub aborted: bool,
}

impl TensorSink for MemorySink {
    fn begin(&mut self, dst: &TensorDescriptor) -> Result<(), String> {
        *self = MemorySink::default();
        self.payload = vec![0; dst.payload_bytes() as usize];
        self.scales = (0..dst.encoding.scales.len()).map(|l| vec![0; dst.scale_bytes(l) as usize]).collect();
        Ok(())
    }

    fn write_tile(&mut self, dst: &TensorDescriptor, tile: &Tile<'_>) -> Result<(), String> {
        let pr = dst.payload_range(&tile.elems);
        self.payload[pr.start as usize..pr.end as usize].copy_from_slice(tile.payload);
        for (l, s) in tile.scales.iter().enumerate() {
            let w = dst.encoding.scales[l].format.bits() as u64 / 8;
            let r: Range<u64> = dst.scale_range(l, &tile.elems);
            self.scales[l][(r.start * w) as usize..(r.end * w) as usize].copy_from_slice(s);
        }
        self.tiles += 1;
        Ok(())
    }

    fn finish(&mut self, _dst: &TensorDescriptor) -> Result<(), String> {
        self.finished = true;
        Ok(())
    }

    fn abort(&mut self, _dst: &TensorDescriptor) {
        self.aborted = true;
    }
}
