// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in layer-streamed full fine-tuning then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Deferred publication of updated values.
//!
//! A layer-streamed trainer updates a tensor as soon as its gradient is
//! complete. That is only safe when nothing later in the same step still reads
//! the old value (tied embeddings, recomputed activations, a second pass over
//! shared weights all do). [`DeferredPublish`] wraps any [`ParamTiles`]: reads
//! and gradients go to the live store, updated values go to a pending store,
//! and [`DeferredPublish::publish`] copies the pending tensors over the live
//! ones once the step's gradients are all computed - so every gradient of a
//! step is taken against one parameter version.
//!
//! The pending store holds one tensor per updated tensor of the step: in RAM or,
//! with a file backing, on disk.

use std::io;

use crate::codec::Rounding;
use crate::storage::{Backing, BlockStateStorage, OptimizerStateStorage};
use crate::streamed::ParamTiles;

pub struct DeferredPublish<'a, B: Backing> {
    live: &'a mut dyn ParamTiles,
    /// One f32 slot per tensor (see `streamed::tensor_layout`).
    pending: BlockStateStorage<B>,
    numels: Vec<usize>,
    written: Vec<usize>,
    /// Rounding the optimiser asked for each tensor's value; applied when the
    /// live store receives it, so deferral does not change the result.
    rounding: Vec<Rounding>,
    tile: usize,
    buf: Vec<f32>,
}

impl<'a, B: Backing> DeferredPublish<'a, B> {
    /// `pending` must hold one f32 slot per tensor with `numels` elements each
    /// and be able to stage tiles of `tile` elements.
    pub fn new(live: &'a mut dyn ParamTiles, pending: BlockStateStorage<B>, numels: Vec<usize>, tile: usize) -> Self {
        let written = vec![0; numels.len()];
        let rounding = vec![Rounding::NEAREST; numels.len()];
        DeferredPublish { live, pending, numels, written, rounding, tile, buf: vec![0.0; tile] }
    }

    /// Copy every fully written pending tensor over its live value and forget
    /// it. Errors if a tensor was only partly updated: publishing half a layer
    /// would mix two parameter versions inside it.
    pub fn publish(&mut self) -> io::Result<usize> {
        let mut published = 0;
        for ti in 0..self.numels.len() {
            let w = self.written[ti];
            if w == 0 {
                continue;
            }
            if w != self.numels[ti] {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("tensor {ti}: {w} of {} elements updated before publish", self.numels[ti])));
            }
            for start in (0..self.numels[ti]).step_by(self.tile) {
                let n = self.tile.min(self.numels[ti] - start);
                self.pending.read_tile(ti, 0, start, &mut self.buf[..n])?;
                self.live.write_value(ti, start, &self.buf[..n], self.rounding[ti])?;
            }
            self.written[ti] = 0;
            published += 1;
        }
        Ok(published)
    }
}

impl<B: Backing> ParamTiles for DeferredPublish<'_, B> {
    fn read_value(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        self.live.read_value(tensor, start, out)
    }

    fn write_value(&mut self, tensor: usize, start: usize, vals: &[f32], rnd: Rounding) -> io::Result<()> {
        // The pending store keeps the optimiser's f32 result; the live store
        // applies the value rounding when the tensor is published.
        self.rounding[tensor] = rnd;
        self.pending.write_tile(tensor, 0, start, vals, Rounding::NEAREST)?;
        self.written[tensor] += vals.len();
        Ok(())
    }

    fn read_grad(&mut self, tensor: usize, start: usize, out: &mut [f32]) -> io::Result<()> {
        self.live.read_grad(tensor, start, out)
    }

    fn write_grad(&mut self, tensor: usize, start: usize, grad: &[f32]) -> io::Result<()> {
        self.live.write_grad(tensor, start, grad)
    }
}
