// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The rows a pipeline stage starts from: the boundary residual the stage
//! before it produced, or the embedding rows of the first stage.
//!
//! A model split across cards hands each stage's last-layer residual to the
//! next. Held as host floats that costs a device-wide wait and two PCIe
//! crossings per boundary per step; held as the previous stage's own device
//! buffer it is one ordered device-to-device copy (`Gpu::copy_to`: straight
//! over the card link when the pair has one, through the host when it does
//! not, with the reason stated once) and the host never waits. The stage's
//! forward takes either through this one type, so there is one implementation
//! of the stage and the host-row entry points are thin wrappers.
//!
//! Swedish Embedded AB implements multi-GPU inference runtimes for its clients.
//! If your team needs expertise in splitting a model across accelerators
//! without paying for a host round trip at every boundary, you can procure our
//! services by sending an email to info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Gpu};

/// A stage's input rows, `[rows, d_model]` row-major.
#[derive(Clone, Copy)]
pub(crate) enum StageRows<'a> {
    /// Rows on the host: the embedding of the first stage, or a caller's own.
    Host(&'a [f32]),
    /// Rows already on a device, usually the previous stage's output.
    Device {
        /// The handle that owns `buf` (the previous stage's card).
        from: &'a Gpu,
        buf: &'a DeviceBuffer,
        /// First word of the rows inside `buf`.
        offset_words: usize,
        words: usize,
    },
    /// No rows: the stage holds the embedding table itself and gathers the step's own tokens on its card
    /// ([`StageRows::provided`] is `None`). Only the entry points that take the step's tokens accept it
    /// (decode step, mixed forward); asking it for rows is a bug.
    Gather,
}

impl<'a> StageRows<'a> {
    /// The first `words` words of `buf`, which lives on `from`.
    pub(crate) fn device(from: &'a Gpu, buf: &'a DeviceBuffer, words: usize) -> StageRows<'a> {
        StageRows::Device { from, buf, offset_words: 0, words }
    }

    /// These rows, or `None` when the stage gathers the step's tokens from its own embedding table.
    pub(crate) fn provided(self) -> Option<StageRows<'a>> {
        (!matches!(self, StageRows::Gather)).then_some(self)
    }

    /// How many f32 words of rows this holds.
    pub(crate) fn words(&self) -> usize {
        match self {
            StageRows::Host(x) => x.len(),
            StageRows::Device { words, .. } => *words,
            StageRows::Gather => unreachable!("StageRows::Gather carries no rows; the entry point must call `provided()` first"),
        }
    }

    /// Words `start..end` of these rows (a group of a batch that runs in several passes). No copy.
    pub(crate) fn slice(&self, start: usize, end: usize) -> StageRows<'a> {
        assert!(start <= end && end <= self.words(), "StageRows::slice {start}..{end} of {} words", self.words());
        match *self {
            StageRows::Host(x) => StageRows::Host(&x[start..end]),
            StageRows::Device { from, buf, offset_words, .. } => StageRows::Device { from, buf, offset_words: offset_words + start, words: end - start },
            StageRows::Gather => unreachable!("StageRows::Gather carries no rows to slice"),
        }
    }

    /// Write the rows into the start of `dst`, a buffer of `g` with room for them. A device source is ordered
    /// behind everything its card has submitted so far and `dst`'s users wait for the copy, so the call does not
    /// wait for either device.
    pub(crate) fn load_into(&self, g: &Gpu, dst: &DeviceBuffer) {
        match *self {
            StageRows::Host(x) => g.write_f32(dst, x),
            StageRows::Device { from, buf, offset_words, words } => {
                from.copy_to(buf, offset_words as u64 * 4, g, dst, 0, words as u64 * 4);
            }
            StageRows::Gather => unreachable!("StageRows::Gather carries no rows to load"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device source loads exactly the words asked for, from the offset a `slice` moved to, and a host source
    /// loads the same way - on the backend the suite runs on (the CPU one under `BRAIN_DEVICE=cpu`).
    #[test]
    fn rows_load_exactly_from_host_and_from_a_device_slice() {
        let g = Gpu::new(&[]);
        let data: Vec<f32> = (0..64).map(|i| f32::from_bits(0x3f80_0000 + i * 977)).collect();
        let src = g.storage_init("src", &data);
        let all = StageRows::device(&g, &src, data.len());
        let part = all.slice(16, 48).slice(8, 24);
        assert_eq!(part.words(), 16);
        let dst = g.storage(16);
        part.load_into(&g, &dst);
        assert_eq!(g.read(&dst, 16), data[24..40]);

        let host = StageRows::Host(&data[..16]);
        let dst = g.storage(16);
        host.load_into(&g, &dst);
        assert_eq!(g.read(&dst, 16), data[..16]);
    }
}
