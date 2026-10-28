// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Collectives over several cards: all-reduce and all-gather on `DeviceBuffer`s.
//!
//! When the backend has a device collective library (NCCL for CUDA) the
//! payload never leaves the cards and the operations are stream-ordered, so a
//! tensor-parallel layer needs no host wait. When it does not - another
//! backend, the library absent, a card set it refuses - the group says why,
//! once, and reduces through host memory instead: the same results, slower.
//! A failure of a device collective that was opened is an error, never a silent
//! retry through the host.
//!
//! Swedish Embedded AB implements multi-GPU training and inference runtimes for
//! its clients. If your team needs expertise in keeping tensor-parallel and
//! data-parallel traffic on the GPU interconnect instead of staging it through
//! host memory, you can procure our services by sending an email to
//! info@swedishembedded.com.

use crate::{DeviceBuffer, Gpu};

/// A group of cards that run collectives together. Rank = position in the slice given to [`DeviceGroup::open`];
/// every operation takes one buffer per rank, in rank order, each on its rank's card.
pub struct DeviceGroup {
    members: Vec<Gpu>,
    device: Option<Box<dyn backend_api::DeviceCollective>>,
}

impl DeviceGroup {
    /// A group over `members` (at least two cards). Never fails: without a device collective for this card set the
    /// group reduces through the host, and the reason is printed once.
    pub fn open(members: &[&Gpu]) -> DeviceGroup {
        assert!(members.len() >= 2, "DeviceGroup::open: a collective needs at least two cards, got {}", members.len());
        let device = match members[0].open_device_collective(members) {
            Ok(c) => Some(c),
            Err(reason) => {
                eprintln!("brain: collectives across {} cards go through host memory ({reason})", members.len());
                None
            }
        };
        DeviceGroup { members: members.iter().map(|g| g.share()).collect(), device }
    }

    /// Who moves the bytes: `"nccl"`, or `"host"` for the fallback.
    pub fn provider(&self) -> &'static str {
        self.device.as_ref().map_or("host", |d| d.provider())
    }

    fn check(&self, what: &str, counts: &[usize]) -> Result<(), String> {
        match counts.iter().find(|&&c| c != self.members.len()) {
            Some(c) => Err(format!("{what}: {c} buffers for a group of {} cards", self.members.len())),
            None => Ok(()),
        }
    }

    /// Every rank's `bufs[i]` (first `words` f32 words) becomes the element-wise sum over the ranks, added in rank
    /// order on the host path.
    pub fn all_reduce_sum(&self, bufs: &[&DeviceBuffer], words: u64) -> Result<(), String> {
        self.check("all_reduce_sum", &[bufs.len()])?;
        if let Some(d) = &self.device {
            self.members.iter().for_each(Gpu::release_held_work);
            return d.all_reduce_sum(bufs, words);
        }
        let n = words as usize;
        let mut sum = self.members[0].read(bufs[0], n);
        for (g, b) in self.members.iter().zip(bufs).skip(1) {
            for (s, x) in sum.iter_mut().zip(g.read(b, n)) {
                *s += x;
            }
        }
        for (g, b) in self.members.iter().zip(bufs) {
            g.write_f32(b, &sum);
        }
        Ok(())
    }

    /// Every rank's `recv[i]` (`ranks * words` f32 words) becomes the concatenation, in rank order, of every rank's
    /// first `words` words of `send`.
    pub fn all_gather(&self, send: &[&DeviceBuffer], recv: &[&DeviceBuffer], words: u64) -> Result<(), String> {
        self.check("all_gather", &[send.len(), recv.len()])?;
        if let Some(d) = &self.device {
            self.members.iter().for_each(Gpu::release_held_work);
            return d.all_gather(send, recv, words);
        }
        let n = words as usize;
        let mut all = Vec::with_capacity(n * self.members.len());
        for (g, b) in self.members.iter().zip(send) {
            all.extend(g.read(b, n));
        }
        for (g, b) in self.members.iter().zip(recv) {
            g.write_f32(b, &all);
        }
        Ok(())
    }
}
