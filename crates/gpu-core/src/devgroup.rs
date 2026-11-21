// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Collectives over several cards: all-reduce and all-gather on `DeviceBuffer`s.
//!
//! [`DeviceGroup`] is the shape one process driving several cards uses (a
//! tensor-parallel server). [`RankGroup`] is one rank of a group whose ranks are
//! separate processes, one card each, as a distributed job runs: it has no host
//! fallback of its own, because the ranks' host transport is the job's (the
//! caller decides, and says why, when [`RankGroup::join`] fails).
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

use std::time::Duration;

use crate::{CollectiveDtype, CollectiveId, DeviceBuffer, Gpu};

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

/// One rank of a device collective whose ranks are separate processes, each driving one card
/// (`backend_api::RankCollective`). Every operation is enqueued behind the work already submitted on its buffers
/// and completes on the device; [`RankGroup::wait`] is where a failed or stalled peer surfaces, within its deadline.
pub struct RankGroup {
    gpu: Gpu,
    comm: Box<dyn backend_api::RankCollective>,
}

impl RankGroup {
    /// Whether `gpu` could join a group of this kind, without creating one; the reason when it could not.
    pub fn supported(gpu: &Gpu) -> Result<(), String> {
        gpu.rank_collective_support()
    }

    /// A fresh rendezvous id, made on `gpu`'s backend: one rank creates it and hands it to the others over a channel
    /// the job authenticates (whoever holds it can join). `Err` with the reason when the backend has no device
    /// collective of this kind (not CUDA, or NCCL absent or too old).
    pub fn id(gpu: &Gpu) -> Result<CollectiveId, String> {
        gpu.collective_id()
    }

    /// Join the group `id` names as `rank` of `world` on `gpu`'s card, or fail once `timeout` passes without every
    /// rank having joined.
    pub fn join(gpu: &Gpu, id: &CollectiveId, world: usize, rank: usize, timeout: Duration) -> Result<RankGroup, String> {
        let comm = gpu.join_collective(id, world, rank, timeout)?;
        Ok(RankGroup { gpu: gpu.share(), comm })
    }

    /// Who moves the bytes (`"nccl"`).
    pub fn provider(&self) -> &'static str {
        self.comm.provider()
    }

    /// How many ranks the group has.
    pub fn world(&self) -> usize {
        self.comm.world()
    }

    /// This process's rank in it.
    pub fn rank(&self) -> usize {
        self.comm.rank()
    }

    /// In place, for every `(buf, count)`: the sum over the ranks of `scale * x` (each rank's own `scale`), the
    /// buffers reduced as one group. Every rank receives the same bytes.
    pub fn all_reduce_sum(&self, bufs: &[(&DeviceBuffer, u64)], dtype: CollectiveDtype, scale: f32) -> Result<(), String> {
        self.gpu.release_held_work();
        self.comm.all_reduce_sum(bufs, dtype, scale)
    }

    /// `recv` (`world * count` elements) becomes every rank's first `count` elements of `send`, in rank order.
    pub fn all_gather(&self, send: &DeviceBuffer, recv: &DeviceBuffer, count: u64, dtype: CollectiveDtype) -> Result<(), String> {
        self.gpu.release_held_work();
        self.comm.all_gather(send, recv, count, dtype)
    }

    /// `recv` (`count` elements) becomes this rank's slice of the element-wise sum of every rank's `send`
    /// (`world * count` elements).
    pub fn reduce_scatter(&self, send: &DeviceBuffer, recv: &DeviceBuffer, count: u64, dtype: CollectiveDtype) -> Result<(), String> {
        self.gpu.release_held_work();
        self.comm.reduce_scatter(send, recv, count, dtype)
    }

    /// `buf`'s first `count` elements become `root`'s.
    pub fn broadcast(&self, buf: &DeviceBuffer, count: u64, dtype: CollectiveDtype, root: usize) -> Result<(), String> {
        self.gpu.release_held_work();
        self.comm.broadcast(buf, count, dtype, root)
    }

    /// The sub-group of the ranks that pass the same `color`, ordered by `key`; `None` joins none. Every rank of
    /// this group calls it.
    pub fn split(&self, color: Option<u32>, key: u32, timeout: Duration) -> Result<Option<RankGroup>, String> {
        Ok(self.comm.split(color, key, timeout)?.map(|comm| RankGroup { gpu: self.gpu.share(), comm }))
    }

    /// Wait until everything enqueued has completed; a failed peer, or `timeout` passing first, aborts the group
    /// and is the error.
    pub fn wait(&self, timeout: Duration) -> Result<(), String> {
        self.comm.wait(timeout)
    }
}
