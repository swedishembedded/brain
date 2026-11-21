// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distributed training runtimes that keep
// gradient traffic on the accelerator interconnect for its clients. If your
// team needs expertise in multi-node collectives and their failure semantics,
// you can procure our services by sending an email to info@swedishembedded.com.

//! A distributed job's collective with a device-resident gradient path.
//!
//! [`JobCollective`] wraps the job's sealed host transport (the coordinator
//! star every rank joined with the job secret). Its payload ops are the host
//! transport's, unchanged: small scalars such as a step's token count or an
//! evaluation broadcast. Its [`Collective::all_reduce_device`] reduces
//! gradients where they live, on the card, over a device collective (NCCL),
//! so a data-parallel step stages none of them through host memory.
//!
//! The device collective is formed on the first device reduction, by
//! [`rendezvous`] over the host transport: every rank states whether it can
//! take part, rank 0 creates the rendezvous id and broadcasts it over the
//! sealed channel - so only holders of the job secret ever see it - and every
//! rank joins with it and reports whether it did. Only when every rank joined
//! does any rank use it; otherwise every rank reduces through the host star,
//! and each says why, once. A plan that asks for [`CollectivePath::Device`]
//! fails instead of falling back.
//!
//! Every device reduction is waited for within the plan's op timeout, so a
//! peer that dies mid-reduction is an error on the surviving ranks (and the
//! communicator is aborted), not a hang.
//!
//! Reproducibility: every rank receives the same reduced bytes from the device
//! all-reduce (NCCL computes each element once and distributes it), so the
//! replicas stay bit-identical. The sum is not the host star's rank-ordered
//! sum, and NCCL may pick another algorithm for another world size, topology or
//! library version, so a device run is not bitwise comparable with a host run;
//! for a fixed world, card set and library it repeats bit for bit (measured by
//! `crates/model/tests/it/nccl_job.rs`, not a guarantee NCCL documents).

use std::sync::Mutex;
use std::time::Duration;

use gpu_core::devgroup::RankGroup;
use gpu_core::{CollectiveDtype, CollectiveId, DeviceBuffer, Gpu};
use serde::{Deserialize, Serialize};

use crate::collective::{Collective, CollectiveError, CollectiveFuture, CollectiveResult, Payload};

/// Which transport a distributed job reduces its gradients over; the plan's
/// `collectives` field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CollectivePath {
    /// On the device when every rank can, otherwise through the host star,
    /// saying why.
    #[default]
    Auto,
    /// Always through the host star.
    Host,
    /// On the device, or the job fails.
    Device,
}

impl CollectivePath {
    /// Whether this is the default, which a plan file leaves out.
    pub fn is_auto(&self) -> bool {
        *self == CollectivePath::Auto
    }
}

fn flag(ok: bool) -> f32 {
    if ok {
        1.0
    } else {
        0.0
    }
}

/// The ranks whose gathered flag is not set.
fn unset(flags: &[f32]) -> Vec<usize> {
    flags.iter().enumerate().filter(|(_, f)| **f != 1.0).map(|(r, _)| r).collect()
}

/// The id as the payload rank 0 broadcasts: a presence flag, then one byte per
/// element (exact in fp32).
fn encode(id: Option<&CollectiveId>) -> Vec<f32> {
    match id {
        Some(id) => std::iter::once(1.0).chain(id.0.iter().map(|&b| b as f32)).collect(),
        None => vec![0.0],
    }
}

/// The inverse of [`encode`], refusing anything it could not have produced.
fn decode(data: &[f32]) -> CollectiveResult<Option<CollectiveId>> {
    let malformed = || CollectiveError::Transport(format!("the rendezvous id rank 0 sent is malformed ({} values)", data.len()));
    match data.split_first() {
        Some((&0.0, [])) => Ok(None),
        Some((&1.0, bytes)) if bytes.len() == CollectiveId::BYTES => {
            let mut id = [0u8; CollectiveId::BYTES];
            for (out, &v) in id.iter_mut().zip(bytes) {
                if !(0.0..=255.0).contains(&v) || v.fract() != 0.0 {
                    return Err(malformed());
                }
                *out = v as u8;
            }
            Ok(Some(CollectiveId(id)))
        }
        _ => Err(malformed()),
    }
}

/// Form a device collective over the ranks of `host`, or agree on why not.
///
/// `local` is whether this rank could take part. Rank 0 calls `make_id` and
/// sends the id over `host` (the job's authenticated channel); every rank then
/// calls `join` with it. Every rank returns the same verdict: `Ok(Ok(group))`
/// only when every rank joined, `Ok(Err(reason))` when any could not (a group
/// this rank did form is dropped), and `Err` when `host` itself failed.
pub fn rendezvous<G>(
    host: &dyn Collective,
    rank: usize,
    local: Result<(), String>,
    make_id: impl FnOnce() -> Result<CollectiveId, String>,
    join: impl FnOnce(&CollectiveId) -> Result<G, String>,
) -> CollectiveResult<Result<G, String>> {
    let ready = pollster::block_on(host.all_gather(rank, Payload::f32(vec![flag(local.is_ok())])))?.data;
    if let Err(why) = local {
        return Ok(Err(why));
    }
    let missing = unset(&ready);
    if !missing.is_empty() {
        return Ok(Err(format!("rank(s) {missing:?} cannot use a device collective")));
    }
    let mut make_error = None;
    let offer = if rank == 0 {
        match make_id() {
            Ok(id) => encode(Some(&id)),
            Err(e) => {
                make_error = Some(e);
                encode(None)
            }
        }
    } else {
        Vec::new()
    };
    let sent = pollster::block_on(host.broadcast(rank, Payload::f32(offer), 0))?.data;
    let Some(id) = decode(&sent)? else {
        return Ok(Err(make_error.map_or_else(|| "rank 0 could not create the rendezvous id".to_string(), |e| format!("rank 0 could not create the rendezvous id: {e}"))));
    };
    let joined = join(&id);
    let joins = pollster::block_on(host.all_gather(rank, Payload::f32(vec![flag(joined.is_ok())])))?.data;
    let failed = unset(&joins);
    Ok(match joined {
        Err(e) => Err(format!("joining the device collective: {e}")),
        Ok(_) if !failed.is_empty() => Err(format!("rank(s) {failed:?} could not join the device collective")),
        Ok(group) => Ok(group),
    })
}

/// What a [`JobCollective`] has settled on for device reductions.
enum DevicePath {
    Undecided,
    Device(Box<RankGroup>),
    Host,
}

/// A job's collective: payload ops over the sealed host transport, gradient
/// reductions on the device when every rank can (module documentation).
pub struct JobCollective<C: Collective> {
    host: C,
    path: CollectivePath,
    join_timeout: Duration,
    op_timeout: Duration,
    device: Mutex<DevicePath>,
}

impl<C: Collective> JobCollective<C> {
    /// Wrap the job's host transport; `join_timeout` bounds forming the device
    /// collective and `op_timeout` every device reduction.
    pub fn new(host: C, path: CollectivePath, join_timeout: Duration, op_timeout: Duration) -> JobCollective<C> {
        JobCollective { host, path, join_timeout, op_timeout, device: Mutex::new(DevicePath::Undecided) }
    }

    /// Who reduces device gradients: the device library's name (`"nccl"`),
    /// `"host"` once the ranks settled on the host star, `None` before the first
    /// device reduction decided it.
    pub fn device_provider(&self) -> Option<&'static str> {
        match &*self.device.lock().unwrap_or_else(|e| e.into_inner()) {
            DevicePath::Undecided => None,
            DevicePath::Device(group) => Some(group.provider()),
            DevicePath::Host => Some("host"),
        }
    }

    /// Settle the path on the first device reduction (every rank calls it at
    /// the same point of the step, so the rendezvous lines up).
    fn decide(&self, rank: usize, gpu: &Gpu) -> CollectiveResult<DevicePath> {
        if self.path == CollectivePath::Host {
            eprintln!("brain: rank {rank}: gradients reduce through the host star (the job plan asks for host collectives)");
            return Ok(DevicePath::Host);
        }
        let world = self.host.world_size();
        let verdict = rendezvous(&self.host, rank, RankGroup::supported(gpu), || RankGroup::id(gpu), |id| RankGroup::join(gpu, id, world, rank, self.join_timeout))?;
        match verdict {
            Ok(group) => {
                eprintln!("brain: rank {rank}: gradients reduce on the device over {} ({world} ranks)", group.provider());
                Ok(DevicePath::Device(Box::new(group)))
            }
            Err(why) if self.path == CollectivePath::Device => Err(CollectiveError::Transport(format!("the job plan asks for device collectives, but {why}"))),
            Err(why) => {
                eprintln!("brain: rank {rank}: gradients reduce through the host star ({why})");
                Ok(DevicePath::Host)
            }
        }
    }
}

impl<C: Collective> Collective for JobCollective<C> {
    fn world_size(&self) -> usize {
        self.host.world_size()
    }

    fn all_reduce<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        self.host.all_reduce(rank, local)
    }

    fn all_gather<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        self.host.all_gather(rank, local)
    }

    fn reduce_scatter<'a>(&'a self, rank: usize, local: Payload) -> CollectiveFuture<'a, Payload> {
        self.host.reduce_scatter(rank, local)
    }

    fn broadcast<'a>(&'a self, rank: usize, local: Payload, root: usize) -> CollectiveFuture<'a, Payload> {
        self.host.broadcast(rank, local, root)
    }

    fn all_reduce_device<'a>(&'a self, rank: usize, gpu: &'a Gpu, bufs: &'a [(&'a DeviceBuffer, u64)], scale: f32) -> CollectiveFuture<'a, bool> {
        Box::pin(async move {
            let mut path = self.device.lock().unwrap_or_else(|e| e.into_inner());
            if matches!(*path, DevicePath::Undecided) {
                *path = self.decide(rank, gpu)?;
            }
            match &*path {
                DevicePath::Device(group) => {
                    group.all_reduce_sum(bufs, CollectiveDtype::F32, scale).and_then(|()| group.wait(self.op_timeout)).map_err(CollectiveError::Transport)?;
                    Ok(true)
                }
                _ => Ok(false),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HostCollective;

    /// Run `body(rank)` on one thread per rank and collect the results in rank order.
    fn ranks<T: Send>(world: usize, body: impl Fn(usize) -> T + Sync) -> Vec<T> {
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..world).map(|r| s.spawn({
                let body = &body;
                move || body(r)
            })).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    fn one_rank() -> HostCollective {
        std::sync::Arc::into_inner(HostCollective::new(1)).expect("a fresh group has one owner")
    }

    fn id(byte: u8) -> CollectiveId {
        let mut bytes = [0u8; CollectiveId::BYTES];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = byte.wrapping_add(i as u8);
        }
        CollectiveId(bytes)
    }

    #[test]
    fn every_rank_joins_with_the_id_rank_zero_made() {
        let host = HostCollective::new(3);
        let joined = ranks(3, |r| rendezvous(&*host, r, Ok(()), || Ok(id(200)), |got| Ok(got.clone())).unwrap());
        for got in joined {
            assert_eq!(got.unwrap(), id(200));
        }
    }

    #[test]
    fn a_rank_that_cannot_take_part_sends_every_rank_to_the_host() {
        let host = HostCollective::new(3);
        let made = std::sync::atomic::AtomicBool::new(false);
        let out = ranks(3, |r| {
            let local = if r == 2 { Err("no CUDA card".to_string()) } else { Ok(()) };
            rendezvous(&*host, r, local, || {
                made.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(id(1))
            }, |_| Ok(()))
            .unwrap()
        });
        assert_eq!(out[2], Err("no CUDA card".to_string()));
        for r in [0, 1] {
            assert!(out[r].as_ref().unwrap_err().contains("[2]"), "{:?}", out[r]);
        }
        assert!(!made.into_inner(), "no id is created when a rank cannot use it");
    }

    #[test]
    fn a_failed_id_or_join_anywhere_is_every_rank_s_fallback() {
        let host = HostCollective::new(2);
        let out = ranks(2, |r| rendezvous(&*host, r, Ok(()), || Err("bootstrap failed".to_string()), |_| Ok(())).unwrap());
        for o in &out {
            assert!(o.as_ref().unwrap_err().contains("rank 0 could not create"), "{o:?}");
        }
        let host = HostCollective::new(2);
        let out = ranks(2, |r| rendezvous(&*host, r, Ok(()), || Ok(id(7)), |_| if r == 1 { Err("timed out".to_string()) } else { Ok(()) }).unwrap());
        assert!(out[0].as_ref().unwrap_err().contains("[1]"), "{:?}", out[0]);
        assert!(out[1].as_ref().unwrap_err().contains("timed out"), "{:?}", out[1]);
    }

    #[test]
    fn a_malformed_id_payload_is_refused() {
        assert_eq!(decode(&encode(Some(&id(9)))).unwrap(), Some(id(9)));
        assert_eq!(decode(&encode(None)).unwrap(), None);
        assert!(decode(&[1.0, 2.0]).is_err());
        let mut bad = encode(Some(&id(9)));
        bad[5] = 256.0;
        assert!(decode(&bad).is_err());
        bad[5] = 1.5;
        assert!(decode(&bad).is_err());
    }

    #[test]
    fn a_plan_that_asks_for_the_host_never_forms_a_device_group() {
        let job = JobCollective::new(one_rank(), CollectivePath::Host, Duration::from_secs(1), Duration::from_secs(1));
        assert_eq!(job.device_provider(), None);
        let gpu = gpu_core::Gpu::new_cpu(&[]);
        let buf = gpu.storage(4);
        let bufs = [(&buf, 4u64)];
        assert!(!pollster::block_on(job.all_reduce_device(0, &gpu, &bufs, 1.0)).unwrap());
        assert_eq!(job.device_provider(), Some("host"));
    }

    #[test]
    fn a_plan_that_requires_the_device_fails_where_there_is_none() {
        let job = JobCollective::new(one_rank(), CollectivePath::Device, Duration::from_secs(1), Duration::from_secs(1));
        let gpu = gpu_core::Gpu::new_cpu(&[]);
        let buf = gpu.storage(4);
        let bufs = [(&buf, 4u64)];
        let err = pollster::block_on(job.all_reduce_device(0, &gpu, &bufs, 1.0)).unwrap_err();
        assert!(err.to_string().contains("asks for device collectives"), "{err}");
        let auto = JobCollective::new(one_rank(), CollectivePath::Auto, Duration::from_secs(1), Duration::from_secs(1));
        assert!(!pollster::block_on(auto.all_reduce_device(0, &gpu, &bufs, 1.0)).unwrap());
        assert_eq!(auto.device_provider(), Some("host"));
    }
}
