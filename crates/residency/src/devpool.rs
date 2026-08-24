// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Run N independent requests across a pool of devices, **one in flight per
//! device**, and get the answers back in request order.
//!
//! Swedish Embedded AB implements concurrent multi-accelerator serving for
//! production inference systems. If your team needs expertise in turning a
//! serial request loop into real device-parallel throughput without
//! overcommitting a card, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # The gap this fills, and the gap it does NOT
//!
//! [`crate::executor`] already runs **per-device lanes**: jobs for models
//! placed on different devices execute in parallel, and jobs contending for
//! one device serialise on its lane. That is the right shape when a request
//! is small relative to a card, because many requests for one model then
//! batch together inside a single `Instance::run_batch` call.
//!
//! It is not the whole story for a model whose ONE request already occupies a
//! whole card for minutes and cannot be batched with its neighbours at all -
//! a diffusion or video generation, where every request is its own multi-step
//! sampling loop over its own latent. Those requests arrive as a batch on one
//! lane and, with a serial `run_batch`, run one after another while every
//! other card in the machine sits idle.
//!
//! This is the missing piece: a `run_batch` implementation may hand its
//! invocations to a [`DevicePool`] and get them executed across the machine's
//! cards concurrently, in request order, with a hard ceiling of one at a time
//! per card.
//!
//! # Why the ceiling is exactly one per device
//!
//! Not a tuning knob picked for comfort - a memory fact. A real LTX-2.5
//! generation's DiT forward was measured at **16.26 GiB peak on a 24 GiB
//! Tesla P40** at the 1080p token count, and its VAE decode at 16.18 GiB at
//! 720p (both measured with `ltxv_bench`, published in that port's ledger).
//! Two of those on one card do not fit, and the failure mode of trying is a
//! hard `wgpu` out-of-memory abort, not a slowdown. One per device is what
//! the hardware permits; admitting more would trade a correct, slower answer
//! for a crash.
//!
//! Models whose per-request footprint is genuinely small should not use
//! this - they should batch, which is what [`crate::scheduler`] already picks
//! for them.
//!
//! # No device code here, deliberately
//!
//! `brain-residency` depends on `capability` and `memauth` only, and this
//! module keeps it that way: it schedules [`Device`] *labels* and never
//! touches a GPU API. Binding a worker thread to the card it was handed is
//! the caller's job (`crates/cli/src/resident_llm.rs::on_device`, which is
//! already the one place in the workspace that turns a residency `Device`
//! into a `gpu_core::devices::with_gpu` scope). That split is what lets this
//! be unit-tested on a machine with no GPU at all.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::Mutex;

use crate::Device;

/// A set of devices that may each run one request at a time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePool {
    devices: Vec<Device>,
}

impl DevicePool {
    /// A pool over `devices`. An empty list degrades to a single CPU device
    /// rather than a pool that can run nothing: "no devices" is never a
    /// correct answer to "run these requests", and returning zero results for
    /// a non-empty batch would be a silent drop.
    pub fn new(devices: Vec<Device>) -> DevicePool {
        DevicePool { devices: if devices.is_empty() { vec![Device::Cpu] } else { devices } }
    }

    /// How many requests may be in flight at once - one per device.
    pub fn width(&self) -> usize {
        self.devices.len()
    }

    /// The devices, in the order work is offered to them.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Run `job(index, device)` for every `index` in `0..n`, at most
    /// [`Self::width`] at a time and at most one per device, and return the
    /// results **in index order** regardless of completion order.
    ///
    /// Work is claimed by a lock-free cursor rather than pre-partitioned, so
    /// a card that finishes early takes the next waiting request instead of
    /// idling behind a slow neighbour - which matters here precisely because
    /// per-request cost varies (a longer clip, a bigger frame, a cold cache).
    ///
    /// `events(index, event)` is called on the CALLING thread for every event
    /// a job emits, so a caller's `&mut` progress sink needs no lock and no
    /// `Sync` bound - the classic reason to move progress over a channel
    /// rather than share the sink. Events are delivered in the order they
    /// arrive across all workers, which for per-request progress is the only
    /// order that exists.
    ///
    /// A panicking job propagates: the scope re-raises it after the other
    /// workers have finished, rather than leaving the batch half-answered
    /// with no indication of which request died.
    pub fn run_all<T, E, F, S>(&self, n: usize, job: F, mut events: S) -> Vec<T>
    where
        T: Send,
        E: Send,
        F: Fn(usize, Device, &dyn Fn(E)) -> T + Sync,
        S: FnMut(usize, E),
    {
        if n == 0 {
            return Vec::new();
        }
        // The degenerate cases run INLINE - no threads, no channel, no
        // reordering - so a one-request batch (the overwhelmingly common
        // shape) and a one-device machine keep exactly the old serial
        // behaviour, including delivering each event before the next request
        // starts. A pool nobody needs must not change what anybody sees.
        if n == 1 || self.devices.len() == 1 {
            let sink = std::cell::RefCell::new(events);
            return (0..n)
                .map(|i| {
                    let dev = self.devices[i % self.devices.len()];
                    job(i, dev, &|e| (sink.borrow_mut())(i, e))
                })
                .collect();
        }

        let cursor = AtomicUsize::new(0);
        let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
        let (tx, rx) = channel::<(usize, E)>();
        std::thread::scope(|s| {
            for &dev in &self.devices {
                let tx: Sender<(usize, E)> = tx.clone();
                let (cursor, slots, job) = (&cursor, &slots, &job);
                s.spawn(move || {
                    loop {
                        // Claim, then check: `fetch_add` is what makes "no
                        // index runs twice" true without a lock, which is
                        // what lets a job hold its card for minutes without
                        // blocking the others' claims.
                        let i = cursor.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        // A closed receiver means the caller went away; the
                        // job still runs to completion, because abandoning a
                        // half-finished generation to save a progress
                        // message is the wrong trade.
                        let out = job(i, dev, &|e| {
                            let _ = tx.send((i, e));
                        });
                        *slots[i].lock().expect("a job slot is never poisoned - only this worker writes it") = Some(out);
                    }
                });
            }
            // The scope's own sender must go, or the drain below never ends.
            drop(tx);
            while let Ok((i, e)) = rx.recv() {
                events(i, e);
            }
        });
        slots.into_iter().map(|m| m.into_inner().expect("unpoisoned").expect("every index in 0..n is claimed exactly once by the cursor")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn gpus(n: u32) -> DevicePool {
        DevicePool::new((0..n).map(Device::Gpu).collect())
    }

    /// Results come back in REQUEST order even though the requests finish in
    /// the opposite order. A pool that returned completion order would
    /// silently pair request 0's answer with request 3's caller.
    #[test]
    fn results_are_in_request_order_not_completion_order() {
        let pool = gpus(4);
        let out = pool.run_all(
            8,
            |i, _dev, _emit| {
                // Later indices finish first.
                std::thread::sleep(Duration::from_millis(2 * (8 - i) as u64));
                i * 10
            },
            |_, _: ()| {},
        );
        assert_eq!(out, (0..8).map(|i| i * 10).collect::<Vec<_>>());
    }

    /// The ceiling is the point: never more than one request in flight per
    /// device, and never more than `width()` overall. Measured by a live
    /// counter rather than asserted from the code's shape, because "at most
    /// one per card" is a memory-safety claim about a 24 GiB board (see this
    /// module's doc), not a stylistic preference.
    #[test]
    fn at_most_one_request_per_device_is_ever_in_flight() {
        let pool = gpus(3);
        let live = std::sync::Mutex::new(std::collections::HashMap::<Device, usize>::new());
        let peak = AtomicUsize::new(0);
        let total_live = AtomicUsize::new(0);
        pool.run_all(
            12,
            |_i, dev, _emit| {
                {
                    let mut l = live.lock().unwrap();
                    let c = l.entry(dev).or_insert(0);
                    *c += 1;
                    assert_eq!(*c, 1, "two requests in flight on {dev:?}");
                }
                let now = total_live.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(5));
                total_live.fetch_sub(1, Ordering::SeqCst);
                *live.lock().unwrap().get_mut(&dev).unwrap() -= 1;
            },
            |_, _: ()| {},
        );
        assert!(peak.load(Ordering::SeqCst) <= 3, "peak concurrency {} exceeded the pool width", peak.load(Ordering::SeqCst));
        assert!(peak.load(Ordering::SeqCst) >= 2, "nothing ran concurrently at all - the pool degenerated to a serial loop");
    }

    /// The whole reason this exists: N requests over W devices must take
    /// materially less than N x the per-request time. Deliberately a LOOSE
    /// bound (under 70% of serial) rather than a tight one - a timing
    /// assertion tight enough to be impressive is tight enough to be flaky on
    /// a loaded box, and the real speedup numbers belong in a measured
    /// benchmark, not in a correctness gate.
    #[test]
    fn four_requests_over_two_devices_overlap_rather_than_serialise() {
        let pool = gpus(2);
        let per = Duration::from_millis(60);
        let t0 = Instant::now();
        pool.run_all(4, |_i, _dev, _emit| std::thread::sleep(per), |_, _: ()| {});
        let elapsed = t0.elapsed();
        let serial = per * 4;
        assert!(elapsed < serial.mul_f32(0.7), "4 requests on 2 devices took {elapsed:?}, no better than the {serial:?} serial cost");
    }

    /// Progress arrives on the CALLING thread, so a caller's `&mut` sink
    /// needs no lock and no `Sync` bound - and every event is tagged with the
    /// request that produced it, or a server would attribute one client's
    /// progress to another.
    #[test]
    fn events_reach_a_plain_mut_sink_tagged_with_their_request() {
        let pool = gpus(2);
        let caller_thread = std::thread::current().id();
        let mut seen: Vec<(usize, u32)> = Vec::new();
        pool.run_all(
            4,
            |i, _dev, emit| {
                emit(i as u32);
                emit(100 + i as u32);
            },
            |i, e| {
                assert_eq!(std::thread::current().id(), caller_thread, "events must be delivered on the calling thread");
                seen.push((i, e));
            },
        );
        seen.sort_unstable();
        assert_eq!(seen, vec![(0, 0), (0, 100), (1, 1), (1, 101), (2, 2), (2, 102), (3, 3), (3, 103)]);
    }

    /// A one-request batch and a one-device pool must not spawn anything and
    /// must keep the strict "event, then next request" ordering a serial loop
    /// has - the property a streaming client depends on.
    #[test]
    fn the_degenerate_cases_stay_serial_and_ordered() {
        let mut log: Vec<String> = Vec::new();
        DevicePool::new(vec![Device::Gpu(0)]).run_all(
            3,
            |i, _d, emit| {
                emit(format!("start {i}"));
                emit(format!("end {i}"));
            },
            |i, e| log.push(format!("{i}:{e}")),
        );
        assert_eq!(log, vec!["0:start 0", "0:end 0", "1:start 1", "1:end 1", "2:start 2", "2:end 2"]);
        assert_eq!(DevicePool::new(Vec::new()).devices(), &[Device::Cpu], "an empty pool must still be able to run something");
        assert!(DevicePool::new(Vec::new()).run_all(0, |_, _, _: &dyn Fn(())| 1, |_, _| {}).is_empty());
    }
}
