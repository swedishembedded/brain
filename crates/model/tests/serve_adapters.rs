// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one model from a single resident copy of its weights for its
// clients. If your team needs expertise in multi-tenant inference scheduling
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec of the scheduler's adapter seam, against a fake decoder with no
//! device: a request for an adapter that is not resident yet is SKIPPED, never
//! waited on, so a cold adapter cannot hold up base traffic; everything else
//! about admission and decode is unchanged; the adapter a sequence runs under
//! is pinned for exactly its lifetime and reaches the decoder as the block
//! table's KV namespace.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use model::paged::{BlockAllocator, BlockTable, NamespaceIds, BASE_KV_NAMESPACE};
use model::serve::{AdapterAvailability, AdapterId, PagedDecoder, RejectReason, Request, Scheduler};

const VOCAB: usize = 64;

/// What a test observes of the fake decoder after handing it to the scheduler.
#[derive(Default)]
struct Probe {
    availability: HashMap<AdapterId, AdapterAvailability>,
    pins: HashMap<AdapterId, i32>,
    /// KV namespace of every prefilled table, in prefill order.
    prefilled: Vec<u32>,
    /// KV namespaces of the rows of every batched decode call.
    decode_batches: Vec<Vec<u32>>,
}

type SharedProbe = Rc<RefCell<Probe>>;

/// Next token = previous token + 1, so a sequence is predictable without a
/// model. Block bookkeeping is the real allocator.
struct Core {
    alloc: BlockAllocator,
    probe: SharedProbe,
}

impl Core {
    fn new(probe: SharedProbe) -> Core {
        Core { alloc: BlockAllocator::new(256, 4), probe }
    }
    fn next_of(&self, tok: u32) -> u32 {
        (tok + 1) % VOCAB as u32
    }
    fn decode(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        self.probe.borrow_mut().decode_batches.push(tables.iter().map(|t| t.kv_namespace()).collect());
        for t in tables.iter_mut() {
            t.append(&mut self.alloc).unwrap();
        }
        inputs.iter().map(|&i| self.next_of(i)).collect()
    }
}

/// Implements the required decoder surface over `self.core`, plus any extra
/// items (the adapter overrides) the caller supplies.
macro_rules! fake_decoder {
    ($ty:ident { $($extra:item)* }) => {
        impl PagedDecoder for $ty {
            fn alloc_mut(&mut self) -> &mut BlockAllocator { &mut self.core.alloc }
            fn max_prefill_tokens(&self) -> u32 { 1024 }
            fn free_blocks(&self) -> u32 { self.core.alloc.free_blocks() }
            fn max_seq_len(&self) -> usize { 512 }
            fn vocab(&self) -> usize { VOCAB }
            fn blocks_for(&self, tokens: u32) -> u32 { tokens.div_ceil(4) }
            fn reclaim_prefix(&mut self, _want: u32) -> u32 { 0 }
            fn release_table(&mut self, t: &mut BlockTable) { t.release(&mut self.core.alloc); }
            fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
                self.core.probe.borrow_mut().prefilled.push(table.kv_namespace());
                table.reserve(prompt.len() as u32, &mut self.core.alloc).unwrap();
                vec![*prompt.last().unwrap() as f32]
            }
            fn logits(&self, hidden: &[f32]) -> Vec<f32> {
                let mut l = vec![0.0; VOCAB];
                l[self.core.next_of(hidden[0] as u32) as usize] = 1.0;
                l
            }
            fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
                self.core.decode(tables, inputs)
            }
            fn forward_batched_greedy_window(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<u32>> {
                unreachable!("the fake reports a window capacity of 1")
            }
            fn prefix_stats(&self) -> (u64, u64, usize) { (0, 0, 0) }
            fn device_stats(&self) -> Option<gpu_core::DeviceStats> { None }
            fn kv_pool_bytes(&self) -> u64 { 0 }
            fn kv_pool_capacity_tokens(&self) -> u64 { 1024 }
            fn decode_window_capacity(&self) -> usize { 1 }
            fn forward_batched_topk(&mut self, _: &mut [&mut BlockTable], _: &[u32], _: usize) -> Vec<Vec<(u32, f32)>> {
                unreachable!("only greedy requests are submitted")
            }
            fn topk_capacity(&self) -> usize { 1 }
            $($extra)*
        }
    };
}

/// A decoder that, like every decoder that has not adopted adapters, relies
/// on the trait's defaults for the adapter methods.
struct BaseOnly {
    core: Core,
}
fake_decoder!(BaseOnly {});

/// A decoder whose adapter residency the test controls through its probe.
struct WithAdapters {
    core: Core,
}
fake_decoder!(WithAdapters {
    fn adapter_availability(&mut self, adapter: AdapterId) -> AdapterAvailability {
        self.core.probe.borrow().availability.get(&adapter).copied().unwrap_or(AdapterAvailability::Unknown)
    }
    fn pin_adapter(&mut self, adapter: AdapterId) {
        assert_eq!(
            self.core.probe.borrow().availability.get(&adapter),
            Some(&AdapterAvailability::Resident),
            "only a resident adapter may be pinned"
        );
        *self.core.probe.borrow_mut().pins.entry(adapter).or_insert(0) += 1;
    }
    fn unpin_adapter(&mut self, adapter: AdapterId) {
        *self.core.probe.borrow_mut().pins.entry(adapter).or_insert(0) -= 1;
    }
});

struct Fixture {
    sched: Scheduler<WithAdapters>,
    probe: SharedProbe,
    ids: NamespaceIds,
}

impl Fixture {
    fn new() -> Fixture {
        let probe = SharedProbe::default();
        let dec = WithAdapters { core: Core::new(probe.clone()) };
        Fixture { sched: Scheduler::new(dec, 8), probe, ids: NamespaceIds::default() }
    }
    fn adapter(&mut self, state: AdapterAvailability) -> AdapterId {
        let id = AdapterId::new(self.ids.fresh()).expect("fresh namespaces are never the base");
        self.probe.borrow_mut().availability.insert(id, state);
        id
    }
    fn set(&self, adapter: AdapterId, state: AdapterAvailability) {
        self.probe.borrow_mut().availability.insert(adapter, state);
    }
    fn pins(&self, adapter: AdapterId) -> i32 {
        self.probe.borrow().pins.get(&adapter).copied().unwrap_or(0)
    }
}

fn req(first: u32, max_new: usize) -> Request {
    Request { prompt: vec![first, first + 1], max_new, stop: Vec::new() }
}

#[test]
fn a_cold_adapter_does_not_delay_base_requests_behind_it() {
    let mut f = Fixture::new();
    let cold = f.adapter(AdapterAvailability::Loading);
    let cold_id = f.sched.submit_for_adapter(req(1, 3), cold, Default::default(), 0);
    let base_a = f.sched.submit(req(10, 3));
    let base_b = f.sched.submit(req(20, 3));

    let report = f.sched.step_report();
    assert_eq!(report.admitted, vec![base_a, base_b], "base requests are admitted past the cold one");
    assert_eq!(f.sched.waiting_len(), 1, "the cold request is still queued, not rejected");
    assert!(report.rejected.is_empty());
    assert_eq!(f.pins(cold), 0, "nothing is pinned for a request that was not admitted");

    // Once the adapter is resident the very next iteration admits it, and
    // it was never moved behind later arrivals.
    f.set(cold, AdapterAvailability::Resident);
    let later = f.sched.submit(req(30, 3));
    let report = f.sched.step_report();
    assert_eq!(report.admitted, vec![cold_id, later]);
    assert_eq!(f.pins(cold), 1);
}

#[test]
fn running_sequences_decode_every_step_while_an_adapter_loads() {
    let mut f = Fixture::new();
    let resident = f.adapter(AdapterAvailability::Resident);
    let cold = f.adapter(AdapterAvailability::Loading);
    let base = f.sched.submit(req(1, 6));
    let warm = f.sched.submit_for_adapter(req(5, 6), resident, Default::default(), 0);
    f.sched.submit_for_adapter(req(9, 6), cold, Default::default(), 0);

    for _ in 0..3 {
        let report = f.sched.step_report();
        let produced: Vec<u64> = report.produced.iter().map(|(id, _)| *id).collect();
        assert!(produced.contains(&base) && produced.contains(&warm), "every running sequence advances each step");
    }
    let batches = f.probe.borrow().decode_batches.clone();
    assert!(batches.len() >= 3);
    for b in &batches {
        assert_eq!(b.len(), 2, "one decode batch per step covering both running sequences");
    }
    assert_eq!(f.sched.waiting_len(), 1);
}

#[test]
fn admission_order_is_arrival_order_when_every_adapter_is_resident() {
    let mut f = Fixture::new();
    let a = f.adapter(AdapterAvailability::Resident);
    let b = f.adapter(AdapterAvailability::Resident);
    let ids = [
        f.sched.submit(req(1, 2)),
        f.sched.submit_for_adapter(req(5, 2), a, Default::default(), 0),
        f.sched.submit(req(9, 2)),
        f.sched.submit_for_adapter(req(13, 2), b, Default::default(), 0),
    ];
    let report = f.sched.step_report();
    assert_eq!(report.admitted, ids.to_vec());
    // The decoder saw each sequence's adapter as its table's KV namespace.
    assert_eq!(f.probe.borrow().prefilled, vec![BASE_KV_NAMESPACE, a.kv_namespace(), BASE_KV_NAMESPACE, b.kv_namespace()]);
}

#[test]
fn a_mixed_batch_decodes_in_one_call_with_each_rows_namespace() {
    let mut f = Fixture::new();
    let a = f.adapter(AdapterAvailability::Resident);
    f.sched.submit(req(1, 4));
    f.sched.submit_for_adapter(req(5, 4), a, Default::default(), 0);
    f.sched.step_report();
    assert_eq!(f.probe.borrow().decode_batches, vec![vec![BASE_KV_NAMESPACE, a.kv_namespace()]]);
}

#[test]
fn the_pin_lasts_exactly_as_long_as_the_sequence() {
    let mut f = Fixture::new();
    let a = f.adapter(AdapterAvailability::Resident);

    // Finishing releases it.
    let done = f.sched.submit_for_adapter(req(1, 2), a, Default::default(), 0);
    let out = f.sched.run();
    assert_eq!(out[&done].len(), 2);
    assert_eq!(f.pins(a), 0);

    // Cancelling mid-decode releases it, and returns the blocks.
    let free = f.sched.free_blocks();
    let doomed = f.sched.submit_for_adapter(req(1, 50), a, Default::default(), 0);
    f.sched.step();
    assert_eq!(f.pins(a), 1);
    f.sched.cancel(doomed);
    assert_eq!(f.pins(a), 0);
    assert_eq!(f.sched.free_blocks(), free);

    // Tearing the scheduler down releases it too.
    f.sched.submit_for_adapter(req(1, 50), a, Default::default(), 0);
    f.sched.step();
    assert_eq!(f.pins(a), 1);
    let probe = f.probe.clone();
    drop(f.sched.into_decoder());
    assert_eq!(probe.borrow().pins[&a], 0);
}

#[test]
fn an_unknown_adapter_is_rejected_without_blocking_the_queue() {
    let mut f = Fixture::new();
    let gone = f.adapter(AdapterAvailability::Unknown);
    let bad = f.sched.submit_for_adapter(req(1, 2), gone, Default::default(), 0);
    let ok = f.sched.submit(req(10, 2));
    let report = f.sched.step_report();
    assert_eq!(report.rejected, vec![(bad, RejectReason::AdapterUnavailable { adapter: gone })]);
    assert_eq!(report.admitted, vec![ok]);
    assert!(RejectReason::AdapterUnavailable { adapter: gone }.to_string().contains("adapter"));
}

#[test]
fn a_decoder_without_adapter_support_rejects_adapter_requests_and_serves_base_unchanged() {
    let probe = SharedProbe::default();
    let mut sched = Scheduler::new(BaseOnly { core: Core::new(probe.clone()) }, 8);
    let adapter = AdapterId::new(NamespaceIds::default().fresh()).unwrap();
    let first = sched.submit(req(1, 3));
    let refused = sched.submit_for_adapter(req(5, 3), adapter, Default::default(), 0);
    let second = sched.submit(req(9, 3));
    let report = sched.step_report();
    assert_eq!(report.admitted, vec![first, second]);
    assert_eq!(report.rejected, vec![(refused, RejectReason::AdapterUnavailable { adapter })]);
    let out = sched.run();
    assert_eq!(out[&first], vec![3, 4, 5], "base decoding is untouched: prompt [1, 2] continues 3, 4, 5");
}
