// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements out-of-core training and inference on GPU
// servers for its clients. If your team needs expertise in streaming models
// that do not fit in device memory without leaving the accelerator idle, you
// can procure our services by sending an email to info@swedishembedded.com.

//! Ordered, bounded prefetch of the units a streaming pass visits (the decoder layers of a model whose weights
//! do not fit on the device): [`for_each_prefetched`] produces unit `i + 1` (and a few after it) on worker
//! threads while the caller consumes unit `i`, hands them over strictly in the requested order, and never has
//! more than `window` units alive at once - so the memory a pass adds is a configured number of units, whatever
//! the model depth. It knows nothing about weights or devices: the producer closure is the model's loader.

use std::collections::BTreeMap;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// What a prefetched pass cost the consumer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefetchStats {
    /// Time the consumer spent blocked waiting for the next unit: a pass bound by the producers shows it here,
    /// one bound by the consumer shows (almost) none.
    pub waited: Duration,
}

struct State<T> {
    /// Position in the order of the next unit to start producing.
    next: usize,
    /// Units fully consumed and dropped.
    done: usize,
    ready: BTreeMap<usize, T>,
    /// The first panic of a producer, re-raised on the consumer's thread.
    failed: Option<Box<dyn std::any::Any + Send>>,
    /// The consumer stopped (returned or panicked): producers stop starting units.
    closed: bool,
}

struct Shared<T> {
    state: Mutex<State<T>>,
    changed: Condvar,
}

impl<T> Shared<T> {
    fn lock(&self) -> std::sync::MutexGuard<'_, State<T>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Closes the stream when the consumer leaves, however it leaves, so no producer waits for a unit nobody takes.
struct CloseOnDrop<'a, T>(&'a Shared<T>);

impl<T> Drop for CloseOnDrop<'_, T> {
    fn drop(&mut self) {
        self.0.lock().closed = true;
        self.0.changed.notify_all();
    }
}

/// Call `consume(unit, &item)` for every `unit` of `order`, in that order, where `item = produce(unit)` was
/// made on one of `workers` producer threads.
///
/// At most `window` items exist at once, counting the one being consumed and any under production (`window` is
/// at least 2 and `workers` is capped at `window - 1`: a worker needs a free place for its item). An item is
/// dropped before the place it held is reused, so `window` bounds what the producers may hold on the device
/// and on the host. A producer or consumer panic propagates to the caller; no thread is left waiting.
pub fn for_each_prefetched<T: Send>(order: &[usize], workers: usize, window: usize, produce: &(impl Fn(usize) -> T + Sync), mut consume: impl FnMut(usize, &T)) -> PrefetchStats {
    let window = window.max(2);
    let workers = workers.clamp(1, window - 1);
    let shared = Shared { state: Mutex::new(State { next: 0, done: 0, ready: BTreeMap::new(), failed: None, closed: false }), changed: Condvar::new() };
    let mut stats = PrefetchStats::default();
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let pos = {
                    let mut st = shared.lock();
                    while !st.closed && st.failed.is_none() && st.next < order.len() && st.next >= st.done + window {
                        st = shared.changed.wait(st).unwrap_or_else(|e| e.into_inner());
                    }
                    if st.closed || st.failed.is_some() || st.next >= order.len() {
                        return;
                    }
                    st.next += 1;
                    st.next - 1
                };
                match catch_unwind(AssertUnwindSafe(|| produce(order[pos]))) {
                    Ok(item) => shared.lock().ready.insert(pos, item),
                    Err(payload) => {
                        shared.lock().failed.get_or_insert(payload);
                        shared.changed.notify_all();
                        return;
                    }
                };
                shared.changed.notify_all();
            });
        }
        let _close = CloseOnDrop(&shared);
        for (pos, &unit) in order.iter().enumerate() {
            let waiting = Instant::now();
            let item = {
                let mut st = shared.lock();
                loop {
                    if let Some(payload) = st.failed.take() {
                        drop(st);
                        resume_unwind(payload);
                    }
                    if let Some(item) = st.ready.remove(&pos) {
                        break item;
                    }
                    st = shared.changed.wait(st).unwrap_or_else(|e| e.into_inner());
                }
            };
            stats.waited += waiting.elapsed();
            consume(unit, &item);
            drop(item);
            shared.lock().done += 1;
            shared.changed.notify_all();
        }
    });
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// An item that counts itself while alive, so a test can read the peak.
    struct Live<'a> {
        id: usize,
        alive: &'a AtomicUsize,
    }

    impl Drop for Live<'_> {
        fn drop(&mut self) {
            self.alive.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn run(order: &[usize], workers: usize, window: usize) -> (Vec<usize>, usize) {
        let alive = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let produce = |i: usize| {
            let now = alive.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            // Uneven production time, so a faster later unit overtakes a slower earlier one.
            std::thread::sleep(Duration::from_micros(if i.is_multiple_of(3) { 900 } else { 100 }));
            Live { id: i, alive: &alive }
        };
        let mut seen = Vec::new();
        for_each_prefetched(order, workers, window, &produce, |unit, item| {
            assert_eq!(unit, item.id);
            seen.push(unit);
        });
        assert_eq!(alive.load(Ordering::SeqCst), 0, "every item is dropped when the pass ends");
        (seen, peak.load(Ordering::SeqCst))
    }

    #[test]
    fn units_are_consumed_in_the_requested_order_whatever_the_worker_count() {
        let order: Vec<usize> = (0..40).rev().collect();
        for workers in [1, 2, 5] {
            let (seen, _) = run(&order, workers, 6);
            assert_eq!(seen, order, "{workers} workers");
        }
    }

    #[test]
    fn no_more_than_window_items_are_ever_alive() {
        let order: Vec<usize> = (0..60).collect();
        for (workers, window) in [(1, 2), (3, 4), (8, 4), (2, 9)] {
            let (_, peak) = run(&order, workers, window);
            assert!(peak <= window, "{workers} workers, window {window}: peak {peak}");
        }
    }

    #[test]
    fn a_producer_panic_reaches_the_caller_and_leaves_no_thread_waiting() {
        let caught = catch_unwind(|| {
            for_each_prefetched(&(0..10).collect::<Vec<_>>(), 3, 4, &|i| if i == 5 { panic!("loader failed on 5") } else { i }, |_, _| {});
        });
        let payload = caught.expect_err("the producer's panic must propagate");
        assert_eq!(payload.downcast_ref::<&str>().copied(), Some("loader failed on 5"));
    }

    #[test]
    fn a_consumer_panic_does_not_hang_the_producers() {
        let caught = catch_unwind(|| {
            for_each_prefetched(&(0..50).collect::<Vec<_>>(), 3, 4, &|i| i, |unit, _| assert!(unit < 3, "stop at 3"));
        });
        assert!(caught.is_err());
    }
}
