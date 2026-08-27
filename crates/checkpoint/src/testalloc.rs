// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A test-scoped counting allocator: peak live heap bytes while armed.
//!
//! Memory bounds in this crate are *the* contract - a reader that promises to
//! stream one tensor at a time is only useful if that is measured rather than
//! read off the source - so the measuring apparatus lives here, once, for
//! every test in the crate that needs it.
//!
//! A `#[global_allocator]` sees EVERY thread in the test binary, and libtest
//! runs tests in parallel by default, so the arming state has to be per-thread
//! rather than one process-wide flag. With a global flag, two things went
//! wrong at once and both produced wrong numbers rather than errors: a second
//! `peak_live` on another test's thread disarmed the first mid-measurement
//! (peaks read back as a few KB instead of megabytes), and unrelated
//! concurrent allocations were charged to whoever happened to be armed. The
//! thread-local `MEASURING` flag below fixes both - only the measuring
//! thread's own allocations are ever counted - and [`peak_live`] additionally
//! serializes on `MEASURE_LOCK` so the shared LIVE/PEAK counters belong to
//! exactly one measurement at a time.
//!
//! `MEASURING` is const-initialized, so reading it allocates nothing (a
//! lazily-initialized thread-local would recurse straight back into `alloc`);
//! `try_with` keeps a late allocation during TLS teardown from panicking.
//!
//! Note what this does and does not see: it counts the **heap**. A memory
//! mapping is not a heap allocation, so a mapped reader's file pages are
//! correctly invisible here - which is exactly why "did this path slurp the
//! file into an owned buffer?" is answerable by this harness at all.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static MEASURE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

fn measuring() -> bool {
    MEASURING.try_with(|m| m.get()).unwrap_or(false)
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() && measuring() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if measuring() {
            // Saturating: frees of pre-arm allocations must not underflow.
            let mut cur = LIVE.load(Ordering::Relaxed);
            loop {
                let next = cur.saturating_sub(layout.size());
                match LIVE.compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(c) => cur = c,
                }
            }
        }
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Run `f` with peak-live-byte tracking armed on THIS thread; returns
/// `(f's value, peak live bytes)`.
pub fn peak_live<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    MEASURING.with(|m| m.set(true));
    let r = f();
    MEASURING.with(|m| m.set(false));
    let peak = PEAK.load(Ordering::Relaxed);
    drop(guard);
    (r, peak)
}
