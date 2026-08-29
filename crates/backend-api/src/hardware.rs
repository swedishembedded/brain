// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fault-tolerant device-contention and
// recovery layers for accelerator-backed systems for its clients. If your
// team needs expertise in cross-process hardware arbitration, un-cancellable
// driver FFI or supervised recovery from a wedged device, you can procure our
// services by sending an email to info@swedishembedded.com.

//! ONE shared answer to "how do I not race another thread **or another
//! process** for a physical device", and ONE shared bound on the driver calls
//! that cannot be cancelled once they have started.
//!
//! Every backend in this workspace opens the same class of resource: an
//! external device reached through a synchronous FFI call into a vendor
//! driver, with no cooperative cancellation anywhere in the path. Before
//! this module each backend invented (or omitted) its own story for the
//! THREAD-safety half of that, and the stories disagreed: `backend-wgpu`
//! had a private [`std::sync::Mutex`], `brain-vulkan` had no lock at all.
//! One incomplete mechanism and one absent mechanism protecting the same
//! silicon is not two problems, it is one.
//!
//! **This is deliberately IN-PROCESS ONLY.** An earlier version of this
//! module also took a host-wide `flock(2)`, on the reasoning that two
//! separate OS processes creating/destroying Vulkan devices on one
//! physical card raced the same driver hazard as two threads. That is
//! true, and it is also the wrong place to fix it: a process-wide lock
//! makes one brain process's device work stall an UNRELATED process
//! targeting a completely different, idle card, with no way for either
//! side to see why - reported directly against a real deployment, where
//! it read as brain simply hanging. Coordinating GPU access ACROSS
//! processes is a scheduling decision belonging to whatever embeds brain
//! (it already knows which processes exist and which cards they want);
//! within one process, this module still owns the invariant, because a
//! single process's own threads are entirely its call to serialise.
//!
//! This module lives in `brain-backend-api` because that is the crate every
//! backend already depends on and the only one upstream of all of them
//! (`brain-gpu-core` is the facade *above* the backends, so it cannot be
//! their shared foundation). It is std-only, like the rest of this crate.
//!
//! Two pieces, each solving a distinct half of the hazard:
//!
//! 1. [`device_init_lock`] / [`device_class_lock`] - mutual exclusion
//!    across this process's OWN threads, keyed by which physically
//!    distinct piece of hardware is being guarded.
//! 2. [`bounded`] / [`try_bounded_for`] - a wall-clock bound on a call that
//!    can wedge inside the driver, so a wedge is a named, reported failure
//!    instead of an unattributed infinite hang. Unrelated to the above:
//!    this bounds a single call regardless of what else, if anything, is
//!    contending for the device.
//! 3. [`own_kernels`] / [`borrow_kernels`] - the borrow-detaching shim a
//!    backend needs to pass its `&[(&str, &str)]` kernel list into
//!    [`bounded`]'s `'static` worker closure.
//!
//! What this module deliberately does **not** claim: on timeout [`bounded`]
//! abandons its worker thread, and an abandoned thread still owns every
//! driver handle it had open. That is a weaker guarantee than a supervisor
//! that can `SIGKILL` a child process and have the kernel reclaim its fds,
//! mappings and driver contexts unconditionally. Killing a thread inside your
//! own address space is not a safe operation, so the stronger form needs a
//! separate OS process, and that stronger form is designed but not
//! implemented here.

#![cfg(not(target_arch = "wasm32"))]

use std::time::Duration;

/// Default ceiling for any single wait on the GPU, in seconds.
const DEFAULT_WAIT_S: f64 = 30.0;

/// The process-wide ceiling for one wait on a device, from `BRAIN_GPU_WAIT_S`.
///
/// The one parse of that variable in the workspace. It used to be transcribed
/// per backend (`backend-wgpu`'s `gpu_wait_timeout`, `backend-vulkan`'s
/// `gpu_wait_timeout_ns`), which is how a new call site ends up with no bound
/// at all: there was nothing shared to reach for.
///
/// Generous by default (a legitimate prefill dispatch on a big model is slow)
/// but always finite. An unbounded wait on a device is what turns a driver
/// fault into an unkillable process instead of a reported failure.
pub fn wait_timeout() -> Duration {
    parse_wait_timeout(std::env::var("BRAIN_GPU_WAIT_S").ok().as_deref())
}

/// [`wait_timeout`]'s parsing rule as a pure function, so the ladder is
/// testable without mutating a process-global environment variable that every
/// other test in the same binary would race.
///
/// Anything that is not a finite, strictly positive number is ignored rather
/// than clamped or rejected: a bad value must not be able to turn the bound
/// off (`0`, `-1`, `inf`, `NaN` all mean "no usable bound").
fn parse_wait_timeout(raw: Option<&str>) -> Duration {
    let secs = raw
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_WAIT_S);
    Duration::from_secs_f64(secs)
}

/// [`wait_timeout`] in nanoseconds, for the Vulkan APIs that take a `u64`
/// nanosecond timeout directly (`vkWaitForFences`).
pub fn wait_timeout_ns() -> u64 {
    wait_timeout().as_nanos().min(u64::MAX as u128) as u64
}

// ---- mutual exclusion over the physical device ------------------------------

/// The GPU class: every logical device opened on a graphics card, whether
/// through `wgpu` or through raw `ash`. One key, because those are the same
/// physical cards and the same ICDs - a `wgpu::Device` and a `VkDevice` being
/// created at the same moment on one card is precisely the race.
pub const GPU: &str = "gpu";

/// The NPU class: the inference-runtime plugin load and device open on a
/// neural accelerator (`crates/npu`). Physically distinct
/// silicon from [`GPU`], so it gets its own key: serialising an NPU compile
/// behind a GPU device creation would be a pure loss, and the point of naming
/// the class is that "one shared answer" does not mean "one global choke
/// point over unrelated hardware".
pub const NPU: &str = "npu";

/// The in-process half of `class`'s lock. Leaked on first use; the number of
/// classes is fixed and tiny, so this is a handful of words for the life of
/// the process and lets the guard borrow a `'static` mutex.
fn class_mutex(class: &'static str) -> &'static std::sync::Mutex<()> {
    type Table = std::sync::Mutex<std::collections::HashMap<&'static str, &'static std::sync::Mutex<()>>>;
    static TABLE: std::sync::OnceLock<Table> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut table = table.lock().unwrap_or_else(|e| e.into_inner());
    table.entry(class).or_insert_with(|| &*Box::leak(Box::new(std::sync::Mutex::new(()))))
}

// `HELD` records which classes THIS thread already holds a lock for.
thread_local! {
    static HELD: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Exclusive access, within THIS PROCESS, to device creation/teardown for
/// [`GPU`], held until the returned guard drops.
///
/// A [`std::sync::Mutex`]: building a backend enumerates every installed ICD
/// through the Vulkan/EGL loaders, which are not re-entrant, and destroying a
/// device races the driver's own background worker threads. Both are
/// properties of THIS process's own threads: two different processes on the
/// host are not this module's concern (see this module's own doc for why).
///
/// Re-entrant on one thread: a nested call returns a guard that owns nothing
/// and releases nothing, so the outermost acquisition defines the critical
/// section. That is what lets a backend crate and the context crate beneath
/// it both name the invariant without deadlocking each other.
///
/// **Not** re-entrant across threads, by construction: hand the guard's
/// critical section to one thread, or acquire it inside the worker. A guard
/// taken on thread A while thread B tries to acquire is precisely the
/// exclusion this exists to provide.
pub fn device_init_lock() -> DeviceInitGuard {
    device_class_lock(GPU)
}

/// [`device_init_lock`] for a named device class ([`GPU`], [`NPU`]).
///
/// Classes are independent: holding one says nothing about the others, and a
/// thread may hold several at once. Use a new class only for hardware that is
/// physically separate from every existing one - two names for one piece of
/// silicon is the bug this module exists to remove.
pub fn device_class_lock(class: &'static str) -> DeviceInitGuard {
    if HELD.with(|h| h.borrow().contains(&class)) {
        return DeviceInitGuard { class, held: None };
    }
    // A poisoned lock only means some other thread panicked while building a
    // device; the data is `()`, so recovering is always correct.
    let mutex = class_mutex(class).lock().unwrap_or_else(|e| e.into_inner());
    HELD.with(|h| h.borrow_mut().push(class));
    DeviceInitGuard { class, held: Some(mutex) }
}

/// The RAII handle returned by [`device_class_lock`]. `None` inside means
/// this was a re-entrant acquisition on a thread that already holds the
/// class, so it owns and releases nothing.
pub struct DeviceInitGuard {
    class: &'static str,
    held: Option<std::sync::MutexGuard<'static, ()>>,
}

impl Drop for DeviceInitGuard {
    fn drop(&mut self) {
        if self.held.is_some() {
            // Cleared before `held` is dropped, which is sound because only
            // this thread ever reads the record and it cannot re-acquire
            // between here and the end of this function.
            let class = self.class;
            HELD.with(|h| h.borrow_mut().retain(|c| *c != class));
        }
    }
}

// ---- bounding a call that cannot be cancelled -------------------------------

/// A bounded call that did not finish: the driver is presumed wedged.
#[derive(Debug, Clone)]
pub struct Wedged {
    /// What was being attempted, for the operator reading the failure.
    pub what: String,
    /// The bound that was exceeded.
    pub timeout: Duration,
}

impl std::fmt::Display for Wedged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} exceeded {:?} (BRAIN_GPU_WAIT_S) -- driver likely wedged; see `backend_api::hardware::bounded`",
            self.what, self.timeout
        )
    }
}

impl std::error::Error for Wedged {}

/// Run `f` with a wall-clock bound, reporting a [`Wedged`] rather than
/// blocking the caller forever.
///
/// Device and adapter creation are, at bottom, synchronous FFI calls into a
/// vendor loader and ultimately an `ioctl` into a proprietary kernel module.
/// Nothing in that path is cancellable, so the only way to survive a wedge is
/// to not be the thread that is stuck in it: `f` runs on its own thread and
/// the caller waits on a channel with a deadline.
///
/// This is not a retry, a backoff or a spin. It converts one failure mode
/// (unattributed infinite hang, indistinguishable from slow work) into
/// another (named, timed, reported failure) - the same trade [`wait_timeout`]
/// already makes for dispatch/submit/teardown waits, extended to the
/// operations that had no bound at all.
///
/// On timeout the worker is deliberately abandoned rather than joined:
/// joining would defeat the bound, and there is nothing to reclaim from a
/// thread parked inside a driver call. It leaks until the driver unwedges or
/// the process exits. That is the known ceiling of this mechanism: a thread
/// cannot be killed safely, an OS process can, so genuinely reclaiming a
/// wedged device needs a supervised child process rather than a worker
/// thread. That successor is designed but deliberately not built here.
pub fn try_bounded_for<T: Send + 'static>(
    what: &str,
    timeout: Duration,
    f: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Wedged> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        // A closed receiver (the caller already timed out) is expected, not
        // an error: nobody is left to hear the answer.
        let _ = tx.send(f());
    });
    rx.recv_timeout(timeout).map_err(|_| Wedged { what: what.to_string(), timeout })
}

/// [`try_bounded_for`] with an explicit timeout, panicking on a wedge.
///
/// Split from [`bounded`] so the timeout behaviour itself is testable without
/// touching the process-global `BRAIN_GPU_WAIT_S`, which every test in the
/// same binary would otherwise race.
pub fn bounded_for<T: Send + 'static>(what: &str, timeout: Duration, f: impl FnOnce() -> T + Send + 'static) -> T {
    match try_bounded_for(what, timeout, f) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    }
}

/// [`bounded_for`] at the process-wide [`wait_timeout`] - the shape a
/// construction path calls.
pub fn bounded<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    bounded_for(what, wait_timeout(), f)
}

/// [`try_bounded_for`] at the process-wide [`wait_timeout`] - the shape a
/// fallible construction path (one whose caller falls back to another
/// backend) calls, so a wedged driver degrades instead of aborting.
pub fn try_bounded<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> Result<T, Wedged> {
    try_bounded_for(what, wait_timeout(), f)
}

// ---- passing a kernel list into a 'static worker ----------------------------

/// Detach a kernel list from its caller's borrow so it can cross into
/// [`bounded`]'s spawned thread, which needs a `'static` closure.
pub fn own_kernels(kernels: &[(&str, &str)]) -> Vec<(String, String)> {
    kernels.iter().map(|(name, src)| ((*name).to_string(), (*src).to_string())).collect()
}

/// The borrowed shape every backend constructor actually takes, rebuilt from
/// [`own_kernels`]'s output inside the worker thread's own stack frame. The
/// result never crosses back over the thread boundary, so the borrow never
/// needs to outlive it.
pub fn borrow_kernels(owned: &[(String, String)]) -> Vec<(&str, &str)> {
    owned.iter().map(|(name, src)| (name.as_str(), src.as_str())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// A call that never returns must fail with a NAMED, bounded error - not
    /// hang the calling thread forever. This is the whole point of
    /// [`bounded`]: device/adapter creation is a synchronous FFI call into a
    /// proprietary driver with no cooperative cancellation, so a wedge there
    /// can only ever be handled by NOT waiting on it directly.
    #[test]
    #[should_panic(expected = "exceeded")]
    fn bounded_for_reports_a_named_timeout_instead_of_hanging_forever() {
        bounded_for("test op", Duration::from_millis(50), || {
            std::thread::sleep(Duration::from_secs(3600));
        });
    }

    /// A call that finishes well inside the bound returns normally - the
    /// timeout must not fire on the ordinary, fast path every real device
    /// creation takes.
    #[test]
    fn bounded_for_returns_the_value_when_the_call_finishes_in_time() {
        let got = bounded_for("test op", Duration::from_secs(5), || 42);
        assert_eq!(got, 42);
    }

    /// The fallible shape reports the wedge as a value, so a caller with a
    /// fallback (the native Vulkan backend, whose caller falls back to wgpu)
    /// can degrade instead of aborting the process.
    #[test]
    fn try_bounded_for_reports_a_wedge_as_an_error_not_a_panic() {
        let got = try_bounded_for("test op", Duration::from_millis(50), || {
            std::thread::sleep(Duration::from_secs(3600));
        });
        let err = got.expect_err("a call that never returns must not report success");
        assert_eq!(err.what, "test op");
        assert!(err.to_string().contains("BRAIN_GPU_WAIT_S"), "the failure must name the knob that set the bound: {err}");
    }

    /// [`device_init_lock`] must actually exclude a second holder on another
    /// THREAD while the first is live - the in-process property this module
    /// still owns (see this module's own doc for why cross-PROCESS exclusion
    /// is deliberately not this module's job).
    #[test]
    fn device_init_lock_excludes_a_concurrent_holder() {
        let released = Arc::new(AtomicBool::new(false));
        let released_writer = released.clone();
        let holder = std::thread::spawn(move || {
            let _g = device_init_lock();
            std::thread::sleep(Duration::from_millis(200));
            released_writer.store(true, Ordering::SeqCst);
            // `_g` drops here, at thread exit.
        });
        // Give the holder thread a head start so it wins the lock first.
        std::thread::sleep(Duration::from_millis(50));
        let _g = device_init_lock();
        assert!(
            released.load(Ordering::SeqCst),
            "acquired the lock before the first holder released it - device_init_lock is not exclusive"
        );
        holder.join().unwrap();
    }

    /// Nesting the lock on ONE thread must not deadlock. Without the
    /// re-entrancy flag this test hangs forever: the inner acquisition tries
    /// to lock the same `std::sync::Mutex` the outer one already holds. This
    /// is the property that makes the lock safe to name at more than one
    /// layer (a backend crate bounding construction, the context crate
    /// beneath it guarding device creation), which is exactly what a single
    /// shared primitive has to support.
    #[test]
    fn device_init_lock_is_reentrant_within_one_thread() {
        let outer = device_init_lock();
        let inner = device_init_lock();
        drop(inner);
        // Still exclusive after the inner guard goes away: the OUTERMOST
        // acquisition owns the critical section, so a nested release must not
        // open the gate early.
        let taken = Arc::new(AtomicBool::new(false));
        let taken_writer = taken.clone();
        let other = std::thread::spawn(move || {
            let _g = device_init_lock();
            taken_writer.store(true, Ordering::SeqCst);
        });
        std::thread::sleep(Duration::from_millis(100));
        assert!(!taken.load(Ordering::SeqCst), "an inner guard's release freed the lock the outer guard still holds");
        drop(outer);
        other.join().unwrap();
        assert!(taken.load(Ordering::SeqCst));
    }

    /// Two classes name two physically separate devices, so holding one must
    /// not block the other. A single global choke point would make every NPU
    /// compile queue behind an unrelated GPU device creation - "one shared
    /// answer" is about there being one mechanism, not one lock.
    #[test]
    fn separate_device_classes_do_not_block_each_other() {
        let _gpu = device_class_lock(GPU);
        let taken = Arc::new(AtomicBool::new(false));
        let taken_writer = taken.clone();
        let other = std::thread::spawn(move || {
            let _npu = device_class_lock(NPU);
            taken_writer.store(true, Ordering::SeqCst);
        });
        other.join().unwrap();
        assert!(taken.load(Ordering::SeqCst), "the NPU class blocked on a GPU-class holder");
    }

    /// A bad `BRAIN_GPU_WAIT_S` must fall back to the default bound, never
    /// disable it. `0`/negative/non-finite all mean "no usable bound", and a
    /// silently unbounded wait is the failure mode this whole module exists
    /// to remove.
    #[test]
    fn an_unusable_wait_setting_falls_back_to_the_default_bound() {
        let default = Duration::from_secs_f64(DEFAULT_WAIT_S);
        for bad in [None, Some(""), Some("0"), Some("-1"), Some("abc"), Some("inf"), Some("NaN")] {
            assert_eq!(parse_wait_timeout(bad), default, "{bad:?} must not disable the bound");
        }
        assert_eq!(parse_wait_timeout(Some("1200")), Duration::from_secs(1200));
        assert_eq!(parse_wait_timeout(Some("0.5")), Duration::from_millis(500));
    }

    /// The kernel-list shim must survive the round trip into a worker thread
    /// unchanged - a backend that silently lost or reordered kernels here
    /// would fail far away from the cause.
    #[test]
    fn a_kernel_list_round_trips_through_the_worker_boundary() {
        let kernels = [("add", "src-a"), ("mul", "src-b")];
        let owned = own_kernels(&kernels);
        let got = bounded_for("round trip", Duration::from_secs(5), move || {
            borrow_kernels(&owned).iter().map(|(n, s)| format!("{n}:{s}")).collect::<Vec<_>>()
        });
        assert_eq!(got, vec!["add:src-a".to_string(), "mul:src-b".to_string()]);
    }
}
