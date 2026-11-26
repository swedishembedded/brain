// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a caught panic was, in words, whatever its payload.
//!
//! Swedish Embedded AB implements long-running inference servers that report
//! their own faults precisely for its clients. If your team needs expertise in
//! diagnosing crashes in GPU serving processes, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! `panic!` throws a `&str` or a `String`, but a step can also unwind with a
//! value: a full card's [`DeviceOutOfMemory`] (`resume_unwind`), a payload
//! caught on one thread and thrown again on another (boxed once more), or
//! anything `panic_any` was given. A catch site that only reads strings turns
//! each of those into "non-string payload" and the cause is gone.
//! [`describe`] names every payload brain throws and says what an unknown one
//! is; [`record_locations`] keeps where the last panic of each thread was
//! raised, because a `resume_unwind` payload carries no location and the
//! panic hook's own line may be far up the log.

use crate::DeviceOutOfMemory;
use std::any::Any;
use std::cell::RefCell;

/// The payload in words: the message of a `panic!`, the report of a known
/// payload type, the inner payload of one thrown again boxed, and otherwise
/// that it is a value of a type brain does not name (which `panic_any` or
/// `resume_unwind` threw; the hook's `panicked at` line, when one ran, says
/// where).
pub fn describe(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    if let Some(oom) = payload.downcast_ref::<DeviceOutOfMemory>() {
        return oom.to_string();
    }
    if let Some(inner) = payload.downcast_ref::<Box<dyn Any + Send>>() {
        return describe(inner.as_ref());
    }
    "a panic payload of a type brain does not name (thrown with panic_any or resume_unwind)".to_string()
}

thread_local! {
    static LAST_LOCATION: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Chain a panic hook that remembers, per thread, where its last panic was
/// raised (`file:line:column`), for [`take_location`]. The hook installed
/// before keeps running first, so `RUST_BACKTRACE` and the usual message are
/// unchanged. Installing twice is harmless: the second call does nothing.
pub fn record_locations() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            previous(info);
            if let Some(at) = info.location() {
                LAST_LOCATION.with(|slot| *slot.borrow_mut() = Some(at.to_string()));
            }
        }));
    });
}

/// Where this thread's last panic was raised, once: `None` without
/// [`record_locations`], or when the unwind bypassed the hook
/// (`resume_unwind`, which [`DeviceOutOfMemory::raise`] uses).
pub fn take_location() -> Option<String> {
    LAST_LOCATION.with(|slot| slot.borrow_mut().take())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_payload_brain_throws_is_named_and_an_unknown_one_says_so() {
        let oom = DeviceOutOfMemory { bytes: 64, what: "pool".into(), detail: "out of memory".into() };
        assert_eq!(describe(&"plain"), "plain");
        assert_eq!(describe(&String::from("owned")), "owned");
        assert_eq!(describe(&oom), oom.to_string());
        let rethrown: Box<dyn Any + Send> = Box::new(String::from("from a worker thread"));
        assert_eq!(describe(&rethrown), "from a worker thread");
        assert!(describe(&3.5f64).contains("type brain does not name"));
    }

    #[test]
    fn a_recorded_panic_leaves_its_location_once() {
        record_locations();
        let caught = std::panic::catch_unwind(|| std::panic::panic_any(11u8)).expect_err("it panics");
        assert!(describe(caught.as_ref()).contains("type brain does not name"));
        let at = take_location().expect("the hook recorded where");
        assert!(at.contains("panic_payload.rs"), "{at}");
        assert_eq!(take_location(), None, "taken once");
    }
}
