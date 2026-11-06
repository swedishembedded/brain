// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for inference servers that accept
// media from the internet without letting it stall them for its clients. If
// your team needs expertise in bounding the cost of untrusted input then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Where a request's media is decoded: off the async runtime, a few at a time.
//!
//! Decoding an image, a recording or a clip costs CPU seconds and up to a
//! gigabyte of memory before the executor has admitted anything. On the async
//! workers, a handful of such requests would stall every surface of the
//! process, keyed or not; unbounded on the blocking pool, a burst would hold
//! that many gigabytes at once. So every decode runs on the blocking pool
//! under one process-wide bound, [`MAX_CONCURRENT_DECODES`]; a request that
//! cannot get a slot within [`DECODE_QUEUE_DEADLINE`] is shed as saturated
//! (503) rather than queued without end.

use std::time::Duration;

use tokio::sync::Semaphore;

use crate::error::ApiError;
use crate::surface::Provider;

/// How many request decodes run at once, across every surface of the process.
/// Each may hold up to the request pixel budget as float (768 MB), so this
/// bounds decode memory to a few gigabytes whatever arrives.
pub const MAX_CONCURRENT_DECODES: usize = 4;

/// How long a request waits for a decode slot before it is shed.
pub const DECODE_QUEUE_DEADLINE: Duration = Duration::from_secs(30);

static DECODERS: Semaphore = Semaphore::const_new(MAX_CONCURRENT_DECODES);

/// Run `decode` on the blocking pool once a decode slot is free, holding the
/// slot until it returns.
///
/// # Errors
///
/// No slot within [`DECODE_QUEUE_DEADLINE`] (503, saturated), or the decode
/// panicked (500, with no detail of it).
pub async fn off_runtime<T: Send + 'static>(provider: Provider, decode: impl FnOnce() -> T + Send + 'static) -> Result<T, ApiError> {
    let slot = tokio::time::timeout(DECODE_QUEUE_DEADLINE, DECODERS.acquire())
        .await
        .map_err(|_| ApiError::saturated(provider, "every media decoder is busy; retry shortly"))?
        .expect("the decoder semaphore is never closed");
    let result = tokio::task::spawn_blocking(decode).await;
    drop(slot);
    result.map_err(|_| ApiError::internal(provider, "the request's media could not be decoded"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_more_than_the_bound_decode_at_once() {
        let (running, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let tasks: Vec<_> = (0..3 * MAX_CONCURRENT_DECODES)
            .map(|_| {
                let (running, peak) = (running.clone(), peak.clone());
                tokio::spawn(off_runtime(Provider::OpenAI, move || {
                    let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(30));
                    running.fetch_sub(1, Ordering::SeqCst);
                }))
            })
            .collect();
        for task in tasks {
            task.await.unwrap().unwrap();
        }
        assert_eq!(peak.load(Ordering::SeqCst), MAX_CONCURRENT_DECODES, "the bound is reached and never passed");
    }

    #[tokio::test]
    async fn a_decode_that_panics_is_an_error_and_frees_its_slot() {
        for _ in 0..=MAX_CONCURRENT_DECODES {
            let failed = off_runtime(Provider::OpenAI, || -> () { panic!("a decoder bug") }).await;
            assert!(failed.is_err());
        }
        // More panics than slots, and a decode still gets one.
        assert_eq!(off_runtime(Provider::OpenAI, || 7).await.unwrap(), 7);
    }
}
