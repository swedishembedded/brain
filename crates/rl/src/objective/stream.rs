// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training runs whose data order
// is a property of the run, not of how often it was probed or restarted, for
// its clients. If your team needs expertise in reproducible fine-tuning then
// you can procure our services by sending an email to info@swedishembedded.com.

//! A record stream an objective walks on its own, addressed by index so that a resumed run can
//! continue it exactly: record `n` is a function of `n` alone.
//!
//! A source that read the training generator or kept private mutable state could not be put back
//! to a position, so the closure a caller supplies is given the index of the record it is asked
//! for and nothing else. After a resume the same index is asked for again.

/// Draws records by index and counts how many were asked for.
pub(crate) struct IndexedStream<T> {
    next: Box<dyn FnMut(u64) -> Option<T>>,
    /// Records asked of `next`, `None` answers included.
    asked: u64,
}

impl<T: 'static> IndexedStream<T> {
    pub(crate) fn new(next: impl FnMut(u64) -> Option<T> + 'static) -> Self {
        IndexedStream { next: Box::new(next), asked: 0 }
    }
}

/// The source of a fixed dataset walked in order, wrapping around.
pub(crate) fn cycle<T: Clone + 'static>(items: Vec<T>) -> impl FnMut(u64) -> Option<T> {
    assert!(!items.is_empty(), "a cycling stream needs at least one record");
    move |n| Some(items[(n % items.len() as u64) as usize].clone())
}

impl<T> IndexedStream<T> {
    /// The next record, `None` when the source has none this round.
    pub(crate) fn draw(&mut self) -> Option<T> {
        let record = (self.next)(self.asked);
        self.asked += 1;
        record
    }

    /// The position to save while `queued` drawn records are still waiting to be trained on: a
    /// resumed run asks for them again.
    pub(crate) fn position(&self, queued: usize) -> serde_json::Value {
        serde_json::json!({ "records": self.asked - queued as u64 })
    }

    /// Continue from a saved [`Self::position`]; the records queued before are dropped by the caller.
    pub(crate) fn seek(&mut self, saved: &serde_json::Value) -> Result<(), std::io::Error> {
        let records = saved.get("records").and_then(|r| r.as_u64()).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("not a stream position: {saved}")))?;
        self.asked = records;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cycling_stream_wraps_and_a_saved_position_is_the_trained_prefix() {
        let mut s = IndexedStream::new(cycle(vec![10, 11, 12]));
        assert_eq!((s.draw(), s.draw(), s.draw(), s.draw()), (Some(10), Some(11), Some(12), Some(10)));
        let at = s.position(1);
        let mut resumed = IndexedStream::new(cycle(vec![10, 11, 12]));
        resumed.seek(&at).unwrap();
        assert_eq!(resumed.draw(), Some(10), "the queued record is asked for again");
        assert!(resumed.seek(&serde_json::json!({"records": "x"})).is_err());
    }
}
