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
//!
//! A data-parallel rank walks its own shard of the stream ([`IndexedStream::shard`]): rank `r` of `world` is asked
//! for record `n` exactly when `n % world == r`, so the ranks' shards are disjoint and their interleaving is the
//! stream one rank walks, which is what `model::Objective::shard` requires.

/// Draws records by index and counts how many were asked for.
pub(crate) struct IndexedStream<T> {
    next: Box<dyn FnMut(u64) -> Option<T>>,
    /// Records this rank asked of `next`, `None` answers included.
    asked: u64,
    /// `(rank, world)`: this rank's shard of the stream; `(0, 1)`, the whole stream.
    shard: (u64, u64),
}

impl<T: 'static> IndexedStream<T> {
    pub(crate) fn new(next: impl FnMut(u64) -> Option<T> + 'static) -> Self {
        IndexedStream { next: Box::new(next), asked: 0, shard: (0, 1) }
    }
}

/// The source of a fixed dataset walked in order, wrapping around.
pub(crate) fn cycle<T: Clone + 'static>(items: Vec<T>) -> impl FnMut(u64) -> Option<T> {
    assert!(!items.is_empty(), "a cycling stream needs at least one record");
    move |n| Some(items[(n % items.len() as u64) as usize].clone())
}

impl<T> IndexedStream<T> {
    /// Walk only rank `rank`'s shard of a stream split over `world` ranks (see the module doc). Set before the first
    /// draw: a stream already walked whole cannot change its shard.
    pub(crate) fn shard(&mut self, rank: usize, world: usize) -> Result<(), std::io::Error> {
        if world == 0 || rank >= world {
            return Err(std::io::Error::other(format!("rank {rank} is not one of a world of {world}")));
        }
        if self.asked > 0 {
            return Err(std::io::Error::other("a stream already drawn from cannot be sharded"));
        }
        self.shard = (rank as u64, world as u64);
        Ok(())
    }

    /// The next record of this rank's shard, `None` when the source has none this round.
    pub(crate) fn draw(&mut self) -> Option<T> {
        let (rank, world) = self.shard;
        let record = (self.next)(self.asked * world + rank);
        self.asked += 1;
        record
    }

    /// The position to save while `queued` drawn records are still waiting to be trained on: a
    /// resumed run asks for them again. Counted in records of the whole stream, which every rank of a
    /// group has walked as far.
    pub(crate) fn position(&self, queued: usize) -> serde_json::Value {
        serde_json::json!({ "records": (self.asked - queued as u64) * self.shard.1 })
    }

    /// Continue from a saved [`Self::position`]; the records queued before are dropped by the caller.
    pub(crate) fn seek(&mut self, saved: &serde_json::Value) -> Result<(), std::io::Error> {
        let invalid = |why: String| std::io::Error::new(std::io::ErrorKind::InvalidData, why);
        let records = saved.get("records").and_then(|r| r.as_u64()).ok_or_else(|| invalid(format!("not a stream position: {saved}")))?;
        let world = self.shard.1;
        if !records.is_multiple_of(world) {
            return Err(invalid(format!("stream position {records} is not a step of a group of {world} ranks")));
        }
        self.asked = records / world;
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

    /// Two ranks walk disjoint shards whose interleaving is the whole stream, and a rank's saved position is the
    /// whole stream's.
    #[test]
    fn sharded_streams_are_disjoint_and_interleave_into_the_stream() {
        let shard = |rank| {
            let mut s = IndexedStream::new(Some);
            s.shard(rank, 2).unwrap();
            s
        };
        let (mut a, mut b) = (shard(0), shard(1));
        let walked: Vec<u64> = (0..3).flat_map(|_| [a.draw().unwrap(), b.draw().unwrap()]).collect();
        assert_eq!(walked, [0, 1, 2, 3, 4, 5]);
        assert_eq!(b.position(1), serde_json::json!({"records": 4}), "the queued record is asked for again");
        let mut resumed = shard(1);
        resumed.seek(&b.position(1)).unwrap();
        assert_eq!(resumed.draw(), Some(5));
        assert!(shard(0).seek(&serde_json::json!({"records": 3})).is_err(), "no rank of two stops mid-step");
        assert!(b.shard(0, 2).is_err(), "a stream walked whole keeps its shard");
        assert!(IndexedStream::new(Some).shard(2, 2).is_err());
    }
}
