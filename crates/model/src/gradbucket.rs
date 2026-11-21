// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements data-parallel training runtimes that keep
// gradient synchronisation cheap at any number of tensors for its clients. If
// your team needs expertise in gradient bucketing and collective scheduling,
// you can procure our services by sending an email to info@swedishembedded.com.

//! Gradient buckets: how a step's many small gradient tensors are packed into
//! a few flat buffers so a reduction moves a few large messages instead of
//! hundreds of small ones.
//!
//! [`plan`] is a pure function of the tensors' element counts and the bucket
//! cap, so every rank of a job, given the same model, computes the same layout:
//! the order in which ranks reduce buckets must agree, and it does because it
//! is never derived from anything rank-local. Tensors are taken in reverse
//! order (the order a backward pass produces them: last layer first), packed
//! greedily up to the cap; a tensor at or above the cap is reduced where it
//! lives, without a copy, since packing it would only add traffic.
//!
//! The layout is shared by every reduction shape: an all-reduce of each bucket
//! today, a reduce-scatter of each bucket for sharded optimizer state later.

/// One unit a reduction moves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Slot {
    /// The tensor at this index, reduced in place (it is at least the cap).
    InPlace(usize),
    /// Several tensors packed into one flat buffer of `elems` elements: each
    /// `(index, offset, count)` member occupies `[offset, offset + count)`.
    Packed { elems: u64, members: Vec<(usize, u64, u64)> },
}

/// The buckets for tensors of `counts` elements under a cap of `cap` elements
/// per packed bucket (`0`: no packing, every tensor in place), each packed
/// member starting at a multiple of `align` elements (bf16 members on an even
/// one, so a copy moves whole 32-bit words).
pub fn plan(counts: &[u64], cap: u64, align: u64) -> Vec<Slot> {
    let align = align.max(1);
    let mut slots = Vec::new();
    let mut open: Vec<(usize, u64, u64)> = Vec::new();
    let mut filled = 0u64;
    let close = |open: &mut Vec<(usize, u64, u64)>, filled: &mut u64, slots: &mut Vec<Slot>| {
        if !open.is_empty() {
            slots.push(Slot::Packed { elems: *filled, members: std::mem::take(open) });
            *filled = 0;
        }
    };
    for (index, &count) in counts.iter().enumerate().rev() {
        if count == 0 {
            continue;
        }
        if cap == 0 || count >= cap {
            // The open bucket is complete before this tensor in backward order, so it goes first.
            close(&mut open, &mut filled, &mut slots);
            slots.push(Slot::InPlace(index));
            continue;
        }
        let start = filled.div_ceil(align) * align;
        if start + count > cap {
            close(&mut open, &mut filled, &mut slots);
        }
        let start = filled.div_ceil(align) * align;
        open.push((index, start, count));
        filled = start + count;
    }
    close(&mut open, &mut filled, &mut slots);
    slots
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every non-empty tensor is in exactly one slot, packed buckets respect
    /// the cap and are contiguous, and tensors come in reverse order.
    #[test]
    fn a_plan_covers_every_tensor_once_in_backward_order_within_the_cap() {
        let mut lcg = data::rng::Lcg::new(11);
        let counts: Vec<u64> = (0..300).map(|i| if i % 37 == 0 { 0 } else { 1 + (lcg.unit() * 5000.0) as u64 + if i % 50 == 1 { 20_000 } else { 0 } }).collect();
        for cap in [0, 1, 4096, 16_384, 1 << 20] {
            let slots = plan(&counts, cap, 1);
            let mut seen = Vec::new();
            for slot in &slots {
                match slot {
                    Slot::InPlace(i) => {
                        assert!(cap == 0 || counts[*i] >= cap, "only tensors at the cap stay in place");
                        seen.push(*i);
                    }
                    Slot::Packed { elems, members } => {
                        assert!(*elems <= cap, "bucket of {elems} over the cap {cap}");
                        let mut at = 0;
                        for &(i, offset, count) in members {
                            assert_eq!((offset, count), (at, counts[i]), "members are contiguous");
                            at += count;
                            seen.push(i);
                        }
                        assert_eq!(at, *elems);
                    }
                }
            }
            let expected: Vec<usize> = (0..counts.len()).rev().filter(|&i| counts[i] > 0).collect();
            assert_eq!(seen, expected, "cap {cap}: every tensor once, last first");
            assert_eq!(plan(&counts, cap, 1), slots, "the layout is a function of the counts and the cap alone");
        }
    }

    #[test]
    fn small_tensors_share_a_few_buckets() {
        let counts = vec![1000u64; 504];
        let slots = plan(&counts, 64_000, 1);
        assert_eq!(slots.len(), 8, "504 tensors of 1000 into buckets of 64 tensors");
    }

    /// With an alignment, members start on its multiples and still fit the cap.
    #[test]
    fn aligned_members_start_on_whole_words() {
        let counts = vec![3u64, 5, 1, 7, 2];
        for slot in plan(&counts, 12, 2) {
            if let Slot::Packed { elems, members } = slot {
                assert!(elems <= 12);
                for (_, offset, _) in members {
                    assert_eq!(offset % 2, 0, "member at {offset}");
                }
            }
        }
    }
}
