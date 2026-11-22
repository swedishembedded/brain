// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models on packed rows with exact document
// boundaries for its clients. If your team needs expertise in training data pipelines then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Packed training rows (`data::packing`, `TokenDataset::packed_batch`): several whole examples per row.
//!
//! - packing never splits an example, never drops or reorders one, and fills rows greedily;
//! - each example's tokens and targets in a packed row are exactly those of the same example laid out alone in a row
//!   (`example_batch`), so its last token, whose next token belongs to another document, is never supervised;
//! - segment ids are constant inside a document, change at every document start (padding included) and never
//!   decrease, and the document positions derived from them restart at 0 for every document;
//! - malformed requests are refused by name.

use data::loader::{BatchConfig, TokenDataset, IGNORE};
use data::packing::{document_positions, PackingError, RowPacker};
use data::rng::Rng;

const ROW: usize = 16;
const ROWS: usize = 3;

fn cfg() -> BatchConfig {
    BatchConfig { batch_size: ROWS, block_size: ROW, ..BatchConfig::default() }
}

/// A dataset of `n` examples of random lengths `1..=ROW`, random tokens and a random supervision mask.
fn dataset(rng: &mut Rng, n: usize) -> (TokenDataset, Vec<usize>) {
    let mut starts = Vec::new();
    let mut data = Vec::new();
    for _ in 0..n {
        starts.push(data.len());
        let len = rng.gen_range_inclusive(1, ROW as i64) as usize;
        data.extend((0..len).map(|_| rng.gen_range_inclusive(1, 50) as u32));
    }
    let mask: Vec<bool> = (0..data.len()).map(|_| rng.gen_range_inclusive(0, 3) > 0).collect();
    (TokenDataset::new_examples(data, mask, &starts, 0, &cfg()).expect("every example fits a row"), starts)
}

#[test]
fn packing_lays_out_every_example_whole_with_its_own_targets_and_segments() {
    let mut rng = Rng::random();
    let n = 40;
    let (ds, _) = dataset(&mut rng, n);
    let mut packer = RowPacker::new();
    let mut order = 0..n;
    let mut seen = Vec::new();
    loop {
        let rows = packer.fill(ROWS, ROW, |e| ds.example_len(e).expect("an example"), || order.next());
        if rows.iter().all(Vec::is_empty) {
            break;
        }
        // Greedy: a row is closed only by an example that does not fit it (or by the end of the stream).
        for (r, row) in rows.iter().enumerate() {
            let used: usize = row.iter().map(|&e| ds.example_len(e).unwrap()).sum();
            let next = rows.get(r + 1).and_then(|next| next.first()).copied().or(packer.pending());
            if let Some(next) = next {
                assert!(used + ds.example_len(next).unwrap() > ROW, "row {r} closed with room for example {next}");
            }
        }
        let batch = ds.packed_batch(&cfg(), &rows).expect("a valid packing");
        let positions = document_positions(&batch.segments, ROW).expect("valid segments");
        let mut useful = 0u64;
        for (r, row) in rows.iter().enumerate() {
            let at = |t: usize| r * ROW + t;
            let mut t = 0usize;
            for &e in row {
                let len = ds.example_len(e).unwrap();
                // The example alone, one per row.
                let mut picks = vec![None; ROWS];
                picks[0] = Some(e);
                let (alone_x, alone_y) = ds.example_batch(&cfg(), &picks).unwrap();
                assert_eq!(&batch.tokens[at(t)..at(t + len)], &alone_x[..len], "example {e}: tokens");
                assert_eq!(&batch.targets[at(t)..at(t + len)], &alone_y[..len], "example {e}: targets");
                assert_eq!(batch.targets[at(t + len - 1)], IGNORE, "example {e}: its last token is not supervised");
                let id = batch.segments[at(t)];
                assert!(batch.segments[at(t)..at(t + len)].iter().all(|&s| s == id), "example {e}: one segment id");
                if t > 0 {
                    assert!(batch.segments[at(t - 1)] < id, "example {e}: a new id at its start");
                }
                assert_eq!(&positions[at(t)..at(t + len)], &(0..len as u32).collect::<Vec<_>>()[..], "example {e}: positions");
                seen.push(e);
                t += len;
                useful += len as u64;
            }
            // Padding: no target, a document of its own.
            assert!(batch.targets[at(t)..at(ROW)].iter().all(|&y| y == IGNORE), "row {r}: padding is not supervised");
            if t > 0 && t < ROW {
                assert!(batch.segments[at(t)] > batch.segments[at(t - 1)], "row {r}: padding has its own segment");
            }
        }
        assert_eq!(batch.useful_tokens, useful);
    }
    assert_eq!(seen, (0..n).collect::<Vec<_>>(), "every example once, in order");
}

#[test]
fn a_resumed_packer_lays_out_the_rows_an_uninterrupted_one_does() {
    let mut rng = Rng::random();
    let (ds, _) = dataset(&mut rng, 30);
    let len = |e: usize| ds.example_len(e).unwrap();
    let mut straight = RowPacker::new();
    let mut order = 0..30;
    let _first = straight.fill(ROWS, ROW, len, || order.next());
    // What a run saves after its first batch: how far the order went, and the example the packer holds.
    let (drawn, held) = (30 - order.len(), straight.pending());
    let second = straight.fill(ROWS, ROW, len, || order.next());
    let mut resumed = RowPacker::resuming(held);
    let mut order = drawn..30;
    assert_eq!(resumed.fill(ROWS, ROW, len, || order.next()), second);
}

#[test]
fn malformed_packings_are_refused_by_name() {
    let mut rng = Rng::random();
    let (ds, _) = dataset(&mut rng, 5);
    let two: Vec<Vec<usize>> = vec![vec![0], vec![1]];
    assert_eq!(ds.packed_batch(&cfg(), &two), Err(PackingError::RowCount { rows: 2, batch_size: ROWS }));
    assert_eq!(ds.packed_batch(&cfg(), &[vec![0], vec![9], vec![]]), Err(PackingError::UnknownExample { row: 1, example: 9, examples: 5 }));
    let full: Vec<usize> = (0..5).collect();
    let tokens: usize = full.iter().map(|&e| ds.example_len(e).unwrap()).sum();
    if tokens > ROW {
        assert_eq!(ds.packed_batch(&cfg(), &[vec![], full, vec![]]), Err(PackingError::RowOverflow { row: 1, tokens, row_len: ROW }));
    }
    let stream = TokenDataset::new((0..100).collect(), &cfg());
    assert_eq!(stream.packed_batch(&cfg(), &[vec![], vec![], vec![]]), Err(PackingError::NoExamples));
}
