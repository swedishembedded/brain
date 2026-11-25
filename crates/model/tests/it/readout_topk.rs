// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::readout::topk_rows`: every row's best candidates, best first, the lower column first among equal values -
//! the native one-launch kernel (CUDA devices) and the chain of argmax rounds it replaces (every other device) give
//! the same candidates as a sort of the row on the host, however the values tie, and the device is told once.
//!
//! Swedish Embedded AB implements low-latency sampling heads for language-model serving for its clients. If your
//! team needs expertise in reading the top candidates of a vocabulary-wide logits row without a launch per
//! candidate, you can procure our services by sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use model::readout::ReadoutKernels;

const KERNELS: &[(&str, &str)] =
    &[("argmax_part", kernels::ARGMAX_PART), ("argmax_final", kernels::ARGMAX_FINAL), ("topk_extract_step", kernels::TOPK_EXTRACT_STEP)];

fn host_topk(row: &[f32], cap: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = row.iter().copied().enumerate().map(|(i, v)| (i as u32, v)).collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.truncate(cap);
    all
}

#[test]
fn the_candidates_are_the_rows_best_in_order_with_ties_to_the_lower_column() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let idx = |n: &str| g.kernel_index(n).unwrap_or_else(|| panic!("kernel '{n}' not registered"));
    let k = ReadoutKernels { part: idx("argmax_part"), fin: idx("argmax_final"), extract: idx("topk_extract_step") };
    let mut rng = Lcg::new(11); // seed: fixed fixture so a failure reproduces on the same values
    // (rows, vocab, cap): a real vocabulary at the serving widths, a tiny one, a vocabulary shorter than the block.
    // A codec codebook as a speech decoder samples it (32 streams, 2048 codes, top 50), and widths around the
    // shared-memory sort's limit.
    for (rows, vocab, cap) in [(16u32, 248_320u32, 32u32), (1, 248_320, 32), (3, 5_000, 7), (2, 700, 32), (5, 1_024, 1), (2, 1_001, 5), (1, 250_001, 32), (32, 2_048, 50), (4, 4_096, 64), (3, 4_097, 9), (2, 50, 50)] {
        // Values drawn from a small set so that many columns tie, including at the top.
        let host: Vec<f32> = (0..rows * vocab).map(|_| (rng.next_u32() % 4000) as f32 / 100.0 - 20.0).collect();
        let logits = g.storage_init("logits", &host);
        let got = model::readout::topk_rows(&g, k, &logits, rows, vocab, cap);
        for (r, row) in got.iter().enumerate() {
            let want = host_topk(&host[r * vocab as usize..(r + 1) * vocab as usize], cap as usize);
            assert_eq!(row, &want, "row {r} of {rows} x {vocab}, cap {cap}");
        }
    }
}

/// `topk_rows_steps` writes into buffers the caller owns, so a decoder can record the extraction beside its own steps and
/// keep its candidate block: all columns of a `[rows, stride]` block when `count == stride` (the one-launch kernel where the
/// device offers it) and the first `count` columns when fewer are wanted (the chain of rounds), the rest untouched.
#[test]
fn the_steps_fill_the_callers_candidate_block_whole_or_in_its_first_columns() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let idx = |n: &str| g.kernel_index(n).unwrap_or_else(|| panic!("kernel '{n}' not registered"));
    let k = ReadoutKernels { part: idx("argmax_part"), fin: idx("argmax_final"), extract: idx("topk_extract_step") };
    let mut rng = Lcg::new(23); // seed: fixed fixture so a failure reproduces on the same values
    let (rows, vocab, stride) = (6u32, 20_000u32, 16u32);
    for count in [stride, 5, 1] {
        let host: Vec<f32> = (0..rows * vocab).map(|_| (rng.next_u32() % 3000) as f32 / 100.0 - 15.0).collect();
        let logits = g.storage_init("logits", &host);
        let sentinel = -777.0f32;
        let vals = g.storage_init("vals", &vec![sentinel; (rows * stride) as usize]);
        let ids = g.storage_init("ids", &vec![sentinel; (rows * stride) as usize]);
        let steps = model::readout::topk_rows_steps(&g, k, &logits, &vals, &ids, rows, vocab, stride, count);
        g.submit(&[], &steps);
        let (vals, ids) = (g.read(&vals, (rows * stride) as usize), g.read(&ids, (rows * stride) as usize));
        for r in 0..rows as usize {
            let want = host_topk(&host[r * vocab as usize..(r + 1) * vocab as usize], count as usize);
            let got: Vec<(u32, f32)> = (0..count as usize).map(|c| (ids[r * stride as usize + c] as u32, vals[r * stride as usize + c])).collect();
            assert_eq!(got, want, "row {r}, {count} of {stride} columns");
            assert!(vals[r * stride as usize + count as usize..(r + 1) * stride as usize].iter().all(|&v| v == sentinel), "row {r}: columns past the {count} wanted stay as they were");
        }
    }
}

/// The extraction leaves every column it did not take as it was (a taken one may be masked to -3.4e38 in place), and a
/// row with fewer real candidates than asked reports the rest at -3.4e38, after its real ones.
#[test]
fn the_taken_columns_are_masked_in_place_and_a_short_row_reports_masked_candidates() {
    let g = gpu_core::testgpu::dev(KERNELS);
    let idx = |n: &str| g.kernel_index(n).unwrap_or_else(|| panic!("kernel '{n}' not registered"));
    let k = ReadoutKernels { part: idx("argmax_part"), fin: idx("argmax_final"), extract: idx("topk_extract_step") };
    let mut rng = Lcg::new(31); // seed: fixed fixture so a failure reproduces on the same values
    for (rows, vocab, cap) in [(8u32, 2_048u32, 50u32), (2, 30_000, 16)] {
        let host: Vec<f32> = (0..rows * vocab).map(|_| (rng.next_u32() % 500) as f32 / 10.0 - 25.0).collect();
        let logits = g.storage_init("logits", &host);
        let got = model::readout::topk_rows(&g, k, &logits, rows, vocab, cap);
        let after = g.read(&logits, (rows * vocab) as usize);
        for (r, row) in got.iter().enumerate() {
            let base = r * vocab as usize;
            let taken: std::collections::HashSet<usize> = row.iter().map(|&(i, _)| i as usize).collect();
            for c in 0..vocab as usize {
                let ok = after[base + c] == host[base + c] || (taken.contains(&c) && after[base + c] == -3.4e38);
                assert!(ok, "row {r} column {c} of {vocab}, cap {cap}: {} became {}", host[base + c], after[base + c]);
            }
        }
    }
    // Three real values and a row of masked ones: the candidates past the real ones are masked.
    let mut host = vec![-3.4e38f32; 2_048];
    host[7] = 1.0;
    host[3] = 2.0;
    host[900] = 1.0;
    let logits = g.storage_init("logits", &host);
    let got = model::readout::topk_rows(&g, k, &logits, 1, 2_048, 5);
    assert_eq!(got[0][..3], [(3, 2.0), (7, 1.0), (900, 1.0)]);
    assert!(got[0][3..].iter().all(|&(_, v)| v == -3.4e38), "{:?}", got[0]);
}
