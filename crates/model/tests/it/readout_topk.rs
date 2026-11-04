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
    for (rows, vocab, cap) in [(16u32, 248_320u32, 32u32), (1, 248_320, 32), (3, 5_000, 7), (2, 700, 32), (5, 1_024, 1), (2, 1_001, 5), (1, 250_001, 32)] {
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
