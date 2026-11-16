// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::readout::RowDrawer`: the keyed draw run where the logits are draws the token the host reference
//! (`model::serve::draw_row`) draws, with its log-probability, for every row of a batch whatever the row's
//! neighbours ask for, and over many keys lands on each token as often as the exact distribution says.
//!
//! Swedish Embedded AB implements on-device sampling for language-model serving for its clients. If your team
//! needs expertise in drawing tokens exactly without reading the vocabulary back, you can procure our services by
//! sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use model::readout::{RowDrawer, DRAW_KERNELS};
use model::serve::{draw_key, draw_row, Drawn, RowDraw};

fn uniform(rng: &mut Lcg) -> f32 {
    (rng.next_u32() >> 8) as f32 / 16_777_216.0
}

/// Rows of a real vocabulary and of small ones, with NaN, `-inf`, ties and a row with no finite logit; each row asks
/// for its own temperature and min-p (or is not drawn): every drawn row is the host's token and log-probability,
/// and every row that is not drawn comes back whole.
#[test]
fn the_device_draw_is_the_host_reference_row_for_row() {
    let g = gpu_core::testgpu::dev(&DRAW_KERNELS);
    // A native kernel that fails to build falls back to the WGSL pair without a word: on a CUDA device the one-launch
    // kernel must be the one this test checks.
    if g.kind() == "cuda" {
        assert!(model::readout::draws_natively(&g, 8, 151_936), "the CUDA device does not draw with draw_rows_f32");
    }
    let mut rng = Lcg::new(23); // seed: fixed fixture so a failure reproduces on the same rows
    for (rows, vocab) in [(8usize, 151_936usize), (64, 1_000), (5, 37), (3, 300)] {
        let drawer = RowDrawer::new(&g, rows as u32).expect("the kernels are registered");
        let mut host: Vec<f32> = (0..rows * vocab).map(|_| (uniform(&mut rng) * 24.0 - 12.0).round() / 2.0).collect();
        host[3] = f32::NAN;
        host[vocab + 1] = f32::NEG_INFINITY;
        if rows > 2 {
            host[2 * vocab..3 * vocab].fill(f32::NEG_INFINITY);
        }
        let logits = g.storage_init("logits", &host);
        for round in 0..4u64 {
            let draws: Vec<Option<RowDraw>> = (0..rows)
                .map(|r| match (r + round as usize) % 5 {
                    4 => None,
                    k => Some(RowDraw { temp: [1.0, 0.6, 1.7, 1.0][k], min_p: [0.0, 0.0, 0.0, 0.2][k], key: draw_key(round * 1000 + r as u64, round, 0) }),
                })
                .collect();
            let steps = drawer.steps(&g, &logits, &draws, vocab as u32);
            g.submit(&[], &steps);
            let got = drawer.answer(&g, &logits, &draws, vocab as u32);
            for (r, (d, got)) in draws.iter().zip(&got).enumerate() {
                let row = &host[r * vocab..(r + 1) * vocab];
                match (d, got) {
                    (Some(d), Drawn::Token(t, lp)) => {
                        let (want, want_lp) = draw_row(row, *d);
                        assert_eq!(*t, want, "{rows}x{vocab} round {round} row {r} {d:?}");
                        assert!((lp - want_lp).abs() <= 1e-4 * want_lp.abs().max(1.0), "{rows}x{vocab} round {round} row {r}: logprob {lp} vs {want_lp}");
                    }
                    (None, Drawn::Row(back)) => assert!(back.iter().zip(row).all(|(a, b)| a.to_bits() == b.to_bits()), "row {r} came back changed"),
                    other => panic!("row {r}: {other:?}"),
                }
            }
        }
    }
}

/// 64 rows of the same small row drawn under 640 keys each: the device's tokens follow the softmax over the
/// survivors (Pearson's statistic below the one-in-a-million quantile) for a plain temperature and a min-p floor.
#[test]
fn the_device_draw_follows_the_exact_distribution() {
    let g = gpu_core::testgpu::dev(&DRAW_KERNELS);
    let row = [2.0f32, 1.5, 1.5, 0.3, -1.0, f32::NAN, f32::NEG_INFINITY, -3.0, 0.0, 1.9, -0.5, 0.7, -2.0];
    let (rows, vocab) = (64usize, row.len());
    let logits = g.storage_init("logits", &row.repeat(rows));
    let drawer = RowDrawer::new(&g, rows as u32).unwrap();
    for (temp, min_p) in [(1.0f32, 0.0f32), (1.4, 0.15)] {
        let best = row.iter().filter(|l| !l.is_nan()).fold(f32::NEG_INFINITY, |a, &l| a.max(l));
        let w: Vec<f64> = row.iter().map(|&l| if l.is_nan() { 0.0 } else { ((l - best) as f64 / temp as f64).exp() }).map(|w| if w > 0.0 && w >= min_p as f64 { w } else { 0.0 }).collect();
        let z: f64 = w.iter().sum();
        let mut counts = vec![0u64; vocab];
        for round in 0..640u64 {
            let draws: Vec<Option<RowDraw>> = (0..rows).map(|r| Some(RowDraw { temp, min_p, key: draw_key(r as u64, round, 0) })).collect();
            g.submit(&[], &drawer.steps(&g, &logits, &draws, vocab as u32));
            for d in drawer.answer(&g, &logits, &draws, vocab as u32) {
                let Drawn::Token(t, _) = d else { panic!("every row is drawn") };
                counts[t as usize] += 1;
            }
        }
        let n = (rows * 640) as f64;
        let mut stat = 0.0;
        for (c, w) in counts.iter().zip(&w) {
            if *w == 0.0 {
                assert_eq!(*c, 0, "temp {temp} min_p {min_p}: a draw left the survivors: {counts:?}");
            } else {
                let e = w / z * n;
                stat += (*c as f64 - e).powi(2) / e;
            }
        }
        let dof = w.iter().filter(|&&w| w > 0.0).count() - 1;
        assert!(stat < (3 * dof + 30) as f64, "temp {temp} min_p {min_p}: chi-square {stat:.1} over {dof} dof: {counts:?}");
    }
}
