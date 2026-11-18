// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::readout::RowDrawer`: the keyed draw run where the logits are draws the token the host reference
//! (`model::serve::draw_row`) draws, with its log-probability, for every row of a batch whatever the row's
//! neighbours ask for (temperature, min-p, nucleus and top-k, from the row's candidates or the row itself), and over
//! many keys lands on each token as often as the exact distribution says.
//!
//! Swedish Embedded AB implements on-device sampling for language-model serving for its clients. If your team
//! needs expertise in drawing tokens exactly without reading the vocabulary back, you can procure our services by
//! sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use model::readout::{RowDrawer, DRAW_KERNELS};
use model::serve::{draw_key, draw_row, Drawn, RowDraw};

/// The distribution `sample_logits_row` defines: best first (equal logits lowest id first), the best `top_k`, the
/// softmax of `logit / temp`, the nucleus against the row's mass (the kept list's with a `top_k`), then the min-p
/// floor; unnormalised weights.
fn exact_truncated(row: &[f32], temp: f32, min_p: f32, top_p: f32, top_k: usize) -> Vec<f64> {
    let mut order: Vec<usize> = (0..row.len()).filter(|&i| !row[i].is_nan()).collect();
    order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
    let best = row[order[0]] as f64;
    let weight = |i: usize| ((row[i] as f64 - best) / temp as f64).exp();
    let kept: Vec<usize> = if top_k > 0 { order[..top_k.min(order.len())].to_vec() } else { order.clone() };
    let whole: f64 = kept.iter().map(|&i| weight(i)).sum();
    let mut cut = kept.len();
    if top_p < 1.0 {
        let mut acc = 0.0;
        for (r, &i) in kept.iter().enumerate() {
            acc += weight(i);
            if acc / whole >= top_p as f64 {
                cut = r + 1;
                break;
            }
        }
    }
    let mut w = vec![0.0; row.len()];
    for &i in &kept[..cut] {
        if weight(i) > 0.0 && weight(i) >= min_p as f64 {
            w[i] = weight(i);
        }
    }
    w
}

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
        for round in 0..4u64 {
            // The candidate extraction masks the logits it takes: every round starts from the row as it was.
            let logits = g.storage_init("logits", &host);
            // Plain draws, a floor, nuclei (the flat half-step rows hold more than the candidates can, so some rows
            // are drawn whole), top_k inside and past the candidate list, and a row not drawn.
            let draws: Vec<Option<RowDraw>> = (0..rows)
                .map(|r| match (r + round as usize) % 9 {
                    8 => None,
                    k => Some(RowDraw {
                        temp: [1.0, 0.6, 1.7, 1.0, 1.0, 0.3, 1.0, 0.8][k],
                        min_p: [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1][k],
                        top_p: [1.0, 1.0, 1.0, 1.0, 0.9, 0.5, 1.0, 0.7][k],
                        top_k: [0, 0, 0, 0, 0, 0, 40, 2000][k],
                        key: draw_key(round * 1000 + r as u64, round, 0),
                    }),
                })
                .collect();
            let shape = drawer.shape(&draws, vocab as u32);
            let steps = drawer.steps(&g, &logits, &draws, vocab as u32);
            g.submit(&[], &steps);
            let got = drawer.answer(&g, &logits, &draws, vocab as u32, shape);
            for (r, (d, got)) in draws.iter().zip(&got).enumerate() {
                let row = &host[r * vocab..(r + 1) * vocab];
                match (d, got) {
                    (Some(d), Drawn::Token(t, lp, whole)) => {
                        let (want, want_lp) = draw_row(row, *d);
                        assert_eq!(*t, want, "{rows}x{vocab} round {round} row {r} {d:?}");
                        // A NaN logit carries no mass (the row fixture plants one), as in the draw's own weights.
                        let clean: Vec<f32> = row.iter().map(|&l| if l.is_nan() { f32::NEG_INFINITY } else { l }).collect();
                        let want_whole = model::logprobs::row_logprob_at(&clean, want as usize, d.temp);
                        // A row of no mass has no distribution to report a figure under (the draw answers 0).
                        assert!(want_whole.is_nan() || (whole - want_whole).abs() <= 1e-4 * want_whole.abs().max(1.0), "{rows}x{vocab} round {round} row {r} {d:?}: whole-row logprob {whole} vs {want_whole}");
                        assert!((lp - want_lp).abs() <= 1e-4 * want_lp.abs().max(1.0), "{rows}x{vocab} round {round} row {r}: logprob {lp} vs {want_lp}");
                    }
                    (None, Drawn::Row(back)) => assert!(back.iter().zip(row).all(|(a, b)| a.to_bits() == b.to_bits()), "row {r} came back changed"),
                    other => panic!("row {r}: {other:?}"),
                }
            }
        }
    }
}

/// 64 rows of the same small row drawn under 640 keys each: the device's tokens follow the row sampler's distribution
/// (Pearson's statistic below the one-in-a-million quantile) for a plain temperature, a min-p floor, a nucleus, a
/// `top_k` and both.
#[test]
fn the_device_draw_follows_the_exact_distribution() {
    let g = gpu_core::testgpu::dev(&DRAW_KERNELS);
    let row = [2.0f32, 1.5, 1.5, 0.3, -1.0, f32::NAN, f32::NEG_INFINITY, -3.0, 0.0, 1.9, -0.5, 0.7, -2.0];
    let (rows, vocab) = (64usize, row.len());
    let drawer = RowDrawer::new(&g, rows as u32).unwrap();
    for (temp, min_p, top_p, top_k) in [(1.0f32, 0.0f32, 1.0f32, 0usize), (1.4, 0.15, 1.0, 0), (1.0, 0.0, 0.8, 0), (1.2, 0.0, 1.0, 4), (1.0, 0.05, 0.9, 6)] {
        let w = exact_truncated(&row, temp, min_p, top_p, top_k);
        let z: f64 = w.iter().sum();
        let mut counts = vec![0u64; vocab];
        for round in 0..640u64 {
            let draws: Vec<Option<RowDraw>> = (0..rows).map(|r| Some(RowDraw { temp, min_p, top_p, top_k, key: draw_key(r as u64, round, 0) })).collect();
            let logits = g.storage_init("logits", &row.repeat(rows));
            let shape = drawer.shape(&draws, vocab as u32);
            g.submit(&[], &drawer.steps(&g, &logits, &draws, vocab as u32));
            for d in drawer.answer(&g, &logits, &draws, vocab as u32, shape) {
                let Drawn::Token(t, ..) = d else { panic!("every row is drawn") };
                counts[t as usize] += 1;
            }
        }
        let n = (rows * 640) as f64;
        let mut stat = 0.0;
        for (c, w) in counts.iter().zip(&w) {
            if *w == 0.0 {
                assert_eq!(*c, 0, "{temp} {min_p} {top_p} {top_k}: a draw left the survivors: {counts:?}");
            } else {
                let e = w / z * n;
                stat += (*c as f64 - e).powi(2) / e;
            }
        }
        let dof = w.iter().filter(|&&w| w > 0.0).count() - 1;
        assert!(stat < (3 * dof + 30) as f64, "{temp} {min_p} {top_p} {top_k}: chi-square {stat:.1} over {dof} dof: {counts:?}");
    }
}
