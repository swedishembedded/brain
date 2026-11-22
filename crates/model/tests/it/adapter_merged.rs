// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-rank adaptation of every learned tensor of a model for its clients. If your team
// needs expertise in parameter-efficient fine-tuning beyond linear layers then you can procure our services by sending
// an email to info@swedishembedded.com.

//! Spec: a merged LoRA pair (`model::adapter::merged`) forms `W + s B A` exactly, and projects a gradient of the
//! effective matrix onto its factors as the chain rule says: `dB += s dW A^T`, `dA += s B^T dW`, both added to what the
//! gradients held, `A`'s skipped when it is frozen.

use data::rng::{random_seed, Lcg};
use model::adapter::merged::{MergedLora, KERNELS};

fn pipes() -> &'static [(&'static str, &'static str)] {
    static P: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let mut v: Vec<(&str, &str)> = model::ops::kernel_list().to_vec();
        model::ops::extend_kernel_list(&mut v, KERNELS);
        v
    })
}

fn close(got: &[f32], want: &[f64], what: &str) {
    let scale = want.iter().fold(1e-6f64, |m, v| m.max(v.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((f64::from(*g) - w).abs() <= 1e-5 * scale, "{what}[{i}]: {g} vs {w}");
    }
}

#[test]
fn a_merged_pair_forms_the_effective_matrix_and_projects_its_gradient_onto_the_factors() {
    let g = gpu_core::testgpu::dev(pipes());
    let ops = model::ops::Ops::new(g.share()).unwrap();
    let mut rng = Lcg::new(random_seed());
    let (out, inn, r, s) = (5usize, 7usize, 3usize, 0.75f32);
    let (w, a, b, dw) = (rng.vec_scaled(out * inn, 1.0), rng.vec_scaled(r * inn, 1.0), rng.vec_scaled(out * r, 1.0), rng.vec_scaled(out * inn, 1.0));
    let (ga0, gb0) = (rng.vec_scaled(r * inn, 1.0), rng.vec_scaled(out * r, 1.0));
    let m = MergedLora::new(&g, out as u32, inn as u32, r as u32, s).unwrap();
    let (wb, ab, bb) = (g.storage_init("w", &w), g.storage_init("a", &a), g.storage_init("b", &b));
    let (ga, gb) = (g.storage_init("ga", &ga0), g.storage_init("gb", &gb0));
    let mut steps = Vec::new();
    m.refresh(&ops, &mut steps, &wb, &ab, &bb);
    g.submit(&[], &steps);
    let eff: Vec<f64> = (0..out * inn).map(|i| f64::from(w[i]) + f64::from(s) * (0..r).map(|k| f64::from(b[(i / inn) * r + k]) * f64::from(a[k * inn + i % inn])).sum::<f64>()).collect();
    close(&g.read(m.eff(), out * inn), &eff, "W_eff");

    g.write_f32(m.d_eff(), &dw);
    let mut steps = Vec::new();
    m.project(&ops, &mut steps, &ab, &bb, Some(&ga), &gb);
    g.submit(&[], &steps);
    let want_gb: Vec<f64> = (0..out * r).map(|i| f64::from(gb0[i]) + f64::from(s) * (0..inn).map(|c| f64::from(dw[(i / r) * inn + c]) * f64::from(a[(i % r) * inn + c])).sum::<f64>()).collect();
    let want_ga: Vec<f64> = (0..r * inn).map(|i| f64::from(ga0[i]) + f64::from(s) * (0..out).map(|o| f64::from(b[o * r + i / inn]) * f64::from(dw[o * inn + i % inn])).sum::<f64>()).collect();
    close(&g.read(&gb, out * r), &want_gb, "dB");
    close(&g.read(&ga, r * inn), &want_ga, "dA");

    // A frozen A: only B's gradient moves.
    let gb2 = g.storage_init("gb2", &gb0);
    let mut steps = Vec::new();
    m.project(&ops, &mut steps, &ab, &bb, None, &gb2);
    g.submit(&[], &steps);
    close(&g.read(&gb2, out * r), &want_gb, "dB with A frozen");
    assert!(MergedLora::new(&g, 0, 3, 2, 1.0).is_err(), "an empty pair is refused");
}
