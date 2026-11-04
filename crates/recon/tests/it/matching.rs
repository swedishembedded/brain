// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Descriptor matching on the device is the host's search: for every
//! descriptor, the same nearest neighbour and the same two best scores, so
//! structure from motion reconstructs the same capture whichever one ran.
//!
//! Swedish Embedded AB implements structure from motion for its clients. If
//! your team needs expertise in photogrammetry then you can procure our
//! services by sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use recon::photogrammetry::{pipelines, DeviceMatcher};
use sfm::matching::{match_with, Host, NearestNeighbours};
use sfm::sift::DESC;

/// `n` unit descriptors, the second half near copies of `like`'s rows so
/// there are true matches to find.
fn descriptors(n: usize, seed: u64, like: Option<&[f32]>) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    let mut out = Vec::with_capacity(n * DESC);
    for i in 0..n {
        let mut d: Vec<f32> = match like {
            Some(l) if i >= n / 2 && (i - n / 2) * DESC < l.len() => l[(i - n / 2) * DESC..(i - n / 2 + 1) * DESC].iter().map(|v| v + 0.02 * r.signed()).collect(),
            _ => (0..DESC).map(|_| r.unit()).collect(),
        };
        let l = d.iter().map(|v| v * v).sum::<f32>().sqrt();
        d.iter_mut().for_each(|v| *v /= l);
        out.extend(d);
    }
    out
}

#[test]
fn the_device_search_is_the_host_search() {
    let pipes: &'static [(&'static str, &'static str)] = Box::leak(pipelines().into_boxed_slice());
    let g = gpu_core::testgpu::dev(pipes);
    if !g.caps().workgroup_reductions {
        eprintln!("the device search needs a GPU; skipped on {}", g.kind());
        return;
    }
    let a = descriptors(301, 1, None);
    let b = descriptors(517, 2, Some(&a));
    let dev = DeviceMatcher::new(&g);
    for (x, y) in [(&a, &b), (&b, &a)] {
        let (want, got) = (Host.best_two(x, y), dev.best_two(x, y));
        assert_eq!(got.len(), want.len());
        for (i, (w, d)) in want.iter().zip(&got).enumerate() {
            assert!((w.1 - d.1).abs() < 1e-5 && (w.2 - d.2).abs() < 1e-5, "row {i}: scores {:?} on the host, {:?} on the device", w, d);
            // a different index only where two candidates tie to rounding
            assert!(w.0 == d.0 || (w.1 - w.2).abs() < 1e-5, "row {i}: nearest {} on the host, {} on the device", w.0, d.0);
        }
    }
    assert_eq!(match_with(&Host, &a, &b, 0.8), match_with(&dev, &a, &b, 0.8), "the mutual ratio-test matches");
    assert!(dev.best_two(&a, &[]).iter().all(|r| r.0 == usize::MAX && r.1 == -1.0), "against nothing, no neighbour");
}
