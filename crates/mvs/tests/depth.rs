// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Halving a range map for a fit that runs coarser than the stereo: exact
//! for planar surfaces through any lens, and never a value between the two
//! sides of a depth edge.

use camera::{Intrinsics, Lens};
use data::rng::Lcg;

/// Two fronto-parallel planes, z = 2 on the left and z = 5 on the right,
/// seen through `k`: the range and normal at every pixel centre.
fn step_map(k: &Intrinsics, g: &mut Lcg) -> mvs::DepthMap {
    let (w, h) = (k.width, k.height);
    let mut m = mvs::DepthMap::empty(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let Some(d) = k.unproject([x as f64 + 0.5, y as f64 + 0.5]) else { continue };
            if d[2] <= 0.05 {
                continue;
            }
            let z = if x < w / 2 + 3 { 2.0 } else { 5.0 };
            m.range[i] = (z / d[2]) as f32;
            m.normal[3 * i + 2] = -1.0;
            m.conf[i] = 0.2 + 0.8 * g.unit();
        }
    }
    m
}

#[test]
fn halving_picks_one_sample_per_block_and_stays_exact_on_planes() {
    let mut g = Lcg::new(9);
    for k in [
        Intrinsics::pinhole(120.0, 96, 64),
        Intrinsics { fx: 40.0, fy: 40.0, cx: 48.5, cy: 31.0, lens: Lens::Fisheye { k: [0.04, -0.01, 0.0, 0.0] }, width: 96, height: 64 },
    ] {
        let full = step_map(&k, &mut g);
        let half = full.halved(&k);
        assert_eq!((half.width, half.height), (48, 32));
        let kh = Intrinsics { fx: k.fx / 2.0, fy: k.fy / 2.0, cx: k.cx / 2.0, cy: k.cy / 2.0, width: 48, height: 32, ..k };
        for y in 0..32u32 {
            for x in 0..48u32 {
                let o = (y * 48 + x) as usize;
                let block: Vec<usize> = [(0, 0), (1, 0), (0, 1), (1, 1)].iter().map(|(dx, dy)| ((2 * y + dy) * 96 + 2 * x + dx) as usize).collect();
                let measured: Vec<usize> = block.iter().copied().filter(|&c| full.range[c] > 0.0).collect();
                if measured.is_empty() {
                    assert_eq!(half.range[o], 0.0);
                    continue;
                }
                // the most confident sample of the block, all of it
                let best = *measured.iter().max_by(|a, b| full.conf[**a].total_cmp(&full.conf[**b])).unwrap();
                assert_eq!(half.conf[o], full.conf[best], "({x}, {y}): confidence is not the block's best");
                assert_eq!(&half.normal[3 * o..3 * o + 3], &full.normal[3 * best..3 * best + 3]);
                // its plane through the half-size pixel's own ray: one of the
                // two planes exactly, never a value between them
                let d = kh.unproject([x as f64 + 0.5, y as f64 + 0.5]).unwrap();
                let z = half.range[o] as f64 * d[2];
                let plane = if (z - 2.0).abs() < (z - 5.0).abs() { 2.0 } else { 5.0 };
                assert!((z - plane).abs() < 1e-4 * plane, "{} ({x}, {y}): z = {z} lies on neither plane", k.lens.name());
                let src_z = if (best as u32 % 96) < 96 / 2 + 3 { 2.0 } else { 5.0 };
                assert_eq!(plane, src_z, "({x}, {y}): the plane is not the chosen sample's");
            }
        }
    }
}
