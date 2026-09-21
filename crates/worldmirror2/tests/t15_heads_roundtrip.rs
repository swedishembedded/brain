// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! T15 gate: a forward pass can be answered offline.
//!
//! Which pixels become geometry is decided by a handful of thresholds - the
//! validity mask, the depth-edge tolerance, the fusion tolerance, the support
//! floor. Choosing one cost a full forward pass, so in practice they were
//! chosen by argument instead of by measurement, and a setting that quietly
//! deleted a thin structure looked the same as one that kept it.
//!
//! Dumping the head outputs makes those questions answerable in seconds, but
//! only if the dump is the same thing the model produced. This gates that.

use splat::types::Camera;
use worldmirror2::gaussians::{assemble_from, AssembleOpts, HeadOutputs};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32
    }
}

fn heads(s: usize, w: u32, h: u32, seed: u64) -> HeadOutputs {
    let mut r = Lcg(seed);
    let hw = (w * h) as usize;
    HeadOutputs {
        // log depth around 1, log confidence, mask logit mostly positive
        gsd: (0..s)
            .map(|_| {
                (0..3 * hw)
                    .map(|i| match i / hw {
                        0 => (r.next() - 0.5) * 0.2,
                        1 => r.next(),
                        _ => r.next() * 4.0 - 1.0,
                    })
                    .collect()
            })
            .collect(),
        gsp: (0..s).map(|_| (0..12 * hw).map(|_| r.next() * 2.0 - 1.0).collect()).collect(),
        rgb: (0..s).map(|_| (0..3 * hw).map(|_| r.next()).collect()).collect(),
        pts: (0..s).map(|_| (0..4 * hw).map(|_| r.next() - 0.5).collect()).collect(),
        norm: (0..s).map(|_| (0..4 * hw).map(|_| r.next() - 0.5).collect()).collect(),
        depth: (0..s).map(|_| (0..3 * hw).map(|_| r.next() - 0.5).collect()).collect(),
        width: w,
        height: h,
    }
}

fn cams(s: usize, w: u32, h: u32) -> Vec<Camera> {
    (0..s)
        .map(|i| {
            let a = i as f32 * 0.4;
            Camera::look_at(
                [0.0, 0.0, 0.0],
                [3.0 * a.sin(), 0.5, 3.0 * a.cos()],
                [0.0, -1.0, 0.0],
                55.0,
                w,
                h,
            )
        })
        .collect()
}

/// A dumped forward pass re-assembles into exactly the scene the live one
/// produced - every gaussian, bit for bit.
#[test]
fn a_dumped_forward_pass_reassembles_into_the_same_scene() {
    let (s, w, h) = (3usize, 16u32, 12u32);
    let hd = heads(s, w, h, 0xbeef);
    let cs = cams(s, w, h);
    let opts = AssembleOpts { min_support: 1, ..Default::default() };

    let (want, _, wwt) = assemble_from(&hd, &cs, &opts);
    assert!(!want.is_empty(), "test is vacuous: the synthetic heads assembled to nothing");

    let dir = std::env::temp_dir().join(format!("wm2_heads_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    hd.save(&dir).expect("save");
    let back = HeadOutputs::load(&dir).expect("load");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(back.width, hd.width);
    assert_eq!(back.height, hd.height);
    assert_eq!(back.len(), hd.len());
    // every head round-trips, not only the two assembly used to read
    assert_eq!(back.pts, hd.pts, "the pointmap head did not survive the dump");
    assert_eq!(back.norm, hd.norm, "the normals head did not survive the dump");
    assert_eq!(back.depth, hd.depth, "the depth head did not survive the dump");
    let (got, _, gwt) = assemble_from(&back, &cs, &opts);
    assert_eq!(got.len(), want.len(), "reloaded heads assembled a different number of gaussians");
    assert_eq!(got.means, want.means, "means differ");
    assert_eq!(got.quats, want.quats, "quats differ");
    assert_eq!(got.scales, want.scales, "scales differ");
    assert_eq!(got.opacities, want.opacities, "opacities differ");
    assert_eq!(got.colors, want.colors, "colours differ");
    assert_eq!(gwt, wwt, "merge weights differ");
}

/// The thresholds actually decide something, and the test above is not
/// comparing two empty scenes because everything was rejected.
#[test]
fn the_rejection_thresholds_each_change_what_survives() {
    let (s, w, h) = (3usize, 16u32, 12u32);
    let hd = heads(s, w, h, 0x5151);
    let cs = cams(s, w, h);
    let base = AssembleOpts { edge_depth_rtol: 0.0, fuse_depth_rtol: 0.0, ..Default::default() };
    let n = |o: &AssembleOpts| assemble_from(&hd, &cs, o).0.len();

    let all = n(&AssembleOpts { gs_mask_threshold: 0.0, ..base });
    assert!(n(&base) < all, "the validity mask rejected nothing");
    assert!(
        n(&AssembleOpts { edge_depth_rtol: 0.03, ..base }) < n(&base),
        "the depth-edge tolerance rejected nothing"
    );
    assert!(
        n(&AssembleOpts { fuse_depth_rtol: 0.05, min_support: 1, ..base }) < n(&base),
        "the support floor rejected nothing"
    );
}
