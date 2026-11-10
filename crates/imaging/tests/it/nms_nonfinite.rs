// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A detector's confidence can be NaN (a NaN logit after a numerical fault).
//! NMS must not keep such a candidate, and must still keep the real ones in
//! confidence order: a NaN breaks a partial-order sort, which can misorder the
//! list or panic in the standard library's sort.

use imaging::nms::{nms, nms_agnostic};

#[test]
fn a_nan_or_infinite_confidence_is_never_kept() {
    let dets = vec![
        [0.0, 0.0, 10.0, 10.0, f32::NAN, 0.0],
        [20.0, 20.0, 30.0, 30.0, 0.9, 0.0],
        [40.0, 40.0, 50.0, 50.0, f32::INFINITY, 0.0],
        [60.0, 60.0, 70.0, 70.0, 0.5, 0.0],
    ];
    for kept in [nms(&dets, 0.5, 300), nms_agnostic(&dets, 0.5, 300)] {
        let confs: Vec<f32> = kept.iter().map(|d| d[4]).collect();
        assert_eq!(confs, vec![0.9, 0.5], "kept {kept:?}");
    }
}

#[test]
fn many_nans_among_real_candidates_do_not_disturb_the_order() {
    let mut dets = Vec::new();
    for i in 0..200 {
        let x = (i * 20) as f32;
        let conf = if i % 3 == 0 { f32::NAN } else { (i % 97) as f32 / 97.0 };
        dets.push([x, 0.0, x + 10.0, 10.0, conf, 0.0]);
    }
    let kept = nms(&dets, 0.5, 300);
    assert!(kept.iter().all(|d| d[4].is_finite()));
    assert!(kept.windows(2).all(|w| w[0][4] >= w[1][4]), "not in descending confidence");
    assert_eq!(kept.len(), dets.iter().filter(|d| d[4].is_finite()).count());
}
