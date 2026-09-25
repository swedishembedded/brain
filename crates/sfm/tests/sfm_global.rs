// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Global structure from motion on synthetic data with known truth: which
//! pairs are worth matching (retrieval by a global image descriptor), every
//! camera's rotation from noisy and partly wrong relative rotations
//! (robust rotation averaging), and every camera centre and point from
//! bearings alone (global positioning), each checked against the truth up to
//! the gauge no set of photographs can fix.
//!
//! Swedish Embedded AB implements camera calibration and multi-view
//! reconstruction for its clients. If your team needs photographs turned into
//! calibrated cameras and geometry, you can procure our services by sending
//! an email to info@swedishembedded.com.

use data::rng::Lcg;
use sfm::linalg::{dot, exp_so3, log_so3, mm, mv, norm, normalize, scale, sub, transpose, M3, V3};
use sfm::positioning::{global_positions, PositioningCfg, RayObservation};
use sfm::retrieval::{select_pairs, PairSelection};
use sfm::rotation::{average_rotations, RelativeRotation, RotationCfg};
use sfm::sift::DESC;

fn unit_descriptor(rng: &mut Lcg) -> Vec<f32> {
    // RootSIFT descriptors are non-negative and unit length
    let d: Vec<f32> = (0..DESC).map(|_| rng.unit().powi(3)).collect();
    let n = d.iter().map(|v| v * v).sum::<f32>().sqrt();
    d.iter().map(|v| v / n).collect()
}

fn perturbed(rng: &mut Lcg, d: &[f32], sigma: f32) -> Vec<f32> {
    let e: Vec<f32> = d.iter().map(|v| (v + sigma * rng.signed()).max(0.0)).collect();
    let n = e.iter().map(|v| v * v).sum::<f32>().sqrt();
    e.iter().map(|v| v / n).collect()
}

/// Images walking along a street of "places", each image seeing three
/// consecutive places plus clutter of its own, presented in shuffled order
/// so that input order says nothing: retrieval alone has to find each
/// image's true neighbours, and the pairs it proposes are far fewer than
/// every pair.
#[test]
fn retrieval_finds_the_overlapping_images_among_all_pairs() {
    let mut rng = Lcg::new(5);
    let places: Vec<Vec<Vec<f32>>> = (0..40).map(|_| (0..60).map(|_| unit_descriptor(&mut rng)).collect()).collect();
    let n = 36;
    // shuffled: image slot k shows street position order[k]
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, rng.next_u32() as usize % (i + 1));
    }
    let images: Vec<Vec<f32>> = order
        .iter()
        .map(|&pos| {
            let mut d = Vec::new();
            for place in &places[pos..pos + 3] {
                for f in place {
                    d.extend(perturbed(&mut rng, f, 0.02));
                }
            }
            for _ in 0..120 {
                d.extend(unit_descriptor(&mut rng));
            }
            d
        })
        .collect();
    let views: Vec<&[f32]> = images.iter().map(|d| d.as_slice()).collect();

    let cfg = PairSelection { exhaustive_up_to: 0, top_k: 3, sequential: 0, words: 32 };
    let pairs = select_pairs(&views, &cfg, 1);
    assert!(pairs.len() < n * (n - 1) / 4, "{} pairs proposed of {}", pairs.len(), n * (n - 1) / 2);
    for &(a, b) in &pairs {
        assert!(a < b, "pairs are ordered and unique");
    }
    // every image's immediate street neighbours are proposed
    for (ka, &pa) in order.iter().enumerate() {
        for (kb, &pb) in order.iter().enumerate() {
            if pa + 1 == pb {
                let key = (ka.min(kb), ka.max(kb));
                assert!(pairs.contains(&key), "street neighbours {pa},{pb} (images {ka},{kb}) not proposed");
            }
        }
    }

    // and a small set is matched exhaustively
    let small = PairSelection { exhaustive_up_to: n, ..cfg };
    assert_eq!(select_pairs(&views, &small, 1).len(), n * (n - 1) / 2);
}

fn rotation_angle(a: &M3, b: &M3) -> f64 {
    norm(log_so3(&mm(a, &transpose(b))))
}

fn random_rotation(rng: &mut Lcg) -> M3 {
    exp_so3([3.0 * rng.signed() as f64, 3.0 * rng.signed() as f64, 3.0 * rng.signed() as f64])
}

/// Thirty cameras, every pair within reach measured with a degree of noise
/// and a fifth of the measurements replaced by arbitrary rotations: the
/// averaged rotations land within a degree and a half of the truth (after
/// the one global rotation the data cannot fix), and the wrong measurements
/// are the ones flagged.
#[test]
fn rotation_averaging_survives_a_fifth_of_the_view_graph_being_wrong() {
    let mut rng = Lcg::new(11);
    let n = 30;
    let truth: Vec<M3> = (0..n).map(|_| random_rotation(&mut rng)).collect();
    let mut edges = Vec::new();
    let mut wrong = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            if (b - a) > 6 && rng.unit() < 0.7 {
                continue;
            }
            let bad = rng.unit() < 0.2;
            let r = if bad {
                random_rotation(&mut rng)
            } else {
                let rel = mm(&truth[b], &transpose(&truth[a]));
                let s = 1.0f64.to_radians();
                mm(&exp_so3([s * rng.signed() as f64, s * rng.signed() as f64, s * rng.signed() as f64]), &rel)
            };
            edges.push(RelativeRotation { a, b, r, weight: 100.0 + 200.0 * rng.unit() as f64 });
            wrong.push(bad);
        }
    }
    let got = average_rotations(n, &edges, &RotationCfg::default());
    let est: Vec<M3> = got.rotations.iter().map(|r| r.expect("every camera is connected")).collect();
    // the gauge: est_i = truth_i G for one rotation G
    let g = mm(&transpose(&truth[0]), &est[0]);
    let worst = (0..n).map(|i| rotation_angle(&est[i], &mm(&truth[i], &g))).fold(0.0, f64::max).to_degrees();
    assert!(worst < 1.5, "worst rotation error {worst:.2} deg");
    let caught = edges.iter().zip(&wrong).zip(&got.inlier).filter(|((_, &bad), &inl)| bad && !inl).count();
    let bad_total = wrong.iter().filter(|&&b| b).count();
    let false_alarm = wrong.iter().zip(&got.inlier).filter(|(&bad, &inl)| !bad && !inl).count();
    assert!(caught * 10 >= bad_total * 9, "{caught} of {bad_total} wrong measurements flagged");
    assert!(false_alarm * 20 <= edges.len(), "{false_alarm} right measurements flagged");
}

/// Cameras known only in rotation and points known only as the bearings in
/// which each camera sees them - a tenth of those bearings pointing
/// anywhere - are placed from a random start: every camera centre within 1%
/// of the rig's size of the truth after the best similarity.
#[test]
fn global_positioning_places_cameras_and_points_from_bearings() {
    let mut rng = Lcg::new(23);
    let ncam = 12;
    let centres: Vec<V3> = (0..ncam)
        .map(|i| {
            let a = i as f64 * 0.45;
            [4.0 * a.cos(), 0.6 * rng.signed() as f64, 4.0 * a.sin()]
        })
        .collect();
    let rots: Vec<M3> = (0..ncam).map(|_| random_rotation(&mut rng)).collect();
    let points: Vec<V3> = (0..800).map(|_| [1.5 * rng.signed() as f64, 1.5 * rng.signed() as f64, 1.5 * rng.signed() as f64]).collect();
    let mut obs = Vec::new();
    for (pi, x) in points.iter().enumerate() {
        for (ci, c) in centres.iter().enumerate() {
            if rng.unit() < 0.5 {
                continue;
            }
            let dir = if rng.unit() < 0.1 {
                normalize([rng.signed() as f64, rng.signed() as f64, rng.signed() as f64])
            } else {
                let d = normalize(sub(*x, *c));
                let s = 0.002;
                normalize([d[0] + s * rng.signed() as f64, d[1] + s * rng.signed() as f64, d[2] + s * rng.signed() as f64])
            };
            // the observation as the camera measured it, in its own frame
            obs.push(RayObservation { cam: ci, point: pi, ray: mv(&rots[ci], dir) });
        }
    }
    let got = global_positions(&rots, points.len(), &obs, &PositioningCfg::default());
    let est: Vec<V3> = got.centres.iter().map(|c| c.expect("every camera observes points")).collect();
    let worst = similarity_error(&est, &centres);
    assert!(worst < 0.01, "worst camera-centre error {worst:.4} of the rig's size");
}

/// The worst distance between `est` and `truth` after the best similarity
/// taking one onto the other (Umeyama), as a fraction of the truth's RMS
/// spread about its centroid.
fn similarity_error(est: &[V3], truth: &[V3]) -> f64 {
    let fit = sfm::georef::umeyama(est, truth, None).expect("a similarity");
    let ct = truth.iter().fold([0.0; 3], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
    let ct = scale(ct, 1.0 / truth.len() as f64);
    let spread = (truth.iter().map(|t| dot(sub(*t, ct), sub(*t, ct))).sum::<f64>() / truth.len() as f64).sqrt();
    est.iter().zip(truth).map(|(e, t)| norm(sub(fit.apply(*e), *t))).fold(0.0, f64::max) / spread
}
