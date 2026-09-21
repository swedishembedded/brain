// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The orchestration that turns a capture of ANY length into one scene.
//!
//! One forward pass holds a bounded number of frames - the trunk's global
//! attention is quadratic in frame count - so a long walk has to be split,
//! reconstructed piecewise, and put back together. Everything that can go
//! wrong in that loop is silent: a chunk plan that leaves a gap, an
//! accumulation that applies a similarity the wrong way round, a chunk that
//! did not really register and lands as a second copy of the same wall.
//!
//! These tests hold the pipeline to the four things the model cannot check
//! for itself: the plan covers the capture, the accumulation composes in the
//! right direction, a registration that failed is an ERROR and not a scene,
//! and the frames that reach the model are the sharp, distinct ones.
//!
//! None of this needs a checkpoint, and none of it names a model. The model
//! sits behind `recon::ReconstructionModel`, so the planner, the selection and
//! the accumulator are all exercised against a synthetic one - which is also
//! the proof that the trait is implementable by something that is not
//! worldmirror2.
//!
//! Swedish Embedded AB implements 3D reconstruction pipelines that scale past
//! one model's context window. If your team needs that, you can procure our
//! services by sending an email to info@swedishembedded.com.

use data::rng::Lcg;
use imaging::Rgb8;
use recon::{
    plan_chunks, reconstruct_plan, select, sharpness, ChunkScene, Dropped, Frame, PipelineError,
    PipelineOpts, ReconstructOpts, ReconstructionModel, SelectOpts, Source, MIN_OVERLAP,
};
use splat::align::{apply_sim3, transform_c2w_sim3, Sim3};
use splat::types::{Camera, Splats};

// ---------------------------------------------------------------- planning

/// The plan is the contract between the budget and the registration: every
/// frame has to be reconstructed by SOME chunk, and every consecutive pair of
/// chunks has to share enough frames for `sim3_from_cameras` to have a fit.
/// A plan that silently drops the tail, or that leaves one shared frame
/// between two chunks, produces a scene rather than a failure.
#[test]
fn every_frame_is_covered_and_adjacent_chunks_share_the_overlap() {
    for &budget in &[3usize, 4, 8, 16] {
        for overlap in MIN_OVERLAP..budget {
            // the awkward ones are in here: fewer frames than one chunk, an
            // exact multiple of the stride, one over, one under.
            for frames in 1..=(4 * budget + 3) {
                let plan = plan_chunks(frames, budget, overlap)
                    .unwrap_or_else(|e| panic!("plan({frames},{budget},{overlap}) refused: {e}"));
                assert!(!plan.chunks.is_empty(), "plan({frames},{budget},{overlap}) is empty");

                let mut seen = vec![false; frames];
                for c in &plan.chunks {
                    assert!(c.len > 0 && c.len <= budget, "chunk {c:?} is not within the budget {budget}");
                    assert!(c.end() <= frames, "chunk {c:?} runs past the {frames} frames");
                    seen[c.range()].fill(true);
                }
                let missing: Vec<usize> = (0..frames).filter(|&f| !seen[f]).collect();
                assert!(missing.is_empty(), "plan({frames},{budget},{overlap}) never reconstructs {missing:?}");

                for (i, pair) in plan.chunks.windows(2).enumerate() {
                    let shared = pair[0].end().saturating_sub(pair[1].start);
                    assert!(
                        pair[1].start >= pair[0].start,
                        "plan({frames},{budget},{overlap}) chunk {} starts before chunk {i}",
                        i + 1
                    );
                    assert!(
                        shared >= overlap.min(frames),
                        "plan({frames},{budget},{overlap}) chunks {i}/{} share {shared} frame(s), \
                         below the {overlap} the registration was asked for",
                        i + 1
                    );
                }
            }
        }
    }
}

/// A capture that fits in one pass is one chunk, not a chunk plus a stub.
#[test]
fn a_capture_shorter_than_one_chunk_is_a_single_chunk() {
    for frames in 1..=8usize {
        let plan = plan_chunks(frames, 8, 3).unwrap();
        assert_eq!(plan.chunks.len(), 1, "{frames} frame(s) in a budget of 8 planned {plan:?}");
        assert_eq!((plan.chunks[0].start, plan.chunks[0].len), (0, frames));
    }
}

/// One shared camera fixes no scale, so a plan that asks for one is refused
/// up front rather than producing chunks that cannot be registered.
#[test]
fn an_overlap_the_registration_cannot_use_is_refused() {
    assert!(matches!(plan_chunks(20, 8, 1), Err(PipelineError::OverlapTooSmall { .. })));
    assert!(matches!(plan_chunks(20, 8, 0), Err(PipelineError::OverlapTooSmall { .. })));
    // an overlap that eats the whole budget leaves no stride: the plan would
    // never advance.
    assert!(matches!(plan_chunks(20, 8, 8), Err(PipelineError::ChunkTooSmall { .. })));
    assert!(matches!(plan_chunks(0, 8, 3), Err(PipelineError::NoFrames)));
}

// ------------------------------------------------------- synthetic capture

fn look_at(eye: [f64; 3], up: [f64; 3]) -> [f64; 16] {
    let n = |v: [f64; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    };
    let z = n([-eye[0], -eye[1], -eye[2]]);
    let x = n(cross(up, z));
    let y = cross(z, x);
    [
        x[0], y[0], z[0], eye[0],
        x[1], y[1], z[1], eye[1],
        x[2], y[2], z[2], eye[2],
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Cameras on an arc looking at the origin - one smooth sweep, which is what
/// a chunk's shared frames actually are.
fn truth_cameras(n: usize) -> Vec<Camera> {
    (0..n)
        .map(|i| {
            let a = 0.35 * i as f64;
            let c2w = look_at([3.0 * a.cos(), 0.6 + 0.05 * i as f64, 3.0 * a.sin()], [0.0, 1.0, 0.0]);
            Camera {
                c2w: std::array::from_fn(|k| c2w[k] as f32),
                fx: 400.0,
                fy: 400.0,
                cx: 160.0,
                cy: 120.0,
                width: 320,
                height: 240,
            }
        })
        .collect()
}

/// A scene whose gaussians are individually identifiable, so the assembled
/// result can be matched back point by point.
fn truth_scene(n: usize) -> Splats {
    let mut s = Splats::default();
    for i in 0..n {
        let t = i as f32 * 0.37;
        s.means.extend_from_slice(&[t.cos(), 0.2 * i as f32, t.sin()]);
        s.quats.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        s.scales.extend_from_slice(&[0.01, 0.02, 0.03]);
        s.opacities.push(0.5);
        s.colors.extend_from_slice(&[0.1 * i as f32, 0.5, 0.25]);
    }
    s
}

/// The gaussians of `all` whose index satisfies `keep`.
fn subset(all: &Splats, n: usize, keep: impl Fn(usize) -> bool) -> Splats {
    let mut s = Splats::default();
    for i in (0..n).filter(|&i| keep(i)) {
        s.means.extend_from_slice(&all.means[i * 3..i * 3 + 3]);
        s.quats.extend_from_slice(&all.quats[i * 4..i * 4 + 4]);
        s.scales.extend_from_slice(&all.scales[i * 3..i * 3 + 3]);
        s.opacities.push(all.opacities[i]);
        s.colors.extend_from_slice(&all.colors[i * 3..i * 3 + 3]);
    }
    s
}

fn c2w64(c: &Camera) -> [f64; 16] {
    std::array::from_fn(|i| c.c2w[i] as f64)
}

fn blank_frames(n: usize) -> Vec<Frame> {
    (0..n).map(|i| Frame::new(Rgb8::new(2, 2, vec![0u8; 12]).unwrap(), format!("f{i:03}"), 0)).collect()
}

/// The similarity each chunk's own world is related to the true one by. The
/// model anchors every chunk to ITS first frame and normalizes ITS own
/// content, so a later chunk comes back rotated, translated and rescaled.
fn chunk_world(i: usize) -> Sim3 {
    match i {
        0 => Sim3::default(),
        1 => Sim3 { s: 2.5, r: [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], t: [1.5, -0.25, 3.0] },
        _ => Sim3 { s: 0.4, r: [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0], t: [-2.0, 0.75, 0.5] },
    }
}

/// A model that is not a model: it hands back a known scene in a per-chunk
/// arbitrary world. Everything the pipeline is allowed to know about a model
/// goes through this trait, so a fixture can be one.
struct Fixture {
    /// Truth poses for the whole capture.
    cams: Vec<Camera>,
    /// What each chunk contributes, in truth coordinates.
    parts: Vec<Splats>,
    /// Where each chunk starts, in call order.
    starts: Vec<usize>,
    next: usize,
    /// Chunk whose shared cameras get bent out of similarity.
    warp: Option<usize>,
    /// The model refuses outright.
    refuse: Option<String>,
    /// What this "model" says it can take in one pass.
    budget: usize,
}

impl Fixture {
    fn new(plan: &recon::ChunkPlan, cams: Vec<Camera>, parts: Vec<Splats>, budget: usize) -> Fixture {
        Fixture {
            cams,
            parts,
            starts: plan.chunks.iter().map(|c| c.start).collect(),
            next: 0,
            warp: None,
            refuse: None,
            budget,
        }
    }
}

impl ReconstructionModel for Fixture {
    fn name(&self) -> &str {
        "fixture"
    }
    /// No preprocessing rule of its own: it runs frames as they come.
    fn input_grid(&self, width: u32, height: u32) -> (u32, u32) {
        (width, height)
    }
    fn frame_budget(&self, _grid: (u32, u32)) -> usize {
        self.budget
    }
    fn reconstruct(&mut self, frames: &[Frame]) -> Result<ChunkScene, String> {
        if let Some(r) = &self.refuse {
            return Err(r.clone());
        }
        let i = self.next;
        self.next += 1;
        let start = self.starts[i];
        let mut scene = local_scene(i, &self.parts[i], &self.cams[start..start + frames.len()]);
        if self.warp == Some(i) {
            // Stretch each shared camera by its own factor. The headings still
            // agree, so a similarity is still SOLVED - it just does not fit,
            // which is exactly the failure that has to be caught by the
            // residual rather than by the solver returning None.
            for (k, cam) in scene.cameras.iter_mut().enumerate() {
                let f = [1.0f32, 1.9, 0.35, 2.4, 0.6][k % 5];
                for r in 0..3 {
                    cam.c2w[r * 4 + 3] *= f;
                }
            }
        }
        Ok(scene)
    }
}

fn local_scene(chunk: usize, splats: &Splats, cams: &[Camera]) -> ChunkScene {
    let w = chunk_world(chunk);
    let cameras = cams
        .iter()
        .map(|c| {
            let m = transform_c2w_sim3(&c2w64(c), &w);
            Camera { c2w: std::array::from_fn(|i| m[i] as f32), ..*c }
        })
        .collect();
    let moved = apply_sim3(splats, &w);
    let weights = moved.opacities.clone();
    ChunkScene { splats: moved, cameras, weights }
}

// ------------------------------------------------------------ accumulation

/// The one that goes wrong in the direction, not in the primitive.
///
/// `splat::align` already proves a similarity can be solved and applied; what
/// this holds is that the accumulator solves for the transform taking a
/// chunk's OWN world onto the world built so far - not the inverse, and not
/// composed in the wrong order - so that three chunks, each handed back in a
/// different arbitrary frame, land as one scene in the first chunk's frame.
#[test]
fn chunks_handed_back_in_their_own_frames_accumulate_into_one_world() {
    let frames = 8usize;
    let cams = truth_cameras(frames);
    let plan = plan_chunks(frames, 5, 3).unwrap();
    assert!(plan.chunks.len() >= 3, "wanted a chained plan, got {plan:?}");

    // Each chunk carries its OWN third of the scene, so the accumulated
    // result is exactly the truth once and can be matched point for point.
    let truth = truth_scene(12);
    let nchunks = plan.chunks.len();
    let parts: Vec<Splats> = (0..nchunks).map(|c| subset(&truth, 12, |i| i % nchunks == c)).collect();

    let opts = ReconstructOpts { voxel: 0.0, max_scale_quantile: 1.0, orient: false, ..Default::default() };
    let mut model = Fixture::new(&plan, cams.clone(), parts, 5);
    let got = reconstruct_plan(&blank_frames(frames), &plan, &mut model, &opts)
        .expect("three consistent chunks must accumulate");

    for r in &got.reports {
        assert!(
            r.relative < 1e-6,
            "chunk {} registered at a relative residual of {:.3e} on an exactly consistent capture",
            r.chunk, r.relative
        );
    }

    // Every camera lands back where the truth put it, in the FIRST chunk's
    // frame (which is the truth frame here, since chunk 0 is the identity).
    assert_eq!(got.cameras.len(), frames);
    for (i, (g, w)) in got.cameras.iter().zip(&cams).enumerate() {
        let e = (0..16).map(|k| (g.c2w[k] - w.c2w[k]).abs()).fold(0.0f32, f32::max);
        assert!(e < 1e-3, "camera {i} came back {e:.3e} from the truth pose");
    }

    // Every gaussian lands back on its truth point, and every truth point is
    // covered exactly once.
    assert_eq!(got.splats.len(), 12, "accumulated {} gaussians for 12 truth points", got.splats.len());
    let mut hit = vec![0usize; 12];
    for i in 0..got.splats.len() {
        let p = [got.splats.means[i * 3], got.splats.means[i * 3 + 1], got.splats.means[i * 3 + 2]];
        let (mut best, mut bd) = (0usize, f32::INFINITY);
        for j in 0..12 {
            let d: f32 = (0..3).map(|k| (p[k] - truth.means[j * 3 + k]).powi(2)).sum();
            if d < bd {
                bd = d;
                best = j;
            }
        }
        assert!(bd.sqrt() < 1e-3, "gaussian {i} landed {:.3e} from any truth point", bd.sqrt());
        hit[best] += 1;
    }
    assert!(hit.iter().all(|&h| h == 1), "truth points covered {hit:?}, wanted one gaussian each");
}

/// A chunk whose shared cameras are not a similarity of the ones already in
/// the world has NOT registered. Concatenating it anyway is the worst
/// available outcome: the scene silently doubles, and nothing downstream can
/// tell that from a wall that really is there twice.
#[test]
fn a_chunk_that_cannot_register_is_a_named_error_and_not_a_scene() {
    let frames = 8usize;
    let cams = truth_cameras(frames);
    let plan = plan_chunks(frames, 5, 3).unwrap();
    let truth = truth_scene(12);
    let parts: Vec<Splats> = plan.chunks.iter().map(|_| truth.clone()).collect();
    let mut model = Fixture::new(&plan, cams, parts, 5);
    model.warp = Some(1);

    let got = reconstruct_plan(&blank_frames(frames), &plan, &mut model, &ReconstructOpts::default());

    match got {
        Err(PipelineError::RegistrationFailed { chunk, shared, relative, tolerance, .. }) => {
            assert_eq!(chunk, 1, "the wrong chunk was blamed");
            assert!(shared >= MIN_OVERLAP, "reported {shared} shared frames");
            assert!(relative > tolerance, "reported {relative:.3e} against a tolerance of {tolerance:.3e}");
        }
        Err(other) => panic!("wanted a registration failure, got {other}"),
        Ok(s) => panic!("a chunk that does not register produced a scene of {} gaussians", s.splats.len()),
    }
}

/// A chunk the model itself could not produce is the model's failure, carried
/// out by name with the chunk that caused it.
///
/// The model here is a `Box<dyn ReconstructionModel>`, which is how a caller
/// that resolved it from a `--model <name>` holds one: the trait has to stay
/// object safe, and the pipeline has to accept the result.
#[test]
fn a_chunk_the_model_refuses_names_itself() {
    let plan = plan_chunks(8, 5, 3).unwrap();
    let mut fixture = Fixture::new(&plan, truth_cameras(8), vec![truth_scene(4); plan.chunks.len()], 5);
    fixture.refuse = Some("out of device memory".to_string());
    let mut model: Box<dyn ReconstructionModel> = Box::new(fixture);
    assert_eq!(model.name(), "fixture");
    match reconstruct_plan(&blank_frames(8), &plan, &mut &mut *model, &ReconstructOpts::default()) {
        Err(PipelineError::ChunkFailed { chunk, reason }) => {
            assert_eq!(chunk, 0);
            assert!(reason.contains("device memory"), "lost the reason: {reason}");
        }
        Err(other) => panic!("wanted a chunk failure, got {other}"),
        Ok(s) => panic!("a chunk the model refused produced a scene of {} gaussians", s.splats.len()),
    }
}

// --------------------------------------------------------------- selection

fn noise_frame(seed: u64, w: u32, h: u32) -> Rgb8 {
    let mut r = Lcg::new(seed);
    let px: Vec<u8> = (0..(w * h * 3) as usize).map(|_| (r.unit() * 255.0) as u8).collect();
    Rgb8::new(w, h, px).unwrap()
}

/// Something photograph-shaped: a smooth background carrying most of the
/// picture, with a few hard-edged objects on it carrying the detail. That
/// split is what the two selection statistics have to separate - a blur
/// destroys the edges and leaves the layout alone, which is exactly why a
/// blurred frame is still the SAME viewpoint.
fn scene_frame(seed: u64, w: u32, h: u32) -> Rgb8 {
    let mut r = Lcg::new(seed);
    let mut px = vec![0u8; (w * h * 3) as usize];
    for y in 0..w * h {
        let (x, yy) = (y % w, y / w);
        let i = (y * 3) as usize;
        px[i] = (40 + 120 * x / w) as u8;
        px[i + 1] = (30 + 140 * yy / h) as u8;
        px[i + 2] = (80 + 60 * (x + yy) / (w + h)) as u8;
    }
    let (bw, bh) = (w / 6, h / 6);
    for _ in 0..4 {
        let x0 = (r.unit() * (w - bw) as f32) as u32;
        let y0 = (r.unit() * (h - bh) as f32) as u32;
        let c = [(r.unit() * 255.0) as u8, (r.unit() * 255.0) as u8, (r.unit() * 255.0) as u8];
        for y in y0..y0 + bh {
            for x in x0..x0 + bw {
                let i = ((y * w + x) * 3) as usize;
                px[i..i + 3].copy_from_slice(&c);
            }
        }
    }
    Rgb8::new(w, h, px).unwrap()
}

/// A 3x3 box blur - a stand-in for the motion blur of a camera that moved
/// while the shutter was open.
fn blurred(img: &Rgb8) -> Rgb8 {
    let (w, h) = (img.w as i32, img.h as i32);
    let mut px = vec![0u8; img.px.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let (mut sum, mut n) = (0u32, 0u32);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (sx, sy) = (x + dx, y + dy);
                        if sx >= 0 && sx < w && sy >= 0 && sy < h {
                            sum += img.px[((sy * w + sx) * 3 + c) as usize] as u32;
                            n += 1;
                        }
                    }
                }
                px[((y * w + x) * 3 + c) as usize] = (sum / n) as u8;
            }
        }
    }
    Rgb8::new(img.w, img.h, px).unwrap()
}

/// Motion blur is the one defect a feed-forward reconstruction cannot recover
/// from: the frame costs a quadratic slot and contributes smeared geometry.
#[test]
fn a_blurred_frame_loses_to_a_sharp_one() {
    let sharp = scene_frame(7, 256, 192);
    let soft = blurred(&sharp);
    let (a, b) = (sharpness(&sharp), sharpness(&soft));
    assert!(a > 2.0 * b, "a 3x3 blur scored {b:.5} against the original's {a:.5}");

    let frames = vec![
        Frame::new(sharp.clone(), "sharp", 0),
        Frame::new(soft, "blurred", 0),
        Frame::new(scene_frame(99, 256, 192), "other", 0),
    ];
    let sel = select(&frames, &SelectOpts::default());
    assert_eq!(sel.keep, vec![0, 2], "the blurred frame survived selection: {sel:?}");
    assert!(
        sel.dropped.iter().any(|(i, d)| *i == 1 && matches!(d, Dropped::Blurred)),
        "the blurred frame was dropped for the wrong reason: {:?}",
        sel.dropped
    );
}

/// Two frames from the same viewpoint cost two quadratic slots and add one
/// view. The sharper of the pair is the one that stays.
#[test]
fn a_near_duplicate_viewpoint_is_dropped_and_the_sharper_one_survives() {
    let a = scene_frame(11, 256, 192);
    let b = scene_frame(12, 256, 192);
    let frames = vec![
        Frame::new(a.clone(), "a", 0),
        Frame::new(a.clone(), "a-again", 0),
        Frame::new(b, "b", 0),
    ];
    let sel = select(&frames, &SelectOpts::default());
    assert_eq!(sel.keep, vec![0, 2], "an identical repeat survived: {sel:?}");
    assert!(sel.dropped.iter().any(|(i, d)| *i == 1 && matches!(d, Dropped::NearDuplicate)));

    // Same viewpoint, different sharpness, blur gate off: the pair collapses
    // onto the sharp one even though the soft one came first.
    let frames = vec![
        Frame::new(blurred(&a), "soft", 0),
        Frame::new(a, "sharp", 0),
        Frame::new(scene_frame(13, 256, 192), "next", 0),
    ];
    let sel = select(&frames, &SelectOpts { blur_ratio: 0.0, ..SelectOpts::default() });
    assert_eq!(sel.keep, vec![1, 2], "the pair did not collapse onto the sharp frame: {sel:?}");
}

/// The budget is a hard ceiling on what reaches the model, and it has to SPAN
/// the capture: a prefix of an orbit is a worse reconstruction than the same
/// number of frames spread over all of it.
#[test]
fn the_budget_spans_the_capture_rather_than_truncating_it() {
    let frames: Vec<Frame> = (0..20).map(|i| Frame::new(noise_frame(100 + i, 32, 24), format!("f{i}"), 0)).collect();
    let sel = select(&frames, &SelectOpts { budget: 6, ..SelectOpts::default() });
    assert_eq!(sel.keep.len(), 6);
    assert_eq!(sel.keep[0], 0);
    assert_eq!(*sel.keep.last().unwrap(), 19, "the cap took a prefix: {:?}", sel.keep);
}

// ------------------------------------------------------------ mixed ingest

/// Photographs and a clip in one reconstruction means frames of different
/// sizes and different aspect ratios reaching a model that takes ONE shape for
/// the whole pass. They have to be brought onto a common one, and the aspect
/// they are brought onto is the MODEL's answer, not this crate's guess.
#[test]
fn frames_of_mixed_sizes_are_brought_onto_one_shape() {
    let frames = vec![
        Frame::new(noise_frame(1, 640, 360), "video-0", 0),
        Frame::new(noise_frame(2, 640, 360), "video-1", 0),
        Frame::new(noise_frame(3, 800, 800), "photo", 1),
    ];
    // A model that wants 16:9, as the majority of this capture already is.
    let got = recon::normalize(&frames, (1920, 1080));
    let shapes: Vec<(u32, u32)> = got.iter().map(|f| (f.image.w, f.image.h)).collect();
    assert!(shapes.iter().all(|&s| s == shapes[0]), "the pass is not one shape: {shapes:?}");
    // The majority size decides it: two video frames outvote one photograph,
    // and the frames that were already right are untouched, pixel for pixel.
    assert_eq!(shapes[0], (640, 360), "the minority shape won: {shapes:?}");
    assert_eq!(got[0].image.px, frames[0].image.px, "a uniform capture was resampled for nothing");
    assert_eq!(got[2].label, "photo", "provenance lost in normalisation");

    // A model that wants square input gets square frames, from the same
    // capture and with no change to this crate.
    let square = recon::normalize(&frames, (518, 518));
    assert_eq!(square[0].image.w, square[0].image.h, "a square model did not get square frames");
}

/// A folder of photographs, a clip, or both - one ordered set, with every frame
/// still able to say where it came from.
#[test]
fn several_sources_become_one_ordered_set() {
    let d = std::env::temp_dir().join(format!("brain-recon-ingest-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    for (i, n) in ["b.ppm", "a.ppm", "c.ppm"].iter().enumerate() {
        imaging::save_ppm(d.join(n), &noise_frame(200 + i as u64, 8, 8)).unwrap();
    }
    let d2 = d.join("more");
    std::fs::create_dir_all(&d2).unwrap();
    let extra = d2.join("later.ppm");
    imaging::save_ppm(&extra, &noise_frame(300, 8, 8)).unwrap();

    let got = recon::ingest(&[Source::Dir(d.clone()), Source::Images(vec![extra.clone()])])
    .expect("two image sources must ingest");
    let labels: Vec<&str> = got.iter().map(|f| f.label.as_str()).collect();
    assert_eq!(labels.len(), 4, "got {labels:?}");
    assert!(labels[0].ends_with("a.ppm") && labels[1].ends_with("b.ppm") && labels[2].ends_with("c.ppm"),
        "a directory is not in its own filename order: {labels:?}");
    assert!(labels[3].ends_with("later.ppm"), "the second source did not follow the first: {labels:?}");
    assert_eq!(got[3].source, 1, "provenance lost");
    let _ = std::fs::remove_dir_all(&d);
}

/// The whole path, and the one question that keeps model knowledge out of the
/// pipeline: how many frames fit in a pass is the MODEL's answer. A model that
/// can take 32 frames gets one chunk; one that can take five gets a plan built
/// around five, from the same call with nothing else changed.
#[test]
fn the_frames_per_pass_come_from_the_model() {
    let d = std::env::temp_dir().join(format!("brain-recon-run-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let n = 9usize;
    for i in 0..n {
        imaging::save_ppm(d.join(format!("f{i:03}.ppm")), &scene_frame(400 + i as u64, 128, 96)).unwrap();
    }
    let cams = truth_cameras(n);
    let truth = truth_scene(9);

    for budget in [5usize, 32] {
        let plan = plan_chunks(n, budget, 3).unwrap();
        let parts: Vec<Splats> = (0..plan.chunks.len())
            .map(|c| subset(&truth, 9, |i| i % plan.chunks.len() == c))
            .collect();
        let mut model = Fixture::new(&plan, cams.clone(), parts, budget);
        let opts = PipelineOpts {
            budget: None,
            overlap: 3,
            // Selection has its own tests; this one is about the plan, so
            // every frame written is a frame reconstructed.
            select: SelectOpts { blur_ratio: 0.0, min_change: 0.0, budget: 0 },
            scene: ReconstructOpts { voxel: 0.0, max_scale_quantile: 1.0, orient: false, ..Default::default() },
        };
        let cap = recon::run(&[Source::Dir(d.clone())], &opts, &mut model).expect("a whole capture");
        assert_eq!(cap.budget, budget, "the pipeline did not ask the model");
        assert_eq!(cap.plan.chunks.len(), plan.chunks.len());
        assert!(cap.plan.chunks.iter().all(|c| c.len <= budget), "a chunk exceeded the model's budget");
        assert_eq!(cap.scene.cameras.len(), n, "not every selected frame came back with a pose");
        assert!(cap.scene.worst_relative() < 1e-6, "worst residual {:.3e}", cap.scene.worst_relative());
    }
    let _ = std::fs::remove_dir_all(&d);
}
