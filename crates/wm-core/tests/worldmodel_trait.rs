// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Unit P1.worldmodel-trait — tests written FROM THE SPEC, never from the
//! implementation.
//!
//! Every test is named `worldmodel_*` so `cargo test worldmodel` selects
//! exactly this unit's tests.
//!
//! Determinism is normative (spec §7): all frame comparisons are BITWISE
//! (`f32::to_bits`), never tolerance-based.
//!
//! Spec §8: this unit has NO backward ops and NO kernels, therefore no
//! gradcheck/FD entry is required.

use wm_core::{FakeWorldModel, WorldModel};

const C: usize = 3;
const H: usize = 64;
const W: usize = 64;
const FRAME_LEN: usize = C * H * W; // 12288 (spec §3)

/// `frame[c*H*W + y*W + x]` (spec §2).
fn idx(c: usize, y: usize, x: usize) -> usize {
    c * H * W + y * W + x
}

fn bits(frame: &[f32]) -> Vec<u32> {
    frame.iter().map(|f| f.to_bits()).collect()
}

fn assert_frame_bits_eq(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: frames differ at index {i}: {x} vs {y}"
        );
    }
}

/// Recover the square origin from a frame using the spec §4.3
/// distinguishability invariant: the blue channel is exactly 1.0 on the 64
/// footprint pixels (8 wrapped columns x 8 wrapped rows) and 0.25 elsewhere.
fn recover_origin(frame: &[f32]) -> (u32, u32) {
    let one = 1.0f32.to_bits();
    let mut cols = [false; W];
    let mut rows = [false; H];
    let mut count = 0usize;
    for y in 0..H {
        for x in 0..W {
            if frame[idx(2, y, x)].to_bits() == one {
                cols[x] = true;
                rows[y] = true;
                count += 1;
            }
        }
    }
    assert_eq!(count, 64, "square footprint must be exactly 64 blue-channel 1.0 pixels");
    assert_eq!(cols.iter().filter(|c| **c).count(), 8, "footprint must span 8 columns");
    assert_eq!(rows.iter().filter(|r| **r).count(), 8, "footprint must span 8 rows");
    // The origin is the unique wrapped-contiguous start: the column whose
    // left neighbour (mod 64) is off-footprint; same for rows.
    let px = (0..W)
        .find(|&x| cols[x] && !cols[(x + W - 1) % W])
        .expect("wrapped column run must have a unique start");
    let py = (0..H)
        .find(|&y| rows[y] && !rows[(y + H - 1) % H])
        .expect("wrapped row run must have a unique start");
    (px as u32, py as u32)
}

/// Reference math transcribed from spec §4 ONLY (not from any implementation).
/// Its own correctness is pinned to the spec's hand-computed checkpoints by
/// `worldmodel_specref_selfcheck_against_spec_constants`.
mod specref {
    use super::{idx, C, FRAME_LEN, H, W};

    pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    pub const FNV_PRIME: u64 = 0x100000001b3;

    pub fn fnv1a<I: IntoIterator<Item = u8>>(bytes: I) -> u64 {
        let mut h = FNV_OFFSET;
        for b in bytes {
            h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
        }
        h
    }

    /// Spec §4.1 byte stream: LE lengths prepended, f32s by bit pattern.
    pub fn ctx_bytes(ctx_frames: &[f32], ctx_actions: &[u32]) -> Vec<u8> {
        let mut s = Vec::new();
        s.extend_from_slice(&(ctx_frames.len() as u64).to_le_bytes());
        for f in ctx_frames {
            s.extend_from_slice(&f.to_bits().to_le_bytes());
        }
        s.extend_from_slice(&(ctx_actions.len() as u64).to_le_bytes());
        for a in ctx_actions {
            s.extend_from_slice(&a.to_le_bytes());
        }
        s
    }

    /// Spec §4.1: px = h & 63, py = (h >> 32) & 63.
    pub fn seed(ctx_frames: &[f32], ctx_actions: &[u32]) -> (u32, u32) {
        let h = fnv1a(ctx_bytes(ctx_frames, ctx_actions));
        ((h & 63) as u32, ((h >> 32) & 63) as u32)
    }

    /// Spec §4.2 move table + toroidal wrap.
    pub fn step_pos(px: u32, py: u32, action: u32) -> (u32, u32) {
        assert!(action < 5);
        let (dx, dy): (i64, i64) = match action {
            0 => (0, 0),
            1 => (0, -1),
            2 => (-1, 0),
            3 => (0, 1),
            4 => (1, 0),
            _ => unreachable!(),
        };
        (
            (px as i64 + dx).rem_euclid(64) as u32,
            (py as i64 + dy).rem_euclid(64) as u32,
        )
    }

    /// Spec §4.3 render (pure function of position).
    pub fn render(px: u32, py: u32) -> Vec<f32> {
        let mut frame = vec![0.0f32; FRAME_LEN];
        for c in 0..C {
            for y in 0..H {
                for x in 0..W {
                    let in_sq = ((x as u32 + 64 - px) % 64) < 8 && ((y as u32 + 64 - py) % 64) < 8;
                    frame[idx(c, y, x)] = if in_sq {
                        1.0
                    } else {
                        match c {
                            0 => x as f32 / 63.0,
                            1 => y as f32 / 63.0,
                            _ => 0.25,
                        }
                    };
                }
            }
        }
        frame
    }
}

/// Pin the test-side reference math to the spec's hand-computed values
/// (spec §5.1, §5.2) so it cannot silently diverge from the spec.
#[test]
fn worldmodel_specref_selfcheck_against_spec_constants() {
    // §5.1 FNV chain over zero bytes.
    assert_eq!(specref::fnv1a(std::iter::repeat_n(0u8, 1)), 0xaf63bd4c8601b7df);
    assert_eq!(specref::fnv1a(std::iter::repeat_n(0u8, 2)), 0x08328807b4eb6fed);
    assert_eq!(specref::fnv1a(std::iter::repeat_n(0u8, 16)), 0x88201fb960ff6465);
    assert_eq!(specref::seed(&[], &[]), (37, 57));

    // §5.2 Case B byte stream and checkpoints.
    let s = specref::ctx_bytes(&[0.5], &[3]);
    assert_eq!(
        s,
        vec![
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // frames len
            0x00, 0x00, 0x00, 0x3F, // 0.5f32 bits LE
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // actions len
            0x03, 0x00, 0x00, 0x00, // action 3
        ]
    );
    assert_eq!(specref::fnv1a(s[..8].iter().copied()), 0x89cd31291d2aefa4);
    assert_eq!(specref::fnv1a(s[..12].iter().copied()), 0x5f245439c2426e29);
    assert_eq!(specref::fnv1a(s.iter().copied()), 0x339bdfb58ed81f5b);
    assert_eq!(specref::seed(&[0.5], &[3]), (27, 53));
}

// ---------------------------------------------------------------------------
// §9.1 Object safety
// ---------------------------------------------------------------------------

#[test]
fn worldmodel_object_safety_box_dyn() {
    // Must compile and be callable entirely through the trait object.
    let mut m: Box<dyn WorldModel> = Box::new(FakeWorldModel::new());
    assert_eq!(m.frame_shape(), (3, 64, 64), "frame_shape (spec §3)");
    assert_eq!(m.num_actions(), 5, "num_actions (spec §3)");
    m.set_nfe(4); // default no-op, callable through the box
    m.reset(&[0.5], &[3]);
    let frame = m.step(0);
    assert_eq!(frame.len(), FRAME_LEN, "step frame length C*H*W (spec §2)");

    // Also callable through &mut dyn.
    fn drive(m: &mut dyn WorldModel) -> Vec<f32> {
        m.reset(&[], &[]);
        m.step(0)
    }
    let frame2 = drive(m.as_mut());
    assert_eq!(frame2.len(), FRAME_LEN);
}

// ---------------------------------------------------------------------------
// §5.5 / §6.7 new() & Default start at (0, 0), stepable without reset
// ---------------------------------------------------------------------------

#[test]
fn worldmodel_new_and_default_start_at_origin() {
    let mut m = FakeWorldModel::new();
    let f = m.step(0); // noop: renders at (0, 0)
    assert_eq!(f.len(), FRAME_LEN);
    // §5.5 hand-computed probes.
    assert_eq!(f[8192].to_bits(), 1.0f32.to_bits(), "B at (0,0) in square");
    assert_eq!(f[8192 + 7 * 64 + 7].to_bits(), 1.0f32.to_bits(), "B(7,7): square is 0..=7 inclusive");
    assert_eq!(f[8192 + 8 * 64 + 8].to_bits(), 0.25f32.to_bits(), "B(8,8): first off-square diagonal pixel");
    assert_eq!(f[63].to_bits(), 1.0f32.to_bits(), "R at (0,63) = 63/63: bg may hit 1.0 in R");
    assert_eq!(recover_origin(&f), (0, 0));

    // Default is new() (spec §2).
    let mut d = FakeWorldModel::default();
    let fd = d.step(0);
    assert_frame_bits_eq(&f, &fd, "Default::default() vs new()");
}

// ---------------------------------------------------------------------------
// §9.3 Seed math: hand-computed Case A / Case B pixel probes
// ---------------------------------------------------------------------------

#[test]
fn worldmodel_case_a_pixel_probes() {
    // reset(&[], &[]) => (px, py) = (37, 57) (spec §5.1); step(0) keeps it.
    let mut m = FakeWorldModel::new();
    m.reset(&[], &[]);
    let f = m.step(0);
    assert_eq!(f.len(), FRAME_LEN);
    // §5.4 hand-computed values, bitwise.
    assert_eq!(f[11877].to_bits(), 1.0f32.to_bits(), "B at square origin (y=57,x=37)");
    assert_eq!(f[8236].to_bits(), 1.0f32.to_bits(), "B at (y=0,x=44): wrapped footprint row");
    assert_eq!(f[8300].to_bits(), 0.25f32.to_bits(), "B at (y=1,x=44): y-off 8 => bg");
    assert_eq!(f[21].to_bits(), (21.0f32 / 63.0f32).to_bits(), "bg R at (0,21)");
    assert_eq!(f[21].to_bits(), 0x3eaaaaab, "21/63 = nearest f32 to 1/3 (spec §5.4)");
    assert_eq!(f[4741].to_bits(), (10.0f32 / 63.0f32).to_bits(), "bg G at (y=10,x=5)");
    assert_eq!(f[0].to_bits(), 0.0f32.to_bits(), "bg R at x=0");
    assert_eq!(recover_origin(&f), (37, 57), "Case A seeded position");
}

#[test]
fn worldmodel_case_b_pixel_probes() {
    // reset(&[0.5], &[3]) => (px, py) = (27, 53) (spec §5.2).
    let mut m = FakeWorldModel::new();
    m.reset(&[0.5], &[3]);
    let f = m.step(0);
    assert_eq!(f[idx(2, 53, 27)].to_bits(), 1.0f32.to_bits(), "B at square origin (y=53,x=27)");
    assert_eq!(f[idx(2, 60, 34)].to_bits(), 1.0f32.to_bits(), "B at (y=60,x=34): offsets (7,7) in square");
    assert_eq!(f[idx(2, 53, 35)].to_bits(), 0.25f32.to_bits(), "B at (y=53,x=35): x-off 8 => bg");
    assert_eq!(f[idx(2, 61, 27)].to_bits(), 0.25f32.to_bits(), "B at (y=61,x=27): y-off 8 => bg");
    assert_eq!(f[idx(0, 0, 10)].to_bits(), (10.0f32 / 63.0f32).to_bits(), "bg R ramp");
    assert_eq!(f[idx(1, 10, 0)].to_bits(), (10.0f32 / 63.0f32).to_bits(), "bg G ramp");
    assert_eq!(recover_origin(&f), (27, 53), "Case B seeded position");
}

// ---------------------------------------------------------------------------
// §9.2 / §7 Determinism (bitwise, fixed inputs, twice-run)
// ---------------------------------------------------------------------------

#[test]
fn worldmodel_determinism_two_instances_bitwise() {
    // Same reset (Case B) + same action sequence on two fresh models =>
    // byte-identical frames (spec §7, §9.2).
    let actions = [4u32, 4, 1, 0, 3, 2, 2, 2];
    let mut m1 = FakeWorldModel::new();
    let mut m2 = FakeWorldModel::new();
    m1.reset(&[0.5], &[3]);
    m2.reset(&[0.5], &[3]);
    for (i, &a) in actions.iter().enumerate() {
        let f1 = m1.step(a);
        let f2 = m2.step(a);
        assert_frame_bits_eq(&f1, &f2, &format!("step {i} (action {a})"));
    }
}

#[test]
fn worldmodel_determinism_reset_twice_same_instance() {
    // reset twice on ONE instance re-produces the same first frame (§9.2),
    // even after wandering in between.
    let mut m = FakeWorldModel::new();
    m.reset(&[0.5], &[3]);
    let first = m.step(0);
    for a in [4u32, 4, 3, 1, 2] {
        let _ = m.step(a);
    }
    m.reset(&[0.5], &[3]);
    let again = m.step(0);
    assert_frame_bits_eq(&first, &again, "re-reset first frame");
}

// ---------------------------------------------------------------------------
// §9.4 Action consequence
// ---------------------------------------------------------------------------

#[test]
fn worldmodel_action_consequence_pairwise_distinct() {
    // From a common reset, one step with each action on 5 fresh instances:
    // all 5 frames pairwise different; each non-noop differs from noop.
    let frames: Vec<Vec<f32>> = (0..5u32)
        .map(|a| {
            let mut m = FakeWorldModel::new();
            m.reset(&[0.5], &[3]);
            m.step(a)
        })
        .collect();
    let noop = bits(&frames[0]);
    for a in 1..5 {
        assert_ne!(bits(&frames[a]), noop, "action {a} frame must differ from noop");
    }
    for i in 0..5 {
        for j in (i + 1)..5 {
            assert_ne!(
                bits(&frames[i]),
                bits(&frames[j]),
                "frames for actions {i} and {j} must differ"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// §9.5 Frame bounds & length (incl. wrapped positions)
// ---------------------------------------------------------------------------

#[test]
fn worldmodel_frame_bounds_and_length() {
    let check = |f: &[f32], what: &str| {
        assert_eq!(f.len(), FRAME_LEN, "{what}: frame length");
        for (i, &v) in f.iter().enumerate() {
            // Written to also reject NaN (any comparison with NaN is false).
            assert!(
                (0.0..=1.0).contains(&v),
                "{what}: frame[{i}] = {v} out of [0,1]"
            );
        }
    };
    // Walk from (0,0) across the left and top edges (both wrap immediately).
    let mut m = FakeWorldModel::new();
    for i in 0..70 {
        let a = if i % 2 == 0 { 2 } else { 1 }; // A then W
        let f = m.step(a);
        check(&f, &format!("origin walk step {i}"));
    }
    // Case A position (37, 57) wraps its footprint across the bottom edge.
    m.reset(&[], &[]);
    check(&m.step(0), "case A wrapped footprint");
    // And across the bottom-row boundary by movement.
    for i in 0..8 {
        check(&m.step(3), &format!("case A down step {i}"));
    }
}

// ---------------------------------------------------------------------------
// §5.3 / §9.6 Motion and wrap-around (blue-channel probes)
// ---------------------------------------------------------------------------

#[test]
fn worldmodel_motion_case_a_dpad_sequence() {
    // §5.3: (37,57) --noop--> (37,57) --D--> (38,57) --W--> (38,56).
    let mut m = FakeWorldModel::new();
    m.reset(&[], &[]);
    assert_eq!(recover_origin(&m.step(0)), (37, 57));
    assert_eq!(recover_origin(&m.step(4)), (38, 57));
    assert_eq!(recover_origin(&m.step(1)), (38, 56));
}

#[test]
fn worldmodel_wrap_down_across_bottom_edge() {
    // §5.3: fresh Case A, 7 x S: py = (57+7) mod 64 = 0 => (37, 0).
    let mut m = FakeWorldModel::new();
    m.reset(&[], &[]);
    let mut last = Vec::new();
    for _ in 0..7 {
        last = m.step(3);
    }
    assert_eq!(recover_origin(&last), (37, 0));
    assert_eq!(last[idx(2, 0, 37)].to_bits(), 1.0f32.to_bits(), "B at new origin (0,37)");
    assert_eq!(last[idx(2, 63, 37)].to_bits(), 0.25f32.to_bits(), "B at (63,37): y-off 63 => bg");
}

#[test]
fn worldmodel_wrap_left_across_zero() {
    // §5.3: fresh Case A, 38 x A: px = (37-38) mod 64 = 63 => (63, 57).
    let mut m = FakeWorldModel::new();
    m.reset(&[], &[]);
    let mut last = Vec::new();
    for _ in 0..38 {
        last = m.step(2);
    }
    assert_eq!(recover_origin(&last), (63, 57));
    // §6.4: a square at px=63 paints columns {63, 0..=6}.
    assert_eq!(last[idx(2, 57, 63)].to_bits(), 1.0f32.to_bits(), "B at (57,63)");
    assert_eq!(last[idx(2, 57, 6)].to_bits(), 1.0f32.to_bits(), "B at (57,6): wrapped column");
    assert_eq!(last[idx(2, 57, 7)].to_bits(), 0.25f32.to_bits(), "B at (57,7): x-off 8 => bg");
}

#[test]
fn worldmodel_wrap_up_from_origin() {
    // §5.3: from new() (0,0), W: py = -1 mod 64 = 63 => (0, 63).
    let mut m = FakeWorldModel::new();
    let f = m.step(1);
    assert_eq!(recover_origin(&f), (0, 63));
    assert_eq!(f[idx(2, 63, 0)].to_bits(), 1.0f32.to_bits(), "B at (63,0)");
    assert_eq!(f[idx(2, 6, 0)].to_bits(), 1.0f32.to_bits(), "B at (6,0): wrapped row");
    assert_eq!(f[idx(2, 7, 0)].to_bits(), 0.25f32.to_bits(), "B at (7,0): y-off 8 => bg");
}

#[test]
fn worldmodel_wrap_left_from_origin() {
    // §5.3: from new() (0,0), A: px = 63 => (63, 0).
    let mut m = FakeWorldModel::new();
    let f = m.step(2);
    assert_eq!(recover_origin(&f), (63, 0));
    assert_eq!(f[idx(2, 0, 63)].to_bits(), 1.0f32.to_bits(), "B at (0,63)");
    assert_eq!(f[idx(2, 0, 6)].to_bits(), 1.0f32.to_bits(), "B at (0,6): wrapped column");
    assert_eq!(f[idx(2, 0, 7)].to_bits(), 0.25f32.to_bits(), "B at (0,7): x-off 8 => bg");
}

// ---------------------------------------------------------------------------
// §9.7 / §6.1 Panic path
// ---------------------------------------------------------------------------

#[test]
#[should_panic]
fn worldmodel_step_action_5_panics() {
    let mut m = FakeWorldModel::new();
    let _ = m.step(5);
}

#[test]
fn worldmodel_step_invalid_action_panics_after_valid_step() {
    // Rigorous variant: prove the model works first, then require the panic
    // (a #[should_panic] alone would also pass if construction panicked).
    let mut m = FakeWorldModel::new();
    let f = m.step(0);
    assert_eq!(f.len(), FRAME_LEN);
    for bad in [5u32, 6, 64, u32::MAX] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.step(bad)));
        assert!(r.is_err(), "step({bad}) must panic (spec §6.1), never wrap or clamp");
    }
}

// ---------------------------------------------------------------------------
// §9.8 / §6.6 set_nfe inert
// ---------------------------------------------------------------------------

#[test]
fn worldmodel_set_nfe_inert() {
    let actions = [4u32, 1, 3, 2, 0];
    let run = |nfe: Option<u32>| -> Vec<Vec<u32>> {
        let mut m = FakeWorldModel::new();
        m.reset(&[0.5], &[3]);
        if let Some(n) = nfe {
            m.set_nfe(n);
        }
        actions
            .iter()
            .map(|&a| {
                let f = m.step(a);
                if let Some(n) = nfe {
                    m.set_nfe(n); // also mid-sequence
                }
                bits(&f)
            })
            .collect()
    };
    let none = run(None);
    assert_eq!(run(Some(1)), none, "set_nfe(1) must not change frames");
    assert_eq!(run(Some(64)), none, "set_nfe(64) must not change frames");
}

// ---------------------------------------------------------------------------
// §6.5 step returns a fresh, unaliased Vec
// ---------------------------------------------------------------------------

#[test]
fn worldmodel_step_returns_fresh_unaliased_frame() {
    let mut m = FakeWorldModel::new();
    m.reset(&[], &[]);
    let f1 = m.step(0);
    assert_eq!(f1.len(), FRAME_LEN);
    // Poison the returned buffer; a later step must be unaffected.
    let mut poisoned = m.step(0);
    assert_frame_bits_eq(&f1, &poisoned, "two noop frames at the same position");
    for v in poisoned.iter_mut() {
        *v = 9.0;
    }
    let f3 = m.step(0);
    assert_eq!(f3.len(), FRAME_LEN);
    assert_frame_bits_eq(&f1, &f3, "frame after mutating a previously returned buffer");
}

// ---------------------------------------------------------------------------
// §6.2 / §6.3 reset edge cases: ragged lengths, bit-pattern hashing
// ---------------------------------------------------------------------------

#[test]
fn worldmodel_reset_ragged_context_lengths() {
    // ctx_frames need not be a multiple of C*H*W; the fake only hashes bits.
    let cases: Vec<(Vec<f32>, Vec<u32>)> = vec![
        (vec![1.0, 2.0, 3.0], vec![0]),
        (vec![], vec![1, 2, 3]),
        (vec![0.25; 5], vec![]),
    ];
    for (frames, actions) in &cases {
        let mut m = FakeWorldModel::new();
        m.reset(frames, actions);
        let f = m.step(0);
        let expect = specref::seed(frames, actions);
        assert_eq!(
            recover_origin(&f),
            expect,
            "seed for ctx_frames={frames:?}, ctx_actions={actions:?}"
        );
    }
}

#[test]
fn worldmodel_reset_hashes_bit_patterns_with_domain_separation() {
    // §4.1 lengths-prepended domain separation:
    // ([], [7]) and ([f32::from_bits(7)], []) must seed differently.
    let seed_of = |frames: &[f32], actions: &[u32]| -> (u32, u32) {
        let mut m = FakeWorldModel::new();
        m.reset(frames, actions);
        recover_origin(&m.step(0))
    };
    let a = seed_of(&[], &[7]);
    let b = seed_of(&[f32::from_bits(7)], &[]);
    assert_eq!(a, specref::seed(&[], &[7]));
    assert_eq!(b, specref::seed(&[f32::from_bits(7)], &[]));
    assert_ne!(a, b, "domain separation: ([], [7]) vs ([bits 7], [])");

    // §6.3: -0.0 and +0.0 hash to different seeds (bit pattern, not value).
    let pz = seed_of(&[0.0], &[]);
    let nz = seed_of(&[-0.0], &[]);
    assert_eq!(pz, specref::seed(&[0.0], &[]));
    assert_eq!(nz, specref::seed(&[-0.0], &[]));
    assert_ne!(pz, nz, "-0.0 vs +0.0 must seed differently");

    // §6.3: NaN payloads are legal input and hash by bit pattern.
    let nan = f32::from_bits(0x7fc00001);
    assert!(nan.is_nan());
    assert_eq!(seed_of(&[nan], &[]), specref::seed(&[nan], &[]));

    // Multi-element mixed context.
    let frames = [1.0f32, 2.0];
    let actions = [0u32, 1, 4];
    assert_eq!(seed_of(&frames, &actions), specref::seed(&frames, &actions));
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

#[test]
fn worldmodel_property_action_inverses() {
    // W/S and A/D are inverses: [a, inv(a)] from any reset lands back on the
    // start position, so the second frame equals the [noop, noop] second
    // frame from the same reset — bitwise.
    let contexts: Vec<(Vec<f32>, Vec<u32>)> = vec![
        (vec![], vec![]),
        (vec![0.5], vec![3]),
        (vec![1.0, 2.0], vec![0, 1, 4]),
    ];
    for (frames, actions) in &contexts {
        for (a, inv) in [(1u32, 3u32), (3, 1), (2, 4), (4, 2)] {
            let mut m = FakeWorldModel::new();
            m.reset(frames, actions);
            let _ = m.step(a);
            let back = m.step(inv);

            let mut n = FakeWorldModel::new();
            n.reset(frames, actions);
            let _ = n.step(0);
            let stay = n.step(0);

            assert_frame_bits_eq(
                &back,
                &stay,
                &format!("inverse pair ({a},{inv}) from ctx {frames:?}/{actions:?}"),
            );
        }
    }
}

#[test]
fn worldmodel_property_square_recoverable_and_frames_injective() {
    // §4.3 distinguishability invariant: every frame has EXACTLY 64 blue-1.0
    // pixels forming one wrapped 8x8 square; distinct positions => distinct
    // frames. Walk a D/S diagonal visiting 80 distinct positions.
    let mut m = FakeWorldModel::new();
    let (mut px, mut py) = (0u32, 0u32); // new() state per spec §2
    let mut seen: Vec<((u32, u32), Vec<u32>)> = Vec::new();
    for i in 0..80 {
        let a = if i % 2 == 0 { 4u32 } else { 3u32 };
        let f = m.step(a);
        let (epx, epy) = specref::step_pos(px, py, a);
        px = epx;
        py = epy;
        let got = recover_origin(&f); // asserts the 64-pixel footprint shape
        assert_eq!(got, (px, py), "recovered origin at walk step {i}");
        seen.push(((px, py), bits(&f)));
    }
    for i in 0..seen.len() {
        for j in (i + 1)..seen.len() {
            assert_ne!(seen[i].0, seen[j].0, "walk positions must be distinct");
            assert_ne!(
                seen[i].1, seen[j].1,
                "frames at distinct positions {:?} and {:?} must differ",
                seen[i].0, seen[j].0
            );
        }
    }
}

#[test]
fn worldmodel_property_full_walk_matches_reference() {
    // Fixed-seed pseudo-random walk: every full frame must be bitwise equal
    // to the spec-§4 reference math, and a twice-run must be identical.
    let mut lcg: u64 = 0x243F6A8885A308D3; // fixed seed
    let mut next_action = move || -> u32 {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 33) % 5) as u32
    };
    let actions: Vec<u32> = (0..50).map(|_| next_action()).collect();

    let ctx_frames = [0.5f32];
    let ctx_actions = [3u32];
    let (mut px, mut py) = specref::seed(&ctx_frames, &ctx_actions);

    let mut m1 = FakeWorldModel::new();
    let mut m2 = FakeWorldModel::new();
    m1.reset(&ctx_frames, &ctx_actions);
    m2.reset(&ctx_frames, &ctx_actions);
    for (i, &a) in actions.iter().enumerate() {
        let f1 = m1.step(a);
        let f2 = m2.step(a);
        assert_frame_bits_eq(&f1, &f2, &format!("twice-run walk step {i}"));
        let (npx, npy) = specref::step_pos(px, py, a);
        px = npx;
        py = npy;
        let expect = specref::render(px, py);
        assert_frame_bits_eq(&f1, &expect, &format!("walk step {i} (action {a}) vs spec reference"));
    }
}
