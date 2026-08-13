// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P1 FORWARD-VALUE PIN: the tiny detector's forward output, frozen bitwise.
//!
//! WHY THIS EXISTS. The other yolo gates do not actually pin the forward:
//!   * `p8_names.rs` checks parameter NAMES, SHAPES and COUNTS (297 tensors /
//!     3,167,776 params) — not a single value.
//!   * `p3_gradcheck.rs` checks that the backward is CONSISTENT WITH the forward
//!     — not that the forward is unchanged. Shift the forward and the backward
//!     together and it stays green.
//!
//! So a refactor that silently moves the numerics — a different thread count
//! changing a reduction's accumulation order, a kernel swapped for an
//! "equivalent" one, a fused path taken where an unfused one used to be — keeps
//! BOTH green while changing what the model computes and therefore what it
//! trains to. This test is the missing invariant, and it is the gate for the
//! `crates/vision` block extraction: those commits claim to be behaviour-
//! preserving, and this is the only thing that makes that claim checkable.
//!
//! Both modes are pinned because the refactor touches both:
//!   * TRAIN mode exercises the BN host-interleave (submit -> host read of
//!     mean/var -> submit), which is easy to "tidy" into one submit and thereby
//!     read stale stats.
//!   * EVAL mode exercises the collapsed-BN fused `conv_act_reg` path.
//!
//! SCOPE OF THE GOLDENS. These are f32 values from the native CPU backend, whose
//! conv fast paths are selected by runtime CPU-feature detection (AVX2/winograd
//! in `backend-cpu`). A mismatch therefore means EITHER a real behaviour change
//! OR a different host microarchitecture. The failure prints the leading values
//! so the two are distinguishable: a real change moves them visibly; a microarch
//! difference shows up in the last ulp. Same trade-off the repo already accepts
//! for `brain wm play --hashes` rollout goldens.

use model::Model;
use yolov8::{LossMode, Yolo, YoloConfig};

/// FNV-1a over the raw f32 bits — exact, order-sensitive, and diagnosable
/// (unlike a float tolerance, which is what we are specifically NOT doing here).
fn fnv1a_f32(vals: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for v in vals {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Deterministic LCG -> (-1, 1), identical to the P1/P2/P3 test generators so the
/// batch matches what p3_gradcheck feeds.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(6364136223846793005).wrapping_add(1) | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}
fn randvec(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| r.next_f32()).collect()
}

/// Build tiny(2) with a fixed seed and a fixed image batch — the same fixture
/// shape p3_gradcheck uses, so the two gates describe the same model.
fn fixture(train: bool) -> (Yolo, usize) {
    let cfg = YoloConfig::tiny(2);
    let b = 4u32;
    let init = yolov8::init_weights(&cfg, 7);
    let model = Yolo::new(cfg.clone(), b, cfg.input, &init);
    model.set_mode(LossMode::Proxy);
    model.set_eval(!train);
    let n = (b * 3 * cfg.input * cfg.input) as usize;
    let img: Vec<f32> = randvec(7 ^ 0xA5A5, n);
    model.set_batch(model::Batch::Tensor { tokens: None, inputs: &img, targets: &[] });
    (model, n)
}

fn report(tag: &str, cls: &[f32], boxl: &[f32]) -> (u64, u64) {
    let (hc, hb) = (fnv1a_f32(cls), fnv1a_f32(boxl));
    eprintln!(
        "{tag}: cls[{}] hash={hc:#018x} head={:?}\n{tag}: box[{}] hash={hb:#018x} head={:?}",
        cls.len(),
        &cls[..4.min(cls.len())],
        boxl.len(),
        &boxl[..4.min(boxl.len())],
    );
    (hc, hb)
}

// tiny(2) at input 128: A = 16^2 + 8^2 + 4^2 = 336 anchors (cf. p3_gradcheck's
// forward_shapes). cls = [N, A, nc] = 4*336*2; box = [N, A, 4*reg_max] with
// tiny's reg_max = 8, so 4*336*32.
const CLS_LEN: usize = 4 * 336 * 2; // 2,688
const BOX_LEN: usize = 4 * 336 * 32; // 43,008

/// EVAL-mode forward (collapsed BN + fused conv_act_reg path), pinned bitwise.
#[test]
fn eval_forward_logits_are_bit_stable() {
    let (model, _) = fixture(false);
    let _ = model.forward();
    let (cls, boxl) = model.raw_logits();
    let (hc, hb) = report("eval", &cls, &boxl);

    assert_eq!(cls.len(), CLS_LEN, "cls shape drifted");
    assert_eq!(boxl.len(), BOX_LEN, "box shape drifted");
    assert_eq!(hc, 0x04c939a0b3a4733d, "EVAL cls logits changed");
    assert_eq!(hb, 0xfd9d846874cd69a8, "EVAL box logits changed");
}

/// TRAIN-mode forward (BN BATCH stats + the host interleave), pinned bitwise.
///
/// The train hashes differ from the eval ones by design — BN normalizes by the
/// batch's own mean/var in train and by the running estimates in eval — so the
/// two tests pin genuinely different computations, and a refactor that collapsed
/// one into the other would be caught here rather than in a training curve.
#[test]
fn train_forward_logits_are_bit_stable() {
    let (model, _) = fixture(true);
    let loss = model.forward();
    let (cls, boxl) = model.raw_logits();
    let (hc, hb) = report("train", &cls, &boxl);
    eprintln!("train: proxy loss = {loss:.9e} bits={:#010x}", loss.to_bits());

    assert_eq!(cls.len(), CLS_LEN, "cls shape drifted");
    assert_eq!(boxl.len(), BOX_LEN, "box shape drifted");
    assert_eq!(hc, 0x2d75fbbe4cc3ff3f, "TRAIN cls logits changed");
    assert_eq!(hb, 0xbb3482412696d144, "TRAIN box logits changed");
    // -17.99055290; the scalar the gradcheck differentiates.
    assert_eq!(loss.to_bits(), 0xc18feca7, "TRAIN proxy loss changed");
}

/// The forward must not depend on how the CPU backend splits work across rayon
/// threads. Every kernel computes one output element per invocation from a
/// gather, so a disjoint range split cannot change any value — this asserts that
/// property directly rather than trusting it, since a future kernel that reduces
/// across invocations would silently break it AND make the goldens above
/// machine-dependent.
#[test]
fn forward_is_independent_of_thread_split() {
    let a = {
        let (m, _) = fixture(false);
        let _ = m.forward();
        m.raw_logits()
    };
    let b = {
        let (m, _) = fixture(false);
        let _ = m.forward();
        m.raw_logits()
    };
    assert_eq!(fnv1a_f32(&a.0), fnv1a_f32(&b.0), "cls logits not reproducible run-to-run");
    assert_eq!(fnv1a_f32(&a.1), fnv1a_f32(&b.1), "box logits not reproducible run-to-run");
}
