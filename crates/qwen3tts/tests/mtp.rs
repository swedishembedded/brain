// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MTP code-predictor gradient check, plus the env-gated real-checkpoint
//! import + forward.
//!
//! The gradient check is the MTP's twin of `tests/talker.rs`'s
//! `talker_analytic_grads_match_finite_differences`, and it covers what that
//! one structurally cannot: the Talker's decoder IS a `qwen3::Qwen`, so its
//! check gates Qwen's own backward, whereas the MTP assembles the same shared
//! `model::block` builders into a different graph (a fixed
//! `num_code_groups`-long sequence of input EMBEDDINGS rather than token ids,
//! `num_code_groups - 1` separate per-position output heads rather than one
//! `lm_head`, and - on the 1.7B family - a `small_to_mtp_projection` on every
//! position's input). Both shapes are checked: `MtpConfig::tiny`
//! (`embedding_dim == d_model`, projection Identity, the 0.6B story) and
//! `MtpConfig::tiny_projected` (`embedding_dim != d_model`, a real projection
//! folded across every position, the 1.7B one).

use std::cell::RefCell;

use qwen3tts::{MtpConfig, MtpModel};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

/// Local newtype so we can implement the external `CheckModel` trait (orphan
/// rule). A `RefCell` rather than `tests/talker.rs`'s plain newtype because
/// three of the MTP's four parameter families - the per-residual `lm_head` and
/// `codec_embedding` tables and `small_to_mtp_projection` - are host
/// `Vec<f32>`s the served model owns outright (its forward reads them with
/// `model::hostmath`, not a device dispatch), so `write_weight` takes `&mut
/// self`. The interior mutability the checker's `&self` surface wants belongs
/// here, in the adapter, not bolted onto the model every served run pays for.
struct Checkable(RefCell<MtpModel>);

impl gradcheck::CheckModel for Checkable {
    fn param_names(&self) -> Vec<String> {
        self.0.borrow().param_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.0.borrow().read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.0.borrow_mut().write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.0.borrow().read_grad(name)
    }
    fn loss(&self) -> f32 {
        self.0.borrow_mut().forward()
    }
    fn zero_grads(&self) {
        self.0.borrow_mut().zero_grads();
    }
    fn backward(&self) {
        self.0.borrow_mut().backward();
    }
}

/// One frame's fixed training example on a freshly-initialised trainable MTP.
fn checkable(cfg: MtpConfig, seed: u64) -> Checkable {
    let e = cfg.embedding_dim as usize;
    let nres = cfg.num_code_groups as usize - 1;
    let vocab = cfg.vocab;
    let mut m = MtpModel::new_trainable_on(
        gpu_core::testgpu::dev(qwen3tts::mtp::TRAIN_PIPELINES),
        cfg,
        seed,
    );
    let mut rng = data::rng::Rng::new(seed ^ 0x5EED);
    let th: Vec<f32> = (0..e).map(|_| rng.next_gaussian() as f32 * 0.5).collect();
    let cb0: Vec<f32> = (0..e).map(|_| rng.next_gaussian() as f32 * 0.5).collect();
    // Distinct per codebook, so a head/position swap changes the loss.
    let targets: Vec<u32> = (0..nres).map(|i| ((i * 7 + 3) as u32) % vocab).collect();
    m.set_frame_batch(&th, &cb0, &targets);
    Checkable(RefCell::new(m))
}

/// Finite-difference step. **1e-3, not the workspace's usual 5e-3**, and the
/// difference is conditioning, not correctness.
///
/// A `num_code_groups`-long sequence of input embeddings is a much shorter,
/// much narrower graph than the Talker's token stream, and its RMSNorms
/// therefore run at a large gain: the residual stream's rms is set by the
/// input rows themselves, not by a trained residual, so a step that is
/// harmless on a real-width model is a several-percent perturbation of the
/// normalised input here. `small_to_mtp_projection.bias` is the extreme case
/// - one bias entry shifts the same channel of EVERY position at once, in
/// phase, which is the most curved direction the graph has.
///
/// The evidence this is a step-size question and not a wrong gradient: sweep
/// that bias entry by entry (`BRAIN_DEVICE=cpu`, `tiny_projected`, seed 7) and
/// the central difference walks monotonically onto the analytic value and
/// stays there, rather than settling somewhere else:
///
/// | entry | analytic | 1e-2  | 3e-3  | 1e-3  | 3e-4  | 1e-4  | 3e-5  |
/// |-------|----------|-------|-------|-------|-------|-------|-------|
/// | 0     | -6.5946  | +0.25 | -1.15 | -4.32 | -6.40 | -6.574| -6.596|
/// | 9     | +7.4344  | +3.36 | +6.06 | +7.13 | +7.41 | +7.432| +7.435|
/// | 14    | -7.9675  | -2.56 | -6.70 | -7.970| -7.974| -7.968| -7.971|
///
/// A wrong gradient converges onto a different number; this one converges onto
/// its own. Combined with the scale-preserving `small_to_mtp_projection` init
/// the synthetic builder now uses (see `mtp::MtpModel::synthetic_weights`),
/// 1e-3 sits inside the linear region for every tensor while staying far above
/// the fp32 cancellation floor. The pass/fail tolerance is unchanged from
/// every other model's gate.
const FD_EPS: f32 = 1e-3;

#[test]
fn mtp_analytic_grads_match_finite_differences() {
    if gpu_disabled() {
        return;
    }
    for (label, cfg) in [
        ("tiny", MtpConfig::tiny()),
        ("tiny_projected", MtpConfig::tiny_projected()),
    ] {
        let model = checkable(cfg, 7);
        let report = gradcheck::directional_check(&model, FD_EPS, 4, 0x7abc);
        println!("--- MTP gradient check: {label} ---");
        report.print();
        // fp32 directional finite differences on the software GPU: the
        // framework's proven combined tolerance (same as every other brain
        // model's gate, including the Talker's next door).
        let (atol, rtol) = (4e-3, 8e-2);
        let fails = report.failures(atol, rtol);
        assert!(
            fails.is_empty(),
            "MTP gradient check failed ({label}) for {:?}",
            fails
                .iter()
                .map(|c| (&c.param, c.abs_err, c.rel_err))
                .collect::<Vec<_>>()
        );
    }
}

/// `small_to_mtp_projection` is the one MTP parameter a reverse pass FOLDS
/// across the whole sequence - every one of the `num_code_groups` input rows
/// goes through the same weight and bias - and `directional_check`'s own doc
/// records that a contraction onto a random direction, best-of-`n_dirs`, is
/// measurably blind to a *partial* gradient error on exactly that shape (a
/// third of T5's `rel_bias` gradient went missing at `rel = 6.2e-4`).
/// Perturbing one entry at a time removes both effects, so the tensor that
/// only the 1.7B family even has - the one whose absence of any gate let a
/// real width-mismatch bug ship - gets the stronger check.
#[test]
fn mtp_projection_grads_match_elementwise_finite_differences() {
    if gpu_disabled() {
        return;
    }
    let model = checkable(MtpConfig::tiny_projected(), 7);
    for name in ["small_to_mtp_projection.bias", "small_to_mtp_projection.weight"] {
        // `elementwise_check`'s doc suggests a LARGER step than the
        // directional one, because a single-entry perturbation has no
        // `sqrt(numel)` amplification. That reasoning is about signal, and it
        // loses here to the conditioning `FD_EPS` documents: one bias entry
        // alone already shifts its channel in every position at once, which is
        // enough to leave the linear region (that entry's central difference
        // reads -2.56 at 1e-2 against an analytic -7.97). Same step, then.
        let report = gradcheck::elementwise_check(&model, name, FD_EPS);
        let fails = report.failures(4e-3, 8e-2);
        assert!(
            fails.is_empty(),
            "{name}: {} / {} entries outside tolerance, worst {:?}",
            fails.len(),
            report.checks.len(),
            fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).take(4).collect::<Vec<_>>()
        );
    }
}

/// A parameter family whose gradient comes back identically zero is the
/// signature of a missing dispatch, not of a small derivative - the shape the
/// directional check's `atol` floor cannot see. Everything here must be live
/// except the residual codebooks past the end of this sequence: at
/// `num_code_groups = 4` only `codec_embedding.0` and `.1` are ever fed as
/// INPUTS (positions 2 and 3), so `codec_embedding.2` - the table that would
/// embed codebook 3 at position 4, which does not exist - is legitimately
/// dead here and is exempted by name rather than by tolerance.
#[test]
fn every_mtp_parameter_family_receives_gradient() {
    if gpu_disabled() {
        return;
    }
    let cfg = MtpConfig::tiny_projected();
    let n_fed = cfg.num_code_groups as usize - 2;
    let model = checkable(cfg, 7);
    let dead = gradcheck::zero_grad_params(&model, |n| {
        !matches!(
            n.strip_prefix("codec_embedding.").and_then(|r| r.strip_suffix(".weight")).and_then(|i| i.parse::<usize>().ok()),
            Some(i) if i >= n_fed
        )
    });
    assert!(dead.is_empty(), "MTP parameters with an all-zero gradient: {dead:?}");
}

#[test]
fn mtp_real_import_and_forward() {
    let Ok(dir) = std::env::var("BRAIN_QWEN3TTS_CKPT") else {
        return brain_testutil::skip("set BRAIN_QWEN3TTS_CKPT to a real Qwen3-TTS checkpoint dir");
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
    }
    let out = std::env::temp_dir().join("brain_mtp_test.safetensors");
    let out = out.to_str().unwrap();
    qwen3tts::import::import_mtp(&dir, out).expect("mtp import");
    let m = MtpModel::load_inference_on(gpu_core::testgpu::dev(qwen3tts::mtp::PIPELINES), out);
    assert_eq!(m.cfg.vocab, 2048);
    assert_eq!(m.cfg.num_code_groups, 16);
    let d = m.cfg.d_model as usize;
    let th = vec![0.1f32; d];
    let cb0 = vec![-0.1f32; d];
    let residual: Vec<u32> = (0..(m.cfg.num_code_groups as usize - 2))
        .map(|i| (i * 7 % 2048) as u32)
        .collect();
    let embeds = m.assemble(&th, &cb0, &residual);
    let logits = m.logits(&embeds);
    assert_eq!(logits.len(), (m.cfg.num_code_groups as usize - 1) * 2048);
    assert!(logits.iter().all(|x| x.is_finite()));
    let _ = std::fs::remove_file(out);
}
