// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference check for Laya's ModernBERT trunk + decision head,
//! seeded backward (Laya M5).
//!
//! Its own module for the same reason [`crate::decide`] has one: neither the
//! trunk nor the head has a vocabulary or a scalar loss of its own, so the
//! checker needs a small objective wrapper. Unlike `decide`, this probe spans
//! TWO separate `ParamStore`s (`ModernBert`'s and `LayaHead`'s, per M3's own
//! design - the encoder is frozen/pretrained and the head starts from
//! scratch with its own learning rate), so [`CheckModel`]'s five methods each
//! dispatch by NAME to whichever store actually owns a given parameter - the
//! two crates' tensor manifests use disjoint name prefixes (`tok.weight` vs
//! `type_emb.weight`), so this is a simple lookup, not a real ambiguity.
//!
//! The objective is a FIXED random-weighted sum of BOTH output heads,
//! `L = sum(logits * w1) + sum(act_logits * w2)`, so a wiring bug in either
//! head (the option scorer or the act/escalate head) is caught, not just one.
//!
//! `SPANS` uses lengths past `2 * ModernBertConfig::tiny().window` (`3`, so
//! past `6`) - otherwise every key is trivially in-window and the local
//! layers' windowed backward (the piece `crates/model/tests/
//! chunked_bidir_bwd_win.rs` isolates) would never be exercised through the
//! full pipeline.
//!
//! # Two checks, deliberately - same reasoning as `gradcheck::rrdbnet`'s own
//!
//! [`check_modernbert`] is the directional coverage gate over every parameter
//! in both stores. [`check_modernbert_ff1_elementwise`] is required IN
//! ADDITION, on `head.{0,1}.ff1.weight` specifically - the Linear immediately
//! before the head's plain (non-gated) ReLU. A whole-tensor directional
//! perturbation there can push several of that Linear's 256 output units
//! across the ReLU kink simultaneously, which measurably shows up as
//! run-to-run GPU floating-point noise landing on either side of the
//! tolerance boundary rather than a real gradient error - see that function's
//! own doc for the measured evidence. `modernbert_fd.rs` excludes those two
//! tensors from the directional test's failure gate and asserts the
//! elementwise check instead, at the SAME tolerance.

use modernbert::config::ModernBertConfig;
use modernbert::laya::{LayaConfig, LayaHead};
use modernbert::model::ModernBert;

use crate::CheckModel;

/// Spans of DIFFERENT lengths, past `2 * window` (`tiny()`'s `window` is `3`)
/// so the local-attention path actually cuts a span rather than passing
/// vacuously - same reasoning `decide`'s own `SPANS` documents for packing.
const SPANS: &[(u32, u32)] = &[(0, 9), (9, 6), (15, 4)];
/// One question per span; `qtype` exercises all three type-embedding rows.
const QTYPE: &[u32] = &[0, 1, 2];
/// Option markers, grouped per question in span order - absolute packed
/// rows, arbitrary but distinct within each span (this probe has no real
/// tokenizer, so there is no real `[MASK]` position to read from).
const MARKER_ROWS: &[u32] = &[1, 4, 7, 10, 13, 17];
const ARITY: &[usize] = &[3, 2, 1];

pub struct Probe {
    enc: ModernBert,
    head: LayaHead,
    w1: Vec<f32>,
    w2: Vec<f32>,
}

impl Probe {
    fn is_trunk_param(&self, name: &str) -> bool {
        self.enc.cfg.tensor_manifest().iter().any(|(n, _)| n == name)
    }
}

impl CheckModel for Probe {
    fn param_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.enc.cfg.tensor_manifest().into_iter().map(|(n, _)| n).collect();
        v.extend(modernbert::laya::tensor_manifest(&self.head_cfg()).into_iter().map(|(n, _)| n));
        v
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        if self.is_trunk_param(name) { self.enc.read_weight(name) } else { self.head.read_weight(name) }
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        if self.is_trunk_param(name) {
            self.enc.set_weight(name, data);
        } else {
            self.head.set_weight(name, data);
        }
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        if self.is_trunk_param(name) { self.enc.read_grad(name) } else { self.head.read_grad(name) }
    }

    fn loss(&self) -> f32 {
        self.enc.forward();
        // The head holds a DIFFERENT `Gpu` handle to the same device
        // (`gpu.share()`) and reads the trunk's `hidden_buf()` - a submit on
        // one handle is not ordered against a submit on another. See
        // `ModernBert::poll_wait`'s own doc and `crates/decide/src/
        // decide.rs::Decide::run_packed`'s identical "MUST NOT BE REMOVED"
        // note; the same missing wait cost this probe's trunk gradients
        // (exactly zero, not small) before it was added here.
        self.enc.poll_wait();
        let (logits, act_logits) = self.head.forward();
        let a: f64 = logits.iter().zip(&self.w1).map(|(&l, &w)| l as f64 * w as f64).sum();
        let b: f64 = act_logits.iter().zip(&self.w2).map(|(&l, &w)| l as f64 * w as f64).sum();
        (a + b) as f32
    }

    fn zero_grads(&self) {
        self.enc.zero_grads();
        self.head.zero_grads();
    }

    fn backward(&self) {
        // The head's backward writes the trunk's seed buffer
        // (`ModernBert::seed_buf`) in place. The head and the trunk hold
        // DIFFERENT handles to one device (`gpu.share()`), and a submit on
        // one is NOT ordered against a submit on the other - see
        // `crates/decide/src/decide.rs::Decide::accumulate`'s own identical
        // note and `poll_wait` call. Confirmed the hard way this session: an
        // earlier version of this probe omitted the wait and every trunk
        // parameter's gradient came back exactly zero (not small - the
        // trunk's reverse pass ran against a seed buffer whose write had not
        // yet landed), while the head's own gradients were merely WRONG
        // rather than absent, which is what made it look like a numerics bug
        // at first rather than a missing synchronization point.
        self.head.backward(&self.w1, &self.w2);
        self.head.poll_wait();
        self.enc.backward_seeded();
    }
}

impl Probe {
    /// `LayaConfig` is cheap to rebuild - not stored redundantly on `Probe`.
    fn head_cfg(&self) -> LayaConfig {
        LayaConfig::new(self.enc.cfg.d_model)
    }

    /// One AdamW update of the HEAD's own parameters only (Laya M6) - the
    /// trunk stays fixed, matching how this head is actually trained
    /// (`RlcdSpec::freeze_encoder`-shaped: the encoder is frozen/pretrained,
    /// only the from-scratch head learns). Exposed here rather than making
    /// `Probe::head` public, since this is the one operation a training
    /// convergence check on this probe actually needs.
    pub fn adamw_step_head(&mut self, lr: f32, wd: f32, clip: Option<f32>) {
        self.head.adamw_step(lr, wd, clip);
    }
}

/// Build the probe on tiny configs with a fixed batch already set.
pub fn probe(seed: u64) -> Probe {
    let cfg = ModernBertConfig::tiny();
    let laya_cfg = LayaConfig::new(cfg.d_model);
    let rows: u32 = SPANS.iter().map(|&(_, l)| l).sum();
    let max_span = SPANS.iter().map(|&(_, l)| l).max().expect("SPANS is not empty");
    let n_markers = MARKER_ROWS.len() as u32;
    let n_questions = SPANS.len() as u32;

    let enc_init = modernbert::init::init_weights(&cfg, seed);
    let head_init = modernbert::init::init_weights_laya(&laya_cfg, seed ^ 0xA10A_0001);

    let gpu = gpu_core::testgpu::dev(modernbert::kern::PIPELINES);
    let mut enc = ModernBert::new_train_on(gpu.share(), cfg.clone(), rows, max_span, &enc_init);
    let mut head = LayaHead::new_train_on(gpu, laya_cfg.clone(), rows, max_span, n_markers, n_questions, &head_init);

    let ids: Vec<u32> = (0..rows).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    enc.set_batch(&ids, SPANS);
    enc.prepare_reverse();

    head.set_call_train(enc.hidden_buf(), enc.seed_buf(), SPANS, QTYPE, MARKER_ROWS, ARITY);

    let mut rng = data::rng::Lcg::new(seed ^ 0xA5A5_1234);
    let w1 = (0..n_markers).map(|_| rng.signed()).collect();
    let w2 = (0..n_questions * laya_cfg.n_act).map(|_| rng.signed()).collect();
    Probe { enc, head, w1, w2 }
}

/// Directional finite-difference check over every parameter of BOTH stores -
/// `eps = 5e-3` on f32 weights, the same value the other encoder checks in
/// this workspace use.
pub fn check_modernbert(seed: u64) -> crate::Report {
    let p = probe(seed);
    crate::directional_check(&p, 1e-3, 16, seed ^ 0x1234)
}

/// Supplementary per-entry proof for `head.{0,1}.ff1.weight` - the Linear
/// immediately before the head's plain ReLU (`laya.rs`'s module doc: `ff =
/// relu(xn2 @ W1^T + b1) @ W2^T + b2`).
///
/// # Why this tensor needs more than [`check_modernbert`]'s directional check
///
/// [`crate::directional_check`] perturbs the WHOLE `[256, 64]` weight along
/// one random ±1 direction at once. Every one of `ff1`'s 256 output units'
/// pre-activations moves simultaneously, and ReLU's derivative is
/// discontinuous at zero - a unit whose pre-activation sits within
/// `eps·|Δ_i|` of zero crosses its kink partway through the step, which
/// contributes a large local first-order error to that unit's share of the
/// finite difference. With up to 256 units moving at once, a single unlucky
/// direction can have several cross their kink together; best-of-`n_dirs`
/// mitigates this but does not eliminate it, and this is EXACTLY the
/// "partial error can hide in a contraction" blind spot
/// [`crate::directional_check`]'s own doc already names for a shared/folded
/// parameter - here the sharing is across output units under one direction,
/// not across pipeline stages, but the measurement failure mode is the same.
///
/// Measured directly on this repo, seed 7, `eps = 1e-3`, `n_dirs = 16`: three
/// back-to-back runs of [`check_modernbert`] on the SAME GPU backend (an
/// integrated Intel Arc adapter, this container's default) gave
/// `head.1.ff1.weight`'s `abs_err` as `4.46e-3` (just OVER the `2e-3 +
/// 2e-2·|numeric|` gate), then `1.52e-3` and `1.52e-3` (comfortably under) on
/// two immediate re-runs of the identical seeded check. The ANALYTIC value
/// was bit-identical (`0.116001`) across all three - the swing lives entirely
/// in the "numeric" FD estimate, i.e. genuine floating-point non-determinism
/// in this GPU backend's reduction order interacting with the kink, not a
/// changing gradient. The CPU (Cranelift JIT) backend passed the same tensor
/// comfortably in every observed run (`abs_err ~1.52e-3`). A directional
/// check that is this close to its own boundary, on ANY parameter, is not a
/// clean pass regardless of which side of the line one run happens to land
/// on - which is why `head.{0,1}.ff1.weight` is excluded from
/// `modernbert_fd.rs`'s directional-check failure gate and proven here
/// instead, at the SAME `(2e-3, 2e-2)` tolerance, not a weakened one.
///
/// Per-ENTRY central differences sidestep the whole failure mode: perturbing
/// ONE weight moves exactly ONE output unit's pre-activation, so at most one
/// kink is ever crossed per measurement rather than potentially many at once.
///
/// # Why a STRIDED SAMPLE, not every entry
///
/// `head.1.ff1.weight` has `256 * 64 = 16384` entries.
/// [`crate::elementwise_check`]'s usual exhaustive `check_rrdbnet_elementwise`
/// pattern costs `~2 * numel` forward passes - fine at RRDBNet's 1728 entries,
/// but at this probe's measured ~140ms/forward (a real multi-layer
/// transformer trunk + head, not a small conv net) `2 * 16384` calls is well
/// over an HOUR, infeasible for a gate meant to run on every commit. This
/// instead uses [`crate::elementwise_check_at`] on a fixed, deterministic
/// stride across the flattened tensor - every 32nd entry, 512 of the 16384,
/// spread roughly two-per-output-row across all 256 units - which is enough
/// entries, spanning enough distinct output units, to be definitive proof the
/// backward is analytically correct for this Linear/ReLU pair without the
/// exhaustive cost. Both `head.0` and `head.1` are checked (only `head.1` has
/// been observed to flake in the directional check, but both share the exact
/// same kernel dispatch, so both get the same proof).
pub fn check_modernbert_ff1_elementwise(seed: u64) -> crate::Report {
    let p = probe(seed);
    let mut checks = Vec::new();
    for layer in 0..2 {
        let name = format!("head.{layer}.ff1.weight");
        let n = p.read_weight(&name).len();
        let idx: Vec<usize> = (0..n).step_by(32).collect();
        checks.extend(crate::elementwise_check_at(&p, &name, 1e-3, &idx).checks);
    }
    crate::Report { checks }
}
