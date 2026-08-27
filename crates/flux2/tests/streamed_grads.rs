// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The host trainer's memory contract, and the proof that meeting it changed
//! nothing about the numbers.
//!
//! A LoRA step reduces every block's dense `dL/dW_eff` to a rank-`r`
//! projection and is finished with it, but `modelgrad::grads` used to hand the
//! trainer the whole stack at once - a third fp32 copy of the model, alongside
//! the frozen base and its `apply`ed effective copy. Measured on this box, one
//! klein-4b step at a 16x16 latent grid peaked at 49.6 GiB against a 14.4 GiB
//! model: three copies plus activations. `modelgrad::grads_into` streams each
//! block to a [`flux2::modelgrad::GradSink`] the moment it is complete, so the
//! trainer never holds more than one block's worth.
//!
//! Two things have to be true and both are gated here:
//!
//! 1. **The adapter is bit-identical.** The host trainer is the oracle
//!    `tests/dev_grad.rs`/`tests/device_train.rs` gate the device trainer
//!    against; a memory fix that moved it would invalidate that. Compared as
//!    raw bytes, not as a tolerance.
//! 2. **The peak is actually lower.** Asserted on the *whole process*
//!    (`VmHWM`), not on a heap counter: an allocation that a heap counter
//!    misses still costs the box its pages.

use flux2::grad::{DoubleGrads, SingleGrads};
use flux2::lora::{LoraAdapter, LoraCfg};
use flux2::modelgrad::{grads, grads_into, init_model, make_flow_batch, Batch, Cfg, GradSink, ModelWeights};

fn batch(c: &Cfg, sigma: f64, seed: u64) -> Batch<f32> {
    let x0 = model::hostmath::randn(c.n_img() * c.in_channels, seed);
    let ctx = model::hostmath::randn(c.txt_len * c.context_in_dim, seed ^ 0x11);
    let noise = model::hostmath::randn(x0.len(), seed ^ 0x22);
    make_flow_batch(c, &x0, &ctx, sigma, &noise)
}

/// Flatten an adapter to the exact bytes it would be saved as.
fn adapter_bits(a: &LoraAdapter) -> Vec<(String, Vec<usize>, Vec<u32>)> {
    a.to_tensors()
        .into_iter()
        .map(|(n, s, d)| (n, s, d.iter().map(|x| x.to_bits()).collect()))
        .collect()
}

/// A tiny-but-not-degenerate training config: every kind of targeted linear
/// present (double img/txt streams, single blocks, both `linear2` column
/// halves), at dims where a transposed or dropped slice cannot hide.
fn cfg() -> Cfg {
    Cfg { depth_double: 2, depth_single: 3, ..Cfg::tiny() }
}

/// The streamed projection must produce the SAME adapter as collecting the
/// whole `ModelGrads` and stepping from it - to the bit, over several steps so
/// a divergence in the Adam moments would compound rather than cancel.
#[test]
fn streaming_the_block_grads_steps_the_adapter_identically() {
    let c = cfg();
    let w: ModelWeights<f32> = init_model(&c, 0x51a7);
    let lc = LoraCfg::new(4);
    let (mut collected, mut streamed) = (LoraAdapter::new(&c, lc), LoraAdapter::new(&c, lc));
    let init = adapter_bits(&streamed);
    assert_eq!(adapter_bits(&collected), adapter_bits(&streamed), "same seed, same init");

    for step in 0..4u64 {
        let b = batch(&c, 0.2 + 0.15 * step as f64, 0xbeef + step);
        // reference: whole-model grads, then project
        let (l_ref, g) = grads(&c, &w, &b);
        collected.step(&g, 1e-3);
        // streamed: project each block as it completes, never hold the stack
        let mut s = streamed.stepper(1e-3);
        let (l_str, globals) = grads_into(&c, &w, &b, &mut s);

        assert_eq!(l_ref.to_bits(), l_str.to_bits(), "step {step}: loss must be the same forward");
        assert!(globals.dbl.is_empty() && globals.sgl.is_empty(), "streamed grads must not carry the block stack");
        assert_eq!(g.img_in, globals.img_in, "step {step}: the global grads are unchanged too");
        assert_eq!(g.mod_img, globals.mod_img, "step {step}");
        assert_eq!(
            adapter_bits(&collected),
            adapter_bits(&streamed),
            "step {step}: streamed projection drifted from the collected one"
        );
    }
    // Agreement alone is not enough: `step` is written in terms of `stepper`,
    // so a fault inside the projection would break both paths equally and
    // still compare equal (a NaN compares equal by bits). These two say the
    // shared path did something real.
    assert!(
        streamed.to_tensors().iter().all(|(_, _, d)| d.iter().all(|x| x.is_finite())),
        "the trained adapter must be finite"
    );
    assert_ne!(adapter_bits(&streamed), init, "four steps must have moved the adapter");
}

/// `backward`'s own output must be untouched: it is the collecting sink over
/// the same walk, so every block must still arrive, in index order, with its
/// `dx` intact (`gradcheck` and the block tests read these).
#[test]
fn the_collecting_backward_still_yields_every_block_in_order() {
    let c = cfg();
    let w: ModelWeights<f32> = init_model(&c, 0x2211);
    let b = batch(&c, 0.5, 7);
    let (_, g) = grads(&c, &w, &b);
    assert_eq!(g.dbl.len(), c.depth_double);
    assert_eq!(g.sgl.len(), c.depth_single);
    for (i, d) in g.dbl.iter().enumerate() {
        assert_eq!(d.dx.len(), c.n() * c.hidden, "double block {i} kept its dx");
    }
    for (i, s) in g.sgl.iter().enumerate() {
        assert_eq!(s.dx.len(), c.n() * c.hidden, "single block {i} kept its dx");
    }

    // The blocks arrive last-to-first and must be filed under their own index:
    // a sink that mixed two blocks up would still produce a full-length Vec.
    struct Order(Vec<usize>);
    impl GradSink<f32> for Order {
        fn double(&mut self, i: usize, _g: DoubleGrads<f32>) {
            self.0.push(i);
        }
        fn single(&mut self, i: usize, _g: SingleGrads<f32>) {
            self.0.push(100 + i);
        }
    }
    let mut o = Order(Vec::new());
    grads_into(&c, &w, &b, &mut o);
    assert_eq!(o.0, vec![102, 101, 100, 1, 0], "single stack in reverse, then the double stack in reverse");
}

/// `apply_into` is `apply` without the clone, so it must land on the same
/// bytes - the whole point of the trainer being allowed to use it.
#[test]
fn apply_into_matches_apply_bit_for_bit() {
    let c = cfg();
    let base: ModelWeights<f32> = init_model(&c, 0x3344);
    let mut ad = LoraAdapter::new(&c, LoraCfg::new(4));
    // B starts at zero, so step it first or `apply` is a no-op and would agree
    // with anything.
    let b = batch(&c, 0.4, 5);
    let (_, g) = grads(&c, &base, &b);
    ad.step(&g, 1e-2);

    let cloned = ad.apply(&base);
    let mut in_place = base.clone();
    ad.apply_into(&mut in_place);
    for (i, ((a, bb), o)) in cloned.dbl.iter().zip(&in_place.dbl).zip(&base.dbl).enumerate() {
        for (n, x, y, z) in [
            ("img.wq", &a.img.wq, &bb.img.wq, &o.img.wq),
            ("img.wk", &a.img.wk, &bb.img.wk, &o.img.wk),
            ("img.wv", &a.img.wv, &bb.img.wv, &o.img.wv),
            ("img.wo", &a.img.wo, &bb.img.wo, &o.img.wo),
            ("img.w1", &a.img.w1, &bb.img.w1, &o.img.w1),
            ("img.w3", &a.img.w3, &bb.img.w3, &o.img.w3),
            ("img.w2", &a.img.w2, &bb.img.w2, &o.img.w2),
            ("txt.wq", &a.txt.wq, &bb.txt.wq, &o.txt.wq),
            ("txt.w2", &a.txt.w2, &bb.txt.w2, &o.txt.w2),
        ] {
            assert_eq!(x, y, "double {i} {n}: in place differs from the cloned apply");
            // ...and both are a real delta, not a copy: a leaf `apply_into`
            // forgot would agree with `apply` (which now calls it) and pass on
            // equality alone.
            assert_ne!(x, z, "double {i} {n}: no delta was applied at all");
        }
    }
    for (i, ((a, bb), o)) in cloned.sgl.iter().zip(&in_place.sgl).zip(&base.sgl).enumerate() {
        for (n, x, y, z) in [
            ("wq", &a.wq, &bb.wq, &o.wq),
            ("wk", &a.wk, &bb.wk, &o.wk),
            ("wv", &a.wv, &bb.wv, &o.wv),
            ("w1", &a.w1, &bb.w1, &o.w1),
            ("w3", &a.w3, &bb.w3, &o.w3),
            ("wo_a", &a.wo_a, &bb.wo_a, &o.wo_a),
            ("wo_b", &a.wo_b, &bb.wo_b, &o.wo_b),
        ] {
            assert_eq!(x, y, "single {i} {n}: in place differs from the cloned apply");
            assert_ne!(x, z, "single {i} {n}: no delta was applied at all");
        }
    }
    // The frozen base itself must be untouched by either route.
    assert_eq!(base.dbl[0].img.wq, init_model::<f32>(&c, 0x3344).dbl[0].img.wq, "apply must not mutate the base");
}

/// The memory contract, on the whole process.
///
/// A host step holds the frozen base and its `apply`ed effective copy - two
/// copies of the model, inherent to a trainer that differentiates a dense
/// `W_eff`. It must NOT hold a third for the gradients. The model here is
/// sized so that one extra copy is hundreds of MB, far outside the noise of a
/// test binary, while a step still runs in seconds. The other tests in this
/// file share the process (libtest runs them in parallel) but all use
/// `Cfg::tiny`, so their footprint is kilobytes against this one's hundreds of
/// megabytes.
#[test]
fn a_host_step_costs_two_copies_of_the_model_not_three() {
    // ~284 MB of fp32 weights: big enough that a third copy is unmistakable.
    let c = Cfg {
        in_channels: 64,
        context_in_dim: 768,
        hidden: 768,
        n_heads: 6,
        depth_double: 2,
        depth_single: 4,
        mlp: 2304,
        txt_len: 64,
        lh: 8,
        lw: 8,
        axes_dim: [32, 32, 32, 32],
        rope_theta: 2000.0,
    };
    let base: ModelWeights<f32> = init_model(&c, 0x9911);
    // `param_bytes` is what `finetune::run` reports the host budget from, so
    // it is checked here against an independent walk of the same fields.
    let bytes = model_bytes(&base);
    assert_eq!(base.param_bytes() as u64, bytes, "param_bytes must count every weight, once");
    let b = batch(&c, 0.5, 13);
    let mut ad = LoraAdapter::new(&c, LoraCfg::new(8));

    let before = brain_testutil::rss_bytes();
    brain_testutil::reset_peak_rss();
    let w_eff = ad.apply(&base);
    let mut s = ad.stepper(1e-4);
    let (loss, _) = grads_into(&c, &w_eff, &b, &mut s);
    let peak = brain_testutil::peak_rss_bytes();
    drop(w_eff);
    let over = peak.saturating_sub(before);
    eprintln!(
        "model {:.0} MB, step peak over baseline {:.0} MB = {:.2} copies (loss {loss:.4})",
        bytes as f64 / 1e6,
        over as f64 / 1e6,
        over as f64 / bytes as f64
    );
    assert!(loss.is_finite(), "the step has to have actually run");
    if brain_testutil::peak_rss_bytes() == 0 {
        return; // no /proc: nothing to assert on, and a report must not fail a build
    }
    // One copy for `w_eff`, plus activations, the global grads and one block's
    // gradients - measured at about 1.55 copies. Collecting the whole stack
    // instead lands above 2.5. The bound sits between the two.
    assert!(
        over < 2 * bytes,
        "a host step allocated {over} bytes over baseline for a {bytes}-byte model - that is a second whole-model buffer, i.e. the block gradients are being collected again"
    );
    // ...and the measurement itself has to have seen the step. `w_eff` alone
    // is one whole copy, so anything much below that means the reset or the
    // baseline is wrong and the upper bound above is passing for free.
    assert!(
        over > 9 * bytes / 10,
        "a host step measured only {over} bytes over baseline for a {bytes}-byte model - the measurement is not seeing the effective-weight copy"
    );
}

fn model_bytes(w: &ModelWeights<f32>) -> u64 {
    let v = |x: &Vec<f32>| (x.len() * 4) as u64;
    let mut n = v(&w.img_in) + v(&w.txt_in) + v(&w.time_a) + v(&w.time_b);
    n += v(&w.mod_img) + v(&w.mod_txt) + v(&w.mod_single) + v(&w.final_adaln) + v(&w.final_w);
    for d in &w.dbl {
        for s in [&d.img, &d.txt] {
            n += v(&s.wq) + v(&s.wk) + v(&s.wv) + v(&s.nq) + v(&s.nk) + v(&s.wo) + v(&s.w1) + v(&s.w3) + v(&s.w2);
        }
    }
    for s in &w.sgl {
        n += v(&s.wq) + v(&s.wk) + v(&s.wv) + v(&s.nq) + v(&s.nk) + v(&s.w1) + v(&s.w3) + v(&s.wo_a) + v(&s.wo_b);
    }
    n
}
