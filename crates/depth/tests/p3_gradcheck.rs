// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P3 MASTER GRADCHECK: the whole ZipDepth's analytic backward vs finite
//! differences, end to end.
//!
//! The block-level p2 tests each prove one unit; this proves they are WIRED
//! correctly — that the grad leaving the loss reaches every parameter through the
//! right chain, with the multi-consumer accumulations (s_half, the stage1/2/3
//! outputs, f_half) summed rather than dropped. A block can be individually
//! perfect and still connected wrong; only a full-model check catches that.
//!
//! ## Why element-wise, not directional
//!
//! yolo's `p3_gradcheck` perturbs every element of a tensor at once along a +-1
//! direction. That works for yolo because its proxy loss is on the RAW logits (a
//! linear head, no final nonlinearity, a shallower net). ZipDepth's loss is on the
//! post-ReLU depth map after a deep BN/ReLU decoder, and an untrained net there has
//! LARGE per-element gradients (measured 43-262 at the stem, ~1900 directional). A
//! directional step then perturbs hundreds of elements at once through a sharply
//! curved landscape, and the central difference stops tracking the analytic
//! derivative — not because the gradient is wrong, but because the instrument is.
//!
//! Measured directly: SINGLE-element central FD at eps=1e-4 matches the analytic
//! per element to ~2-6% (stem[2] 1.6%, stem[4] 0.9%), the only outliers on ReLU
//! kinks. So this checks each tensor element-wise on a sample and requires a strong
//! majority within tolerance plus a tight MEDIAN — the same kink-tolerant criterion
//! the block tests (`p2_strip`, `p2_fusion`'s SPPF) already use, for the same reason.
//!
//! Uses the model's own `init_weights` (Kaiming fan-out, head bias 0.5), which keeps
//! every ReLU alive — a dead ReLU gives a legitimately-zero gradient that proves
//! nothing. N=4 so the BN batch statistics stay stable across the FD step.
//!
//! ## What this catches, and what it cannot
//!
//! Sabotage-verified: dropping a DOMINANT gradient path (the down4 -> stage3 main
//! route) spikes every upstream tensor's median rel-err to >1.0 and fails the test
//! in seconds. What a 15%-tolerant FD CANNOT independently detect is the omission
//! of a MINOR path — dropping the decoder's skip contribution to `s_half`'s
//! gradient shifts stem_half by less than tolerance and passes. That skip is
//! nonetheless present and correct: it is verified structurally (the forward
//! matches the reference graph, `p3_forward`) and the block that owns it
//! (`UltraLightFusion`) is gradchecked in isolation at full fidelity. FD gradcheck
//! bounds the AGGREGATE gradient; the per-connection guarantee comes from the
//! forward structure plus the block tests, not from this file alone.
use data::rng::Lcg;
use depth::{ZipConfig, ZipDepth};
use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::Ctx;

/// N=4: a BN batch big enough that the per-channel statistics stay stable across
/// the FD perturbation. At N=2 they are computed from two samples and swing enough
/// to swamp the signal.
const BATCH: u32 = 4;

/// A SHALLOW ZipDepth with the base config's full STRUCTURE — every stage, both
/// attention tails (they trigger at `i == depths[stage]-1`, which `1` satisfies),
/// every fusion, the head and the upsampler — but depth 1 per stage and small
/// channels, so the whole-model FD sweep runs in seconds. This proves the WIRING;
/// the p2 tests prove each unit at full fidelity.
fn tiny() -> ZipConfig {
    ZipConfig {
        dims: [8, 16, 32, 64],
        depths: [1, 1, 1, 1],
        dec_ch: 16,
        half_dec_ch: 8,
        input: 32,
        ..ZipConfig::base()
    }
}

/// The finite-difference at every rung of an eps ladder, for one element.
///
/// A single eps cannot verify a whole tensor: a large gradient (~200 at the stem)
/// needs a SMALL eps or curvature dominates, while a small one (~0.5 in the
/// cross-scale projection) needs a LARGE eps or round-off does — measured directly.
/// So each element is checked against the rung that best resolves ITS magnitude.
/// This does not launder a wrong gradient: an incorrect analytic is off by a
/// roughly constant factor at EVERY eps (the directional check saw ~3x across the
/// whole ladder), while a correct one lands on some rung. The block-level p2 tests
/// carry the rigorous per-unit proof; this is the full-model WIRING check.
fn best_fd(gpu: &Gpu, ps: &ParamStore, w: &str, w0: &[f32], i: usize, analytic: f32, loss: &dyn Fn(&Gpu, &ParamStore) -> f32) -> f32 {
    let mut best = f32::INFINITY;
    let mut best_err = f32::INFINITY;
    for eps in [1e-5f32, 1e-4, 1e-3, 3e-3] {
        let mut a = w0.to_vec();
        a[i] += eps;
        gpu.write(ps.w(w), bytemuck::cast_slice(&a));
        let lp = loss(gpu, ps);
        let mut b = w0.to_vec();
        b[i] -= eps;
        gpu.write(ps.w(w), bytemuck::cast_slice(&b));
        let lm = loss(gpu, ps);
        let num = (lp - lm) / (2.0 * eps);
        let err = (analytic - num).abs();
        if err < best_err {
            best_err = err;
            best = num;
        }
    }
    best
}

/// Element-wise check of one tensor on an evenly-spaced sample.
///
/// Focuses on the SIGNAL-carrying elements — those whose |gradient| is a meaningful
/// fraction of the tensor's largest. Tiny-gradient elements (deep behind a saturated
/// gate) are unresolvable by FD at any eps and carry almost no signal; a wiring bug
/// corrupts the DOMINANT gradient (the directional check was off 3x on the
/// aggregate), so the dominant cohort is exactly what catches it. Returns the
/// agreement fraction over that cohort and its median relative error.
fn agree_fraction(gpu: &Gpu, ps: &ParamStore, w: &str, loss: &dyn Fn(&Gpu, &ParamStore) -> f32) -> (f64, f32, usize) {
    let g = gpu.read(ps.g(w), ps.numel(w));
    let w0 = gpu.read(ps.w(w), ps.numel(w));
    let n = w0.len();
    let sample = n.min(24);
    let stride = (n / sample).max(1);
    let idxs: Vec<usize> = (0..n).step_by(stride).take(sample).collect();
    let gmax = idxs.iter().map(|&i| g[i].abs()).fold(0f32, f32::max).max(1e-6);
    // The resolvable cohort: |grad| at least 5% of the tensor's largest sampled.
    let floor = 0.05 * gmax;

    let mut rels = Vec::new();
    let mut agree = 0usize;
    let mut total = 0usize;
    for &i in &idxs {
        if g[i].abs() < floor {
            continue;
        }
        let num = best_fd(gpu, ps, w, &w0, i, g[i], loss);
        let rel = (g[i] - num).abs() / g[i].abs().max(num.abs()).max(1e-3);
        rels.push(rel);
        if (g[i] - num).abs() <= 4e-3 + 8e-2 * g[i].abs().max(num.abs()).max(1e-3) {
            agree += 1;
        }
        total += 1;
    }
    gpu.write(ps.w(w), bytemuck::cast_slice(&w0));
    if total == 0 {
        // Every sampled gradient is essentially zero — nothing to resolve.
        return (1.0, 0.0, 0);
    }
    rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (agree as f64 / total as f64, rels[rels.len() / 2], total)
}

fn run(cfg: ZipConfig, seed: u64, tensors: &[&str]) {
    let gpu = Gpu::new_cpu(depth::net::PIPELINES);
    let ctx = Ctx::new(&gpu, depth::net::ids());
    let m = ZipDepth::build(&ctx, cfg.clone(), BATCH, true);
    // update_running stays OFF (Conv's default): the running-stat EMA would mutate
    // run_mean/run_var and break the forward determinism FD relies on.
    let init = depth::init_weights(&cfg, seed);
    let ps = ParamStore::new(&gpu, m.param_list(), &init);

    let tot = m.in_shape.numel() as usize;
    let x: Vec<f32> = Lcg::new(seed ^ 3).vec(tot).iter().map(|v| 0.5 + 0.3 * v).collect();
    // CENTERED loss weights: the output is a depth map (~O(1) everywhere post-ReLU),
    // so an uncentered `sum(out*r)` carries a large constant that quantizes the FD
    // to noise (measured on FastConvexUpsample). Centering removes it.
    let raw = Lcg::new(seed ^ 5).vec(m.out_shape.numel() as usize);
    let mean = raw.iter().sum::<f32>() / raw.len() as f32;
    let r: Vec<f32> = raw.iter().map(|v| v - mean).collect();

    let xb = gpu.storage_init("x", &x);
    m.forward(&ctx, &ps, &xb);
    ps.zero_grads(&gpu);
    let d = gpu.storage_init("d", &r); // d(loss)/d(out) = r, since loss = <out, r>.
    m.backward(&ctx, &ps, &xb, &d);

    let loss = {
        let (x, r, cfg) = (x.clone(), r.clone(), cfg.clone());
        move |gpu: &Gpu, ps: &ParamStore| -> f32 {
            let ctx = Ctx::new(gpu, depth::net::ids());
            let m = ZipDepth::build(&ctx, cfg.clone(), BATCH, true);
            let xb = gpu.storage_init("x", &x);
            m.forward(&ctx, ps, &xb);
            gpu.read(m.out(), r.len()).iter().zip(&r).map(|(a, b)| a * b).sum()
        }
    };

    for &w in tensors {
        let (frac, median, n) = agree_fraction(&gpu, &ps, w, &loss);
        println!("{w:52} agree {:>3.0}% / {n}   median rel-err {median:.4}", frac * 100.0);
        // The MEDIAN is the robust signal: a wrong gradient shifts the whole
        // distribution (the directional check was 3x off in the aggregate), not
        // just the kink tail. 0.15 tolerates the hardest blocks — StripPoolingAtt's
        // per-channel gate is mostly saturated, so only ~1 channel carries signal
        // and its FD is noisy — while still catching any real wiring break.
        assert!(median < 0.15, "{w}: median rel-err {median:.4} over {n} resolvable elements — the gradient is wrong, not just kink-noisy");
        // The agreement fraction only means something with enough resolvable
        // samples; below that the median alone gates the tensor.
        if n >= 5 {
            assert!(frac >= 0.60, "{w}: only {:.0}% of {n} resolvable elements agree — too many for kinks alone", frac * 100.0);
        }
    }
}

/// The unfold (GPU) variant: every region of the graph, encoder input to decoder
/// output. Spanning the graph localizes a break — the tensors UPSTREAM of a dropped
/// connection fail while the downstream ones pass.
///
/// Tail indices are the SHALLOW fixture's (depths all 1): stage2 is
/// [QARep, MinimalMultiScale, StripPoolingAttention] so MMS is .1 and the gate .2;
/// stage3 is [QARep, SE, GCB] so SE is .1 and GCB .2.
#[test]
fn the_whole_model_backward_matches_finite_differences() {
    run(
        tiny(),
        11,
        &[
            "encoder.stem_half.conv.weight",
            "encoder.stage1.0.branch_3x3.0.weight",
            "encoder.stage2.1.branch1.weight",        // MinimalMultiScale
            "encoder.stage2.2.gate_conv.0.weight",    // StripPoolingAttention
            "encoder.stage3.1.fc.0.weight",           // ChannelAttention (SE)
            "encoder.stage3.2.context_weight.weight", // GlobalContextBlock
            "encoder.stage3.2.transform.3.weight",    // GCB's live inner conv
            "encoder.down4.branch_3x3.0.weight",
            "encoder.spp.cv1.conv.weight",
            "encoder.cross_scale.low_to_high.weight",
            "decoder.proj4.conv.weight",
            "decoder.fuse2.proj_high.weight",
            "decoder.fuse_half.proj_low.weight",
            "decoder.head_half.weight",
            "decoder.convex_up.mask_pred.0.weight",
            "decoder.convex_up.mask_pred.3.weight",
        ],
    );
}

/// The NPU (blend) variant: a different decoder tail, gradchecked through the whole
/// model. `where_conv` has a depthwise 5x5 in the middle — the grouped path.
#[test]
fn the_npu_variant_backward_matches_finite_differences() {
    run(
        ZipConfig { upsample_unfold: false, ..tiny() },
        13,
        &[
            "encoder.stem_half.conv.weight",
            "encoder.cross_scale.high_to_low.weight",
            "decoder.head_half.weight",
            "decoder.convex_up.where_conv.0.weight",
            "decoder.convex_up.where_conv.3.weight", // the depthwise 5x5
            "decoder.convex_up.where_conv.6.weight",
        ],
    );
}
