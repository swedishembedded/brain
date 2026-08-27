// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every shipped FLUX.2 variant's REAL widths, through the same trainer a real
//! run uses.
//!
//! The existing device gates (`dev_grad.rs`, `device_train.rs`) prove the
//! *math*: they run at a tiny synthetic topology (hidden 64, 4 heads, mlp 192)
//! chosen so every sliced binding offset lands on the storage alignment. What
//! they cannot see is the thing that separates klein-4B from klein-9B, because
//! the two share every line of code and differ only in numbers: hidden
//! 3072→4096, heads 24→32, SwiGLU inner 9216→12288, conditioning width
//! 7680→12288, depths 5/20→8/24. A kernel that tiles, packs heads, or slices a
//! fused linear correctly at hidden 64 can still be wrong at 4096.
//!
//! So this file gates the two things a "klein-9B is wired" claim needs:
//!
//! 1. [`shapes_are_derived_from_the_variant_not_assumed`] - CPU, full depth,
//!    no weights. Every trainable shape the run builds is a function of the
//!    variant, and the two variants really do differ in each of them. This is
//!    the anti-hardcode check: it fails if anything downstream of
//!    `Flux2Config` starts assuming 4B's numbers.
//! 2. `device_matches_host_at_{klein_4b,klein_9b}_widths` - GPU, real widths,
//!    reduced depth. The WGSL trainer's adapter gradients against the
//!    FD-gradchecked host reference at the shipped hidden/heads/mlp/context, at
//!    cosine AND rel_l2 (cosine alone is not a gate here: an epsilon mutation
//!    scores cosine 1.000000000 - see `dev_grad.rs`). Only the two depths and
//!    the token counts are cut, because those are loop bounds: a stack of N
//!    blocks is the same block N times, and `depth_double`/`depth_single` are
//!    read straight out of the config by code this test does exercise.
//!
//! Swedish Embedded AB implements validated multi-scale GPU training kernels
//! for its clients. If your team needs expertise in gradient-parity gating
//! across model configurations, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! The device tests are `#[ignore]`d: each wants a card and several GiB of host
//! memory, which is a lot to take from a shared box. Run them with
//!
//! ```text
//! BRAIN_DEV_GPU=1 BRAIN_GPU_INDEX=<i> cargo test -p brain-flux2 --release \
//!     --test variant_widths -- --ignored --nocapture
//! ```

use flux2::devtrain::DeviceTrainer;
use flux2::lora::{LoraAdapter, LoraCfg};
use flux2::modelgrad::{self, Cfg, ModelWeights};
use flux2::Flux2Config;

const RANK: usize = 8;

/// The two variants that carry weights. `base-*` differ from `klein-*` only in
/// the sampling recipe (`distilled`), which no training shape reads.
const VARIANTS: [&str; 2] = ["klein-4b", "klein-9b"];

/// One variant's real per-block widths at reduced depth and a small token grid.
///
/// `hidden`, `n_heads` (hence `head_dim`), `mlp`, `in_channels`,
/// `context_in_dim` and `axes_dim` are the SHIPPED values - those are the
/// numbers every kernel shapes a dispatch from. `depth_double`, `depth_single`
/// and the token counts are cut: they set how many times the stack runs, not
/// what one run of it computes.
fn width_cfg(variant: &str) -> Cfg {
    let fc = Flux2Config::from_name(variant).expect("variant");
    Cfg { depth_double: 1, depth_single: 1, txt_len: 64, ..Cfg::from_flux2(&fc, 4, 4) }
}

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    }
}

fn vof(n: usize, r: &mut impl FnMut() -> f32, s: f32) -> Vec<f32> {
    (0..n).map(|_| r() * s).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-300)
}

fn rel_l2(dev: &[f32], host: &[f32]) -> f64 {
    let nh: f64 = host.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let diff: f64 = dev.iter().zip(host).map(|(&a, &b)| ((a - b) as f64) * ((a - b) as f64)).sum::<f64>().sqrt();
    diff / nh.max(1e-12)
}

fn skip() -> bool {
    if std::env::var("BRAIN_DEV_GPU").as_deref() != Ok("1") {
        brain_testutil::skip_unavailable("set BRAIN_DEV_GPU=1 (needs a GPU) for the FLUX.2 variant-width gates");
        return true;
    }
    false
}

// --------------------------------------------------------------------------
// 1. Shapes come from the variant (CPU, full depth, no weights)
// --------------------------------------------------------------------------

/// The closed form for the adapter's trainable parameter count: every targeted
/// linear contributes `r * (out + in)`, and the seven leaves per stream are
/// four `d x d`, two `mlp x d` and one `d x mlp`.
fn expected_lora_params(c: &Cfg, r: usize) -> usize {
    let (d, mlp) = (c.hidden, c.mlp);
    let stream = r * (4 * (d + d) + 2 * (mlp + d) + (d + mlp));
    // A single block has the same seven shapes, split differently across the
    // two fused linears (3 qkv + 2 mlp-in from `linear1`, 2 column halves of
    // `linear2`): four `d x d`, two `mlp x d`, one `d x mlp`.
    let single = r * (4 * (d + d) + 2 * (mlp + d) + (d + mlp));
    2 * c.depth_double * stream + c.depth_single * single
}

/// Everything that must hold of ONE variant's derived training shapes.
///
/// Returns the trainable parameter count, or the FIRST violation as an error
/// string. It returns rather than asserts so that a negative test can feed it
/// a deliberately wrong config and check it is rejected - which is how this
/// file proves its own assertions bite. Mutating `config.rs` proves the same
/// thing, but only for as long as the mutation is in the tree, and it leaves a
/// window in which every other build in this workspace compiles a wrong
/// klein-9B.
fn check_shapes(name: &str, fc: &Flux2Config, c: &Cfg) -> Result<usize, String> {
    let eq = |what: &str, got: usize, want: usize| -> Result<(), String> {
        (got == want).then_some(()).ok_or_else(|| format!("{name}: {what} is {got}, config says {want}"))
    };
    eq("hidden", c.hidden, fc.hidden)?;
    eq("n_heads", c.n_heads, fc.n_heads)?;
    eq("mlp", c.mlp, fc.mlp_hidden())?;
    eq("context_in_dim", c.context_in_dim, fc.context_in_dim)?;
    eq("depth_double", c.depth_double, fc.depth_double)?;
    eq("depth_single", c.depth_single, fc.depth_single)?;
    eq("head_dim", c.head_dim(), fc.head_dim())?;
    // The 4-axis interleaved RoPE has to tile the head exactly, or the
    // rotation runs off the end of the head at a width nobody tested.
    eq("sum(axes_dim)", c.axes_dim.iter().sum::<usize>(), c.head_dim())?;
    if !c.hidden.is_multiple_of(c.n_heads) {
        return Err(format!("{name}: hidden {} does not divide into {} heads", c.hidden, c.n_heads));
    }

    // The conditioning width is three concatenated Qwen3 hidden states, and
    // `pipeline::build_text_encoder` picks WHICH Qwen from exactly this
    // number. That branch is the one magic constant on the path, so gate its
    // premise rather than trusting it.
    let te = if c.context_in_dim == 12288 { qwen3::QwenConfig::qwen3_8b() } else { qwen3::QwenConfig::qwen3_4b() };
    eq("context_in_dim vs 3x the selected encoder's hidden", c.context_in_dim, 3 * te.d_model as usize)?;
    for t in [9usize, 18, 27] {
        if t >= te.n_layers as usize {
            return Err(format!("{name}: tap layer {t} past the encoder's {} layers", te.n_layers));
        }
    }

    // The adapter must be built to the variant's widths.
    let ad = LoraAdapter::new(c, LoraCfg::new(RANK));
    let pairs = ad.pairs();
    eq("adapter pair count", pairs.len(), 7 * (2 * c.depth_double + c.depth_single))?;
    let mut params = 0usize;
    for p in &pairs {
        eq("rank", p.r, RANK)?;
        // The three shapes a FLUX.2 LoRA target can have, written as shapes
        // rather than as a conjunction so the set is readable.
        let shape = (p.out, p.inn);
        if !(shape == (c.hidden, c.hidden) || shape == (c.mlp, c.hidden) || shape == (c.hidden, c.mlp)) {
            return Err(format!("{name}: adapter pair {}x{} matches none of the variant's linear shapes (d {}, mlp {})", p.out, p.inn, c.hidden, c.mlp));
        }
        params += p.r * (p.out + p.inn);
    }
    eq("trainable parameter count", params, expected_lora_params(c, RANK))?;
    Ok(params)
}

/// What must DIFFER between the two variants. The anti-hardcode half: any of
/// these comparing equal means something quietly adopted 4B's number for 9B,
/// and `check_shapes` alone would not notice, because a consistently-wrong
/// config is still self-consistent.
fn check_variants_differ(c4: &Cfg, c9: &Cfg) -> Result<(), String> {
    let ne = |what: &str, a: usize, b: usize| -> Result<(), String> {
        (a != b).then_some(()).ok_or_else(|| format!("the two variants must differ in {what}, both are {a}"))
    };
    ne("hidden", c4.hidden, c9.hidden)?;
    ne("n_heads", c4.n_heads, c9.n_heads)?;
    ne("mlp", c4.mlp, c9.mlp)?;
    ne("context_in_dim", c4.context_in_dim, c9.context_in_dim)?;
    ne("depth_double", c4.depth_double, c9.depth_double)?;
    ne("depth_single", c4.depth_single, c9.depth_single)?;
    if expected_lora_params(c9, RANK) <= expected_lora_params(c4, RANK) {
        return Err("klein-9b's adapter must be the larger of the two".into());
    }
    Ok(())
}

/// Full shipped depth at a 64x64 latent grid (1024x1024). Costs nothing
/// without weights, so there is no reason to reduce it here.
fn full_cfg(variant: &str) -> (Flux2Config, Cfg) {
    let fc = Flux2Config::from_name(variant).expect("variant");
    let c = Cfg::from_flux2(&fc, 64, 64);
    (fc, c)
}

#[test]
fn shapes_are_derived_from_the_variant_not_assumed() {
    let mut cfgs = Vec::new();
    for name in VARIANTS {
        let (fc, c) = full_cfg(name);
        let params = check_shapes(name, &fc, &c).unwrap_or_else(|e| panic!("{e}"));
        eprintln!(
            "{name}: hidden {} heads {} head_dim {} mlp {} ctx {} depth {}+{} -> {} adapter pairs, {params} trainable params at rank {RANK}",
            c.hidden,
            c.n_heads,
            c.head_dim(),
            c.mlp,
            c.context_in_dim,
            c.depth_double,
            c.depth_single,
            7 * (2 * c.depth_double + c.depth_single)
        );
        cfgs.push(c);
    }
    check_variants_differ(&cfgs[0], &cfgs[1]).unwrap_or_else(|e| panic!("{e}"));
}

/// The negative half: a klein-9B that kept the 4B conditioning width.
///
/// This is not a hypothetical. It is one number copied from the sibling
/// constructor, and it leaves the tensor COUNT unchanged - only `txt_in`'s
/// shape moves - so it was measured to pass both of `config.rs`'s own tests,
/// `derived_dims` and `manifest_counts_match_the_reference_checkpoints`.
///
/// It also passes [`check_shapes`], and that is the point worth writing down
/// rather than papering over: with `context_in_dim` 7680 the `== 12288` branch
/// selects Qwen3-4B, whose hidden is 2560, and 3 x 2560 IS 7680. The config is
/// then internally consistent - every per-variant invariant still holds of it.
/// Nothing about a DiT hidden of 4096 *mathematically* requires an 8B encoder;
/// that pairing is a fact about the released checkpoints, not an identity, so
/// no single-variant check can derive it. Inventing a threshold that happened
/// to separate the two would be fitting a rule to the answer.
///
/// What does catch it is the CROSS-variant check: klein-9B carrying 4B's
/// conditioning width collides with klein-4B, and two variants that agree on a
/// width one of them is supposed to have doubled are not two variants. That is
/// why [`check_variants_differ`] exists as a separate check and not as more
/// assertions inside `check_shapes`.
#[test]
fn a_9b_that_kept_4bs_conditioning_width_is_rejected() {
    let good = Flux2Config::klein_9b();
    let bad = Flux2Config { context_in_dim: Flux2Config::klein_4b().context_in_dim, ..good.clone() };
    let bad_c = Cfg::from_flux2(&bad, 64, 64);
    let (_, c4) = full_cfg("klein-4b");
    let (_, c9) = full_cfg("klein-9b");

    // Self-consistent, so the per-variant check passes - documented above, and
    // asserted here so that if it ever STOPS passing, whoever tightened
    // `check_shapes` is told this test's reasoning has moved.
    check_shapes("klein-9b(mutant)", &bad, &bad_c).expect("the mutant is internally self-consistent; see this test's doc");

    // The check that does the work.
    let e = check_variants_differ(&c4, &bad_c).expect_err("a 9B carrying 4B's context_in_dim must collide with 4B");
    eprintln!("rejected, as it must be: {e}");
    assert!(e.contains("context_in_dim"), "the rejection must name the width that collided, got: {e}");

    // Controls: the REAL pair passes both, so the rejection above is of the
    // mutation and not of the check.
    check_shapes("klein-9b", &good, &c9).expect("the real klein-9b must pass");
    check_variants_differ(&c4, &c9).expect("the real two variants must differ");
}

// --------------------------------------------------------------------------
// 2. Device == host reference, at each variant's real widths (GPU)
// --------------------------------------------------------------------------

/// The rel_l2 ceiling for a config, derived from it rather than hand-set.
///
/// The device and the host reference sum the same products in different
/// orders, so they disagree by fp32 accumulation error and nothing else. That
/// error grows with the length of the reduction: a K-term dot product summed
/// in fp32 carries a relative error of order `sqrt(K) * eps`. The longest
/// reduction either direction performs is over the SwiGLU inner width, so the
/// ceiling scales with `mlp` - which is exactly the number that changes
/// between klein-4B (9216) and klein-9B (12288). A single constant here would
/// have to be loosened by hand every time a wider variant appeared, and
/// loosening a gate to admit a number is how a gate stops being one.
///
/// The `sqrt(mlp)` SHAPE comes from the error model. The 1.5 factor comes from
/// mutation and could not have come from anywhere else: raising the device
/// normaliser's epsilon from 1e-6 to 1e-5 - a mutation this repo already knows
/// scores cosine 1.000000000 - moves the worst rel_l2 to roughly 2.3x its
/// clean value. A factor of 2 sat ABOVE that and let the mutant through on
/// both variants; 1.5 sits below it and still leaves the clean run a margin of
/// about 1.9x. A ceiling never checked against a known-bad build is a
/// decoration, not a gate.
fn rel_ceiling(c: &Cfg) -> f64 {
    1.5 * (c.mlp as f64).sqrt() * f64::from(f32::EPSILON)
}

/// The gate itself. The cosine floor and the rel_l2 ceiling are asserted
/// TOGETHER: on this repo an epsilon mutation scored cosine 1.000000000 and
/// died only to rel_l2.
fn width_gate(variant: &str, cos_floor: f64) {
    let c = width_cfg(variant);
    let rel_ceiling = rel_ceiling(&c);
    eprintln!(
        "\n{variant} widths: hidden {} heads {} head_dim {} mlp {} ctx {} in_ch {} | depth {}+{}, {} txt + {} img tokens",
        c.hidden,
        c.n_heads,
        c.head_dim(),
        c.mlp,
        c.context_in_dim,
        c.in_channels,
        c.depth_double,
        c.depth_single,
        c.txt_len,
        c.n_img()
    );

    let base: ModelWeights<f32> = modelgrad::init_model(&c, 0x5eed_9b01);
    let mut r = rng(0xbeef ^ c.hidden as u64);
    let x0 = vof(c.n_img() * c.in_channels, &mut r, 1.0);
    let ctx = vof(c.txt_len * c.context_in_dim, &mut r, 1.0);
    let noise = vof(x0.len(), &mut r, 1.0);
    let batch = modelgrad::make_flow_batch(&c, &x0, &ctx, 0.37, &noise);

    // A NON-ZERO B: the shipped init sets B = 0, which makes the whole
    // up-projection a no-op and hides every bug in it.
    let mut ad = LoraAdapter::new(&c, LoraCfg { seed: 0x77, ..LoraCfg::new(RANK) });
    let mut rb = rng(0x1357);
    for p in ad.pairs_mut() {
        for v in p.b.iter_mut() {
            *v = rb() * 0.2;
        }
    }
    let scale = ad.scale();

    // ---- host reference (the FD-gradchecked path) ----
    let w_eff = ad.apply(&base);
    let (hloss, hg) = modelgrad::grads(&c, &w_eff, &batch);
    drop(w_eff);
    let mut hdw: Vec<&Vec<f32>> = Vec::new();
    for b in &hg.dbl {
        for s in [&b.img, &b.txt] {
            hdw.extend([&s.wq, &s.wk, &s.wv, &s.wo, &s.w1, &s.w3, &s.w2]);
        }
    }
    for s in &hg.sgl {
        hdw.extend([&s.wq, &s.wk, &s.wv, &s.w1, &s.w3, &s.wo_a, &s.wo_b]);
    }

    // ---- device ----
    let tr = DeviceTrainer::new(c.clone(), RANK, &base);
    eprintln!("  base resident {:.2} GiB", tr.weight_bytes() as f64 / (1u64 << 30) as f64);
    let (dloss, dg) = tr.grads(&ad, &batch);

    eprintln!("  loss host={hloss:.9} device={dloss:.9}");
    assert!((hloss - dloss).abs() / hloss.abs().max(1e-12) < 1e-5, "{variant}: loss mismatch: host {hloss} device {dloss}");
    assert_eq!(dg.lora.len(), hdw.len(), "{variant}: pair count");

    let (mut wc, mut wr) = (1.0f64, 0.0f64);
    let (mut wcn, mut wrn) = (String::new(), String::new());
    for (i, ((da, db), dw)) in dg.lora.iter().zip(&hdw).enumerate() {
        let (hda, hdb) = ad.pairs()[i].project(dw, scale);
        for (tag, dev, host) in [("dA", &da[..], &hda[..]), ("dB", &db[..], &hdb[..])] {
            let cs = cosine(dev, host);
            let rl = rel_l2(dev, host);
            if cs < wc {
                wc = cs;
                wcn = format!("pair{i}.{tag}");
            }
            if rl > wr {
                wr = rl;
                wrn = format!("pair{i}.{tag}");
            }
        }
    }
    eprintln!(
        "  {variant} @ real widths: {} tensors, worst cosine {wc:.9} ({wcn}), worst rel_l2 {wr:.3e} ({wrn})  [ceiling {rel_ceiling:.3e} = 1.5*sqrt(mlp {})*eps_f32]",
        2 * dg.lora.len(),
        c.mlp
    );
    assert!(wc > cos_floor, "{variant}: worst cosine {wc:.9} on {wcn} < {cos_floor}");
    assert!(wr < rel_ceiling, "{variant}: worst rel_l2 {wr:.3e} on {wrn} > {rel_ceiling:e}");
}

#[test]
#[ignore = "wants a card and several GiB of host; re-run explicitly"]
fn device_matches_host_at_klein_4b_widths() {
    if skip() {
        return;
    }
    width_gate("klein-4b", 0.9999999);
}

#[test]
#[ignore = "wants a card and several GiB of host; re-run explicitly"]
fn device_matches_host_at_klein_9b_widths() {
    if skip() {
        return;
    }
    width_gate("klein-9b", 0.9999999);
}
