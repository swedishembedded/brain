// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `ClipVision` - the vanilla CLIP-L/14 image tower, its `PatchSource` bypass
//! seam, its non-native-grid position embedding, and its backward.
//!
//! ## What is actually gated here
//!
//! 1. **The seam is a seam.** `PatchSource::Pixels` and `PatchSource::Tokens`
//!    must produce **bit-identical** tower output when the tokens fed to the
//!    second are the ones the first's conv produced from the same pixels. Bit
//!    equality (not "close") is the whole point: it proves the two paths share
//!    one downstream graph rather than being two implementations that agree to
//!    5 decimals. The conv itself is independently re-computed on the host, so
//!    the test does not merely compare the tower to itself.
//! 2. **The learned position table is resampled, not assumed.** The tower runs
//!    at a grid the checkpoint was never trained at - that is CLIP's entire
//!    reason for existing inside DeepSeek-OCR - so the class-token row must
//!    survive untouched, the patch rows must go through the bicubic resample,
//!    and at the NATIVE grid the resample must be the identity bit for bit
//!    (otherwise every real-checkpoint forward silently drifts).
//! 3. **The backward is the adjoint of the forward.** Directional finite
//!    differences on both `PatchSource` paths and on both backends.
//! 4. **The training build's forward is the inference build's forward**, bit for
//!    bit - a gradient check compares a model against itself and cannot see a
//!    forward that drifted between the two builds.
//!
//! No fixtures: the weights come from `clip::init`, so nothing here skips.

use std::collections::HashMap;

use data::rng::Rng;
use gpu_core::Gpu;

use clip::config::{ClipVisionConfig, TextAct};
use clip::model::{ClipVision, PatchSource, CLIP_VISION_PIPELINES};

// ---------------------------------------------------------------------------
// fixture
// ---------------------------------------------------------------------------

/// Deliberately pairwise-distinct dimensions, so a swapped extent cannot
/// cancel: `d_model 32`, `heads 4`, `head_dim 8`, `mlp 20`, `patch 2`,
/// native grid `5x5` (26 learned positions), 2 blocks.
///
/// The one arithmetic constraint the fixture must respect is the attention
/// `ctx` binding: `attn_apply_cross` has no output-offset Param, so each
/// sample's span is bound at `si*seq*D` and that must be 64-float aligned.
/// `26*32 = 832 = 13*64` (pixels path) and `22*32 = 704 = 11*64` (tokens path),
/// which is what lets the fixture run `B = 2` instead of a single sample.
fn tiny(act: TextAct) -> ClipVisionConfig {
    ClipVisionConfig {
        shape: gguf::deepseek_ocr_vision::ClipConfig {
            d_model: 32,
            n_layers: 2,
            n_heads: 4,
            ffn_hidden: 20,
            patch_size: 2,
            image_size: 10, // native grid 5x5
            n_positions: 26,
            layer_norm_eps: 1e-5,
        },
        act,
    }
}

/// The NON-native grid the `Tokens` path runs at: `3x7`, so the position
/// resample DOWNsamples the height (5 -> 3) and UPsamples the width (5 -> 7) in
/// one call, and `gh != gw` catches a transposed grid.
const TOKEN_GRID: (u32, u32) = (3, 7);
const B: u32 = 2;

/// Which backend a case runs on. A *factory*, not a handle: `Gpu` is not
/// `Clone`, and several cases build two towers side by side. The default
/// backend's handles all come from the pooled test device
/// (`gpu_core::testgpu::dev`, keyed on the pipeline slice address), never a
/// fresh `Gpu::new` per model - that pattern deadlocks the driver under
/// `--test-threads`.
#[derive(Clone, Copy)]
enum Dev {
    Default,
    CpuJit,
}

impl Dev {
    fn gpu(self) -> Gpu {
        match self {
            Dev::Default => gpu_core::testgpu::dev(CLIP_VISION_PIPELINES),
            Dev::CpuJit => Gpu::new_cpu(CLIP_VISION_PIPELINES),
        }
    }
}

/// Both backends the workspace gates on: the default (wgpu/Vulkan) device and
/// the CPU Cranelift JIT.
fn devices() -> Vec<(&'static str, Dev)> {
    let mut v: Vec<(&'static str, Dev)> = Vec::new();
    if std::env::var("MOE_SKIP_GPU_TESTS").is_err() {
        v.push(("default", Dev::Default));
    }
    v.push(("cpu-jit", Dev::CpuJit));
    v
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.next_f32() - 0.5).collect()
}

// ---------------------------------------------------------------------------
// 1. the PatchSource seam
// ---------------------------------------------------------------------------

/// `Conv2d(3, D, k=patch, stride=patch, bias=False)` followed by NCHW -> NLC,
/// on the host. Written out longhand so the device path is compared against an
/// independent computation, not against itself.
fn host_patch_tokens(cfg: &ClipVisionConfig, b: u32, px: &[f32], w: &[f32]) -> Vec<f32> {
    let (d, p, s) = (cfg.d_model() as usize, cfg.patch() as usize, cfg.image_size() as usize);
    let g = cfg.native_grid() as usize;
    let mut out = vec![0f32; b as usize * g * g * d];
    for si in 0..b as usize {
        for gy in 0..g {
            for gx in 0..g {
                for c in 0..d {
                    let mut acc = 0f32;
                    for ci in 0..3 {
                        for kh in 0..p {
                            for kw in 0..p {
                                let px_i = ((si * 3 + ci) * s + gy * p + kh) * s + gx * p + kw;
                                let w_i = ((c * 3 + ci) * p + kh) * p + kw;
                                acc += px[px_i] * w[w_i];
                            }
                        }
                    }
                    out[((si * g + gy) * g + gx) * d + c] = acc;
                }
            }
        }
    }
    out
}

/// **The required test.** `encode(Pixels(px))` and
/// `encode(Tokens{ conv_patch_embed(px), grid })` are bit-identical.
#[test]
fn pixels_and_tokens_paths_are_bit_identical() {
    let cfg = tiny(TextAct::QuickGelu);
    let init = clip::init::init_vision_weights(&cfg, 11);
    let px = clip::init::fixed_pixels(&cfg, B, 5);
    let native = (cfg.native_grid(), cfg.native_grid());

    for (label, dev) in devices() {
        let a = ClipVision::new_on(dev.gpu(), cfg.clone(), B, PatchSource::Pixels, &init);
        a.set_pixels(&px);
        a.forward();
        let tokens = a.read_tokens();
        let out_a = a.read_output();

        // The conv is real: recompute it on the host from the same weights.
        let host = host_patch_tokens(&cfg, B, &px, &init["patch_embed.weight"]);
        assert_eq!(host.len(), tokens.len(), "[{label}] token count");
        let worst = host.iter().zip(&tokens).map(|(h, t)| (h - t).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "[{label}] device patch tokens differ from the host conv by {worst:e}");
        assert!(tokens.iter().any(|v| v.abs() > 1e-4), "[{label}] patch tokens are ~zero");

        // Same pixels, same weights, tokens injected through the bypass.
        let b_tower = ClipVision::new_on(dev.gpu(), cfg.clone(), B, PatchSource::Tokens { grid: native }, &init);
        b_tower.set_tokens(&tokens);
        b_tower.forward();
        let out_b = b_tower.read_output();

        assert_eq!(out_a.len(), out_b.len(), "[{label}] output length");
        assert!(out_a.iter().any(|v| v.abs() > 1e-4), "[{label}] tower output is ~zero");
        for (i, (x, y)) in out_a.iter().zip(&out_b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "[{label}] out[{i}]: Pixels {x} vs Tokens {y} - the bypass is not the same graph"
            );
        }
    }
}

/// The seam refuses the two ways it could silently lie: uploading tokens into a
/// build whose conv would overwrite them, and uploading pixels into a build that
/// has no conv.
#[test]
fn the_seam_refuses_the_wrong_input() {
    let cfg = tiny(TextAct::QuickGelu);
    let init = clip::init::init_vision_weights(&cfg, 3);
    let dev = Dev::CpuJit;

    let pixels = ClipVision::new_on(dev.gpu(), cfg.clone(), 1, PatchSource::Pixels, &init);
    let n = pixels.token_count();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pixels.set_tokens(&vec![0.0; n]))).is_err());

    let tokens = ClipVision::new_on(dev.gpu(), cfg.clone(), 1, PatchSource::Tokens { grid: TOKEN_GRID }, &init);
    let k = 3 * (cfg.image_size() * cfg.image_size()) as usize;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tokens.set_pixels(&vec![0.0; k]))).is_err());
    assert_eq!(tokens.grid(), TOKEN_GRID);
    assert_eq!(tokens.seq_len(), 1 + TOKEN_GRID.0 * TOKEN_GRID.1);
}

// ---------------------------------------------------------------------------
// 2. the learned position embedding at a non-native grid
// ---------------------------------------------------------------------------

/// At the checkpoint's own grid the bicubic resample must be the IDENTITY, bit
/// for bit - `align_corners = 0` puts every output sample exactly on an input
/// sample, where the 4-tap cubic kernel collapses to weight 1 on one tap. If
/// this ever stops holding, every real-checkpoint forward drifts silently.
#[test]
fn native_grid_pos_embed_is_the_table_bit_for_bit() {
    let cfg = tiny(TextAct::QuickGelu);
    let init = clip::init::init_vision_weights(&cfg, 17);
    for (label, dev) in devices() {
        let m = ClipVision::new_on(dev.gpu(), cfg.clone(), 1, PatchSource::Pixels, &init);
        m.forward();
        let got = m.read_pos_full();
        let want = &init["pos_embed"];
        assert_eq!(got.len(), want.len(), "[{label}] pos table length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "[{label}] pos_full[{i}]: {g} vs {w}");
        }
    }
}

/// At a grid the checkpoint was never trained at: the class-token row passes
/// through untouched, the patch rows are a real resample (they differ from any
/// table row), and a CONSTANT table resamples to that same constant - the
/// cheapest proof the four cubic weights sum to 1 in both axes.
#[test]
fn non_native_grid_pos_embed_is_a_real_resample() {
    let cfg = tiny(TextAct::QuickGelu);
    let d = cfg.d_model() as usize;
    let seq = (1 + TOKEN_GRID.0 * TOKEN_GRID.1) as usize;
    let init = clip::init::init_vision_weights(&cfg, 19);

    for (label, dev) in devices() {
        let m = ClipVision::new_on(dev.gpu(), cfg.clone(), 1, PatchSource::Tokens { grid: TOKEN_GRID }, &init);
        m.forward();
        let got = m.read_pos_full();
        assert_eq!(got.len(), seq * d, "[{label}] resampled table is [1 + gh*gw, D]");
        // Class-token row: never part of the patch grid, never resampled.
        for c in 0..d {
            assert_eq!(got[c].to_bits(), init["pos_embed"][c].to_bits(), "[{label}] class pos row changed");
        }
        assert!(got.iter().all(|v| v.is_finite()), "[{label}] resampled table has non-finite entries");
        // A resample onto a different grid cannot reproduce the source rows.
        let src_rows = cfg.native_patches() as usize;
        let same = (0..src_rows).any(|r| {
            (0..d).all(|c| (init["pos_embed"][(1 + r) * d + c] - got[d + c]).abs() < 1e-9)
        });
        assert!(!same, "[{label}] first resampled row equals a source row - the resample did nothing");

        // Constant table -> constant output (the cubic weights sum to 1).
        let mut flat = init.clone();
        flat.insert("pos_embed".into(), vec![0.375f32; cfg.n_positions() as usize * d]);
        let f = ClipVision::new_on(dev.gpu(), cfg.clone(), 1, PatchSource::Tokens { grid: TOKEN_GRID }, &flat);
        f.forward();
        let worst = f.read_pos_full().iter().map(|v| (v - 0.375).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-6, "[{label}] constant table resampled to something else (worst {worst:e})");
    }
}

// ---------------------------------------------------------------------------
// 3. backward
// ---------------------------------------------------------------------------

/// One `ClipVision`, one fixed input, and the fixed proxy direction that turns
/// the tower into a scalar: `L = <r, x_L>`, exactly linear in the output, so
/// `backward()` is seeded with `r` itself.
struct Harness {
    m: ClipVision,
    r: Vec<f32>,
    input: Vec<f32>,
}

impl Harness {
    fn new(gpu: Gpu, cfg: ClipVisionConfig, src: PatchSource, seed: u64) -> Harness {
        let init = clip::init::init_vision_weights(&cfg, seed);
        let m = ClipVision::new_train_on(gpu, cfg.clone(), B, src, &init);
        let input = match src {
            PatchSource::Pixels => clip::init::fixed_pixels(&cfg, B, seed ^ 0x5EED),
            PatchSource::Tokens { grid } => clip::init::fixed_tokens_grid(&cfg, B, grid, seed ^ 0x5EED),
        };
        let r = rand_vec(m.out_len(), seed ^ 0xC11F);
        let h = Harness { m, r, input };
        h.upload();
        h
    }

    fn upload(&self) {
        match self.m.source() {
            PatchSource::Pixels => self.m.set_pixels(&self.input),
            PatchSource::Tokens { .. } => self.m.set_tokens(&self.input),
        }
    }

    /// f64 accumulation: the finite difference is a small difference of two
    /// large-ish sums, and an f32 accumulator's round-off would land straight in
    /// the numerator.
    fn loss(&self) -> f64 {
        self.m.forward();
        self.m.read_output().iter().zip(&self.r).map(|(y, r)| *y as f64 * *r as f64).sum()
    }
}

/// Directional finite differences, in the shape of `gradcheck::directional_check`
/// (which this crate cannot call - `brain-gradcheck` depends on `brain-clip`,
/// so the gate for a NEW tower has to start here and be lifted into
/// `gradcheck::clip` by whoever owns that crate).
///
/// Per tensor: `dirs` random ±1 directions, analytic `<g, v>` against the
/// central difference, keeping the BEST - the same selection rule the shared
/// checker uses, because a single unlucky direction can contract a correct
/// gradient onto ~0.
fn directional(h: &Harness, eps: f32, dirs: usize, seed: u64) -> Vec<(String, f32, f32)> {
    h.m.zero_grads();
    let _ = h.loss();
    h.m.backward(&h.r);
    h.m.poll_wait();

    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    for name in h.m.dispatched_params() {
        let w0 = h.m.read_weight(&name);
        let g = h.m.read_grad(&name);
        let mut best: Option<(f32, f32)> = None;
        for _ in 0..dirs {
            let v: Vec<f32> = (0..w0.len()).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect();
            let analytic: f64 = g.iter().zip(&v).map(|(a, b)| *a as f64 * *b as f64).sum();
            let step = |s: f32| -> f64 {
                let w: Vec<f32> = w0.iter().zip(&v).map(|(a, b)| a + s * eps * b).collect();
                h.m.write_weight(&name, &w);
                h.loss()
            };
            let numeric = (step(1.0) - step(-1.0)) / (2.0 * eps as f64);
            h.m.write_weight(&name, &w0);
            let abs = (analytic - numeric).abs() as f32;
            let rel = abs / (analytic.abs().max(numeric.abs()).max(1e-6) as f32);
            best = Some(match best {
                Some((a, r)) if r <= rel => (a, r),
                _ => (abs, rel),
            });
        }
        let (abs, rel) = best.expect("at least one direction");
        assert!(g.iter().any(|x| *x != 0.0), "{name}: analytic gradient is exactly zero everywhere");
        out.push((name, abs, rel));
    }
    // Restore the forward the caller may inspect afterwards.
    let _ = h.loss();
    out
}

const ATOL: f32 = 4e-3;
const RTOL: f32 = 8e-2;

fn gate(checks: Vec<(String, f32, f32)>, what: &str) {
    let mut worst = 0f32;
    for (name, abs, rel) in &checks {
        println!("  {what:<28} {name:<28} abs {abs:.3e}  rel {rel:.3e}");
        worst = worst.max(*rel);
    }
    let fails: Vec<&(String, f32, f32)> =
        checks.iter().filter(|(_, abs, rel)| *abs > ATOL && *rel > RTOL).collect();
    println!("  {what}: {} tensors, worst rel {worst:.3e}", checks.len());
    assert!(fails.is_empty(), "{what} gradient check failed for {fails:?}");
}

/// **The gate.** The bypass path - a `3x7` token grid the checkpoint's `5x5`
/// position table has to be resampled onto, which is the configuration
/// DeepSeek-OCR actually runs.
///
/// `eps = 5e-4`, not the workspace default `5e-3`: the largest tensor here is
/// `blocks.N.attn.qkv.weight` at 3072 entries, where a ±1 direction at `5e-3` is
/// an L2 step of 0.28 in weight space - outside the region where a softmax is
/// locally linear.
#[test]
fn clip_vision_tokens_path_grads_match_finite_differences() {
    for (label, dev) in devices() {
        let h = Harness::new(dev.gpu(), tiny(TextAct::QuickGelu), PatchSource::Tokens { grid: TOKEN_GRID }, 7);
        gate(directional(&h, 5e-4, 4, 0x1234), &format!("ClipVision/tokens [{label}]"));
    }
}

/// The `Pixels` path: the same tower plus `conv2d` / `conv2d_dw`, at the
/// checkpoint's native grid.
#[test]
fn clip_vision_pixels_path_grads_match_finite_differences() {
    for (label, dev) in devices() {
        let h = Harness::new(dev.gpu(), tiny(TextAct::QuickGelu), PatchSource::Pixels, 7);
        gate(directional(&h, 5e-4, 4, 0x1234), &format!("ClipVision/pixels [{label}]"));
    }
}

/// The second activation. `quick_gelu_bwd` standing in for `gelu_erf_bwd` (or
/// the reverse) is a real, otherwise-invisible bug - see
/// `crates/gradcheck/tests/gelu_erf_fd.rs`.
#[test]
fn clip_vision_gelu_erf_grads_match_finite_differences() {
    let h = Harness::new(Dev::CpuJit.gpu(), tiny(TextAct::GeluErf), PatchSource::Tokens { grid: TOKEN_GRID }, 7);
    gate(directional(&h, 5e-4, 4, 0x1234), "ClipVision/gelu_erf [cpu-jit]");
}

/// The gradient w.r.t. the INJECTED tokens - what DeepSeek-OCR's SAM branch
/// needs to keep training through the seam, and the one gradient no parameter
/// check covers.
#[test]
fn injected_token_gradient_matches_finite_differences() {
    let cfg = tiny(TextAct::QuickGelu);
    let h = Harness::new(Dev::CpuJit.gpu(), cfg, PatchSource::Tokens { grid: TOKEN_GRID }, 7);
    h.m.zero_grads();
    let _ = h.loss();
    h.m.backward(&h.r);
    h.m.poll_wait();
    let g = h.m.read_token_grad();
    assert_eq!(g.len(), h.input.len());

    let eps = 1e-3f32;
    for &i in &[0usize, 37, 129, 400] {
        let probe = |s: f32| -> f64 {
            let mut t = h.input.clone();
            t[i] += s * eps;
            h.m.set_tokens(&t);
            h.loss()
        };
        let numeric = ((probe(1.0) - probe(-1.0)) / (2.0 * eps as f64)) as f32;
        h.upload();
        let (a, n) = (g[i], numeric);
        assert!(
            (a - n).abs() <= ATOL + RTOL * n.abs(),
            "d tokens[{i}]: analytic {a} vs numeric {n}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. the training build does not perturb the forward
// ---------------------------------------------------------------------------

#[test]
fn training_build_forward_is_identical_to_the_inference_build() {
    let cfg = tiny(TextAct::QuickGelu);
    let init: HashMap<String, Vec<f32>> = clip::init::init_vision_weights(&cfg, 23);
    let tokens = clip::init::fixed_tokens_grid(&cfg, B, TOKEN_GRID, 29);
    let dev = Dev::CpuJit;

    let infer = ClipVision::new_on(dev.gpu(), cfg.clone(), B, PatchSource::Tokens { grid: TOKEN_GRID }, &init);
    infer.set_tokens(&tokens);
    infer.forward();
    let a = infer.read_output();
    assert!(!infer.is_trainable());
    drop(infer);

    let train = ClipVision::new_train_on(dev.gpu(), cfg, B, PatchSource::Tokens { grid: TOKEN_GRID }, &init);
    train.set_tokens(&tokens);
    // TWICE: `vit_block_fwd_cached` accumulates its qkv copy with `axpy`, so a
    // forward that forgot the zero-clear would double the second run.
    train.forward();
    train.forward();
    assert!(train.is_trainable());
    let b = train.read_output();

    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "out[{i}]: inference {x} vs training {y}");
    }
}

/// One host gradient-descent step must reduce the same proxy objective the
/// gradient was taken of. A backward whose sign or scale is wrong passes a
/// finite-difference check that compares it against nothing else.
#[test]
fn a_gradient_step_reduces_the_proxy_objective() {
    let h = Harness::new(Dev::CpuJit.gpu(), tiny(TextAct::QuickGelu), PatchSource::Tokens { grid: TOKEN_GRID }, 31);
    let l0 = h.loss();
    h.m.zero_grads();
    h.m.backward(&h.r);
    h.m.poll_wait();

    let names = h.m.dispatched_params();
    let gnorm: f32 = names
        .iter()
        .map(|k| h.m.read_grad(k).iter().map(|g| g * g).sum::<f32>())
        .sum::<f32>()
        .sqrt();
    assert!(gnorm > 1e-3, "gradient is ~zero (norm {gnorm}) - nothing was written");
    let lr = 1e-2 / gnorm;
    for k in &names {
        let w = h.m.read_weight(k);
        let g = h.m.read_grad(k);
        let stepped: Vec<f32> = w.iter().zip(&g).map(|(w, g)| w - lr * g).collect();
        h.m.write_weight(k, &stepped);
    }
    let l1 = h.loss();
    eprintln!("proxy loss {l0:.6} -> {l1:.6} (grad norm {gnorm:.4})");
    assert!(l1 < l0, "a descent step did not decrease the loss: {l0} -> {l1}");
}
