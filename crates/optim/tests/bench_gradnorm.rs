// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The optimiser's global grad-norm: `gradnorm_sq` (ONE thread walks the whole
//! tensor) vs `gradnorm_part` + `clip_coef_wg` (cooperative tree reduction) —
//! parity + achieved bandwidth at the tensor-size distribution real models have.
//!
//! `gradnorm_sq.wgsl` dispatches **one invocation** per parameter tensor and
//! loops `numel` times inside it. That is not a coalescing bug like `rmsnorm` /
//! `layernorm` (thread `t` gets row `t`, eight-way amplification) - it is a
//! *parallelism* bug one level worse: a 38.6 M-element embedding gradient
//! becomes 38.6 M dependent scalar loads on one lane of a 3840-core card.
//! Measured in situ it was the overwhelming majority of all GPU time in
//! `brain gpt train`.
//!
//! Both param COUNT and size SKEW matter, so the sizes below are the real ones:
//! dozens of 768-element LayerNorm/bias tensors alongside a couple of ~39 M
//! embedding matrices. The aggregate row is what the optimiser step actually
//! pays (per-size cost × how many tensors of that size the model has).
//!
//! ```text
//! DISPLAY= BRAIN_DEVICE=gpu0 cargo test --release -p brain-gpu-core \
//!     --test bench_gradnorm -- --ignored --nocapture
//! ```
//!
//! `PEAK_GBPS` is the Tesla P40's datasheet bandwidth; on another card set it
//! to that card's own figure and read the achieved column, not the percentage.

use gpu_core::Gpu;

const PEAK_GBPS: f64 = 346.0;

/// `(numel, count)` — one distinct tensor size and how many tensors of that
/// size the model has. Timing is per size; the aggregate weights by `count`.
type ParamDist = &'static [(usize, usize)];

/// GPT-2-small (12 × 768, ff 3072, block 1024, vocab 50257, untied head) —
/// 148 tensors, 124 M params, of which 77 M sit in the two embedding matrices.
const GPT2_SMALL: ParamDist = &[
    (768, 74),           // ln1/ln2 weight+bias, attn.out.bias, mlp.proj.bias, final ln
    (2304, 12),          // attn.qkv.bias
    (3072, 12),          // mlp.fc.bias
    (786_432, 1),        // pos.weight  (1024 × 768)
    (589_824, 12),       // attn.out.weight (768 × 768)
    (1_769_472, 12),     // attn.qkv.weight (2304 × 768)
    (2_359_296, 24),     // mlp.fc.weight + mlp.proj.weight (3072 × 768)
    (38_597_376, 2),     // tok.weight + lm_head.weight (50257 × 768)
];

/// Qwen3-0.6B-shaped (28 layers, hidden 1024, GQA 16/8 heads × 128, ffn 3072,
/// vocab 151936, tied embedding) — 311 tensors, ~596 M params. Far more tiny
/// tensors (q_norm/k_norm are 128 elements each) and one enormous embedding.
const QWEN_0B6: ParamDist = &[
    (128, 56),           // q_norm / k_norm (head_dim)
    (1024, 57),          // input/post attention RMSNorm gains + final norm
    (1_048_576, 56),     // k_proj / v_proj … (1024 × 1024-ish)
    (2_097_152, 56),     // q_proj / o_proj  (2048 × 1024)
    (3_145_728, 84),     // gate / up / down (3072 × 1024)
    (155_582_464, 1),    // embed_tokens (151936 × 1024), tied head
];

fn fill(n: usize, s: usize) -> Vec<f32> {
    (0..n).map(|i| ((((i * 37 + s * 13) % 197) as f32 / 197.0) - 0.5) * 0.01).collect()
}

/// Min-of-N wall clock for ONE dispatch. Long dispatches (the serial kernel on a
/// big tensor takes seconds) are timed once, singly; short ones are timed as 4
/// back-to-back dispatches × 8 reps so launch overhead is amortised, as in
/// `bench_layernorm`.
fn time(gpu: &Gpu, kind: usize, ub: &gpu_core::DeviceBuffer, bufs: &[&gpu_core::DeviceBuffer], threads: u32) -> f64 {
    let warm = std::time::Instant::now();
    let s = gpu.step_buf(kind, ub, bufs, threads);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();
    let first = warm.elapsed().as_secs_f64();
    let (batch, reps) = if first > 0.02 { (1, 2) } else { (4, 8) };
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        let steps: Vec<_> = (0..batch).map(|_| gpu.step_buf(kind, ub, bufs, threads)).collect();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        best = best.min(t.elapsed().as_secs_f64() / batch as f64);
    }
    best
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

#[test]
#[ignore]
fn bench_gradnorm() {
    let ks = &[
        ("gradnorm_sq", kernels::GRADNORM_SQ),
        ("gradnorm_part", kernels::GRADNORM_PART),
        ("clip_coef", kernels::CLIP_COEF),
        ("clip_coef_wg", kernels::CLIP_COEF_WG),
    ];
    let (sq, part, cc, ccw) = (0usize, 1, 2, 3);
    let g = Gpu::new_wgpu(ks);

    for (model, dist) in [("GPT-2-small (124 M, 148 tensors)", GPT2_SMALL), ("Qwen3-0.6B (596 M, 311 tensors)", QWEN_0B6)] {
        println!(
            "\n{model}\n{:>12} {:>6} {:>12} {:>10} {:>12} {:>10} {:>9} {:>10}",
            "numel", "count", "gradnorm_sq", "GB/s", "grad*_part", "GB/s", "speedup", "rel diff"
        );
        println!("{}", "-".repeat(90));
        let (mut tot_ref, mut tot_new, mut n_params, mut n_parts, mut n_elems) = (0.0, 0.0, 0usize, 0u32, 0usize);
        for &(numel, count) in dist {
            // Values do not affect bandwidth, so only tensors small enough to
            // build on the host cheaply are filled; parity is checked
            // separately below on filled buffers.
            let gb = if numel <= 4_000_000 {
                g.storage_init("grad", &fill(numel, 1))
            } else {
                g.storage(numel as u64)
            };
            let nwg = paramstore::gradnorm_parts(numel);
            let norms = g.storage(nwg.max(1) as u64);

            let u_sq = g.uniform_dynamic(2);
            g.write(&u_sq, &[numel as u32, 0]);
            let t_ref = time(&g, sq, &u_sq, &[&gb, &norms], 1);
            let v_ref = g.read(&norms, 1)[0];

            let u_pt = g.uniform_dynamic(3);
            g.write(&u_pt, &[numel as u32, 0, nwg]);
            let t_new = time(&g, part, &u_pt, &[&gb, &norms], nwg * 64);
            let v_new: f32 = g.read(&norms, nwg as usize).iter().sum();

            let bytes = numel as f64 * 4.0;
            let rel = if v_ref.abs() > 0.0 { ((v_new - v_ref) / v_ref).abs() } else { 0.0 };
            println!(
                "{numel:>12} {count:>6} {:>12.3} {:>10.1} {:>12.3} {:>10.1} {:>8.1}x {:>10.1e}",
                t_ref * 1e3,
                bytes / t_ref / 1e9,
                t_new * 1e3,
                bytes / t_new / 1e9,
                t_ref / t_new,
                rel
            );
            // A DIFFERENCE column, not an error column: the f64 oracle in
            // `gradnorm_part_matches_gradnorm_sq` shows the serial walk is the
            // inaccurate side (2.3e-3 relative at 4 M elements vs the tree's
            // 2.4e-7), because a single fp32 accumulator over millions of adds
            // loses the small terms. The gate here only catches a wrong answer.
            assert!(rel < 1e-2, "gradnorm_part disagrees at numel {numel}: {v_ref} vs {v_new}");
            tot_ref += t_ref * count as f64;
            tot_new += t_new * count as f64;
            n_params += count;
            n_parts += nwg * count as u32;
            n_elems += numel * count;
        }
        // The clip-coefficient fold: `clip_coef` walks its input on one thread
        // (n_params entries on the reference path, n_parts on the cooperative
        // one — which is why the cooperative path needs `clip_coef_wg`).
        let parts = g.storage(n_parts.max(n_params as u32) as u64);
        let coef = g.storage(1);
        let u = g.uniform_dynamic(3);
        g.write(&u, &[n_params as u32, f(1.0), f(1.0)]);
        let t_cc = time(&g, cc, &u, &[&parts, &coef], 1);
        g.write(&u, &[n_parts, f(1.0), f(1.0)]);
        let t_cc_serial_parts = time(&g, cc, &u, &[&parts, &coef], 1);
        let t_ccw = time(&g, ccw, &u, &[&parts, &coef], 64);

        let agg_ref = tot_ref + t_cc;
        let agg_new = tot_new + t_ccw;
        println!("{}", "-".repeat(90));
        println!(
            "clip fold      : clip_coef over {n_params} tensors {:.3} ms | clip_coef over {n_parts} partials {:.3} ms | clip_coef_wg {:.3} ms",
            t_cc * 1e3,
            t_cc_serial_parts * 1e3,
            t_ccw * 1e3
        );
        println!(
            "WHOLE GRAD-NORM: {:.1} ms ({:.1} GB/s)  ->  {:.1} ms ({:.1} GB/s)   {:.0}x   [{} dispatches -> {}]",
            agg_ref * 1e3,
            n_elems as f64 * 4.0 / agg_ref / 1e9,
            agg_new * 1e3,
            n_elems as f64 * 4.0 / agg_new / 1e9,
            agg_ref / agg_new,
            n_params + 1,
            n_params + 1,
        );
        println!(
            "                 peak {PEAK_GBPS} GB/s -> cooperative reaches {:.0}% of it",
            100.0 * (n_elems as f64 * 4.0 / agg_new / 1e9) / PEAK_GBPS
        );
    }
}

/// Parity of the whole two-stage reduction across the size range, against an
/// **f64 oracle** — neither kernel is the reference for accuracy.
///
/// The tree is not merely "as good as" the serial walk, it is far better:
/// `gradnorm_sq` keeps one fp32 accumulator across `numel` sequential adds, so
/// once the running sum is large the individual squares round away. Measured on
/// a P40: at 4.19 M elements the serial walk is 2.3e-3 relative off the exact
/// value while the tree (64 partials per workgroup, then a second pass) is
/// 2.4e-7 - four orders of magnitude. The clip coefficient is therefore not
/// only far cheaper to compute, it is also the more correct one.
#[test]
#[ignore]
fn gradnorm_part_matches_gradnorm_sq() {
    let ks = &[("gradnorm_sq", kernels::GRADNORM_SQ), ("gradnorm_part", kernels::GRADNORM_PART)];
    let g = Gpu::new_wgpu(ks);
    for &numel in &[1usize, 63, 64, 65, 768, 8191, 8192, 8193, 100_000, 1_769_472, 4_194_304] {
        let data = fill(numel, 7);
        let gb = g.storage_init("grad", &data);
        let nwg = paramstore::gradnorm_parts(numel);
        let norms = g.storage(nwg as u64 + 1);

        let u_sq = g.uniform_dynamic(2);
        g.write(&u_sq, &[numel as u32, 0]);
        let s = g.step_buf(0, &u_sq, &[&gb, &norms], 1);
        g.submit(&[], &[s]);
        let v_ref = g.read(&norms, 1)[0];

        let u_pt = g.uniform_dynamic(3);
        g.write(&u_pt, &[numel as u32, 0, nwg]);
        let s = g.step_buf(1, &u_pt, &[&gb, &norms], nwg * 64);
        g.submit(&[], &[s]);
        let v_new: f32 = g.read(&norms, nwg as usize).iter().sum();

        // f64 oracle — neither kernel is the reference for accuracy.
        let exact: f64 = data.iter().map(|&v| v as f64 * v as f64).sum();
        let e_ref = ((v_ref as f64 - exact) / exact).abs();
        let e_new = ((v_new as f64 - exact) / exact).abs();
        println!("numel {numel:>9}: serial {v_ref:.9e} (err {e_ref:.2e})  tree {v_new:.9e} (err {e_new:.2e})");
        assert!(e_new < 1e-5, "tree reduction err {e_new} at numel {numel}");
    }
}
