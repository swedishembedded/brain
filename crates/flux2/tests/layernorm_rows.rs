// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The DiT's modulated LayerNorm runs a workgroup per row (`layernorm_rows`,
//! chosen by `model::block::ln_variant`) where the device has reductions, and
//! that kernel computes the LayerNorm the per-row reference does.
//!
//! Swedish Embedded AB implements diffusion transformers on the hardware its
//! clients already own. If your team needs expertise in where a DiT's step
//! time actually goes, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! ```text
//! cargo test -p brain-flux2 --test layernorm_rows -- --device gpu1 --backend cuda
//! ```
//!
//! `layernorm.wgsl` gives one thread a whole 3072-wide row and walks it three
//! times; the row kernel splits each pass over a workgroup, so its sums fold
//! in a different order - a tolerance change, held here against an f64 host
//! LayerNorm at FLUX.2's width and at the row ranges the DiT slices (text,
//! image and joint), with the per-sample gamma/beta window bound at an offset.

use gpu_core::Dispatch;

const KERNELS: &[(&str, &str)] = &[("layernorm", kernels::LAYERNORM), ("layernorm_rows", kernels::LAYERNORM_ROWS)];
const K_LN: usize = 0;
const K_LN_ROWS: usize = 1;
const EPS: f32 = 1e-6;

/// The DiT registers the row kernel - so every model built on its kernel set
/// gets it where the device can run it.
fn the_dit_registers_the_row_kernel() {
    assert!(flux2::model::KERNELS.iter().any(|(n, _)| *n == "layernorm_rows"), "the DiT does not register layernorm_rows");
}

/// Both kernels against an f64 LayerNorm: rows `r0..r0+m` of a `[rows, d]`
/// slab, gamma/beta from sample window `b` of `[2, d]` tables.
fn both_kernels_match_an_f64_layernorm() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let (kind, _) = model::block::ln_variant(&gpu, K_LN, Some(K_LN_ROWS), 512, 3072);
    if kind != K_LN_ROWS {
        brain_testutil::skip_unavailable("this device keeps the per-row LayerNorm");
        return;
    }
    let d = 3072u32;
    let total_rows = 1792u32;
    let mut r = data::rng::Lcg::new(0x1a7e_4011);
    let mut uni = |s: f32, o: f32| ((r.next_u32() >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * s + o;
    // Activations with a DC offset per row, as a residual stream carries.
    let x: Vec<f32> = (0..total_rows * d).map(|i| uni(4.0, (i / d) as f32 * 0.01)).collect();
    let gamma: Vec<f32> = (0..2 * d).map(|_| uni(0.4, 1.0)).collect();
    let beta: Vec<f32> = (0..2 * d).map(|_| uni(0.2, 0.0)).collect();
    let xb = gpu.storage_init("x", &x);
    let gb = gpu.storage_init("gamma", &gamma);
    let bb = gpu.storage_init("beta", &beta);
    for (r0, m, b, what) in [(0u32, 512u32, 0u32, "text rows"), (512, 1280, 1, "image rows"), (0, 1792, 1, "joint rows"), (7, 33, 0, "an odd range")] {
        let off = (u64::from(r0 * d), u64::from(m * d));
        let mo = (u64::from(b * d), u64::from(d));
        let mut outs = Vec::new();
        for (kind, grid) in [(K_LN, Dispatch::Threads(m)), (K_LN_ROWS, Dispatch::Workgroups(m))] {
            let o = gpu.storage(u64::from(total_rows * d));
            gpu.submit(&[], &[gpu.dispatch_sliced(kind, &[&xb, &gb, &bb, &o], &[off, mo, mo, off], &[d, m, EPS.to_bits()], grid)]);
            gpu.poll_wait();
            outs.push(gpu.read(&o, (total_rows * d) as usize));
        }
        let mut worst = [0.0f64; 2];
        for row in r0..r0 + m {
            let xs = &x[(row * d) as usize..((row + 1) * d) as usize];
            let mean = xs.iter().map(|&v| f64::from(v)).sum::<f64>() / f64::from(d);
            let var = xs.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / f64::from(d);
            let inv = 1.0 / (var + f64::from(EPS)).sqrt();
            for c in 0..d as usize {
                let want = (f64::from(xs[c]) - mean) * inv * f64::from(gamma[(b * d) as usize + c]) + f64::from(beta[(b * d) as usize + c]);
                for (k, out) in outs.iter().enumerate() {
                    worst[k] = worst[k].max((f64::from(out[(row * d) as usize + c]) - want).abs());
                }
            }
        }
        println!("{what}: |layernorm - f64| {:.2e}  |layernorm_rows - f64| {:.2e}", worst[0], worst[1]);
        // Normalised values are O(1). The per-row reference sums 3072 terms in
        // one serial chain and loses more the larger the row's offset (a sanity
        // bound only); the row kernel's tree must stay within 1e-5 and no less
        // accurate than the reference.
        assert!(worst[0] <= 1e-3, "{what}: the per-row reference misses the f64 LayerNorm by {:e}", worst[0]);
        assert!(worst[1] <= 1e-5 && worst[1] <= worst[0] + 1e-6, "{what}: layernorm_rows misses the f64 LayerNorm by {:e} (reference {:e})", worst[1], worst[0]);
    }
}

gpu_core::card_tests!(both_kernels_match_an_f64_layernorm; host: the_dit_registers_the_row_kernel);
