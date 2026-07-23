// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Conv-as-GEMM (im2col + `matmul_reg2`) vs the direct register-tiled conv
//! (`conv_act_reg`), on YOLOv8n's layer shapes — on the P40.
//!
//! ```text
//! DISPLAY= cargo test --release -p brain-backend-wgpu --test bench_conv_gemm -- --ignored --nocapture
//! ```
//!
//! `docs/performance/overview.md` rejected im2col+GEMM for conv on the bandwidth-bound
//! Intel Arc but flagged it "worth it on a compute-bound discrete GPU". The P40
//! is that GPU (34 FLOP/byte ridge). This measures both paths' wall-clock and
//! GFLOP/s, and checks the GEMM output matches the direct conv (parity).

use backend_wgpu::WgpuBackend;

struct Shape { label: &'static str, cin: u32, cout: u32, h: u32, w: u32, k: u32, stride: u32, pad: u32 }

const SHAPES: &[Shape] = &[
    Shape { label: "stage1 48->48 3x3 @152x96", cin: 48, cout: 48, h: 96, w: 152, k: 3, stride: 1, pad: 1 },
    Shape { label: "stage2 96->96 3x3 @76x48", cin: 96, cout: 96, h: 48, w: 76, k: 3, stride: 1, pad: 1 },
    Shape { label: "stage3 192->192 3x3 @38x24", cin: 192, cout: 192, h: 24, w: 38, k: 3, stride: 1, pad: 1 },
    Shape { label: "stage4 384->384 3x3 @19x12", cin: 384, cout: 384, h: 12, w: 19, k: 3, stride: 1, pad: 1 },
    Shape { label: "stem   24->48 3x3 s2 @304x192", cin: 24, cout: 48, h: 192, w: 304, k: 3, stride: 2, pad: 1 },
];

const K_IM2COL: usize = 0;
const K_REG2: usize = 1;
const K_CONV_REG: usize = 2;
const K_EPI: usize = 3;

fn kernels() -> Vec<(&'static str, &'static str)> {
    vec![
        ("im2col", kernels::IM2COL),
        ("matmul_reg2", kernels::MATMUL_REG2),
        ("conv_act_reg", kernels::CONV_ACT_REG),
        ("conv_epilogue", kernels::CONV_EPILOGUE),
    ]
}

fn best_secs(f: impl Fn(), reps: usize) -> f64 {
    f(); // warm
    let mut b = f64::INFINITY;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

#[test]
#[ignore]
fn bench_conv_gemm() {
    let g = WgpuBackend::new(&kernels());
    let reps = 20;
    println!("\n{:<32} {:>9} {:>12} {:>12} {:>12} {:>9}", "shape", "GFLOP", "conv_reg", "im2col+reg2", "of which col", "speedup");
    println!("{}", "-".repeat(92));

    for s in SHAPES {
        let ho = (s.h + 2 * s.pad - s.k) / s.stride + 1;
        let wo = (s.w + 2 * s.pad - s.k) / s.stride + 1;
        let hw = (ho * wo) as usize;
        let cinkk = (s.cin * s.k * s.k) as usize;
        let flops = 2.0 * s.cout as f64 * cinkk as f64 * hw as f64;

        let x: Vec<f32> = (0..(s.cin * s.h * s.w) as usize).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
        let wt: Vec<f32> = (0..(s.cout as usize * cinkk)).map(|i| ((i % 13) as f32) * 0.05 - 0.3).collect();
        let sb: Vec<f32> = (0..2 * s.cout as usize).map(|i| if i % 2 == 0 { 0.8 } else { 0.1 }).collect();

        let xb = g.storage_init("x", &x);
        let wb = g.storage_init("w", &wt);
        let sbb = g.storage_init("sb", &sb);
        let colb = WgpuBackend::storage(&g, (hw * cinkk) as u64);
        let y_conv = WgpuBackend::storage(&g, (s.cout as usize * hw) as u64);
        let y_gemm = WgpuBackend::storage(&g, (s.cout as usize * hw) as u64);

        // --- direct register-tiled conv (act=0 identity, bias sb=1/0) ---
        let ntc = s.cout.div_ceil(8);
        let npq = (ho * wo).div_ceil(4);
        let conv_p = [1, s.cin, s.h, s.w, s.cout, s.k, s.stride, s.pad, ho, wo, 2u32]; // act=SiLU
        let t_conv = best_secs(|| {
            let st = g.step(K_CONV_REG, &[&xb, &wb, &sbb, &y_conv], &conv_p, ntc * npq);
            g.submit(&[], &[st]);
            g.poll_wait();
        }, reps);

        // --- im2col + reg2 GEMM:  y[Cout,HW] = W[Cout,CinKK] · col[HW,CinKK] ---
        let col_p = [s.cin, s.h, s.w, s.k, s.stride, s.pad, ho, wo, cinkk as u32];
        let reg_p = [s.cout, cinkk as u32, hw as u32];
        let reg_threads = (s.cout as usize).div_ceil(128) as u32 * (hw.div_ceil(128) as u32) * 256;
        let t_col = best_secs(|| {
            let st = g.step(K_IM2COL, &[&xb, &colb], &col_p, (hw * cinkk) as u32);
            g.submit(&[], &[st]);
            g.poll_wait();
        }, reps);
        let epi_p = [s.cout, hw as u32, 2u32]; // act=SiLU
        let t_gemm = best_secs(|| {
            let st = g.step(K_IM2COL, &[&xb, &colb], &col_p, (hw * cinkk) as u32);
            let st2 = g.step(K_REG2, &[&wb, &colb, &y_gemm], &reg_p, reg_threads);
            let st3 = g.step(K_EPI, &[&sbb, &y_gemm], &epi_p, (s.cout as usize * hw) as u32);
            g.submit(&[], &[st, st2, st3]);
            g.poll_wait();
        }, reps);

        // parity: GEMM path vs direct conv (identity act, bias 0)
        let a = g.read(&y_conv, s.cout as usize * hw);
        let b = g.read(&y_gemm, s.cout as usize * hw);
        let maxd = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        let scale = a.iter().fold(1e-3f32, |m, &v| m.max(v.abs()));
        let rel = maxd / scale;

        println!(
            "{:<32} {:>9.2} {:>9.0} GF {:>9.0} GF {:>9.2} ms {:>7.2}x {}",
            s.label,
            flops / 1e9,
            flops / t_conv / 1e9,
            flops / t_gemm / 1e9,
            t_col * 1e3,
            t_conv / t_gemm,
            if rel < 2e-3 { "ok" } else { "PARITY-FAIL" },
        );
        assert!(rel < 2e-3, "{}: im2col+GEMM diverges from conv (rel {rel:.2e})", s.label);
    }
}
