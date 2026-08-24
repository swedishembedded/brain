// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Direct vs GEMM-lowered 1D convolution, swept over `Cout` - the measurement
//! behind `backend_api::select::GEMM_CONV1D_MIN_COUT`.
//!
//! `#[ignore]`d: it is a benchmark, not a gate. Run it with
//!
//! ```text
//! cargo test --release -p brain-audio --test bench_conv1d_lowering -- --ignored --nocapture
//! ```
//!
//! Method, following this repo's kernel-measurement discipline:
//!
//! * **Best-of-N**, warm-up excluded, every timed region `poll_wait()`-bracketed:
//!   an unbracketed loop times host-side recording and reports it as device
//!   throughput.
//! * **`max|delta|` beside every timing**, against the OTHER path, plus each
//!   path's own agreement with the host reference. A faster kernel that
//!   disagrees is not a faster kernel.
//! * **Each kernel pair gets its own threshold.** `conv1d` and `convtr1d` are
//!   swept separately because they are different GEMMs with different
//!   epilogues; reusing one crossover for the other is a mistake this repo has
//!   already paid for on a forward/backward pair.
//!
//! The shapes are the MiniMax-Music-3 vocoder's own: a k=7 pad-3 residual conv
//! and a `K = 2·stride` upsampling transposed conv, at the lengths those run at.

use audio::conv::{conv1d_bias_fwd, conv1d_ref, convtr1d_bias_fwd, convtr1d_ref, Conv1d, ConvGemmKernels, ConvKernels, ConvScratch};
use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu, Step};

const PIPELINES: &[(&str, &str)] = &[
    ("conv1d", kernels::CONV1D),
    ("convtr1d", kernels::CONVTR1D),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("im2col1d_at", kernels::IM2COL1D_AT),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg_splitk", kernels::MATMUL_DW_REG_SPLITK),
    ("nlc_bias_nchw", kernels::NLC_BIAS_NCHW),
    ("col2im1d_bias", kernels::COL2IM1D_BIAS),
];

fn kern(fwd: usize) -> ConvGemmKernels {
    ConvGemmKernels {
        direct: ConvKernels { fwd, dx: 0, dw: 0 },
        bias: 2,
        im2col: 3,
        matmul: 4,
        matmul_nn: 5,
        matmul_tn: 6,
        nlc_bias: 7,
        col2im: 8,
    }
}

const REPS: usize = 5;

fn best_ms(g: &Gpu, steps: &[Step]) -> f64 {
    // Warm-up (pipeline/bind-group residency), never counted.
    g.submit(&[], steps);
    g.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = std::time::Instant::now();
        g.submit(&[], steps);
        g.poll_wait();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| f64::from((x - y).abs())).fold(0.0, f64::max)
}

fn with_bias(c: &Conv1d, mut y: Vec<f32>, bias: &[f32]) -> Vec<f32> {
    for (i, v) in y.iter_mut().enumerate() {
        *v += bias[(((i as u32) / c.lo) % c.cout) as usize];
    }
    y
}

struct Operands {
    x: DeviceBuffer,
    w: DeviceBuffer,
    b: DeviceBuffer,
    y: DeviceBuffer,
    want: Vec<f32>,
}

fn operands(g: &Gpu, c: &Conv1d, transposed: bool, rng: &mut Lcg) -> Operands {
    let x = rng.vec_scaled((c.n * c.cin * c.l) as usize, 1.0);
    let wv = rng.vec_scaled(if transposed { c.weight_numel_transposed() } else { c.weight_numel() }, 0.5);
    let bias = rng.vec_scaled(c.cout as usize, 0.25);
    let want = with_bias(c, if transposed { convtr1d_ref(c, &x, &wv) } else { conv1d_ref(c, &x, &wv) }, &bias);
    Operands {
        x: g.storage_init("x", &x),
        w: g.storage_init("w", &wv),
        b: g.storage_init("b", &bias),
        y: g.storage(u64::from(c.n) * u64::from(c.cout) * u64::from(c.lo)),
        want,
    }
}

/// Force the direct path regardless of the selector: the bench needs BOTH
/// sides at every `Cout`, and above the threshold the selector only offers
/// one. `groups` is the structural exclusion the lowering cannot express, so
/// `groups = 1` with a grouped-shaped clone is the honest way to reach the
/// same arithmetic on the other path - here, just the raw kernel pair.
fn direct_steps(g: &Gpu, k: &ConvGemmKernels, c: &Conv1d, o: &Operands, transposed: bool) -> Vec<Step> {
    let p = [c.n, c.cin, c.l, c.cout, c.k, c.stride, c.pad, c.dilation, c.groups, c.lo];
    let total = c.n * c.cout * c.lo;
    let fwd = if transposed { 1 } else { 0 };
    vec![
        g.step(fwd, &[&o.x, &o.w, &o.y], &p, total),
        g.step(k.bias, &[&o.y, &o.b], &[total, c.cout, c.lo], total),
    ]
}

fn sweep(label: &str, g: &Gpu, transposed: bool, shape: impl Fn(u32) -> Conv1d) {
    println!("\n{label}");
    println!("{:>6} {:>10} {:>10} {:>9} {:>12} {:>12}", "Cout", "direct ms", "lowered ms", "speedup", "max|d| ref", "max|d| A/B");
    let mut rng = Lcg::new(0xBE0C);
    for cout in [4u32, 8, 12, 16, 24, 32, 48, 64, 128, 256] {
        let c = shape(cout);
        let o = operands(g, &c, transposed, &mut rng);
        let k = kern(usize::from(transposed));

        let d = direct_steps(g, &k, &c, &o, transposed);
        let t_direct = best_ms(g, &d);
        let got_direct = g.read(&o.y, o.want.len());

        let mut scratch = ConvScratch::new();
        let l = if transposed {
            convtr1d_bias_fwd(g, &k, &c, &o.x, &o.w, &o.b, &o.y, &mut scratch)
        } else {
            conv1d_bias_fwd(g, &k, &c, &o.x, &o.w, &o.b, &o.y, &mut scratch)
        };
        // Below `GEMM_CONV1D_MIN_COUT` the selector answers "direct" for both
        // columns, which reads as a meaningless 1.0x - run this sweep with
        // `BRAIN_CONV1D_GEMM=force` to see the sub-threshold side, which is
        // the half that says where the crossover actually is.
        let t_lowered = best_ms(g, &l);
        let got_lowered = g.read(&o.y, o.want.len());

        println!(
            "{cout:>6} {t_direct:>10.3} {t_lowered:>10.3} {:>8.2}x {:>12.2e} {:>12.2e}",
            t_direct / t_lowered,
            max_abs(&got_lowered, &o.want),
            max_abs(&got_lowered, &got_direct),
        );
    }
}

#[test]
#[ignore]
fn bench_conv1d_lowering_crossover() {
    let g = Gpu::new_wgpu(PIPELINES);
    if !g.caps().workgroup_reductions {
        println!("no workgroup reductions on this device - the lowering is never selected");
        return;
    }
    // The vocoder's residual conv: k=7, pad=3, stride 1, at its widest length.
    sweep("conv1d k=7 pad=3, N=2, Cin=Cout, L=44096", &g, false, |cout| {
        let l = 44096;
        Conv1d { n: 2, cin: cout, l, cout, k: 7, stride: 1, pad: 3, dilation: 1, groups: 1, lo: Conv1d::out_len(l, 7, 1, 3, 3, 1) }
    });
    // The 1x1 projection (the `matmul_dx_reg` NN path, no im2col).
    sweep("conv1d k=1, N=2, Cin=Cout, L=44096", &g, false, |cout| {
        let l = 44096;
        Conv1d { n: 2, cin: cout, l, cout, k: 1, stride: 1, pad: 0, dilation: 1, groups: 1, lo: l }
    });
    // The upsampling transposed conv: K = 2*stride, pad = stride/2.
    sweep("convtr1d stride=4 K=8, N=2, Cin=2*Cout, L=11024", &g, true, |cout| {
        let l = 11024;
        Conv1d { n: 2, cin: 2 * cout, l, cout, k: 8, stride: 4, pad: 2, dilation: 1, groups: 1, lo: Conv1d::out_len_transposed(l, 8, 4, 2, 0, 1) }
    });
}
