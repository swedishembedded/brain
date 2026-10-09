// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the native dense-conv kernels (`kernels_cuda`,
//! `cu/conv2d_f32.cu`): the forward and the input gradient that
//! `gpu_core::native_upgrade` substitutes for `conv2d` / `conv2d_dx` on a CUDA
//! device, and the split weight gradient `Gpu::conv2d_dw_steps` builds.
//!
//! Swedish Embedded AB implements training kernels for convolutional networks.
//! If your team needs expertise in fast and verifiably correct GPU
//! convolution backward passes, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! # The bar, and where it comes from
//!
//! The native kernels do not reproduce the WGSL kernels' bits: they accumulate
//! with a fused multiply-add (one rounding where the reference has two) and the
//! weight gradient re-associates its position sum into slices. So the gate is
//! the fp32 summation error bound, derived rather than tuned. Each output is a
//! sum of `L` products `t_i`; evaluated in fp32 in ANY order (any summation
//! tree of height at most `L`, with or without fused multiply-adds) it obeys
//!
//! ```text
//! |computed - exact| <= gamma_L * sum_i |t_i|,   gamma_L = L u / (1 - L u),  u = 2^-24
//! ```
//!
//! (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §3.1 and
//! §4.2). `L` is the reduction length: `Cin*K*K` for the forward, `Cout*K*K`
//! for the input gradient, `N*Ho*Wo` for the weight gradient - plus one for the
//! weight gradient's accumulation into the existing `dw`, whose magnitude joins
//! the sum. The host oracle evaluates the exact sum and `sum |t_i|` in f64, and
//! BOTH the native kernel and the WGSL reference are held to the bound: the
//! reference passing is what shows the oracle and the bound are right, the
//! native kernel passing is the claim.
//!
//! At the real model geometry the host oracle is too slow, so the second test
//! compares native to reference on the device, each within `gamma_L` of the
//! exact value, i.e. within `2 gamma_L * sum |t_i|` of each other, with
//! `sum |t_i|` itself computed by the reference kernel from `|inputs|`.

use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[
    ("conv2d", kernels::CONV2D),
    ("conv2d_dx", kernels::CONV2D_DX),
    ("conv2d_dw", kernels::CONV2D_DW),
    // The same sources under names no redirect knows: the WGSL reference,
    // dispatched on the same device from the same inputs.
    ("conv2d_ref", kernels::CONV2D),
    ("conv2d_dx_ref", kernels::CONV2D_DX),
    ("conv2d_dw_ref", kernels::CONV2D_DW),
    ("conv_bias", kernels::CONV_BIAS),
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("conv_bias_ref", kernels::CONV_BIAS),
];
const CONV: usize = 0;
const CONV_DX: usize = 1;
const CONV_DW: usize = 2;
const CONV_REF: usize = 3;
const CONV_DX_REF: usize = 4;
const CONV_DW_REF: usize = 5;
const CONV_BIAS: usize = 6;
const CONV_BIAS_REG: usize = 7;
const CONV_BIAS_REF: usize = 8;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

/// One dense convolution: `[N, Cin, H, W]` in, `Cout` channels of `K x K`.
#[derive(Clone, Copy, Debug)]
struct Conv {
    n: u32,
    cin: u32,
    h: u32,
    w: u32,
    cout: u32,
    k: u32,
    s: u32,
    pad: u32,
}

impl Conv {
    fn ho(&self) -> u32 {
        (self.h + 2 * self.pad - self.k) / self.s + 1
    }
    fn wo(&self) -> u32 {
        (self.w + 2 * self.pad - self.k) / self.s + 1
    }
    fn params(&self) -> [u32; 10] {
        [self.n, self.cin, self.h, self.w, self.cout, self.k, self.s, self.pad, self.ho(), self.wo()]
    }
    fn x_len(&self) -> usize {
        (self.n * self.cin * self.h * self.w) as usize
    }
    fn y_len(&self) -> usize {
        (self.n * self.cout * self.ho() * self.wo()) as usize
    }
    fn w_len(&self) -> usize {
        (self.cout * self.cin * self.k * self.k) as usize
    }
}

/// Every distinct dense conv unit of YOLOv8n (nc = 3): `(Cin, Cout, K, stride,
/// pad, input side at a 512 input)` - backbone, neck, and both detect-head
/// branches including the biased 1x1 projections, whose input and weight
/// gradients run through these kernels too.
const YOLOV8N: &[(u32, u32, u32, u32, u32, u32)] = &[
    (3, 16, 3, 2, 1, 512),
    (16, 32, 3, 2, 1, 256),
    (32, 32, 1, 1, 0, 128),
    (16, 16, 3, 1, 1, 128),
    (48, 32, 1, 1, 0, 128),
    (32, 64, 3, 2, 1, 128),
    (64, 64, 1, 1, 0, 64),
    (32, 32, 3, 1, 1, 64),
    (128, 64, 1, 1, 0, 64),
    (64, 128, 3, 2, 1, 64),
    (128, 128, 1, 1, 0, 32),
    (64, 64, 3, 1, 1, 32),
    (256, 128, 1, 1, 0, 32),
    (128, 256, 3, 2, 1, 32),
    (256, 256, 1, 1, 0, 16),
    (128, 128, 3, 1, 1, 16),
    (384, 256, 1, 1, 0, 16),
    (256, 128, 1, 1, 0, 16),
    (512, 256, 1, 1, 0, 16),
    (384, 128, 1, 1, 0, 32),
    (192, 128, 1, 1, 0, 32),
    (192, 64, 1, 1, 0, 64),
    (96, 64, 1, 1, 0, 64),
    (64, 64, 3, 2, 1, 64),
    (128, 128, 3, 2, 1, 32),
    (64, 64, 3, 1, 1, 64),
    (64, 3, 1, 1, 0, 64),
    (128, 64, 3, 1, 1, 32),
    (256, 64, 3, 1, 1, 16),
];

const U: f64 = 1.0 / (1u64 << 24) as f64;

fn gamma(l: usize) -> f64 {
    let lu = l as f64 * U;
    lu / (1.0 - lu)
}

/// The exact (f64) result and `sum |t_i|` of each output, plus the reduction
/// length the bound is taken at.
struct Oracle {
    exact: Vec<f64>,
    abs: Vec<f64>,
    len: usize,
}

fn oracle_fwd(c: &Conv, x: &[f32], w: &[f32]) -> Oracle {
    let (ho, wo) = (c.ho() as usize, c.wo() as usize);
    let (n, cin, h, wd, cout, k, s, pad) =
        (c.n as usize, c.cin as usize, c.h as usize, c.w as usize, c.cout as usize, c.k as usize, c.s as usize, c.pad as usize);
    let mut exact = vec![0.0; c.y_len()];
    let mut abs = vec![0.0; c.y_len()];
    for b in 0..n {
        for co in 0..cout {
            for oy in 0..ho {
                for ox in 0..wo {
                    let (mut e, mut a) = (0.0f64, 0.0f64);
                    for ci in 0..cin {
                        for kh in 0..k {
                            let hi = (oy * s + kh) as isize - pad as isize;
                            if hi < 0 || hi >= h as isize {
                                continue;
                            }
                            for kw in 0..k {
                                let wi = (ox * s + kw) as isize - pad as isize;
                                if wi < 0 || wi >= wd as isize {
                                    continue;
                                }
                                let t = f64::from(x[((b * cin + ci) * h + hi as usize) * wd + wi as usize]) * f64::from(w[((co * cin + ci) * k + kh) * k + kw]);
                                e += t;
                                a += t.abs();
                            }
                        }
                    }
                    let i = ((b * cout + co) * ho + oy) * wo + ox;
                    exact[i] = e;
                    abs[i] = a;
                }
            }
        }
    }
    Oracle { exact, abs, len: cin * k * k }
}

fn oracle_dx(c: &Conv, dy: &[f32], w: &[f32]) -> Oracle {
    let (ho, wo) = (c.ho() as usize, c.wo() as usize);
    let (n, cin, h, wd, cout, k, s, pad) =
        (c.n as usize, c.cin as usize, c.h as usize, c.w as usize, c.cout as usize, c.k as usize, c.s as usize, c.pad as usize);
    let mut exact = vec![0.0; c.x_len()];
    let mut abs = vec![0.0; c.x_len()];
    for b in 0..n {
        for co in 0..cout {
            for oy in 0..ho {
                for ox in 0..wo {
                    let g = f64::from(dy[((b * cout + co) * ho + oy) * wo + ox]);
                    for ci in 0..cin {
                        for kh in 0..k {
                            let hi = (oy * s + kh) as isize - pad as isize;
                            if hi < 0 || hi >= h as isize {
                                continue;
                            }
                            for kw in 0..k {
                                let wi = (ox * s + kw) as isize - pad as isize;
                                if wi < 0 || wi >= wd as isize {
                                    continue;
                                }
                                let t = g * f64::from(w[((co * cin + ci) * k + kh) * k + kw]);
                                let i = ((b * cin + ci) * h + hi as usize) * wd + wi as usize;
                                exact[i] += t;
                                abs[i] += t.abs();
                            }
                        }
                    }
                }
            }
        }
    }
    Oracle { exact, abs, len: cout * k * k }
}

/// The weight gradient ACCUMULATES into `dw0`, so the oracle starts there and
/// counts it as one more term.
fn oracle_dw(c: &Conv, dy: &[f32], x: &[f32], dw0: &[f32]) -> Oracle {
    let (ho, wo) = (c.ho() as usize, c.wo() as usize);
    let (n, cin, h, wd, cout, k, s, pad) =
        (c.n as usize, c.cin as usize, c.h as usize, c.w as usize, c.cout as usize, c.k as usize, c.s as usize, c.pad as usize);
    let mut exact: Vec<f64> = dw0.iter().map(|&v| f64::from(v)).collect();
    let mut abs: Vec<f64> = dw0.iter().map(|&v| f64::from(v).abs()).collect();
    for b in 0..n {
        for co in 0..cout {
            for oy in 0..ho {
                for ox in 0..wo {
                    let g = f64::from(dy[((b * cout + co) * ho + oy) * wo + ox]);
                    for ci in 0..cin {
                        for kh in 0..k {
                            let hi = (oy * s + kh) as isize - pad as isize;
                            if hi < 0 || hi >= h as isize {
                                continue;
                            }
                            for kw in 0..k {
                                let wi = (ox * s + kw) as isize - pad as isize;
                                if wi < 0 || wi >= wd as isize {
                                    continue;
                                }
                                let t = g * f64::from(x[((b * cin + ci) * h + hi as usize) * wd + wi as usize]);
                                let i = ((co * cin + ci) * k + kh) * k + kw;
                                exact[i] += t;
                                abs[i] += t.abs();
                            }
                        }
                    }
                }
            }
        }
    }
    Oracle { exact, abs, len: n * ho * wo + 1 }
}

/// Every element within `gamma_L * sum |t_i|` of the exact value.
fn check_bound(got: &[f32], o: &Oracle, what: &str) {
    assert_eq!(got.len(), o.exact.len(), "{what}: length");
    let g = gamma(o.len);
    let mut worst = 0.0f64;
    for (i, (&v, (&e, &a))) in got.iter().zip(o.exact.iter().zip(o.abs.iter())).enumerate() {
        let err = (f64::from(v) - e).abs();
        let bound = g * a;
        assert!(v.is_finite() && err <= bound, "{what}: element {i} = {v}, exact {e}, |err| {err:e} > bound {bound:e} (L = {})", o.len);
        if a > 0.0 {
            worst = worst.max(err / (g * a));
        }
    }
    eprintln!("{what}: worst error at {:.3} of the bound (L = {})", worst, o.len);
}

struct Bufs {
    x: Vec<f32>,
    w: Vec<f32>,
    dy: Vec<f32>,
    dw0: Vec<f32>,
}

fn inputs(c: &Conv, seed: u64) -> Bufs {
    let mut r = data::rng::Lcg::new(seed);
    Bufs { x: r.vec(c.x_len()), w: r.vec(c.w_len()), dy: r.vec(c.y_len()), dw0: r.vec_scaled(c.w_len(), 0.5) }
}

/// Run forward, input gradient and weight gradient through the given kernel
/// slots (`dw_native` picks `Gpu::conv2d_dw_steps` over the WGSL slot).
fn run(gpu: &Gpu, c: &Conv, b: &Bufs, fwd: usize, dx: usize, dw: Option<usize>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let p = c.params();
    let x = gpu.storage_init("x", &b.x);
    let w = gpu.storage_init("w", &b.w);
    let dy = gpu.storage_init("dy", &b.dy);
    let dwb = gpu.storage_init("dw", &b.dw0);
    let y = gpu.storage(c.y_len() as u64);
    let dxb = gpu.storage(c.x_len() as u64);
    let mut steps = vec![
        gpu.step(fwd, &[&x, &w, &y], &p, c.y_len() as u32),
        gpu.step(dx, &[&dy, &w, &dxb], &p, c.x_len() as u32),
    ];
    let scratch;
    match dw {
        Some(slot) => steps.push(gpu.step(slot, &[&dy, &x, &dwb], &p, c.w_len() as u32)),
        None => {
            let words = gpu.conv2d_dw_scratch_words(&p).expect("the native weight gradient is offered on a CUDA device");
            scratch = (words > 0).then(|| gpu.storage(words));
            steps.extend(gpu.conv2d_dw_steps(&dy, &x, &dwb, scratch.as_ref(), &p).expect("the native weight gradient serves every dense conv"));
        }
    }
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    (gpu.read(&y, c.y_len()), gpu.read(&dxb, c.x_len()), gpu.read(&dwb, c.w_len()))
}

/// Shapes for the oracle test: every YOLOv8n unit at a small random spatial
/// size (odd and even, non-square, down to a single pixel) and batch 1 or 8,
/// plus the geometry the model never uses but the kernels claim: a 5x5 kernel,
/// stride 3, an even kernel, and a 1x1 stride-2 conv whose odd stride classes
/// receive no tap at all.
fn oracle_shapes() -> Vec<Conv> {
    let mut r = data::rng::Lcg::new(0x5eed_c0f2);
    let mut v = Vec::new();
    for (i, &(cin, cout, k, s, pad, _)) in YOLOV8N.iter().enumerate() {
        // Batch 8 for the narrow units, 1 for the wide ones: the f64 oracle's
        // cost grows with Cin*Cout and the shapes stay a few-second test.
        let n = if cin * cout <= 64 * 64 && i % 2 == 0 { 8 } else { 1 };
        let h = 1 + r.next_u32() % 13;
        let w = 1 + r.next_u32() % 13;
        // A 3x3 pad-1 conv is defined down to a 1x1 map; a 1x1 one at any size.
        v.push(Conv { n, cin, h, w, cout, k, s, pad });
    }
    v.extend([
        // Wide enough in positions that the weight gradient splits (S > 1).
        Conv { n: 8, cin: 16, h: 16, w: 16, cout: 16, k: 3, s: 1, pad: 1 },
        Conv { n: 2, cin: 32, h: 33, w: 31, cout: 16, k: 3, s: 2, pad: 1 },
        Conv { n: 1, cin: 3, h: 1, w: 1, cout: 16, k: 3, s: 2, pad: 1 },
        Conv { n: 2, cin: 5, h: 9, w: 11, cout: 7, k: 5, s: 1, pad: 2 },
        Conv { n: 1, cin: 4, h: 13, w: 10, cout: 9, k: 3, s: 3, pad: 0 },
        Conv { n: 3, cin: 6, h: 8, w: 9, cout: 33, k: 2, s: 2, pad: 0 },
        Conv { n: 2, cin: 70, h: 7, w: 5, cout: 65, k: 1, s: 2, pad: 0 },
    ]);
    v
}

fn the_native_conv_kernels_are_redirected_to_only_on_cuda() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    let p = Conv { n: 1, cin: 16, h: 8, w: 8, cout: 32, k: 3, s: 2, pad: 1 }.params();
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(CONV, &p), None, "only a CUDA device takes a native kernel");
        assert_eq!(gpu.native_kernel_for(CONV_DX, &p), None);
        assert!(gpu.conv2d_dw_split(&p).is_none());
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert_eq!(gpu.native_kernel_for(CONV, &p), Some("conv2d_fwd_f32"));
    assert_eq!(gpu.native_kernel_for(CONV_DX, &p), Some("conv2d_dx_f32"));
    assert!(gpu.conv2d_dw_split(&p).is_some());
    // The reference slots are never redirected, the weight-gradient slot is
    // not redirected either (its native form needs scratch the slot cannot
    // carry), and the grouped/dilated twelve-word ABI is not served.
    assert_eq!(gpu.native_kernel_for(CONV_REF, &p), None);
    assert_eq!(gpu.native_kernel_for(CONV_DW, &p), None);
    let mut gd = p.to_vec();
    gd.splice(8..8, [1, 1]);
    assert_eq!(gpu.native_kernel_for(CONV, &gd), None);
    // A uniform whose output extent contradicts its input is not served.
    let mut bad = p;
    bad[8] += 1;
    assert_eq!(gpu.native_kernel_for(CONV, &bad), None);
}

fn every_yolov8n_conv_shape_is_within_the_fp32_summation_bound_of_an_f64_oracle() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for (i, c) in oracle_shapes().iter().enumerate() {
        let b = inputs(c, 1000 + i as u64);
        let p = c.params();
        let fwd = if direct_3x3(c) { "conv2d_fwd3x3_f32" } else { "conv2d_fwd_f32" };
        assert_eq!(gpu.native_kernel_for(CONV, &p), Some(fwd), "{c:?}");
        assert_eq!(gpu.native_kernel_for(CONV_DX, &p), Some("conv2d_dx_f32"), "{c:?}");
        let (y, dx, dw) = run(&gpu, c, &b, CONV, CONV_DX, None);
        let (y_r, dx_r, dw_r) = run(&gpu, c, &b, CONV_REF, CONV_DX_REF, Some(CONV_DW_REF));
        let of = oracle_fwd(c, &b.x, &b.w);
        let od = oracle_dx(c, &b.dy, &b.w);
        let ow = oracle_dw(c, &b.dy, &b.x, &b.dw0);
        let splits = gpu.conv2d_dw_split(&p).map(|s| s.0).unwrap_or(0);
        let tag = format!("{c:?} (dw slices {splits})");
        check_bound(&y_r, &of, &format!("reference forward {tag}"));
        check_bound(&dx_r, &od, &format!("reference dx {tag}"));
        check_bound(&dw_r, &ow, &format!("reference dw {tag}"));
        check_bound(&y, &of, &format!("native forward {tag}"));
        check_bound(&dx, &od, &format!("native dx {tag}"));
        check_bound(&dw, &ow, &format!("native dw {tag}"));
    }
}

/// Small integers in `[-2, 2]`: every product and every partial sum of every
/// order is an integer far below `2^24`, so fp32 evaluates each output EXACTLY
/// however it is associated, fused or split. Whatever the summation order, a
/// correct kernel produces the same bits as the reference - and a kernel that
/// drops, repeats or misplaces even one term of a half-million-term reduction
/// does not, which a rounding bound that long could not show.
fn integer_inputs(c: &Conv, seed: u64) -> Bufs {
    let mut r = data::rng::Lcg::new(seed);
    let mut ints = |n: usize| (0..n).map(|_| (r.next_u32() % 5) as f32 - 2.0).collect::<Vec<f32>>();
    Bufs { x: ints(c.x_len()), w: ints(c.w_len()), dy: ints(c.y_len()), dw0: ints(c.w_len()) }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn the_real_yolov8n_geometry_is_exact_on_integer_data() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    // Batch 8 at a 256 input: every tile count, stride class and slice count
    // of the real network, at a quarter of its positions. The largest weight-
    // gradient sum is 8 * 128 * 128 terms of magnitude at most 4 - 2^19, well
    // inside fp32's exact integers.
    for (i, &(cin, cout, k, s, pad, side)) in YOLOV8N.iter().enumerate() {
        let c = Conv { n: 8, cin, h: side / 2, w: side / 2, cout, k, s, pad };
        let b = integer_inputs(&c, 7000 + i as u64);
        let (y, dx, dw) = run(&gpu, &c, &b, CONV, CONV_DX, None);
        let (y_r, dx_r, dw_r) = run(&gpu, &c, &b, CONV_REF, CONV_DX_REF, Some(CONV_DW_REF));
        let splits = gpu.conv2d_dw_split(&c.params()).map(|s| s.0).unwrap_or(0);
        let tag = format!("{c:?} (dw slices {splits})");
        assert!(bits(&y) == bits(&y_r), "forward differs: {tag}");
        assert!(bits(&dx) == bits(&dx_r), "dx differs: {tag}");
        assert!(bits(&dw) == bits(&dw_r), "dw differs: {tag}");
    }
}

/// `y = conv(x, w) + bias[co]` from the slot `slot`.
fn run_bias(gpu: &Gpu, c: &Conv, x: &[f32], w: &[f32], bias: &[f32], slot: usize) -> Vec<f32> {
    let xb = gpu.storage_init("x", x);
    let wb = gpu.storage_init("w", w);
    let bb = gpu.storage_init("bias", bias);
    let y = gpu.storage(c.y_len() as u64);
    gpu.submit(&[], &[gpu.step(slot, &[&xb, &wb, &bb, &y], &c.params(), c.y_len() as u32)]);
    gpu.poll_wait();
    gpu.read(&y, c.y_len())
}

/// The biased forward (`conv_bias`, and its register-tiled sibling
/// `conv_bias_reg`, both redirected to the native kernel): within the
/// summation bound of the f64 oracle, the bias counting as one more term, and
/// bit-exact with the WGSL reference on integer data at the detection head's
/// real geometry.
fn the_biased_forward_is_within_the_bound_and_exact_on_integer_data() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        assert_eq!(gpu.native_kernel_for(CONV_BIAS, &Conv { n: 1, cin: 4, h: 4, w: 4, cout: 3, k: 1, s: 1, pad: 0 }.params()), None);
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    for (i, c) in oracle_shapes().iter().enumerate() {
        let b = inputs(c, 2000 + i as u64);
        let bias = data::rng::Lcg::new(3000 + i as u64).vec(c.cout as usize);
        let p = c.params();
        let want = if direct_3x3(c) { "conv2d_bias_fwd3x3_f32" } else { "conv2d_bias_fwd_f32" };
        assert_eq!(gpu.native_kernel_for(CONV_BIAS, &p), Some(want), "{c:?}");
        assert_eq!(gpu.native_kernel_for(CONV_BIAS_REG, &p), Some(want), "{c:?}");
        let mut o = oracle_fwd(c, &b.x, &b.w);
        let plane = (c.ho() * c.wo()) as usize;
        for (k, (e, a)) in o.exact.iter_mut().zip(o.abs.iter_mut()).enumerate() {
            let co = (k / plane) % c.cout as usize;
            *e += f64::from(bias[co]);
            *a += f64::from(bias[co]).abs();
        }
        o.len += 1;
        check_bound(&run_bias(&gpu, c, &b.x, &b.w, &bias, CONV_BIAS_REF), &o, &format!("reference conv_bias {c:?}"));
        check_bound(&run_bias(&gpu, c, &b.x, &b.w, &bias, CONV_BIAS), &o, &format!("native conv_bias {c:?}"));
        check_bound(&run_bias(&gpu, c, &b.x, &b.w, &bias, CONV_BIAS_REG), &o, &format!("native conv_bias_reg {c:?}"));
    }
    // The detection head's biased projections, batch 8 at a 512 input.
    for (i, &(cin, cout, side)) in [(64u32, 64u32, 64u32), (64, 3, 64), (64, 64, 32), (64, 3, 32), (64, 64, 16), (64, 3, 16)].iter().enumerate() {
        let c = Conv { n: 8, cin, h: side, w: side, cout, k: 1, s: 1, pad: 0 };
        let b = integer_inputs(&c, 4000 + i as u64);
        let bias = integer_inputs(&Conv { n: 1, cin: 1, h: 1, w: 1, cout, k: 1, s: 1, pad: 0 }, 5000 + i as u64).dy;
        let reference = run_bias(&gpu, &c, &b.x, &b.w, &bias, CONV_BIAS_REF);
        assert!(bits(&run_bias(&gpu, &c, &b.x, &b.w, &bias, CONV_BIAS)) == bits(&reference), "conv_bias differs: {c:?}");
        assert!(bits(&run_bias(&gpu, &c, &b.x, &b.w, &bias, CONV_BIAS_REG)) == bits(&reference), "conv_bias_reg differs: {c:?}");
    }
}

/// Whether a forward takes the direct 3x3 kernel: stride 1, pad 1, the same
/// extent out as in, and at least one full 64-channel output tile.
fn direct_3x3(c: &Conv) -> bool {
    c.k == 3 && c.s == 1 && c.pad == 1 && c.cout >= 64
}

/// The general implicit-GEMM forward (`conv2d_bias_fwd_f32`), launched by
/// name, for a shape the redirect now sends to the direct kernel.
fn run_bias_implicit_gemm(gpu: &Gpu, c: &Conv, x: &[f32], w: &[f32], bias: &[f32]) -> Vec<f32> {
    static BIND: [backend_api::BindKind; 5] = [
        backend_api::BindKind::Uniform,
        backend_api::BindKind::StorageRead,
        backend_api::BindKind::StorageRead,
        backend_api::BindKind::StorageRead,
        backend_api::BindKind::StorageReadWrite,
    ];
    let k = kernels_cuda::get("conv2d_bias_fwd_f32").expect("registry entry");
    let id = gpu.native_kernel(k, &BIND).expect("the backend compiles the implicit GEMM");
    let bm = [16u32, 32, 64].into_iter().find(|b| c.cout <= *b).unwrap_or(64);
    let blocks = c.cout.div_ceil(bm) * (c.n * c.ho() * c.wo()).div_ceil(k.tile.1);
    let xb = gpu.storage_init("x", x);
    let wb = gpu.storage_init("w", w);
    let bb = gpu.storage_init("bias", bias);
    let y = gpu.storage(c.y_len() as u64);
    gpu.submit(&[], &[gpu.step_native(id, &[&xb, &wb, &bb, &y], &c.params(), blocks).expect("step")]);
    gpu.poll_wait();
    gpu.read(&y, c.y_len())
}

/// The FLUX.2 VAE's convs are 3x3, stride 1, pad 1 and 64-512 channels wide:
/// those take the direct kernel, which sums every output in the implicit
/// GEMM's own order with the same fused multiply-adds - so it must match it
/// BIT FOR BIT, at every tile edge (rows and columns past a 16-wide tile,
/// channel counts past a 64-wide tile and off a 4-channel slice, a batch, a
/// single pixel), and stay within the f64 oracle's bound. A narrower output
/// keeps the implicit GEMM.
fn the_vae_3x3_convs_take_the_direct_kernel_bit_identical_to_the_implicit_gemm() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    let narrow = Conv { n: 1, cin: 64, h: 16, w: 16, cout: 63, k: 3, s: 1, pad: 1 };
    assert_eq!(gpu.native_kernel_for(CONV_BIAS, &narrow.params()), Some("conv2d_bias_fwd_f32"));
    let strided = Conv { n: 1, cin: 64, h: 16, w: 16, cout: 64, k: 3, s: 2, pad: 1 };
    assert_eq!(gpu.native_kernel_for(CONV_BIAS, &strided.params()), Some("conv2d_bias_fwd_f32"));
    let shapes = [
        Conv { n: 1, cin: 64, h: 16, w: 16, cout: 64, k: 3, s: 1, pad: 1 },
        Conv { n: 2, cin: 67, h: 17, w: 33, cout: 65, k: 3, s: 1, pad: 1 },
        Conv { n: 1, cin: 3, h: 20, w: 18, cout: 128, k: 3, s: 1, pad: 1 },
        Conv { n: 1, cin: 130, h: 5, w: 40, cout: 192, k: 3, s: 1, pad: 1 },
        Conv { n: 3, cin: 64, h: 1, w: 1, cout: 64, k: 3, s: 1, pad: 1 },
        Conv { n: 1, cin: 32, h: 31, w: 15, cout: 512, k: 3, s: 1, pad: 1 },
    ];
    for (i, c) in shapes.iter().enumerate() {
        let p = c.params();
        assert_eq!(gpu.native_kernel_for(CONV_BIAS, &p), Some("conv2d_bias_fwd3x3_f32"), "{c:?}");
        assert_eq!(gpu.native_kernel_for(CONV_BIAS_REG, &p), Some("conv2d_bias_fwd3x3_f32"), "{c:?}");
        assert_eq!(gpu.native_kernel_for(CONV, &p), Some("conv2d_fwd3x3_f32"), "{c:?}");
        let b = inputs(c, 7000 + i as u64);
        let bias = data::rng::Lcg::new(8000 + i as u64).vec(c.cout as usize);
        let direct = run_bias(&gpu, c, &b.x, &b.w, &bias, CONV_BIAS);
        let gemm = run_bias_implicit_gemm(&gpu, c, &b.x, &b.w, &bias);
        let first = bits(&direct).iter().zip(bits(&gemm).iter()).position(|(a, g)| a != g);
        assert!(first.is_none(), "direct 3x3 differs from the implicit GEMM at {first:?}: {c:?}");
        let mut o = oracle_fwd(c, &b.x, &b.w);
        let plane = (c.ho() * c.wo()) as usize;
        for (k, (e, a)) in o.exact.iter_mut().zip(o.abs.iter_mut()).enumerate() {
            let co = (k / plane) % c.cout as usize;
            *e += f64::from(bias[co]);
            *a += f64::from(bias[co]).abs();
        }
        o.len += 1;
        check_bound(&direct, &o, &format!("direct conv_bias {c:?}"));
        // The unbiased entry is the same kernel without the epilogue.
        let (y, _, _) = run(&gpu, c, &b, CONV, CONV_DX_REF, Some(CONV_DW_REF));
        check_bound(&y, &oracle_fwd(c, &b.x, &b.w), &format!("direct conv2d {c:?}"));
    }
}

gpu_core::card_tests!(
    the_native_conv_kernels_are_redirected_to_only_on_cuda,
    every_yolov8n_conv_shape_is_within_the_fp32_summation_bound_of_an_f64_oracle,
    the_real_yolov8n_geometry_is_exact_on_integer_data,
    the_biased_forward_is_within_the_bound_and_exact_on_integer_data,
    the_vae_3x3_convs_take_the_direct_kernel_bit_identical_to_the_implicit_gemm,
);
