// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Numerical gradient checks for Moondream's net-new device kernels — each
//! validated in isolation (analytic backward vs a directional finite difference
//! of its own forward) before being composed into the model. Built up
//! kernel-by-kernel as they are added.

#[cfg(test)]
mod tests {
    use data::rng::Rng;
    use gpu_core::Gpu;
    use kernels::{GEGLU_SHIFT, GEGLU_SHIFT_DA, GEGLU_SHIFT_DB};

    // Pipeline: 0 = geglu_shift, 1 = geglu_shift_da, 2 = geglu_shift_db.
    fn gpu() -> Gpu {
        gpu_core::testgpu::dev(&[("geglu_shift", GEGLU_SHIFT), ("geglu_shift_da", GEGLU_SHIFT_DA), ("geglu_shift_db", GEGLU_SHIFT_DB)])
    }

    fn geglu(g: &Gpu, h: &[f32], gg: &[f32]) -> Vec<f32> {
        let n = h.len() as u32;
        let (hb, gb, ob) = (g.storage_init("h", h), g.storage_init("g", gg), g.storage(n as u64));
        g.submit(&[], &[g.step(0, &[&hb, &gb, &ob], &[n], n)]);
        g.read(&ob, n as usize)
    }
    fn geglu_dh(g: &Gpu, dy: &[f32], gg: &[f32], h: &[f32]) -> Vec<f32> {
        let n = h.len() as u32;
        let (a, b, c, o) = (g.storage_init("dy", dy), g.storage_init("g", gg), g.storage_init("h", h), g.storage(n as u64));
        g.submit(&[], &[g.step(1, &[&a, &b, &c, &o], &[n], n)]);
        g.read(&o, n as usize)
    }
    fn geglu_dg(g: &Gpu, dy: &[f32], h: &[f32]) -> Vec<f32> {
        let n = h.len() as u32;
        let (a, c, o) = (g.storage_init("dy", dy), g.storage_init("h", h), g.storage(n as u64));
        g.submit(&[], &[g.step(2, &[&a, &c, &o], &[n], n)]);
        g.read(&o, n as usize)
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
    fn loss(out: &[f32], dy: &[f32]) -> f32 {
        dot(out, dy)
    }

    #[test]
    fn rope_partial_inverse_and_grad() {
        use gpu_core::f;
        use kernels::{ROPE_PARTIAL, ROPE_PARTIAL_BWD};
        let g = gpu_core::testgpu::dev(&[("rope_partial", ROPE_PARTIAL), ("rope_partial_bwd", ROPE_PARTIAL_BWD)]);
        // 4 rows (T=4), 1 head, head_dim 8, rot_dim 4 → rotate first 4 channels.
        let (rows, heads, hd, rot) = (4u32, 1u32, 8u32, 4u32);
        let n = (rows * heads * hd) as usize;
        let par = [rows, heads, hd, heads * hd, 0, rows, f(1.5e6), rot];
        let disp = rows * heads * (rot / 2);
        let rope = |x: &[f32], bwd: bool| -> Vec<f32> {
            let b = g.storage_init("b", x);
            g.submit(&[], &[g.step(if bwd { 1 } else { 0 }, &[&b], &par, disp)]);
            g.read(&b, n)
        };
        let mut rng = Rng::new(41);
        let x: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();

        // Inverse: bwd(fwd(x)) == x (rotated pairs round-trip; tail untouched).
        let rt = rope(&rope(&x, false), true);
        for (a, b) in x.iter().zip(&rt) {
            assert!((a - b).abs() < 1e-4, "rope_partial not invertible: {a} vs {b}");
        }

        // Directional grad of L = <fwd(x), dy>: d_x = bwd(dy) (rotation adjoint).
        let dy: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();
        let v: Vec<f32> = (0..n).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect();
        let dx = rope(&dy, true);
        let an = dot(&dx, &v);
        let eps = 1e-3f32;
        let l = |y: &[f32]| dot(y, &dy);
        let px = |sg: f32| x.iter().zip(&v).map(|(b, d)| b + sg * eps * d).collect::<Vec<f32>>();
        let num = (l(&rope(&px(1.0), false)) - l(&rope(&px(-1.0), false))) / (2.0 * eps);
        let rel = (an - num).abs() / an.abs().max(num.abs()).max(1e-3);
        assert!(rel < 1e-2, "rope_partial grad: analytic {an} vs numeric {num} (rel {rel})");
    }

    #[test]
    fn attn_prefix_mask_pattern() {
        use kernels::ATTN_PREFIX_MASK;
        let g = gpu_core::testgpu::dev(&[("attn_prefix_mask", ATTN_PREFIX_MASK)]);
        // T=4, prefix P=2, 1 head. allow = (i<2 && j<2) || (j<=i).
        let (bsz, heads, t, p) = (1u32, 1u32, 4u32, 2u32);
        let n = (bsz * heads * t * t) as usize;
        let sb = g.storage_init("s", &vec![1.0f32; n]);
        g.submit(&[], &[g.step(0, &[&sb], &[bsz, heads, t, p], n as u32)]);
        let s = g.read(&sb, n);
        let at = |i: u32, j: u32| s[(i * t + j) as usize];
        // allowed (unchanged at 1.0): prefix pairs + causal
        for (i, j) in [(0u32, 1u32), (1, 0), (3, 0), (2, 2), (0, 0)] {
            assert!((at(i, j) - 1.0).abs() < 1e-6, "({i},{j}) should be allowed");
        }
        // masked (large negative): non-prefix, non-causal
        for (i, j) in [(0u32, 2u32), (0, 3), (1, 2), (1, 3), (2, 3)] {
            assert!(at(i, j) < -1.0e29, "({i},{j}) should be masked");
        }
    }

    #[test]
    fn adaptive_avgpool2d_grads_match_finite_differences() {
        use kernels::{ADAPTIVE_AVGPOOL2D, ADAPTIVE_AVGPOOL2D_DX};
        let g = gpu_core::testgpu::dev(&[("adaptive_avgpool2d", ADAPTIVE_AVGPOOL2D), ("adaptive_avgpool2d_dx", ADAPTIVE_AVGPOOL2D_DX)]);
        // 4×4 → 3×3: non-integer ratio → overlapping bins (the general case).
        let (nn, cc, h, w, oh, ow) = (1u32, 1u32, 4u32, 4u32, 3u32, 3u32);
        let nx = (nn * cc * h * w) as usize;
        let ny = (nn * cc * oh * ow) as usize;
        let par = [nn, cc, h, w, oh, ow];
        let pool = |x: &[f32]| -> Vec<f32> {
            let (xb, yb) = (g.storage_init("x", x), g.storage(ny as u64));
            g.submit(&[], &[g.step(0, &[&xb, &yb], &par, ny as u32)]);
            g.read(&yb, ny)
        };
        let pool_dx = |dy: &[f32]| -> Vec<f32> {
            let (a, o) = (g.storage_init("dy", dy), g.storage(nx as u64));
            g.submit(&[], &[g.step(1, &[&a, &o], &par, nx as u32)]);
            g.read(&o, nx)
        };
        let mut rng = Rng::new(31);
        let x: Vec<f32> = (0..nx).map(|_| rng.next_f32() - 0.5).collect();
        let dy: Vec<f32> = (0..ny).map(|_| rng.next_f32() - 0.5).collect();
        let v: Vec<f32> = (0..nx).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect();

        let dx = pool_dx(&dy);
        let an = dot(&dx, &v);
        let eps = 1e-3f32;
        let l = |y: &[f32]| dot(y, &dy);
        let px = |sg: f32| x.iter().zip(&v).map(|(b, d)| b + sg * eps * d).collect::<Vec<f32>>();
        let num = (l(&pool(&px(1.0))) - l(&pool(&px(-1.0)))) / (2.0 * eps);
        let rel = (an - num).abs() / an.abs().max(num.abs()).max(1e-3);
        assert!(rel < 1e-2, "adaptive_avgpool2d dx: analytic {an} vs numeric {num} (rel {rel})");
    }

    #[test]
    fn tau_scale_grads_match_finite_differences() {
        use kernels::{TAU_SCALE, TAU_SCALE_DS};
        let g = gpu_core::testgpu::dev(&[("tau_scale", TAU_SCALE), ("tau_scale_ds", TAU_SCALE_DS)]);
        let (rows, heads, hd) = (4u32, 2u32, 8u32);
        let n = (rows * heads * hd) as usize;
        let ns = (heads * rows) as usize;
        // out[row,h,d] = in[row,h,d] * s[h,row]
        let scale = |inp: &[f32], s: &[f32]| -> Vec<f32> {
            let (ib, sb, ob) = (g.storage_init("i", inp), g.storage_init("s", s), g.storage(n as u64));
            g.submit(&[], &[g.step(0, &[&ib, &sb, &ob], &[rows, heads, hd], n as u32)]);
            g.read(&ob, n)
        };
        let ds_of = |dout: &[f32], inp: &[f32]| -> Vec<f32> {
            let (a, b, o) = (g.storage_init("do", dout), g.storage_init("i", inp), g.storage(ns as u64));
            g.submit(&[], &[g.step(1, &[&a, &b, &o], &[rows, heads, hd], ns as u32)]);
            g.read(&o, ns)
        };
        let mut rng = Rng::new(21);
        let inp: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();
        let s: Vec<f32> = (0..ns).map(|_| rng.next_f32() - 0.5).collect();
        let dy: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();
        let vin: Vec<f32> = (0..n).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect();
        let vs: Vec<f32> = (0..ns).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect();

        let d_in = scale(&dy, &s); // d_in = dy * s (same op as forward)
        let d_s = ds_of(&dy, &inp);
        let an_in = dot(&d_in, &vin);
        let an_s = dot(&d_s, &vs);

        let eps = 1e-3f32;
        let l = |o: &[f32]| dot(o, &dy);
        let pin = |sg: f32| inp.iter().zip(&vin).map(|(b, d)| b + sg * eps * d).collect::<Vec<f32>>();
        let ps = |sg: f32| s.iter().zip(&vs).map(|(b, d)| b + sg * eps * d).collect::<Vec<f32>>();
        let num_in = (l(&scale(&pin(1.0), &s)) - l(&scale(&pin(-1.0), &s))) / (2.0 * eps);
        let num_s = (l(&scale(&inp, &ps(1.0))) - l(&scale(&inp, &ps(-1.0)))) / (2.0 * eps);
        for (name, a, nn) in [("d_in", an_in, num_in), ("d_s", an_s, num_s)] {
            let rel = (a - nn).abs() / a.abs().max(nn.abs()).max(1e-3);
            assert!(rel < 1e-2, "tau_scale {name}: analytic {a} vs numeric {nn} (rel {rel})");
        }
    }

    #[test]
    fn geglu_shift_grads_match_finite_differences() {
        let g = gpu();
        let n = 32usize;
        let mut rng = Rng::new(11);
        let r = |rng: &mut Rng| (0..n).map(|_| rng.next_f32() - 0.5).collect::<Vec<f32>>();
        let (h, gg, dy) = (r(&mut rng), r(&mut rng), r(&mut rng));
        let (vh, vg): (Vec<f32>, Vec<f32>) = (
            (0..n).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect(),
            (0..n).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect(),
        );
        // Analytic directional derivatives of L = <geglu(h,g), dy>.
        let dh = geglu_dh(&g, &dy, &gg, &h);
        let dg = geglu_dg(&g, &dy, &h);
        let (an_h, an_g) = (dot(&dh, &vh), dot(&dg, &vg));

        let eps = 1e-3f32;
        let perturb = |base: &[f32], v: &[f32], s: f32| base.iter().zip(v).map(|(b, d)| b + s * eps * d).collect::<Vec<f32>>();
        // d/dh
        let num_h = (loss(&geglu(&g, &perturb(&h, &vh, 1.0), &gg), &dy) - loss(&geglu(&g, &perturb(&h, &vh, -1.0), &gg), &dy)) / (2.0 * eps);
        // d/dg
        let num_g = (loss(&geglu(&g, &h, &perturb(&gg, &vg, 1.0)), &dy) - loss(&geglu(&g, &h, &perturb(&gg, &vg, -1.0)), &dy)) / (2.0 * eps);

        for (name, a, nnum) in [("dh", an_h, num_h), ("dg", an_g, num_g)] {
            let rel = (a - nnum).abs() / a.abs().max(nnum.abs()).max(1e-3);
            assert!(rel < 1e-2, "geglu_shift {name}: analytic {a} vs numeric {nnum} (rel {rel})");
        }
    }
}
