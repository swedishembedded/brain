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
        Gpu::new_cpu(&[("geglu_shift", GEGLU_SHIFT), ("geglu_shift_da", GEGLU_SHIFT_DA), ("geglu_shift_db", GEGLU_SHIFT_DB)])
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
    fn tau_scale_grads_match_finite_differences() {
        use kernels::{TAU_SCALE, TAU_SCALE_DS};
        let g = Gpu::new_cpu(&[("tau_scale", TAU_SCALE), ("tau_scale_ds", TAU_SCALE_DS)]);
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
