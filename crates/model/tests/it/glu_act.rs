// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec of the declared expert activation (`kernels::GLU_ACT` / `GLU_ACT_BWD`): for every activation and layout the
//! kernels equal the host reference `GatedAct::{apply, grad}`, with gate and up read from where the layout puts them.
//! The default SwiGLU over separate buffers is `silu_mul`, `silu_bwd_da` and `silu_bwd_db` to the bit.
//!
//! Swedish Embedded AB implements solutions for serving and training mixture-of-experts models whose expert
//! activation differs from the common SwiGLU recipe. If your team needs expertise in gated expert FFNs then you can
//! procure our services by sending an email to info@swedishembedded.com.

use data::rng::{random_seed, Lcg};
use model::glu::{glu_act_bwd_step, glu_act_step, GatedAct, GluIds, GluLayout, GluSpec};

const PIPES: &[(&str, &str)] = &[
    ("glu_act", kernels::GLU_ACT),
    ("glu_act_bwd", kernels::GLU_ACT_BWD),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
];

const ROWS: usize = 3;
const FF: usize = 70; // not a multiple of the workgroup size

fn position(layout: GluLayout, row: usize, j: usize) -> (usize, usize) {
    match layout {
        GluLayout::Separate => (row * FF + j, row * FF + j),
        GluLayout::Concat => (row * 2 * FF + j, row * 2 * FF + FF + j),
        GluLayout::Interleaved => (row * 2 * FF + 2 * j, row * 2 * FF + 2 * j + 1),
    }
}

#[test]
fn kernels_match_the_host_reference_in_every_layout() {
    let g = gpu_core::testgpu::dev(PIPES);
    let ids = GluIds { act: g.kernel_index("glu_act").expect("registered"), act_bwd: g.kernel_index("glu_act_bwd").expect("registered") };
    let mut rng = Lcg::new(random_seed());
    let dh = rng.vec_scaled(ROWS * FF, 1.0);
    let dhb = g.storage_init("dh", &dh);
    for act in [GatedAct::SwiGlu, GatedAct::SoftcapSigmoid { gate_cap: 0.7, up_cap: 1.3 }] {
        for layout in [GluLayout::Separate, GluLayout::Concat, GluLayout::Interleaved] {
            let spec = GluSpec { act, layout };
            let width = if spec.is_fused() { 2 * FF } else { FF };
            let a = rng.vec_scaled(ROWS * width, 2.0);
            let b = rng.vec_scaled(ROWS * FF, 2.0);
            let (ab, bb) = (g.storage_init("a", &a), g.storage_init("b", &b));
            let (h, da, db) = (g.storage((ROWS * FF) as u64), g.storage((ROWS * width) as u64), g.storage(if spec.is_fused() { 1 } else { (ROWS * FF) as u64 }));
            let up = (!spec.is_fused()).then_some(&bb);
            g.submit(&[], &[glu_act_step(&g, &ids, &spec, ROWS as u32, FF as u32, &ab, up, &h), glu_act_bwd_step(&g, &ids, &spec, ROWS as u32, FF as u32, &ab, up, &dhb, &da, &db)]);
            let (h, da, db) = (g.read(&h, ROWS * FF), g.read(&da, ROWS * width), g.read(&db, if spec.is_fused() { 1 } else { ROWS * FF }));
            for row in 0..ROWS {
                for j in 0..FF {
                    let (gi, ui) = position(layout, row, j);
                    let (gv, uv) = (a[gi], if spec.is_fused() { a[ui] } else { b[ui] });
                    let i = row * FF + j;
                    let (dg, du) = act.grad(gv, uv);
                    let what = format!("{act:?} {layout:?} row {row} j {j}");
                    assert!((h[i] - act.apply(gv, uv)).abs() <= 1e-5, "{what}: h {} vs {}", h[i], act.apply(gv, uv));
                    assert!((da[gi] - dh[i] * dg).abs() <= 1e-5, "{what}: dgate {} vs {}", da[gi], dh[i] * dg);
                    let got_du = if spec.is_fused() { da[ui] } else { db[ui] };
                    assert!((got_du - dh[i] * du).abs() <= 1e-5, "{what}: dup {got_du} vs {}", dh[i] * du);
                }
            }
        }
    }
}

#[test]
fn default_swiglu_is_silu_mul_and_its_backward_to_the_bit() {
    let g = gpu_core::testgpu::dev(PIPES);
    let ids = GluIds { act: g.kernel_index("glu_act").expect("registered"), act_bwd: g.kernel_index("glu_act_bwd").expect("registered") };
    let (silu_mul, da_k, db_k) = (g.kernel_index("silu_mul").unwrap(), g.kernel_index("silu_bwd_da").unwrap(), g.kernel_index("silu_bwd_db").unwrap());
    let mut rng = Lcg::new(random_seed());
    let n = ROWS * FF;
    let (a, b, dh) = (rng.vec_scaled(n, 3.0), rng.vec_scaled(n, 3.0), rng.vec_scaled(n, 1.0));
    let (ab, bb, dhb) = (g.storage_init("a", &a), g.storage_init("b", &b), g.storage_init("dh", &dh));
    let (h, da, db) = (g.storage(n as u64), g.storage(n as u64), g.storage(n as u64));
    let (h0, da0, db0) = (g.storage(n as u64), g.storage(n as u64), g.storage(n as u64));
    let spec = GluSpec::default();
    g.submit(&[], &[
        glu_act_step(&g, &ids, &spec, ROWS as u32, FF as u32, &ab, Some(&bb), &h),
        glu_act_bwd_step(&g, &ids, &spec, ROWS as u32, FF as u32, &ab, Some(&bb), &dhb, &da, &db),
        g.step(silu_mul, &[&ab, &bb, &h0], &[n as u32], n as u32),
        g.step(da_k, &[&ab, &bb, &dhb, &da0], &[n as u32], n as u32),
        g.step(db_k, &[&ab, &dhb, &db0], &[n as u32], n as u32),
    ]);
    for (got, want, what) in [(&h, &h0, "h"), (&da, &da0, "d_gate"), (&db, &db0, "d_up")] {
        let (x, y) = (g.read(got, n), g.read(want, n));
        assert!(x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()), "{what} differs from the existing SwiGLU kernels");
    }
}
