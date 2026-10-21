// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded attention training kernels for its clients. If your team needs
// expertise in long-sequence fine-tuning of diffusion and language models on a single GPU then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The flash trio's bidirectional form (`block::flash_bidir_kernels`: `flash_attn_causal_gqa*` specialised with
//! `CAUSAL = 0`, optionally `UNIT_SCALE = 1` for a model that folded `1/sqrt(head_dim)` into its QK norm) as a
//! TRAINING attention: forward context, row log-sum-exp and `dq`/`dk`/`dv` against an f64 host reference that
//! materialises the whole `[T, T]` score matrix, so the device trio is shown to need none of it. Odd `T`, head
//! widths 40/64/128/256, MHA and GQA, with and without the unit scale.
//!
//! The kernels are barrier-cooperative, so these run on the pooled GPU test device (the CPU backend cannot run
//! them, which is why `FlashGqaIds::supported` is false there).

use data::rng::Lcg;
use gpu_core::Gpu;
use model::block::{flash_bidir_kernels, flash_gqa_causal_bwd, flash_gqa_causal_fwd, FlashGqaIds, Gqa};

struct Case {
    b: usize,
    t: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    unit_scale: bool,
}

/// `(ctx, lse, dq, dk, dv)` of bidirectional attention from `q [b*t, nh*hd]`, `k`/`v [b*t, nkv*hd]` and the
/// context gradient, in f64 over the full score matrix.
fn reference(c: &Case, q: &[f32], k: &[f32], v: &[f32], d_ctx: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let Case { b, t, nh, nkv, hd, unit_scale } = *c;
    let (group, scale) = (nh / nkv, if unit_scale { 1.0 } else { 1.0 / (hd as f64).sqrt() });
    let (qr, kr) = (nh * hd, nkv * hd);
    let (mut ctx, mut lse) = (vec![0f32; b * t * qr], vec![0f32; b * nh * t]);
    let (mut dq, mut dk, mut dv) = (vec![0f32; b * t * qr], vec![0f32; b * t * kr], vec![0f32; b * t * kr]);
    for bi in 0..b {
        for h in 0..nh {
            let kvh = h / group;
            let at = |buf: &[f32], row: usize, width: usize, head: usize| -> Vec<f64> { buf[(bi * t + row) * width + head * hd..][..hd].iter().map(|&x| x as f64).collect() };
            for i in 0..t {
                let qi = at(q, i, qr, h);
                let s: Vec<f64> = (0..t).map(|j| scale * qi.iter().zip(at(k, j, kr, kvh)).map(|(a, b)| a * b).sum::<f64>()).collect();
                let m = s.iter().fold(f64::MIN, |a, &x| a.max(x));
                let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
                lse[(bi * nh + h) * t + i] = (m + z.ln()) as f32;
                let p: Vec<f64> = s.iter().map(|x| (x - m).exp() / z).collect();
                let doi = at(d_ctx, i, qr, h);
                // dP_ij = dO_i . v_j ; D_i = sum_j P_ij dP_ij
                let dp: Vec<f64> = (0..t).map(|j| doi.iter().zip(at(v, j, kr, kvh)).map(|(a, b)| a * b).sum()).collect();
                let dsum: f64 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
                for j in 0..t {
                    let (vj, kj) = (at(v, j, kr, kvh), at(k, j, kr, kvh));
                    let ds = p[j] * (dp[j] - dsum);
                    for d in 0..hd {
                        ctx[(bi * t + i) * qr + h * hd + d] += (p[j] * vj[d]) as f32;
                        dq[(bi * t + i) * qr + h * hd + d] += (scale * ds * kj[d]) as f32;
                        dk[(bi * t + j) * kr + kvh * hd + d] += (scale * ds * qi[d]) as f32;
                        dv[(bi * t + j) * kr + kvh * hd + d] += (p[j] * doi[d]) as f32;
                    }
                }
            }
        }
    }
    (ctx, lse, dq, dk, dv)
}

fn assert_close(what: &str, got: &[f32], want: &[f32], rel: f32) {
    let scale = want.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
    let worst = got.iter().zip(want).fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
    assert!(got.iter().all(|x| x.is_finite()), "{what}: non-finite output");
    assert!(worst <= rel * scale.max(1.0), "{what}: worst abs diff {worst:e} at scale {scale:e}");
}

fn ids_for(g: &Gpu, unit_scale: bool) -> FlashGqaIds {
    let at = |name: &str| g.kernel_index(name).unwrap_or_else(|| panic!("{name} is not registered"));
    let [fwd, bwd_dq, bwd_dkv] = flash_bidir_kernels(unit_scale).map(|(name, _)| at(name));
    FlashGqaIds { fwd, bwd_dq, bwd_dkv, native: false }
}

#[test]
fn the_bidirectional_flash_trio_matches_the_full_matrix_reference() {
    let mut all: Vec<(&str, &str)> = Vec::new();
    for unit_scale in [false, true] {
        all.extend(flash_bidir_kernels(unit_scale));
    }
    let g = gpu_core::testgpu::dev(Box::leak(all.into_boxed_slice()));
    let cases = [
        Case { b: 1, t: 37, nh: 2, nkv: 2, hd: 40, unit_scale: false },
        Case { b: 2, t: 70, nh: 4, nkv: 2, hd: 64, unit_scale: false },
        Case { b: 1, t: 53, nh: 2, nkv: 2, hd: 128, unit_scale: true },
        Case { b: 1, t: 19, nh: 6, nkv: 3, hd: 256, unit_scale: true },
    ];
    for (n, c) in cases.iter().enumerate() {
        assert!(FlashGqaIds::supported(&g, c.hd as u32), "case {n}: the test device must run the flash trio at head_dim {}", c.hd);
        let (qr, kr) = (c.nh * c.hd, c.nkv * c.hd);
        let mut rng = Lcg::random();
        let (q, k, v, d_ctx) = (rng.vec_scaled(c.b * c.t * qr, 1.0), rng.vec_scaled(c.b * c.t * kr, 1.0), rng.vec_scaled(c.b * c.t * kr, 1.0), rng.vec_scaled(c.b * c.t * qr, 1.0));
        let (want_ctx, want_lse, want_dq, want_dk, want_dv) = reference(c, &q, &k, &v, &d_ctx);

        let ga = Gqa { b: c.b as u32, t: c.t as u32, n_heads: c.nh as u32, n_kv_heads: c.nkv as u32, head_dim: c.hd as u32 };
        let ids = ids_for(&g, c.unit_scale);
        let (qb, kb, vb, dcb) = (g.storage_init("q", &q), g.storage_init("k", &k), g.storage_init("v", &v), g.storage_init("dctx", &d_ctx));
        let ctx = g.storage((c.b * c.t * qr) as u64);
        let (lse, dsum) = (g.storage((c.b * c.nh * c.t) as u64), g.storage((c.b * c.nh * c.t) as u64));
        let (dq, dk, dv) = (g.storage((c.b * c.t * qr) as u64), g.storage((c.b * c.t * kr) as u64), g.storage((c.b * c.t * kr) as u64));
        let mut steps = vec![flash_gqa_causal_fwd(&g, &ids, &ga, &qb, &kb, &vb, &ctx, &lse)];
        steps.extend(flash_gqa_causal_bwd(&g, &ids, &ga, &qb, &kb, &vb, &ctx, &lse, &dcb, &dsum, &dq, &dk, &dv));
        g.submit(&[], &steps);

        let tag = format!("case {n} (b={} t={} nh={} nkv={} hd={} unit_scale={})", c.b, c.t, c.nh, c.nkv, c.hd, c.unit_scale);
        assert_close(&format!("{tag} ctx"), &g.read(&ctx, want_ctx.len()), &want_ctx, 2e-4);
        assert_close(&format!("{tag} lse"), &g.read(&lse, want_lse.len()), &want_lse, 2e-4);
        assert_close(&format!("{tag} dq"), &g.read(&dq, want_dq.len()), &want_dq, 2e-4);
        assert_close(&format!("{tag} dk"), &g.read(&dk, want_dk.len()), &want_dk, 2e-4);
        assert_close(&format!("{tag} dv"), &g.read(&dv, want_dv.len()), &want_dv, 2e-4);
    }
}

/// The three specialisations are three different kernels registered under three different names, in
/// forward, `dq`, `dkv` order (a name derived from the source's address once collapsed them in a release build).
#[test]
fn the_specialised_kernels_have_distinct_names_in_trio_order() {
    for unit_scale in [false, true] {
        let [fwd, dq, dkv] = flash_bidir_kernels(unit_scale).map(|(name, _)| name);
        assert!(fwd.starts_with("flash_attn_causal_gqa#") && dq.starts_with("flash_attn_causal_gqa_bwd_dq#") && dkv.starts_with("flash_attn_causal_gqa_bwd_dkv#"), "{fwd} {dq} {dkv}");
        assert!(fwd.contains("CAUSAL=0") && fwd.contains(&format!("UNIT_SCALE={}", u32::from(unit_scale))));
    }
    assert_ne!(flash_bidir_kernels(false)[0].0, flash_bidir_kernels(true)[0].0);
}
