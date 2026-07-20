// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Block-level gradcheck for the ViT builder: analytic grads from
//! `vit_block_bwd` vs central finite differences of the cached forward, on a
//! tiny trunk-like block (2 frames × 8 tokens, dim 32, QK-norm + 2D RoPE +
//! LayerScale all ON) and a DINOv2-like block (hooks off). All ops are
//! smooth, so finite differences are a valid oracle here (unlike the
//! rasterizer's truncated gaussians). CPU backend.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu};
use model::vit::{
    vit_block_bwd, vit_block_fwd, vit_block_fwd_cached, QkNorm, RopeTables, VitBlockCache,
    VitBlockGrads, VitBlockWeights, VitBwdIds, VitBwdScratch, VitKernelIds, VitScratch, VitShape,
};

const PIPES: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("gelu_erf", kernels::GELU_ERF),
    ("scale_chan", kernels::SCALE_CHAN),
    ("add2", kernels::ADD2),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("ln_head", kernels::LN_HEAD),
    ("rope2d", kernels::ROPE2D),
    ("layernorm_dx", kernels::LAYERNORM_DX),
    ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
    ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("bias_grad", kernels::BIAS_GRAD),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    ("scale_chan_dg", kernels::SCALE_CHAN_DG),
    ("ln_head_dx", kernels::LN_HEAD_DX),
    ("ln_head_dgb", kernels::LN_HEAD_DGB),
    ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
    ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS),
    ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
    ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS),
    ("ln_stats", kernels::LN_STATS),
    ("region_copy", kernels::REGION_COPY),
    ("axpy", kernels::AXPY),
];

fn ids() -> (VitKernelIds, VitBwdIds) {
    (
        VitKernelIds {
            layernorm: 0,
            matmul: 1,
            bias_add: 2,
            gelu_erf: 3,
            scale_chan: 4,
            add2: 5,
            attn_scores_cross: 6,
            attn_softmax_cross: 7,
            attn_apply_cross: 8,
            ln_head: 9,
            rope2d: 10,
        },
        VitBwdIds {
            layernorm_dx: 11,
            ln_dgamma: 12,
            ln_dbeta: 13,
            matmul_dx: 14,
            matmul_dw: 15,
            bias_grad: 16,
            gelu_erf_bwd: 17,
            scale_chan_dg: 18,
            ln_head_dx: 19,
            ln_head_dgb: 20,
            attn_bwd_dscores_cross: 21,
            attn_bwd_dv_cross: 22,
            attn_bwd_dq_cross: 23,
            attn_bwd_dk_cross: 24,
            ln_stats: 25,
            region_copy: 26,
            axpy: 27,
        },
    )
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

const C: usize = 32;
const HEADS: usize = 2; // head_dim 16, rope quarter 4
const M: usize = 64;
const SPAN: usize = 8;
const ROWS: usize = 16; // 2 frames

fn param_shapes(qk_norm: bool, ls: bool) -> Vec<(&'static str, usize)> {
    let hd = C / HEADS;
    let mut v = vec![
        ("norm1_w", C),
        ("norm1_b", C),
        ("qkv_w", 3 * C * C),
        ("qkv_b", 3 * C),
        ("proj_w", C * C),
        ("proj_b", C),
        ("norm2_w", C),
        ("norm2_b", C),
        ("fc1_w", M * C),
        ("fc1_b", M),
        ("fc2_w", C * M),
        ("fc2_b", C),
    ];
    if qk_norm {
        v.extend([("q_norm_w", hd), ("q_norm_b", hd), ("k_norm_w", hd), ("k_norm_b", hd)]);
    }
    if ls {
        v.extend([("ls1", C), ("ls2", C)]);
    }
    v
}

struct Setup {
    weights: HashMap<&'static str, Vec<f32>>,
    x: Vec<f32>,
    wloss: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

fn setup(qk_norm: bool, ls: bool, seed: u64) -> Setup {
    let mut r = Lcg(seed);
    let mut weights = HashMap::new();
    for (name, n) in param_shapes(qk_norm, ls) {
        let mut v = r.vec(n);
        if name.ends_with("_w") && name.contains("norm") || name.starts_with("ls") {
            // norms/scales near 1
            for x in v.iter_mut() {
                *x = 1.0 + 0.3 * *x;
            }
        } else if name.ends_with("_w") {
            // fan-in-ish scaling keeps scores O(1): saturated softmax makes
            // finite differences noise-dominated (loss >> perturbation delta)
            for x in v.iter_mut() {
                *x *= 0.35;
            }
        }
        weights.insert(name, v);
    }
    Setup {
        weights,
        x: r.vec(ROWS * C),
        wloss: r.vec(ROWS * C),
        cos: r.vec(SPAN * C / HEADS / 2).iter().map(|v| (v * 2.0).cos()).collect(),
        sin: r.vec(SPAN * C / HEADS / 2).iter().map(|v| (v * 2.0).sin()).collect(),
    }
}

/// Run the cached forward with the given weight overrides; return loss and
/// (optionally) the analytic grads.
fn run(
    g: &Gpu,
    su: &Setup,
    qk_norm: bool,
    ls: bool,
    with_bwd: bool,
) -> (f64, HashMap<&'static str, Vec<f32>>, Vec<f32>) {
    let (kf, kb) = ids();
    let sh = VitShape { dim: C as u32, heads: HEADS as u32, mlp: M as u32, eps: 1e-5 };
    let hd = C / HEADS;
    let b = |v: &[f32]| g.storage_init("w", v);
    let wb: HashMap<&'static str, DeviceBuffer> =
        su.weights.iter().map(|(k, v)| (*k, b(v))).collect();
    let cos = b(&su.cos);
    let sin = b(&su.sin);
    let rope = || RopeTables { cos: &cos, sin: &sin, tmod: SPAN as u32 };
    let w = VitBlockWeights {
        norm1_w: &wb["norm1_w"],
        norm1_b: &wb["norm1_b"],
        qkv_w: &wb["qkv_w"],
        qkv_b: &wb["qkv_b"],
        qk_norm: if qk_norm {
            Some(QkNorm { q_w: &wb["q_norm_w"], q_b: &wb["q_norm_b"], k_w: &wb["k_norm_w"], k_b: &wb["k_norm_b"] })
        } else {
            None
        },
        rope: if qk_norm { Some(rope()) } else { None },
        proj_w: &wb["proj_w"],
        proj_b: &wb["proj_b"],
        ls1: if ls { Some(&wb["ls1"]) } else { None },
        norm2_w: &wb["norm2_w"],
        norm2_b: &wb["norm2_b"],
        fc1_w: &wb["fc1_w"],
        fc1_b: &wb["fc1_b"],
        fc2_w: &wb["fc2_w"],
        fc2_b: &wb["fc2_b"],
        ls2: if ls { Some(&wb["ls2"]) } else { None },
    };
    let cache = VitBlockCache::new(g, &sh, ROWS as u32, SPAN as u32);
    g.write(&cache.x_in, cast(&su.x));
    let x_out = g.storage((ROWS * C) as u64);
    let scr_tmp = g.storage((ROWS * C) as u64);
    let scores = g.storage((HEADS * SPAN * SPAN) as u64);
    let spans = [(0u32, SPAN as u32), (SPAN as u32, SPAN as u32)];
    let mut steps = Vec::new();
    vit_block_fwd_cached(g, &kf, &kb, &sh, &w, &cache, &x_out, ROWS as u32, &spans, &scr_tmp, &scores, &mut steps);
    g.submit(&[&cache.qkv], &steps);
    let y = g.read(&x_out, ROWS * C);
    if std::env::var("VIT_BWD_DEBUG").is_ok() {
        // cross-check the cached forward against the verified in-place one
        let scr = VitScratch::new(g, &sh, ROWS as u32, SPAN as u32, SPAN as u32);
        let xb = g.storage_init("xip", &su.x);
        let mut st2 = Vec::new();
        vit_block_fwd(g, &kf, &sh, &w, &xb, ROWS as u32, &spans, SPAN as u32, &scr, &mut st2);
        g.submit(&[], &st2);
        let y2 = g.read(&xb, ROWS * C);
        let md = y.iter().zip(&y2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("cached-vs-inplace fwd max abs diff: {md}");
    }
    let loss: f64 = y.iter().zip(&su.wloss).map(|(a, b)| (a * b) as f64).sum();

    if !with_bwd {
        return (loss, HashMap::new(), Vec::new());
    }
    let gb: HashMap<&'static str, DeviceBuffer> = su
        .weights
        .iter()
        .map(|(k, v)| (*k, g.storage(v.len() as u64)))
        .collect();
    let gr = VitBlockGrads {
        norm1_w: &gb["norm1_w"],
        norm1_b: &gb["norm1_b"],
        qkv_w: &gb["qkv_w"],
        qkv_b: &gb["qkv_b"],
        q_norm_w: gb.get("q_norm_w"),
        q_norm_b: gb.get("q_norm_b"),
        k_norm_w: gb.get("k_norm_w"),
        k_norm_b: gb.get("k_norm_b"),
        proj_w: &gb["proj_w"],
        proj_b: &gb["proj_b"],
        ls1: gb.get("ls1"),
        norm2_w: &gb["norm2_w"],
        norm2_b: &gb["norm2_b"],
        fc1_w: &gb["fc1_w"],
        fc1_b: &gb["fc1_b"],
        fc2_w: &gb["fc2_w"],
        fc2_b: &gb["fc2_b"],
        ls2: gb.get("ls2"),
    };
    let d_out = g.storage_init("dout", &su.wloss);
    let d_x_in = g.storage((ROWS * C) as u64);
    let sb = VitBwdScratch::new(g, &sh, ROWS as u32, SPAN as u32);
    let mut steps = Vec::new();
    vit_block_bwd(g, &kf, &kb, &sh, &w, &gr, &cache, &d_out, &d_x_in, ROWS as u32, &spans, &sb, &mut steps);
    let clears: Vec<&DeviceBuffer> = gb.values().chain([&sb.d_qkv, &sb.d_qkv_pre]).collect();
    g.submit(&clears, &steps);
    if std::env::var("VIT_BWD_DEBUG").is_ok() {
        let dq = g.read(&sb.d_qkv, ROWS * 3 * C);
        let mut norms = [0.0f64; 3];
        for row in 0..ROWS {
            for reg in 0..3 {
                for i in 0..C {
                    let v = dq[row * 3 * C + reg * C + i] as f64;
                    norms[reg] += v * v;
                }
            }
        }
        eprintln!("d_qkv region rms: q {:.4} k {:.4} v {:.4}",
            (norms[0] / (ROWS*C) as f64).sqrt(), (norms[1] / (ROWS*C) as f64).sqrt(), (norms[2] / (ROWS*C) as f64).sqrt());
        let ds = g.read(&sb.dscores, HEADS * SPAN * SPAN);
        eprintln!("dscores rms {:.4}", (ds.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / ds.len() as f64).sqrt());
    }
    let grads: HashMap<&'static str, Vec<f32>> =
        gb.iter().map(|(k, buf)| (*k, g.read(buf, su.weights[k].len()))).collect();
    let dx = g.read(&d_x_in, ROWS * C);
    let _ = hd;
    (loss, grads, dx)
}

fn cast(v: &[f32]) -> &[u32] {
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u32, v.len()) }
}

fn gradcheck(qk_norm: bool, ls: bool, seed: u64, cfg_key: &str) {
    let g = Gpu::new_cpu(PIPES);
    let base = setup(qk_norm, ls, seed);
    let (_, grads, dx) = run(&g, &base, qk_norm, ls, true);

    let golden: serde_json::Value =
        serde_json::from_str(include_str!("golden/vit_gradcheck.json")).unwrap();
    let gold = &golden[cfg_key];
    let get = |k: &str| -> Vec<f32> {
        gold[k].as_array().unwrap_or_else(|| panic!("golden missing {k}"))
            .iter().map(|v| v.as_f64().unwrap() as f32).collect()
    };
    let mut errs: Vec<String> = Vec::new();
    let mut checked = 0usize;
    let mut cmp = |name: &str, got: &[f32], want: &[f32]| {
        assert_eq!(got.len(), want.len(), "{name} len");
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-3);
        for (i, (a, w)) in got.iter().zip(want).enumerate() {
            let rel = (a - w).abs() / w.abs().max(0.02 * scale);
            if rel > 2e-2 {
                errs.push(format!("{name}[{i}]: analytic {a} vs autograd {w} (rel {rel:.4})"));
                if errs.len() > 12 { return; }
            }
        }
        checked += got.len();
    };
    cmp("dx", &dx, &get("dx"));
    for (name, _) in param_shapes(qk_norm, ls) {
        cmp(name, &grads[name], &get(name));
    }
    assert!(errs.is_empty(), "gradient mismatches:\n{}", errs.join("\n"));
    assert!(checked > 3000, "too few gradients compared ({checked})");
}


/// Trunk-like: QK-norm + 2D RoPE + LayerScale all on.
#[test]
fn vit_block_gradcheck_trunk() {
    gradcheck(true, true, 0x7a11, "trunk");
}

/// DINOv2-like: LayerScale only.
#[test]
fn vit_block_gradcheck_dino() {
    gradcheck(false, true, 0xd1a0, "dino");
}
