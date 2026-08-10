// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity gate for the Kronos **cached** s1 NPU rollout: prefill fills a fixed-`cap`
//! KV cache, then the single-token decode graph appends tokens attending the cache
//! (O(cap)/step). This must reproduce the full-window s1 graph run over the same
//! growing context — i.e. the KV-cache math == full causal attention. Both compile
//! on the OpenVINO **CPU** device (fp32). Skips without an OpenVINO runtime.
//!
//! Run: LD_LIBRARY_PATH=<openvino/libs> cargo test -p brain-npu --test kronos_kvcache -- --nocapture

use std::collections::HashMap;

use kronos::KronosConfig;
use npu::kronos_topology::{
    build_kronos_dep_decode_graph, build_kronos_dep_graph, build_kronos_dep_prefill_graph, build_kronos_decoder_graph,
    build_kronos_s1_decode_graph, build_kronos_s1_prefill_graph,
};
use npu::openvino::{available_devices, Feed, NpuConfig, NpuDevice, NpuGraph, PerfHint};
use onnx::GraphBuilder;

type W = HashMap<String, Vec<f32>>;

fn cpu() -> NpuConfig {
    NpuConfig { device: NpuDevice::Cpu, perf_hint: PerfHint::Latency, allow_fallback: true, ..Default::default() }
}

fn rand_weights(cfg: &KronosConfig) -> W {
    // The unified deterministic LCG (audit F39/F40).
    let mut lcg = data::rng::Lcg::new(0x1234_5678);
    cfg.param_list()
        .into_iter()
        .map(|(k, s)| {
            let n: usize = s.iter().product();
            let v = (0..n).map(|_| lcg.scaled(0.05)).collect();
            (k, v)
        })
        .collect()
}

/// Look up a named output (name, shape, data) from an `NpuGraph::run` result.
fn get<'a>(out: &'a [(String, Vec<usize>, Vec<f32>)], name: &str) -> &'a [f32] {
    &out.iter().find(|(n, _, _)| n == name).unwrap_or_else(|| panic!("missing output {name}")).2
}

use model::hostmath::cosine;

#[test]
fn s1_cached_rollout_matches_full_window() {
    if available_devices().map(|d| d.is_empty()).unwrap_or(true) {
        eprintln!("skip: no OpenVINO runtime");
        return;
    }
    let cfg = KronosConfig::tiny();
    let w = rand_weights(&cfg);
    let (d, heads, nl, s1v) = (cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.s1_vocab());
    let hd = d / heads;
    let half = hd / 2;
    let (cap, t_ctx) = (8usize, 5usize); // prefill 0..5, decode 5..8

    // random host-assembled token embeddings [cap, d].
    let x: Vec<f32> = (0..cap * d).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect();

    // ---- 1) full-window reference at T=cap ----
    let mut gf = GraphBuilder::new("s1_full");
    build_kronos_decoder_graph(&cfg, &w, cap, &mut gf);
    let mut full = NpuGraph::compile_bytes(&gf.finish(), &cpu()).expect("compile full");
    let fout = full.run(&[("x", Feed::F32(&x, vec![1, cap as i64, d as i64]))]).expect("run full");
    let full_ctx = get(&fout, "ctx").to_vec();
    let full_lg = get(&fout, "s1_logits").to_vec();

    // ---- 2) prefill over 0..t_ctx ----
    let mut gp = GraphBuilder::new("s1_prefill");
    build_kronos_s1_prefill_graph(&cfg, &w, t_ctx, &mut gp);
    let mut pre = NpuGraph::compile_bytes(&gp.finish(), &cpu()).expect("compile prefill");
    let pout = pre.run(&[("x", Feed::F32(&x[..t_ctx * d], vec![1, t_ctx as i64, d as i64]))]).expect("run prefill");

    // seed cache buffers [heads, cap, hd] per layer with prefill K/V at positions 0..t_ctx.
    let mut pk: Vec<Vec<f32>> = vec![vec![0.0f32; heads * cap * hd]; nl];
    let mut pv: Vec<Vec<f32>> = vec![vec![0.0f32; heads * cap * hd]; nl];
    for l in 0..nl {
        let kl = get(&pout, &format!("k_{l}")); // [heads, t_ctx, hd]
        let vl = get(&pout, &format!("v_{l}"));
        for h in 0..heads {
            for p in 0..t_ctx {
                for j in 0..hd {
                    pk[l][(h * cap + p) * hd + j] = kl[(h * t_ctx + p) * hd + j];
                    pv[l][(h * cap + p) * hd + j] = vl[(h * t_ctx + p) * hd + j];
                }
            }
        }
    }
    // prefill's last-position ctx/logits should already match the full graph.
    let cos_pre = cosine(&full_lg[(t_ctx - 1) * s1v..t_ctx * s1v], &get(&pout, "s1_logits")[(t_ctx - 1) * s1v..t_ctx * s1v]);
    eprintln!("prefill last-pos s1_logits cosine {cos_pre:.6}");
    assert!(cos_pre > 0.999, "prefill last position diverged: {cos_pre}");

    // ---- 3) decode positions t_ctx..cap, appending to the cache ----
    let mut gd = GraphBuilder::new("s1_decode");
    build_kronos_s1_decode_graph(&cfg, &w, cap, &mut gd);
    let mut dec = NpuGraph::compile_bytes(&gd.finish(), &cpu()).expect("compile decode");

    let mut worst = 1.0f32;
    for p in t_ctx..cap {
        let cos_t: Vec<f32> = (0..half).map(|j| (p as f32 * 10000f32.powf(-(2.0 * j as f32) / hd as f32)).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|j| (p as f32 * 10000f32.powf(-(2.0 * j as f32) / hd as f32)).sin()).collect();
        let mask: Vec<f32> = (0..cap).map(|j| if j < p { 0.0 } else { -1e9 }).collect();

        // build feeds + run in a scope so the immutable borrows of `pk`/`pv` end
        // before we append this step's K/V to them below.
        let keys: Vec<(String, String)> = (0..nl).map(|l| (format!("past_k_{l}"), format!("past_v_{l}"))).collect();
        let dout = {
            let mut feeds: Vec<(&str, Feed)> = vec![
                ("x", Feed::F32(&x[p * d..(p + 1) * d], vec![1, 1, d as i64])),
                ("rope_cos", Feed::F32(&cos_t, vec![1, 1, 1, half as i64])),
                ("rope_sin", Feed::F32(&sin_t, vec![1, 1, 1, half as i64])),
                ("past_mask", Feed::F32(&mask, vec![1, 1, 1, cap as i64])),
            ];
            for l in 0..nl {
                feeds.push((keys[l].0.as_str(), Feed::F32(&pk[l], vec![1, heads as i64, cap as i64, hd as i64])));
                feeds.push((keys[l].1.as_str(), Feed::F32(&pv[l], vec![1, heads as i64, cap as i64, hd as i64])));
            }
            dec.run(&feeds).expect("run decode")
        };

        // parity vs full window at this position.
        let c_ctx = cosine(get(&dout, "ctx"), &full_ctx[p * d..(p + 1) * d]);
        let c_lg = cosine(get(&dout, "s1_logits"), &full_lg[p * s1v..(p + 1) * s1v]);
        worst = worst.min(c_ctx).min(c_lg);
        eprintln!("pos {p}: ctx cosine {c_ctx:.6}, s1_logits cosine {c_lg:.6}");
        assert!(c_ctx > 0.999 && c_lg > 0.999, "cached decode diverged at pos {p}: ctx {c_ctx}, logits {c_lg}");

        // append this token's K/V to the cache at slot p (for the next step).
        for l in 0..nl {
            let nk = get(&dout, &format!("new_k_{l}")); // [heads,1,hd]
            let nv = get(&dout, &format!("new_v_{l}"));
            for h in 0..heads {
                for j in 0..hd {
                    pk[l][(h * cap + p) * hd + j] = nk[h * hd + j];
                    pv[l][(h * cap + p) * hd + j] = nv[h * hd + j];
                }
            }
        }
    }
    eprintln!("s1 cached rollout vs full window: worst cosine {worst:.6} (cap={cap}, ctx={t_ctx})");
}

#[test]
fn dep_cached_rollout_matches_full_window() {
    if available_devices().map(|d| d.is_empty()).unwrap_or(true) {
        eprintln!("skip: no OpenVINO runtime");
        return;
    }
    let cfg = KronosConfig::tiny();
    let w = rand_weights(&cfg);
    let (d, s2v) = (cfg.d_model, cfg.s2_vocab());
    let heads = cfg.dep_n_heads;
    let hd = d / heads;
    let half = hd / 2;
    let (cap, t_ctx) = (8usize, 5usize);

    // random s1 context + per-position sibling embeddings [cap, d].
    let ctx: Vec<f32> = (0..cap * d).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    let sib: Vec<f32> = (0..cap * d).map(|i| ((i as f32) * 0.019 + 1.0).cos() * 0.5).collect();

    // dep prefill over 0..t_ctx → cache [heads, cap, hd].
    let mut gp = GraphBuilder::new("dep_prefill");
    build_kronos_dep_prefill_graph(&cfg, &w, t_ctx, &mut gp);
    let mut pre = NpuGraph::compile_bytes(&gp.finish(), &cpu()).expect("compile dep prefill");
    let pout = pre.run(&[("ctx", Feed::F32(&ctx[..t_ctx * d], vec![1, t_ctx as i64, d as i64]))]).expect("run dep prefill");
    let mut dk = vec![0.0f32; heads * cap * hd];
    let mut dv = vec![0.0f32; heads * cap * hd];
    for (buf, name) in [(&mut dk, "dep_k"), (&mut dv, "dep_v")] {
        let src = get(&pout, name); // [heads, t_ctx, hd]
        for h in 0..heads {
            for p in 0..t_ctx {
                for j in 0..hd {
                    buf[(h * cap + p) * hd + j] = src[(h * t_ctx + p) * hd + j];
                }
            }
        }
    }

    let mut gd = GraphBuilder::new("dep_decode");
    build_kronos_dep_decode_graph(&cfg, &w, cap, &mut gd);
    let mut dec = NpuGraph::compile_bytes(&gd.finish(), &cpu()).expect("compile dep decode");

    let mut worst = 1.0f32;
    for p in t_ctx..cap {
        // reference: the full dep graph over the growing context 0..p+1, last row.
        let tref = p + 1;
        let mut gr = GraphBuilder::new("dep_full");
        build_kronos_dep_graph(&cfg, &w, tref, &mut gr);
        let mut refg = NpuGraph::compile_bytes(&gr.finish(), &cpu()).expect("compile dep full");
        let rout = refg
            .run(&[
                ("ctx", Feed::F32(&ctx[..tref * d], vec![1, tref as i64, d as i64])),
                ("sib", Feed::F32(&sib[..tref * d], vec![1, tref as i64, d as i64])),
            ])
            .expect("run dep full");
        let ref_last = &get(&rout, "s2_logits")[p * s2v..(p + 1) * s2v];

        let cos_t: Vec<f32> = (0..half).map(|j| (p as f32 * 10000f32.powf(-(2.0 * j as f32) / hd as f32)).cos()).collect();
        let sin_t: Vec<f32> = (0..half).map(|j| (p as f32 * 10000f32.powf(-(2.0 * j as f32) / hd as f32)).sin()).collect();
        let mask: Vec<f32> = (0..cap).map(|j| if j < p { 0.0 } else { -1e9 }).collect();
        let dout = {
            let feeds: Vec<(&str, Feed)> = vec![
                ("sib", Feed::F32(&sib[p * d..(p + 1) * d], vec![1, 1, d as i64])),
                ("ctx_last", Feed::F32(&ctx[p * d..(p + 1) * d], vec![1, 1, d as i64])),
                ("rope_cos", Feed::F32(&cos_t, vec![1, 1, 1, half as i64])),
                ("rope_sin", Feed::F32(&sin_t, vec![1, 1, 1, half as i64])),
                ("dep_mask", Feed::F32(&mask, vec![1, 1, 1, cap as i64])),
                ("past_dep_k", Feed::F32(&dk, vec![1, heads as i64, cap as i64, hd as i64])),
                ("past_dep_v", Feed::F32(&dv, vec![1, heads as i64, cap as i64, hd as i64])),
            ];
            dec.run(&feeds).expect("run dep decode")
        };
        let c_lg = cosine(get(&dout, "s2_logits"), ref_last);
        worst = worst.min(c_lg);
        eprintln!("pos {p}: s2_logits cosine {c_lg:.6}");
        assert!(c_lg > 0.999, "cached dep decode diverged at pos {p}: {c_lg}");
        // append this position's dep K/V to the cache.
        for (buf, name) in [(&mut dk, "new_dep_k"), (&mut dv, "new_dep_v")] {
            let nn = get(&dout, name); // [heads,1,hd]
            for h in 0..heads {
                for j in 0..hd {
                    buf[(h * cap + p) * hd + j] = nn[h * hd + j];
                }
            }
        }
    }
    eprintln!("dep cached rollout vs full window: worst cosine {worst:.6} (cap={cap}, ctx={t_ctx})");
}

/// End-to-end driver logic: the interleaved cached rollout (prefill, then per step
/// dep_decode using the latest s1 ctx, then s1_decode) must match the full-window
/// s1 + dep graphs run over the growing context — i.e. the ctx threading + position
/// / dep_valid counters in `KronosCachedNpu` are correct. Mirrors that struct's
/// logic with the graphs directly (no tokenizer/checkpoint needed).
#[test]
fn cached_rollout_driver_matches_full_window() {
    if available_devices().map(|d| d.is_empty()).unwrap_or(true) {
        eprintln!("skip: no OpenVINO runtime");
        return;
    }
    let cfg = KronosConfig::tiny();
    let w = rand_weights(&cfg);
    let (d, s1v, s2v, nl) = (cfg.d_model, cfg.s1_vocab(), cfg.s2_vocab(), cfg.n_layers);
    let (heads, dep_heads) = (cfg.n_heads, cfg.dep_n_heads);
    let (hd, dhd) = (d / heads, d / dep_heads);
    let (half, dhalf) = (hd / 2, dhd / 2);
    let (cap, t) = (8usize, 5usize);
    let x: Vec<f32> = (0..cap * d).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect();
    let sibs: Vec<f32> = (0..cap * d).map(|i| ((i as f32) * 0.021 + 0.3).cos() * 0.5).collect();

    // full-window references: s1 over all x, and dep over the growing s1 ctx.
    let mut gf = GraphBuilder::new("s1_full");
    build_kronos_decoder_graph(&cfg, &w, cap, &mut gf);
    let mut full = NpuGraph::compile_bytes(&gf.finish(), &cpu()).expect("compile s1 full");
    let fout = full.run(&[("x", Feed::F32(&x, vec![1, cap as i64, d as i64]))]).expect("run s1 full");
    let ctx_full = get(&fout, "ctx").to_vec();
    let s1lg_full = get(&fout, "s1_logits").to_vec();
    let dep_ref = |m: usize| -> Vec<f32> {
        let tref = m + 1;
        let mut gr = GraphBuilder::new("dep_full");
        build_kronos_dep_graph(&cfg, &w, tref, &mut gr);
        let mut refg = NpuGraph::compile_bytes(&gr.finish(), &cpu()).expect("compile dep full");
        let rout = refg
            .run(&[
                ("ctx", Feed::F32(&ctx_full[..tref * d], vec![1, tref as i64, d as i64])),
                ("sib", Feed::F32(&sibs[..tref * d], vec![1, tref as i64, d as i64])),
            ])
            .expect("run dep full");
        get(&rout, "s2_logits")[m * s2v..(m + 1) * s2v].to_vec()
    };

    // cached backend (mirrors KronosCachedNpu).
    let mut s1p = GraphBuilder::new("s1p");
    build_kronos_s1_prefill_graph(&cfg, &w, t, &mut s1p);
    let mut s1_prefill = NpuGraph::compile_bytes(&s1p.finish(), &cpu()).expect("s1p");
    let mut s1d = GraphBuilder::new("s1d");
    build_kronos_s1_decode_graph(&cfg, &w, cap, &mut s1d);
    let mut s1_decode = NpuGraph::compile_bytes(&s1d.finish(), &cpu()).expect("s1d");
    let mut dpp = GraphBuilder::new("dpp");
    build_kronos_dep_prefill_graph(&cfg, &w, t - 1, &mut dpp);
    let mut dep_prefill = NpuGraph::compile_bytes(&dpp.finish(), &cpu()).expect("dpp");
    let mut dpd = GraphBuilder::new("dpd");
    build_kronos_dep_decode_graph(&cfg, &w, cap, &mut dpd);
    let mut dep_decode = NpuGraph::compile_bytes(&dpd.finish(), &cpu()).expect("dpd");

    let rope = |pos: usize, hf: usize, h: usize| -> (Vec<f32>, Vec<f32>) {
        let c = (0..hf).map(|j| (pos as f32 * 10000f32.powf(-(2.0 * j as f32) / h as f32)).cos()).collect();
        let s = (0..hf).map(|j| (pos as f32 * 10000f32.powf(-(2.0 * j as f32) / h as f32)).sin()).collect();
        (c, s)
    };

    // prefill
    let pout = s1_prefill.run(&[("x", Feed::F32(&x[..t * d], vec![1, t as i64, d as i64]))]).expect("run s1p");
    let mut pk = vec![vec![0.0f32; heads * cap * hd]; nl];
    let mut pv = pk.clone();
    for l in 0..nl {
        let (kl, vl) = (get(&pout, &format!("k_{l}")), get(&pout, &format!("v_{l}")));
        for h in 0..heads {
            for p in 0..t {
                for j in 0..hd {
                    pk[l][(h * cap + p) * hd + j] = kl[(h * t + p) * hd + j];
                    pv[l][(h * cap + p) * hd + j] = vl[(h * t + p) * hd + j];
                }
            }
        }
    }
    let dpre = dep_prefill.run(&[("ctx", Feed::F32(&get(&pout, "ctx")[..(t - 1) * d], vec![1, (t - 1) as i64, d as i64]))]).expect("run dpp");
    let mut dk = vec![0.0f32; dep_heads * cap * dhd];
    let mut dv = dk.clone();
    for (buf, nm) in [(&mut dk, "dep_k"), (&mut dv, "dep_v")] {
        let src = get(&dpre, nm);
        for h in 0..dep_heads {
            for p in 0..(t - 1) {
                for j in 0..dhd {
                    buf[(h * cap + p) * dhd + j] = src[(h * (t - 1) + p) * dhd + j];
                }
            }
        }
    }
    let mut ctx_last = get(&pout, "ctx")[(t - 1) * d..t * d].to_vec();
    let mut s1_pos = t;
    let mut dep_valid = t - 1;
    let mut worst = 1.0f32;

    for k in 0..(cap - t) {
        // dep_step at position s1_pos-1 using ctx_last.
        let mdep = s1_pos - 1;
        let (dc, ds) = rope(mdep, dhalf, dhd);
        let dmask: Vec<f32> = (0..cap).map(|j| if j < dep_valid { 0.0 } else { -1e9 }).collect();
        let dout = {
            let feeds: Vec<(&str, Feed)> = vec![
                ("sib", Feed::F32(&sibs[mdep * d..(mdep + 1) * d], vec![1, 1, d as i64])),
                ("ctx_last", Feed::F32(&ctx_last, vec![1, 1, d as i64])),
                ("rope_cos", Feed::F32(&dc, vec![1, 1, 1, dhalf as i64])),
                ("rope_sin", Feed::F32(&ds, vec![1, 1, 1, dhalf as i64])),
                ("dep_mask", Feed::F32(&dmask, vec![1, 1, 1, cap as i64])),
                ("past_dep_k", Feed::F32(&dk, vec![1, dep_heads as i64, cap as i64, dhd as i64])),
                ("past_dep_v", Feed::F32(&dv, vec![1, dep_heads as i64, cap as i64, dhd as i64])),
            ];
            dep_decode.run(&feeds).expect("run dpd")
        };
        let c_s2 = cosine(get(&dout, "s2_logits"), &dep_ref(mdep));
        worst = worst.min(c_s2);
        assert!(c_s2 > 0.999, "driver dep at pos {mdep} diverged: {c_s2}");
        for (buf, nm) in [(&mut dk, "new_dep_k"), (&mut dv, "new_dep_v")] {
            let nn = get(&dout, nm);
            for h in 0..dep_heads {
                for j in 0..dhd {
                    buf[(h * cap + dep_valid) * dhd + j] = nn[h * dhd + j];
                }
            }
        }
        dep_valid += 1;

        // s1_step at position s1_pos.
        let pos = s1_pos;
        let (sc, ss) = rope(pos, half, hd);
        let smask: Vec<f32> = (0..cap).map(|j| if j < pos { 0.0 } else { -1e9 }).collect();
        let keys: Vec<(String, String)> = (0..nl).map(|l| (format!("past_k_{l}"), format!("past_v_{l}"))).collect();
        let sout = {
            let mut feeds: Vec<(&str, Feed)> = vec![
                ("x", Feed::F32(&x[pos * d..(pos + 1) * d], vec![1, 1, d as i64])),
                ("rope_cos", Feed::F32(&sc, vec![1, 1, 1, half as i64])),
                ("rope_sin", Feed::F32(&ss, vec![1, 1, 1, half as i64])),
                ("past_mask", Feed::F32(&smask, vec![1, 1, 1, cap as i64])),
            ];
            for l in 0..nl {
                feeds.push((keys[l].0.as_str(), Feed::F32(&pk[l], vec![1, heads as i64, cap as i64, hd as i64])));
                feeds.push((keys[l].1.as_str(), Feed::F32(&pv[l], vec![1, heads as i64, cap as i64, hd as i64])));
            }
            s1_decode.run(&feeds).expect("run s1d")
        };
        let c_s1 = cosine(get(&sout, "s1_logits"), &s1lg_full[pos * s1v..(pos + 1) * s1v]);
        worst = worst.min(c_s1);
        assert!(c_s1 > 0.999, "driver s1 at pos {pos} diverged: {c_s1}");
        for l in 0..nl {
            let (nk, nv) = (get(&sout, &format!("new_k_{l}")), get(&sout, &format!("new_v_{l}")));
            for h in 0..heads {
                for j in 0..hd {
                    pk[l][(h * cap + pos) * hd + j] = nk[h * hd + j];
                    pv[l][(h * cap + pos) * hd + j] = nv[h * hd + j];
                }
            }
        }
        ctx_last = get(&sout, "ctx").to_vec();
        s1_pos += 1;
        eprintln!("step k={k}: dep@{mdep} cos {c_s2:.6}, s1@{pos} cos {c_s1:.6}");
    }
    eprintln!("cached-rollout DRIVER vs full window: worst cosine {worst:.6}");
}
