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
use npu::kronos_topology::{build_kronos_decoder_graph, build_kronos_s1_decode_graph, build_kronos_s1_prefill_graph};
use npu::openvino::{available_devices, Feed, NpuConfig, NpuDevice, NpuGraph, PerfHint};
use onnx::GraphBuilder;

type W = HashMap<String, Vec<f32>>;

fn cpu() -> NpuConfig {
    NpuConfig { device: NpuDevice::Cpu, perf_hint: PerfHint::Latency, allow_fallback: true, ..Default::default() }
}

fn rand_weights(cfg: &KronosConfig) -> W {
    let mut seed = 0x1234_5678u64;
    cfg.param_list()
        .into_iter()
        .map(|(k, s)| {
            let n: usize = s.iter().product();
            let v = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05
                })
                .collect();
            (k, v)
        })
        .collect()
}

/// Look up a named output (name, shape, data) from an `NpuGraph::run` result.
fn get<'a>(out: &'a [(String, Vec<usize>, Vec<f32>)], name: &str) -> &'a [f32] {
    &out.iter().find(|(n, _, _)| n == name).unwrap_or_else(|| panic!("missing output {name}")).2
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

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
