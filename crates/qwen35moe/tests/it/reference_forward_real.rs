// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3.6-35B-A3B's first four decoder layers (three Gated-DeltaNet, one GQA,
//! every one with its sparse MoE) on the REAL checkpoint, against an
//! independent host implementation of the published architecture written in
//! plain f64 loops - no kernel, no shared code with `qwen35moe::model`.
//!
//! Why it exists: every other test of this model's forward is structural (finite,
//! deterministic, decode == prefill, gradient-checked); none says the forward is
//! the architecture the checkpoint was trained as. The checkpoint's own tensors
//! are the only oracle on this box (no Python, no reference runtime), so the
//! oracle is the architecture's equations, transcribed from the reference
//! modelling code:
//!
//! * RMSNorm `x * rsqrt(mean(x^2) + eps) * w` with the reference's `1 + w`
//!   already folded into the stored weight (llama.cpp and brain both store it
//!   folded - see `gguf_vs_hf_real`);
//! * Gated DeltaNet: `silu(causal depthwise conv(in_proj_qkv x))`, q/k L2-normalised
//!   and q scaled by `1/sqrt(dk)`, `beta = sigmoid(b)`, `g = -exp(A_log) *
//!   softplus(a + dt_bias)`, the delta-rule recurrence `S <- S exp(g)`,
//!   `delta = (v - S^T k) beta`, `S <- S + k delta^T`, `o = S^T q`, a gated
//!   RMSNorm `norm(o) * w * silu(z)` and `out_proj`; value head `h` reads key
//!   head `h / group`;
//! * GQA: `q_proj` rows are `[n_heads, 2 * head_dim]` = per head `[q | gate]`,
//!   per-head RMSNorm on q and k, rotate-half RoPE on the first `rotary_dim`
//!   dims, causal softmax attention with 1/sqrt(head_dim), the output gated by
//!   `sigmoid(gate)`, then `o_proj`;
//! * MoE: softmax over all experts, top-k, renormalised, SwiGLU experts plus a
//!   shared SwiGLU expert scaled by `sigmoid(shared_expert_gate . x)`.
//!
//! The model under test runs in fp32 (an int8 model would add its own,
//! separately measured, quantisation error), so agreement should be to
//! rounding. Needs the Q8_0 GGUF under `BRAIN_MODELS_DIR`; skips loudly
//! otherwise.

use checkpoint::gguf::MmapGguf;
use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use model::ops::TierPolicy;
use qwen35moe::config::LayerType;
use qwen35moe::model::{pipelines, Qwen35};
use qwen35moe::{gguf_load, Qwen35Config};

/// Depth compared. Four layers is the default: fp32 weights of more do not fit
/// a card next to the other tenants of this box, and the reference is host f64.
const DEFAULT_LAYERS: u32 = 4;
const TOKENS: [u32; 8] = [760, 6511, 3177, 314, 9338, 369, 11751, 13];

struct Ref<'a> {
    src: &'a dyn TensorSource,
    cfg: &'a Qwen35Config,
}

impl Ref<'_> {
    fn w(&self, name: &str) -> Vec<f64> {
        let mut out = Vec::new();
        assert!(self.src.with_tensor(name, &mut |d| out = d.iter().map(|&v| v as f64).collect()), "{name}");
        out
    }

    /// `y = W x` for a row-major `[n, k]` weight.
    fn linear(&self, name: &str, x: &[f64]) -> Vec<f64> {
        let w = self.w(name);
        let k = x.len();
        w.chunks(k).map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
    }

    fn rmsnorm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
        let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        x.iter().zip(w).map(|(v, w)| v * inv * w).collect()
    }
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}
fn l2norm(x: &[f64]) -> Vec<f64> {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt();
    x.iter().map(|v| v * inv).collect()
}

/// Per-sequence state of one Gated-DeltaNet layer: the conv window (the last
/// `kernel - 1` inputs per channel) and the `[nvh, dk, dv]` recurrent state.
struct GdnState {
    hist: Vec<Vec<f64>>, // [kernel-1][conv_dim], oldest first
    s: Vec<f64>,
}

fn gdn_layer(r: &Ref, l: usize, x: &[f64], st: &mut GdnState) -> Vec<f64> {
    let c = r.cfg;
    let (nkh, nvh, dk, dv) = (c.linear_num_key_heads as usize, c.linear_num_value_heads as usize, c.linear_key_head_dim as usize, c.linear_value_head_dim as usize);
    let (key_dim, conv_dim, kernel) = (nkh * dk, c.linear_conv_dim() as usize, c.linear_conv_kernel_dim as usize);
    let p = |s: &str| format!("blocks.{l}.linear_attn.{s}");

    let mixed = r.linear(&p("in_proj_qkv.weight"), x);
    let z = r.linear(&p("in_proj_z.weight"), x);
    let b = r.linear(&p("in_proj_b.weight"), x);
    let a = r.linear(&p("in_proj_a.weight"), x);

    // Causal depthwise conv over [history..., current], taps oldest first.
    let cw = r.w(&p("conv1d.weight")); // [conv_dim, kernel]
    let mut conv = vec![0.0; conv_dim];
    for ch in 0..conv_dim {
        let mut acc = cw[ch * kernel + kernel - 1] * mixed[ch];
        for (j, h) in st.hist.iter().enumerate() {
            acc += cw[ch * kernel + j] * h[ch];
        }
        conv[ch] = silu(acc);
    }
    st.hist.remove(0);
    st.hist.push(mixed);

    let (q_all, rest) = conv.split_at(key_dim);
    let (k_all, v_all) = rest.split_at(key_dim);
    let a_log = r.w(&p("A_log"));
    let dt_bias = r.w(&p("dt_bias"));
    let group = nvh / nkh;
    let scale = 1.0 / (dk as f64).sqrt();

    let mut core = vec![0.0; nvh * dv];
    for h in 0..nvh {
        let kh = h / group;
        let q: Vec<f64> = l2norm(&q_all[kh * dk..(kh + 1) * dk]).iter().map(|v| v * scale).collect();
        let k = l2norm(&k_all[kh * dk..(kh + 1) * dk]);
        let v = &v_all[h * dv..(h + 1) * dv];
        let beta = sigmoid(b[h]);
        let g = -a_log[h].exp() * softplus(a[h] + dt_bias[h]);
        let s = &mut st.s[h * dk * dv..(h + 1) * dk * dv];
        let decay = g.exp();
        for e in s.iter_mut() {
            *e *= decay;
        }
        let mut delta = vec![0.0; dv];
        for j in 0..dv {
            let kv_mem: f64 = (0..dk).map(|i| s[i * dv + j] * k[i]).sum();
            delta[j] = (v[j] - kv_mem) * beta;
        }
        for i in 0..dk {
            for j in 0..dv {
                s[i * dv + j] += k[i] * delta[j];
            }
        }
        for j in 0..dv {
            core[h * dv + j] = (0..dk).map(|i| s[i * dv + j] * q[i]).sum();
        }
    }

    // Gated RMSNorm per value head, then out_proj.
    let nw = r.w(&p("norm.weight"));
    let mut gated = vec![0.0; nvh * dv];
    for h in 0..nvh {
        let normed = Ref::rmsnorm(&core[h * dv..(h + 1) * dv], &nw, c.rms_eps as f64);
        for j in 0..dv {
            gated[h * dv + j] = normed[j] * silu(z[h * dv + j]);
        }
    }
    r.linear(&p("out_proj.weight"), &gated)
}

/// One GQA layer for the token at position `pos`, appending its k/v to `cache`.
fn gqa_layer(r: &Ref, l: usize, x: &[f64], pos: usize, cache: &mut Vec<(Vec<f64>, Vec<f64>)>) -> Vec<f64> {
    let c = r.cfg;
    let (nh, nkv, hd, rot) = (c.n_heads as usize, c.n_kv_heads as usize, c.head_dim as usize, c.rotary_dim() as usize);
    let p = |s: &str| format!("blocks.{l}.self_attn.{s}");
    let eps = c.rms_eps as f64;

    let qg = r.linear(&p("q_proj.weight"), x); // [nh, 2*hd]: per head [q | gate]
    let kk = r.linear(&p("k_proj.weight"), x);
    let vv = r.linear(&p("v_proj.weight"), x);
    let (qn, kn) = (r.w(&p("q_norm.weight")), r.w(&p("k_norm.weight")));

    let rope = |v: &mut [f64]| {
        let half = rot / 2;
        let orig = v[..rot].to_vec();
        for i in 0..half {
            let inv_freq = (c.rope_theta as f64).powf(-(2.0 * i as f64) / rot as f64);
            let (s, co) = (pos as f64 * inv_freq).sin_cos();
            v[i] = orig[i] * co - orig[i + half] * s;
            v[i + half] = orig[i + half] * co + orig[i] * s;
        }
    };
    let mut q: Vec<Vec<f64>> = (0..nh).map(|h| Ref::rmsnorm(&qg[h * 2 * hd..h * 2 * hd + hd], &qn, eps)).collect();
    let gate: Vec<f64> = (0..nh).flat_map(|h| qg[h * 2 * hd + hd..(h + 1) * 2 * hd].to_vec()).collect();
    let mut k: Vec<Vec<f64>> = (0..nkv).map(|h| Ref::rmsnorm(&kk[h * hd..(h + 1) * hd], &kn, eps)).collect();
    q.iter_mut().for_each(|v| rope(v));
    k.iter_mut().for_each(|v| rope(v));
    cache.push((k.concat(), vv));

    let group = nh / nkv;
    let mut ctx = vec![0.0; nh * hd];
    for h in 0..nh {
        let kvh = h / group;
        let scores: Vec<f64> = cache.iter().map(|(ck, _)| q[h].iter().zip(&ck[kvh * hd..(kvh + 1) * hd]).map(|(a, b)| a * b).sum::<f64>() / (hd as f64).sqrt()).collect();
        let mx = scores.iter().cloned().fold(f64::MIN, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for (t, (_, cv)) in cache.iter().enumerate() {
            for j in 0..hd {
                ctx[h * hd + j] += exps[t] / sum * cv[kvh * hd + j];
            }
        }
    }
    let gated: Vec<f64> = ctx.iter().zip(&gate).map(|(c, g)| c * sigmoid(*g)).collect();
    r.linear(&p("o_proj.weight"), &gated)
}

fn moe(r: &Ref, l: usize, x: &[f64]) -> Vec<f64> {
    let c = r.cfg;
    let p = |s: &str| format!("blocks.{l}.mlp.{s}");
    let logits = r.linear(&p("router.weight"), x);
    let mx = logits.iter().cloned().fold(f64::MIN, f64::max);
    let exps: Vec<f64> = logits.iter().map(|v| (v - mx).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let top = &order[..c.top_k as usize];
    let norm: f64 = top.iter().map(|&e| probs[e]).sum();

    let swiglu = |gate: &str, up: &str, down: &str| {
        let (g, u) = (r.linear(gate, x), r.linear(up, x));
        let h: Vec<f64> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
        r.linear(down, &h)
    };
    let mut out = vec![0.0; x.len()];
    for &e in top {
        let y = swiglu(&p(&format!("experts.{e}.gate.weight")), &p(&format!("experts.{e}.up.weight")), &p(&format!("experts.{e}.down.weight")));
        for (o, v) in out.iter_mut().zip(&y) {
            *o += probs[e] / norm * v;
        }
    }
    let shared = swiglu(&p("shared_expert.gate.weight"), &p("shared_expert.up.weight"), &p("shared_expert.down.weight"));
    let sg = sigmoid(r.w(&p("shared_expert_gate.weight")).iter().zip(x).map(|(a, b)| a * b).sum());
    for (o, v) in out.iter_mut().zip(&shared) {
        *o += sg * v;
    }
    out
}

#[test]
fn the_first_four_layers_match_an_independent_host_implementation_of_the_architecture() {
    let Some(models) = std::env::var_os("BRAIN_MODELS_DIR").map(std::path::PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".local/share/brain/models"))) else {
        return brain_testutil::skip("no models directory");
    };
    let path = models.join("unsloth/Qwen3.6-35B-A3B-GGUF/Q8_0.gguf");
    if !path.exists() {
        return brain_testutil::skip("the Qwen3.6-35B-A3B Q8_0 GGUF is not under BRAIN_MODELS_DIR");
    }
    let mg = MmapGguf::open(path.to_str().unwrap()).unwrap();
    let mut cfg = gguf_load::resident_config(&mg, 64).unwrap();
    let layers: u32 = std::env::var("BRAIN_REFERENCE_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_LAYERS);
    cfg.n_layers = layers;
    let src = gguf_load::source(&mg, &cfg).unwrap();
    let r = Ref { src: &src, cfg: &cfg };

    let d = cfg.d_model as usize;
    let embed = |tok: u32| -> Vec<f64> {
        let mut row = Vec::new();
        assert!(src.with_tensor_range("tok.weight", tok as usize * d, d, &mut |v| row = v.iter().map(|&x| x as f64).collect()));
        row
    };
    let eps = cfg.rms_eps as f64;
    let types = cfg.layer_types();
    let mut gdn: Vec<Option<GdnState>> = types
        .iter()
        .map(|t| (*t == LayerType::Linear).then(|| GdnState { hist: vec![vec![0.0; cfg.linear_conv_dim() as usize]; cfg.linear_conv_kernel_dim as usize - 1], s: vec![0.0; (cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim) as usize] }))
        .collect();
    let mut kv: Vec<Vec<(Vec<f64>, Vec<f64>)>> = vec![Vec::new(); types.len()];
    let final_norm = r.w("norm.weight");

    let mut want: Vec<Vec<f64>> = Vec::new();
    for (pos, &tok) in TOKENS.iter().enumerate() {
        let mut h = embed(tok);
        for l in 0..layers as usize {
            let xn1 = Ref::rmsnorm(&h, &r.w(&format!("blocks.{l}.ln1.weight")), eps);
            let attn = match types[l] {
                LayerType::Linear => gdn_layer(&r, l, &xn1, gdn[l].as_mut().unwrap()),
                LayerType::Full => gqa_layer(&r, l, &xn1, pos, &mut kv[l]),
            };
            let mid: Vec<f64> = h.iter().zip(&attn).map(|(a, b)| a + b).collect();
            let xn2 = Ref::rmsnorm(&mid, &r.w(&format!("blocks.{l}.ln2.weight")), eps);
            let ffn = moe(&r, l, &xn2);
            h = mid.iter().zip(&ffn).map(|(a, b)| a + b).collect();
        }
        want.push(Ref::rmsnorm(&h, &final_norm, eps));
    }

    // `BRAIN_REFERENCE_TIER=i8x` runs int8 experts under fp32 mixers, `i8` all
    // int8: an exploration of the quantisation error at depth (printed, not
    // asserted - it is a measurement, not the architecture check).
    let tier = match std::env::var("BRAIN_REFERENCE_TIER").as_deref() {
        Ok("i8x") => Some(TierPolicy::uniform(Dtype::I8).with(&["linear_attn", "self_attn"], Dtype::F32)),
        Ok("i8") => Some(TierPolicy::uniform(Dtype::I8)),
        _ => None,
    };
    let gpu = gpu_core::Gpu::new(pipelines());
    let model = match &tier {
        Some(t) => Qwen35::new_on_tier_src(gpu, cfg.clone(), 1, 64, &src, t),
        None => Qwen35::new_on_src(gpu, cfg.clone(), 1, 64, &src),
    };
    let mut worst = 0f64;
    let mut model_hidden: Vec<Vec<f32>> = Vec::new();
    for (pos, &tok) in TOKENS.iter().enumerate() {
        let got = model.step(tok);
        model_hidden.push(got.clone());
        let (mut num, mut den) = (0f64, 0f64);
        for (g, w) in got.iter().zip(&want[pos]) {
            num += (*g as f64 - w).powi(2);
            den += w * w;
        }
        let rel = (num / den).sqrt();
        println!("position {pos}: relative L2 error vs the host reference {rel:.3e}");
        worst = worst.max(rel);
    }
    // Exploration (BRAIN_REFERENCE_TOP=1, meaningful at full depth): what the
    // HOST reference and the model each predict next, from the same hidden
    // states - the reference's own distribution says whether the equations
    // above describe a language model at all.
    if std::env::var_os("BRAIN_REFERENCE_TOP").is_some() {
        use data::tokenizer::Tokenizer;
        let tok = data::qwen_tokenizer::QwenBpe::from_gguf(&mg.tokenizer().unwrap()).unwrap();
        let vocab = cfg.vocab as usize;
        let mut head = Vec::new();
        assert!(src.with_tensor("lm_head.weight", &mut |d| head = d.to_vec()));
        let top5 = |h: &[f64]| {
            let logits: Vec<f64> = (0..vocab).map(|v| head[v * d..(v + 1) * d].iter().zip(h).map(|(a, b)| *a as f64 * b).sum()).collect();
            let mut idx: Vec<usize> = (0..vocab).collect();
            idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            idx[..5].iter().map(|&i| (tok.decode(&[i as u32]), (logits[i] * 100.0).round() / 100.0)).collect::<Vec<_>>()
        };
        for pos in 4..TOKENS.len() {
            let got: Vec<f64> = model_hidden[pos].iter().map(|&v| v as f64).collect();
            println!("after position {pos}: host reference {:?}\n                       model          {:?}", top5(&want[pos]), top5(&got));
        }
    }
    if tier.is_some() {
        return;
    }
    assert!(worst < 2e-3, "the model's first {layers} layers disagree with the architecture's equations: worst relative L2 error {worst:.3e}");
}
