// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reference-parity gates for sparse mixture-of-experts decoders for its clients. If
// your team needs expertise in proving a hybrid linear-attention MoE port computes what its reference computes then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Numerical parity of Qwen3.5-35B-A3B against `transformers`' `Qwen3_5MoeForCausalLM`, on the tiny random-weight
//! model `tools/goldens/qwen35moe_dump_reference.py` writes in the release's checkpoint layout.
//!
//! The checkpoint is read the way the release is (`hf_source::hf_config` + `hf_source::open_dir`: the
//! `model.language_model.*` names, one tensor per expert projection, `1 + w` norm gains), so the import is under test
//! as well as the forward. Covered: the full-sequence prefill (every layer's residual output and the logits, three
//! prompts, one of them longer than the 64-row Gated DeltaNet chunk), the incremental decode over the cached recurrent
//! state and KV after a chunked prefill against the reference's own cached decode, the serving engine (paged KV,
//! recurrent state pool, one sequence and a batch of three), and the int8 tier.
//!
//! ## The tolerance
//!
//! Both sides compute in binary32, so the fp32 comparisons are held to the binary32 round-off bound
//! ([`brain_testutil::rounding::agreement_bound`]), per element: `2 gamma_k S`, with `k` the rounded operations on the
//! longest chain from a token to that element ([`chain_to`]) and `S` the element's absolute-term sum, which the
//! reference records per residual element (`layer{l}.abs_sum`: the embedding plus every mixer output projection's and
//! every MoE combine's absolute terms) and from which the head's is formed. As for `gpt2`'s logit bound, the count
//! takes each stage to pass relative error on with gain at most one; that is the modelling assumption the measured
//! ratios printed below check. The bound presumes both sides route every token to the same experts: the reference
//! records each token's routing margin, and a token routed differently would move its row by a large fraction of an
//! expert's output, far past the bound, so a routing difference fails the comparison instead of hiding in it.
//!
//! The products are pinned to binary32 ([`GemmPrecision::Fp32Reference`]): TF32 operands would put a 2^-11 rounding
//! in every product, which is not the precision the bound is stated for.
//!
//! The int8 tier is compared with the reference evaluated under the same W8A8 rounding, not with fp32: on these
//! weights quantization alone moves the reference's logits by 0.2 to 0.46 relative L2 (knowledge #456), so the
//! fp32 bound says nothing about it (see [`the_int8_tier_computes_the_reference_w8a8_function`]).

use std::collections::HashMap;
use std::path::PathBuf;

use brain_testutil::rounding::agreement_bound;
use gpu_core::provider::library::GemmPrecision;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::ops::TierPolicy;
use model::paged::BlockTable;
use qwen35moe::config::{LayerType, Qwen35Config};
use qwen35moe::model::{pipelines, Qwen35};
use qwen35moe::serve::{Engine, EngineOptions};

const DUMPER: &str = "tools/goldens/qwen35moe_dump_reference.py";
/// The model's sequence length for the full-sequence forward: every prompt is padded to it (the model is causal), and
/// it is a multiple of the 64-row chunk the reference uses, so the longest prompt spans two chunks on both sides.
const T: u32 = 128;
/// The largest Gated DeltaNet chunk either side forms.
const GDN_CHUNK: usize = 64;

struct Prompt {
    tokens: Vec<u32>,
    decode_from: usize,
}

struct Golden {
    ckpt: PathBuf,
    cfg: Qwen35Config,
    taps: HashMap<String, Vec<f32>>,
    prompts: Vec<Prompt>,
    /// `lm_head.weight`, `[vocab, d]`.
    head: Vec<f32>,
    /// The final norm's applied gain `1 + w`.
    final_gain: Vec<f32>,
}

impl Golden {
    fn tap(&self, name: &str) -> &[f32] {
        self.taps.get(name).unwrap_or_else(|| panic!("golden has no tap {name}; re-run {DUMPER}"))
    }
}

fn golden() -> Option<Golden> {
    let dir = brain_testutil::testdata_path("golden/qwen35moe/tiny");
    let src = brain_testutil::golden::Source::open(&dir, DUMPER)?;
    let ckpt = dir.join("checkpoint");
    let cfg = qwen35moe::hf_source::hf_config(&ckpt).unwrap_or_else(|e| panic!("{e}"));
    let identity = [
        ("vocab_size", cfg.vocab),
        ("hidden_size", cfg.d_model),
        ("num_hidden_layers", cfg.n_layers),
        ("num_attention_heads", cfg.n_heads),
        ("num_key_value_heads", cfg.n_kv_heads),
        ("head_dim", cfg.head_dim),
        ("linear_key_head_dim", cfg.linear_key_head_dim),
        ("linear_value_head_dim", cfg.linear_value_head_dim),
        ("linear_num_key_heads", cfg.linear_num_key_heads),
        ("linear_num_value_heads", cfg.linear_num_value_heads),
        ("moe_intermediate_size", cfg.moe_intermediate_size),
        ("shared_expert_intermediate_size", cfg.shared_expert_intermediate_size),
        ("num_experts", cfg.n_experts),
        ("num_experts_per_tok", cfg.top_k),
    ];
    if !src.require(&identity.map(|(k, v)| (k, v as i64))) {
        return None;
    }
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest")).expect("manifest json");
    let meta = &manifest["_meta"];
    // The reference names each layer's mixer explicitly; the config derives them from the interval.
    let want: Vec<LayerType> = meta["layer_types"]
        .as_array()
        .expect("layer_types")
        .iter()
        .map(|t| match t.as_str() {
            Some("linear_attention") => LayerType::Linear,
            Some("full_attention") => LayerType::Full,
            other => panic!("unknown layer type {other:?}"),
        })
        .collect();
    assert_eq!(cfg.layer_types(), want, "the config's layer schedule is not the reference's");

    let read = |path: PathBuf| -> HashMap<String, Vec<f32>> {
        checkpoint::safetensors::read(&path.to_string_lossy()).unwrap_or_else(|e| panic!("{e}")).into_iter().map(|t| (t.name, t.data)).collect()
    };
    let taps = read(dir.join("qwen35moe_tiny.safetensors"));
    let mut weights = read(ckpt.join("model.safetensors"));
    let head = weights.remove("lm_head.weight").expect("an untied lm_head");
    let final_gain = weights.remove("model.language_model.norm.weight").expect("final norm").iter().map(|w| 1.0 + w).collect();
    let prompts = meta["prompts"]
        .as_array()
        .expect("prompts")
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let tokens: Vec<u32> = taps[&format!("p{i}.tokens")].iter().map(|&v| v.round() as u32).collect();
            assert_eq!(tokens.len() as u64, p["len"].as_u64().expect("len"));
            Prompt { tokens, decode_from: p["decode_from"].as_u64().expect("decode_from") as usize }
        })
        .collect();
    Some(Golden { ckpt, cfg, taps, prompts, head, final_gain })
}

fn gpu() -> Option<Gpu> {
    let gpu = Gpu::new(pipelines());
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable("the MoE router kernel needs workgroup barriers (not the CPU JIT)");
        return None;
    }
    Some(gpu)
}

/// Rounded operations on the longest chain through decoder layer `ty`, input residual to output residual, for a
/// sequence of `t` positions. Every reduction counts its length, an elementwise step a few. The Gated DeltaNet term
/// takes the longer of the chunked form (a 64-row chunk's forward substitution, `c^2 / 2`, plus the state carried
/// across every chunk) and the recurrent form (the state updated once per position), since prefill takes one and
/// decode the other.
fn layer_chain(cfg: &Qwen35Config, ty: LayerType, t: usize) -> usize {
    let d = cfg.d_model as usize;
    let norm = 2 * d + 4;
    let mixer = match ty {
        LayerType::Linear => {
            let (dk, dv) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
            let c = GDN_CHUNK.min(t);
            let chunked = c * c / 2 + 2 * c + t.div_ceil(c) * (c + dk + 4) + dk + c;
            let recurrent = t * (dk + dv + 6);
            d + cfg.linear_conv_kernel_dim as usize + dk + 16 + chunked.max(recurrent) + 2 * dv + 8 + cfg.linear_value_dim() as usize
        }
        LayerType::Full => {
            let hd = cfg.head_dim as usize;
            d + 2 * hd + 8 + hd + 2 * t + 8 + cfg.q_dim() as usize
        }
    };
    let (e, k) = (cfg.n_experts as usize, cfg.top_k as usize);
    let ff = cfg.moe_intermediate_size.max(cfg.shared_expert_intermediate_size) as usize;
    let moe = d + e + 4 + 2 * k + 2 + d + 4 + ff + k + 4 + d;
    norm + mixer + 1 + norm + moe + 1
}

/// The chain length to the residual after `layers` layers.
fn chain_to(cfg: &Qwen35Config, layers: usize, t: usize) -> usize {
    cfg.layer_types().iter().take(layers).map(|ty| layer_chain(cfg, *ty, t)).sum()
}

/// What one comparison found: the worst error as a fraction of its element's bound, and the relative L2.
struct Check {
    name: String,
    worst_ratio: f64,
    rel_l2: f64,
}

#[derive(Default)]
struct Ledger {
    rows: Vec<Check>,
}

impl Ledger {
    /// Compare `got` against `want` element by element, each against `bound[i]`.
    fn check(&mut self, name: &str, got: &[f32], want: &[f32], bound: &[f64]) {
        assert_eq!(got.len(), want.len(), "{name}: length");
        assert_eq!(bound.len(), want.len(), "{name}: bound length");
        assert!(got.iter().all(|v| v.is_finite()), "{name}: non-finite output");
        let worst_ratio = got.iter().zip(want).zip(bound).map(|((g, w), b)| f64::from((g - w).abs()) / b.max(f64::MIN_POSITIVE)).fold(0.0, f64::max);
        self.rows.push(Check { name: name.to_string(), worst_ratio, rel_l2: brain_testutil::parity::rel_l2(got, want) });
    }

    fn finish(self, what: &str) {
        let mut failed = Vec::new();
        for r in &self.rows {
            eprintln!("{what}: {:<28} worst |err|/bound {:.3e}  rel-L2 {:.3e}", r.name, r.worst_ratio, r.rel_l2);
            if r.worst_ratio > 1.0 {
                failed.push(r.name.as_str());
            }
        }
        assert!(failed.is_empty(), "{what}: past the binary32 bound: {failed:?}");
    }
}

/// Per-element bound of the residual after `layers` layers of prompt `p`: `layers == 0` is the embedding (a gather,
/// exact up to one rounding).
fn residual_bound(g: &Golden, p: usize, layers: usize, t: usize) -> Vec<f64> {
    if layers == 0 {
        return g.tap(&format!("p{p}.embed")).iter().map(|e| agreement_bound(1, f64::from(e.abs()))).collect();
    }
    let k = chain_to(&g.cfg, layers, t);
    g.tap(&format!("p{p}.layer{}.abs_sum", layers - 1)).iter().map(|&s| agreement_bound(k, f64::from(s))).collect()
}

/// Per-logit bound for the rows of prompt `p` at positions `rows`. The final norm scales each residual element by a
/// positive factor, so the absolute-term sum of a normed element is the residual's scaled the same way; the head's
/// terms sum `|W[v, j]|` times those.
fn logit_bound(g: &Golden, p: usize, rows: std::ops::Range<usize>, t: usize) -> Vec<f64> {
    let cfg = &g.cfg;
    let (d, v, l) = (cfg.d_model as usize, cfg.vocab as usize, cfg.n_layers as usize);
    let k = chain_to(cfg, l, t) + 2 * d + 4 + d;
    let res = g.tap(&format!("p{p}.layer{}.out", l - 1));
    let abs = g.tap(&format!("p{p}.layer{}.abs_sum", l - 1));
    let mut out = Vec::with_capacity(rows.len() * v);
    for r in rows {
        let x = &res[r * d..(r + 1) * d];
        let rinv = 1.0 / ((x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / d as f64) + f64::from(cfg.rms_eps)).sqrt();
        let s_norm: Vec<f64> = (0..d).map(|j| f64::from(g.final_gain[j].abs()) * rinv * f64::from(abs[r * d + j])).collect();
        for row in g.head.chunks(d) {
            out.push(agreement_bound(k, row.iter().zip(&s_norm).map(|(w, s)| f64::from(w.abs()) * s).sum()));
        }
    }
    out
}

/// The smallest routing margin the reference recorded over every layer and token of every prompt.
fn min_router_gap(g: &Golden) -> f32 {
    (0..g.prompts.len())
        .flat_map(|p| (0..g.cfg.n_layers as usize).map(move |l| (p, l)))
        .flat_map(|(p, l)| g.tap(&format!("p{p}.layer{l}.router_gap")).iter().copied())
        .fold(f32::INFINITY, f32::min)
}

fn source(g: &Golden) -> checkpoint::packed::PackedSource {
    qwen35moe::hf_source::open_dir(&g.ckpt, &g.cfg).unwrap_or_else(|e| panic!("{e}"))
}

fn fp32_model(gpu: &Gpu, g: &Golden, t: u32) -> Qwen35 {
    let mut m = Qwen35::new_on_src(gpu.share(), g.cfg.clone(), 1, t, &source(g));
    m.set_gemm_precision(GemmPrecision::Fp32Reference);
    m
}

/// Spec: brain's full-sequence forward on the reference's own weights, read from the release layout, reproduces the
/// reference's embedding, every decoder layer's output and the logits of every prompt within the binary32 bound.
#[test]
fn prefill_matches_the_reference_layer_by_layer() {
    let Some(g) = golden() else { return };
    let Some(gpu) = gpu() else { return };
    let (d, v) = (g.cfg.d_model as usize, g.cfg.vocab as usize);
    eprintln!("reference routing margin: smallest top-k gap {:.3e}", min_router_gap(&g));
    let m = fp32_model(&gpu, &g, T);
    let mut ledger = Ledger::default();
    for (p, prompt) in g.prompts.iter().enumerate() {
        let n = prompt.tokens.len();
        let logits = m.logits_all(&prompt.tokens);
        ledger.check(&format!("p{p} (L={n}) embed"), &m.read_residual(0)[..n * d], g.tap(&format!("p{p}.embed")), &residual_bound(&g, p, 0, n));
        for l in 0..g.cfg.n_layers as usize {
            let got = m.read_residual(l + 1);
            ledger.check(&format!("p{p} layer{l}.out"), &got[..n * d], g.tap(&format!("p{p}.layer{l}.out")), &residual_bound(&g, p, l + 1, n));
        }
        assert_eq!(logits.len(), n * v);
        ledger.check(&format!("p{p} logits"), &logits, g.tap(&format!("p{p}.logits")), &logit_bound(&g, p, 0..n, n));
    }
    ledger.finish("prefill");
}

/// Spec: a chunked prefill of the first `decode_from` tokens followed by one [`Qwen35::step`] per remaining token
/// (the recurrent Gated DeltaNet state, the conv window and the GQA KV cache carried between calls) gives the logits of
/// the reference's own cached decode at every position, within the binary32 bound.
#[test]
fn cached_decode_matches_the_reference_cached_decode() {
    let Some(g) = golden() else { return };
    let Some(gpu) = gpu() else { return };
    let m = fp32_model(&gpu, &g, T);
    let v = g.cfg.vocab as usize;
    let mut ledger = Ledger::default();
    for (p, prompt) in g.prompts.iter().enumerate() {
        let (n, from) = (prompt.tokens.len(), prompt.decode_from);
        m.reset_decode_cache();
        let mut logits = m.head_logits(&m.prefill_chunked(&prompt.tokens[..from], 32));
        for &tok in &prompt.tokens[from..] {
            logits.extend(m.head_logits(&m.step(tok)));
        }
        assert_eq!(m.decode_pos() as usize, n);
        assert_eq!(logits.len(), (n - from + 1) * v);
        ledger.check(&format!("p{p} (L={n}, prefill {from})"), &logits, g.tap(&format!("p{p}.decode_logits")), &logit_bound(&g, p, from - 1..n, n));
    }
    ledger.finish("cached decode");
}

/// Spec: the serving engine (paged KV pool, recurrent state pool, chunked prefill rounds, the batched decode step) on
/// the release-layout checkpoint reproduces the reference: each prompt's last prefill hidden state and every decode
/// step's logits, served alone and as a batch of three sequences of different lengths decoded together.
#[test]
fn the_served_engine_matches_the_reference() {
    let Some(g) = golden() else { return };
    let Some(gpu) = gpu() else { return };
    let (d, v, l) = (g.cfg.d_model as usize, g.cfg.vocab as usize, g.cfg.n_layers as usize);
    let n_prompts = g.prompts.len();
    let opts = EngineOptions::new(T, n_prompts as u32).with_gemm_precision(GemmPrecision::Fp32Reference);
    let mut engine = Engine::from_source_on_parent(&gpu, g.cfg.clone(), &source(&g), opts);
    let hidden_bound = |p: usize, row: usize, t: usize| -> Vec<f64> {
        // The final norm of row `row`, bounded as `logit_bound` bounds its input to the head.
        let res = &g.tap(&format!("p{p}.layer{}.out", l - 1))[row * d..(row + 1) * d];
        let abs = &g.tap(&format!("p{p}.layer{}.abs_sum", l - 1))[row * d..(row + 1) * d];
        let rinv = 1.0 / ((res.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / d as f64) + f64::from(g.cfg.rms_eps)).sqrt();
        let k = chain_to(&g.cfg, l, t) + 2 * d + 4;
        (0..d).map(|j| agreement_bound(k, f64::from(g.final_gain[j].abs()) * rinv * f64::from(abs[j]))).collect()
    };
    let final_row = |p: usize, row: usize| g.tap(&format!("p{p}.final_hidden"))[row * d..(row + 1) * d].to_vec();
    let decode_row = |p: usize, s: usize| g.tap(&format!("p{p}.decode_logits"))[s * v..(s + 1) * v].to_vec();

    let mut ledger = Ledger::default();
    for (p, prompt) in g.prompts.iter().enumerate() {
        let (n, from) = (prompt.tokens.len(), prompt.decode_from);
        let mut table = BlockTable::new();
        let hidden = engine.prefill(&mut table, &prompt.tokens[..from]);
        ledger.check(&format!("alone p{p} prefill {from} hidden"), &hidden, &final_row(p, from - 1), &hidden_bound(p, from - 1, n));
        let (mut got, mut want) = (Vec::new(), Vec::new());
        for (s, &tok) in prompt.tokens[from..].iter().enumerate() {
            let mut refs = [&mut table];
            got.extend(engine.forward_batched_logit_rows(&mut refs, &[tok]).remove(0));
            want.extend(decode_row(p, s + 1));
        }
        ledger.check(&format!("alone p{p} decode {} steps", n - from), &got, &want, &logit_bound(&g, p, from..n, n));
        engine.release_table(&mut table);
    }

    // The batch: every prompt prefilled, then decoded together for as many steps as the shortest has left.
    let mut tables: Vec<BlockTable> = g.prompts.iter().map(|_| BlockTable::new()).collect();
    for (p, (prompt, table)) in g.prompts.iter().zip(tables.iter_mut()).enumerate() {
        let from = prompt.decode_from;
        let hidden = engine.prefill(table, &prompt.tokens[..from]);
        ledger.check(&format!("batch p{p} prefill hidden"), &hidden, &final_row(p, from - 1), &hidden_bound(p, from - 1, prompt.tokens.len()));
    }
    let steps = g.prompts.iter().map(|p| p.tokens.len() - p.decode_from).min().expect("prompts");
    let mut rows: Vec<(Vec<f32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); n_prompts];
    for s in 0..steps {
        let inputs: Vec<u32> = g.prompts.iter().map(|p| p.tokens[p.decode_from + s]).collect();
        let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
        for (p, out) in engine.forward_batched_logit_rows(&mut refs, &inputs).into_iter().enumerate() {
            rows[p].0.extend(out);
            rows[p].1.extend(decode_row(p, s + 1));
        }
    }
    for (p, (got, want)) in rows.iter().enumerate() {
        let from = g.prompts[p].decode_from;
        ledger.check(&format!("batch p{p} decode {steps} steps"), got, want, &logit_bound(&g, p, from..from + steps, g.prompts[p].tokens.len()));
    }
    for t in &mut tables {
        engine.release_table(t);
    }
    ledger.finish("engine");
}

/// Spec: the int8 tier computes the reference's W8A8 function. The dumper evaluates the reference with brain's int8
/// rounding applied at every site the tier quantizes (`p{i}.int8.*`: mixer projections, routed and shared experts and
/// the untied head, weight and activation alike in 32-element blocks); the tier's logits must lie nearer that function
/// than the function itself lies from the fp32 reference. A different quantization (another scale granularity, an
/// activation left unrounded, a quantized router) is a different function and lands as far from the reference's int8
/// as int8 is from fp32. Agreement tighter than this is not on offer: the W8A8 function is discontinuous, and two
/// evaluations of it that differ only in binary32 rounding move a code across a rounding midpoint and diverge layer by
/// layer (the dumper's own binary32 against binary64 evaluation, `_meta.int8[*].fp32_vs_fp64`, shows it).
#[test]
fn the_int8_tier_computes_the_reference_w8a8_function() {
    let Some(g) = golden() else { return };
    let Some(gpu) = gpu() else { return };
    if !gpu.caps().numeric.int8_dot {
        return brain_testutil::skip_unavailable("no packed int8 dot on this device");
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(brain_testutil::testdata_path("golden/qwen35moe/tiny/manifest.json")).expect("manifest")).expect("manifest json");
    let mut m = Qwen35::new_on_tier_src(gpu.share(), g.cfg.clone(), 1, T, &source(&g), &TierPolicy::uniform(Dtype::I8));
    m.set_gemm_precision(GemmPrecision::Fp32Reference);
    assert!(m.moe_int8_active(), "the int8 tier must run the int8 expert banks");
    assert!(m.moe_shared_in_bank(), "the shared expert has the routed experts' shape and rides in the bank");
    assert!(m.head_is_quantized(), "the untied head is int8 under an int8 policy");
    let d = g.cfg.d_model as usize;
    for (p, prompt) in g.prompts.iter().enumerate() {
        let n = prompt.tokens.len();
        let logits = m.logits_all(&prompt.tokens);
        let layers: Vec<String> = (0..g.cfg.n_layers as usize)
            .map(|l| format!("{:.1e}", brain_testutil::parity::rel_l2(&m.read_residual(l + 1)[..n * d], g.tap(&format!("p{p}.int8.layer{l}.out")))))
            .collect();
        let w8a8 = g.tap(&format!("p{p}.int8.logits"));
        let to_w8a8 = brain_testutil::parity::rel_l2(&logits, w8a8);
        let loss = brain_testutil::parity::rel_l2(w8a8, g.tap(&format!("p{p}.logits")));
        let to_fp32 = brain_testutil::parity::rel_l2(&logits, g.tap(&format!("p{p}.logits")));
        let spread = manifest["_meta"]["int8"][p]["fp32_vs_fp64"].as_f64().expect("int8 spread");
        eprintln!(
            "int8 p{p} (L={n}): residual vs W8A8 reference per layer [{}], logits vs W8A8 reference {to_w8a8:.3e}, \
             W8A8 reference vs fp32 {loss:.3e}, tier vs fp32 {to_fp32:.3e}, W8A8 reference binary32 vs binary64 {spread:.3e}",
            layers.join(" ")
        );
        assert!(to_w8a8 < loss, "int8 p{p}: the tier is {to_w8a8:.3e} from the reference's W8A8 function, which is itself {loss:.3e} from fp32");
    }
}
