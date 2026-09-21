// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity vs the REAL `convaiinnovations/laya` `DecisionModel`
//! (`rl_common.py`, Apache-2.0): `scripts/parity-dump/modernbert.py` wraps
//! the same tiny random-weight `ModernBertModel` `tests/parity.rs` checks
//! with an inline copy of the real head module and dumps its weights, a
//! synthetic marker/qtype batch, and both `logits` and `act_logits`.
//!
//! This is the test that actually catches the head's two traps (see
//! `crates/modernbert/src/laya.rs`'s own module doc):
//!
//! - the 2 head layers' FFN uses **RELU**, not GELU (opposite of the
//!   scorer's own explicit GELU and of the trunk's own GELU);
//! - the 2 head layers' LayerNorm/Linear/attention are all **biased**
//!   (opposite of the frozen trunk's `norm_bias: false`).
//!
//! Both were deliberately introduced and confirmed to fail this test while
//! building it - see the M3 commit message for which one and the measured
//! failure.
//!
//! Fixtures are NOT in git, same convention as `tests/parity.rs` - this test
//! reads the SAME fixture directory (`modernbert.py` dumps both the trunk's
//! and the head's fixture in one manifest) and SKIPS when it is absent.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use gpu_core::DeviceBuffer;
use modernbert::config::ModernBertConfig;
use modernbert::kern::PIPELINES;
use modernbert::laya::{LayaConfig, LayaHead};
use modernbert::model::ModernBert;

fn fixture_dir() -> PathBuf {
    if let Ok(d) = std::env::var("MODERNBERT_FIXTURE_DIR") {
        return PathBuf::from(d);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/modernbert")
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

struct Fixture {
    manifest: serde_json::Value,
    dir: PathBuf,
}

impl Fixture {
    fn load() -> Option<Fixture> {
        let dir = fixture_dir();
        let m = dir.join("manifest.json");
        if !m.exists() {
            brain_testutil::skip(&format!(
                "{} absent - run `python3 scripts/parity-dump/modernbert.py --out <scratch>` \
                 and copy the output into {}",
                m.display(),
                dir.display()
            ));
            return None;
        }
        let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&m).unwrap()).unwrap();
        if manifest["head_weights"].as_object().is_none_or(|o| o.is_empty()) {
            brain_testutil::skip(
                "fixture predates Laya M3's head dump - regenerate with the current \
                 scripts/parity-dump/modernbert.py",
            );
            return None;
        }
        Some(Fixture { manifest, dir })
    }

    fn tensor(&self, section: &str, name: &str) -> (Vec<usize>, Vec<f32>) {
        let e = &self.manifest[section][name];
        assert!(!e.is_null(), "fixture missing {section}/{name}");
        let shape: Vec<usize> = e["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let data = read_f32(&self.dir.join(e["file"].as_str().unwrap()));
        assert_eq!(data.len(), shape.iter().product::<usize>().max(1), "{section}/{name}");
        (shape, data)
    }

    fn config(&self) -> ModernBertConfig {
        ModernBertConfig::from_json(&self.manifest["config"])
    }

    fn spans(&self) -> Vec<(u32, u32)> {
        self.manifest["spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                let a = s.as_array().unwrap();
                (a[0].as_u64().unwrap() as u32, a[1].as_u64().unwrap() as u32)
            })
            .collect()
    }

    /// The trunk's own weights, renamed exactly like `tests/parity.rs`'s own
    /// `weights()` - duplicated here rather than shared, since the two test
    /// binaries do not share code today and this crate's tests are small.
    fn trunk_weights(&self, cfg: &ModernBertConfig) -> HashMap<String, Vec<f32>> {
        let mut w = HashMap::new();
        let mut put = |brain_name: &str, hf_name: &str| {
            let (_, data) = self.tensor("weights", hf_name);
            w.insert(brain_name.to_string(), data);
        };
        put("tok.weight", "embeddings.tok_embeddings.weight");
        put("emb_norm.weight", "embeddings.norm.weight");
        for l in 0..cfg.n_layers as usize {
            if cfg.has_attn_norm(l) {
                put(&format!("blocks.{l}.attn_norm.weight"), &format!("layers.{l}.attn_norm.weight"));
            }
            put(&format!("blocks.{l}.qkv.weight"), &format!("layers.{l}.attn.Wqkv.weight"));
            put(&format!("blocks.{l}.proj.weight"), &format!("layers.{l}.attn.Wo.weight"));
            put(&format!("blocks.{l}.mlp_norm.weight"), &format!("layers.{l}.mlp_norm.weight"));
            put(&format!("blocks.{l}.mlp.wi.weight"), &format!("layers.{l}.mlp.Wi.weight"));
            put(&format!("blocks.{l}.mlp.wo.weight"), &format!("layers.{l}.mlp.Wo.weight"));
        }
        put("final_norm.weight", "final_norm.weight");
        w
    }

    /// The head's own weights, renamed from the real `DecisionModel`'s own
    /// dotted names (`head.layers.{l}.*`, `scorer.*`, `act_head.*`,
    /// `type_emb.*` - the SAME names the real released checkpoint uses, per
    /// the Laya plan's own verified ground truth) to this crate's short
    /// `laya::tensor_manifest` names. This mapping is the seam Laya M4's real
    /// importer will need, just ad hoc here.
    fn head_weights(&self, cfg: &LayaConfig) -> HashMap<String, Vec<f32>> {
        let mut w = HashMap::new();
        let mut put = |brain_name: &str, hf_name: &str| {
            let (_, data) = self.tensor("head_weights", hf_name);
            w.insert(brain_name.to_string(), data);
        };
        put("type_emb.weight", "type_emb.weight");
        for l in 0..cfg.head_layers as usize {
            let p = format!("head.{l}");
            let hf = format!("head.layers.{l}");
            put(&format!("{p}.attn.in_proj.weight"), &format!("{hf}.self_attn.in_proj_weight"));
            put(&format!("{p}.attn.in_proj.bias"), &format!("{hf}.self_attn.in_proj_bias"));
            put(&format!("{p}.attn.out_proj.weight"), &format!("{hf}.self_attn.out_proj.weight"));
            put(&format!("{p}.attn.out_proj.bias"), &format!("{hf}.self_attn.out_proj.bias"));
            // PyTorch's `nn.TransformerEncoderLayer` names its FFN
            // `linear1`/`linear2` - this crate calls them `ff1`/`ff2`.
            put(&format!("{p}.ff1.weight"), &format!("{hf}.linear1.weight"));
            put(&format!("{p}.ff1.bias"), &format!("{hf}.linear1.bias"));
            put(&format!("{p}.ff2.weight"), &format!("{hf}.linear2.weight"));
            put(&format!("{p}.ff2.bias"), &format!("{hf}.linear2.bias"));
            put(&format!("{p}.norm1.weight"), &format!("{hf}.norm1.weight"));
            put(&format!("{p}.norm1.bias"), &format!("{hf}.norm1.bias"));
            put(&format!("{p}.norm2.weight"), &format!("{hf}.norm2.weight"));
            put(&format!("{p}.norm2.bias"), &format!("{hf}.norm2.bias"));
        }
        // `scorer = nn.Sequential(LayerNorm, Linear, GELU, Linear)` - index 2
        // is the parameter-free GELU, so indices 0/1/3.
        put("scorer.norm.weight", "scorer.0.weight");
        put("scorer.norm.bias", "scorer.0.bias");
        put("scorer.fc1.weight", "scorer.1.weight");
        put("scorer.fc1.bias", "scorer.1.bias");
        put("scorer.fc2.weight", "scorer.3.weight");
        put("scorer.fc2.bias", "scorer.3.bias");
        // `act_head = nn.Sequential(Linear, GELU, Linear)` - index 1 is GELU.
        put("act.fc1.weight", "act_head.0.weight");
        put("act.fc1.bias", "act_head.0.bias");
        put("act.fc2.weight", "act_head.2.weight");
        put("act.fc2.bias", "act_head.2.bias");
        w
    }
}

#[test]
fn parity_tiny_head_matches_the_reference_decision_model() {
    let Some(fx) = Fixture::load() else { return };
    let cfg = fx.config();
    let trunk_weights = fx.trunk_weights(&cfg);

    let laya_cfg = LayaConfig::new(cfg.d_model);
    let head_weights = fx.head_weights(&laya_cfg);

    // Full-coverage discipline, same as `tests/parity.rs`.
    let expected = modernbert::laya::tensor_manifest(&laya_cfg);
    assert_eq!(expected.len(), head_weights.len(), "laya::tensor_manifest vs fixture head weight count");
    for (name, shape) in &expected {
        let got = head_weights.get(name).unwrap_or_else(|| panic!("fixture lacks head weight {name}"));
        assert_eq!(got.len(), shape.iter().product::<usize>(), "{name} numel");
    }

    let spans = fx.spans();
    let rows: u32 = spans.iter().map(|&(_, l)| l).sum();
    let max_span = spans.iter().map(|&(_, l)| l).max().unwrap();

    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut enc = ModernBert::new_on(gpu.share(), cfg.clone(), rows, max_span, &trunk_weights);
    let (_, ids_f32) = fx.tensor("inputs", "ids");
    let ids: Vec<u32> = ids_f32.iter().map(|&v| v.round() as u32).collect();
    enc.set_batch(&ids, &spans);
    enc.forward();
    enc.gpu.poll_wait();

    let (_, qtype_f32) = fx.tensor("head_inputs", "qtype");
    let qtype: Vec<u32> = qtype_f32.iter().map(|&v| v.round() as u32).collect();
    let arity: Vec<usize> = fx.manifest["head_inputs"]["arity"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let (_, marker_rows_f32) = fx.tensor("head_inputs", "marker_rows_absolute");
    let marker_rows: Vec<u32> = marker_rows_f32.iter().map(|&v| v.round() as u32).collect();

    let n_markers = marker_rows.len() as u32;
    let n_questions = spans.len() as u32;
    let mut head = LayaHead::new_on(gpu, laya_cfg, rows, max_span, n_markers, n_questions, &head_weights);
    let hidden_buf: &DeviceBuffer = enc.hidden_buf();
    head.set_call(hidden_buf, &spans, &qtype, &marker_rows, &arity);
    let (got_logits, got_act_logits) = head.forward();

    let (_, want_logits) = fx.tensor("head_output", "logits");
    let (_, want_act_logits) = fx.tensor("head_output", "act_logits");
    assert_eq!(got_logits.len(), want_logits.len());
    assert_eq!(got_act_logits.len(), want_act_logits.len());

    // Tolerance reasoning, same shape as `tests/parity.rs`'s own: both sides
    // fp32, tiny std=0.1 random weights, no bf16 quantization (that bar is
    // for the real released checkpoint, Laya M4/M8, not this fixture). The
    // head is shallower than the trunk (2 layers vs 4) but sits on top of
    // it, so the same `1e-4` bound the trunk measured ~3e-8 against applies
    // here with the same headroom for a slower device's FMA codegen.
    let max_abs_logits = got_logits.iter().zip(&want_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_abs_logits < 1e-4, "option logits parity: max abs diff {max_abs_logits} (tolerance 1e-4)");
    let max_abs_act = got_act_logits.iter().zip(&want_act_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_abs_act < 1e-4, "act logits parity: max abs diff {max_abs_act} (tolerance 1e-4)");
}
