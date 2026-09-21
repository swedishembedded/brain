// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity vs the REAL `transformers.models.modernbert.
//! modeling_modernbert.ModernBertModel`: a tiny random-weight ModernBERT was
//! run by `scripts/parity-dump/modernbert.py` on a fixed-seed packed
//! multi-span batch (two different-length sequences, both past `2*window`),
//! and brain must reproduce its final hidden states.
//!
//! This is the test that actually catches the GeGLU chunk-order trap (see
//! `crates/modernbert/src/model.rs`'s module doc): getting `wi_u`/`wi_v`
//! backwards produces a plausible-looking but numerically wrong forward that
//! no fixture-free structural test (`tests/structural.rs`) can catch, since
//! that one only asserts "finite and non-zero", not "correct".
//!
//! Fixtures are NOT in git (`crates/modernbert/tests/fixtures/` is
//! gitignored, same `brain-never-commit-goldens` convention as
//! `crates/diamond/tests/fixtures`). Regenerate with:
//!   python3 scripts/parity-dump/modernbert.py --out <scratch>/fixtures-modernbert
//!   cp <scratch>/fixtures-modernbert/* crates/modernbert/tests/fixtures/modernbert/
//! The test SKIPS (with that message) when the fixture directory is absent.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use modernbert::config::ModernBertConfig;
use modernbert::kern::PIPELINES;
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

    /// Every HF weight, renamed to this crate's own short tensor-manifest
    /// names (`ModernBertConfig::tensor_manifest`) - the dump script writes
    /// the REAL `ModernBertModel` module names on purpose (see its own
    /// module doc), so this mapping is the seam between "what the real
    /// checkpoint calls a tensor" and "what `crates/modernbert` calls it",
    /// the same seam Laya M4's real importer will need, just ad hoc here.
    fn weights(&self, cfg: &ModernBertConfig) -> HashMap<String, Vec<f32>> {
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
}

#[test]
fn parity_tiny_packed_multi_span_forward_matches_reference() {
    let Some(fx) = Fixture::load() else { return };
    let cfg = fx.config();
    let weights = fx.weights(&cfg);

    // Full-coverage discipline: the manifest must supply every tensor
    // `tensor_manifest()` expects, no more, no less - the same check
    // `crates/diamond/tests/parity.rs` runs before trusting a fixture.
    let expected = cfg.tensor_manifest();
    assert_eq!(expected.len(), weights.len(), "tensor_manifest vs fixture weight count");
    for (name, shape) in &expected {
        let got = weights.get(name).unwrap_or_else(|| panic!("fixture lacks {name}"));
        assert_eq!(got.len(), shape.iter().product::<usize>(), "{name} numel");
    }

    let spans = fx.spans();
    let rows: u32 = spans.iter().map(|&(_, l)| l).sum();
    let max_span = spans.iter().map(|&(_, l)| l).max().unwrap();

    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut m = ModernBert::new_on(gpu, cfg.clone(), rows, max_span, &weights);

    let (_, ids_f32) = fx.tensor("inputs", "ids");
    let ids: Vec<u32> = ids_f32.iter().map(|&v| v.round() as u32).collect();
    m.set_batch(&ids, &spans);
    m.forward();

    let got = m.hidden();
    let (_, want) = fx.tensor("output", "hidden");
    assert_eq!(got.len(), want.len());

    // Tolerance reasoning (following `crates/gradcheck`'s convention of
    // stating why, not a bare magic number): both sides are fp32, the
    // weights are tiny random values with no bf16 quantization involved (see
    // the module-level parity-tolerance note in the Laya plan - that bar is
    // for the REAL bf16 checkpoint, not this one), so agreement should be
    // close to fp32 epsilon accumulated over 4 pre-LN layers of GEMMs,
    // GeGLU, RoPE and windowed attention. Measured on this device: max abs
    // diff ~3e-8, essentially fp32 rounding noise from a different (but
    // equally valid) reduction order between brain's WGSL kernels and
    // PyTorch's own BLAS/SDPA kernels over the same real numbers. `1e-4` is
    // roughly three orders of magnitude looser than that measured number -
    // enough headroom for a slower device's different FMA codegen (the
    // `~3e-8` cross-kernel-file gap Laya M1's own windowed-attention tests
    // hit on the CPU JIT) without being loose enough to hide a real forward
    // bug.
    let max_abs = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1e-4, "forward parity: max abs diff {max_abs} (tolerance 1e-4)");
}
