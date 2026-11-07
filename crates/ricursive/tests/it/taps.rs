// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared by the two parity tests: reading the reference's taps and comparing
//! the forward's against them.
//!
//! Swedish Embedded AB implements solutions for proving that ported language
//! models reproduce their reference implementations for its clients. If your
//! team needs expertise in layer-by-layer parity validation then you can
//! procure our services by sending an email to info@swedishembedded.com.

use ricursive::Forward;

/// The reference's tensors (`tools/goldens/ricursive_dump_reference.py`).
pub struct Taps(Vec<checkpoint::safetensors::StTensor>);

impl Taps {
    pub fn read(path: &str) -> Taps {
        Taps(checkpoint::safetensors::read(path).unwrap_or_else(|e| panic!("reading golden {path}: {e}")))
    }

    pub fn get(&self, name: &str) -> &[f32] {
        &self.0.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden has no tensor {name}")).data
    }

    pub fn input_ids(&self) -> Vec<u32> {
        self.get("input_ids").iter().map(|&v| v as u32).collect()
    }
}

/// `max|got - want|` over `max|want|`: the error relative to the tap's own
/// scale, so a tap of magnitude 1 and one of magnitude 100 share a bound.
pub fn scaled_error(label: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{label}: {} values, reference has {}", got.len(), want.len());
    assert!(got.iter().all(|v| v.is_finite()), "{label}: non-finite value");
    let max_abs = got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
    let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
    let err = max_abs / scale.max(f32::MIN_POSITIVE);
    eprintln!("{label:>14}: max_abs {max_abs:.3e}  scale {scale:.3e}  scaled {err:.3e}  n={}", got.len());
    err
}

/// Assert every tap of `out` is within `tolerance` of the reference and return
/// the worst scaled error. Taps: `hidden.0` (embeddings), `hidden.{k+1}`,
/// `attn_out.{k}`, `final_hidden`, `logits`.
pub fn check(out: &Forward, taps: &Taps, tolerance: f32) -> f32 {
    let mut worst = 0.0f32;
    let mut one = |label: String, got: &[f32], want: &[f32]| {
        let err = scaled_error(&label, got, want);
        assert!(err <= tolerance, "{label}: scaled error {err:.3e} exceeds {tolerance:.1e}");
        worst = worst.max(err);
    };
    one("hidden.0".into(), &out.embeddings, taps.get("hidden.0"));
    for (k, (hidden, attn)) in out.layer_outputs.iter().zip(&out.attn_outputs).enumerate() {
        one(format!("attn_out.{k}"), attn, taps.get(&format!("attn_out.{k}")));
        one(format!("hidden.{}", k + 1), hidden, taps.get(&format!("hidden.{}", k + 1)));
    }
    one("final_hidden".into(), &out.final_hidden, taps.get("final_hidden"));
    one("logits".into(), &out.logits, taps.get("logits"));
    worst
}
