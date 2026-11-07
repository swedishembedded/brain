// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding for clients serving large models on modest hardware.
// If your team needs expertise in gating a ported drafter against its reference then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Shared by the gates: the reference's tensors and the scaled error.

use checkpoint::safetensors::StTensor;

/// The tensors of a `golden.safetensors` written by `tools/goldens/dflash2_reference_forward.py`.
pub struct Golden(Vec<StTensor>);

impl Golden {
    pub fn read(path: &str) -> Golden {
        Golden(checkpoint::safetensors::read(path).unwrap_or_else(|e| panic!("reading golden {path}: {e}")))
    }

    pub fn get(&self, name: &str) -> &[f32] {
        &self.0.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden has no tensor {name}")).data
    }

    pub fn ids(&self, name: &str) -> Vec<u32> {
        self.get(name).iter().map(|&v| v as u32).collect()
    }
}

/// `max|got - want|` over `max|want|`: the error relative to the tensor's own scale.
pub fn scaled_error(label: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{label}: {} values, the reference has {}", got.len(), want.len());
    assert!(got.iter().all(|v| v.is_finite()), "{label}: non-finite value");
    let worst = got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
    let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
    let err = worst / scale.max(f32::MIN_POSITIVE);
    eprintln!("{label:>22}: max_abs {worst:.3e}  scale {scale:.3e}  scaled {err:.3e}  n={}", got.len());
    err
}

/// `rows` of `table` (`[n, width]`) for `ids`.
pub fn gather(table: &[f32], width: usize, ids: &[u32]) -> Vec<f32> {
    ids.iter().flat_map(|&i| table[i as usize * width..(i as usize + 1) * width].iter().copied()).collect()
}

/// `x @ head^T` for `x` of `[rows, width]` and `head` of `[vocab, width]`.
pub fn logits(x: &[f32], head: &[f32], width: usize) -> Vec<f32> {
    x.chunks(width).flat_map(|row| head.chunks(width).map(move |h| row.iter().zip(h).map(|(a, b)| a * b).sum::<f32>())).collect()
}
