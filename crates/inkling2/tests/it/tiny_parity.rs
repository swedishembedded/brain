// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements parity-proven ports of looped mixture-of-experts models for its clients. If your
// team needs expertise in reproducing a reference forward operator by operator then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: the forward reproduces the reference (`modular_inkling2_loop.py`, f32) on the tiny looped config at every
//! logical visit - its input and output partial, block states, depth-attention inputs, messages and message-gate
//! outputs, attention, convolutions and the state they leave, relative states, router logits, choices and weights,
//! the MoE output - and in the readout and the soft-capped logits.

use std::collections::HashMap;

use inkling2::{Inkling2, Probe};

/// The f32 reference's own spread: its cached decode differs from its uncached forward by up to 1.7e-5 on these
/// logits (`decode_vs_full_max_abs` in the golden's manifest); a value agrees when it is within that order, relative to
/// the tensor's scale.
const TOL: f32 = 5e-5;

#[derive(Default)]
struct Recorder(HashMap<String, Vec<f32>>);

impl Probe for Recorder {
    fn record(&mut self, key: &str, values: Vec<f32>) {
        assert!(self.0.insert(key.to_string(), values).is_none(), "{key} recorded twice");
    }
}

fn golden(path: &std::path::Path) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    checkpoint::safetensors::read(&path.to_string_lossy()).expect("read golden").into_iter().map(|t| (t.name, (t.shape, t.data))).collect()
}

/// The largest difference relative to the reference tensor's own scale.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / scale
}

pub fn ops() -> model::ops::Ops {
    static LIST: std::sync::OnceLock<&'static [(&'static str, &'static str)]> = std::sync::OnceLock::new();
    let list = LIST.get_or_init(|| Box::leak(inkling2::kernels().into_boxed_slice()));
    model::ops::Ops::new(gpu_core::testgpu::dev(list)).expect("Ops::new")
}

#[test]
fn the_kernel_list_is_complete_for_ops() {
    model::ops::assert_kernel_list_complete(&inkling2::kernels());
}

#[test]
fn the_tiny_reference_is_reproduced_at_every_visit_and_in_the_logits() {
    let dir = brain_testutil::testdata_path("golden/inkling2/tiny");
    if !dir.join("golden.safetensors").is_file() {
        brain_testutil::skip("testdata/golden/inkling2/tiny absent (make testdata/fetch, or run tools/goldens/inkling2_dump_reference.py tiny)");
        return;
    }
    let gold = golden(&dir.join("golden.safetensors"));
    let m = Inkling2::load(ops(), &dir.join("checkpoint")).unwrap();
    let tokens: Vec<u32> = gold["tokens"].1.iter().map(|&v| v as u32).collect();
    let mut rec = Recorder::default();
    let logits = m.forward_probed(&tokens, &mut rec).unwrap();
    let got = rec.0;
    assert_eq!(logits, got["logits"]);

    let cfg = m.config();
    let mut report = Vec::new();
    let mut check = |key: &str, mine: &[f32], want: &[f32]| {
        let e = rel_err(mine, want);
        report.push(format!("{key}: {e:.2e}"));
        assert!(e <= TOL, "{key}: relative error {e:.3e} against the reference");
    };
    for key in ["embed", "final_in", "final_norm", "logits"] {
        check(key, &got[key], &gold[key].1);
    }
    let mut checked = 0;
    for v in m.plan().visits() {
        let p = |name: &str| format!("visit{}.{name}", v.index);
        for name in ["in", "out", "block_states", "attn_in", "mlp_in", "messages", "gated_q", "gated_k", "gated_v", "gated_mlp", "attn", "attn_conv", "mlp", "mlp_conv", "router_logits", "router_weights"] {
            let key = p(name);
            match (got.get(&key), gold.get(&key)) {
                (Some(mine), Some((_, want))) => {
                    check(&key, mine, want);
                    checked += 1;
                }
                (None, None) => {}
                (mine, want) => panic!("{key}: recorded {} / in the golden {}", mine.is_some(), want.is_some()),
            }
        }
        assert_eq!(got[&p("router_indices")], gold[&p("router_indices")].1, "{}: router choices", p(""));
        // Relative states: the golden's are after the log-position scale on full layers, which the attention kernel
        // applies to the bias it forms from them.
        let (rel, want) = (&got[&p("rel_states")], &gold[&p("rel_states")].1);
        let per_row = rel.len() / tokens.len();
        let scaled: Vec<f32> = match v.attention {
            model::looped::Attention::Full => rel.iter().enumerate().map(|(i, r)| r * model::flash_ext::log_position_scale((i / per_row) as u32, cfg.log_alpha, cfg.log_floor as f32)).collect(),
            model::looped::Attention::Window(_) => rel.clone(),
        };
        check(&p("rel_states"), &scaled, want);
        // Convolution states: the reference keeps the last `taps` inputs, a decode reads the last `taps - 1`.
        for j in 0..4 {
            let (shape, want) = &gold[&format!("prefill.visit{}.conv{j}", v.index)];
            let (c, k) = (shape[0], shape[1]);
            let tail: Vec<f32> = (0..c).flat_map(|ch| want[ch * k + 1..(ch + 1) * k].iter().copied()).collect();
            check(&p(&format!("conv{j}")), &got[&p(&format!("conv{j}"))], &tail);
        }
    }
    assert!(checked >= 10 * 8, "only {checked} visit tensors compared");
    eprintln!("{}", report.join("\n"));
}
