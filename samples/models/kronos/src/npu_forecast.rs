// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `npu-forecast`: an autoregressive Kronos forecast with both decoder graphs
//! (`decode_s1` gives the coarse token, `decode_s2` its refinement) running on
//! the Intel NPU through OpenVINO. The host does the BSQ tokenization, token
//! embedding, argmax and the sliding window. `verify` also runs the identical
//! rollout on the device core and reports how many tokens agree: the
//! NPU-versus-device parity check.
//!
//! The context is a synthetic OHLCV series of one window, since the graphs are
//! fixed-length. Needs OpenVINO and an Intel NPU at run time, and says so when
//! it has neither.
//!
//! Swedish Embedded AB implements autoregressive forecasting on NPUs and other
//! edge accelerators for its clients. If your team needs expertise in
//! exporting transformer decoders for OpenVINO then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::time::Instant;

use capability::{Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType, Progress};
use npu::openvino::{KronosS1Session, KronosS2Session};
use serde_json::json;

use crate::actions::Checkpoints;

/// Tokens generated when the caller names no horizon.
const DEFAULT_HORIZON: i64 = 8;
/// Longest horizon one run may ask for.
const MAX_HORIZON: f64 = 10_000.0;

pub fn spec() -> ActionSpec {
    let mut spec = ActionSpec::new("npu-forecast", "an autoregressive Kronos forecast with both decoder graphs on the Intel NPU through OpenVINO (a synthetic OHLCV context of one window)")
        .param(ParamSpec::new("horizon", ParamType::Int, "bars to generate").min(1.0).max(MAX_HORIZON).default(json!(DEFAULT_HORIZON)))
        .param(ParamSpec::new("context_len", ParamType::Int, "bars of context, which is the graphs' fixed window; 0 = the decoder's own max_context").min(0.0).max(MAX_HORIZON).default(json!(0)))
        .param(ParamSpec::new("verify", ParamType::Bool, "also run the rollout on the device core and report how many tokens agree").default(json!(false)))
        .output(BlobSpec::new("report", Media::Text, "what ran: the device, the time per step, the generated close"));
    spec.params.extend(npu::run_options::params());
    spec
}

pub struct NpuForecast(Checkpoints);

impl NpuForecast {
    pub fn new(checkpoints: Checkpoints) -> NpuForecast {
        NpuForecast(checkpoints)
    }
}

fn argmax(x: &[f32]) -> u32 {
    (0..x.len()).max_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap()).unwrap_or(0) as u32
}

/// The Kronos AR rollout, parameterized by the s1/s2 **core** so the identical
/// loop runs on the device (`core_forward_s1/s2`) or the NPU (the two sessions):
/// per step, host-embed the window, s1 core, argmax, sibling-embed, s2 core,
/// argmax, append, slide the `t_win` window. Returns the generated `(s1, s2)`
/// token tails. `s1_core(x) -> (ctx, s1_logits)`; `s2_core(ctx, sib) -> s2_logits`.
fn rollout<F1, F2>(dec: &kronos::decoder::KronosDecoder, s1_ctx: &[u32], s2_ctx: &[u32], horizon: usize, t_win: usize, mut s1_core: F1, mut s2_core: F2) -> (Vec<u32>, Vec<u32>)
where
    F1: FnMut(&[f32]) -> (Vec<f32>, Vec<f32>),
    F2: FnMut(&[f32], &[f32]) -> Vec<f32>,
{
    let (vs1, vs2) = (dec.config().s1_vocab(), dec.config().s2_vocab());
    let ctx_len = s1_ctx.len();
    let mut s1 = s1_ctx.to_vec();
    let mut s2 = s2_ctx.to_vec();
    for _ in 0..horizon {
        let len = s1.len();
        let w0 = len - t_win;
        let (s1w, s2w) = (&s1[w0..], &s2[w0..]);
        let x = dec.embed_tokens(s1w, s2w, &[]);
        let (ctx, s1_logits) = s1_core(&x);
        let samp_s1 = argmax(&s1_logits[(t_win - 1) * vs1..t_win * vs1]);
        let mut s1_cond = s1w.to_vec();
        *s1_cond.last_mut().unwrap() = samp_s1;
        let sib = dec.sib_embed(&s1_cond);
        let s2_logits = s2_core(&ctx, &sib);
        let samp_s2 = argmax(&s2_logits[(t_win - 1) * vs2..t_win * vs2]);
        s1.push(samp_s1);
        s2.push(samp_s2);
    }
    (s1[ctx_len..].to_vec(), s2[ctx_len..].to_vec())
}

/// A synthetic OHLCV(+amount) context of `t_win` bars, `[t_win, feat]`.
fn synthetic_bars(t_win: usize, feat: usize) -> Vec<f32> {
    let mut bars = Vec::with_capacity(t_win * feat);
    for i in 0..t_win {
        let p = 100.0 + (i as f32 * 0.1).sin() * 5.0;
        let (o, h, l, c, v) = (p, p + 0.5, p - 0.5, p + 0.2, 1000.0 + i as f32);
        bars.extend_from_slice(&[o, h, l, c, v]);
        if feat >= 6 {
            bars.push(v * (o + h + l + c) / 4.0);
        }
        bars.extend(std::iter::repeat_n(0.0, feat.saturating_sub(6)));
    }
    bars
}

impl Action for NpuForecast {
    fn spec(&self) -> ActionSpec {
        spec()
    }

    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let horizon = inv.get_i64("horizon").unwrap_or(DEFAULT_HORIZON).max(1) as usize;
        let context_len = inv.get_i64("context_len").unwrap_or(0).max(0) as usize;
        let cfg = npu::run_options::config(inv)?;
        let model = kronos::import::load_model(&self.0.tokenizer, &self.0.decoder).map_err(|e| format!("kronos npu-forecast: load: {e}"))?;
        // The graph is fixed-length; keep the window at exactly `t_win` so the
        // device and NPU rollouts slide identically.
        let t_win = if context_len > 0 { context_len } else { model.max_context() };
        let feat = model.feat();
        let (s1_ctx, s2_ctx) = model.tokenize(&synthetic_bars(t_win, feat), t_win);
        let dec = model.decoder();

        let quant = npu::qwen_topology::Quant::F32;
        let s1_bytes = npu::kronos_export::export_onnx(&self.0.decoder, t_win, quant).map_err(|e| format!("kronos npu-forecast: export decode_s1: {e}"))?;
        let s2_bytes = npu::kronos_export::export_dep_onnx(&self.0.decoder, t_win, quant).map_err(|e| format!("kronos npu-forecast: export decode_s2: {e}"))?;
        let mut s1_sess = KronosS1Session::load_bytes(&s1_bytes, &cfg).map_err(|e| format!("kronos npu-forecast: compile decode_s1: {e}"))?;
        let mut s2_sess = KronosS2Session::load_bytes(&s2_bytes, &cfg).map_err(|e| format!("kronos npu-forecast: compile decode_s2: {e}"))?;
        let device = s1_sess.device().to_string();

        let t0 = Instant::now();
        // An infer failure surfaces as a panic out of the rollout's closures;
        // it is turned into this action's error rather than unwinding the
        // program.
        let (g1, g2) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rollout(dec, &s1_ctx, &s2_ctx, horizon, t_win, |x| s1_sess.run(x).expect("npu decode_s1"), |ctx, sib| s2_sess.run(ctx, sib).expect("npu decode_s2"))
        }))
        .map_err(|_| "kronos npu-forecast: the NPU rollout failed (infer error: see stderr)".to_string())?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let per_step = ms / horizon as f64;

        // Detokenize the generated tail into normalized bars; report the close column.
        let recon = model.tokenizer().decode(&g1, &g2); // [horizon, feat] normalized
        let close: Vec<f32> = (0..horizon).map(|k| (recon[k * feat + 3] * 1000.0).round() / 1000.0).collect();
        let mut report = format!("kronos AR forecast on {device} · T={t_win} · horizon {horizon} · {ms:.1} ms ({per_step:.1} ms/step)\n");
        report.push_str(&format!("  generated {} s1 + {} s2 tokens; recon close (normalized): {close:?}\n", g1.len(), g2.len()));

        let mut out = Outcome::new().set("device", json!(device)).set("window", json!(t_win)).set("ms", json!(ms)).set("ms_per_step", json!(per_step));
        if inv.get_bool("verify").unwrap_or(false) {
            let (d1, d2) = rollout(dec, &s1_ctx, &s2_ctx, horizon, t_win, |x| {
                let (l, c) = dec.core_forward_s1(x, t_win);
                (c, l)
            }, |ctx, sib| dec.core_forward_s2(ctx, sib, t_win));
            let a1 = g1.iter().zip(&d1).filter(|(a, b)| a == b).count();
            let a2 = g2.iter().zip(&d2).filter(|(a, b)| a == b).count();
            report.push_str(&format!("  vs device core: s1 {a1}/{} match, s2 {a2}/{} match\n", g1.len(), g2.len()));
            out = out.set("s1_match", json!(a1)).set("s2_match", json!(a2)).set("tokens", json!(g1.len()));
        }
        Ok(out.blob("report", Blob::new(Media::Text, report.into_bytes())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_action_offers_the_npu_options_and_the_rollout_flags() {
        let names: Vec<String> = spec().params.into_iter().map(|p| p.name).collect();
        for want in ["horizon", "context_len", "verify", "ov_device", "allow_fallback", "cache_dir"] {
            assert!(names.iter().any(|n| n == want), "{want} in {names:?}");
        }
    }

    #[test]
    fn the_synthetic_context_has_one_row_per_bar_with_the_amount_column_when_the_model_has_one() {
        assert_eq!(synthetic_bars(4, 5).len(), 20);
        let six = synthetic_bars(3, 6);
        assert_eq!(six.len(), 18);
        assert!((six[5] - 1000.0 * (100.0 + 100.5 + 99.5 + 100.2) / 4.0).abs() < 1e-3, "amount = volume * mean(OHLC)");
    }

    #[test]
    fn argmax_picks_the_index_of_the_largest_logit() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
    }
}
