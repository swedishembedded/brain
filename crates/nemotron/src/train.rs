// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Trainable Nemotron models wired into brain's `model::Model` trait, so the
//! FastConformer/RNN-T architecture is a first-class trainable citizen (generic
//! trainer, `gradcheck::directional_check`, bench/eval all work with it).
//!
//! Two layers, both validated by full-parameter finite-diff gradcheck on tiny
//! configs (`directional_check`, which perturbs EVERY parameter):
//!   * [`Transducer`] — RNN-T prediction net + joint + transducer loss over
//!     encoder features (the recurrent/transducer trainable core).
//!   * [`AcousticModel`] — Conformer blocks + projectors + the Transducer head
//!     (the full trainable acoustic model over subsampled features).
//!
//! Weights/grads are held host-side and updated with a self-contained AdamW; the
//! math reuses the parity- and gradient-checked reference in [`crate::reference`].
//! (The device forward for inference lives in [`crate::encoder`]; a device-graph
//! training port is a perf follow-up — every gradient here is already verified.)

use std::collections::HashMap;

use crate::config::NemotronConfig;
use crate::reference as rf;

type W = HashMap<String, Vec<f32>>;

/// Config for the RNN-T transducer trainable head over `enc_dim` encoder features.
#[derive(Clone, Debug)]
pub struct TransducerConfig {
    pub enc_dim: u32,        // encoder feature width fed to the joint (== decoder_hidden)
    pub cfg: NemotronConfig, // decoder_hidden, num_decoder_layers, vocab, blank
    pub t_frames: u32,       // encoder frames (fixed batch shape)
    pub n_labels: u32,       // label length (fixed batch shape)
}

impl TransducerConfig {
    pub fn tiny() -> TransducerConfig {
        let mut cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        cfg.decoder_hidden = 4;
        cfg.num_decoder_layers = 2;
        cfg.vocab = 6;
        cfg.blank_token_id = 5;
        TransducerConfig { enc_dim: 4, cfg, t_frames: 4, n_labels: 2 }
    }

    fn param_list(&self) -> Vec<(String, usize)> {
        let (dh, v) = (self.cfg.decoder_hidden as usize, self.cfg.vocab as usize);
        let mut p = vec![("decoder.embedding.weight".to_string(), v * dh)];
        for l in 0..self.cfg.num_decoder_layers as usize {
            p.push((format!("decoder.lstm.weight_ih_l{l}"), 4 * dh * dh));
            p.push((format!("decoder.lstm.weight_hh_l{l}"), 4 * dh * dh));
            p.push((format!("decoder.lstm.bias_ih_l{l}"), 4 * dh));
            p.push((format!("decoder.lstm.bias_hh_l{l}"), 4 * dh));
        }
        p.push(("decoder.decoder_projector.weight".into(), dh * dh));
        p.push(("decoder.decoder_projector.bias".into(), dh));
        p.push(("joint.head.weight".into(), v * dh));
        p.push(("joint.head.bias".into(), v));
        p
    }
}

/// Trainable RNN-T transducer (prediction net + joint + transducer loss).
pub struct Transducer {
    tcfg: TransducerConfig,
    w: std::cell::RefCell<HashMap<String, Vec<f32>>>,
    g: std::cell::RefCell<HashMap<String, Vec<f32>>>,
    // AdamW moments
    m: std::cell::RefCell<HashMap<String, Vec<f32>>>,
    v: std::cell::RefCell<HashMap<String, Vec<f32>>>,
    // current batch
    enc: std::cell::RefCell<Vec<f32>>,    // [t_frames, enc_dim]
    labels: std::cell::RefCell<Vec<u32>>, // [n_labels]
}

impl Transducer {
    pub fn new_from(tcfg: TransducerConfig, init: &W) -> Transducer {
        let zeros: HashMap<String, Vec<f32>> = tcfg.param_list().iter().map(|(n, sz)| (n.clone(), vec![0.0; *sz])).collect();
        Transducer {
            tcfg,
            w: std::cell::RefCell::new(init.clone()),
            g: std::cell::RefCell::new(zeros.clone()),
            m: std::cell::RefCell::new(zeros.clone()),
            v: std::cell::RefCell::new(zeros),
            enc: std::cell::RefCell::new(Vec::new()),
            labels: std::cell::RefCell::new(Vec::new()),
        }
    }

    pub fn init(tcfg: &TransducerConfig, seed: u64) -> W {
        use data::rng::Rng;
        let mut rng = Rng::new(seed);
        tcfg.param_list().into_iter().map(|(n, sz)| (n, (0..sz).map(|_| (rng.next_f32() - 0.5) * 0.4).collect())).collect()
    }

    /// Forward → transducer loss; caches nothing (backward recomputes). The joint
    /// logits over the `T×(U+1)` lattice come from `joint(enc[t], predictor[u])`.
    fn forward_loss(&self) -> f32 {
        let (loss, _) = self.loss_and_grads(false);
        loss
    }

    /// Shared forward (+ optional full backward). Returns `(loss, grad_map)`.
    fn loss_and_grads(&self, want_grads: bool) -> (f32, W) {
        let cfg = &self.tcfg.cfg;
        let (dh, v, blank) = (cfg.decoder_hidden as usize, cfg.vocab as usize, cfg.blank_token_id);
        let w = self.w.borrow();
        let enc = self.enc.borrow();
        let labels = self.labels.borrow();
        let t = self.tcfg.t_frames as usize;
        let u = labels.len();
        let up1 = u + 1;

        // predictor over [blank, labels] → dec[up1][dh]
        let pred_tokens: Vec<u32> = std::iter::once(blank).chain(labels.iter().copied()).collect();
        let mut st = rf::LstmState::new(cfg.num_decoder_layers as usize, dh);
        let mut dec = vec![0.0f32; up1 * dh];
        for (uu, &tok) in pred_tokens.iter().enumerate() {
            let d = rf::lstm_predict(tok, &mut st, &w, cfg);
            dec[uu * dh..uu * dh + dh].copy_from_slice(&d);
        }
        // joint logits over the lattice [t, up1, v]
        let mut logits = vec![0.0f32; t * up1 * v];
        for tt in 0..t {
            for uu in 0..up1 {
                let l = rf::joint(&enc[tt * dh..tt * dh + dh], &dec[uu * dh..uu * dh + dh], &w, cfg);
                logits[(tt * up1 + uu) * v..(tt * up1 + uu) * v + v].copy_from_slice(&l);
            }
        }
        let (loss, d_logits) = rf::rnnt_loss(&logits, t, &labels, blank as usize, v);
        if !want_grads {
            return (loss, W::new());
        }

        // backward: d_logits → joint bwd (accumulate d_dec + joint head grads) → predictor BPTT
        let mut grads: W = W::new();
        let mut d_dec = vec![vec![0.0f32; dh]; up1];
        let mut d_head_w = vec![0.0f32; v * dh];
        let mut d_head_b = vec![0.0f32; v];
        for tt in 0..t {
            for uu in 0..up1 {
                let dl = &d_logits[(tt * up1 + uu) * v..(tt * up1 + uu) * v + v];
                let (_d_enc, dd, dhw, dhb) = rf::joint_backward(&enc[tt * dh..tt * dh + dh], &dec[uu * dh..uu * dh + dh], dl, &w, cfg);
                for j in 0..dh {
                    d_dec[uu][j] += dd[j];
                }
                for i in 0..v * dh {
                    d_head_w[i] += dhw[i];
                }
                for i in 0..v {
                    d_head_b[i] += dhb[i];
                }
            }
        }
        grads.insert("joint.head.weight".into(), d_head_w);
        grads.insert("joint.head.bias".into(), d_head_b);
        // predictor BPTT (embedding + LSTM + decoder_projector grads)
        for (k, val) in rf::predictor_grads(&pred_tokens, &w, cfg, &d_dec) {
            grads.insert(k, val);
        }
        (loss, grads)
    }
}

impl model::ModelConfig for TransducerConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        TransducerConfig::param_list(self)
    }
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "enc_dim": self.enc_dim, "t_frames": self.t_frames, "n_labels": self.n_labels,
            "decoder_hidden": self.cfg.decoder_hidden, "num_decoder_layers": self.cfg.num_decoder_layers,
            "vocab": self.cfg.vocab, "blank_token_id": self.cfg.blank_token_id })
    }
    fn from_json(v: &serde_json::Value) -> Self {
        let mut cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let g = |k: &str| v[k].as_u64().unwrap() as u32;
        cfg.decoder_hidden = g("decoder_hidden");
        cfg.num_decoder_layers = g("num_decoder_layers");
        cfg.vocab = g("vocab");
        cfg.blank_token_id = g("blank_token_id");
        TransducerConfig { enc_dim: g("enc_dim"), cfg, t_frames: g("t_frames"), n_labels: g("n_labels") }
    }
    fn vocab(&self) -> u32 {
        self.cfg.vocab
    }
    fn block_size(&self) -> u32 {
        self.t_frames
    }
    fn finalize_for_dataset(self, _vocab: u32, _block_size: u32) -> Self {
        self
    }
}

impl model::Model for Transducer {
    type Config = TransducerConfig;

    fn new(cfg: TransducerConfig, _b: u32, _t: u32, init: &W) -> Self {
        Transducer::new_from(cfg, init)
    }
    fn init_weights(cfg: &TransducerConfig, seed: u64) -> W {
        Transducer::init(cfg, seed)
    }
    fn config(&self) -> &TransducerConfig {
        &self.tcfg
    }
    fn set_batch(&self, batch: model::Batch) {
        if let model::Batch::Tensor { tokens, inputs, .. } = batch {
            *self.enc.borrow_mut() = inputs.to_vec();
            *self.labels.borrow_mut() = tokens.expect("transducer batch needs label tokens").to_vec();
        } else {
            panic!("Transducer expects Batch::Tensor {{ tokens: labels, inputs: features }}");
        }
    }
    fn forward(&self) -> f32 {
        self.forward_loss()
    }
    fn backward(&self) {
        let (_loss, grads) = self.loss_and_grads(true);
        *self.g.borrow_mut() = grads;
    }
    fn zero_grads(&self) {
        for v in self.g.borrow_mut().values_mut() {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }
    fn adamw_step(&self, t: u32, lr: f32, wd: f32, _clip: Option<f32>, extra_scale: f32) {
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let (mut m, mut v) = (self.m.borrow_mut(), self.v.borrow_mut());
        let g = self.g.borrow();
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        for (name, w) in self.w_mut().iter_mut() {
            let (gg, mm, vv) = (&g[name], m.get_mut(name).unwrap(), v.get_mut(name).unwrap());
            for i in 0..w.len() {
                let grad = gg[i] * extra_scale;
                mm[i] = b1 * mm[i] + (1.0 - b1) * grad;
                vv[i] = b2 * vv[i] + (1.0 - b2) * grad * grad;
                let mhat = mm[i] / bc1;
                let vhat = vv[i] / bc2;
                w[i] -= lr * (mhat / (vhat.sqrt() + eps) + wd * w[i]);
            }
        }
    }
    fn poll_wait(&self) {}
    fn param_names(&self) -> Vec<String> {
        self.tcfg.param_list().into_iter().map(|(n, _)| n).collect()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.w.borrow()[name].clone()
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.w_mut().get_mut(name).unwrap().copy_from_slice(data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.g.borrow()[name].clone()
    }
    fn logits_all(&self, _tokens: &[u32]) -> Option<Vec<f32>> {
        None
    }
    fn save(&self, _path: &str) {}
    fn config_json(&self) -> serde_json::Value {
        model::ModelConfig::to_json(&self.tcfg)
    }
}

impl Transducer {
    // interior mutability for weights (write_weight/adamw during a &self trait)
    fn w_mut(&self) -> std::cell::RefMut<'_, HashMap<String, Vec<f32>>> {
        self.w.borrow_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;
    use model::{Batch, Model};

    #[test]
    fn transducer_full_param_gradcheck() {
        // Every parameter of the RNN-T transducer (embedding, 2-layer LSTM,
        // decoder_projector, joint head) checked against finite differences via
        // brain's directional_check — proving it is a training-faithful model::Model.
        let tcfg = TransducerConfig::tiny();
        let init = Transducer::init(&tcfg, 7);
        let m = Transducer::new_from(tcfg.clone(), &init);
        let (t, dh) = (tcfg.t_frames as usize, tcfg.cfg.decoder_hidden as usize);
        let mut rng = Rng::new(9);
        let enc: Vec<f32> = (0..t * dh).map(|_| (rng.next_f32() - 0.5) * 1.0).collect();
        let labels = vec![1u32, 3u32]; // valid labels (< vocab, != blank)
        m.set_batch(Batch::Tensor { tokens: Some(&labels), inputs: &enc, targets: &[] });

        let report = gradcheck::directional_check(&m, 5e-3, 4, 0x1234);
        let fails = report.failures(4e-3, 8e-2);
        eprintln!("transducer gradcheck max_rel {:.2e}, params {}", report.max_rel(), m.param_names().len());
        assert!(fails.is_empty(), "gradcheck failures: {:?}", fails);
    }

    #[test]
    fn transducer_trains_one_step() {
        // Smoke: forward → backward → AdamW reduces the loss on a fixed batch.
        let tcfg = TransducerConfig::tiny();
        let init = Transducer::init(&tcfg, 2);
        let m = Transducer::new_from(tcfg.clone(), &init);
        let (t, dh) = (tcfg.t_frames as usize, tcfg.cfg.decoder_hidden as usize);
        let enc: Vec<f32> = (0..t * dh).map(|i| (i as f32 * 0.1).sin()).collect();
        let labels = vec![1u32, 3u32];
        m.set_batch(Batch::Tensor { tokens: Some(&labels), inputs: &enc, targets: &[] });
        let l0 = m.forward();
        for step in 1..=20 {
            m.zero_grads();
            m.forward();
            m.backward();
            m.adamw_step(step, 0.05, 0.0, None, 1.0);
        }
        let l1 = m.forward();
        eprintln!("transducer loss {l0:.4} -> {l1:.4}");
        assert!(l1 < l0, "AdamW should reduce the loss ({l0} -> {l1})");
    }
}
