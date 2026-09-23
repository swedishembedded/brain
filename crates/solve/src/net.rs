// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The policy network: a residual MLP, trained and evaluated on the device.
//!
//! Both dispatch graphs are built ONCE and resubmitted. The shapes never move
//! between steps - a fixed row count, a fixed feature width, a fixed move
//! count - so rebuilding them per step would buy nothing and cost the whole
//! encode each time.
//!
//! Every row of a batch is an independent example that shares one forward
//! pass. That is the difference that makes this tractable: a per-example loop
//! leaves a GPU almost entirely idle on a network this small, and no amount
//! of training budget rescues a pipeline that spends it one example at a
//! time.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};
use optim::Optim;
use paramstore::ParamStore;

use model::block::pick_gemm;

use crate::config::Config;
use crate::kern::*;

pub struct Net {
    pub gpu: Gpu,
    pub ps: ParamStore,
    opt: Optim,
    pub cfg: Config,
    /// Rows per batch. Fixed, because the graphs are.
    pub rows: u32,

    x: DeviceBuffer,
    targets: DeviceBuffer,
    stem_pre: DeviceBuffer,
    res: Vec<DeviceBuffer>,
    up_pre: Vec<DeviceBuffer>,
    up_act: Vec<DeviceBuffer>,
    down: Vec<DeviceBuffer>,
    logits: DeviceBuffer,
    probs: DeviceBuffer,
    ce: DeviceBuffer,

    d_logits: DeviceBuffer,
    d_res: Vec<DeviceBuffer>,
    d_skip: DeviceBuffer,
    d_up_act: DeviceBuffer,
    d_up_pre: DeviceBuffer,
    d_stem: DeviceBuffer,

    fwd: Vec<Step>,
    bwd: Vec<Step>,
    infer: Vec<Step>,
}

impl Net {
    pub fn new(cfg: Config, rows: u32, seed: u64) -> Net {
        Net::from_weights(cfg.clone(), rows, &init_weights(&cfg, seed))
    }

    pub fn from_weights(cfg: Config, rows: u32, init: &HashMap<String, Vec<f32>>) -> Net {
        let gpu = Gpu::new(PIPELINES);
        let params: Vec<(String, usize)> = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>()))
            .collect();
        let ps = ParamStore::new(&gpu, params, init);
        let opt = Optim::new(K_ADAMW, K_GRADNORM_SQ, K_GRAD_SCALE, K_CLIP_COEF, K_GRAD_SCALE_BUF);

        let (n, d, ff, m, i) = (
            rows as u64,
            cfg.d_model as u64,
            cfg.d_ff as u64,
            cfg.moves as u64,
            cfg.in_dim as u64,
        );
        let st = |x: u64| gpu.storage(x);
        let blocks = cfg.blocks as usize;

        let mut net = Net {
            x: st(n * i),
            targets: st(n),
            stem_pre: st(n * d),
            res: (0..=blocks).map(|_| st(n * d)).collect(),
            up_pre: (0..blocks).map(|_| st(n * ff)).collect(),
            up_act: (0..blocks).map(|_| st(n * ff)).collect(),
            down: (0..blocks).map(|_| st(n * d)).collect(),
            logits: st(n * m),
            probs: st(n * m),
            ce: st(n),
            d_logits: st(n * m),
            d_res: (0..=blocks).map(|_| st(n * d)).collect(),
            d_skip: st(n * d),
            d_up_act: st(n * ff),
            d_up_pre: st(n * ff),
            d_stem: st(n * d),
            fwd: Vec::new(),
            bwd: Vec::new(),
            infer: Vec::new(),
            gpu,
            ps,
            opt,
            cfg,
            rows,
        };
        net.fwd = net.build_forward(true);
        net.infer = net.build_forward(false);
        net.bwd = net.build_backward();
        net
    }

    fn w(&self, n: &str) -> &DeviceBuffer {
        self.ps.w(n)
    }
    fn g(&self, n: &str) -> &DeviceBuffer {
        self.ps.g(n)
    }

    /// The forward graph. `with_loss` appends the cross-entropy; inference
    /// appends a row softmax instead, so a rollout never needs a label.
    fn build_forward(&self, with_loss: bool) -> Vec<Step> {
        let c = &self.cfg;
        let (n, d, ff, m, i) = (self.rows, c.d_model, c.d_ff, c.moves, c.in_dim);
        let g = &self.gpu;
        let mut s = Vec::new();

        // A matmul against a one-hot row is a table lookup; the width of the
        // feature vector is bought at almost no run-time cost.
        let gemm = |m: u32, out: u32| pick_gemm(m as usize, out as usize, K_MATMUL, K_MATMUL_REG, false);

        let (k, th) = gemm(n, d);
        s.push(g.step(k, &[&self.x, self.w("stem.weight"), &self.stem_pre], &[n, i, d], th));
        s.push(g.step(K_BIAS_ADD, &[&self.stem_pre, self.w("stem.bias")], &[n, d], n * d));
        s.push(g.step(K_GELU, &[&self.stem_pre, &self.res[0]], &[n * d], n * d));

        for b in 0..c.blocks as usize {
            let p = |x: &str| format!("blocks.{b}.{x}");
            let (k, th) = gemm(n, ff);
            s.push(g.step(k, &[&self.res[b], self.w(&p("up.weight")), &self.up_pre[b]], &[n, d, ff], th));
            s.push(g.step(K_BIAS_ADD, &[&self.up_pre[b], self.w(&p("up.bias"))], &[n, ff], n * ff));
            s.push(g.step(K_GELU, &[&self.up_pre[b], &self.up_act[b]], &[n * ff], n * ff));
            let (k, th) = gemm(n, d);
            s.push(g.step(k, &[&self.up_act[b], self.w(&p("down.weight")), &self.down[b]], &[n, ff, d], th));
            s.push(g.step(K_BIAS_ADD, &[&self.down[b], self.w(&p("down.bias"))], &[n, d], n * d));
            s.push(g.step(K_ADD2, &[&self.res[b], &self.down[b], &self.res[b + 1]], &[n * d], n * d));
        }

        let last = &self.res[c.blocks as usize];
        let (k, th) = gemm(n, m);
        s.push(g.step(k, &[last, self.w("head.weight"), &self.logits], &[n, d, m], th));
        s.push(g.step(K_BIAS_ADD, &[&self.logits, self.w("head.bias")], &[n, m], n * m));
        if with_loss {
            s.push(g.step(K_CE_VALUE, &[&self.logits, &self.targets, &self.ce], &[n, m], n));
        } else {
            // One WORKGROUP per row, not one thread: `softmax_rows` has 64
            // threads cooperate on a row, so the dispatch has to ask for
            // `rows * 64` threads to get `rows` workgroups. Asking for `rows`
            // covers only the first `rows / 64` of them and leaves the rest
            // holding whatever was in the buffer.
            s.push(g.step(K_SOFTMAX, &[&self.logits, &self.probs], &[n, m], n * 64));
        }
        s
    }

    fn build_backward(&self) -> Vec<Step> {
        let c = &self.cfg;
        let (n, d, ff, m, i) = (self.rows, c.d_model, c.d_ff, c.moves, c.in_dim);
        let g = &self.gpu;
        let blocks = c.blocks as usize;
        let last = blocks;
        let dw = |a: u32, b: u32| pick_gemm(a as usize, b as usize, K_MATMUL_DW, K_MATMUL_DW_REG, false);
        let dx = |a: u32, b: u32| pick_gemm(a as usize, b as usize, K_MATMUL_DX, K_MATMUL_DX_REG, false);

        // `ce_grad` divides by the row count, so what accumulates here is
        // already the batch MEAN gradient.
        let mut s = vec![
            g.step(K_CE_GRAD, &[&self.logits, &self.targets, &self.d_logits], &[n, m], n * m),
            { let (k, th) = dw(m, d); g.step(k, &[&self.d_logits, &self.res[last], self.g("head.weight")], &[n, d, m], th) },
            g.step(K_BIAS_GRAD, &[&self.d_logits, self.g("head.bias")], &[n, m], m),
            { let (k, th) = dx(n, d); g.step(k, &[&self.d_logits, self.w("head.weight"), &self.d_res[last]], &[n, d, m, 0], th) },
        ];

        for b in (0..blocks).rev() {
            let p = |x: &str| format!("blocks.{b}.{x}");
            // The residual adds, so the gradient arriving at res[b+1] reaches
            // both the block's output and the skip unchanged.
            s.push(g.step(K_BIAS_GRAD, &[&self.d_res[b + 1], self.g(&p("down.bias"))], &[n, d], d));
            let (k, th) = dw(d, ff);
            s.push(g.step(k, &[&self.d_res[b + 1], &self.up_act[b], self.g(&p("down.weight"))], &[n, ff, d], th));
            let (k, th) = dx(n, ff);
            s.push(g.step(k, &[&self.d_res[b + 1], self.w(&p("down.weight")), &self.d_up_act], &[n, ff, d, 0], th));
            s.push(g.step(K_GELU_BWD, &[&self.up_pre[b], &self.d_up_act, &self.d_up_pre], &[n * ff], n * ff));
            s.push(g.step(K_BIAS_GRAD, &[&self.d_up_pre, self.g(&p("up.bias"))], &[n, ff], ff));
            let (k, th) = dw(ff, d);
            s.push(g.step(k, &[&self.d_up_pre, &self.res[b], self.g(&p("up.weight"))], &[n, d, ff], th));
            let (k, th) = dx(n, d);
            s.push(g.step(k, &[&self.d_up_pre, self.w(&p("up.weight")), &self.d_skip], &[n, d, ff, 0], th));
            // ...and the skip path is the other half of that sum. Written to
            // a separate buffer first because a kernel that read and wrote
            // one buffer in the same dispatch would race.
            s.push(g.step(K_ADD2, &[&self.d_skip, &self.d_res[b + 1], &self.d_res[b]], &[n * d], n * d));
        }

        s.push(g.step(K_GELU_BWD, &[&self.stem_pre, &self.d_res[0], &self.d_stem], &[n * d], n * d));
        s.push(g.step(K_BIAS_GRAD, &[&self.d_stem, self.g("stem.bias")], &[n, d], d));
        let (k, th) = dw(d, i);
        s.push(g.step(k, &[&self.d_stem, &self.x, self.g("stem.weight")], &[n, i, d], th));
        s
    }

    /// Load one batch of features and labels onto the device.
    pub fn load_batch(&self, features: &[f32], labels: &[u32]) {
        let need = (self.rows * self.cfg.in_dim) as usize;
        assert_eq!(features.len(), need, "features must be rows x in_dim");
        assert_eq!(labels.len(), self.rows as usize, "one label per row");
        self.gpu.write_f32(&self.x, features);
        self.gpu.write(&self.targets, labels);
    }

    /// Forward + backward for the loaded batch.
    ///
    /// Reads NOTHING back. A device read is a synchronisation point, and at
    /// these shapes the pipeline is short enough that syncing once per step
    /// to fetch a number nobody prints costs more than the arithmetic does.
    /// Call [`Net::loss`] on the steps you actually report.
    ///
    /// Gradients ACCUMULATE; the caller zeroes once per optimizer step, which
    /// is what lets several batches make one update.
    pub fn accumulate(&self) {
        self.gpu.submit(&[], &self.fwd);
        self.gpu.submit(&[], &self.bwd);
    }

    /// Mean cross-entropy of the batch last run through [`Net::accumulate`].
    pub fn loss(&self) -> f32 {
        let ce = self.gpu.read(&self.ce, self.rows as usize);
        ce.iter().sum::<f32>() / self.rows as f32
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    pub fn adamw(&self, t: u32, lr: f32, wd: f32) {
        self.opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, Some(1.0), 1.0);
    }

    /// Move probabilities for a batch of states. No label, no loss, no
    /// search - one forward pass is the whole inference path.
    pub fn policy(&self, features: &[f32]) -> Vec<f32> {
        let need = (self.rows * self.cfg.in_dim) as usize;
        assert_eq!(features.len(), need, "features must be rows x in_dim");
        self.gpu.write_f32(&self.x, features);
        self.gpu.submit(&[], &self.infer);
        self.gpu.read(&self.probs, (self.rows * self.cfg.moves) as usize)
    }

    /// Overwrite one parameter in place. Exists for the gradient check,
    /// which has to perturb a weight and re-measure without rebuilding the
    /// device and recompiling every pipeline to do it.
    pub fn set_weight(&self, name: &str, v: &[f32]) {
        self.gpu.write_f32(self.ps.w(name), v);
    }

    pub fn grad(&self, name: &str) -> Vec<f32> {
        self.ps.read_grad(&self.gpu, name)
    }

    /// Write the policy as one safetensors file, config and all.
    ///
    /// The fit that produced it travels with it: a network whose training
    /// settings were not recorded cannot later be told apart from a sibling
    /// that differed in every way that mattered, and its published numbers
    /// then cannot be defended or discarded.
    pub fn save(&self, path: &str, fit: &serde_json::Value) -> Result<(), String> {
        let shapes: HashMap<String, Vec<usize>> = self.cfg.tensor_manifest().into_iter().collect();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = self
            .cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, _)| {
                let shape = shapes[&n].iter().map(|&d| d as u64).collect();
                let data = self.ps.read_weight(&self.gpu, &n);
                (n, shape, data)
            })
            .collect();
        let mut config = self.cfg.to_json();
        if let Some(o) = config.as_object_mut() {
            o.insert("trained_for".into(), fit.clone());
        }
        checkpoint::st::save_safetensors(path, &tensors, &config, None)
            .map_err(|e| format!("write {path}: {e}"))
    }

    /// Read one back. `rows` is the batch this instance will run, and is NOT
    /// stored in the file: it is a property of how the network is being used,
    /// not of what it learned, so a policy trained at one batch rolls out at
    /// any other.
    pub fn load(path: &str, rows: u32) -> Result<Net, String> {
        let m = checkpoint::st::load_safetensors(path).map_err(|e| format!("read {path}: {e}"))?;
        let cfg_text = m.config().to_string();
        let cfg = Config::from_json(&cfg_text)?;
        for (name, shape) in cfg.tensor_manifest() {
            let want: usize = shape.iter().product();
            match m.tensors.get(&name) {
                None => return Err(format!("{path}: missing tensor {name:?}")),
                Some(v) if v.len() != want => {
                    return Err(format!("{path}: {name:?} has {} elements, config says {want}", v.len()))
                }
                Some(_) => {}
            }
        }
        Ok(Net::from_weights(cfg, rows, &m.tensors))
    }

    pub fn weights(&self) -> HashMap<String, Vec<f32>> {
        self.cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, _)| {
                let v = self.ps.read_weight(&self.gpu, &n);
                (n, v)
            })
            .collect()
    }
}

/// Weights at initialisation.
///
/// Fan-in scaled normal, and the LAST projection of each residual block
/// scaled down by the block count. A residual stream whose blocks all start
/// at full variance grows with depth before a single step is taken, which
/// shows up as a loss that rises for the first few hundred steps and is
/// usually misread as a learning rate that is too high.
pub fn init_weights(cfg: &Config, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = crate::data::Rng::new(seed);
    let mut normal = move || {
        // Box-Muller, from two uniforms.
        let u1 = ((rng.next() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let damp = 1.0 / (cfg.blocks.max(1) as f32).sqrt();
    cfg.tensor_manifest()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            if name.ends_with(".bias") {
                return (name, vec![0.0f32; n]);
            }
            let fan_in = shape[0] as f32;
            let mut s = (2.0 / fan_in).sqrt();
            if name.ends_with("down.weight") {
                s *= damp;
            }
            let v = (0..n).map(|_| normal() as f32 * s).collect();
            (name, v)
        })
        .collect()
}
