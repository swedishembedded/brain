// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The MLP projector that carries vision features into a language model's
//! embedding space: LLaVA's and FastVLM's `mlp2x_gelu`, DeepSeek-VL's and
//! Janus-Pro's `MlpProjector`.
//!
//! One input linear, then `depth - 1` times an exact (erf) GELU followed by
//! a linear. The hybrid split form (DeepSeek-VL's
//! `low_high_hybrid_split_mlp_gelu`) takes two feature streams, projects each
//! to half the width and concatenates the halves (high first) before the
//! GELU stack. The last linear may widen past `n_embed` ([`ProjectorConfig::
//! with_out_dim`]): Janus-Pro's generation head is this stack ending in the
//! image vocabulary.
//!
//! [`MlpProjector`] runs it on the device, forward and backward (the aligner
//! is what a VLM fine-tune trains); [`forward_host`] is the same function on
//! the host, for a one-off projection too small to dispatch and as the
//! reference the device path is tested against.
//!
//! Swedish Embedded AB implements vision-language model integration like this
//! for its clients. If your team needs expertise in multimodal model
//! architectures, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Every kernel [`MlpProjector`] dispatches; register these on its device.
pub const PROJECTOR_PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("gelu_erf", kernels::GELU_ERF),
    ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("bias_grad", kernels::BIAS_GRAD),
    ("concat2", kernels::CONCAT2),
    ("concat_split", kernels::CONCAT_SPLIT),
];

/// How the projector's first linear reads its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectorKind {
    /// One feature stream (`mlp_gelu`, `mlp2x_gelu`).
    Mlp,
    /// Two feature streams, each projected to `n_embed / 2` and concatenated
    /// high-then-low (`low_high_hybrid_split_mlp_gelu`).
    HybridSplit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectorConfig {
    pub kind: ProjectorKind,
    /// Linears in the stack, the input one included (`mlp2x_gelu` is 2).
    pub depth: u32,
    pub input_dim: u32,
    pub n_embed: u32,
    /// The last linear's output width: `n_embed` for an aligner.
    pub out_dim: u32,
}

impl ProjectorConfig {
    /// A config from its upstream `projector_type` name and sizes.
    pub fn from_type(projector_type: &str, depth: u32, input_dim: u32, n_embed: u32) -> Result<ProjectorConfig, String> {
        let kind = match projector_type {
            "mlp_gelu" | "mlp2x_gelu" => ProjectorKind::Mlp,
            "low_high_hybrid_split_mlp_gelu" => ProjectorKind::HybridSplit,
            other => return Err(format!("projector_type {other:?} is not implemented")),
        };
        if depth == 0 {
            return Err("a projector has at least one linear".to_string());
        }
        if kind == ProjectorKind::HybridSplit && n_embed % 2 != 0 {
            return Err(format!("a hybrid split projector needs an even n_embed, got {n_embed}"));
        }
        Ok(ProjectorConfig { kind, depth, input_dim, n_embed, out_dim: n_embed })
    }

    /// The same stack with its last linear producing `out_dim` columns. A
    /// one-linear hybrid split's output is its two halves, so its width is
    /// fixed.
    pub fn with_out_dim(self, out_dim: u32) -> Result<ProjectorConfig, String> {
        if self.kind == ProjectorKind::HybridSplit && self.depth == 1 && out_dim != self.n_embed {
            return Err(format!("a one-linear hybrid split projector outputs its two {}-wide halves, not {out_dim} columns", self.n_embed / 2));
        }
        Ok(ProjectorConfig { out_dim, ..self })
    }

    /// Linear `k`'s output width: `out_dim` for the last, `n_embed` before.
    fn width(&self, k: u32) -> u32 {
        if k + 1 == self.depth {
            self.out_dim
        } else {
            self.n_embed
        }
    }

    /// The input linears: `(name, out, in)`.
    fn input_linears(&self) -> Vec<(&'static str, u32)> {
        match self.kind {
            ProjectorKind::Mlp => vec![("in", self.width(0))],
            ProjectorKind::HybridSplit => vec![("in_high", self.n_embed / 2), ("in_low", self.n_embed / 2)],
        }
    }

    /// Every parameter as `(name, element count)`: `in.{weight,bias}` (or
    /// `in_high.*` and `in_low.*`), then `layers.{k}.{weight,bias}` for the
    /// linears after each GELU, `k` from 1. Weights are `[out, in]`.
    pub fn param_list(&self) -> Vec<(String, usize)> {
        let (i, e) = (self.input_dim as usize, self.n_embed as usize);
        let mut out = Vec::new();
        for (name, o) in self.input_linears() {
            out.push((format!("{name}.weight"), o as usize * i));
            out.push((format!("{name}.bias"), o as usize));
        }
        for k in 1..self.depth {
            let o = self.width(k) as usize;
            out.push((format!("layers.{k}.weight"), o * e));
            out.push((format!("layers.{k}.bias"), o));
        }
        out
    }

    /// How many feature streams the projector reads.
    pub fn inputs(&self) -> usize {
        self.input_linears().len()
    }
}

/// `x [rows, in] · W [out, in]ᵀ + b` on the host.
fn linear_host(x: &[f32], w: &[f32], b: &[f32], rows: usize, inn: usize, out: usize) -> Vec<f32> {
    let mut y = crate::hostmath::linear_rows(x, w, rows, inn, out);
    for r in 0..rows {
        for (v, bias) in y[r * out..(r + 1) * out].iter_mut().zip(b) {
            *v += bias;
        }
    }
    y
}

/// The projector on the host: `inputs` is one `[rows, input_dim]` stream,
/// or two (high, low) for the hybrid split; the result is `[rows, out_dim]`.
pub fn forward_host(cfg: &ProjectorConfig, weights: &HashMap<String, Vec<f32>>, inputs: &[&[f32]], rows: usize) -> Vec<f32> {
    assert_eq!(inputs.len(), cfg.inputs(), "projector: {} input stream(s) expected", cfg.inputs());
    let (i, e) = (cfg.input_dim as usize, cfg.n_embed as usize);
    let parts: Vec<Vec<f32>> = cfg
        .input_linears()
        .iter()
        .zip(inputs)
        .map(|((name, o), x)| linear_host(x, &weights[&format!("{name}.weight")], &weights[&format!("{name}.bias")], rows, i, *o as usize))
        .collect();
    let mut h: Vec<f32> = if parts.len() == 1 {
        parts.into_iter().next().expect("one part")
    } else {
        let half = e / 2;
        (0..rows).flat_map(|r| parts[0][r * half..(r + 1) * half].iter().chain(&parts[1][r * half..(r + 1) * half]).copied().collect::<Vec<_>>()).collect()
    };
    for k in 1..cfg.depth {
        let g: Vec<f32> = h.iter().map(|&v| crate::hostmath::gelu_exact(v)).collect();
        h = linear_host(&g, &weights[&format!("layers.{k}.weight")], &weights[&format!("layers.{k}.bias")], rows, e, cfg.width(k) as usize);
    }
    h
}

/// A projector's weights held on the host, run by [`forward_host`]: the
/// form a model that projects one image's features at a time keeps.
#[derive(Clone, Debug)]
pub struct HostProjector {
    pub cfg: ProjectorConfig,
    pub weights: HashMap<String, Vec<f32>>,
}

impl HostProjector {
    /// LLaVA's and FastVLM's `mlp2x_gelu` from its two linears
    /// (`[n_embed, input_dim]` then `[n_embed, n_embed]`, torch layout).
    pub fn mlp2x(fc1_w: Vec<f32>, fc1_b: Vec<f32>, fc2_w: Vec<f32>, fc2_b: Vec<f32>, input_dim: usize, n_embed: usize) -> HostProjector {
        let cfg = ProjectorConfig { kind: ProjectorKind::Mlp, depth: 2, input_dim: input_dim as u32, n_embed: n_embed as u32, out_dim: n_embed as u32 };
        let weights = HashMap::from([
            ("in.weight".to_string(), fc1_w),
            ("in.bias".to_string(), fc1_b),
            ("layers.1.weight".to_string(), fc2_w),
            ("layers.1.bias".to_string(), fc2_b),
        ]);
        for (name, n) in cfg.param_list() {
            assert_eq!(weights[&name].len(), n, "HostProjector::mlp2x: {name}");
        }
        HostProjector { cfg, weights }
    }

    /// Project `[rows, input_dim]` (a single stream) to `[rows, n_embed]`.
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        forward_host(&self.cfg, &self.weights, &[x], rows)
    }
}

/// The kernels [`MlpProjector`] dispatches, resolved by name on its device.
#[derive(Clone, Copy, Debug)]
struct Ids {
    matmul: usize,
    bias_add: usize,
    gelu: usize,
    concat2: Option<usize>,
    /// The backward's kernels: `None` on a device registered for inference
    /// only, where [`MlpProjector::backward`] refuses to run.
    bwd: Option<BwdIds>,
}

#[derive(Clone, Copy, Debug)]
struct BwdIds {
    gelu_bwd: usize,
    matmul_dx: usize,
    matmul_dw: usize,
    bias_grad: usize,
    concat_split: usize,
}

impl Ids {
    /// The forward's kernels are required; the backward's are taken when the
    /// device registers them.
    fn resolve(g: &Gpu, kind: ProjectorKind) -> Result<Ids, String> {
        let k = |name: &str| g.kernel_index(name).ok_or_else(|| format!("projector: kernel {name:?} is not registered (see PROJECTOR_PIPELINES)"));
        let bwd = (|| -> Result<BwdIds, String> {
            Ok(BwdIds { gelu_bwd: k("gelu_erf_bwd")?, matmul_dx: k("matmul_dx")?, matmul_dw: k("matmul_dw")?, bias_grad: k("bias_grad")?, concat_split: k("concat_split")? })
        })()
        .ok();
        Ok(Ids {
            matmul: k("matmul")?,
            bias_add: k("bias_add")?,
            gelu: k("gelu_erf")?,
            concat2: if kind == ProjectorKind::HybridSplit { Some(k("concat2")?) } else { None },
            bwd,
        })
    }
}

/// The projector on the device for `rows` feature rows: parameters with
/// their gradients, and the activations its backward reads.
pub struct MlpProjector {
    pub cfg: ProjectorConfig,
    rows: u32,
    ids: Ids,
    params: HashMap<String, DeviceBuffer>,
    grads: HashMap<String, DeviceBuffer>,
    /// The hybrid split's two half-width projections.
    halves: Vec<DeviceBuffer>,
    /// `pre[k]` is the input to GELU `k` (the output of linear `k - 1`);
    /// `post[k]` is its GELU. `pre[0]`/`post[0]` are unused.
    pre: Vec<DeviceBuffer>,
    post: Vec<DeviceBuffer>,
    out: DeviceBuffer,
    /// Scratch for the gradient flowing back through the stack.
    d_a: DeviceBuffer,
    d_b: DeviceBuffer,
    d_half: DeviceBuffer,
}

impl MlpProjector {
    /// Upload `weights` (every [`ProjectorConfig::param_list`] name) onto
    /// `gpu`, which must register [`PROJECTOR_PIPELINES`].
    pub fn new(gpu: &Gpu, cfg: ProjectorConfig, rows: u32, weights: &HashMap<String, Vec<f32>>) -> Result<MlpProjector, String> {
        Self::build(gpu, cfg, rows, weights, true)
    }

    /// [`Self::new`] for inference: no gradient or backward scratch is
    /// allocated, and [`Self::backward`] refuses to run.
    pub fn new_frozen(gpu: &Gpu, cfg: ProjectorConfig, rows: u32, weights: &HashMap<String, Vec<f32>>) -> Result<MlpProjector, String> {
        Self::build(gpu, cfg, rows, weights, false)
    }

    fn build(gpu: &Gpu, cfg: ProjectorConfig, rows: u32, weights: &HashMap<String, Vec<f32>>, trainable: bool) -> Result<MlpProjector, String> {
        let ids = Ids::resolve(gpu, cfg.kind)?;
        let mut params = HashMap::new();
        let mut grads = HashMap::new();
        for (name, n) in cfg.param_list() {
            let w = weights.get(&name).ok_or_else(|| format!("projector: missing {name}"))?;
            if w.len() != n {
                return Err(format!("projector: {name} holds {} values, expected {n}", w.len()));
            }
            params.insert(name.clone(), gpu.storage_init(&name, w));
            if trainable {
                grads.insert(name, gpu.storage(n as u64));
            }
        }
        let e = (rows * cfg.n_embed) as u64;
        // A frozen build's backward scratch is a placeholder word.
        let scratch = |n: u64| gpu.storage(if trainable { n } else { 1 });
        let halves = if cfg.kind == ProjectorKind::HybridSplit { vec![gpu.storage(e / 2), gpu.storage(e / 2)] } else { Vec::new() };
        let layer = |_| gpu.storage(e);
        Ok(MlpProjector {
            cfg,
            rows,
            ids,
            params,
            grads,
            halves,
            pre: (0..cfg.depth).map(layer).collect(),
            post: (0..cfg.depth).map(layer).collect(),
            out: gpu.storage((rows * cfg.out_dim) as u64),
            d_a: scratch(e),
            d_b: scratch(e),
            d_half: scratch(e / 2),
        })
    }

    /// The `[rows, out_dim]` output buffer [`Self::forward`] writes.
    pub fn out(&self) -> &DeviceBuffer {
        &self.out
    }

    /// A parameter's gradient buffer (accumulated by [`Self::backward`]).
    pub fn grad(&self, name: &str) -> &DeviceBuffer {
        &self.grads[name]
    }

    /// A parameter's buffer.
    pub fn param(&self, name: &str) -> &DeviceBuffer {
        &self.params[name]
    }

    /// The buffer linear `k`'s output lands in: the next GELU's input, or
    /// the projector's output for the last one.
    fn linear_out(&self, k: u32) -> &DeviceBuffer {
        if k + 1 == self.cfg.depth {
            &self.out
        } else {
            &self.pre[k as usize + 1]
        }
    }

    fn linear(&self, g: &Gpu, x: &DeviceBuffer, name: &str, y: &DeviceBuffer, inn: u32, out: u32) -> [Step; 2] {
        let (m, id) = (self.rows, &self.ids);
        [
            g.step(id.matmul, &[x, &self.params[&format!("{name}.weight")], y], &[m, inn, out], m * out),
            g.step(id.bias_add, &[y, &self.params[&format!("{name}.bias")]], &[m, out], m * out),
        ]
    }

    /// The forward over `inputs` (`[rows, input_dim]` each: one stream, or
    /// high and low for the hybrid split) into [`Self::out`].
    pub fn forward(&self, g: &Gpu, inputs: &[&DeviceBuffer]) -> Vec<Step> {
        assert_eq!(inputs.len(), self.cfg.inputs(), "projector: {} input stream(s) expected", self.cfg.inputs());
        let (m, i, e) = (self.rows, self.cfg.input_dim, self.cfg.n_embed);
        let first = self.linear_out(0);
        let mut s = Vec::new();
        match self.cfg.kind {
            ProjectorKind::Mlp => s.extend(self.linear(g, inputs[0], "in", first, i, self.cfg.width(0))),
            ProjectorKind::HybridSplit => {
                s.extend(self.linear(g, inputs[0], "in_high", &self.halves[0], i, e / 2));
                s.extend(self.linear(g, inputs[1], "in_low", &self.halves[1], i, e / 2));
                let concat2 = self.ids.concat2.expect("resolved for the hybrid split");
                s.push(g.step(concat2, &[&self.halves[0], &self.halves[1], first], &[m, e / 2, e / 2, 1, 1], m * e));
            }
        }
        for k in 1..self.cfg.depth {
            let (pre, post) = (&self.pre[k as usize], &self.post[k as usize]);
            s.push(g.step(self.ids.gelu, &[pre, post], &[m * e], m * e));
            s.extend(self.linear(g, post, &format!("layers.{k}"), self.linear_out(k), e, self.cfg.width(k)));
        }
        s
    }

    /// Zero every parameter gradient.
    pub fn zero_grads(&self, g: &Gpu) {
        for (name, n) in self.cfg.param_list() {
            g.write_f32(&self.grads[&name], &vec![0.0f32; n]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_bwd(&self, g: &Gpu, id: &BwdIds, s: &mut Vec<Step>, dy: &DeviceBuffer, x: &DeviceBuffer, name: &str, dx: Option<&DeviceBuffer>, inn: u32, out: u32) {
        let m = self.rows;
        s.push(g.step(id.matmul_dw, &[dy, x, &self.grads[&format!("{name}.weight")]], &[m, inn, out], out * inn));
        s.push(g.step(id.bias_grad, &[dy, &self.grads[&format!("{name}.bias")]], &[m, out], out));
        if let Some(dx) = dx {
            s.push(g.step(id.matmul_dx, &[dy, &self.params[&format!("{name}.weight")], dx], &[m, inn, out, 0], m * inn));
        }
    }

    /// The backward of the last [`Self::forward`] for the output gradient
    /// `d_out` (`[rows, out_dim]`): accumulates every parameter gradient and,
    /// when `d_inputs` is given (one buffer per input stream), writes the
    /// input gradients there. `inputs` are the forward's own inputs.
    ///
    /// # Panics
    /// On a device that does not register the backward's kernels.
    pub fn backward(&self, g: &Gpu, inputs: &[&DeviceBuffer], d_out: &DeviceBuffer, d_inputs: Option<&[&DeviceBuffer]>) -> Vec<Step> {
        assert!(!self.grads.is_empty(), "projector backward: this projector was built frozen (MlpProjector::new_frozen)");
        let id = self.ids.bwd.as_ref().expect("projector backward: the device does not register the backward kernels (see PROJECTOR_PIPELINES)");
        let (m, i, e) = (self.rows, self.cfg.input_dim, self.cfg.n_embed);
        let mut s = Vec::new();
        // `dy` is the gradient at linear k's output, ping-ponged between two
        // scratch buffers on the way down.
        let mut dy: &DeviceBuffer = d_out;
        for k in (1..self.cfg.depth).rev() {
            let (pre, post) = (&self.pre[k as usize], &self.post[k as usize]);
            let d_post = if std::ptr::eq(dy, &self.d_a) { &self.d_b } else { &self.d_a };
            self.linear_bwd(g, id, &mut s, dy, post, &format!("layers.{k}"), Some(d_post), e, self.cfg.width(k));
            let d_pre = if std::ptr::eq(d_post, &self.d_a) { &self.d_b } else { &self.d_a };
            s.push(g.step(id.gelu_bwd, &[pre, d_post, d_pre], &[m * e], m * e));
            dy = d_pre;
        }
        match self.cfg.kind {
            ProjectorKind::Mlp => self.linear_bwd(g, id, &mut s, dy, inputs[0], "in", d_inputs.map(|d| d[0]), i, self.cfg.width(0)),
            ProjectorKind::HybridSplit => {
                for (half, name) in [(0u32, "in_high"), (1, "in_low")] {
                    s.push(g.step(id.concat_split, &[dy, &self.d_half], &[m, e, e / 2, half * (e / 2), 1, 1], m * (e / 2)));
                    self.linear_bwd(g, id, &mut s, &self.d_half, inputs[half as usize], name, d_inputs.map(|d| d[half as usize]), i, e / 2);
                }
            }
        }
        s
    }
}

/// A projector (an aligner, a generation head) on the device with its
/// host-side AdamW state: what a trainable `model::projector::MlpProjector`
/// needs around it.
pub struct TrainableProjector {
    gpu: Gpu,
    projector: MlpProjector,
    inputs: Vec<DeviceBuffer>,
    d_out: DeviceBuffer,
    d_inputs: Vec<DeviceBuffer>,
    /// `(master weights, m, v)` by parameter name.
    state: HashMap<String, (Vec<f32>, Vec<f32>, Vec<f32>)>,
    rows: usize,
}

impl TrainableProjector {
    /// `weights` (every [`ProjectorConfig::param_list`] name) for `rows`
    /// rows of feature streams.
    pub fn new(cfg: ProjectorConfig, weights: HashMap<String, Vec<f32>>, rows: usize) -> Result<TrainableProjector, String> {
        let gpu = Gpu::new(PROJECTOR_PIPELINES);
        let projector = MlpProjector::new(&gpu, cfg, rows as u32, &weights)?;
        let stream = |_| gpu.storage(rows as u64 * cfg.input_dim as u64);
        let inputs = (0..cfg.inputs()).map(stream).collect();
        let d_inputs = (0..cfg.inputs()).map(stream).collect();
        let d_out = gpu.storage(rows as u64 * cfg.out_dim as u64);
        let state = weights.into_iter().map(|(n, w)| (n, (w.clone(), vec![0.0; w.len()], vec![0.0; w.len()]))).collect();
        Ok(TrainableProjector { gpu, projector, inputs, d_out, d_inputs, state, rows })
    }

    pub fn cfg(&self) -> ProjectorConfig {
        self.projector.cfg
    }

    /// The projector's `[rows, out_dim]` output for `streams`
    /// (`[rows, input_dim]` each).
    pub fn forward(&self, streams: &[Vec<f32>]) -> Vec<f32> {
        assert_eq!(streams.len(), self.inputs.len(), "the projector reads {} feature stream(s)", self.inputs.len());
        for (buf, s) in self.inputs.iter().zip(streams) {
            self.gpu.write_f32(buf, s);
        }
        let refs: Vec<&DeviceBuffer> = self.inputs.iter().collect();
        self.gpu.submit(&[], &self.projector.forward(&self.gpu, &refs));
        self.gpu.read(self.projector.out(), self.rows * self.projector.cfg.out_dim as usize)
    }

    pub fn zero_grads(&self) {
        self.projector.zero_grads(&self.gpu);
    }

    /// Accumulate the parameter gradients for `d_rows`, the loss's gradient
    /// at the output (after a [`Self::forward`] on the same streams), and
    /// return the gradient at each input stream.
    pub fn backward(&self, d_rows: &[f32]) -> Vec<Vec<f32>> {
        self.gpu.write_f32(&self.d_out, d_rows);
        let refs: Vec<&DeviceBuffer> = self.inputs.iter().collect();
        let d_refs: Vec<&DeviceBuffer> = self.d_inputs.iter().collect();
        self.gpu.submit(&[], &self.projector.backward(&self.gpu, &refs, &self.d_out, Some(&d_refs)));
        let n = self.rows * self.projector.cfg.input_dim as usize;
        self.d_inputs.iter().map(|b| self.gpu.read(b, n)).collect()
    }

    /// The accumulated gradient of every parameter.
    pub fn grads(&self) -> HashMap<String, Vec<f32>> {
        self.projector.cfg.param_list().into_iter().map(|(n, len)| (n.clone(), self.gpu.read(self.projector.grad(&n), len))).collect()
    }

    /// The parameters as trained so far.
    pub fn weights(&self) -> HashMap<String, Vec<f32>> {
        self.state.iter().map(|(n, (w, _, _))| (n.clone(), w.clone())).collect()
    }

    /// Replace the parameters (all of them).
    pub fn set_weights(&mut self, weights: &HashMap<String, Vec<f32>>) {
        for (name, w) in weights {
            self.gpu.write_f32(self.projector.param(name), w);
            self.state.get_mut(name).expect("a parameter of the projector").0 = w.clone();
        }
    }

    /// The row-major shape of parameter `name`: `[out, in]` for a weight,
    /// `[out]` for a bias.
    pub fn shape(&self, name: &str) -> Vec<u64> {
        let cfg = self.projector.cfg;
        let len = self.state[name].0.len() as u64;
        if name.ends_with(".bias") {
            return vec![len];
        }
        let inputs = if name.starts_with("in") { cfg.input_dim } else { cfg.n_embed } as u64;
        vec![len / inputs, inputs]
    }

    /// One AdamW step (1-based `t`) on the accumulated gradients, clipped to
    /// `grad_clip` in global norm when it is positive.
    pub fn step(&mut self, t: u32, lr: f32, weight_decay: f32, grad_clip: f32) {
        self.step_scaled(t, lr, weight_decay, grad_clip, 1.0);
    }

    /// [`Self::step`] on the accumulated gradients multiplied by `mean`
    /// (`1/K` after `K` examples), the clip applied to the scaled norm.
    pub fn step_scaled(&mut self, t: u32, lr: f32, weight_decay: f32, grad_clip: f32, mean: f32) {
        let grads = self.grads();
        let sum_sq: f64 = grads.values().flatten().map(|g| (*g as f64).powi(2)).sum();
        let scale = crate::grad_multiplier(sum_sq, (grad_clip > 0.0).then_some(grad_clip), mean);
        let adam = crate::Adam::default();
        for (name, g) in &grads {
            let (w, m, v) = self.state.get_mut(name).expect("a parameter of the projector");
            adam.update_slice(t, lr, weight_decay, scale, w, m, v, g);
            self.gpu.write_f32(self.projector.param(name), w);
        }
    }
}

