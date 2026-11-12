// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The recorded forward and backward graphs of the Ripple trainer, and the build that allocates what they bind.
//!
//! Swedish Embedded AB implements solutions for fine-tuning very large dense decoders on the memory a team
//! actually has, from full-weight training to adapters over a 4-bit base. If your team needs expertise in
//! training hybrid-attention language models then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::cell::Cell;
use std::collections::HashMap;

use gpu_core::{f, BufUsage, DeviceBuffer, Gpu, Step};
use model::adapter::exec::{AdapterRuntime, TieredBase};
use model::block::{self, Gqa, KernelIds};
use model::head_ce::{GradScale, HeadBackward, HeadCe, HeadCeConfig, HeadKernels, RowWeights};
use model::ops::{Act, Ops, Weight};
use model::saved::{Residuals, SavedFormat};
use model::segmented::SegmentedPass;
use model::Shard;
use paramstore::ParamStore;

use super::{Ripple, TrainConfig, TrainOpts, IGNORE};
use crate::config::LayerType;
use crate::import::layer_tensor;
use crate::weights::Loader;

/// Words of logits one chunk of the training head may hold (1 GiB of f32): rows beyond it are processed in chunks.
pub(super) const HEAD_BUDGET_WORDS: u64 = 1 << 28;

/// Kernel pipeline indices, resolved by name from the device's table once.
pub(super) struct Kernels {
    pub ids: KernelIds,
    pub gqa_scores_win: usize,
    pub sigmoid: usize,
    pub sigmoid_bwd: usize,
    pub mul: usize,
    pub add2: usize,
    pub clamp: usize,
    pub clamp_bwd: usize,
    pub lerp: usize,
    pub lerp_bwd: usize,
    pub lerp_dlam: usize,
    pub bias_grad: usize,
    pub axpy: usize,
    pub grad_scale: usize,
    pub ce_value: usize,
    pub ce_grad: usize,
    pub emb_bwd: usize,
    pub embed: usize,
    pub scale_row: usize,
}

impl Kernels {
    fn resolve(gpu: &Gpu) -> Kernels {
        let need = |name: &str| gpu.kernel_index(name).unwrap_or_else(|| panic!("ricursive training: kernel {name:?} is not registered"));
        Kernels {
            ids: KernelIds {
                rmsnorm: need("rmsnorm"),
                rms_inv: need("rms_inv"),
                rmsnorm_dx: need("rmsnorm_dx"),
                rmsnorm_dw: need("rmsnorm_dw"),
                rope: need("rope_base"),
                rope_bwd: need("rope_base_bwd"),
                gqa_scores: need("gqa_scores_win"),
                gqa_apply: need("gqa_apply"),
                attn_softmax: need("attn_softmax"),
                gqa_dscores: need("gqa_bwd_dscores"),
                gqa_dv: need("gqa_bwd_dv"),
                gqa_dq: need("gqa_bwd_dq"),
                gqa_dk: need("gqa_bwd_dk"),
                silu_mul: need("silu_mul"),
                silu_da: need("silu_bwd_da"),
                silu_db: need("silu_bwd_db"),
                rmsnorm_rows: block::UNREGISTERED,
                rmsnorm_dx_rows: block::UNREGISTERED,
            },
            gqa_scores_win: need("gqa_scores_win"),
            sigmoid: need("sigmoid"),
            sigmoid_bwd: need("sigmoid_bwd"),
            mul: need("mul"),
            add2: need("add2"),
            clamp: need("clamp_range"),
            clamp_bwd: need("clamp_range_bwd"),
            lerp: need("value_lerp"),
            lerp_bwd: need("value_lerp_bwd"),
            lerp_dlam: need("value_lerp_dlam"),
            bias_grad: need("bias_grad"),
            axpy: need("axpy"),
            grad_scale: need("grad_scale"),
            ce_value: need("ce_value"),
            ce_grad: need("ce_grad"),
            emb_bwd: need("emb_bwd"),
            embed: need("embed"),
            scale_row: need("scale_row"),
        }
    }
}

/// One linear: its checkpoint name, its adapter leaf, and the weight at the tier it is held in.
pub(super) struct Lin {
    pub name: String,
    pub leaf: &'static str,
    pub w: Weight,
}

impl Lin {
    fn n(&self) -> u32 {
        self.w.n()
    }
    fn k(&self) -> u32 {
        self.w.k()
    }
}

/// Names and linears of one layer.
pub(super) struct LayerParams {
    pub kind: LayerType,
    pub q: Lin,
    pub k: Lin,
    pub v: Lin,
    pub o: Lin,
    pub g: Option<Lin>,
    pub gate: Lin,
    pub up: Lin,
    pub down: Lin,
    pub input_ln: String,
    pub post_attn_ln: String,
    pub pre_ff_ln: String,
    pub post_ff_ln: String,
    pub q_norm: String,
    pub k_norm: String,
    pub lam: Option<String>,
}

/// One layer's activations, shared by every layer (the forward is rebuilt per layer in the backward).
pub(super) struct Scratch {
    pub xn: DeviceBuffer,
    pub v_in: DeviceBuffer,
    pub q_pre: DeviceBuffer,
    pub k_pre: DeviceBuffer,
    pub q: DeviceBuffer,
    pub k: DeviceBuffer,
    pub v: DeviceBuffer,
    pub scores: DeviceBuffer,
    pub probs: DeviceBuffer,
    pub ctx: DeviceBuffer,
    pub g_pre: DeviceBuffer,
    pub sg: DeviceBuffer,
    pub ctx2: DeviceBuffer,
    pub attn_out: DeviceBuffer,
    pub attn_normed: DeviceBuffer,
    pub x1: DeviceBuffer,
    pub m: DeviceBuffer,
    pub gate_pre: DeviceBuffer,
    pub up_pre: DeviceBuffer,
    /// The clamped branches; the pre-activations themselves when the model does not clip.
    pub gate_c: DeviceBuffer,
    pub up_c: DeviceBuffer,
    pub act: DeviceBuffer,
    pub mlp: DeviceBuffer,
    pub mlp_normed: DeviceBuffer,
}

/// The backward's working set.
pub(super) struct Grad {
    /// Gradient of the stream at each layer boundary, two buffers alternating by parity.
    pub dres: [DeviceBuffer; 2],
    /// Gradient of `x0` accumulated over every layer's value path, zeroed at the start of each backward.
    pub dx0: DeviceBuffer,
    pub dlogits: DeviceBuffer,
    pub d_hs: DeviceBuffer,
    pub rms_inv: DeviceBuffer,
    pub d_mlp: DeviceBuffer,
    pub d_act: DeviceBuffer,
    pub d_gate_c: DeviceBuffer,
    pub d_up_c: DeviceBuffer,
    pub d_gate_pre: DeviceBuffer,
    pub d_up_pre: DeviceBuffer,
    pub d_m: DeviceBuffer,
    pub d_x1_part: DeviceBuffer,
    pub d_x1: DeviceBuffer,
    pub d_attn_out: DeviceBuffer,
    pub d_ctx2: DeviceBuffer,
    pub d_ctx: DeviceBuffer,
    pub d_sg: DeviceBuffer,
    pub d_g_pre: DeviceBuffer,
    pub d_scores: DeviceBuffer,
    pub d_q: DeviceBuffer,
    pub d_k: DeviceBuffer,
    pub d_v: DeviceBuffer,
    pub d_q_pre: DeviceBuffer,
    pub d_k_pre: DeviceBuffer,
    pub d_xn: DeviceBuffer,
    pub d_vin: DeviceBuffer,
    pub d_hin_part: DeviceBuffer,
    pub lam_part: DeviceBuffer,
}

/// Floats of device memory the activations of a build hold: the boundary residuals, one layer's scratch, the
/// attention chain, the logits, and (training) the backward's working set. Parameters are not counted.
pub(super) fn activation_bytes(cfg: &TrainConfig, b: u32, t: u32, boundaries: SavedFormat, train: bool) -> u64 {
    let c = &cfg.model;
    let (n, d, hq, hkv, ff, v) = (u64::from(b) * u64::from(t), c.hidden_size as u64, c.q_dim() as u64, c.kv_dim() as u64, c.intermediate_size as u64, c.vocab_size as u64);
    let attn = u64::from(b) * c.num_attention_heads as u64 * u64::from(t) * u64::from(t);
    // The narrow store holds the boundaries; two f32 buffers serve every working copy.
    let res = if boundaries == SavedFormat::F32 { boundaries.bytes(n * c.num_hidden_layers as u64, d) } else { boundaries.bytes(n * c.num_hidden_layers as u64, d) + 8 * n * d };
    // A training build's head keeps one chunk of logits and of their gradient alive ([`HeadCe`]); the inference build
    // keeps every row's.
    let head_rows = if train { model::head_ce::rows_for_budget(n, v, HEAD_BUDGET_WORDS) } else { n };
    let scratch = 4 * (n * (8 * d + 2 * hq + 2 * hq + 2 * hq + hkv * 3 + 5 * ff) + 2 * attn) + 4 * head_rows * v + 4 * n * (3 * d);
    let grad = if train { 4 * (n * (16 * d + 6 * hq + 4 * hkv + 5 * ff) + attn + head_rows * v + n * d + n * c.num_attention_heads as u64) } else { 0 };
    res + scratch + grad
}

pub(super) fn build(cfg: TrainConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, opts: &TrainOpts, train: bool) -> Result<Ripple, String> {
    let shard = Shard::whole(cfg.model.num_hidden_layers);
    build_shard(cfg, b, t, src, opts, train, shard)
}

/// The stage `shard` of the model: only its layers (and the embedding, norm and head it needs) are allocated.
pub(super) fn build_shard(cfg: TrainConfig, b: u32, t: u32, src: &dyn checkpoint::TensorSource, opts: &TrainOpts, train: bool, shard: Shard) -> Result<Ripple, String> {
    let c = cfg.model.clone();
    if shard.start >= shard.end || shard.end > c.num_hidden_layers || (shard.start == 0) != shard.embed {
        return Err(format!("ricursive training: stage {}..{} (embed {}) does not fit {} layers", shard.start, shard.end, shard.embed, c.num_hidden_layers));
    }
    if b == 0 || t == 0 {
        return Err("ricursive training: batch and sequence must be positive".into());
    }
    if t > cfg.block_size {
        return Err(format!("ricursive training: sequence {t} exceeds the configured block of {}", cfg.block_size));
    }
    if !opts.saved.recomputes() {
        return Err("ricursive training recomputes each layer in the backward; SavedPolicy::Keep is not available".into());
    }
    let attn_elems = u64::from(b) * c.num_attention_heads as u64 * u64::from(t) * u64::from(t);
    if attn_elems > u64::from(u32::MAX) {
        return Err(format!("ricursive training: the materialised attention of {attn_elems} scores per layer exceeds the 32-bit index space; shorten the row"));
    }

    let gpu = Gpu::new(super::pipelines());
    let mut ops = Ops::new(gpu.share()).map_err(|e| format!("ricursive training: {e}"))?;
    ops.set_gemm_precision(opts.gemm);

    // Which linears are held packed (outside the store). Training a packed linear is a straight-through dead end:
    // only an adapter build, whose base is frozen, may hold one.
    let linear_shapes: Vec<(String, usize, usize)> = crate::weights::all_linears(&c).into_iter().filter(|(name, ..)| cfg.shard_owns(name, &shard)).collect();
    // A linear the adapter also saves in full is trained, so it lives in the store as f32 whatever tier the rest is held at.
    let saved_in_full = |name: &str| cfg.lora.as_ref().is_some_and(|lora| lora.saves(name));
    let packed: std::collections::HashSet<String> = linear_shapes.iter().filter(|(name, ..)| opts.tiers.want(name) != gpu_core::select::Dtype::F32 && !saved_in_full(name)).map(|(name, ..)| name.clone()).collect();
    if train && cfg.lora.is_none() && !packed.is_empty() {
        return Err(format!("ricursive training: a full fine-tune trains f32 weights; {} linear(s) are asked for as {} (train an adapter over a packed base instead)", packed.len(), opts.tiers.describe()));
    }

    // A resident full fine-tune holds the linears and the two tables in its own state, with their master copies and
    // optimizer state off the device; the store keeps what is left (the norms and the gates).
    let resident_specs = opts.resident.as_ref().map(|_| super::resident_specs(&cfg, &shard)).unwrap_or_default();
    if let Some(plan) = &opts.resident {
        if !train || cfg.lora.is_some() {
            return Err("ricursive training: a resident low-precision build is a full fine-tune: training, no adapter".into());
        }
        if opts.tiers.dtypes().any(|t| t != gpu_core::select::Dtype::F32) {
            return Err(format!("ricursive training: a resident full fine-tune holds its linears in the encoding of its plan, not in {} (give the encoding as weights=)", opts.tiers.describe()));
        }
        super::check_resident_fits(&cfg, &shard, plan)?;
    }
    let resident_names: std::collections::HashSet<&str> = resident_specs.iter().map(|spec| spec.name.as_str()).collect();
    let roles: Vec<(String, usize, paramstore::Role)> = cfg
        .shard_param_list(&shard)
        .into_iter()
        .filter(|(name, _)| !packed.contains(name) && !resident_names.contains(name.as_str()))
        .map(|(name, numel)| {
            let role = if train { Ripple::role(&cfg, &name) } else { paramstore::Role::Frozen };
            (name, numel, role)
        })
        .collect();
    let mut ps = match &opts.resident {
        // The store trains as the store runs it: the run's optimizer-state format, f32 values and gradients.
        Some(plan) => {
            let policy = paramstore::precision::PrecisionPolicy::uniform(paramstore::precision::ParamPrecision { state: plan.cfg.state, ..paramstore::precision::ParamPrecision::FP32 });
            ParamStore::new_with_policy_src(&gpu, roles, src, &policy)
        }
        None => ParamStore::new_with_roles_src(&gpu, roles, src),
    };
    if let Some(lora) = cfg.lora.as_ref().filter(|_| train) {
        model::adapter::device::apply_groups(&mut ps, &lora.hp());
    }
    let resident = opts.resident.as_ref().map(|plan| model::resident_ft::ResidentFullFt::new(&ops, plan.cfg.clone(), resident_specs.clone(), src, &plan.budget())).transpose()?;

    // Every linear as a `Weight` at its tier.
    let loader = Loader::new(&ops, src, &opts.tiers);
    let mut weights: HashMap<String, Weight> = HashMap::new();
    for (name, n, k) in &linear_shapes {
        let w = if let Some(ft) = &resident {
            ft.weight(ft.index_of(name).expect("every linear of a resident build is a resident tensor")).clone()
        } else if packed.contains(name) {
            loader.linear(name, *n, *k)?
        } else {
            Weight::dense_f32(ps.w(name).clone(), *n as u32, *k as u32)
        };
        weights.insert(name.clone(), w);
    }
    let lin = |leaf: &'static str, name: String, weights: &mut HashMap<String, Weight>| -> Lin {
        let w = weights.get(&name).unwrap_or_else(|| panic!("ricursive training: no weight {name}")).clone();
        Lin { name, leaf, w }
    };
    let mut layers = Vec::with_capacity(c.num_hidden_layers);
    for l in shard.start..shard.end {
        let mut linears: HashMap<&'static str, Lin> = cfg.layer_linears(l).into_iter().map(|(leaf, name, ..)| (leaf, lin(leaf, name, &mut weights))).collect();
        let take = |leaf: &str, linears: &mut HashMap<&'static str, Lin>| linears.remove(leaf).unwrap_or_else(|| panic!("ricursive training: layer {l} has no {leaf}"));
        let name = |suffix: &str| layer_tensor(l, suffix);
        layers.push(LayerParams {
            kind: c.layer_type(l),
            q: take("q_proj", &mut linears),
            k: take("k_proj", &mut linears),
            v: take("v_proj", &mut linears),
            o: take("o_proj", &mut linears),
            g: c.use_gated_attention.then(|| take("g_proj", &mut linears)),
            gate: take("gate_proj", &mut linears),
            up: take("up_proj", &mut linears),
            down: take("down_proj", &mut linears),
            input_ln: name("input_layernorm.weight"),
            post_attn_ln: name("post_attention_layernorm.weight"),
            pre_ff_ln: name("pre_feedforward_layernorm.weight"),
            post_ff_ln: name("post_feedforward_layernorm.weight"),
            q_norm: name("self_attn.q_norm.weight"),
            k_norm: name("self_attn.k_norm.weight"),
            lam: c.use_valres.then(|| name("self_attn.value_res_lambda")),
        });
    }
    let head = shard.head.then(|| Lin { name: "lm_head.weight".into(), leaf: "lm_head", w: weights["lm_head.weight"].clone() });

    let (nn, d, hq, hkv, ff, v) = (u64::from(b) * u64::from(t), c.hidden_size as u64, c.q_dim() as u64, c.kv_dim() as u64, c.intermediate_size as u64, c.vocab_size as u64);
    let max_out = hq.max(ff).max(d).max(hkv);
    let adapter = cfg.lora.as_ref().map(|lora| AdapterRuntime::new(&gpu, lora, nn as u32, max_out as u32));
    let st = |x: u64| gpu.storage(x);

    let tokens = gpu.buffer("tokens", nn * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
    // The embedding gradient of a resident build scatters by the distinct ids of the batch.
    let tok_uniq = (resident.is_some() && (shard.embed || c.use_valres)).then(|| gpu.storage(nn));
    let targets = gpu.buffer("targets", nn * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
    let x0 = st(nn * d);
    // Boundary `l` is held by the stage that writes it (`shard.end` included) and by the one that reads it as its input.
    let live = shard.start.max(1)..shard.end + 1;
    let mut bufs = vec![st(1)];
    bufs.extend((1..=c.num_hidden_layers).map(|l| if live.contains(&l) { st(nn * d) } else { st(1) }));
    let mut res = Residuals::new(bufs, live);
    if opts.saved.boundary_format() != SavedFormat::F32 {
        res.quantize(&gpu, opts.saved.boundary_format(), nn as u32, d as u32)?;
    }

    let gate_pre = st(nn * ff);
    let up_pre = st(nn * ff);
    let clip = c.ffn_clamp_threshold.is_some();
    let scratch = Scratch {
        xn: st(nn * d),
        v_in: if c.use_valres { st(nn * d) } else { st(1) },
        q_pre: st(nn * hq),
        k_pre: st(nn * hkv),
        q: st(nn * hq),
        k: st(nn * hkv),
        v: st(nn * hkv),
        scores: st(attn_elems),
        probs: st(attn_elems),
        ctx: st(nn * hq),
        g_pre: if c.use_gated_attention { st(nn * hq) } else { st(1) },
        sg: if c.use_gated_attention { st(nn * hq) } else { st(1) },
        ctx2: if c.use_gated_attention { st(nn * hq) } else { st(1) },
        attn_out: st(nn * d),
        attn_normed: st(nn * d),
        x1: st(nn * d),
        m: st(nn * d),
        gate_c: if clip { st(nn * ff) } else { gate_pre.clone() },
        up_c: if clip { st(nn * ff) } else { up_pre.clone() },
        gate_pre,
        up_pre,
        act: st(nn * ff),
        mlp: st(nn * d),
        mlp_normed: st(nn * d),
    };
    // The training head never holds every row's logits: [`HeadCe`] computes them a chunk at a time, in the forward for
    // the loss and again in the backward for the gradient.
    let head_ce = match (&head, train) {
        (Some(head), true) => {
            let rows = opts.head_chunk_rows.unwrap_or_else(|| model::head_ce::rows_for_budget(nn, v, HEAD_BUDGET_WORDS) as u32);
            let kernels = HeadKernels::from_gpu(&gpu)?;
            Some(HeadCe::new(&ops, d as u32, v as u32, &head.w, HeadCeConfig { rows_per_chunk: rows, tile_budget_words: None }, kernels)?)
        }
        _ => None,
    };
    let (hs, logits, ce_rows) = match (shard.head, head_ce.is_some()) {
        (true, true) => (st(nn * d), st(1), st(nn)),
        (true, false) => (st(nn * d), st(nn * v), st(nn)),
        _ => (st(1), st(1), st(1)),
    };

    let grad = train.then(|| {
        let d_gate_c = st(nn * ff);
        let d_up_c = st(nn * ff);
        Grad {
            dres: [st(nn * d), st(nn * d)],
            dx0: st(nn * d),
            dlogits: if shard.head && head_ce.is_none() { st(nn * v) } else { st(1) },
            d_hs: if shard.head { st(nn * d) } else { st(1) },
            rms_inv: st(nn * c.num_attention_heads as u64),
            d_mlp: st(nn * d),
            d_act: st(nn * ff),
            d_gate_pre: if clip { st(nn * ff) } else { d_gate_c.clone() },
            d_up_pre: if clip { st(nn * ff) } else { d_up_c.clone() },
            d_gate_c,
            d_up_c,
            d_m: st(nn * d),
            d_x1_part: st(nn * d),
            d_x1: st(nn * d),
            d_attn_out: st(nn * d),
            d_ctx2: st(nn * hq),
            d_ctx: if c.use_gated_attention { st(nn * hq) } else { st(1) },
            d_sg: if c.use_gated_attention { st(nn * hq) } else { st(1) },
            d_g_pre: if c.use_gated_attention { st(nn * hq) } else { st(1) },
            d_scores: st(attn_elems),
            d_q: st(nn * hq),
            d_k: st(nn * hkv),
            d_v: st(nn * hkv),
            d_q_pre: st(nn * hq),
            d_k_pre: st(nn * hkv),
            d_xn: st(nn * d),
            d_vin: st(nn * d),
            d_hin_part: st(nn * d),
            lam_part: st(d),
        }
    });

    let k = Kernels::resolve(&gpu);
    let opt = train.then(|| {
        let idx = |name: &str| gpu.kernel_index(name).unwrap_or_else(|| panic!("ricursive training: kernel {name:?} is not registered"));
        optim::Optim::new(idx("adamw"), idx("gradnorm_sq"), idx("grad_scale"), idx("clip_coef"), idx("grad_scale_buf"))
    });
    let ce_grad_uniform = gpu.uniform_dynamic(4);
    let inv_count = gpu.storage(1);

    let mut model = Ripple {
        gpu,
        ops,
        cfg,
        shard,
        b,
        t,
        train,
        count: Cell::new(1.0),
        ps,
        weights,
        layers,
        head,
        adapter,
        k,
        tokens,
        targets,
        x0,
        res,
        hs,
        logits,
        ce_rows,
        scratch,
        grad,
        ce_grad_uniform,
        inv_count,
        head_ce,
        fwd: SegmentedPass::default(),
        bwd: SegmentedPass::default(),
        resident: resident.map(std::cell::RefCell::new),
        resident_norm: Cell::new(None),
        backward_pending: Cell::new(false),
        tok_uniq,
        opt,
        nlsa: None,
        weighted: None,
        logit_scale: 1.0,
    };
    model.record();
    Ok(model)
}

impl Ripple {
    /// Record the forward and (for a training build) the backward over the buffers as they are now. Called again
    /// whenever one of them is replaced (the boundary store narrowing).
    pub(super) fn record(&mut self) {
        self.fwd = self.forward_tape();
        self.bwd = if self.train { self.backward_tape() } else { SegmentedPass::default() };
    }

    fn boundary(&self, i: usize) -> &DeviceBuffer {
        if i == 0 {
            &self.x0
        } else {
            &self.res[i]
        }
    }

    /// `x`'s activation operand for the products of `weights`: the f32 rows, or the rows quantized for an int8 tier.
    fn act(&self, s: &mut Vec<Step>, x: &DeviceBuffer, k: u32, weights: &[&Weight]) -> Act {
        if weights.iter().all(|w| w.reads_f32_activations()) {
            self.ops.act_f32(x, 0, self.rows(), k)
        } else {
            self.ops.act(s, x, 0, self.rows(), k)
        }
    }

    /// `y = x W^T` (+ the adapter's correction when `lin` is targeted).
    fn lin_fwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, lin: &Lin, y: &DeviceBuffer) {
        let act = self.act(s, x, lin.k(), &[&lin.w]);
        self.ops.matmul(s, &lin.w, &act, y, 0);
        if let Some(rt) = self.adapter.as_ref().filter(|rt| rt.hits(lin.leaf)) {
            let params = TieredBase { params: &self.ps, ops: &self.ops, weights: &self.weights };
            rt.forward(self.ops.tensor_gemm(), s, &params, &lin.name, x, y, self.rows(), lin.k(), lin.n());
        }
    }

    /// The backward of [`Self::lin_fwd`]: the weight (or adapter) gradient from `d_out` and the saved input `x`, and
    /// the gradient of `x` into `dx` (added when `acc`, else written).
    fn lin_bwd(&self, s: &mut Vec<Step>, lin: &Lin, d_out: &DeviceBuffer, x: &DeviceBuffer, dx: &DeviceBuffer, acc: bool) {
        let m = self.rows();
        let base_dx = |s: &mut Vec<Step>, dy: &DeviceBuffer| self.ops.matmul_dx(s, &lin.w, dy, m, dx, acc);
        if let Some(rt) = self.adapter.as_ref().filter(|rt| rt.hits(lin.leaf)) {
            let params = TieredBase { params: &self.ps, ops: &self.ops, weights: &self.weights };
            rt.backward(self.ops.tensor_gemm(), s, &params, &lin.name, d_out, x, dx, m, lin.k(), lin.n(), &base_dx);
            // A linear the adapter also saves in full has its own weight gradient beside the factors'.
            self.weight_grad(s, lin, d_out, x);
            return;
        }
        self.weight_grad(s, lin, d_out, x);
        base_dx(s, d_out);
    }

    /// The weight gradient `dW += d_out^T x` of `lin` where the weight trains: in the store's f32 buffer, or in the
    /// resident state's accumulating one against the stored tier (the shared op). Nothing for a frozen weight.
    fn weight_grad(&self, s: &mut Vec<Step>, lin: &Lin, d_out: &DeviceBuffer, x: &DeviceBuffer) {
        if self.is_trainable(&lin.name) {
            self.ops.matmul_dw(s, x, d_out, self.rows(), lin.n(), lin.k(), self.ps.g(&lin.name));
        } else if let Some(ft) = &self.resident {
            let ft = ft.borrow();
            if let Some(i) = ft.index_of(&lin.name) {
                ft.weight_grad(&self.ops, s, i, x, d_out, self.rows());
            }
        }
    }

    /// The gain gradient buffer of a norm, `None` when the gain is frozen.
    fn gain_grad(&self, name: &str) -> Option<&DeviceBuffer> {
        self.is_trainable(name).then(|| self.ps.g(name))
    }

    fn rmsnorm_fwd(&self, x: &DeviceBuffer, gain: &str, out: &DeviceBuffer, dim: u32, rows: u32) -> Step {
        block::rmsnorm_fwd(&self.gpu, &self.k.ids, x, self.ps.w(gain), out, dim, rows, self.cfg.model.rms_norm_eps)
    }

    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_bwd(&self, s: &mut Vec<Step>, x: &DeviceBuffer, gain: &str, dy: &DeviceBuffer, dx: &DeviceBuffer, dim: u32, rows: u32) {
        let g = self.grad.as_ref().expect("a backward needs the training build");
        s.extend(block::rmsnorm_bwd(&self.gpu, &self.k.ids, x, self.ps.w(gain), dy, dx, &g.rms_inv, self.gain_grad(gain), dim, rows, self.cfg.model.rms_norm_eps));
    }

    fn elementwise(&self, kernel: usize, bufs: &[&DeviceBuffer], n: u32) -> Step {
        self.gpu.step(kernel, bufs, &[n], n)
    }

    fn window(&self, kind: LayerType) -> u32 {
        match kind {
            LayerType::Sliding => (self.cfg.model.sliding_window as u32).min(self.t),
            LayerType::Full => self.t,
        }
    }

    /// The forward of layer `l` from the stream `h_in` into `h_out`, over the shared scratch.
    pub(super) fn layer_fwd(&self, s: &mut Vec<Step>, l: usize, h_in: &DeviceBuffer, h_out: &DeviceBuffer, recompute: bool) {
        let c = &self.cfg.model;
        let (lp, sc, gpu, ids) = (&self.layers[l - self.shard.start], &self.scratch, &self.gpu, &self.k.ids);
        let (n, d, hq, hkv, hd, ff) = (self.rows(), c.hidden_size as u32, c.q_dim() as u32, c.kv_dim() as u32, c.head_dim as u32, c.intermediate_size as u32);
        let (heads, kv_heads) = (c.num_attention_heads as u32, c.num_key_value_heads as u32);

        s.push(self.rmsnorm_fwd(h_in, &lp.input_ln, &sc.xn, d, n));
        self.lin_fwd(s, &sc.xn, &lp.q, &sc.q_pre);
        self.lin_fwd(s, &sc.xn, &lp.k, &sc.k_pre);
        let v_in = match &lp.lam {
            Some(lam) => {
                s.push(gpu.step(self.k.lerp, &[&self.x0, &sc.xn, self.ps.w(lam), &sc.v_in], &[n * d], n * d));
                &sc.v_in
            }
            None => &sc.xn,
        };
        self.lin_fwd(s, v_in, &lp.v, &sc.v);
        s.push(self.rmsnorm_fwd(&sc.q_pre, &lp.q_norm, &sc.q, hd, n * heads));
        s.push(self.rmsnorm_fwd(&sc.k_pre, &lp.k_norm, &sc.k, hd, n * kv_heads));
        if lp.kind == LayerType::Sliding {
            s.push(block::rope_fwd(gpu, ids, &sc.q, n, heads, hd, hq, self.t, c.rope_theta));
            s.push(block::rope_fwd(gpu, ids, &sc.k, n, kv_heads, hd, hkv, self.t, c.rope_theta));
        }
        // A global layer under native sparse attention reads its selection; every other layer reads its window (or all keys).
        if self.native_fwd(s, l, recompute).is_none() {
            let ga = Gqa { b: self.b, t: self.t, n_heads: heads, n_kv_heads: kv_heads, head_dim: hd };
            s.extend(block::gqa_fwd_win(gpu, self.k.gqa_scores_win, ids, &ga, self.window(lp.kind), &sc.q, &sc.k, &sc.v, &sc.scores, &sc.probs, &sc.ctx));
        }
        let attended = match &lp.g {
            Some(g) => {
                self.lin_fwd(s, &sc.xn, g, &sc.g_pre);
                s.push(gpu.step(self.k.sigmoid, &[&sc.g_pre, &sc.sg], &[n * hq], n * hq));
                s.push(self.elementwise(self.k.mul, &[&sc.ctx, &sc.sg, &sc.ctx2], n * hq));
                &sc.ctx2
            }
            None => &sc.ctx,
        };
        self.lin_fwd(s, attended, &lp.o, &sc.attn_out);
        s.push(self.rmsnorm_fwd(&sc.attn_out, &lp.post_attn_ln, &sc.attn_normed, d, n));
        s.push(self.elementwise(self.k.add2, &[h_in, &sc.attn_normed, &sc.x1], n * d));

        s.push(self.rmsnorm_fwd(&sc.x1, &lp.pre_ff_ln, &sc.m, d, n));
        self.lin_fwd(s, &sc.m, &lp.gate, &sc.gate_pre);
        self.lin_fwd(s, &sc.m, &lp.up, &sc.up_pre);
        if let Some(limit) = c.ffn_clamp_threshold {
            s.push(gpu.step(self.k.clamp, &[&sc.gate_pre, &sc.gate_c], &[n * ff, f(-f32::MAX), f(limit)], n * ff));
            s.push(gpu.step(self.k.clamp, &[&sc.up_pre, &sc.up_c], &[n * ff, f(-limit), f(limit)], n * ff));
        }
        s.push(block::swiglu_fwd(gpu, ids, &sc.gate_c, &sc.up_c, &sc.act, n * ff));
        self.lin_fwd(s, &sc.act, &lp.down, &sc.mlp);
        s.push(self.rmsnorm_fwd(&sc.mlp, &lp.post_ff_ln, &sc.mlp_normed, d, n));
        s.push(self.elementwise(self.k.add2, &[&sc.x1, &sc.mlp_normed, h_out], n * d));
    }

    /// The backward of layer `l` after [`Self::layer_fwd`] rebuilt the scratch from `h_in`: `dh_out` is the gradient of
    /// the layer's output, `dh_in` receives the gradient of `h_in`. `x0`'s gradient accumulates into `dx0`.
    pub(super) fn layer_bwd(&self, s: &mut Vec<Step>, l: usize, h_in: &DeviceBuffer, dh_out: &DeviceBuffer, dh_in: &DeviceBuffer) {
        let c = &self.cfg.model;
        let (lp, sc, gpu, ids) = (&self.layers[l - self.shard.start], &self.scratch, &self.gpu, &self.k.ids);
        let g = self.grad.as_ref().expect("a backward needs the training build");
        let (n, d, hq, hkv, hd, ff) = (self.rows(), c.hidden_size as u32, c.q_dim() as u32, c.kv_dim() as u32, c.head_dim as u32, c.intermediate_size as u32);
        let (heads, kv_heads) = (c.num_attention_heads as u32, c.num_key_value_heads as u32);

        // h_out = x1 + post_ff_norm(down(act))
        self.rmsnorm_bwd(s, &sc.mlp, &lp.post_ff_ln, dh_out, &g.d_mlp, d, n);
        self.lin_bwd(s, &lp.down, &g.d_mlp, &sc.act, &g.d_act, false);
        s.extend(block::swiglu_bwd(gpu, ids, &sc.gate_c, &sc.up_c, &g.d_act, &g.d_gate_c, &g.d_up_c, n * ff));
        if let Some(limit) = c.ffn_clamp_threshold {
            s.push(gpu.step(self.k.clamp_bwd, &[&sc.gate_pre, &g.d_gate_c, &g.d_gate_pre], &[n * ff, f(-f32::MAX), f(limit)], n * ff));
            s.push(gpu.step(self.k.clamp_bwd, &[&sc.up_pre, &g.d_up_c, &g.d_up_pre], &[n * ff, f(-limit), f(limit)], n * ff));
        }
        self.lin_bwd(s, &lp.gate, &g.d_gate_pre, &sc.m, &g.d_m, false);
        self.lin_bwd(s, &lp.up, &g.d_up_pre, &sc.m, &g.d_m, true);
        self.rmsnorm_bwd(s, &sc.x1, &lp.pre_ff_ln, &g.d_m, &g.d_x1_part, d, n);
        s.push(self.elementwise(self.k.add2, &[dh_out, &g.d_x1_part, &g.d_x1], n * d));

        // x1 = h_in + post_attn_norm(o(attended))
        self.rmsnorm_bwd(s, &sc.attn_out, &lp.post_attn_ln, &g.d_x1, &g.d_attn_out, d, n);
        let attended = if lp.g.is_some() { &sc.ctx2 } else { &sc.ctx };
        self.lin_bwd(s, &lp.o, &g.d_attn_out, attended, &g.d_ctx2, false);
        let d_ctx = match &lp.g {
            Some(_) => {
                // ctx2 = ctx * sigmoid(g_pre)
                s.push(self.elementwise(self.k.mul, &[&g.d_ctx2, &sc.sg, &g.d_ctx], n * hq));
                s.push(self.elementwise(self.k.mul, &[&g.d_ctx2, &sc.ctx, &g.d_sg], n * hq));
                s.push(gpu.step(self.k.sigmoid_bwd, &[&sc.g_pre, &g.d_sg, &g.d_g_pre], &[n * hq], n * hq));
                &g.d_ctx
            }
            None => &g.d_ctx2,
        };
        if self.native_bwd(s, l, d_ctx).is_none() {
            let ga = Gqa { b: self.b, t: self.t, n_heads: heads, n_kv_heads: kv_heads, head_dim: hd };
            s.extend(block::gqa_bwd(gpu, ids, &ga, &sc.q, &sc.k, &sc.v, &sc.probs, d_ctx, &g.d_scores, &g.d_q, &g.d_k, &g.d_v));
        }
        if lp.kind == LayerType::Sliding {
            s.push(block::rope_bwd(gpu, ids, &g.d_q, n, heads, hd, hq, self.t, c.rope_theta));
            s.push(block::rope_bwd(gpu, ids, &g.d_k, n, kv_heads, hd, hkv, self.t, c.rope_theta));
        }
        self.rmsnorm_bwd(s, &sc.q_pre, &lp.q_norm, &g.d_q, &g.d_q_pre, hd, n * heads);
        self.rmsnorm_bwd(s, &sc.k_pre, &lp.k_norm, &g.d_k, &g.d_k_pre, hd, n * kv_heads);
        // The normed stream xn feeds q, k, the gate and (through the lerp or directly) v.
        self.lin_bwd(s, &lp.q, &g.d_q_pre, &sc.xn, &g.d_xn, false);
        self.lin_bwd(s, &lp.k, &g.d_k_pre, &sc.xn, &g.d_xn, true);
        if let Some(gp) = &lp.g {
            self.lin_bwd(s, gp, &g.d_g_pre, &sc.xn, &g.d_xn, true);
        }
        match &lp.lam {
            Some(lam) => {
                self.lin_bwd(s, &lp.v, &g.d_v, &sc.v_in, &g.d_vin, false);
                s.push(gpu.step(self.k.lerp_bwd, &[&g.d_vin, self.ps.w(lam), &g.dx0, &g.d_xn], &[n * d], n * d));
                if self.is_trainable(lam) {
                    s.push(gpu.step(self.k.lerp_dlam, &[&g.d_vin, &self.x0, &sc.xn, self.ps.w(lam), &g.lam_part], &[n, d], d));
                    s.push(gpu.step(self.k.bias_grad, &[&g.lam_part, self.ps.g(lam)], &[d, 1], 1));
                }
            }
            None => self.lin_bwd(s, &lp.v, &g.d_v, &sc.xn, &g.d_xn, true),
        }
        self.rmsnorm_bwd(s, h_in, &lp.input_ln, &g.d_xn, &g.d_hin_part, d, n);
        s.push(self.elementwise(self.k.add2, &[&g.d_x1, &g.d_hin_part, dh_in], n * d));
    }

    /// The recorded forward in segments: the embedding, one per layer, then the head and the loss.
    fn forward_tape(&self) -> SegmentedPass {
        let c = &self.cfg.model;
        let (n, d, v) = (self.rows(), c.hidden_size as u32, c.vocab_size as u32);
        let layers = c.num_hidden_layers;
        let mut pass = SegmentedPass::default();
        let mut s = Vec::new();
        // The embedding feeds the stream on the first stage and every layer's value residual on any stage.
        if self.shard.embed || c.use_valres {
            match self.resident_table("model.embed_tokens.weight") {
                Some(table) => self.ops.embed(&mut s, &table, &self.tokens, n, &self.x0),
                None => s.push(self.gpu.step(self.k.embed, &[&self.tokens, self.ps.w("model.embed_tokens.weight"), &self.x0], &[d, n], n * d)),
            }
            if let Some(scale) = c.embed_scale {
                s.push(self.gpu.step(self.k.grad_scale, &[&self.x0], &[n * d, f(scale)], n * d));
            }
        }
        pass.segment(&mut s, None);
        for l in self.shard.start..self.shard.end {
            self.layer_fwd(&mut s, l, self.boundary(l), &self.res[l + 1], false);
            self.res.round_trip(&self.gpu, &mut s, l + 1);
            pass.segment(&mut s, Some(l));
        }
        let Some(head) = &self.head else { return pass };
        s.push(self.rmsnorm_fwd(&self.res[layers], "model.norm.weight", &self.hs, d, n));
        if let Some(chunked) = &self.head_ce {
            chunked.forward_steps(&self.ops, &mut s, &self.hs, n, &self.targets, &self.ce_rows, self.logit_scale);
            pass.segment(&mut s, None);
            return pass;
        }
        let act = self.act(&mut s, &self.hs, d, &[&head.w]);
        self.ops.matmul(&mut s, &head.w, &act, &self.logits, 0);
        if self.logit_scale != 1.0 {
            s.push(self.gpu.step(self.k.grad_scale, &[&self.logits], &[n * v, f(self.logit_scale)], n * v));
        }
        s.push(self.gpu.step(self.k.ce_value, &[&self.logits, &self.targets, &self.ce_rows], &[n, v, IGNORE], n));
        pass.segment(&mut s, None);
        pass
    }

    /// The table `name` of a resident build as the `Weight` its products read, `None` when the store holds it.
    pub(super) fn resident_table(&self, name: &str) -> Option<Weight> {
        let ft = self.resident.as_ref()?.borrow();
        ft.index_of(name).map(|i| ft.weight(i).clone())
    }

    /// The gradient buffer of tensor `name` and the dtype it accumulates in: the resident state's on a resident build,
    /// the store's f32 one where the tensor trains there, `None` for a frozen tensor.
    pub(super) fn table_grad(&self, name: &str) -> Option<(DeviceBuffer, model::ops::GradDtype)> {
        if let Some(ft) = &self.resident {
            let ft = ft.borrow();
            if let Some(i) = ft.index_of(name) {
                return Some((ft.grad_buffer(i).clone(), ft.grad_dtype()));
            }
        }
        self.is_trainable(name).then(|| (self.ps.g(name).clone(), model::ops::GradDtype::F32))
    }

    /// The recorded backward in segments: the head and final norm, one per layer from the last down (each rebuilds its
    /// layer from the saved boundary before differentiating it), then the embedding.
    fn backward_tape(&self) -> SegmentedPass {
        let c = &self.cfg.model;
        let g = self.grad.as_ref().expect("a backward needs the training build");
        let (n, d, v) = (self.rows(), c.hidden_size as u32, c.vocab_size as u32);
        let layers = c.num_hidden_layers;
        let mut pass = SegmentedPass::default();
        pass.clear_on_submit(&g.dx0);
        let mut s = Vec::new();
        // The head stage starts the chain from the loss; any other receives the gradient of its output boundary.
        if let (Some(head), Some(chunked)) = (&self.head, &self.head_ce) {
            // The chunked head recomputes each chunk's logits, turns them into the loss gradient (scaled by 1/count, then by
            // the per-token weights of a weighted loss) and applies the head: no row's logits outlive its chunk.
            let dw = self.table_grad(&head.name);
            let dw = dw.as_ref().map(|(buf, dtype)| model::head_ce::HeadGrad { buf, dtype: *dtype });
            let row_weights = self.weighted.as_ref().map(|w| RowWeights { weights: w.weights(), scratch: w.scratch() });
            chunked.backward_steps(&self.ops, &mut s, &HeadBackward { hidden: &self.hs, d_hidden: &g.d_hs, n, labels: &self.targets, grad_scale: GradScale::Buf(&self.inv_count), logit_scale: self.logit_scale, row_weights, dw, forward_state_held: false });
            self.res.restore(&self.gpu, &mut s, layers);
            self.rmsnorm_bwd(&mut s, &self.res[layers], "model.norm.weight", &g.d_hs, &g.dres[layers % 2], d, n);
        } else if let Some(head) = &self.head {
            s.push(self.gpu.step_buf(self.k.ce_grad, &self.ce_grad_uniform, &[&self.logits, &self.targets, &g.dlogits], n * v));
            if self.logit_scale != 1.0 {
                s.push(self.gpu.step(self.k.grad_scale, &[&g.dlogits], &[n * v, f(self.logit_scale)], n * v));
            }
            let dlogits = match &self.weighted {
                Some(w) => w.hook(&self.gpu, &mut s, self.k.scale_row, &g.dlogits),
                None => &g.dlogits,
            };
            if let Some((buf, dtype)) = self.table_grad(&head.name) {
                assert_eq!(dtype, model::ops::GradDtype::F32, "ricursive: the unchunked head accumulates an f32 gradient");
                self.ops.matmul_dw(&mut s, &self.hs, dlogits, n, v, d, &buf);
            }
            self.ops.matmul_dx(&mut s, &head.w, dlogits, n, &g.d_hs, false);
            self.res.restore(&self.gpu, &mut s, layers);
            self.rmsnorm_bwd(&mut s, &self.res[layers], "model.norm.weight", &g.d_hs, &g.dres[layers % 2], d, n);
        }
        // The head and final norm are complete; the layers follow, the last first.
        pass.segment(&mut s, None);
        for l in (self.shard.start..self.shard.end).rev() {
            if l >= 1 {
                self.res.restore(&self.gpu, &mut s, l);
            }
            let h_in = self.boundary(l);
            self.layer_fwd(&mut s, l, h_in, &self.res[l + 1], true);
            self.layer_bwd(&mut s, l, h_in, &g.dres[(l + 1) % 2], &g.dres[l % 2]);
            pass.segment(&mut s, Some(l));
        }
        // x0 is both layer 0's input and every layer's value source: its gradient is the stream's plus the sum of
        // the value paths', scaled back through the embedding's multiplier. A later stage holds its own copy of the
        // embedding for the value paths alone; the pipeline sums the copies' gradients.
        let dx0 = if self.shard.embed {
            s.push(self.gpu.step(self.k.axpy, &[&g.dres[0], &g.dx0], &[n * d, f(1.0)], n * d));
            &g.dres[0]
        } else {
            &g.dx0
        };
        if let Some((dw, grad)) = self.table_grad("model.embed_tokens.weight") {
            if let Some(scale) = c.embed_scale {
                s.push(self.gpu.step(self.k.grad_scale, &[dx0], &[n * d, f(scale)], n * d));
            }
            match &self.tok_uniq {
                // The shared op scatters by the batch's distinct ids: a vocabulary-wide kernel per token would cost
                // `vocab * hidden * rows` at the real model's size.
                Some(uniq) => self.ops.embedding_grad(&mut s, &model::ops::EmbeddingGrad { tokens: &self.tokens, uniq, n_uniq: n, d_x: dx0, n_rows: n, width: d, dw: &dw, grad }),
                None => {
                    assert_eq!(grad, model::ops::GradDtype::F32, "ricursive: the store accumulates an f32 embedding gradient");
                    s.push(self.gpu.step(self.k.emb_bwd, &[&self.tokens, dx0, &dw], &[n, d, v], v * d));
                }
            }
        }
        pass.segment(&mut s, None);
        pass
    }

    /// Run the recorded forward on the batch already set. A layer-streamed build loads each layer's weights into its
    /// device slots as the forward reaches it.
    pub(super) fn run_forward(&self) {
        if !self.layer_streamed() {
            self.fwd.submit(&self.gpu);
            return;
        }
        self.fwd.submit_hooked(&self.gpu, &mut |edge| {
            if let model::segmented::SegmentEdge::Before { layer: Some(layer), .. } = edge {
                self.stream_load(layer);
            }
        });
    }

    /// Run the recorded backward on the batch already set. A layer-streamed build runs it inside the update instead
    /// ([`model::resident_ft::adamw_step`]), where the layers it loads are the ones it updates.
    pub(super) fn run_backward(&self) {
        self.gpu.write(&self.ce_grad_uniform, &[self.rows(), self.cfg.model.vocab_size as u32, IGNORE, f(self.count.get())]);
        self.gpu.write(&self.inv_count, &[(1.0 / self.count.get()).to_bits()]);
        if self.layer_streamed() {
            assert!(!self.backward_pending.replace(true), "ricursive layer-streamed full fine-tune: a second backward before the update (gradient accumulation) is not supported; every micro-batch is its own step");
            return;
        }
        self.bwd.submit(&self.gpu);
    }
}
