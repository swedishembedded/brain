// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Full training pipeline (forward + backprop + AdamW) as a WGSL compute
//! pipeline. Mirrors `tiny_sparse_moe.py`'s training step exactly, so the Rust
//! executable can train the model from scratch on the GPU.
//!
//! Two entry points:
//!   * `validate(path)` — load the PyTorch golden reference (init weights, a
//!     fixed batch, per-parameter grads, post-AdamW weights), run one Rust step
//!     and report the max gradient / weight error. This is the correctness gate.
//!   * `train(args)`    — generate the toy corpus, init weights, and run the
//!     optimisation loop, then save weights in the inference engine's format.
//!
//! Design notes:
//!   * fp32 only; <=5 storage buffers per kernel; single bind group. We request
//!     `max_storage_buffers_per_shader_stage = 8` (well within Pascal/sm_61).
//!   * The forward pass is written SSA-style: every stage writes a fresh buffer
//!     which doubles as its activation cache, so backprop has everything it
//!     needs and no mid-pass copies are required.
//!   * MoE uses the dense top-k formulation (all experts evaluated, masked by
//!     the renormalised gate). With capacity dropping disabled this is exactly
//!     what the Python reference computes.

use std::collections::HashMap;

use wgpu::util::DeviceExt;

// ---- kernel indices (order matches `PIPELINES`) ----
const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const ROPE: usize = 3;
const ATTN_SCORES: usize = 4;
const ATTN_SOFTMAX: usize = 5;
const ATTN_APPLY: usize = 6;
const ROUTER: usize = 7;
const SILU: usize = 8;
const SCALE_ADD: usize = 9;
const ADD2: usize = 10;
const CE_GRAD: usize = 11;
const CE_VALUE: usize = 12;
const MATMUL_DX: usize = 13;
const MATMUL_DW: usize = 14;
const RMS_INV: usize = 15;
const RMSNORM_DX: usize = 16;
const RMSNORM_DW: usize = 17;
const SILU_DA: usize = 18;
const SILU_DB: usize = 19;
const SCALE_ADD_DEXP: usize = 20;
const SCALE_ADD_DGATE: usize = 21;
const EXPERT_COUNTS: usize = 22;
const ROUTER_BWD: usize = 23;
const ROPE_BWD: usize = 24;
const ATTN_BWD_DSCORES: usize = 25;
const ATTN_BWD_DV: usize = 26;
const ATTN_BWD_DQ: usize = 27;
const ATTN_BWD_DK: usize = 28;
const EMB_BWD: usize = 29;
const ADAMW: usize = 30;

const PIPELINES: &[(&str, &str)] = &[
    ("embed", include_str!("shaders/embed.wgsl")),
    ("matmul", include_str!("shaders/matmul.wgsl")),
    ("rmsnorm", include_str!("shaders/rmsnorm.wgsl")),
    ("rope_train", include_str!("shaders/rope_train.wgsl")),
    ("attn_scores", include_str!("shaders/attn_scores.wgsl")),
    ("attn_softmax", include_str!("shaders/attn_softmax.wgsl")),
    ("attn_apply", include_str!("shaders/attn_apply.wgsl")),
    ("router_gate_train", include_str!("shaders/router_gate_train.wgsl")),
    ("silu_mul", include_str!("shaders/silu_mul.wgsl")),
    ("scale_add", include_str!("shaders/scale_add.wgsl")),
    ("add2", include_str!("shaders/add2.wgsl")),
    ("ce_grad", include_str!("shaders/ce_grad.wgsl")),
    ("ce_value", include_str!("shaders/ce_value.wgsl")),
    ("matmul_dx", include_str!("shaders/matmul_dx.wgsl")),
    ("matmul_dw", include_str!("shaders/matmul_dw.wgsl")),
    ("rms_inv", include_str!("shaders/rms_inv.wgsl")),
    ("rmsnorm_dx", include_str!("shaders/rmsnorm_dx.wgsl")),
    ("rmsnorm_dw", include_str!("shaders/rmsnorm_dw.wgsl")),
    ("silu_bwd_da", include_str!("shaders/silu_bwd_da.wgsl")),
    ("silu_bwd_db", include_str!("shaders/silu_bwd_db.wgsl")),
    ("scale_add_dexp", include_str!("shaders/scale_add_dexp.wgsl")),
    ("scale_add_dgate", include_str!("shaders/scale_add_dgate.wgsl")),
    ("expert_counts", include_str!("shaders/expert_counts.wgsl")),
    ("router_bwd", include_str!("shaders/router_bwd.wgsl")),
    ("rope_train_bwd", include_str!("shaders/rope_train_bwd.wgsl")),
    ("attn_bwd_dscores", include_str!("shaders/attn_bwd_dscores.wgsl")),
    ("attn_bwd_dv", include_str!("shaders/attn_bwd_dv.wgsl")),
    ("attn_bwd_dq", include_str!("shaders/attn_bwd_dq.wgsl")),
    ("attn_bwd_dk", include_str!("shaders/attn_bwd_dk.wgsl")),
    ("emb_bwd", include_str!("shaders/emb_bwd.wgsl")),
    ("adamw", include_str!("shaders/adamw.wgsl")),
];

#[derive(Clone)]
pub struct Config {
    pub vocab: u32,
    pub block_size: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_experts: u32,
    pub top_k: u32,
    pub d_ff: u32,
    pub aux_coef: f32,
    pub z_coef: f32,
}

impl Config {
    fn head_dim(&self) -> u32 {
        self.d_model / self.n_heads
    }
}

type Step = (usize, wgpu::BindGroup, u32);

/// Names + element counts of the unique (trainable) parameters.
fn param_list(c: &Config) -> Vec<(String, usize)> {
    let d = c.d_model as usize;
    let ff = c.d_ff as usize;
    let mut v = vec![("token_emb.weight".to_string(), c.vocab as usize * d)];
    for l in 0..c.n_layers {
        let p = |s: &str| format!("blocks.{l}.{s}");
        v.push((p("norm1.weight"), d));
        v.push((p("attn.qkv.weight"), 3 * d * d));
        v.push((p("attn.out.weight"), d * d));
        v.push((p("norm2.weight"), d));
        v.push((p("moe.router.weight"), c.n_experts as usize * d));
        for e in 0..c.n_experts {
            v.push((format!("blocks.{l}.moe.experts.{e}.w_gate.weight"), ff * d));
            v.push((format!("blocks.{l}.moe.experts.{e}.w_up.weight"), ff * d));
            v.push((format!("blocks.{l}.moe.experts.{e}.w_down.weight"), d * ff));
        }
    }
    v.push(("norm.weight".to_string(), d));
    v
}

struct LayerBufs {
    xn1: wgpu::Buffer,
    qkv: wgpu::Buffer,
    probs: wgpu::Buffer,
    attn_out: wgpu::Buffer,
    xmid: wgpu::Buffer,
    xn2: wgpu::Buffer,
    router_logits: wgpu::Buffer,
    router_probs: wgpu::Buffer,
    gate: wgpu::Buffer,
    gate_pre: Vec<wgpu::Buffer>,
    up: Vec<wgpu::Buffer>,
    h: Vec<wgpu::Buffer>,
    expert_out: Vec<wgpu::Buffer>,
    dxmid: wgpu::Buffer,
}

pub struct Trainer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    cfg: Config,
    b: u32,
    t: u32,

    weights: HashMap<String, wgpu::Buffer>,
    grads: HashMap<String, wgpu::Buffer>,
    adam_m: HashMap<String, wgpu::Buffer>,
    adam_v: HashMap<String, wgpu::Buffer>,
    params: Vec<(String, usize)>,

    tokens: wgpu::Buffer,
    targets: wgpu::Buffer,

    res: Vec<wgpu::Buffer>,  // residual stream, len n_layers+1 (res[0]=emb out, res[L]=x_final)
    dres: Vec<wgpu::Buffer>, // its gradient
    layers: Vec<LayerBufs>,
    xn_final: wgpu::Buffer,
    logits: wgpu::Buffer,

    // forward temporaries
    scores: wgpu::Buffer,
    proj: wgpu::Buffer,
    moe_acc: wgpu::Buffer,
    ce_buf: wgpu::Buffer,
    fe: wgpu::Buffer,

    // backward temporaries
    d_logits: wgpu::Buffer,
    d_xn: wgpu::Buffer,
    d_tmp: wgpu::Buffer,
    d_qkv: wgpu::Buffer,
    d_attn_out: wgpu::Buffer,
    d_scores: wgpu::Buffer,
    d_gate: wgpu::Buffer,
    d_router_logits: wgpu::Buffer,
    d_gate_pre: wgpu::Buffer,
    d_up: wgpu::Buffer,
    d_h: wgpu::Buffer,
    d_expert_out: wgpu::Buffer,
    inv: wgpu::Buffer,
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

impl Trainer {
    fn storage(device: &wgpu::Device, n: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn new(cfg: Config, b: u32, t: u32, init: &HashMap<String, Vec<f32>>) -> Trainer {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("no suitable GPU adapter found");
        eprintln!("adapter: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);

        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = 8;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("trainer"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .expect("request_device failed");

        let pipelines = PIPELINES
            .iter()
            .map(|(name, src)| {
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl((*src).into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            })
            .collect();

        let c = cfg.clone();
        let params = param_list(&c);

        // weights / grads / adam state
        let mut weights = HashMap::new();
        let mut grads = HashMap::new();
        let mut adam_m = HashMap::new();
        let mut adam_v = HashMap::new();
        for (name, numel) in &params {
            let data = init.get(name).unwrap_or_else(|| panic!("missing init weight {name}"));
            assert_eq!(data.len(), *numel, "size mismatch for {name}");
            weights.insert(
                name.clone(),
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                }),
            );
            grads.insert(name.clone(), Self::storage(&device, *numel as u64));
            adam_m.insert(name.clone(), Self::storage(&device, *numel as u64));
            adam_v.insert(name.clone(), Self::storage(&device, *numel as u64));
        }
        // zero adam state
        for name in weights.keys() {
            let z = vec![0u8; (params.iter().find(|(n, _)| n == name).unwrap().1) * 4];
            queue.write_buffer(&adam_m[name], 0, &z);
            queue.write_buffer(&adam_v[name], 0, &z);
        }

        let n = (b * t) as u64;
        let d = c.d_model as u64;
        let ff = c.d_ff as u64;
        let e = c.n_experts as u64;
        let bht2 = (b * c.n_heads * t * t) as u64;

        let tokens = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tokens"),
            size: n * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let targets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("targets"),
            size: n * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let st = |x: u64| Self::storage(&device, x);
        let mut res = Vec::new();
        let mut dres = Vec::new();
        for _ in 0..=c.n_layers {
            res.push(st(n * d));
            dres.push(st(n * d));
        }
        let mut layers = Vec::new();
        for _ in 0..c.n_layers {
            layers.push(LayerBufs {
                xn1: st(n * d),
                qkv: st(n * 3 * d),
                probs: st(bht2),
                attn_out: st(n * d),
                xmid: st(n * d),
                xn2: st(n * d),
                router_logits: st(n * e),
                router_probs: st(n * e),
                gate: st(n * e),
                gate_pre: (0..e).map(|_| st(n * ff)).collect(),
                up: (0..e).map(|_| st(n * ff)).collect(),
                h: (0..e).map(|_| st(n * ff)).collect(),
                expert_out: (0..e).map(|_| st(n * d)).collect(),
                dxmid: st(n * d),
            });
        }

        Trainer {
            cfg: c,
            b,
            t,
            params,
            weights,
            grads,
            adam_m,
            adam_v,
            tokens,
            targets,
            res,
            dres,
            layers,
            xn_final: st(n * d),
            logits: st(n * cfg.vocab as u64),
            scores: st(bht2),
            proj: st(n * d),
            moe_acc: st(n * d),
            ce_buf: st(n),
            fe: st(e),
            d_logits: st(n * cfg.vocab as u64),
            d_xn: st(n * d),
            d_tmp: st(n * d),
            d_qkv: st(n * 3 * d),
            d_attn_out: st(n * d),
            d_scores: st(bht2),
            d_gate: st(n * e),
            d_router_logits: st(n * e),
            d_gate_pre: st(n * ff),
            d_up: st(n * ff),
            d_h: st(n * ff),
            d_expert_out: st(n * d),
            inv: st(n),
            pipelines,
            device,
            queue,
        }
    }

    fn uniform(&self, data: &[u32]) -> wgpu::Buffer {
        let mut bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
        while bytes.len() % 16 != 0 {
            bytes.push(0);
        }
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    fn step(&self, kind: usize, bufs: &[&wgpu::Buffer], params: &[u32], threads: u32) -> Step {
        let ubuf = self.uniform(params);
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: ubuf.as_entire_binding(),
        }];
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: b.as_entire_binding(),
            });
        }
        let layout = self.pipelines[kind].get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        (kind, bg, ((threads + 63) / 64).max(1))
    }

    fn submit(&self, clears: &[&wgpu::Buffer], steps: &[Step]) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for c in clears {
            enc.clear_buffer(c, 0, None);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for (kind, bg, groups) in steps {
                pass.set_pipeline(&self.pipelines[*kind]);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(*groups, 1, 1);
            }
        }
        self.queue.submit(Some(enc.finish()));
    }

    fn w(&self, name: &str) -> &wgpu::Buffer {
        self.weights.get(name).unwrap_or_else(|| panic!("no weight {name}"))
    }
    fn g(&self, name: &str) -> &wgpu::Buffer {
        self.grads.get(name).unwrap()
    }

    fn read(&self, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, (n * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        out
    }

    pub fn set_batch(&self, x: &[u32], y: &[u32]) {
        self.queue.write_buffer(&self.tokens, 0, bytemuck::cast_slice(x));
        self.queue.write_buffer(&self.targets, 0, bytemuck::cast_slice(y));
    }

    /// Forward pass; caches all activations. Returns mean cross-entropy.
    pub fn forward(&self) -> f32 {
        let c = &self.cfg;
        let n = self.b * self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let e = c.n_experts;
        let hd = c.head_dim();
        let half = hd / 2;
        let mut s: Vec<Step> = Vec::new();

        // embedding -> res[0]
        s.push(self.step(EMBED, &[&self.tokens, self.w("token_emb.weight"), &self.res[0]], &[d, n], n * d));

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let pn = |name: &str| format!("blocks.{l}.{name}");
            // attention
            s.push(self.step(RMSNORM, &[&self.res[l], self.w(&pn("norm1.weight")), &lb.xn1], &[d, n], n));
            s.push(self.step(MATMUL, &[&lb.xn1, self.w(&pn("attn.qkv.weight")), &lb.qkv], &[n, d, 3 * d], n * 3 * d));
            s.push(self.step(ROPE, &[&lb.qkv], &[n, c.n_heads, hd, 3 * d, 0, self.t], n * c.n_heads * half));
            s.push(self.step(ROPE, &[&lb.qkv], &[n, c.n_heads, hd, 3 * d, d, self.t], n * c.n_heads * half));
            s.push(self.step(ATTN_SCORES, &[&lb.qkv, &self.scores], &[self.b, c.n_heads, self.t, hd, 3 * d, 0, d], self.b * c.n_heads * self.t * self.t));
            s.push(self.step(ATTN_SOFTMAX, &[&self.scores, &lb.probs], &[self.b, c.n_heads, self.t], self.b * c.n_heads * self.t));
            s.push(self.step(ATTN_APPLY, &[&lb.probs, &lb.qkv, &lb.attn_out], &[self.b, c.n_heads, self.t, hd, 3 * d, 2 * d, d], self.b * c.n_heads * self.t * hd));
            s.push(self.step(MATMUL, &[&lb.attn_out, self.w(&pn("attn.out.weight")), &self.proj], &[n, d, d], n * d));
            s.push(self.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));
            // moe
            s.push(self.step(RMSNORM, &[&lb.xmid, self.w(&pn("norm2.weight")), &lb.xn2], &[d, n], n));
            s.push(self.step(MATMUL, &[&lb.xn2, self.w(&pn("moe.router.weight")), &lb.router_logits], &[n, d, e], n * e));
            s.push(self.step(ROUTER, &[&lb.router_logits, &lb.gate, &lb.router_probs], &[n, e, c.top_k], n));
            for ei in 0..e as usize {
                let ep = |name: &str| format!("blocks.{l}.moe.experts.{ei}.{name}");
                s.push(self.step(MATMUL, &[&lb.xn2, self.w(&ep("w_gate.weight")), &lb.gate_pre[ei]], &[n, d, ff], n * ff));
                s.push(self.step(MATMUL, &[&lb.xn2, self.w(&ep("w_up.weight")), &lb.up[ei]], &[n, d, ff], n * ff));
                s.push(self.step(SILU, &[&lb.gate_pre[ei], &lb.up[ei], &lb.h[ei]], &[n * ff], n * ff));
                s.push(self.step(MATMUL, &[&lb.h[ei], self.w(&ep("w_down.weight")), &lb.expert_out[ei]], &[n, ff, d], n * d));
                let acc = if ei == 0 { 0 } else { 1 };
                s.push(self.step(SCALE_ADD, &[&lb.gate, &lb.expert_out[ei], &self.moe_acc], &[n, d, e, ei as u32, acc], n * d));
            }
            s.push(self.step(ADD2, &[&lb.xmid, &self.moe_acc, &self.res[l + 1]], &[n * d], n * d));
        }

        // final norm + tied lm_head
        s.push(self.step(RMSNORM, &[&self.res[c.n_layers as usize], self.w("norm.weight"), &self.xn_final], &[d, n], n));
        s.push(self.step(MATMUL, &[&self.xn_final, self.w("token_emb.weight"), &self.logits], &[n, d, c.vocab], n * c.vocab));
        s.push(self.step(CE_VALUE, &[&self.logits, &self.targets, &self.ce_buf], &[n, c.vocab], n));

        self.submit(&[], &s);
        let losses = self.read(&self.ce_buf, n as usize);
        losses.iter().sum::<f32>() / n as f32
    }

    /// Backward pass: zero grads, then accumulate every parameter gradient.
    pub fn backward(&self) {
        let c = &self.cfg;
        let n = self.b * self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let e = c.n_experts;
        let hd = c.head_dim();
        let half = hd / 2;
        let n_layers = c.n_layers as usize;
        let mut s: Vec<Step> = Vec::new();

        // ---- output: cross-entropy grad, lm_head, final norm ----
        s.push(self.step(CE_GRAD, &[&self.logits, &self.targets, &self.d_logits], &[n, c.vocab], n * c.vocab));
        // lm_head dW (tied -> grad_emb) and dX -> d_xn (=d_xn_final)
        s.push(self.step(MATMUL_DW, &[&self.d_logits, &self.xn_final, self.g("token_emb.weight")], &[n, d, c.vocab], c.vocab * d));
        s.push(self.step(MATMUL_DX, &[&self.d_logits, self.w("token_emb.weight"), &self.d_xn], &[n, d, c.vocab, 0], n * d));
        // final norm backward -> dres[L]
        s.push(self.step(RMS_INV, &[&self.res[n_layers], &self.inv], &[d, n], n));
        s.push(self.step(RMSNORM_DW, &[&self.d_xn, &self.res[n_layers], &self.inv, self.g("norm.weight")], &[d, n], d));
        s.push(self.step(RMSNORM_DX, &[&self.res[n_layers], self.w("norm.weight"), &self.d_xn, &self.dres[n_layers]], &[d, n], n));

        for l in (0..n_layers).rev() {
            let lb = &self.layers[l];
            let pn = |name: &str| format!("blocks.{l}.{name}");

            // ===== MoE backward (d_moe_acc = dres[l+1]) =====
            // Phase A: per-expert gate gradient
            for ei in 0..e as usize {
                s.push(self.step(SCALE_ADD_DGATE, &[&lb.expert_out[ei], &self.dres[l + 1], &self.d_gate], &[n, d, e, ei as u32], n));
            }
            // Phase B: router gradient -> d_xn (init), grad_Wrouter
            s.push(self.step(EXPERT_COUNTS, &[&lb.gate, &self.fe], &[n, e, c.top_k], e));
            s.push(self.step(ROUTER_BWD, &[&lb.router_logits, &lb.gate, &self.d_gate, &self.fe, &self.d_router_logits], &[n, e, c.top_k, 0, f(c.aux_coef), f(c.z_coef)], n));
            s.push(self.step(MATMUL_DW, &[&self.d_router_logits, &lb.xn2, self.g(&pn("moe.router.weight"))], &[n, d, e], e * d));
            s.push(self.step(MATMUL_DX, &[&self.d_router_logits, self.w(&pn("moe.router.weight")), &self.d_xn], &[n, d, e, 0], n * d));
            // Phase C: per-expert SwiGLU backward, accumulate into d_xn
            for ei in 0..e as usize {
                let ep = |name: &str| format!("blocks.{l}.moe.experts.{ei}.{name}");
                s.push(self.step(SCALE_ADD_DEXP, &[&lb.gate, &self.dres[l + 1], &self.d_expert_out], &[n, d, e, ei as u32], n * d));
                s.push(self.step(MATMUL_DW, &[&self.d_expert_out, &lb.h[ei], self.g(&ep("w_down.weight"))], &[n, ff, d], d * ff));
                s.push(self.step(MATMUL_DX, &[&self.d_expert_out, self.w(&ep("w_down.weight")), &self.d_h], &[n, ff, d, 0], n * ff));
                s.push(self.step(SILU_DA, &[&lb.gate_pre[ei], &lb.up[ei], &self.d_h, &self.d_gate_pre], &[n * ff], n * ff));
                s.push(self.step(SILU_DB, &[&lb.gate_pre[ei], &self.d_h, &self.d_up], &[n * ff], n * ff));
                s.push(self.step(MATMUL_DW, &[&self.d_up, &lb.xn2, self.g(&ep("w_up.weight"))], &[n, d, ff], ff * d));
                s.push(self.step(MATMUL_DX, &[&self.d_up, self.w(&ep("w_up.weight")), &self.d_xn], &[n, d, ff, 1], n * d));
                s.push(self.step(MATMUL_DW, &[&self.d_gate_pre, &lb.xn2, self.g(&ep("w_gate.weight"))], &[n, d, ff], ff * d));
                s.push(self.step(MATMUL_DX, &[&self.d_gate_pre, self.w(&ep("w_gate.weight")), &self.d_xn], &[n, d, ff, 1], n * d));
            }
            // norm2 backward -> d_tmp ; dxmid = dres[l+1] + d_tmp
            s.push(self.step(RMS_INV, &[&lb.xmid, &self.inv], &[d, n], n));
            s.push(self.step(RMSNORM_DW, &[&self.d_xn, &lb.xmid, &self.inv, self.g(&pn("norm2.weight"))], &[d, n], d));
            s.push(self.step(RMSNORM_DX, &[&lb.xmid, self.w(&pn("norm2.weight")), &self.d_xn, &self.d_tmp], &[d, n], n));
            s.push(self.step(ADD2, &[&self.dres[l + 1], &self.d_tmp, &lb.dxmid], &[n * d], n * d));

            // ===== attention backward (d_proj = dxmid) =====
            s.push(self.step(MATMUL_DW, &[&lb.dxmid, &lb.attn_out, self.g(&pn("attn.out.weight"))], &[n, d, d], d * d));
            s.push(self.step(MATMUL_DX, &[&lb.dxmid, self.w(&pn("attn.out.weight")), &self.d_attn_out], &[n, d, d, 0], n * d));
            s.push(self.step(ATTN_BWD_DSCORES, &[&self.d_attn_out, &lb.qkv, &lb.probs, &self.d_scores], &[self.b, c.n_heads, self.t, hd, 3 * d, 2 * d, d], self.b * c.n_heads * self.t));
            s.push(self.step(ATTN_BWD_DV, &[&lb.probs, &self.d_attn_out, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 2 * d, d], self.b * c.n_heads * self.t * hd));
            s.push(self.step(ATTN_BWD_DQ, &[&self.d_scores, &lb.qkv, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 0, d], self.b * c.n_heads * self.t * hd));
            s.push(self.step(ATTN_BWD_DK, &[&self.d_scores, &lb.qkv, &self.d_qkv], &[self.b, c.n_heads, self.t, hd, 3 * d, 0, d], self.b * c.n_heads * self.t * hd));
            // rope backward on q and k regions of d_qkv
            s.push(self.step(ROPE_BWD, &[&self.d_qkv], &[n, c.n_heads, hd, 3 * d, 0, self.t], n * c.n_heads * half));
            s.push(self.step(ROPE_BWD, &[&self.d_qkv], &[n, c.n_heads, hd, 3 * d, d, self.t], n * c.n_heads * half));
            // qkv matmul backward -> grad_Wqkv, d_xn (=d_xn1)
            s.push(self.step(MATMUL_DW, &[&self.d_qkv, &lb.xn1, self.g(&pn("attn.qkv.weight"))], &[n, d, 3 * d], 3 * d * d));
            s.push(self.step(MATMUL_DX, &[&self.d_qkv, self.w(&pn("attn.qkv.weight")), &self.d_xn], &[n, d, 3 * d, 0], n * d));
            // norm1 backward -> d_tmp ; dres[l] = dxmid + d_tmp
            s.push(self.step(RMS_INV, &[&self.res[l], &self.inv], &[d, n], n));
            s.push(self.step(RMSNORM_DW, &[&self.d_xn, &self.res[l], &self.inv, self.g(&pn("norm1.weight"))], &[d, n], d));
            s.push(self.step(RMSNORM_DX, &[&self.res[l], self.w(&pn("norm1.weight")), &self.d_xn, &self.d_tmp], &[d, n], n));
            s.push(self.step(ADD2, &[&lb.dxmid, &self.d_tmp, &self.dres[l]], &[n * d], n * d));
        }

        // embedding backward (accumulates onto grad_emb which holds lm_head grad)
        s.push(self.step(EMB_BWD, &[&self.tokens, &self.dres[0], self.g("token_emb.weight")], &[n, d, c.vocab], c.vocab * d));

        // zero every weight grad, then run the whole backward in one pass
        let clears: Vec<&wgpu::Buffer> = self.params.iter().map(|(name, _)| self.g(name)).collect();
        self.submit(&clears, &s);
    }

    /// One AdamW step. `t` is the (1-based) step index for bias correction.
    pub fn adamw_step(&self, t: u32, lr: f32, wd: f32, beta1: f32, beta2: f32, eps: f32) {
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);
        let mut s: Vec<Step> = Vec::new();
        for (name, numel) in &self.params {
            s.push(self.step(
                ADAMW,
                &[self.w(name), self.g(name), &self.adam_m[name], &self.adam_v[name]],
                &[*numel as u32, 0, f(lr), f(beta1), f(beta2), f(eps), f(wd), f(bc1), f(bc2)],
                *numel as u32,
            ));
        }
        self.submit(&[], &s);
    }

    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        let numel = self.params.iter().find(|(n, _)| n == name).unwrap().1;
        self.read(self.w(name), numel)
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        let numel = self.params.iter().find(|(n, _)| n == name).unwrap().1;
        self.read(self.g(name), numel)
    }
}

// ===========================================================================
// CLI: `validate <ref.bin>`  and  `train [flags]`
// ===========================================================================

fn cfg_from_json(c: &serde_json::Value) -> Config {
    let g = |k: &str| c[k].as_u64().unwrap_or_else(|| panic!("missing config.{k}")) as u32;
    let gf = |k: &str, d: f32| c[k].as_f64().map(|v| v as f32).unwrap_or(d);
    Config {
        vocab: g("vocab_size"),
        block_size: g("block_size"),
        n_layers: g("n_layers"),
        d_model: g("d_model"),
        n_heads: g("n_heads"),
        n_experts: g("n_experts"),
        top_k: g("top_k"),
        d_ff: g("d_ff"),
        aux_coef: gf("aux_loss_coef", 0.01),
        z_coef: gf("z_loss_coef", 1e-4),
    }
}

fn max_err(a: &[f32], b: &[f32]) -> (f32, f32) {
    // returns (max abs error, max relative error over entries with |b|>1e-3)
    let mut mae = 0.0f32;
    let mut mre = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let ae = (x - y).abs();
        mae = mae.max(ae);
        if y.abs() > 1e-3 {
            mre = mre.max(ae / y.abs());
        }
    }
    (mae, mre)
}

pub fn validate(path: &str) {
    let bytes = std::fs::read(path).expect("cannot read ref file");
    let jlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let json: serde_json::Value = serde_json::from_str(
        std::str::from_utf8(&bytes[8..8 + jlen]).unwrap(),
    )
    .unwrap();
    let data = &bytes[8 + jlen..];
    let cfg = cfg_from_json(&json["config"]);
    let bsz = json["B"].as_u64().unwrap() as u32;
    let t = json["T"].as_u64().unwrap() as u32;

    let read_tensor = |offset: usize, numel: usize| -> Vec<f32> {
        data[offset * 4..(offset + numel) * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };

    let mut init = HashMap::new();
    let mut grad = HashMap::new();
    let mut updated = HashMap::new();
    let mut batch_x = Vec::new();
    let mut batch_y = Vec::new();
    for t in json["tensors"].as_array().unwrap() {
        let name = t["name"].as_str().unwrap().to_string();
        let role = t["role"].as_str().unwrap();
        let vals = read_tensor(t["offset"].as_u64().unwrap() as usize, t["numel"].as_u64().unwrap() as usize);
        match role {
            "init" => { init.insert(name, vals); }
            "grad" => { grad.insert(name, vals); }
            "updated" => { updated.insert(name, vals); }
            "data" if name == "batch_x" => batch_x = vals,
            "data" if name == "batch_y" => batch_y = vals,
            _ => {}
        }
    }

    let opt = &json["opt"];
    let (lr, wd) = (opt["lr"].as_f64().unwrap() as f32, opt["weight_decay"].as_f64().unwrap() as f32);
    let (beta1, beta2, eps) = (
        opt["beta1"].as_f64().unwrap() as f32,
        opt["beta2"].as_f64().unwrap() as f32,
        opt["eps"].as_f64().unwrap() as f32,
    );

    let trainer = Trainer::new(cfg, bsz, t, &init);
    let xs: Vec<u32> = batch_x.iter().map(|v| *v as u32).collect();
    let ys: Vec<u32> = batch_y.iter().map(|v| *v as u32).collect();
    trainer.set_batch(&xs, &ys);

    let ce = trainer.forward();
    let total = json["losses"]["total"].as_f64().unwrap() as f32;
    let moe = json["losses"]["moe"].as_f64().unwrap() as f32;
    println!("loss: rust_ce={:.6}  py_ce(total-moe)={:.6}  (py total={:.6})", ce, total - moe, total);

    trainer.backward();
    println!("\n== gradient check (Rust vs PyTorch autograd) ==");
    let mut g_mae = 0.0f32;
    let mut g_mre = 0.0f32;
    let mut worst = String::new();
    for (name, _) in trainer.params.iter() {
        let r = trainer.read_grad(name);
        let p = &grad[name];
        let (mae, mre) = max_err(&r, p);
        if mae > g_mae { g_mae = mae; worst = name.clone(); }
        g_mre = g_mre.max(mre);
    }
    println!("max abs grad error = {:.3e} (worst: {})", g_mae, worst);
    println!("max rel grad error = {:.3e}", g_mre);

    trainer.adamw_step(1, lr, wd, beta1, beta2, eps);
    println!("\n== weight check after one AdamW step ==");
    let mut w_mae = 0.0f32;
    let mut w_mre = 0.0f32;
    for (name, _) in trainer.params.iter() {
        let r = trainer.read_weight(name);
        let (mae, mre) = max_err(&r, &updated[name]);
        w_mae = w_mae.max(mae);
        w_mre = w_mre.max(mre);
    }
    println!("max abs weight error = {:.3e}", w_mae);
    println!("max rel weight error = {:.3e}", w_mre);

    let ok = g_mae < 2e-3 && w_mae < 2e-4;
    println!("\n{}", if ok { "VALIDATION PASSED" } else { "VALIDATION FAILED" });
}

// ---- toy corpus + init for from-scratch training (Rust side) ----

fn xorshift(s: &mut u64) -> u64 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}
fn randf(s: &mut u64) -> f32 {
    (xorshift(s) >> 40) as f32 / (1u64 << 24) as f32
}
fn randn(s: &mut u64) -> f32 {
    // Box-Muller
    let u1 = randf(s).max(1e-7);
    let u2 = randf(s);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// The toy corpus and its substitution table (the rule's ground truth).
/// `data[i] = (data[i-2] + table[data[i-1]]) % vocab`, with a reset every 257th.
pub fn corpus_and_table(n: usize, vocab: u32, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let mut s = seed.max(1);
    let mut table: Vec<u32> = (0..vocab).collect();
    for i in (1..vocab as usize).rev() {
        let j = (xorshift(&mut s) % (i as u64 + 1)) as usize;
        table.swap(i, j);
    }
    let mut d = vec![0u32; n];
    d[0] = (xorshift(&mut s) % vocab as u64) as u32;
    d[1] = (xorshift(&mut s) % vocab as u64) as u32;
    for i in 2..n {
        if i % 257 == 0 {
            d[i] = (xorshift(&mut s) % vocab as u64) as u32;
        } else {
            d[i] = (d[i - 2] + table[d[i - 1] as usize]) % vocab;
        }
    }
    (d, table)
}

fn make_corpus(n: usize, vocab: u32, seed: u64) -> Vec<u32> {
    corpus_and_table(n, vocab, seed).0
}

/// A reset-free orbit of the same rule starting from (s0, s1) — used to test
/// generalisation to a never-seen trajectory.
pub fn orbit(table: &[u32], vocab: u32, n: usize, s0: u32, s1: u32) -> Vec<u32> {
    let mut d = vec![s0 % vocab, s1 % vocab];
    for i in 2..n {
        d.push((d[i - 2] + table[d[i - 1] as usize]) % vocab);
    }
    d
}

fn init_weights(cfg: &Config, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut s = seed.max(1);
    let mut map = HashMap::new();
    for (name, numel) in param_list(cfg) {
        let vals: Vec<f32> = if name.contains("norm") {
            vec![1.0; numel] // RMSNorm gain initialised to ones
        } else {
            (0..numel).map(|_| 0.02 * randn(&mut s)).collect()
        };
        map.insert(name, vals);
    }
    map
}

pub struct TrainArgs {
    pub steps: u32,
    pub b: u32,
    pub t: u32,
    pub lr: f32,
    pub wd: f32,
    pub seed: u64,
    pub out: String,
}

pub fn train(args: TrainArgs) {
    let cfg = Config {
        vocab: 64, block_size: 64, n_layers: 2, d_model: 64, n_heads: 4,
        n_experts: 4, top_k: 2, d_ff: 128, aux_coef: 0.01, z_coef: 1e-4,
    };
    assert!(args.t <= cfg.block_size, "T must be <= block_size");

    let corpus = make_corpus(20_000, cfg.vocab, 123);
    let init = init_weights(&cfg, args.seed);
    let trainer = Trainer::new(cfg.clone(), args.b, args.t, &init);

    let mut rng = args.seed.max(1) ^ 0x9E3779B97F4A7C15;
    let tt = args.t as usize;
    for step in 1..=args.steps {
        // sample B random windows
        let mut xs = Vec::with_capacity((args.b * args.t) as usize);
        let mut ys = Vec::with_capacity((args.b * args.t) as usize);
        for _ in 0..args.b {
            let start = (xorshift(&mut rng) as usize) % (corpus.len() - tt - 1);
            xs.extend_from_slice(&corpus[start..start + tt]);
            ys.extend_from_slice(&corpus[start + 1..start + 1 + tt]);
        }
        trainer.set_batch(&xs, &ys);
        let loss = trainer.forward();
        trainer.backward();
        trainer.adamw_step(step, args.lr, args.wd, 0.9, 0.95, 1e-8);
        if step == 1 || step % 50 == 0 || step == args.steps {
            println!("step {:5} | loss {:.4}", step, loss);
        }
    }

    save_weights(&trainer, &cfg, &args.out);
    println!("saved {}", args.out);
}

fn save_weights(trainer: &Trainer, cfg: &Config, path: &str) {
    use std::io::Write;
    let mut tensors = Vec::new();
    let mut blob: Vec<f32> = Vec::new();
    let add = |name: &str, shape: Vec<u64>, data: &[f32], tensors: &mut Vec<serde_json::Value>, blob: &mut Vec<f32>| {
        tensors.push(serde_json::json!({
            "name": name, "shape": shape, "offset": blob.len(), "numel": data.len()
        }));
        blob.extend_from_slice(data);
    };
    let d = cfg.d_model as u64;
    for (name, _) in trainer.params.iter() {
        let data = trainer.read_weight(name);
        add(name, vec![data.len() as u64], &data, &mut tensors, &mut blob);
    }
    // tied head expected by the inference loader
    let emb = trainer.read_weight("token_emb.weight");
    add("lm_head.weight", vec![cfg.vocab as u64, d], &emb, &mut tensors, &mut blob);

    let header = serde_json::json!({
        "config": {
            "vocab_size": cfg.vocab, "block_size": cfg.block_size, "n_layers": cfg.n_layers,
            "d_model": cfg.d_model, "n_heads": cfg.n_heads, "n_experts": cfg.n_experts,
            "top_k": cfg.top_k, "d_ff": cfg.d_ff
        },
        "tensors": tensors
    });
    let hbytes = serde_json::to_vec(&header).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(hbytes.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&hbytes).unwrap();
    for v in &blob {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}
