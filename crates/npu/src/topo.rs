// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The** ONNX graph-emission base every topology builder composes.
//!
//! Eight `*_topology.rs` files each carried a copy-pasted mini-DSL (`tmp`,
//! `node`, `unary`, `mul_t`, …) plus their own emitters for the *same math*:
//! `rmsnorm` was emitted in seven places, `layernorm` in four. Per the
//! workspace's one-implementation rule (AGENTS.md), the DSL and the shared
//! emitters live here once; a model's builder embeds a [`TopoBase`] (via
//! `Deref`, so call sites read exactly as before) and keeps only its
//! model-specific graph logic.
//!
//! Emission conventions the shared code preserves — these keep the generated
//! graphs structurally identical to what the per-model copies produced:
//! * temporaries are `<tag>_<counter>` with one counter per builder;
//! * initializers are deduplicated by name (`f32`/`i64` are idempotent — the
//!   strictest of the previous per-file variants, and safe for all);
//! * `rmsnorm`/`layernorm` take the epsilon **initializer name**, so each model
//!   keeps registering its own `c_eps` exactly as it always has.

use onnx::{GraphBuilder, Node};

pub use crate::qwen_topology::Quant;
use crate::topology::WeightSource;

/// The shared graph-builder core: the wrapped graph, the temp-name counter and
/// the emission DSL. Model builders hold one and `Deref` to it.
pub(crate) struct TopoBase<'a> {
    pub g: &'a mut GraphBuilder,
    pub n: usize,
}

impl<'a> TopoBase<'a> {
    pub fn new(g: &'a mut GraphBuilder) -> TopoBase<'a> {
        TopoBase { g, n: 0 }
    }

    /// Fresh unique tensor name `<tag>_<counter>`.
    pub fn tmp(&mut self, tag: &str) -> String {
        self.n += 1;
        format!("{tag}_{}", self.n)
    }

    /// True when an initializer of this name is already registered.
    pub fn has(&self, name: &str) -> bool {
        self.g.graph().initializers.iter().any(|t| t.name == name)
    }

    /// Register an f32 initializer, idempotently.
    pub fn f32(&mut self, name: &str, dims: &[i64], data: Vec<f32>) {
        if !self.has(name) {
            self.g.init_f32(name, dims, data);
        }
    }

    /// Register an i64 initializer, idempotently.
    pub fn i64(&mut self, name: &str, dims: &[i64], data: Vec<i64>) {
        if !self.has(name) {
            self.g.init_i64(name, dims, data);
        }
    }

    /// Add a plain node `op(ins) -> out`.
    pub fn node(&mut self, op: &str, ins: &[&str], out: &str) {
        self.g.add(Node::new(op, ins, &[out]));
    }

    pub fn unary(&mut self, op: &str, x: &str) -> String {
        let o = self.tmp(&op.to_lowercase());
        self.node(op, &[x], &o);
        o
    }

    /// `Mul` with an initializer/const second operand (by name).
    pub fn mul(&mut self, x: &str, c: &str) -> String {
        let o = self.tmp("mul");
        self.node("Mul", &[x, c], &o);
        o
    }

    /// `Add` with an initializer/const second operand (by name).
    pub fn add(&mut self, x: &str, c: &str) -> String {
        let o = self.tmp("add");
        self.node("Add", &[x, c], &o);
        o
    }

    /// Tensor-tensor `Mul`.
    pub fn mul_t(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("mul");
        self.node("Mul", &[a, b], &o);
        o
    }

    /// Tensor-tensor `Add` (residual naming, matching the previous copies).
    pub fn add_t(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("res");
        self.node("Add", &[a, b], &o);
        o
    }

    /// Tensor-tensor `Sub`.
    pub fn sub_t(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("sub");
        self.node("Sub", &[a, b], &o);
        o
    }

    pub fn matmul(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("mm");
        self.node("MatMul", &[a, b], &o);
        o
    }

    pub fn reshape(&mut self, x: &str, shape: &str) -> String {
        let o = self.tmp("rs");
        self.node("Reshape", &[x, shape], &o);
        o
    }

    pub fn transpose(&mut self, x: &str, perm: &[i64]) -> String {
        let o = self.tmp("tr");
        self.g.add(Node::new("Transpose", &[x], &[&o]).attr_ints("perm", perm));
        o
    }

    pub fn softmax(&mut self, x: &str, axis: i64) -> String {
        let o = self.tmp("sm");
        self.g.add(Node::new("Softmax", &[x], &[&o]).attr_int("axis", axis));
        o
    }

    pub fn gather(&mut self, data: &str, idx: &str, axis: i64, tag: &str) -> String {
        let o = self.tmp(tag);
        self.g.add(Node::new("Gather", &[data, idx], &[&o]).attr_int("axis", axis));
        o
    }

    /// `Slice(x, lo, hi, axis)` by initializer names.
    pub fn slice(&mut self, x: &str, lo: &str, hi: &str, ax: &str) -> String {
        let o = self.tmp("sl");
        self.g.add(Node::new("Slice", &[x, lo, hi, ax], &[&o]));
        o
    }

    /// `Concat([a, b], axis)`.
    pub fn concat2(&mut self, a: &str, b: &str, axis: i64) -> String {
        let o = self.tmp("cat");
        self.g.add(Node::new("Concat", &[a, b], &[&o]).attr_int("axis", axis));
        o
    }

    /// SiLU / swish: `x * sigmoid(x)`.
    pub fn silu(&mut self, x: &str) -> String {
        let s = self.unary("Sigmoid", x);
        self.mul_t(x, &s)
    }

    // ---- shared math emitters ----------------------------------------------
    //
    // One emission of each normalisation, replacing seven per-model copies of
    // rmsnorm and four of layernorm. `eps_name` is the model's own epsilon
    // initializer (conventionally "c_eps"), registered by the model as before.

    /// RMSNorm over the last axis, writing to a fresh temp:
    /// `x * rsqrt(mean(x², -1) + eps) * gain`. The gain initializer is
    /// registered idempotently under `gain_name`.
    pub fn rmsnorm(
        &mut self,
        x: &str,
        gain_name: &str,
        gain: Vec<f32>,
        dim: usize,
        eps_name: &str,
    ) -> String {
        let o = self.tmp("rmsn");
        self.rmsnorm_to(x, gain_name, gain, dim, eps_name, &o);
        o
    }

    /// [`Self::rmsnorm`] writing its final scaled tensor to `out_name`.
    pub fn rmsnorm_to(
        &mut self,
        x: &str,
        gain_name: &str,
        gain: Vec<f32>,
        dim: usize,
        eps_name: &str,
        out_name: &str,
    ) {
        self.f32(gain_name, &[dim as i64], gain);
        let sq = self.mul_t(x, x);
        let ms = {
            let o = self.tmp("rms_mean");
            self.g.add(
                Node::new("ReduceMean", &[&sq], &[&o]).attr_ints("axes", &[-1]).attr_int("keepdims", 1),
            );
            o
        };
        let mse = self.add(&ms, eps_name);
        let rms = self.unary("Sqrt", &mse);
        let nrm = {
            let o = self.tmp("rms_div");
            self.node("Div", &[x, &rms], &o);
            o
        };
        self.node("Mul", &[&nrm, gain_name], out_name);
    }

    /// LayerNorm over the last axis:
    /// `(x - mean) * rsqrt(var + eps) * gamma + beta`. Gamma/beta initializers
    /// are registered idempotently under their names.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm(
        &mut self,
        x: &str,
        gamma_name: &str,
        gamma: Vec<f32>,
        beta_name: &str,
        beta: Vec<f32>,
        dim: usize,
        eps_name: &str,
    ) -> String {
        self.f32(gamma_name, &[dim as i64], gamma);
        self.f32(beta_name, &[dim as i64], beta);
        let mean = {
            let o = self.tmp("ln_mean");
            self.g.add(
                Node::new("ReduceMean", &[x], &[&o]).attr_ints("axes", &[-1]).attr_int("keepdims", 1),
            );
            o
        };
        let cent = {
            let o = self.tmp("ln_sub");
            self.node("Sub", &[x, &mean], &o);
            o
        };
        let sq = self.mul_t(&cent, &cent);
        let var = {
            let o = self.tmp("ln_var");
            self.g.add(
                Node::new("ReduceMean", &[&sq], &[&o]).attr_ints("axes", &[-1]).attr_int("keepdims", 1),
            );
            o
        };
        let veps = self.add(&var, eps_name);
        let sd = self.unary("Sqrt", &veps);
        let nrm = {
            let o = self.tmp("ln_div");
            self.node("Div", &[&cent, &sd], &o);
            o
        };
        let scaled = self.mul(&nrm, gamma_name);
        self.add(&scaled, beta_name)
    }
}

/// Bias-free linear `y = x @ Wᵀ` with per-output-channel weight-only
/// quantization — the ONE emitter every topology's `linear` delegates to
/// (the per-model copies it replaces drifted exactly the way AGENTS.md's
/// one-implementation rule warns about). Brain weights are `[out,in]`
/// row-major; ONNX wants `[in,out]`, transposed here once. `Quant::F32`
/// stores the fp32 initializer; `Int8`/`Int4` store symmetric integers +
/// scales and dequantise in-graph (`DequantizeLinear` → MatMul).
#[allow(clippy::too_many_arguments)]
pub fn linear_quant(
    b: &mut TopoBase,
    x: &str,
    name: &str,
    winit: &str,
    w: &dyn WeightSource,
    out: usize,
    inp: usize,
    quant: Quant,
    y: &str,
) {
    let transpose = |m: &[f32]| -> Vec<f32> {
        let mut t = vec![0f32; m.len()];
        for r in 0..out {
            for c in 0..inp {
                t[c * out + r] = m[r * inp + c];
            }
        }
        t
    };
    let (q4, qmax) = match quant {
        Quant::F32 => {
            if !b.has(winit) {
                let wt = transpose(&w.get(name));
                b.f32(winit, &[inp as i64, out as i64], wt);
            }
            b.node("MatMul", &[x, winit], y);
            return;
        }
        Quant::Int8 => (false, 127.0f32),
        Quant::Int4 => (true, 7.0f32),
    };
    let wq = format!("{winit}.q");
    if !b.has(&wq) {
        let wt = transpose(&w.get(name));
        let mut scales = vec![0f32; out];
        let mut q = vec![0i8; inp * out];
        for o in 0..out {
            let mut mx = 0f32;
            for i in 0..inp {
                mx = mx.max(wt[i * out + o].abs());
            }
            let s = if mx > 0.0 { mx / qmax } else { 1.0 };
            scales[o] = s;
            for i in 0..inp {
                q[i * out + o] = (wt[i * out + o] / s).round().clamp(-qmax, qmax) as i8;
            }
        }
        let zp = format!("{winit}.zp");
        if q4 {
            b.g.init_i4(&wq, &[inp as i64, out as i64], q);
            b.g.init_i4(&zp, &[out as i64], vec![0i8; out]);
        } else {
            b.g.init_i8(&wq, &[inp as i64, out as i64], q);
            b.g.init_i8(&zp, &[out as i64], vec![0i8; out]);
        }
        b.f32(&format!("{winit}.s"), &[out as i64], scales);
        b.g.add(
            Node::new("DequantizeLinear", &[&wq, &format!("{winit}.s"), &zp], &[winit]).attr_int("axis", 1),
        );
    }
    b.node("MatMul", &[x, winit], y);
}
