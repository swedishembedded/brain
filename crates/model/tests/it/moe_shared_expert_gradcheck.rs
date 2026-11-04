// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host central-finite-difference gradcheck for `model::moe::shared_expert_bwd`
//! - BOTH arms, on a tiny shape.
//!
//! Why this file and not `check_glm` alone: `check_glm` exercises only the
//! UNWEIGHTED (`shared_gate_w: None`) arm, because GLM-5.2/DeepSeek-V3 has no
//! sigmoid shared-expert gate. The gated (`Some`) arm - Qwen3-Omni's Talker and
//! Qwen3.5-35B-A3B - reaches `shared_expert_bwd` through no model-level
//! gradcheck at all today, so `scale_row`-as-its-own-backward, `row_dot`'s
//! row-scale gradient and `sigmoid_bwd` would be UNGATED without this.
//!
//! The oracle is this file: central differences of the same cached forward
//! (`shared_expert_fwd`), the shape `moe_block_gradcheck.rs` already
//! established.

use std::collections::HashMap;

use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu};
use gradcheck::{elementwise_check, CheckModel};
use model::moe::{
    shared_expert_bwd, shared_expert_fwd, SharedExpertBwdIds, SharedExpertBwdScratch, SharedExpertGrads, SharedExpertIds, SharedExpertScratch,
};

const PIPES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("matmul_dx", kernels::MATMUL_DX),
    ("matmul_dw", kernels::MATMUL_DW),
    ("silu_mul", kernels::SILU_MUL),
    ("silu_bwd_da", kernels::SILU_BWD_DA),
    ("silu_bwd_db", kernels::SILU_BWD_DB),
    ("sigmoid", kernels::SIGMOID),
    ("sigmoid_bwd", kernels::SIGMOID_BWD),
    ("scale_row", kernels::SCALE_ROW),
    ("row_dot", kernels::ROW_DOT),
    ("add2", kernels::ADD2),
];

const ROWS: u32 = 4;
const D: u32 = 6;
const FF: u32 = 5;

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

struct SharedExpertCheck {
    g: Gpu,
    gated: bool,
    fwd_ids: SharedExpertIds,
    bwd_ids: SharedExpertBwdIds,
    // weights + their grads
    gate_w: DeviceBuffer,
    up_w: DeviceBuffer,
    down_w: DeviceBuffer,
    shared_gate_w: DeviceBuffer,
    gate_g: DeviceBuffer,
    up_g: DeviceBuffer,
    down_g: DeviceBuffer,
    shared_gate_g: DeviceBuffer,
    sizes: HashMap<String, usize>,
    // activations
    x: DeviceBuffer,
    acc: DeviceBuffer,
    out: DeviceBuffer,
    s_gate_pre: DeviceBuffer,
    s_up: DeviceBuffer,
    s_h: DeviceBuffer,
    s_mlp_out: DeviceBuffer,
    s_gate_logits: DeviceBuffer,
    s_gate_scalar: DeviceBuffer,
    s_scaled: DeviceBuffer,
    // backward scratch
    d_out: DeviceBuffer, // == wloss, constant: loss = sum(out .* wloss)
    wloss: Vec<f32>,
    d_h: DeviceBuffer,
    d_gate_pre: DeviceBuffer,
    d_up: DeviceBuffer,
    d_mlp_out: DeviceBuffer,
    d_gate_scalar: DeviceBuffer,
    d_gate_logits: DeviceBuffer,
    d_x: DeviceBuffer,
}

impl SharedExpertCheck {
    fn new(gated: bool, seed: u64) -> SharedExpertCheck {
        let g = Gpu::new_cpu(PIPES);
        let mut r = Lcg::new(seed);
        let st = |n: u32| g.storage(n as u64);
        let init = |v: &[f32]| g.storage_init("w", v);

        let mut sizes = HashMap::new();
        sizes.insert("gate_w".to_string(), (FF * D) as usize);
        sizes.insert("up_w".to_string(), (FF * D) as usize);
        sizes.insert("down_w".to_string(), (D * FF) as usize);
        if gated {
            sizes.insert("shared_gate_w".to_string(), D as usize);
        }

        SharedExpertCheck {
            gated,
            fwd_ids: SharedExpertIds {
                matmul: idx(&g, "matmul"),
                silu_mul: idx(&g, "silu_mul"),
                sigmoid: idx(&g, "sigmoid"),
                scale_row: idx(&g, "scale_row"),
                add2: idx(&g, "add2"),
            },
            bwd_ids: SharedExpertBwdIds {
                linear_dx: idx(&g, "matmul_dx"),
                linear_dw: idx(&g, "matmul_dw"),
                silu_da: idx(&g, "silu_bwd_da"),
                silu_db: idx(&g, "silu_bwd_db"),
                scale_row: idx(&g, "scale_row"),
                row_dot: idx(&g, "row_dot"),
                sigmoid_bwd: idx(&g, "sigmoid_bwd"),
            },
            x: init(&r.vec_scaled((ROWS * D) as usize, 0.7)),
            gate_w: init(&r.vec_scaled((FF * D) as usize, 0.5)),
            up_w: init(&r.vec_scaled((FF * D) as usize, 0.5)),
            down_w: init(&r.vec_scaled((D * FF) as usize, 0.5)),
            shared_gate_w: init(&r.vec_scaled(D as usize, 0.5)),
            acc: init(&r.vec_scaled((ROWS * D) as usize, 0.4)),
            wloss: r.vec_scaled((ROWS * D) as usize, 0.6),
            gate_g: st(FF * D),
            up_g: st(FF * D),
            down_g: st(D * FF),
            shared_gate_g: st(D),
            out: st(ROWS * D),
            s_gate_pre: st(ROWS * FF),
            s_up: st(ROWS * FF),
            s_h: st(ROWS * FF),
            s_mlp_out: st(ROWS * D),
            s_gate_logits: st(ROWS),
            s_gate_scalar: st(ROWS),
            s_scaled: st(ROWS * D),
            d_out: st(ROWS * D),
            d_h: st(ROWS * FF),
            d_gate_pre: st(ROWS * FF),
            d_up: st(ROWS * FF),
            d_mlp_out: st(ROWS * D),
            d_gate_scalar: st(ROWS),
            d_gate_logits: st(ROWS),
            d_x: st(ROWS * D),
            sizes,
            g,
        }
    }

    fn scratch(&self) -> SharedExpertScratch<'_> {
        SharedExpertScratch {
            gate_pre: &self.s_gate_pre,
            up: &self.s_up,
            h: &self.s_h,
            mlp_out: &self.s_mlp_out,
            gate_logits: &self.s_gate_logits,
            gate_scalar: &self.s_gate_scalar,
            scaled: &self.s_scaled,
        }
    }

    fn weight_buf(&self, name: &str) -> &DeviceBuffer {
        match name {
            "gate_w" => &self.gate_w,
            "up_w" => &self.up_w,
            "down_w" => &self.down_w,
            "shared_gate_w" => &self.shared_gate_w,
            _ => unreachable!("unknown param {name}"),
        }
    }

    fn grad_buf(&self, name: &str) -> &DeviceBuffer {
        match name {
            "gate_w" => &self.gate_g,
            "up_w" => &self.up_g,
            "down_w" => &self.down_g,
            "shared_gate_w" => &self.shared_gate_g,
            _ => unreachable!("unknown param {name}"),
        }
    }
}

impl CheckModel for SharedExpertCheck {
    fn param_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.sizes.keys().cloned().collect();
        v.sort();
        v
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.g.read(self.weight_buf(name), self.sizes[name])
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        self.g.write_f32(self.weight_buf(name), data);
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.g.read(self.grad_buf(name), self.sizes[name])
    }

    fn loss(&self) -> f32 {
        let steps = shared_expert_fwd(
            &self.g,
            &self.fwd_ids,
            ROWS,
            D,
            FF,
            &self.x,
            &self.gate_w,
            &self.up_w,
            &self.down_w,
            self.gated.then_some(&self.shared_gate_w),
            &self.scratch(),
            &self.acc,
            &self.out,
        );
        self.g.submit(&[], &steps);
        let out = self.g.read(&self.out, (ROWS * D) as usize);
        // loss = sum(out .* wloss)  =>  d(loss)/d(out) = wloss, a constant.
        out.iter().zip(&self.wloss).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() as f32
    }

    fn zero_grads(&self) {
        for name in self.param_names() {
            self.g.write_f32(self.grad_buf(&name), &vec![0.0; self.sizes[&name]]);
        }
    }

    fn backward(&self) {
        // Forward first: `shared_expert_bwd` reads the SAVED activations.
        self.loss();
        self.g.write_f32(&self.d_out, &self.wloss);
        let steps = shared_expert_bwd(
            &self.g,
            &self.bwd_ids,
            ROWS,
            D,
            FF,
            &self.x,
            &self.gate_w,
            &self.up_w,
            &self.down_w,
            self.gated.then_some(&self.shared_gate_w),
            &SharedExpertGrads {
                gate_w: Some(&self.gate_g),
                up_w: Some(&self.up_g),
                down_w: Some(&self.down_g),
                shared_gate_w: self.gated.then_some(&self.shared_gate_g),
            },
            &self.scratch().acts(),
            &SharedExpertBwdScratch {
                d_h: &self.d_h,
                d_gate_pre: &self.d_gate_pre,
                d_up: &self.d_up,
                d_mlp_out: Some(&self.d_mlp_out),
                d_gate_scalar: Some(&self.d_gate_scalar),
                d_gate_logits: Some(&self.d_gate_logits),
            },
            &self.d_out,
            &self.d_x,
            false, // nothing has written d_x yet
        );
        self.g.submit(&[], &steps);
    }
}

fn run_case(gated: bool, seed: u64) {
    let m = SharedExpertCheck::new(gated, seed);
    let mut checked = 0usize;
    let mut worst_rel = 0.0f32;
    for name in m.param_names() {
        let report = elementwise_check(&m, &name, 5e-3);
        for c in &report.checks {
            worst_rel = worst_rel.max(c.rel_err);
            assert!(
                c.rel_err < 2e-2 || c.abs_err < 1e-4,
                "{}: analytic {} vs numeric {} (rel {:.4}, abs {:.6}) [gated={gated}]",
                c.param,
                c.analytic,
                c.numeric,
                c.rel_err,
                c.abs_err,
            );
            // A silently-dead gradient is the failure this composition is most
            // exposed to: drop `row_dot` and `shared_gate_w`'s grad stays 0.0
            // while the numeric derivative is clearly not.
            assert!(
                !(c.analytic == 0.0 && c.numeric.abs() > 1e-4),
                "{}: analytic is exactly 0 but numeric is {} -- a missing backward kernel",
                c.param,
                c.numeric
            );
        }
        checked += report.checks.len();
    }
    assert!(checked > 20, "too few gradients compared ({checked})");
    eprintln!("shared_expert_bwd(gated={gated}): {checked} grads checked, worst rel_err {worst_rel:.5}");
}

/// The arm `crates/glm` uses (and now dispatches through this function).
#[test]
fn shared_expert_bwd_unweighted_matches_finite_differences() {
    run_case(false, 0x5E01);
}

/// The arm Qwen3-Omni's Talker / Qwen3.5-35B-A3B use - `scale_row` +
/// `row_dot` + `sigmoid_bwd` on top of the same dense SwiGLU backward.
#[test]
fn shared_expert_bwd_sigmoid_gated_matches_finite_differences() {
    run_case(true, 0x5E02);
}
