// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The pre-LN residual MLP sublayer shared by DaViT's spatial and channel
//! blocks: `x = x + fc2(gelu(fc1(LN(x))))`, hidden width `mlp_ratio * dim`
//! (Florence-2-base: 4x, `Mlp`/`PreNorm` in the reference).

use gpu_core::{DeviceBuffer, Gpu, Step};

pub struct MlpKernelIds {
    pub layernorm: usize,
    pub matmul_rows: usize,
    pub bias_add: usize,
    pub gelu_erf: usize,
    pub add2: usize,
}

impl MlpKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> MlpKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        MlpKernelIds { layernorm: k("layernorm"), matmul_rows: k("matmul_rows"), bias_add: k("bias_add"), gelu_erf: k("gelu_erf"), add2: k("add2") }
    }
}

pub struct Mlp {
    dim: u32,
    hidden: u32,
    eps: f32,
    prefix: String,
    normed: DeviceBuffer,
    h: DeviceBuffer,
    h_act: DeviceBuffer,
    fc2_out: DeviceBuffer,
    out: DeviceBuffer,
}

impl Mlp {
    pub fn new(gpu: &Gpu, prefix: &str, dim: u32, hidden: u32, rows: u32, eps: f32) -> Mlp {
        Mlp {
            dim,
            hidden,
            eps,
            prefix: prefix.to_string(),
            normed: gpu.storage((rows * dim) as u64),
            h: gpu.storage((rows * hidden) as u64),
            h_act: gpu.storage((rows * hidden) as u64),
            fc2_out: gpu.storage((rows * dim) as u64),
            out: gpu.storage((rows * dim) as u64),
        }
    }

    pub fn forward(&self, gpu: &Gpu, k: &MlpKernelIds, ps: &paramstore::ParamStore, x_in: &DeviceBuffer, rows: u32) -> &DeviceBuffer {
        // Same PreNorm sibling-not-nested layout as window_attn.rs
        // (`ffn.norm.*` vs `ffn.fn.net.{fc1,fc2}.*` - the extra `.net.` is
        // `Mlp`'s own `nn.Sequential(OrderedDict(...))` wrapper), confirmed
        // against the real checkpoint's tensor names.
        let ln = model::block::LayerNormIds::resolve_fwd(gpu, k.layernorm);
        let norm_w = ps.w(&format!("{}.norm.weight", self.prefix));
        let norm_b = ps.w(&format!("{}.norm.bias", self.prefix));
        let fc1_w = ps.w(&format!("{}.fn.net.fc1.weight", self.prefix));
        let fc1_b = ps.w(&format!("{}.fn.net.fc1.bias", self.prefix));
        let fc2_w = ps.w(&format!("{}.fn.net.fc2.weight", self.prefix));
        let fc2_b = ps.w(&format!("{}.fn.net.fc2.bias", self.prefix));

        let s: Vec<Step> = vec![
            model::block::layernorm_fwd(gpu, &ln, x_in, norm_w, norm_b, &self.normed, self.dim, rows, self.eps),
            gpu.step(k.matmul_rows, &[&self.normed, fc1_w, &self.h], &[rows, self.dim, self.hidden], rows.div_ceil(8) * self.hidden),
            gpu.step(k.bias_add, &[&self.h, fc1_b], &[rows, self.hidden], rows * self.hidden),
            gpu.step(k.gelu_erf, &[&self.h, &self.h_act], &[rows * self.hidden], rows * self.hidden),
            gpu.step(k.matmul_rows, &[&self.h_act, fc2_w, &self.fc2_out], &[rows, self.hidden, self.dim], rows.div_ceil(8) * self.dim),
            gpu.step(k.bias_add, &[&self.fc2_out, fc2_b], &[rows, self.dim], rows * self.dim),
            gpu.step(k.add2, &[x_in, &self.fc2_out, &self.out], &[rows * self.dim], rows * self.dim),
        ];
        gpu.submit(&[], &s);
        &self.out
    }
}
