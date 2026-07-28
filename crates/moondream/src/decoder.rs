// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream text decoder pieces. Built up incrementally; today: the sparse-MoE
//! FFN (GeGLU-shift experts + top-k router), which mirrors `crates/moe`'s dense-
//! over-all-experts FFN but swaps SwiGLU for Moondream's GeGLU-with-+1-shift
//! (`geglu_shift`) and a single fc1 split into its `h`/`g` halves (`w_h`/`w_g`).

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Decoder kernel pipeline (indices used below).
pub fn pipelines() -> &'static [(&'static str, &'static str)] {
    &[
        ("matmul", kernels::MATMUL),           // 0
        ("router_gate", kernels::ROUTER_GATE), // 1
        ("geglu_shift", kernels::GEGLU_SHIFT), // 2
        ("scale_add", kernels::SCALE_ADD),     // 3
    ]
}

/// Sparse-MoE FFN: `router → for each expert (w_h, w_g → geglu_shift → w_down) →
/// gate-weighted accumulate`. Returns the mixed output `[t, d]` (no residual — the
/// parallel block owns the 3-way residual). Weight keys: `router.weight` `[e, d]`,
/// and per expert `experts.{e}.{w_h,w_g}.weight` `[inner, d]`, `w_down.weight`
/// `[d, inner]`.
pub struct MoeFfn<'g> {
    gpu: &'g Gpu,
    w: HashMap<String, DeviceBuffer>,
    e: u32,
    top_k: u32,
    d: u32,
    inner: u32,
    // scratch
    logits: DeviceBuffer,
    gate: DeviceBuffer,
    h: DeviceBuffer,
    g: DeviceBuffer,
    act: DeviceBuffer,
    eout: DeviceBuffer,
    acc: DeviceBuffer,
    t: u32,
}

impl<'g> MoeFfn<'g> {
    pub fn new(gpu: &'g Gpu, weights: &HashMap<String, Vec<f32>>, t: u32, d: u32, inner: u32, e: u32, top_k: u32) -> MoeFfn<'g> {
        let w = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        MoeFfn {
            gpu,
            w,
            e,
            top_k,
            d,
            inner,
            logits: gpu.storage((t * e) as u64),
            gate: gpu.storage((t * e) as u64),
            h: gpu.storage((t * inner) as u64),
            g: gpu.storage((t * inner) as u64),
            act: gpu.storage((t * inner) as u64),
            eout: gpu.storage((t * d) as u64),
            acc: gpu.storage((t * d) as u64),
            t,
        }
    }
    fn wb(&self, n: &str) -> &DeviceBuffer {
        self.w.get(n).unwrap_or_else(|| panic!("moe weight missing: {n}"))
    }
    pub fn forward(&self, xn: &DeviceBuffer) -> &DeviceBuffer {
        let (t, d, inner, e) = (self.t, self.d, self.inner, self.e);
        let mut s: Vec<Step> = Vec::new();
        // Router: logits = xn·router.weight^T, then top-k softmax gate.
        s.push(self.gpu.step(0, &[xn, self.wb("router.weight"), &self.logits], &[t, d, e], t * e));
        s.push(self.gpu.step(1, &[&self.logits, &self.gate], &[t, e, self.top_k], t));
        for ei in 0..e {
            let ep = |leaf: &str| self.wb(&format!("experts.{ei}.{leaf}"));
            s.push(self.gpu.step(0, &[xn, ep("w_h.weight"), &self.h], &[t, d, inner], t * inner));
            s.push(self.gpu.step(0, &[xn, ep("w_g.weight"), &self.g], &[t, d, inner], t * inner));
            s.push(self.gpu.step(2, &[&self.h, &self.g, &self.act], &[t * inner], t * inner)); // gelu(h)·(g+1)
            s.push(self.gpu.step(0, &[&self.act, ep("w_down.weight"), &self.eout], &[t, inner, d], t * d));
            let acc = if ei == 0 { 0u32 } else { 1u32 };
            s.push(self.gpu.step(3, &[&self.gate, &self.eout, &self.acc], &[t, d, e, ei, acc], t * d));
        }
        self.gpu.submit(&[], &s);
        &self.acc
    }
    /// Number of output elements (`t·d`).
    pub fn numel(&self) -> usize {
        (self.t * self.d) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;

    #[test]
    fn moe_ffn_geglu_runs() {
        let gpu = Gpu::new_cpu(pipelines());
        let (t, d, inner, e, top_k) = (4u32, 8u32, 4u32, 3u32, 2u32);
        let mut rng = Rng::new(5);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("router.weight".into(), r((e * d) as usize));
        for ei in 0..e {
            w.insert(format!("experts.{ei}.w_h.weight"), r((inner * d) as usize));
            w.insert(format!("experts.{ei}.w_g.weight"), r((inner * d) as usize));
            w.insert(format!("experts.{ei}.w_down.weight"), r((d * inner) as usize));
        }
        let ffn = MoeFfn::new(&gpu, &w, t, d, inner, e, top_k);
        let xn = gpu.storage_init("xn", &(0..(t * d) as usize).map(|_| rng.next_f32() - 0.5).collect::<Vec<f32>>());
        let out = gpu.read(ffn.forward(&xn), ffn.numel());
        assert_eq!(out.len(), (t * d) as usize);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-9));
    }
}
