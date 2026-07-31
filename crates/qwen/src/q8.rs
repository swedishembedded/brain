// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Int8 (DP4A) inference path for the Qwen encoder's 7 per-layer linears.
//!
//! Purpose: the fp32 Qwen3-4B encoder is ~16 GB of weights, but on a non-ReBAR
//! Pascal card each storage buffer carries ~2× resident overhead, so the fp32
//! encoder needs ~30 GB and does not fit one 24 GB P40 (nor split alongside the
//! 13 GB int8 DiT). Quantizing the linears to int8 (per-channel symmetric,
//! packed 4-per-`u32`) drops the linear weights ~4× (~12.6 GB → ~3.2 GB), so the
//! whole encoder is ~4.8 GB of weights → ~9.5 GB resident and fits GPU 1 alone,
//! leaving the DiT its own card. The encode then runs on-GPU (~1-2 s) instead of
//! ~38 s on the CPU — the point of a hot, resident pipeline.
//!
//! Same recipe as the DiT's `zimage::int8`: weights quantized once at build;
//! activations quantized on-device each forward with a dynamic per-token scale
//! (`max_abs_row` → `quant_pack`), then the DP4A GEMM (`matmul_i8`, ~4× the fp32
//! rate on Pascal) dequantizes with `sx·sw`. Norms/RoPE/attention stay f32 (not
//! matmuls). Inference-only (frozen, no LoRA, no backward).

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};

/// Per-channel symmetric int8 weight quantization — the engine-wide shared
/// implementation (`model::int8`, also used by `zimage` and `flux2`),
/// re-exported so `q8::quantize_weight` callers keep their path. The packed
/// layout is what `matmul_i8*.wgsl` consume.
pub use model::int8::quantize_weight;

/// One int8 linear: packed int8 weight (`[n, k/4]` u32) + per-channel scale `[n]`.
pub struct Lin8 {
    pub packed: DeviceBuffer,
    pub scale: DeviceBuffer,
    pub k: u32, // input width (contraction dim)
    pub n: u32, // output width
}

/// The 7 int8 linears of one transformer layer (attention q/k/v/o + SwiGLU
/// gate/up/down). Norms/RoPE stay f32 and live in the fp32 `ParamStore`.
pub struct Q8Layer {
    pub wq: Lin8,
    pub wk: Lin8,
    pub wv: Lin8,
    pub wo: Lin8,
    pub gate: Lin8,
    pub up: Lin8,
    pub down: Lin8,
}

/// Resident int8 linears for every owned layer + shared activation-quant scratch.
pub struct Q8 {
    pub layers: HashMap<usize, Q8Layer>,
    pub sx: DeviceBuffer, // [n_tokens] per-token activation scale
    pub xq: DeviceBuffer, // [n_tokens * max_k/4] packed activation (reused per linear)
    // kernel indices in the model's pipeline table.
    k_max_abs_row: usize,
    k_quant_pack: usize,
    k_matmul_i8: usize,
}

impl Q8 {
    /// The 7 leaf names that become int8 (everything else stays fp32).
    pub const LINEARS: [&'static str; 7] = [
        "attn.wq.weight",
        "attn.wk.weight",
        "attn.wv.weight",
        "attn.wo.weight",
        "mlp.gate.weight",
        "mlp.up.weight",
        "mlp.down.weight",
    ];

    /// Is `name` (e.g. `blocks.5.attn.wq.weight`) one of the int8 linears?
    pub fn is_i8_linear(name: &str) -> bool {
        name.strip_prefix("blocks.").is_some_and(|rest| {
            rest.split_once('.').is_some_and(|(_, leaf)| Self::LINEARS.contains(&leaf))
        })
    }

    /// Quantize+upload the owned layers' 7 linears from `init`, allocate scratch.
    /// `owned` are the absolute layer indices this shard holds; `dims(l, leaf)`
    /// gives `(n_out, k_in)` for each linear. `n_tokens = b*t`, `max_k` the widest
    /// contraction dim (= d_ff) so one activation-quant buffer serves every linear.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        gpu: &Gpu,
        source: &dyn checkpoint::TensorSource,
        owned: impl Iterator<Item = usize>,
        dims: impl Fn(&str) -> (usize, usize),
        n_tokens: u32,
        max_k: u32,
        k_max_abs_row: usize,
        k_quant_pack: usize,
        k_matmul_i8: usize,
    ) -> Q8 {
        let mk = |name: &str| -> Lin8 {
            let leaf = name.strip_prefix("blocks.").and_then(|r| r.split_once('.')).map(|(_, l)| l).unwrap_or(name);
            let (n, k) = dims(leaf);
            // Pull, quantize+upload, and drop this weight before the next — one
            // tensor of host f32 at a time, whatever the source.
            let mut lin: Option<Lin8> = None;
            let found = source.with_tensor(name, &mut |raw| {
                let (packed, sw) = quantize_weight(raw, n, k);
                let pb = gpu.storage(packed.len() as u64);
                gpu.write(&pb, &packed);
                gpu.poll_wait(); // reclaim staging before the next weight (see paramstore)
                let sb = gpu.storage(sw.len() as u64);
                gpu.write(&sb, &sw.iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
                gpu.poll_wait();
                lin = Some(Lin8 { packed: pb, scale: sb, k: k as u32, n: n as u32 });
            });
            if !found {
                panic!("q8: missing init weight {name}");
            }
            lin.unwrap()
        };
        let mut layers = HashMap::new();
        for l in owned {
            let p = |leaf: &str| format!("blocks.{l}.{leaf}");
            layers.insert(
                l,
                Q8Layer {
                    wq: mk(&p("attn.wq.weight")),
                    wk: mk(&p("attn.wk.weight")),
                    wv: mk(&p("attn.wv.weight")),
                    wo: mk(&p("attn.wo.weight")),
                    gate: mk(&p("mlp.gate.weight")),
                    up: mk(&p("mlp.up.weight")),
                    down: mk(&p("mlp.down.weight")),
                },
            );
        }
        let sx = gpu.storage(n_tokens.max(1) as u64);
        let xq = gpu.storage((n_tokens * max_k / 4).max(1) as u64);
        Q8 { layers, sx, xq, k_max_abs_row, k_quant_pack, k_matmul_i8 }
    }

    /// Quantize activation `x` `[n_tokens · k]` into `self.xq` with fresh per-token
    /// scales `self.sx`. Emits the two prep steps; call once per distinct input
    /// (shared by all linears reading that input, e.g. xn1 → q/k/v).
    pub fn quant(&self, gpu: &Gpu, s: &mut Vec<Step>, x: &DeviceBuffer, k: u32, n_tokens: u32) {
        s.push(gpu.step(self.k_max_abs_row, &[x, &self.sx], &[n_tokens, k], n_tokens));
        s.push(gpu.step(self.k_quant_pack, &[x, &self.sx, &self.xq], &[n_tokens, k], n_tokens * k / 4));
    }

    /// `out = dequant(xq @ wᵀ)`: dynamic per-token scale `self.sx` × per-channel
    /// weight scale. Must be preceded by a matching [`Q8::quant`] on the same input.
    pub fn mm8(&self, gpu: &Gpu, s: &mut Vec<Step>, w: &Lin8, out: &DeviceBuffer, n_tokens: u32) {
        s.push(gpu.step(
            self.k_matmul_i8,
            &[&self.xq, &w.packed, &self.sx, &w.scale, out],
            &[n_tokens, w.k / 4, w.n],
            n_tokens.div_ceil(128) * w.n.div_ceil(128) * 256,
        ));
    }
}
