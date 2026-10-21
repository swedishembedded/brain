// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements fast serving of NVFP4 mixture-of-experts models for its clients. If your team needs
// expertise in 4-bit expert banks and MoE routing on one accelerator then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The NVFP4 expert tier of the Qwen3.5/3.6 MoE: every routed expert (and the shared expert, as block `n_experts`) of a
//! layer lives in three fused banks ([`model::ops::MicroBank`]) holding the checkpoint's own bytes - E2M1 codes, E4M3
//! block scales, one f32 `weight_scale_2` per expert matrix. The expert GEMMs are float-read (`moe_nvfp4_gemv_gather`
//! at decode, `moe_nvfp4_grouped` at prefill): a ModelOpt `W4A16_NVFP4` release has no activation quantization, so
//! nothing is quantized on the way in and the stored `input_scale`s are unused. No expert is ever expanded to a wider
//! format (invariant M).

use checkpoint::packed::{BoundedWorkspace, MemoryBudget, QuantEncoding};
use checkpoint::TensorSource;
use gpu_core::Gpu;
use model::ops::{upload_micro_bank, MicroBank, Weight};

use crate::config::Qwen35Config;

/// Host staging one tile of an expert load may take.
const STAGING_BYTES: u64 = 16 << 20;
/// Elements of the router (and shared gate) decoded per step.
const ROUTER_CHUNK_ELEMS: usize = 1 << 20;

/// Whether `enc` is one of the float-scaled formats a checkpoint stores weights in that the device holds as stored (FP8
/// with tile or tensor scales, NVFP4): such a tensor is uploaded in that format and never widened.
pub fn is_stored_float_quant(enc: &QuantEncoding) -> bool {
    use checkpoint::packed::{Element, FloatFormat};
    matches!(enc.element, Element::Float(FloatFormat::E4M3Fn | FloatFormat::E5M2 | FloatFormat::E2M1)) && !enc.scales.is_empty()
}

/// One layer's routed experts and the shared expert (the banks' last block).
pub struct NvMoeLayer {
    pub gate: MicroBank,
    pub up: MicroBank,
    pub down: MicroBank,
    /// `[n_experts + 1, d_model]` f32 router weight with the shared expert's gate row appended, so one matmul gives
    /// the routed logits and the shared gate's logit (see `moe_router_topk.wgsl`).
    pub router_ext: Weight,
}

pub struct Qwen35Nv {
    pub moe: Vec<NvMoeLayer>,
}

impl Qwen35Nv {
    /// Whether `src` stores this model's experts (and its shared expert) as NVFP4, so the NVFP4 tier applies. A source
    /// that stores some other encoding keeps the model's other expert tiers.
    pub fn applies(src: &dyn TensorSource, cfg: &Qwen35Config) -> bool {
        let nvfp4 = QuantEncoding::nvfp4();
        let is = |name: &str| src.packed_descriptor(name).is_some_and(|d| d.encoding == nvfp4);
        cfg.shared_expert_intermediate_size == cfg.moe_intermediate_size
            && is("blocks.0.mlp.experts.0.gate.weight")
            && is("blocks.0.mlp.shared_expert.gate.weight")
    }

    /// Upload every layer's banks from `src`, one tensor tile at a time.
    pub fn build(gpu: &Gpu, ops: &model::ops::Ops, src: &dyn TensorSource, cfg: &Qwen35Config) -> Result<Qwen35Nv, String> {
        let staging = BoundedWorkspace::new(MemoryBudget::new(STAGING_BYTES, 0));
        let (d, ff, e) = (cfg.d_model as usize, cfg.moe_intermediate_size as usize, cfg.n_experts as usize);
        let mut moe = Vec::with_capacity(cfg.n_layers as usize);
        for l in 0..cfg.n_layers as usize {
            let bank = |proj: &str, n: usize, k: usize| -> Result<MicroBank, String> {
                let names: Vec<String> = (0..e)
                    .map(|x| format!("blocks.{l}.mlp.experts.{x}.{proj}.weight"))
                    .chain(std::iter::once(format!("blocks.{l}.mlp.shared_expert.{proj}.weight")))
                    .collect();
                upload_micro_bank(ops, src, &names, n, k, &staging).map_err(|err| format!("qwen35moe layer {l} {proj} experts: {err}"))
            };
            let (gate, up, down) = (bank("gate", ff, d)?, bank("up", ff, d)?, bank("down", d, ff)?);
            let mut w = Vec::with_capacity((e + 1) * d);
            for name in [format!("blocks.{l}.mlp.router.weight"), format!("blocks.{l}.mlp.shared_expert_gate.weight")] {
                if src.numel(&name).is_none() {
                    return Err(format!("qwen35moe: missing {name}"));
                }
                src.with_tensor_chunks(&name, ROUTER_CHUNK_ELEMS, &mut |_, chunk| w.extend_from_slice(chunk));
            }
            let router_ext = Weight::dense_f32(gpu.storage_init(&format!("blocks.{l}.mlp.router_ext"), &w), cfg.n_experts + 1, cfg.d_model);
            moe.push(NvMoeLayer { gate, up, down, router_ext });
        }
        Ok(Qwen35Nv { moe })
    }

    /// Whether `name` is a tensor the banks hold (a routed or shared expert projection, the router, the shared
    /// expert's gate): the fp32 parameter store does not carry them.
    pub fn holds(name: &str) -> bool {
        crate::q8::Qwen35Q8::is_i8_linear(name) && name.contains(".mlp.experts.")
            || crate::q8::Qwen35Q8::is_shared_expert_linear(name)
            || name.ends_with("mlp.router.weight")
            || name.ends_with("mlp.shared_expert_gate.weight")
    }
}
