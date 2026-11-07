// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where the draft's tensors come from, under one canonical vocabulary.
//!
//! Swedish Embedded AB implements speculative decoding for clients serving large models on modest hardware.
//! If your team needs expertise in loading drafters from the formats they ship in then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The published draft is bf16 safetensors ([`HfDraft`]); the Qwen3.8 draft this repository first ported is a Q8_0
//! GGUF ([`GgufDraft`]). Each names its tensors its own way, so the model asks for a [`Tensor`] and the source maps
//! it: [`Tensor::name_hf`] and [`Tensor::name_gguf`] are the two spellings, kept side by side so a disagreement
//! between them is one diff.

use std::path::Path;

use checkpoint::gguf::MmapGguf;
use checkpoint::packed::{BoundedWorkspace, CodecRegistry, MemoryBudget, PackedReader, PackedSource};
use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use model::ops::{stored_tier, upload_from_source, Ops, Weight};

const MODEL: &str = "dflash2";

/// Host staging one tile of a packed load may take.
const STAGING_BYTES: u64 = 64 << 20;

/// A tensor of one draft layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leaf {
    InputNorm,
    PostAttnNorm,
    Q,
    K,
    V,
    O,
    QNorm,
    KNorm,
    Gate,
    Up,
    Down,
    /// `[2, taps, d_model]`: the static base kernels of the convolutions entering and leaving the attention.
    AttnConvBase,
    AttnConvProj,
    MlpConvBase,
    MlpConvProj,
}

/// A tensor of the draft model, by role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tensor {
    /// `[d_model, taps * d_model]`: fuses the target's tapped hidden states.
    Fc,
    /// RMSNorm applied after `fc` (not before it).
    HiddenNorm,
    /// The draft's final RMSNorm: applied to the denoised block before the target's head.
    OutNorm,
    SelectorHidden,
    SelectorPredecessor,
    SelectorSuccessor,
    Layer(usize, Leaf),
}

impl Tensor {
    /// The name in the Hugging Face safetensors.
    pub fn name_hf(self) -> String {
        match self {
            Tensor::Fc => "fc.weight".into(),
            Tensor::HiddenNorm => "hidden_norm.weight".into(),
            Tensor::OutNorm => "norm.weight".into(),
            Tensor::SelectorHidden => "candidate_selector.hidden_projection.weight".into(),
            Tensor::SelectorPredecessor => "candidate_selector.predecessor_codebook".into(),
            Tensor::SelectorSuccessor => "candidate_selector.successor_codebook".into(),
            Tensor::Layer(l, leaf) => {
                let tail = match leaf {
                    Leaf::InputNorm => "input_layernorm.weight",
                    Leaf::PostAttnNorm => "post_attention_layernorm.weight",
                    Leaf::Q => "self_attn.q_proj.weight",
                    Leaf::K => "self_attn.k_proj.weight",
                    Leaf::V => "self_attn.v_proj.weight",
                    Leaf::O => "self_attn.o_proj.weight",
                    Leaf::QNorm => "self_attn.q_norm.weight",
                    Leaf::KNorm => "self_attn.k_norm.weight",
                    Leaf::Gate => "mlp.gate_proj.weight",
                    Leaf::Up => "mlp.up_proj.weight",
                    Leaf::Down => "mlp.down_proj.weight",
                    Leaf::AttnConvBase => "attention_conv.base_kernel",
                    Leaf::AttnConvProj => "attention_conv.kernel_projection.weight",
                    Leaf::MlpConvBase => "mlp_conv.base_kernel",
                    Leaf::MlpConvProj => "mlp_conv.kernel_projection.weight",
                };
                format!("layers.{l}.{tail}")
            }
        }
    }

    /// The name in the GGUF.
    pub fn name_gguf(self) -> String {
        match self {
            Tensor::Fc => "fc.weight".into(),
            Tensor::HiddenNorm => "enc.output_norm.weight".into(),
            Tensor::OutNorm => "output_norm.weight".into(),
            Tensor::SelectorHidden => "selector_hidden.weight".into(),
            Tensor::SelectorPredecessor => "selector_predecessor.weight".into(),
            Tensor::SelectorSuccessor => "selector_successor.weight".into(),
            Tensor::Layer(l, leaf) => {
                let tail = match leaf {
                    Leaf::InputNorm => "attn_norm.weight",
                    Leaf::PostAttnNorm => "ffn_norm.weight",
                    Leaf::Q => "attn_q.weight",
                    Leaf::K => "attn_k.weight",
                    Leaf::V => "attn_v.weight",
                    Leaf::O => "attn_output.weight",
                    Leaf::QNorm => "attn_q_norm.weight",
                    Leaf::KNorm => "attn_k_norm.weight",
                    Leaf::Gate => "ffn_gate.weight",
                    Leaf::Up => "ffn_up.weight",
                    Leaf::Down => "ffn_down.weight",
                    Leaf::AttnConvBase => "attn_conv_base",
                    Leaf::AttnConvProj => "attn_conv_proj.weight",
                    Leaf::MlpConvBase => "ffn_conv_base",
                    Leaf::MlpConvProj => "ffn_conv_proj.weight",
                };
                format!("blk.{l}.{tail}")
            }
        }
    }
}

/// The draft's tensors, from whatever format it was published in.
pub trait DraftSource: Send + Sync {
    /// The logical shape of `t` when the format records one.
    fn shape(&self, t: Tensor) -> Option<Vec<usize>>;

    /// `t` as f32 on the host (norm gains, convolution bases, the selector's projection).
    fn vector(&self, t: Tensor) -> Result<Vec<f32>, String>;

    /// The `[n, k]` linear `t` resident on `ops`'s device at the source's tier.
    fn linear(&self, ops: &Ops, t: Tensor, n: usize, k: usize) -> Result<Weight, String>;

    /// `count` rows of `width` elements of the `[rows, width]` matrix `t`, from row `first`: the selector's
    /// codebooks are read by row (a handful of 100k) and never materialized.
    fn rows(&self, t: Tensor, first: usize, count: usize, width: usize) -> Result<Vec<f32>, String>;
}

/// The published draft: Hugging Face safetensors, held on the device as the file stores them (bf16 stays bf16).
pub struct HfDraft {
    src: PackedSource,
    staging: BoundedWorkspace,
}

impl HfDraft {
    /// The draft in `path`, a safetensors file or a Hugging Face directory of them.
    pub fn open(path: &Path) -> Result<HfDraft, String> {
        let reader = PackedReader::open(path).map_err(|e| format!("{MODEL}: cannot open {}: {e}", path.display()))?;
        let src = PackedSource::identity_lenient(Box::new(reader), CodecRegistry::reference());
        Ok(HfDraft { src, staging: BoundedWorkspace::new(MemoryBudget::new(STAGING_BYTES, 0)) })
    }
}

impl DraftSource for HfDraft {
    fn shape(&self, t: Tensor) -> Option<Vec<usize>> {
        self.src.packed_descriptor(&t.name_hf()).map(|d| d.shape.iter().map(|&x| x as usize).collect())
    }

    fn vector(&self, t: Tensor) -> Result<Vec<f32>, String> {
        let name = t.name_hf();
        let numel = self.src.packed_descriptor(&name).map(|d| d.numel()).ok_or_else(|| format!("{MODEL}: the draft checkpoint has no tensor {name}"))?;
        self.src.read_range(&name, 0, numel)?.ok_or_else(|| format!("{MODEL}: the draft checkpoint has no tensor {name}"))
    }

    fn linear(&self, ops: &Ops, t: Tensor, n: usize, k: usize) -> Result<Weight, String> {
        let name = t.name_hf();
        let want = stored_tier(&self.src, &name).unwrap_or(Dtype::F32);
        let limit = ops.gpu().max_storage_binding_bytes();
        let bytes = Weight::bytes_for(want, n, k);
        if bytes > limit {
            return Err(format!("{MODEL}: {name} needs a {bytes}-byte binding as {} but this device binds at most {limit} bytes", model::policy::format_name(want)));
        }
        let weight = upload_from_source(ops, &self.src, &name, n, k, want, &self.staging).ok_or_else(|| format!("{MODEL}: the draft checkpoint has no tensor {name}"))?;
        if weight.dtype() != want {
            return Err(format!("{MODEL}: {name} is stored as {} but this device holds that tier as {}", model::policy::format_name(want), model::policy::format_name(weight.dtype())));
        }
        Ok(weight)
    }

    fn rows(&self, t: Tensor, first: usize, count: usize, width: usize) -> Result<Vec<f32>, String> {
        let name = t.name_hf();
        self.src.read_range(&name, (first * width) as u64, (count * width) as u64)?.ok_or_else(|| format!("{MODEL}: {name} has no rows {first}..{} of width {width}", first + count))
    }
}

/// The Q8_0 GGUF draft of the Qwen3.8 target: the linears become group-wise int8 straight from the blocks
/// ([`Dtype::I8`], the serving tier) or f32 for the host-reference gate.
pub struct GgufDraft {
    mg: MmapGguf,
    dt: Dtype,
}

impl GgufDraft {
    pub fn open(path: &str, dt: Dtype) -> Result<GgufDraft, String> {
        Ok(GgufDraft { mg: MmapGguf::open(path).map_err(|e| format!("{MODEL}: {path}: {e}"))?, dt })
    }

    /// The file, for [`crate::Dflash2Config::from_gguf`].
    pub fn gguf(&self) -> &MmapGguf {
        &self.mg
    }

    pub fn dtype(&self) -> Dtype {
        self.dt
    }
}

impl DraftSource for GgufDraft {
    fn shape(&self, t: Tensor) -> Option<Vec<usize>> {
        self.mg.shape(&t.name_gguf()).map(<[usize]>::to_vec)
    }

    fn vector(&self, t: Tensor) -> Result<Vec<f32>, String> {
        let name = t.name_gguf();
        self.mg.tensor(&name).ok_or_else(|| format!("{MODEL}: {name} missing"))?
    }

    fn linear(&self, ops: &Ops, t: Tensor, n: usize, k: usize) -> Result<Weight, String> {
        let name = t.name_gguf();
        if self.dt == Dtype::I8 {
            let mut up = paramstore::upload::Uploader::new(ops.gpu());
            let (w, s) = model::int8::upload_quantized(&mut up, &self.mg, &name, n, k)?;
            return Ok(Weight::from_packed_int8(w, s, n, k));
        }
        let raw = self.mg.tensor(&name).ok_or_else(|| format!("{MODEL}: {name} missing"))??;
        Ok(Weight::upload(ops, &raw, n, k, self.dt))
    }

    fn rows(&self, t: Tensor, first: usize, count: usize, width: usize) -> Result<Vec<f32>, String> {
        let name = t.name_gguf();
        self.mg.tensor_range(&name, first * width, count * width).ok_or_else(|| format!("{MODEL}: {name} has no rows {first}..{}", first + count))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_spellings_name_the_same_roles() {
        assert_eq!(Tensor::Layer(3, Leaf::AttnConvBase).name_hf(), "layers.3.attention_conv.base_kernel");
        assert_eq!(Tensor::Layer(3, Leaf::AttnConvBase).name_gguf(), "blk.3.attn_conv_base");
        assert_eq!(Tensor::HiddenNorm.name_hf(), "hidden_norm.weight");
        assert_eq!(Tensor::HiddenNorm.name_gguf(), "enc.output_norm.weight");
        assert_eq!(Tensor::SelectorSuccessor.name_hf(), "candidate_selector.successor_codebook");
    }
}
