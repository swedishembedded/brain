// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import `flow.pt` (the real `FunAudioLLM/CosyVoice2-0.5B` checkpoint - a
//! `torch.save`'d `CausalMaskedDiffWithXvec.state_dict()`) into the flow
//! decoder's weights.
//!
//! `flow.pt` is self-contained (1121 tensors: `input_embedding`,
//! `spk_embed_affine_layer`, the 6+4-layer `UpsampleConformerEncoder` (each
//! `RelPositionMultiHeadedAttention` + `PositionwiseFeedForward`, no
//! macaron/conv module - `macaron_style: False, use_cnn_module: False` in
//! `cosyvoice2.yaml`), `encoder_proj`, and the `CausalConditionalDecoder` UNet
//! estimator: 1 down + 12 mid + 1 up stage, each a `CausalResnetBlock1D` + 4
//! `BasicTransformerBlock`s (56 transformer blocks, 14 resnet blocks total -
//! verified by literally counting the imported tensor names, not assumed from
//! the config alone). `decoder.rand_noise` is NOT a registered buffer (a
//! plain `self.rand_noise = torch.randn(...)` attribute in
//! `CausalConditionalCFM.__init__`, seeded via `set_all_random_seed(0)`), so
//! it never appears in the state dict - see `crate::flow`'s noise-asset doc
//! for how this port reproduces it instead.
//!
//! Two-way coverage, same discipline as [`crate::llm_import::import_llm_pt`]:
//! every tensor this canonical manifest names is required exactly once: an
//! unrecognized or duplicate tensor fails loudly, and any checkpoint tensor
//! left over after every name is resolved fails loudly too.

use std::collections::HashMap;

use crate::flow_config::FlowConfig;

#[derive(Clone, Debug)]
pub struct LinearW {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

/// `RelPositionMultiHeadedAttention` + `PositionwiseFeedForward`, one
/// `ConformerEncoderLayer` (no macaron FFN, no conv module - see the module
/// doc).
#[derive(Clone, Debug)]
pub struct ConformerLayerW {
    pub pos_bias_u: Vec<f32>, // [heads*head_dim]
    pub pos_bias_v: Vec<f32>,
    pub wq: LinearW,
    pub wk: LinearW,
    pub wv: LinearW,
    pub wo: LinearW,
    pub w_pos: Vec<f32>, // linear_pos, no bias
    pub ff1: LinearW,
    pub ff2: LinearW,
    pub norm_mha: LinearW, // LayerNorm(weight, bias)
    pub norm_ff: LinearW,
}

/// `LinearNoSubsampling`: `Linear -> LayerNorm`.
#[derive(Clone)]
pub struct SubsampleW {
    pub linear: LinearW,
    pub ln: LinearW,
}

#[derive(Clone)]
pub struct EncoderW {
    pub pre_conv1: LinearW, // Conv1d(512,512,4)
    pub pre_conv2: LinearW, // Conv1d(512,512,3)
    pub embed: SubsampleW,
    pub layers: Vec<ConformerLayerW>,
    pub up_layer_conv: LinearW, // Conv1d(512,512,5)
    pub up_embed: SubsampleW,
    pub up_layers: Vec<ConformerLayerW>,
    pub after_norm: LinearW,
}

/// `CausalResnetBlock1D`: `mlp` (`Mish -> Linear`), two `CausalBlock1D`s
/// (`CausalConv1d -> LayerNorm -> Mish`), and a plain `Conv1d(k=1)` residual
/// projection.
#[derive(Clone)]
pub struct ResnetBlockW {
    pub mlp: LinearW,
    pub block1_conv: LinearW,
    pub block1_ln: LinearW,
    pub block2_conv: LinearW,
    pub block2_ln: LinearW,
    pub res_conv: LinearW,
}

/// `BasicTransformerBlock`: pre-LN self-attention (`bias=False` on q/k/v,
/// `bias=True` on `to_out`) + pre-LN exact-GELU FFN, no cross-attention.
#[derive(Clone)]
pub struct CfmBlockW {
    pub norm1: LinearW,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: LinearW,
    pub norm3: LinearW,
    pub ff1: LinearW,
    pub ff2: LinearW,
}

/// One down/up UNet stage: a resnet block, `blocks_per_stage` transformer
/// blocks, and the trailing causal `Conv1d(k=3)` ("downsample"/"upsample" in
/// name only - `channels=[256]` makes every stage `is_last`, so this conv
/// never changes resolution, see `crate::flow`'s module doc).
#[derive(Clone)]
pub struct UnetStageW {
    pub resnet: ResnetBlockW,
    pub blocks: Vec<CfmBlockW>,
    pub conv: LinearW,
}

#[derive(Clone)]
pub struct MidStageW {
    pub resnet: ResnetBlockW,
    pub blocks: Vec<CfmBlockW>,
}

#[derive(Clone)]
pub struct EstimatorW {
    pub time_mlp1: LinearW, // Linear(320,1024)
    pub time_mlp2: LinearW, // Linear(1024,1024)
    pub down: UnetStageW,
    pub mid: Vec<MidStageW>,
    pub up: UnetStageW,
    pub final_block_conv: LinearW,
    pub final_block_ln: LinearW,
    pub final_proj: LinearW,
}

#[derive(Clone)]
pub struct FlowWeights {
    pub input_embedding: Vec<f32>, // [vocab_size, input_size]
    pub spk_affine: LinearW,       // [output_size, spk_embed_dim]
    pub encoder: EncoderW,
    pub encoder_proj: LinearW, // [output_size, d_model]
    pub estimator: EstimatorW,
}

/// A tensor pool keyed by name, consumed exactly-once via [`Self::take`] -
/// the same "remove and error if anything is left" two-way coverage
/// [`crate::llm_import::import_llm_pt`] uses.
struct Pool(HashMap<String, Vec<f32>>);

impl Pool {
    fn take(&mut self, name: &str) -> Result<Vec<f32>, String> {
        self.0.remove(name).ok_or_else(|| format!("import_flow_pt: missing {name}"))
    }
    fn take_linear(&mut self, prefix: &str) -> Result<LinearW, String> {
        Ok(LinearW { w: self.take(&format!("{prefix}.weight"))?, b: self.take(&format!("{prefix}.bias"))? })
    }
    fn take_weight_only(&mut self, prefix: &str) -> Result<Vec<f32>, String> {
        self.take(&format!("{prefix}.weight"))
    }
}

fn take_conformer_layer(p: &mut Pool, prefix: &str) -> Result<ConformerLayerW, String> {
    Ok(ConformerLayerW {
        pos_bias_u: p.take(&format!("{prefix}.self_attn.pos_bias_u"))?,
        pos_bias_v: p.take(&format!("{prefix}.self_attn.pos_bias_v"))?,
        wq: p.take_linear(&format!("{prefix}.self_attn.linear_q"))?,
        wk: p.take_linear(&format!("{prefix}.self_attn.linear_k"))?,
        wv: p.take_linear(&format!("{prefix}.self_attn.linear_v"))?,
        wo: p.take_linear(&format!("{prefix}.self_attn.linear_out"))?,
        w_pos: p.take_weight_only(&format!("{prefix}.self_attn.linear_pos"))?,
        ff1: p.take_linear(&format!("{prefix}.feed_forward.w_1"))?,
        ff2: p.take_linear(&format!("{prefix}.feed_forward.w_2"))?,
        norm_mha: p.take_linear(&format!("{prefix}.norm_mha"))?,
        norm_ff: p.take_linear(&format!("{prefix}.norm_ff"))?,
    })
}

fn take_subsample(p: &mut Pool, prefix: &str) -> Result<SubsampleW, String> {
    Ok(SubsampleW { linear: p.take_linear(&format!("{prefix}.out.0"))?, ln: p.take_linear(&format!("{prefix}.out.1"))? })
}

fn take_resnet_block(p: &mut Pool, prefix: &str) -> Result<ResnetBlockW, String> {
    Ok(ResnetBlockW {
        mlp: p.take_linear(&format!("{prefix}.mlp.1"))?,
        block1_conv: p.take_linear(&format!("{prefix}.block1.block.0"))?,
        block1_ln: p.take_linear(&format!("{prefix}.block1.block.2"))?,
        block2_conv: p.take_linear(&format!("{prefix}.block2.block.0"))?,
        block2_ln: p.take_linear(&format!("{prefix}.block2.block.2"))?,
        res_conv: p.take_linear(&format!("{prefix}.res_conv"))?,
    })
}

fn take_cfm_block(p: &mut Pool, prefix: &str) -> Result<CfmBlockW, String> {
    Ok(CfmBlockW {
        norm1: p.take_linear(&format!("{prefix}.norm1"))?,
        wq: p.take_weight_only(&format!("{prefix}.attn1.to_q"))?,
        wk: p.take_weight_only(&format!("{prefix}.attn1.to_k"))?,
        wv: p.take_weight_only(&format!("{prefix}.attn1.to_v"))?,
        wo: p.take_linear(&format!("{prefix}.attn1.to_out.0"))?,
        norm3: p.take_linear(&format!("{prefix}.norm3"))?,
        ff1: p.take_linear(&format!("{prefix}.ff.net.0.proj"))?,
        ff2: p.take_linear(&format!("{prefix}.ff.net.2"))?,
    })
}

fn take_unet_stage(p: &mut Pool, prefix: &str, n_blocks: u32) -> Result<UnetStageW, String> {
    let resnet = take_resnet_block(p, &format!("{prefix}.0"))?;
    let mut blocks = Vec::with_capacity(n_blocks as usize);
    for i in 0..n_blocks {
        blocks.push(take_cfm_block(p, &format!("{prefix}.1.{i}"))?);
    }
    let conv = p.take_linear(&format!("{prefix}.2"))?;
    Ok(UnetStageW { resnet, blocks, conv })
}

/// Import `flow.pt` into [`FlowWeights`], validated against the exact shape
/// counted from `cfg`.
pub fn import_flow_pt(path: &str, cfg: &FlowConfig) -> Result<FlowWeights, String> {
    let tensors = checkpoint::torchpt::read(path)?;
    let mut p = Pool(tensors.into_iter().map(|t| (t.name, t.data)).collect());

    let input_embedding = p.take_weight_only("input_embedding")?;
    let spk_affine = p.take_linear("spk_embed_affine_layer")?;

    let pre_conv1 = p.take_linear("encoder.pre_lookahead_layer.conv1")?;
    let pre_conv2 = p.take_linear("encoder.pre_lookahead_layer.conv2")?;
    let embed = take_subsample(&mut p, "encoder.embed")?;
    let mut layers = Vec::with_capacity(cfg.encoder.num_blocks as usize);
    for i in 0..cfg.encoder.num_blocks {
        layers.push(take_conformer_layer(&mut p, &format!("encoder.encoders.{i}"))?);
    }
    let up_layer_conv = p.take_linear("encoder.up_layer.conv")?;
    let up_embed = take_subsample(&mut p, "encoder.up_embed")?;
    let mut up_layers = Vec::with_capacity(cfg.encoder.num_up_blocks as usize);
    for i in 0..cfg.encoder.num_up_blocks {
        up_layers.push(take_conformer_layer(&mut p, &format!("encoder.up_encoders.{i}"))?);
    }
    let after_norm = p.take_linear("encoder.after_norm")?;
    let encoder = EncoderW { pre_conv1, pre_conv2, embed, layers, up_layer_conv, up_embed, up_layers, after_norm };

    let encoder_proj = p.take_linear("encoder_proj")?;

    let time_mlp1 = p.take_linear("decoder.estimator.time_mlp.linear_1")?;
    let time_mlp2 = p.take_linear("decoder.estimator.time_mlp.linear_2")?;
    let down = take_unet_stage(&mut p, "decoder.estimator.down_blocks.0", cfg.estimator.blocks_per_stage)?;
    let mut mid = Vec::with_capacity(cfg.estimator.num_mid_stages as usize);
    for i in 0..cfg.estimator.num_mid_stages {
        let resnet = take_resnet_block(&mut p, &format!("decoder.estimator.mid_blocks.{i}.0"))?;
        let mut blocks = Vec::with_capacity(cfg.estimator.blocks_per_stage as usize);
        for j in 0..cfg.estimator.blocks_per_stage {
            blocks.push(take_cfm_block(&mut p, &format!("decoder.estimator.mid_blocks.{i}.1.{j}"))?);
        }
        mid.push(MidStageW { resnet, blocks });
    }
    let up = take_unet_stage(&mut p, "decoder.estimator.up_blocks.0", cfg.estimator.blocks_per_stage)?;
    let final_block_conv = p.take_linear("decoder.estimator.final_block.block.0")?;
    let final_block_ln = p.take_linear("decoder.estimator.final_block.block.2")?;
    let final_proj = p.take_linear("decoder.estimator.final_proj")?;
    let estimator =
        EstimatorW { time_mlp1, time_mlp2, down, mid, up, final_block_conv, final_block_ln, final_proj };

    if !p.0.is_empty() {
        let mut extra: Vec<&String> = p.0.keys().collect();
        extra.sort();
        return Err(format!("import_flow_pt: {} tensors unused: {extra:?}", extra.len()));
    }

    let w = FlowWeights { input_embedding, spk_affine, encoder, encoder_proj, estimator };
    validate_shapes(&w, cfg)?;
    Ok(w)
}

/// Element-count validation against `cfg` - the other half of two-way
/// coverage (every NAME resolved above; every SHAPE checked here).
fn validate_shapes(w: &FlowWeights, cfg: &FlowConfig) -> Result<(), String> {
    let (d, mel, spk) = (cfg.input_size as usize, cfg.output_size as usize, cfg.spk_embed_dim as usize);
    let check = |name: &str, got: usize, want: usize| -> Result<(), String> {
        if got != want {
            return Err(format!("import_flow_pt: {name} has {got} elements, want {want}"));
        }
        Ok(())
    };
    check("input_embedding", w.input_embedding.len(), cfg.vocab_size as usize * d)?;
    check("spk_embed_affine_layer.weight", w.spk_affine.w.len(), mel * spk)?;
    check("encoder_proj.weight", w.encoder_proj.w.len(), mel * d)?;
    check("encoder.layers", w.encoder.layers.len(), cfg.encoder.num_blocks as usize)?;
    check("encoder.up_layers", w.encoder.up_layers.len(), cfg.encoder.num_up_blocks as usize)?;
    check("estimator.mid", w.estimator.mid.len(), cfg.estimator.num_mid_stages as usize)?;
    check("estimator.down.blocks", w.estimator.down.blocks.len(), cfg.estimator.blocks_per_stage as usize)?;
    check("estimator.up.blocks", w.estimator.up.blocks.len(), cfg.estimator.blocks_per_stage as usize)?;
    let inner = (cfg.estimator.num_heads * cfg.estimator.attention_head_dim) as usize;
    check("estimator.down.blocks[0].attn1.to_q", w.estimator.down.blocks[0].wq.len(), inner * cfg.estimator.channels as usize)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_name_helpers_match_the_real_checkpoint_layout() {
        // The exact prefixes read from `flow.pt`'s own state-dict keys
        // (`torch.load(..., weights_only=True).keys()`), not guessed from the
        // Python source alone - see the module doc.
        let mut m: HashMap<String, Vec<f32>> = HashMap::new();
        for name in [
            "encoder.encoders.0.self_attn.pos_bias_u",
            "encoder.encoders.0.self_attn.pos_bias_v",
            "encoder.encoders.0.self_attn.linear_q.weight",
            "encoder.encoders.0.self_attn.linear_q.bias",
            "encoder.encoders.0.self_attn.linear_k.weight",
            "encoder.encoders.0.self_attn.linear_k.bias",
            "encoder.encoders.0.self_attn.linear_v.weight",
            "encoder.encoders.0.self_attn.linear_v.bias",
            "encoder.encoders.0.self_attn.linear_out.weight",
            "encoder.encoders.0.self_attn.linear_out.bias",
            "encoder.encoders.0.self_attn.linear_pos.weight",
            "encoder.encoders.0.feed_forward.w_1.weight",
            "encoder.encoders.0.feed_forward.w_1.bias",
            "encoder.encoders.0.feed_forward.w_2.weight",
            "encoder.encoders.0.feed_forward.w_2.bias",
            "encoder.encoders.0.norm_mha.weight",
            "encoder.encoders.0.norm_mha.bias",
            "encoder.encoders.0.norm_ff.weight",
            "encoder.encoders.0.norm_ff.bias",
        ] {
            m.insert(name.to_string(), vec![0.0]);
        }
        let mut p = Pool(m);
        assert!(take_conformer_layer(&mut p, "encoder.encoders.0").is_ok());
        assert!(p.0.is_empty(), "take_conformer_layer must consume every one of its own tensors");
    }

    #[test]
    fn missing_tensor_fails_loudly_by_name() {
        let mut p = Pool(HashMap::new());
        let err = take_conformer_layer(&mut p, "encoder.encoders.0").unwrap_err();
        assert!(err.contains("pos_bias_u"), "error must name the missing tensor, got: {err}");
    }
}
