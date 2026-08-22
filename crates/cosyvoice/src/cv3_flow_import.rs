// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import `flow.pt` (the real `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
//! checkpoint - a `torch.save`'d `CausalMaskedDiffWithDiT.state_dict()`) into
//! the DiT flow decoder's weights.
//!
//! Real, checked-not-assumed tensor layout (listed by loading the actual
//! `flow.pt` and grouping its 330 keys by name pattern): `input_embedding`
//! (`[6561, 80]`), `spk_embed_affine_layer` (`[80, 192]` + bias),
//! `pre_lookahead_layer.{conv1,conv2}` (`[1024, 80, 4]` and `[80, 1024, 3]`),
//! and `decoder.estimator.*` - `input_embed.proj` (`[1024, 320]`),
//! `input_embed.conv_pos_embed.conv{1,2}.0` (`[1024, 64, 31]`, `groups=16`
//! depthwise-block), `time_embed.time_mlp.{0,2}` (`[1024,256]`/`[1024,1024]`),
//! `rotary_embed.inv_freq` (`[32]` - a registered, persistent buffer, but this
//! importer recomputes it from the closed-form `1/theta^(2i/dim_head)`
//! formula rather than reading it, the same "regenerable, don't ship a data
//! asset" call `crate::flow::torch_rng` makes for the CFM noise buffer; not
//! read here at all), 22x `transformer_blocks.{i}.{attn_norm.linear,
//! attn.to_{q,k,v}, attn.to_out.0, ff.ff.0.0, ff.ff.2}`, `norm_out.linear`,
//! `proj_out`. **No `attn_norm.norm`/`ff_norm`/`norm_out.norm` tensors exist**
//! (confirmed absent from the real checkpoint's own key list, not assumed
//! from `elementwise_affine=False` alone) - `AdaLayerNormZero`/
//! `AdaLayerNormZero_Final`/`DiTBlock.ff_norm`'s `LayerNorm`s carry no
//! learnable affine, so there is nothing to import for them; `crate::cv3_flow`
//! applies them as a bare (mean/var, no weight/bias) normalization.
//!
//! Two-way coverage, same discipline as [`crate::flow_import::import_flow_pt`]:
//! every tensor this canonical manifest names is required exactly once,
//! `rotary_embed.inv_freq` is the one explicit, documented drop; any other
//! unrecognized or duplicate tensor fails loudly, and any checkpoint tensor
//! left over after every name is resolved fails loudly too.

use std::collections::HashMap;

use crate::cv3_flow_config::Cv3FlowConfig;
use crate::flow_import::LinearW;

/// One `DiTBlock`'s weights.
#[derive(Clone)]
pub struct DitBlockW {
    /// `AdaLayerNormZero.linear`: `Linear(dim, 6*dim)`.
    pub attn_norm_linear: LinearW,
    pub wq: LinearW,
    pub wk: LinearW,
    pub wv: LinearW,
    /// `attn.to_out.0`: `Linear(inner_dim, dim)`.
    pub wo: LinearW,
    /// `ff.ff.0.0`: `Linear(dim, ff_hidden)`.
    pub ff1: LinearW,
    /// `ff.ff.2`: `Linear(ff_hidden, dim)`.
    pub ff2: LinearW,
}

/// `TimestepEmbedding`: `SinusPositionEmbedding(freq_embed_dim) ->
/// Linear(freq_embed_dim, dim) -> SiLU -> Linear(dim, dim)`.
#[derive(Clone)]
pub struct TimeEmbedW {
    pub mlp1: LinearW,
    pub mlp2: LinearW,
}

/// `InputEmbedding`: `Linear(mel_dim*2+mu_dim+spk_dim, dim)` +
/// `CausalConvPositionEmbedding(dim, kernel=31, groups=16)` (two grouped
/// causal convs, each followed by `Mish`).
#[derive(Clone)]
pub struct InputEmbedW {
    pub proj: LinearW,
    pub conv1: LinearW,
    pub conv2: LinearW,
}

/// `AdaLayerNormZero_Final.linear`: `Linear(dim, 2*dim)`.
#[derive(Clone)]
pub struct NormOutW {
    pub linear: LinearW,
}

#[derive(Clone)]
pub struct DitW {
    pub time_embed: TimeEmbedW,
    pub input_embed: InputEmbedW,
    pub blocks: Vec<DitBlockW>,
    pub norm_out: NormOutW,
    pub proj_out: LinearW,
}

#[derive(Clone)]
pub struct Cv3FlowWeights {
    pub input_embedding: Vec<f32>, // [vocab_size, input_size]
    pub spk_affine: LinearW,       // [output_size, spk_embed_dim]
    pub pre_lookahead_conv1: LinearW, // [pre_lookahead_channels, input_size, pre_lookahead_len+1]
    pub pre_lookahead_conv2: LinearW, // [input_size, pre_lookahead_channels, 3]
    pub dit: DitW,
}

struct Pool(HashMap<String, Vec<f32>>);

impl Pool {
    fn take(&mut self, name: &str) -> Result<Vec<f32>, String> {
        self.0.remove(name).ok_or_else(|| format!("import_cv3_flow_pt: missing {name}"))
    }
    fn take_linear(&mut self, prefix: &str) -> Result<LinearW, String> {
        Ok(LinearW { w: self.take(&format!("{prefix}.weight"))?, b: self.take(&format!("{prefix}.bias"))? })
    }
}

fn take_dit_block(p: &mut Pool, prefix: &str) -> Result<DitBlockW, String> {
    Ok(DitBlockW {
        attn_norm_linear: p.take_linear(&format!("{prefix}.attn_norm.linear"))?,
        wq: p.take_linear(&format!("{prefix}.attn.to_q"))?,
        wk: p.take_linear(&format!("{prefix}.attn.to_k"))?,
        wv: p.take_linear(&format!("{prefix}.attn.to_v"))?,
        wo: p.take_linear(&format!("{prefix}.attn.to_out.0"))?,
        ff1: p.take_linear(&format!("{prefix}.ff.ff.0.0"))?,
        ff2: p.take_linear(&format!("{prefix}.ff.ff.2"))?,
    })
}

/// Import `flow.pt` into [`Cv3FlowWeights`], validated against the exact
/// shape counted from `cfg`.
pub fn import_cv3_flow_pt(path: &str, cfg: &Cv3FlowConfig) -> Result<Cv3FlowWeights, String> {
    let tensors = checkpoint::torchpt::read(path)?;
    let mut p = Pool(tensors.into_iter().map(|t| (t.name, t.data)).collect());

    // `rotary_embed.inv_freq` is a registered-but-regenerable buffer - see
    // this module's doc for why it is dropped rather than imported.
    p.0.remove("decoder.estimator.rotary_embed.inv_freq");

    let input_embedding = p.take("input_embedding.weight")?;
    let spk_affine = p.take_linear("spk_embed_affine_layer")?;
    let pre_lookahead_conv1 = p.take_linear("pre_lookahead_layer.conv1")?;
    let pre_lookahead_conv2 = p.take_linear("pre_lookahead_layer.conv2")?;

    let time_embed = TimeEmbedW {
        mlp1: p.take_linear("decoder.estimator.time_embed.time_mlp.0")?,
        mlp2: p.take_linear("decoder.estimator.time_embed.time_mlp.2")?,
    };
    let input_embed = InputEmbedW {
        proj: p.take_linear("decoder.estimator.input_embed.proj")?,
        conv1: p.take_linear("decoder.estimator.input_embed.conv_pos_embed.conv1.0")?,
        conv2: p.take_linear("decoder.estimator.input_embed.conv_pos_embed.conv2.0")?,
    };
    let mut blocks = Vec::with_capacity(cfg.dit.depth as usize);
    for i in 0..cfg.dit.depth {
        blocks.push(take_dit_block(&mut p, &format!("decoder.estimator.transformer_blocks.{i}"))?);
    }
    let norm_out = NormOutW { linear: p.take_linear("decoder.estimator.norm_out.linear")? };
    let proj_out = p.take_linear("decoder.estimator.proj_out")?;

    if !p.0.is_empty() {
        let mut extra: Vec<&String> = p.0.keys().collect();
        extra.sort();
        return Err(format!("import_cv3_flow_pt: {} tensors unused: {extra:?}", extra.len()));
    }

    let dit = DitW { time_embed, input_embed, blocks, norm_out, proj_out };
    let w = Cv3FlowWeights { input_embedding, spk_affine, pre_lookahead_conv1, pre_lookahead_conv2, dit };
    validate_shapes(&w, cfg)?;
    Ok(w)
}

fn validate_shapes(w: &Cv3FlowWeights, cfg: &Cv3FlowConfig) -> Result<(), String> {
    let (d, mel, spk) = (cfg.input_size as usize, cfg.output_size as usize, cfg.spk_embed_dim as usize);
    let check = |name: &str, got: usize, want: usize| -> Result<(), String> {
        if got != want {
            return Err(format!("import_cv3_flow_pt: {name} has {got} elements, want {want}"));
        }
        Ok(())
    };
    check("input_embedding", w.input_embedding.len(), cfg.vocab_size as usize * d)?;
    check("spk_embed_affine_layer.weight", w.spk_affine.w.len(), mel * spk)?;
    let plc = cfg.pre_lookahead_channels as usize;
    check("pre_lookahead_layer.conv1.weight", w.pre_lookahead_conv1.w.len(), plc * d * (cfg.pre_lookahead_len as usize + 1))?;
    check("pre_lookahead_layer.conv2.weight", w.pre_lookahead_conv2.w.len(), d * plc * 3)?;
    check("dit.blocks", w.dit.blocks.len(), cfg.dit.depth as usize)?;
    let dim = cfg.dit.dim as usize;
    check("dit.input_embed.proj.weight", w.dit.input_embed.proj.w.len(), dim * cfg.dit.input_embed_in() as usize)?;
    check("dit.transformer_blocks[0].attn.to_q.weight", w.dit.blocks[0].wq.w.len(), dim * dim)?;
    check("dit.transformer_blocks[0].attn_norm.linear.weight", w.dit.blocks[0].attn_norm_linear.w.len(), 6 * dim * dim)?;
    check("dit.proj_out.weight", w.dit.proj_out.w.len(), mel * dim)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_cv3_flow_pt_from_the_real_checkpoint_covers_every_tensor_and_matches_expected_shapes() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights3/flow.pt");
        if !std::path::Path::new(path).is_file() {
            return;
        }
        let cfg = Cv3FlowConfig::cosyvoice3();
        let w = import_cv3_flow_pt(path, &cfg).unwrap_or_else(|e| panic!("import_cv3_flow_pt: {e}"));
        assert_eq!(w.dit.blocks.len(), 22);
        assert_eq!(w.input_embedding.len(), 6561 * 80);
        assert_eq!(w.pre_lookahead_conv1.w.len(), 1024 * 80 * 4);
        assert_eq!(w.pre_lookahead_conv2.w.len(), 80 * 1024 * 3);
        assert_eq!(w.dit.input_embed.proj.w.len(), 1024 * 320);
        assert_eq!(w.dit.input_embed.conv1.w.len(), 1024 * 64 * 31);
        assert_eq!(w.dit.time_embed.mlp1.w.len(), 1024 * 256);
        assert_eq!(w.dit.proj_out.w.len(), 80 * 1024);
        assert_eq!(w.dit.norm_out.linear.w.len(), 2048 * 1024);
    }
}
