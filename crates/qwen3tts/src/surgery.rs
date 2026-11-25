// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for shrinking trained speech
// synthesizers into smaller models that still load in their upstream tooling,
// for its clients. If your team needs expertise in compressing text-to-speech
// models then you can procure our services by sending an email to
// info@swedishembedded.com.

//! What a Qwen3-TTS checkpoint (upstream layout, upstream names) declares for
//! weight surgery ([`model::surgery::SurgeryGraph`]): every dimension the
//! upstream config can state narrower, and the tensors coupled to it.
//!
//! * **Layer stacks** `talker.layers` and `code_predictor.layers`; a layer
//!   writes the stream through its `o_proj` and `down_proj`.
//! * **Channel dimensions** per layer of either stack: `*.mlp` (the MLP's
//!   hidden channels: `gate_proj`/`up_proj` rows, `down_proj` columns) and
//!   `*.kv_heads` (whole KV groups with their query heads: `head_dim` rows of
//!   `k_proj`/`v_proj`, `heads per group * head_dim` rows of `q_proj` and
//!   columns of `o_proj`). `head_dim` never changes: the multimodal rotary
//!   sections are laid out over 128 dimensions.
//! * **Streams** `talker` (written by the codec embedding, the text
//!   projection's output, every layer, the code predictor's codec embeddings,
//!   which are talker-wide, and the speaker encoder's last projection when the
//!   checkpoint has one; read by every layer, `codec_head` and
//!   `small_to_mtp_projection`) and `code_predictor` (written by
//!   `small_to_mtp_projection` and its layers, read by its layers and its 15
//!   heads). The talker's final norm is shared: its output and the raw codec
//!   embeddings both enter `small_to_mtp_projection`. A checkpoint whose code
//!   predictor is as wide as the talker has no projection, and its code
//!   predictor joins the talker stream.
//! * **Fold** `talker.text`: the text embedding feeds `linear_fc1`, which is
//!   square at `text_hidden_size` (upstream sizes it so) and linear up to its
//!   output channels.
//! * **Vocabulary** `text`: the text embedding's rows, `text_vocab_size` and
//!   every special token id of the top-level config.
//!
//! [`check_upstream`] refuses a config the upstream code would build
//! differently from the tensors (a code predictor as wide as the talker
//! while a projection exists, heads that do not divide into KV groups).

use model::surgery::{Axis, ChannelDim, ConfigWidth, EmbeddingFold, LayerStack, Lines, Stream, StreamNorm, StreamReader, SurgeryGraph, Vocabulary};
use serde_json::Value;

const TALKER: &str = "talker_config";
const CODE_PREDICTOR: &str = "talker_config.code_predictor_config";
const TALKER_LAYERS: &str = "talker.model.layers.";
const CP_LAYERS: &str = "talker.code_predictor.model.layers.";
const PROJECTION: &str = "talker.code_predictor.small_to_mtp_projection";
/// The speaker encoder's last projection: a 1x1 convolution onto the talker width.
const SPEAKER_FC: &str = "speaker_encoder.fc";
/// The top-level config keys holding a text token id.
pub const TOKEN_ID_KEYS: [&str; 6] = ["im_start_token_id", "im_end_token_id", "assistant_token_id", "tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id"];

fn get<'a>(config: &'a Value, path: &str) -> Result<&'a Value, String> {
    path.split('.').try_fold(config, |v, k| v.get(k).ok_or_else(|| format!("config.json has no {path}")))
}

fn width(config: &Value, path: &str) -> Result<usize, String> {
    get(config, path)?.as_u64().map(|v| v as usize).ok_or_else(|| format!("config.json: {path} is not an unsigned integer"))
}

/// One decoder stack's shape, from its config section.
struct Decoder {
    section: &'static str,
    prefix: &'static str,
    name: &'static str,
    layers: usize,
    hidden: usize,
    ff: usize,
    heads: usize,
    kv: usize,
    head_dim: usize,
}

impl Decoder {
    fn read(config: &Value, section: &'static str, prefix: &'static str, name: &'static str) -> Result<Decoder, String> {
        let w = |k: &str| width(config, &format!("{section}.{k}"));
        let d = Decoder { section, prefix, name, layers: w("num_hidden_layers")?, hidden: w("hidden_size")?, ff: w("intermediate_size")?, heads: w("num_attention_heads")?, kv: w("num_key_value_heads")?, head_dim: w("head_dim")? };
        if d.kv == 0 || !d.heads.is_multiple_of(d.kv) {
            return Err(format!("config.json: {section} has {} attention heads, not whole groups of its {} KV heads", d.heads, d.kv));
        }
        Ok(d)
    }

    fn key(&self, k: &str) -> String {
        format!("{}.{k}", self.section)
    }

    fn t(&self, l: usize, leaf: &str) -> String {
        format!("{}{l}.{leaf}", self.prefix)
    }

    fn stack(&self, config: &Value) -> LayerStack {
        let lists = ["layer_types"].iter().map(|k| self.key(k)).filter(|k| get(config, k).is_ok_and(Value::is_array)).collect();
        LayerStack { name: format!("{}.layers", self.name), prefix: self.prefix.into(), count: self.layers, key: self.key("num_hidden_layers"), outputs: vec!["self_attn.o_proj.weight".into(), "mlp.down_proj.weight".into()], lists }
    }

    fn dims(&self) -> Vec<ChannelDim> {
        let group = self.heads / self.kv * self.head_dim;
        (0..self.layers)
            .flat_map(|l| {
                let mlp = ChannelDim {
                    name: format!("{}.layers.{l}.mlp", self.name),
                    width: self.ff,
                    keys: vec![ConfigWidth::new(self.key("intermediate_size"), 1)],
                    lines: vec![Lines::new(self.t(l, "mlp.gate_proj.weight"), Axis::Rows, 1), Lines::new(self.t(l, "mlp.up_proj.weight"), Axis::Rows, 1), Lines::new(self.t(l, "mlp.down_proj.weight"), Axis::Cols, 1)],
                    masks: vec![self.t(l, "mlp.down_proj.weight")],
                };
                let kv = ChannelDim {
                    name: format!("{}.layers.{l}.kv_heads", self.name),
                    width: self.kv,
                    keys: vec![ConfigWidth::new(self.key("num_key_value_heads"), 1), ConfigWidth::new(self.key("num_attention_heads"), self.heads / self.kv)],
                    lines: vec![
                        Lines::new(self.t(l, "self_attn.q_proj.weight"), Axis::Rows, group),
                        Lines::new(self.t(l, "self_attn.k_proj.weight"), Axis::Rows, self.head_dim),
                        Lines::new(self.t(l, "self_attn.v_proj.weight"), Axis::Rows, self.head_dim),
                        Lines::new(self.t(l, "self_attn.o_proj.weight"), Axis::Cols, group),
                    ],
                    masks: vec![self.t(l, "self_attn.o_proj.weight")],
                };
                [mlp, kv]
            })
            .collect()
    }

    /// This stack's layers as part of a stream: writers, readers and norms.
    fn join(&self, s: &mut Stream) {
        for l in 0..self.layers {
            s.writers.push((self.t(l, "self_attn.o_proj.weight"), Axis::Rows));
            s.writers.push((self.t(l, "mlp.down_proj.weight"), Axis::Rows));
            for (leaf, norm) in [("self_attn.q_proj", "input_layernorm"), ("self_attn.k_proj", "input_layernorm"), ("self_attn.v_proj", "input_layernorm"), ("mlp.gate_proj", "post_attention_layernorm"), ("mlp.up_proj", "post_attention_layernorm")] {
                s.readers.push(StreamReader { tensor: self.t(l, &format!("{leaf}.weight")), norm: Some(self.t(l, &format!("{norm}.weight"))) });
            }
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                s.norms.push(StreamNorm { tensor: self.t(l, &format!("{norm}.weight")), shared: false });
            }
        }
    }
}

/// The code predictor's per-codebook tables: `talker.code_predictor.{kind}.{i}.weight`.
fn tables(kind: &str, n: usize) -> impl Iterator<Item = String> + '_ {
    (0..n).map(move |i| format!("talker.code_predictor.{kind}.{i}.weight"))
}

/// The surgery graph of the Qwen3-TTS checkpoint whose `config.json` is
/// `config` and which holds the tensors `has` answers for.
pub fn graph(config: &Value, has: &dyn Fn(&str) -> bool) -> Result<SurgeryGraph, String> {
    let talker = Decoder::read(config, TALKER, TALKER_LAYERS, "talker")?;
    let cp = Decoder::read(config, CODE_PREDICTOR, CP_LAYERS, "code_predictor")?;
    let groups = width(config, &format!("{CODE_PREDICTOR}.num_code_groups"))?;
    let residual = groups.checked_sub(1).ok_or("config.json: num_code_groups is 0")?;
    let projected = has(&format!("{PROJECTION}.weight"));
    if projected == (talker.hidden == cp.hidden) {
        return Err(format!("the talker is {} wide and the code predictor {}, but the checkpoint {} small_to_mtp_projection", talker.hidden, cp.hidden, if projected { "has a" } else { "has no" }));
    }

    let mut keys = vec![talker.key("hidden_size")];
    if get(config, "speaker_encoder_config.enc_dim").is_ok() {
        keys.push("speaker_encoder_config.enc_dim".into());
    }
    let mut t = Stream { name: "talker".into(), width: talker.hidden, keys, writers: vec![("talker.model.codec_embedding.weight".into(), Axis::Cols)], readers: Vec::new(), norms: Vec::new() };
    for leaf in ["weight", "bias"] {
        t.writers.push((format!("talker.text_projection.linear_fc2.{leaf}"), Axis::Rows));
        if has(&format!("{SPEAKER_FC}.{leaf}")) {
            t.writers.push((format!("{SPEAKER_FC}.{leaf}"), Axis::Rows));
        }
    }
    t.writers.extend(tables("model.codec_embedding", residual).map(|n| (n, Axis::Cols)));
    talker.join(&mut t);
    t.readers.push(StreamReader { tensor: "talker.codec_head.weight".into(), norm: Some("talker.model.norm.weight".into()) });
    // The final norm's output and the raw codec embeddings both enter the code predictor's first map.
    t.norms.push(StreamNorm { tensor: "talker.model.norm.weight".into(), shared: true });
    let mut streams = Vec::new();
    let cp_norm = || StreamNorm { tensor: "talker.code_predictor.model.norm.weight".into(), shared: false };
    let heads = || tables("lm_head", residual).map(|n| StreamReader { tensor: n, norm: Some("talker.code_predictor.model.norm.weight".into()) });
    if projected {
        t.readers.push(StreamReader { tensor: format!("{PROJECTION}.weight"), norm: None });
        let mut c = Stream { name: "code_predictor".into(), width: cp.hidden, keys: vec![cp.key("hidden_size")], writers: vec![(format!("{PROJECTION}.weight"), Axis::Rows), (format!("{PROJECTION}.bias"), Axis::Rows)], readers: Vec::new(), norms: Vec::new() };
        cp.join(&mut c);
        c.readers.extend(heads());
        c.norms.push(cp_norm());
        streams.push(t);
        streams.push(c);
    } else {
        // One stream through both decoders: the code predictor reads the talker's rows directly.
        t.keys.push(cp.key("hidden_size"));
        cp.join(&mut t);
        t.readers.extend(heads());
        t.norms.push(cp_norm());
        streams.push(t);
    }

    let text_hidden = width(config, &format!("{TALKER}.text_hidden_size"))?;
    let fc1 = "talker.text_projection.linear_fc1";
    let hidden = ChannelDim {
        name: "talker.text".into(),
        width: text_hidden,
        keys: vec![ConfigWidth::new(format!("{TALKER}.text_hidden_size"), 1)],
        lines: vec![Lines::new(format!("{fc1}.weight"), Axis::Rows, 1), Lines::new(format!("{fc1}.bias"), Axis::Rows, 1), Lines::new("talker.text_projection.linear_fc2.weight", Axis::Cols, 1)],
        masks: vec!["talker.text_projection.linear_fc2.weight".into()],
    };
    let fold = EmbeddingFold { name: "talker.text".into(), embedding: "talker.model.text_embedding.weight".into(), linear: format!("{fc1}.weight"), bias: Some(format!("{fc1}.bias")), hidden };
    let vocab = Vocabulary {
        name: "text".into(),
        size: width(config, &format!("{TALKER}.text_vocab_size"))?,
        key: format!("{TALKER}.text_vocab_size"),
        tables: vec!["talker.model.text_embedding.weight".into()],
        ids: TOKEN_ID_KEYS.iter().filter(|k| config.get(**k).is_some()).map(|k| k.to_string()).collect(),
    };
    let mut dims = talker.dims();
    dims.extend(cp.dims());
    Ok(SurgeryGraph { dims, stacks: vec![talker.stack(config), cp.stack(config)], streams, folds: vec![fold], vocabularies: vec![vocab] })
}

/// Refuse a config the upstream package would build differently from the
/// tensors it describes: a projection between equal widths (upstream builds
/// none), heads that are not whole KV groups, a `head_dim` other than the
/// multimodal rotary sections cover.
pub fn check_upstream(config: &Value, has: &dyn Fn(&str) -> bool) -> Result<(), String> {
    let talker = Decoder::read(config, TALKER, TALKER_LAYERS, "talker")?;
    let cp = Decoder::read(config, CODE_PREDICTOR, CP_LAYERS, "code_predictor")?;
    let projected = has(&format!("{PROJECTION}.weight"));
    if projected && talker.hidden == cp.hidden {
        return Err(format!("the talker and the code predictor are both {} wide: upstream then builds no small_to_mtp_projection, and this checkpoint needs one; choose another width for either", talker.hidden));
    }
    let sections: usize = get(config, &format!("{TALKER}.rope_scaling.mrope_section"))?.as_array().ok_or("config.json: mrope_section is not a list")?.iter().filter_map(Value::as_u64).sum::<u64>() as usize;
    if talker.head_dim != 2 * sections {
        return Err(format!("the talker's head_dim is {}, the rotary sections cover {}", talker.head_dim, 2 * sections));
    }
    Ok(())
}
