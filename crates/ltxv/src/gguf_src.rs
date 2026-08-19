// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Direct GGUF loading for the LTX-2.5 AV DiT - a [`checkpoint::TensorSource`]
//! over a [`checkpoint::gguf::MmapGguf`] that never materializes a converted
//! safetensors file on disk.
//!
//! Modelled on `crates/wan/src/gguf_src.rs`'s `WanGgufSource` (the only other
//! model in this repo that already does GGUF-direct-to-device loading), with
//! ONE structural simplification: Wan needs a reference<->source NAME table
//! because a Wan GGUF may carry either the native or the diffusers tensor
//! spelling. LTX-2.5's real checkpoint carries only ONE spelling - its own,
//! which IS `crate::block::LtxBlock`/`LtxAvBlock`'s `tget` names verbatim
//! (confirmed by range-reading the real 4349-tensor header) - so there is no
//! renaming step here at all: every [`checkpoint::TensorSource`] method just
//! forwards the caller's name straight to the mmap.
//!
//! What still has to agree, and does: [`LtxvGgufSource::from_mmap`] and
//! [`crate::import::import_gguf`] both validate two-way coverage through
//! [`crate::import::validate_av_dit_gguf_shapes`], which itself is built from
//! [`crate::dit::av_dit_tensor_manifest`] - ONE manifest function, so a
//! shape this loader can read and a shape the ahead-of-time converter writes
//! can never silently drift apart. Peak host memory is one tensor's fp32
//! expansion (dominated by a block's own `4*inner_dim x inner_dim` FFN
//! weight), never the whole 22B model.

use checkpoint::gguf::MmapGguf;

use crate::config::LtxAvDitConfig;
use crate::import::{av_dit_config_from_kv, validate_av_dit_gguf_shapes};

/// An LTX-2.5 AV DiT GGUF, opened for direct streaming access - no
/// ahead-of-time conversion, no whole-model host materialization.
pub struct LtxvGgufSource {
    mg: MmapGguf,
    cfg: LtxAvDitConfig,
}

impl LtxvGgufSource {
    /// Open `path`, read the AV DiT config off the file's own embedded
    /// `config` KV, and validate two-way tensor coverage on shapes alone (no
    /// dequant). Errors if the file is not an `ltxv`-architecture checkpoint,
    /// its config KV cannot be parsed, or its tensors don't exactly cover
    /// [`crate::dit::av_dit_tensor_manifest`] at that config.
    pub fn open(path: &str) -> Result<LtxvGgufSource, String> {
        let mg = MmapGguf::open(path)?;
        Self::from_mmap(mg)
    }

    /// [`Self::open`] over an already-mapped [`MmapGguf`] - the seam a caller
    /// that already opened the file (e.g. to read its `ModelCard`) uses to
    /// avoid a second `mmap()`.
    pub fn from_mmap(mg: MmapGguf) -> Result<LtxvGgufSource, String> {
        let cfg = av_dit_config_from_kv(&mg)?;
        validate_av_dit_gguf_shapes(&mg, &cfg)?;
        Ok(LtxvGgufSource { mg, cfg })
    }

    /// The AV DiT config this checkpoint's own `config` KV names.
    pub fn config(&self) -> &LtxAvDitConfig {
        &self.cfg
    }
}

impl checkpoint::TensorSource for LtxvGgufSource {
    fn with_tensor(&self, name: &str, f: &mut dyn FnMut(&[f32])) -> bool {
        self.mg.with_tensor(name, f)
    }

    fn raw_words(&self, name: &str) -> Option<&[u32]> {
        self.mg.raw_words(name)
    }

    fn with_tensor_chunks(&self, name: &str, max_elems: usize, f: &mut dyn FnMut(u64, &[f32])) -> bool {
        self.mg.with_tensor_chunks(name, max_elems, f)
    }

    fn numel(&self, name: &str) -> Option<usize> {
        self.mg.numel(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit::av_dit_tensor_manifest;
    use checkpoint::gguf_write::{write, TensorOut};
    use checkpoint::TensorSource;

    /// [`av_dit_tensor_manifest`] does not gate ANY tensor family (gate
    /// logits, both connectors, the prompt-adaLN tables) on
    /// `apply_gated_attention`/`use_embeddings_connector` - it always
    /// enumerates the full real-checkpoint shape family, by design (this
    /// milestone makes the real config REPRESENTABLE, not conditionally
    /// present - see `crate::dit::av_dit_tensor_manifest`'s doc). So
    /// `LtxAvDitConfig::tiny` already exercises every tensor family this
    /// test needs, at toy dims.
    fn small_cfg() -> LtxAvDitConfig {
        LtxAvDitConfig::tiny()
    }

    /// The `config` KV JSON a real LTX-2.x GGUF embeds, built from `cfg` -
    /// mirrors `qwen35moe::import::testing::write_synthetic_gguf`'s "write a
    /// tiny but structurally real fixture" approach, applied to a JSON KV
    /// value instead of tensor names.
    fn config_kv_json(cfg: &LtxAvDitConfig) -> String {
        serde_json::json!({
            "transformer": {
                "num_attention_heads": cfg.video.num_heads,
                "attention_head_dim": cfg.video.head_dim(),
                "num_layers": cfg.video.num_layers,
                "in_channels": cfg.video.in_channels,
                "out_channels": cfg.video.out_channels,
                "cross_attention_dim": cfg.video.cross_attention_dim,
                "ff_bias": cfg.video.ff_bias,
                "cross_attention_adaln": cfg.video.cross_attention_adaln,
                "use_keyframes_abs_pos_embedding": cfg.video.use_keyframes_abs_pos_embedding,
                "norm_eps": cfg.video.norm_eps,
                "positional_embedding_theta": cfg.video.positional_embedding_theta,
                "positional_embedding_max_pos": cfg.video.positional_embedding_max_pos,
                "timestep_scale_multiplier": cfg.video.timestep_scale_multiplier,
                "use_middle_indices_grid": cfg.video.use_middle_indices_grid,
                "apply_gated_attention": cfg.video.apply_gated_attention,
                "connector_apply_gated_attention": cfg.video.connector_apply_gated_attention,
                "connector_num_layers": cfg.video.connector_num_layers,
                "connector_num_attention_heads": cfg.video.connector_num_attention_heads,
                "connector_attention_head_dim": cfg.video.connector_attention_head_dim,
                "connector_num_learnable_registers": cfg.video.connector_num_learnable_registers,
                "connector_positional_embedding_max_pos": cfg.video.connector_positional_embedding_max_pos,
                "connector_norm_output": cfg.video.connector_norm_output,
                "caption_proj_before_connector": cfg.video.caption_proj_before_connector,
                "use_embeddings_connector": cfg.video.use_embeddings_connector,
                "audio_num_attention_heads": cfg.audio.num_heads,
                "audio_attention_head_dim": cfg.audio.head_dim(),
                "audio_out_channels": cfg.audio.out_channels,
                "audio_cross_attention_dim": cfg.audio.cross_attention_dim,
                "audio_positional_embedding_max_pos": cfg.audio.positional_embedding_max_pos,
                "audio_connector_num_attention_heads": cfg.audio.connector_num_attention_heads,
                "audio_connector_attention_head_dim": cfg.audio.connector_attention_head_dim,
                "av_ca_timestep_scale_multiplier": cfg.av_ca_timestep_scale_multiplier,
            },
        })
        .to_string()
    }

    /// Build a tiny native-spelled GGUF for `cfg`'s full AV manifest, each
    /// tensor filled with distinct deterministic data, plus the embedded
    /// `config` KV [`av_dit_config_from_kv`] must be able to parse back.
    fn synthetic_gguf(path: &str, cfg: &LtxAvDitConfig) {
        use checkpoint::gguf::GgufValue;
        let manifest = av_dit_tensor_manifest(cfg);
        let mut seed = 0u64;
        let tensors: Vec<TensorOut> = manifest
            .into_iter()
            .map(|(name, shape)| {
                seed += 1;
                let n: usize = shape.iter().product();
                let data: Vec<u8> = (0..n).flat_map(|i| (((i as u64 + seed) % 997) as f32 * 0.01 - 5.0).to_le_bytes()).collect();
                TensorOut { name, shape, ty: 0u32 /* ggml type id: F32 */, data }
            })
            .collect();
        let kvs = vec![
            ("general.architecture".to_string(), GgufValue::String("ltxv".to_string())),
            ("config".to_string(), GgufValue::String(config_kv_json(cfg))),
        ];
        write(path, &kvs, &tensors, 32).unwrap();
    }

    fn tmp_path(tag: &str) -> String {
        std::env::temp_dir().join(format!("ltxv-gguf-src-test-{tag}-{}.gguf", std::process::id())).to_string_lossy().into_owned()
    }

    #[test]
    fn open_derives_the_config_and_resolves_every_manifest_tensor() {
        let cfg = small_cfg();
        let path = tmp_path("open");
        synthetic_gguf(&path, &cfg);

        let src = LtxvGgufSource::open(&path).unwrap();
        assert_eq!(*src.config(), cfg);

        for (name, shape) in av_dit_tensor_manifest(&cfg) {
            let n: usize = shape.iter().product();
            let mut got = None;
            assert!(src.with_tensor(&name, &mut |d| got = Some(d.to_vec())), "missing {name}");
            assert_eq!(got.unwrap().len(), n, "{name}");
            assert_eq!(src.numel(&name), Some(n), "{name}");
        }
        assert!(!src.with_tensor("does.not.exist", &mut |_| panic!("must not call")));
        assert_eq!(src.numel("does.not.exist"), None);
        assert!(src.raw_words("does.not.exist").is_none());

        std::fs::remove_file(&path).ok();
    }

    /// Every F32 tensor in the synthetic file must be zero-copyable (the
    /// point of `raw_words` needing no name translation here at all).
    #[test]
    fn raw_words_zero_copies_with_no_rename_step() {
        let cfg = small_cfg();
        let path = tmp_path("rawwords");
        synthetic_gguf(&path, &cfg);
        let src = LtxvGgufSource::open(&path).unwrap();

        for (name, _) in av_dit_tensor_manifest(&cfg) {
            let words = src.raw_words(&name).unwrap_or_else(|| panic!("{name} should be zero-copyable (plain F32)"));
            let mut via_with_tensor = None;
            src.with_tensor(&name, &mut |d| via_with_tensor = Some(d.to_vec()));
            let via_words: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();
            assert_eq!(via_words, via_with_tensor.unwrap(), "{name}");
        }
        std::fs::remove_file(&path).ok();
    }

    /// A file whose architecture tag or tensor coverage doesn't match must
    /// fail at `open`, not silently succeed with a bogus config or a
    /// half-covered model - the same two-way-coverage discipline
    /// `import_av_dit`/`import_gguf` apply, checked at the loader boundary
    /// too since a caller may reach `LtxvGgufSource::open` directly without
    /// ever going through `import_gguf`.
    #[test]
    fn open_rejects_a_checkpoint_that_is_not_an_ltxv_av_dit() {
        let path = tmp_path("bad-arch");
        let tensors = vec![TensorOut { name: "not.an.ltxv.tensor".to_string(), shape: vec![4], ty: 0u32, data: vec![0u8; 16] }];
        write(&path, &[], &tensors, 32).unwrap();
        let e = LtxvGgufSource::open(&path).err().expect("must be rejected");
        assert!(e.contains("general.architecture"), "{e}");
        std::fs::remove_file(&path).ok();

        // Right architecture tag, but a tensor missing from the manifest.
        let cfg = small_cfg();
        let path2 = tmp_path("missing-tensor");
        {
            use checkpoint::gguf::GgufValue;
            let mut manifest = av_dit_tensor_manifest(&cfg);
            manifest.pop(); // drop the last tensor - an incomplete checkpoint
            let tensors: Vec<TensorOut> = manifest
                .into_iter()
                .map(|(name, shape)| {
                    let n: usize = shape.iter().product();
                    TensorOut { name, shape, ty: 0u32, data: vec![0u8; n * 4] }
                })
                .collect();
            let kvs = vec![
                ("general.architecture".to_string(), GgufValue::String("ltxv".to_string())),
                ("config".to_string(), GgufValue::String(config_kv_json(&cfg))),
            ];
            write(&path2, &kvs, &tensors, 32).unwrap();
        }
        let e2 = LtxvGgufSource::open(&path2).err().expect("must be rejected");
        assert!(e2.contains("missing tensor"), "{e2}");
        std::fs::remove_file(&path2).ok();
    }
}
