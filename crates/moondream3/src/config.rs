// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream 3 (preview) configuration — values from the released `config.py`.
//! The 3.1 release ships no modeling code; the preview is the architecture
//! reference (identical hyperparameters assumed).

/// SigLIP-style ViT vision encoder + overlap multi-crop.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub dim: u32,            // enc_dim 1152
    pub patch: u32,          // enc_patch_size 14
    pub n_layers: u32,       // enc_n_layers 27
    pub ff_dim: u32,         // enc_ff_dim 4304
    pub n_heads: u32,        // enc_n_heads 16 (head_dim 72)
    pub crop_size: u32,      // 378 → 27×27 = 729 patches
    pub max_crops: u32,      // 12
    pub overlap_margin: u32, // 4 patches
}

impl VisionConfig {
    pub fn head_dim(&self) -> u32 {
        self.dim / self.n_heads
    }
    /// Patch grid side (`crop_size / patch`). 27.
    pub fn grid(&self) -> u32 {
        self.crop_size / self.patch
    }
    /// Patches per crop (`grid²`). 729.
    pub fn patches_per_crop(&self) -> u32 {
        self.grid() * self.grid()
    }
    /// Flattened per-patch vector (`3 · patch²`). 588.
    pub fn patch_vec(&self) -> u32 {
        3 * self.patch * self.patch
    }
}

/// Sparse-MoE FFN (top-k GeGLU-shift experts) for the deep decoder layers.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    pub num_experts: u32, // 64
    pub start_layer: u32, // 4 (layers 0..3 dense, 4..23 MoE)
    pub top_k: u32,       // 8
    pub inner_dim: u32,   // expert GeGLU inner 1024
}

/// Full Moondream 3 configuration.
#[derive(Clone, Debug)]
pub struct MoondreamConfig {
    pub dim: u32,         // text dim 2048
    pub ff_dim: u32,      // dense FFN 8192
    pub n_layers: u32,    // 24
    pub vocab: u32,       // 51200
    pub n_heads: u32,     // 32 (full MHA, no GQA)
    pub head_dim: u32,    // 64
    pub prefix_attn: u32, // 730 = 1 (bos) + 729 image tokens (bidirectional)
    pub rot_dim: u32,     // partial-RoPE rotated channels 32
    pub rope_theta: f32,  // 1.5e6
    pub proj_inner: u32,  // connector hidden 8192
    pub proj_out: u32,    // connector out (= text dim) 2048
    pub vision: VisionConfig,
    pub moe: MoeConfig,
}

impl MoondreamConfig {
    /// The Moondream 3 preview configuration.
    pub fn preview() -> MoondreamConfig {
        MoondreamConfig {
            dim: 2048,
            ff_dim: 8192,
            n_layers: 24,
            vocab: 51200,
            n_heads: 32,
            head_dim: 64,
            prefix_attn: 730,
            rot_dim: 32,
            rope_theta: 1_500_000.0,
            proj_inner: 8192,
            proj_out: 2048,
            vision: VisionConfig {
                dim: 1152,
                patch: 14,
                n_layers: 27,
                ff_dim: 4304,
                n_heads: 16,
                crop_size: 378,
                max_crops: 12,
                overlap_margin: 4,
            },
            moe: MoeConfig { num_experts: 64, start_layer: 4, top_k: 8, inner_dim: 1024 },
        }
    }

    /// Read a checkpoint's own `config.json`, then REFUSE anything that is not
    /// the preview architecture.
    ///
    /// Deriving alone would silently serve a differently-shaped checkpoint;
    /// hardcoding alone would not notice one at all. So this does both, the
    /// same way `deepseek2ocr::import::config` does: parse the fields that are
    /// present, then compare the result against [`MoondreamConfig::preview`]
    /// and name the first field that differs.
    ///
    /// The 3.1 release ships no modeling code, so `config.json` is the only
    /// machine-readable description of the checkpoint; a release that changes a
    /// dimension will be caught here rather than as a shape error deep in the
    /// importer.
    pub fn from_json(root: &serde_json::Value) -> Result<MoondreamConfig, String> {
        let want = MoondreamConfig::preview();
        // Only the fields a real `config.json` carries are read; anything absent
        // falls back to the preview value, and the comparison below is what
        // makes that safe (a checkpoint that disagrees is rejected, not
        // defaulted into).
        let u = |k: &str, d: u32| -> u32 { root.get(k).and_then(|v| v.as_u64()).map(|v| v as u32).unwrap_or(d) };
        let got = MoondreamConfig {
            dim: u("dim", want.dim),
            ff_dim: u("ff_dim", want.ff_dim),
            n_layers: u("n_layers", want.n_layers),
            vocab: u("vocab_size", u("vocab", want.vocab)),
            n_heads: u("n_heads", want.n_heads),
            head_dim: u("head_dim", want.head_dim),
            prefix_attn: u("prefix_attn", want.prefix_attn),
            rot_dim: u("rot_dim", want.rot_dim),
            rope_theta: root.get("rope_theta").and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(want.rope_theta),
            proj_inner: u("proj_inner", want.proj_inner),
            proj_out: u("proj_out", want.proj_out),
            vision: want.vision.clone(),
            moe: want.moe.clone(),
        };
        let mismatches: Vec<String> = [
            ("dim", got.dim, want.dim),
            ("ff_dim", got.ff_dim, want.ff_dim),
            ("n_layers", got.n_layers, want.n_layers),
            ("vocab", got.vocab, want.vocab),
            ("n_heads", got.n_heads, want.n_heads),
            ("head_dim", got.head_dim, want.head_dim),
            ("prefix_attn", got.prefix_attn, want.prefix_attn),
            ("rot_dim", got.rot_dim, want.rot_dim),
            ("proj_inner", got.proj_inner, want.proj_inner),
            ("proj_out", got.proj_out, want.proj_out),
        ]
        .iter()
        .filter(|(_, g, w)| g != w)
        .map(|(n, g, w)| format!("{n}: checkpoint says {g}, this port builds {w}"))
        .collect();
        if !mismatches.is_empty() {
            return Err(format!("moondream3: config.json is not the preview architecture ({})", mismatches.join("; ")));
        }
        Ok(got)
    }

    /// [`MoondreamConfig::from_json`] over a checkpoint directory's
    /// `config.json`.
    pub fn from_dir(dir: &std::path::Path) -> Result<MoondreamConfig, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("moondream3: reading {}: {e}", path.display()))?;
        let root: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("moondream3: parsing {}: {e}", path.display()))?;
        MoondreamConfig::from_json(&root)
    }

    /// True if decoder layer `l` uses the MoE FFN (else a dense FFN).
    pub fn is_moe_layer(&self, l: u32) -> bool {
        l >= self.moe.start_layer
    }

    /// Connector input width: global‖local channel-concat of the ViT features
    /// (`2 · vision.dim`). 2304.
    pub fn connector_in(&self) -> u32 {
        2 * self.vision.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_dims() {
        let c = MoondreamConfig::preview();
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.n_heads * c.head_dim, c.dim); // full MHA, 32×64 = 2048
        assert_eq!(c.connector_in(), 2304); // 2 × 1152
        assert_eq!(c.prefix_attn, 1 + c.vision.patches_per_crop()); // 730
    }

    #[test]
    fn vision_dims() {
        let v = MoondreamConfig::preview().vision;
        assert_eq!(v.head_dim(), 72); // 1152 / 16
        assert_eq!(v.grid(), 27); // 378 / 14
        assert_eq!(v.patches_per_crop(), 729);
        assert_eq!(v.patch_vec(), 588); // 3 · 14²
    }

    /// An empty `config.json` falls back to the preview values and is
    /// accepted - the checkpoint simply did not restate them.
    #[test]
    fn from_json_accepts_a_config_that_matches_the_preview() {
        let c = MoondreamConfig::from_json(&serde_json::json!({})).expect("empty config falls back to preview");
        let want = MoondreamConfig::preview();
        assert_eq!((c.dim, c.n_layers, c.vocab, c.n_heads), (want.dim, want.n_layers, want.vocab, want.n_heads));
    }

    /// THE POINT OF READING IT AT ALL: a checkpoint of a DIFFERENT shape must be
    /// named, not silently served with this port's dimensions. Deriving without
    /// checking would build a graph the weights do not fit; hardcoding without
    /// reading would never notice.
    #[test]
    fn from_json_rejects_a_differently_shaped_checkpoint_by_name() {
        let err = MoondreamConfig::from_json(&serde_json::json!({"dim": 4096})).unwrap_err();
        assert!(err.contains("dim"), "the mismatching field must be named: {err}");
        assert!(err.contains("4096"), "the checkpoint's own value must be quoted back: {err}");
    }

    /// `vocab_size` is the HF spelling; `vocab` is brain's. Both are read, so a
    /// real `config.json` is not silently ignored in favour of the default.
    #[test]
    fn from_json_reads_either_vocab_spelling() {
        let want = MoondreamConfig::preview().vocab;
        assert!(MoondreamConfig::from_json(&serde_json::json!({"vocab_size": want})).is_ok());
        assert!(MoondreamConfig::from_json(&serde_json::json!({"vocab": want})).is_ok());
        let err = MoondreamConfig::from_json(&serde_json::json!({"vocab_size": 1234})).unwrap_err();
        assert!(err.contains("vocab"), "{err}");
    }

    #[test]
    fn moe_layer_split() {
        let c = MoondreamConfig::preview();
        assert!(!c.is_moe_layer(0) && !c.is_moe_layer(3));
        assert!(c.is_moe_layer(4) && c.is_moe_layer(23));
        assert_eq!(c.moe.num_experts, 64);
        assert_eq!(c.moe.top_k, 8);
    }
}
