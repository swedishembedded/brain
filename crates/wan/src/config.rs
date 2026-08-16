// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Wan variant hyperparameters, transcribed from upstream.
//!
//! Two sources, because upstream splits them:
//!
//! - **Architecture** (`dim`, `ffn_dim`, heads, layers, …) comes from
//!   `wan/configs/wan_{t2v_1_3B,t2v_14B,i2v_14B}.py` and agrees with the HF
//!   `transformer/config.json` in the `-Diffusers` repos.
//! - **Sampling defaults** (shift, steps, guidance, frame count) come from
//!   `generate.py`'s *argument defaults*, not from any config file. Nothing in
//!   the checkpoint records them, so a port that reads only `config.json`
//!   silently invents its own schedule.

/// Which task a checkpoint was trained for. This is not cosmetic: it changes
/// the input channel count and the sampling defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    /// Text to video. The DiT consumes the 16-channel latent directly.
    T2v,
    /// Image to video. The DiT input is widened to 36 channels - 16 latent, 4
    /// mask, 16 for the VAE-encoded conditioning frame - and a CLIP ViT-H/14
    /// **vision tower** supplies 257 extra context tokens through a separate
    /// `img_emb` MLP. (The `clip_xlm_roberta_vit_h_14` checkpoint's XLM-RoBERTa
    /// text side is unused: only `clip.visual(...)` is called.)
    I2v,
}

/// One Wan variant.
///
/// Fields mirror upstream's names so a reader can diff this against
/// `wan/configs/*.py` without a translation table.
#[derive(Clone, Debug, PartialEq)]
pub struct WanConfig {
    /// Upstream's task name, e.g. `"t2v-1.3B"` - the key into `SUPPORTED_SIZES`.
    pub name: &'static str,
    pub task: Task,

    // -- transformer -------------------------------------------------------
    pub dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    /// 16 for T2V; 36 for I2V (16 latent + 4 mask + 16 conditioning frame).
    pub in_channels: usize,
    pub out_channels: usize,
    /// `(t, h, w)`. Always `(1, 2, 2)`: the temporal patch is 1, so patchify is
    /// a per-frame 2x2 space-to-depth, not a true 3D convolution.
    pub patch_size: (usize, usize, usize),
    /// Width of the sinusoidal timestep embedding before its MLP.
    pub freq_dim: usize,
    /// Width of the text encoding the cross-attention consumes (umT5-XXL).
    pub text_dim: usize,
    /// Text tokens are masked and then zero-padded to exactly this length.
    pub text_len: usize,
    pub eps: f32,
    /// RMSNorm applied to q and k inside self-attention.
    pub qk_norm: bool,
    /// LayerNorm on the cross-attention input.
    pub cross_attn_norm: bool,

    // -- VAE ---------------------------------------------------------------
    /// `(t, h, w)` compression. 4x temporal and 8x spatial, with the `1 + 4k`
    /// frame rule: 81 frames encode to 21 latent frames, not 20.
    pub vae_stride: (usize, usize, usize),

    // -- sampling defaults -------------------------------------------------
    /// Flow-matching sigma shift: `s' = shift*s / (1 + (shift-1)*s)`.
    ///
    /// 5.0 everywhere **except** I2V at 480p, which uses 3.0. Note this is a
    /// task-and-size rule, not a resolution rule - a T2V run at 480p still
    /// uses 5.0 (`generate.py:76-81`).
    pub sample_shift: f32,
    /// 50 for T2V, 40 for I2V (`generate.py:70-74`).
    pub sample_steps: usize,
    /// Classifier-free guidance scale. 5.0 for every task
    /// (`generate.py:242-244`) - upstream exposes no per-task default.
    pub guide_scale: f32,
    pub sample_fps: usize,
    /// Default frame count. The VAE's `1 + 4k` rule means only `1 + 4k` values
    /// are representable; 81 = 1 + 4*20.
    pub frame_num: usize,
    pub num_train_timesteps: usize,

    /// `(width, height)` pairs upstream lists as supported for this variant.
    /// T2V-1.3B is **480p only** - 720p exists solely on the 14B tier.
    pub sizes: &'static [(usize, usize)],
}

const P480: &[(usize, usize)] = &[(832, 480), (480, 832)];
const P480_720: &[(usize, usize)] = &[(1280, 720), (720, 1280), (832, 480), (480, 832)];

impl WanConfig {
    /// The shared spine. Every variant differs from this only in the fields it
    /// overrides, which keeps the diff against upstream's `shared_config.py`
    /// readable.
    const fn base() -> Self {
        Self {
            name: "",
            task: Task::T2v,
            dim: 0,
            ffn_dim: 0,
            num_heads: 0,
            num_layers: 0,
            in_channels: 16,
            out_channels: 16,
            patch_size: (1, 2, 2),
            freq_dim: 256,
            text_dim: 4096,
            text_len: 512,
            eps: 1e-6,
            qk_norm: true,
            cross_attn_norm: true,
            vae_stride: (4, 8, 8),
            sample_shift: 5.0,
            sample_steps: 50,
            guide_scale: 5.0,
            sample_fps: 16,
            frame_num: 81,
            num_train_timesteps: 1000,
            sizes: P480,
        }
    }

    /// Wan2.1-T2V-1.3B - the 8.19 GB-VRAM variant, and the demo target.
    pub const fn t2v_1_3b() -> Self {
        Self { name: "t2v-1.3B", dim: 1536, ffn_dim: 8960, num_heads: 12, num_layers: 30, sizes: P480, ..Self::base() }
    }

    /// Wan2.1-T2V-14B.
    pub const fn t2v_14b() -> Self {
        Self { name: "t2v-14B", dim: 5120, ffn_dim: 13824, num_heads: 40, num_layers: 40, sizes: P480_720, ..Self::base() }
    }

    /// Wan2.1-I2V-14B at 480p. Note the 3.0 shift - the one variant that
    /// deviates from upstream's 5.0 default.
    pub const fn i2v_14b_480p() -> Self {
        Self {
            name: "i2v-14B",
            task: Task::I2v,
            in_channels: 36,
            dim: 5120,
            ffn_dim: 13824,
            num_heads: 40,
            num_layers: 40,
            sample_shift: 3.0,
            sample_steps: 40,
            sizes: P480,
            ..Self::base()
        }
    }

    /// Wan2.1-I2V-14B at 720p. Same weights as the 480p entry upstream treats
    /// as a separate checkpoint; the shift reverts to 5.0 at this size.
    pub const fn i2v_14b_720p() -> Self {
        Self { sample_shift: 5.0, sizes: P480_720, ..Self::i2v_14b_480p() }
    }

    /// Every variant, in the order `brain caps` should list them.
    pub const fn all() -> [Self; 4] {
        [Self::t2v_1_3b(), Self::t2v_14b(), Self::i2v_14b_480p(), Self::i2v_14b_720p()]
    }

    /// `dim / num_heads`.
    pub const fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }

    /// How `head_dim` splits across the (frame, height, width) RoPE axes.
    ///
    /// Upstream (`wan/modules/model.py`, `rope_apply`) splits as
    /// `[c - 2*(c/3), c/3, c/3]` where `c = head_dim / 2` is the number of
    /// complex pairs. The frame axis takes the remainder, so an indivisible
    /// `head_dim` biases toward time rather than truncating.
    ///
    /// Returned in **real** components (2 per complex pair), which is what
    /// `dit::rope::RopeConfig::axes_dims` wants.
    pub const fn rope_axes_dims(&self) -> [usize; 3] {
        let c = self.head_dim() / 2;
        let hw = c / 3;
        [(c - 2 * hw) * 2, hw * 2, hw * 2]
    }

    /// Latent extent for a pixel-space request: `(frames, height, width)`.
    ///
    /// The temporal rule is `1 + (frames - 1) / 4`, not `frames / 4` - the
    /// causal VAE maps the first frame to a latent frame of its own. Returns
    /// `None` when `frames` is not of the form `1 + 4k`, because a silently
    /// rounded frame count is a bug that only shows up as a truncated video.
    pub fn latent_shape(&self, frames: usize, width: usize, height: usize) -> Option<(usize, usize, usize)> {
        let (st, sh, sw) = self.vae_stride;
        if frames == 0 || !(frames - 1).is_multiple_of(st) {
            return None;
        }
        Some((1 + (frames - 1) / st, height / sh, width / sw))
    }

    /// Number of transformer tokens for a request - the number that decides
    /// whether attention fits in memory. See the crate doc.
    pub fn token_count(&self, frames: usize, width: usize, height: usize) -> Option<usize> {
        let (f, h, w) = self.latent_shape(frames, width, height)?;
        let (pt, ph, pw) = self.patch_size;
        Some((f / pt) * (h / ph) * (w / pw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_dims_match_upstream_configs() {
        // wan/configs/wan_t2v_1_3B.py:21-25 and the HF transformer/config.json
        let c = WanConfig::t2v_1_3b();
        assert_eq!((c.dim, c.ffn_dim, c.num_heads, c.num_layers), (1536, 8960, 12, 30));
        // wan/configs/wan_t2v_14B.py:21-25
        let c = WanConfig::t2v_14b();
        assert_eq!((c.dim, c.ffn_dim, c.num_heads, c.num_layers), (5120, 13824, 40, 40));
    }

    #[test]
    fn i2v_widens_the_input_but_t2v_does_not() {
        // The 36 channels are 16 latent + 4 mask + 16 conditioning frame. A
        // T2V importer that accepts a 36-channel patch_embedding, or vice
        // versa, is loading the wrong checkpoint.
        assert_eq!(WanConfig::t2v_1_3b().in_channels, 16);
        assert_eq!(WanConfig::t2v_14b().in_channels, 16);
        assert_eq!(WanConfig::i2v_14b_480p().in_channels, 36);
        assert_eq!(WanConfig::i2v_14b_720p().in_channels, 36);
        // Output is always the 16-channel latent.
        assert!(WanConfig::all().iter().all(|c| c.out_channels == 16));
    }

    #[test]
    fn shift_is_a_task_and_size_rule_not_a_resolution_rule() {
        // generate.py:76-81 - the easy misreading is "3.0 at 480p, 5.0 at
        // 720p", which would put T2V-1.3B (a 480p-only variant) on the wrong
        // sigma schedule for every single generation it ever does.
        assert_eq!(WanConfig::t2v_1_3b().sample_shift, 5.0);
        assert_eq!(WanConfig::t2v_14b().sample_shift, 5.0);
        assert_eq!(WanConfig::i2v_14b_480p().sample_shift, 3.0);
        assert_eq!(WanConfig::i2v_14b_720p().sample_shift, 5.0);
    }

    #[test]
    fn sampling_defaults_match_generate_py() {
        assert_eq!(WanConfig::t2v_1_3b().sample_steps, 50);
        assert_eq!(WanConfig::i2v_14b_480p().sample_steps, 40);
        // Upstream has ONE guidance default across every task (generate.py:244).
        assert!(WanConfig::all().iter().all(|c| c.guide_scale == 5.0));
    }

    #[test]
    fn only_the_14b_tier_claims_720p() {
        // wan/configs/__init__.py SUPPORTED_SIZES. This is what keeps the
        // 75,600-token / 22.9 GB-per-head attention case off the 1.3B path.
        assert_eq!(WanConfig::t2v_1_3b().sizes, P480);
        assert!(!WanConfig::t2v_1_3b().sizes.contains(&(1280, 720)));
        assert!(WanConfig::t2v_14b().sizes.contains(&(1280, 720)));
        assert!(WanConfig::i2v_14b_720p().sizes.contains(&(1280, 720)));
    }

    #[test]
    fn rope_axes_split_the_head_dim_exactly() {
        for c in WanConfig::all() {
            let a = c.rope_axes_dims();
            assert_eq!(a[0] + a[1] + a[2], c.head_dim(), "{}: axes must tile head_dim", c.name);
            assert_eq!(a[1], a[2], "{}: height and width axes are equal upstream", c.name);
            assert!(a.iter().all(|d| d % 2 == 0), "{}: each axis is complex pairs", c.name);
        }
        // 1.3B: head_dim 128 -> 64 pairs -> [64-2*21, 21, 21] = [22,21,21] pairs
        assert_eq!(WanConfig::t2v_1_3b().head_dim(), 128);
        assert_eq!(WanConfig::t2v_1_3b().rope_axes_dims(), [44, 42, 42]);
        // 14B: head_dim 128 as well (5120/40), so the same split.
        assert_eq!(WanConfig::t2v_14b().head_dim(), 128);
    }

    #[test]
    fn latent_shape_follows_the_one_plus_four_k_rule() {
        let c = WanConfig::t2v_1_3b();
        // 81 frames -> 21 latent frames, NOT 20: the causal VAE gives the
        // first frame its own latent frame.
        assert_eq!(c.latent_shape(81, 832, 480), Some((21, 60, 104)));
        assert_eq!(c.latent_shape(1, 832, 480), Some((1, 60, 104)));
        assert_eq!(c.latent_shape(5, 832, 480), Some((2, 60, 104)));
        // A frame count that is not 1 + 4k is rejected rather than rounded.
        assert_eq!(c.latent_shape(80, 832, 480), None);
        assert_eq!(c.latent_shape(0, 832, 480), None);
    }

    #[test]
    fn token_counts_are_the_ones_the_memory_plan_assumes() {
        // These two numbers decide that dense attention is impossible and
        // chunked/flash attention is a correctness prerequisite. If either
        // changes, the attention strategy needs revisiting.
        assert_eq!(WanConfig::t2v_1_3b().token_count(81, 832, 480), Some(32_760));
        assert_eq!(WanConfig::t2v_14b().token_count(81, 1280, 720), Some(75_600));
    }
}
