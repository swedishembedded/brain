// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 model configuration and the canonical (BFL-named) tensor manifest.

/// Architecture hyperparameters for one FLUX.2 variant.
///
/// `distilled` selects the sampling recipe only (4-step no-CFG klein vs
/// 50-step CFG base) — base and klein weights are shape-identical.
#[derive(Clone, Debug, PartialEq)]
pub struct Flux2Config {
    /// Latent channels entering `img_in` (VAE 32ch × 2×2 pixel-unshuffle).
    pub in_channels: usize,
    /// Text-conditioning width: 3 concatenated Qwen3 hidden states.
    pub context_in_dim: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub depth_double: usize,
    pub depth_single: usize,
    /// SwiGLU inner width = `hidden * mlp_ratio`.
    pub mlp_ratio: f32,
    /// Per-axis RoPE dims in (t, h, w, l) order; sums to `head_dim()`.
    pub axes_dim: [usize; 4],
    pub rope_theta: f64,
    /// LayerNorm and QK-RMSNorm epsilon (both 1e-6 in every variant).
    pub norm_eps: f32,
    /// Fixed text token count (pad tokens attend un-masked).
    pub txt_len: usize,
    /// `guidance_in` MLP present (FLUX.2-dev only; every klein variant: false).
    pub guidance_embed: bool,
    /// Step+guidance distilled → 4 Euler steps, no CFG. Base variants: false.
    pub distilled: bool,
}

impl Flux2Config {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }

    /// SwiGLU inner width (the `mlp_hidden` of the reference).
    pub fn mlp_hidden(&self) -> usize {
        (self.hidden as f32 * self.mlp_ratio) as usize
    }

    pub fn klein_4b() -> Flux2Config {
        Flux2Config {
            in_channels: 128,
            context_in_dim: 7680, // 3 × Qwen3-4B hidden 2560
            hidden: 3072,
            n_heads: 24,
            depth_double: 5,
            depth_single: 20,
            mlp_ratio: 3.0,
            axes_dim: [32, 32, 32, 32],
            rope_theta: 2000.0,
            norm_eps: 1e-6,
            txt_len: 512,
            guidance_embed: false,
            distilled: true,
        }
    }

    pub fn klein_9b() -> Flux2Config {
        Flux2Config {
            in_channels: 128,
            context_in_dim: 12288, // 3 × Qwen3-8B hidden 4096
            hidden: 4096,
            n_heads: 32,
            depth_double: 8,
            depth_single: 24,
            ..Flux2Config::klein_4b()
        }
    }

    /// Resolve a user-facing variant name (the CLI/capability enum:
    /// `klein-4b | klein-9b | base-4b | base-9b`) — the ONE name→config map
    /// shared by `brain flux2` and the capability provider.
    pub fn from_name(v: &str) -> Result<Flux2Config, String> {
        Ok(match v {
            "klein-4b" => Flux2Config::klein_4b(),
            "klein-9b" => Flux2Config::klein_9b(),
            "base-4b" => Flux2Config::klein_base_4b(),
            "base-9b" => Flux2Config::klein_base_9b(),
            other => return Err(format!("unknown variant {other} (klein-4b|klein-9b|base-4b|base-9b)")),
        })
    }

    pub fn klein_base_4b() -> Flux2Config {
        Flux2Config { distilled: false, ..Flux2Config::klein_4b() }
    }

    pub fn klein_base_9b() -> Flux2Config {
        Flux2Config { distilled: false, ..Flux2Config::klein_9b() }
    }

    /// The full canonical tensor manifest: BFL names with shapes, in a fixed
    /// order. Import validates full two-way coverage against this list; the
    /// trainer derives its parameter layout from it.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let d = self.hidden;
        let hd = self.head_dim();
        let mlp = self.mlp_hidden();
        let mut v: Vec<(String, Vec<usize>)> = vec![
            ("img_in.weight".into(), vec![d, self.in_channels]),
            ("txt_in.weight".into(), vec![d, self.context_in_dim]),
            ("time_in.in_layer.weight".into(), vec![d, 256]),
            ("time_in.out_layer.weight".into(), vec![d, d]),
        ];
        if self.guidance_embed {
            v.push(("guidance_in.in_layer.weight".into(), vec![d, 256]));
            v.push(("guidance_in.out_layer.weight".into(), vec![d, d]));
        }
        v.push(("double_stream_modulation_img.lin.weight".into(), vec![6 * d, d]));
        v.push(("double_stream_modulation_txt.lin.weight".into(), vec![6 * d, d]));
        v.push(("single_stream_modulation.lin.weight".into(), vec![3 * d, d]));
        for n in 0..self.depth_double {
            for s in ["img", "txt"] {
                let p = format!("double_blocks.{n}.{s}");
                v.push((format!("{p}_attn.qkv.weight"), vec![3 * d, d]));
                v.push((format!("{p}_attn.norm.query_norm.scale"), vec![hd]));
                v.push((format!("{p}_attn.norm.key_norm.scale"), vec![hd]));
                v.push((format!("{p}_attn.proj.weight"), vec![d, d]));
                v.push((format!("{p}_mlp.0.weight"), vec![2 * mlp, d]));
                v.push((format!("{p}_mlp.2.weight"), vec![d, mlp]));
            }
        }
        for n in 0..self.depth_single {
            let p = format!("single_blocks.{n}");
            v.push((format!("{p}.linear1.weight"), vec![3 * d + 2 * mlp, d]));
            v.push((format!("{p}.linear2.weight"), vec![d, d + mlp]));
            v.push((format!("{p}.norm.query_norm.scale"), vec![hd]));
            v.push((format!("{p}.norm.key_norm.scale"), vec![hd]));
        }
        v.push(("final_layer.adaLN_modulation.1.weight".into(), vec![2 * d, d]));
        v.push(("final_layer.linear.weight".into(), vec![self.in_channels, d]));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_counts_match_the_reference_checkpoints() {
        // BFL klein-4b: 149 tensors; klein-9b: 201; dev-shaped (guidance): +2.
        assert_eq!(Flux2Config::klein_4b().tensor_manifest().len(), 149);
        assert_eq!(Flux2Config::klein_9b().tensor_manifest().len(), 201);
        let dev_ish = Flux2Config { guidance_embed: true, ..Flux2Config::klein_9b() };
        assert_eq!(dev_ish.tensor_manifest().len(), 203);
    }

    #[test]
    fn derived_dims() {
        let c4 = Flux2Config::klein_4b();
        assert_eq!(c4.head_dim(), 128);
        assert_eq!(c4.mlp_hidden(), 9216);
        assert_eq!(c4.axes_dim.iter().sum::<usize>(), c4.head_dim());
        let c9 = Flux2Config::klein_9b();
        assert_eq!(c9.head_dim(), 128);
        assert_eq!(c9.mlp_hidden(), 12288);
        // base differs only in the sampling recipe
        assert_eq!(
            Flux2Config::klein_base_9b().tensor_manifest(),
            c9.tensor_manifest()
        );
    }
}
