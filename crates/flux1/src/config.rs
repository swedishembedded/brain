// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.1 model configuration and the canonical (BFL-named) tensor manifest.
//!
//! FLUX.1 is FLUX.2's *ancestor*: the same double-stream → single-stream MMDiT
//! skeleton, differing in exactly four bounded ways (each verified against
//! `black-forest-labs/flux` `src/flux/modules/layers.py`, the architecture
//! authority):
//!
//! 1. **Per-block modulation.** Every double block owns two `Modulation`
//!    linears (`img_mod`, `txt_mod`, `[6D, D]`) and every single block owns one
//!    (`[3D, D]`) — where FLUX.2 hoisted three *global* ones. That is ~3.2 B of
//!    FLUX.1-dev's 11.9 B parameters.
//! 2. **3 RoPE axes at theta 10000** (`axes_dim = [16, 56, 56]`), not 4 at 2000.
//! 3. **T5-XXL sequence + CLIP-L pooled** conditioning: `txt_in` reads a
//!    `[txt, 4096]` T5 sequence and `vector_in` reads a `[768]` CLIP pooled
//!    vector into the modulation vector, alongside `time_in` and (dev/Kontext
//!    only) `guidance_in`.
//! 4. **16-channel VAE latent** → `in_channels = 16 * 2 * 2 = 64` after the
//!    2×2 patchify, not 128.
//!
//! Two further differences follow from the same generation gap and matter to
//! the kernels: the MLPs are plain **GELU(tanh)** two-layer stacks, not SwiGLU
//! (so `mlp_hidden = 4 D` and `linear1` emits `3D + mlp`, not `3D + 2·mlp`),
//! and **every linear is biased** (FLUX.2's are bias-free).

/// Architecture hyperparameters for one FLUX.1 variant.
///
/// The text length is deliberately **not** here: FLUX.1's T5 sequence length is
/// a pipeline argument (`max_sequence_length`, 256 or 512), so the forward
/// derives it from the conditioning it is handed.
#[derive(Clone, Debug, PartialEq)]
pub struct Flux1Config {
    /// Latent channels entering `img_in` (VAE 16ch × 2×2 patchify).
    pub in_channels: usize,
    /// Text-conditioning width: the T5-XXL encoder hidden size.
    pub context_in_dim: usize,
    /// Pooled-vector width entering `vector_in`: the CLIP-L projection dim.
    pub vec_in_dim: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub depth_double: usize,
    pub depth_single: usize,
    /// GELU MLP inner width = `hidden * mlp_ratio`.
    pub mlp_ratio: f32,
    /// Per-axis RoPE dims in (t, h, w) order; sums to `head_dim()`.
    pub axes_dim: [usize; 3],
    pub rope_theta: f64,
    /// LayerNorm and QK-RMSNorm epsilon (1e-6 in every variant).
    pub norm_eps: f32,
    /// `guidance_in` MLP present (dev + Kontext: true; schnell: false).
    pub guidance_embed: bool,
}

impl Flux1Config {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }

    /// GELU MLP inner width (the `mlp_hidden_dim` of the reference).
    pub fn mlp_hidden(&self) -> usize {
        (self.hidden as f32 * self.mlp_ratio) as usize
    }

    /// The 12 B guidance-distilled model: `FLUX.1-dev`.
    pub fn dev() -> Flux1Config {
        Flux1Config {
            in_channels: 64,
            context_in_dim: 4096,
            vec_in_dim: 768,
            hidden: 3072,
            n_heads: 24,
            depth_double: 19,
            depth_single: 38,
            mlp_ratio: 4.0,
            axes_dim: [16, 56, 56],
            rope_theta: 10000.0,
            norm_eps: 1e-6,
            guidance_embed: true,
        }
    }

    /// `FLUX.1-Kontext-dev` — byte-identical topology to [`Flux1Config::dev`];
    /// Kontext is a *training* difference (reference-image tokens appended with
    /// an axis-0 offset), not an architecture one.
    pub fn kontext_dev() -> Flux1Config {
        Flux1Config::dev()
    }

    /// `FLUX.1-schnell`: timestep-distilled, **no** guidance embedding.
    pub fn schnell() -> Flux1Config {
        Flux1Config { guidance_embed: false, ..Flux1Config::dev() }
    }

    /// Resolve a user-facing variant name — the ONE name→config map.
    pub fn from_name(v: &str) -> Result<Flux1Config, String> {
        Ok(match v {
            "dev" => Flux1Config::dev(),
            "kontext-dev" => Flux1Config::kontext_dev(),
            "schnell" => Flux1Config::schnell(),
            other => {
                return Err(format!("unknown variant {other} (dev|kontext-dev|schnell)"))
            }
        })
    }

    /// A reduced-depth copy for gating the math on hardware that cannot hold
    /// the full 12 B model. Goldens must be dumped at the SAME depth
    /// (`tools/flux1_dump_reference.py --small-double/--small-single`).
    pub fn with_depth(&self, depth_double: usize, depth_single: usize) -> Flux1Config {
        Flux1Config { depth_double, depth_single, ..self.clone() }
    }

    /// The full canonical tensor manifest: BFL names with shapes, in a fixed
    /// order. Import validates full two-way coverage against this list.
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let d = self.hidden;
        let hd = self.head_dim();
        let mlp = self.mlp_hidden();
        let mut v: Vec<(String, Vec<usize>)> = Vec::new();
        // a biased linear contributes `.weight [out, k]` and `.bias [out]`
        fn lin(v: &mut Vec<(String, Vec<usize>)>, name: &str, out: usize, k: usize) {
            v.push((format!("{name}.weight"), vec![out, k]));
            v.push((format!("{name}.bias"), vec![out]));
        }
        macro_rules! lin {
            ($name:expr, $out:expr, $k:expr) => {
                lin(&mut v, &$name, $out, $k)
            };
        }
        lin!("img_in", d, self.in_channels);
        lin!("txt_in", d, self.context_in_dim);
        lin!("time_in.in_layer", d, 256);
        lin!("time_in.out_layer", d, d);
        lin!("vector_in.in_layer", d, self.vec_in_dim);
        lin!("vector_in.out_layer", d, d);
        if self.guidance_embed {
            lin!("guidance_in.in_layer", d, 256);
            lin!("guidance_in.out_layer", d, d);
        }
        for n in 0..self.depth_double {
            for s in ["img", "txt"] {
                let p = format!("double_blocks.{n}.{s}");
                lin!(format!("{p}_mod.lin"), 6 * d, d);
                lin!(format!("{p}_attn.qkv"), 3 * d, d);
                v.push((format!("{p}_attn.norm.query_norm.scale"), vec![hd]));
                v.push((format!("{p}_attn.norm.key_norm.scale"), vec![hd]));
                lin!(format!("{p}_attn.proj"), d, d);
                lin!(format!("{p}_mlp.0"), mlp, d);
                lin!(format!("{p}_mlp.2"), d, mlp);
            }
        }
        for n in 0..self.depth_single {
            let p = format!("single_blocks.{n}");
            lin!(format!("{p}.modulation.lin"), 3 * d, d);
            lin!(format!("{p}.linear1"), 3 * d + mlp, d);
            lin!(format!("{p}.linear2"), d, d + mlp);
            v.push((format!("{p}.norm.query_norm.scale"), vec![hd]));
            v.push((format!("{p}.norm.key_norm.scale"), vec![hd]));
        }
        lin!("final_layer.adaLN_modulation.1", 2 * d, d);
        lin!("final_layer.linear", self.in_channels, d);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_counts_match_the_reference_checkpoints() {
        // BFL FLUX.1-dev / Kontext-dev: 780 tensors after q/k/v(+mlp) fusion;
        // schnell drops the 4 guidance_in tensors.
        assert_eq!(Flux1Config::dev().tensor_manifest().len(), 780);
        assert_eq!(Flux1Config::kontext_dev().tensor_manifest().len(), 780);
        assert_eq!(Flux1Config::schnell().tensor_manifest().len(), 776);
    }

    #[test]
    fn derived_dims() {
        let c = Flux1Config::dev();
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.mlp_hidden(), 12288);
        assert_eq!(c.axes_dim.iter().sum::<usize>(), c.head_dim());
        // every width the model binds a storage sub-range at must be a
        // multiple of 64 floats (256-byte min_storage_buffer_offset_alignment)
        for w in [c.hidden, c.mlp_hidden(), c.in_channels, 3 * c.hidden + c.mlp_hidden()] {
            assert_eq!(w % 64, 0, "width {w} breaks storage-binding alignment");
        }
    }

    #[test]
    fn reduced_depth_keeps_everything_else() {
        let small = Flux1Config::dev().with_depth(2, 2);
        assert_eq!(small.depth_double, 2);
        assert_eq!(small.depth_single, 2);
        assert_eq!(small.hidden, 3072);
        // 20 fixed + 2*24 + 2*8
        assert_eq!(small.tensor_manifest().len(), 20 + 48 + 16);
    }
}
