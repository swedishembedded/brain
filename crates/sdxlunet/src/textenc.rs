// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SDXL's dual CLIP text conditioning, and the two file-reading helpers every
//! SDXL-family pipeline needs to load one.
//!
//! [`TextEncoders`] holds the two CLIP towers' tokenizers and the device/root
//! they are built against; [`TextEncoders::encode_all`] builds both towers,
//! encodes every prompt, and drops them, so the conditional and unconditional
//! passes never pay for 3.3 GB of encoder twice. `crate::pipeline::Sdxl` and
//! `controlnet::caps::Controlled` both use this - it used to be two
//! byte-identical copies, one per crate.
//!
//! # SDXL's conditioning is two encoders, and the layer index matters
//!
//! `prompt_embeds` is `concat(CLIP-L penultimate, OpenCLIP-bigG penultimate)`
//! along the feature axis - 768 + 1280 = 2048 - and `pooled_prompt_embeds` is
//! bigG's **projected** `text_embeds` alone. The PENULTIMATE hidden state, not
//! the last: diffusers passes `output_hidden_states=True` and takes
//! `hidden_states[-2]`. Taking the last layer instead runs, produces an image,
//! and is not SDXL.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use clip::config::ClipTextConfig;
use clip::model::ClipText;
use gpu_core::Gpu;

/// SDXL's context length: both CLIP towers are padded/truncated to 77 tokens.
pub const CONTEXT: usize = 77;

/// One prompt's SDXL conditioning: the `77 x 2048` sequence and the 1280-d
/// pooled vector.
pub type Conditioning = (Vec<f32>, Vec<f32>);

/// Read every `*.safetensors` in `dir`. The diffusers layout names a component's
/// weights after the component (`diffusion_pytorch_model[.fp16].safetensors`),
/// and a variant suffix is normal - so match on the extension, not the stem.
pub fn read_any_safetensors(dir: &Path) -> Result<Vec<checkpoint::safetensors::StTensor>, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("sdxl: reading {}: {e}", dir.display()))?;
    let mut files: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("sdxl: no *.safetensors under {}", dir.display()));
    }
    let mut out = Vec::new();
    for f in files {
        out.extend(checkpoint::safetensors::read(f.to_str().ok_or("sdxl: non-UTF8 path")?)?);
    }
    Ok(out)
}

pub fn read_json(p: &Path) -> Result<serde_json::Value, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("sdxl: reading {}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("sdxl: parsing {}: {e}", p.display()))
}

/// The two SDXL CLIP towers' tokenizers, plus the device and checkpoint root
/// each encode call rebuilds the towers from.
///
/// # Only the tokenizers stay resident
///
/// A tower is ~3.3 GB (CLIP-L) or ~10.2 GB (OpenCLIP-bigG) at fp32 and is
/// needed only for the handful of encode calls at the start of a generation,
/// so [`Self::encode_all`] builds both, encodes every prompt, and drops them -
/// never held alongside the UNet, which is what a caller decides to keep
/// resident across the whole denoise loop.
pub struct TextEncoders {
    gpu: Gpu,
    root: PathBuf,
    tok_l: data::clip_bpe::ClipBpe,
    tok_g: data::clip_bpe::ClipBpe,
}

impl TextEncoders {
    /// `gpu` should already be a fresh handle for this purpose (`Gpu::share`
    /// at the call site) - [`Self::tower`] builds a further `new_like` handle
    /// per encode, on the same device.
    pub fn load(gpu: Gpu, root: impl Into<PathBuf>) -> Result<TextEncoders, String> {
        let root = root.into();
        let tok_l = data::clip_bpe::ClipBpe::from_dir(&root.join("tokenizer")).map_err(|e| format!("sdxl: tokenizer: {e}"))?;
        let tok_g = data::clip_bpe::ClipBpe::from_dir(&root.join("tokenizer_2")).map_err(|e| format!("sdxl: tokenizer_2: {e}"))?;
        Ok(TextEncoders { gpu, root, tok_l, tok_g })
    }

    fn tower(&self, sub: &str, cfg: &ClipTextConfig) -> Result<ClipText, String> {
        let t = clip::import::read_text_encoder(&self.root.join(sub))?;
        let init = clip::import::import_text(t, cfg)?;
        let map: HashMap<String, Vec<f32>> = init.into_iter().map(|(k, (_, d))| (k, d)).collect();
        // `new_like`: a DIFFERENT kernel set on the SAME device. Each crate
        // resolves kernel indices against the list it registered, so building a
        // ClipText on a Gpu made from sdxlunet::KERNELS binds the wrong pipelines.
        Ok(ClipText::new_on(self.gpu.new_like(clip::model::TEXT_PIPELINES), cfg.clone(), 1, CONTEXT as u32, &map))
    }

    /// Encode every prompt in one pass, so the towers are built and dropped ONCE
    /// rather than once per prompt.
    pub fn encode_all(&self, prompts: &[&str]) -> Result<Vec<Conditioning>, String> {
        let l_tower = self.tower("text_encoder", &ClipTextConfig::clip_l())?;
        let g_tower = self.tower("text_encoder_2", &ClipTextConfig::openclip_bigg())?;
        Ok(prompts.iter().map(|p| self.encode_with(&l_tower, &g_tower, p)).collect())
    }

    fn encode_with(&self, l_tower: &ClipText, g_tower: &ClipText, prompt: &str) -> Conditioning {
        l_tower.set_tokens(&self.tok_l.encode_with_context(prompt, CONTEXT).ids);
        l_tower.forward();
        let l = l_tower.read_penultimate();

        g_tower.set_tokens(&self.tok_g.encode_with_context(prompt, CONTEXT).ids);
        g_tower.forward();
        let g = g_tower.read_penultimate();
        let pooled = g_tower.read_text_embeds().unwrap_or_else(|| g_tower.read_pooled());

        let (dl, dg) = (l.len() / CONTEXT, g.len() / CONTEXT);
        let mut embeds = Vec::with_capacity(CONTEXT * (dl + dg));
        for t in 0..CONTEXT {
            embeds.extend_from_slice(&l[t * dl..(t + 1) * dl]);
            embeds.extend_from_slice(&g[t * dg..(t + 1) * dg]);
        }
        (embeds, pooled)
    }
}
