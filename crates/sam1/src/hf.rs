// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The SAM tower's upstream PyTorch tensor names**, and the one map from
//! them onto this crate's manifest ([`SamViTConfig::param_list`]).
//!
//! Every release that ships this tower as a `transformers`/PyTorch
//! safetensors checkpoint spells it the way the reference `ImageEncoderViT`
//! module's attribute path does, under its own prefix. The spellings differ
//! only in how the compressor (and, where it exists, the HD branch) is
//! named, so one map with a [`Spelling`] switch covers every release, and a
//! composite importer strips its own prefix and calls [`brain_name`] rather
//! than restating the ~20 leaf names.
//!
//! No transpose, slice or fusion anywhere: every upstream tensor has the
//! element order brain's graph reads.
//!
//! Swedish Embedded AB implements checkpoint importers like this one for its
//! clients. If your team needs a published vision model brought onto its own
//! inference stack with every tensor accounted for, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::collections::HashMap;

use checkpoint::remap::{Fetch, RemapSource};
use checkpoint::TensorSource;

use crate::config::{SamViTConfig, HD_ALPHA};

/// Which release's spelling of the tower's non-block tensors a checkpoint uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spelling {
    /// `deepseek-ai/DeepSeek-OCR` (`model.sam_model.*`): the two compressor
    /// convs are `net_2` / `net_3`.
    DeepseekOcr,
    /// `deepseek-ai/deepseek-vl-*`'s high-resolution tower
    /// (`vision_model.vision_tower_high.vision_tower.*`, `sam_b_downsample`):
    /// the compressor convs are `downsamples.0` / `downsamples.1`, and the HD
    /// branch adds `neck_hd.*` and `hd_alpha_downsamples`.
    DeepseekVl,
}

/// The brain-side name of one upstream SAM tensor.
///
/// `leaf` is the name with the checkpoint's own tower prefix already stripped
/// (`blocks.3.mlp.lin1.weight`, not `model.sam_model.blocks.3.mlp.lin1.weight`).
/// An unrecognized leaf, or a block index outside `cfg.n_layers`, is an error:
/// a checkpoint that grew a tensor must stop the import, never load a model
/// missing it.
pub fn brain_name(leaf: &str, cfg: &SamViTConfig, spelling: Spelling) -> Result<String, String> {
    let unknown = || format!("unrecognized SAM tensor {leaf:?}");
    let p = |s: &str| Ok(format!("vision.sam.{s}"));
    match leaf {
        "pos_embed" => return p("pos_embed"),
        "patch_embed.proj.weight" => return p("patch_embed.weight"),
        "patch_embed.proj.bias" => return p("patch_embed.bias"),
        _ => {}
    }
    if let Some(rest) = leaf.strip_prefix("neck.") {
        return neck_leaf(rest, "neck").ok_or_else(unknown);
    }
    // The HD branch's tensors exist only in a config that runs the branch; in
    // any other they are a tensor the graph would silently not read.
    if spelling == Spelling::DeepseekVl && cfg.hd_branch {
        if let Some(rest) = leaf.strip_prefix("neck_hd.") {
            return neck_leaf(rest, "neck_hd").ok_or_else(unknown);
        }
        if leaf == "hd_alpha_downsamples" {
            return Ok(HD_ALPHA.to_string());
        }
    }
    let compress = match (spelling, leaf) {
        (Spelling::DeepseekOcr, "net_2.weight") | (Spelling::DeepseekVl, "downsamples.0.weight") => Some("compress.conv1.weight"),
        (Spelling::DeepseekOcr, "net_3.weight") | (Spelling::DeepseekVl, "downsamples.1.weight") => Some("compress.conv2.weight"),
        _ => None,
    };
    if let Some(c) = compress {
        return p(c);
    }
    let rest = leaf.strip_prefix("blocks.").ok_or_else(unknown)?;
    let (idx, block_leaf) = rest.split_once('.').ok_or_else(unknown)?;
    let l: u32 = idx.parse().map_err(|_| format!("malformed SAM block index in {leaf:?}"))?;
    if l >= cfg.n_layers {
        return Err(format!("{leaf}: SAM block index {l} beyond {}", cfg.n_layers));
    }
    let b = |s: &str| Ok(format!("vision.sam.blocks.{l}.{s}"));
    match block_leaf {
        "norm1.weight" | "norm1.bias" | "norm2.weight" | "norm2.bias" => b(block_leaf),
        "attn.qkv.weight" | "attn.qkv.bias" | "attn.rel_pos_h" | "attn.rel_pos_w" => b(block_leaf),
        "attn.proj.weight" | "attn.proj.bias" => b(block_leaf),
        "mlp.lin1.weight" => b("mlp.fc1.weight"),
        "mlp.lin1.bias" => b("mlp.fc1.bias"),
        "mlp.lin2.weight" => b("mlp.fc2.weight"),
        "mlp.lin2.bias" => b("mlp.fc2.bias"),
        _ => Err(unknown()),
    }
}

/// A neck is an `nn.Sequential(conv, LayerNorm2d, conv, LayerNorm2d)`, so
/// upstream indexes its parts positionally and brain names them.
fn neck_leaf(rest: &str, neck: &str) -> Option<String> {
    let part = match rest {
        "0.weight" => "conv1.weight",
        "1.weight" => "norm1.weight",
        "1.bias" => "norm1.bias",
        "2.weight" => "conv2.weight",
        "3.weight" => "norm2.weight",
        "3.bias" => "norm2.bias",
        _ => return None,
    };
    Some(format!("vision.sam.{neck}.{part}"))
}

/// The fetch plan for the tower inside a checkpoint whose SAM tensors all sit
/// under `prefix`, keyed by brain-side name.
///
/// Names outside `prefix` belong to other towers and are ignored; every name
/// inside it must map ([`brain_name`]), and no brain-side name may be produced
/// twice. Coverage of the manifest is checked by [`source`].
pub fn plan<'a>(names: impl IntoIterator<Item = &'a str>, prefix: &str, cfg: &SamViTConfig, spelling: Spelling) -> Result<HashMap<String, Fetch>, String> {
    let mut out = HashMap::new();
    for name in names {
        let Some(leaf) = name.strip_prefix(prefix) else { continue };
        let brain = brain_name(leaf, cfg, spelling).map_err(|e| format!("sam1 hf import: {name}: {e}"))?;
        if let Some(Fetch::Whole(prev)) = out.insert(brain.clone(), Fetch::Whole(name.to_string())) {
            return Err(format!("sam1 hf import: {prev} and {name} both map to {brain}"));
        }
    }
    Ok(out)
}

/// The tower's weights as a streaming [`TensorSource`] under brain's names,
/// borrowing `src` -- what [`crate::SamEncoder::new_on`] takes.
///
/// Two-way coverage: every upstream tensor under `prefix` maps ([`plan`]), and
/// every tensor of `cfg`'s manifest is produced at its declared size.
pub fn source<'a>(src: &'a dyn TensorSource, names: impl IntoIterator<Item = &'a str>, prefix: &str, cfg: &SamViTConfig, spelling: Spelling) -> Result<RemapSource<'a>, String> {
    let remap = RemapSource::new(src, plan(names, prefix, cfg, spelling)?);
    remap.validate(&cfg.param_list()).map_err(|e| format!("sam1 hf import ({prefix}): {e}"))?;
    Ok(remap)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The DeepSeek-OCR release's SAM header, written as an EMITTER so a typo
    /// here and one in the MATCHER above fail the coverage test rather than
    /// cancel out. Shapes from the real `model.sam_model.*` tensors.
    fn ocr_header() -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = vec![
            ("pos_embed".into(), 64 * 64 * 768),
            ("patch_embed.proj.weight".into(), 768 * 3 * 16 * 16),
            ("patch_embed.proj.bias".into(), 768),
        ];
        for l in 0..12 {
            let rows = if [2, 5, 8, 11].contains(&l) { 127 } else { 27 };
            for (leaf, n) in [
                ("norm1.weight", 768),
                ("norm1.bias", 768),
                ("attn.rel_pos_h", rows * 64),
                ("attn.rel_pos_w", rows * 64),
                ("attn.qkv.weight", 2304 * 768),
                ("attn.qkv.bias", 2304),
                ("attn.proj.weight", 768 * 768),
                ("attn.proj.bias", 768),
                ("norm2.weight", 768),
                ("norm2.bias", 768),
                ("mlp.lin1.weight", 3072 * 768),
                ("mlp.lin1.bias", 3072),
                ("mlp.lin2.weight", 768 * 3072),
                ("mlp.lin2.bias", 768),
            ] {
                v.push((format!("blocks.{l}.{leaf}"), n));
            }
        }
        for (leaf, n) in [("0.weight", 256 * 768), ("1.weight", 256), ("1.bias", 256), ("2.weight", 256 * 256 * 9), ("3.weight", 256), ("3.bias", 256)] {
            v.push((format!("neck.{leaf}"), n));
        }
        v.push(("net_2.weight".into(), 512 * 256 * 9));
        v.push(("net_3.weight".into(), 1024 * 512 * 9));
        v
    }

    /// Map an upstream header onto `cfg`'s manifest and demand an exact
    /// bijection with matching element counts.
    fn assert_bijection(header: &[(String, usize)], cfg: &SamViTConfig, spelling: Spelling) {
        let mut got: HashMap<String, usize> = HashMap::new();
        for (leaf, n) in header {
            let brain = brain_name(leaf, cfg, spelling).unwrap_or_else(|e| panic!("{e}"));
            assert!(got.insert(brain.clone(), *n).is_none(), "two upstream tensors map to {brain}");
        }
        let want: HashMap<String, usize> = cfg.param_list().into_iter().collect();
        assert_eq!(got, want, "the upstream header and the manifest must be the same set at the same sizes");
    }

    #[test]
    fn the_deepseek_ocr_spelling_covers_the_ocr_manifest_exactly() {
        let header = ocr_header();
        assert_eq!(header.len(), 179, "the real release carries 179 SAM tensors");
        assert_bijection(&header, &SamViTConfig::deepseek_ocr(), Spelling::DeepseekOcr);
    }

    /// The DeepSeek-VL high-resolution tower's header: DeepSeek-OCR's with the
    /// compressor renamed, plus the HD branch's seven tensors.
    fn vl_header() -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = ocr_header()
            .into_iter()
            .map(|(n, s)| match n.as_str() {
                "net_2.weight" => ("downsamples.0.weight".to_string(), s),
                "net_3.weight" => ("downsamples.1.weight".to_string(), s),
                _ => (n, s),
            })
            .collect();
        for (leaf, n) in [("0.weight", 256 * 768), ("1.weight", 256), ("1.bias", 256), ("2.weight", 256 * 256 * 9), ("3.weight", 256), ("3.bias", 256)] {
            v.push((format!("neck_hd.{leaf}"), n));
        }
        v.push(("hd_alpha_downsamples".into(), 1));
        v
    }

    #[test]
    fn the_deepseek_vl_spelling_covers_the_vl_manifest_exactly() {
        let header = vl_header();
        assert_eq!(header.len(), 186, "the real release carries 186 high-resolution tower tensors");
        assert_bijection(&header, &SamViTConfig::deepseek_vl(), Spelling::DeepseekVl);
    }

    /// An HD tensor in a checkpoint loaded without the HD branch is refused,
    /// not dropped: the graph would never read it.
    #[test]
    fn hd_tensors_need_a_config_that_runs_the_hd_branch() {
        let no_hd = SamViTConfig { hd_branch: false, ..SamViTConfig::deepseek_vl() };
        for leaf in ["neck_hd.0.weight", "hd_alpha_downsamples"] {
            assert!(brain_name(leaf, &no_hd, Spelling::DeepseekVl).is_err(), "{leaf}");
            assert!(brain_name(leaf, &SamViTConfig::deepseek_ocr(), Spelling::DeepseekOcr).is_err(), "{leaf}");
        }
    }

    /// Header-only coverage against the REAL checkpoint when it is in the
    /// model store: every tensor under the tower prefix maps and every
    /// manifest entry is produced at its size.
    #[test]
    fn the_real_deepseek_vl_header_covers_the_manifest() {
        const REPO: &str = "deepseek-ai/deepseek-vl-7b-chat";
        let Some(dir) = brain_testutil::model_dir(REPO).filter(|d| std::path::Path::new(d).join("model.safetensors.index.json").exists()) else {
            return brain_testutil::skip(&format!("{REPO} not in the model store"));
        };
        let reader = checkpoint::weightio::WeightReader::open_hf_dir(std::path::Path::new(&dir)).expect("open checkpoint");
        let cfg = SamViTConfig::deepseek_vl();
        let prefix = "vision_model.vision_tower_high.vision_tower.";
        let n = reader.names().filter(|n| n.starts_with(prefix)).count();
        assert_eq!(n, cfg.param_list().len(), "tensors under {prefix}");
        source(&reader, reader.names(), prefix, &cfg, Spelling::DeepseekVl).unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn an_unknown_leaf_or_an_out_of_range_block_is_refused() {
        let cfg = SamViTConfig::deepseek_ocr();
        for leaf in ["blocks.0.attn.bias", "blocks.12.norm1.weight", "neck.4.weight", "downsamples.0.weight", "blocks.x.norm1.weight"] {
            assert!(brain_name(leaf, &cfg, Spelling::DeepseekOcr).is_err(), "{leaf} must not map");
        }
    }
}
