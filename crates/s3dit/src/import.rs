// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import the original / Comfy Z-Image checkpoint layout into brain's
//! (diffusers-named) tensor map.
//!
//! The shipped weights (`z_image_turbo_bf16.safetensors`) use the original
//! naming: a fused `attention.qkv.weight`, `attention.out`, `q_norm`/`k_norm`,
//! top-level `x_embedder`/`final_layer`. brain's model uses the diffusers names
//! ([`crate::ZImageModel`]). This reverses the official
//! `z_image_convert_original_to_comfy.py` map (verified against it):
//!   - `attention.qkv.weight` → split (chunk-3, q|k|v row-blocks) into
//!     `attention.{to_q,to_k,to_v}.weight`;
//!   - `attention.out` → `attention.to_out.0`;
//!   - `attention.{q,k}_norm.weight` → `attention.norm_{q,k}.weight`;
//!   - `x_embedder.` → `all_x_embedder.{ps}-{pf}.`;
//!   - `final_layer.` → `all_final_layer.{ps}-{pf}.`.

use std::collections::HashMap;

use checkpoint::gguf::MmapGguf;
use checkpoint::safetensors::StTensor;
use checkpoint::st::ModelCard;

use crate::block::Tensors;
use crate::grad::WeightsF32;
use crate::modelgrad::ModelWeightsF32;
use crate::model::ZImageConfig;

/// The one place the Comfy → brain rename rules live, so the eager
/// [`import_comfy`], the streaming [`comfy_source`] and the GGUF importer
/// cannot drift apart. Pure renaming: the fused `qkv.weight` needs a SPLIT and
/// is handled by each caller in the way its data access allows (a copy, a
/// zero-copy slice fetch, or a slice of one dequantized buffer).
///
/// `xk`/`fk` are the patch-size-qualified prefixes
/// (`all_x_embedder.{ps}-{pf}.` / `all_final_layer.{ps}-{pf}.`) - passed in
/// rather than re-derived per tensor, since they are the same for every name in
/// one checkpoint.
fn comfy_rename(name: &str, xk: &str, fk: &str) -> String {
    let mut k = name.replace(".attention.out.", ".attention.to_out.0.");
    k = k.replace(".attention.k_norm.weight", ".attention.norm_k.weight");
    k = k.replace(".attention.q_norm.weight", ".attention.norm_q.weight");
    if let Some(rest) = k.strip_prefix("x_embedder.") {
        k = format!("{xk}{rest}");
    } else if let Some(rest) = k.strip_prefix("final_layer.") {
        k = format!("{fk}{rest}");
    }
    k
}

/// Remap original/Comfy Z-Image tensors → brain's diffusers-named map. Splits
/// the fused qkv (Z-Image is full MHA, so q/k/v are equal `dim`-row thirds).
pub fn import_comfy(tensors: Vec<StTensor>, cfg: &ZImageConfig) -> Tensors {
    let xk = format!("all_x_embedder.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let fk = format!("all_final_layer.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let dim = cfg.dim as usize;
    let mut out = Tensors::new();
    for t in tensors {
        if let Some(base) = t.name.strip_suffix("qkv.weight") {
            // base = "…attention." ; split [3·dim, dim] into q|k|v.
            let dd = dim * dim;
            assert_eq!(t.data.len(), 3 * dd, "qkv {} has {} != 3·dim² elems", t.name, t.data.len());
            out.insert(format!("{base}to_q.weight"), (vec![dim, dim], t.data[0..dd].to_vec()));
            out.insert(format!("{base}to_k.weight"), (vec![dim, dim], t.data[dd..2 * dd].to_vec()));
            out.insert(format!("{base}to_v.weight"), (vec![dim, dim], t.data[2 * dd..3 * dd].to_vec()));
            continue;
        }
        let k = comfy_rename(&t.name, &xk, &fk);
        out.insert(k, (t.shape, t.data));
    }
    out
}

/// The width of the sinusoidal timestep features, and of `t_embedder`'s hidden
/// layer. `model::timestep_cond` hardcodes the same two numbers (Z-Image's
/// `TimestepEmbedder` is `sinusoid(256) -> Linear(256, 1024) -> SiLU ->
/// Linear(1024, cdim)`), so the manifest names them rather than inventing
/// [`ZImageConfig`] fields the forward would not read.
const T_FREQ_DIM: usize = 256;
const T_HIDDEN_DIM: usize = 1024;

/// Append one transformer block's tensors, in brain's (post-rename) spelling.
///
/// `modulated` is false for exactly the `context_refiner` blocks - the caption
/// stream carries no timestep conditioning, so those blocks have no
/// `adaLN_modulation`, which is why `block::NormBufs::new` and
/// [`model_weights_from_comfy`] both take the same flag.
fn push_block(v: &mut Vec<(String, Vec<usize>)>, prefix: &str, cfg: &ZImageConfig, modulated: bool) {
    let dim = cfg.dim as usize;
    let cdim = dim.min(256);
    let hidden = dim * 8 / 3;
    let head_dim = dim / cfg.n_heads as usize;
    let mut p = |leaf: &str, shape: Vec<usize>| v.push((format!("{prefix}.{leaf}"), shape));
    for leaf in ["attention.to_q.weight", "attention.to_k.weight", "attention.to_v.weight", "attention.to_out.0.weight"] {
        p(leaf, vec![dim, dim]);
    }
    p("attention.norm_q.weight", vec![head_dim]);
    p("attention.norm_k.weight", vec![head_dim]);
    p("feed_forward.w1.weight", vec![hidden, dim]);
    p("feed_forward.w2.weight", vec![dim, hidden]);
    p("feed_forward.w3.weight", vec![hidden, dim]);
    for leaf in ["attention_norm1.weight", "attention_norm2.weight", "ffn_norm1.weight", "ffn_norm2.weight"] {
        p(leaf, vec![dim]);
    }
    if modulated {
        // Four `dim`-wide modulations per block (scale/gate for attention and
        // for the FFN), projected from the `cdim`-wide timestep vector - the
        // exact split `block::fold_adaln` slices back out.
        p("adaLN_modulation.0.weight", vec![4 * dim, cdim]);
        p("adaLN_modulation.0.bias", vec![4 * dim]);
    }
}

/// Every tensor a Z-Image DiT checkpoint carries, in brain's spelling, with the
/// shape derived from `cfg` alone - the same role `wan::import::dit_manifest`
/// plays for Wan, and for the same reason: a name→shape list that needs no
/// weights is what lets [`import_gguf`] decide whether a file is the model it
/// claims to be BEFORE dequantizing a single block.
///
/// Verified against the released `Tongyi-MAI/Z-Image-Turbo` transformer, whose
/// 521 tensors this reproduces exactly, name for name and shape for shape.
///
/// Two of those 521 are `cap_pad_token` and `x_pad_token`, which brain's
/// forward never reads - Z-Image uses them to pad a batch's image and caption
/// streams to a common length, and brain runs one sequence at a time. They are
/// in the manifest because they are in the checkpoint: dropping them would make
/// a GGUF import produce a different tensor set than a safetensors import of
/// the same weights, and the point of a two-way check is that the two agree.
pub fn dit_manifest(cfg: &ZImageConfig) -> Vec<(String, Vec<usize>)> {
    let dim = cfg.dim as usize;
    let cdim = dim.min(256);
    let cap = cfg.cap_feat_dim as usize;
    let patch_dim = (cfg.in_channels * cfg.patch_size * cfg.patch_size * cfg.f_patch_size) as usize;
    let xk = format!("all_x_embedder.{}-{}", cfg.patch_size, cfg.f_patch_size);
    let fk = format!("all_final_layer.{}-{}", cfg.patch_size, cfg.f_patch_size);
    let mut v: Vec<(String, Vec<usize>)> = vec![
        ("cap_embedder.0.weight".to_string(), vec![cap]),
        ("cap_embedder.1.weight".to_string(), vec![dim, cap]),
        ("cap_embedder.1.bias".to_string(), vec![dim]),
        ("cap_pad_token".to_string(), vec![1, dim]),
        ("x_pad_token".to_string(), vec![1, dim]),
        (format!("{xk}.weight"), vec![dim, patch_dim]),
        (format!("{xk}.bias"), vec![dim]),
        ("t_embedder.mlp.0.weight".to_string(), vec![T_HIDDEN_DIM, T_FREQ_DIM]),
        ("t_embedder.mlp.0.bias".to_string(), vec![T_HIDDEN_DIM]),
        ("t_embedder.mlp.2.weight".to_string(), vec![cdim, T_HIDDEN_DIM]),
        ("t_embedder.mlp.2.bias".to_string(), vec![cdim]),
        (format!("{fk}.adaLN_modulation.1.weight"), vec![dim, cdim]),
        (format!("{fk}.adaLN_modulation.1.bias"), vec![dim]),
        (format!("{fk}.linear.weight"), vec![patch_dim, dim]),
        (format!("{fk}.linear.bias"), vec![patch_dim]),
    ];
    for l in 0..cfg.n_layers {
        push_block(&mut v, &format!("layers.{l}"), cfg, true);
    }
    for l in 0..cfg.n_refiner_layers {
        push_block(&mut v, &format!("noise_refiner.{l}"), cfg, true);
    }
    for l in 0..cfg.n_refiner_layers {
        push_block(&mut v, &format!("context_refiner.{l}"), cfg, false);
    }
    v
}

/// Check a name→shape view (brain's spelling) against [`dit_manifest`] in both
/// directions - shapes only, so the caller may run it on a GGUF header before
/// any tensor is decoded.
///
/// Element counts rather than exact shapes, the same latitude
/// `wan::import::validate_dit_shapes` gives: a repacker is free to store a 2-D
/// weight flattened, and the graph depends on how many values there are, not on
/// how the source chose to punctuate them. The declared shape is still what
/// gets reported, so a genuine width mismatch stays readable.
fn validate_dit_shapes(shapes: &HashMap<String, Vec<usize>>, cfg: &ZImageConfig) -> Result<(), String> {
    let manifest = dit_manifest(cfg);
    for (name, want) in &manifest {
        let Some(got) = shapes.get(name) else {
            return Err(format!("z-image dit import: missing tensor {name}"));
        };
        let n: usize = want.iter().product();
        if got.iter().product::<usize>() != n {
            return Err(format!("z-image dit import: {name} shape {got:?}, expected {want:?}"));
        }
    }
    if shapes.len() != manifest.len() {
        let expected: std::collections::HashSet<&str> = manifest.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&str> = shapes.keys().map(String::as_str).filter(|k| !expected.contains(k)).collect();
        extra.sort_unstable();
        return Err(format!("z-image dit import: unused source tensors: {extra:?}"));
    }
    Ok(())
}

/// Read the config off the checkpoint's own tensor shapes instead of assuming
/// [`ZImageConfig::turbo`], the way `wan::import::dit_config_from_shapes` reads
/// Wan's variant off `patch_embedding.weight`.
///
/// `shapes` is in the SOURCE (Comfy) spelling, i.e. what a GGUF header lists.
///
/// **What the weights can answer, and what they cannot.** `dim` and
/// `cap_feat_dim` come from `cap_embedder.1.weight`; `n_heads` from `dim`
/// divided by the `q_norm` width (Z-Image's per-head RMSNorm gain is one head's
/// worth); `n_layers` and `n_refiner_layers` by counting block prefixes. The
/// RoPE axes, `rope_theta`, `t_scale` and `norm_eps` appear in no tensor at
/// all, and `in_channels`/`patch_size`/`f_patch_size` only ever appear as their
/// product in `x_embedder.weight`'s second dimension - so those come from
/// [`ZImageConfig::turbo`] and the product is CHECKED against it. A file whose
/// patch geometry differs is refused rather than imported under the wrong
/// `all_x_embedder.{ps}-{pf}` name, which is what an assumed config would do
/// silently.
pub fn dit_config_from_shapes(shapes: &[(String, Vec<usize>)]) -> Result<ZImageConfig, String> {
    let get = |n: &str| shapes.iter().find(|(k, _)| k == n).map(|(_, s)| s.as_slice());
    let need = |n: &str| get(n).ok_or_else(|| format!("z-image dit import: no {n} to derive the config from"));

    let cap1 = need("cap_embedder.1.weight")?;
    if cap1.len() != 2 {
        return Err(format!("z-image dit import: cap_embedder.1.weight is {cap1:?}, expected [dim, cap_feat_dim]"));
    }
    let (dim, cap_feat_dim) = (cap1[0], cap1[1]);

    let qn = need("layers.0.attention.q_norm.weight")?;
    let head_dim: usize = qn.iter().product();
    if head_dim == 0 || !dim.is_multiple_of(head_dim) {
        return Err(format!("z-image dit import: dim {dim} is not a whole number of {head_dim}-wide heads"));
    }

    let count = |prefix: &str| shapes.iter().filter(|(k, _)| k.starts_with(prefix) && k.ends_with(".attention.qkv.weight")).count();
    let n_layers = count("layers.");
    let n_refiner_layers = count("noise_refiner.");
    let ctx = count("context_refiner.");
    if n_layers == 0 || n_refiner_layers == 0 {
        return Err(format!("z-image dit import: {n_layers} main and {n_refiner_layers} noise-refiner blocks, expected at least one of each"));
    }
    if ctx != n_refiner_layers {
        return Err(format!("z-image dit import: {n_refiner_layers} noise_refiner blocks but {ctx} context_refiner blocks"));
    }

    let mut cfg = ZImageConfig::turbo();
    cfg.dim = u32::try_from(dim).map_err(|_| format!("z-image dit import: implausible dim {dim}"))?;
    cfg.cap_feat_dim = u32::try_from(cap_feat_dim).map_err(|_| format!("z-image dit import: implausible cap_feat_dim {cap_feat_dim}"))?;
    cfg.n_heads = cfg.dim / head_dim as u32;
    cfg.n_layers = u32::try_from(n_layers).map_err(|_| format!("z-image dit import: implausible layer count {n_layers}"))?;
    cfg.n_refiner_layers = u32::try_from(n_refiner_layers).map_err(|_| format!("z-image dit import: implausible refiner count {n_refiner_layers}"))?;

    // The one thing that would otherwise be assumed silently: the patch
    // geometry is baked into the OUTPUT tensor NAMES, so getting it wrong
    // produces a checkpoint the model cannot find its embedder in.
    let xemb = need("x_embedder.weight")?;
    let patch_dim = (cfg.in_channels * cfg.patch_size * cfg.patch_size * cfg.f_patch_size) as usize;
    if xemb.iter().product::<usize>() != dim * patch_dim {
        return Err(format!(
            "z-image dit import: x_embedder.weight is {xemb:?}, not [{dim}, {patch_dim}] - this checkpoint's \
             in_channels/patch_size/f_patch_size are not ({}, {}, {}), and nothing in the weights says what they are",
            cfg.in_channels, cfg.patch_size, cfg.f_patch_size
        ));
    }
    Ok(cfg)
}

/// One brain tensor's values inside one source tensor: the whole of it, or the
/// `[start, start + len)` row-block a fused `qkv` contributes.
#[derive(Debug)]
struct Piece {
    out: String,
    shape: Vec<usize>,
    span: Option<(usize, usize)>,
}

/// The brain tensor(s) a single source tensor carries, with their output
/// shapes - [`comfy_rename`] plus the qkv split, expressed on SHAPES so a
/// caller can build the whole output plan from a GGUF header.
fn comfy_targets(name: &str, shape: &[usize], cfg: &ZImageConfig, xk: &str, fk: &str) -> Result<Vec<Piece>, String> {
    let dim = cfg.dim as usize;
    if let Some(base) = name.strip_suffix("qkv.weight") {
        let dd = dim * dim;
        let n: usize = shape.iter().product();
        if n != 3 * dd {
            return Err(format!("z-image dit import: {name} has {n} values, expected 3*{dim}^2 = {}", 3 * dd));
        }
        return Ok(["to_q", "to_k", "to_v"]
            .iter()
            .enumerate()
            .map(|(i, leaf)| Piece { out: format!("{base}{leaf}.weight"), shape: vec![dim, dim], span: Some((i * dd, dd)) })
            .collect());
    }
    Ok(vec![Piece { out: comfy_rename(name, xk, fk), shape: shape.to_vec(), span: None }])
}

/// The `general.architecture` unsloth's Z-Image Q8_0 GGUF release
/// (`unsloth/Z-Image-Turbo-GGUF`) declares. **Not a Z-Image-specific
/// spelling** - Z-Image is architecturally Lumina2-adjacent and unsloth
/// reused that tag for RoPE/metadata purposes, so on its own this cannot
/// distinguish a Z-Image GGUF from a genuine Lumina2 one the way
/// `crates/gguf::registry`'s `clip.projector_type` discriminator tells
/// DeepSeek-OCR's mmproj apart from every other CLIP-shaped GGUF.
/// `crates/cli/src/gguf_import.rs`'s `GgufArchitectureImporter` registry has
/// no such discriminator mechanism, so [`import_gguf`] carries its own guard
/// (see [`DISCRIMINATOR_TENSOR`]) instead of silently trusting the tag.
pub const GGUF_ARCHITECTURE: &str = "lumina2";

/// A tensor name unique to Z-Image's checkpoint (the caption-conditioning
/// embedder) and absent from a real Lumina2 release - [`import_gguf`]
/// requires it before proceeding, so a genuine Lumina2 GGUF reaching this
/// importer (mis-routed at the registry level, since both share
/// [`GGUF_ARCHITECTURE`]) fails loudly with a clear message instead of
/// silently producing a wrong-but-plausible checkpoint.
const DISCRIMINATOR_TENSOR: &str = "cap_embedder.0.weight";

/// Import a Z-Image GGUF (any block-quant this crate's `checkpoint::gguf`
/// dequant supports) into a brain-native single-file safetensors checkpoint
/// that `BRAIN_S3DIT_DIT` can point at directly - the same tensor names
/// [`import_comfy`] already remaps, since unsloth's GGUF conversion kept the
/// original/Comfy layout unchanged (`layers.N.attention.qkv.weight`,
/// `context_refiner.*`, `noise_refiner.*`, `t_embedder.*`, `x_embedder.*`,
/// `cap_embedder.*`, `final_layer.*`).
///
/// A released quantization is not one ggml type: unsloth's Q2_K leaves every
/// 1-D tensor at F32, keeps the refiner blocks and the wrapper linears at BF16,
/// and mixes Q2_K with Q4_K and Q5_K across the main blocks - so "which quant"
/// is a property of each tensor, never of the file.
///
/// **DiT-only.** The GGUF release does not bundle the VAE or the Qwen-4B text
/// encoder; `BRAIN_S3DIT_VAE`/`BRAIN_S3DIT_QWEN` still need their own source,
/// same as the safetensors path.
///
/// **Streaming, one source tensor at a time.** The names and shapes come from
/// the header, the config is derived from those shapes
/// ([`dit_config_from_shapes`]) and the two-way manifest check runs on them
/// alone - only then is each tensor dequantized from the mmap, written through
/// [`checkpoint::weightio::StWriter`] and dropped. Peak host memory is one
/// tensor's fp32 expansion (169 MiB, each block's fused
/// `attention.qkv.weight`, 11520x3840), not the whole model.
///
/// It used to hold about **three copies of the whole thing at once**: the DiT
/// is 6,154,908,736 parameters, so 24.6 GB as fp32, and the dequantized map was
/// live while [`checkpoint::st::save_safetensors`] copied each tensor into its
/// own little-endian `Vec<u8>` and `safetensors::serialize` concatenated those
/// into one more contiguous blob - about 74 GB, on an input whose whole reason
/// for arriving as an [`MmapGguf`] was to avoid even the first copy.
/// `crates/wan`'s importer had the identical shape and the identical fix.
///
/// The fused `qkv.weight` is what makes the loop walk SOURCE tensors rather
/// than the manifest: one dequant feeds all three of `to_q`/`to_k`/`to_v`,
/// where a manifest walk would decode the same 169 MiB three times.
/// `StWriter::write` takes its tensors in any order, so the output plan is laid
/// out in the GGUF's own order and every write is sequential.
pub fn import_gguf(mg: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
    if mg.shape(DISCRIMINATOR_TENSOR).is_none() {
        return Err(format!(
            "not a Z-Image checkpoint: missing tensor {DISCRIMINATOR_TENSOR:?} \
             (general.architecture={:?} is shared with real Lumina2 GGUFs, \
             which this importer refuses to guess at)",
            mg.kv().get("general.architecture")
        ));
    }
    let shapes: Vec<(String, Vec<usize>)> =
        mg.names().iter().map(|n| (n.clone(), mg.shape(n).map(<[usize]>::to_vec).unwrap_or_default())).collect();
    let cfg = dit_config_from_shapes(&shapes)?;

    // Source order, so the write loop is one pass over the mmap and one
    // forward pass over the output file.
    let xk = format!("all_x_embedder.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let fk = format!("all_final_layer.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let mut pieces: Vec<(String, Vec<Piece>)> = Vec::with_capacity(shapes.len());
    let mut out_shapes: HashMap<String, Vec<usize>> = HashMap::with_capacity(shapes.len());
    for (name, shape) in &shapes {
        let targets = comfy_targets(name, shape, &cfg, &xk, &fk)?;
        for p in &targets {
            if out_shapes.insert(p.out.clone(), p.shape.clone()).is_some() {
                return Err(format!("z-image dit import: two source tensors map to {}", p.out));
            }
        }
        pieces.push((name.clone(), targets));
    }
    validate_dit_shapes(&out_shapes, &cfg)?;
    let manifest: HashMap<String, Vec<usize>> = dit_manifest(&cfg).into_iter().collect();

    let id = id_override.unwrap_or("brain/s3dit-gguf");
    let mut card = ModelCard::new(id, "s3dit");
    card.param_count = Some(manifest.values().map(|s| s.iter().product::<usize>() as u64).sum());
    let config = serde_json::json!({
        "dim": cfg.dim,
        "n_layers": cfg.n_layers,
        "n_refiner_layers": cfg.n_refiner_layers,
        "n_heads": cfg.n_heads,
        "cap_feat_dim": cfg.cap_feat_dim,
        "in_channels": cfg.in_channels,
        "patch_size": cfg.patch_size,
        "f_patch_size": cfg.f_patch_size,
        "rope_theta": cfg.rope_theta,
        "t_scale": cfg.t_scale,
        "norm_eps": cfg.norm_eps,
    });
    // The MANIFEST shape is what gets declared, not the source's: a repacker is
    // free to store a weight flattened (the element count is all
    // `validate_dit_shapes` demands of it), and the checkpoint brain writes has
    // to name the rank its loader indexes with.
    let plan: Vec<(String, Vec<u64>)> = pieces
        .iter()
        .flat_map(|(_, targets)| targets.iter().map(|p| (p.out.clone(), manifest[&p.out].iter().map(|&d| d as u64).collect())))
        .collect();
    let mut w = checkpoint::weightio::StWriter::create(out_path, &plan, &config, Some(&card)).map_err(|e| e.to_string())?;
    for (name, targets) in &pieces {
        let data = mg
            .tensor(name)
            .ok_or_else(|| format!("{name}: missing tensor data"))?
            .map_err(|e| format!("{name}: dequant failed: {e}"))?;
        for p in targets {
            let slice = match p.span {
                Some((start, len)) => data.get(start..start + len).ok_or_else(|| format!("{name}: dequant produced {} values, too few to split", data.len()))?,
                None => &data[..],
            };
            w.write(&p.out, slice).map_err(|e| e.to_string())?;
        }
    }
    w.finish().map_err(|e| e.to_string())
}

/// The streaming sibling of [`import_comfy`]: a `checkpoint::remap::RemapSource`
/// over `r` resolving every brain (diffusers-named) tensor to its Comfy source
/// via the SAME rename/qkv-split rules - reading no tensor data up front. A
/// `qkv.weight` still resolves to three zero-copy [`checkpoint::remap::Fetch::Slice`]s
/// (slicing a borrow is still a borrow); every renamed tensor is a
/// [`checkpoint::remap::Fetch::Whole`] pass-through. `ZImageDit{,I8,Shard}::
/// build_from_source` accept the result directly, so a build from this never
/// materializes the whole DiT checkpoint on the host - peak allocation is one
/// tensor (up to ~157 MB for `feed_forward.w1`, once converted from BF16), not
/// the whole ~24 GB model.
pub fn comfy_source<'a>(r: &'a checkpoint::weightio::WeightReader, cfg: &ZImageConfig) -> checkpoint::remap::RemapSource<'a> {
    let xk = format!("all_x_embedder.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let fk = format!("all_final_layer.{}-{}.", cfg.patch_size, cfg.f_patch_size);
    let dim = cfg.dim as usize;
    let mut plan: HashMap<String, checkpoint::remap::Fetch> = HashMap::new();
    for name in r.names() {
        if let Some(base) = name.strip_suffix("qkv.weight") {
            // base = "…attention."; split [3·dim, dim] into q|k|v row-blocks -
            // three Slice fetches over the SAME source tensor, zero-copy.
            let dd = dim * dim;
            plan.insert(format!("{base}to_q.weight"), checkpoint::remap::Fetch::Slice { name: name.to_string(), start: 0, len: dd });
            plan.insert(format!("{base}to_k.weight"), checkpoint::remap::Fetch::Slice { name: name.to_string(), start: dd, len: dd });
            plan.insert(format!("{base}to_v.weight"), checkpoint::remap::Fetch::Slice { name: name.to_string(), start: 2 * dd, len: dd });
            continue;
        }
        plan.insert(comfy_rename(name, &xk, &fk), checkpoint::remap::Fetch::Whole(name.to_string()));
    }
    checkpoint::remap::RemapSource::new(r, plan)
}

/// Bridge the (already-`import_comfy`'d) inference tensor map into the **training**
/// weight format [`ModelWeightsF32`] - the piece that lets a real shipped Z-Image
/// checkpoint be fine-tuned (LoRA or full). The inference path and this share one
/// source of truth: the exact same tensor keys the inference model reads
/// ([`crate::block::BlockWeights`]/[`NormBufs`], `model.rs` embedders/final) are
/// read here. Blocks stay f32 (the 24 GB runtime type); the ~100 MB wrapper linears
/// widen to f64 (where the host reference math runs). Errors name the first missing
/// tensor so a layout mismatch fails loudly instead of silently zero-filling.
pub fn model_weights_from_comfy(t: &Tensors, cfg: &ZImageConfig) -> Result<ModelWeightsF32, String> {
    let f32v = |k: &str| -> Result<Vec<f32>, String> {
        t.get(k).map(|(_, d)| d.clone()).ok_or_else(|| format!("import: missing tensor {k}"))
    };
    let f64v = |k: &str| -> Result<Vec<f64>, String> { Ok(f32v(k)?.iter().map(|&x| x as f64).collect()) };

    // One transformer block (15 tensors; adaLN present only when modulated - the
    // context_refiner blocks are UNmodulated, matching the inference model).
    let block = |prefix: &str, modulated: bool| -> Result<WeightsF32, String> {
        let g = |leaf: &str| f32v(&format!("{prefix}.{leaf}"));
        Ok(WeightsF32 {
            wq: g("attention.to_q.weight")?,
            wk: g("attention.to_k.weight")?,
            wv: g("attention.to_v.weight")?,
            wo: g("attention.to_out.0.weight")?,
            w1: g("feed_forward.w1.weight")?,
            w2: g("feed_forward.w2.weight")?,
            w3: g("feed_forward.w3.weight")?,
            nq: g("attention.norm_q.weight")?,
            nk: g("attention.norm_k.weight")?,
            an1: g("attention_norm1.weight")?,
            an2: g("attention_norm2.weight")?,
            fn1: g("ffn_norm1.weight")?,
            fn2: g("ffn_norm2.weight")?,
            adaln_w: if modulated { g("adaLN_modulation.0.weight")? } else { Vec::new() },
            adaln_b: if modulated { g("adaLN_modulation.0.bias")? } else { Vec::new() },
        })
    };

    let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
    let xk = format!("all_x_embedder.{ps}-{pf}");
    let fk = format!("all_final_layer.{ps}-{pf}");
    let mut noise_ref = Vec::with_capacity(cfg.n_refiner_layers as usize);
    let mut ctx_ref = Vec::with_capacity(cfg.n_refiner_layers as usize);
    for l in 0..cfg.n_refiner_layers {
        noise_ref.push(block(&format!("noise_refiner.{l}"), true)?);
        ctx_ref.push(block(&format!("context_refiner.{l}"), false)?);
    }
    let mut main = Vec::with_capacity(cfg.n_layers as usize);
    for l in 0..cfg.n_layers {
        main.push(block(&format!("layers.{l}"), true)?);
    }

    Ok(ModelWeightsF32 {
        t0_w: f64v("t_embedder.mlp.0.weight")?, t0_b: f64v("t_embedder.mlp.0.bias")?,
        t2_w: f64v("t_embedder.mlp.2.weight")?, t2_b: f64v("t_embedder.mlp.2.bias")?,
        xemb_w: f64v(&format!("{xk}.weight"))?, xemb_b: f64v(&format!("{xk}.bias"))?,
        capn_w: f64v("cap_embedder.0.weight")?,
        cap1_w: f64v("cap_embedder.1.weight")?, cap1_b: f64v("cap_embedder.1.bias")?,
        noise_ref, ctx_ref, main,
        fadaln_w: f64v(&format!("{fk}.adaLN_modulation.1.weight"))?, fadaln_b: f64v(&format!("{fk}.adaLN_modulation.1.bias"))?,
        flin_w: f64v(&format!("{fk}.linear.weight"))?, flin_b: f64v(&format!("{fk}.linear.bias"))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(name: &str, data: Vec<f32>) -> StTensor {
        StTensor { name: name.to_string(), shape: vec![data.len()], data }
    }

    /// A whole checkpoint's names and shapes in the SOURCE (Comfy/GGUF)
    /// spelling, derived by INVERTING the rename rules over [`dit_manifest`]
    /// rather than by re-listing them - so a manifest entry with no source
    /// name, or a source name that maps nowhere, shows up as a set difference.
    ///
    /// Synthetic, and only as good as the inverse: what certifies the manifest
    /// against a file someone actually published is `tests/gguf_import_real.rs`.
    fn comfy_shapes(cfg: &ZImageConfig) -> Vec<(String, Vec<usize>)> {
        let dim = cfg.dim as usize;
        let xk = format!("all_x_embedder.{}-{}.", cfg.patch_size, cfg.f_patch_size);
        let fk = format!("all_final_layer.{}-{}.", cfg.patch_size, cfg.f_patch_size);
        let mut out = Vec::new();
        for (name, shape) in dit_manifest(cfg) {
            if let Some(base) = name.strip_suffix("attention.to_q.weight") {
                out.push((format!("{base}attention.qkv.weight"), vec![3 * dim, dim]));
                continue;
            }
            if name.ends_with("attention.to_k.weight") || name.ends_with("attention.to_v.weight") {
                continue; // folded into the qkv above
            }
            let mut k = name.replace(".attention.to_out.0.", ".attention.out.");
            k = k.replace(".attention.norm_k.weight", ".attention.k_norm.weight");
            k = k.replace(".attention.norm_q.weight", ".attention.q_norm.weight");
            if let Some(rest) = k.strip_prefix(xk.as_str()) {
                k = format!("x_embedder.{rest}");
            } else if let Some(rest) = k.strip_prefix(fk.as_str()) {
                k = format!("final_layer.{rest}");
            }
            out.push((k, shape));
        }
        out
    }

    /// Resolve a whole source-spelling checkpoint the way [`import_gguf`] does,
    /// to the name→shape map [`validate_dit_shapes`] takes.
    fn resolve(shapes: &[(String, Vec<usize>)], cfg: &ZImageConfig) -> Result<HashMap<String, Vec<usize>>, String> {
        let xk = format!("all_x_embedder.{}-{}.", cfg.patch_size, cfg.f_patch_size);
        let fk = format!("all_final_layer.{}-{}.", cfg.patch_size, cfg.f_patch_size);
        let mut out = HashMap::new();
        for (name, shape) in shapes {
            for p in comfy_targets(name, shape, cfg, &xk, &fk)? {
                out.insert(p.out, p.shape);
            }
        }
        Ok(out)
    }

    /// The shipped variant's source spelling derives back to the shipped
    /// config, and resolves to exactly the manifest - the round trip
    /// [`import_gguf`] performs, minus the bytes.
    ///
    /// The counts are the released `Tongyi-MAI/Z-Image-Turbo` transformer's:
    /// 521 tensors, of which the 34 fused `qkv` collapse 68 away, leaving the
    /// 453 a GGUF repack of it carries.
    #[test]
    fn the_source_spelling_derives_the_config_and_resolves_to_the_manifest() {
        let cfg = ZImageConfig::turbo();
        let src = comfy_shapes(&cfg);
        assert_eq!(src.len(), 453, "34 fused qkv collapse 68 of the manifest's 521");

        let derived = dit_config_from_shapes(&src).expect("derive the config");
        assert_eq!(
            (derived.dim, derived.n_heads, derived.n_layers, derived.n_refiner_layers, derived.cap_feat_dim),
            (cfg.dim, cfg.n_heads, cfg.n_layers, cfg.n_refiner_layers, cfg.cap_feat_dim)
        );

        let out = resolve(&src, &derived).expect("resolve");
        assert_eq!(out.len(), 521);
        validate_dit_shapes(&out, &derived).expect("the manifest is covered in both directions");
    }

    /// A checkpoint at a DIFFERENT shape still derives and validates - the
    /// point of deriving rather than assuming [`ZImageConfig::turbo`]. Under
    /// the old importer this file would have been written out under turbo's
    /// tensor names and layer count, silently.
    #[test]
    fn a_smaller_variant_derives_its_own_config_rather_than_turbos() {
        let mut cfg = ZImageConfig::turbo();
        cfg.dim = 1280;
        cfg.n_heads = 10;
        cfg.n_layers = 4;
        cfg.n_refiner_layers = 1;
        cfg.cap_feat_dim = 512;
        let src = comfy_shapes(&cfg);
        let derived = dit_config_from_shapes(&src).expect("derive");
        assert_eq!((derived.dim, derived.n_heads, derived.n_layers, derived.n_refiner_layers, derived.cap_feat_dim), (1280, 10, 4, 1, 512));
        validate_dit_shapes(&resolve(&src, &derived).expect("resolve"), &derived).expect("covered");
    }

    /// The three ways a file can fail to be the checkpoint it claims, each
    /// reported by NAME rather than half-imported.
    #[test]
    fn a_mismatched_checkpoint_is_refused_with_the_offending_tensor_named() {
        let cfg = ZImageConfig::turbo();

        // 1. Nothing to derive from at all (a Lumina2 GGUF that got this far).
        let err = dit_config_from_shapes(&[("tok_embeddings.weight".to_string(), vec![32000, 2304])]).unwrap_err();
        assert!(err.contains("cap_embedder.1.weight"), "unhelpful error: {err}");

        // 2. A patch geometry that is not turbo's - the one config field baked
        // into the OUTPUT tensor names, so guessing it wrong is unrecoverable.
        let mut src = comfy_shapes(&cfg);
        let xe = src.iter_mut().find(|(n, _)| n == "x_embedder.weight").expect("x_embedder");
        xe.1 = vec![cfg.dim as usize, 144]; // in_channels 16 at patch 3, say
        let err = dit_config_from_shapes(&src).unwrap_err();
        assert!(err.contains("x_embedder.weight") && err.contains("patch_size"), "unhelpful error: {err}");

        // 3. A file short one block: derived config says 30 layers, the
        // manifest wants layers.29's other 14 tensors.
        let mut src = comfy_shapes(&cfg);
        src.retain(|(n, _)| n != "layers.29.feed_forward.w2.weight");
        let derived = dit_config_from_shapes(&src).expect("derive");
        let err = validate_dit_shapes(&resolve(&src, &derived).expect("resolve"), &derived).unwrap_err();
        assert!(err.contains("layers.29.feed_forward.w2.weight"), "unhelpful error: {err}");

        // 4. A file with a tensor the manifest has no slot for.
        let mut src = comfy_shapes(&cfg);
        src.push(("layers.0.attention.rope_freqs".to_string(), vec![64]));
        let derived = dit_config_from_shapes(&src).expect("derive");
        let err = validate_dit_shapes(&resolve(&src, &derived).expect("resolve"), &derived).unwrap_err();
        assert!(err.contains("unused source tensors") && err.contains("rope_freqs"), "unhelpful error: {err}");
    }

    /// A fused `qkv` whose element count is not three square blocks is a shape
    /// error, not a panic three quarters of the way through a 24 GB write.
    #[test]
    fn a_fused_qkv_of_the_wrong_width_is_an_error_not_a_slice_panic() {
        let cfg = ZImageConfig::turbo();
        let err = comfy_targets("layers.0.attention.qkv.weight", &[2 * cfg.dim as usize, cfg.dim as usize], &cfg, "x.", "f.").unwrap_err();
        assert!(err.contains("layers.0.attention.qkv.weight"), "unhelpful error: {err}");
    }

    #[test]
    fn remap_and_qkv_split() {
        let mut cfg = ZImageConfig::turbo();
        cfg.dim = 2; // tiny: qkv = [6, 2] = 12 elems, split into 3× [2,2]=4.
        let qkv: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensors = vec![
            st("layers.0.attention.qkv.weight", qkv),
            st("layers.0.attention.out.weight", vec![9.0; 4]),
            st("layers.0.attention.q_norm.weight", vec![1.0; 2]),
            st("layers.0.attention.k_norm.weight", vec![1.0; 2]),
            st("x_embedder.weight", vec![2.0; 8]),
            st("final_layer.linear.bias", vec![3.0; 4]),
            st("cap_embedder.0.weight", vec![4.0; 2]),
        ];
        let m = import_comfy(tensors, &cfg);
        // qkv split: q=[0,1,2,3], k=[4,5,6,7], v=[8,9,10,11].
        assert_eq!(m["layers.0.attention.to_q.weight"].1, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(m["layers.0.attention.to_k.weight"].1, vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(m["layers.0.attention.to_v.weight"].1, vec![8.0, 9.0, 10.0, 11.0]);
        // renames
        assert!(m.contains_key("layers.0.attention.to_out.0.weight"));
        assert!(m.contains_key("layers.0.attention.norm_q.weight"));
        assert!(m.contains_key("layers.0.attention.norm_k.weight"));
        assert!(m.contains_key("all_x_embedder.2-1.weight"));
        assert!(m.contains_key("all_final_layer.2-1.linear.bias"));
        // untouched
        assert!(m.contains_key("cap_embedder.0.weight"));
    }

    /// Insert the per-block tensor keys (post-import names) for `prefix`.
    fn ins_block(m: &mut Tensors, prefix: &str, modulated: bool) {
        for leaf in [
            "attention.to_q.weight", "attention.to_k.weight", "attention.to_v.weight",
            "attention.to_out.0.weight", "feed_forward.w1.weight", "feed_forward.w2.weight",
            "feed_forward.w3.weight", "attention.norm_q.weight", "attention.norm_k.weight",
            "attention_norm1.weight", "attention_norm2.weight", "ffn_norm1.weight", "ffn_norm2.weight",
        ] {
            m.insert(format!("{prefix}.{leaf}"), (vec![1], vec![1.0]));
        }
        if modulated {
            m.insert(format!("{prefix}.adaLN_modulation.0.weight"), (vec![1], vec![2.0]));
            m.insert(format!("{prefix}.adaLN_modulation.0.bias"), (vec![1], vec![3.0]));
        }
    }

    #[test]
    fn bridge_to_training_weights_covers_blocks_and_wrapper() {
        let mut cfg = ZImageConfig::turbo();
        cfg.dim = 2;
        cfg.n_layers = 3;
        cfg.n_refiner_layers = 2;
        let (ps, pf) = (cfg.patch_size, cfg.f_patch_size);
        let mut m = Tensors::new();
        for l in 0..cfg.n_layers {
            ins_block(&mut m, &format!("layers.{l}"), true);
        }
        for l in 0..cfg.n_refiner_layers {
            ins_block(&mut m, &format!("noise_refiner.{l}"), true);
            ins_block(&mut m, &format!("context_refiner.{l}"), false); // UNmodulated
        }
        for k in [
            "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias", "t_embedder.mlp.2.weight",
            "t_embedder.mlp.2.bias", "cap_embedder.0.weight", "cap_embedder.1.weight",
            "cap_embedder.1.bias",
            &format!("all_x_embedder.{ps}-{pf}.weight"), &format!("all_x_embedder.{ps}-{pf}.bias"),
            &format!("all_final_layer.{ps}-{pf}.adaLN_modulation.1.weight"),
            &format!("all_final_layer.{ps}-{pf}.adaLN_modulation.1.bias"),
            &format!("all_final_layer.{ps}-{pf}.linear.weight"), &format!("all_final_layer.{ps}-{pf}.linear.bias"),
        ] {
            m.insert(k.to_string(), (vec![1], vec![7.0]));
        }

        let w = model_weights_from_comfy(&m, &cfg).expect("bridge");
        assert_eq!(w.main.len(), 3);
        assert_eq!(w.noise_ref.len(), 2);
        assert_eq!(w.ctx_ref.len(), 2);
        // context_refiner is unmodulated → empty adaLN; noise_refiner/main have it.
        assert!(w.ctx_ref[0].adaln_w.is_empty() && w.ctx_ref[0].adaln_b.is_empty());
        assert!(!w.noise_ref[0].adaln_w.is_empty());
        assert!(!w.main[0].adaln_w.is_empty());
        assert!(!w.main[0].wq.is_empty() && !w.xemb_w.is_empty() && !w.flin_w.is_empty());

        // A missing tensor must error (loudly, named) - not silently zero-fill.
        m.remove("layers.1.feed_forward.w2.weight");
        let err = match model_weights_from_comfy(&m, &cfg) {
            Ok(_) => panic!("expected a missing-tensor error"),
            Err(e) => e,
        };
        assert!(err.contains("layers.1.feed_forward.w2.weight"), "unhelpful error: {err}");
    }

    /// [`comfy_source`] must be byte-for-byte identical to the eager
    /// [`import_comfy`] for every renamed AND every qkv-split tensor - the
    /// same tiny fixture `remap_and_qkv_split` above uses, round-tripped
    /// through a real safetensors file so `comfy_source` reads it via a
    /// genuine `WeightReader`, not an in-memory shortcut.
    #[test]
    fn comfy_source_streaming_matches_eager_import_comfy() {
        use checkpoint::TensorSource;

        let mut cfg = ZImageConfig::turbo();
        cfg.dim = 2; // tiny: qkv = [6, 2] = 12 elems, split into 3x [2,2]=4.
        let qkv: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let named: Vec<(&str, Vec<f32>)> = vec![
            ("layers.0.attention.qkv.weight", qkv),
            ("layers.0.attention.out.weight", vec![9.0; 4]),
            ("layers.0.attention.q_norm.weight", vec![1.0; 2]),
            ("layers.0.attention.k_norm.weight", vec![1.5; 2]),
            ("x_embedder.weight", vec![2.0; 8]),
            ("final_layer.linear.bias", vec![3.0; 4]),
            ("cap_embedder.0.weight", vec![4.0; 2]),
        ];
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = named.iter().map(|(n, d)| (n.to_string(), vec![d.len() as u64], d.clone())).collect();
        let path = std::env::temp_dir().join(format!("brain-zimage-comfy-streaming-{}.safetensors", std::process::id()));
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();

        // Eager reference, over the exact same source tensors.
        let eager = import_comfy(named.into_iter().map(|(n, d)| st(n, d)).collect(), &cfg);

        let reader = checkpoint::weightio::WeightReader::open(path.to_str().unwrap()).unwrap();
        let streamed = comfy_source(&reader, &cfg);

        // Same tensors `remap_and_qkv_split` checks explicitly, so a rename or
        // a qkv-split slice-boundary regression fails here identically.
        for name in eager.keys() {
            let mut got = None;
            assert!(streamed.with_tensor(name, &mut |d| got = Some(d.to_vec())), "missing {name}");
            assert_eq!(got.unwrap(), eager[name].1, "{name}");
        }

        std::fs::remove_file(&path).ok();
    }
}
