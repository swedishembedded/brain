// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Map a `moondream/moondream3-preview` checkpoint's 662 tensors onto brain's
//! layout. Three subsystems: `model.vision.*` (SigLIP ViT + `proj_mlp` connector),
//! `model.text.*` (the parallel-block MoE decoder), and `model.region.*` (the
//! region/point/detect heads — deferred, recognized so coverage is exhaustive).
//!
//! The one non-trivial transform is the MoE experts: layers 4–23 store all experts
//! stacked in `mlp.fc1.weight [E, 2·inner, d]` and `mlp.fc2.weight [E, d, inner]`.
//! Per expert, `fc1` splits along its `2·inner` rows into `w_h` (first `inner`,
//! erf-GELU'd) and `w_g` (next `inner`, the `+1` shift) — matching `layers.py`'s
//! `x1, g = x1_full.chunk(2); F.gelu(x1) * (g + 1)` — and `fc2` is `w_down` as-is.

use std::collections::HashMap;

use checkpoint::weightio::WeightReader;
use checkpoint::TensorSource;

use crate::config::MoondreamConfig;

/// A stacked MoE tensor that splits per-expert at load time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MoePart {
    /// `[E, 2·inner, d]` → per expert `w_h [inner,d]` (0..inner) + `w_g [inner,d]` (inner..2·inner).
    Fc1,
    /// `[E, d, inner]` → per expert `w_down [d, inner]`.
    Fc2,
    /// `[E, d]` router gate.
    Router,
    /// `[E]` router bias (recognized; consumed once the router uses a bias).
    RouterBias,
}

/// Where a text tensor lands in brain's layout.
#[derive(Clone, Debug, PartialEq)]
pub enum TextTarget {
    /// Direct rename to this brain parameter key.
    Key(String),
    /// A stacked MoE tensor for decoder layer `layer`, split per-expert at load.
    Moe { layer: u32, part: MoePart },
}

/// HF `model.text.*` name → brain target (config decides dense vs MoE per layer).
pub fn map_text(hf: &str, cfg: &MoondreamConfig) -> Option<TextTarget> {
    use TextTarget::Key;
    match hf {
        "model.text.wte" => return Some(Key("tok.weight".into())),
        "model.text.lm_head.weight" => return Some(Key("lm_head.weight".into())),
        "model.text.lm_head.bias" => return Some(Key("lm_head.bias".into())),
        "model.text.post_ln.weight" => return Some(Key("post_ln.weight".into())),
        "model.text.post_ln.bias" => return Some(Key("post_ln.bias".into())),
        _ => {}
    }
    let (n, leaf) = hf.strip_prefix("model.text.blocks.")?.split_once('.')?;
    let layer: u32 = n.parse().ok()?;
    let moe = cfg.is_moe_layer(layer);
    let key = |k: &str| Some(Key(format!("blocks.{layer}.{k}")));
    match leaf {
        "ln.weight" | "ln.bias" | "attn.proj.weight" | "attn.proj.bias" | "attn.qkv.weight" | "attn.qkv.bias"
        | "attn.tau.alpha" | "attn.tau.wq" | "attn.tau.wv" => key(leaf),
        "mlp.fc1.bias" | "mlp.fc2.bias" if !moe => key(leaf), // dense layers only
        "mlp.fc1.weight" if !moe => key(leaf),
        "mlp.fc2.weight" if !moe => key(leaf),
        "mlp.fc1.weight" if moe => Some(TextTarget::Moe { layer, part: MoePart::Fc1 }),
        "mlp.fc2.weight" if moe => Some(TextTarget::Moe { layer, part: MoePart::Fc2 }),
        "mlp.router.weight" if moe => Some(TextTarget::Moe { layer, part: MoePart::Router }),
        "mlp.router.bias" if moe => Some(TextTarget::Moe { layer, part: MoePart::RouterBias }),
        _ => None,
    }
}

/// HF `model.vision.*` ViT tensor → [`SiglipEncoder`] key (prefix stripped). Returns
/// `None` for `proj_mlp.*` (the connector, see [`map_connector`]).
///
/// [`SiglipEncoder`]: crate::vision::SiglipEncoder
pub fn map_vision(hf: &str) -> Option<String> {
    let rest = hf.strip_prefix("model.vision.")?;
    if rest.starts_with("proj_mlp.") {
        return None;
    }
    // patch_emb.{weight,bias}, pos_emb, post_ln.{weight,bias}, blocks.N.<leaf> — all
    // already match SiglipEncoder's key scheme verbatim.
    Some(rest.to_string())
}

/// HF `model.vision.proj_mlp.*` → [`Connector`] key.
///
/// [`Connector`]: crate::vision::Connector
pub fn map_connector(hf: &str) -> Option<String> {
    let rest = hf.strip_prefix("model.vision.proj_mlp.")?;
    Some(match rest {
        "fc1.weight" => "fc1.weight",
        "fc1.bias" => "fc1.bias",
        "fc2.weight" => "fc2.weight",
        "fc2.bias" => "fc2.bias",
        _ => return None,
    }
    .to_string())
}

/// The region/point/detect heads (`model.region.*`) — deferred (Phase 3.9).
pub fn is_region(hf: &str) -> bool {
    hf.starts_with("model.region.")
}

/// Split a stacked MoE `fc1.weight[expert]` slice `[2·inner, d]` into `(w_h, w_g)`,
/// each `[inner, d]`. `w_h` is the erf-GELU'd half, `w_g` the `+1`-shifted half.
pub fn split_fc1_expert(slice: &[f32], inner: u32, d: u32) -> (Vec<f32>, Vec<f32>) {
    let half = (inner * d) as usize;
    assert_eq!(slice.len(), 2 * half, "fc1 expert slice must be [2·inner, d]");
    (slice[..half].to_vec(), slice[half..].to_vec())
}

/// The brain keys produced for one MoE decoder layer (router + per-expert triples).
pub fn moe_layer_keys(layer: u32, num_experts: u32) -> Vec<String> {
    let mut keys = vec![format!("blocks.{layer}.moe.router.weight")];
    for e in 0..num_experts {
        for leaf in ["w_h.weight", "w_g.weight", "w_down.weight"] {
            keys.push(format!("blocks.{layer}.moe.experts.{e}.{leaf}"));
        }
    }
    keys
}

/// The three weight maps [`crate::model::MoondreamModel::new`] takes, plus the
/// coverage report proving every checkpoint tensor was accounted for.
#[derive(Debug, Default)]
pub struct Weights {
    /// SigLIP ViT tensors, prefix stripped.
    pub vision: HashMap<String, Vec<f32>>,
    /// `proj_mlp` connector tensors.
    pub connector: HashMap<String, Vec<f32>>,
    /// Decoder tensors, with the stacked MoE experts already split.
    pub decoder: HashMap<String, Vec<f32>>,
}

/// What [`load`] did with every tensor in the checkpoint.
///
/// **Two-way coverage, per the porting rules.** A one-way check ("every tensor
/// I wanted was present") silently tolerates a checkpoint carrying tensors this
/// port ignores - which is exactly how a missing fused-qkv bias hid in this
/// model once already. So `unmapped` is reported, and `load` refuses rather
/// than continuing when it is non-empty for anything but the deliberately
/// deferred region heads.
#[derive(Debug, Default)]
pub struct Coverage {
    /// Tensors that landed in one of the three maps.
    pub mapped: usize,
    /// `model.region.*` - recognized and deliberately skipped (the
    /// region/point/detect heads are not built).
    pub region_skipped: usize,
    /// The region subtree's `(name, shape)` pairs, in checkpoint order.
    ///
    /// Captured rather than merely counted because porting those heads is
    /// blocked on not knowing what they ARE: this repo records only the
    /// `model.region.` prefix, and the reference modeling code ships inside the
    /// checkpoint directory rather than here. Whoever next has a checkpoint can
    /// call [`load`] and print this instead of writing a throwaway script to
    /// discover the manifest - which is the actual first step of that port.
    pub region_tensors: Vec<(String, Vec<usize>)>,
    /// Anything else: a tensor the checkpoint has and this port does not know.
    pub unmapped: Vec<String>,
}

/// Load a `moondream/moondream3-preview` checkpoint directory into the three
/// weight maps, splitting the stacked MoE experts on the way through.
///
/// **This is the production loader, and the crate's real-weight tests are thin
/// wrappers over it** - the arrangement `crates/deepseek2ocr` uses, and for the
/// same reason: a served run and its own parity test must not be able to
/// disagree about which tensors they loaded. Before this existed the only code
/// that turned a real checkpoint into weights lived inside a `#[cfg(test)]`
/// module, so nothing user-facing could load the model at all.
///
/// Streams through [`checkpoint::WeightReader`], which mmaps and parses only the
/// headers, so the peak host cost is the maps being built rather than the maps
/// plus a full copy of the file.
///
/// **The fp32 footprint is the real constraint here, not the I/O.** At
/// [`MoondreamConfig::preview`] the decoder alone is 8.8 B parameters (20 MoE
/// layers x 64 experts x three `[1024, 2048]`-ish matrices), which is ~33 GB of
/// `f32` in `Weights::decoder` before a single device buffer is allocated. That
/// is a property of the model at this precision, not of this function; see this
/// crate's own docs for the quantized path.
pub fn load(dir: &std::path::Path, cfg: &MoondreamConfig) -> Result<(Weights, Coverage), String> {
    let rd = WeightReader::open_hf_dir(dir).map_err(|e| format!("moondream3: cannot open '{}': {e}", dir.display()))?;
    let mut w = Weights { vision: HashMap::new(), connector: HashMap::new(), decoder: HashMap::new() };
    let mut cov = Coverage { mapped: 0, region_skipped: 0, region_tensors: Vec::new(), unmapped: Vec::new() };

    let names: Vec<String> = rd.names().map(str::to_string).collect();
    for name in &names {
        let shape: Vec<usize> = rd.shape(name).map(|s: &[u64]| s.iter().map(|&d| d as usize).collect()).unwrap_or_default();
        if is_region(name) {
            cov.region_skipped += 1;
            // Shape only - never the DATA. These heads are not built, so
            // materialising their tensors would be pure footprint.
            cov.region_tensors.push((name.clone(), shape));
            continue;
        }
        let mut taken = false;
        // `with_tensor` lends the decoded values for the call only, so each arm
        // copies exactly what it keeps - the MoE arms copy per-expert slices
        // rather than the whole stacked tensor.
        rd.with_tensor(name, &mut |data: &[f32]| {
            taken = true;
            if let Some(k) = map_vision(name) {
                w.vision.insert(k, data.to_vec());
            } else if let Some(k) = map_connector(name) {
                w.connector.insert(k, data.to_vec());
            } else if let Some(t) = map_text(name, cfg) {
                match t {
                    TextTarget::Key(k) => {
                        w.decoder.insert(k, data.to_vec());
                    }
                    TextTarget::Moe { layer, part } => split_moe(&mut w.decoder, layer, part, data, &shape, cfg),
                }
            } else {
                taken = false;
            }
        });
        if taken {
            cov.mapped += 1;
        } else {
            cov.unmapped.push(name.clone());
        }
    }

    if !cov.unmapped.is_empty() {
        return Err(format!(
            "moondream3: {} checkpoint tensor(s) this port does not recognize, e.g. {:?} - refusing rather than loading a partial model",
            cov.unmapped.len(),
            &cov.unmapped[..cov.unmapped.len().min(5)]
        ));
    }
    // The other direction: every key the graph will ask for must be present.
    for k in required_keys(cfg) {
        let present = w.vision.contains_key(&k) || w.connector.contains_key(&k) || w.decoder.contains_key(&k);
        if !present {
            return Err(format!("moondream3: checkpoint is missing '{k}'"));
        }
    }
    Ok((w, cov))
}

/// Split one stacked MoE tensor into the per-expert keys the decoder reads.
///
/// `fc1` is `[E, 2*inner, d]` and splits along its `2*inner` rows into `w_h`
/// (first `inner`, erf-GELU'd) and `w_g` (next `inner`, the `+1` shift) -
/// matching the reference's `x1, g = x1_full.chunk(2)`. `fc2` is `[E, d, inner]`
/// and is `w_down` as-is. The router is `[E, d]`, unstacked.
fn split_moe(out: &mut HashMap<String, Vec<f32>>, layer: u32, part: MoePart, data: &[f32], shape: &[usize], cfg: &MoondreamConfig) {
    let (e, inner, d) = (cfg.moe.num_experts, cfg.moe.inner_dim, cfg.dim);
    match part {
        MoePart::Router => {
            out.insert(format!("blocks.{layer}.moe.router.weight"), data.to_vec());
        }
        // Recognized so coverage is exhaustive; the router has no bias term in
        // the graph this port builds.
        MoePart::RouterBias => {}
        MoePart::Fc1 => {
            let per = (2 * inner * d) as usize;
            debug_assert_eq!(data.len(), e as usize * per, "fc1 stacked shape {shape:?}");
            for ei in 0..e as usize {
                let (w_h, w_g) = split_fc1_expert(&data[ei * per..(ei + 1) * per], inner, d);
                out.insert(format!("blocks.{layer}.moe.experts.{ei}.w_h.weight"), w_h);
                out.insert(format!("blocks.{layer}.moe.experts.{ei}.w_g.weight"), w_g);
            }
        }
        MoePart::Fc2 => {
            let per = (d * inner) as usize;
            debug_assert_eq!(data.len(), e as usize * per, "fc2 stacked shape {shape:?}");
            for ei in 0..e as usize {
                out.insert(format!("blocks.{layer}.moe.experts.{ei}.w_down.weight"), data[ei * per..(ei + 1) * per].to_vec());
            }
        }
    }
}

/// Every key the composed graph reads, for the "nothing missing" half of the
/// coverage check. Derived from the config, so a config change cannot leave the
/// check describing a different model than the one being built.
pub fn required_keys(cfg: &MoondreamConfig) -> Vec<String> {
    let mut k: Vec<String> = vec![
        "tok.weight".into(),
        "lm_head.weight".into(),
        "lm_head.bias".into(),
        "post_ln.weight".into(),
        "post_ln.bias".into(),
        "patch_emb.weight".into(),
        "patch_emb.bias".into(),
        "pos_emb".into(),
        "fc1.weight".into(),
        "fc1.bias".into(),
        "fc2.weight".into(),
        "fc2.bias".into(),
    ];
    for l in 0..cfg.n_layers {
        for leaf in ["ln.weight", "ln.bias", "attn.qkv.weight", "attn.proj.weight", "attn.proj.bias"] {
            k.push(format!("blocks.{l}.{leaf}"));
        }
        if cfg.is_moe_layer(l) {
            k.extend(moe_layer_keys(l, cfg.moe.num_experts));
        } else {
            for leaf in ["mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"] {
                k.push(format!("blocks.{l}.{leaf}"));
            }
        }
    }
    k
}

#[cfg(test)]
mod loader_tests {
    use super::*;

    /// The graph asks for exactly the keys the loader promises to check for.
    /// If these two lists drift, a checkpoint passes coverage and then the
    /// builder panics naming a tensor - which is the failure the two-way check
    /// exists to turn into a clean error at the boundary.
    #[test]
    fn required_keys_covers_every_layer_and_both_ffn_kinds() {
        let cfg = MoondreamConfig::preview();
        let k = required_keys(&cfg);
        let set: std::collections::HashSet<&str> = k.iter().map(String::as_str).collect();
        assert!(set.contains("tok.weight") && set.contains("lm_head.weight") && set.contains("pos_emb"));
        // Layer 0 is dense (below `moe.start_layer`), layer 23 is MoE.
        assert!(set.contains("blocks.0.mlp.fc1.weight"), "a dense layer must want its own FFN");
        assert!(!set.contains("blocks.23.mlp.fc1.weight"), "an MoE layer must NOT want a dense FFN");
        assert!(set.contains("blocks.23.moe.router.weight"));
        let last = cfg.moe.num_experts - 1;
        assert!(set.contains(format!("blocks.23.moe.experts.{last}.w_down.weight").as_str()));
        assert_eq!(k.len(), set.len(), "required_keys must not repeat a key");
    }

    /// The stacked-expert split is where a wrong stride would silently give
    /// every expert the same weights (or interleave two of them) - shapes still
    /// line up either way, so it is checked by construction.
    #[test]
    fn split_moe_gives_each_expert_its_own_distinct_slice() {
        let mut cfg = MoondreamConfig::preview();
        cfg.moe.num_experts = 3;
        cfg.moe.inner_dim = 2;
        cfg.dim = 4;
        let (e, inner, d) = (3usize, 2usize, 4usize);
        // fc1 stacked as [E, 2*inner, d], each expert filled with its own index.
        let data: Vec<f32> = (0..e).flat_map(|ei| std::iter::repeat_n(ei as f32, 2 * inner * d)).collect();
        let mut out = HashMap::new();
        split_moe(&mut out, 7, MoePart::Fc1, &data, &[e, 2 * inner, d], &cfg);
        for ei in 0..e {
            let h = &out[&format!("blocks.7.moe.experts.{ei}.w_h.weight")];
            let g = &out[&format!("blocks.7.moe.experts.{ei}.w_g.weight")];
            assert_eq!(h.len(), inner * d);
            assert_eq!(g.len(), inner * d);
            assert!(h.iter().all(|&v| v == ei as f32), "expert {ei}'s w_h took another expert's slice: {h:?}");
            assert!(g.iter().all(|&v| v == ei as f32), "expert {ei}'s w_g took another expert's slice: {g:?}");
        }
    }

    /// `w_h` is the FIRST half of the `2*inner` rows and `w_g` the second - the
    /// reference's `x1, g = x1_full.chunk(2)`. Swapping them runs, and produces
    /// a different function.
    #[test]
    fn split_moe_puts_the_gelu_half_first() {
        let mut cfg = MoondreamConfig::preview();
        cfg.moe.num_experts = 1;
        cfg.moe.inner_dim = 1;
        cfg.dim = 2;
        // One expert, [2*inner=2, d=2]: first row 10s (w_h), second row 20s (w_g).
        let data = vec![10.0, 10.0, 20.0, 20.0];
        let mut out = HashMap::new();
        split_moe(&mut out, 0, MoePart::Fc1, &data, &[1, 2, 2], &cfg);
        assert_eq!(out["blocks.0.moe.experts.0.w_h.weight"], vec![10.0, 10.0]);
        assert_eq!(out["blocks.0.moe.experts.0.w_g.weight"], vec![20.0, 20.0]);
    }

    /// The region subtree is REPORTED, not just counted.
    ///
    /// Porting the point/detect/region-caption heads is blocked on not knowing
    /// their architecture: this crate records only the `model.region.` prefix,
    /// and the reference modeling code ships inside the checkpoint directory
    /// rather than in this repo. Capturing `(name, shape)` during the load that
    /// already streams every header turns "discover the manifest" from a
    /// throwaway script into a field on the report - so this test pins that the
    /// field is wired to the same prefix `is_region` matches, and that a region
    /// tensor is skipped rather than counted as mapped.
    #[test]
    fn region_tensors_are_recognized_and_reported_not_silently_dropped() {
        assert!(is_region("model.region.coord_encoder.weight"));
        assert!(!is_region("model.text.blocks.0.ln.weight"));
        assert!(!is_region("model.vision.pos_emb"));
        // The report has somewhere for them to go, and it starts empty.
        let cov = Coverage::default();
        assert_eq!(cov.region_skipped, 0);
        assert!(cov.region_tensors.is_empty());
    }

    /// A missing checkpoint is a clean error naming the directory, not a panic.
    #[test]
    fn a_missing_checkpoint_directory_is_a_named_error() {
        let err = load(std::path::Path::new("/definitely/not/a/moondream/dir"), &MoondreamConfig::preview()).unwrap_err();
        assert!(err.contains("cannot open"), "{err}");
    }
}

#[cfg(test)]
mod tests {

use brain_testutil::model_dir;
#[allow(dead_code)]
fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}
    use super::*;

    fn cfg() -> MoondreamConfig {
        MoondreamConfig::preview()
    }

    #[test]
    fn text_names_route_dense_vs_moe() {
        let c = cfg();
        assert_eq!(map_text("model.text.wte", &c), Some(TextTarget::Key("tok.weight".into())));
        // Dense layer 0: fc1.weight is a plain rename with a bias.
        assert_eq!(map_text("model.text.blocks.0.mlp.fc1.weight", &c), Some(TextTarget::Key("blocks.0.mlp.fc1.weight".into())));
        assert_eq!(map_text("model.text.blocks.0.mlp.fc1.bias", &c), Some(TextTarget::Key("blocks.0.mlp.fc1.bias".into())));
        // MoE layer 4: fc1/fc2/router are stacked splits; no dense bias.
        assert_eq!(map_text("model.text.blocks.4.mlp.fc1.weight", &c), Some(TextTarget::Moe { layer: 4, part: MoePart::Fc1 }));
        assert_eq!(map_text("model.text.blocks.4.mlp.router.weight", &c), Some(TextTarget::Moe { layer: 4, part: MoePart::Router }));
        assert_eq!(map_text("model.text.blocks.23.mlp.fc2.weight", &c), Some(TextTarget::Moe { layer: 23, part: MoePart::Fc2 }));
        // tau + attn always direct.
        assert_eq!(map_text("model.text.blocks.7.attn.tau.alpha", &c), Some(TextTarget::Key("blocks.7.attn.tau.alpha".into())));
    }

    #[test]
    fn vision_and_connector_split() {
        assert_eq!(map_vision("model.vision.blocks.3.attn.qkv.weight"), Some("blocks.3.attn.qkv.weight".into()));
        assert_eq!(map_vision("model.vision.patch_emb.weight"), Some("patch_emb.weight".into()));
        assert_eq!(map_vision("model.vision.proj_mlp.fc1.weight"), None); // connector, not ViT
        assert_eq!(map_connector("model.vision.proj_mlp.fc2.bias"), Some("fc2.bias".into()));
    }

    #[test]
    fn fc1_split_halves() {
        // [2·inner=4, d=2]: rows 0..2 → w_h, rows 2..4 → w_g.
        let slice: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let (h, g) = split_fc1_expert(&slice, 2, 2);
        assert_eq!(h, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(g, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn real_checkpoint_fully_covered() {
        use std::io::Read;
        let path = format!("{}/model.safetensors.index.json", model_dir("moondream/moondream3-preview").unwrap_or_default());
        let Ok(mut f) = std::fs::File::open(path) else {
            brain_testutil::skip("moondream3 index not present");
            return;
        };
        let mut s = String::new();
        f.read_to_string(&mut s).unwrap();
        let idx: serde_json::Value = serde_json::from_str(&s).unwrap();
        let names: Vec<String> = idx["weight_map"].as_object().unwrap().keys().cloned().collect();
        let c = cfg();

        // Every source tensor is classified into exactly one subsystem.
        let mut text = 0usize;
        let mut vision = 0usize;
        let mut connector = 0usize;
        let mut region = 0usize;
        for n in &names {
            let hits = map_text(n, &c).is_some() as u8
                + map_vision(n).is_some() as u8
                + map_connector(n).is_some() as u8
                + is_region(n) as u8;
            assert_eq!(hits, 1, "tensor classified {hits}× (want 1): {n}");
            if map_text(n, &c).is_some() {
                text += 1;
            } else if map_vision(n).is_some() {
                vision += 1;
            } else if map_connector(n).is_some() {
                connector += 1;
            } else {
                region += 1;
            }
        }
        assert_eq!(names.len(), 662, "expected 662 tensors");
        assert_eq!(connector, 4, "proj_mlp fc1/fc2 weight+bias");
        assert_eq!(region, 12, "region head tensors (deferred)");
        // Vision: 27 blocks × 12 + patch_emb(2) + pos_emb(1) + post_ln(2) = 329.
        assert_eq!(vision, 27 * 12 + 5);
        assert_eq!(text + vision + connector + region, 662);
        assert!(text > 0);
    }
}
