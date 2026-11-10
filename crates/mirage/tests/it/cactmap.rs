// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The positional `.cact` -> canonical tensor mapping: derived from the
//! configuration, self-consistent with the manifest, invertible on a synthetic
//! file, and numerically equal to the safetensors weights on the real one.

use std::collections::{BTreeMap, BTreeSet};

use data::rng::Lcg;
use mirage::cact::CactFile;
use mirage::cactmap::{layout, CactModel, Layout, SlotKind, VISION_MANIFEST_LEN};
use mirage::config::{MirageConfig, READER_CLASSES};
use mirage::import::{self, ImportOptions};
use mirage::spec::Manifest;

use crate::cactgen::{build_with_header, Blob};
use crate::common::{released_or_skip, RELEASED_ARCHITECTURE_JSON};

fn released() -> MirageConfig {
    MirageConfig::from_json(RELEASED_ARCHITECTURE_JSON).unwrap()
}

fn tensor_name(c: &MirageConfig, index: usize) -> (String, Option<usize>) {
    match &layout(c)[index].kind {
        SlotKind::Tensor { name, layer, .. } => (name.clone(), *layer),
        other => (format!("{other:?}"), None),
    }
}

#[test]
fn the_released_config_lays_out_806_entries_in_the_published_order() {
    let c = released();
    let slots = layout(&c);
    assert_eq!(slots.len(), 806);
    assert!(slots.iter().enumerate().all(|(i, s)| s.entry == i));
    let at = |i: usize| tensor_name(&c, i);
    assert_eq!(at(0), ("embedding/embedding".to_string(), None));
    assert_eq!(at(1), ("stack/layers/block/ZCRMSNorm_0/scale".to_string(), Some(0)));
    assert_eq!(at(2).0, "stack/layers/block/self_attn/q_proj/kernel");
    assert_eq!(at(28), ("stack/layers/block/ZCRMSNorm_0/scale".to_string(), Some(1)));
    assert_eq!(at(216), ("stack/layers/block/hadamard_mlp/cond_u".to_string(), Some(7)));
    assert_eq!(at(217).0, "stack/mhc_a_pre");
    assert_eq!(at(222).0, "stack/mhc_b_res");
    assert_eq!(at(223).0, "stack/mhc_phi_pre");
    assert_eq!(at(225).0, "stack/mhc_phi_res");
    assert!(matches!(slots[226].kind, SlotKind::HadamardPermutation(1)));
    assert!(matches!(slots[227].kind, SlotKind::HadamardPermutation(2)));
    assert_eq!(at(228).0, "engrams_0/embedding");
    assert_eq!(at(235).0, "engrams_1/taps");
    assert_eq!(at(236).0, "stack/final_norm/scale");
    assert!(matches!(slots[237].kind, SlotKind::VisionManifest));
    assert_eq!(at(238), ("stack/layers/block/cross_norm/scale".to_string(), Some(0)));
    assert_eq!(at(317), ("stack/layers/block/cross_gate".to_string(), Some(7)));
    assert_eq!(at(318).0, "pe_gate");
    assert_eq!(at(319).0, "stem/w0");
    assert_eq!(at(333).0, "body_0/mix_norm/scale");
    assert_eq!(at(540).0, "body_15/grn/beta");
    assert_eq!(at(541).0, "top_0/attn_norm/scale");
    assert_eq!(at(720).0, "top_11/hconv/out_conv");
    assert_eq!(at(721).0, "enc_norm/scale");
    assert_eq!(at(735).0, "level_gates");
    assert_eq!(at(736).0, "dense/coarse/kernel");
    assert_eq!(at(753).0, "dense/bias");
    assert_eq!(at(770).0, "dense/coef/kernel");
    assert_eq!(at(771).0, "slot_proj/kernel");
    assert_eq!(at(774).0, "reader/w0");
    assert_eq!(at(784).0, "reader/stage_norm_3/scale");
    assert_eq!(at(802).0, "reader/out/kernel");
    assert_eq!(at(803).0, "reader/out/bias");
    assert!(matches!(slots[804].kind, SlotKind::Charset));
    assert!(matches!(slots[805].kind, SlotKind::Tokenizer));
    assert!(matches!(slots[223].kind, SlotKind::Tensor { layout: Layout::MhcPhi, .. }));
    assert!(matches!(slots[2].kind, SlotKind::Tensor { layout: Layout::Transpose, .. }));
    assert!(matches!(slots[0].kind, SlotKind::Tensor { layout: Layout::Same, .. }));
}

/// Every deployment tensor is covered exactly once (a stacked one once per layer),
/// and nothing outside the deployment set is mapped.
fn assert_covers_deployment_set(c: &MirageConfig) {
    let manifest = Manifest::new(c);
    let mut seen: BTreeMap<String, Vec<Option<usize>>> = BTreeMap::new();
    for s in layout(c) {
        if let SlotKind::Tensor { name, layer, .. } = s.kind {
            let spec = manifest.get(&name).unwrap_or_else(|| panic!("mapped `{name}` is not in the manifest"));
            assert!(!spec.training_only, "{name} is training-only but mapped");
            seen.entry(name).or_default().push(layer);
        }
    }
    for spec in manifest.deployment() {
        let layers = seen.remove(&spec.name).unwrap_or_else(|| panic!("deployment tensor `{}` is not mapped", spec.name));
        let stacked = spec.name.starts_with("stack/layers/");
        if stacked {
            let want: Vec<Option<usize>> = (0..c.dec_layers).map(Some).collect();
            let mut got = layers.clone();
            got.sort();
            assert_eq!(got, want, "{}", spec.name);
        } else {
            assert_eq!(layers, vec![None], "{}", spec.name);
        }
    }
    assert!(seen.is_empty(), "mapped but unknown: {:?}", seen.keys().collect::<Vec<_>>());
}

#[test]
fn the_layout_covers_exactly_the_deployment_manifest() {
    assert_covers_deployment_set(&released());
    assert_covers_deployment_set(&MirageConfig::tiny());
    let mut deeper = MirageConfig::tiny();
    deeper.dec_layers = 3;
    deeper.sites = vec![0, 2];
    deeper.body_layers = 3;
    deeper.attn_layers = 1;
    assert_covers_deployment_set(&deeper);
}

// ---- a synthetic file for the tiny config: the layout transforms are invertible ----

/// Header fields a file exported for `c` carries.
fn header_fields(c: &MirageConfig, tensors: usize) -> Vec<(usize, u32)> {
    let mut f = vec![
        (1, tensors as u32),
        (5, c.embedding_rows() as u32),
        (7, c.dim as u32),
        (8, c.heads() as u32),
        (9, c.kv_heads as u32),
        (10, c.dec_layers as u32),
        (11, c.qk_dim as u32),
        (12, c.v_dim as u32),
        (13, c.max_tokens as u32),
        (14, c.hadamard_width() as u32),
        (15, c.lanes as u32),
        (19, c.taps as u32),
        (20, c.slots as u32),
        (21, c.engram_sub_dim() as u32),
        (22, c.engram_tables() as u32),
        (23, 4),
        (24, *c.orders.iter().max().unwrap() as u32),
        (26, c.orders.len() as u32),
        (31, c.sites.len() as u32),
    ];
    f.extend(c.orders.iter().enumerate().map(|(i, &o)| (27 + i, o as u32)));
    f.extend(c.sites.iter().enumerate().map(|(i, &s)| (32 + i, s as u32)));
    f
}

struct Synthetic {
    bytes: Vec<u8>,
    canonical: BTreeMap<String, Vec<f32>>,
    permutations: (Vec<u32>, Vec<u32>),
    vision: Vec<f32>,
    charset: Vec<char>,
    tokenizer: Vec<u8>,
}

/// Write the cact form of random canonical weights, inverting each layout by hand.
fn synthetic(c: &MirageConfig) -> Synthetic {
    let manifest = Manifest::new(c);
    let mut rng = Lcg::new(77); // seed: filler; the test compares bit-exact against what it generated
    let canonical: BTreeMap<String, Vec<f32>> = manifest.deployment().map(|t| (t.name.clone(), (0..t.numel()).map(|_| rng.signed()).collect())).collect();
    let n = c.hadamard_width();
    let permutation = |shift: usize| -> Vec<u32> { (0..n).map(|i| ((i * 5 + shift) % n) as u32).collect() };
    let permutations = (permutation(1), permutation(3));
    let vision: Vec<f32> = (0..VISION_MANIFEST_LEN).map(|i| i as f32 * 0.5).collect();
    let charset: Vec<char> = (0..READER_CLASSES - 1).map(|i| char::from_u32(0x20 + i as u32).unwrap()).collect();
    let tokenizer = b"tokenizer-bytes".to_vec();

    let mut blobs = Vec::new();
    for slot in layout(c) {
        match slot.kind {
            SlotKind::Tensor { name, layer, layout } => {
                let spec = manifest.get(&name).unwrap();
                let (unit_shape, whole) = match layer {
                    Some(l) => {
                        let unit: usize = spec.shape[1..].iter().product();
                        (spec.shape[1..].to_vec(), canonical[&name][l * unit..(l + 1) * unit].to_vec())
                    }
                    None => (spec.shape.clone(), canonical[&name].clone()),
                };
                let (shape, data): (Vec<u32>, Vec<f32>) = match layout {
                    Layout::Same => {
                        let shape = match unit_shape.as_slice() {
                            [] => vec![1],
                            [k1, k2, 1, ch] => vec![(k1 * k2) as u32, *ch as u32],
                            other => other.iter().map(|&d| d as u32).collect(),
                        };
                        (shape, whole)
                    }
                    Layout::Transpose => {
                        let (inn, out) = (unit_shape[0], unit_shape[1]);
                        let mut t = vec![0f32; whole.len()];
                        for i in 0..inn {
                            for o in 0..out {
                                t[o * inn + i] = whole[i * out + o];
                            }
                        }
                        (vec![out as u32, inn as u32], t)
                    }
                    Layout::MhcPhi => {
                        let (layers, width, lanes) = (unit_shape[0], unit_shape[1], unit_shape[2]);
                        let mut t = vec![0f32; whole.len()];
                        for l in 0..layers {
                            for w in 0..width {
                                for j in 0..lanes {
                                    t[(l * lanes + j) * width + w] = whole[(l * width + w) * lanes + j];
                                }
                            }
                        }
                        (vec![(layers * lanes) as u32, width as u32], t)
                    }
                };
                blobs.push(Blob::F32(data, shape));
            }
            SlotKind::HadamardPermutation(k) => {
                let p = if k == 1 { &permutations.0 } else { &permutations.1 };
                blobs.push(Blob::F32(p.iter().map(|&v| v as f32).collect(), vec![n as u32]));
            }
            SlotKind::VisionManifest => blobs.push(Blob::F32(vision.clone(), vec![VISION_MANIFEST_LEN as u32])),
            SlotKind::Charset => blobs.push(Blob::F32(charset.iter().map(|&ch| ch as u32 as f32).collect(), vec![charset.len() as u32])),
            SlotKind::Tokenizer => blobs.push(Blob::Raw(tokenizer.clone())),
        }
    }
    let bytes = build_with_header(&blobs, &header_fields(c, blobs.len()), c.rope);
    Synthetic { bytes, canonical, permutations, vision, charset, tokenizer }
}

#[test]
fn a_synthetic_file_maps_back_to_the_canonical_tensors_exactly() {
    let c = MirageConfig::tiny();
    let s = synthetic(&c);
    let model = CactModel::new(CactFile::parse(s.bytes).unwrap(), c.clone()).expect("a consistent file maps");
    let tensors = model.canonical_tensors().unwrap();
    assert_eq!(tensors.keys().collect::<BTreeSet<_>>(), s.canonical.keys().collect::<BTreeSet<_>>());
    for (name, want) in &s.canonical {
        let got = &tensors[name];
        assert_eq!(got.shape, Manifest::new(&c).get(name).unwrap().shape, "{name}");
        assert!(got.data.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()), "{name} differs after the layout round trip");
    }
    assert_eq!(model.hadamard_permutations().unwrap(), s.permutations);
    assert_eq!(model.vision_manifest().unwrap(), s.vision);
    assert_eq!(model.charset().unwrap(), s.charset);
    assert_eq!(model.tokenizer_blob().unwrap(), &s.tokenizer[..]);
}

fn expect_model_error(file: CactFile, c: &MirageConfig, needle: &str) {
    let err = CactModel::new(file, c.clone()).map(|_| ()).unwrap_err();
    assert!(err.to_string().contains(needle), "expected `{needle}` in `{err}`");
}

#[test]
fn a_header_that_disagrees_with_the_config_is_refused_by_field() {
    let c = MirageConfig::tiny();
    let good = synthetic(&c).bytes;
    for (field_index, name) in [(7usize, "d_model"), (10, "num_layers"), (5, "vocab"), (14, "hada_n"), (22, "num_engram_tables"), (15, "mhc_lanes")] {
        let mut b = good.clone();
        let current = u32::from_le_bytes(b[field_index * 4..field_index * 4 + 4].try_into().unwrap());
        b[field_index * 4..field_index * 4 + 4].copy_from_slice(&(current + 1).to_le_bytes());
        expect_model_error(CactFile::parse(b).unwrap(), &c, name);
    }
    let mut b = good.clone();
    b[192..196].copy_from_slice(&5.0f32.to_le_bytes());
    expect_model_error(CactFile::parse(b).unwrap(), &c, "rope");
}

#[test]
fn an_entry_of_the_wrong_size_is_refused_with_its_position_and_tensor() {
    let c = MirageConfig::tiny();
    // A different config produces a file with the same entry count but other shapes.
    let mut other = c.clone();
    other.dense_dim = 16;
    let file = CactFile::parse(synthetic(&other).bytes).unwrap();
    let err = CactModel::new(file, c).map(|_| ()).unwrap_err().to_string();
    assert!(err.contains("dense/") && err.contains("entry"), "{err}");
}

#[test]
fn a_file_with_the_wrong_number_of_entries_is_refused() {
    let c = MirageConfig::tiny();
    let mut other = c.clone();
    other.body_layers = 3;
    expect_model_error(CactFile::parse(synthetic(&other).bytes).unwrap(), &c, "entries");
}

#[test]
fn derived_entries_are_validated() {
    let c = MirageConfig::tiny();
    let good = CactFile::parse(synthetic(&c).bytes).unwrap();
    let model = CactModel::new(good, c.clone()).unwrap();
    assert_eq!(model.hadamard_permutations().unwrap().0.len(), c.hadamard_width());

    // Entry 0 and 1 of the permutation both become 1.0.
    // A permutation with a repeat, a charset with a surrogate: both refused when read.
    let slots = layout(&c);
    let perm_entry = slots.iter().position(|s| matches!(s.kind, SlotKind::HadamardPermutation(1))).unwrap();
    let charset_entry = slots.iter().position(|s| matches!(s.kind, SlotKind::Charset)).unwrap();
    let mut bytes = synthetic(&c).bytes;
    let file = CactFile::parse(bytes.clone()).unwrap();
    let off = file.entries()[perm_entry].offset;
    bytes[off..off + 4].copy_from_slice(&1.0f32.to_le_bytes());
    bytes[off + 4..off + 8].copy_from_slice(&1.0f32.to_le_bytes());
    let model = CactModel::new(CactFile::parse(bytes.clone()).unwrap(), c.clone()).unwrap();
    assert!(model.hadamard_permutations().is_err());

    let off = file.entries()[charset_entry].offset;
    bytes[off..off + 4].copy_from_slice(&(0xD800u32 as f32).to_le_bytes());
    let model = CactModel::new(CactFile::parse(bytes).unwrap(), c).unwrap();
    assert!(model.charset().is_err());
}

// ---- the released file against the released safetensors ----

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().map(|&v| v as f64).sum::<f64>() / n, b.iter().map(|&v| v as f64).sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    sab / (saa.sqrt() * sbb.sqrt())
}

#[test]
fn the_released_cact_agrees_with_the_released_safetensors() {
    let Some(dir) = released_or_skip("released cact vs safetensors") else { return };
    let ck = import::load_dir(&dir, &ImportOptions::default()).unwrap();
    let c = ck.config.clone();
    let file = CactFile::open(&dir.join("mirage.cact")).expect("the released .cact parses");
    let model = CactModel::new(file, c.clone()).expect("the released header and shapes match the config");

    let (mut quantised, mut small, mut worst_corr) = (0usize, 0usize, 1f64);
    let mut embedding_corr = 0f64;
    for slot in model.slots() {
        let SlotKind::Tensor { name, layer, .. } = &slot.kind else { continue };
        let decoded = model.slot_tensor(slot).unwrap();
        let full = &ck.tensors[name].data;
        let reference: &[f32] = match layer {
            Some(l) => {
                let unit = full.len() / c.dec_layers;
                &full[l * unit..(l + 1) * unit]
            }
            None => full,
        };
        assert_eq!(decoded.len(), reference.len(), "{name}");
        let entry = &model.file().entries()[slot.entry];
        match entry.dtype {
            mirage::cact::EntryDtype::Cq => {
                let r = correlation(&decoded, reference);
                assert!(r > 0.8, "{name} (layer {layer:?}, {} bit) correlates {r:.4} with the safetensors weights", entry.bits);
                worst_corr = worst_corr.min(r);
                quantised += 1;
                if name == "embedding/embedding" {
                    embedding_corr = r;
                }
            }
            mirage::cact::EntryDtype::Fp16 => {
                for (i, (&a, &b)) in decoded.iter().zip(reference).enumerate() {
                    // binary16 has 11 significant bits: half an ulp is 2^-11 relative; subnormal spacing is 2^-24.
                    assert!((a - b).abs() <= b.abs() * (1.0 / 1024.0) + 1.0 / 16_777_216.0, "{name}[{i}]: {a} vs {b}");
                }
                small += 1;
            }
            mirage::cact::EntryDtype::Fp32 => {
                assert!(decoded.iter().zip(reference).all(|(a, b)| a == b), "{name} is stored exactly");
                small += 1;
            }
            mirage::cact::EntryDtype::Raw => panic!("{name} maps a raw blob"),
        }
    }
    eprintln!("cact: {quantised} quantised matrices (worst correlation {worst_corr:.3}), {small} unquantised tensors");
    assert!(embedding_corr > 0.99, "embedding correlation {embedding_corr}");
    assert_eq!(quantised, 159 + 4, "the released file quantises 159 two-bit and 4 four-bit entries, all of them mapped");
}

#[test]
fn the_released_derived_entries_are_what_the_format_says() {
    let Some(dir) = released_or_skip("released cact derived entries") else { return };
    let model = CactModel::new(CactFile::open(&dir.join("mirage.cact")).unwrap(), released()).unwrap();
    let (p1, p2) = model.hadamard_permutations().unwrap();
    for p in [&p1, &p2] {
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..512).collect::<Vec<u32>>(), "a permutation of 0..512");
    }
    assert_ne!(p1, p2);
    let vision = model.vision_manifest().unwrap();
    assert_eq!(vision.len(), 36);
    assert!(vision.iter().all(|v| v.is_finite()));
    let charset = model.charset().unwrap();
    assert_eq!(charset.len(), 306);
    assert_eq!(charset[0], '0');
    assert_eq!(charset.iter().collect::<BTreeSet<_>>().len(), 306, "code points are distinct");
    assert!(model.tokenizer_blob().unwrap().len() > 100_000);
}
