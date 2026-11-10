// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running aggressively quantised perception models on constrained
// devices for its clients. If your team needs expertise in low-bit deployment of vision models then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The deployment archive served natively: its CQ matrices stay packed on the device and a describe request through
//! the rotated-codebook kernels equals the same model with those matrices dequantised to dense f32.

use mirage::cact::{CactFile, EntryDtype};
use mirage::describe::Options;
use mirage::device::{Device, Image, MirageModel};
use mirage::model::{load_dir, Mirage, WeightSource};
use serde_json::Value;

use crate::common::released_dir;

fn json<T: serde::Serialize, E: std::fmt::Display>(r: Result<T, E>) -> Value {
    serde_json::to_value(r.unwrap_or_else(|e| panic!("{e}"))).unwrap()
}

#[test]
fn packed_matrices_are_the_archive_matrices_in_the_orientation_the_model_reads() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("device cact: the released checkpoint is absent");
        return;
    };
    let loaded = load_dir(&dir, WeightSource::Cact).expect("the archive loads");
    let file = CactFile::open(&dir.join("mirage.cact")).unwrap();
    let cq_entries = file.entries().iter().filter(|e| e.dtype == EntryDtype::Cq).count();
    // One entry per matrix, except the three routing matrices, which are split per decoder layer.
    let layers = loaded.checkpoint.config.dec_layers;
    assert_eq!(loaded.checkpoint.packed.len(), cq_entries - 3 + 3 * layers, "every CQ entry is packed");
    let by_width = |bits| loaded.checkpoint.packed.values().filter(|c| c.bits == bits).count();
    assert!(by_width(2) > 100 && by_width(4) > 0 && by_width(3) == 0, "2-bit projections and 4-bit embedding and routing");
    // The packed form dequantises to exactly the Linear the host builds from the same tensors.
    let model = Mirage::new(&loaded.checkpoint, &loaded.tokenizer, loaded.charset.clone()).unwrap();
    let embedding = &model.decoder().embedding_linear();
    let cq = embedding.cq.as_ref().expect("the embedding is packed");
    assert_eq!(cq.dequantize(), embedding.wt, "packed embedding == host embedding");
    let bytes: usize = loaded.checkpoint.packed.values().map(|c| c.bytes()).sum();
    eprintln!("device cact: {} packed matrices, {:.2} MB resident", loaded.checkpoint.packed.len(), bytes as f64 / 1e6);
}

#[test]
fn a_packed_model_describes_like_the_same_model_dequantised() {
    let _g = gpu_core::testgpu::exclusive();
    let Some(dir) = released_dir() else {
        brain_testutil::skip("device cact: the released checkpoint is absent");
        return;
    };
    let Some(cats) = crate::decoder::golden("device cact", "cats") else { return };
    let rgb: Vec<u8> = cats["input_rgb"].iter().map(|&v| v as u8).collect();
    let opts = Options::from_json(r#"{"fields":["bbox","position","description"],"max_tokens":600}"#).unwrap();

    let packed = load_dir(&dir, WeightSource::Cact).unwrap();
    let mut dense = load_dir(&dir, WeightSource::Cact).unwrap();
    dense.checkpoint.packed.clear();
    let build = |l: &mirage::model::Loaded| {
        let gpu = gpu_core::Gpu::new(&Device::kernels());
        MirageModel::new(Mirage::new(&l.checkpoint, &l.tokenizer, l.charset.clone()).unwrap(), gpu).unwrap()
    };
    let (m_packed, m_dense) = (build(&packed), build(&dense));
    let (wp, wd) = (m_packed.device().weight_bytes(), m_dense.device().weight_bytes());
    eprintln!("device cact: resident matrix weights packed model {wp:?}, dequantised model {wd:?}");
    assert!(wp.packed > 0 && wd.packed == 0 && wp.dense < wd.dense, "the packed model keeps its CQ matrices packed");

    let images = [Image { rgb: &rgb, height: 480, width: 640 }];
    let a = json(m_packed.describe_batch(&images, &opts).pop().unwrap());
    let b = json(m_dense.describe_batch(&images, &opts).pop().unwrap());
    let strip = |mut v: Value| {
        for list in ["objects", "text"] {
            for item in v[list].as_array_mut().into_iter().flatten() {
                item.as_object_mut().unwrap().remove("score");
            }
        }
        v
    };
    assert_eq!(strip(a), strip(b), "the rotated-codebook kernels reproduce the dequantised model's answer");
}

/// The quantiser against the archive: re-quantising the FP32 weights of a projection with the archive's codebook
/// reproduces its stored indices and norms (up to ties and fp16 rounding of the norm).
#[test]
fn quantising_the_fp32_weights_reproduces_the_archive_matrices() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("device cact: the released checkpoint is absent");
        return;
    };
    let archive = load_dir(&dir, WeightSource::Cact).unwrap();
    let fp32 = load_dir(&dir, WeightSource::Safetensors).unwrap();
    let mut worst_codes = 1.0f64;
    let mut worst_norms = 1.0f64;
    let mut checked = 0;
    for ((name, layer), stored) in archive.checkpoint.packed.iter().take(40) {
        let Some(layer_index) = *layer else { continue };
        let Some(t) = fp32.checkpoint.tensors.get(name) else { continue };
        // Flax `[layers, in, out]` kernel layer -> `[out, in]`.
        let (inn, out) = (t.shape[1], t.shape[2]);
        if stored.n != out || stored.k != inn {
            continue;
        }
        let kernel = &t.data[layer_index * inn * out..(layer_index + 1) * inn * out];
        let wt = mirage::vision::ops::Linear::from_flax(kernel, inn, out).wt;
        let mine = model::cq::Cq::quantize(&wt, out, inn, stored.bits, &stored.book).unwrap();
        let same_codes = mine.packed_bytes().iter().zip(stored.packed_bytes()).filter(|(a, b)| **a == *b).count() as f64 / stored.packed_bytes().len() as f64;
        let same_norms = mine.norm_bytes().iter().zip(stored.norm_bytes()).filter(|(a, b)| **a == *b).count() as f64 / stored.norm_bytes().len() as f64;
        eprintln!("{name}[{layer_index}] {out}x{inn} {}-bit: {:.4} of index bytes and {:.4} of norm bytes identical", stored.bits, same_codes, same_norms);
        (worst_codes, worst_norms, checked) = (worst_codes.min(same_codes), worst_norms.min(same_norms), checked + 1);
    }
    assert!(checked > 0, "no projection was compared");
    eprintln!("quantiser vs archive over {checked} matrices: worst index-byte agreement {worst_codes:.4}, worst norm-byte agreement {worst_norms:.4}");
    assert!(worst_codes > 0.97 && worst_norms > 0.99, "the quantiser disagrees with the archive: indices {worst_codes:.4}, norms {worst_norms:.4}");
}

/// The container writer against the released file: every entry pushed as stored rebuilds the file byte for byte
/// (header, codebooks, directory records, 64-byte blob alignment, padding).
#[test]
fn the_container_writer_rebuilds_the_released_archive_byte_for_byte() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("device cact: the released checkpoint is absent");
        return;
    };
    let original = std::fs::read(dir.join("mirage.cact")).unwrap();
    let file = CactFile::parse(original.clone()).unwrap();
    let mut builder = mirage::cact::CactBuilder::new(file.header(), file.codebooks());
    for (i, e) in file.entries().iter().enumerate() {
        builder.push_as(e, file.entry_bytes(i).unwrap());
    }
    let rebuilt = builder.finish();
    assert_eq!(rebuilt.len(), original.len(), "file length");
    let first = rebuilt.iter().zip(&original).position(|(a, b)| a != b);
    assert!(first.is_none(), "first differing byte at {first:?}");
}

/// Quantising the released FP32 weights against the released archive's plan writes a loadable archive that agrees
/// with the original: every non-quantised entry is identical and the quantised matrices match to the tie level.
#[test]
fn quantize_writes_an_archive_that_matches_the_released_one() {
    let Some(dir) = released_dir() else {
        brain_testutil::skip("device cact: the released checkpoint is absent");
        return;
    };
    let fp32 = load_dir(&dir, WeightSource::Safetensors).unwrap();
    let original = CactFile::open(&dir.join("mirage.cact")).unwrap();
    let template = mirage::cactmap::CactModel::new(CactFile::open(&dir.join("mirage.cact")).unwrap(), fp32.checkpoint.config.clone()).unwrap();
    let bytes = template.quantize(&fp32.checkpoint).expect("quantises");
    let made = mirage::cactmap::CactModel::new(CactFile::parse(bytes.clone()).unwrap(), fp32.checkpoint.config.clone()).expect("the new archive validates against the configuration");
    let (mut cq_total, mut cq_same, mut norm_same, mut other_total, mut other_same) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for (i, e) in original.entries().iter().enumerate() {
        let (a, b) = (original.entry_bytes(i).unwrap(), made.file().entry_bytes(i).unwrap());
        if e.dtype == EntryDtype::Cq {
            let packed = original.cq(i).unwrap().packed.len();
            cq_total += packed;
            cq_same += a[..packed].iter().zip(&b[..packed]).filter(|(x, y)| x == y).count();
            norm_same += a[packed..].iter().zip(&b[packed..]).filter(|(x, y)| x == y).count() * 100 / (a.len() - packed).max(1);
        } else {
            other_total += 1;
            other_same += (a == b) as usize;
        }
    }
    let cq_entries = original.entries().iter().filter(|e| e.dtype == EntryDtype::Cq).count();
    eprintln!(
        "quantize: {} bytes (original {}); {cq_entries} CQ entries with {:.4} of index bytes identical and {:.2}% of norm bytes (mean); {other_same}/{other_total} other entries identical",
        bytes.len(),
        std::fs::metadata(dir.join("mirage.cact")).unwrap().len(),
        cq_same as f64 / cq_total as f64,
        norm_same as f64 / cq_entries as f64
    );
    assert_eq!(bytes.len(), std::fs::metadata(dir.join("mirage.cact")).unwrap().len() as usize, "same layout, same size");
    assert!(cq_same as f64 / cq_total as f64 > 0.97);
    assert_eq!(other_same, other_total, "tokenizer, charset, permutations, vision constants and fp16 tensors carry over exactly");
}

/// Writes the re-quantised archive to the temporary directory for a check by the reference engine's harness
/// (`mirage-requantised.cact`); run by hand.
#[test]
#[ignore = "writes a 21 MB file for a manual check by the reference engine"]
fn write_the_requantised_archive_for_the_reference_engine() {
    let Some(dir) = released_dir() else { return };
    let fp32 = load_dir(&dir, WeightSource::Safetensors).unwrap();
    let template = mirage::cactmap::CactModel::new(CactFile::open(&dir.join("mirage.cact")).unwrap(), fp32.checkpoint.config.clone()).unwrap();
    let path = std::env::temp_dir().join("mirage-requantised.cact");
    std::fs::write(&path, template.quantize(&fp32.checkpoint).unwrap()).unwrap();
    eprintln!("wrote {}", path.display());
}
