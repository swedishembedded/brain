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
