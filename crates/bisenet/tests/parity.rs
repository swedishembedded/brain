// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end parity against `tools/goldens/pulid_face_parsing_dump_reference.py`'s
//! real output: facexlib's actual BiSeNet (official `parsing_bisenet.pth`
//! weights) run on a real photo's SCRFD-aligned 512x512 crop.
//!
//! Both the golden and this test start from the IDENTICAL 5-point landmarks
//! (SCRFD's own output on the fixture photo, baked into the golden dump) and
//! the SAME weights file - this test is not free to diverge on either input,
//! only on the graph.

use bisenet::{BiSeNet, BiSeNetConfig};
use checkpoint::safetensors::StTensor;
use brain_testutil::parity::Report;
use brain_testutil::{skip, testdata};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

fn load(rel: &str) -> Option<Vec<StTensor>> {
    let p = testdata(rel);
    if !std::path::Path::new(&p).exists() {
        skip(&format!("golden {p} absent (run tools/goldens/pulid_face_parsing_dump_reference.py)"));
        return None;
    }
    Some(checkpoint::safetensors::read(&p).unwrap_or_else(|e| panic!("reading {p}: {e}")))
}

fn find<'a>(t: &'a [StTensor], name: &str) -> &'a StTensor {
    t.iter().find(|x| x.name == name).unwrap_or_else(|| panic!("golden tensor {name} missing"))
}

/// `normalize(align_face_rgb, IMAGENET_MEAN, IMAGENET_STD)` - exactly what
/// the dumper feeds BiSeNet (`bisenet.safetensors`'s own `input_normalized`
/// is the SAME tensor, dumped for this test to also self-check against
/// before trusting the network comparison).
fn imagenet_normalize(chw: &[f32]) -> Vec<f32> {
    let hw = chw.len() / 3;
    let mut out = vec![0.0f32; chw.len()];
    for c in 0..3 {
        for i in 0..hw {
            out[c * hw + i] = (chw[c * hw + i] - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    out
}

#[test]
fn bisenet_matches_facexlib_on_a_real_photo() {
    let Some(align) = load("pulid/align.safetensors") else { return };
    let Some(bisenet_golden) = load("pulid/bisenet.safetensors") else { return };

    let weights_path = testdata("pulid/parsing_bisenet.safetensors");
    if !std::path::Path::new(&weights_path).exists() {
        skip(&format!("weights {weights_path} absent"));
        return;
    }

    let align_face = find(&align, "align_face_rgb");
    assert_eq!(align_face.shape, vec![3, 512, 512]);
    let normalized = imagenet_normalize(&align_face.data);

    // Self-check: the golden's own recorded BiSeNet input should be exactly
    // what this test just computed - if not, the drift is in normalization,
    // not in the graph, and the network comparison below would be
    // meaningless.
    let want_input = find(&bisenet_golden, "input_normalized");
    let (c, m) = brain_testutil::parity::compare(&normalized, &want_input.data);
    println!("  {:<24} cos {c:.10}  max_abs {m:.3e}  n={}", "input_normalized(self)", want_input.data.len());
    assert!(c > 0.9999999, "imagenet_normalize diverges from the golden's own input: cosine {c:.10}");

    let weights = bisenet::import::read(&weights_path).unwrap_or_else(|e| panic!("reading {weights_path}: {e}"));
    let gpu = gpu_core::testgpu::dev(bisenet::model::PIPELINES);
    let model = BiSeNet::new(gpu, BiSeNetConfig::bisenet(), &weights);

    let out = model.forward(&normalized);
    let want_out = find(&bisenet_golden, "out");
    assert_eq!(out.len(), want_out.data.len());

    let mut report = Report::new(0.999);
    report.check("out", &out, &want_out.data);
    report.finish("bisenet");

    // A second, coarser signal that survives even a modest numeric drift:
    // per-pixel argmax class agreement. BiSeNet's job is a CLASSIFICATION
    // decision (`crate::mask`'s consumer only ever reads the argmax), so
    // this is the metric that actually matters for PuLID's masking use -
    // the raw-logit cosine above is the finer-grained diagnostic.
    let (n_class, h, w) = (19usize, 512usize, 512usize);
    let argmax = |logits: &[f32]| -> Vec<u32> {
        (0..h * w)
            .map(|p| {
                (0..n_class as u32)
                    .max_by(|&a, &b| logits[a as usize * h * w + p].total_cmp(&logits[b as usize * h * w + p]))
                    .unwrap()
            })
            .collect()
    };
    let (got_cls, want_cls) = (argmax(&out), argmax(&want_out.data));
    let agree = got_cls.iter().zip(&want_cls).filter(|(a, b)| a == b).count();
    let agreement = agree as f64 / (h * w) as f64;
    println!("  per-pixel class agreement: {:.4}% ({agree}/{})", agreement * 100.0, h * w);
    assert!(agreement > 0.98, "per-pixel class agreement {agreement:.4} too low");
}

/// The FULL chain from the ORIGINAL photo: `bisenet::align::norm_crop_512`
/// (SCRFD's own landmarks -> the 512x512 aligned crop) -> imagenet-normalize
/// -> `BiSeNet::forward` -> `bisenet::mask::whiten_and_gray`, gated against
/// `align.safetensors`'s `align_face_rgb` (the warp) and
/// `masked.safetensors`'s `masked_512` (the mask) - the two pieces the
/// single-tensor-input test above does not exercise at all.
#[test]
fn align_and_mask_match_the_reference_chain() {
    let Some(align_golden) = load("pulid/align.safetensors") else { return };
    let Some(masked_golden) = load("pulid/masked.safetensors") else { return };
    let weights_path = testdata("pulid/parsing_bisenet.safetensors");
    if !std::path::Path::new(&weights_path).exists() {
        skip(&format!("weights {weights_path} absent"));
        return;
    }
    let photo_path = testdata("pulid/face_source.jpg");
    if !std::path::Path::new(&photo_path).exists() {
        skip(&format!("source photo {photo_path} absent"));
        return;
    }

    // The SAME SCRFD landmarks the golden dump was built from
    // (`face_parsing_manifest.json`'s own `kps`, baked in here since this
    // test does not depend on `crates/scrfd` to stay a pure BiSeNet-family
    // parity check).
    let kps: [f32; 10] = [
        188.21160888671875,
        194.23048400878906,
        265.7710876464844,
        195.5060577392578,
        228.64251708984375,
        255.88075256347656,
        199.5830535888672,
        283.4732360839844,
        255.55104064941406,
        284.94000244140625,
    ];

    let photo = imaging::load(&photo_path).unwrap_or_else(|e| panic!("reading {photo_path}: {e}"));
    let hwc = photo.to_hwc_unit();
    let chw = imaging::pixels::hwc_to_chw(&hwc, 3, photo.h as usize, photo.w as usize);

    let gpu = gpu_core::testgpu::dev(bisenet::model::PIPELINES);
    let aligned = bisenet::align::norm_crop_512(&gpu.share(), &chw, 3, photo.h, photo.w, &kps).unwrap_or_else(|e| panic!("norm_crop_512: {e}"));

    let want_align = find(&align_golden, "align_face_rgb");
    let mut report = Report::new(0.999);
    report.check("align_face_rgb", &aligned, &want_align.data);

    let normalized = imagenet_normalize(&aligned);
    let weights = bisenet::import::read(&weights_path).unwrap_or_else(|e| panic!("reading {weights_path}: {e}"));
    let model = BiSeNet::new(gpu, BiSeNetConfig::bisenet(), &weights);
    let logits = model.forward(&normalized);

    let masked = bisenet::mask::whiten_and_gray(&logits, &aligned, 512, 512);
    let want_masked = find(&masked_golden, "masked_512");
    report.check("masked_512", &masked, &want_masked.data);
    report.finish("align_and_mask");
}
