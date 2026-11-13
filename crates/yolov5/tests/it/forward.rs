// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixture-free coverage of the reference forward on a tiny synthetic network:
//! shapes at several input sizes, output range, determinism, and the input
//! contract. Numerical correctness against the real network is the job of
//! `tests/parity.rs`.

use yolov5::{Activation, Error, Graph, Weights, Yolov5, Yolov5Config};

fn tiny_cfg(act: Activation) -> Yolov5Config {
    let mut cfg = Yolov5Config::yolov5s(act);
    cfg.nc = 3;
    cfg.width_multiple = 0.125;
    cfg
}

fn weights_for(cfg: &Yolov5Config, seed: u64) -> Weights {
    let graph = Graph::build(cfg);
    let mut rng = data::rng::Lcg::new(seed);
    let tensors = graph
        .param_list()
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let fan_in = shape.iter().skip(1).product::<usize>().max(1) as f32;
            (name, (0..n).map(|_| rng.scaled((2.0 / fan_in).sqrt())).collect())
        })
        .collect();
    Weights::new(&graph, tensors).unwrap()
}

fn model(cfg: &Yolov5Config, h: u32, w: u32) -> Yolov5 {
    let gpu = gpu_core::testgpu::dev(yolov5::model::PIPELINES);
    Yolov5::new_on(gpu, cfg.clone(), &weights_for(cfg, 11), h, w).unwrap()
}

fn image(h: u32, w: u32) -> Vec<f32> {
    let mut rng = data::rng::Lcg::new(3);
    (0..3 * h * w).map(|_| rng.unit()).collect()
}

#[test]
fn every_supported_size_yields_the_three_scales_with_sigmoid_range_values() {
    let _device = gpu_core::testgpu::exclusive();
    for act in [Activation::Relu, Activation::Silu] {
        let cfg = tiny_cfg(act);
        for (h, w) in [(64, 64), (96, 64)] {
            let out = model(&cfg, h, w).forward(&image(h, w)).unwrap();
            let (h, w) = (h as usize, w as usize);
            assert_eq!(out.dims, [(h / 8, w / 8), (h / 16, w / 16), (h / 32, w / 32)], "{act:?} {h}x{w}");
            for (map, (mh, mw)) in out.maps.iter().zip(out.dims) {
                assert_eq!(map.len(), cfg.head_channels() as usize * mh * mw);
                assert!(map.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)), "{act:?}: head values must be sigmoid outputs");
            }
        }
    }
}

#[test]
fn the_forward_is_deterministic_and_input_dependent() {
    let _device = gpu_core::testgpu::exclusive();
    let cfg = tiny_cfg(Activation::Relu);
    let m = model(&cfg, 64, 64);
    let a = m.forward(&image(64, 64)).unwrap();
    let again = m.forward(&image(64, 64)).unwrap();
    assert_eq!(a.maps, again.maps);
    let other: Vec<f32> = image(64, 64).iter().map(|v| 1.0 - v).collect();
    assert_ne!(a.maps, m.forward(&other).unwrap().maps);
}

#[test]
fn the_input_contract_is_enforced_by_name() {
    let _device = gpu_core::testgpu::exclusive();
    let cfg = tiny_cfg(Activation::Relu);
    let gpu = || gpu_core::testgpu::dev(yolov5::model::PIPELINES);
    let w = weights_for(&cfg, 1);
    assert!(matches!(Yolov5::new_on(gpu(), cfg.clone(), &w, 100, 64), Err(Error::Input(m)) if m.contains("multiples of 32")));
    assert!(matches!(Yolov5::new_on(gpu(), cfg.clone(), &w, 0, 64), Err(Error::Input(_))));
    let m = model(&cfg, 64, 64);
    assert!(matches!(m.forward(&[0.0; 5]), Err(Error::Input(_))));
    assert!(matches!(m.detect(&[0.0; 5], 4, 4, 0.25, 0.45), Err(Error::Input(_))));
    let rect = model(&cfg, 96, 64);
    assert!(matches!(rect.detect(&image(8, 8), 8, 8, 0.25, 0.45), Err(Error::Input(m)) if m.contains("square")));
}

#[test]
fn detect_returns_in_range_boxes_in_original_image_pixels() {
    let _device = gpu_core::testgpu::exclusive();
    let cfg = tiny_cfg(Activation::Relu);
    let m = model(&cfg, 64, 64);
    // A non-square image, so the letterbox pads: boxes must come back clamped to it.
    let (w0, h0) = (100u32, 40u32);
    let dets = m.detect(&image(h0, w0), w0, h0, 0.0, 0.45).unwrap();
    assert!(!dets.is_empty(), "a zero threshold keeps candidates");
    for d in &dets {
        assert!(d[0] >= 0.0 && d[2] <= w0 as f32 && d[1] >= 0.0 && d[3] <= h0 as f32, "{d:?}");
        assert!(d[0] <= d[2] && d[1] <= d[3] && (d[5] as u32) < cfg.nc, "{d:?}");
    }
}

/// `detect` letterboxes on the device; the old path letterboxed on the host and
/// uploaded. Both must give the same detections bit for bit, for odd pads in
/// both orientations (95x40 and 40x95 onto the 64 square).
#[test]
fn device_letterbox_detections_equal_the_host_letterbox_path_bitwise() {
    let _device = gpu_core::testgpu::exclusive();
    let cfg = tiny_cfg(Activation::Relu);
    let m = model(&cfg, 64, 64);
    for (w0, h0) in [(95u32, 40u32), (40, 95)] {
        let hwc = image(h0, w0);
        let (chw, lb) = imaging::letterbox_rgb(&hwc, w0, h0, 64, 114.0 / 255.0);
        let out = m.forward(&chw).unwrap();
        let host = yolov5::decode::detections(&cfg, out.heads(), &lb, (w0, h0), 0.0, 0.45, yolov5::decode::MAX_DETECTIONS).unwrap();
        let device = m.detect(&hwc, w0, h0, 0.0, 0.45).unwrap();
        assert!(!host.is_empty(), "{w0}x{h0}: a zero threshold keeps candidates");
        assert_eq!(host, device, "{w0}x{h0}");
    }
}
