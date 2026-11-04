// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Not a gate — a visual dump: run the NPU-exported ZipDepth on a PPM and write the
//! colorized depth, so the NPU output can be eyeballed. Env-gated (ZIPDEPTH_NPU_PTH
//! + DEPTH_RENDER_IMG + DEPTH_RENDER_OUT).
use std::collections::HashMap;
use zipdepth::import;
use zipdepth::viz::{colorize, Bounds, Colormap};
use npu::build_depth_graph;
use npu::openvino::{NpuConfig, NpuDevice, NpuSession};
use onnx::GraphBuilder;

#[test]
fn render() {
    let (Ok(pth), Ok(img), Ok(out)) = (
        std::env::var("ZIPDEPTH_NPU_PTH"),
        std::env::var("DEPTH_RENDER_IMG"),
        std::env::var("DEPTH_RENDER_OUT"),
    ) else {
        return brain_testutil::skip(
            "set ZIPDEPTH_NPU_PTH to the ZipDepth checkpoint, DEPTH_RENDER_IMG to an input image and DEPTH_RENDER_OUT to the output path",
        );
    };
    let cfg = zipdepth::ZipConfig { upsample_unfold: false, ..zipdepth::ZipConfig::base() };
    let sz = cfg.input as usize;
    let init: HashMap<String, Vec<f32>> = import::load(&pth, &cfg).unwrap();
    let mut g = GraphBuilder::new("zipdepth");
    build_depth_graph(&cfg, &init, &mut g);
    let mut sess = NpuSession::load_bytes(&g.finish(), &NpuConfig { device: NpuDevice::Npu, allow_fallback: true, ..Default::default() }).unwrap();

    let (px, w, h) = events::ppm::decode_p6(&std::fs::read(&img).unwrap()).unwrap();
    // center-crop/resize to sz square via nearest (visual only).
    let mut chw = vec![0.5f32; 3 * sz * sz];
    for y in 0..sz { for x in 0..sz {
        let sx = x * w as usize / sz; let sy = y * h as usize / sz;
        let o = (sy * w as usize + sx) * 3;
        for c in 0..3 { chw[c*sz*sz + y*sz + x] = px[o+c] as f32 / 255.0; }
    }}
    let res = sess.run(&chw, [1,3,sz,sz]).unwrap();
    let depth = &res.tensors[0].2;
    let b = Bounds::from_percentiles(depth, 0.02, 0.98);
    let rgb = colorize(depth, b, Colormap::Turbo);
    std::fs::write(&out, events::ppm::encode_p6(&rgb, sz as u32, sz as u32)).unwrap();
    eprintln!("wrote {out} ({sz}x{sz}) on {}", sess.device());
}
