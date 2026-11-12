// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The exported static-shape graph, compiled by OpenVINO, against the same
//! goldens as brain's own forward: every network, both inputs, on each
//! OpenVINO device this machine has (CPU, the Intel GPU, the NPU), and the
//! decoded landmarks against the reference tracker.
//!
//! The GPU and NPU compute in fp16, so their tolerance is that of half
//! precision on these activations; the CPU plugin computes in fp32.

use openseeface::config::{LANDMARKS_BEST, LANDMARKS_FAST};
use openseeface::decode;
use openseeface::openvino::OvNet;
use openseeface::Infer;
use openvino_rt::NpuDevice;

/// The OpenVINO devices present, or `None` (having skipped) when the runtime
/// is not installed.
fn devices() -> Option<Vec<NpuDevice>> {
    match openvino_rt::available_devices() {
        Ok(d) => Some([NpuDevice::Cpu, NpuDevice::Gpu, NpuDevice::Npu].into_iter().filter(|x| d.iter().any(|n| n.starts_with(x.ov_str()))).collect()),
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no OpenVINO runtime ({e}); samples/vision/fishtank/fetch-data.sh installs one"));
            None
        }
    }
}

/// Half-precision devices round every activation; fp32 CPU does not.
fn tolerance(d: NpuDevice) -> (f32, f32) {
    match d {
        NpuDevice::Cpu => (0.99999, 1e-3),
        _ => (0.9995, 6e-2),
    }
}

#[test]
fn the_exported_graph_matches_the_reference_on_every_openvino_device() {
    let (Some(dir), Some(devs)) = (crate::weights(), devices()) else { return };
    for dev in devs {
        let (cos, abs) = tolerance(dev);
        for (name, cfg) in crate::NETS {
            let Some(g) = crate::golden(&format!("{name}.safetensors")) else { return };
            let mut net = OvNet::load(&dir, cfg, dev).unwrap_or_else(|e| panic!("{name} on {dev:?}: {e}"));
            println!("{name} on {}:", net.device());
            for input in ["syn", "real"] {
                let o = net.infer(&g[&format!("{input}.input")].1).unwrap();
                crate::gate(&format!("{input} out"), &o.map, &g[&format!("{input}.out")].1, cos, abs);
                if let Some(p) = o.pooled {
                    crate::gate(&format!("{input} maxpool"), &p, &g[&format!("{input}.maxpool")].1, cos, abs);
                }
            }
        }
    }
}

#[test]
fn landmarks_from_the_compiled_graph_land_where_the_reference_tracker_puts_them() {
    let (Some(dir), Some(devs), Some(g)) = (crate::weights(), devices(), crate::golden("pipeline.safetensors")) else { return };
    let c = &g["lm.crop"].1;
    let crop = [c[0] as i32, c[1] as i32, c[2] as i32, c[3] as i32];
    for dev in devs {
        for (key, cfg) in [("lm0", LANDMARKS_FAST), ("lm3", LANDMARKS_BEST)] {
            let mut net = OvNet::load(&dir, cfg, dev).unwrap();
            let lm = decode::landmarks(&net.infer(&g["lm.input"].1).unwrap().map, crop);
            let want = &g[&format!("{key}.landmarks")].1;
            let worst = (0..decode::POINTS).map(|i| (lm.points[i][0] - want[i * 3 + 1]).abs().max((lm.points[i][1] - want[i * 3]).abs())).fold(0.0f32, f32::max);
            println!("  {key} on {dev:?}: worst landmark {worst:.3} px");
            // half a heatmap cell is 4 px in this crop; an fp16 argmax may
            // pick a neighbouring cell for a flat peak, which the offset
            // then mostly corrects
            assert!(worst < 2.0, "{key} on {dev:?}: a landmark is {worst} px from the reference");
        }
    }
}
