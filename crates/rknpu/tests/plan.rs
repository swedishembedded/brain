// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The compiler on the host: a YOLO-shaped graph is quantized, laid out and
//! simulated with the NPU's exact integers, and must track the float graph.

mod common;

use common::mini_yolo::mini_yolo;
use data::rng::Rng;
use rknpu::graph::Calibration;
use rknpu::plan::{Plan, Step};
use rknpu::quant::QParams;

#[test]
fn quantized_plan_tracks_the_float_graph() {
    let mut rng = Rng::new(11);
    let size = 32;
    let g = mini_yolo(&mut rng, size);
    let images: Vec<Vec<f32>> = (0..4).map(|_| (0..size * size * 3).map(|_| rng.next_f32()).collect()).collect();
    let calibration = Calibration::measure(&g, images.iter().map(|v| v.as_slice())).unwrap();
    let input_q = QParams { scale: 1.0 / 255.0, zero_point: -128 };
    let plan = Plan::new(&g, &calibration, input_q).unwrap();

    // The schedule: the Add is lowered onto the NPU, so only the two max
    // pools and the upsample run on the CPU.
    let cpu = plan.steps().iter().filter(|s| !matches!(s, Step::Npu { .. })).count();
    assert_eq!(cpu, 3, "two max pools, one upsample");

    let image = &images[0];
    let pixels: Vec<u8> = image.iter().map(|&v| (v * 255.0).round() as u8).collect();
    let quantized: Vec<f32> = pixels.iter().map(|&p| f32::from(p) / 255.0).collect();
    let want = g.run(&quantized, |_, _| {}).unwrap();

    let mut data = vec![0u8; plan.activation_bytes()];
    let input = plan.placement(plan.input());
    let codes: Vec<i8> = quantized.iter().map(|&v| input_q.quantize(v)).collect();
    input.shape.pack(&codes, &mut data[input.offset..]);
    plan.simulate(&mut data).unwrap();

    for (i, &out) in plan.outputs().iter().enumerate() {
        let p = plan.placement(out);
        let got: Vec<f32> = p.shape.unpack(&data[p.offset..]).into_iter().map(|q| p.q.dequantize(q)).collect();
        let range = want[i].iter().fold(0f32, |m, v| m.max(v.abs()));
        let mean_err = got.iter().zip(&want[i]).map(|(a, b)| (a - b).abs()).sum::<f32>() / got.len() as f32;
        assert!(mean_err < 0.03 * range, "output {i}: mean abs error {mean_err} vs range {range}");
    }
}
