// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Runs convolution layers on the board's NPU and compares every output byte
//! with the host reference (`conv::reference`). Run it on the board; it needs
//! `/dev/rknpu`.
//!
//! Each case is a layer shape YOLOv5 has at a 320x320 input, so a pass here
//! is a pass for the layers the detector runs. Prints per case: bytes that
//! differ (must be 0), task count, NPU time (median of the timed runs).

use std::time::Instant;

use data::rng::Rng;
use rknpu::conv::{self, ConvAddrs, ConvGeometry, ENABLE, INT_MASK};
use rknpu::driver::Device;
use rknpu::job::{Job, Task};
use rknpu::layout::Shape;
use rknpu::quant::{ConvQuant, QParams};

struct Case {
    name: &'static str,
    geometry: ConvGeometry,
    relu: bool,
}

fn case(name: &'static str, h: usize, w: usize, c: usize, oc: usize, k: usize, stride: usize, pad: [usize; 4], relu: bool) -> Case {
    Case { name, geometry: ConvGeometry { input: Shape::new(h, w, c), out_channels: oc, kernel: k, stride, pad }, relu }
}

fn main() {
    let device = Device::open().unwrap_or_else(|e| panic!("{e}"));
    let cases = [
        case("smoke 8x8x16->16 k3", 8, 8, 16, 16, 3, 1, [1; 4], true),
        case("stem 320x320x3->12 k2s2 as k3", 320, 320, 3, 12, 3, 2, [0, 0, 1, 1], false),
        case("160x160x12->32 k3", 160, 160, 12, 32, 3, 1, [1; 4], true),
        case("160x160x32->64 k3s2", 160, 160, 32, 64, 3, 2, [1; 4], true),
        case("40x40x128->64 k1", 40, 40, 128, 64, 1, 1, [0; 4], true),
        case("20x20x256->256 k3", 20, 20, 256, 256, 3, 1, [1; 4], true),
        case("10x10x1024->512 k1", 10, 10, 1024, 512, 1, 1, [0; 4], true),
        case("head 40x40x128->255 k1", 40, 40, 128, 255, 1, 1, [0; 4], false),
    ];
    // Operands are random; the seed is printed so a failure reproduces with
    // `rknpu-board-check <seed>`.
    let seed = match std::env::args().nth(1) {
        Some(s) => s.parse().unwrap_or_else(|_| panic!("seed must be an integer, got {s:?}")),
        None => Rng::random().state(),
    };
    println!("seed {seed}");
    let mut rng = Rng::new(seed);
    let mut failures = 0;
    for c in &cases {
        let g = &c.geometry;
        let out = g.output();
        let fan_in = g.input.c * g.kernel * g.kernel;
        let weights: Vec<f32> = (0..g.out_channels * fan_in).map(|_| rng.uniform(-0.5, 0.5) as f32).collect();
        let bias: Vec<f32> = (0..g.out_channels).map(|_| rng.uniform(-0.2, 0.2) as f32).collect();
        let input_q = QParams::from_range(-1.0, 1.0).unwrap();
        let bound = 0.6 * (fan_in as f32).sqrt();
        let output_q = QParams::from_range(if c.relu { 0.0 } else { -bound }, bound).unwrap();
        let q = ConvQuant::quantize(&weights, &bias, g.out_channels, g.input.c, g.kernel, input_q, output_q, c.relu).unwrap();
        let input: Vec<i8> = (0..g.input.h * g.input.w * g.input.c).map(|_| rng.gen_range_inclusive(-128, 127) as i8).collect();

        // Data region: input, output, weights, bias, each 64-byte aligned.
        let input_at = 0usize;
        let output_at = input_at + g.input.bytes();
        let weights_at = output_at + out.bytes();
        let bias_at = weights_at + g.weight_bytes().next_multiple_of(64);
        let data_bytes = bias_at + g.bias_bytes().next_multiple_of(64);
        let addrs = ConvAddrs {
            input: input_at as u32,
            input_plane_bytes: g.input.plane_bytes(),
            output: output_at as u32,
            output_plane_bytes: out.plane_bytes(),
            weights: weights_at as u32,
            bias: bias_at as u32,
        };
        let programs = match conv::programs(g, &q, input_q.zero_point, &addrs) {
            Ok(p) => p,
            Err(e) => {
                println!("{:<34} UNSUPPORTED {e}", c.name);
                failures += 1;
                continue;
            }
        };
        let tasks: Vec<Task> = programs.into_iter().map(|program| Task { program, enable: ENABLE, int_mask: INT_MASK }).collect();
        let mut job = Job::new(&device, data_bytes, &tasks).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let data = job.data_mut();
        g.input.pack(&input, &mut data[input_at..]);
        conv::pack_weights(g, &q, &mut data[weights_at..weights_at + g.weight_bytes()]);
        conv::pack_bias(&q, &mut data[bias_at..bias_at + g.bias_bytes()]);
        job.flush_all().unwrap();

        let mut times = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            job.run(&device, None).unwrap_or_else(|e| panic!("{}: {e}", c.name));
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(f64::total_cmp);
        job.invalidate(output_at..output_at + out.bytes()).unwrap();
        let got = out.unpack(&job.data()[output_at..]);
        let want = conv::reference(g, &q, input_q.zero_point, &input);
        let differ = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        let max_diff = got.iter().zip(&want).map(|(&a, &b)| (i16::from(a) - i16::from(b)).abs()).max().unwrap_or(0);
        if differ != 0 {
            failures += 1;
        }
        println!(
            "{:<34} {} differ={differ}/{} max|d|={max_diff} tasks={} npu={:.2}ms arena={}KiB",
            c.name,
            if differ == 0 { "OK  " } else { "FAIL" },
            want.len(),
            job.task_count(),
            times[times.len() / 2],
            job.resident_bytes() / 1024
        );
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines().filter(|l| l.starts_with("VmHWM")) {
        println!("{line}");
    }
    std::process::exit(if failures == 0 { 0 } else { 1 });
}
