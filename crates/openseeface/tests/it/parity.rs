// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity, stage by stage, for each released network on two inputs:
//! a seeded N(0, 1) tensor and the real photo preprocessed by the reference.
//! The input tensors are replayed from the goldens, so this rung measures the
//! network alone; preprocessing has its own rung in `pipeline.rs`.

use openseeface::model::{Net, PIPELINES};

/// Stage tap names in the order the goldens hold them.
fn taps(g: &crate::Golden, input: &str) -> Vec<String> {
    let mut t: Vec<String> = g.keys().filter_map(|k| k.strip_prefix(&format!("{input}.")).map(str::to_string)).filter(|k| k != "input" && k != "out" && k != "maxpool").collect();
    let order = |k: &str| -> (u32, u32, u32) {
        if k == "stem" {
            return (0, 0, 0);
        }
        if let Some(up) = k.strip_prefix("up") {
            return (2, up.parse().unwrap_or(0), 0);
        }
        let mut it = k.trim_start_matches("blocks.").split('.').map(|v| v.parse::<u32>().unwrap_or(0));
        (1, it.next().unwrap_or(0), it.next().unwrap_or(0))
    };
    t.sort_by_key(|k| order(k));
    t
}

#[test]
fn every_network_matches_the_reference_at_every_stage() {
    let Some(dir) = crate::weights() else { return };
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    for (name, cfg) in crate::NETS {
        let Some(g) = crate::golden(&format!("{name}.safetensors")) else { return };
        let net = Net::load(&gpu, &dir, cfg).unwrap_or_else(|e| panic!("{name}: {e}"));
        for input in ["real", "syn"] {
            println!("{name} / {input}:");
            net.forward(&g[&format!("{input}.input")].1);
            for t in taps(&g, input) {
                let got = net.read_tap(&t).unwrap_or_else(|| panic!("{name}: no tap {t}"));
                crate::gate(&t, &got, &g[&format!("{input}.{t}")].1, 0.9999, 2e-2);
            }
            crate::gate("out", &net.read_output(), &g[&format!("{input}.out")].1, 0.99999, 5e-3);
            if let Some(p) = net.read_pooled() {
                crate::gate("maxpool", &p, &g[&format!("{input}.maxpool")].1, 0.99999, 5e-3);
            }
        }
    }
}
