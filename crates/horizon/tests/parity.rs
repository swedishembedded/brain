// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device the environment selects (`BRAIN_BACKEND`) computes what the CPU
//! backend computes: the same loss and every parameter's gradient, on one
//! batch, for the set encoder and both visit backbones. Gradient checks hold
//! each backend to finite differences of itself; this holds the backends to
//! each other, so a kernel that is wrong in the same way as its own
//! derivative on one device (a transpilation fault, a missing barrier) is
//! still caught. On a machine whose selected device is the CPU it compares
//! the CPU with itself.

use data::rng::Rng;
use gpu_core::Gpu;
use horizon::batch::assemble;
use horizon::encode::{encode, Encoded};
use horizon::synthetic::drifting::{self, Gaps};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Backbone, Horizon, HorizonConfig, PIPELINES};

fn models(visits: u32, backbone: Backbone) -> (Horizon, Horizon) {
    let gaps = Gaps {
        last: (0.0, 2.0),
        between: (0.5, 3.0),
        visits: (1, 5),
    };
    let (subjects, _) = drifting::population(400, 5, &gaps, 6.0);
    let codes = vec![drifting::CODE.to_string()];
    let vocab = Vocab::fit(
        &subjects,
        &codes,
        &codes,
        &FitOptions {
            knots: 9,
            min_count: 1,
        },
    )
    .unwrap();
    let mut cfg = HorizonConfig::tiny(vocab.len(), 1);
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.max_tokens = 8;
    cfg.visits = visits;
    cfg.backbone = backbone;
    let enc: Vec<Encoded> = subjects
        .iter()
        .take(30)
        .map(|s| encode(s, &vocab, &cfg))
        .collect();
    let refs: Vec<&Encoded> = enc.iter().collect();
    let hb = assemble(&cfg, &refs, 32, 0.4, &mut Rng::new(9));
    let init = horizon::init_weights(&cfg, 3);
    let cpu = Horizon::new_on(Gpu::new_cpu(PIPELINES), cfg.clone(), 32, &init);
    let dev = Horizon::new(cfg, 32, &init);
    for m in [&cpu, &dev] {
        m.set_batch(&hb);
        m.zero_grads();
    }
    (cpu, dev)
}

#[test]
fn the_selected_device_matches_the_cpu() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    for (visits, backbone) in [
        (0, Backbone::State),
        (3, Backbone::State),
        (3, Backbone::Attention),
    ] {
        let (cpu, dev) = models(visits, backbone);
        let (a, b) = (cpu.forward(), dev.forward());
        assert!(
            (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
            "visits {visits} {backbone:?}: loss {a} vs {b}"
        );
        cpu.backward();
        dev.backward();
        let names: Vec<String> = cpu.ps.params.iter().map(|(n, _)| n.clone()).collect();
        for name in names {
            let (x, y) = (
                cpu.ps.read_grad(&cpu.gpu, &name),
                dev.ps.read_grad(&dev.gpu, &name),
            );
            let scale = x.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
            let worst = x
                .iter()
                .zip(&y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max);
            assert!(worst <= 2e-3 * scale, "visits {visits} {backbone:?}: gradient of {name} differs by {worst} (scale {scale})");
        }
    }
}
