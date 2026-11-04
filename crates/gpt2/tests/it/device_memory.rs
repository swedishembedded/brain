// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of language models for its clients. If your team needs
// expertise in proving what a training configuration saves on the device, you can procure our services by sending an
// email to info@swedishembedded.com.

//! The device memory a precision choice saves, measured by the backend's own accounting rather than computed: one
//! test in its own binary, so no other test allocates on the device while it measures.

use gpt2::{Gpt, GptConfig};
use model::Model;

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(gpt2::model::PIPELINES))
}

/// Moving the optimizer state to host memory releases exactly the two f32 moment buffers of every trainable tensor,
/// and int8 blocks hold them in about a quarter of the bytes.
#[test]
fn host_moments_release_exactly_the_device_moments() {
    let cfg = GptConfig { vocab: 23, block_size: 16, ..GptConfig::tiny() };
    let init = gpt2::init_weights(&cfg, 11);
    let live = |m: &Gpt| -> Option<u64> { m.optimizer_parts().and_then(|p| p.gpu.memory_usage()).map(|u| u.live) };
    let mut m = Gpt::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 2, 8, &init);
    let Some(before) = live(&m) else {
        brain_testutil::skip_unavailable("device memory accounting: this backend does not account for its allocations");
        return;
    };
    let numel: u64 = m.optimizer_parts().unwrap().store.trainable.iter().map(|(_, n)| *n as u64).sum();
    model::apply_train_precision(&mut m, &model::TrainPrecision::parse("moments=host").unwrap(), "gpt2").unwrap();
    m.poll_wait();
    assert_eq!(before - live(&m).unwrap(), 2 * 4 * numel, "host moments release two f32 words per trainable element");
    drop(m);

    let mut m = Gpt::new_on(gpu_core::testgpu::dev(pipelines()), cfg, 2, 8, &init);
    let before = live(&m).unwrap();
    model::apply_train_precision(&mut m, &model::TrainPrecision::parse("state=int8").unwrap(), "gpt2").unwrap();
    m.poll_wait();
    let saved = before - live(&m).unwrap();
    assert!(saved * 10 >= 2 * 4 * numel * 7, "int8 blocks save about three quarters of the f32 moments: {saved} of {}", 8 * numel);
}
