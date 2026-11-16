// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Gpu::new_like` on CUDA opens the card of the handle it is called on, not
//! the ambient one. A pipeline places a part on a card and then builds that
//! part's sub-models with `new_like` outside the placement scope; landing them
//! on the ambient card instead puts their bytes on a card the plan charged
//! for something else.
//!
//! Swedish Embedded AB implements multi-model device placement for its
//! clients. If your team needs expertise in keeping several models' memory on
//! the accelerator a plan chose, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! Skip-if-absent: a box with fewer than two CUDA cards has no other card for
//! the handle to wrongly land on.

use gpu_core::{devices, Gpu};

const KERNELS: &[(&str, &str)] = &[("axpy", kernels::AXPY)];
const OTHER_KERNELS: &[(&str, &str)] = &[("add2", kernels::ADD2)];

#[test]
fn a_handle_for_another_kernel_set_stays_on_the_card_it_was_made_from() {
    if gpu_core::testgpu::cards().len() < 2 || devices::cuda_identities().map_or(true, |v| v.len() < 2) {
        brain_testutil::skip_unavailable("needs two CUDA cards");
        return;
    }
    let cards = gpu_core::testgpu::cards();
    let last = cards.last().expect("two cards").index;
    let Ok(on_last) = devices::with_gpu(last, || Gpu::try_new_cuda(KERNELS)).expect("registry card") else {
        brain_testutil::skip_unavailable("no usable CUDA backend");
        return;
    };
    let first = cards[0].index;
    let like = devices::with_gpu(first, || on_last.new_like(OTHER_KERNELS)).expect("registry card");
    assert_eq!(like.kind(), "cuda");
    let (want, got) = (on_last.identity().expect("CUDA identity"), like.identity().expect("CUDA identity"));
    assert!(got.same_device(&want), "new_like opened {:?} instead of the handle's own card {:?}", got.uuid, want.uuid);

    let a = like.storage_init("a", &[1.0f32, 2.0]);
    let b = like.storage_init("b", &[10.0f32, 20.0]);
    let out = like.storage(2);
    like.submit(&[], &[like.step(0, &[&a, &b, &out], &[2], 2)]);
    assert_eq!(like.read(&out, 2), vec![11.0, 22.0], "the new kernel set runs on the handle");
}
