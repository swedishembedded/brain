// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `gpu-core` half of automatic placement: this crate ASKS, it does not
//! decide. An installed [`gpu_core::devices::Placer`] answers the `None` arm
//! of `selected_device()` - the case where the user expressed no preference -
//! and nothing else.
//!
//! Swedish Embedded AB implements device-placement plumbing for its clients.
//! If your team needs expertise in making a model land on hardware that can
//! hold it, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! One test, deliberately: `selected_device()` reads process-global state, so
//! the ordering of these assertions IS the specification and splitting them
//! into parallel tests would make them race.

use std::sync::Arc;

use gpu_core::devices::{self, Home, Need, Placer};

/// A placer that puts the first part on `self.0` and every later part on the
/// OTHER card - stands in for the real capacity-driven one so this gate needs
/// no particular machine state, and, critically, gives the later parts a
/// device that differs from the ambient default. A `run` assertion whose part
/// happens to sit on the ambient card cannot fail when `run` forgets to scope
/// at all.
struct Fixed(u32);

impl Placer for Fixed {
    fn place(&self, needs: &[Need]) -> Result<Vec<Home>, String> {
        Ok(needs.iter().enumerate().map(|(i, _)| if i == 0 { Home::Gpu(self.0) } else { Home::Gpu(1 - self.0) }).collect())
    }
}

#[test]
fn a_placer_answers_only_when_the_user_expressed_no_preference() {
    let n = devices::gpus().len();
    if n < 2 {
        eprintln!("skip: needs 2 GPUs, found {n}");
        return;
    }

    // 1. With nothing installed, the default is exactly what it always was.
    devices::set_ambient_gpu(None);
    assert_eq!(devices::selected_device().map(|d| d.index), Some(0), "with no placer, card 0 stays the default");

    devices::install_placer(Arc::new(Fixed(1)));

    // 2. A scoped selection wins, unchanged.
    let scoped = devices::with_gpu(0, || devices::selected_device().map(|d| d.index)).expect("scope");
    assert_eq!(scoped, Some(0), "with_gpu must still pin the card it names");

    // 3. The `--device gpu0` / BRAIN_DEVICE pin wins, unchanged.
    devices::set_ambient_gpu(Some(0));
    assert_eq!(devices::selected_device().map(|d| d.index), Some(0), "an explicit pin must still win");

    // 4. Only with no preference at all does the placer decide.
    devices::set_ambient_gpu(None);
    assert_eq!(devices::selected_device().map(|d| d.index), Some(1), "the placer must place the no-preference case");

    // 5. A multi-part plan runs each part on the device it was given, and
    //    says so.
    let homes = devices::place(&[Need::unsized_("dit"), Need::unsized_("te").apart()]).expect("plan");
    assert_eq!(homes.of("dit"), Some(Home::Gpu(1)));
    assert_eq!(homes.of("te"), Some(Home::Gpu(0)));
    assert_eq!(homes.describe(), "dit=gpu1 te=gpu0", "the placement must be printable, in declaration order");
    // Deliberately the part that is NOT on the ambient default card: this is
    // what makes the assertion capable of failing when `run` does not scope.
    let inside = homes.run("te", || devices::selected_device().map(|d| d.index)).expect("run");
    assert_eq!(inside, Some(0), "running a part must scope every Gpu::new under it to that part's device");
    assert_eq!(devices::selected_device().map(|d| d.index), Some(1), "the scope must not outlive the part");

    // 6. Installing a placer must not leak into an explicit run afterwards.
    devices::set_ambient_gpu(Some(0));
    assert_eq!(devices::selected_device().map(|d| d.index), Some(0));
    devices::set_ambient_gpu(None);
}

/// With no card at all the seam is inert: no placer call, no panic, and the
/// backend's own default still applies.
#[test]
fn a_gpu_less_box_is_unaffected() {
    if !devices::gpus().is_empty() {
        return;
    }
    assert_eq!(devices::selected_device(), None);
    let homes = devices::place(&[Need::unsized_("model")]).expect("a plan must still resolve");
    assert_eq!(homes.of("model"), Some(Home::Cpu));
}
