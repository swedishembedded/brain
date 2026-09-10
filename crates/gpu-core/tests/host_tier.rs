// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The host tier has to be real.** Automatic placement is allowed to answer
//! "the CPU" - on a GPU-less box, and equally on a box whose cards a
//! neighbouring process has filled - and that answer only means anything if
//! the model then actually builds on the CPU backend.
//!
//! Swedish Embedded AB implements graceful accelerator fallback for its
//! clients. If your team needs expertise in keeping a model running when the
//! hardware it wanted is busy, you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! Hardware-free: everything here drives an installed [`Placer`] and reads the
//! scoping state, so it runs identically on a 2-GPU box and in CI. One test
//! function, deliberately - the placer, the automatic memo and the panic hook
//! are all process-global, so the ORDER of these assertions is the
//! specification and splitting them into parallel tests would make them race.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use gpu_core::devices::{self, Home, Homes, Need, Placer};

/// A placer that always answers the host tier - what the real one returns when
/// no card can hold anything right now.
struct AllHost;

impl Placer for AllHost {
    fn place(&self, needs: &[Need]) -> Result<Vec<Home>, String> {
        Ok(needs.iter().map(|_| Home::Cpu).collect())
    }
}

/// A placer that refuses. The real one used to do this whenever every card was
/// contended, because the host tier was gated on there being no cards at all.
struct Refuses;

impl Placer for Refuses {
    fn place(&self, _needs: &[Need]) -> Result<Vec<Home>, String> {
        Err("nothing fits".into())
    }
}

#[test]
fn the_host_tier_is_a_real_placement_not_a_label() {
    devices::set_ambient_gpu(None);

    // 1. Outside any scope, nothing claims the host tier.
    assert!(!devices::on_host_tier());

    // 2. `Homes::run` on a `Home::Cpu` part SCOPES that part to the CPU
    //    backend. It used to match only `Some(Home::Gpu(i))` and let
    //    `Home::Cpu` fall through to a bare `f()`, which ran unscoped - so
    //    every `Gpu::new` inside still built on whatever card was ambient, and
    //    a part "placed on the CPU" allocated exactly the VRAM the placer had
    //    just decided it could not have.
    let homes = Homes::new(vec![("te".into(), Home::Cpu), ("dit".into(), Home::Gpu(0))]);
    assert!(homes.run("te", devices::on_host_tier).expect("run"), "a Home::Cpu part must build on the host tier");
    assert!(!devices::on_host_tier(), "the scope must not outlive the part");
    assert!(!homes.run("dit", devices::on_host_tier).expect("run"), "a carded part must not be dragged onto the host tier");
    // An unknown name expresses no preference and is left alone.
    assert!(!homes.run("nobody", devices::on_host_tier).expect("run"));

    // 3. Nested scopes restore correctly (a host-tier part building a
    //    sub-part that also asks).
    devices::with_host_tier(|| {
        assert!(devices::on_host_tier());
        devices::with_host_tier(|| assert!(devices::on_host_tier()));
        assert!(devices::on_host_tier(), "an inner scope ending must not end the outer one");
    });
    assert!(!devices::on_host_tier());

    // 4. A policy answering the host tier is REPORTED as the host tier.
    //    `selected_device()` used to fall through to `devs.first()` - card 0,
    //    typically the most contended card on the machine, which is exactly
    //    backwards - so the policy's decision was silently undone.
    devices::install_placer(Arc::new(AllHost));
    assert!(devices::auto_host_tier(), "the placer said cpu; the default must say so too");
    assert_eq!(devices::selected_device().map(|d| d.index), None, "there is no card to return for a host-tier answer");
    assert_eq!(devices::place(&[Need::unsized_("model")]).expect("plan").of("model"), Some(Home::Cpu));

    // 5. A placer that ERRORS falls back to the registry default, and that
    //    negative answer is NOT cached for the life of the process: the memo
    //    is cleared when a new policy is installed, and re-asked on a TTL.
    let asked = Arc::new(AtomicUsize::new(0));
    struct Counting(Arc<AtomicUsize>);
    impl Placer for Counting {
        fn place(&self, needs: &[Need]) -> Result<Vec<Home>, String> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(needs.iter().map(|_| Home::Cpu).collect())
        }
    }
    devices::install_placer(Arc::new(Refuses));
    assert!(!devices::auto_host_tier(), "a policy that errors is not a host-tier answer");
    devices::install_placer(Arc::new(Counting(asked.clone())));
    assert!(devices::auto_host_tier());
    let before = asked.load(Ordering::SeqCst);
    devices::forget_auto_placement();
    assert!(devices::auto_host_tier());
    assert!(asked.load(Ordering::SeqCst) > before, "forgetting the memo must make the next call re-ask the machine");

    // Leave the process as we found it.
    devices::install_placer(Arc::new(AllHost));
    devices::forget_auto_placement();
    devices::set_ambient_gpu(None);

    a_lost_memory_race_is_retried_rather_than_fatal();
}

/// A build that fails because another process took the memory is retried
/// against a fresh plan, not reported as a dead run. What it does NOT do is
/// promise a reservation - see `build_with_retry`'s own doc. Part of the same
/// test function because it swaps the process-global panic hook.
fn a_lost_memory_race_is_retried_rather_than_fatal() {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // the deliberate panics below are not test failures

    let attempts = AtomicUsize::new(0);
    let got = devices::build_with_retry("fixture", || {
        if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
            panic!("brain: device memory exhausted");
        }
        Ok(7)
    });
    assert_eq!(got, Ok(7), "the second attempt must be allowed to succeed");
    assert_eq!(attempts.load(Ordering::SeqCst), 2);

    // A model's own error is NOT a capacity fault and must not be retried -
    // retrying a missing checkpoint three times just makes it slower to fail.
    let tries = AtomicUsize::new(0);
    let err: Result<u32, String> = devices::build_with_retry("fixture", || {
        tries.fetch_add(1, Ordering::SeqCst);
        Err("no such checkpoint".into())
    });
    assert_eq!(err, Err("no such checkpoint".into()));
    assert_eq!(tries.load(Ordering::SeqCst), 1, "a real error must fail on the first attempt");

    std::panic::set_hook(hook);
}
