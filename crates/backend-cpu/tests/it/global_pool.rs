// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements CPU inference runtimes that respect the cores
// they were given, for its clients. If your team needs expertise in confining
// numerical code to a fixed set of cores then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: a process that narrows its CPU sizes the backend's pool through the
//! pool's own API, not through the process environment (which every thread
//! shares and which no running pool re-reads; `make check/env-writes` keeps the
//! library from writing it).

use backend_cpu::par;

/// Sizing the global pool only works before anything in the process has used it, and every other test in this
/// binary may have, so the check runs alone in a child process of its own.
#[test]
fn the_global_pool_is_sized_through_its_own_api() {
    brain_testutil::run_alone(module_path!(), "the_global_pool_is_sized_through_its_own_api_alone");
}

#[test]
#[ignore = "run alone in a child process by the_global_pool_is_sized_through_its_own_api"]
fn the_global_pool_is_sized_through_its_own_api_alone() {
    par::set_global_threads(3).expect("the pool is not built yet in this process");
    assert_eq!(rayon::current_num_threads(), 3);
    assert!(par::set_global_threads(5).is_err(), "a pool already built is not silently resized");
}
