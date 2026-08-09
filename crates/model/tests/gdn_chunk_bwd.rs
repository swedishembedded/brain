// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gated DeltaNet backward — NOT YET IMPLEMENTED.
//!
//! `model::gdn` ships forward only (`gdn_chunk_fwd`, gated at
//! `crates/model/tests/gdn_chunk_fwd.rs`). A hand-written reverse-mode
//! backward through all eleven steps of `torch_chunk_gated_delta_rule` —
//! including a REVERSE sequential sweep through the UT-transform's forward
//! substitution and the across-chunk state recurrence — is real, separate
//! work that this porting pass did not have the budget to derive and
//! gradient-check with confidence. Per `docs/porting-playbook.md`'s own
//! discipline (ship a correct, honestly-scoped forward over a rushed
//! backward that is subtly wrong), that work is deferred rather than
//! attempted here.
//!
//! What "done" looks like for this file, when someone picks this up:
//!
//! 1. A `model::gdn::gdn_chunk_bwd` step-list builder, dispatching backward
//!    kernels for each forward step in REVERSE order (most of `bmm.wgsl`'s
//!    backward is just another `bmm`/`bmm_acc` call with permuted operands
//!    and swapped `trans_a`/`trans_b` — see `bmm.wgsl`'s header; the
//!    UT-transform's backward needs a genuinely new reverse sweep, `i` from
//!    `chunk-1` down to `1`, mirroring `gdn_ut_step.wgsl`'s forward sweep).
//! 2. A finite-difference gradcheck AT THE SAME tiny shape as
//!    `gdn_chunk_fwd.rs` (`B=1,H=2,T=8,Dk=3,Dv=4,C=4`), perturbing each of
//!    `query`/`key`/`value`/`raw_g`/`beta`/`initial_state` by `+-eps`,
//!    central differences, f64, comparing against the analytic backward's
//!    device output. `docs/porting-playbook.md` §8's own numbers (block FD
//!    < 1e-4, model FD < 1e-3) are the right order of magnitude to target.
//! 3. Both backends (`BRAIN_DEVICE=cpu` and default), per `docs/lessons.md`
//!    #5 — a barrier-crossing backward kernel can return all-zero gradients
//!    on exactly one backend with no error.
//!
//! This test is `#[ignore]`d rather than deleted so `cargo test -p
//! brain-model` lists it (and its reason) instead of it silently not
//! existing.

#[test]
#[ignore = "model::gdn backward is not implemented -- see this file's module doc"]
fn gdn_chunk_bwd_gradcheck() {
    unimplemented!(
        "model::gdn::gdn_chunk_bwd does not exist yet -- forward-only, see \
         crates/model/src/gdn.rs's \"Backward\" doc section and this file's own module doc"
    );
}
