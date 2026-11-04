// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fault-tolerant distributed training for its
// clients. If your team needs expertise in collective communication failure
// semantics or multi-node training runtimes, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A collective failure is an error the caller can handle, not a panic.
//!
//! The transport can fail for reasons that say nothing about the model: a peer
//! rank died, a socket closed, a coordinator went away. A training step must
//! report that as an error and leave the replica's weights untouched, so the
//! driver can stop, checkpoint the last committed step and restart the job
//! instead of tearing the process down.

use model::collective::{Collective, CollectiveError, CollectiveFuture, CollectiveResult, Payload};
use model::distributed::{fed_avg_flat, FlatAdam};

/// Every op fails the way a lost peer does.
struct LostPeer;

impl Collective for LostPeer {
    fn world_size(&self) -> usize {
        2
    }

    fn all_reduce<'a>(&'a self, _rank: usize, _local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async { Err(CollectiveError::Transport("rank 1 closed the connection".into())) })
    }

    fn all_gather<'a>(&'a self, _rank: usize, _local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async { Err(CollectiveError::Transport("rank 1 closed the connection".into())) })
    }

    fn reduce_scatter<'a>(&'a self, _rank: usize, _local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async { Err(CollectiveError::Transport("rank 1 closed the connection".into())) })
    }

    fn broadcast<'a>(&'a self, _rank: usize, _local: Payload, _root: usize) -> CollectiveFuture<'a, Payload> {
        Box::pin(async { Err(CollectiveError::Transport("rank 1 closed the connection".into())) })
    }
}

#[test]
fn a_lost_peer_fails_the_optimiser_step_without_touching_weights() {
    let mut opt = FlatAdam::new(vec![1.0, -2.0, 0.5]);
    let before = opt.weights().to_vec();
    let err = opt
        .step(&LostPeer, 0, vec![0.25, 0.25, 0.25], 1, 1e-2, 0.0, model::Adam::default(), None)
        .expect_err("a transport failure must surface as an error");
    assert!(matches!(err, CollectiveError::Transport(_)), "unexpected error: {err}");
    assert_eq!(opt.weights(), before.as_slice(), "a failed step must leave the weights unchanged");
}

#[test]
fn a_lost_peer_fails_the_federated_average() {
    let err = fed_avg_flat(&LostPeer, 0, &[1.0, 2.0], 4.0).expect_err("a transport failure must surface as an error");
    assert!(matches!(err, CollectiveError::Transport(_)), "unexpected error: {err}");
}

/// A validation failure the local rank detects is an error too, and is not
/// mistaken for a transport problem.
struct RejectsRank;

impl Collective for RejectsRank {
    fn world_size(&self) -> usize {
        2
    }

    fn all_reduce<'a>(&'a self, rank: usize, _local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async move { Err(CollectiveError::RankOutOfRange { rank, world_size: 2 }) })
    }

    fn all_gather<'a>(&'a self, _rank: usize, _local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async { unreachable!("not used") })
    }

    fn reduce_scatter<'a>(&'a self, _rank: usize, _local: Payload) -> CollectiveFuture<'a, Payload> {
        Box::pin(async { unreachable!("not used") })
    }

    fn broadcast<'a>(&'a self, _rank: usize, _local: Payload, _root: usize) -> CollectiveFuture<'a, Payload> {
        Box::pin(async { unreachable!("not used") })
    }
}

#[test]
fn a_validation_failure_keeps_its_own_error_kind() {
    let mut opt = FlatAdam::new(vec![1.0]);
    let err = opt
        .step(&RejectsRank, 7, vec![1.0], 1, 1e-2, 0.0, model::Adam::default(), None)
        .expect_err("an out-of-range rank must surface as an error");
    assert_eq!(err, CollectiveError::RankOutOfRange { rank: 7, world_size: 2 });
}

/// A one-rank group still succeeds and updates the weights (the regression
/// guard for the error path not swallowing the normal one).
#[test]
fn a_one_rank_group_still_updates_the_weights() {
    let coll = model::HostCollective::new(1);
    let mut opt = FlatAdam::new(vec![0.0]);
    let w = opt
        .step(&*coll, 0, vec![2.0], 1, 0.5, 0.0, model::Adam::default(), None)
        .expect("a one-rank collective cannot fail");
    assert!(w[0] < 0.0, "the step did not descend: {}", w[0]);
}

#[allow(dead_code)]
fn _assert_result_type(_: CollectiveResult<Payload>) {}
