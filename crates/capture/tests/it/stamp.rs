// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which capture time a dequeued buffer gets: the driver's, when it is a
//! plausible monotonic reading, and the dequeue time, honestly marked,
//! otherwise.
#![cfg(target_os = "linux")]

use capture::clock::monotonic_ns;
use capture::v4l2::{stamp, V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC, V4L2_BUF_FLAG_TSTAMP_SRC_SOE};
use capture::StampSource;

const NOW: u64 = 5_000_000_000_000;

fn tv(ns: u64) -> (i64, i64) {
    ((ns / 1_000_000_000) as i64, ((ns % 1_000_000_000) / 1_000) as i64)
}

#[test]
fn a_recent_monotonic_driver_stamp_is_used_with_its_source() {
    let (s, u) = tv(NOW - 33_000_000);
    let eof = stamp(V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC, s, u, NOW);
    assert_eq!((eof.ns, eof.source), (NOW - 33_000_000, StampSource::FrameEnd));
    let soe = stamp(V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC | V4L2_BUF_FLAG_TSTAMP_SRC_SOE, s, u, NOW);
    assert_eq!(soe.source, StampSource::Exposure);
}

#[test]
fn an_implausible_driver_stamp_falls_back_to_the_dequeue_time() {
    let dequeue = |flags, s, u| stamp(flags, s, u, NOW).source == StampSource::Dequeue;
    let (s, u) = tv(NOW - 33_000_000);
    assert!(dequeue(0, s, u), "no monotonic flag: some other clock");
    let (fs, fu) = tv(NOW + 1_000_000);
    assert!(dequeue(V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC, fs, fu), "from the future");
    let (os, ou) = tv(NOW - 10_000_000_000);
    assert!(dequeue(V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC, os, ou), "older than any queue holds a frame");
    assert!(dequeue(V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC, 0, 0), "unset");
    assert!(dequeue(V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC, s, 2_000_000), "microseconds out of range");
    assert_eq!(stamp(0, s, u, NOW).ns, NOW);
}

#[test]
fn the_clock_is_monotonic() {
    let a = monotonic_ns();
    let b = monotonic_ns();
    assert!(b >= a && a > 0);
}
