// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Capture-to-send latency statistics for the video path.
//!
//! For every frame this keeps the time from the sensor timestamp on the frame
//! to the moment its last RTP packet was written to the socket, the part of
//! that spent inside this process after the frame was dequeued from the
//! camera, and that part broken down by stage. The difference between the
//! first two is the sensor readout and ISP, which the application cannot
//! change.
//!
//! Swedish Embedded AB implements low-latency camera and video pipelines for
//! embedded Linux boards for its clients. If your team needs expertise in
//! measuring and cutting video latency then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::fmt;
use std::time::Instant;

/// Percentiles of a series, in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Percentiles {
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

/// What one frame cost, in microseconds. The stages run in the order listed
/// and sum to `in_process`, apart from the bookkeeping between them.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    /// Sensor timestamp to last packet written.
    pub capture_to_send: u64,
    /// Frame dequeued from the camera to last packet written.
    pub in_process: u64,
    /// Detector tap: scale and convert in the 2D accelerator.
    pub tap: u64,
    /// Drawing the detection boxes.
    pub overlay: u64,
    /// Hardware encoder, from sending the frame to getting its bytes back.
    pub encode: u64,
    /// Packetizing and writing to the clients' sockets.
    pub send: u64,
}

const SERIES: usize = 6;

impl Sample {
    fn series(&self) -> [u64; SERIES] {
        [self.capture_to_send, self.in_process, self.tap, self.overlay, self.encode, self.send]
    }
}

/// One periodic summary.
#[derive(Clone, Copy, Debug)]
pub struct Report {
    pub frames: usize,
    pub fps: f64,
    pub capture_to_send: Percentiles,
    pub in_process: Percentiles,
    pub tap: Percentiles,
    pub overlay: Percentiles,
    pub encode: Percentiles,
    pub send: Percentiles,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} frames, {:.1} fps | capture->send p50 {:.1} ms p95 {:.1} ms max {:.1} ms \
             | in-process p50 {:.1} ms p95 {:.1} ms \
             | p50 per stage: tap {:.1} overlay {:.1} encode {:.1} send {:.1} ms",
            self.frames,
            self.fps,
            self.capture_to_send.p50,
            self.capture_to_send.p95,
            self.capture_to_send.max,
            self.in_process.p50,
            self.in_process.p95,
            self.tap.p50,
            self.overlay.p50,
            self.encode.p50,
            self.send.p50,
        )
    }
}

/// Collects per-frame costs and emits a [`Report`] every `report_every`
/// frames, over exactly those frames.
pub struct LatencyStats {
    report_every: usize,
    samples: [Vec<u64>; SERIES],
    window_start: Instant,
}

impl LatencyStats {
    pub fn new(report_every: usize) -> Self {
        assert!(report_every > 0, "a report needs at least one frame");
        LatencyStats {
            report_every,
            samples: std::array::from_fn(|_| Vec::with_capacity(report_every)),
            window_start: Instant::now(),
        }
    }

    /// Records one frame; returns a report when the window is full.
    pub fn record(&mut self, sample: Sample) -> Option<Report> {
        for (series, value) in self.samples.iter_mut().zip(sample.series()) {
            series.push(value);
        }
        if self.samples[0].len() < self.report_every {
            return None;
        }
        let frames = self.samples[0].len();
        let elapsed = self.window_start.elapsed().as_secs_f64();
        let [capture_to_send, in_process, tap, overlay, encode, send] =
            std::array::from_fn(|i| percentiles(&mut self.samples[i]));
        for series in &mut self.samples {
            series.clear();
        }
        self.window_start = Instant::now();
        Some(Report {
            frames,
            fps: if elapsed > 0.0 { frames as f64 / elapsed } else { 0.0 },
            capture_to_send,
            in_process,
            tap,
            overlay,
            encode,
            send,
        })
    }
}

/// Nearest-rank percentiles of microsecond samples, reported in milliseconds.
/// Sorts `samples` in place.
fn percentiles(samples: &mut [u64]) -> Percentiles {
    samples.sort_unstable();
    let rank = |p: f64| samples[((p * samples.len() as f64).ceil() as usize).clamp(1, samples.len()) - 1];
    let ms = |us: u64| us as f64 / 1000.0;
    Percentiles { p50: ms(rank(0.50)), p95: ms(rank(0.95)), max: ms(samples[samples.len() - 1]) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_use_nearest_rank() {
        let mut s: Vec<u64> = (1..=100).map(|i| i * 1000).collect();
        s.reverse();
        assert_eq!(percentiles(&mut s), Percentiles { p50: 50.0, p95: 95.0, max: 100.0 });
    }

    #[test]
    fn a_report_is_emitted_once_per_window_and_the_window_restarts() {
        let mut stats = LatencyStats::new(3);
        let frame = |capture_to_send, in_process| Sample { capture_to_send, in_process, ..Sample::default() };
        assert!(stats.record(frame(10_000, 5_000)).is_none());
        assert!(stats.record(frame(20_000, 6_000)).is_none());
        let r = stats.record(frame(30_000, 7_000)).expect("third frame completes the window");
        assert_eq!((r.frames, r.capture_to_send.max, r.in_process.p50), (3, 30.0, 6.0));
        assert!(stats.record(frame(1_000, 1_000)).is_none(), "the next window starts empty");
    }
}
