// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Capture-to-send latency statistics for the video path.
//!
//! The sensor timestamp on a frame is the **start** of its readout. This
//! sensor reads a frame out line by line over most of the frame period (33 ms
//! of 33.3 ms at 30 fps: the capture driver reports `readout: 33 ms`), and the
//! ISP runs online on the lines as they arrive, finishing when the readout
//! does (it reports `delay: 32 ms`). So a frame is captured, all of it, at
//! about the moment it is dequeued here, and every series is split there:
//!
//! - `readout`: sensor timestamp to dequeue, the sensor's readout and the
//!   ISP's tail, fixed by the sensor mode;
//! - `captured_to_send`: dequeue to the last RTP packet written to the
//!   socket, the pipeline's own latency, broken down by stage;
//! - `readout_start_to_send`: the two together.
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
/// and sum to `captured_to_send`, apart from the bookkeeping between them.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sample {
    /// Sensor timestamp (start of readout) to last packet written.
    pub readout_start_to_send: u64,
    /// Sensor timestamp to the complete frame dequeued from the camera.
    pub readout: u64,
    /// Complete frame dequeued to last packet written.
    pub captured_to_send: u64,
    /// Detector tap: scale and convert in the 2D accelerator.
    pub tap: u64,
    /// Drawing the detection boxes.
    pub overlay: u64,
    /// Hardware encoder, from sending the frame to getting its bytes back.
    pub encode: u64,
    /// Packetizing and writing to the clients' sockets.
    pub send: u64,
}

const SERIES: usize = 7;

impl Sample {
    fn series(&self) -> [u64; SERIES] {
        [self.readout_start_to_send, self.readout, self.captured_to_send, self.tap, self.overlay, self.encode, self.send]
    }
}

/// One periodic summary.
#[derive(Clone, Copy, Debug)]
pub struct Report {
    pub frames: usize,
    pub fps: f64,
    pub readout_start_to_send: Percentiles,
    pub readout: Percentiles,
    pub captured_to_send: Percentiles,
    pub tap: Percentiles,
    pub overlay: Percentiles,
    pub encode: Percentiles,
    pub send: Percentiles,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} frames, {:.1} fps | frame captured -> sent p50 {:.1} ms p95 {:.1} ms max {:.1} ms \
             | sensor readout+ISP p50 {:.1} ms | readout start -> sent p50 {:.1} ms p95 {:.1} ms \
             | p50 per stage: tap {:.1} overlay {:.1} encode {:.1} send {:.1} ms",
            self.frames,
            self.fps,
            self.captured_to_send.p50,
            self.captured_to_send.p95,
            self.captured_to_send.max,
            self.readout.p50,
            self.readout_start_to_send.p50,
            self.readout_start_to_send.p95,
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
        let [readout_start_to_send, readout, captured_to_send, tap, overlay, encode, send] =
            std::array::from_fn(|i| percentiles(&mut self.samples[i]));
        for series in &mut self.samples {
            series.clear();
        }
        self.window_start = Instant::now();
        Some(Report {
            frames,
            fps: if elapsed > 0.0 { frames as f64 / elapsed } else { 0.0 },
            readout_start_to_send,
            readout,
            captured_to_send,
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
        let frame = |readout_start_to_send, captured_to_send| Sample { readout_start_to_send, captured_to_send, ..Sample::default() };
        assert!(stats.record(frame(10_000, 5_000)).is_none());
        assert!(stats.record(frame(20_000, 6_000)).is_none());
        let r = stats.record(frame(30_000, 7_000)).expect("third frame completes the window");
        assert_eq!((r.frames, r.readout_start_to_send.max, r.captured_to_send.p50), (3, 30.0, 6.0));
        assert!(stats.record(frame(1_000, 1_000)).is_none(), "the next window starts empty");
    }
}
