// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-time object detection on a Rockchip RV1103/RV1106 board.
//!
//! `--stream` runs the camera -> H.264 -> RTSP pipeline with detection boxes
//! drawn on the picture; see `stream` for how the video path is arranged and
//! `detect` for the seam the detector plugs into. Without `--stream` the
//! program loads the detection model and reports its memory use.

mod camera;
mod cli;
mod config;
mod demo;
mod detect;
mod encoder;
mod error;
mod ffi;
mod latency;
mod media;
mod memory;
mod overlay;
mod rtsp;
mod signals;
mod stream;
mod tap;

use cli::{Command, StreamOptions};
use error::{Context, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() {
    let command = match cli::parse(std::env::args().skip(1)) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("error: {message}\n\n{}", cli::USAGE);
            std::process::exit(2);
        }
    };
    let result = match command {
        Command::Help => {
            println!("{}", cli::USAGE);
            Ok(())
        }
        Command::Detect { model } => {
            load_model(model);
            Ok(())
        }
        Command::Stream(options) => run_stream(options),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn load_model(model: String) {
    let pipe = brain::DetectionPipeline::from_pretrained(model);
    println!("load ok: {}", pipe.is_ok());
    if let Err(e) = pipe {
        println!("{e}");
    }
    println!("{}", memory::report());
}

/// How often the stream's memory use is reported.
const MEMORY_REPORT_EVERY: Duration = Duration::from_secs(5);

fn run_stream(options: StreamOptions) -> Result<()> {
    ffi::dl::ensure_vendor_search_path()?;
    signals::install()?;
    eprintln!("before start: {}", memory::report());
    let StreamOptions { config, duration, dump_tap, demo_box } = options;
    let stream = stream::Stream::start(config.clone())?;

    // The helpers use only the seam a real detector uses.
    let helpers_stop = Arc::new(AtomicBool::new(false));
    let mut helpers = Vec::new();
    if demo_box {
        let (board, stop) = (Arc::clone(&stream.board), Arc::clone(&helpers_stop));
        helpers.push(std::thread::spawn(move || demo::sweep_box(board, config.width, config.height, stop)));
    }
    let tap_thread = {
        let (frames, stop) = (stream.frames.clone(), Arc::clone(&helpers_stop));
        std::thread::spawn(move || demo::drain_tap(frames, dump_tap.as_deref(), stop))
    };

    let started = Instant::now();
    let mut last_report = started;
    while !signals::requested() && stream.is_running() && duration.is_none_or(|d| started.elapsed() < d) {
        std::thread::sleep(Duration::from_millis(100));
        if last_report.elapsed() >= MEMORY_REPORT_EVERY {
            eprintln!("memory: {}", memory::report());
            last_report = Instant::now();
        }
    }

    helpers_stop.store(true, Ordering::Release);
    for h in helpers {
        let _ = h.join();
    }
    let tapped = tap_thread.join().map_err(|_| error::Error::new("the tap consumer panicked"))?;
    let result = stream.stop();
    eprintln!("after stop: {}", memory::report());
    eprintln!("tap delivered {} frames", tapped.context("draining the frame tap")?);
    result
}
