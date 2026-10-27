// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-time object detection on a Rockchip RV1103/RV1106 board, with brain's
//! `DetectionPipeline` running YOLOv5 on the NPU.
//!
//! `--image` detects in one picture and prints the boxes as JSON. `--stream`
//! runs the camera -> H.264 -> RTSP pipeline with the detector beside it: see
//! `stream` for how the video path is arranged, `detect` for the seam between
//! the two and `detector` for the detection thread.

mod camera;
mod cli;
mod config;
mod detect;
mod detector;
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

use cli::{Command, ModelChoice, StreamOptions, Thresholds};
use error::{Context, Error, Result};
use std::path::Path;
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
        Command::Image { model, path, thresholds } => detect_image(&model, &path, thresholds),
        Command::Stream(options) => run_stream(options),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Load the model onto the NPU. Model ids resolve in the `models` directory
/// beside this program unless told otherwise, so a copied `brain/` directory
/// (the program and its `models/`) runs from wherever it is put.
fn load_pipeline(model: &ModelChoice) -> Result<brain::DetectionPipeline> {
    let models_dir = match &model.models_dir {
        Some(dir) => dir.clone(),
        None => std::env::current_exe().context("locating this program")?.with_file_name("models"),
    };
    let started = Instant::now();
    let pipeline = brain::DetectionPipeline::builder(&model.id)
        .models_dir(models_dir.to_string_lossy())
        .download_policy(brain::DownloadPolicy::Offline)
        .load()
        .map_err(|e| Error::new(format!("loading {} from {}: {e}", model.id, models_dir.display())))?;
    eprintln!("model {} loaded in {:.1} s; {}", model.id, started.elapsed().as_secs_f64(), memory::report());
    Ok(pipeline)
}

/// Detect in one image and print one JSON object per detection, best first.
fn detect_image(model: &ModelChoice, path: &Path, thresholds: Thresholds) -> Result<()> {
    let pipeline = load_pipeline(model)?;
    let image = brain::Image::open(path).map_err(|e| Error::new(format!("{}: {e}", path.display())))?;
    let started = Instant::now();
    let found = pipeline.detect_with(&image, thresholds.options()).map_err(|e| Error::new(format!("detection: {e}")))?;
    eprintln!("{}x{} image: {} detections in {:.1} ms", image.width(), image.height(), found.len(), started.elapsed().as_secs_f64() * 1e3);
    for d in &found {
        let label = pipeline.class_name(d.class).unwrap_or("?");
        println!(
            "{{\"label\":\"{label}\",\"class\":{},\"confidence\":{:.3},\"box\":[{:.1},{:.1},{:.1},{:.1}]}}",
            d.class, d.confidence, d.x1, d.y1, d.x2, d.y2
        );
    }
    Ok(())
}

/// How often the stream's memory use is reported.
const MEMORY_REPORT_EVERY: Duration = Duration::from_secs(5);

fn run_stream(options: StreamOptions) -> Result<()> {
    ffi::dl::ensure_vendor_search_path()?;
    signals::install()?;
    let StreamOptions { model, thresholds, config, duration, dump_tap } = options;
    // The model first: its NPU memory is the largest contiguous allocation,
    // and it is most likely to succeed before the camera fragments the pool.
    let pipeline = load_pipeline(&model)?;
    let stream = stream::Stream::start(config)?;
    eprintln!("streaming: {}", memory::report());

    let stop = Arc::new(AtomicBool::new(false));
    let detector = {
        let (frames, board, stop) = (stream.frames.clone(), Arc::clone(&stream.board), Arc::clone(&stop));
        std::thread::spawn(move || detector::run(pipeline, thresholds.options(), frames, board, dump_tap, stop))
    };

    let started = Instant::now();
    let mut last_report = started;
    while !signals::requested() && stream.is_running() && !detector.is_finished() && duration.is_none_or(|d| started.elapsed() < d) {
        std::thread::sleep(Duration::from_millis(100));
        if last_report.elapsed() >= MEMORY_REPORT_EVERY {
            eprintln!("memory: {}", memory::report());
            last_report = Instant::now();
        }
    }

    stop.store(true, Ordering::Release);
    let detected = detector.join().map_err(|_| Error::new("the detector thread panicked"))?;
    let result = stream.stop();
    eprintln!("after stop: {}", memory::report());
    let stats = detected.context("detecting")?;
    eprintln!("detector: {} frames, {} detections", stats.frames, stats.detections);
    result
}
