// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Command line of the sample. Hand-written because the board has no room for
//! an argument-parsing dependency and the surface is a dozen flags.

use crate::config::StreamConfig;
use std::path::PathBuf;
use std::time::Duration;

pub const USAGE: &str = "\
usage: sample-npu-rockchip [MODEL_DIR]
       sample-npu-rockchip --stream [options]

Without --stream, loads the detection model in MODEL_DIR and reports memory.

--stream streams the camera over RTSP with detection boxes drawn on it:
  --width N            stream width in pixels (default 800, max 2304)
  --height N           stream height in pixels (default 448, max 1296)
  --fps N              the sensor's frame rate (default 30, what the SC3336 delivers)
  --bitrate KBPS       H.264 bit rate in kbit/s (default 2048)
  --gop N              frames between IDR frames (default 250)
  --refresh-rows N     intra-refresh macroblock rows per frame, 0 = off (default 2)
  --port N             RTSP port (default 8554)
  --path PATH          RTSP mount point (default /live/0)
  --iq-dir DIR         sensor tuning files (default /etc/iqfiles)
  --report-every N     frames between latency reports (default 100)
  --duration SECS      stop after this long (default: run until Ctrl-C)
  --dump-tap FILE      write a detector tap image (taken 60 frames in) to FILE as a PPM
  --no-demo-box        do not draw the moving demo box";

/// What the program was asked to do.
#[derive(Debug, PartialEq)]
pub enum Command {
    Help,
    /// Load the detection model in this directory.
    Detect { model: String },
    Stream(StreamOptions),
}

#[derive(Debug, PartialEq)]
pub struct StreamOptions {
    pub config: StreamConfig,
    pub duration: Option<Duration>,
    pub dump_tap: Option<PathBuf>,
    pub demo_box: bool,
}

pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let mut stream = false;
    let mut model = None;
    let mut opts = StreamOptions { config: StreamConfig::default(), duration: None, dump_tap: None, demo_box: true };
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--stream" => stream = true,
            "--no-demo-box" => opts.demo_box = false,
            "--width" => opts.config.width = number(&arg, &value(&arg)?)?,
            "--height" => opts.config.height = number(&arg, &value(&arg)?)?,
            "--fps" => opts.config.fps = number(&arg, &value(&arg)?)?,
            "--bitrate" => opts.config.bitrate_kbps = number(&arg, &value(&arg)?)?,
            "--gop" => opts.config.gop = number(&arg, &value(&arg)?)?,
            "--refresh-rows" => opts.config.intra_refresh_rows = number(&arg, &value(&arg)?)?,
            "--port" => opts.config.rtsp_port = number(&arg, &value(&arg)?)?,
            "--path" => opts.config.rtsp_path = value(&arg)?,
            "--iq-dir" => opts.config.iq_dir = value(&arg)?,
            "--report-every" => opts.config.report_every = number(&arg, &value(&arg)?)?,
            "--duration" => opts.duration = Some(Duration::from_secs(number(&arg, &value(&arg)?)?)),
            "--dump-tap" => opts.dump_tap = Some(PathBuf::from(value(&arg)?)),
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            positional if model.is_none() => model = Some(positional.to_string()),
            extra => return Err(format!("unexpected argument {extra}")),
        }
    }
    if stream {
        opts.config.validate()?;
        Ok(Command::Stream(opts))
    } else {
        Ok(Command::Detect { model: model.unwrap_or_default() })
    }
}

fn number<T: std::str::FromStr>(flag: &str, text: &str) -> Result<T, String> {
    text.parse().map_err(|_| format!("{flag} expects a non-negative whole number, got {text:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_args(args: &[&str]) -> Result<Command, String> {
        parse(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn stream_defaults_fit_the_cma_budget_and_serve_rtsp_on_8554() {
        let Ok(Command::Stream(o)) = parse_args(&["--stream"]) else { panic!("not a stream command") };
        assert_eq!((o.config.width, o.config.height, o.config.rtsp_port), (800, 448, 8554));
        assert!(o.demo_box && o.duration.is_none());
    }

    #[test]
    fn options_override_the_defaults() {
        let Ok(Command::Stream(o)) =
            parse_args(&["--stream", "--width", "1920", "--height", "1080", "--fps", "30", "--duration", "20", "--no-demo-box"])
        else {
            panic!("not a stream command")
        };
        assert_eq!((o.config.width, o.config.height, o.config.fps), (1920, 1080, 30));
        assert_eq!(o.duration, Some(Duration::from_secs(20)));
        assert!(!o.demo_box);
    }

    #[test]
    fn bad_input_is_rejected_with_a_reason() {
        assert!(parse_args(&["--stream", "--width", "wide"]).unwrap_err().contains("--width"));
        assert!(parse_args(&["--stream", "--width"]).unwrap_err().contains("needs a value"));
        assert!(parse_args(&["--stream", "--width", "4000"]).unwrap_err().contains("exceeds the sensor"));
        assert!(parse_args(&["--stream", "--fps", "0"]).unwrap_err().contains("fps"));
        assert!(parse_args(&["--bogus"]).unwrap_err().contains("unknown option"));
    }

    #[test]
    fn without_stream_the_first_argument_is_the_model() {
        assert_eq!(parse_args(&["models/yolo"]), Ok(Command::Detect { model: "models/yolo".into() }));
        assert_eq!(parse_args(&[]), Ok(Command::Detect { model: String::new() }));
    }
}
