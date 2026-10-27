// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Command line of the sample. Hand-written because the board has no room for
//! an argument-parsing dependency and the surface is a dozen flags.

use crate::config::StreamConfig;
use std::path::PathBuf;
use std::time::Duration;

pub const USAGE: &str = "\
usage: sample-npu-rockchip --model MODEL --image FILE [--conf X] [--iou X]
       sample-npu-rockchip --model MODEL --stream [options]

MODEL is a brain model id (`local/yolov5s-relu`): a YOLOv5 checkpoint
(model.safetensors) with its NPU calibration (model.npu-calibration.json)
in <models dir>/<id>/.

--image FILE detects objects in one JPEG/PNG image and prints them as JSON,
in the image's pixels.

--stream streams the camera over RTSP with the detected boxes drawn on it:
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
  --dump-tap FILE      write the first detector input image to FILE as a PPM

Both modes:
  --models-dir DIR     where model ids resolve (default: models/ beside this program)
  --conf X             minimum confidence to keep a detection (default 0.25)
  --iou X              NMS overlap threshold (default 0.45)";

/// What the program was asked to do.
#[derive(Debug, PartialEq)]
pub enum Command {
    Help,
    /// Detect in one image file and print the result.
    Image { model: ModelChoice, path: PathBuf, thresholds: Thresholds },
    Stream(StreamOptions),
}

/// Which model, and the models directory its id resolves in (`None`: the
/// `models` directory beside the program).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelChoice {
    pub id: String,
    pub models_dir: Option<PathBuf>,
}

/// Detection thresholds, applied by `brain::DetectOptions`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    pub confidence: f32,
    pub iou: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds { confidence: 0.25, iou: 0.45 }
    }
}

impl Thresholds {
    pub fn options(&self) -> brain::DetectOptions {
        brain::DetectOptions::new().confidence(self.confidence).iou(self.iou)
    }
}

#[derive(Debug, PartialEq)]
pub struct StreamOptions {
    pub model: ModelChoice,
    pub thresholds: Thresholds,
    pub config: StreamConfig,
    pub duration: Option<Duration>,
    pub dump_tap: Option<PathBuf>,
}

pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let (mut stream, mut image, mut model, mut models_dir) = (false, None, None, None);
    let mut thresholds = Thresholds::default();
    let mut config = StreamConfig::default();
    let (mut duration, mut dump_tap) = (None, None);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--model" => model = Some(value(&arg)?),
            "--models-dir" => models_dir = Some(PathBuf::from(value(&arg)?)),
            "--image" => image = Some(PathBuf::from(value(&arg)?)),
            "--stream" => stream = true,
            "--conf" => thresholds.confidence = fraction(&arg, &value(&arg)?)?,
            "--iou" => thresholds.iou = fraction(&arg, &value(&arg)?)?,
            "--width" => config.width = number(&arg, &value(&arg)?)?,
            "--height" => config.height = number(&arg, &value(&arg)?)?,
            "--fps" => config.fps = number(&arg, &value(&arg)?)?,
            "--bitrate" => config.bitrate_kbps = number(&arg, &value(&arg)?)?,
            "--gop" => config.gop = number(&arg, &value(&arg)?)?,
            "--refresh-rows" => config.intra_refresh_rows = number(&arg, &value(&arg)?)?,
            "--port" => config.rtsp_port = number(&arg, &value(&arg)?)?,
            "--path" => config.rtsp_path = value(&arg)?,
            "--iq-dir" => config.iq_dir = value(&arg)?,
            "--report-every" => config.report_every = number(&arg, &value(&arg)?)?,
            "--duration" => duration = Some(Duration::from_secs(number(&arg, &value(&arg)?)?)),
            "--dump-tap" => dump_tap = Some(PathBuf::from(value(&arg)?)),
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            extra => return Err(format!("unexpected argument {extra}")),
        }
    }
    let model = ModelChoice { id: model.ok_or("--model is required")?, models_dir };
    match (stream, image) {
        (true, Some(_)) => Err("--stream and --image are separate modes; pick one".into()),
        (true, None) => {
            config.validate()?;
            Ok(Command::Stream(StreamOptions { model, thresholds, config, duration, dump_tap }))
        }
        (false, Some(path)) => Ok(Command::Image { model, path, thresholds }),
        (false, None) => Err("give --image FILE or --stream".into()),
    }
}

fn number<T: std::str::FromStr>(flag: &str, text: &str) -> Result<T, String> {
    text.parse().map_err(|_| format!("{flag} expects a non-negative whole number, got {text:?}"))
}

fn fraction(flag: &str, text: &str) -> Result<f32, String> {
    match text.parse::<f32>() {
        Ok(v) if (0.0..=1.0).contains(&v) => Ok(v),
        _ => Err(format!("{flag} expects a number from 0 to 1, got {text:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_args(args: &[&str]) -> Result<Command, String> {
        parse(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn stream_defaults_fit_the_cma_budget_and_serve_rtsp_on_8554() {
        let Ok(Command::Stream(o)) = parse_args(&["--model", "m", "--stream"]) else { panic!("not a stream command") };
        assert_eq!((o.config.width, o.config.height, o.config.rtsp_port), (800, 448, 8554));
        assert_eq!((o.model.id.as_str(), o.model.models_dir, o.thresholds, o.duration), ("m", None, Thresholds::default(), None));
    }

    #[test]
    fn options_override_the_defaults() {
        let Ok(Command::Stream(o)) = parse_args(&["--model", "m", "--stream", "--width", "1920", "--height", "1080", "--fps", "30", "--duration", "20", "--conf", "0.5"]) else {
            panic!("not a stream command")
        };
        assert_eq!((o.config.width, o.config.height, o.config.fps), (1920, 1080, 30));
        assert_eq!(o.duration, Some(Duration::from_secs(20)));
        assert_eq!(o.thresholds.confidence, 0.5);
    }

    #[test]
    fn image_mode_takes_a_file() {
        assert_eq!(
            parse_args(&["--model", "m", "--models-dir", "d", "--image", "a.jpg", "--iou", "0.6"]),
            Ok(Command::Image { model: ModelChoice { id: "m".into(), models_dir: Some("d".into()) }, path: "a.jpg".into(), thresholds: Thresholds { confidence: 0.25, iou: 0.6 } })
        );
    }

    #[test]
    fn bad_input_is_rejected_with_a_reason() {
        assert!(parse_args(&["--model", "m", "--stream", "--width", "wide"]).unwrap_err().contains("--width"));
        assert!(parse_args(&["--model", "m", "--stream", "--width"]).unwrap_err().contains("needs a value"));
        assert!(parse_args(&["--model", "m", "--stream", "--width", "4000"]).unwrap_err().contains("exceeds the sensor"));
        assert!(parse_args(&["--model", "m", "--stream", "--fps", "0"]).unwrap_err().contains("fps"));
        assert!(parse_args(&["--model", "m", "--image", "a", "--conf", "2"]).unwrap_err().contains("0 to 1"));
        assert!(parse_args(&["--bogus"]).unwrap_err().contains("unknown option"));
        assert!(parse_args(&["--stream"]).unwrap_err().contains("--model"));
        assert!(parse_args(&["--model", "m"]).unwrap_err().contains("--image FILE or --stream"));
        assert!(parse_args(&["--model", "m", "--stream", "--image", "a"]).unwrap_err().contains("separate modes"));
    }
}
