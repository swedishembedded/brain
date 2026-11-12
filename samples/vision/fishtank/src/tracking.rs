// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Webcam head tracking: the threads between the camera and the eye the
//! renderer reads.
//!
//! ```text
//! capture thread   V4L2 DQBUF -> RGB + driver timestamp -> latest-frame slot
//! tracking thread  slot -> OpenSeeFace landmarks -> headpose::fit_head
//!                  -> camera mount -> ViewerEstimator::observe(capture time)
//! render thread    ViewerEstimator::predict(presentation time)
//! ```
//!
//! Neither side waits for the other: the slot keeps only the newest frame
//! (an inference slower than the camera drops frames instead of building
//! latency), and the renderer queries the estimator at its own rate. Every
//! time is `capture::clock::monotonic_ns`, the clock the driver stamps frames
//! with, so a frame's age is a subtraction.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use capture::{Frame, FrameSlot};
use headpose::calibration::Profile;
use headpose::{fit_head, Calibration, EstimatorConfig, FaceModel, TrackedViewer, ViewerEstimator};
use openseeface::config::{NetConfig, LANDMARKS_BEST, LANDMARKS_FAST};
use openseeface::{Tracker, TrackerConfig};

/// The model-store id the weights live under (see `fetch-data.sh`).
pub const WEIGHTS_ID: &str = "emilianavt/OpenSeeFace";

/// Landmarks below this confidence do not enter the head fit.
const MIN_POINT_CONFIDENCE: f64 = 0.4;
/// A fit that misses its landmarks by more than this fraction of the outer
/// eye corners' distance in the image is rejected: a residual means the same
/// thing at every distance from the camera only relative to the face's size.
const MAX_RMSE_OF_FACE: f64 = 0.08;

/// What the tracker last did, for the HUD.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub camera: String,
    pub frames: u64,
    pub dropped: u64,
    pub detections: u64,
    pub inference_ms: f64,
    pub fit_rmse_px: f64,
    pub confidence: f64,
    pub last_problem: String,
}

pub struct Tracking {
    pub estimator: Arc<Mutex<ViewerEstimator>>,
    pub calibration: Arc<Mutex<Calibration>>,
    pub stats: Arc<Mutex<Stats>>,
    /// The flat profile the calibration came from, rewritten on recentre.
    profile: Mutex<Profile>,
    save_to: PathBuf,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
}

/// Where the weights are, or an error that says how to get them.
pub fn weights_dir() -> Result<PathBuf, String> {
    let root = brain_modelstore::default_root().ok_or("no models directory: set BRAIN_MODELS_DIR")?;
    let id = brain_modelref::ModelRef::parse(WEIGHTS_ID).map_err(|e| format!("{WEIGHTS_ID}: {e}"))?;
    let dir = brain_modelstore::Store::new(root).repo_dir(&id);
    for f in [openseeface::config::DETECTOR.file, LANDMARKS_FAST.file, LANDMARKS_BEST.file] {
        if !dir.join(f).exists() {
            return Err(format!("{} is missing; fetch the face tracker first:\n    samples/vision/fishtank/fetch-data.sh", dir.join(f).display()));
        }
    }
    Ok(dir)
}

/// Where frames come from.
#[derive(Clone, Debug)]
pub enum Camera {
    /// A V4L2 device.
    Device(String),
    /// A still image replayed at 30 Hz, panned sideways by up to
    /// `pan_px` pixels so the tracked eye moves: the whole chain exercised
    /// without a camera. Stamped with the replay clock, and always labelled
    /// as a replay.
    Image { path: PathBuf, pan_px: u32 },
}

pub struct Options {
    pub camera: Camera,
    pub size: (u32, u32),
    pub landmarks: NetConfig,
    /// Where the face networks run (`cpu`, `gpu`, `gpu<i>`).
    pub device: String,
    pub profile: Profile,
    pub save_to: PathBuf,
}

impl Tracking {
    pub fn start(o: Options) -> Result<Tracking, String> {
        let dir = weights_dir()?;
        let source = FrameSource::open(&o.camera, o.size)?;
        let (w, h) = source.size();
        // the lens is described at the size the driver actually gave
        let calibration = o.profile.calibration_for(w, h).map_err(|e| format!("calibration: {e}"))?;
        let estimator = Arc::new(Mutex::new(ViewerEstimator::new(EstimatorConfig::default())));
        let calibration = Arc::new(Mutex::new(calibration));
        let stats = Arc::new(Mutex::new(Stats { camera: format!("{} {w}x{h}", source.label()), ..Stats::default() }));
        let stop = Arc::new(AtomicBool::new(false));
        let slot = FrameSlot::new();

        let capture = {
            let (slot, stop, stats) = (slot.clone(), stop.clone(), stats.clone());
            std::thread::Builder::new()
                .name("fishtank-capture".into())
                .spawn(move || {
                    let mut source = source;
                    let mut seq = 0;
                    while !stop.load(Ordering::Relaxed) {
                        match source.next(seq) {
                            Ok((rgb, stamp)) => {
                                seq += 1;
                                slot.push(Frame { rgb, w, h, seq, stamp });
                            }
                            Err(e) => {
                                stats.lock().unwrap().last_problem = format!("camera: {e}");
                                break;
                            }
                        }
                    }
                })
                .map_err(|e| format!("starting the capture thread: {e}"))?
        };

        // the tracker loads on its own thread (it owns its GPU device), and
        // reports a load failure through the handshake
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let tracking = {
            let (slot, stop, stats, est, cal) = (slot.clone(), stop.clone(), stats.clone(), estimator.clone(), calibration.clone());
            let (lm, dev) = (o.landmarks, o.device.clone());
            std::thread::Builder::new()
                .name("fishtank-track".into())
                .spawn(move || {
                    let device = gpu_core::Gpu::open(Some(&dev), openseeface::model::PIPELINES);
                    let mut tracker = match Tracker::load(&device, &dir, lm, TrackerConfig::default()) {
                        Ok(t) => {
                            let _ = ready_tx.send(Ok(()));
                            t
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    let mut prior = None;
                    while !stop.load(Ordering::Relaxed) {
                        let Some(frame) = slot.take() else {
                            std::thread::sleep(Duration::from_millis(2));
                            continue;
                        };
                        let t0 = std::time::Instant::now();
                        let found = tracker.track(&frame.rgb, frame.w, frame.h);
                        let ms = t0.elapsed().as_secs_f64() * 1e3;
                        let mut s = stats.lock().unwrap();
                        s.frames += 1;
                        s.dropped = slot.stats().dropped;
                        s.inference_ms = 0.9 * s.inference_ms + 0.1 * ms;
                        let Some(found) = found else {
                            prior = None;
                            s.last_problem = "no face".into();
                            continue;
                        };
                        s.detections += found.detected as u64;
                        let pts: Vec<[f64; 2]> = found.landmarks.points.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
                        let conf: Vec<f64> = found.landmarks.confidence.iter().map(|&c| c as f64).collect();
                        let c = cal.lock().unwrap().clone();
                        let model = FaceModel::canonical(c.ipd_m);
                        let span = ((pts[45][0] - pts[36][0]).powi(2) + (pts[45][1] - pts[36][1]).powi(2)).sqrt();
                        match fit_head(&pts, &conf, &c.camera, &model, prior.as_ref(), MIN_POINT_CONFIDENCE, MAX_RMSE_OF_FACE * span) {
                            Ok(fit) => {
                                prior = Some(fit.head_to_camera);
                                let t = frame.stamp.ns as f64 * 1e-9;
                                let score = found.landmarks.score() as f64;
                                let _ = est.lock().unwrap().observe(t, c.to_screen(fit.right_eye), c.to_screen(fit.left_eye), score, fit.rmse_px);
                                s.fit_rmse_px = fit.rmse_px;
                                s.confidence = score;
                                s.last_problem.clear();
                            }
                            Err(e) => {
                                prior = None;
                                s.last_problem = e.to_string();
                            }
                        }
                    }
                })
                .map_err(|e| format!("starting the tracking thread: {e}"))?
        };
        let me = Tracking { estimator, calibration, stats, profile: Mutex::new(o.profile), save_to: o.save_to, stop, threads: vec![capture, tracking] };
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(me),
            Ok(Err(e)) => Err(format!("loading the face tracker: {e}")),
            Err(_) => Err("the tracking thread died while loading".into()),
        }
    }

    /// The eye to render a frame presented at `t` (monotonic seconds) for.
    pub fn viewer(&self, t: f64) -> TrackedViewer {
        self.estimator.lock().unwrap().predict(t)
    }

    /// "I am sitting centred in front of the screen": absorb the current
    /// horizontal offset into the camera mount, and save the profile.
    pub fn recentre(&self, now: f64) -> Result<String, String> {
        let v = self.viewer(now);
        if v.status != headpose::TrackingStatus::Tracking {
            return Err("no face is tracked right now".into());
        }
        let mut c = self.calibration.lock().unwrap();
        c.recentre(v.cyclopean());
        self.save(&c)
    }

    /// "My eyes are `true_m` from the glass": rescale the face model, and
    /// save the profile.
    pub fn set_distance(&self, now: f64, true_m: f64) -> Result<String, String> {
        let v = self.viewer(now);
        if v.status != headpose::TrackingStatus::Tracking {
            return Err("no face is tracked right now".into());
        }
        let mut c = self.calibration.lock().unwrap();
        c.set_distance(v.cyclopean()[2], true_m).map_err(|e| e.to_string())?;
        self.save(&c)
    }

    fn save(&self, c: &Calibration) -> Result<String, String> {
        let mut p = self.profile.lock().unwrap();
        *p = p.updated(c);
        let json = serde_json::to_string_pretty(&*p).map_err(|e| e.to_string())?;
        if let Some(dir) = self.save_to.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
        }
        std::fs::write(&self.save_to, json + "\n").map_err(|e| format!("saving {}: {e}", self.save_to.display()))?;
        Ok(format!("calibration saved to {}", self.save_to.display()))
    }
}

impl Drop for Tracking {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

enum FrameSource {
    Device(capture::v4l2::Device, String),
    Image { img: imaging::Rgb8, pan_px: u32, label: String, start: u64 },
}

impl FrameSource {
    fn open(c: &Camera, size: (u32, u32)) -> Result<FrameSource, String> {
        match c {
            Camera::Device(d) => capture::v4l2::Device::open(d, size.0, size.1, 4)
                .map(|dev| FrameSource::Device(dev, d.clone()))
                .map_err(|e| format!("opening {d}: {e}\n(the camera must offer YUYV and this process must be allowed to open it; check `v4l2-ctl -d {d} --list-formats-ext`)")),
            Camera::Image { path, pan_px } => {
                let img = imaging::codec::load(path).map_err(|e| format!("--camera-image {}: {e}", path.display()))?;
                if img.w % 2 == 1 {
                    return Err(format!("--camera-image {}: width {} must be even, like a camera's", path.display(), img.w));
                }
                Ok(FrameSource::Image { img, pan_px: *pan_px, label: format!("replay of {}", path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()), start: capture::clock::monotonic_ns() })
            }
        }
    }

    fn size(&self) -> (u32, u32) {
        match self {
            FrameSource::Device(d, _) => (d.width, d.height),
            FrameSource::Image { img, .. } => (img.w, img.h),
        }
    }

    fn label(&self) -> String {
        match self {
            FrameSource::Device(_, d) => d.clone(),
            FrameSource::Image { label, .. } => label.clone(),
        }
    }

    fn next(&mut self, seq: u64) -> Result<(Vec<u8>, capture::Stamp), String> {
        match self {
            FrameSource::Device(d, _) => d.next_frame(imaging::color::yuyv_to_rgb),
            FrameSource::Image { img, pan_px, start, .. } => {
                // pace like a 30 Hz camera, on the clock the estimator uses
                let due = *start + (seq + 1) * 33_333_333;
                let now = capture::clock::monotonic_ns();
                if due > now {
                    std::thread::sleep(Duration::from_nanos(due - now));
                }
                let t = (due - *start) as f64 * 1e-9;
                let shift = (*pan_px as f64 * (0.6 * t).sin()).round() as i64;
                let (w, h) = (img.w as i64, img.h as i64);
                let mut out = vec![0u8; img.px.len()];
                for y in 0..h {
                    for x in 0..w {
                        let sx = x - shift;
                        if (0..w).contains(&sx) {
                            let (d, s) = (((y * w + x) * 3) as usize, ((y * w + sx) * 3) as usize);
                            out[d..d + 3].copy_from_slice(&img.px[s..s + 3]);
                        }
                    }
                }
                Ok((out, capture::Stamp { ns: due, source: capture::StampSource::Dequeue }))
            }
        }
    }
}
