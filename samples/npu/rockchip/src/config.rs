// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Settings of the live stream, with defaults suited to the Luckfox Pico Plus
//! (RV1103, SC3336 sensor, a 33 MB Linux view of RAM).

/// The sensor's largest output, which bounds the stream size.
pub const SENSOR_MAX: (u32, u32) = (2304, 1296);

#[derive(Clone, Debug, PartialEq)]
pub struct StreamConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate_kbps: u32,
    /// Frames between IDR frames.
    pub gop: u32,
    /// Macroblock rows of intra refresh per frame; 0 disables it.
    pub intra_refresh_rows: u32,
    pub rtsp_port: u16,
    pub rtsp_path: String,
    /// Directory of the sensor's ISP tuning files.
    pub iq_dir: String,
    /// Frames between latency reports.
    pub report_every: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        StreamConfig {
            width: 800,
            height: 448,
            fps: 30,
            bitrate_kbps: 2048,
            gop: 250,
            intra_refresh_rows: 2,
            rtsp_port: 8554,
            rtsp_path: "/live/0".to_string(),
            iq_dir: "/etc/iqfiles".to_string(),
            report_every: 100,
        }
    }
}

impl StreamConfig {
    /// Rejects settings the camera or encoder cannot honour.
    pub fn validate(&self) -> Result<(), String> {
        let (max_w, max_h) = SENSOR_MAX;
        if !self.width.is_multiple_of(2) || !self.height.is_multiple_of(2) || self.width < 64 || self.height < 64 {
            return Err(format!("{}x{} is not a usable stream size (even, at least 64x64)", self.width, self.height));
        }
        if self.width > max_w || self.height > max_h {
            return Err(format!("{}x{} exceeds the sensor's {max_w}x{max_h}", self.width, self.height));
        }
        if !(1..=60).contains(&self.fps) {
            return Err(format!("{} fps is outside 1..=60", self.fps));
        }
        if !(64..=100_000).contains(&self.bitrate_kbps) {
            return Err(format!("{} kbit/s is outside 64..=100000", self.bitrate_kbps));
        }
        if self.gop == 0 {
            return Err("the GOP length must be at least 1".to_string());
        }
        if !self.rtsp_path.starts_with('/') {
            return Err(format!("RTSP path {:?} must start with '/'", self.rtsp_path));
        }
        if self.report_every == 0 {
            return Err("--report-every must be at least 1".to_string());
        }
        Ok(())
    }
}
