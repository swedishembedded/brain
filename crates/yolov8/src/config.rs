// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! YOLOv8-style detector configuration.
//!
//! A YOLOv8 detector is an anchor-free, single-stage object detector: a CSP
//! backbone produces a feature pyramid at three strides (8/16/32), a PAN-style
//! neck fuses them, and a decoupled head emits, per pyramid cell, `4*reg_max`
//! DFL box-distribution logits + `nc` class scores. Unlike the token models,
//! its output sizing comes from the *image* geometry (`input` resolution and
//! `strides`) and the dataset's class count `nc`, not from a token vocabulary.
//!
//! Following the [`model::ModelConfig`] convention used by the autoencoder
//! (another non-token model), `vocab()` carries the class count and
//! `block_size()` carries the input resolution so the generic trainer can
//! reason uniformly about the model. Detection-specific sizing comes from the
//! dataset meta, so `finalize_for_dataset` leaves the config unchanged.
//!
//! ## Canonical layout (P12)
//!
//! [`YoloConfig::yolov8n`] is the BYTE-COMPATIBLE canonical Ultralytics
//! `yolov8n` graph (`width_mult 0.25`, `depth_mult 0.33`, `max_channels 1024`):
//! per-stage backbone channels `[16,32,32,64,64,128,128,256,256,256]` with C2f
//! bottleneck depths `[1,2,2,1]` (stages 2/4/6/8), a PAN-FPN neck, and a biased
//! decoupled DFL head (`reg` hidden 64, `cls` hidden 80). The struct carries the
//! layout EXPLICITLY (`backbone_ch`/`backbone_depth`/`neck_ch`/`cls_mid`/
//! `reg_mid`) rather than deriving it from the multipliers, so it can express
//! the per-stage depth `[1,2,2,1]` + 256-wide deep channels + distinct neck/head
//! widths that the canonical graph needs (and that a single `depth_mult`/
//! `channels` pair cannot). [`YoloConfig::tiny`] is a small fast variant for the
//! gradient checks; it shares the exact same biased-head graph shape.

use serde_json::Value;

/// YOLOv8-style detector configuration.
///
/// - `input`         — square input resolution in pixels (e.g. 640).
/// - `nc`            — number of object classes.
/// - `reg_max`       — DFL distribution bins per box side (box coord = expectation
///   over `reg_max` logits per side).
/// - `depth_mult`    — CSP block-depth multiplier (informational; the explicit
///   `backbone_depth` is authoritative).
/// - `width_mult`    — channel-width multiplier (informational).
/// - `channels`      — legacy per-stage stem widths (informational / JSON).
/// - `strides`       — the three pyramid strides (P3/P4/P5), e.g. `[8, 16, 32]`.
/// - `backbone_ch`   — output channels of backbone stages `0..=9`.
/// - `backbone_depth`— C2f bottleneck count for stages `2,4,6,8` (in that order).
/// - `neck_ch`       — output channels of neck stages `neck.0..=neck.5`.
/// - `neck_depth`    — C2f bottleneck count in every neck C2f (shortcut=false).
/// - `cls_mid`       — head cls-branch hidden width.
/// - `reg_mid`       — head reg-branch hidden width.
#[derive(Clone, Debug)]
pub struct YoloConfig {
    pub input: u32,
    pub nc: u32,
    pub reg_max: u32,
    pub depth_mult: f32,
    pub width_mult: f32,
    pub channels: Vec<u32>,
    pub strides: [u32; 3],
    pub backbone_ch: [u32; 10],
    pub backbone_depth: [u32; 4],
    pub neck_ch: [u32; 6],
    pub neck_depth: u32,
    pub cls_mid: u32,
    pub reg_mid: u32,
}

impl YoloConfig {
    /// The canonical `yolov8n` (nano) variant: 640px input, 80 COCO classes.
    ///
    /// BYTE-COMPATIBLE with the official Ultralytics `yolov8n.pt` graph
    /// (`width_mult 0.25`, `depth_mult 0.33`, `max_channels 1024`):
    ///
    /// - backbone out-channels `[16,32,32,64,64,128,128,256,256,256]`
    ///   (stages 0..=9; P3=stage4=64, P4=stage6=128, P5=stage9=256),
    /// - C2f bottleneck depths `[1,2,2,1]` for stages `2,4,6,8`,
    /// - neck out-channels `[128,64,64,128,128,256]` (neck.0..=neck.5;
    ///   neck.1→N3=64, neck.3→N4=128, neck.5→N5=256), all C2f depth 1,
    /// - biased decoupled head, reg hidden 64, cls hidden 80.
    pub fn yolov8n() -> YoloConfig {
        YoloConfig {
            input: 640,
            nc: 80,
            reg_max: 16,
            depth_mult: 0.33,
            width_mult: 0.25,
            channels: vec![16, 32, 64, 128, 256],
            strides: [8, 16, 32],
            backbone_ch: [16, 32, 32, 64, 64, 128, 128, 256, 256, 256],
            backbone_depth: [1, 2, 2, 1],
            neck_ch: [128, 64, 64, 128, 128, 256],
            neck_depth: 1,
            cls_mid: 80,
            reg_mid: 64,
        }
    }

    /// A tiny config for tests / gradient checks: small input, one bottleneck
    /// per C2f, narrow channels, and a caller-supplied class count. Shares the
    /// exact biased-head graph shape with `yolov8n` (just smaller widths).
    pub fn tiny(nc: u32) -> YoloConfig {
        // Stem widths [c0,c1,c2,c3] = [8,16,32,64]; P3=32, P4=64, P5=64.
        let (c0, c1, c2, c3) = (8u32, 16, 32, 64);
        let reg_max = 8u32;
        YoloConfig {
            input: 128,
            nc,
            reg_max,
            depth_mult: 1.0,
            width_mult: 0.25,
            channels: vec![c0, c1, c2, c3],
            strides: [8, 16, 32],
            // 10 backbone stages; deep channels capped at c3 (the tiny variant
            // keeps the original reduced widths so the FD gradcheck stays fast).
            backbone_ch: [c0, c1, c1, c2, c2, c3, c3, c3, c3, c3],
            backbone_depth: [1, 1, 1, 1],
            // neck.0→T4=c3, neck.1→N3=c2, neck.2(down)=c2, neck.3→N4=c3,
            // neck.4(down)=c3, neck.5→N5=c3.
            neck_ch: [c3, c2, c2, c3, c3, c3],
            neck_depth: 1,
            // Head hidden widths (small for tiny): cls=max(c2,nc), reg=max(c2,4*rm).
            cls_mid: c2.max(nc),
            reg_mid: c2.max(4 * reg_max),
        }
    }

    /// The REAL full parameter layout (backbone + PAN-FPN neck + decoupled
    /// head), composed in graph order. Param counts depend only on channel
    /// widths + kernel sizes + bottleneck depths, so this reproduces the exact
    /// prefixes/shapes that [`crate::model::Yolo::new`] registers — without
    /// needing a GPU. Both `yolov8n` and `tiny` compose through the SAME builder.
    pub fn full_param_list(&self) -> Vec<(String, usize)> {
        let mut out: Vec<(String, usize)> = Vec::new();
        let bch = &self.backbone_ch;
        let bd = &self.backbone_depth;

        // --- conv-param helpers (param counts are H/W-independent) ---
        let conv = |out: &mut Vec<(String, usize)>, p: &str, cin: u32, cout: u32, k: u32| {
            let w = (cout * cin * k * k) as usize;
            out.push((format!("{p}.conv.weight"), w));
            out.push((format!("{p}.bn.gamma"), cout as usize));
            out.push((format!("{p}.bn.beta"), cout as usize));
            out.push((format!("{p}.bn.run_mean"), cout as usize));
            out.push((format!("{p}.bn.run_var"), cout as usize));
        };
        let bottleneck = |out: &mut Vec<(String, usize)>, p: &str, cin: u32, cout: u32| {
            conv(out, &format!("{p}.cv1"), cin, cout, 3);
            conv(out, &format!("{p}.cv2"), cout, cout, 3);
        };
        let c2f = |out: &mut Vec<(String, usize)>, p: &str, cin: u32, cout: u32, n: u32| {
            let c = cout / 2;
            conv(out, &format!("{p}.cv1"), cin, 2 * c, 1);
            let mut prev = c;
            for i in 0..n {
                bottleneck(out, &format!("{p}.m.{i}"), prev, c);
                prev = c;
            }
            conv(out, &format!("{p}.cv2"), (2 + n) * c, cout, 1);
        };
        let sppf = |out: &mut Vec<(String, usize)>, p: &str, cin: u32, cout: u32| {
            let c = cout / 2;
            conv(out, &format!("{p}.cv1"), cin, c, 1);
            conv(out, &format!("{p}.cv2"), 4 * c, cout, 1);
        };
        // A head branch: two K3 Convs (cin->mid->mid) then a BIASED 1x1 (mid->out_c).
        let branch = |out: &mut Vec<(String, usize)>, p: &str, cin: u32, mid: u32, out_c: u32| {
            conv(out, &format!("{p}.0"), cin, mid, 3);
            conv(out, &format!("{p}.1"), mid, mid, 3);
            out.push((format!("{p}.2.weight"), (out_c * mid) as usize));
            out.push((format!("{p}.2.bias"), out_c as usize));
        };

        // --- backbone (stages 0..=9) ---
        conv(&mut out, "backbone.0", 3, bch[0], 3);
        conv(&mut out, "backbone.1", bch[0], bch[1], 3);
        c2f(&mut out, "backbone.2", bch[1], bch[2], bd[0]);
        conv(&mut out, "backbone.3", bch[2], bch[3], 3);
        c2f(&mut out, "backbone.4", bch[3], bch[4], bd[1]); // P3
        conv(&mut out, "backbone.5", bch[4], bch[5], 3);
        c2f(&mut out, "backbone.6", bch[5], bch[6], bd[2]); // P4
        conv(&mut out, "backbone.7", bch[6], bch[7], 3);
        c2f(&mut out, "backbone.8", bch[7], bch[8], bd[3]);
        sppf(&mut out, "backbone.9", bch[8], bch[9]); // P5

        // P3/P4/P5 feature widths (backbone stages 4 / 6 / 9).
        let p3 = bch[4];
        let p4 = bch[6];
        let p5 = bch[9];

        // --- neck (PAN-FPN) ---
        let nc_ = &self.neck_ch;
        let nd = self.neck_depth;
        c2f(&mut out, "neck.0", p5 + p4, nc_[0], nd); // [up(P5) | P4] -> T4
        c2f(&mut out, "neck.1", nc_[0] + p3, nc_[1], nd); // [up(T4) | P3] -> N3
        conv(&mut out, "neck.2", nc_[1], nc_[2], 3); // down N3
        c2f(&mut out, "neck.3", nc_[2] + nc_[0], nc_[3], nd); // [dn3 | T4] -> N4
        conv(&mut out, "neck.4", nc_[3], nc_[4], 3); // down N4
        c2f(&mut out, "neck.5", nc_[4] + p5, nc_[5], nd); // [dn4 | P5] -> N5

        // --- head (3 scales on N3=neck.1, N4=neck.3, N5=neck.5) ---
        let scale_in = [nc_[1], nc_[3], nc_[5]];
        for (s, &cin) in scale_in.iter().enumerate() {
            branch(&mut out, &format!("head.{s}.cls"), cin, self.cls_mid, self.nc);
            branch(&mut out, &format!("head.{s}.reg"), cin, self.reg_mid, 4 * self.reg_max);
        }
        out
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "model": "yolo",
            "input": self.input,
            "nc": self.nc,
            "reg_max": self.reg_max,
            "depth_mult": self.depth_mult,
            "width_mult": self.width_mult,
            "channels": self.channels,
            "strides": self.strides,
            "backbone_ch": self.backbone_ch,
            "backbone_depth": self.backbone_depth,
            "neck_ch": self.neck_ch,
            "neck_depth": self.neck_depth,
            "cls_mid": self.cls_mid,
            "reg_mid": self.reg_mid,
        })
    }

    pub fn from_json(c: &Value) -> YoloConfig {
        let u = |k: &str, d: u32| c[k].as_u64().map(|v| v as u32).unwrap_or(d);
        let f = |k: &str, d: f32| c[k].as_f64().map(|v| v as f32).unwrap_or(d);
        let channels = c["channels"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect())
            .unwrap_or_default();
        let strides = c["strides"]
            .as_array()
            .map(|a| {
                let mut s = [8u32, 16, 32];
                for (i, v) in a.iter().take(3).enumerate() {
                    if let Some(x) = v.as_u64() {
                        s[i] = x as u32;
                    }
                }
                s
            })
            .unwrap_or([8, 16, 32]);
        // Default explicit layout = the canonical yolov8n one (so a config JSON
        // written before P12 still loads as canonical yolov8n).
        let def = YoloConfig::yolov8n();
        let arr10 = |k: &str, d: [u32; 10]| -> [u32; 10] {
            c[k].as_array()
                .map(|a| {
                    let mut o = d;
                    for (i, v) in a.iter().take(10).enumerate() {
                        if let Some(x) = v.as_u64() {
                            o[i] = x as u32;
                        }
                    }
                    o
                })
                .unwrap_or(d)
        };
        let arr4 = |k: &str, d: [u32; 4]| -> [u32; 4] {
            c[k].as_array()
                .map(|a| {
                    let mut o = d;
                    for (i, v) in a.iter().take(4).enumerate() {
                        if let Some(x) = v.as_u64() {
                            o[i] = x as u32;
                        }
                    }
                    o
                })
                .unwrap_or(d)
        };
        let arr6 = |k: &str, d: [u32; 6]| -> [u32; 6] {
            c[k].as_array()
                .map(|a| {
                    let mut o = d;
                    for (i, v) in a.iter().take(6).enumerate() {
                        if let Some(x) = v.as_u64() {
                            o[i] = x as u32;
                        }
                    }
                    o
                })
                .unwrap_or(d)
        };
        YoloConfig {
            input: u("input", 640),
            nc: u("nc", 80),
            reg_max: u("reg_max", 16),
            depth_mult: f("depth_mult", 0.33),
            width_mult: f("width_mult", 0.25),
            channels,
            strides,
            backbone_ch: arr10("backbone_ch", def.backbone_ch),
            backbone_depth: arr4("backbone_depth", def.backbone_depth),
            neck_ch: arr6("neck_ch", def.neck_ch),
            neck_depth: u("neck_depth", def.neck_depth),
            cls_mid: u("cls_mid", def.cls_mid),
            reg_mid: u("reg_mid", def.reg_mid),
        }
    }
}

// ---- the architecture-agnostic Model seam (ADR 0001 §2.2/§2.3) ----

impl model::ModelConfig for YoloConfig {
    fn param_list(&self) -> Vec<(String, usize)> {
        self.full_param_list()
    }
    fn to_json(&self) -> Value {
        YoloConfig::to_json(self)
    }
    fn from_json(v: &Value) -> Self {
        YoloConfig::from_json(v)
    }
    /// No token head: `vocab` carries the object-class count.
    fn vocab(&self) -> u32 {
        self.nc
    }
    /// `block_size` carries the square input resolution for the generic trainer.
    fn block_size(&self) -> u32 {
        self.input
    }
    /// Detection sizing comes from the dataset meta (image geometry + class
    /// count baked into this config), not from a token vocab/block_size, so the
    /// config is returned unchanged.
    fn finalize_for_dataset(self, _vocab: u32, _block_size: u32) -> Self {
        self
    }
}
