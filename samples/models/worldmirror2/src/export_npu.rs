// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `export-npu`: WorldMirror-2's stages as fp32 ONNX graphs with external
//! weights (`model.onnx` + `model.onnx.data`), for OpenVINO on the Intel NPU or
//! CPU.
//!
//! The model is exported in pieces because the whole of it does not fit one
//! NPU graph: `dino` is the per-frame encoder, `trunk` the fixed-frame-count
//! alternating-attention trunk with its four taps, `heads` one graph per dense
//! head (the Gaussian head carries the RGB-merge branch). `--levels` and
//! `--tap-levels` shorten the trunk and move its taps, for bisecting a device
//! parity problem one level at a time.
//!
//! Swedish Embedded AB implements model deployment to NPUs and other edge
//! accelerators for its clients. If your team needs expertise in exporting
//! transformer models to ONNX and OpenVINO then you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use worldmirror2::config::MirrorConfig;

/// A stage of the model that exports as its own graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Dino,
    Trunk,
    Heads,
}

/// Every flag of `export-npu`, parsed and checked.
#[derive(Debug, PartialEq)]
pub struct ExportArgs {
    pub stage: Stage,
    /// The graph file for `dino` and `trunk`; the directory the per-head
    /// graphs go in for `heads`.
    pub out: PathBuf,
    pub frames: usize,
    pub hp: usize,
    pub wp: usize,
    pub levels: usize,
    pub taps: Vec<usize>,
}

/// The value after `flag`, removed from `args` together with it.
fn take(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, String> {
    let Some(i) = args.iter().position(|a| a == flag) else { return Ok(None) };
    let value = args.get(i + 1).cloned().ok_or_else(|| format!("{flag} needs a value"))?;
    args.drain(i..=i + 1);
    Ok(Some(value))
}

fn count(args: &mut Vec<String>, flag: &str, default: usize) -> Result<usize, String> {
    match take(args, flag)? {
        None => Ok(default),
        Some(v) => v.parse::<usize>().ok().filter(|n| *n > 0).ok_or_else(|| format!("{flag} {v}: want a count of at least 1")),
    }
}

impl ExportArgs {
    pub fn parse(args: &[String], cfg: &MirrorConfig) -> Result<ExportArgs, String> {
        let mut rest = args.to_vec();
        let stage = match take(&mut rest, "--stage")?.as_deref().unwrap_or("dino") {
            "dino" => Stage::Dino,
            "trunk" => Stage::Trunk,
            "heads" => Stage::Heads,
            other => return Err(format!("unknown --stage {other} (dino|trunk|heads)")),
        };
        let out = take(&mut rest, "--out")?.map(PathBuf::from).unwrap_or_else(|| match stage {
            Stage::Dino => PathBuf::from("out/mirror-dino.onnx"),
            Stage::Trunk => PathBuf::from("out/mirror-trunk.onnx"),
            Stage::Heads => PathBuf::from("out"),
        });
        let frames = count(&mut rest, "--frames", 1)?;
        let hp = count(&mut rest, "--hp", 37)?;
        let wp = count(&mut rest, "--wp", 37)?;
        let levels = count(&mut rest, "--levels", cfg.depth)?;
        let taps = match take(&mut rest, "--tap-levels")? {
            None => cfg.tap_levels.to_vec(),
            Some(spec) => spec
                .split(',')
                .map(|v| v.trim().parse::<usize>().map_err(|_| format!("--tap-levels {spec}: want N,N,..")))
                .collect::<Result<_, _>>()?,
        };
        if let Some(stray) = rest.first() {
            return Err(format!("unexpected argument {stray}"));
        }
        Ok(ExportArgs { stage, out, frames, hp, wp, levels, taps })
    }
}

fn write(g: &onnx::builder::GraphBuilder, path: &Path) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    g.finish_external(&path.to_string_lossy(), 1 << 20).map_err(|e| format!("ONNX write failed for {}: {e}", path.display()))?;
    println!("wrote {} (+ external weight data)", path.display());
    Ok(())
}

pub fn run(call: &invoke::WorkflowCall) -> Result<(), String> {
    let cfg = MirrorConfig::default();
    let args = ExportArgs::parse(call.args, &cfg)?;
    let weights = call.assembly.role_path("weights")?;
    eprintln!("loading {weights} ...");
    let init = worldmirror2::import::load_weights(&weights, &cfg)?;
    match args.stage {
        Stage::Heads => {
            // One graph per DPT head; the Gaussian head carries the rgb-merge branch.
            for (name, out_ch, gs) in [("depth_head", 3i64, false), ("pts_head", 4, false), ("norm_head", 4, false), ("gs_head", 3, true)] {
                let mut g = onnx::builder::GraphBuilder::new(&format!("mirror_{name}"));
                npu::mirror_topology::build_dpt_head_graph(&init, &mut g, &cfg, name, out_ch, args.hp, args.wp, gs);
                write(&g, &args.out.join(format!("mirror-{name}.onnx")))?;
            }
        }
        Stage::Dino => {
            let mut g = onnx::builder::GraphBuilder::new("mirror_dino");
            npu::mirror_topology::build_dinov2_graph(&init, &mut g, cfg.depth);
            write(&g, &args.out)?;
            println!("verify: python3 tools/goldens/worldmirror2_check_onnx.py {}   (OpenVINO CPU/NPU)", args.out.display());
        }
        Stage::Trunk => {
            let mut g = onnx::builder::GraphBuilder::new("mirror_trunk");
            npu::mirror_topology::build_trunk_graph(&init, &mut g, args.frames, args.hp, args.wp, args.levels, &args.taps);
            write(&g, &args.out)?;
            println!("verify: python3 tools/goldens/worldmirror2_check_onnx.py {}   (OpenVINO CPU/NPU)", args.out.display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<ExportArgs, String> {
        ExportArgs::parse(&args.iter().map(|a| a.to_string()).collect::<Vec<_>>(), &MirrorConfig::default())
    }

    /// With no flags the encoder is exported at the native 37x37 patch grid.
    #[test]
    fn defaults_export_the_encoder_at_the_native_grid() {
        let a = parse(&[]).unwrap();
        let cfg = MirrorConfig::default();
        assert_eq!((a.stage, a.frames, a.hp, a.wp, a.levels), (Stage::Dino, 1, 37, 37, cfg.depth));
        assert_eq!(a.taps, cfg.tap_levels.to_vec());
        assert_eq!(a.out, PathBuf::from("out/mirror-dino.onnx"));
    }

    #[test]
    fn every_flag_is_taken() {
        let a = parse(&["--stage", "trunk", "--out", "g/trunk.onnx", "--frames", "4", "--hp", "20", "--wp", "30", "--levels", "6", "--tap-levels", "0, 2,4,5"]).unwrap();
        assert_eq!((a.stage, a.frames, a.hp, a.wp, a.levels), (Stage::Trunk, 4, 20, 30, 6));
        assert_eq!(a.taps, vec![0, 2, 4, 5]);
        assert_eq!(a.out, PathBuf::from("g/trunk.onnx"));
        assert_eq!(parse(&["--stage", "heads"]).unwrap().out, PathBuf::from("out"), "heads writes several graphs, so --out is a directory");
    }

    #[test]
    fn bad_values_are_named_not_defaulted() {
        assert!(parse(&["--stage", "vae"]).unwrap_err().contains("dino|trunk|heads"));
        assert!(parse(&["--frames", "0"]).unwrap_err().contains("--frames"));
        assert!(parse(&["--hp"]).unwrap_err().contains("needs a value"));
        assert!(parse(&["--tap-levels", "1,x"]).unwrap_err().contains("--tap-levels"));
        assert!(parse(&["--bogus", "1"]).unwrap_err().contains("--bogus"));
    }
}
