// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain flux2 …` — FLUX.2 Klein text-to-image + image editing.
//!
//! Weights via env (`BRAIN_FLUX2_{DIT,VAE,TE,TOKENIZER}`); images in/out are
//! binary PPM P6 (the CLI-wide convention).

use flux2::{Flux2Config, GenOpts, Paths, Pipeline};

const HELP: &str = "brain flux2 <cmd>
  generate --prompt <text> --out <out.ppm> [--width W] [--height H]
           [--steps N] [--seed S] [--guidance G] [--variant klein-4b|klein-9b|base-4b|base-9b]
           [--ref <in.ppm>]...      # reference images => editing mode
Weights (env): BRAIN_FLUX2_DIT, BRAIN_FLUX2_VAE, BRAIN_FLUX2_TE, BRAIN_FLUX2_TOKENIZER";

pub fn run_flux2(args: &[String]) {
    if args.is_empty() || args[0] == "--help" {
        eprintln!("{HELP}");
        return;
    }
    match args[0].as_str() {
        "generate" => {
            if let Err(e) = generate(&args[1..]) {
                eprintln!("flux2 generate: {e}");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("flux2: unknown subcommand {other}\n{HELP}");
            std::process::exit(2);
        }
    }
}

fn generate(args: &[String]) -> Result<(), String> {
    let mut prompt = None;
    let mut out = None;
    let mut o = GenOpts { width: 512, height: 512, ..GenOpts::default() };
    let mut variant_name = "klein-4b".to_string();
    let mut refs: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let need = |i: usize| -> Result<&String, String> {
            args.get(i + 1).ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--prompt" => prompt = Some(need(i)?.clone()),
            "--out" => out = Some(need(i)?.clone()),
            "--width" => o.width = need(i)?.parse().map_err(|e| format!("--width: {e}"))?,
            "--height" => o.height = need(i)?.parse().map_err(|e| format!("--height: {e}"))?,
            "--steps" => o.steps = Some(need(i)?.parse().map_err(|e| format!("--steps: {e}"))?),
            "--seed" => o.seed = need(i)?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--guidance" => o.guidance = need(i)?.parse().map_err(|e| format!("--guidance: {e}"))?,
            "--variant" => variant_name = need(i)?.clone(),
            "--ref" => refs.push(need(i)?.clone()),
            other => return Err(format!("unknown flag {other}\n{HELP}")),
        }
        i += 2;
    }
    let prompt = prompt.ok_or("--prompt is required")?;
    let out = out.ok_or("--out is required")?;
    let variant = Flux2Config::from_name(&variant_name)?;
    flux2::caps::check_license(&variant_name)?; // 9B = FLUX Non-Commercial license

    // load refs as [-1,1] CHW, center-cropped to /16 (shared helper — the
    // capability provider uses the same one)
    let mut ref_imgs: Vec<(Vec<f32>, u32, u32)> = Vec::new();
    for r in &refs {
        let (hwc, w, h) = crate::image_io::load_image(r)?;
        ref_imgs.push(flux2::pipeline::ref_from_hwc(&hwc, w, h)?);
    }

    let paths = Paths::from_env()?;
    let n_gen = (o.height / 16) * (o.width / 16);
    let n_ref: u32 = ref_imgs.iter().map(|(_, h, w)| (h / 16) * (w / 16)).sum();
    eprintln!("flux2: building pipeline ({} + {} tokens) ...", n_gen, n_ref);
    let pipe = Pipeline::build(&variant, &paths, n_gen + n_ref)?;
    let t0 = std::time::Instant::now();
    // The CLI has no cancel front-end — an unarmed Default token never fires.
    let (rgb, w, h) = pipe.generate(&prompt, &ref_imgs, &o, &Default::default(), |step, total, msg| {
        eprint!("\rflux2 [{step}/{total}] {msg}          ");
    })?;
    eprintln!("\nflux2: {:.1}s", t0.elapsed().as_secs_f32());
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    ppm.extend_from_slice(&rgb);
    std::fs::write(&out, ppm).map_err(|e| format!("writing {out}: {e}"))?;
    eprintln!("flux2: wrote {out} ({w}x{h})");
    Ok(())
}
