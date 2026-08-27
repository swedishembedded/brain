// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 latent laboratory: encode an image to the VAE latent, operate on it,
//! decode it back, and measure what moved.
//!
//! The DiT is not involved and is never loaded - every run here is
//! encode → operate → decode against the FLUX.2 autoencoder alone, which is
//! what makes latent-space questions answerable in seconds instead of minutes.
//!
//! The latent that travels between subcommands is the VAE posterior **mean**,
//! `[32, H/8, W/8]`, in a self-describing file (see
//! [`flux2::latentops::Latent`]). A decoded image can be put in the *same*
//! container (`img2lat`), so a pixel-space edit and a latent-space edit run
//! through literally the same operator and cannot differ by an accident of
//! implementation - which is the only way the comparison between them means
//! anything.
//!
//! ```text
//!   flux2_latent encode  --outdir D [--device gpu1] IMG...     image -> .lat
//!   flux2_latent decode  --outdir D [--device gpu1] LAT...     .lat  -> png
//!   flux2_latent img2lat --out F.lat IMG                       image -> .lat (RGB)
//!   flux2_latent lat2img --out F.png LAT                       .lat (RGB) -> image
//!   flux2_latent stats   LAT                                   per-channel mean/std
//!   flux2_latent metrics A.png B.png                           A against reference B
//!   flux2_latent thumb   --outdir D --width N IMG...           contact-sheet tiles
//!   flux2_latent op OP [flags] --out F.lat                     see `op` below
//! ```
//!
//! `op` is host math, needs no weights and no device:
//!
//! ```text
//!   blend  --a A --b B --alpha F
//!   splice --a A --b B --rect X,Y,W,H [--feather PX]   rect in IMAGE pixels
//!   flip   --in A --axis h|v
//!   rot90  --in A --k N                                N quarter-turns clockwise
//!   rotate --in A --deg F                              bilinear, edge-clamped
//!   roll   --in A --dy N --dx N                        circular, in latent cells
//!   chan   --in A --c N --op zero|mean|scale:F|cscale:F
//!   noise  --in A --sigma F [--units std|global|raw] [--c N] [--seed N]
//!                        [--smooth K]   give the noise a K-cell correlation
//!                                       length at the same amplitude
//! ```
//!
//! The VAE checkpoint comes from `BRAIN_FLUX2_VAE` (a diffusers `vae/`
//! directory or a single safetensors file), the same variable the generation
//! pipeline reads. Nothing here bakes in a path.
//!
//! Swedish Embedded AB implements diffusion latent-space tooling and analysis
//! for its clients. If your team needs expertise in generative imaging pipelines
//! then you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::HashMap;

use flux2::latentops::{
    self, add_noise, blend, channel_op, compare, splice, ChanOp, Kind, Latent, NoiseUnits, Rect,
};

type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

fn main() {
    if let Err(e) = run() {
        eprintln!("flux2_latent: {e}");
        std::process::exit(1);
    }
}

/// Flags parsed off the tail of the command line: `--key value` pairs plus the
/// bare positional arguments, in order.
struct Args {
    flags: HashMap<String, String>,
    pos: Vec<String>,
}

impl Args {
    fn parse(argv: &[String]) -> Args {
        let (mut flags, mut pos) = (HashMap::new(), Vec::new());
        let mut i = 0;
        while i < argv.len() {
            match argv[i].strip_prefix("--") {
                Some(k) if i + 1 < argv.len() => {
                    flags.insert(k.to_string(), argv[i + 1].clone());
                    i += 2;
                }
                Some(k) => {
                    flags.insert(k.to_string(), String::new());
                    i += 1;
                }
                None => {
                    pos.push(argv[i].clone());
                    i += 1;
                }
            }
        }
        Args { flags, pos }
    }

    fn get(&self, k: &str) -> Result<&str, String> {
        self.flags.get(k).map(String::as_str).ok_or_else(|| format!("--{k} is required"))
    }

    fn opt(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(String::as_str)
    }

    fn num<T: std::str::FromStr>(&self, k: &str) -> Result<T, String> {
        self.get(k)?.parse::<T>().map_err(|_| format!("--{k}: not a number"))
    }

    fn num_or<T: std::str::FromStr>(&self, k: &str, d: T) -> Result<T, String> {
        match self.opt(k) {
            Some(v) => v.parse::<T>().map_err(|_| format!("--{k}: not a number")),
            None => Ok(d),
        }
    }
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().cloned().unwrap_or_default();
    let a = Args::parse(&argv[argv.len().min(1)..]);
    match cmd.as_str() {
        "encode" => encode(&a),
        "decode" => decode(&a),
        "img2lat" => img2lat(&a),
        "lat2img" => lat2img(&a),
        "stats" => stats(&a),
        "metrics" => metrics(&a),
        "thumb" => thumb(&a),
        "op" => op(&a),
        _ => Err("usage: flux2_latent {encode|decode|img2lat|lat2img|stats|metrics|thumb|op} ... \
                  (see the module docs)"
            .into()),
    }
}

// ===================== image <-> host buffers =====================

/// Interleaved RGB u8 → CHW f32 in `[-1, 1]`, the VAE's input convention.
fn to_chw(px: &[u8], h: usize, w: usize) -> Vec<f32> {
    let n = h * w;
    let mut out = vec![0.0f32; 3 * n];
    for c in 0..3 {
        for i in 0..n {
            out[c * n + i] = px[i * 3 + c] as f32 / 127.5 - 1.0;
        }
    }
    out
}

/// CHW f32 in `[-1, 1]` → interleaved RGB u8. Clamp first, then rescale - the
/// reference order; reversed it produces artifacts. Truncating, which is what
/// [`flux2::Pipeline::decode_tokens`] does, so a lab decode and a generated
/// image are the same bytes.
fn to_rgb8(chw: &[f32], h: usize, w: usize) -> Vec<u8> {
    let n = h * w;
    let mut out = vec![0u8; 3 * n];
    for c in 0..3 {
        for i in 0..n {
            out[i * 3 + c] = (127.5 * (chw[c * n + i].clamp(-1.0, 1.0) + 1.0)) as u8;
        }
    }
    out
}

/// Interleaved RGB u8 → CHW f32 **levels** (0..255) for a [`Kind::Rgb`]
/// container, and back. Exact in both directions - see [`Kind::Rgb`].
fn to_levels(px: &[u8], h: usize, w: usize) -> Vec<f32> {
    let n = h * w;
    let mut out = vec![0.0f32; 3 * n];
    for c in 0..3 {
        for i in 0..n {
            out[c * n + i] = px[i * 3 + c] as f32;
        }
    }
    out
}

fn from_levels(chw: &[f32], h: usize, w: usize) -> Vec<u8> {
    let n = h * w;
    let mut out = vec![0u8; 3 * n];
    for c in 0..3 {
        for i in 0..n {
            out[i * 3 + c] = chw[c * n + i].round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn load_image(path: &str) -> Result<(Vec<u8>, usize, usize), String> {
    let img = imaging::load(path)?;
    Ok((img.px, img.h as usize, img.w as usize))
}

fn save_image(path: &str, px: Vec<u8>, h: usize, w: usize) -> Result<(), String> {
    imaging::save_png(path, &imaging::Rgb8::new(w as u32, h as u32, px)?)
}

/// `--out` for a single input, or `<--outdir>/<stem><ext>` for many.
fn out_path(a: &Args, input: &str, ext: &str, many: bool) -> Result<String, String> {
    match (a.opt("out"), a.opt("outdir")) {
        (Some(p), _) if !many => Ok(p.to_string()),
        (_, Some(d)) => {
            let stem = std::path::Path::new(input)
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| format!("{input}: no file stem"))?;
            Ok(format!("{d}/{stem}{ext}"))
        }
        _ => Err("--out (one input) or --outdir (any number) is required".into()),
    }
}

// ===================== VAE =====================

/// Read the VAE checkpoint named by `BRAIN_FLUX2_VAE` into name → (shape, data).
fn load_vae() -> Result<(vae::VaeConfig, Tensors), String> {
    let path = std::env::var("BRAIN_FLUX2_VAE").map_err(|_| "BRAIN_FLUX2_VAE not set")?;
    let p = std::path::Path::new(&path);
    let (file, json) = if p.is_dir() {
        (p.join("diffusion_pytorch_model.safetensors"), std::fs::read_to_string(p.join("config.json")).ok())
    } else {
        (p.to_path_buf(), None)
    };
    let cfg = match json {
        Some(j) => vae::VaeConfig::from_json(&serde_json::from_str(&j).map_err(|e| e.to_string())?),
        None => vae::VaeConfig::flux2(),
    };
    let mut map = Tensors::new();
    for t in checkpoint::safetensors::read(file.to_str().ok_or("vae path is not UTF-8")?)? {
        map.insert(t.name, (t.shape, t.data));
    }
    Ok((cfg, map))
}

fn encode(a: &Args) -> Result<(), String> {
    let (cfg, ts) = load_vae()?;
    let many = a.pos.len() > 1;
    // One graph per distinct image size; the inputs are almost always one size,
    // so this builds the encoder once for the whole batch.
    let mut graph: Option<(usize, usize, vae::VaeEncoder)> = None;
    for input in &a.pos {
        let (px, h, w) = load_image(input)?;
        if h % 8 != 0 || w % 8 != 0 {
            return Err(format!("{input}: {w}x{h} is not a multiple of 8"));
        }
        if graph.as_ref().is_none_or(|(gh, gw, _)| (*gh, *gw) != (h, w)) {
            let enc = vae::VaeEncoder::from_diffusers(cfg.clone(), &ts, h as u32, w as u32, a.opt("device"));
            graph = Some((h, w, enc));
        }
        let enc = &graph.as_ref().unwrap().2;
        let (lh, lw) = (h / 8, w / 8);
        let mean = enc.encode_mean(&to_chw(&px, h, w), lh as u32, lw as u32);
        let lat = Latent::new(Kind::VaeMean, cfg.latent_channels as usize, lh, lw, mean)?;
        let out = out_path(a, input, ".lat", many)?;
        lat.save(&out)?;
        println!("encode {input} {w}x{h} -> {out} [{},{lh},{lw}]", lat.c);
    }
    Ok(())
}

fn decode(a: &Args) -> Result<(), String> {
    let (cfg, ts) = load_vae()?;
    let many = a.pos.len() > 1;
    let mut graph: Option<(usize, usize, vae::VaeDecoder)> = None;
    for input in &a.pos {
        let lat = Latent::load(input)?;
        if lat.kind != Kind::VaeMean {
            return Err(format!("{input}: decode wants a VaeMean latent, got {:?}", lat.kind));
        }
        if graph.as_ref().is_none_or(|(gh, gw, _)| (*gh, *gw) != (lat.h, lat.w)) {
            let dec =
                vae::VaeDecoder::from_diffusers(cfg.clone(), &ts, lat.h as u32, lat.w as u32, a.opt("device"));
            graph = Some((lat.h, lat.w, dec));
        }
        let chw = graph.as_ref().unwrap().2.decode(&lat.data);
        let (h, w) = (lat.h * 8, lat.w * 8);
        let out = out_path(a, input, ".png", many)?;
        save_image(&out, to_rgb8(&chw, h, w), h, w)?;
        println!("decode {input} [{},{},{}] -> {out} {w}x{h}", lat.c, lat.h, lat.w);
    }
    Ok(())
}

fn img2lat(a: &Args) -> Result<(), String> {
    let many = a.pos.len() > 1;
    for input in &a.pos {
        let (px, h, w) = load_image(input)?;
        let lat = Latent::new(Kind::Rgb, 3, h, w, to_levels(&px, h, w))?;
        let out = out_path(a, input, ".lat", many)?;
        lat.save(&out)?;
        println!("img2lat {input} -> {out} {w}x{h}");
    }
    Ok(())
}

fn lat2img(a: &Args) -> Result<(), String> {
    let many = a.pos.len() > 1;
    for input in &a.pos {
        let lat = Latent::load(input)?;
        if lat.kind != Kind::Rgb {
            return Err(format!("{input}: lat2img wants an Rgb latent, got {:?}", lat.kind));
        }
        let out = out_path(a, input, ".png", many)?;
        save_image(&out, from_levels(&lat.data, lat.h, lat.w), lat.h, lat.w)?;
        println!("lat2img {input} -> {out} {}x{}", lat.w, lat.h);
    }
    Ok(())
}

fn stats(a: &Args) -> Result<(), String> {
    for input in &a.pos {
        let lat = Latent::load(input)?;
        println!("{input}: {:?} [{},{},{}]", lat.kind, lat.c, lat.h, lat.w);
        println!("  ch      mean       std       min       max");
        let plane = lat.h * lat.w;
        for (ci, (m, s)) in lat.channel_stats().iter().enumerate() {
            let p = &lat.data[ci * plane..(ci + 1) * plane];
            let lo = p.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            println!("  {ci:>3} {m:>9.4} {s:>9.4} {lo:>9.4} {hi:>9.4}");
        }
    }
    Ok(())
}

fn metrics(a: &Args) -> Result<(), String> {
    if a.pos.len() != 2 {
        return Err("metrics wants exactly two images: <a.png> <reference.png>".into());
    }
    let (x, h, w) = load_image(&a.pos[0])?;
    let (y, rh, rw) = load_image(&a.pos[1])?;
    if (h, w) != (rh, rw) {
        return Err(format!("size mismatch: {w}x{h} vs {rw}x{rh}"));
    }
    let m = compare(&x, &y, h, w)?;
    println!(
        "{}\t{}\tmad={:.4}\trel_l2={:.6}\tcosine={:.8}\tedge_corr={:.6}",
        a.pos[0], a.pos[1], m.mad, m.rel_l2, m.cosine, m.edge_corr
    );
    Ok(())
}

fn thumb(a: &Args) -> Result<(), String> {
    let tw: usize = a.num("width")?;
    for input in &a.pos {
        let (px, h, w) = load_image(input)?;
        let th = (h * tw).div_ceil(w).max(1);
        let f: Vec<f32> = px.iter().map(|&v| v as f32).collect();
        let small = imaging::resize_bilinear_hwc(&f, 3, w as u32, h as u32, tw as u32, th as u32);
        let out = out_path(a, input, ".png", a.pos.len() > 1)?;
        save_image(&out, small.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect(), th, tw)?;
    }
    Ok(())
}

// ===================== host-side operations =====================

fn op(a: &Args) -> Result<(), String> {
    let name = a.pos.first().ok_or("op: which operation?")?.clone();
    let load_in = || Latent::load(a.get("in")?);
    let load_ab = || Ok::<_, String>((Latent::load(a.get("a")?)?, Latent::load(a.get("b")?)?));
    let out = match name.as_str() {
        "blend" => {
            let (x, y) = load_ab()?;
            blend(&x, &y, a.num("alpha")?)?
        }
        "splice" => {
            let (x, y) = load_ab()?;
            let r: Vec<i64> = a
                .get("rect")?
                .split(',')
                .map(|v| v.trim().parse::<i64>().map_err(|_| "--rect wants X,Y,W,H".to_string()))
                .collect::<Result<_, _>>()?;
            if r.len() != 4 {
                return Err("--rect wants X,Y,W,H".into());
            }
            splice(&x, &y, Rect { x: r[0], y: r[1], w: r[2], h: r[3] }, a.num_or("feather", 0.0)?)?
        }
        "flip" => {
            let x = load_in()?;
            match a.get("axis")? {
                "h" => latentops::flip_h(&x),
                "v" => latentops::flip_v(&x),
                other => return Err(format!("--axis {other}: want h or v")),
            }
        }
        "rot90" => latentops::rot90(&load_in()?, a.num("k")?),
        "rotate" => latentops::rotate(&load_in()?, a.num("deg")?),
        "roll" => latentops::roll(&load_in()?, a.num_or("dy", 0)?, a.num_or("dx", 0)?),
        "chan" => {
            let spec = a.get("op")?;
            let cop = match spec.split_once(':') {
                Some(("scale", f)) => ChanOp::Scale(f.parse().map_err(|_| "--op scale:F")?),
                Some(("cscale", f)) => ChanOp::ScaleCentred(f.parse().map_err(|_| "--op cscale:F")?),
                None if spec == "zero" => ChanOp::Zero,
                None if spec == "mean" => ChanOp::Mean,
                _ => return Err(format!("--op {spec}: want zero|mean|scale:F|cscale:F")),
            };
            channel_op(&load_in()?, a.num("c")?, cop)?
        }
        "noise" => {
            let units = match a.opt("units").unwrap_or("std") {
                "std" => NoiseUnits::PerChannelStd,
                "global" => NoiseUnits::GlobalStd,
                "raw" => NoiseUnits::Raw,
                other => return Err(format!("--units {other}: want std|global|raw")),
            };
            let only = match a.opt("c") {
                Some(v) => Some(v.parse::<usize>().map_err(|_| "--c: not a number")?),
                None => None,
            };
            add_noise(&load_in()?, a.num("sigma")?, units, a.num_or("seed", 0)?, only, a.num_or("smooth", 1)?)
        }
        other => return Err(format!("op {other}: unknown")),
    };
    let path = a.get("out")?;
    out.save(path)?;
    println!("op {name} -> {path} [{},{},{}]", out.c, out.h, out.w);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pixel conversions are the boundary every measured number crosses
    /// twice, so they are pinned rather than assumed.
    ///
    /// The `Kind::Rgb` container MUST be exactly invertible: it is the
    /// pixel-space control arm, and a control that quietly loses a level is
    /// worse than no control. The VAE's own `[-1, 1]` convention is NOT
    /// invertible - that is a property of the reference's truncating rescale,
    /// pinned here so nobody re-derives it as a surprise mid-experiment.
    #[test]
    fn the_pixel_container_is_exact_and_the_vae_convention_is_not() {
        let px: Vec<u8> = (0..=255u8).flat_map(|v| [v, 255 - v, v / 2]).collect();
        let (h, w) = (16, 16);
        assert_eq!(px.len(), h * w * 3);
        assert_eq!(from_levels(&to_levels(&px, h, w), h, w), px);

        let via_vae = to_rgb8(&to_chw(&px, h, w), h, w);
        assert_ne!(via_vae, px, "if this now round-trips, drop the separate levels container");
        let worst = via_vae.iter().zip(&px).map(|(&a, &b)| a as i32 - b as i32).min().unwrap();
        assert_eq!(worst, -1, "the truncating rescale should lose at most one level");
    }

    #[test]
    fn flags_and_positionals_split_cleanly() {
        let argv: Vec<String> =
            ["op", "blend", "--a", "x.lat", "--b", "y.lat", "--alpha", "0.5", "--out", "z.lat"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        let a = Args::parse(&argv[1..]);
        assert_eq!(a.pos, vec!["blend".to_string()]);
        assert_eq!(a.get("a").unwrap(), "x.lat");
        assert_eq!(a.num::<f32>("alpha").unwrap(), 0.5);
        assert_eq!(a.num_or::<i64>("dx", 7).unwrap(), 7);
        assert!(a.get("nope").is_err());
    }
}
