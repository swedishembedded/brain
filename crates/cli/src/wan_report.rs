// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain wan finetune --report <dir>` (G3): the human-readable artifact that
//! proves a trained LoRA adapter actually changed generation, not just the
//! training loss number. Writes, under `<dir>`, with only relative paths:
//!
//! * `index.md` - run summary, the loss curve, and (base, adapted) pairs for
//!   a few held-out prompts/seeds with links to their clips/contact sheets.
//! * `loss.csv` - `step,loss`.
//! * `clips/{base,adapted}_<prompt>_<seed>.mp4` - short clips from the SAME
//!   seed/steps/prompt, with and without the adapter folded in.
//! * `sheets/{base,adapted}_<prompt>_<seed>.png` - first/middle/last frame
//!   side by side, so the difference is visible without a video player.
//! * a lightweight EVA-CLIP image-tower score table (`s_base`, `s_adapted`,
//!   `delta`) when `BRAIN_CLIP_DIR` names an EVA-CLIP checkpoint directory -
//!   the FULL statistically-controlled comparison (random-adapter control,
//!   paired significance test) is `crates/wan/tests/finetune_ab.rs`'s job,
//!   not this report's; this table is a quick visual/numeric sanity check.

use std::collections::HashMap;
use std::path::Path;

use wan::pipeline::{GenOpts, Paths};
use wan::WanConfig;

/// Paraphrases of the concept, deliberately NOT the exact training captions
/// (`data::gen_clips::CONCEPT_CAPTIONS`) - a held-out prompt has to exercise
/// generalisation, not recite a memorised string.
// ONE held-out prompt / seed: each `generate_one` call reloads BOTH the
// umT5-XXL encoder and the DiT from disk (no cross-call cache here - see the
// module doc), and a umT5-XXL CPU forward alone measures in minutes
// (`tests/lora_train.rs`'s G1 gate: ~7.3 min for 4 short captions).
// This report's own wall-clock is `2 * len(HELD_OUT_PROMPTS) * len(SEEDS)`
// full generations, so it stays at the minimum that still shows a real base
// vs. adapted pair; the statistically powered comparison across many
// (prompt, seed) pairs is `crates/wan/tests/finetune_ab.rs`'s job, which
// amortises the umT5 reload to ONE session for everything it needs.
const HELD_OUT_PROMPTS: [&str; 1] = ["a magenta triangle circling a white dot on a black background"];
const SEEDS: [u64; 1] = [1000];

struct ClipOut {
    frames_hwc: Vec<(Vec<f32>, u32, u32)>,
    width: u32,
    height: u32,
    fps: usize,
}

fn generate_one(cfg: &WanConfig, paths: &Paths, prompt: &str, seed: u64, frames: usize, adapter: Option<String>) -> Result<ClipOut, String> {
    let o = GenOpts {
        frames,
        width: 256,
        height: 256,
        steps: 6,
        shift: 5.0,
        guidance: 1.0, // one forward/step - the A/B contrast rides the fold, not CFG
        seed,
        negative_prompt: None,
        solver: wan::pipeline::Solver::UniPc,
        fps: 8,
        device: None,
        te_device: None,
        adapter,
        dit_dtype: wan::WanDtype::F32,
    };
    let cancel = capability::CancelToken::default();
    let (video, _) = wan::generate(cfg, paths, prompt, &o, &cancel, |_, _, _| {})?;
    let frames_hwc = video.frames.iter().map(|px| (px.iter().map(|&b| b as f32 / 255.0).collect::<Vec<f32>>(), video.width, video.height)).collect();
    Ok(ClipOut { frames_hwc, width: video.width, height: video.height, fps: video.fps })
}

fn write_clip_mp4(clip: &ClipOut, path: &Path) -> Result<(), String> {
    let frames: Vec<imaging::Rgb8> = clip
        .frames_hwc
        .iter()
        .map(|(hwc, w, h)| imaging::Rgb8::new(*w, *h, hwc.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect()))
        .collect::<Result<_, _>>()?;
    imaging::video::encode_frames(&frames, path, clip.fps as f64, &Default::default())?;
    Ok(())
}

/// First/middle/last frame, concatenated left-to-right, as one PNG.
fn write_contact_sheet(clip: &ClipOut, path: &Path) -> Result<(), String> {
    let n = clip.frames_hwc.len();
    if n == 0 {
        return Err("wan report: a clip with no frames cannot make a contact sheet".into());
    }
    let idxs = [0, n / 2, n - 1];
    let (w, h) = (clip.width, clip.height);
    let mut px = vec![0u8; (w as usize * 3) * h as usize * 3];
    for (slot, &fi) in idxs.iter().enumerate() {
        let (hwc, _, _) = &clip.frames_hwc[fi];
        for y in 0..h as usize {
            for x in 0..w as usize {
                for c in 0..3 {
                    let src = (y * w as usize + x) * 3 + c;
                    let dst = (y * (w as usize * 3) + slot * w as usize + x) * 3 + c;
                    px[dst] = (hwc[src].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
    }
    let img = imaging::Rgb8::new(w * 3, h, px)?;
    imaging::codec::save_png(path, &img)
}

/// One held-out EVA-CLIP image embedding session, built once and reused
/// across every frame this report embeds. `None` when `BRAIN_CLIP_DIR` is
/// unset or holds no EVA checkpoint - the G2 table is then omitted rather
/// than failing the whole report over an optional check.
struct EvaSession {
    gpu: gpu_core::Gpu,
    model: clip::model::EvaVision,
    side: u32,
}

impl EvaSession {
    fn from_env() -> Option<EvaSession> {
        let dir = std::env::var("BRAIN_CLIP_DIR").ok().filter(|s| !s.is_empty())?;
        let path = Path::new(&dir).join(clip::caps::EVA_FILE);
        if !path.exists() {
            return None;
        }
        let cfg = clip::config::EvaVisionConfig::eva02_l336();
        let tensors = checkpoint::torchpt::read(path.to_str()?).ok()?;
        let (init, _report) = clip::import::import_eva_visual(tensors, &cfg).ok()?;
        let map: HashMap<String, Vec<f32>> = init.into_iter().map(|(k, (_, d))| (k, d)).collect();
        // EVA's own kernels (conv2d/bidir-attention/rope2d/...) live in
        // `VISION_PIPELINES`, not `TEXT_PIPELINES` (`crates/clip/tests/parity.rs`
        // builds `EvaVision` against `VISION_PIPELINES` for exactly this reason);
        // `imaging::PIPELINES` is added for the resize this struct's own `embed`
        // dispatches - a Gpu built from only one of the two produces a wrong
        // kernel-id resolution for whichever half is missing, which surfaces as
        // a wgpu bind-group-size mismatch rather than a clean "not registered".
        let kernels: Vec<(&str, &str)> = clip::model::VISION_PIPELINES.iter().chain(imaging::PIPELINES.iter()).copied().collect();
        let gpu = gpu_core::Gpu::new(&kernels);
        let side = cfg.image_size;
        let model = clip::model::EvaVision::new_on(gpu.share(), cfg, 1, &map);
        Some(EvaSession { gpu, model, side })
    }

    fn embed(&self, hwc: &[f32], w: u32, h: u32) -> Vec<f32> {
        let chw = imaging::pixels::hwc_to_chw(hwc, 3, h as usize, w as usize);
        let ctx = imaging::Ctx::new(&self.gpu);
        let src = ctx.upload("wan_report.eva", &chw);
        let (dst, _) = ctx.resize(&src, imaging::Shape::new(1, 3, h, w), self.side, self.side, imaging::Filter::Bilinear, imaging::AlignCorners::HalfPixel);
        let resized = ctx.download(&dst, 3 * self.side * self.side);
        self.model.set_pixels(&resized);
        self.model.forward();
        self.model.read_cls_embed_l2norm()
    }
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let (na, nb) = (a.iter().map(|x| x * x).sum::<f32>().sqrt(), b.iter().map(|x| x * x).sum::<f32>().sqrt());
    if na <= 1e-12 || nb <= 1e-12 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn centroid(eva: &EvaSession, clips: &[(String, Vec<Vec<f32>>)], w: u32, h: u32, max_clips: usize) -> Vec<f32> {
    let mut acc: Vec<f32> = Vec::new();
    let mut n = 0usize;
    for (_, frames) in clips.iter().take(max_clips) {
        let mid = frames.len() / 2;
        let e = eva.embed(&frames[mid], w, h);
        if acc.is_empty() {
            acc = vec![0.0; e.len()];
        }
        for (a, b) in acc.iter_mut().zip(&e) {
            *a += b;
        }
        n += 1;
    }
    if n > 0 {
        for a in acc.iter_mut() {
            *a /= n as f32;
        }
    }
    acc
}

/// `data_dir`'s sibling `distractor/` directory, by the convention
/// `gen_wan_clips` writes (`<out>/concept`, `<out>/distractor`).
fn sibling_distractor_dir(data_dir: &Path) -> Option<std::path::PathBuf> {
    let parent = data_dir.parent()?;
    let d = parent.join("distractor");
    d.is_dir().then_some(d)
}

#[allow(clippy::too_many_arguments)]
pub fn write_report(
    paths: &Paths,
    cfg: &WanConfig,
    opts: &wan::finetune::TrainOpts,
    data_dir: &Path,
    adapter_path: &Path,
    losses: &[(u32, f32)],
    out_dir: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(out_dir.join("clips")).map_err(|e| format!("wan report: {e}"))?;
    std::fs::create_dir_all(out_dir.join("sheets")).map_err(|e| format!("wan report: {e}"))?;

    let mut loss_csv = String::from("step,loss\n");
    for (s, l) in losses {
        loss_csv.push_str(&format!("{s},{l}\n"));
    }
    std::fs::write(out_dir.join("loss.csv"), &loss_csv).map_err(|e| format!("wan report: {e}"))?;

    let adapter_str = adapter_path.to_string_lossy().to_string();
    let mut rows: Vec<(String, u64, String, String, String, String)> = Vec::new(); // (prompt, seed, base_mp4, adapted_mp4, base_png, adapted_png)
    let mut scores: Vec<(String, u64, f32, f32, f32)> = Vec::new(); // (prompt, seed, s_base, s_adapted, delta)

    let eva = EvaSession::from_env();
    let concept_centroid_and_w_h = eva.as_ref().map(|e| {
        // `data_dir` IS the concept set this adapter trained on; reusing it
        // for the centroid is the honest tradeoff this report makes (the held
        // out/train split for the centroid lives in the G2 gate test, which
        // is what actually GATES the claim - this table is a sanity check).
        let clips = data::gen_clips::generate_concept_set(6, opts.frames, 256, 256, opts.seed ^ 0x51de);
        (centroid(e, &clips, 256, 256, 6), sibling_distractor_dir(data_dir))
    });

    for (pi, prompt) in HELD_OUT_PROMPTS.iter().enumerate() {
        for &seed in &SEEDS {
            let base = generate_one(cfg, paths, prompt, seed, opts.frames, None)?;
            let adapted = generate_one(cfg, paths, prompt, seed, opts.frames, Some(adapter_str.clone()))?;

            let base_mp4 = format!("clips/base_{pi}_{seed}.mp4");
            let adapted_mp4 = format!("clips/adapted_{pi}_{seed}.mp4");
            let base_png = format!("sheets/base_{pi}_{seed}.png");
            let adapted_png = format!("sheets/adapted_{pi}_{seed}.png");
            write_clip_mp4(&base, &out_dir.join(&base_mp4))?;
            write_clip_mp4(&adapted, &out_dir.join(&adapted_mp4))?;
            write_contact_sheet(&base, &out_dir.join(&base_png))?;
            write_contact_sheet(&adapted, &out_dir.join(&adapted_png))?;
            rows.push((prompt.to_string(), seed, base_mp4, adapted_mp4, base_png, adapted_png));

            if let Some((concept_c, Some(distractor_dir))) = &concept_centroid_and_w_h {
                if let Some(e) = &eva {
                    let distractor_set = data::episode::EpisodeDataset::open(distractor_dir).ok();
                    if let Some(ds) = distractor_set {
                        let mut d_centroid: Vec<f32> = vec![0.0; concept_c.len()];
                        let take = ds.episodes.len().min(6);
                        for ep in &ds.episodes[..take] {
                            let mid = ep.start + ep.len / 2;
                            let hwc = ds.frame_f32(mid).unwrap();
                            let chw_to_hwc = imaging::pixels::chw_to_hwc(&hwc, 3, ds.h as usize, ds.w as usize);
                            let emb = e.embed(&chw_to_hwc, ds.w, ds.h);
                            for (a, b) in d_centroid.iter_mut().zip(&emb) {
                                *a += b;
                            }
                        }
                        if take > 0 {
                            for a in d_centroid.iter_mut() {
                                *a /= take as f32;
                            }
                        }
                        let score = |c: &ClipOut| -> f32 {
                            let mid = c.frames_hwc.len() / 2;
                            let (hwc, w, h) = &c.frames_hwc[mid];
                            let emb = e.embed(hwc, *w, *h);
                            cos(&emb, concept_c) - cos(&emb, &d_centroid)
                        };
                        let (sb, sa) = (score(&base), score(&adapted));
                        scores.push((prompt.to_string(), seed, sb, sa, sa - sb));
                    }
                }
            }
        }
    }

    let mut md = String::new();
    md.push_str("# Wan LoRA finetune report\n\n");
    md.push_str(&format!(
        "variant `{}`, rank {}, steps {}, frames {}, samples {}, lr {}, seed {}\n\ndata: `{}`\nadapter: `{}`\n\n",
        cfg.name,
        opts.rank,
        opts.steps,
        opts.frames,
        opts.samples,
        opts.lr,
        opts.seed,
        data_dir.display(),
        adapter_path.display()
    ));
    if let Some((first, last)) = losses.first().zip(losses.last()) {
        md.push_str(&format!("loss: step {} = {:.5} -> step {} = {:.5} (full curve: `loss.csv`)\n\n", first.0, first.1, last.0, last.1));
    }
    md.push_str("## Base vs. adapted, held-out prompts\n\n");
    md.push_str("| prompt | seed | base | adapted | base sheet | adapted sheet |\n|---|---|---|---|---|---|\n");
    for (p, seed, bm, am, bp, ap) in &rows {
        md.push_str(&format!("| {p} | {seed} | [{bm}]({bm}) | [{am}]({am}) | ![base]({bp}) | ![adapted]({ap}) |\n"));
    }
    if scores.is_empty() {
        md.push_str("\n_G2 score table omitted: set `BRAIN_CLIP_DIR` to an EVA-CLIP checkpoint directory and keep a sibling `distractor/` set next to `--data` to compute it. The statistically-controlled A/B comparison (random-adapter control, significance test) is `crates/wan/tests/finetune_ab.rs`._\n");
    } else {
        md.push_str("\n## G2 sanity scores (EVA-CLIP image tower; concept-centroid vs distractor-centroid margin)\n\n");
        md.push_str("| prompt | seed | s_base | s_adapted | delta |\n|---|---|---|---|---|\n");
        for (p, seed, sb, sa, d) in &scores {
            md.push_str(&format!("| {p} | {seed} | {sb:.4} | {sa:.4} | {d:+.4} |\n"));
        }
        let mean_delta: f32 = scores.iter().map(|(_, _, _, _, d)| d).sum::<f32>() / scores.len() as f32;
        md.push_str(&format!("\nmean delta = {mean_delta:+.4} ({} pairs; too few for a significance test here - see `finetune_ab.rs`).\n", scores.len()));
    }
    std::fs::write(out_dir.join("index.md"), md).map_err(|e| format!("wan report: {e}"))?;
    Ok(())
}
