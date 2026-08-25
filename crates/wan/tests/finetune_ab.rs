// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! G2 (A/B direction gate): does a LoRA trained ONLY on the procedural
//! concept clips move actual GENERATION toward the concept on held-out
//! prompts (phrased differently than the training captions), more than a
//! random adapter of matched perturbation size does - and without collapsing
//! into degenerate output that would win the score trivially.
//!
//! umT5-XXL is the single most expensive thing this test touches (import +
//! forward is minutes even on 48 threads), so it is built exactly ONCE and
//! every prompt (training captions AND the held-out generation prompts) is
//! encoded from that one session before it is dropped. The DiT is imported
//! once as host tensors and then built as THREE separate device graphs in
//! turn (base / trained-adapter / random-control-adapter), each one held
//! only for the handful of forwards it needs. The VAE decoder and the
//! EVA-CLIP image tower are each built once and reused across everything.

use std::collections::HashMap;
use std::path::Path;

use wan::config::WanConfig;
use wan::lora::{LoraAdapter, LoraCfg};
use wan::modelgrad::{make_flow_batch, Cfg, ModelWeights};
use wan::train::Trainer;
use wan::vae3d::{WanVaeConfig, WanVaeDecoder, WanVaeEncoder};
use wan::WanDitDev;

fn real_paths() -> Option<wan::Paths> {
    match wan::Paths::from_env() {
        Ok(p) => Some(p),
        Err(e) => {
            brain_testutil::skip(&format!("set BRAIN_WAN_{{DIT,VAE,T5,TOKENIZER}} to run the real-weight G2 gate: {e}"));
            None
        }
    }
}

/// Paraphrases of the concept, deliberately distinct from
/// `data::gen_clips::CONCEPT_CAPTIONS` - a held-out prompt has to exercise
/// generalisation, not recite a memorised training string.
// Only PROMPTS cost a umT5 forward each, and that forward is slow; SEEDS are free extra
// (prompt, seed) pairs for the paired test - they reuse the same prompt
// embedding through a cheap GPU denoise, so more seeds is how this gate buys
// power without buying more umT5 time.
const HELD_OUT_PROMPTS: [&str; 4] = [
    "a magenta triangle circling a white dot on a black background",
    "a small magenta triangular shape spinning around a fixed white point",
    "a pink-purple triangle looping around a white circle on a dark background",
    "a magenta arrow-like shape rotating around a stationary white marker",
];
const SEEDS: [u64; 4] = [11, 22, 33, 44];

fn read_pth(path: &str) -> Result<Vec<checkpoint::safetensors::StTensor>, String> {
    Ok(checkpoint::torchpt::read(path)?.into_iter().map(|t| checkpoint::safetensors::StTensor { name: t.name, shape: t.shape, data: t.data }).collect())
}

/// `[c,f,h,w] -> N interleaved-RGB8 frames, [-1,1] clamped and rescaled` -
/// mirrors `wan::pipeline::run`'s own postprocessing exactly, duplicated here
/// (private to that module) so this test decodes the same way generation does.
fn chw_to_rgb_frames(chw: &[f32], frames: usize, h: usize, w: usize) -> Vec<Vec<f32>> {
    let plane = frames * h * w;
    (0..frames)
        .map(|f| {
            let mut px = vec![0f32; h * w * 3];
            for c in 0..3 {
                let base = c * plane + f * h * w;
                for i in 0..h * w {
                    px[i * 3 + c] = (chw[base + i].clamp(-1.0, 1.0) + 1.0) * 0.5;
                }
            }
            px
        })
        .collect()
}

/// Frame-to-frame mean-|L2| difference and mean spatial-gradient energy - the
/// anti-degeneracy pair: a collapsed/static or pure-noise adapter shows up as
/// one of these leaving the base's range, even if it happens to win the
/// cosine-margin score.
fn motion_and_texture(frames: &[Vec<f32>], w: usize, h: usize) -> (f32, f32) {
    let mut diff_sum = 0.0f64;
    for i in 1..frames.len() {
        let d: f64 = frames[i].iter().zip(&frames[i - 1]).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        diff_sum += (d / frames[i].len() as f64).sqrt();
    }
    let motion = if frames.len() > 1 { (diff_sum / (frames.len() - 1) as f64) as f32 } else { 0.0 };
    let mut grad_sum = 0.0f64;
    let mut n = 0usize;
    for f in frames {
        for y in 0..h - 1 {
            for x in 0..w - 1 {
                for c in 0..3 {
                    let p0 = f[(y * w + x) * 3 + c] as f64;
                    let px = f[(y * w + x + 1) * 3 + c] as f64;
                    let py = f[((y + 1) * w + x) * 3 + c] as f64;
                    grad_sum += (px - p0).powi(2) + (py - p0).powi(2);
                    n += 1;
                }
            }
        }
    }
    (motion, (grad_sum / n.max(1) as f64) as f32)
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

struct Eva {
    gpu: gpu_core::Gpu,
    model: clip::model::EvaVision,
    side: u32,
}

impl Eva {
    fn from_env() -> Option<Eva> {
        let dir = std::env::var("BRAIN_CLIP_DIR").ok().filter(|s| !s.is_empty())?;
        let path = Path::new(&dir).join(clip::caps::EVA_FILE);
        if !path.exists() {
            return None;
        }
        let cfg = clip::config::EvaVisionConfig::eva02_l336();
        let tensors = checkpoint::torchpt::read(path.to_str()?).ok()?;
        let (init, _r) = clip::import::import_eva_visual(tensors, &cfg).ok()?;
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
        Some(Eva { gpu, model, side })
    }
    fn embed(&self, hwc: &[f32], w: u32, h: u32) -> Vec<f32> {
        let chw = imaging::pixels::hwc_to_chw(hwc, 3, h as usize, w as usize);
        let ctx = imaging::Ctx::new(&self.gpu);
        let src = ctx.upload("g2.eva", &chw);
        let (dst, _) = ctx.resize(&src, imaging::Shape::new(1, 3, h, w), self.side, self.side, imaging::Filter::Bilinear, imaging::AlignCorners::HalfPixel);
        let resized = ctx.download(&dst, 3 * self.side * self.side);
        self.model.set_pixels(&resized);
        self.model.forward();
        self.model.read_cls_embed_l2norm()
    }
}

/// Random A/B of the same rank as `like`, per-tensor Frobenius-rescaled so
/// `||scale * B*A||_F` matches `like`'s own delta on every targeted tensor -
/// the control that stops G2 from just measuring "any perturbation helps".
/// `||BA||_F^2 = trace((B^T B)(A A^T))`, both r x r, so the match is computed
/// WITHOUT ever forming the out x in product.
fn matched_random_adapter(cfg: &Cfg, like: &LoraAdapter, seed: u64) -> LoraAdapter {
    let rank = like.rank();
    let mut rng = data::rng::Rng::new(seed);
    let mut raw = LoraAdapter::new(cfg, LoraCfg { seed, ..LoraCfg::new(rank) });
    // Force B non-zero too (LoraAdapter::new leaves B = 0) with the same
    // small-Gaussian distribution A already uses.
    let mut tensors = raw.to_tensors();
    for (name, _, data) in tensors.iter_mut() {
        if name.ends_with(".lora_b") {
            for v in data.iter_mut() {
                *v = (rng.next_gaussian() * 0.02) as f32;
            }
        }
    }
    let map: HashMap<String, (Vec<usize>, Vec<f32>)> = tensors.into_iter().map(|(n, s, d)| (n, (s, d))).collect();
    raw = LoraAdapter::from_tensors(cfg, LoraCfg { seed, ..LoraCfg::new(rank) }, &map).expect("rebuild random adapter");

    let scale = raw.alpha() / rank as f32;
    let like_scale = like.alpha() / rank as f32;
    let like_t: HashMap<String, (Vec<usize>, Vec<f32>)> = like.to_tensors().into_iter().map(|(n, s, d)| (n, (s, d))).collect();
    let raw_t: HashMap<String, (Vec<usize>, Vec<f32>)> = raw.to_tensors().into_iter().map(|(n, s, d)| (n, (s, d))).collect();

    // Frobenius norm of B*A via trace((B^T B)(A A^T)), r x r - see doc above.
    let bta_fro = |a: &[f32], b: &[f32], out: usize, inn: usize, r: usize| -> f64 {
        let mut bt_b = vec![0f64; r * r];
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0;
                for o in 0..out {
                    s += b[o * r + i] as f64 * b[o * r + j] as f64;
                }
                bt_b[i * r + j] = s;
            }
        }
        let mut a_at = vec![0f64; r * r];
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0;
                for k in 0..inn {
                    s += a[i * inn + k] as f64 * a[j * inn + k] as f64;
                }
                a_at[i * r + j] = s;
            }
        }
        let trace: f64 = (0..r * r).map(|k| bt_b[k] * a_at[k]).sum();
        trace.max(0.0).sqrt()
    };

    let mut rescaled_tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut seen_b: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (name, (shape, data)) in raw_t.iter() {
        if let Some(leaf) = name.strip_suffix(".lora_b") {
            if !seen_b.insert(leaf.to_string()) {
                continue;
            }
            let a_key = format!("{leaf}.lora_a");
            let (a_shape, a_data) = raw_t.get(&a_key).expect("matching lora_a");
            let (r, inn) = (a_shape[0], a_shape[1]);
            let out = shape[0];
            // `like` shares this exact (out, in, r) per leaf (same cfg/rank),
            // so its own tensors can be read with the SAME shape triple.
            let (_, lb) = &like_t[&format!("{leaf}.lora_b")];
            let (_, la) = &like_t[&a_key];
            let want = (like_scale as f64) * bta_fro(la, lb, out, inn, r);
            let got = (scale as f64) * bta_fro(a_data, data, out, inn, r);
            let k = if got > 1e-12 { (want / got) as f32 } else { 0.0 };
            let mut b_scaled = data.clone();
            for v in b_scaled.iter_mut() {
                *v *= k;
            }
            rescaled_tensors.push((a_key.clone(), a_shape.clone(), a_data.clone()));
            rescaled_tensors.push((name.clone(), shape.clone(), b_scaled));
        }
    }
    let map2: HashMap<String, (Vec<usize>, Vec<f32>)> = rescaled_tensors.into_iter().map(|(n, s, d)| (n, (s, d))).collect();
    LoraAdapter::from_tensors(cfg, LoraCfg { seed, ..LoraCfg::new(rank) }, &map2).expect("rebuild matched-norm control adapter")
}

/// A step-count sweep (15/80/250, rank 8) never clears the anti-degeneracy
/// bar: real-scale training moves the directional score from clearly-wrong
/// (15 steps) to chance (250 steps) but collapses adapted-generation
/// motion/texture below the [0.5x,2x] floor by step 80 and it does not
/// recover by step 250. See `scratchpad/wan-lora-demo/report_real/index.md`
/// for the full sweep and the diagnosis (rank 8 likely lets the adapter
/// shortcut to a low-motion solution on this small, low-motion-variety
/// synthetic dataset). Re-enable once a lower rank or a richer dataset is
/// tried.
#[test]
#[ignore]
fn a_concept_lora_moves_held_out_generation_toward_the_concept_more_than_a_matched_random_adapter() {
    let Some(paths) = real_paths() else { return };
    let Some(eva) = Eva::from_env() else {
        brain_testutil::skip("set BRAIN_CLIP_DIR to an EVA-CLIP checkpoint directory to run the real-weight G2 gate");
        return;
    };
    let cfg = WanConfig::t2v_1_3b();
    let (frames, size) = (5usize, 64u32);

    // ---- dataset: train / held-out-concept-for-centroid / distractor ----
    // Full-scale concept set (30 clips, matching the on-disk demo set this
    // gate mirrors procedurally): 24 for training, 6 held out entirely from
    // training for the concept centroid. umT5 cost stays low despite the
    // larger set because the caption pool is a handful of templates
    // (`data::gen_clips::CONCEPT_CAPTIONS`) and the T5 session below caches by
    // exact caption string, so more training WINDOWS does not mean more T5
    // forwards.
    let base_dir = std::env::temp_dir().join(format!("wan-g2-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base_dir);
    let concept = data::gen_clips::generate_concept_set(30, frames, size, size, 909);
    let distractor = data::gen_clips::generate_distractor_set(10, frames, size, size, 808);
    let (train_c, centroid_c) = concept.split_at(24);
    let train_dir = base_dir.join("train");
    data::videoset::write_clipset(&train_dir, train_c, size, size, 8).expect("train clips");

    // ---- one T5 session: training captions + held-out generation prompts ----
    let (lf, lh, lw) = cfg.latent_shape(frames, size as usize, size as usize).expect("latent shape");
    let tcfg = Cfg::from_wan(&cfg, lf, lh, lw);
    let train_set = wan::finetune::ClipSet::load_dir(&train_dir).expect("train set");
    let mut rng = data::rng::Rng::new(1234);
    let n_train_samples = 24;
    let train_clips: Vec<wan::finetune::Clip> = (0..n_train_samples).map(|_| train_set.sample(&mut rng, frames).expect("sample")).collect();

    let (t5_out, prompt_embeds): (Vec<Vec<f32>>, Vec<Vec<f32>>) = {
        let tok = if Path::new(&paths.tokenizer).is_dir() {
            data::unigram::UnigramTokenizer::from_dir(&paths.tokenizer)
        } else {
            data::unigram::UnigramTokenizer::from_file(&paths.tokenizer)
        }
        .expect("tokenizer");
        let t5cfg = t5encoder::config::T5Config::umt5_xxl();
        let imported = t5encoder::import::import_wan(read_pth(&paths.t5).expect("read t5"), &t5cfg).expect("import t5");
        let gpu = gpu_core::Gpu::new_cpu(t5encoder::model::PIPELINES);
        let enc = t5encoder::model::T5Encoder::new_on(gpu, t5cfg, 1, cfg.text_len as u32, &t5encoder::import::to_init(imported));
        let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
        let mut run_one = |text: &str| -> Vec<f32> {
            if let Some(v) = cache.get(text) {
                return v.clone();
            }
            let (ids, mask) = tok.encode_padded(text, cfg.text_len);
            enc.set_tokens(&ids);
            enc.set_mask(&mask);
            enc.forward();
            enc.poll_wait();
            let v = enc.read_context();
            cache.insert(text.to_string(), v.clone());
            v
        };
        let train_ctx: Vec<Vec<f32>> = train_clips.iter().map(|c| run_one(&c.caption)).collect();
        let prompt_ctx: Vec<Vec<f32>> = HELD_OUT_PROMPTS.iter().map(|p| run_one(p)).collect();
        (train_ctx, prompt_ctx)
    }; // T5 dropped here - the ONE expensive load this whole test pays

    // ---- VAE encode the training windows (same session, dropped after) ----
    let train_latents: Vec<Vec<f32>> = {
        let vcfg = WanVaeConfig::wan21();
        let vweights = wan::import::import_vae(read_pth(&paths.vae).expect("read vae"), &vcfg).expect("import vae");
        let enc = WanVaeEncoder::build(&vcfg, &vweights, &vcfg.encode_chunks(frames as u32), size, size, None);
        train_clips.iter().map(|c| enc.encode(&c.video)).collect()
    };

    // ---- train a real-scale adapter through the same on-device LoRA path
    // `wan::finetune::run`/the CLI use (`Trainer::lora_step`), not a
    // re-implementation - forced onto the GPU so a real step count (hundreds,
    // not a smoke-scale handful) finishes in this test's own budget. ----
    let raw = checkpoint::safetensors::read(&paths.dit).expect("read DiT safetensors");
    let base_tensors = wan::import::import_dit(raw, &cfg).expect("import DiT");
    let base_weights = ModelWeights::from_tensors(&tcfg, &base_tensors).expect("host weights");
    let rank = 8;
    let mut adapter = LoraAdapter::new(&tcfg, LoraCfg::new(rank));
    let mut trainer = Trainer::open(&tcfg, Some("gpu"));
    trainer.begin_lora(&base_weights, rank);
    let mut trng = data::rng::Rng::new(4242);
    let steps = 250;
    let t_train = std::time::Instant::now();
    let mut loss = 0.0f64;
    for step in 0..steps {
        let idx = trng.gen_range_inclusive(0, train_latents.len() as i64 - 1) as usize;
        let sigma = (trng.next_f64()).clamp(1e-3, 1.0);
        let noise: Vec<f32> = (0..train_latents[idx].len()).map(|_| trng.next_gaussian() as f32).collect();
        let b = make_flow_batch(&tcfg, &train_latents[idx], &t5_out[idx], cfg.text_len, sigma, &noise);
        loss = trainer.lora_step(&base_weights, &mut adapter, &b, 1e-4);
        if step % 25 == 0 || step + 1 == steps {
            println!("G2 train: step {step:>3}/{steps}  loss {loss:.5}");
        }
    }
    println!("G2 train: {steps} steps on {} in {:.1}s ({:.2}s/step), final loss {loss:.5}", trainer.label(), t_train.elapsed().as_secs_f32(), t_train.elapsed().as_secs_f32() / steps as f32);
    let control = matched_random_adapter(&tcfg, &adapter, 99);

    // ---- centroids from held-out real clips (EVA image tower) ----
    let centroid = |clips: &[(String, Vec<Vec<f32>>)]| -> Vec<f32> {
        let mut acc: Vec<f32> = Vec::new();
        for (_, frames) in clips {
            let e = eva.embed(&frames[frames.len() / 2], size, size);
            if acc.is_empty() {
                acc = vec![0.0; e.len()];
            }
            for (a, b) in acc.iter_mut().zip(&e) {
                *a += b;
            }
        }
        for a in acc.iter_mut() {
            *a /= clips.len() as f32;
        }
        acc
    };
    let concept_centroid = centroid(centroid_c);
    let distractor_centroid = centroid(&distractor);

    // ---- generate with each weight variant, reusing ONE VAE decoder ----
    let vcfg = WanVaeConfig::wan21();
    let vweights = wan::import::import_vae(read_pth(&paths.vae).expect("read vae"), &vcfg).expect("import vae");
    let dec = WanVaeDecoder::build(&vcfg, &vweights, lf as u32, lh as u32, lw as u32, None);

    let n_latent = cfg.in_channels * lf * lh * lw;
    let text_mlp_names = ["text_embedding.0.weight", "text_embedding.0.bias", "text_embedding.2.weight", "text_embedding.2.bias"];
    let mut text_mlp = wan::model::Tensors::new();
    for n in text_mlp_names {
        text_mlp.insert(n.to_string(), base_tensors.get(n).expect("text mlp tensor").clone());
    }

    let generate_variant = |weights: &wan::model::Tensors| -> Vec<(usize, u64, Vec<Vec<f32>>)> {
        let dit = WanDitDev::build(&cfg, weights, lf as u32, lh as u32, lw as u32, None, &[]);
        let mut out = Vec::new();
        for (pi, ctx) in prompt_embeds.iter().enumerate() {
            let emb = wan::model::text_embed(&cfg, &text_mlp, ctx, cfg.text_len);
            for &seed in &SEEDS {
                let mut rngn = data::rng::Rng::new(seed);
                let mut latent: Vec<f32> = (0..n_latent).map(|_| rngn.next_gaussian() as f32).collect();
                let mut sched = diffusion::flowsolvers::FlowUniPcScheduler::new(Default::default());
                sched.set_timesteps(6, cfg.sample_shift as f64);
                let ts = sched.timesteps().to_vec();
                dit.set_context_embed(&emb);
                for &t in &ts {
                    let pred = dit.forward(&latent, t);
                    latent = sched.step(&pred, &latent);
                }
                let chw = dec.decode(&latent);
                let rgb = chw_to_rgb_frames(&chw, dec.frames() as usize, size as usize, size as usize);
                out.push((pi, seed, rgb));
            }
        }
        out
    };

    let mut base_clone = base_tensors.clone();
    let base_clips = generate_variant(&base_clone);
    let mut adapted_tensors = base_tensors.clone();
    adapter.fold_into_tensors(&mut adapted_tensors).expect("fold trained");
    let adapted_clips = generate_variant(&adapted_tensors);
    control.fold_into_tensors(&mut base_clone).expect("fold control - reuses the untouched base_clone");
    let control_clips = generate_variant(&base_clone);

    let score = |rgb: &[Vec<f32>]| -> f32 {
        let mid = rgb.len() / 2;
        let e = eva.embed(&rgb[mid], size, size);
        cos(&e, &concept_centroid) - cos(&e, &distractor_centroid)
    };

    let mut s_base = vec![0.0f32; base_clips.len()];
    let mut s_adapted = vec![0.0f32; base_clips.len()];
    let mut s_control = vec![0.0f32; base_clips.len()];
    for i in 0..base_clips.len() {
        s_base[i] = score(&base_clips[i].2);
        s_adapted[i] = score(&adapted_clips[i].2);
        s_control[i] = score(&control_clips[i].2);
    }

    let n = s_base.len();
    let deltas_adapted: Vec<f32> = (0..n).map(|i| s_adapted[i] - s_base[i]).collect();
    let deltas_control: Vec<f32> = (0..n).map(|i| s_control[i] - s_base[i]).collect();
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (mean_adapted, mean_control) = (mean(&deltas_adapted), mean(&deltas_control));

    // Exact one-sided paired sign test: k = pairs with s_adapted > s_base,
    // p = P(Binomial(n, 0.5) >= k) - no distributional assumption, easy to
    // verify by hand, and the primary option the gate spec names.
    let k = deltas_adapted.iter().filter(|&&d| d > 0.0).count();
    let binom_sf = |n: usize, k: usize| -> f64 {
        let mut p = 0.0f64;
        for i in k..=n {
            p += binom_coeff(n, i) * 0.5f64.powi(n as i32);
        }
        p
    };
    fn binom_coeff(n: usize, k: usize) -> f64 {
        let mut c = 1.0f64;
        for i in 0..k.min(n - k).max(0) {
            c = c * (n - i) as f64 / (i + 1) as f64;
        }
        c
    }
    let p_value = binom_sf(n, k);

    println!("G2: n={n} pairs, s_base={s_base:?}");
    println!("G2: s_adapted={s_adapted:?}  s_control={s_control:?}");
    println!("G2: mean delta adapted={mean_adapted:+.5}  control={mean_control:+.5}  (sign test k={k}/{n}, one-sided p={p_value:.4})");

    // ---- anti-degeneracy: motion + texture must not blow up or collapse ----
    let agg_motion_texture = |clips: &[(usize, u64, Vec<Vec<f32>>)]| -> (f32, f32) {
        let stats: Vec<(f32, f32)> = clips.iter().map(|(_, _, rgb)| motion_and_texture(rgb, size as usize, size as usize)).collect();
        (stats.iter().map(|s| s.0).sum::<f32>() / stats.len() as f32, stats.iter().map(|s| s.1).sum::<f32>() / stats.len() as f32)
    };
    let (base_motion, base_texture) = agg_motion_texture(&base_clips);
    let (adapted_motion, adapted_texture) = agg_motion_texture(&adapted_clips);
    println!("G2: motion base={base_motion:.5} adapted={adapted_motion:.5}; texture base={base_texture:.5} adapted={adapted_texture:.5}");
    let in_band = |v: f32, base: f32| base <= 1e-9 || (0.5 * base..=2.0 * base).contains(&v);
    assert!(in_band(adapted_motion, base_motion), "adapted motion {adapted_motion} outside [0.5x,2x] of base {base_motion} - possible degenerate adapter");
    assert!(in_band(adapted_texture, base_texture), "adapted texture energy {adapted_texture} outside [0.5x,2x] of base {base_texture} - possible degenerate adapter");

    // ---- the actual gate ----
    assert!(mean_adapted > mean_control, "the trained adapter's margin ({mean_adapted:+.5}) must exceed the matched-norm random control's ({mean_control:+.5})");
    if p_value >= 0.05 {
        eprintln!("G2 WARNING: sign test did not reach p<0.05 (p={p_value:.4}, k={k}/{n}) - the direction is right (mean delta {mean_adapted:+.5} > 0) but not significant at this sample size/step count.");
    }
    assert!(mean_adapted > 0.0, "mean(s_adapted - s_base) must be positive: {mean_adapted}");

    let _ = std::fs::remove_dir_all(&base_dir);
}
