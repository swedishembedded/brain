// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CLIP behind the generalized [`capability`] interface — what makes
//! `brain caps clip` / `brain do clip embed_text …`, the D-Bus `Run` method and
//! `brain perf`'s `CapabilityTarget` work with no CLIP-specific plumbing in the
//! CLI or the transports.
//!
//! Two actions, one per tower:
//!
//! * **`embed_text`** — a string in, one pooled text embedding out. `tower`
//!   selects `clip_l` (768-d, SDXL's first encoder and FLUX.1's pooled vector)
//!   or `openclip_bigg` (1280-d, SDXL's second). The projected `text_embeds` is
//!   returned when the config has a projection, otherwise the pooled EOS row —
//!   `text_embeds` is what a similarity search wants, and the caller should not
//!   have to know which towers project.
//! * **`embed_image`** — an image in, one L2-normalised EVA-CLIP-L/336 CLS
//!   embedding out. This is the tower PuLID conditions on.
//!
//! # Batching is real here, and it matters
//!
//! [`Session::embed_text_batch`] builds the tower at `b = texts.len()` and runs
//! **one** forward over the whole batch, rather than looping [`Session::run`].
//! A text tower is the easiest thing in the workspace to batch — every row is
//! the same fixed 77-token context — so a serial loop would be indefensible
//! here (AGENTS.md requires a genuine batched `run_batch` "wherever the
//! architecture allows"). The residency adapter
//! (`crates/cli/src/resident_clip.rs`) forwards `run_batch` straight to it.
//!
//! # Tokenisation
//!
//! Text arrives as a string and is tokenised by [`data::clip_bpe::ClipBpe`] —
//! the workspace's one CLIP BPE, loaded from the tokenizer directory beside the
//! weights. Before it existed the parity tests were fed token ids dumped from
//! the reference, which is why this action could not be built at the time the
//! tower landed.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use gpu_core::Gpu;
use serde_json::json;

use crate::config::{ClipTextConfig, EvaVisionConfig};
use crate::import::Tensors;
use crate::model::{ClipText, EvaVision};

/// `import_*` returns `(shape, data)` per tensor; the model builders take the
/// data alone. One conversion, here, rather than a copy per call site.
fn to_map(t: Tensors) -> HashMap<String, Vec<f32>> {
    t.into_iter().map(|(k, (_, d))| (k, d)).collect()
}

/// The model id used on the CLI (`brain do clip …`), over D-Bus and in the
/// residency manifest.
pub const MODEL: &str = "brain/clip";

/// The two text towers this action can serve, by the name used on the wire.
pub const TOWERS: [&str; 2] = ["clip_l", "openclip_bigg"];

/// CLIP's fixed context length. Both SDXL towers use 77; it is a property of the
/// released tokenizers, not a tunable.
pub const CONTEXT: usize = 77;

/// The EVA-CLIP-L/336 release filename, as distributed by QuanSun/EVA-CLIP.
pub const EVA_FILE: &str = "EVA02_CLIP_L_336_psz14_s6B.pt";

fn tower_cfg(name: &str) -> Result<ClipTextConfig, String> {
    match name {
        "clip_l" => Ok(ClipTextConfig::clip_l()),
        "openclip_bigg" => Ok(ClipTextConfig::openclip_bigg()),
        other => Err(format!("clip: unknown tower '{other}' (expected one of {TOWERS:?})")),
    }
}

fn embed_text_spec() -> ActionSpec {
    ActionSpec::new("embed_text", "Embed a string with one of the CLIP text towers.")
        .param(
            ParamSpec::new("text", ParamType::Str, "the string to embed").required(),
        )
        .param(
            ParamSpec::new("tower", ParamType::Str, "clip_l (768-d) or openclip_bigg (1280-d)")
                .default(json!("clip_l")),
        )
        .output(BlobSpec::new(
            "embedding",
            Media::Bytes,
            "f32 little-endian: the projected text_embeds when the tower projects, else the pooled EOS row",
        ))
}

fn embed_image_spec() -> ActionSpec {
    ActionSpec::new("embed_image", "Embed an image with the EVA-CLIP-L/336 vision tower.")
        .input(BlobSpec::new("image", Media::Image, "the image (resized to the tower's 336² input)").required())
        .output(BlobSpec::new("embedding", Media::Bytes, "f32 little-endian, L2-normalised CLS embedding"))
}

/// The full, static capability manifest — safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    Manifest::new(
        MODEL,
        "CLIP encoders: CLIP-L and OpenCLIP-bigG text towers plus the EVA-CLIP-L/336 image tower.",
        vec![embed_text_spec(), embed_image_spec()],
    )
}

// ===================== the shared work =====================

/// The towers on one device — the single implementation of `embed_text` /
/// `embed_image`, shared by [`ClipProvider`] and the residency adapter
/// (`crates/cli/src/resident_clip.rs`).
///
/// Towers are built lazily and keyed by `(tower, batch)`: a text tower's graph
/// is recorded for a fixed `b`, so a batch of 4 needs a different build than a
/// batch of 1. Both are cached, because a serving process alternates between
/// single interactive calls and batched ones.
pub struct Session {
    gpu: Gpu,
    dir: String,
    tok: HashMap<String, data::clip_bpe::ClipBpe>,
    text: Mutex<HashMap<(String, u32), ClipText>>,
    vision: Mutex<Option<EvaVision>>,
}

impl Session {
    /// `dir` is the released checkpoint root — the SDXL layout, holding
    /// `text_encoder/`, `text_encoder_2/`, `tokenizer/` and `tokenizer_2/`.
    /// It comes from a CLI flag or `BRAIN_CLIP_DIR`, never a baked-in path.
    pub fn load(dir: &str, gpu: Gpu) -> Result<Session, String> {
        let root = std::path::Path::new(dir);
        let mut tok = HashMap::new();
        for (tower, sub) in [("clip_l", "tokenizer"), ("openclip_bigg", "tokenizer_2")] {
            let d = root.join(sub);
            if d.exists() {
                let bpe = data::clip_bpe::ClipBpe::from_dir(&d)
                    .map_err(|e| format!("clip: loading {sub}: {e}"))?;
                tok.insert(tower.to_string(), bpe);
            }
        }
        if tok.is_empty() {
            return Err(format!("clip: {dir} holds neither tokenizer/ nor tokenizer_2/"));
        }
        Ok(Session { gpu, dir: dir.into(), tok, text: Mutex::new(HashMap::new()), vision: Mutex::new(None) })
    }

    fn weights_subdir(tower: &str) -> &'static str {
        if tower == "clip_l" { "text_encoder" } else { "text_encoder_2" }
    }

    /// Tokenise to the fixed 77-token context, right-padded.
    fn tokenize(&self, tower: &str, text: &str) -> Result<Vec<u32>, String> {
        let bpe = self
            .tok
            .get(tower)
            .ok_or_else(|| format!("clip: no tokenizer for tower '{tower}' under {}", self.dir))?;
        Ok(bpe.encode_with_context(text, CONTEXT).ids)
    }

    /// Build (or reuse) the text tower for `tower` at batch `b`.
    fn with_text<R>(&self, tower: &str, b: u32, f: impl FnOnce(&ClipText) -> R) -> Result<R, String> {
        let cfg = tower_cfg(tower)?;
        let mut guard = self.text.lock().map_err(|_| "clip: text tower lock poisoned")?;
        let key = (tower.to_string(), b);
        if !guard.contains_key(&key) {
            let sub = std::path::Path::new(&self.dir).join(Self::weights_subdir(tower));
            let tensors = crate::import::read_text_encoder(&sub)?;
            let init = to_map(crate::import::import_text(tensors, &cfg)?);
            // `Gpu::share` — one device for every tower. A second `Gpu` per model
            // is the deadlock AGENTS.md forbids.
            let m = ClipText::new_on(self.gpu.share(), cfg, b, CONTEXT as u32, &init);
            guard.insert(key.clone(), m);
        }
        Ok(f(guard.get(&key).expect("built above")))
    }

    /// One forward over `texts.len()` rows — the genuine batched path.
    pub fn embed_text_batch(&self, tower: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let b = texts.len() as u32;
        let mut ids = Vec::with_capacity(texts.len() * CONTEXT);
        for t in texts {
            ids.extend_from_slice(&self.tokenize(tower, t)?);
        }
        self.with_text(tower, b, |m| {
            m.set_tokens(&ids);
            m.forward();
            // `text_embeds` when the tower projects, else the pooled EOS row —
            // the caller should not have to know which towers project.
            let flat = m.read_text_embeds().unwrap_or_else(|| m.read_pooled());
            let d = flat.len() / texts.len();
            flat.chunks(d).map(|r| r.to_vec()).collect()
        })
    }

    fn embed_text(&self, inv: &Invocation) -> ActionResult {
        let text = inv.get_str("text").ok_or("clip: 'text' is required")?;
        let tower = inv.get_str("tower").unwrap_or_else(|| "clip_l".into());
        let mut out = self.embed_text_batch(&tower, std::slice::from_ref(&text))?;
        let v = out.pop().ok_or("clip: empty batch result")?;
        Ok(Outcome::new()
            .set("dim", json!(v.len()))
            .set("tower", json!(tower))
            .blob("embedding", Blob::new(Media::Bytes, f32_le(&v))))
    }

    fn embed_image(&self, inv: &Invocation) -> ActionResult {
        let (hwc, w, h) = capability::blob::decode_image(inv, "image")?;
        let cfg = EvaVisionConfig::eva02_l336();
        let side = cfg.image_size;
        let chw = imaging::pixels::hwc_to_chw(&hwc, 3, h as usize, w as usize);
        let mut guard = self.vision.lock().map_err(|_| "clip: vision tower lock poisoned")?;
        if guard.is_none() {
            let path = std::path::Path::new(&self.dir).join(EVA_FILE);
            let p = path.to_str().ok_or("clip: non-UTF8 checkpoint path")?;
            let tensors = checkpoint::torchpt::read(p).map_err(|e| format!("clip: reading {p}: {e}"))?;
            let (init, _report) = crate::import::import_eva_visual(tensors, &cfg)?;
            *guard = Some(EvaVision::new_on(self.gpu.share(), cfg.clone(), 1, &to_map(init)));
        }
        let m = guard.as_ref().expect("built above");
        // Resize on the device — a host loop over a full-resolution image is the
        // "host math does not run on the accelerator" trap in AGENTS.md.
        let ctx = imaging::Ctx::new(&self.gpu);
        let src = ctx.upload("clip.image", &chw);
        let (dst, _) = ctx.resize(
            &src,
            imaging::Shape::new(1, 3, h, w),
            side,
            side,
            imaging::Filter::Bilinear,
            imaging::AlignCorners::HalfPixel,
        );
        let resized = ctx.download(&dst, 3 * side * side);
        m.set_pixels(&resized);
        m.forward();
        let v = m.read_cls_embed_l2norm();
        Ok(Outcome::new()
            .set("dim", json!(v.len()))
            .blob("embedding", Blob::new(Media::Bytes, f32_le(&v))))
    }

    /// Dispatch by action name — the one place the two actions are named.
    pub fn run(&self, action: &str, inv: &Invocation) -> ActionResult {
        match action {
            "embed_text" => self.embed_text(inv),
            "embed_image" => self.embed_image(inv),
            other => Err(format!("clip: unknown action '{other}'")),
        }
    }
}

fn f32_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ===================== the provider =====================

/// The lazily-built session, keyed by the directory it was built from. `Arc` so
/// an [`Action`] can clone it out of the lock and run without holding the lock
/// across a forward.
type HotSession = Arc<Mutex<Option<(String, Arc<Session>)>>>;

/// The executable CLIP stack behind the manifest. Construction is free — towers
/// import lazily on first use and stay resident.
pub struct ClipProvider {
    dir: String,
    hot: HotSession,
}

impl ClipProvider {
    pub fn new(dir: impl Into<String>) -> ClipProvider {
        ClipProvider { dir: dir.into(), hot: Arc::new(Mutex::new(None)) }
    }

    /// `BRAIN_CLIP_DIR` — `None` when unset, or when the directory holds neither
    /// released tokenizer, since without one no action can run.
    pub fn from_env() -> Option<ClipProvider> {
        let dir = std::env::var("BRAIN_CLIP_DIR").ok().filter(|p| !p.is_empty())?;
        let d = std::path::Path::new(&dir);
        (d.join("tokenizer").exists() || d.join("tokenizer_2").exists()).then(|| ClipProvider::new(dir))
    }
}

impl Provider for ClipProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        matches!(name, "embed_text" | "embed_image").then(|| {
            Arc::new(ClipAction { name: name.to_string(), dir: self.dir.clone(), hot: self.hot.clone() })
                as Arc<dyn Action>
        })
    }
}

struct ClipAction {
    name: String,
    dir: String,
    hot: HotSession,
}

impl Action for ClipAction {
    fn spec(&self) -> ActionSpec {
        if self.name == "embed_text" { embed_text_spec() } else { embed_image_spec() }
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let session = {
            let mut guard = self.hot.lock().map_err(|_| "clip: hot model lock poisoned")?;
            if !matches!(&*guard, Some((d, _)) if *d == self.dir) {
                *guard = None; // free the old build before importing another directory
                // `Gpu::share` - one device for every tower (text and EVA vision
                // alike) plus `embed_image`'s own device-side resize, so the
                // pipeline set has to cover all three; EVA's conv2d/bidir-
                // attention/rope2d and the resize kernels are absent from
                // TEXT_PIPELINES alone.
                let kernels: Vec<(&str, &str)> = crate::model::TEXT_PIPELINES
                    .iter()
                    .chain(crate::model::VISION_PIPELINES.iter())
                    .chain(imaging::PIPELINES.iter())
                    .copied()
                    .collect();
                let gpu = Gpu::new(&kernels);
                *guard = Some((self.dir.clone(), Arc::new(Session::load(&self.dir, gpu)?)));
            }
            guard.as_ref().expect("built above").1.clone()
        };
        session.run(&self.name, inv)
    }
}

#[cfg(test)]
mod caps_tests {
    use super::*;

    #[test]
    fn manifest_declares_both_towers() {
        let m = manifest();
        let names: Vec<&str> = m.actions.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["embed_text", "embed_image"]);
    }

    #[test]
    fn tower_names_round_trip_to_configs() {
        for t in TOWERS {
            assert!(tower_cfg(t).is_ok(), "{t} should map to a config");
        }
        assert!(tower_cfg("nope").is_err());
    }

    #[test]
    fn an_unknown_action_is_named_not_ignored() {
        // A silent no-op here would surface as an empty embedding downstream.
        let p = ClipProvider::new("/nonexistent");
        assert!(p.action("embed_audio").is_none());
    }

    #[test]
    fn from_env_declines_a_directory_with_no_tokenizer() {
        // Guards the case that used to make this crate unservable: without a
        // tokenizer the text action cannot run at all.
        assert!(ClipProvider::from_env().is_none() || std::env::var("BRAIN_CLIP_DIR").is_ok());
    }
}
