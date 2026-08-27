// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end FLUX.2 Klein: prompt → text conditioning → 4-step (klein) or
//! CFG (base) rectified-flow Euler → FLUX.2 VAE decode; image editing via
//! reference-image token concatenation (RoPE t-offsets 10·(i+1)).
//!
//! Weight locations come from env only (`Paths::from_env`) — never baked-in
//! paths. The text encoder runs the parity-proven masked-pad path
//! (`Qwen::encode_hiddens_padded`, layers 9/18/27 concatenated).

use crate::config::Flux2Config;
use crate::model::{position_ids, Flux2Model};
use data::Tokenizer;

/// Qwen3 hidden-state taps concatenated per token (also used by
/// [`crate::finetune`]'s standalone caption encoder).
pub const TAP_LAYERS: [usize; 3] = [9, 18, 27];
/// Right-pad token for the masked-pad text-encoder path.
pub const PAD_TOKEN: u32 = 151643;

/// Weight locations, env-only:
/// `BRAIN_FLUX2_DIT` (diffusers `transformer/` dir, BFL single-file
/// safetensors, or BF16 GGUF), `BRAIN_FLUX2_VAE` (diffusers `vae/` dir or
/// file), `BRAIN_FLUX2_TE` (HF text-encoder dir), `BRAIN_FLUX2_TOKENIZER`
/// (`tokenizer.json`).
pub struct Paths {
    pub dit: String,
    pub vae: String,
    pub te: String,
    pub tokenizer: String,
}

impl Paths {
    pub fn from_env() -> Result<Paths, String> {
        let get = |k: &str| std::env::var(k).map_err(|_| format!("{k} not set"));
        Ok(Paths {
            dit: get("BRAIN_FLUX2_DIT")?,
            vae: get("BRAIN_FLUX2_VAE")?,
            te: get("BRAIN_FLUX2_TE")?,
            tokenizer: get("BRAIN_FLUX2_TOKENIZER")?,
        })
    }
}

/// The pixel size at which each reference is encoded as **conditioning**, in
/// order; `None` for a reference that contributes nothing.
///
/// A supplied reference always conditions the model. `strength` decides how
/// much of the denoise starts from the init latent, not whether the DiT can
/// see the photograph - so under `strength < 1` the first reference does
/// double duty: it is the init latent *and* it is attended to. Because the
/// init role pins it to the output size, its conditioning copy is downscaled
/// by [`GenOpts::ref_cond_scale`]; reference tokens cost attention
/// quadratically, and a full-size copy of a same-size reference doubles the
/// image half of the joint sequence.
///
/// This is the ONE place the rule is written. The sizing entry point
/// ([`ref_tokens`]), the position-id builder and the denoise loop all read it,
/// so a pipeline cannot be sized for a sequence different from the one it is
/// handed.
pub fn cond_sizes(refs: &[(Vec<f32>, u32, u32)], opts: &GenOpts) -> Vec<Option<(u32, u32)>> {
    let init = opts.strength.is_some_and(|s| s < 1.0);
    refs.iter()
        .enumerate()
        .map(|(i, &(_, h, w))| {
            if i == 0 && init {
                init_cond_size(opts.ref_cond_scale, h, w)
            } else {
                Some((h, w))
            }
        })
        .collect()
}

/// Conditioning tokens `refs` actually contribute under `opts` -- what a
/// pipeline must be sized for, in latent tokens.
pub fn ref_tokens(refs: &[(Vec<f32>, u32, u32)], opts: &GenOpts) -> u32 {
    cond_sizes(refs, opts).into_iter().flatten().map(|(h, w)| (h / 16) * (w / 16)).sum()
}

/// A LoRA adapter to fold in before the model is built.
///
/// `path` selects the family by extension: a `.safetensors` is a third-party
/// (ai-toolkit / ComfyUI / diffusers) adapter over the fused matrices,
/// anything else is brain's own trained checkpoint container.
#[derive(Clone, Debug, PartialEq)]
pub struct AdapterSpec {
    pub path: String,
    /// ComfyUI's `strength_model`: multiplies the whole delta. 1.0 is the
    /// reference default. Meaningful for third-party adapters, whose files
    /// carry no alpha; brain's own adapters bake their scale into the
    /// checkpoint header and ignore this.
    pub scale: f32,
}

impl AdapterSpec {
    /// An adapter at the reference default strength.
    pub fn new(path: impl Into<String>) -> AdapterSpec {
        AdapterSpec { path: path.into(), scale: 1.0 }
    }
}

#[derive(Clone, Debug)]
pub struct GenOpts {
    pub width: u32,
    pub height: u32,
    /// Image-to-image init strength in `(0, 1]`. `None` (or 1.0) starts the
    /// denoise from pure noise, so the result keeps the composition but is a
    /// fresh generation (this is why a reference-only "colorize" reinterprets
    /// the scene). With `Some(s)` the first reference is VAE-encoded and the
    /// trajectory starts partway down the schedule from
    /// `x_σ = (1−σ)·x₀ + σ·ε` - the rectified-flow forward process - so
    /// structure is anchored to the source. Small `s` = faithful.
    ///
    /// This is how much of the denoise starts from the init latent, NOT
    /// whether the model can see the reference: the reference images
    /// condition the DiT through their tokens at **every** value, including
    /// under `strength`, where the first one is both the init latent and a
    /// conditioning input ([`GenOpts::ref_cond_scale`], [`cond_sizes`]).
    pub strength: Option<f32>,
    /// None → the variant default (4 distilled / 50 base).
    pub steps: Option<u32>,
    /// CFG scale — only meaningful for the undistilled base variants.
    pub guidance: f32,
    pub seed: u64,
    /// Spatial preservation mask over the output canvas: **white regenerates,
    /// black preserves** the first reference image, which must then be at the
    /// output size. Where `strength` decides how much of the source survives
    /// *everywhere*, this decides *where* it survives - after every Euler step
    /// the masked-out region is replaced by the source latent renoised to that
    /// step's sigma ([`crate::mask::blend`]), so it tracks the source exactly
    /// instead of being softly guided toward it.
    ///
    /// `None` - and an all-white mask - are bit-for-bit the unmasked
    /// behaviour.
    pub mask: Option<crate::mask::Mask>,
    /// Linear scale of the **conditioning copy** of the init reference, in
    /// `[0, 1]`. Only `refs[0]` under `strength < 1` is affected: that is the
    /// one reference whose resolution the caller cannot choose, because the
    /// init-latent role pins it to the output size. Every other reference
    /// conditions at whatever size it was supplied at, and `strength >= 1`
    /// (or `None`) consumes no init latent at all, so this dial does not
    /// apply there.
    ///
    /// `1.0` conditions at the full output size - the largest, most faithful
    /// and most expensive setting, and the one that makes `strength 0.999`
    /// cost exactly what `strength 1.0` costs. `0.0` switches the
    /// conditioning copy off, which is the explicit opt-in to the cheap
    /// behaviour where the reference reaches the model only through the
    /// init latent. The default is a downscale, because reference tokens are
    /// quadratic in the attention and a full-size copy of a same-size
    /// reference doubles the image half of the joint sequence.
    pub ref_cond_scale: f32,
}

/// Default [`GenOpts::ref_cond_scale`]: the conditioning copy of the init
/// reference is three quarters of its linear size, i.e. a bit over half its
/// tokens. Reference *resolution* is the architecture-preservation dial, so
/// this is a fidelity/cost trade and not an implementation detail; the value
/// is the one that produced the staging results this behaviour was built
/// for. Raise it with `--ref-cond-scale` when the card has room.
pub const DEFAULT_REF_COND_SCALE: f32 = 0.75;

impl Default for GenOpts {
    fn default() -> Self {
        GenOpts {
            width: 1024,
            height: 1024,
            strength: None,
            steps: None,
            guidance: 4.0,
            seed: 0,
            mask: None,
            ref_cond_scale: DEFAULT_REF_COND_SCALE,
        }
    }
}

/// Pixel size of the conditioning copy of an init reference that is `h x w`,
/// or `None` when `scale` switches conditioning off.
///
/// The result is floored to a multiple of 16 (one latent token) on each axis
/// independently, so the aspect ratio is preserved up to one token and a
/// non-square canvas is not a special case. `scale` is clamped to `[0, 1]`:
/// upscaling a reference past the size it was encoded at buys nothing the VAE
/// did not already throw away, and costs tokens quadratically.
pub fn init_cond_size(scale: f32, h: u32, w: u32) -> Option<(u32, u32)> {
    let s = scale.clamp(0.0, 1.0);
    let q = |d: u32| (((d as f32 * s) as u32) / 16 * 16).max(16);
    let (ch, cw) = (q(h), q(w));
    // Below one latent token in either axis there is nothing to condition on;
    // `q` floors at 16, so the off switch is the scale itself.
    (s > 0.0).then_some((ch, cw))
}

/// Bilinear resize of a reference image (`[-1,1]` CHW, the layout
/// [`Pipeline::generate`] takes) to `th x tw`.
///
/// Each channel plane is contiguous `[h, w]`, which is exactly a 1-channel
/// interleaved image, so this is the shared host resize applied three times
/// rather than a fourth resampler in the workspace.
fn resize_ref(chw: &[f32], h: u32, w: u32, th: u32, tw: u32) -> Vec<f32> {
    if (th, tw) == (h, w) {
        return chw.to_vec();
    }
    let plane = (h * w) as usize;
    let mut out = Vec::with_capacity(3 * (th * tw) as usize);
    for c in 0..3usize {
        out.extend(imaging::resize_bilinear_hwc(&chw[c * plane..(c + 1) * plane], 1, w, h, tw, th));
    }
    out
}

/// Read DiT weights from a diffusers `transformer/` dir, a BFL single-file
/// safetensors, or a BF16 GGUF, onto the canonical BFL names. Public so
/// [`crate::finetune`] loads the frozen base through the same importer.
pub fn read_dit_tensors(path: &str, cfg: &Flux2Config) -> Result<crate::Tensors, String> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        let mut files: Vec<_> = std::fs::read_dir(p)
            .map_err(|e| format!("{path}: {e}"))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|q| q.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(format!("no .safetensors under {path}"));
        }
        let mut ts = Vec::new();
        for f in files {
            ts.extend(checkpoint::safetensors::read(f.to_str().unwrap())?);
        }
        crate::import_diffusers(ts, cfg)
    } else if p.extension().is_some_and(|x| x == "gguf") {
        crate::import_bfl(checkpoint::gguf::read(path)?, cfg)
    } else {
        let ts = checkpoint::safetensors::read(path)?;
        // single-file releases use BFL names; a consolidated diffusers dump
        // would carry `transformer_blocks.` — probe and route
        if ts.iter().any(|t| t.name.starts_with("transformer_blocks.")) {
            crate::import_diffusers(ts, cfg)
        } else {
            crate::import_bfl(ts, cfg)
        }
    }
}

/// A ready-to-generate model: DiT + VAE + text encoder held together.
pub struct Pipeline {
    pub cfg: Flux2Config,
    model: Flux2Model,
    tok: data::qwen_tokenizer::QwenBpe,
    te: qwen3::Qwen,
    vae_cfg: vae::VaeConfig,
    vae_tensors: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>,
    bn_mean: Vec<f32>,
    bn_var: Vec<f32>,
}

impl Pipeline {
    /// Build for a maximum joint sequence (txt + generated + reference
    /// tokens). `n_img_max` in latent tokens, e.g. 4096 for 1024×1024.
    pub fn build(cfg: &Flux2Config, paths: &Paths, n_img_max: u32) -> Result<Pipeline, String> {
        Pipeline::build_adapted(cfg, paths, n_img_max, None)
    }

    /// [`Pipeline::build`] with an optional trained LoRA adapter
    /// ([`crate::finetune`] output) folded into the DiT tensors before the
    /// model is built — a plain generation run then produces
    /// adapter-conditioned images with no model change.
    pub fn build_adapted(cfg: &Flux2Config, paths: &Paths, n_img_max: u32, adapter: Option<&AdapterSpec>) -> Result<Pipeline, String> {
        Pipeline::build_with(cfg, paths, n_img_max, adapter, crate::Precision::F32)
    }

    /// [`Pipeline::build_adapted`] with a DiT numeric tier: `Precision::Int8`
    /// builds the DP4A DiT (~4x smaller than f32 - DiT + int8 TE fit ONE
    /// 24 GB card). A LoRA adapter (if any) is folded into the f32 tensors
    /// BEFORE quantization, so adapters work at either tier - the same order
    /// ComfyUI uses (patch the weights, then run).
    pub fn build_with(cfg: &Flux2Config, paths: &Paths, n_img_max: u32, adapter: Option<&AdapterSpec>, precision: crate::Precision) -> Result<Pipeline, String> {
        Pipeline::build_batched(cfg, paths, n_img_max, adapter, precision, 1)
    }

    /// The DiT half of a build: the weight source decision, any LoRA fold,
    /// and the model construction.
    ///
    /// A Q8_0 GGUF at the int8 tier never needs the fp32 model. The
    /// checkpoint already holds int8, and `DitWeights::Gguf` requantizes each
    /// matrix straight to this engine's per-row packing, one at a time;
    /// routing it through the fp32 map instead materializes the whole model
    /// (36.3 GB on klein-9b) purely as an intermediate, reads it back twice
    /// to quantize, and frees it again. The result is BIT-IDENTICAL either
    /// way - see `crate::weights` for why that is provable rather than
    /// approximate - so this is a pure cost decision, not a fidelity one.
    ///
    /// A third-party LoRA still needs a float domain, but per tensor rather
    /// than over a resident map, so it rides the same streamed path. brain's
    /// own adapter container does not: it folds through
    /// `LoraAdapter::fold_into_tensors`, which is written against the whole
    /// map. Everything else - safetensors, diffusers dirs, the fp32 tier, a
    /// GGUF whose tensors are not Q8_0 - takes the map route unchanged.
    fn build_dit(
        cfg: &Flux2Config,
        paths: &Paths,
        n_max: u32,
        adapter: Option<&AdapterSpec>,
        precision: crate::Precision,
        max_batch: u32,
        gpu: gpu_core::Gpu,
    ) -> Result<Flux2Model, String> {
        let external = adapter.filter(|a| a.path.ends_with(".safetensors"));
        // `BRAIN_FLUX2_NO_STREAM=1` forces the fp32-map route. Both produce
        // the same bytes, so this is not a correctness switch - it is what
        // lets the two be A/B'd on a real checkpoint (which is how the
        // byte-identity of a real 9B generation was checked, adapter and
        // all), and a valve if a checkpoint ever trips the streamed path.
        let no_stream = std::env::var("BRAIN_FLUX2_NO_STREAM").is_ok_and(|v| v != "0");
        let streamable = !no_stream
            && precision == crate::Precision::Int8
            && paths.dit.ends_with(".gguf")
            && adapter.is_none_or(|a| a.path.ends_with(".safetensors"));
        if streamable {
            let g = checkpoint::gguf::MmapGguf::open(&paths.dit)?;
            // Two-way coverage still has to hold, and it has to hold BEFORE
            // any weight is read: it is what catches a wrong checkpoint, and
            // skipping it because the load got cheaper would trade the one
            // check that matters for the saving.
            crate::import::validate_manifest(&|n| g.shape(n).map(<[usize]>::to_vec), g.names(), cfg)?;
            let lora = match external {
                Some(ap) => {
                    let l = crate::weights::PendingLora::open(&ap.path, ap.scale, &|n| g.shape(n).map(<[usize]>::to_vec))?;
                    let (pairs, rank, scale) = l.summary();
                    // Loud on success too: a run that claims to be adapted
                    // should say how much of the model it actually moved, so
                    // a silent no-op cannot hide behind a clean exit.
                    eprintln!("flux2: folded external LoRA {} - {pairs} linears, rank {rank}, strength {scale}", ap.path);
                    Some(l)
                }
                None => None,
            };
            let src = crate::weights::DitWeights::gguf_adapted(&g, lora.as_ref());
            return Ok(Flux2Model::new_from(cfg, &src, gpu, n_max, max_batch, precision));
        }

        let mut dit_ts = read_dit_tensors(&paths.dit, cfg)?;
        if let Some(ap) = adapter {
            // Two adapter families reach this point, told apart by extension:
            // a `.safetensors` is a THIRD-PARTY (ai-toolkit / ComfyUI) file
            // over the fused matrices, anything else is brain's own trained
            // checkpoint container. Both fold into the same f32 tensor map.
            if ap.path.ends_with(".safetensors") {
                let info = crate::lora::fold_external_adapter(&ap.path, &mut dit_ts, ap.scale)?;
                eprintln!(
                    "flux2: folded external LoRA {} - {} linears, rank {}, strength {}",
                    ap.path, info.pairs, info.rank, info.scale
                );
            } else {
                // The adapter's tensor shapes depend only on the architecture, not
                // the latent grid - any (lh, lw) loads it.
                let tcfg = crate::modelgrad::Cfg::from_flux2(cfg, 1, 1);
                let ad = crate::lora::load_adapter(&ap.path, &tcfg)?;
                ad.fold_into_tensors(&mut dit_ts)?;
                eprintln!("flux2: folded brain LoRA {} - rank {}", ap.path, ad.rank());
            }
        }
        let model = Flux2Model::new_batched(cfg, &dit_ts, gpu, n_max, max_batch, precision);
        drop(dit_ts);
        Ok(model)
    }

    /// [`Pipeline::build_with`] sized for up to `max_batch` concurrent
    /// generations sharing one denoise loop ([`Pipeline::generate_batch`]).
    /// Only the DiT activation scratch grows; the text encoder and VAE stay
    /// single-stream.
    pub fn build_batched(cfg: &Flux2Config, paths: &Paths, n_img_max: u32, adapter: Option<&AdapterSpec>, precision: crate::Precision, max_batch: u32) -> Result<Pipeline, String> {
        let n_max = cfg.txt_len as u32 + n_img_max;
        let gpu = gpu_core::Gpu::new(crate::model::KERNELS);
        let model = Self::build_dit(cfg, paths, n_max, adapter, precision, max_batch.max(1), gpu)?;

        let tok = data::qwen_tokenizer::QwenBpe::from_file(&paths.tokenizer)?;
        let te_cfg = if cfg.context_in_dim == 12288 {
            qwen3::QwenConfig::qwen3_8b()
        } else {
            qwen3::QwenConfig::qwen3_4b()
        };
        // Streamed, not slurped. `read_model_dir` + `brain_init_from_hf` built
        // the WHOLE encoder as an fp32 `HashMap` on the host - for Qwen3-8B
        // that is the largest host allocation the process makes, and it is
        // made only to be read once, tensor by tensor, into device buffers and
        // then dropped. A mapped `WeightReader` + `RemapSource` hands
        // `new_shard`/`new_shard_i8` the same bytes through the
        // `checkpoint::TensorSource` seam they already accept, so the host
        // holds one tensor at a time instead of the model.
        //
        // `BRAIN_FLUX2_TE_NO_STREAM=1` forces the old whole-map route. Not a
        // correctness switch - both produce the same bytes, pinned per tensor
        // in `qwen3` - but it is how the two are A/B'd on a real checkpoint,
        // and a valve if a configuration ever needs the map route back.
        let te_no_stream = std::env::var("BRAIN_FLUX2_TE_NO_STREAM").is_ok_and(|v| v != "0");
        let te_eager: Option<std::collections::HashMap<String, Vec<f32>>> = if te_no_stream {
            let ts = checkpoint::safetensors::read_model_dir(std::path::Path::new(&paths.te))?;
            Some(qwen3::import::brain_init_from_hf(ts, &te_cfg)?)
        } else {
            None
        };
        let te_reader = checkpoint::weightio::WeightReader::open_hf_dir(std::path::Path::new(&paths.te))
            .map_err(|e| format!("flux2: open text encoder {}: {e}", paths.te))?;
        // TE placement: default = ambient device; `BRAIN_FLUX2_TE_DEVICE=gpu<i>`
        // builds a truncated fp32 shard on that card, so the DiT can own the
        // other card whole. `Shard::owns` is `l < end`, so the shard keeps
        // layers `[0, deepest)` - the residual stream the deepest tap reads has
        // passed through exactly those, and the remaining layers, the final
        // norm and the LM head are never read. A `:i8` suffix (`gpu<i>:i8`)
        // uses the int8 (DP4A) shard instead, which is several times smaller.
        // The masked-pad kmask path is shared by both shard graphs, so parity
        // is unchanged (int8 is the lossy tier, gated in its own test).
        //
        // This is a TWO-CARD layout. Putting the encoder on the DiT's own card
        // is not a supported configuration at klein-9b/1024x768: measured, the
        // DiT plus VAE alone comes close to filling a 24 GB card, and even the
        // truncated int8 encoder is far too large to join it. See the FLUX.2
        // roadmap for the measured budgets. Today that combination fails as a
        // raw device out-of-memory rather than a refusal naming the two
        // budgets, which is a known gap recorded there.
        let deepest = *TAP_LAYERS.iter().max().unwrap();
        // `BRAIN_FLUX2_TE_DEVICE=gpu<i>[:i8]` is user input, parsed to a
        // canonical card index at this edge; the shard's gpu_index is what
        // places the build (device registry) — later device creation (VAE)
        // stays on the ambient card beside the DiT.
        let te = match std::env::var("BRAIN_FLUX2_TE_DEVICE").ok().as_deref() {
            Some(s) if s.starts_with("gpu") => {
                let (idx_s, te_i8) = match s[3..].strip_suffix(":i8") {
                    Some(p) => (p, true),
                    None => (&s[3..], false),
                };
                let idx: usize = idx_s.parse().map_err(|_| format!("bad BRAIN_FLUX2_TE_DEVICE {s} (gpu<i>[:i8])"))?;
                let shard = qwen3::Shard { start: 0, end: deepest, embed: true, head: false, gpu_index: idx };
                // Shard-aware coverage: this build reads the embedding and
                // layers `[0, deepest)` and nothing else, so the checkpoint is
                // required to carry exactly those. The layers past the tap,
                // the final norm and the LM head are neither read nor demanded
                // - previously they had to be present (and, on a checkpoint
                // still being fetched, downloaded) purely to be validated and
                // discarded. Narrowed, not weakened: a tensor that IS present
                // is still element-count checked, and a tensor mapping outside
                // the config's full parameter list is still a hard error.
                let streamed;
                let src: &dyn checkpoint::TensorSource = match &te_eager {
                    Some(m) => m,
                    None => {
                        streamed = qwen3::import::hf_shard_source(&te_reader, &te_cfg, &shard)?;
                        &streamed
                    }
                };
                if te_i8 {
                    qwen3::Qwen::new_shard_i8(te_cfg, 1, cfg.txt_len as u32, src, shard)
                } else {
                    qwen3::Qwen::new_shard(te_cfg, 1, cfg.txt_len as u32, src, false, shard)
                }
            }
            // No explicit placement: the whole encoder on the ambient device,
            // exactly what `Qwen::new` builds (whole shard, `train`, fp32) -
            // only the weights now arrive one tensor at a time. A whole shard
            // requires the whole `param_list()`, so the coverage check here is
            // identical to the one this path always ran.
            _ => {
                let shard = qwen3::Shard::whole(te_cfg.n_layers as usize);
                let streamed;
                let src: &dyn checkpoint::TensorSource = match &te_eager {
                    Some(m) => m,
                    None => {
                        streamed = qwen3::import::hf_shard_source(&te_reader, &te_cfg, &shard)?;
                        &streamed
                    }
                };
                qwen3::Qwen::new_shard(te_cfg, 1, cfg.txt_len as u32, src, true, shard)
            }
        };

        let vp = std::path::Path::new(&paths.vae);
        let (vae_file, vae_json) = if vp.is_dir() {
            (
                vp.join("diffusion_pytorch_model.safetensors"),
                std::fs::read_to_string(vp.join("config.json")).ok(),
            )
        } else {
            (vp.to_path_buf(), None)
        };
        let vae_cfg = match vae_json {
            Some(j) => vae::VaeConfig::from_json(
                &serde_json::from_str(&j).map_err(|e| e.to_string())?,
            ),
            None => vae::VaeConfig::flux2(),
        };
        let vae_ts = checkpoint::safetensors::read(vae_file.to_str().unwrap())?;
        let mut map = std::collections::HashMap::new();
        let (mut bn_mean, mut bn_var) = (Vec::new(), Vec::new());
        for t in vae_ts {
            if t.name == "bn.running_mean" {
                bn_mean = t.data.clone();
            }
            if t.name == "bn.running_var" {
                bn_var = t.data.clone();
            }
            map.insert(t.name, (t.shape, t.data));
        }
        if bn_mean.is_empty() || bn_var.is_empty() {
            return Err("vae checkpoint missing bn.running_{mean,var}".into());
        }

        Ok(Pipeline { cfg: cfg.clone(), model, tok, te, vae_cfg, vae_tensors: map, bn_mean, bn_var })
    }

    /// Prompt → `[txt_len, context_in_dim]` conditioning (masked-pad,
    /// layers 9/18/27 concatenated per token).
    pub fn encode_prompt(&self, prompt: &str) -> Vec<f32> {
        let templated = self.tok.apply_chat_template_no_think(&[("user", prompt)]);
        let mut ids = self.tok.encode(&templated);
        if ids.len() > self.cfg.txt_len {
            // Loud, not silent: the conditioning is computed from a PREFIX of
            // the user's prompt (audit F18).
            eprintln!("flux2: prompt is {} tokens but the model's text window is {} -- conditioning on the first {} tokens only", ids.len(), self.cfg.txt_len, self.cfg.txt_len);
        }
        ids.truncate(self.cfg.txt_len);
        let content = ids.len();
        ids.resize(self.cfg.txt_len, PAD_TOKEN);
        let taps = self.te.encode_hiddens_padded(&ids, content, &TAP_LAYERS);
        let d = taps[0].len() / self.cfg.txt_len;
        let mut ctx = Vec::with_capacity(self.cfg.txt_len * 3 * d);
        for row in 0..self.cfg.txt_len {
            for tap in &taps {
                ctx.extend_from_slice(&tap[row * d..(row + 1) * d]);
            }
        }
        ctx
    }

    /// VAE-encode an RGB image (`[-1,1]` CHW) to packed+normalized latent
    /// tokens `[lh*lw, 128]` (row-major, matching `position_ids`).
    pub fn encode_image(&self, chw: &[f32], h: u32, w: u32) -> Result<Vec<f32>, String> {
        let enc = vae::VaeEncoder::from_diffusers(self.vae_cfg.clone(), &self.vae_tensors, h, w, None);
        let (lh8, lw8) = ((h / 8) as usize, (w / 8) as usize);
        let mean = enc.encode_mean(chw, lh8 as u32, lw8 as u32);
        let eps = self.vae_cfg.batch_norm_eps;
        let packed = vae::latent::pack(&mean, 32, lh8, lw8, &self.bn_mean, &self.bn_var, eps);
        // [128, lh, lw] -> tokens [lh*lw, 128]
        let (lh, lw) = (lh8 / 2, lw8 / 2);
        let mut tokens = vec![0.0f32; lh * lw * 128];
        for c in 0..128 {
            for y in 0..lh {
                for x in 0..lw {
                    tokens[(y * lw + x) * 128 + c] = packed[(c * lh + y) * lw + x];
                }
            }
        }
        Ok(tokens)
    }

    /// Latent tokens `[lh*lw, 128]` → RGB u8 HWC.
    pub fn decode_tokens(&self, tokens: &[f32], lh: usize, lw: usize) -> Result<Vec<u8>, String> {
        // tokens -> [128, lh, lw]
        let mut packed = vec![0.0f32; 128 * lh * lw];
        for c in 0..128 {
            for y in 0..lh {
                for x in 0..lw {
                    packed[(c * lh + y) * lw + x] = tokens[(y * lw + x) * 128 + c];
                }
            }
        }
        let eps = self.vae_cfg.batch_norm_eps;
        let unpacked = vae::latent::unpack(&packed, 32, lh * 2, lw * 2, &self.bn_mean, &self.bn_var, eps);
        let (h, w) = ((lh * 16) as u32, (lw * 16) as u32);
        let dec = vae::VaeDecoder::from_diffusers(self.vae_cfg.clone(), &self.vae_tensors, (lh * 2) as u32, (lw * 2) as u32, None);
        let chw = dec.decode(&unpacked);
        // clamp FIRST, then rescale (reference order — reversed produces artifacts)
        let n = (h * w) as usize;
        let mut out = vec![0u8; n * 3];
        for c in 0..3 {
            for i in 0..n {
                let v = chw[c * n + i].clamp(-1.0, 1.0);
                out[i * 3 + c] = (127.5 * (v + 1.0)) as u8;
            }
        }
        Ok(out)
    }

    /// The largest batch [`Pipeline::generate_batch`] can put in one DiT
    /// forward (what the model's scratch was sized for at build time).
    pub fn max_batch(&self) -> u32 {
        self.model.max_batch()
    }

    /// Text-to-image (optionally with reference images for editing).
    /// `refs`: RGB `[-1,1]` CHW images, each with its (h, w) — pre-cropped to
    /// multiples of 16 (see [`ref_from_hwc`]). Returns (rgb8 HWC, width,
    /// height). `cancel` is polled once per denoise step (multi-minute CPU
    /// runs must be abortable); a `Default` token never fires.
    ///
    /// This is [`Pipeline::generate_batch`] with one request — the two share
    /// one denoise loop, so there is no second sampling implementation to
    /// drift.
    pub fn generate(
        &self,
        prompt: &str,
        refs: &[(Vec<f32>, u32, u32)],
        o: &GenOpts,
        cancel: &capability::CancelToken,
        mut progress: impl FnMut(u32, u32, &str),
    ) -> BatchOutcome {
        let req = BatchRequest { prompt: prompt.to_string(), refs: refs.to_vec(), opts: o.clone(), cancel: cancel.clone() };
        self.generate_batch(std::slice::from_ref(&req), &mut progress)
            .pop()
            .expect("one request in, one result out")
    }

    /// Generate `reqs.len()` images through ONE denoise loop: at every step the
    /// still-running requests are packed into a single batched DiT forward
    /// ([`Flux2Model::forward_batch`]).
    ///
    /// Per-request **seed, steps, guidance and prompt** are all honoured:
    ///
    /// * the seed only picks the initial latent, which is per-sample anyway;
    /// * different `steps` mean different sigma schedules, so at loop index `i`
    ///   two samples sit at *different timesteps* — which costs nothing because
    ///   modulation is a per-sample condition group. A request that runs out of
    ///   steps simply leaves the batch, which shrinks for the remainder;
    /// * CFG (undistilled `base` variants) enters as a **second slot** in the
    ///   same batch — the conditional and unconditional evaluations of one
    ///   request are two samples at the same timestep with different `ctx`,
    ///   which used to be two sequential forwards;
    /// * `cancel` is polled per request per step; a cancelled request leaves
    ///   the batch immediately with `Err("cancelled")` and the others continue.
    ///
    /// Requests whose **position ids** differ (a different reference-image
    /// layout at the same total token count) cannot share a slab, so they are
    /// partitioned into id-groups and the groups run one after another. The
    /// text encoder and the VAE stay per request (they are separate models with
    /// their own single-sequence graphs) — only the DiT, which is the whole
    /// denoise cost, batches.
    ///
    /// Results are returned in request order.
    pub fn generate_batch(
        &self,
        reqs: &[BatchRequest],
        progress: &mut dyn FnMut(u32, u32, &str),
    ) -> Vec<BatchOutcome> {
        generate_batch_on(self, reqs, progress)
    }
}

/// What the sampler needs from the models underneath it: the DiT forward, plus
/// the two codecs that bracket it.
///
/// The seam exists so the sampling logic - the sigma schedule, the img2img
/// init, the mask blending, the batching and the per-request cancellation - can
/// be exercised without a multi-gigabyte checkpoint on disk and a card to put
/// it on. [`Pipeline`] is the one production implementation; there is no second
/// sampler behind it to drift.
trait Denoiser {
    fn cfg(&self) -> &Flux2Config;
    fn encode_prompt(&self, prompt: &str) -> Vec<f32>;
    fn encode_image(&self, chw: &[f32], h: u32, w: u32) -> Result<Vec<f32>, String>;
    fn decode_tokens(&self, tokens: &[f32], lh: usize, lw: usize) -> Result<Vec<u8>, String>;
    fn max_batch(&self) -> u32;
    fn forward_batch(&self, samples: &[crate::model::Sample<'_>], ids: &[u32], n_pred: usize) -> Vec<Vec<f32>>;
}

impl Denoiser for Pipeline {
    fn cfg(&self) -> &Flux2Config {
        &self.cfg
    }
    fn encode_prompt(&self, prompt: &str) -> Vec<f32> {
        Pipeline::encode_prompt(self, prompt)
    }
    fn encode_image(&self, chw: &[f32], h: u32, w: u32) -> Result<Vec<f32>, String> {
        Pipeline::encode_image(self, chw, h, w)
    }
    fn decode_tokens(&self, tokens: &[f32], lh: usize, lw: usize) -> Result<Vec<u8>, String> {
        Pipeline::decode_tokens(self, tokens, lh, lw)
    }
    fn max_batch(&self) -> u32 {
        self.model.max_batch()
    }
    fn forward_batch(&self, samples: &[crate::model::Sample<'_>], ids: &[u32], n_pred: usize) -> Vec<Vec<f32>> {
        self.model.forward_batch(samples, ids, n_pred)
    }
}

/// [`Pipeline::generate_batch`] over any [`Denoiser`].
fn generate_batch_on<D: Denoiser>(
    d: &D,
    reqs: &[BatchRequest],
    progress: &mut dyn FnMut(u32, u32, &str),
) -> Vec<BatchOutcome> {
    let mut out: Vec<BatchOutcome> = (0..reqs.len()).map(|_| Err("not run".to_string())).collect();
    // Partition by position ids: one slab layout per group.
    let mut groups: Vec<(Vec<u32>, Vec<usize>)> = Vec::new();
    for (i, r) in reqs.iter().enumerate() {
        match plan_on(d, r) {
            Err(e) => out[i] = Err(e),
            Ok(ids) => match groups.iter_mut().find(|(g, _)| *g == ids) {
                Some((_, v)) => v.push(i),
                None => groups.push((ids, vec![i])),
            },
        }
    }
    for (ids, members) in groups {
        denoise_group_on(d, reqs, &ids, &members, &mut out, progress);
    }
    out
}

/// Validate one request and return its joint position ids (the key that
/// decides which requests can share a batched forward).
fn plan_on<D: Denoiser>(d: &D, r: &BatchRequest) -> Result<Vec<u32>, String> {
    let o = &r.opts;
    if !o.width.is_multiple_of(16) || !o.height.is_multiple_of(16) {
        return Err(format!("width/height must be multiples of 16 (got {}×{})", o.width, o.height));
    }
    let (lh, lw) = ((o.height / 16) as usize, (o.width / 16) as usize);
    // Keep in step with the token builder: a reference contributes position
    // ids at the size its CONDITIONING copy is encoded at, which under
    // `strength` is a downscale of the first reference rather than its own
    // dimensions.
    let ref_dims: Vec<(usize, usize)> = cond_sizes(&r.refs, o)
        .into_iter()
        .flatten()
        .map(|(rh, rw)| ((rh / 16) as usize, (rw / 16) as usize))
        .collect();
    Ok(position_ids(d.cfg().txt_len, lh, lw, &ref_dims))
}

/// The source content a masked lane preserves, in latent space.
///
/// Held per lane because the blend needs all three at every step: the source
/// latent, the *same* noise draw the init used (so the preserved region walks
/// the source's own forward trajectory rather than a fresh one each step), and
/// the mask resampled to this lane's latent grid.
struct Preserve {
    /// One weight per latent token, `n_gen` of them.
    mask: Vec<f32>,
    /// The source latent `x₀`, `n_gen * in_channels`.
    src: Vec<f32>,
    /// The lane's init noise `ε`, same layout.
    noise: Vec<f32>,
}

/// One id-group's shared denoise loop.
fn denoise_group_on<D: Denoiser>(
    d: &D,
    reqs: &[BatchRequest],
    ids: &[u32],
    members: &[usize],
    out: &mut [BatchOutcome],
    progress: &mut dyn FnMut(u32, u32, &str),
) {
    let cfg = d.cfg();
    // Per-member state; a member that fails to encode drops out here.
    struct Lane {
        idx: usize,
        lh: usize,
        lw: usize,
        n_gen: usize,
        steps: usize,
        guidance: f32,
        ctx: Vec<f32>,
        ctx_uncond: Option<Vec<f32>>,
        ref_tokens: Vec<f32>,
        sigmas: Vec<f32>,
        lat: Vec<f32>,
        /// First schedule index this lane runs; > 0 for img2img inits.
        start: usize,
        /// Set only when the request carries a mask; `None` leaves the
        /// trajectory bit-for-bit what it was before masking existed.
        preserve: Option<Preserve>,
    }
    let max_steps_hint = members.iter().map(|&i| reqs[i].steps_for(cfg.distilled)).max().unwrap_or(0) as u32;
    let mut lanes: Vec<Lane> = Vec::new();
    for &i in members {
        let r = &reqs[i];
        let o = &r.opts;
        let (lh, lw) = ((o.height / 16) as usize, (o.width / 16) as usize);
        let n_gen = lh * lw;
        let steps = r.steps_for(cfg.distilled);
        progress(0, max_steps_hint + 2, "encoding prompt");
        let ctx = d.encode_prompt(&r.prompt);
        let cf = !cfg.distilled && o.guidance > 1.0;
        let ctx_uncond = if cf { Some(d.encode_prompt("")) } else { None };
        let mut ref_tokens: Vec<f32> = Vec::new();
        let mut failed = None;
        // Every supplied reference conditions the model. Under `strength` the
        // first one does double duty - it is also the init latent below - and
        // is encoded a second time at its conditioning size, which is a
        // downscale of itself. That second encode is the price of the model
        // being able to SEE the reference at all: without it `strength`
        // silently turns off conditioning, and the reference reaches the
        // denoiser only as leftover signal in a partially-noised latent.
        //
        // A mask does not change this either way: it reads the source latent
        // for its preserved region but leaves the token budget alone.
        for ((chw, rh, rw), size) in r.refs.iter().zip(cond_sizes(&r.refs, o)) {
            let Some((ch, cw)) = size else { continue };
            progress(0, max_steps_hint + 2, "encoding reference");
            let small;
            let src = if (ch, cw) == (*rh, *rw) {
                chw
            } else {
                small = resize_ref(chw, *rh, *rw, ch, cw);
                &small
            };
            match d.encode_image(src, ch, cw) {
                Ok(t) => ref_tokens.extend(t),
                Err(e) => failed = Some(e),
            }
        }
        if let Some(e) = failed {
            out[i] = Err(e);
            continue;
        }
        let sigmas = diffusion::scheduler::klein_sigmas(steps, n_gen);
        let noise = model::hostmath::randn(n_gen * cfg.in_channels, o.seed);
        // The source latent `x₀`. `strength` needs it as the init, a mask
        // needs it as the preserved content at every step, and both need
        // it at the output size - so it is encoded ONCE here rather than
        // once per consumer.
        let img2img = o.strength.is_some_and(|s| s < 1.0);
        let want_src = img2img || o.mask.is_some();
        let src = if want_src {
            let why = match (img2img, o.mask.is_some()) {
                (true, true) => "strength/mask",
                (true, false) => "strength",
                _ => "mask",
            };
            let Some((chw, rh, rw)) = r.refs.first() else {
                out[i] = Err(format!("{why} needs a reference image"));
                continue;
            };
            if (*rh as usize / 16) * (*rw as usize / 16) != n_gen {
                out[i] = Err(format!(
                    "{why} needs the reference at the output size ({}x{}, got {rw}x{rh})",
                    o.width, o.height
                ));
                continue;
            }
            match d.encode_image(chw, *rh, *rw) {
                Ok(x0) => Some(x0),
                Err(e) => {
                    out[i] = Err(e);
                    continue;
                }
            }
        } else {
            None
        };
        // Resample the mask to THIS lane's latent grid; the clones are the
        // price of keeping the source and its noise draw alive for the
        // whole trajectory, and are only paid when a mask is present.
        let preserve = match (&o.mask, &src) {
            (Some(m), Some(x0)) => {
                Some(Preserve { mask: m.to_latent(lh, lw), src: x0.clone(), noise: noise.clone() })
            }
            _ => None,
        };
        // img2img: start partway down the schedule from the source latent.
        // `x_σ = (1−σ)·x₀ + σ·ε` is the same forward process the trainer
        // uses (`modelgrad::make_flow_batch`), so the model sees exactly
        // the distribution it was trained on at that σ.
        let (mut lat, start, sigmas) = if img2img {
            let st = o.strength.unwrap_or(1.0).clamp(1e-3, 1.0);
            // Do NOT slice the distilled schedule: `klein_sigmas` is
            // shifted so hard for few-step sampling that its lowest
            // non-zero entry is 0.56 at 8 steps (0.75 at 4) - there is
            // no low-noise entry point to start an img2img from, and
            // starting at 0.84 with 3 steps left resolves to noise.
            // The velocity field is defined at every σ, so integrate
            // the requested number of Euler steps over [strength, 0]
            // instead; `strength` IS the starting noise level.
            let sigmas: Vec<f32> =
                (0..=steps).map(|k| st * (1.0 - k as f32 / steps as f32)).collect();
            let x0 = src.as_ref().expect("img2img encodes the source above");
            let lat: Vec<f32> =
                x0.iter().zip(&noise).map(|(&a, &e)| (1.0 - st) * a + st * e).collect();
            (lat, 0usize, sigmas)
        } else {
            (noise, 0usize, sigmas)
        };
        // Seed the preserved region on the source's own trajectory before
        // the first forward, not just after each step: otherwise the model
        // spends step 1 looking at pure noise where the walls should be.
        if let Some(p) = &preserve {
            crate::mask::blend(&mut lat, &p.mask, &p.src, &p.noise, sigmas[start], cfg.in_channels);
        }
        lanes.push(Lane {
            idx: i,
            lh,
            lw,
            n_gen,
            steps,
            guidance: o.guidance,
            ctx,
            ctx_uncond,
            ref_tokens,
            sigmas,
            lat,
            start,
            preserve,
        });
    }
    if lanes.is_empty() {
        return;
    }
    let max_steps = lanes.iter().map(|l| l.steps).max().unwrap_or(0);
    let cap = d.max_batch() as usize;

    for i in 0..max_steps {
        // Cancellation is per request: a cancelled lane leaves the batch and
        // the others keep going (the scheduler handed us N independent jobs).
        lanes.retain(|l| {
            if reqs[l.idx].cancel.is_cancelled() {
                out[l.idx] = Err("cancelled".into());
                false
            } else {
                true
            }
        });
        let active: Vec<usize> =
            (0..lanes.len()).filter(|&k| i >= lanes[k].start && i < lanes[k].steps).collect();
        if active.is_empty() {
            break;
        }
        progress(i as u32 + 1, max_steps as u32 + 2, "denoising");

        // Build one slot per DiT evaluation: (lane, ctx, t). CFG adds the
        // unconditional pass as a second slot at the same timestep.
        let mut joints: Vec<Vec<f32>> = Vec::with_capacity(active.len());
        let mut slots: Vec<(usize, bool, f32)> = Vec::new(); // (active index, is_uncond, t)
        for (a, &k) in active.iter().enumerate() {
            let l = &lanes[k];
            let mut joint = Vec::with_capacity(l.lat.len() + l.ref_tokens.len());
            joint.extend_from_slice(&l.lat);
            joint.extend_from_slice(&l.ref_tokens);
            joints.push(joint);
            slots.push((a, false, l.sigmas[i]));
            if l.ctx_uncond.is_some() {
                slots.push((a, true, l.sigmas[i]));
            }
        }
        // One forward per chunk of at most `max_batch` slots.
        let mut preds: Vec<Vec<f32>> = Vec::with_capacity(slots.len());
        for chunk in slots.chunks(cap) {
            let samples: Vec<crate::model::Sample<'_>> = chunk
                .iter()
                .map(|&(a, unc, t)| {
                    let l = &lanes[active[a]];
                    let ctx = if unc { l.ctx_uncond.as_ref().unwrap() } else { &l.ctx };
                    crate::model::Sample { img_tokens: &joints[a], ctx, t }
                })
                .collect();
            preds.extend(d.forward_batch(&samples, ids, lanes[active[0]].n_gen));
        }
        // Fold CFG and take the Euler step, per lane.
        for (a, &k) in active.iter().enumerate() {
            let cond = slots.iter().position(|&(sa, unc, _)| sa == a && !unc).expect("cond slot");
            let pred: Vec<f32> = match slots.iter().position(|&(sa, unc, _)| sa == a && unc) {
                None => preds[cond].clone(),
                Some(u) => preds[cond].iter().zip(&preds[u]).map(|(&c, &un)| un + lanes[k].guidance * (c - un)).collect(),
            };
            let l = &mut lanes[k];
            let dt = l.sigmas[i + 1] - l.sigmas[i];
            for (x, v) in l.lat.iter_mut().zip(&pred) {
                *x += dt * v;
            }
            // Blended latent diffusion. Outside the mask the latent is not
            // *guided* toward the source, it IS the source renoised to the
            // sigma this step just landed on - so the preserved region is
            // re-anchored every step and reaches σ = 0 as the source
            // exactly, instead of drifting a little further with each
            // forward the way `strength` alone lets it.
            if let Some(p) = &l.preserve {
                crate::mask::blend(&mut l.lat, &p.mask, &p.src, &p.noise, l.sigmas[i + 1], cfg.in_channels);
            }
        }
    }

    progress(max_steps as u32 + 2, max_steps as u32 + 2, "decoding");
    for l in &lanes {
        let o = &reqs[l.idx].opts;
        out[l.idx] = d.decode_tokens(&l.lat, l.lh, l.lw).map(|rgb| (rgb, o.width, o.height));
    }
}


/// One generated image `(rgb8, width, height)`, or why it failed. Named because
/// it appears in the batch entry point, its per-group helper and the
/// single-image wrapper, which must not drift apart.
pub type BatchOutcome = Result<(Vec<u8>, u32, u32), String>;

/// One generation in a [`Pipeline::generate_batch`] call: everything
/// `Pipeline::generate` takes, owned, plus its cancellation token.
#[derive(Clone)]
pub struct BatchRequest {
    pub prompt: String,
    /// RGB `[-1,1]` CHW reference images with their (h, w), pre-cropped to /16.
    pub refs: Vec<(Vec<f32>, u32, u32)>,
    pub opts: GenOpts,
    /// Polled once per denoise step; a `Default` token never fires.
    pub cancel: capability::CancelToken,
}

impl BatchRequest {
    /// Resolved step count (`opts.steps` or the variant default).
    fn steps_for(&self, distilled: bool) -> usize {
        self.opts.steps.unwrap_or(if distilled { 4 } else { 50 }) as usize
    }
}

/// Convert an interleaved HWC RGB image in `[0,1]` (the shared
/// `capability::blob` wire format, also what the CLI's PPM loader produces) to
/// the reference-image layout [`Pipeline::generate`] expects: `[-1,1]` CHW,
/// **center-cropped** to multiples of 16. Returns `(chw, h, w)` with the
/// cropped dims — the ONE implementation shared by the CLI and the capability
/// provider.
pub fn ref_from_hwc(hwc: &[f32], w: u32, h: u32) -> Result<(Vec<f32>, u32, u32), String> {
    let (cw, ch) = (w - w % 16, h - h % 16);
    if cw == 0 || ch == 0 {
        return Err(format!("reference image {w}×{h} is smaller than 16×16"));
    }
    let (x0, y0) = (((w - cw) / 2) as usize, ((h - ch) / 2) as usize);
    let mut chw = vec![0.0f32; 3 * (cw * ch) as usize];
    for c in 0..3usize {
        for y in 0..ch as usize {
            for x in 0..cw as usize {
                let v = hwc[((y + y0) * w as usize + (x + x0)) * 3 + c];
                chw[(c * ch as usize + y) * cw as usize + x] = 2.0 * v - 1.0;
            }
        }
    }
    Ok((chw, ch, cw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(h: u32, w: u32) -> (Vec<f32>, u32, u32) {
        (Vec::new(), h, w)
    }

    /// A [`Denoiser`] with no checkpoint behind it, so the sampler itself can
    /// be gated: schedule, img2img init, mask blending, decode.
    ///
    /// * `encode_image` / `decode_tokens` are a real, lossy round trip - the
    ///   mean of each 16x16 pixel block per channel, broadcast back over the
    ///   block. Like the VAE it is many-to-one, so "reproduces the source"
    ///   has to be asserted against the round trip AND against the original
    ///   with a stated tolerance, exactly as it does on real weights.
    /// * `forward_batch` is the exact velocity field `v = (x − target)/σ`,
    ///   whose Euler solution is `x = target + C·σ`. One integration therefore
    ///   lands on `target` - a fixed, deterministic "generated image" that
    ///   depends on the prompt and not at all on the source, which is what
    ///   makes "this region was regenerated" unambiguous.
    ///
    /// Unlike the real VAE this decoder is block-local, so the pixel-level
    /// equalities below are exact. On real weights the same guarantees hold in
    /// *latent* space, and the decoder's receptive field smears them across
    /// the mask seam by a few pixels: preserved regions are exact latents, not
    /// exact pixels, within a few pixels of a mask boundary.
    struct Stub {
        cfg: Flux2Config,
        /// Every `(joint image sequence, position-id count)` the sampler
        /// handed to the DiT. What the model *attends to* is not observable
        /// from the returned image, so the gates below read it here.
        seen: std::cell::RefCell<Vec<(Vec<f32>, usize)>>,
    }

    impl Stub {
        fn new() -> Stub {
            Stub {
                cfg: Flux2Config { in_channels: 4, txt_len: 8, ..Flux2Config::klein_4b() },
                seen: Default::default(),
            }
        }
    }

    impl Denoiser for Stub {
        fn cfg(&self) -> &Flux2Config {
            &self.cfg
        }
        fn encode_prompt(&self, prompt: &str) -> Vec<f32> {
            let d = self.cfg.context_in_dim;
            (0..self.cfg.txt_len * d)
                .map(|i| ((i + prompt.len()) as f32 * 0.017).sin())
                .collect()
        }
        fn encode_image(&self, chw: &[f32], h: u32, w: u32) -> Result<Vec<f32>, String> {
            let (h, w) = (h as usize, w as usize);
            let (lh, lw) = (h / 16, w / 16);
            let ch = self.cfg.in_channels;
            let mut t = vec![0.0f32; lh * lw * ch];
            for c in 0..3.min(ch) {
                for y in 0..lh {
                    for x in 0..lw {
                        let mut s = 0.0f32;
                        for dy in 0..16 {
                            for dx in 0..16 {
                                s += chw[c * h * w + (y * 16 + dy) * w + x * 16 + dx];
                            }
                        }
                        t[(y * lw + x) * ch + c] = s / 256.0;
                    }
                }
            }
            Ok(t)
        }
        fn decode_tokens(&self, tokens: &[f32], lh: usize, lw: usize) -> Result<Vec<u8>, String> {
            let (h, w) = (lh * 16, lw * 16);
            let ch = self.cfg.in_channels;
            let mut out = vec![0u8; h * w * 3];
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let v = tokens[((y / 16) * lw + x / 16) * ch + c].clamp(-1.0, 1.0);
                        out[(y * w + x) * 3 + c] = (127.5 * (v + 1.0)) as u8;
                    }
                }
            }
            Ok(out)
        }
        fn max_batch(&self) -> u32 {
            4
        }
        fn forward_batch(&self, samples: &[crate::model::Sample<'_>], ids: &[u32], n_pred: usize) -> Vec<Vec<f32>> {
            let ch = self.cfg.in_channels;
            samples
                .iter()
                .map(|s| {
                    self.seen.borrow_mut().push((s.img_tokens.to_vec(), ids.len()));
                    // The conditioning tail shifts the target's phase. Without
                    // this the stub's output is blind to the reference tokens
                    // and a byte-identity gate over a rendered image could not
                    // see a conditioning change at all - which is the very
                    // thing being fenced. The summary is deliberately
                    // POSITION-WEIGHTED: a plain mean is almost invariant to
                    // resampling the same photograph, so it cannot tell a
                    // full-size conditioning copy from a downscaled one. It is
                    // still one scalar shared by every token, so the velocity
                    // reads only its own latent and the mask gates' exact
                    // equalities hold.
                    let tail = &s.img_tokens[n_pred * ch..];
                    let cond = tail
                        .iter()
                        .enumerate()
                        .map(|(j, &v)| v * (j as f32 * 0.37).sin())
                        .sum::<f32>()
                        / (tail.len().max(1)) as f32;
                    (0..n_pred * ch)
                        .map(|i| {
                            // A fixed "generated image", prompt- and
                            // conditioning-dependent, init-latent-independent.
                            let target = (i as f32 * 0.031 + s.ctx[0] + cond).sin() * 0.8;
                            (s.img_tokens[i] - target) / s.t.max(1e-6)
                        })
                        .collect()
                })
                .collect()
        }
    }

    /// A deterministic source photo: `[-1,1]` CHW, structured enough that a
    /// left half and a right half are visibly different.
    fn source(h: u32, w: u32) -> (Vec<f32>, u32, u32) {
        let (hu, wu) = (h as usize, w as usize);
        let mut chw = vec![0.0f32; 3 * hu * wu];
        for c in 0..3 {
            for y in 0..hu {
                for x in 0..wu {
                    // Low spatial frequency, like a real photograph relative to
                    // a 16x16 latent cell: the codec's block average is then a
                    // faithful round trip rather than a blur beyond
                    // recognition, which is what the VAE's own fidelity looks
                    // like and what the tolerances below are calibrated to.
                    chw[c * hu * wu + y * wu + x] =
                        ((x as f32 * 0.006 + y as f32 * 0.004 + c as f32).sin()) * 0.9;
                }
            }
        }
        (chw, h, w)
    }

    fn run(mask: Option<crate::mask::Mask>, w: u32, h: u32) -> Vec<u8> {
        let d = Stub::new();
        let opts = GenOpts {
            width: w,
            height: h,
            strength: Some(0.9),
            steps: Some(4),
            guidance: 4.0,
            seed: 11,
            mask,
            ref_cond_scale: DEFAULT_REF_COND_SCALE,
        };
        let req = BatchRequest {
            prompt: "a staged living room".into(),
            refs: vec![source(h, w)],
            opts,
            cancel: Default::default(),
        };
        generate_batch_on(&d, std::slice::from_ref(&req), &mut |_, _, _| {})
            .pop()
            .unwrap()
            .expect("stub generation")
            .0
    }

    fn solid(v: f32, w: u32, h: u32) -> crate::mask::Mask {
        crate::mask::Mask::new(vec![v; (w * h) as usize], w, h).unwrap()
    }

    /// **Gate 1 - masking is free when it is not used.** An all-white mask must
    /// be BIT-IDENTICAL to no mask at all. Anything weaker leaves every
    /// existing unmasked generation one rounding step away from its previous
    /// output, and this feature would be a silent regression for everyone who
    /// never asked for it.
    #[test]
    fn an_all_white_mask_is_bit_identical_to_no_mask() {
        let (w, h) = (128u32, 96u32); // 4:3, the aspect this was built for
        assert_eq!(run(None, w, h), run(Some(solid(1.0, w, h)), w, h));
    }

    /// **Gate 2 - an all-black mask reproduces the source.** Exactly, against
    /// the codec round trip (the best any latent-space edit can do), and within
    /// a stated tolerance against the original - asserted on rel_l2 as well as
    /// cosine, because cosine alone is scale-invariant and would pass a
    /// uniformly brightened image.
    #[test]
    fn an_all_black_mask_reproduces_the_source() {
        let (w, h) = (128u32, 96u32);
        let d = Stub::new();
        let (chw, sh, sw) = source(h, w);
        let round_trip = d
            .decode_tokens(&d.encode_image(&chw, sh, sw).unwrap(), (h / 16) as usize, (w / 16) as usize)
            .unwrap();
        let got = run(Some(solid(0.0, w, h)), w, h);
        assert_eq!(got, round_trip, "black must land on the source latent exactly");

        // ... and that round trip is genuinely the source, not a grey field.
        let n = (h * w) as usize;
        let mut src8 = vec![0u8; n * 3];
        for i in 0..n {
            for c in 0..3 {
                src8[i * 3 + c] = (127.5 * (chw[c * n + i].clamp(-1.0, 1.0) + 1.0)) as u8;
            }
        }
        let (cos, rel) = agreement(&got, &src8);
        assert!(cos > 0.99, "cosine {cos}");
        assert!(rel < 0.15, "rel_l2 {rel}");

        // A generation with no preservation at all must NOT pass that bar -
        // otherwise the gate above proves nothing about the mask.
        let (cos_free, rel_free) = agreement(&run(Some(solid(1.0, w, h)), w, h), &src8);
        assert!(cos_free < 0.99 || rel_free > 0.15, "free run: cosine {cos_free}, rel_l2 {rel_free}");
    }

    /// **Gate 3 - a mask is spatial.** With the left half white the right half
    /// must match the preserved baseline and the left half must not; and the
    /// mirror image must hold against the regenerated baseline. Both directions
    /// are asserted, because a mask that changes nothing and a mask that
    /// changes everything are both failures and each assertion alone catches
    /// only one of them.
    ///
    /// Neither baseline is itself produced by a mask: "preserved" is the codec
    /// round trip of the source and "regenerated" is a plain unmasked run. An
    /// earlier version of this test compared against all-black and all-white
    /// *mask* runs and was therefore blind to a global mask inversion, which
    /// flips the baselines in lockstep with the result.
    #[test]
    fn a_split_mask_regenerates_one_half_and_preserves_the_other() {
        let (w, h) = (128u32, 96u32);
        let mut v = vec![0.0f32; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..(w / 2) as usize {
                v[y * w as usize + x] = 1.0;
            }
        }
        let split = run(Some(crate::mask::Mask::new(v, w, h).unwrap()), w, h);
        let d = Stub::new();
        let (chw, sh, sw) = source(h, w);
        let kept = d
            .decode_tokens(&d.encode_image(&chw, sh, sw).unwrap(), (h / 16) as usize, (w / 16) as usize)
            .unwrap();
        let freed = run(None, w, h);

        let half = |img: &[u8], left: bool| -> Vec<u8> {
            let mut o = Vec::new();
            for y in 0..h as usize {
                let (a, b) = if left { (0, w as usize / 2) } else { (w as usize / 2, w as usize) };
                for x in a..b {
                    o.extend_from_slice(&img[(y * w as usize + x) * 3..][..3]);
                }
            }
            o
        };
        assert_eq!(half(&split, false), half(&kept, false), "the black half must be the source");
        assert_ne!(half(&split, true), half(&kept, true), "the white half must NOT be the source");
        assert_ne!(half(&split, false), half(&freed, false), "the black half must not be a free generation");
        // Gate 1, restricted to a region: the white half is untouched by the
        // blend, so it reproduces the unmasked run exactly. (Exact here because
        // this stub's velocity reads only its own token; with real attention
        // the white half still *sees* the preserved half, which is the point of
        // masking, so on real weights this is a resemblance, not an equality.)
        assert_eq!(half(&split, true), half(&freed, true), "the white half must be the unmasked generation");
    }

    /// Cosine and relative L2 between two u8 images, on the `[-1,1]` scale the
    /// latents live on.
    fn agreement(a: &[u8], b: &[u8]) -> (f32, f32) {
        let f = |x: &[u8]| -> Vec<f32> { x.iter().map(|&v| v as f32 / 127.5 - 1.0).collect() };
        let (a, b) = (f(a), f(b));
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let d2: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
        (dot / (na * nb).max(1e-12), d2 / nb.max(1e-12))
    }

    /// **Gate 4 - a supplied reference always conditions the model.**
    /// `strength` decides how much denoising starts from the init latent; it
    /// must NOT decide whether the DiT can see the photograph. Under
    /// `strength < 1` the first reference does double duty: it is the init
    /// latent AND it contributes conditioning tokens, at
    /// [`GenOpts::ref_cond_scale`] of its own size (the init role pins it to
    /// the output size, so it is the one reference whose conditioning
    /// resolution the caller cannot pick by choosing a file).
    #[test]
    fn a_strength_reference_still_contributes_conditioning_tokens() {
        let refs = vec![img(768, 1024), img(768, 1024)];
        let base = GenOpts { width: 1024, height: 768, ..GenOpts::default() };

        // No strength: every reference conditions at its own size.
        let no_str = GenOpts { strength: None, ..base.clone() };
        assert_eq!(ref_tokens(&refs, &no_str), 2 * 48 * 64);

        // With strength: the first is BOTH the init latent and conditioning,
        // downscaled by the default 0.75 (1024x768 -> 768x576 -> 48x36).
        let with_str = GenOpts { strength: Some(0.4), ..base.clone() };
        assert_eq!(ref_tokens(&refs, &with_str), 36 * 48 + 48 * 64);

        // A lone reference under strength conditions on ITSELF - this is the
        // whole point. It used to contribute nothing.
        assert_eq!(ref_tokens(&refs[..1], &with_str), 36 * 48);

        // The dial reaches both ends: 1.0 is the full-size conditioning copy
        // (exactly what strength 1.0 costs), 0.0 switches it off entirely -
        // the documented escape hatch back to the old, cheap behaviour.
        let full_cond = GenOpts { ref_cond_scale: 1.0, ..with_str.clone() };
        assert_eq!(ref_tokens(&refs[..1], &full_cond), 48 * 64);
        let off = GenOpts { ref_cond_scale: 0.0, ..with_str.clone() };
        assert_eq!(ref_tokens(&refs[..1], &off), 0);

        // strength == 1.0 consumes no init latent, so nothing is downscaled
        // and the dial does not apply. This is the path that already works.
        for scale in [0.0, 0.75, 1.0] {
            let full = GenOpts { strength: Some(1.0), ref_cond_scale: scale, ..base.clone() };
            assert_eq!(ref_tokens(&refs, &full), 2 * 48 * 64, "scale {scale}");
        }
    }

    /// **Gate 5 - a pipeline is sized for exactly the tokens it attends to.**
    /// The invariant that motivated the removed
    /// `the_strength_init_reference_contributes_no_conditioning_tokens`: the
    /// attention scratch is allocated from [`ref_tokens`], so if the denoise
    /// loop builds a joint sequence of any other length the graph is either
    /// too small (a wrong-shaped forward) or wastefully too large. Only the
    /// answer changed; the invariant did not. Asserted against what the DiT
    /// was actually handed, on both sides of the sizing decision.
    #[test]
    fn the_joint_sequence_is_exactly_what_the_pipeline_was_sized_for() {
        let (w, h) = (128u32, 96u32);
        let refs = [source(h, w), source(h, w)];
        let base = GenOpts { width: w, height: h, steps: Some(2), seed: 5, ..GenOpts::default() };
        let cases = [
            GenOpts { strength: None, ..base.clone() },
            GenOpts { strength: Some(0.4), ..base.clone() },
            GenOpts { strength: Some(0.4), ref_cond_scale: 1.0, ..base.clone() },
            GenOpts { strength: Some(0.4), ref_cond_scale: 0.0, ..base.clone() },
            GenOpts { strength: Some(1.0), ..base.clone() },
        ];
        for (n, opts) in cases.iter().enumerate() {
            for k in 1..=refs.len() {
                let d = Stub::new();
                let req = BatchRequest {
                    prompt: "a staged living room".into(),
                    refs: refs[..k].to_vec(),
                    opts: opts.clone(),
                    cancel: Default::default(),
                };
                generate_batch_on(&d, std::slice::from_ref(&req), &mut |_, _, _| {})
                    .pop()
                    .unwrap()
                    .expect("stub generation");
                let n_gen = ((h / 16) * (w / 16)) as usize;
                let want = n_gen + ref_tokens(&refs[..k], opts) as usize;
                let seen = d.seen.borrow();
                assert!(!seen.is_empty(), "case {n}/{k}: no forward ran");
                for (joint, n_ids) in seen.iter() {
                    assert_eq!(joint.len(), want * d.cfg.in_channels, "case {n}/{k}: joint tokens");
                    assert_eq!(*n_ids, 4 * (d.cfg.txt_len + want), "case {n}/{k}: position ids");
                }
            }
        }
    }

    /// **Gate 6 - the model actually receives the photograph.** Gate 5 pins
    /// the *length* of the joint sequence; a pipeline that padded it with
    /// zeros would pass. This pins the *content*: the tail of what the DiT
    /// attends to is the encoding of the reference, downscaled by the
    /// conditioning dial. Under the old behaviour that tail was empty - which
    /// is exactly the defect: at `--strength 0.95` the DiT never saw the
    /// user's photograph at all, and only the leftover signal in a
    /// partially-noised init latent stood between the result and a fresh
    /// generation.
    #[test]
    fn the_denoiser_attends_to_the_downscaled_init_reference() {
        let (w, h) = (128u32, 96u32);
        let d = Stub::new();
        let src = source(h, w);
        let opts = GenOpts {
            width: w,
            height: h,
            strength: Some(0.95),
            steps: Some(2),
            seed: 3,
            ..GenOpts::default()
        };
        let req = BatchRequest {
            prompt: "a staged living room".into(),
            refs: vec![src.clone()],
            opts: opts.clone(),
            cancel: Default::default(),
        };
        generate_batch_on(&d, std::slice::from_ref(&req), &mut |_, _, _| {})
            .pop()
            .unwrap()
            .expect("stub generation");

        let ch = d.cfg.in_channels;
        let n_gen = ((h / 16) * (w / 16)) as usize;
        let seen = d.seen.borrow();
        let (joint, _) = seen.first().expect("at least one forward");
        let tail = &joint[n_gen * ch..];
        assert!(
            !tail.is_empty(),
            "a supplied reference must be attended to, not merely renoised into the init latent"
        );

        // 96x128 at the default 0.75 -> 72x96 floored to /16 -> 64x96 -> 4x6.
        let (ch_px, cw_px) = init_cond_size(opts.ref_cond_scale, h, w).expect("dial is on");
        assert_eq!((ch_px, cw_px), (64, 96));
        let small = resize_ref(&src.0, h, w, ch_px, cw_px);
        let want = d.encode_image(&small, ch_px, cw_px).expect("stub encode");
        assert_eq!(tail, &want[..], "the conditioning tail must BE the reference");
    }

    /// **Gate 7 - `strength == 1.0` is byte-for-byte what it always was.**
    /// The digest below was taken on the code as it stood *before* the
    /// conditioning change, so it is a genuine before/after fence on the one
    /// path users already depend on. If a future edit to the reference
    /// pipeline moves this, it moved a rendered image, not an abstraction.
    #[test]
    fn a_strength_one_run_is_byte_identical_to_the_pre_change_output() {
        let (w, h) = (128u32, 96u32);
        let refs = vec![source(h, w), source(h, w)];
        let d = Stub::new();
        let opts = GenOpts {
            width: w,
            height: h,
            strength: Some(1.0),
            steps: Some(4),
            seed: 11,
            ..GenOpts::default()
        };
        let req = BatchRequest {
            prompt: "a staged living room".into(),
            refs,
            opts,
            cancel: Default::default(),
        };
        let rgb = generate_batch_on(&d, std::slice::from_ref(&req), &mut |_, _, _| {})
            .pop()
            .unwrap()
            .expect("stub generation")
            .0;
        assert_eq!(fnv1a(&rgb), 0x0d96_f927_7211_6425u64);
    }

    /// FNV-1a 64 over the rendered bytes. A whole reference image is too large
    /// to inline and a tolerance would defeat the purpose, so the fence is a
    /// digest - written here rather than pulled in as a dependency because a
    /// hash of 36 kB in a unit test needs nothing stronger.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}
