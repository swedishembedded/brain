// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Inkling 2 image path: processor configuration, the vision tower over brain's shared ViT blocks, and the image
//! features spliced over a prompt's placeholder tokens.
//!
//! The pieces, in the order an image meets them:
//!
//! 1. [`ImageProcessorSpec`] is `processor_config.json` read as hostile input and held to the one recipe this tower
//!    was trained with; [`ImageProcessorSpec::tiling`] is `imaging`'s pad-and-tile preprocessor configured with it:
//!    40-pixel regions padded with -1 on the 0-255 scale, RGBA composited onto an 8-pixel 204/153 checkerboard, mean and
//!    std 0.5, no resize. One region is one language-model token.
//! 2. [`VisionWeights`] reads the tower's tensors from a checkpoint (`checkpoint::packed::PackedSource`, so bf16 and any
//!    per-tensor declaration are applied once, in the checkpoint layer) and checks each against the name map.
//! 3. [`VisionTower`] runs the tower. Every 2x2 block of 20-pixel patches is one region: a bias-free patch embedding,
//!    a learned 32x32 position table resampled bilinearly onto the image's patch grid (`model::postable`), one class
//!    token in front of every image, `ln_pre`, then 23 pre-LayerNorm blocks (LayerNorm with bias, layer scale, axial
//!    2-D RoPE over interleaved channel pairs) with attention block-diagonal per image, and last the merger: the four
//!    patch states of a region concatenated, `proj_in`, GELU, `proj_out` to the decoder's width. The blocks are
//!    `model::vit`'s; the RoPE is its `rope2d` slot filled with `rope2d_pairs`.
//! 4. [`expand_placeholders`] and [`splice_embeds`] map the features onto a rendered prompt exactly as the reference
//!    processor and `get_placeholder_mask` do: the chat template renders ONE placeholder token per image, the
//!    processor expands it to one per region, and the features of all images, in image order, fill the placeholder
//!    positions in order. [`VisionTower::embed_prompt`] runs the whole chain for encoded image files.
//!
//! The projector is the shared `model::projector::MlpProjector`; the published merger has no biases, so zero biases
//! are fed (an exact no-op) and the checkpoint is held to having none. The tower is inference-only here: the shared
//! blocks have their backward (`vit_block_fwd_cached` / `vit_block_bwd`, the `rope2d` slot with sign -1 is the same
//! kernel), and what a training tower would still need is recorded in the roadmap ledger.

use std::collections::HashMap;
use std::sync::Mutex;

use checkpoint::packed::PackedSource;
use gpu_core::{DeviceBuffer, Gpu, Step};
use imaging::{Checkerboard, RegionImage, RegionTiling};
use model::block::LayerNormIds;
use model::postable::{bilinear_grid_plan, DeviceTaps, PosTableKernels};
use model::projector::{MlpProjector, ProjectorConfig, PROJECTOR_PIPELINES};
use model::serve::PromptEmbeds;
use model::vit::{
    attn_chunk_for_device, gather_rows, linear_rows_step, row_index_buffer, scatter_rows, vit_block_fwd, VitBlockWeights, VitKernelIds, VitPermuteIds, VitScratch, VitShape,
    RopeTables, UNREGISTERED,
};
use paramstore::{ParamStore, Role};
use serde::Deserialize;

use crate::config::VisionConfig;
use crate::names::{vision_manifest_of, VISION_PREFIX};

mod train;

pub use train::{train_pipelines, LoraTraining, VisionPass};

/// The default id of the image placeholder token (`<|unused_200054|>`), the reference configuration's
/// `image_token_id`. Callers that hold the tokenizer should pass the id they resolve from the processor's
/// `image_token` instead.
pub const IMAGE_TOKEN_ID: u32 = 200_054;

/// The base of the tower's rotary frequencies. `vision_config` does not carry it; it is the reference configuration's
/// `rope_parameters` default for the axial rotary embedding.
pub const ROPE_THETA: f32 = 10_000.0;

/// Longest side, in pixels, of an accepted image: four thousand and ninety-six pixels, which makes 103 regions a side.
pub const MAX_SIDE: u32 = 4096;

/// The most sequence rows (class tokens and patches) one tower dispatch group holds. A larger batch of images is split
/// into groups at image boundaries, which changes nothing because attention is block-diagonal per image; one
/// maximum-size image is about 42 thousand rows.
pub const MAX_GROUP_ROWS: usize = 1 << 16;

/// The 8-pixel checkerboard transparency is composited onto, in the reference processor's code (not its config file).
const CHECKERBOARD: Checkerboard = Checkerboard { cell: 8, light: 204, dark: 153 };
/// The value padding takes on the 0-255 scale, in the reference processor's code (not its config file).
const PAD_FILL: f32 = -1.0;

// ---------------------------------------------------------------------------------------------------------------
// processor_config.json
// ---------------------------------------------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProcessorConfig {
    image_processor: RawImageProcessor,
    image_token: String,
    processor_class: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawImageProcessor {
    do_convert_rgb: bool,
    do_normalize: bool,
    do_rescale: bool,
    image_mean: [f32; 3],
    image_processor_type: String,
    image_std: [f32; 3],
    rescale_factor: f64,
    size: RawSize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSize {
    height: u32,
    width: u32,
}

/// `processor_config.json` held to the recipe the tower was trained with.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageProcessorSpec {
    pub region: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale_factor: f64,
    /// The placeholder token's text (`<|unused_200054|>`), for resolving its id in the tokenizer.
    pub image_token: String,
}

impl ImageProcessorSpec {
    /// Parse `processor_config.json`. Unknown fields, a processor that is not the tile-per-token one, a disabled
    /// conversion, rescale or normalisation, a non-square region or a non-positive scale are refused by name: each
    /// would load and silently change what the tower sees.
    pub fn from_json(text: &str) -> Result<ImageProcessorSpec, String> {
        let raw: RawProcessorConfig = serde_json::from_str(text).map_err(|e| format!("processor_config.json: {e}"))?;
        let p = &raw.image_processor;
        if raw.processor_class != "Inkling2Processor" || p.image_processor_type != "Inkling2ImageProcessor" {
            return Err(format!("processor_config.json: {:?} / {:?} is not the Inkling 2 processor", raw.processor_class, p.image_processor_type));
        }
        for (field, on) in [("do_convert_rgb", p.do_convert_rgb), ("do_normalize", p.do_normalize), ("do_rescale", p.do_rescale)] {
            if !on {
                return Err(format!("processor_config.json: image_processor.{field} is false; the tower was trained with it on"));
            }
        }
        if p.size.height != p.size.width || p.size.height == 0 {
            return Err(format!("processor_config.json: image_processor.size {}x{} is not a positive square", p.size.height, p.size.width));
        }
        if !(p.rescale_factor.is_finite() && p.rescale_factor > 0.0) || p.image_std.iter().any(|&s| !(s.is_finite() && s > 0.0)) || p.image_mean.iter().any(|m| !m.is_finite()) {
            return Err("processor_config.json: image_mean, image_std and rescale_factor must be finite, std and rescale positive".to_string());
        }
        Ok(ImageProcessorSpec { region: p.size.height, mean: p.image_mean, std: p.image_std, rescale_factor: p.rescale_factor, image_token: raw.image_token })
    }

    /// The preprocessor for `vision`: its region must be the tower's (patch size times merge size), since one region
    /// is exactly one token of `vision.adapter_merge_size` squared patches.
    pub fn tiling(&self, vision: &VisionConfig) -> Result<RegionTiling, String> {
        let want = vision.native_patch_size * vision.adapter_merge_size;
        if self.region != want {
            return Err(format!("processor region {} differs from the tower's patch size {} times merge size {}", self.region, vision.native_patch_size, vision.adapter_merge_size));
        }
        Ok(RegionTiling {
            region: self.region,
            checkerboard: CHECKERBOARD,
            pad_fill: PAD_FILL,
            rescale_factor: self.rescale_factor,
            mean: self.mean,
            std: self.std,
            max_side: MAX_SIDE,
        })
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------------------------------------------

/// The tower's tensors as host f32, keyed by their published names (`model.visual.*`).
pub struct VisionWeights {
    tensors: HashMap<String, Vec<f32>>,
}

impl VisionWeights {
    /// Read every tensor of the name map of `cfg` from `src`, checking shape and that nothing is missing. A tensor the
    /// map does not list under the tower's prefix is refused too: a published merger bias, say, would otherwise be
    /// dropped while the tower ran without it.
    pub fn load(src: &PackedSource, cfg: &VisionConfig) -> Result<VisionWeights, String> {
        let manifest = vision_manifest_of(cfg);
        let known: std::collections::HashSet<&str> = manifest.iter().map(|e| e.name.as_str()).collect();
        if let Some(extra) = src.names().into_iter().find(|n| n.starts_with(VISION_PREFIX) && !known.contains(n)) {
            return Err(format!("inkling2 vision: checkpoint tensor {extra} is not in the name map of the vision config"));
        }
        let mut tensors = HashMap::with_capacity(manifest.len());
        for e in &manifest {
            let desc = src.descriptor(&e.name).ok_or_else(|| format!("inkling2 vision: missing tensor {}", e.name))?;
            if desc.shape != e.shape {
                return Err(format!("inkling2 vision: {} has shape {:?}, the config makes it {:?}", e.name, desc.shape, e.shape));
            }
            let numel: u64 = e.shape.iter().product();
            let data = src.read_range(&e.name, 0, numel)?.ok_or_else(|| format!("inkling2 vision: could not read tensor {}", e.name))?;
            if data.iter().any(|v| !v.is_finite()) {
                return Err(format!("inkling2 vision: tensor {} holds a non-finite value", e.name));
            }
            tensors.insert(e.name.clone(), data);
        }
        Ok(VisionWeights { tensors })
    }

    /// Weights from tensors already in memory, held to the same name map as [`Self::load`]: every tensor of `cfg`
    /// present at its element count and finite, nothing else.
    pub fn from_tensors(cfg: &VisionConfig, tensors: HashMap<String, Vec<f32>>) -> Result<VisionWeights, String> {
        let manifest = vision_manifest_of(cfg);
        for e in &manifest {
            let data = tensors.get(&e.name).ok_or_else(|| format!("inkling2 vision: missing tensor {}", e.name))?;
            let numel: u64 = e.shape.iter().product();
            if data.len() as u64 != numel {
                return Err(format!("inkling2 vision: {} holds {} values, the config makes it {:?}", e.name, data.len(), e.shape));
            }
            if data.iter().any(|v| !v.is_finite()) {
                return Err(format!("inkling2 vision: tensor {} holds a non-finite value", e.name));
            }
        }
        if let Some(extra) = tensors.keys().find(|n| !manifest.iter().any(|e| &e.name == *n)) {
            return Err(format!("inkling2 vision: tensor {extra} is not in the name map of the vision config"));
        }
        Ok(VisionWeights { tensors })
    }

    fn get(&self, name: &str) -> &[f32] {
        &self.tensors[name]
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------------------------------------------

/// Every kernel the tower dispatches, for the device it runs on to register. The fused attention and register-tiled
/// GEMM rows are optional speed-ups the shared blocks resolve by name.
pub fn pipelines() -> &'static [(&'static str, &'static str)] {
    static PIPELINES: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    PIPELINES.get_or_init(build_pipelines)
}

fn build_pipelines() -> Vec<(&'static str, &'static str)> {
    let mut p = vec![
        ("layernorm", kernels::LAYERNORM),
        ("matmul", kernels::MATMUL),
        ("matmul_rows", kernels::MATMUL_ROWS),
        ("bias_add", kernels::BIAS_ADD),
        ("gelu_erf", kernels::GELU_ERF),
        ("scale_chan", kernels::SCALE_CHAN),
        ("add2", kernels::ADD2),
        ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
        ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
        ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
        ("kv_k_headt", kernels::KV_K_HEADT),
        ("attn_scores_cross_kt", kernels::ATTN_SCORES_CROSS_KT),
        ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
        ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT),
        ("flash_attn_bidir_reg", kernels::FLASH_ATTN_BIDIR_REG),
        ("flash_attn_bidir_reg2", kernels::FLASH_ATTN_BIDIR_REG2),
        ("matmul_reg3", kernels::MATMUL_REG3),
        ("region_copy", kernels::REGION_COPY),
        ("embed", kernels::EMBED),
        ("row_scatter", kernels::ROW_SCATTER),
        ("rope2d_pairs", kernels::ROPE2D_PAIRS),
        ("rows_gather_weighted", kernels::ROWS_GATHER_WEIGHTED),
        ("rows_gather_weighted_bwd", kernels::ROWS_GATHER_WEIGHTED_BWD),
    ];
    for &(name, src) in PROJECTOR_PIPELINES {
        if !p.iter().any(|&(n, _)| n == name) {
            p.push((name, src));
        }
    }
    p
}

// ---------------------------------------------------------------------------------------------------------------
// The tower
// ---------------------------------------------------------------------------------------------------------------

/// The features of a group of images: one row of the decoder's width per region, the images' rows concatenated in the
/// order given.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageFeatures {
    width: usize,
    tokens: Vec<usize>,
    rows: Vec<f32>,
}

impl ImageFeatures {
    /// Features from their parts: `tokens[i]` rows of `width` floats for image `i`, concatenated in `rows`.
    pub fn new(width: usize, tokens: Vec<usize>, rows: Vec<f32>) -> Result<ImageFeatures, String> {
        let need = width * tokens.iter().sum::<usize>();
        if width == 0 || rows.len() != need {
            return Err(format!("image features: {} floats for {} region(s) of width {width} (need {need})", rows.len(), tokens.iter().sum::<usize>()));
        }
        Ok(ImageFeatures { width, tokens, rows })
    }

    /// The decoder width of a row.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Regions (rows) per image.
    pub fn tokens_per_image(&self) -> &[usize] {
        &self.tokens
    }

    /// All rows, `sum(tokens) * width` floats, images in order.
    pub fn rows(&self) -> &[f32] {
        &self.rows
    }

    /// Image `i`'s rows.
    pub fn image(&self, i: usize) -> &[f32] {
        let start: usize = self.tokens[..i].iter().sum();
        &self.rows[start * self.width..(start + self.tokens[i]) * self.width]
    }
}

/// Intermediate activations of one group, for parity work: every tensor is the reference's, `[sequence rows, width]`
/// unless noted, the sequence being image 0's class token and patches, then image 1's, and so on.
#[derive(Clone, Debug, Default)]
pub struct VisionTrace {
    /// After the embeddings (class tokens in, position table added), before `ln_pre`.
    pub embeddings: Vec<f32>,
    pub pre_layernorm: Vec<f32>,
    /// The output of layer `i`, for each requested layer (ascending).
    pub layers: Vec<(usize, Vec<f32>)>,
    pub last_hidden: Vec<f32>,
    /// The patch states regrouped for the merger, `[regions, merge^2 * width]`.
    pub merger_in: Vec<f32>,
}

/// The patch grid, sequence layout and tables of a group of images.
struct GroupPlan {
    /// Patch grids `(height, width)` of every image, in patches.
    grids: Vec<(u32, u32)>,
    /// First sequence row of every image (its class token).
    starts: Vec<u32>,
    /// Sequence row of every patch, image by image, merge-block order.
    patch_rows: Vec<u32>,
    rows: u32,
    patches: u32,
}

/// The device inputs of one group of images.
struct GroupSetup {
    plan: GroupPlan,
    sh: VitShape,
    /// The `[patches, patch_size^2 * 3]` patch matrix.
    pix: DeviceBuffer,
    taps: DeviceTaps,
    /// One zero per image: gathering the single class row through it repeats it per image.
    zero_idx: DeviceBuffer,
    patch_idx: DeviceBuffer,
    cls_idx: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    spans: Vec<(u32, u32)>,
    max_span: u32,
}

/// Regions the resident projector has room for until a larger group makes it grow.
pub const PROJECTOR_REGIONS: u32 = 1024;

/// The store name of a backbone tensor.
fn backbone(tail: &str) -> String {
    format!("{VISION_PREFIX}backbone.{tail}")
}

/// The store name of a tensor of layer `l`.
fn layer_param(l: u32, tail: &str) -> String {
    backbone(&format!("layers.{l}.{tail}"))
}

/// The store name of the fused `[3C, C]` query/key/value weight (`.weight`) or `[3C]` bias (`.bias`) of layer `l`. The
/// published file keeps the three projections apart; the shared block reads one fused tensor, so the store does too.
/// An adapter still addresses the three by their published names (see [`crate::vision::LoraTraining`]).
pub fn fused_qkv_name(l: u32, kind: &str) -> String {
    layer_param(l, &format!("self_attn.qkv_proj.{kind}"))
}

/// The vision tower on a device.
pub struct VisionTower {
    cfg: VisionConfig,
    ids: VitKernelIds,
    pos_kernels: PosTableKernels,
    permute: VitPermuteIds,
    region_copy: usize,
    /// Every tower tensor, in the published naming except the fused attention input (see [`fused_qkv_name`]). The
    /// merger's two weights live here too and the projector shares their buffers.
    store: ParamStore,
    /// The resident merger: its weights are the store's, its activations have room for a group's regions and grow when
    /// a larger group arrives (the weights are never uploaded again).
    projector: Mutex<MlpProjector>,
    /// What training adds: backward kernels, the saved-activation policy, gradient sinks and the adapters.
    train: Option<train::TowerTraining>,
}

impl VisionTower {
    /// Upload the tower for inference. `gpu` must register [`pipelines`] (the rows the shared blocks do not strictly
    /// need may be absent); a missing required kernel is an error naming it.
    pub fn new(gpu: &Gpu, cfg: &VisionConfig, w: &VisionWeights) -> Result<VisionTower, String> {
        Self::assemble(gpu, cfg, w, &|_| false, None, false)
    }

    /// The store entries in a fixed order, with their sizes.
    fn param_list(cfg: &VisionConfig) -> Vec<(String, usize)> {
        let (c, ff) = (cfg.hidden_size as usize, cfg.mlp_width() as usize);
        let patch_dim = 3 * (cfg.native_patch_size as usize).pow(2);
        let merged = (cfg.adapter_merge_size * cfg.adapter_merge_size * cfg.hidden_size) as usize;
        let mut out = vec![
            (backbone("embeddings.class_embedding"), c),
            (backbone("embeddings.cls_position_embedding"), c),
            (backbone("embeddings.patch_embedding.weight"), c * patch_dim),
            (backbone("embeddings.position_embedding"), (cfg.posemb_grid_size * cfg.posemb_grid_size) as usize * c),
            (backbone("ln_pre.weight"), c),
            (backbone("ln_pre.bias"), c),
        ];
        for l in 0..cfg.num_hidden_layers {
            out.extend([
                (layer_param(l, "ln_1.weight"), c),
                (layer_param(l, "ln_1.bias"), c),
                (fused_qkv_name(l, "weight"), 3 * c * c),
                (fused_qkv_name(l, "bias"), 3 * c),
                (layer_param(l, "self_attn.out_proj.weight"), c * c),
                (layer_param(l, "self_attn.out_proj.bias"), c),
                (layer_param(l, "ls_1.gamma"), c),
                (layer_param(l, "ln_2.weight"), c),
                (layer_param(l, "ln_2.bias"), c),
                (layer_param(l, "mlp.c_fc.weight"), ff * c),
                (layer_param(l, "mlp.c_fc.bias"), ff),
                (layer_param(l, "mlp.c_proj.weight"), c * ff),
                (layer_param(l, "mlp.c_proj.bias"), c),
                (layer_param(l, "ls_2.gamma"), c),
            ]);
        }
        out.push((format!("{VISION_PREFIX}proj_in.weight"), cfg.projection_hidden_dim as usize * merged));
        out.push((format!("{VISION_PREFIX}proj_out.weight"), cfg.text_hidden_size as usize * cfg.projection_hidden_dim as usize));
        out
    }

    /// The store's initial tensors: the published ones, with each layer's q, k and v fused.
    fn initial_tensors(cfg: &VisionConfig, w: &VisionWeights) -> HashMap<String, Vec<f32>> {
        let mut init: HashMap<String, Vec<f32>> = Self::param_list(cfg)
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| !n.contains("qkv_proj"))
            .map(|n| {
                let data = w.get(&n).to_vec();
                (n, data)
            })
            .collect();
        for l in 0..cfg.num_hidden_layers {
            for kind in ["weight", "bias"] {
                let fused = ["q_proj", "k_proj", "v_proj"].iter().flat_map(|p| w.get(&layer_param(l, &format!("self_attn.{p}.{kind}"))).iter().copied()).collect();
                init.insert(fused_qkv_name(l, kind), fused);
            }
        }
        init
    }

    /// The projector's parameter names for the store's merger weights.
    fn projector_names() -> [(&'static str, String); 2] {
        [("in.weight", format!("{VISION_PREFIX}proj_in.weight")), ("layers.1.weight", format!("{VISION_PREFIX}proj_out.weight"))]
    }

    pub(crate) fn projector_config(cfg: &VisionConfig) -> Result<ProjectorConfig, String> {
        let merge = cfg.adapter_merge_size;
        Ok(ProjectorConfig::from_type("mlp2x_gelu", 2, merge * merge * cfg.hidden_size, cfg.projection_hidden_dim)?.with_out_dim(cfg.text_hidden_size)?.without_bias())
    }

    /// Build the tower: `trainable` names the store tensors that train (all others are frozen), `lora` the adapters over
    /// the frozen weights, `training` whether the backward is available at all.
    fn assemble(gpu: &Gpu, cfg: &VisionConfig, w: &VisionWeights, trainable: &dyn Fn(&str) -> bool, lora: Option<LoraTraining<'_>>, training: bool) -> Result<VisionTower, String> {
        let mut ids = VitKernelIds::by_name(gpu, "gelu_erf");
        let pairs = gpu.kernel_index("rope2d_pairs").ok_or("inkling2 vision: kernel \"rope2d_pairs\" is not registered (see vision::pipelines)")?;
        ids.rope2d = pairs;
        for (slot, name) in [
            (ids.layernorm, "layernorm"),
            (ids.matmul, "matmul"),
            (ids.matmul_rows, "matmul_rows"),
            (ids.bias_add, "bias_add"),
            (ids.mlp_act, "gelu_erf"),
            (ids.scale_chan, "scale_chan"),
            (ids.add2, "add2"),
        ] {
            if slot == UNREGISTERED {
                return Err(format!("inkling2 vision: kernel {name:?} is not registered (see vision::pipelines)"));
            }
        }
        let kernel = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("inkling2 vision: kernel {name:?} is not registered (see vision::pipelines)"));
        let permute = VitPermuteIds { embed: kernel("embed")?, row_scatter: kernel("row_scatter")? };

        let roles = Self::param_list(cfg).into_iter().map(|(n, len)| (n.clone(), len, if trainable(&n) { Role::Trainable } else { Role::Frozen })).collect();
        let store = ParamStore::new_with_roles(gpu, roles, &Self::initial_tensors(cfg, w));
        let train = if training { Some(train::TowerTraining::new(gpu, cfg, &store, lora)?) } else { None };

        let (mut weights, mut grads) = (HashMap::new(), HashMap::new());
        for (inner, name) in Self::projector_names() {
            weights.insert(inner.to_string(), store.w(&name).clone());
            if let Some(g) = store.grad.get(&name).or_else(|| train.as_ref().and_then(|t| t.weight_grad(&name))) {
                grads.insert(inner.to_string(), g.clone());
            }
        }
        let projector = MlpProjector::with_buffers(gpu, Self::projector_config(cfg)?, PROJECTOR_REGIONS, weights, grads, training)?;

        Ok(VisionTower { cfg: cfg.clone(), ids, pos_kernels: PosTableKernels::by_name(gpu)?, permute, region_copy: kernel("region_copy")?, store, projector: Mutex::new(projector), train })
    }

    /// The tower with a resident merger that has room for `regions` regions from the start, for a caller that knows
    /// its images' size and wants no growth on the first large group.
    pub fn with_projector_capacity(self, gpu: &Gpu, regions: u32) -> VisionTower {
        {
            let mut p = self.projector.lock().unwrap_or_else(|e| e.into_inner());
            *p = p.resized(gpu, regions.max(1));
        }
        self
    }

    /// The resident merger with room for at least `regions` regions: grown (activations only, the weights are shared)
    /// when the group is larger than what it holds.
    fn projector_for(&self, gpu: &Gpu, regions: u32) -> std::sync::MutexGuard<'_, MlpProjector> {
        let mut p = self.projector.lock().unwrap_or_else(|e| e.into_inner());
        if p.capacity() < regions {
            *p = p.resized(gpu, regions.max(p.capacity().saturating_mul(2)));
        }
        p
    }

    /// Every tower tensor, by store name.
    pub fn store(&self) -> &ParamStore {
        &self.store
    }

    /// The decoder width of the features.
    pub fn out_width(&self) -> usize {
        self.cfg.text_hidden_size as usize
    }

    /// Features for `images` (each a [`RegionImage`] from [`ImageProcessorSpec::tiling`]'s preprocessor).
    pub fn encode(&self, gpu: &Gpu, images: &[RegionImage]) -> Result<ImageFeatures, String> {
        self.encode_group_batches(gpu, images, None).map(|(features, _)| features)
    }

    /// [`Self::encode`] that also returns the activations named by `trace_layers` (layer indices) and the embedding,
    /// pre-norm, last-hidden and merger-input taps of every group, concatenated in group order.
    pub fn encode_traced(&self, gpu: &Gpu, images: &[RegionImage], trace_layers: &[usize]) -> Result<(ImageFeatures, VisionTrace), String> {
        self.encode_group_batches(gpu, images, Some(trace_layers))
    }

    /// Every image holds the pixels of its region grid.
    fn check_images(&self, images: &[RegionImage]) -> Result<(), String> {
        let region = self.cfg.native_patch_size * self.cfg.adapter_merge_size;
        let cell = (region * region * 3) as usize;
        for (i, img) in images.iter().enumerate() {
            if img.rows == 0 || img.cols == 0 || img.pixels.len() != img.regions() * cell {
                return Err(format!("inkling2 vision: image {i} holds {} floats, a {}x{} grid of {region}-pixel regions needs {}", img.pixels.len(), img.rows, img.cols, img.regions() * cell));
            }
        }
        Ok(())
    }

    fn encode_group_batches(&self, gpu: &Gpu, images: &[RegionImage], trace_layers: Option<&[usize]>) -> Result<(ImageFeatures, VisionTrace), String> {
        let merge = self.cfg.adapter_merge_size;
        self.check_images(images)?;
        if let Some(bad) = trace_layers.unwrap_or(&[]).iter().find(|&&l| l >= self.cfg.num_hidden_layers as usize) {
            return Err(format!("inkling2 vision: traced layer {bad} of {}", self.cfg.num_hidden_layers));
        }
        let mut tokens = Vec::with_capacity(images.len());
        let mut rows_out = Vec::new();
        let mut trace = VisionTrace::default();
        let per_image_rows = |img: &RegionImage| img.regions() * (merge * merge) as usize + 1;
        let mut start = 0;
        while start < images.len() {
            let mut end = start;
            let mut total = 0;
            while end < images.len() && (end == start || total + per_image_rows(&images[end]) <= MAX_GROUP_ROWS) {
                total += per_image_rows(&images[end]);
                end += 1;
            }
            let (rows, group_trace) = self.encode_group(gpu, &images[start..end], trace_layers)?;
            rows_out.extend(rows);
            tokens.extend(images[start..end].iter().map(RegionImage::regions));
            trace.merge_group(group_trace);
            start = end;
        }
        Ok((ImageFeatures { width: self.out_width(), tokens, rows: rows_out }, trace))
    }

    fn plan(&self, images: &[RegionImage]) -> GroupPlan {
        let merge = self.cfg.adapter_merge_size;
        let mut grids = Vec::with_capacity(images.len());
        let mut starts = Vec::with_capacity(images.len());
        let mut patch_rows = Vec::new();
        let mut next = 0u32;
        for img in images {
            let (h, w) = (img.rows * merge, img.cols * merge);
            grids.push((h, w));
            starts.push(next);
            patch_rows.extend(next + 1..next + 1 + h * w);
            next += 1 + h * w;
        }
        GroupPlan { grids, starts, rows: next, patches: patch_rows.len() as u32, patch_rows }
    }

    /// The `[patches, patch_size^2 * 3]` patch matrix of a group: each region's `merge x merge` patches in
    /// (row, column) order, each patch `[y][x][channel]`.
    fn patch_matrix(&self, images: &[RegionImage]) -> Vec<f32> {
        let (p, m) = (self.cfg.native_patch_size as usize, self.cfg.adapter_merge_size as usize);
        let region = p * m;
        let mut out = Vec::new();
        for img in images {
            for token in img.pixels.chunks_exact(region * region * 3) {
                for mh in 0..m {
                    for mw in 0..m {
                        for ph in 0..p {
                            let y = mh * p + ph;
                            let row = (y * region + mw * p) * 3;
                            out.extend_from_slice(&token[row..row + p * 3]);
                        }
                    }
                }
            }
        }
        out
    }

    /// The rotary tables `[rows, head_dim / 2]`: one angle per channel pair, the first half of the pairs on the
    /// column coordinate and the second half on the row coordinate (the reference concatenates width then height
    /// frequencies), patches at 1-indexed (row, column) and class tokens at (0, 0).
    fn rope_tables(&self, plan: &GroupPlan) -> (Vec<f32>, Vec<f32>) {
        let head_dim = (self.cfg.hidden_size / self.cfg.num_attention_heads) as usize;
        let spatial = head_dim / 2;
        let n_inv = spatial / 2;
        let inv: Vec<f32> = (0..n_inv).map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / spatial as f32)).collect();
        let half = head_dim / 2;
        let mut cos = vec![1.0f32; plan.rows as usize * half];
        let mut sin = vec![0.0f32; plan.rows as usize * half];
        let merge = self.cfg.adapter_merge_size;
        let mut row = 0usize;
        for (img, &(h, w)) in plan.grids.iter().enumerate() {
            debug_assert_eq!(plan.starts[img] as usize, row);
            row += 1; // the class token keeps angle 0
            for br in 0..h / merge {
                for bc in 0..w / merge {
                    for ir in 0..merge {
                        for ic in 0..merge {
                            let (r, c) = ((br * merge + ir + 1) as f32, (bc * merge + ic + 1) as f32);
                            for j in 0..half {
                                let (pos, f) = if j < n_inv { (c, inv[j]) } else { (r, inv[j - n_inv]) };
                                let angle = pos * f;
                                cos[row * half + j] = angle.cos();
                                sin[row * half + j] = angle.sin();
                            }
                            row += 1;
                        }
                    }
                }
            }
        }
        (cos, sin)
    }

    /// The block's weights from the store, with the group's rotary tables.
    fn block_weights<'a>(&'a self, l: u32, rope: &'a RopeTables<'a>) -> VitBlockWeights<'a> {
        let w = |tail: &str| self.store.w(&layer_param(l, tail));
        VitBlockWeights {
            norm1_w: w("ln_1.weight"),
            norm1_b: w("ln_1.bias"),
            qkv_w: self.store.w(&fused_qkv_name(l, "weight")),
            qkv_b: self.store.w(&fused_qkv_name(l, "bias")),
            qk_norm: None,
            rope: Some(RopeTables { cos: rope.cos, sin: rope.sin, tmod: rope.tmod }),
            proj_w: w("self_attn.out_proj.weight"),
            proj_b: w("self_attn.out_proj.bias"),
            ls1: Some(w("ls_1.gamma")),
            norm2_w: w("ln_2.weight"),
            norm2_b: w("ln_2.bias"),
            fc1_w: w("mlp.c_fc.weight"),
            fc1_b: w("mlp.c_fc.bias"),
            fc2_w: w("mlp.c_proj.weight"),
            fc2_b: w("mlp.c_proj.bias"),
            ls2: Some(w("ls_2.gamma")),
        }
    }

    /// The device inputs of a group of images: pixels, position taps, row maps of the sequence layout, rotary tables
    /// and attention spans.
    fn group_setup(&self, g: &Gpu, images: &[RegionImage]) -> GroupSetup {
        let c = self.cfg.hidden_size;
        let merge = self.cfg.adapter_merge_size;
        let plan = self.plan(images);
        let pos_plan = bilinear_grid_plan(&plan.grids, self.cfg.posemb_grid_size, merge, false).expect("image grids are multiples of the merge size by construction");
        let taps = DeviceTaps::upload(g, &pos_plan, self.cfg.posemb_grid_size * self.cfg.posemb_grid_size).expect("a plan built from the table's own side is in range");
        let (cos, sin) = self.rope_tables(&plan);
        let spans: Vec<(u32, u32)> = plan.starts.iter().zip(&plan.grids).map(|(&s, &(h, w))| (s, 1 + h * w)).collect();
        GroupSetup {
            sh: VitShape { dim: c, heads: self.cfg.num_attention_heads, mlp: self.cfg.mlp_width(), eps: self.cfg.layer_norm_eps as f32 },
            pix: g.storage_init("vit.pixels", &self.patch_matrix(images)),
            taps,
            zero_idx: row_index_buffer(g, "vit.cls_source", &vec![0; images.len()]),
            patch_idx: row_index_buffer(g, "vit.patch_rows", &plan.patch_rows),
            cls_idx: row_index_buffer(g, "vit.cls_rows", &plan.starts),
            cos: g.storage_init("vit.rope.cos", &cos),
            sin: g.storage_init("vit.rope.sin", &sin),
            max_span: spans.iter().map(|&(_, n)| n).max().unwrap_or(1),
            spans,
            plan,
        }
    }

    /// Embeddings: patch projection (no bias) + resampled position table, patches and class rows scattered into the
    /// sequence into `emb`, then `ln_pre` into `x`.
    fn embedding_steps(&self, g: &Gpu, s: &GroupSetup, emb: &DeviceBuffer, x: &DeviceBuffer) -> Vec<Step> {
        let c = s.sh.dim;
        let patch_dim = 3 * self.cfg.native_patch_size.pow(2);
        let (rows, patches, images) = (s.plan.rows, s.plan.patches, s.plan.starts.len() as u32);
        let (pe, pos, pe_pos) = (g.storage(u64::from(patches * c)), g.storage(u64::from(patches * c)), g.storage(u64::from(patches * c)));
        let (cls_sum, cls_rows) = (g.storage(u64::from(c)), g.storage(u64::from(images * c)));
        let ln = LayerNormIds::resolve_fwd(g, self.ids.layernorm);
        vec![
            linear_rows_step(g, &self.ids, &s.pix, self.store.w(&backbone("embeddings.patch_embedding.weight")), &pe, (patches, patch_dim, c)),
            s.taps.gather_step(g, &self.pos_kernels, self.store.w(&backbone("embeddings.position_embedding")), &pos, c),
            g.step(self.ids.add2, &[&pe, &pos, &pe_pos], &[patches * c], patches * c),
            scatter_rows(g, &self.permute, &s.patch_idx, &pe_pos, emb, patches, c, rows),
            // The class token is the class embedding plus its position embedding, one row per image.
            g.step(self.ids.add2, &[self.store.w(&backbone("embeddings.class_embedding")), self.store.w(&backbone("embeddings.cls_position_embedding")), &cls_sum], &[c], c),
            gather_rows(g, &self.permute, &s.zero_idx, &cls_sum, &cls_rows, images, c),
            scatter_rows(g, &self.permute, &s.cls_idx, &cls_rows, emb, images, c, rows),
            model::block::layernorm_fwd(g, &ln, emb, self.store.w(&backbone("ln_pre.weight")), self.store.w(&backbone("ln_pre.bias")), x, c, rows, s.sh.eps),
        ]
    }

    fn encode_group(&self, g: &Gpu, images: &[RegionImage], trace_layers: Option<&[usize]>) -> Result<(Vec<f32>, VisionTrace), String> {
        self.fold_adapters(g);
        let setup = self.group_setup(g, images);
        let (c, cw) = (setup.sh.dim, setup.sh.dim as usize);
        let (rows, patches) = (setup.plan.rows, setup.plan.patches);
        let regions = patches / (self.cfg.adapter_merge_size * self.cfg.adapter_merge_size);

        let emb = g.storage(u64::from(rows) * u64::from(c));
        let x = g.storage(u64::from(rows) * u64::from(c));
        let chunk = attn_chunk_for_device(g, &setup.sh, setup.max_span);
        let scratch = VitScratch::new(g, &setup.sh, rows, chunk, setup.max_span);
        let mut steps = self.embedding_steps(g, &setup, &emb, &x);

        let tap = |steps: &mut Vec<Step>, src: &DeviceBuffer| -> DeviceBuffer {
            let dst = g.storage(u64::from(rows) * u64::from(c));
            steps.push(g.step(self.region_copy, &[src, &dst], &[rows, c, c, 0], rows * c));
            dst
        };
        let emb_tap = trace_layers.map(|_| tap(&mut steps, &emb));
        let pre_tap = trace_layers.map(|_| tap(&mut steps, &x));
        let mut layer_taps = Vec::new();
        let rope = RopeTables { cos: &setup.cos, sin: &setup.sin, tmod: rows };
        for l in 0..self.cfg.num_hidden_layers {
            let bw = self.block_weights(l, &rope);
            vit_block_fwd(g, &self.ids, &setup.sh, &bw, &x, rows, &setup.spans, chunk, &scratch, &mut steps);
            if trace_layers.is_some_and(|t| t.contains(&(l as usize))) {
                layer_taps.push((l as usize, tap(&mut steps, &x)));
            }
        }
        // Merger: the patch states of each region (consecutive patches) side by side, then proj_in / GELU / proj_out.
        let states = g.storage(u64::from(patches) * u64::from(c));
        steps.push(gather_rows(g, &self.permute, &setup.patch_idx, &x, &states, patches, c));
        let projector = self.projector_for(g, regions);
        steps.extend(projector.forward_rows(g, &[&states], regions));
        g.submit(&[], &steps);

        let features = g.read(projector.out(), regions as usize * self.out_width());
        let trace = if trace_layers.is_none() {
            VisionTrace::default()
        } else {
            VisionTrace {
                embeddings: g.read(emb_tap.as_ref().expect("traced"), rows as usize * cw),
                pre_layernorm: g.read(pre_tap.as_ref().expect("traced"), rows as usize * cw),
                layers: layer_taps.iter().map(|(l, b)| (*l, g.read(b, rows as usize * cw))).collect(),
                last_hidden: g.read(&x, rows as usize * cw),
                merger_in: g.read(&states, patches as usize * cw),
            }
        };
        Ok((features, trace))
    }

    /// The whole image chain for a prompt: preprocess each encoded image (`tiling` from the processor spec), expand the
    /// rendered prompt's placeholders to one per region, encode, and return the expanded prompt with the features
    /// spliced as [`PromptEmbeds`] at the placeholder positions.
    pub fn embed_prompt(&self, gpu: &Gpu, tiling: &RegionTiling, prompt: &[u32], image_token: u32, images: &[&[u8]]) -> Result<(Vec<u32>, PromptEmbeds), String> {
        let regions: Vec<RegionImage> = images.iter().enumerate().map(|(i, bytes)| tiling.preprocess_bytes(bytes).map_err(|e| format!("image {i}: {e}"))).collect::<Result<_, _>>()?;
        let counts: Vec<usize> = regions.iter().map(RegionImage::regions).collect();
        let expanded = expand_placeholders(prompt, image_token, &counts)?;
        let features = self.encode(gpu, &regions)?;
        let embeds = splice_embeds(&expanded, image_token, &features)?;
        Ok((expanded, embeds))
    }
}

impl VisionTrace {
    fn merge_group(&mut self, other: VisionTrace) {
        self.embeddings.extend(other.embeddings);
        self.pre_layernorm.extend(other.pre_layernorm);
        self.last_hidden.extend(other.last_hidden);
        self.merger_in.extend(other.merger_in);
        for (l, rows) in other.layers {
            match self.layers.iter_mut().find(|(k, _)| *k == l) {
                Some((_, acc)) => acc.extend(rows),
                None => self.layers.push((l, rows)),
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Prompt splice
// ---------------------------------------------------------------------------------------------------------------

/// The reference processor's placeholder expansion on token ids: the `i`-th occurrence of `image_token` in the rendered
/// prompt (the chat template renders one per image) becomes `tokens_per_image[i]` copies. The occurrences must match
/// the images one to one, and every image must stand for at least one token.
pub fn expand_placeholders(prompt: &[u32], image_token: u32, tokens_per_image: &[usize]) -> Result<Vec<u32>, String> {
    let found = prompt.iter().filter(|&&t| t == image_token).count();
    if found != tokens_per_image.len() {
        return Err(format!("the prompt holds {found} image placeholder(s) (token {image_token}) for {} image(s)", tokens_per_image.len()));
    }
    if let Some(i) = tokens_per_image.iter().position(|&n| n == 0) {
        return Err(format!("image {i} has no regions"));
    }
    let mut out = Vec::with_capacity(prompt.len() + tokens_per_image.iter().sum::<usize>());
    let mut next = 0;
    for &t in prompt {
        if t == image_token {
            out.extend(std::iter::repeat_n(image_token, tokens_per_image[next]));
            next += 1;
        } else {
            out.push(t);
        }
    }
    u32::try_from(out.len()).map_err(|_| "the expanded prompt is longer than a u32 position".to_string())?;
    Ok(out)
}

/// The image features as [`PromptEmbeds`]: the rows of all images, in image order, fill the positions of `prompt`'s
/// `image_token`s in order (`get_placeholder_mask` + `masked_scatter`). The placeholder count must equal the number of
/// feature rows, else the error names both.
pub fn splice_embeds(prompt: &[u32], image_token: u32, features: &ImageFeatures) -> Result<PromptEmbeds, String> {
    let positions: Vec<u32> = prompt.iter().enumerate().filter(|&(_, &t)| t == image_token).map(|(i, _)| i as u32).collect();
    let rows = features.rows().len() / features.width();
    if positions.len() != rows {
        return Err(format!("image features and placeholder tokens do not match: {} placeholder(s), {rows} feature row(s)", positions.len()));
    }
    PromptEmbeds::new(features.width(), positions, features.rows().to_vec())
}
