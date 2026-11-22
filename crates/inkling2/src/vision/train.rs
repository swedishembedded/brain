// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements image-text fine-tuning for its clients: the gradient of a language-model loss reaches
// the vision tower and its adapters through shared, gradient-checked blocks. If your team needs expertise in training a
// multimodal model from its published release then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Training the Inkling 2 vision tower: the backward of [`VisionTower`]'s forward, and LoRA over its linears.
//!
//! **What trains** is decided when the tower is built ([`VisionTower::new_trainable`]): a store tensor with a
//! gradient buffer is differentiated, one without is frozen. Any subset is allowed, from the merger alone to the whole
//! tower; the backward runs only as far down the tower as the lowest tensor that needs a gradient (a merger-only
//! build stops at the merger, an adapter on layer 20 stops there). Gradients accumulate (`+=`) across passes;
//! [`VisionTower::zero_grads`] clears them.
//!
//! **A pass** is [`VisionTower::encode_train`] (features for a group of images, keeping what the backward needs)
//! followed by [`VisionTower::backward`] with the loss's gradient with respect to those features. The decoder's side of
//! that gradient is the splice's backward, `PromptEmbeds::take_row_gradients`: the decoder's gradient with respect to
//! the prompt's embeddings, read at the placeholder positions, IS the gradient with respect to the features, in the
//! same order. A pass owns its merger activations, so passes of several examples may be pending at once.
//!
//! **Memory** follows `model::saved::SavedPolicy`: [`SavedPolicy::Keep`] holds every layer's cached activations
//! (attention probabilities included), [`SavedPolicy::Recompute`] holds only the layer boundaries and rebuilds one
//! layer's cache at a time while walking back. The gradients are the same bit for bit. Attention probabilities are
//! cached per image (`heads x span^2`), so training bounds an image's size ([`MAX_CACHE_BYTES`]); serving does not.
//!
//! **Adapters** ([`LoraTraining`]) use the shared fold choreography (`model::adapter::fold`): the live weight of an
//! adapted linear is `W0 + scale * B A`, written before each forward (training or serving, so a tower never runs on
//! stale adapters), and the dense weight gradient the backward leaves is projected onto the two factors afterwards.
//! The sites are the published linears: q, k and v separately (rows of the fused weight), the output projection, both
//! MLP linears and the merger's two; [`lora_sites`](model::adapter::sites::LoraSites) lists them. A weight is trained
//! in full or adapted, never both. Weights without a gradient buffer still get their dense gradient computed (the
//! shared block always writes one) into one shared sink, an inefficiency the roadmap records.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu};
use model::adapter::device::{apply_groups, init_of, role_of, LoraCfg};
use model::adapter::fold::{fold_fused, project, FoldKernels, FusedTarget, GradTarget, Slice};
use model::adapter::kind::DeviceKind;
use model::adapter::sites::{tensor_names, LoraSites};
use model::adapter::{LinearSite, ParamInit, TargetSpec};
use model::block::LayerNormIds;
use model::projector::MlpProjector;
use model::saved::{BlockStack, SavedPolicy};
use model::serve::PromptEmbeds;
use model::vit::{gather_rows, scatter_rows, vit_block_bwd, vit_block_fwd_cached, RopeTables, VitBlockCache, VitBlockGrads, VitBwdIds, VitBwdScratch};
use paramstore::ParamStore;

use super::{backbone, expand_placeholders, fused_qkv_name, layer_param, pipelines, splice_embeds, GroupSetup, ImageFeatures, VisionTower, MAX_GROUP_ROWS};
use crate::config::VisionConfig;
use crate::names::VISION_PREFIX;

/// The largest cache of attention probabilities and activations a training pass may hold, over all layers it keeps.
pub const MAX_CACHE_BYTES: u64 = 8 << 30;

/// Every kernel a training build dispatches: the forward's ([`pipelines`]) plus the backward's. A trainer adds its
/// optimizer's.
pub fn train_pipelines() -> &'static [(&'static str, &'static str)] {
    static ALL: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    ALL.get_or_init(|| {
        let mut all: Vec<(&'static str, &'static str)> = pipelines().to_vec();
        for &(name, src) in &[
            ("layernorm_dx", kernels::LAYERNORM_DX),
            ("layernorm_dgamma", kernels::LAYERNORM_DGAMMA),
            ("layernorm_dbeta", kernels::LAYERNORM_DBETA),
            ("ln_stats", kernels::LN_STATS),
            ("matmul_dx", kernels::MATMUL_DX),
            ("matmul_dw", kernels::MATMUL_DW),
            ("bias_grad", kernels::BIAS_GRAD),
            ("gelu_erf_bwd", kernels::GELU_ERF_BWD),
            ("scale_chan_dg", kernels::SCALE_CHAN_DG),
            ("attn_bwd_dscores_cross", kernels::ATTN_BWD_DSCORES_CROSS),
            ("attn_bwd_dv_cross", kernels::ATTN_BWD_DV_CROSS),
            ("attn_bwd_dq_cross", kernels::ATTN_BWD_DQ_CROSS),
            ("attn_bwd_dk_cross", kernels::ATTN_BWD_DK_CROSS),
            ("axpy", kernels::AXPY),
        ] {
            if !all.iter().any(|&(n, _)| n == name) {
                all.push((name, src));
            }
        }
        all
    })
}

/// The linears an adapter may correct, with the published tensor names as site names. q, k and v are three sites
/// even though the tower fuses them; the leaves are `q_proj`, `k_proj`, `v_proj`, `out_proj`, `c_fc`, `c_proj` in
/// every layer and `proj_in`, `proj_out` in the merger.
impl LoraSites for VisionConfig {
    fn lora_sites(&self) -> Vec<LinearSite> {
        let (c, ff) = (self.hidden_size as usize, self.mlp_width() as usize);
        let (merged, hidden, text) = ((self.adapter_merge_size * self.adapter_merge_size * self.hidden_size) as usize, self.projection_hidden_dim as usize, self.text_hidden_size as usize);
        let site = |name: String, leaf: &'static str, layer: Option<usize>, spec: TargetSpec| LinearSite { name, leaf, layer, spec, save_name: None };
        let mut out = Vec::new();
        for l in 0..self.num_hidden_layers {
            let at = Some(l as usize);
            for (i, leaf) in ["q_proj", "k_proj", "v_proj"].into_iter().enumerate() {
                out.push(site(layer_param(l, &format!("self_attn.{leaf}.weight")), leaf, at, TargetSpec::row_slice(c, c, i * c, c)));
            }
            out.push(site(layer_param(l, "self_attn.out_proj.weight"), "out_proj", at, TargetSpec::whole(c, c)));
            out.push(site(layer_param(l, "mlp.c_fc.weight"), "c_fc", at, TargetSpec::whole(ff, c)));
            out.push(site(layer_param(l, "mlp.c_proj.weight"), "c_proj", at, TargetSpec::whole(c, ff)));
        }
        out.push(site(format!("{VISION_PREFIX}proj_in.weight"), "proj_in", None, TargetSpec::whole(hidden, merged)));
        out.push(site(format!("{VISION_PREFIX}proj_out.weight"), "proj_out", None, TargetSpec::whole(text, hidden)));
        out
    }
}

/// An adapter configuration for the tower and the draw that starts `A` (`B` starts at zero, so an adapted tower
/// begins as the base). Adapters are plain LoRA with one rank and scale; other kinds, per-leaf overrides and targets
/// that match no site of the tower are refused by name.
pub struct LoraTraining<'a> {
    pub cfg: &'a LoraCfg,
    pub init: &'a mut dyn FnMut() -> f32,
}

/// One adapted linear: where it lives in the store.
struct AdaptedSite {
    site: LinearSite,
    /// The store tensor holding its weight (the fused qkv for q, k and v).
    weight: String,
    /// The first element of the site inside that tensor.
    offset: u64,
}

/// The adapters of a tower: factors in a store of their own, the untouched base weights, and the dense gradient
/// buffers of the adapted weights.
pub(super) struct TowerLora {
    cfg: LoraCfg,
    store: ParamStore,
    sites: Vec<AdaptedSite>,
    /// The adapted weights (store names) with the indices of their sites, in a fixed order.
    by_weight: Vec<(String, Vec<usize>)>,
    base: HashMap<String, DeviceBuffer>,
    dw: HashMap<String, DeviceBuffer>,
    weight_len: HashMap<String, u64>,
    /// Where `A`'s gradient goes when `A` is frozen (the projection always computes it).
    ga_sink: Option<DeviceBuffer>,
    kernels: FoldKernels,
}

/// What a training build adds to a tower.
pub(super) struct TowerTraining {
    bwd: VitBwdIds,
    saved: SavedPolicy,
    /// Receives the dense gradient of every weight nothing reads: a shared block always writes them all.
    sink: DeviceBuffer,
    lora: Option<TowerLora>,
}

impl TowerTraining {
    pub(super) fn new(gpu: &Gpu, cfg: &VisionConfig, store: &ParamStore, lora: Option<LoraTraining<'_>>) -> Result<TowerTraining, String> {
        let bwd = VitBwdIds::by_name(gpu, "gelu_erf_bwd");
        for (id, name) in [
            (bwd.layernorm_dx, "layernorm_dx"),
            (bwd.ln_dgamma, "layernorm_dgamma"),
            (bwd.ln_dbeta, "layernorm_dbeta"),
            (bwd.matmul_dx, "matmul_dx"),
            (bwd.matmul_dw, "matmul_dw"),
            (bwd.bias_grad, "bias_grad"),
            (bwd.mlp_act_bwd, "gelu_erf_bwd"),
            (bwd.scale_chan_dg, "scale_chan_dg"),
            (bwd.attn_bwd_dscores_cross, "attn_bwd_dscores_cross"),
            (bwd.attn_bwd_dv_cross, "attn_bwd_dv_cross"),
            (bwd.attn_bwd_dq_cross, "attn_bwd_dq_cross"),
            (bwd.attn_bwd_dk_cross, "attn_bwd_dk_cross"),
            (bwd.ln_stats, "ln_stats"),
            (bwd.axpy, "axpy"),
        ] {
            if id == model::vit::UNREGISTERED {
                return Err(format!("inkling2 vision: kernel {name:?} is not registered (see vision::train_pipelines)"));
            }
        }
        let widest = store.params.iter().map(|(_, n)| *n).max().unwrap_or(1);
        let lora = lora.map(|l| TowerLora::new(gpu, cfg, store, l)).transpose()?;
        Ok(TowerTraining { bwd, saved: SavedPolicy::Recompute, sink: gpu.storage(widest as u64), lora })
    }

    /// The dense gradient buffer of an adapted weight, for the merger to share.
    pub(super) fn weight_grad(&self, name: &str) -> Option<&DeviceBuffer> {
        self.lora.as_ref().and_then(|l| l.dw.get(name))
    }

    /// Where the backward writes `name`'s gradient: its own buffer, the adapters' dense buffer, or the sink.
    fn grad<'a>(&'a self, store: &'a ParamStore, name: &str) -> &'a DeviceBuffer {
        store.grad.get(name).or_else(|| self.weight_grad(name)).unwrap_or(&self.sink)
    }
}

impl TowerLora {
    fn new(gpu: &Gpu, cfg: &VisionConfig, store: &ParamStore, l: LoraTraining<'_>) -> Result<TowerLora, String> {
        let lc = l.cfg;
        if lc.kind != DeviceKind::Lora {
            return Err(format!("inkling2 vision: the tower folds plain LoRA adapters, not {}", lc.kind.name()));
        }
        if !lc.is_uniform() {
            return Err("inkling2 vision: the tower's adapter fold takes one rank and scale; per-leaf overrides are not supported".to_string());
        }
        if lc.rank == 0 {
            return Err("inkling2 vision: a LoRA rank must be at least 1".to_string());
        }
        let c = u64::from(cfg.hidden_size);
        let mut sites = Vec::new();
        for site in cfg.lora_sites().into_iter().filter(|s| lc.targets_leaf(s.leaf)) {
            let (weight, offset) = match site.leaf {
                "q_proj" | "k_proj" | "v_proj" => (fused_qkv_name(site.layer.expect("a layer site") as u32, "weight"), (site.spec.row0 * site.spec.row_stride + site.spec.col0) as u64),
                _ => (site.name.clone(), 0),
            };
            if store.grad.contains_key(&weight) {
                return Err(format!("inkling2 vision: {weight} is trained in full, so its adapter site {} cannot also be adapted", site.name));
            }
            if !offset.is_multiple_of(64) || (offset != 0 && !c.is_multiple_of(8)) {
                return Err(format!("inkling2 vision: the adapter site {} starts at element {offset} of {weight}, which is not on a 64-word boundary (hidden size {c} must be a multiple of 8)", site.name));
            }
            sites.push(AdaptedSite { site, weight, offset });
        }
        if sites.is_empty() {
            return Err(format!("inkling2 vision: the adapter targets {:?} match no site of the tower (leaves: q_proj, k_proj, v_proj, out_proj, c_fc, c_proj, proj_in, proj_out)", lc.targets));
        }

        let mut roles = Vec::new();
        let mut init = HashMap::new();
        for s in &sites {
            for (name, numel) in lc.param_list(&s.site.name, s.site.spec.out, s.site.spec.inn) {
                let hp = lc.hp_for(&name);
                roles.push((name.clone(), numel, role_of(&name, &hp)));
                let values = match init_of(&name) {
                    Some(ParamInit::Drawn) => (0..numel).map(|_| (l.init)()).collect(),
                    _ => vec![0.0; numel],
                };
                init.insert(name, values);
            }
        }
        let mut lora_store = ParamStore::new_with_roles(gpu, roles, &init);
        apply_groups(&mut lora_store, &lc.hp());

        let mut by_weight: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, s) in sites.iter().enumerate() {
            match by_weight.iter_mut().find(|(w, _)| *w == s.weight) {
                Some((_, v)) => v.push(i),
                None => by_weight.push((s.weight.clone(), vec![i])),
            }
        }
        let kernels = FoldKernels::resolve(gpu);
        let (mut base, mut dw, mut weight_len) = (HashMap::new(), HashMap::new(), HashMap::new());
        for (w, _) in &by_weight {
            let len = store.numel(w) as u64;
            let copy = gpu.storage(len);
            gpu.submit(&[&copy], &[gpu.step(kernels.axpy, &[&copy, store.w(w)], &[len as u32, gpu_core::f(1.0)], len as u32)]);
            base.insert(w.clone(), copy);
            dw.insert(w.clone(), gpu.storage(len));
            weight_len.insert(w.clone(), len);
        }
        let ga_sink = lc.freeze_a.then(|| gpu.storage(sites.iter().map(|s| lc.rank as u64 * s.site.spec.inn as u64).max().unwrap_or(1)));
        Ok(TowerLora { cfg: lc.clone(), store: lora_store, sites, by_weight, base, dw, weight_len, ga_sink, kernels })
    }

    fn scale(&self) -> f32 {
        self.cfg.scale()
    }
}

/// What the backward of one encoded group reads.
pub struct VisionPass {
    setup: GroupSetup,
    emb: DeviceBuffer,
    /// The residual stream entering each layer, then leaving the last: `layers + 1` buffers.
    inputs: Vec<DeviceBuffer>,
    /// Each layer's cache under [`SavedPolicy::Keep`]; empty under [`SavedPolicy::Recompute`].
    caches: Vec<VitBlockCache>,
    /// The patch states regrouped for the merger, `[regions, merge^2 * width]`.
    states: DeviceBuffer,
    /// The merger sized for this group: the pass's own activations over the tower's shared weights.
    projector: MlpProjector,
    regions: u32,
}

impl VisionPass {
    /// How many feature rows the group produced; [`VisionTower::backward`] takes this many.
    pub fn regions(&self) -> usize {
        self.regions as usize
    }
}

impl VisionTower {
    /// A tower that can be trained: the store tensors `trainable` accepts (by store name: the published names, with
    /// each layer's q, k and v fused as [`fused_qkv_name`]) get gradient buffers and the rest are frozen. `lora`
    /// adds adapters over frozen weights. `gpu` must register [`train_pipelines`].
    pub fn new_trainable(gpu: &Gpu, cfg: &VisionConfig, w: &super::VisionWeights, trainable: &dyn Fn(&str) -> bool, lora: Option<LoraTraining<'_>>) -> Result<VisionTower, String> {
        Self::assemble(gpu, cfg, w, trainable, lora, true)
    }

    fn training(&self) -> Result<&TowerTraining, String> {
        self.train.as_ref().ok_or_else(|| "inkling2 vision: this tower was built for inference (VisionTower::new_trainable builds one that trains)".to_string())
    }

    /// What the forward keeps for the backward: [`SavedPolicy::Keep`] (every layer's cache) or
    /// [`SavedPolicy::Recompute`] (the default: layer boundaries only). A narrow saved format is refused by name.
    pub fn set_saved_policy(&mut self, policy: SavedPolicy) -> Result<(), String> {
        BlockStack::<(), ()>::new("inkling2 vision tower", policy)?;
        let t = self.train.as_mut().ok_or("inkling2 vision: an inference tower keeps nothing for a backward")?;
        t.saved = policy;
        Ok(())
    }

    /// The adapter factors (`{site}.lora_a`, `{site}.lora_b` by the published site names) and their gradients, when
    /// the tower has adapters.
    pub fn lora_store(&self) -> Option<&ParamStore> {
        self.train.as_ref().and_then(|t| t.lora.as_ref()).map(|l| &l.store)
    }

    /// Zero every gradient: the tower's, the adapters', the dense buffers and the sink.
    pub fn zero_grads(&self, gpu: &Gpu) {
        self.store.zero_grads(gpu);
        if let Some(t) = &self.train {
            let mut clears: Vec<&DeviceBuffer> = vec![&t.sink];
            if let Some(l) = &t.lora {
                l.store.zero_grads(gpu);
                clears.extend(l.dw.values());
                clears.extend(l.ga_sink.as_ref());
            }
            gpu.submit(&clears, &[]);
            self.projector.lock().unwrap_or_else(|e| e.into_inner()).zero_grads(gpu);
        }
    }

    /// Write `W0 + scale * B A` into every adapted weight. Called before every forward; a tower without adapters has
    /// nothing to do.
    pub(super) fn fold_adapters(&self, g: &Gpu) {
        let Some(l) = self.train.as_ref().and_then(|t| t.lora.as_ref()) else { return };
        let targets: Vec<FusedTarget<'_>> = l
            .by_weight
            .iter()
            .map(|(w, idx)| FusedTarget {
                weight: self.store.w(w),
                base: &l.base[w],
                len: l.weight_len[w],
                slices: idx
                    .iter()
                    .map(|&i| {
                        let s = &l.sites[i];
                        let (a, b) = tensor_names(&s.site);
                        Slice { a: l.store.w(&a), b: l.store.w(&b), out: s.site.spec.out as u32, inn: s.site.spec.inn as u32, offset: s.offset }
                    })
                    .collect(),
            })
            .collect();
        fold_fused(g, &l.kernels, &targets, l.cfg.rank, l.scale());
    }

    /// Accumulate the adapters' gradients from the dense gradients the backward left in the adapted weights' buffers.
    fn project_adapters(&self, g: &Gpu) {
        let Some(l) = self.train.as_ref().and_then(|t| t.lora.as_ref()) else { return };
        let targets: Vec<GradTarget<'_>> = l
            .sites
            .iter()
            .map(|s| {
                let (a, b) = tensor_names(&s.site);
                GradTarget {
                    dw: &l.dw[&s.weight],
                    offset: s.offset,
                    a: l.store.w(&a),
                    b: l.store.w(&b),
                    ga: l.store.grad.get(&a).or(l.ga_sink.as_ref()).expect("A has a gradient buffer or the sink"),
                    gb: l.store.g(&b),
                    out: s.site.spec.out as u32,
                    inn: s.site.spec.inn as u32,
                }
            })
            .collect();
        project(g, &l.kernels, &targets, l.cfg.rank, l.scale());
    }

    fn layer_trains(&self, l: u32) -> bool {
        let t = self.train.as_ref().expect("a training tower");
        let prefix = layer_param(l, "");
        self.store.grad.keys().any(|n| n.starts_with(&prefix)) || t.lora.as_ref().is_some_and(|a| a.sites.iter().any(|s| s.site.layer == Some(l as usize)))
    }

    fn embeddings_train(&self) -> bool {
        ["embeddings.class_embedding", "embeddings.cls_position_embedding", "embeddings.patch_embedding.weight", "embeddings.position_embedding", "ln_pre.weight", "ln_pre.bias"]
            .iter()
            .any(|n| self.store.grad.contains_key(&backbone(n)))
    }

    /// [`Self::encode`] for training: the features of one group of images in a single pass (a training example is one
    /// request), and what [`Self::backward`] needs. `gpu` must register [`train_pipelines`].
    pub fn encode_train(&self, g: &Gpu, images: &[imaging::RegionImage]) -> Result<(ImageFeatures, VisionPass), String> {
        let train = self.training()?;
        self.check_images(images)?;
        if images.is_empty() {
            return Err("inkling2 vision: a training pass needs at least one image".to_string());
        }
        let merge = self.cfg.adapter_merge_size;
        let rows: usize = images.iter().map(|i| i.regions() * (merge * merge) as usize + 1).sum();
        if rows > MAX_GROUP_ROWS {
            return Err(format!("inkling2 vision: a training pass holds {rows} sequence rows, at most {MAX_GROUP_ROWS}"));
        }
        // The cached attention backward binds each image's rows of the context gradient at an offset, which must sit on a
        // 64-float boundary (`model::vit::WindowPlan::ctx_bindable`): always true for a width that is a multiple of 64.
        let width = u64::from(self.cfg.hidden_size);
        let mut start = 0u64;
        for (i, img) in images.iter().enumerate() {
            if !(start * width).is_multiple_of(64) {
                return Err(format!("inkling2 vision: image {i} starts at sequence row {start}, which is not on a 64-float boundary for width {width}; training needs a hidden size that is a multiple of 64"));
            }
            start += (img.regions() * (merge * merge) as usize + 1) as u64;
        }
        let heads = u64::from(self.cfg.num_attention_heads);
        let widest = images.iter().map(|i| i.regions() * (merge * merge) as usize + 1).max().unwrap_or(1) as u64;
        let layers = u64::from(self.cfg.num_hidden_layers);
        let per_layer = 4 * (heads * rows as u64 * widest + rows as u64 * (u64::from(self.cfg.hidden_size) * 12 + 2 * u64::from(self.cfg.mlp_width())));
        let held = if train.saved.recomputes() { per_layer + 4 * (layers + 1) * rows as u64 * u64::from(self.cfg.hidden_size) } else { per_layer * layers };
        if held > MAX_CACHE_BYTES {
            return Err(format!(
                "inkling2 vision: this group needs about {held} bytes of saved activations under {} (attention caches {heads} x {rows} x {widest} floats a layer), more than the {MAX_CACHE_BYTES} a training pass may hold; use fewer or smaller images{}",
                train.saved.label(),
                if train.saved.recomputes() { "" } else { " or the recompute policy" }
            ));
        }

        self.fold_adapters(g);
        let setup = self.group_setup(g, images);
        let (c, rows32, patches) = (setup.sh.dim, setup.plan.rows, setup.plan.patches);
        let regions = patches / (merge * merge);
        let n = u64::from(rows32) * u64::from(c);

        let emb = g.storage(n);
        let x0 = g.storage(n);
        g.submit(&[], &self.embedding_steps(g, &setup, &emb, &x0));

        let rope = RopeTables { cos: &setup.cos, sin: &setup.sin, tmod: rows32 };
        let scr_tmp = g.storage(n);
        let scores = g.storage(u64::from(self.cfg.num_attention_heads) * u64::from(setup.max_span) * u64::from(setup.max_span));
        let mut inputs = vec![x0];
        let mut caches = Vec::new();
        // Recomputing reuses one cache for every layer; keeping makes one per layer.
        let mut shared = train.saved.recomputes().then(|| VitBlockCache::new(g, &setup.sh, rows32, setup.max_span));
        for l in 0..self.cfg.num_hidden_layers {
            let out = g.storage(n);
            let mut cache = shared.is_none().then(|| VitBlockCache::new(g, &setup.sh, rows32, setup.max_span));
            let cache_ref = cache.as_mut().or(shared.as_mut()).expect("a cache for the layer");
            cache_ref.x_in = inputs[l as usize].clone();
            let bw = self.block_weights(l, &rope);
            let mut steps = Vec::new();
            vit_block_fwd_cached(g, &self.ids, &train.bwd, &setup.sh, &bw, cache_ref, &out, rows32, &setup.spans, &scr_tmp, &scores, &mut steps);
            g.submit(&[&cache_ref.qkv], &steps);
            caches.extend(cache);
            inputs.push(out);
        }

        let states = g.storage(u64::from(patches) * u64::from(c));
        let projector = self.projector.lock().unwrap_or_else(|e| e.into_inner()).resized(g, regions);
        let mut steps = vec![gather_rows(g, &self.permute, &setup.patch_idx, inputs.last().expect("a layer ran"), &states, patches, c)];
        steps.extend(projector.forward_rows(g, &[&states], regions));
        g.submit(&[], &steps);
        let features = g.read(projector.out(), regions as usize * self.out_width());
        let tokens = images.iter().map(imaging::RegionImage::regions).collect();
        Ok((ImageFeatures { width: self.out_width(), tokens, rows: features }, VisionPass { setup, emb, inputs, caches, states, projector, regions }))
    }

    /// The backward of [`Self::encode_train`]: `d_rows` is the loss's gradient with respect to the pass's feature rows
    /// (`pass.regions() * out_width()` floats, in feature order). Accumulates the gradient of every differentiated
    /// tensor, and of the adapters' factors.
    pub fn backward(&self, g: &Gpu, pass: VisionPass, d_rows: &[f32]) -> Result<(), String> {
        let train = self.training()?;
        let want = pass.regions() * self.out_width();
        if d_rows.len() != want {
            return Err(format!("inkling2 vision: the backward takes {want} gradient values ({} rows of {}), got {}", pass.regions(), self.out_width(), d_rows.len()));
        }
        let VisionPass { setup, emb, inputs, caches, states, projector, regions } = pass;
        let (c, rows, patches) = (setup.sh.dim, setup.plan.rows, setup.plan.patches);
        let n = u64::from(rows) * u64::from(c);

        // Merger.
        let d_out = g.storage_init("vit.d_features", d_rows);
        let d_states = g.storage(u64::from(patches) * u64::from(c));
        g.submit(&[], &projector.backward_rows(g, &[&states], &d_out, Some(&[&d_states]), regions));

        let lowest = (0..self.cfg.num_hidden_layers).find(|&l| self.layer_trains(l));
        let embeddings = self.embeddings_train();
        let stop = if embeddings { Some(0) } else { lowest };
        if let Some(stop) = stop {
            let mut dx = g.storage(n);
            g.submit(&[&dx], &[scatter_rows(g, &self.permute, &setup.patch_idx, &d_states, &dx, patches, c, rows)]);
            let mut dx_next = g.storage(n);
            let rope = RopeTables { cos: &setup.cos, sin: &setup.sin, tmod: rows };
            let sb = VitBwdScratch::new(g, &setup.sh, rows, setup.max_span);
            let (scr_tmp, scores, x_out) = (g.storage(n), g.storage(u64::from(self.cfg.num_attention_heads) * u64::from(setup.max_span) * u64::from(setup.max_span)), g.storage(n));
            let mut work = train.saved.recomputes().then(|| VitBlockCache::new(g, &setup.sh, rows, setup.max_span));
            for l in (stop..self.cfg.num_hidden_layers).rev() {
                let bw = self.block_weights(l, &rope);
                let cache = match &mut work {
                    Some(w) => {
                        w.x_in = inputs[l as usize].clone();
                        let mut steps = Vec::new();
                        vit_block_fwd_cached(g, &self.ids, &train.bwd, &setup.sh, &bw, w, &x_out, rows, &setup.spans, &scr_tmp, &scores, &mut steps);
                        g.submit(&[&w.qkv], &steps);
                        &*w
                    }
                    None => &caches[l as usize],
                };
                let grads = self.block_grads(train, l);
                let mut steps = Vec::new();
                vit_block_bwd(g, &self.ids, &train.bwd, &setup.sh, &bw, &grads, cache, &dx, &dx_next, rows, &setup.spans, &sb, &mut steps);
                g.submit(&[&sb.d_qkv, &sb.d_qkv_pre], &steps);
                std::mem::swap(&mut dx, &mut dx_next);
            }
            if embeddings {
                self.embeddings_backward(g, train, &setup, &emb, &dx);
            }
        }
        self.project_adapters(g);
        Ok(())
    }

    /// The gradient buffers of layer `l`: each tensor's own, the adapters' dense buffer for an adapted weight, or the
    /// sink.
    fn block_grads<'a>(&'a self, t: &'a TowerTraining, l: u32) -> VitBlockGrads<'a> {
        let s = &self.store;
        let p = |tail: &str| t.grad(s, &layer_param(l, tail));
        VitBlockGrads {
            norm1_w: p("ln_1.weight"),
            norm1_b: p("ln_1.bias"),
            qkv_w: t.grad(s, &fused_qkv_name(l, "weight")),
            qkv_b: t.grad(s, &fused_qkv_name(l, "bias")),
            q_norm_w: None,
            q_norm_b: None,
            k_norm_w: None,
            k_norm_b: None,
            proj_w: p("self_attn.out_proj.weight"),
            proj_b: p("self_attn.out_proj.bias"),
            ls1: Some(p("ls_1.gamma")),
            norm2_w: p("ln_2.weight"),
            norm2_b: p("ln_2.bias"),
            fc1_w: p("mlp.c_fc.weight"),
            fc1_b: p("mlp.c_fc.bias"),
            fc2_w: p("mlp.c_proj.weight"),
            fc2_b: p("mlp.c_proj.bias"),
            ls2: Some(p("ls_2.gamma")),
        }
    }

    /// `ln_pre`, then the patch projection, the position table and the class token: `d_x0` is the gradient at
    /// `ln_pre`'s output.
    fn embeddings_backward(&self, g: &Gpu, t: &TowerTraining, s: &GroupSetup, emb: &DeviceBuffer, d_x0: &DeviceBuffer) {
        let c = s.sh.dim;
        let (rows, patches, images) = (s.plan.rows, s.plan.patches, s.plan.starts.len() as u32);
        let patch_dim = 3 * self.cfg.native_patch_size.pow(2);
        let ln = LayerNormIds::resolve(g, self.ids.layernorm, t.bwd.ln_stats, t.bwd.layernorm_dx);
        let (mean, inv) = (g.storage(u64::from(rows)), g.storage(u64::from(rows)));
        let d_emb = g.storage(u64::from(rows) * u64::from(c));
        g.submit(
            &[],
            &[
                model::block::ln_stats_fwd(g, &ln, emb, &mean, &inv, c, rows, s.sh.eps),
                g.step(t.bwd.ln_dgamma, &[d_x0, emb, &mean, &inv, t.grad(&self.store, &backbone("ln_pre.weight"))], &[c, rows], c),
                g.step(t.bwd.ln_dbeta, &[d_x0, t.grad(&self.store, &backbone("ln_pre.bias"))], &[c, rows], c),
                model::block::layernorm_dx_bwd(g, &ln, emb, self.store.w(&backbone("ln_pre.weight")), d_x0, &d_emb, c, rows, s.sh.eps),
            ],
        );
        let (d_patch, d_cls) = (g.storage(u64::from(patches) * u64::from(c)), g.storage(u64::from(images) * u64::from(c)));
        let mut steps = vec![gather_rows(g, &self.permute, &s.patch_idx, &d_emb, &d_patch, patches, c), gather_rows(g, &self.permute, &s.cls_idx, &d_emb, &d_cls, images, c)];
        // The patch embedding's weight, the position table (the transpose of the bilinear taps), and the class row,
        // which is the sum of two parameters and so gives both the column sum of the class rows' gradient.
        if let Some(gw) = self.store.grad.get(&backbone("embeddings.patch_embedding.weight")) {
            steps.push(g.step(t.bwd.matmul_dw, &[&d_patch, &s.pix, gw], &[patches, patch_dim, c], c * patch_dim));
        }
        if let Some(gp) = self.store.grad.get(&backbone("embeddings.position_embedding")) {
            steps.push(s.taps.table_grad_step(g, &self.pos_kernels, &d_patch, gp, c).expect("train_pipelines registers rows_gather_weighted_bwd"));
        }
        for name in ["embeddings.class_embedding", "embeddings.cls_position_embedding"] {
            if let Some(gc) = self.store.grad.get(&backbone(name)) {
                steps.push(g.step(t.bwd.bias_grad, &[&d_cls, gc], &[images, c], c));
            }
        }
        g.submit(&[], &steps);
    }

    /// [`Self::embed_prompt`] for training: the expanded prompt, the features spliced as [`PromptEmbeds`], and the
    /// pass whose backward takes the decoder's gradient at those rows (`PromptEmbeds::take_row_gradients`).
    pub fn embed_prompt_train(
        &self,
        gpu: &Gpu,
        tiling: &imaging::RegionTiling,
        prompt: &[u32],
        image_token: u32,
        images: &[&[u8]],
    ) -> Result<(Vec<u32>, PromptEmbeds, VisionPass), String> {
        let regions: Vec<imaging::RegionImage> = images.iter().enumerate().map(|(i, bytes)| tiling.preprocess_bytes(bytes).map_err(|e| format!("image {i}: {e}"))).collect::<Result<_, _>>()?;
        let counts: Vec<usize> = regions.iter().map(imaging::RegionImage::regions).collect();
        let expanded = expand_placeholders(prompt, image_token, &counts)?;
        let (features, pass) = self.encode_train(gpu, &regions)?;
        let embeds = splice_embeds(&expanded, image_token, &features)?;
        Ok((expanded, embeds, pass))
    }
}
