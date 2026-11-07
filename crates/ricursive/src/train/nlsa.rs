// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Native latent sparse attention (N-LSA) on Ripple's global layers: each query reads only its selection, in time
//! and memory linear in the context, and the gradients flow exactly through the selected entries.
//!
//! Swedish Embedded AB implements long-context models trained with sparse attention for clients whose sequences
//! outgrow quadratic attention. If your team needs expertise in continuing the training of a dense model with
//! sparse attention then you can procure our services by sending an email to info@swedishembedded.com.
//!
//! The machinery is the shared one ([`model::lsa::native`]); this file only says where Ripple plugs in. The global
//! layers attend NoPE: their queries and keys leave the QK norm unrotated, so the selector reads them as they
//! are (`Rope::None`) and the index space is the attention's own, `n_kv_heads * head_dim` wide. The sliding
//! layers already read a window and stay dense. A selection is made each forward on the device from the layer's own
//! queries and keys, held fixed (no gradient flows through the choice), and kept for the backward's recompute of
//! the layer.

use gpu_core::DeviceBuffer;
use model::block::Gqa;
use model::lsa::device::Rope;
use model::lsa::native::{self, NativeBackward, NativeKernels, NativeSelector, NativeShape, SelectKernels};
use model::lsa::LsaConfig;

use super::Ripple;
use crate::config::LayerType;

/// Native sparse attention as the trainer holds it.
pub(super) struct NativeLsa {
    pub cfg: LsaConfig,
    pub kernels: NativeKernels,
    /// `None` for the window-only baseline, whose selection never changes and is packed once.
    pub selector: Option<(NativeSelector, SelectKernels)>,
    /// One index per layer (a one-word placeholder for the layers that read densely): filled by the forward, read by
    /// the backward.
    pub index: Vec<DeviceBuffer>,
    /// The row log-sum-exp and the backward's `dO . O`; the backward's recompute of a layer rewrites them.
    pub stats: DeviceBuffer,
}

impl Ripple {
    /// Train (and run) the global layers with native sparse attention: each query attends its selection
    /// ([`LsaConfig`]: window, sinks and the top-k clusters the index finds), exactly as dense attention restricted
    /// to those keys. `long_range: false` keeps only the window and sinks, the baseline under the same machinery.
    ///
    /// The kernels need workgroup barriers, so this runs on a GPU or the software Vulkan adapter and is refused by
    /// name where they are not available.
    pub fn enable_native_lsa(&mut self, cfg: LsaConfig, long_range: bool) -> Result<(), String> {
        let c = &self.cfg.model;
        if !(self.shard.start..self.shard.end).any(|l| c.layer_type(l) == LayerType::Full) {
            return Err("ricursive native sparse attention: this build has no global layer".into());
        }
        let cfg = LsaConfig { index_dim: c.kv_dim(), ..cfg };
        cfg.validate()?;
        let name = |k: usize| {
            let n = native::KERNELS[k].0;
            self.gpu.kernel_index(n).ok_or_else(|| format!("ricursive native sparse attention: kernel {n} is not registered"))
        };
        let kernels = NativeKernels { fwd: name(0)?, bwd_dq: name(1)?, bwd_dkv: name(2)? };
        let shape = self.native_shape(&cfg);
        let index: Vec<DeviceBuffer> = (0..c.num_hidden_layers).map(|l| if c.layer_type(l) == LayerType::Full && self.shard.owns(l) { self.gpu.storage(shape.index_words()) } else { self.gpu.storage(1) }).collect();
        let stats = self.gpu.storage(shape.stats_words());
        let selector = if long_range {
            let pipe = |n: &str| self.gpu.kernel_index(n).ok_or_else(|| format!("ricursive native sparse attention: kernel {n} is not registered"));
            let [vectors, read, ward] = model::lsa::device::KERNELS.map(|(n, _)| n);
            let [scan_block, scan_add, sort_hist, sort_scatter] = gpu_core::sort::KERNELS.map(|(n, _)| n);
            let sort = gpu_core::sort::SortKernels { scan: gpu_core::sort::ScanKernels { scan_block: pipe(scan_block)?, scan_add: pipe(scan_add)? }, sort_hist: pipe(sort_hist)?, sort_scatter: pipe(sort_scatter)? };
            let ids = SelectKernels { index_vectors: pipe(vectors)?, read: pipe(read)?, ward: pipe(ward)?, pairs: name(3)?, offsets: name(4)?, sort };
            Some((NativeSelector::new(&self.gpu, cfg, shape, Rope::None)?, ids))
        } else {
            let packed = shape.pack(&vec![native::window_selection(&cfg, self.t as usize); self.b as usize])?;
            for l in (self.shard.start..self.shard.end).filter(|&l| c.layer_type(l) == LayerType::Full) {
                self.gpu.write(&index[l], &packed);
            }
            None
        };
        self.nlsa = Some(NativeLsa { cfg, kernels, selector, index, stats });
        self.record();
        Ok(())
    }

    /// The attention shape of a global layer under `cfg`.
    pub(super) fn native_shape(&self, cfg: &LsaConfig) -> NativeShape {
        let (c, ga) = (&self.cfg.model, self.gqa());
        NativeShape { b: ga.b, t: ga.t, n_heads: ga.n_heads, n_kv_heads: ga.n_kv_heads, head_dim: c.head_dim as u32, width: native::row_width(cfg) }
    }

    pub(super) fn gqa(&self) -> Gqa {
        let c = &self.cfg.model;
        Gqa { b: self.b, t: self.t, n_heads: c.num_attention_heads as u32, n_kv_heads: c.num_key_value_heads as u32, head_dim: c.head_dim as u32 }
    }

    /// The forward of global layer `l`'s attention over its selection: the selection first (a recompute reuses the one
    /// the forward made), then the sparse read into `ctx`. `None` when layer `l` reads densely.
    pub(super) fn native_fwd(&self, s: &mut Vec<gpu_core::Step>, l: usize, recompute: bool) -> Option<()> {
        let nl = self.nlsa.as_ref().filter(|_| self.cfg.model.layer_type(l) == LayerType::Full)?;
        let sc = &self.scratch;
        if let (Some((selector, ids)), false) = (&nl.selector, recompute) {
            s.extend(selector.record(&self.gpu, ids, &sc.q, &sc.k, &nl.index[l]).unwrap_or_else(|e| panic!("ricursive layer {l}: {e}")));
        }
        s.push(self.native_shape(&nl.cfg).forward(&self.gpu, &nl.kernels, &sc.q, &sc.k, &sc.v, &nl.index[l], &sc.ctx, &nl.stats).unwrap_or_else(|e| panic!("ricursive layer {l}: {e}")));
        Some(())
    }

    /// The backward of [`Self::native_fwd`], from the gradient of the context `d_ctx`.
    pub(super) fn native_bwd(&self, s: &mut Vec<gpu_core::Step>, l: usize, d_ctx: &DeviceBuffer) -> Option<()> {
        let nl = self.nlsa.as_ref().filter(|_| self.cfg.model.layer_type(l) == LayerType::Full)?;
        let (sc, g) = (&self.scratch, self.grad.as_ref().expect("a backward needs the training build"));
        let bufs = NativeBackward { q: &sc.q, k: &sc.k, v: &sc.v, ctx: &sc.ctx, d_ctx, index: &nl.index[l], stats: &nl.stats, d_q: &g.d_q, d_k: &g.d_k, d_v: &g.d_v };
        s.extend(self.native_shape(&nl.cfg).backward(&self.gpu, &nl.kernels, &bufs).unwrap_or_else(|e| panic!("ricursive layer {l}: {e}")));
        Some(())
    }
}
