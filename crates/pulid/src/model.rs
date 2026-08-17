// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The two PuLID graphs: [`IdFormer`] (ArcFace ‖ EVA-CLIP → 32 projected ID
//! tokens) and [`PulidCa`] (the cross-attention injected into the FLUX.1
//! residual stream).
//!
//! **No kernel and no shared block is added.** Everything here is composed from
//! kernels that already existed:
//!
//! | reference op | brain |
//! |---|---|
//! | `nn.LayerNorm` | `layernorm` / `layernorm_rows` via `model::block::ln_variant` — the shared selection rule, keyed on the queried `DeviceCaps` |
//! | `nn.Linear` | `matmul` / `matmul_gemv` / `matmul_reg3` via `model::block::gemm_variant` — the same tier and the same selection rule `crates/flux1` uses — then `bias_add` |
//! | `nn.LeakyReLU()` | `leaky_relu` (slope 0.01) |
//! | `nn.GELU()` | `gelu_erf` — the **erf** form, *not* `gelu` (tanh): the reference constructs `nn.GELU()`, whose default is `approximate='none'` |
//! | Perceiver attention | `attn_{scores,softmax,apply}_cross` |
//! | residual add | `add2` |
//! | `img + id_weight * ca(...)` | `axpy` (`out += s·in`, `s` = `id_weight`) |
//! | `cat` of a parameter into a slab | `region_copy` |
//!
//! Two layout facts make the cross trio a drop-in with no repacking:
//!
//! * `to_kv` emits `[rows, 2·inner]` and the reference splits it with
//!   `chunk(2, -1)`, so k sits at column 0 and v at column `inner` — exactly
//!   the fused-KV layout `attn_scores_cross` / `attn_apply_cross` bind
//!   (`kv_stride = 2·inner`, `k_off = 0`, `v_off = inner`).
//! * the reference scales q and k by `dim_head**-0.25` each, so the product is
//!   `1/sqrt(dim_head)` — the scale `attn_scores_cross` applies itself.
//!
//! Concatenations are expressed as **row ranges of one buffer**, not a copy:
//! `cat(id_tokens, mapped_vit)` is written by the two producing linears at row
//! offsets 0 and `num_id_token`, and `cat(norm1(ctx), norm2(latents))` by the
//! two LayerNorms at row offsets 0 and `ctx_rows`. Every such offset is a whole
//! number of 1024- or 3072-float rows, clearing the 64-float (256-byte)
//! storage-binding alignment by construction.

use std::cell::Cell;
use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};
use paramstore::{ParamStore, Role};

use crate::config::PulidConfig;
use crate::import::Tensors;

/// The kernels PuLID dispatches. When PuLID rides a FLUX.1 forward the device
/// handle is built from [`joint_kernels`] instead, and indices are resolved by
/// name — see [`Ki::resolve`].
pub const KERNELS: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("layernorm_rows", kernels::LAYERNORM_ROWS),
    ("matmul", kernels::MATMUL),
    // The SAME three-kernel fp32 GEMM tier `crates/flux1` registers, dispatched
    // through the SAME rule (`model::block::gemm_variant`). Two of PuLID's
    // linears are skinny-M — the `id_map` chain runs at m = 1 and every
    // injected `to_kv` at m = 32 (the ID tokens) — which is exactly the regime
    // `matmul_gemv` exists for; without it each of those re-streams the whole
    // weight once per output element. Registering `matmul_reg3` rather than
    // `matmul_reg2` also keeps `joint_kernels` from compiling a second
    // register-tiled matmul next to flux1's.
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("bias_add", kernels::BIAS_ADD),
    ("leaky_relu", kernels::LEAKY_RELU),
    ("gelu_erf", kernels::GELU_ERF),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    // The coalesced score path `model::rowemit` dispatches: K transposed to
    // key-minor once, then a score sweep whose lanes read contiguous keys.
    ("kv_k_headt", kernels::KV_K_HEADT),
    ("attn_scores_cross_kt", kernels::ATTN_SCORES_CROSS_KT),
    ("add2", kernels::ADD2),
    ("axpy", kernels::AXPY),
    ("region_copy", kernels::REGION_COPY),
];

/// `flux1::KERNELS` followed by whatever PuLID needs on top, de-duplicated by
/// name and **without moving any flux1 index** — so one `Gpu` built from this
/// list runs both graphs, and a PuLID step can be appended straight into
/// `Flux1Model`'s dispatch list (a `Step` is only meaningful to the handle that
/// created it, so there is no second device and no second kernel set).
///
/// De-duplication matters beyond tidiness: `gpu_core::upgrade` resolves its
/// slow→fast redirects by the FIRST matching name, so a second
/// `("layernorm", …)` entry would be a pipeline that silently never upgrades.
///
/// Built once into a `OnceLock` and returned as a `'static` slice: kernel sets
/// are identified by slice ADDRESS (`gpu_core::testgpu::dev`'s pool key and
/// `Gpu::new_like`'s identity), so a fresh `Vec` per call would build a fresh
/// device per call.
pub fn joint_kernels() -> &'static [(&'static str, &'static str)] {
    static JOINT: std::sync::OnceLock<Vec<(&'static str, &'static str)>> =
        std::sync::OnceLock::new();
    JOINT.get_or_init(|| {
        let mut v: Vec<(&'static str, &'static str)> = flux1::KERNELS.to_vec();
        for (n, s) in KERNELS {
            if !v.iter().any(|(m, _)| m == n) {
                v.push((n, s));
            }
        }
        v
    })
}

/// Pipeline indices, resolved by name from whatever list the handle was built
/// with.
#[derive(Clone, Copy, Debug)]
pub struct Ki {
    /// The shared row-range emitter's kernels — hoisted to `model::rowemit` so
    /// the second IP-Adapter-lineage model (InstantID's `Resampler`) does not
    /// carry a second copy of this dispatch logic.
    row: model::rowemit::RowKernels,
    // PuLID-only, dispatched directly rather than through the emitter.
    leaky: usize,
    add2: usize,
    axpy: usize,
    gelu: usize,
}

impl Ki {
    /// Resolve against the pipeline list the [`Gpu`] was constructed with.
    /// Panics naming the kernel if one is absent — a wrong index is silently
    /// wrong output, not a crash.
    pub fn resolve(names: &[(&str, &str)]) -> Ki {
        let f = |k: &str| {
            names
                .iter()
                .position(|(n, _)| *n == k)
                .unwrap_or_else(|| panic!("pulid: the Gpu was built without the `{k}` kernel"))
        };
        Ki {
            row: model::rowemit::RowKernels::resolve("pulid", names),
            leaky: f("leaky_relu"),
            add2: f("add2"),
            axpy: f("axpy"),
            gelu: f("gelu_erf"),
        }
    }
}

use model::rowemit::{fbits as f, rows};

/// The shared row-range emitter, now `model::rowemit::RowEmit`.
///
/// It used to live here. It was hoisted because InstantID's `Resampler` is the
/// same `PerceiverAttention` from the same IP-Adapter lineage — `cat(x, latents)`
/// as one buffer written at row offsets, `to_q` over the latent rows, `to_kv`
/// over the whole thing — and a second copy of the row arithmetic and the
/// fused-kv strides is exactly the drift AGENTS.md's one-implementation rule
/// targets.
type Emit<'a> = model::rowemit::RowEmit<'a>;

/// A named stage snapshot: the buffer it lives in, its float offset and
/// length, and the step index just after the step that wrote it.
struct Tap {
    name: String,
    buf: DeviceBuffer,
    off: usize,
    len: usize,
    step: usize,
}

// ---------------------------------------------------------------------------
// IDFormer
// ---------------------------------------------------------------------------

/// The ID encoder: `(id_cond, id_vit_hidden[5]) -> [num_queries, output_dim]`.
///
/// `id_cond` is `cat(ArcFace 512, L2-normalised EVA-CLIP cls 768)` and
/// `id_vit_hidden[j]` the EVA-CLIP tower's block `4j+3` output — both produced
/// by brain's own parity-gated `arcface` and `clip` crates
/// (`clip::EvaVisionConfig::PULID_TAPS`); this graph adds no second copy of
/// either.
pub struct IdFormer {
    gpu: Gpu,
    pub cfg: PulidConfig,
    ps: ParamStore,
    ki: Ki,
    id_cond: DeviceBuffer,
    vit: Vec<DeviceBuffer>,
    /// `[num_id_token, dim]`-sized scratch for the 1-row `id_map` chain (its
    /// last linear emits `num_id_token * dim` from ONE row).
    s0: DeviceBuffer,
    s1: DeviceBuffer,
    /// `[vit_tokens, dim]` scratch for the 5 per-scale mapping MLPs.
    t0: DeviceBuffer,
    t1: DeviceBuffer,
    /// `cat(id_tokens, mapped_vit)` — `[ctx_rows, dim]`.
    ctxf: DeviceBuffer,
    /// Ping-pong latents `[latent_rows, dim]`.
    lat_a: DeviceBuffer,
    lat_b: DeviceBuffer,
    /// `cat(norm1(ctx), norm2(latents))` — `[kv_rows, dim]`.
    nkv: DeviceBuffer,
    q: DeviceBuffer,
    kv: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    /// `[inner, t_enc]` key-minor K for the coalesced score path.
    xkt: DeviceBuffer,
    actx: DeviceBuffer,
    aout: DeviceBuffer,
    fh: DeviceBuffer,
    fg: DeviceBuffer,
    fo: DeviceBuffer,
    out: DeviceBuffer,
    steps: Vec<Step>,
    taps: Vec<Tap>,
}

impl IdFormer {
    /// Build on a handle whose pipeline list is `kernel_names` (either
    /// [`KERNELS`] or [`joint_kernels`]).
    pub fn new_on(gpu: Gpu, cfg: PulidConfig, kernel_names: &[(&str, &str)], w: Tensors) -> IdFormer {
        let ki = Ki::resolve(kernel_names);
        let init: HashMap<String, Vec<f32>> = w.into_iter().map(|(n, (_, d))| (n, d)).collect();
        let roles: Vec<(String, usize, Role)> = cfg
            .encoder_manifest()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &init);

        let d = cfg.dim as u64;
        let (lr, cr, kr) = (cfg.latent_rows() as u64, cfg.ctx_rows() as u64, cfg.kv_rows() as u64);
        let vt = cfg.vit_tokens as u64;
        let inner = cfg.inner_dim() as u64;
        let ff = cfg.ff_hidden() as u64;
        let nid = cfg.num_id_token as u64;
        let mut m = IdFormer {
            id_cond: gpu.storage(cfg.id_cond_dim as u64),
            vit: (0..cfg.scales).map(|_| gpu.storage(vt * d)).collect(),
            s0: gpu.storage(nid * d),
            s1: gpu.storage(nid * d),
            t0: gpu.storage(vt * d),
            t1: gpu.storage(vt * d),
            ctxf: gpu.storage(cr * d),
            lat_a: gpu.storage(lr * d),
            lat_b: gpu.storage(lr * d),
            nkv: gpu.storage(kr * d),
            q: gpu.storage(lr * inner),
            kv: gpu.storage(kr * 2 * inner),
            scores: gpu.storage(cfg.heads as u64 * lr * kr),
            probs: gpu.storage(cfg.heads as u64 * lr * kr),
            xkt: gpu.storage(inner * kr),
            actx: gpu.storage(lr * inner),
            aout: gpu.storage(lr * d),
            fh: gpu.storage(lr * ff),
            fg: gpu.storage(lr * ff),
            fo: gpu.storage(lr * d),
            out: gpu.storage(cfg.num_queries as u64 * cfg.output_dim as u64),
            gpu,
            cfg,
            ps,
            ki,
            steps: Vec::new(),
            taps: Vec::new(),
        };
        let (steps, taps) = m.build_steps();
        m.steps = steps;
        m.taps = taps;
        m
    }

    /// Build with a device handle carrying PuLID's own kernel set.
    pub fn new(gpu: Gpu, cfg: PulidConfig, w: Tensors) -> IdFormer {
        IdFormer::new_on(gpu, cfg, KERNELS, w)
    }

    fn w(&self, n: &str) -> &DeviceBuffer {
        self.ps.w(n)
    }

    /// One `Linear -> LN -> LeakyReLU -> Linear -> LN -> LeakyReLU -> Linear`
    /// mapping MLP over `m` rows, the final linear writing `y` at row `yr0`.
    #[allow(clippy::too_many_arguments)]
    fn mapping(
        &self,
        e: &Emit,
        s: &mut Vec<Step>,
        pfx: &str,
        x: &DeviceBuffer,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
        y: &DeviceBuffer,
        yr0: usize,
        m: usize,
        k0: usize,
        n2: usize,
    ) {
        let d = self.cfg.dim;
        let (lin0w, lin0b) = (self.w(&format!("{pfx}.lin0.weight")), self.w(&format!("{pfx}.lin0.bias")));
        e.linear(s, x, 0, lin0w, Some(lin0b), a, 0, m, k0, d);
        e.ln(s, a, 0, self.w(&format!("{pfx}.ln0.weight")), self.w(&format!("{pfx}.ln0.bias")), b, 0, m, d);
        s.push(self.gpu.step(self.ki.leaky, &[b, a], &[(m * d) as u32, f(self.cfg.leaky_slope)], (m * d) as u32));
        let (lin1w, lin1b) = (self.w(&format!("{pfx}.lin1.weight")), self.w(&format!("{pfx}.lin1.bias")));
        e.linear(s, a, 0, lin1w, Some(lin1b), b, 0, m, d, d);
        e.ln(s, b, 0, self.w(&format!("{pfx}.ln1.weight")), self.w(&format!("{pfx}.ln1.bias")), a, 0, m, d);
        s.push(self.gpu.step(self.ki.leaky, &[a, b], &[(m * d) as u32, f(self.cfg.leaky_slope)], (m * d) as u32));
        // The last linear writes DIRECTLY into the concatenation target. For
        // `id_map` that is one row of `num_id_token * dim` at row 0 of `ctxf`,
        // which IS the reference's `reshape(-1, num_id_token, dim)`.
        let (lin2w, lin2b) = (self.w(&format!("{pfx}.lin2.weight")), self.w(&format!("{pfx}.lin2.bias")));
        e.linear(s, b, 0, lin2w, Some(lin2b), y, yr0, m, d, n2);
    }

    fn build_steps(&self) -> (Vec<Step>, Vec<Tap>) {
        let c = &self.cfg;
        let e = Emit::new(&self.gpu, self.ki.row, c.eps);
        let (d, nid, lr, cr, kr) = (c.dim, c.num_id_token, c.latent_rows(), c.ctx_rows(), c.kv_rows());
        let mut s: Vec<Step> = Vec::new();
        let mut taps: Vec<Tap> = Vec::new();
        let tap = |taps: &mut Vec<Tap>, s: &[Step], name: &str, buf: &DeviceBuffer, off: usize, len: usize| {
            taps.push(Tap { name: name.into(), buf: buf.clone(), off, len, step: s.len() });
        };

        // id_cond -> `num_id_token` ID tokens, written into rows 0..nid of ctxf.
        self.mapping(&e, &mut s, "id_map", &self.id_cond, &self.s0, &self.s1, &self.ctxf, 0, 1, c.id_cond_dim, nid * d);
        tap(&mut taps, &s, "id_tokens", &self.ctxf, 0, nid * d);

        // latents = cat(learned latents, ID tokens)
        e.copy_rows(&mut s, self.w("latents"), 0, &self.lat_a, 0, c.num_queries, d);
        e.copy_rows(&mut s, &self.ctxf, 0, &self.lat_a, c.num_queries, nid, d);
        tap(&mut taps, &s, "latents_in", &self.lat_a, 0, lr * d);

        let lps = c.layers_per_scale();
        for i in 0..c.scales {
            // ctx = cat(id_tokens, mapping_i(vit_hidden[i])): the mapping's last
            // linear writes STRAIGHT into rows nid.. of ctxf, so the concat is
            // an output offset, not a copy.
            let vit = self.vit[i].clone();
            self.mapping(&e, &mut s, &format!("map{i}"), &vit, &self.t0, &self.t1, &self.ctxf, nid, c.vit_tokens, d, d);
            tap(&mut taps, &s, &format!("map{i}_out"), &self.ctxf, nid * d, c.vit_tokens * d);
            if i == 0 {
                tap(&mut taps, &s, "ctx0", &self.ctxf, 0, cr * d);
            }
            for j in 0..lps {
                let l = i * lps + j;
                let b = format!("layers.{l}");
                // kv input = cat(norm1(ctx), norm2(latents)) in ONE buffer
                e.ln(&mut s, &self.ctxf, 0, self.w(&format!("{b}.attn.norm1.weight")), self.w(&format!("{b}.attn.norm1.bias")), &self.nkv, 0, cr, d);
                e.ln(&mut s, &self.lat_a, 0, self.w(&format!("{b}.attn.norm2.weight")), self.w(&format!("{b}.attn.norm2.bias")), &self.nkv, cr, lr, d);
                e.linear(&mut s, &self.nkv, cr, self.w(&format!("{b}.attn.to_q.weight")), None, &self.q, 0, lr, d, c.inner_dim());
                e.linear(&mut s, &self.nkv, 0, self.w(&format!("{b}.attn.to_kv.weight")), None, &self.kv, 0, kr, d, 2 * c.inner_dim());
                e.cross_attn(&mut s, &self.q, 0, &self.kv, &self.scores, &self.probs, &self.actx, &self.xkt, c.heads, c.dim_head, lr, kr);
                e.linear(&mut s, &self.actx, 0, self.w(&format!("{b}.attn.to_out.weight")), None, &self.aout, 0, lr, c.inner_dim(), d);
                s.push(self.gpu.step(self.ki.add2, &[&self.lat_a, &self.aout, &self.lat_b], &[(lr * d) as u32], (lr * d) as u32));
                tap(&mut taps, &s, &format!("layer{l}_attn"), &self.lat_b, 0, lr * d);

                e.ln(&mut s, &self.lat_b, 0, self.w(&format!("{b}.ff.norm.weight")), self.w(&format!("{b}.ff.norm.bias")), &self.fo, 0, lr, d);
                e.linear(&mut s, &self.fo, 0, self.w(&format!("{b}.ff.w1.weight")), None, &self.fh, 0, lr, d, c.ff_hidden());
                s.push(self.gpu.step(self.ki.gelu, &[&self.fh, &self.fg], &[(lr * c.ff_hidden()) as u32], (lr * c.ff_hidden()) as u32));
                e.linear(&mut s, &self.fg, 0, self.w(&format!("{b}.ff.w2.weight")), None, &self.fo, 0, lr, c.ff_hidden(), d);
                s.push(self.gpu.step(self.ki.add2, &[&self.lat_b, &self.fo, &self.lat_a], &[(lr * d) as u32], (lr * d) as u32));
                tap(&mut taps, &s, &format!("layer{l}_ff"), &self.lat_a, 0, lr * d);
            }
        }
        // Only the first `num_queries` rows survive. `proj_out` was transposed
        // at import, so this is the ordinary `x @ Wᵀ`.
        e.linear(&mut s, &self.lat_a, 0, self.w("proj_out"), None, &self.out, 0, c.num_queries, d, c.output_dim);
        tap(&mut taps, &s, "id_embedding", &self.out, 0, c.num_queries * c.output_dim);
        (s, taps)
    }

    /// Upload the ID condition and the 5 tapped EVA-CLIP hidden states.
    pub fn set_inputs(&self, id_cond: &[f32], vit_hidden: &[Vec<f32>]) {
        let c = &self.cfg;
        assert_eq!(id_cond.len(), c.id_cond_dim, "id_cond width");
        assert_eq!(vit_hidden.len(), c.scales, "expected {} tapped hidden states", c.scales);
        self.gpu.write_f32(&self.id_cond, id_cond);
        for (b, h) in self.vit.iter().zip(vit_hidden) {
            assert_eq!(h.len(), c.vit_tokens * c.dim, "vit hidden size");
            self.gpu.write_f32(b, h);
        }
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
    }

    /// The 32 projected ID tokens `[num_queries, output_dim]`, ready to be
    /// handed to [`PulidCa`].
    pub fn read_id_embedding(&self) -> Vec<f32> {
        self.gpu.read(&self.out, self.cfg.num_queries * self.cfg.output_dim)
    }

    pub fn tap_names(&self) -> Vec<String> {
        self.taps.iter().map(|t| t.name.clone()).collect()
    }

    /// Re-run the forward up to the step that produced `name`, then read it.
    ///
    /// Every scratch buffer here is REUSED across layers, so a stage is only
    /// readable immediately after the step that wrote it — hence the replay
    /// rather than one forward and 20 reads. Cheap: the whole graph is ~140 M
    /// params and a few hundred dispatches.
    pub fn read_tap(&self, name: &str) -> Vec<f32> {
        let t = self.taps.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("pulid: no tap `{name}`"));
        self.gpu.submit(&[], &self.steps[..t.step]);
        self.gpu.read(&t.buf, t.off + t.len)[t.off..].to_vec()
    }
}

// ---------------------------------------------------------------------------
// PerceiverAttentionCA
// ---------------------------------------------------------------------------

/// The `n_ca` cross-attention modules injected into the FLUX.1 backbone.
///
/// One object holds all of them (they share the scratch — only one fires at a
/// time) plus the device-resident ID tokens. [`PulidCa::inject_steps`] appends
/// one module's dispatches to a caller's step list, so the whole conditioned
/// forward is still ONE submit.
pub struct PulidCa {
    gpu: Gpu,
    pub cfg: PulidConfig,
    ps: ParamStore,
    ki: Ki,
    n_ca: usize,
    max_img: usize,
    /// `[num_queries, output_dim]` — the IDFormer output, uploaded once.
    id: DeviceBuffer,
    /// `norm1(id)` `[num_queries, output_dim]`.
    idn: DeviceBuffer,
    /// `norm2(img)` `[max_img, ca_dim]`.
    imn: DeviceBuffer,
    q: DeviceBuffer,
    kv: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    /// `[inner, t_enc]` key-minor K for the coalesced score path.
    xkt: DeviceBuffer,
    ctx: DeviceBuffer,
    out: DeviceBuffer,
    /// Image rows of the most recent `inject_steps`, so [`PulidCa::read_stage`]
    /// knows how much of the scratch is live.
    last_n_img: Cell<usize>,
}

impl PulidCa {
    /// `max_img` is the largest image-token count a forward will inject into.
    pub fn new_on(
        gpu: Gpu,
        cfg: PulidConfig,
        kernel_names: &[(&str, &str)],
        n_ca: usize,
        max_img: usize,
        w: Tensors,
    ) -> PulidCa {
        let ki = Ki::resolve(kernel_names);
        let init: HashMap<String, Vec<f32>> = w.into_iter().map(|(n, (_, d))| (n, d)).collect();
        let roles: Vec<(String, usize, Role)> = cfg
            .ca_manifest(n_ca)
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &init);
        let (nq, kvd, dm, inner) =
            (cfg.num_queries as u64, cfg.output_dim as u64, cfg.ca_dim as u64, cfg.ca_inner_dim() as u64);
        let mi = max_img as u64;
        PulidCa {
            id: gpu.storage(nq * kvd),
            idn: gpu.storage(nq * kvd),
            imn: gpu.storage(mi * dm),
            q: gpu.storage(mi * inner),
            kv: gpu.storage(nq * 2 * inner),
            scores: gpu.storage(cfg.ca_heads as u64 * mi * nq),
            probs: gpu.storage(cfg.ca_heads as u64 * mi * nq),
            xkt: gpu.storage(inner * nq),
            ctx: gpu.storage(mi * inner),
            out: gpu.storage(mi * dm),
            last_n_img: Cell::new(0),
            gpu,
            cfg,
            ps,
            ki,
            n_ca,
            max_img,
        }
    }

    pub fn new(gpu: Gpu, cfg: PulidConfig, n_ca: usize, max_img: usize, w: Tensors) -> PulidCa {
        PulidCa::new_on(gpu, cfg, KERNELS, n_ca, max_img, w)
    }

    pub fn n_ca(&self) -> usize {
        self.n_ca
    }

    /// Upload the IDFormer output. Once per identity, not once per step.
    pub fn set_id(&self, id: &[f32]) {
        assert_eq!(id.len(), self.cfg.num_queries * self.cfg.output_dim, "id token slab size");
        self.gpu.write_f32(&self.id, id);
    }

    /// Append module `k`'s dispatches: `x[r0..r0+n_img] += id_weight ·
    /// ca_k(id, x[r0..r0+n_img])`, in place on the caller's residual slab.
    ///
    /// `x` is `[.., ca_dim]` row-major; `r0` is a ROW offset (for FLUX.1's
    /// joint slab that is the text length, since the image rows follow the text
    /// rows). In-place is correct here because the backbone is inference-only —
    /// a training-mode forward would need the SSA fresh-buffer discipline.
    #[allow(clippy::too_many_arguments)]
    pub fn inject_steps(&self, s: &mut Vec<Step>, k: usize, x: &DeviceBuffer, r0: usize, n_img: usize, id_weight: f32) {
        assert!(k < self.n_ca, "pulid: ca index {k} >= {}", self.n_ca);
        assert!(n_img <= self.max_img, "pulid: {n_img} image rows > max_img {}", self.max_img);
        self.last_n_img.set(n_img);
        let c = &self.cfg;
        let e = Emit::new(&self.gpu, self.ki.row, c.eps);
        let (dm, kvd, inner, nq) = (c.ca_dim, c.output_dim, c.ca_inner_dim(), c.num_queries);
        let b = format!("ca.{k}");
        let w = |n: &str| self.ps.w(&format!("{b}.{n}"));
        // norm1 over the ID tokens (kv side), norm2 over the image rows (q side)
        e.ln(s, &self.id, 0, w("norm1.weight"), w("norm1.bias"), &self.idn, 0, nq, kvd);
        e.ln(s, x, r0, w("norm2.weight"), w("norm2.bias"), &self.imn, 0, n_img, dm);
        e.linear(s, &self.imn, 0, w("to_q.weight"), None, &self.q, 0, n_img, dm, inner);
        e.linear(s, &self.idn, 0, w("to_kv.weight"), None, &self.kv, 0, nq, kvd, 2 * inner);
        e.cross_attn(s, &self.q, 0, &self.kv, &self.scores, &self.probs, &self.ctx, &self.xkt, c.ca_heads, c.ca_dim_head, n_img, nq);
        e.linear(s, &self.ctx, 0, w("to_out.weight"), None, &self.out, 0, n_img, inner, dm);
        // `axpy` Params: [n, s] — `out += s * in`, out bound to the image rows.
        s.push(self.gpu.step_sliced(
            self.ki.axpy,
            &[x, &self.out],
            &[rows(r0, n_img, dm), (0, 0)],
            &[(n_img * dm) as u32, f(id_weight)],
            (n_img * dm) as u32,
        ));
    }

    /// Standalone evaluation of module `k` on a host-supplied image slab —
    /// the parity-test entry point. Returns `img + id_weight * ca_k(id, img)`.
    pub fn apply_host(&self, k: usize, img: &[f32], id_weight: f32) -> Vec<f32> {
        let dm = self.cfg.ca_dim;
        assert_eq!(img.len() % dm, 0, "img not a multiple of ca_dim");
        let n_img = img.len() / dm;
        let x = self.gpu.storage((n_img * dm) as u64);
        self.gpu.write_f32(&x, img);
        let mut s = Vec::new();
        self.inject_steps(&mut s, k, &x, 0, n_img, id_weight);
        self.gpu.submit(&[], &s);
        self.gpu.read(&x, n_img * dm)
    }

    /// Internal stages of the module that fired most recently — the parity
    /// ladder's per-stage taps. Only one module runs at a time and the scratch
    /// is not touched afterwards, so these are live until the next injection.
    ///
    /// Names: `norm1_id`, `norm2_img`, `q`, `kv`, `ctx`.
    pub fn read_stage(&self, name: &str) -> Vec<f32> {
        let c = &self.cfg;
        let n = self.last_n_img.get();
        let (b, len) = match name {
            "norm1_id" => (&self.idn, c.num_queries * c.output_dim),
            "norm2_img" => (&self.imn, n * c.ca_dim),
            "q" => (&self.q, n * c.ca_inner_dim()),
            "kv" => (&self.kv, c.num_queries * 2 * c.ca_inner_dim()),
            "ctx" => (&self.ctx, n * c.ca_inner_dim()),
            _ => panic!("pulid: no CA stage `{name}`"),
        };
        self.gpu.read(b, len)
    }
}
