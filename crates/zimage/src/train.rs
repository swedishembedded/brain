// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device (GPU) full-model training step for the Z-Image S³-DiT. The expensive
//! 34-block DiT core runs on the GPU through the persistent [`crate::devgrad::BlockDev`]
//! engine (forward-sweep saving per-block inputs, then a reverse backward-sweep);
//! the thin wrapper — timestep MLP, image/caption embedders, adaLN final layer,
//! flow-matching loss — runs on the host (it is a handful of small linears; you
//! would not shard it anyway). Gradients from every stage are assembled into the
//! same [`ModelGrads`] the host reference produces.
//!
//! This is the device counterpart of [`crate::modelgrad`]: `tests/device_train.rs`
//! checks its grads match the gradchecked host reference (cosine ~1) and that it
//! overfits one batch on the GPU. The block-stack orchestration is identical for
//! the 4-block small config and the 34-block 6B — it scales by construction.

use crate::devgrad::BlockDev;
use crate::grad::{Dims, Grads};
use crate::modelgrad::{dsilu, layernorm, layernorm_bwd, linb, linb_bwd, patchify, rmsnorm, rmsnorm_dw, silu, timestep_embedding, Cfg, ModelGrads, ModelWeights, TDIM, TH};

/// One training batch (host f64), mirroring the reference.
pub struct Batch {
    pub latent: Vec<f64>,
    pub cap: Vec<f64>,
    pub t: f64,
    pub img_cos: Vec<f64>,
    pub img_sin: Vec<f64>,
    pub cap_cos: Vec<f64>,
    pub cap_sin: Vec<f64>,
    pub target: Vec<f64>,
}

/// Persistent device trainer: owns the GPU block engine sized for the model.
pub struct DeviceTrainer {
    eng: BlockDev,
    cfg: Cfg,
}

fn to32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}
fn to64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

impl DeviceTrainer {
    pub fn new(cfg: Cfg) -> DeviceTrainer {
        let eng = BlockDev::new(cfg.ntot(), cfg.dim, cfg.nh);
        DeviceTrainer { eng, cfg }
    }

    fn dims(&self, t: usize) -> Dims {
        Dims::new(t, self.cfg.dim, self.cfg.nh)
    }

    /// Full forward+backward for one batch. Returns `(loss, grads)`.
    pub fn grads(&self, w: &ModelWeights, b: &Batch) -> (f64, ModelGrads) {
        let c = &self.cfg;
        let (dim, cdim, pd) = (c.dim, c.cdim(), c.patch_dim());
        let (n_img, ncap, ntot) = (c.n_img(), c.ncap, c.ntot());

        // ---- host: timestep conditioning ----
        let te = timestep_embedding(b.t * c.t_scale);
        let h0pre = linb(&te, 1, TDIM, &w.t0_w, &w.t0_b, TH);
        let h0: Vec<f64> = h0pre.iter().map(|&v| silu(v)).collect();
        let cvec = linb(&h0, 1, TH, &w.t2_w, &w.t2_b, cdim);
        let c32 = to32(&cvec);

        // ---- host: embedders ----
        let patches = patchify(&b.latent, c);
        let img = linb(&patches, n_img, pd, &w.xemb_w, &w.xemb_b, dim);
        let (capn, inv_capn) = rmsnorm(&b.cap, ncap, c.cap_feat_dim, &w.capn_w);
        let capt = linb(&capn, ncap, c.cap_feat_dim, &w.cap1_w, &w.cap1_b, dim);

        let (ic, is) = (to32(&b.img_cos), to32(&b.img_sin));
        let (cc, cs) = (to32(&b.cap_cos), to32(&b.cap_sin));

        // ---- device: block stack forward (save each block's input) ----
        let mut img32 = to32(&img);
        let mut noise_in = Vec::new();
        for bw in &w.noise_ref {
            noise_in.push(img32.clone());
            img32 = self.eng.forward(bw, self.dims(n_img), &img32, &c32, &ic, &is, true);
        }
        let mut capt32 = to32(&capt);
        let mut ctx_in = Vec::new();
        for bw in &w.ctx_ref {
            ctx_in.push(capt32.clone());
            capt32 = self.eng.forward(bw, self.dims(ncap), &capt32, &c32, &cc, &cs, false);
        }
        let mut uni32 = img32.clone();
        uni32.extend_from_slice(&capt32);
        let mut uni_cos = ic.clone();
        uni_cos.extend_from_slice(&cc);
        let mut uni_sin = is.clone();
        uni_sin.extend_from_slice(&cs);
        let mut main_in = Vec::new();
        for bw in &w.main {
            main_in.push(uni32.clone());
            uni32 = self.eng.forward(bw, self.dims(ntot), &uni32, &c32, &uni_cos, &uni_sin, true);
        }

        // ---- host: final layer + loss ----
        let uni = to64(&uni32);
        let silu_c: Vec<f64> = cvec.iter().map(|&v| silu(v)).collect();
        let adaln = linb(&silu_c, 1, cdim, &w.fadaln_w, &w.fadaln_b, dim);
        let scale: Vec<f64> = adaln.iter().map(|&v| 1.0 + v).collect();
        let (normed, inv_ln) = layernorm(&uni, ntot, dim);
        let mut scaled = vec![0f64; ntot * dim];
        for r in 0..ntot {
            for cc2 in 0..dim {
                scaled[r * dim + cc2] = normed[r * dim + cc2] * scale[cc2];
            }
        }
        let final_out = linb(&scaled, ntot, dim, &w.flin_w, &w.flin_b, pd);
        let pred = &final_out[..n_img * pd];
        let n = pred.len() as f64;
        let mut loss = 0.0;
        let mut dpred = vec![0f64; pred.len()];
        for i in 0..pred.len() {
            let e = pred[i] - b.target[i];
            loss += e * e / n;
            dpred[i] = 2.0 * e / n;
        }

        // ---- host: final layer backward ----
        let mut dc = vec![0f64; cdim];
        let mut d_final_out = vec![0f64; ntot * pd];
        d_final_out[..n_img * pd].copy_from_slice(&dpred);
        let (d_scaled, g_flin_w, g_flin_b) = linb_bwd(&scaled, ntot, dim, &w.flin_w, pd, &d_final_out);
        let mut d_normed = vec![0f64; ntot * dim];
        let mut d_scale = vec![0f64; dim];
        for r in 0..ntot {
            for cc2 in 0..dim {
                d_normed[r * dim + cc2] = d_scaled[r * dim + cc2] * scale[cc2];
                d_scale[cc2] += d_scaled[r * dim + cc2] * normed[r * dim + cc2];
            }
        }
        let d_uni_host = layernorm_bwd(&uni, ntot, dim, &inv_ln, &d_normed);
        let (d_silu_c, g_fadaln_w, g_fadaln_b) = linb_bwd(&silu_c, 1, cdim, &w.fadaln_w, dim, &d_scale);
        for j in 0..cdim {
            dc[j] += d_silu_c[j] * dsilu(cvec[j]);
        }

        // ---- device: block stack backward (reverse) ----
        let mut d_uni32 = to32(&d_uni_host);
        let mut main_g: Vec<Grads> = Vec::new();
        for (bw, inp) in w.main.iter().zip(&main_in).rev() {
            let g = self.eng.backward(bw, self.dims(ntot), inp, &c32, &uni_cos, &uni_sin, true, &d_uni32);
            d_uni32 = to32(&g.dx);
            for j in 0..cdim {
                dc[j] += g.dc[j];
            }
            main_g.push(g);
        }
        main_g.reverse();
        let mut d_img32 = d_uni32[..n_img * dim].to_vec();
        let mut d_capt32 = d_uni32[n_img * dim..].to_vec();

        let mut ctx_g: Vec<Grads> = Vec::new();
        for (bw, inp) in w.ctx_ref.iter().zip(&ctx_in).rev() {
            let g = self.eng.backward(bw, self.dims(ncap), inp, &c32, &cc, &cs, false, &d_capt32);
            d_capt32 = to32(&g.dx);
            ctx_g.push(g);
        }
        ctx_g.reverse();
        let mut noise_g: Vec<Grads> = Vec::new();
        for (bw, inp) in w.noise_ref.iter().zip(&noise_in).rev() {
            let g = self.eng.backward(bw, self.dims(n_img), inp, &c32, &ic, &is, true, &d_img32);
            d_img32 = to32(&g.dx);
            for j in 0..cdim {
                dc[j] += g.dc[j];
            }
            noise_g.push(g);
        }
        noise_g.reverse();

        // ---- host: embedders backward ----
        let (_dp, g_xemb_w, g_xemb_b) = linb_bwd(&patches, n_img, pd, &w.xemb_w, dim, &to64(&d_img32));
        let (d_capn, g_cap1_w, g_cap1_b) = linb_bwd(&capn, ncap, c.cap_feat_dim, &w.cap1_w, dim, &to64(&d_capt32));
        let g_capn_w = rmsnorm_dw(&b.cap, ncap, c.cap_feat_dim, &inv_capn, &d_capn);

        // ---- host: timestep MLP backward ----
        let (d_h0, g_t2_w, g_t2_b) = linb_bwd(&h0, 1, TH, &w.t2_w, cdim, &dc);
        let mut d_h0pre = vec![0f64; TH];
        for i in 0..TH {
            d_h0pre[i] = d_h0[i] * dsilu(h0pre[i]);
        }
        let (_dte, g_t0_w, g_t0_b) = linb_bwd(&te, 1, TDIM, &w.t0_w, TH, &d_h0pre);

        let grads = ModelGrads {
            t0_w: g_t0_w, t0_b: g_t0_b, t2_w: g_t2_w, t2_b: g_t2_b,
            xemb_w: g_xemb_w, xemb_b: g_xemb_b, capn_w: g_capn_w, cap1_w: g_cap1_w, cap1_b: g_cap1_b,
            noise_ref: noise_g, ctx_ref: ctx_g, main: main_g,
            fadaln_w: g_fadaln_w, fadaln_b: g_fadaln_b, flin_w: g_flin_w, flin_b: g_flin_b,
        };
        (loss, grads)
    }
}
