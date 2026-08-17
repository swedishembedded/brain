// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Multi-head bidirectional attention backward vs a hand-backprop reference.
//! Regression for the heads>1 training-grad bug (the tiny model fixtures are
//! all heads=1). T small so the numpy-free reference is cheap and exact.
use gpu_core::Gpu;

fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z=s;
        z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9); z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);
        ((( (z^(z>>31))>>40) as f32)/(1u64<<24) as f32 - 0.5)*2.0 }).collect()
}

#[test]
fn attn_multihead_backward_matches_reference() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() { return; }
    let (heads, t, hd) = (8usize, 16usize, 8usize);
    let c = heads * hd;
    let gpu = Gpu::new_cpu(&[
        ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
        ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
        ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
        ("attn_bwd_dscores_bidir", kernels::ATTN_BWD_DSCORES_BIDIR),
        ("attn_bwd_dv_bidir", kernels::ATTN_BWD_DV_BIDIR),
        ("attn_bwd_dq_bidir", kernels::ATTN_BWD_DQ_BIDIR),
        ("attn_bwd_dk_bidir", kernels::ATTN_BWD_DK_BIDIR),
    ]);
    let (ks, ksm, kap, kds, kdv, kdq, kdk) = (0,1,2,3,4,5,6);
    // qkv rows [T, 3C]; d_out [T, C].
    let qkv = rand(1, t * 3 * c);
    let d_out = rand(2, t * c);
    let wf = |b: &gpu_core::DeviceBuffer, d: &[f32]| gpu.write(b, &d.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    let qkv_b = gpu.storage_init("qkv", &qkv);
    let scores = gpu.storage((heads*t*t) as u64);
    let probs = gpu.storage((heads*t*t) as u64);
    let out = gpu.storage((t*c) as u64);
    let d_out_b = gpu.storage_init("dout", &d_out);
    let d_scores = gpu.storage((heads*t*t) as u64);
    let d_qkv = gpu.storage((t*3*c) as u64);
    let sp = [1, heads as u32, t as u32, hd as u32, (3*c) as u32, 0, c as u32];
    let ap = [1, heads as u32, t as u32, hd as u32, (3*c) as u32, (2*c) as u32, c as u32];
    gpu.submit(&[], &[
        gpu.step(ks, &[&qkv_b,&scores], &sp, (heads*t*t) as u32),
        gpu.step(ksm, &[&scores,&probs], &[1,heads as u32,t as u32], (heads*t) as u32),
        gpu.step(kap, &[&probs,&qkv_b,&out], &ap, (heads*t*hd) as u32),
        gpu.step(kds, &[&d_out_b,&qkv_b,&probs,&d_scores], &ap, (heads*t) as u32),
        gpu.step(kdv, &[&probs,&d_out_b,&d_qkv], &ap, (heads*t*hd) as u32),
        gpu.step(kdq, &[&d_scores,&qkv_b,&d_qkv], &sp, (heads*t*hd) as u32),
        gpu.step(kdk, &[&d_scores,&qkv_b,&d_qkv], &sp, (heads*t*hd) as u32),
    ]);
    let _ = wf; let _ = ksm;
    let dq_gpu = gpu.read(&d_qkv, t*3*c);

    // Reference backprop (per head, [T,hd]).
    let scale = 1.0/(hd as f32).sqrt();
    let mut dq_ref = vec![0.0f32; t*3*c];
    for h in 0..heads {
        let g = |off: usize, i: usize, d: usize| qkv[i*3*c + off + h*hd + d];
        // fwd probs
        let mut p = vec![0.0f32; t*t];
        for i in 0..t {
            let mut sc = vec![0.0f32; t];
            for (j, scj) in sc.iter_mut().enumerate() { let mut s=0.0; for d in 0..hd { s+=g(0,i,d)*g(c,j,d); } *scj=s*scale; }
            let mx = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut den=0.0; for scj in &mut sc { *scj=(*scj-mx).exp(); den+=*scj; }
            for j in 0..t { p[i*t+j]=sc[j]/den; }
        }
        // d_out for this head
        let dob = |i: usize, d: usize| d_out[i*c + h*hd + d];
        // dv[j,d] = sum_i p[i,j]*dob[i,d]
        for j in 0..t { for d in 0..hd { let mut s=0.0; for i in 0..t { s+=p[i*t+j]*dob(i,d); } dq_ref[j*3*c + 2*c + h*hd + d]=s; } }
        // d_prob[i,j]=sum_d dob[i,d]*v[j,d]; dscore=softmax_jac
        let mut dsc=vec![0.0f32;t*t];
        for i in 0..t {
            let mut dp=vec![0.0f32;t];
            for (j, dpj) in dp.iter_mut().enumerate() { let mut s=0.0; for d in 0..hd { s+=dob(i,d)*g(2*c,j,d); } *dpj=s; }
            let dot:f32=(0..t).map(|j| p[i*t+j]*dp[j]).sum();
            for j in 0..t { dsc[i*t+j]=p[i*t+j]*(dp[j]-dot); }
        }
        // dq[i,d]=scale*sum_j dsc[i,j]*k[j,d]; dk[j,d]=scale*sum_i dsc[i,j]*q[i,d]
        for i in 0..t { for d in 0..hd { let mut s=0.0; for j in 0..t { s+=dsc[i*t+j]*g(c,j,d); } dq_ref[i*3*c + h*hd + d]=s*scale; } }
        for j in 0..t { for d in 0..hd { let mut s=0.0; for i in 0..t { s+=dsc[i*t+j]*g(0,i,d); } dq_ref[j*3*c + c + h*hd + d]=s*scale; } }
    }
    let max = dq_gpu.iter().zip(&dq_ref).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
    assert!(max < 1e-4, "heads=2 attention d_qkv max abs diff {max}");
}
