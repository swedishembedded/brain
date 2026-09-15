// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `ImageProject` (`_encode_image`'s tail - position/temporal embed, spatial
//! pool, projection, LayerNorm) against the real checkpoint: input is
//! `unpooled`'s golden (DaViT's own output, already parity-checked end to
//! end in `davit_full_forward_parity.rs`), expected output is `projected`'s
//! golden - isolates this module from the DaViT tower the same way the
//! block-level tests isolate each block from its neighbors.

use std::collections::HashMap;

use gpu_core::Gpu;
use paramstore::{ParamStore, Role};

use florence2::vision::project::{build_pos_embed_table, transpose_2d};
use florence2::vision::{pipelines::PIPELINES, ImageProject, ImageProjectKernelIds};
use model::hostmath::cosine;

const PROJ_EPS: f32 = 1e-5;
const GRID: u32 = 24;
const DAVIT_DIM: u32 = 1024;
const D_MODEL: u32 = 768;
const PREFIX: &str = "vision_projector";

#[test]
fn image_project_matches_real_checkpoint() {
    let Some(ckpt_dir) = std::env::var("FLORENCE2_DIR").ok() else {
        brain_testutil::skip("FLORENCE2_DIR unset");
        return;
    };
    let golden_path = brain_testutil::testdata("florence2/davit/stages.safetensors");
    if !std::path::Path::new(&golden_path).exists() {
        brain_testutil::skip("florence2 davit golden fixture absent - regenerate with tools/goldens/florence2_dump_reference.py");
        return;
    }

    let ckpt = checkpoint::safetensors::read(&format!("{ckpt_dir}/model.safetensors")).expect("read checkpoint");
    let mut raw: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for t in ckpt {
        raw.insert(t.name, (t.shape, t.data));
    }

    // Synthesize the two derived tensors ImageProject needs (see project.rs's
    // module doc): the static [24,24,1024] position table, and image_projection
    // transposed from the checkpoint's [in,out] to matmul_rows' [out,in].
    let (_, column) = raw.get("image_pos_embed.column_embeddings.weight").expect("column_embeddings");
    let (_, row) = raw.get("image_pos_embed.row_embeddings.weight").expect("row_embeddings");
    let pos_table = build_pos_embed_table(column, row, GRID, GRID, DAVIT_DIM / 2);

    let (proj_shape, proj_data) = raw.get("image_projection").expect("image_projection");
    assert_eq!(proj_shape, &vec![DAVIT_DIM as usize, D_MODEL as usize]);
    let proj_t = transpose_2d(proj_data, DAVIT_DIM as usize, D_MODEL as usize);

    let mut source: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    source.insert(format!("{PREFIX}.pos_embed_table"), (vec![(GRID * GRID) as usize, DAVIT_DIM as usize], pos_table));
    source.insert(format!("{PREFIX}.image_projection_t"), (vec![D_MODEL as usize, DAVIT_DIM as usize], proj_t));
    for name in ["visual_temporal_embed.pos_idx_to_embed", "image_proj_norm.weight", "image_proj_norm.bias"] {
        let (shape, data) = raw.get(name).unwrap_or_else(|| panic!("missing {name}")).clone();
        source.insert(format!("{PREFIX}.{name}"), (shape, data));
    }

    let roles: Vec<_> = source.iter().map(|(name, (_, data))| (name.clone(), data.len(), Role::Frozen)).collect();

    let gpu = Gpu::new_cpu(PIPELINES);
    let ps = ParamStore::new_with_roles_src(&gpu, roles, &source);
    let k = ImageProjectKernelIds::resolve(PIPELINES);
    let m = ImageProject::new(&gpu, PREFIX, GRID, GRID, DAVIT_DIM, D_MODEL, PROJ_EPS);

    let golden = checkpoint::safetensors::read(&golden_path).expect("read golden");
    let gmap: HashMap<String, checkpoint::safetensors::StTensor> = golden.into_iter().map(|t| (t.name.clone(), t)).collect();

    let unpooled_host = &gmap["unpooled"].data;
    let unpooled_buf = gpu.storage(unpooled_host.len() as u64);
    gpu.write_f32(&unpooled_buf, unpooled_host);

    let out_ref = m.forward(&gpu, &k, &ps, &unpooled_buf);
    gpu.poll_wait();
    let want = &gmap["projected"].data;
    let out = gpu.read(out_ref, want.len());

    let c = cosine(&out, want);
    assert!(c >= 0.999, "cosine {c:.6} (want >= 0.999); got[0..4]={:?} want[0..4]={:?}", &out[..4], &want[..4]);
}
