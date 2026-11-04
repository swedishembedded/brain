// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec gates for `model::lora`'s shared adapter-fold layer: the placement
//! fold every architecture's own `fold_into_tensors` reduces to, and the
//! multi-adapter driver above it.
//!
//! Two properties carry the whole feature and are asserted on real numbers
//! here rather than on "it did not crash":
//!
//! 1. **Order.** Adapters fold one after another onto the SAME map, so
//!    adapter *n+1* adds onto adapter *n*'s already-folded result. With
//!    additive low-rank deltas that means the final weight is
//!    `W + Σ sᵢ·(αᵢ/rᵢ)·Bᵢ·Aᵢ` - each adapter's own strength multiplies only
//!    its own delta, and no adapter can scale another's.
//! 2. **Nothing is written unless everything validates.** A rejected adapter
//!    leaves the map pristine, never half folded - the map is what a model is
//!    then built from, and a half-folded map builds a model nobody can
//!    describe.

use model::lora::{fold_adapter_files, fold_external_into, fold_placements, Pair, Placement, Tensors};

/// A pair whose `B·A` is exactly the outer product `b ⊗ a` (rank 1), so every
/// expected value below can be written out by hand.
fn rank1(out: usize, inn: usize, a: &[f32], b: &[f32]) -> Pair {
    assert_eq!(a.len(), inn);
    assert_eq!(b.len(), out);
    Pair::from_ab(out, inn, 1, a.to_vec(), b.to_vec())
}

fn map(entries: &[(&str, Vec<usize>, Vec<f32>)]) -> Tensors {
    entries.iter().map(|(n, s, d)| ((*n).to_string(), (s.clone(), d.clone()))).collect()
}

/// The unfused case every migrated architecture uses: one whole `[out, in]`
/// tensor per pair, at offset 0.
#[test]
fn a_whole_tensor_placement_adds_scale_times_b_times_a() {
    // B·A = [[1,2,3],[2,4,6]]
    let p = rank1(2, 3, &[1.0, 2.0, 3.0], &[1.0, 2.0]);
    let mut ts = map(&[("w", vec![2, 3], vec![10.0; 6])]);
    fold_placements(&mut ts, 1.0, &[Placement::whole("w", &p)]).expect("folds");
    assert_eq!(ts["w"].1, vec![11.0, 12.0, 13.0, 12.0, 14.0, 16.0]);

    // The scale multiplies the whole delta; 0 is a bit-exact no-op.
    let mut ts = map(&[("w", vec![2, 3], vec![10.0; 6])]);
    fold_placements(&mut ts, 0.5, &[Placement::whole("w", &p)]).expect("folds");
    assert_eq!(ts["w"].1, vec![10.5, 11.0, 11.5, 11.0, 12.0, 13.0]);

    let mut ts = map(&[("w", vec![2, 3], vec![10.0; 6])]);
    fold_placements(&mut ts, 0.0, &[Placement::whole("w", &p)]).expect("folds");
    assert_eq!(ts["w"].1, vec![10.0; 6], "strength 0 must reproduce the base bit-for-bit");
}

/// The fused case (`flux2`'s `qkv`/`linear1`/`linear2`): several pairs share
/// one stored tensor, each owning a rectangle of it. A placement must touch
/// exactly its own rectangle.
#[test]
fn a_fused_placement_touches_only_its_own_rectangle() {
    // A [6, 2] tensor holding three stacked [2, 2] row blocks.
    let p = rank1(2, 2, &[1.0, 1.0], &[1.0, 1.0]); // B·A = all ones
    let mut ts = map(&[("qkv", vec![6, 2], vec![0.0; 12])]);
    // The middle block: rows 2..4, row stride 2, column 0.
    fold_placements(&mut ts, 1.0, &[Placement::fused("qkv", &p, 12, 2, 2, 0)]).expect("folds");
    assert_eq!(ts["qkv"].1, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    // A column split: a [2, 5] tensor whose second pair owns columns 2..5.
    let q = rank1(2, 3, &[1.0, 1.0, 1.0], &[1.0, 1.0]);
    let mut ts = map(&[("l2", vec![2, 5], vec![0.0; 10])]);
    fold_placements(&mut ts, 1.0, &[Placement::fused("l2", &q, 10, 0, 5, 2)]).expect("folds");
    assert_eq!(ts["l2"].1, vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
}

/// A missing or mis-sized target is a hard error NAMING the tensor, and the
/// map is left exactly as it was - not half folded.
#[test]
fn a_rejected_fold_names_the_tensor_and_writes_nothing() {
    let p = rank1(2, 3, &[1.0, 2.0, 3.0], &[1.0, 2.0]);
    let q = rank1(2, 3, &[1.0, 1.0, 1.0], &[1.0, 1.0]);

    let base = map(&[("w", vec![2, 3], vec![10.0; 6])]);
    let mut ts = base.clone();
    let e = fold_placements(&mut ts, 1.0, &[Placement::whole("w", &p), Placement::whole("absent", &q)])
        .expect_err("a missing base tensor must fail");
    assert!(e.contains("absent"), "the error must name the tensor: {e}");
    assert_eq!(ts, base, "a rejected fold must leave the map pristine");

    let mut ts = map(&[("w", vec![2, 3], vec![10.0; 5])]);
    let before = ts.clone();
    let e = fold_placements(&mut ts, 1.0, &[Placement::whole("w", &p)]).expect_err("wrong size must fail");
    assert!(e.contains('w'), "the error must name the tensor: {e}");
    assert_eq!(ts, before);
}

/// An empty placement list is a no-op, byte for byte. The zero-adapter case is
/// the common one (every generation with no LoRA at all), so it has to cost
/// nothing and change nothing.
#[test]
fn an_empty_placement_list_changes_nothing() {
    let base = map(&[("w", vec![2, 3], vec![10.0; 6])]);
    let mut ts = base.clone();
    fold_placements(&mut ts, 1.0, &[]).expect("folds");
    assert_eq!(ts, base);
}

// ---------------------------------------------------------------- multi-file

/// Write a minimal F32 safetensors file: 8-byte LE header length, JSON header,
/// then the payloads back to back - enough to stand in for a third-party
/// (ai-toolkit / ComfyUI) adapter.
fn write_st(path: &std::path::Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut blob: Vec<u8> = Vec::new();
    for (name, shape, data) in tensors {
        let start = blob.len();
        for v in data {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            (*name).to_string(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, blob.len()]}),
        );
    }
    let hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&blob);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, out).unwrap();
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("brain-model-lora-fold-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

/// One rank-1 external adapter over `key`, delta `b ⊗ a`.
fn external(tag: &str, stem: &str, a: &[f32], b: &[f32]) -> String {
    let p = tmp(&format!("{tag}.safetensors"));
    write_st(
        &p,
        &[
            (&format!("diffusion_model.{stem}.lora_A.weight"), vec![1, a.len()], a.to_vec()),
            (&format!("diffusion_model.{stem}.lora_B.weight"), vec![b.len(), 1], b.to_vec()),
        ],
    );
    p.to_str().unwrap().to_string()
}

/// TWO adapters over the SAME tensor, each at its own strength, folded in a
/// known order. The result must be `W + s₁·Δ₁ + s₂·Δ₂` exactly: sequential,
/// additive, and with neither strength touching the other's delta.
#[test]
fn two_adapters_fold_in_order_each_at_its_own_scale() {
    let one = external("stack-1", "img_in", &[1.0, 1.0, 1.0], &[1.0, 1.0]); // Δ₁ = all 1
    let two = external("stack-2", "img_in", &[2.0, 2.0, 2.0], &[1.0, 1.0]); // Δ₂ = all 2
    let mut ts = map(&[("img_in.weight", vec![2, 3], vec![0.0; 6])]);
    let reports = fold_adapter_files(
        &mut ts,
        &[(one.as_str(), 1.0), (two.as_str(), 0.5)],
        "test",
        |_, _, _| panic!("both files are .safetensors - the native loader must not be reached"),
    )
    .expect("folds");
    // 1.0*1 + 0.5*2 = 2 everywhere.
    assert_eq!(ts["img_in.weight"].1, vec![2.0; 6]);
    assert_eq!(reports.len(), 2);
    assert!(reports.iter().all(|r| r.external() && r.pairs == 1 && r.rank == 1));
    assert_eq!(reports[0].strength, 1.0);
    assert_eq!(reports[1].strength, 0.5);

    // Swapping the strengths must swap which delta is halved - proof the pairing
    // is positional and not, say, "the last scale wins".
    let mut ts = map(&[("img_in.weight", vec![2, 3], vec![0.0; 6])]);
    fold_adapter_files(&mut ts, &[(one.as_str(), 0.5), (two.as_str(), 1.0)], "test", |_, _, _| unreachable!())
        .expect("folds");
    assert_eq!(ts["img_in.weight"].1, vec![2.5; 6]);
}

/// Distinct targets stack too, and a file that is not `.safetensors` is
/// dispatched to the caller's own native loader - the format decision the
/// shared driver makes on the caller's behalf.
#[test]
fn adapters_over_distinct_tensors_stack_and_native_files_reach_the_callers_loader() {
    let ext = external("distinct-a", "img_in", &[1.0, 1.0, 1.0], &[1.0, 1.0]);
    let mut ts = map(&[
        ("img_in.weight", vec![2, 3], vec![0.0; 6]),
        ("txt_in.weight", vec![2, 3], vec![0.0; 6]),
    ]);
    let mut native_calls = Vec::new();
    let reports = fold_adapter_files(
        &mut ts,
        &[(ext.as_str(), 1.0), ("some/adapter.brain", 2.0)],
        "test",
        |path, ts, strength| {
            native_calls.push((path.to_string(), strength));
            let (_, w) = ts.get_mut("txt_in.weight").expect("present");
            for v in w.iter_mut() {
                *v += strength;
            }
            Ok((7, 4))
        },
    )
    .expect("folds");
    assert_eq!(ts["img_in.weight"].1, vec![1.0; 6]);
    assert_eq!(ts["txt_in.weight"].1, vec![2.0; 6]);
    assert_eq!(native_calls, vec![("some/adapter.brain".to_string(), 2.0)]);
    assert!(reports[0].external());
    assert!(!reports[1].external());
    assert_eq!((reports[1].pairs, reports[1].rank), (7, 4));
}

/// No adapters at all: the map must come back byte-identical, and no loader of
/// either family may run.
#[test]
fn no_adapters_leaves_the_map_untouched() {
    let base = map(&[("img_in.weight", vec![2, 3], vec![3.5; 6])]);
    let mut ts = base.clone();
    let reports =
        fold_adapter_files(&mut ts, &[], "test", |_, _, _| unreachable!("no adapter, no loader")).expect("folds");
    assert!(reports.is_empty());
    assert_eq!(ts, base);
}

// -------------------------------------------------------------------- LoKr

/// `W1 ⊗ W2` written out by hand, so the fold is checked against arithmetic
/// rather than against a second implementation of itself.
///
/// `W1 = [[1,2],[3,4]]`, `W2 = [[5,6],[7,8]]`:
/// ```text
///   [ 1*W2  2*W2 ]   [  5  6 | 10 12 ]
///   [ 3*W2  4*W2 ] = [  7  8 | 14 16 ]
///                    [ 15 18 | 20 24 ]
///                    [ 21 24 | 28 32 ]
/// ```
/// Both factors are stored FULL, which is the case the real ai-toolkit
/// adapters use - and the case in which the file's own `.alpha` is NOT a
/// scale: both reference implementations resolve the multiplier to exactly
/// 1.0 there (ComfyUI leaves `dim` unset so `alpha = 1.0`; LyCORIS writes
/// `alpha = lora_dim` so `alpha/dim == 1`). A file whose stored alpha is a
/// large sentinel - and real ones are - must therefore be folded at 1.0, not
/// at the sentinel.
#[test]
fn a_full_factor_lokr_folds_the_kronecker_product_and_ignores_the_stored_alpha() {
    let p = tmp("lokr-full.safetensors");
    write_st(
        &p,
        &[
            ("diffusion_model.img_in.lokr_w1", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            ("diffusion_model.img_in.lokr_w2", vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]),
            ("diffusion_model.img_in.alpha", vec![], vec![1.0e10]),
        ],
    );
    let want = vec![
        5.0, 6.0, 10.0, 12.0, //
        7.0, 8.0, 14.0, 16.0, //
        15.0, 18.0, 20.0, 24.0, //
        21.0, 24.0, 28.0, 32.0,
    ];
    let mut ts = map(&[("img_in.weight", vec![4, 4], vec![0.0; 16])]);
    let info = fold_external_into(p.to_str().unwrap(), &mut ts, 1.0, "test").expect("folds");
    assert_eq!(ts["img_in.weight"].1, want, "W + kron(W1, W2), exactly");
    assert_eq!(info.pairs, 1);
    assert_eq!(info.family, "lokr");

    // Strength still multiplies the whole delta, and 0 is a bit-exact no-op.
    let mut ts = map(&[("img_in.weight", vec![4, 4], vec![0.0; 16])]);
    fold_external_into(p.to_str().unwrap(), &mut ts, 0.5, "test").expect("folds");
    assert_eq!(ts["img_in.weight"].1, want.iter().map(|v| v * 0.5).collect::<Vec<_>>());

    let mut ts = map(&[("img_in.weight", vec![4, 4], vec![3.0; 16])]);
    fold_external_into(p.to_str().unwrap(), &mut ts, 0.0, "test").expect("folds");
    assert_eq!(ts["img_in.weight"].1, vec![3.0; 16]);
}

/// A factor may itself be stored low-rank (`lokr_w1_a @ lokr_w1_b`), and THEN
/// the file's `.alpha` is a real scale: `alpha / dim`, `dim` being the
/// decomposition's own inner dimension.
///
/// `W1_a = [[1],[3]]`, `W1_b = [[1,2]]` -> `W1 = [[1,2],[3,6]]`, `dim = 1`.
/// `W2 = I₂`, `alpha = 2` -> multiplier 2, and
/// `2·(W1 ⊗ I₂) = [[2,0,4,0],[0,2,0,4],[6,0,12,0],[0,6,0,12]]`.
#[test]
fn a_decomposed_lokr_factor_is_reconstructed_and_scaled_by_alpha_over_dim() {
    let p = tmp("lokr-decomposed.safetensors");
    write_st(
        &p,
        &[
            ("diffusion_model.img_in.lokr_w1_a", vec![2, 1], vec![1.0, 3.0]),
            ("diffusion_model.img_in.lokr_w1_b", vec![1, 2], vec![1.0, 2.0]),
            ("diffusion_model.img_in.lokr_w2", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            ("diffusion_model.img_in.alpha", vec![], vec![2.0]),
        ],
    );
    let mut ts = map(&[("img_in.weight", vec![4, 4], vec![0.0; 16])]);
    fold_external_into(p.to_str().unwrap(), &mut ts, 1.0, "test").expect("folds");
    assert_eq!(
        ts["img_in.weight"].1,
        vec![
            2.0, 0.0, 4.0, 0.0, //
            0.0, 2.0, 0.0, 4.0, //
            6.0, 0.0, 12.0, 0.0, //
            0.0, 6.0, 0.0, 12.0,
        ]
    );
}

/// A LoRA adapter and a LoKr adapter in the SAME stack. The format decision is
/// per adapter, not once for the run, so each one folds by its own family's
/// math at its own strength - and the shared target ends up moved by the sum.
#[test]
fn a_stack_may_mix_lora_and_lokr_adapters() {
    let lora = external("mix-lora", "img_in", &[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0, 1.0, 1.0]); // delta 1 everywhere
    let kr = tmp("mix-lokr.safetensors");
    write_st(
        &kr,
        &[
            ("diffusion_model.img_in.lokr_w1", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]),
            ("diffusion_model.img_in.lokr_w2", vec![2, 2], vec![2.0, 2.0, 2.0, 2.0]),
        ],
    );
    // kron(I₂, 2·J₂) is 2 on the two diagonal 2x2 blocks, 0 off them.
    let kron = vec![
        2.0, 2.0, 0.0, 0.0, //
        2.0, 2.0, 0.0, 0.0, //
        0.0, 0.0, 2.0, 2.0, //
        0.0, 0.0, 2.0, 2.0,
    ];
    let mut ts = map(&[("img_in.weight", vec![4, 4], vec![0.0; 16])]);
    let reports = fold_adapter_files(
        &mut ts,
        &[(lora.as_str(), 1.0), (kr.to_str().unwrap(), 0.5)],
        "test",
        |_, _, _| unreachable!(),
    )
    .expect("folds");
    let want: Vec<f32> = kron.iter().map(|k| 1.0 + 0.5 * k).collect();
    assert_eq!(ts["img_in.weight"].1, want);
    assert_eq!(reports[0].family, "lora");
    assert_eq!(reports[1].family, "lokr");
}

/// A LoKr key this loader does not implement (`lokr_t2`, the convolutional
/// CP-decomposition factor) must be refused by name. FLUX.2's targets are all
/// linears, so a file carrying one is not what we think it is - and folding
/// the rest of it would produce a plausible image from a partly-applied
/// adapter.
#[test]
fn an_unimplemented_lokr_key_is_refused_by_name() {
    let p = tmp("lokr-t2.safetensors");
    write_st(
        &p,
        &[
            ("diffusion_model.img_in.lokr_w1", vec![2, 2], vec![1.0; 4]),
            ("diffusion_model.img_in.lokr_t2", vec![2, 2], vec![1.0; 4]),
            ("diffusion_model.img_in.lokr_w2_a", vec![2, 1], vec![1.0; 2]),
            ("diffusion_model.img_in.lokr_w2_b", vec![1, 2], vec![1.0; 2]),
        ],
    );
    let mut ts = map(&[("img_in.weight", vec![4, 4], vec![0.0; 16])]);
    let e = fold_external_into(p.to_str().unwrap(), &mut ts, 1.0, "test").expect_err("must refuse");
    assert!(e.contains("lokr_t2"), "the error must name the key it cannot handle: {e}");
}

/// A stem carrying BOTH families is a file we do not understand, not a merge
/// opportunity: guessing which one is authoritative would silently halve or
/// double an adapter's effect.
#[test]
fn a_stem_with_both_lora_and_lokr_keys_is_refused() {
    let p = tmp("lokr-mixed-stem.safetensors");
    write_st(
        &p,
        &[
            ("diffusion_model.img_in.lokr_w1", vec![2, 2], vec![1.0; 4]),
            ("diffusion_model.img_in.lokr_w2", vec![2, 2], vec![1.0; 4]),
            ("diffusion_model.img_in.lora_A.weight", vec![1, 4], vec![1.0; 4]),
            ("diffusion_model.img_in.lora_B.weight", vec![4, 1], vec![1.0; 4]),
        ],
    );
    let mut ts = map(&[("img_in.weight", vec![4, 4], vec![0.0; 16])]);
    let e = fold_external_into(p.to_str().unwrap(), &mut ts, 1.0, "test").expect_err("must refuse");
    assert!(e.contains("img_in"), "the error must name the stem: {e}");
}

/// The shape a LoKr adapter implies for its target is `W1.rows·W2.rows` by
/// `W1.cols·W2.cols`, and a mismatch against the base is the wrong-model case
/// - caught before anything is written, as for LoRA.
#[test]
fn a_lokr_shape_mismatch_fails_before_writing() {
    let p = tmp("lokr-badshape.safetensors");
    write_st(
        &p,
        &[
            ("diffusion_model.img_in.lokr_w1", vec![2, 2], vec![1.0; 4]),
            ("diffusion_model.img_in.lokr_w2", vec![3, 2], vec![1.0; 6]),
        ],
    );
    let mut ts = map(&[("img_in.weight", vec![4, 4], vec![0.0; 16])]);
    let e = fold_external_into(p.to_str().unwrap(), &mut ts, 1.0, "test").expect_err("must refuse");
    assert!(e.contains("img_in.weight"), "the error must name the tensor: {e}");
    assert_eq!(ts["img_in.weight"].1, vec![0.0; 16], "a rejected adapter writes nothing");
}

/// An adapter key that matches no base tensor is a hard error naming the
/// tensor AND the architecture that lacks it - never a quiet skip returning
/// base-model output from a run the caller believes is adapted.
#[test]
fn an_unmatched_external_key_fails_loudly_naming_the_architecture() {
    let p = external("unmatched", "no_such_module", &[1.0, 1.0], &[1.0]);
    let mut ts = map(&[("img_in.weight", vec![2, 3], vec![0.0; 6])]);
    let e = fold_external_into(&p, &mut ts, 1.0, "FLUX.2").expect_err("must reject");
    assert!(e.contains("no_such_module"), "the error must name the tensor: {e}");
    assert!(e.contains("FLUX.2"), "the error must name the architecture: {e}");
    assert_eq!(ts["img_in.weight"].1, vec![0.0; 6], "a rejected adapter writes nothing");
}
