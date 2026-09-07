# worldmirror2 - roadmap

Port of HY-WorldMirror-2.0, a DINOv2-based multi-frame 3D scene
reconstruction model that predicts Gaussian splats, depth/normal maps, and
camera pose from one or more images, with an NPU export path. Forward parity
against the reference is verified.

## Not yet done

- [ ] NPU throughput/latency benchmarking (only numerical parity has been
      measured so far)
- [ ] wgpu memory-budget autotuning and frame pipelining
- [ ] Further CPU matmul blocking (column tiling) if forward speed needs
      another step

Training backward for cross-attention currently accumulates gradients only
within a chunk rather than across the full sequence, so training requires
the chunk size to cover the full attention span. The NPU backend executes
only in fp16, and precision drift grows with network depth - acceptable for
preview-quality geometry, less so for higher-fidelity uses.

## Serving contract - done

`reconstruct` (one-shot) is now a [`capability::Provider`] action
(`crates/worldmirror2/src/caps.rs`), registered in the residency scheduler
(`crates/cli/src/resident_worldmirror2.rs`) and the CLI catalog
(`crates/cli/src/catalog.rs`) under `brain/worldmirror2` - reachable over
`brain do`, D-Bus, and the event API with no worldmirror2-specific plumbing
in any transport. `images` (N unposed frames) mirrors the video-blob
convention every other served video input uses; `min_opacity`/`max_depth`/
`prune_voxel`/`maps` mirror `brain mirror infer`'s own CLI flags exactly - a
single feed-forward pass has no iteration count to expose. `weights` carries
`ParamSpec::host_env("BRAIN_WORLDMIRROR2_WEIGHTS")`, so the served manifest
(`manifest_resident`) drops it, following `glmdsa::caps`'s existing pattern.
A mismatched-size (non-patch-aligned) image batch is a clean `Err`, never the
`assert_eq!` panic `mirror_cli.rs`'s own file-loading path would otherwise hit.

### One resident instance, keyed on checkpoint identity only

`Mirror` is shape-ADAPTIVE, not shape-fixed: `Built` (its per-`(frames,H,W)`
shape buffers plus the recorded forward) lives inside `Option<Built>`, kept
OUTSIDE the ~5GB `ParamStore`, and `Mirror::forward` lazily rebuilds `Built`
only when the requested `(s,hp,wp)` differs from what is cached. So
`resident_worldmirror2.rs::WorldMirror2Resident::instance_key` returns ONE
key regardless of the request - keying on frames/width/height too (the
Wan-style per-shape fingerprint) would duplicate the whole ~5GB `ParamStore`
once per distinct request shape for no benefit, fighting `Mirror`'s own
internal shape cache instead of using it.

The honest caveat: `Built` is `Option<Built>`, so exactly ONE shape's buffers
are cached at a time. A workload that keeps ALTERNATING between two shapes on
this one instance pays a full rebuild on every single call - a latency cost,
not a correctness bug. `crates/worldmirror2/tests/t9_caps_matches_direct.rs`'s
`shape_cycle_one_instance_matches_independent_single_shape_runs` test proves
this is still CORRECT (every result in an A→B→A cycle on one instance is
bit-identical to the equivalent freshly-built single-shape run), which is
what actually justifies the one-instance design rather than just reading the
source and assuming it.

### `Mirror` now owns its `Gpu` (sub-blocker, step 0)

Before this item, `Mirror<'g>` BORROWED its `Gpu` (`gpu: &'g Gpu`) - unique
among this repo's served models (`supir::model`, `controlnet::model`,
`vqgan::model` all take and HOLD `Gpu` BY VALUE), and incompatible with a
resident `Instance` holding a built model across calls with nothing else
keeping the `Gpu` alive. `Mirror::new` now takes `gpu: Gpu` and holds it
(`Mirror::gpu(&self) -> &Gpu` for callers that need it back, e.g.
`gaussians::assemble`). The three call sites that construct a `Mirror`
(`crates/cli/src/mirror_cli.rs`, `crates/worldmirror2/tests/t1_t2_encode.rs`,
`crates/worldmirror2/tests/t8_multiframe_tiny.rs`) were updated accordingly.
Re-ran this crate's five pre-existing `t*_*.rs` tests before and after the
refactor - identical pass/skip output both times (`MIRROR_CKPT`/
`MIRROR_DPT_TINY`/`MIRROR_BICUBIC_REFS` are all unset in this environment, so
`t1`'s `t2_dinov2_patch_tokens`, `t3`'s `dpt_tiny_stages` and `t6`'s
`matches_torch_refs` skip identically before and after - no numeric parity
regression is observable here either way, and the checkpoint-free tests in
`t0`, `t1`, `t3`, `t6`, `t8` all pass identically too), confirming the
ownership change caused no behavioural drift.
