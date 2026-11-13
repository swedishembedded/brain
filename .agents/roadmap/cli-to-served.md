<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# cli-to-served

Scope: Moving every model-specific CLI verb into its model's served actions, turning the per-architecture CLI modules into per-model Rust samples, and reducing the `brain` binary to its generic verbs (a major release: no compatibility aliases).

## The design

- **A model capability is a served action, and only that.** Inference, training, evaluation and conversion of one model are `capability::Action`s on that model's provider, reachable from `brain serve` (HTTP, D-Bus) and from `brain <model> <action>` on the host. A param naming a file on the host is `.host_resolved()`; a trained or converted artifact also comes back as an output blob where its size allows.
- **`brain <model> <action>` is the only per-model command.** It is `crates/invoke` over the catalog: the action's manifest is its command line (every param, typed, with its default and bounds), files become input blobs, output blobs become files. No `crates/cli/src/<arch>_cli.rs` module and no `ARCH_HANDLERS` row remains.
- **Per-model samples** live at `samples/models/<arch>/` (package `sample-models-<arch>`): a Rust program over that model's crate and `crates/invoke`. `<sample> <action> [flags]` runs any served action with every knob and the manifest's defaults; on top of that a sample carries what is an application rather than a capability (an interactive window, a camera viewer, a game loop, a multi-step workflow written in Rust). Flags a user must give are only those an action requires; everything else defaults in the manifest, and the action validates every value before a weight loads.
- **Shell samples** that wrapped a per-model verb either move into that model's Rust sample or call `brain <model> <action>`; a wrapper that only renamed flags is deleted.
- **Generic verbs** stay only where they are not about one model: `serve`, `caps`, `pull`, `models`, `import`, `quantize`, `finetune`, `optimize`, `prune`, `perf`, `devices`, `doctor`, `roofline`, `plan`, `gguf`. The rest go into the model or sample that owns them.

## Done

- `crates/invoke`: the action runner `brain <model> <action>` and every sample share. `invoke::main` is a per-model program's whole `main` (global `--device`/`--backend`/`--brain-data-dir`, help from the static manifest, flags checked before any weight lookup, weights from the store, a clean device drain at exit). Two checkpoints of one model that tie in the store are settled by the action's `variant` (given, else its default). The catalog is the one table of which model resolves its weights through which spec.
- ltxv: `replace` is an action of `brain/ltxv` (`ltxv::replace`): SAM 2, FLUX.2 (through its served `edit`) and ArcFace come through `capability::Components`, which the catalog and samples implement over the model store (`loader::components::StoreComponents`); its work directory resumes as before. `t2v` gained `scenes`, `context_frames`, `max_window_tokens`, the reference video (`reference`, `reference_attention_mask`, their strengths) and `stage_2_ic_lora`; `v2v` a mask sequence, `strength` and `keyframes`; `extend` `seconds`; `upscale` `context_frames`; every generating action the runtime knobs (resident blocks, two-stage, VAE tiling, text-encoder precision and cache, latent dumps, the fp32 audio-video reference), host-only. A clip's frame rate from its container is honoured (it was read as an integer and dropped). `samples/models/ltxv` carries the `chain` workflow (`invoke::Workflow`); the shell videogen samples are gone.
- flux2: `text2image`/`edit` take everything the CLI verb did - inpainting (`mask`), up to eight references with the default 512 px bound (`ref_size`), the init image's size as the canvas, tiled canvases, `experimental_steps`, `ref_resolution_scale`, several adapters with their own strengths (`adapter` is a `path@strength` list, `model::adapter::AdapterSpec::parse_list`, for every model) - planned from the images' sizes alone (`flux2::request`), which is also what the scheduler keys a resident on; `lora_train` gained `ckpt_every`/`resume` (they sat on the generation actions), `precision`, `train_base`, `cache_dir`. An unset `precision` is the checkpoint's own, so a served GGUF DiT runs. `samples/models/flux2`.
- wan: `t2v` gained `dit_dtype` (it was read but not declared) and `te_device`; the global `--device` places the DiT and VAE. `lora_train`'s `report` writes the base-vs-adapted evidence (`wan::report`), its EVA-CLIP table from `report_clip` instead of an environment variable. `--variant` settles two DiTs of different sizes in one store. `samples/models/wan`.
- zipdepth: `infer` draws any view of the depth (`zipdepth::view`: side by side, colorized, fog, depth of field, the stereograms) with its colormap and stripes, and times warm inference (`bench`); `train` (from the store's checkpoint, or from scratch: `ActionSpec::from_scratch`, which `invoke` honours when the store has none) and `calib` (the INT8 outlier report) are actions; the serving adapter offers the same three, `infer` also on an NPU. A trained checkpoint keeps its tensors' shapes and loads like a released `.pth`. `samples/models/zipdepth` carries the `show` and `camera` windows, on the engine or the NPU (`npu::DepthSession`); `invoke::WorkflowCall` hands a workflow the resolved checkpoints.
- splat: `info`, `orient`, `merge` and `prune` are actions of `brain/splat` (`splat::tools` holds the scene logic), `fit` and `render` take every knob the verb had (camera files, `view`, Inria low-pass, `eps2d`, the reference rasterizer, the tile-instance budget, timing), and `sfm`/`train` sit next to the photogrammetry they run (`recon::caps`, which the catalog serves as `brain/splat`). The fly-through window is the `view` workflow of `samples/models/splat`.
- sam2: `track` is a served action (`sam2::track`); mask sequences are one format, `imaging::maskseq` (a one-channel clip blob between actions, a directory on disk), read by LTX's masked pass as well; `samples/models/sam2`.

## Decisions for what is left

- **Knobs only a host can set** (a file on the serving machine, a memory placement such as LTX's resident blocks or VAE tiling) are params declared `.host_resolved()`: `brain <model> <action>` and samples set them, a remote caller cannot.
- **A checkpoint role is a `--<role> PATH` flag**, never an action param: `invoke` takes every role the model's spec declares (LTX's `spatial_upsampler`, `detailing_lora`, ...).
- **A pipeline over several models** (LTX `replace`: SAM 2, FLUX.2, ArcFace and LTX) is one served action whose provider resolves each component model from the store; a component's checkpoint is a role named `<model>.<role>`.
- kronos and timesfm3 have a resolver spec in the catalog but no `brain <model>` provider yet (they are served by their residency adapters only); the forecast migration gives them one.

## Open, in order

Each row is one or more self-contained commits: the served action gains what the CLI verb had, the sample replaces the module, the module, its `ARCH_HANDLERS` row, its tests and its doc page move together.

| arch | CLI-only work to move | into |
|---|---|---|
| worldmirror2 | `infer` knobs and directory/video input, `import`, `export-npu` | `reconstruct`; conversion actions; sample |
| yolov8 | identity verification, NPU route, `train`/`finetune`/`eval` | `detect`; training actions; sample |
| qwen3tts | residual sampling knobs, NPU route, `import`, `finetune`, `sim`, `serve` | actions; sample |
| lfm2 | `data`, `finetune`, `eval`, `import` (`fill_mask`/`embed` already served) | actions; sample |
| glmdsa | `train`, `finetune`, `eval`, `import`, `export`, NPU decode, char vocab | actions; sample |
| ouro, huginn | engine sizing (`ctx`, `block`, `steps`), `import`, ouro `fit` | actions; samples |
| qwen35, qwen35moe | adapter overlay, `sample-adapter`, `finetune` modes, NPU `export` (imports are GGUF, generic) | actions; samples |
| qwen3 | `import`, `export`, `precompile`, `calib`, `train`, `finetune` extras, `eval`, `toolcall`, NPU decode, scheduler demo | actions; sample |
| deepseekvl, januspro, minicpmv | `finetune`, `export-peft` | training actions; samples |
| qwen3omnimoe | `import`; its served actions missing from `brain caps` | catalog registration; actions; sample |
| mirage | `quantize` | action; sample |
| gpt2, toypid, toymoe | train, sample text, eval, rollout, profile | a provider each; samples |
| diamond | `play`, `replay`, `bench`, `import`, `finetune`, `export` | game loop in the sample; actions |
| top level | `npu`, `label`, `forecast`, `decide` lifecycle, `federated`, `data`, `gradcheck` | the owning model's actions and samples; generic tools stay where generic |
| cli | `resolver_cli`'s private arch-to-spec table duplicates `catalog::resolver_spec_for`; help text; `docs/reference/cli.md` | one table; regenerated reference |
