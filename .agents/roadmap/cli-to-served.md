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
- flux2: `text2image`/`edit` take everything the CLI verb did - inpainting (`mask`), up to eight references with the default 512 px bound (`ref_size`), the init image's size as the canvas, tiled canvases, `experimental_steps`, `ref_resolution_scale`, several adapters with their own strengths (`adapter` is a `path@strength` list, `model::adapter::AdapterSpec::parse_list`, for every model) - planned from the images' sizes alone (`flux2::request`), which is also what the scheduler keys a resident on; `lora_train` gained `ckpt_every`/`resume` (they sat on the generation actions), `precision`, `train_base`, `cache_dir`. An unset `precision` is the checkpoint's own, so a served GGUF DiT runs. `samples/models/flux2`.
- wan: `t2v` gained `dit_dtype` (it was read but not declared) and `te_device`; the global `--device` places the DiT and VAE. `lora_train`'s `report` writes the base-vs-adapted evidence (`wan::report`), its EVA-CLIP table from `report_clip` instead of an environment variable. `--variant` settles two DiTs of different sizes in one store. `samples/models/wan`.
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
| ltxv | `replace`; knobs missing from `t2v`/`v2v`/`upscale`/`dfr`/`extend` (scene, reference video, keyframes, window, strength, factor, role files); mp4 output | actions; `replace` as an action of its own; sample; delete `samples/shell/videogen/{ltxv,character_replace}` |
| splat | `orient`, `merge`, `prune`, `sfm`, `train`, `fit`/`render` knobs; `view`, `info` | actions; `view`/`info` in the sample |
| worldmirror2 | `infer` knobs and directory/video input, `import`, `export-npu` | `reconstruct`; conversion actions; sample |
| zipdepth | views, colormaps, headless composite; `camera`; `calib`; `train` | actions; camera viewer and window in the sample |
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
