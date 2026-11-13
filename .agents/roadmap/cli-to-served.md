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

- `crates/invoke`: the action runner `brain <model> <action>` and every sample share.

## Open, in order

Each row is one or more self-contained commits: the served action gains what the CLI verb had, the sample replaces the module, the module, its `ARCH_HANDLERS` row, its tests and its doc page move together.

| arch | CLI-only work to move | into |
|---|---|---|
| sam2 | `track` (mask sequence from clicks) | `brain/sam2` action `track`; sample |
| ltxv | `replace`; knobs missing from `t2v`/`v2v`/`upscale`/`dfr`/`extend` (scene, reference video, keyframes, window, strength, factor, role files); mp4 output | actions; `replace` as an action of its own; sample; delete `samples/shell/videogen/{ltxv,character_replace}` |
| flux2 | inpaint mask, tiling, reference scales, `finetune` knobs (precision, cache dir, train base, previews) | `brain/flux2-klein` actions; sample; shell imagegen samples call the action |
| wan | device placement, dtype, role files, `finetune --report` | `t2v`/`lora_train`; sample |
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
