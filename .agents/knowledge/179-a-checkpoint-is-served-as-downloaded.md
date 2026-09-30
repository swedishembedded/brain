<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 179. A checkpoint is served as downloaded

`brain pull` used to finish a Qwen3-family fetch in two steps. It wrote an
fp32 `model.brain.safetensors` beside the download, then deleted the
download. Both steps were wrong:

- **The copy was bigger than what it replaced.** The upstream files are
  bf16 or fp16, and the copy was fp32: 28 GB for a 7B, where the download
  was 14 GB.
- **Deleting the download destroyed the only faithful record.** Parity
  tooling, the reference dumps and any later re-import read the upstream
  files, and after the pull there was nothing left for them to read.

The qwen3 decoder and its `llama`/`qwen2` variant rows are now served from
the files as downloaded, safetensors or `pytorch_model*.bin`:

- The pull writes one `brain.manifest.json` whose `weights` role is the
  repo directory itself (role paths are normalized, so `"."` is the
  directory).
- `qwen3::open_checkpoint` opens a Hugging Face directory through the
  renaming source (`import::owned_source`, backed by
  `RemapSource::owning`), with the config read by `hf::decoder_config`.
- `serve::Engine::tensors_from` and the resident read each tensor through
  that source, converting it to f32 as it is read. That conversion happens
  only because the engine holds f32 or int8 weights; the half-precision
  tensors are never written back to disk as f32.
- The resident's fp32 weight estimate comes from the config, since file
  size means nothing for a directory, or for a bf16 file that doubles on
  the device.

glmdsa and lfm2 read their downloads the same way: a fetch plan per
parameter over the Hugging Face tensors (`glmdsa::import::plan`, where the
per-head de-interleave and the packed experts are plain slices, and a
rename for `lfm2::import::plan`). qwen3omnimoe's int8 resident reads its
download through `import::Int8View`, a `checkpoint::weightio::DerivedCheckpoint`.
That view packs each weight to int8 when it is read, the conversion the
30B model needs in order to fit the cards, and it never writes to disk.
Each crate's explicit `import` command writes out the same view, so there
is one mapping per family.

yolo reads its downloaded `.pt` directly. qwen3tts reads its four
components from the download in the same way:

- the Talker and the MTP through `qwen3tts::import::{talker_view, mtp_view}`,
  each a `checkpoint::weightio::Renamed` view checked against the
  component's parameter list;
- the speaker encoder through `ecapatdnn::import::view`;
- the codec through `mimi::import::view`, which collapses each codebook to
  its table as it is read.

A view carries its loader's config (`DerivedCheckpoint::config`), so
`checkpoint::load_reader` and the NPU exporters, which now take a
`WeightReader` rather than a path, treat a view exactly as they treat a
brain file. `qwen3tts::TtsPaths::new` is the one place that knows the
layout. A directory holding `talker.safetensors` is read as imported files;
any other is read as the download, with its codec in `speech_tokenizer/`.
NPU graph caches for a download go under brain's cache directory, never
into the model store.

No family's pull converts on disk any more, so `brain-loader` links no
model crate.

`a_hugging_face_directory_is_served_as_downloaded` generates from a tiny
Llama directory and asserts that nothing is written beside it. The loader
test pulls a `.bin`-only Llama and checks that its original files are kept
and read back exactly.
