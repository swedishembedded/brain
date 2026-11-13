# Configuration

Ordinary use needs no configuration: `brain serve` serves what the model store
holds on every device it finds, and a local client finds the server by itself
(see [Serving](../guides/serving/serving.md#local-clients)). Choices are flags,
which a config file can persist. Some settings are still `BRAIN_*` environment
variables; this page lists every one of them, grouped by purpose.

Unset variables take documented defaults; a model whose weights variable is
unset simply isn't served (no error, it's just absent from `brain caps`).

## Config file (optional)

`~/.config/brain/config.yaml` (`$XDG_CONFIG_HOME/brain/config.yaml`) is read
when it exists; `--config <FILE>` (global, valid on any subcommand) names
another. It holds two kinds of key:

- a subcommand (`serve`, `qwen3 infer`, ...) mapping that command's flags,
  written without the leading dashes, to values: a list for a repeatable
  flag, `true` for a switch. A flag given on the command line always wins
  over the file. The longest matching subcommand applies.
- a `BRAIN_*` variable from the tables below, set for the run. It fills gaps:
  a variable the caller already exported always wins over the file.

```yaml
# ~/.config/brain/config.yaml
serve:
  api-key-file: /etc/brain/key
  checkpoint: checkpoints/qwen3.8-27b
BRAIN_QWEN35MOE_CTX: 8192
```

A variable brain no longer reads (`BRAIN_API_KEY`, `BRAIN_QWEN_WEIGHTS`,
`BRAIN_QWEN_TOKENIZER`, `BRAIN_RUNTIME_DIR`) is refused in the file, naming
the flag that replaced it, and has no effect in the environment.

## Device selection

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_DEVICE` | for programs with no CLI in the loop (test binaries): the schedulable compute set (`cpu`, `gpu`, `npu`, `gpu0`, `gpu,cpu`, …); same values as `--device`; a value that does not resolve on the machine stops the program, naming it. The `brain` CLI ignores it (and says so); pass `--device` | all detected devices |
| `BRAIN_BACKEND` | for programs with no CLI in the loop: how the selected hardware is driven: `wgpu`, `vulkan`, `cuda` or `cpu`; same values as `--backend`, which the `brain` CLI takes instead. `cuda` needs a GPU in the device set; an unknown or unrunnable backend stops the program, naming it, rather than falling back | chosen from the hardware: `cuda` where every card is NVIDIA's and CUDA can run (driver plus NVRTC or the ahead-of-time kernels), `wgpu` on any other GPU, else `cpu`; `brain devices` prints the choice and why |
| `BRAIN_VK_DEVICE` | forces a specific Vulkan physical-device index, overriding brain's discrete-GPU-first ranking | automatic ranking |
| `BRAIN_QWEN_STEP_CACHE` | `0` disables the Qwen capability path's recorded-step cache, re-recording every dispatch - an A/B seam for measuring what the recording buys | on |
| `BRAIN_SPLAT_PROFILE` | set (any value) to print `splat fit`'s per-stage timing breakdown | off |
| `BRAIN_GPU_NO_READ_STAGING_REUSE` | `1` makes every device-to-host readback allocate its own staging buffer instead of reusing the device's, trading throughput for a smaller resident host footprint | off (the buffer is reused) |

The Qwen3-TTS placement, streaming and diagnostics variables (`BRAIN_QWEN3TTS_TALKER`,
`BRAIN_QWEN3TTS_MTP`, `BRAIN_QWEN3TTS_CODEC`, `BRAIN_QWEN3TTS_STREAM_CHUNK`,
`BRAIN_QWEN3TTS_STREAM_WIN`, `BRAIN_QWEN3TTS_NPU_DEVICE`, `BRAIN_QWEN3TTS_REF`,
`BRAIN_QWEN3TTS_REF_TEXT`, `BRAIN_QWEN3TTS_RES`, `BRAIN_NPU_TURBO`, `TTS_PLAN`,
`TTS_PROFILE`, `TTS_NPU_PARITY`) are no longer read.
They are parameters now: `ov_device`, `turbo`, `talker`, `mtp`, `parity` and `profile` on
`sample-models-qwen3tts`'s `npu-synth`/`npu-clone`/`npu-design` actions, `--npu-device`,
`--turbo`, `--mtp`, `--codec`, `--stream-chunk`, `--stream-window` and `--profile` on its
`npu-serve` workflow, `ref` and `ref_text` on the `clone` action, and `stream_chunk` and
`profile` on the served ones.

`BRAIN_GPU_INDEX` is no longer read. Pin one card with `--device gpu<i>`, or with
`BRAIN_DEVICE=gpu<i>` in a program that has no CLI.

### Tracing

`--trace-<family> <0-5>` turns on structured, per-component tracing for one
family of crates (`brain help` lists the families and levels; `--trace
<family>=<level>` reaches any family generically, `--trace-format text|json`
and `--trace-output -|PATH` control rendering and destination). One variable
sets the same levels for entry points with no CLI in the loop:

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_TRACE` | comma-separated `<family>=<level>` trace levels, e.g. `ltxv=5,gpu=3`; any `--trace*` flag on the command line overrides it entirely | unset (no tracing) |

This is separate from the `--kernel-profile` flag, which prints per-kernel dispatch
timing at exit: tracing is an event stream over a run, profiling is a summary
table over its kernels.

### Backend workarounds & debugging

Most users never need these - they exist for GPU-driver quirks and profiling.

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_VK_SERIAL` | forces one-dispatch-at-a-time submission on the native-Vulkan backend; works around a hang on some Intel integrated-GPU drivers | auto-detected by vendor |
| `BRAIN_VK_NO_SERIAL` | forces the opposite of `BRAIN_VK_SERIAL` even on a vendor that's normally auto-serialized | unset |
| `BRAIN_VK_VALIDATE` | enables Vulkan validation layers on the native-Vulkan backend | off |
| `BRAIN_GPU_GL` | forces the OpenGL backend instead of Vulkan (wgpu) | off |
| `BRAIN_GPU_VALIDATION` | enables wgpu debug/validation instance flags | off |
| `BRAIN_GPU_CHECKED` | enables extra wgpu-side checked-mode assertions | off |
| `BRAIN_GPU_MEM_PERF` | switches wgpu's memory allocator to its performance-hint mode | off |
| `BRAIN_VERBOSE` | CLI/serving log verbosity (`0` = quiet) | 0 |

**NPU capability boundary.** brain can run a subset of models on an Intel NPU
through a separate export -> compile -> run path (ONNX export, OpenVINO
compile, named-tensor inference), with fp16 as the default precision and
INT8/INT4 as opt-in. The residency scheduler (`residency::place::pick_device`)
**does** auto-place a model on the NPU whenever that model advertises an NPU
footprint (`MemCost::with_npu`) - NPU is tried before GPU and CPU whenever a
model declares one. Today that covers ZipDepth (depth), the two ASR models
(Nemotron, Qwen3-ASR), and the forecasting models (chronos2, fincast, kronos)
- check a given model's own resident wiring (or its page under `docs/reference/models/`)
for whether it declares an NPU footprint, since this is per-model, not every
served model. `--device npu` does not bypass this: it constrains/forces the
*schedulable device set* for a request rather than being the only way to
reach the NPU, so it is how you force NPU placement (or rule it out) rather
than the sole trigger for ever using it.

## Model weights & gating

`brain serve` resolves each model's weights from the model store
(`--models-dir`); `--pin ARCH.ROLE=PATH` serves one role from an explicit path
instead, and settles a store that holds two candidates for it. A model whose
weights resolve to nothing is not served, with no error. The variables below
are still read by the models that have not moved onto the store; unset, such a
model is not served.

| Variable | Serves | Value |
| --- | --- | --- |
| `BRAIN_QWEN35MOE_WEIGHTS` + `BRAIN_QWEN35MOE_TOKENIZER` | Qwen3.5 MoE chat | checkpoint (produced by `brain import`) + `tokenizer.json` |
| `BRAIN_GPT2_WEIGHTS` | char-level GPT baseline | checkpoint (embeds its vocab) |
| `BRAIN_GLMDSA_WEIGHTS` | GLM decoder | checkpoint (char-level) |
| `BRAIN_S3DIT_DIT`, `BRAIN_S3DIT_VAE`, `BRAIN_S3DIT_QWEN`, `BRAIN_S3DIT_TOKENIZER` | Z-Image text2image/edit | optional overrides, one per component -- unset, each is resolved from the model store instead (`s3dit::spec::S3ditSpec`); set, it pins that exact file |
| `BRAIN_SCRFD_DIR` | antelopev2 face detection | dir holding `scrfd_10g_bnkps.onnx` |
| `BRAIN_ARCFACE_DIR` | antelopev2 face identity embedding | dir holding `glintr100.onnx` (plus `scrfd_10g_bnkps.onnx` for the default `align=true` path) |
| `BRAIN_CLIP_DIR` | CLIP text/image embeddings | SDXL-layout root (`tokenizer[_2]/`, `text_encoder[_2]/`, EVA `.pt`) |
| `BRAIN_FLUX1_DIR` | FLUX.1 text2image | diffusers-layout root; must hold `transformer/`, or the model is not served |
| `BRAIN_FLUX1_DIR` + `BRAIN_PULID_DIR` + `BRAIN_ARCFACE_DIR` + `BRAIN_CLIP_DIR` + `BRAIN_BISENET_DIR` | PuLID identity-conditioned FLUX.1 text2image (all five required) | the FLUX.1 root plus the PuLID weights dir, the face/CLIP dirs above, and a dir holding `parsing_bisenet.safetensors` (see `pulid::caps::BISENET_FILE`'s doc for how to produce one) |
| _(none)_ | Qwen3.8-27B dense hybrid decoder, FastVLM vision-language, Moondream 3 vision-language, DeepSeek-OCR and DeepSeek-OCR-2 document image → text/markdown - all resolved automatically from the models directory (`qwen35`/`fastvlm`/`moondream3`/`deepseek2ocr::spec`/`deepseekocr2::spec`); `--weights`/`weights` (and `--tokenizer` for qwen35) still name a checkpoint outright per call | n/a |
| `BRAIN_CHRONOS2` | Chronos-2 forecasting | weights |
| `BRAIN_FINCAST` | FinCast forecasting | weights |
| `BRAIN_TIMESFM3` | TimesFM-3 forecasting (natively multivariate; target-only over this served path). `brain pull google/timesfm-3.0-pytorch` fetches the checkpoint, but no caller sets this variable for you yet - point it at the fetched directory (or an imported `.safetensors`) yourself | weights |
| `BRAIN_KRONOS_TOKENIZER` + `BRAIN_KRONOS_DECODER` | Kronos OHLCV forecasting (the served path, `brain serve`, and `brain forecast compare`/`finetune`). `brain forecast predict` instead resolves both roles from the models directory (`NeoQuasar/Kronos-Tokenizer-base` + `NeoQuasar/Kronos-base`, told apart by their own `config.json` content) via `--tokenizer-dir`/`--decoder-dir` or automatically when the store holds exactly one candidate each | the two checkpoint dirs (the decoder also accepts a `.safetensors` fine-tune file) |
| `BRAIN_KRONOS_ARGMAX` | force Kronos's deterministic modal rollout (argmax over the token distribution) instead of nucleus sampling: one reproducible path, N times cheaper. Set by `brain forecast predict --samples 1` | `0` (sample) |
| `BRAIN_QWEN3TTS_WEIGHTS` (+ `BRAIN_QWEN3TTS_CKPT`) | Qwen3-TTS in the `qwen3tts-synth` stdio controller only; `brain serve` and `brain qwen3tts` resolve the checkpoint from the model store (`--weights-dir`/`--ckpt` name one outright) | the checkpoint dir, or a dir of files from `brain qwen3tts import` (+ HF checkpoint dir for config/tokenizer) |
| `BRAIN_QWEN3OMNIMOE_HF_DIR` | Qwen3-Omni Thinker, validation tier - full chat + audio/image/video + `speak`, but weights re-stream from the checkpoint per generated token | HF checkpoint dir |
| `BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT` | Qwen3-Omni Thinker, **GPU-resident** - W8A16 (int8 weight-only, MoE expert linears, full-precision activations) sharded across as many GPUs as they need, loaded with bounded host memory. A separate model id (`brain/Qwen3-Omni-30B-A3B-Instruct-W8A16`), not a variant of the above; the downloaded checkpoint directory (quantized as it loads), or a file `brain qwen3omnimoe import` wrote. Text-only, but ~25x the tokens/second of the streaming path above | unset (not served) |
| `BRAIN_QWEN3OMNIMOE_INT8_TOKENIZER_DIR` | where the int8 model reads `tokenizer.json` (or `vocab.json` + `merges.txt`) so it can serve ordinary chat requests - an int8 checkpoint is a single file with no tokenizer sibling. Without one it still serves raw token ids, but is not on `/v1/chat/completions` | the checkpoint's own directory if it has tokenizer files, else `BRAIN_QWEN3OMNIMOE_HF_DIR` |
| `BRAIN_MOCK` | deterministic, weight-free mock model (for exercising the serving stack without real weights) | any non-empty value |

## Serving & admission

The two admission deadlines are `brain serve` flags, not variables: `--admit-deadline-ms` and
`--cold-build-admit-deadline-ms` (defaults and meaning in
[`serving.md`](../guides/serving/serving.md#admission-and-backpressure)).

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_MODELS_DIR` | model directory scanned at startup (also `--models-dir`) | `$XDG_DATA_HOME/brain/models` |
| `BRAIN_AUTO_FETCH` | `1`/`true`/`on` enables on-demand fetching of missing model files (`--autofetch` publishes the same thing and wins over an exported `0`); a pulled model still resolves with the gate off | disabled |
| `BRAIN_SCHED_MAX_BATCH` | maximum jobs the scheduler groups into one batch | 8 |
| `BRAIN_SCHED_AGE_WEIGHT` | scheduler priority weight per millisecond a job has waited | 1.0 |
| `BRAIN_SCHED_BATCH_WEIGHT` | scheduler priority weight favoring larger batches | 200.0 |
| `BRAIN_SCHED_MAX_WAIT_MS` | a group whose oldest job has waited longer than this is force-picked regardless of batch size | 2000 |
| `BRAIN_LIMIT_VRAM_TOTAL` | environment fallback for `--limit-vram-total`: a ceiling on VRAM the whole process may hold, summed over every GPU (not per card) | no ceiling |
| `BRAIN_LIMIT_RAM_TOTAL` | environment fallback for `--limit-ram-total`: a ceiling on host RAM the process may hold (CPU + every NPU) | no ceiling |

See [`docs/guides/serving/serving.md`](../guides/serving/serving.md) for what admission/backpressure means in practice.

## Precision & tuning

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_QWEN35MOE_CTX` | Qwen3.5 MoE built context length | model default |
| `BRAIN_QWEN35MOE_MAX_BATCH` | Qwen3.5 MoE serving batch slots | model default |
| `BRAIN_FLUX2_MAX_BATCH` | FLUX.2 concurrent same-size batch cap | 4 |
| `BRAIN_FLUX2_FAMILY` | `klein` or `base`, combined with the DiT's own sniffed size (`4b`/`9b`) to pick the served variant when a resident is bound to explicit weight paths rather than a `--variant` flag | `klein` |
| `BRAIN_FLUX2_TE_DEVICE` | **Override** for FLUX.2 text-encoder placement (`gpu<i>[:i8]` for a truncated shard on that card). Unset, placement is automatic: the pipeline declares the DiT, encoder and VAE with their costs and the engine places them on cards that can hold them (`residency::plan`, reached through `gpu_core::devices::place`), printing what it chose | automatic |
| `BRAIN_FLUX1_TE_DEVICE` | Same override, for FLUX.1's T5-XXL/CLIP-L text-encoder placement (`cpu` or `gpu<i>`) - the escape hatch for when the automatic plan doesn't fit some other box | automatic |
| `BRAIN_FLUX2_NO_STREAM` | `1` forces a non-GGUF FLUX.2 DiT to build from a whole fp32 tensor map. It is rejected for a `.gguf` DiT: decoding a quantized GGUF into a whole fp32 map is not an allowed implicit conversion | off |
| `BRAIN_FLUX2_TE_NO_STREAM` | `1` forces the FLUX.2 text encoder to be imported as one whole fp32 map instead of streamed per tensor from a mapping. Same character as `BRAIN_FLUX2_NO_STREAM`: identical weights either way, kept as an A/B instrument and a fallback | off (streamed) |
| `BRAIN_FLUX2_ALLOW_NC` | `1` opts in to the FLUX Non-Commercial-licensed 9B variants | required, not set |
| `BRAIN_FLUX2_LORA_FOLD` | `1` puts an INT8 FLUX.2 build back on the old LoRA deployment: the adapter is baked into the weights and the result requantized, instead of being applied beside the base matmul in fp32. Unlike `BRAIN_FLUX2_NO_STREAM` the two are NOT the same bytes - folding into a 256-level grid discards most of a trained delta (measured: 94.8% of adapted weights round back to their base code at a delta 0.5% of the weight magnitude). An A/B instrument, not a tier to ship | off (runtime correction) | <!-- perf-number: measured result; reproduced by the named model binary/test (`brain perf run` for timings) -->
| `BRAIN_FLUX1_I8_KEEP_F32` / `BRAIN_FLUX2_I8_KEEP_F32` | keeps a specific sub-layer at fp32 under INT8 inference, trading a little memory for accuracy | off |
| `BRAIN_S3DIT_RETAIN_INT8_CACHE` | `1` retains the int8 DiT cache across demote/promote (trades multi-GB host RAM for faster reactivation) | off |
| `BRAIN_S3DIT_ENCODER_GPU` | which GPU card hosts the Z-Image text encoder | shares the DiT's card |
| `BRAIN_S3DIT_ENCODER_CUT` | fp32 2-card encoder split point (layer index) | ~2/3 of layers |
| `BRAIN_S3DIT_ENCODER_FP32SPLIT` | `1` enables the fp32 2-card encoder split | off |
| `BRAIN_S3DIT_ENCODER_RESIDENT` | `1` keeps the encoder resident even when it shares the DiT's card | off (build on demand) |
| `BRAIN_S3DIT_WINDOW_BLOCKS` | blocks kept resident in the fp32 single-GPU decode window | 2 |
| `BRAIN_S3DIT_FLASH` | `1`/`0` forces flash attention on/off for Z-Image | automatic |
| `BRAIN_WAN_ATTN` | `flash`/`chunked` forces the Wan DiT's self-attention implementation | `flash` where the device supports it |
| `BRAIN_WAN_T5_DEVICE` | where the umT5-XXL text encoder runs (`cpu`/`gpu`); it is 22.72 GB in fp32 and does not fit a 24 GB card | `cpu` |
| `BRAIN_FLUX1_VAE_DEVICE` | where the FLUX.1 VAE decode runs (`cpu`/`gpu<i>`) | `cpu` |
| `BRAIN_QWEN3TTS_LANG` | the `qwen3tts-synth` stdio controller's default TTS language (the `lang` param of the served actions) | `english` |
| `BRAIN_MIMI_WEIGHTS` | TTS codec weights override | derived from the resolved qwen3tts `weights_dir` role |
| `BRAIN_QWEN3TTS_SPEAKER` | overrides the speaker-encoder weights used for voice-clone evaluation | derived from the resolved qwen3tts `weights_dir` role |
| `BRAIN_MINILM_DIR` | encoder checkpoint directory for the `salesconv_embed` developer binary | the MiniLM checkpoint under the models directory |
| `BRAIN_FORECAST_HORIZON` / `BRAIN_FORECAST_SAMPLES` | forecast horizon / sample count | 64 / 1 |
| `BRAIN_MOCK_DELAY_MS` | mock model artificial latency | 0 |

The Qwen3 and Qwen3.8 serving sizes are `brain serve` flags, not variables:
`--ctx` (default: auto-sized to the card), `--max-batch` (default
16), the KV pool size (default: batch x half a context when it fits), `--kv-calibration`, `--kv-cache fp8`, `--ctx` (default: the longest the cards hold)
and `--max-batch` (default 8); `--exact-decode` makes a decode row's tokens and
per-token log-probabilities independent of its co-batch (default off); a
Qwen3.8 GGUF's unset `--kv-cache`,
`--speculative` and prompt cache are chosen from the hardware and logged at startup. See the
[Qwen3](models/qwen3.md) and [Qwen3.8](models/qwen35.md) pages.

Deeper per-model knobs not listed here are documented on that model's own
page under `docs/reference/models/`.

## Paths

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_MODELS_DIR` | model directory scanned at startup | `$XDG_DATA_HOME/brain/models` |
| `BRAIN_MINILM_DIR` | encoder checkpoint directory for the `salesconv_embed` developer binary | the MiniLM checkpoint under the models directory |
| `BRAIN_HUB_ENDPOINT` | additional allowed host for model downloads (checked before the standard `HF_ENDPOINT`), both as a base URL and as a redirect target - never a blanket "trust any redirect" | the standard hub host only |

### Which models directory wins

There is one answer to "where do models live", and everything reads it:
`brain pull`, auto-fetch, the served catalog scan, and every capability
provider. Highest precedence first:

1. `--models-dir DIR` - the per-subcommand flag, where a subcommand has one.
   It names a models directory directly.
2. `--brain-data-dir DIR` - the **global** flag (valid on any subcommand). It
   names brain's *data root*; models live in `DIR/models`. Default
   `~/.local/share/brain`.
3. `$BRAIN_MODELS_DIR`
4. `$XDG_DATA_HOME/brain/models`
5. `$HOME/.local/share/brain/models`

Note that `--brain-data-dir` **does** outrank an explicitly-set
`$BRAIN_MODELS_DIR` - a flag the user typed beats an inherited environment
variable, the same rule `--models-dir` and `--device` follow. Because a models
directory is where tens of gigabytes land, brain says so on stderr when the
flag actually overrules a differing `$BRAIN_MODELS_DIR`, rather than choosing
silently.
