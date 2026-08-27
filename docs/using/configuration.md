# Configuration

brain has no config file. Every choice - which models `brain serve` activates,
where they run, and how they're tuned - is a `BRAIN_*` environment variable,
set before you run `brain serve` or a `brain <model>` subcommand. This page is
the complete reference, grouped by purpose.

Unset variables take documented defaults; a model whose weights variable is
unset simply isn't served (no error, it's just absent from `brain caps`).

## Device selection

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_DEVICE` | the schedulable compute set (`cpu`, `gpu`, `npu`, `gpu0`, `gpu,cpu`, …); same values as `--device` | all detected devices |
| `BRAIN_GPU_INDEX` | pins a specific GPU card index (parsed once, at first use) | first/best card |
| `BRAIN_VK_DEVICE` | forces a specific Vulkan physical-device index, overriding brain's discrete-GPU-first ranking | automatic ranking |
| `BRAIN_GPU_WAIT_S` | seconds to wait for a GPU submit to complete before treating the device as wedged | backend default |
| `BRAIN_GPU_NO_READ_STAGING_REUSE` | `1` makes every device-to-host readback allocate its own staging buffer instead of reusing the device's, trading throughput for a smaller resident host footprint | off (the buffer is reused) |
| `BRAIN_NPU_TURBO` | `1`/`yes` requests the Intel NPU's turbo clock during inference | off |

### Tracing

`--trace-<family> <0-5>` turns on structured, per-component tracing for one
family of crates (`brain help` lists the families and levels; `--trace
<family>=<level>` reaches any family generically, `--trace-format text|json`
and `--trace-output -|PATH` control rendering and destination). One variable
sets the same levels for entry points with no CLI in the loop:

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_TRACE` | comma-separated `<family>=<level>` trace levels, e.g. `ltxv=5,gpu=3`; any `--trace*` flag on the command line overrides it entirely | unset (no tracing) |

This is separate from `BRAIN_PROFILE` below, which prints per-kernel dispatch
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
| `BRAIN_PROFILE` | enables per-kernel dispatch timing, printed at exit | off |
| `BRAIN_VERBOSE` | CLI/serving log verbosity (`0` = quiet) | 0 |

**NPU capability boundary.** brain can run a subset of models on an Intel NPU
through a separate export -> compile -> run path (ONNX export, OpenVINO
compile, named-tensor inference), with fp16 as the default precision and
INT8/INT4 as opt-in. The residency scheduler (`residency::place::pick_device`)
**does** auto-place a model on the NPU whenever that model advertises an NPU
footprint (`MemCost::with_npu`) - NPU is tried before GPU and CPU whenever a
model declares one. Today that covers ZipDepth (depth), the two ASR models
(Nemotron, Qwen3-ASR), and the forecasting models (chronos2, fincast, kronos)
- check a given model's own resident wiring (or its page under `docs/models/`)
for whether it declares an NPU footprint, since this is per-model, not every
served model. `--device npu` does not bypass this: it constrains/forces the
*schedulable device set* for a request rather than being the only way to
reach the NPU, so it is how you force NPU placement (or rule it out) rather
than the sole trigger for ever using it.

## Model weights & gating

Whether a model shows up in `brain caps` / is servable by `brain serve` is
gated entirely by whether its weights variable(s) are set. Unset ⇒ not
served, with no error.

| Variable | Serves | Value |
| --- | --- | --- |
| `BRAIN_QWEN_WEIGHTS` + `BRAIN_QWEN_TOKENIZER` | Qwen3 chat (`generate`) | `.brain` checkpoint + `tokenizer.json` |
| `BRAIN_QWEN35MOE_WEIGHTS` + `BRAIN_QWEN35MOE_TOKENIZER` | Qwen3.5 MoE chat | checkpoint (produced by `brain import`) + `tokenizer.json` |
| `BRAIN_GPT2_WEIGHTS` | char-level GPT baseline | checkpoint (embeds its vocab) |
| `BRAIN_GLMDSA_WEIGHTS` | GLM decoder | checkpoint (char-level) |
| `BRAIN_LFM2` + `BRAIN_LFM2_TOKENIZER` | LFM2.5-Encoder (`fill-mask`/`embed`) | weights + `tokenizer.json` |
| `BRAIN_FLUX2_DIT`, `BRAIN_FLUX2_VAE`, `BRAIN_FLUX2_TE`, `BRAIN_FLUX2_TOKENIZER` | FLUX.2 Klein text2image/edit | the four component paths (all required) |
| `BRAIN_S3DIT_DIT`, `BRAIN_S3DIT_VAE`, `BRAIN_S3DIT_QWEN`, `BRAIN_S3DIT_TOKENIZER` | Z-Image text2image/edit | the four component paths (all required) |
| `BRAIN_YOLOV8` | YOLOv8 detection | checkpoint |
| `BRAIN_ZIPDEPTH_WEIGHTS` | ZipDepth monocular depth | `.pth` checkpoint |
| `BRAIN_SAM2_WEIGHTS` | SAM 2.1 segmentation | `sam2.1_hiera_*.pt` checkpoint |
| `BRAIN_SCRFD_DIR` | antelopev2 face detection | dir holding `scrfd_10g_bnkps.onnx` |
| `BRAIN_ARCFACE_DIR` | antelopev2 face identity embedding | dir holding `glintr100.onnx` (plus `scrfd_10g_bnkps.onnx` for the default `align=true` path) |
| `BRAIN_ESRGAN_WEIGHTS` | Real-ESRGAN upscale | `RealESRGAN_x4plus.pth` (or any RRDBNet) |
| `BRAIN_CODEFORMER_WEIGHTS` | CodeFormer face restore | `codeformer.pth` (or its dir) |
| `BRAIN_VQGAN_WEIGHTS` | CodeFormer VQ encode/decode | checkpoint (or its dir) |
| `BRAIN_CLIP_DIR` | CLIP text/image embeddings | SDXL-layout root (`tokenizer[_2]/`, `text_encoder[_2]/`, EVA `.pt`) |
| `BRAIN_T5ENCODER_DIR` | T5-XXL / umT5-XXL text encoding (`encode`) | root holding `text_encoder_2/`+`tokenizer_2/` (the `flux_xxl` variant) and/or `wan/` (the `wan_umt5` variant) |
| `BRAIN_SDXL_DIR` | SDXL text2image | diffusers-layout root; must hold `unet/`, or the model is not served |
| `BRAIN_SDXL_DIR` + `BRAIN_CONTROLNET_DIR` | SDXL + ControlNet text2image (both required) | the SDXL root above plus a diffusers ControlNet root |
| `BRAIN_FLUX1_DIR` | FLUX.1 text2image | diffusers-layout root; must hold `transformer/`, or the model is not served |
| `BRAIN_FLUX1_DIR` + `BRAIN_PULID_DIR` + `BRAIN_ARCFACE_DIR` + `BRAIN_CLIP_DIR` | PuLID identity-conditioned FLUX.1 text2image (all four required) | the FLUX.1 root plus the PuLID weights dir, and the face/CLIP dirs above |
| `BRAIN_QWEN35_DIR` | Qwen3.8-27B dense hybrid decoder | HF checkpoint dir |
| `BRAIN_FASTVLM_WEIGHTS` | FastVLM vision-language | checkpoint directory |
| `BRAIN_QWEN3VL_WEIGHTS` | Qwen-VL vision-language (`brain caps`/`brain do` only - not yet residency-scheduled) | checkpoint directory |
| `BRAIN_DEEPSEEK_OCR_DIR` | DeepSeek-OCR document image → text/markdown (CPU-resident, ~22 GiB) | dir holding `mmproj-DeepSeek-OCR-Q8_0.gguf` + `DeepSeek-OCR-Q8_0.gguf` |
| `BRAIN_CHRONOS2` | Chronos-2 forecasting | weights |
| `BRAIN_FINCAST` | FinCast forecasting | weights |
| `BRAIN_KRONOS_TOKENIZER` + `BRAIN_KRONOS_DECODER` | Kronos OHLCV forecasting. Auto-fetched from `NeoQuasar/Kronos-Tokenizer-base` + `NeoQuasar/Kronos-base` - one model, two upstream repos - so both are normally unset | the two checkpoint dirs (the decoder also accepts a `.safetensors` fine-tune file) |
| `BRAIN_KRONOS_ARGMAX` | force Kronos's deterministic modal rollout (argmax over the token distribution) instead of nucleus sampling: one reproducible path, N times cheaper. Set by `brain forecast predict --samples 1` | `0` (sample) |
| `BRAIN_QWEN3TTS_WEIGHTS` (+ `BRAIN_QWEN3TTS_CKPT`) | Qwen3-TTS `speak` | brain-format weights dir (+ HF checkpoint dir for config/tokenizer) |
| `BRAIN_NEMOTRONASR` | Nemotron 3.5 streaming ASR | HF checkpoint dir |
| `BRAIN_QWEN3ASR` | Qwen3-ASR offline ASR | HF checkpoint dir |
| `BRAIN_QWEN3OMNIMOE_HF_DIR` | Qwen3-Omni Thinker, validation tier - full chat + audio/image/video + `speak`, but weights re-stream from the checkpoint per generated token | HF checkpoint dir |
| `BRAIN_QWEN3OMNIMOE_INT8_CHECKPOINT` | Qwen3-Omni Thinker, **GPU-resident** - W8A16 (int8 weight-only, MoE expert linears, full-precision activations) sharded across as many GPUs as they need, loaded with bounded host memory. A separate model id (`brain/Qwen3-Omni-30B-A3B-Instruct-W8A16`), not a variant of the above; build the file with `brain omni import`. Text-only, but ~25x the tokens/second of the streaming path above | unset (not served) |
| `BRAIN_QWEN3OMNIMOE_INT8_TOKENIZER_DIR` | where the int8 model reads `tokenizer.json` (or `vocab.json` + `merges.txt`) so it can serve ordinary chat requests - an int8 checkpoint is a single file with no tokenizer sibling. Without one it still serves raw token ids, but is not on `/v1/chat/completions` | the checkpoint's own directory if it has tokenizer files, else `BRAIN_QWEN3OMNIMOE_HF_DIR` |
| `BRAIN_MOCK` | deterministic, weight-free mock model (for exercising the serving stack without real weights) | any non-empty value |

## Serving & admission

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_MODELS_DIR` | model directory scanned at startup (also `--models-dir`) | `$XDG_DATA_HOME/brain/models` |
| `BRAIN_AUTO_FETCH` | `0`/`false`/`off` disables transparent auto-fetch of missing model files | enabled |
| `BRAIN_CONF` | stdio-loop YOLO confidence threshold (also `--conf`) | 0.25 |
| `BRAIN_ADMIT_DEADLINE_MS` | how long a request waits for a free lane before it's shed with 429 | 10000 |
| `BRAIN_COLD_BUILD_ADMIT_DEADLINE_MS` | longer deadline applied instead, but only while the request's own model is still cold-building (its first-ever activation) | 180000 |
| `BRAIN_SCHED_MAX_BATCH` | maximum jobs the scheduler groups into one batch | 8 |
| `BRAIN_SCHED_AGE_WEIGHT` | scheduler priority weight per millisecond a job has waited | 1.0 |
| `BRAIN_SCHED_BATCH_WEIGHT` | scheduler priority weight favoring larger batches | 200.0 |
| `BRAIN_SCHED_MAX_WAIT_MS` | a group whose oldest job has waited longer than this is force-picked regardless of batch size | 2000 |
| `BRAIN_CONF` | see above | 0.25 |

See [`docs/using/serving.md`](serving.md) for what admission/backpressure means in practice.

## Precision & tuning

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_NO_AUTOTUNE` | `1` skips runtime kernel autotuning and uses the static best-guess policy | autotune on |
| `BRAIN_PIPELINE_CACHE_DIR` | directory for the GPU pipeline/shader cache | backend default |
| `BRAIN_QWEN_CTX` | Qwen built context length | 24576 |
| `BRAIN_QWEN_MAX_BATCH` | Qwen serving batch slots | 16 |
| `BRAIN_QWEN_KV_INT8` | int8 KV cache (`0` opts out) | on |
| `BRAIN_QWEN_KV_CALIB` | per-head KV clip ranges from `brain qwen calib` | unset |
| `BRAIN_QWEN35MOE_CTX` | Qwen3.5 MoE built context length | model default |
| `BRAIN_QWEN35MOE_MAX_BATCH` | Qwen3.5 MoE serving batch slots | model default |
| `BRAIN_QWEN35_CTX` | Qwen3.8-27B dense built context length | 4096 |
| `BRAIN_QWEN35_MAX_BATCH` | Qwen3.8-27B dense serving batch slots | 4 |
| `BRAIN_LFM2_BATCH` | LFM batched-forward slots per instance | 2 |
| `BRAIN_FLUX2_MAX_BATCH` | FLUX.2 concurrent same-size batch cap | 4 |
| `BRAIN_FLUX2_TE_DEVICE` | FLUX.2 text-encoder placement (`gpu<i>[:i8]` for a truncated int8 shard on that card) | co-located with the DiT |
| `BRAIN_FLUX2_NO_STREAM` | `1` forces FLUX.2 to build its DiT from a whole fp32 tensor map instead of requantising a Q8_0 GGUF one tensor at a time. Not a correctness switch - both routes produce the same weights - but it is how the two are A/B'd, and a valve if a checkpoint ever trips the streamed path | off (stream when the tier and file allow it) |
| `BRAIN_FLUX2_TE_NO_STREAM` | `1` forces the FLUX.2 text encoder to be imported as one whole fp32 map instead of streamed per tensor from a mapping. Same character as `BRAIN_FLUX2_NO_STREAM`: identical weights either way, kept as an A/B instrument and a fallback | off (streamed) |
| `BRAIN_FLUX2_ALLOW_NC` | `1` opts in to the FLUX Non-Commercial-licensed 9B variants | required, not set |
| `BRAIN_FLUX1_I8_KEEP_F32` / `BRAIN_FLUX2_I8_KEEP_F32` | keeps a specific sub-layer at fp32 under INT8 inference, trading a little memory for accuracy | off |
| `BRAIN_LTXV_TEXT_CACHE` | reuse of a previously encoded LTX-2.5 text context for the same prompt and text encoder (`0` opts out) | on |
| `BRAIN_LTXV_TEXT_CACHE_MAX_BYTES` | disk budget for that cache, in bytes; after each write the least-recently-used entries are deleted until the directory fits (a non-positive or unparseable value is ignored with a warning, and the single most recently used entry is never evicted). One entry is tens of MB at the real checkpoint's dimensions, and a distinct prompt is a distinct entry | 2 GiB (`2147483648`) |
| `BRAIN_LTXV_TEXT_PRECISION` | `fp32` or `int8` for the LTX-2.5 text encoder's projections; a device with no packed-int8 path falls back to `fp32` regardless | `int8` for a quantized encoder, else `fp32` |
| `BRAIN_LTXV_AV_FP32` | `1` denoises an LTX-2.5 `--audio` run on the eager host-fp32 audio-visual reference arm instead of the quantized, device-resident one: the same model at higher precision, needing the whole checkpoint expanded to fp32 in host RAM and several times the wall clock. It is the A/B arm the quantized path is measured and gated against, not a tuning knob; the run's own banner names whichever arm is active | off (quantized) |
| `BRAIN_YOLOV8_BATCH` | YOLO true-batch forward width | 1 |
| `BRAIN_SAM2_VARIANT` | SAM 2.1 variant (`tiny`/`large`) | `tiny` |
| `BRAIN_S3DIT_RETAIN_INT8_CACHE` | `1` retains the int8 DiT cache across demote/promote (trades multi-GB host RAM for faster reactivation) | off |
| `BRAIN_S3DIT_ENCODER_GPU` | which GPU card hosts the Z-Image text encoder | shares the DiT's card |
| `BRAIN_S3DIT_ENCODER_CUT` | fp32 2-card encoder split point (layer index) | ~2/3 of layers |
| `BRAIN_S3DIT_ENCODER_FP32SPLIT` | `1` enables the fp32 2-card encoder split | off |
| `BRAIN_S3DIT_ENCODER_RESIDENT` | `1` keeps the encoder resident even when it shares the DiT's card | off (build on demand) |
| `BRAIN_S3DIT_WINDOW_BLOCKS` | blocks kept resident in the fp32 single-GPU decode window | 2 |
| `BRAIN_S3DIT_FLASH` | `1`/`0` forces flash attention on/off for Z-Image | automatic |
| `BRAIN_WAN_ATTN` | `flash`/`chunked` forces the Wan DiT's self-attention implementation | `flash` where the device supports it |
| `BRAIN_WAN_T5_DEVICE` | where the umT5-XXL text encoder runs (`cpu`/`gpu`); it is 22.72 GB in fp32 and does not fit a 24 GB card | `cpu` |
| `BRAIN_WAN_DIT_DTYPE` | the Wan DiT's weight dtype; `brain wan t2v --dit-dtype` wins over it | checkpoint's own dtype |
| `BRAIN_FLUX1_VAE_DEVICE` | where the FLUX.1 VAE decode runs (`cpu`/`gpu<i>`) | `cpu` |
| `BRAIN_QWEN3TTS_LANG` / `BRAIN_QWEN3TTS_REF` / `BRAIN_QWEN3TTS_REF_TEXT` | TTS language / reference voice `.wav` / its transcript | `english` / none / none |
| `BRAIN_MIMI_WEIGHTS` | TTS codec weights override | derived from `BRAIN_QWEN3TTS_WEIGHTS` |
| `BRAIN_QWEN3TTS_TALKER` | TTS talker placement (`cpu`, `npu`/`npu-fp32`, or an NPU int4 KV mode) | model-size default |
| `BRAIN_QWEN3TTS_MTP` | TTS next-token-prediction placement (`cpu`, `npu`, `fused`) | model-size default |
| `BRAIN_QWEN3TTS_CODEC` | TTS codec placement (`windowed`, `cpu-stream`, `npu-stream`) | default engine path |
| `BRAIN_QWEN3TTS_STREAM_CHUNK` | frames per chunk in `cpu-stream` codec mode | 16 |
| `BRAIN_QWEN3TTS_STREAM_WIN` | frames kept resident in the streaming decode window (rounds up to a multiple of the chunk size) | 32 |
| `BRAIN_QWEN3TTS_SPEAKER` | overrides the speaker-encoder weights used for voice-clone evaluation | derived from `BRAIN_QWEN3TTS_WEIGHTS` |
| `BRAIN_QWEN3TTS_NPU_DEVICE` | OpenVINO device for the TTS NPU talker | auto |
| `BRAIN_QWEN3TTS_RES` | resources base for `brain qwen3tts serve`'s default engine paths | unset (flags supply paths) |
| `BRAIN_QWEN3ASR_WINDOW` / `BRAIN_QWEN3ASR_MAXNEW` | Qwen3-ASR window (s) / max tokens | 30 / 200 |
| `BRAIN_FORECAST_HORIZON` / `BRAIN_FORECAST_SAMPLES` | forecast horizon / sample count | 64 / 1 |
| `BRAIN_MOCK_DELAY_MS` | mock model artificial latency | 0 |
| `BRAIN_OV_CACHE` | OpenVINO compiled-graph cache dir (ASR/NPU) | `$TMPDIR/brain_ov_cache` |
| `BRAIN_GPT2` | stdio-loop GPT checkpoint (also `--gpt`) | fake echo model |

Deeper per-model knobs not listed here are documented on that model's own
page under `docs/models/`.

## Paths

| Variable | Meaning | Default |
| --- | --- | --- |
| `BRAIN_MODELS_DIR` | model directory scanned at startup | `$XDG_DATA_HOME/brain/models` |
| `BRAIN_PIPELINE_CACHE_DIR` | GPU pipeline/shader cache directory | backend default |
| `BRAIN_OV_CACHE` | OpenVINO compiled-graph cache directory | `$TMPDIR/brain_ov_cache` |
| `BRAIN_QWEN3TTS_RES` | resources base for `brain qwen3tts serve`'s default paths | unset |
