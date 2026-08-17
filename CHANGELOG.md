# Changelog

All notable changes to brain (https://github.com/swedishembedded/brain) are
documented here. Generated with git-cliff from conventional-commit history;
see CONTRIBUTING or AGENTS.md for the commit-message convention.
## [1.0.0] - 2026-08-17

brain is a small, dependency-light framework for training and evaluating
neural networks from scratch on the GPU: a from-scratch **Rust + raw-WGSL**
engine (forward + backprop + AdamW, hand-written as compute kernels) gradient
checked against a PyTorch reference, running on essentially any GPU -
including in the browser via WebGPU. The engine is architecture-agnostic: 320+
WGSL kernels are reusable building blocks, not a fixed model, and every
supported architecture is composed from them and gradient-check gated.

This is the first stable release. What 1.0.0 ships:

- **Engine core** - fp32-only, core-compute-only WGSL kernels; CPU (Cranelift
  JIT), wgpu, and Vulkan cooperative-matrix backends validated against one
  another; an in-repo finite-difference gradient checker gates every backward
  pass instead of relying on an external oracle.
- **Language / decoder LMs** - dense GPT2 and Qwen3 (GQA, QK-norm, RoPE,
  SwiGLU, LoRA, INT8, tensor/expert sharding, concurrent paged-KV serving);
  Qwen3.5-35B-A3B (Gated DeltaNet + GQA hybrid, 256-expert sparse MoE);
  GLM-5.2 (MLA + sigmoid MoE + DSA + MTP); the toy sparse-MoE Transformer with
  federated/sharded training; LFM2.5 bidirectional hybrid encoder;
  encoder-decoder Seq2seq; a bottleneck autoencoder.
- **Vision & 3D** - YOLOv8-style detector; ZipDepth monocular depth (realtime
  demo incl. Intel NPU); SCRFD + ArcFace face recognition; SAM 2.1 promptable
  segmentation; WorldMirror-2 multi-view 3D reconstruction; a from-scratch
  tiled 3D Gaussian Splatting rasterizer with forward and backward.
- **Image generation** - Z-Image (S3-DiT), FLUX.2 Klein, FLUX.1/Kontext, the
  T5-XXL text encoder, VQGAN/CodeFormer, Real-ESRGAN super-resolution.
- **Audio / speech** - Qwen3-TTS voice cloning; Nemotron and Qwen3 streaming
  ASR; Qwen3-Omni-30B (text/speech/image/video in, synthesized speech out,
  with a layer-sharded int8 dual-GPU serving path); FastVLM/Moondream 3 VLMs;
  DeepSeek-OCR document understanding.
- **Forecasting** - Chronos-2, Kronos, and FinCast time-series models behind
  one model-agnostic seam, with a rolling-origin backtester.
- **World models** - DIAMOND (Atari-100k EDM diffusion, playable) and
  GenieRedux-G.
- **Serving & runtime** - concurrent paged-KV LLM serving (continuous
  batching, chunked prefill, int8 KV as the default), a capability-manifest
  dispatch shared by the CLI/HTTP/D-Bus surfaces, GPU/RAM/disk model
  residency under a memory budget, a live-monitoring TUI (`braintop`), and a
  D-Bus control surface with a vetted system-bus policy.
- **Packaging** - a self-contained Debian package (`make deb`) and a
  reproducible `make release/{patch,minor,major}` version-bump flow.

See the sections below for the full commit history.

### Bug Fixes

- *(bench)* Make mod_add informational (grokking is seed-flaky); p=17 default ([d4673006](https://github.com/swedishembedded/brain/commit/d4673006106cf1e8c0386ac8386283de40032a8f))

- *(runtime)* Return to Idle after detection ([bdaea8d2](https://github.com/swedishembedded/brain/commit/bdaea8d296f8600dc60d7e63207102a92e32c36e))

- *(bench)* Real per-side device selection + honest labels ([53a90bce](https://github.com/swedishembedded/brain/commit/53a90bce3b8e67af060c591e488c67b842a50e6c))

- *(bench)* Honor shell BRAIN_DEVICE + surface engine adapter line ([5fc1d3a7](https://github.com/swedishembedded/brain/commit/5fc1d3a7f43c4e2e081ac590ea4267ccc372282b))

- *(yolo)* Honor --device so brain GPU mode actually uses wgpu ([8aff9e5c](https://github.com/swedishembedded/brain/commit/8aff9e5c8ffedc72b0dd2ffdd104f00ffe3389b8))

- *(flux2)* Img2img init image must not also be a reference token ([a78665ac](https://github.com/swedishembedded/brain/commit/a78665ac107bb16bb823a47cbe823f1ab53e116a))

- *(cli)* Delete dead all_providers, run_residency wrapper and render alias ([77722ceb](https://github.com/swedishembedded/brain/commit/77722cebe43810dd201710f9ba1961580b6cfe9d))

- *(flux2)* Drop the never-dispatched matmul_reg2 pipeline slot ([7ececce5](https://github.com/swedishembedded/brain/commit/7ececce557dbee409114c92f499b6c777cbea9e0))

- *(zimage)* Drop dead cfg accessor, unused import and stale pf binding ([954d914c](https://github.com/swedishembedded/brain/commit/954d914c65a77982a8a8461d984f1d3194461b06))

- *(lfm)* Drop needless mut in Lfm::new; document the unwired flash slot ([25c63acc](https://github.com/swedishembedded/brain/commit/25c63acc57a5f2ceb52ca884d20eca62ca6fa7d1))

- *(qwen)* Delete the Engine::w alias for ParamStore::w ([464697eb](https://github.com/swedishembedded/brain/commit/464697eb8a3f240a9dc683cb8c3521b7322dc9c2))

- *(qwenvl)* Make BLOCK_LEAVES pub ([64c6a48a](https://github.com/swedishembedded/brain/commit/64c6a48a18ca31f435cb8a0ff81302a27b8c666f))

- *(nemotron)* Scope ff_bwd_pipelines to the tests that use it ([0ed09ac6](https://github.com/swedishembedded/brain/commit/0ed09ac61954068d5c160c0244f93c23655d966d))

- *(tts)* Remove the orphaned #[inline] and stale matvec doc in gen_kv ([c236c045](https://github.com/swedishembedded/brain/commit/c236c0456f54959728f447fd00f68ad8c2b066ef))

- *(chronos2)* Drop the never-read HeadCache::normed field ([141296c4](https://github.com/swedishembedded/brain/commit/141296c4161c2ec168a70d9b22477d1dcbadc5f7))

- *(tests)* Clear the remaining unused-binding and import warnings ([88dc08b1](https://github.com/swedishembedded/brain/commit/88dc08b1a6c836f33cf279d01dcd8530296ecddc))

- *(backend-cpu)* Compute the exact extent for the cross-attention fast path ([1d2e1189](https://github.com/swedishembedded/brain/commit/1d2e1189f1d65d68e111880549f395b59399d3fc))

- *(backend-cpu)* A denied clippy lint here blocked the lint pass everywhere ([799dde1c](https://github.com/swedishembedded/brain/commit/799dde1c62145d2a1f733c241794908bcd1c46ab))

- *(apiserve)* Security-audit findings - constant-time key, error hygiene, body limit, key perms (P17) ([ea82ba1b](https://github.com/swedishembedded/brain/commit/ea82ba1b6c5cb7b1d175f483d20f2653f278e432))

- *(residency)* Place vram-cost models on CPU when no accelerator exists ([de7b0131](https://github.com/swedishembedded/brain/commit/de7b0131a131ebd6d29cfb23c3e21d6777d00707))

- Schedule the integrated GPU by default; actionable serve diagnostics ([8de85251](https://github.com/swedishembedded/brain/commit/8de85251b07f8a89aea3fbeb8809b891e5ec8469))

- Attach a ModelCard on HF import so auto-serve actually works ([a295ff38](https://github.com/swedishembedded/brain/commit/a295ff38a7edb2df7166a9476dee0be22911736b))

- Pad per-head strides in GEMM-attention packed buffers (real alignment bug) ([bda58efb](https://github.com/swedishembedded/brain/commit/bda58efb13205f72c2a045607eb3b1ca6fbed381))

- *(dbus,apiserve)* Brain serve --dbus now actually stops on Ctrl-C/SIGTERM ([65f2bc0b](https://github.com/swedishembedded/brain/commit/65f2bc0b7d4e23e063d0d5781b4e07310a05ae1a))

- *(brain-py)* Stop masking D-Bus server errors as JSON parse failures ([7dbab97c](https://github.com/swedishembedded/brain/commit/7dbab97c2ce25ad262cb12701d769a07c07d0d47))

- *(examples)* Repair every example against the P19 brain-py API ([72feb731](https://github.com/swedishembedded/brain/commit/72feb731221b784fcf5593597fb69cdbf020aa57))

- *(modelstore)* Follow real HF redirects correctly; rename examples to fully-qualified auto-fetchable names ([dd02c63e](https://github.com/swedishembedded/brain/commit/dd02c63e08c30d092d92cec975e21e1b260953e7))

- *(flux2)* Host/device parity test used dims the device path cannot bind ([c1969b35](https://github.com/swedishembedded/brain/commit/c1969b3565831937178cbb84053cbf10a8f5016b))

- *(clippy)* Unblock the workspace gate, which aborted before it linted anything ([510c817f](https://github.com/swedishembedded/brain/commit/510c817fa493c354d2afec6b8bb1df641c6f61b5))

- *(depth)* The QARep gradcheck step sat above the FD truncation knee ([2a71c098](https://github.com/swedishembedded/brain/commit/2a71c098280259a1fa9eb51cdf89378d78909461))

- *(depth)* Gate the fused-eval cross-implementation checks on a scale-relative metric ([fe0d1db6](https://github.com/swedishembedded/brain/commit/fe0d1db6f6feb3b4b7fe84509305abafa66be9b0))

- *(cli)* A bool capability param takes a value, not SetTrue ([bc4d80a0](https://github.com/swedishembedded/brain/commit/bc4d80a01be9b2bab9cd731d47c4deb35b4328a8))

- *(depth)* Calibrate INT8 on the preprocessing inference actually uses ([0b9bf123](https://github.com/swedishembedded/brain/commit/0b9bf12352240b12935cbdae0245c8189eda907d))

- *(vae)* SDXL's post_quant_conv was silently skipped - decode was uncorrelated ([a38dd31c](https://github.com/swedishembedded/brain/commit/a38dd31c997c98a4b3297766849bc7eba927cdc3))

- *(wm-core)* Spell out the NaN rejection clippy would have removed ([aadcdfb6](https://github.com/swedishembedded/brain/commit/aadcdfb641c6f1c58d766af8ab830e2d98fa1d34))

- *(qwen)* A LoRA fine-tune had no effect on inference -- the config round-trip dropped it ([114ea106](https://github.com/swedishembedded/brain/commit/114ea1061dc7771086fbad1fb7302dd8ab6e5cfc))

- *(data)* A mistyped/missing field in the bench wire contract was a silent default, not an error ([789d9c2f](https://github.com/swedishembedded/brain/commit/789d9c2fcda8a2cfdb857439aa8683f5bbff53d2))

- *(scripts)* Build-deb.sh ROOT resolved to scripts/, not the repo root ([9084539b](https://github.com/swedishembedded/brain/commit/9084539ba99a605991e118c56d9c5a6942e536e9))

- *(checkpoint)* Rewrap open_hf_dir's doc comment to drop a phantom markdown list ([8ec7672b](https://github.com/swedishembedded/brain/commit/8ec7672b2b9693194b7eae64be0538fe347a885f))

- *(model)* Reject a token split shorter than block_size at load, not mid-batch ([d82f39cc](https://github.com/swedishembedded/brain/commit/d82f39ccb82112fa38e89bb76247a88c4d0e5419))

- *(wm-diamond)* Drop two kernels the model registered and never dispatched ([da028659](https://github.com/swedishembedded/brain/commit/da0286590c8678fad13696d25b40f3c77c273cf2))

- *(backend-wgpu)* 0 enumerated adapters is not "wrong card" - fall back instead of panicking ([5ef59c0a](https://github.com/swedishembedded/brain/commit/5ef59c0ab7941b748caf11f3b7ad0f0a234a5390))

- *(vae/blocks)* The attention tape dropped the head split ([e10959f0](https://github.com/swedishembedded/brain/commit/e10959f009c6abe77df8fa26b9588b5fad02347b))

- *(apiserve)* Name mid-stream errors, distinguish capacity rejections, advertise real context length ([3f4ca85c](https://github.com/swedishembedded/brain/commit/3f4ca85c241e321bd1fedea1ebf4dc85d0928781))

- *(cli)* Stop reserving worst-case KV capacity for every concurrent batch slot ([0b358976](https://github.com/swedishembedded/brain/commit/0b3589769c932a78898da7978f5884a8cf4151ae))

- *(qwen)* Decode/batched-forward reject an out-of-vocab token id instead of reading past the embedding table ([7afa6aed](https://github.com/swedishembedded/brain/commit/7afa6aedcab50b321234e2b9c13f9a2d7db1f7aa))

- *(backend-cpu,backend-vulkan)* Implement Backend::stats - two backends never counted ([26b074e1](https://github.com/swedishembedded/brain/commit/26b074e1a73e73655866ba39ad47a8b62df50850))

- *(kernels/dw_splitk_reduce)* 2D-grid safe index - it dropped outputs past 4.19M ([3ad494a1](https://github.com/swedishembedded/brain/commit/3ad494a1262202d1f493803b1ff733edf34c3a11))

- *(gpu-core)* Measure the roofline hierarchically, and flag the profiler's time source ([e13693ba](https://github.com/swedishembedded/brain/commit/e13693ba7197e5dcf6c844cdbcf490789ca33f44))

- *(scripts)* Count barriers in code, not in comments - 4 catalogue rows were wrong ([83548b46](https://github.com/swedishembedded/brain/commit/83548b466399bb83ce60b3439ea3551ec2cfa11d))

- *(backend-api,scripts)* Parse the code, not the comments - and stop duplicating the derivation ([c7f90220](https://github.com/swedishembedded/brain/commit/c7f902208b8110284ed13abb39dbdaeb57e9a19b))

- *(cli)* Repair clippy warnings that drifted in since the last baseline ([2fdfb4de](https://github.com/swedishembedded/brain/commit/2fdfb4dec6805310e5dacffca09802483da721f9))

- *(qwen)* Kv_calibrated() no longer lies on an fp32 engine; add the int8-KV boundary policy ([87091282](https://github.com/swedishembedded/brain/commit/87091282064363864a90f3f38c44cc8fb54fcd76))

- *(cli)* Repair clippy warnings surfaced by the origin/main rebase ([768dac84](https://github.com/swedishembedded/brain/commit/768dac84e6b31f8e221940ff83f30418d76e5092))

- *(model)* EOS token must not leak into served output text ([23394663](https://github.com/swedishembedded/brain/commit/2339466322992dfab7f1dd74eee66f68bfa2d336))

- *(modelstore,cli)* Stop family_of_architecture routing Omni to the qwen importer ([d744c4da](https://github.com/swedishembedded/brain/commit/d744c4dab83fb51283b9c0ff4569a679449687a7))

- *(omni)* Run the goldens dumper end to end against the real checkpoint ([6bb884fb](https://github.com/swedishembedded/brain/commit/6bb884fbfe8cba50248d8e981477de100730dcea))

- *(kernels)* Router_gate.wgsl/router_gate_train.wgsl OOB write above 64 experts ([e3019d19](https://github.com/swedishembedded/brain/commit/e3019d191d6430d6e603e4e8fd5f8a6f00fcd8f0))

- *(omni)* Talker's use_qk_norm default was false; reference always applies it ([5121cfbc](https://github.com/swedishembedded/brain/commit/5121cfbcbe05e0887cf62dfb5df16b076cbcd64d))

- *(omni)* Generate_spec's param shape blocked OpenAI/Anthropic exposure (M11/M12) ([c7ccaa67](https://github.com/swedishembedded/brain/commit/c7ccaa6752283721a85aa100e6356d75ad6924d1))

- *(omni)* Code-predictor and Code2Wav loader-naming gaps ([16ac315f](https://github.com/swedishembedded/brain/commit/16ac315f5d654779a8d48156b68579bfd3ad4384))

- *(apiserve)* Image_url/input_audio/image content parts were silently dropped ([c0e56d37](https://github.com/swedishembedded/brain/commit/c0e56d374369987cf9727498c85c33f88479a2b4))

- *(gpu-core)* Roof::ensure self-deadlocks on a cold cache ([30555255](https://github.com/swedishembedded/brain/commit/30555255879e25d551e6b61f29671ea31e085559))

- *(wgsl-cpu)* Cranelift-JIT dropped a local-variable assignment in work-group kernels ([9b9c9c00](https://github.com/swedishembedded/brain/commit/9b9c9c00325c5a25fcfa956ff3193117d8efc77b))

- *(omni)* Add the brain-residency dependency the int8 dual-GPU Thinker needs ([5a16ca61](https://github.com/swedishembedded/brain/commit/5a16ca6178f34cfa33e08f09eb0c896be2fe0e73))

- *(perf)* --smoke must cap prompt length too, not just output length ([da4e1eed](https://github.com/swedishembedded/brain/commit/da4e1eed540235d566343fd5597d64f4d22854e7))

- *(perf)* Pool_for must fit the fidelity probe; name a zero-comparison failure honestly ([6c3222ac](https://github.com/swedishembedded/brain/commit/6c3222ac4442f445001a4439df6640d0393322b0))

- *(perf)* Remove the 'fake' perf target -- no synthetic-harness stand-in ([607dce9c](https://github.com/swedishembedded/brain/commit/607dce9c67af751712a100ec7c315ef3b0be5eb6))

- *(residency)* Fix scheduler collapsing to FIFO-of-one on slow hardware ([def4bc65](https://github.com/swedishembedded/brain/commit/def4bc659884a3b0cf9f927596c5b5a7acf9823f))

- *(cli)* Prefer MemAvailable over MemTotal when budgeting scheduler RAM ([eaded512](https://github.com/swedishembedded/brain/commit/eaded51217abf09e844f556ac97da3b3698832bf))

- *(cli)* Resident depth must build its engine on the assigned device ([d053cdbc](https://github.com/swedishembedded/brain/commit/d053cdbcdee9e575f8b2eb920a33ded1a83ff993))

- *(vulkan)* Stop staging on unified memory ([f6d17c2a](https://github.com/swedishembedded/brain/commit/f6d17c2a928683903331c175df783b3395f79a91))

- *(vulkan)* Stage every readback, even on host-visible buffers ([762b64d7](https://github.com/swedishembedded/brain/commit/762b64d7c7441c9262b79fbf96cf04964dca2dd4))

- *(apiserve)* Stop advertising HTTP capabilities the dispatcher can't reach ([9c4129f3](https://github.com/swedishembedded/brain/commit/9c4129f3f8d1962fdaa787cca693dc11183ec5de))

- *(e2e)* Exclude the forward-only qwenvl VLM from the model-table drift guard ([df54aa2b](https://github.com/swedishembedded/brain/commit/df54aa2bc54ce07a02b9726378a478a5dcb775f9))

- *(docs)* Correct the YOLO auto-fetch quick-start, verified live end-to-end ([7ef6f564](https://github.com/swedishembedded/brain/commit/7ef6f5641fb3ccb4e7fe215f5a67b2d405c97051))

- *(wgsl-cpu)* Enforce the barrier-crossing-local invariant compile_one_wg claimed but never checked ([77f96d77](https://github.com/swedishembedded/brain/commit/77f96d77315c9299c403ce12c292096da6cc6ba3))

- *(kernels)* Port matmul_reg3's bank-conflict and register-tiling fixes into the int8 GEMMs ([027e446d](https://github.com/swedishembedded/brain/commit/027e446d61d502832138995893e570bf108dd317))

- *(kernels)* Make the softmax top-k MoE router array-free past 128 experts ([595d1ad7](https://github.com/swedishembedded/brain/commit/595d1ad7296f4cec5fc3c7c0252754015fe643c8))

- *(qwen3)* DeepStack decode splice violated wgpu buffer-offset alignment ([8d69f8b1](https://github.com/swedishembedded/brain/commit/8d69f8b1108457adccbd2b7a8199d9df1497d6ce))

- *(zimage)* Stop retaining the whole checkpoint host-side after build ([318798c6](https://github.com/swedishembedded/brain/commit/318798c662cdd2fe33d63e85a49a49a51e6bb255))

- *(zimage)* Stream the DiT and Qwen-4B encoder loads, killing the acute host OOM ([3cd70c27](https://github.com/swedishembedded/brain/commit/3cd70c27b30280e8b6d539b67d0eb25d4fff7241))

- *(zimage)* Never hold the int8 encoder and DiT resident on the same card ([9b24ab18](https://github.com/swedishembedded/brain/commit/9b24ab18ea314c070cbd27a2801940217457c984))

- *(zimage)* Real end-to-end int8 256x256 generation without OOM ([badc1241](https://github.com/swedishembedded/brain/commit/badc1241902e9fe482d982647eeb1bb61f8f7ad7))

- *(zimage,cli)* ZImageResident::estimate reflects the real windowed-fp32 cost ([14749d6e](https://github.com/swedishembedded/brain/commit/14749d6ebb77c5d945af8328fcaa4b343f657eed))

- *(checkpoint)* Validate mmap header offsets; refuse OOB RemapSource slices on every path ([53a6a5f5](https://github.com/swedishembedded/brain/commit/53a6a5f543a05f0db6d24df3c70d55f5eb082489))

- *(residency)* Panic-isolate executor lanes; Warm-demote fits check; cross-registry claim guard ([d01b8fca](https://github.com/swedishembedded/brain/commit/d01b8fcaee972dd852bbacc44d63718c7b7fc275))

- *(kernels)* Array-free router_gate fwd - no expert cap, silent-OOB literal gone ([b9e42a66](https://github.com/swedishembedded/brain/commit/b9e42a66f8cfcc7879246ddc63b648ae9af03de6))

- *(perf)* Gate refuses floor 0; register weights scenario; carry failure reasons into artifacts ([c77ce9e5](https://github.com/swedishembedded/brain/commit/c77ce9e589ae4b340643ed5b0f0ac55028e9181f))

- *(data)* HF-faithful chat templating; tokenizer/binio robustness on untrusted files ([c8353e1c](https://github.com/swedishembedded/brain/commit/c8353e1cf26e5123d78eba7511172189cc248eef))

- *(model)* Wasm-gate the parallelism re-exports; harden netcollective; stop masking pipeline panics ([94725ebe](https://github.com/swedishembedded/brain/commit/94725ebee477d46ebe522978a30e13c2531f2c69))

- *(dbus)* Never drop terminal done/error/blob stream frames under backpressure ([97ba39ea](https://github.com/swedishembedded/brain/commit/97ba39ea34cb75330671df3333409ddc0d70a7f5))

- *(dbus)* Put StreamTranscribe behind the edge-concurrency ceiling; bound fd blobs and window_ms ([9510752a](https://github.com/swedishembedded/brain/commit/9510752abb241324edafa2b9003b6b8d014f116c))

- *(apiserve)* Batch /v1/embeddings submissions; one catalog snapshot per request ([a172614f](https://github.com/swedishembedded/brain/commit/a172614f728a1794a4de8c33f629fca5aacaaace))

- *(cli)* Kill the perf-target env round-trips; harden the perf gate CLI ([434c3afa](https://github.com/swedishembedded/brain/commit/434c3afa22e7778b329331cbca2d57ae7892f14a))

- *(cli)* Fold ASR, forecasting and TTS into the served-model catalog ([3587ede3](https://github.com/swedishembedded/brain/commit/3587ede3814f37e489779b8b11d891f6f3b4dadc))

- *(cli)* Honest z-image Hot budget with a retained int8 cache; surface calib encode errors ([206e7167](https://github.com/swedishembedded/brain/commit/206e7167d2e80fa31f4174125a3064a2cbb760f3))

- *(cli)* Ship a vetted D-Bus system policy instead of advertising --dbus-system with none ([42884db5](https://github.com/swedishembedded/brain/commit/42884db5a011800c3a06a8a327901241eaecfc87))

- *(omni)* Served speak silently ran generate - dispatch by action name, error on unknown ([3ed7717b](https://github.com/swedishembedded/brain/commit/3ed7717b174be1a4c8301e6d3a7fd4c1db42ad10))

- *(omni)* Talker per-expert submit storm; hoist the GQA attention sublayer into model::block ([66e449f7](https://github.com/swedishembedded/brain/commit/66e449f7b81431e1ac3b55bcf5f2664db611bfc2))

- *(omni)* Request-time tensor-load panics on the served path become errors ([6dc76d69](https://github.com/swedishembedded/brain/commit/6dc76d69c71b8c5404829a9e9704d7b3eea27b48))

- *(qwen)* Eval no longer silently swallows chat-template encode errors ([fbb2ac10](https://github.com/swedishembedded/brain/commit/fbb2ac103119570f69bc6b2cf06cfff429a8c261))

- *(tts)* Resident manifest lives in tts::caps - no more parallel spec in crates/cli ([5bed51ca](https://github.com/swedishembedded/brain/commit/5bed51caa61dba8558a3c6039d167106967444c2))

- *(vqgan)* Bench oracle asserts; card/ledger stop claiming backward is deferred ([c3d45390](https://github.com/swedishembedded/brain/commit/c3d453908540ca5e5cbd7173490a536e67bf6037))

- *(models)* Silent input truncation becomes loud - flux2/zimage prompts, qwen-asr audio ([bd5765e2](https://github.com/swedishembedded/brain/commit/bd5765e20b77a23a25ecf6b30477f859338ec67a))

- *(zimage)* Masked prompt padding in the resident pipeline - no more unmasked repeat-last-token pad ([1159a869](https://github.com/swedishembedded/brain/commit/1159a8692e6d64076680f097c9c80b6657fc4e23))

- *(qwenvl)* The declared .streaming() emits real per-token deltas ([84ae7da2](https://github.com/swedishembedded/brain/commit/84ae7da2e53474f3b6361183294f8a3b465ac480))

- *(omni)* Estimate_multi fails the claim instead of panicking; shard count has one source; drop dead image_row0 fields ([77eb7f2d](https://github.com/swedishembedded/brain/commit/77eb7f2d7eb96e97fa7b95ee5ba2e2967193a0e0))

- *(modelstore)* Delete upstream weights after a successful convert ([6ef19ab5](https://github.com/swedishembedded/brain/commit/6ef19ab557decaf7328c9f6b85f89130f1068215))

- *(apiserve,dbus,residency)* Give an in-flight cold build its own admission deadline ([09ff3b47](https://github.com/swedishembedded/brain/commit/09ff3b47c066c998ffe2d33b128344a34bd08e5f))

- *(residency)* Fix building-flag scoping broken by the rebase onto multi-device dispatch ([478b2a00](https://github.com/swedishembedded/brain/commit/478b2a00c28a3066505a4a4e9396cab6b2236abb))

- *(residency,cli)* Make StoreSupplier::ensure genuinely idempotent ([69cf57eb](https://github.com/swedishembedded/brain/commit/69cf57eb416dcdfa93db16592695ce93e50cbb5c))

- *(cli,residency)* --device gpu no longer starves the shared unified-RAM pool ([6d289c1c](https://github.com/swedishembedded/brain/commit/6d289c1c0d7f923abfbcc02b49788c551c1a305f))

- *(backend-wgpu)* A lost-device readback callback must never panic ([5a5adff2](https://github.com/swedishembedded/brain/commit/5a5adff25f65185e69ff94bb52e57914fffa41ac))

- *(residency)* The dispatcher thread must never panic either, not just lanes ([6575c341](https://github.com/swedishembedded/brain/commit/6575c3410b76ee9c6ae109cafcc864cde2e385d3))

- *(examples)* Show reasoning_content, don't abort the whole demo on a per-route 404 ([d680ecf7](https://github.com/swedishembedded/brain/commit/d680ecf792fec8d2a6746500e9579646d9155c0f))

- Recover from poisoned mutexes after a device-lost panic ([982025f8](https://github.com/swedishembedded/brain/commit/982025f8d8fc35f1347663a3f0244c1bb74f713a))

- *(codec)* Apply the sliding-window mask sliding_window was parsed into but never dispatched ([0893002c](https://github.com/swedishembedded/brain/commit/0893002c308690456fa72786648ce8c1e619d62c))

- *(residency,cli)* Stop permanently-unplaceable jobs hanging silently; fix NPU diagnostics ([e045c6c3](https://github.com/swedishembedded/brain/commit/e045c6c32f68d629b62829964c6bc73909302296))

- *(npu)* Whole-channel INT8 quantization corrupts outlier-heavy channels ([1f4c3b6e](https://github.com/swedishembedded/brain/commit/1f4c3b6e72ee7804862b4be4bc5e2bb06b3f2c59))

- Fix(hooks): scope diff-scoped checks to pre-commit, give pre-push a real
full-tree doc-citation backstop

spdx-license-headers/no-doc-citations-in-code/no-em-dash had no explicit
stages:, so default_install_hook_types made them ALSO re-run at push time --
diff-scoped against whatever pre-commit's own batching decided was
"changed", producing several confusing repeated clean/fail blocks per push
and, worse, still missing things: a push just failed with
crates/imaging/src/video.rs (among others) citing docs/models/omni/status.md,
a citation that predates today's docs-tree reorg and should have been caught
when this morning's rebase replayed that commit, but pre-commit hooks are
not guaranteed to fire on every commit `git rebase --continue` creates.

Now: those three hooks are stages: [pre-commit] only (fast, diff-scoped, as
intended). scripts/hooks/pre-push (already the trailer-check backstop for
exactly this class of gap) gained a second check: a full working-tree run of
check-no-doc-citations.sh with no file args, once, regardless of how the
tree got that way. That is the actual thing about to be shared with the
remote, so it is the one scan nothing can slip past. ([d27d2635](https://github.com/swedishembedded/brain/commit/d27d2635470f2c5dc741fdc5cdffa818ef2ee91b))

- Remove dangling docs/ citations left over from the doc-tree reorg ([2f9a4f44](https://github.com/swedishembedded/brain/commit/2f9a4f44e39af79ad640be59a9a477c02eb4a39f))

- *(data)* QwenBpe digit-run pre-tokenizer width was hardcoded to 1 ([a3ec8288](https://github.com/swedishembedded/brain/commit/a3ec82882db95e7521e9211eb80eaf0ff4f3480c))

- *(moondream)* Pass explicit norm=1,scale=1.0 to the generalized router_gate ([1570b259](https://github.com/swedishembedded/brain/commit/1570b259bd88e15d9fe2fb6414dd39b234f3bc26))

- *(sam1,model)* Mitigate (not fully fix) the wgpu attn_relpos_add over-dispatch ([c3c11f6e](https://github.com/swedishembedded/brain/commit/c3c11f6e413b58559f21e854ccd9643f57afd50c))

- *(backend-wgpu)* Serialize Intel ANV sliced-binding batches, mirroring backend-vulkan ([293b8a58](https://github.com/swedishembedded/brain/commit/293b8a58014a8a3de9ee0c36258f642221d2927d))

- *(apiserve)* Emit full text on chat streams that never send a delta ([22078d3b](https://github.com/swedishembedded/brain/commit/22078d3b8959a62e3b4cc5d6d9e2ccea083b3c30))

- *(omni)* Capacity-aware int8 GPU sharding, bounded disk->VRAM streaming ([b64dac1a](https://github.com/swedishembedded/brain/commit/b64dac1a19ed079c7210ff0cbcbea65127bc3fee))

- *(omni)* Dtype-agnostic sharded placement, correct VRAM accounting ([8f95b2f2](https://github.com/swedishembedded/brain/commit/8f95b2f237a37b4e9e57db8080bb46a6a32d068b))

- *(omni)* Close rename gaps missed by the prior reconstruction commit ([c2208ad8](https://github.com/swedishembedded/brain/commit/c2208ad89346570d20d4210580f0b080c5e01f7f))

- *(gpu-core)* Unify --device/BRAIN_DEVICE grammar, one source of truth ([b3af2a34](https://github.com/swedishembedded/brain/commit/b3af2a34cea168376be8c1175ad09dacbea46d59))

- *(backend-api)* Strip em dashes B1 added to doc comments and ledger ([07abfa88](https://github.com/swedishembedded/brain/commit/07abfa88a53a263ef4416bc50cede9c4d2132044))

- *(qwen3)* Register B8/B9/B10's new kernel names in pipelines() ([ff78ab16](https://github.com/swedishembedded/brain/commit/ff78ab169252f9e9ac4fa362bce2d29678515a48))

- *(omni)* Correct M-RoPE audio position ids (real HF diagonal, not a pinned-H/W meshgrid) ([e39f5fb0](https://github.com/swedishembedded/brain/commit/e39f5fb08d96af07f78e26fe98c1bda656a92701))

- *(omni)* Reclaim tower VRAM between multimodal encode calls ([e2f502cc](https://github.com/swedishembedded/brain/commit/e2f502cc2a303dc00ec16c638395c770b5d360d1))

- *(omni)* Reclaim VRAM periodically between layers during a forward pass ([77e3b242](https://github.com/swedishembedded/brain/commit/77e3b2428b2401973f9cf87901894e33be115cfa))

- *(residency)* Give Executor a real joinable shutdown, fix an exit-time SIGSEGV ([cdc629f4](https://github.com/swedishembedded/brain/commit/cdc629f49b33c550fa7fe2707ce58ed71a65300d))

- *(omni)* Splice multimodal media after the system turn, not before it ([36aa4c69](https://github.com/swedishembedded/brain/commit/36aa4c69788baf72bf51d833bfb3a757f0a7736b))

- *(cli)* Accept infer as the canonical verb across every dedicated handler ([9dc335ef](https://github.com/swedishembedded/brain/commit/9dc335efeeafc09e18002458bd6ebf05fd5eb9a7))

- *(brain-py)* Brain run subprocess spawn is now brain serve --stdio ([9e6507ac](https://github.com/swedishembedded/brain/commit/9e6507acd4ffeec2991361347b39e2b1942452db))

- *(tests)* Document DeepSeek-OCR's deliberate reserved-vendor exception ([3ab19f8a](https://github.com/swedishembedded/brain/commit/3ab19f8aa40c96bac588e27aca09a7cd8097a83a))

- *(s3dit)* Inpaint masked region no longer leaks original-content signal ([7c5c3b89](https://github.com/swedishembedded/brain/commit/7c5c3b89638e7cbb3972521892dcbb11cfadfc11))

- *(s3dit)* Timestep embedding used sin-first ordering, should be cos-first ([ca12c1f9](https://github.com/swedishembedded/brain/commit/ca12c1f910ca969cd14e603fb463bccfe9b443d8))

- *(tests)* Serialize GPU device construction across 11 files that raced under cargo test's default thread pool ([8fdeb0d2](https://github.com/swedishembedded/brain/commit/8fdeb0d2aba949a4b0fc5bc9129213459cbd3b12))

- *(imaging)* Codec test cleanup deleted the shared parent temp dir, not its own leaf ([b0465a31](https://github.com/swedishembedded/brain/commit/b0465a31b97e019b6e0e61f94cff5fed9066df06))

- *(qwen3vl)* Qwen3Vl::new's decode-only switch broke the batched forward()/backward() path ([bfd56770](https://github.com/swedishembedded/brain/commit/bfd567707153b1690ac01eb03447cfa3ae85d351))

- *(yolov8)* Resolve two real clippy warnings in the P5 overfit gate test ([76197647](https://github.com/swedishembedded/brain/commit/761976477602288dedfb166a9c27f07b5143d243))

- *(clippy)* Resolve 6 doc-comment/loop-index warnings, lower the gate baseline 294 -> 292 ([8512c18d](https://github.com/swedishembedded/brain/commit/8512c18df845932b0d2416016638654abfb7037d))

- *(lfm2, qwen3asr)* Correct hardcoded tensor-name prefixes against real checkpoints ([56506d5e](https://github.com/swedishembedded/brain/commit/56506d5e103d847b61ea9a44603484c5d062c7e1))

- *(cli)* Widen auto-fetch gating so lfm2's default_ref is reachable ([c60a65dd](https://github.com/swedishembedded/brain/commit/c60a65dddc1c28fe26e6bcc77abf26927a45c24e))

- *(docs)* Drive inpainting from sam2's real mask instead of a guessed rectangle ([49d43a20](https://github.com/swedishembedded/brain/commit/49d43a20de983899f1643ba11715b8d9dfa3d568))

- *(docs)* Stop the inpaint mask from leaking the original apple into the cake ([c6a5734a](https://github.com/swedishembedded/brain/commit/c6a5734ae2dd02a736f48b566689b5c2a5913207))

- *(gates)* Catch a stale Domain alternation instead of reporting orphan docs ([da80c585](https://github.com/swedishembedded/brain/commit/da80c585b13745129f19c4955b9e64218430ed7d))

- *(docs)* Regenerate the kernel table, which was missing flash_attn_causal_gqa ([8aa1c69d](https://github.com/swedishembedded/brain/commit/8aa1c69d1addfaac1058d1c7c41325560a45cd36))

- *(diffusion)* Guard klein_sigmas against a lookalike refactor ([a8bdc4ee](https://github.com/swedishembedded/brain/commit/a8bdc4ee22703612da1d0b276527fe049c4cc684))

- *(diamond)* Inherit the workspace version and license ([84edd1c4](https://github.com/swedishembedded/brain/commit/84edd1c46aac9809d0b8ac1a60a128f863b238d8))


### Build

- Add Brain Debian package targets ([be4dc750](https://github.com/swedishembedded/brain/commit/be4dc750979675a03ff198751ca0b6e21cce5658))

- A clippy gate that checks the EXIT CODE, not just the output ([51d9a8ff](https://github.com/swedishembedded/brain/commit/51d9a8ffd4663440b664c1e1303301ad109603a5))

- *(clippy-gate)* Re-baseline 179 -> 207 after the rebase onto 73 upstream commits ([8b4c426b](https://github.com/swedishembedded/brain/commit/8b4c426b48485bf5912799399b70c09b9d74b869))

- *(clippy-gate)* Ratchet 194 -> 185 ([59c4ad7c](https://github.com/swedishembedded/brain/commit/59c4ad7cbf98dd743e3192b71d342cbe2c61d4a1))

- *(clippy-gate)* Ratchet 185 -> 183 (the type aliases) ([f52486a8](https://github.com/swedishembedded/brain/commit/f52486a83a6f4023d27a83d0e944fd4558323c1d))

- *(npu)* Auto-detect Intel NPU and install/verify OpenVINO via make environment ([a6854f99](https://github.com/swedishembedded/brain/commit/a6854f997f41dd6db92bcabc7ed2e15d3e8444c7))

- Repoint Makefile's kernel-table comments at the new location ([4e4e0dcb](https://github.com/swedishembedded/brain/commit/4e4e0dcba8a4ce6a5988d2dee9eb3f1f2ea0af59))

- Widen the env-var documentation gate to the whole workspace ([4613161c](https://github.com/swedishembedded/brain/commit/4613161cda9dad5cf215f551643c71bb1fa72080))

- Add a pre-commit gate against docs/.agents citations in code ([3dbe80db](https://github.com/swedishembedded/brain/commit/3dbe80db7f49caabd963cde8f6e43af67acc03c6))

- Gate new em-dash (U+2014) usage in pre-commit, not existing ones ([8d65ed62](https://github.com/swedishembedded/brain/commit/8d65ed621f02ee5655d85ad2d9d5fd584f030188))

- Fix results/ artifact enumeration and gitignore hygiene (A3) ([695379bd](https://github.com/swedishembedded/brain/commit/695379bd650e06b96f64a20713bd5c22a98ef93b))

- Create results/.gitkeep, sync AGENTS.md's stale results/ note ([89b90630](https://github.com/swedishembedded/brain/commit/89b90630e6d41a24c5a47d5d2c320970d162f57a))

- *(clippy-gate)* Ratchet baseline 262 -> 279 after rebasing onto origin/main ([306618a1](https://github.com/swedishembedded/brain/commit/306618a189e048438560b2ef8f2542550d412b32))

- *(scripts)* Sweep Makefile and scripts/ for arch renames ([c8c0926f](https://github.com/swedishembedded/brain/commit/c8c0926f8872043f8bde1e334af056a666bd710c))


### Documentation

- Makefile targets + AGENTS routing for YOLO + event controller ([57b7b0aa](https://github.com/swedishembedded/brain/commit/57b7b0aa7549cbd5f7ff5d6b09fc616d7e7a996b))

- *(yolo)* Comprehensive training + inference guide with rendered test datasets ([81a95511](https://github.com/swedishembedded/brain/commit/81a9551126e1529b225badcabd29372206eb2ee9))

- PERFORMANCE.md - the full CPU & GPU inference optimization list ([0f22e326](https://github.com/swedishembedded/brain/commit/0f22e326fc6a07e6a01ac0949e651ef6a53224df))

- Record head conv+bias fusion + the producer-consumer fusion model ([ffd9457b](https://github.com/swedishembedded/brain/commit/ffd9457bf725234abf97c305c6b7723051e2da07))

- Record hoisted-bounds win + honest assessment of remaining GPU levers ([965ed5e6](https://github.com/swedishembedded/brain/commit/965ed5e6f271ef210c9af7bbb841f21c16fc99d2))

- Rewrite the user-facing README for the full feature set ([20965d42](https://github.com/swedishembedded/brain/commit/20965d429811d936023928578b2f172734f234e5))

- Acceleration write-up + backend comparison (ACCELERATION.md) ([be8d9d5e](https://github.com/swedishembedded/brain/commit/be8d9d5e55bb02dacab3a971e2ca0735b798b646))

- NPU dense-expert export design (Phase 4) ([404e397f](https://github.com/swedishembedded/brain/commit/404e397f69a4972512f1d943c4b06356f3adf630))

- GLM orientation + status + indexer-distillation plan ([a7fc5ede](https://github.com/swedishembedded/brain/commit/a7fc5edef22fb0d887f23e875e7a073bd14ca908))

- World-model performance section (measured levers + honest caveats) ([4f3f887d](https://github.com/swedishembedded/brain/commit/4f3f887d7e86f018eb972782e993bfd4504c6150))

- NPU row in the world-model performance table ([2c5dd0de](https://github.com/swedishembedded/brain/commit/2c5dd0de4babe65c9451e4fed3c2797609466164))

- Verified GenieRedux architecture spec from real checkpoints + source ([25f05f06](https://github.com/swedishembedded/brain/commit/25f05f069efb8fb84bf90f1307b99ae9473f578a))

- Depth guide + perf ledger; AGENTS routing for depth/vision/capture ([4dd6487c](https://github.com/swedishembedded/brain/commit/4dd6487c842b56c337fd2279955aa31227cde95b))

- *(depth)* Training section in guide + STATUS P4 ledger entry ([e5b23529](https://github.com/swedishembedded/brain/commit/e5b2352993920d5787ad5b6f5cb3fb5b34534ef1))

- *(depth)* P5.6 ledger - the dispatch-serialization finding + graph collapse ([367de989](https://github.com/swedishembedded/brain/commit/367de989dbb1ff8ae584a9cd05de03be091fa265))

- Mirror + splat guides, workstream ledgers, AGENTS routing ([707da969](https://github.com/swedishembedded/brain/commit/707da96902197a90499028be35d492075d04fdbd))

- Mirror + splat ledgers - T7 rect, S>1 fix, prune, NPU 6b/6c ([41b1d374](https://github.com/swedishembedded/brain/commit/41b1d3746b4322750ed8ce7b2b6224f6fe20b63a))

- Record the verified 3-frame reconstruction + honest S=3 timings ([a25de828](https://github.com/swedishembedded/brain/commit/a25de8281f7c875d610f6c21fe35d323dcb25477))

- Note the pre-existing GLM wgpu buffer-aliasing failure ([19d8ccd4](https://github.com/swedishembedded/brain/commit/19d8ccd4d07a207e4452523ed6d410346b0130ff))

- Fine-tuning datasets + fetch/train commands; tools/export_ohlcv.py ([069f30bd](https://github.com/swedishembedded/brain/commit/069f30bd355eac04bd702225818fddb8d48b4d8d))

- SCALING.md -- umbrella guide to scaling brain across GPUs ([5abc1cd6](https://github.com/swedishembedded/brain/commit/5abc1cd6c68828325af6716adbeb76461b5c6dae))

- TENSOR_PARALLEL.md -- TP mechanic, collectives, grid, planner ([3ef1a5ab](https://github.com/swedishembedded/brain/commit/3ef1a5ab5fe35a0fe62da3263811aa8d423896b2))

- Index the scaling docs from README + ARCHITECTURE ([4a5ad1c5](https://github.com/swedishembedded/brain/commit/4a5ad1c5cd25248296ef072f0628e10f5c41f417))

- Reorganize into a lowercase, hierarchical layout ([c6ce333a](https://github.com/swedishembedded/brain/commit/c6ce333a786b68c27b83d712360ac87ec72466bc))

- Build system for a single Markdown + professional PDF bundle ([395e69ee](https://github.com/swedishembedded/brain/commit/395e69ee36f944185d99021886c1c3c968a7d7a7))

- Refresh AGENTS.md + architecture.md; add the perf design and ledger ([4ee9286f](https://github.com/swedishembedded/brain/commit/4ee9286f7c550a1e0df0ede139309ede37338546))

- Record the Tier-2 scenarios and the second wave of findings ([ed313b22](https://github.com/swedishembedded/brain/commit/ed313b22f478118592053b8d6e09286af6ab74dc))

- Mitigation plan for every benchmark finding ([eae65bef](https://github.com/swedishembedded/brain/commit/eae65befaaca919d3204e324e67844074335665c))

- Replan mitigations around portable peak utilisation ([d8f04da4](https://github.com/swedishembedded/brain/commit/d8f04da4f7b64b388365b3832c0ca425c9aa9ce8))

- Record suite-reliability and dedup closures in the perf ledger ([ac228566](https://github.com/swedishembedded/brain/commit/ac228566c6e46ca3be7acd519229991e6dabd5d3))

- Device-sharing policy in AGENTS + un-stale the test-lane comment ([b3ef25ac](https://github.com/swedishembedded/brain/commit/b3ef25acdd26038a91f6438e5535849388b0fec5))

- Record the portability-spine + serving-mitigation landings in the ledger ([cb48f73d](https://github.com/swedishembedded/brain/commit/cb48f73d9f9bd916f9a0a9c54f641f567e5e85d0))

- Ledger - S3/S5, J2 gate, K stats, G/H/I scenario closures ([1b0c3571](https://github.com/swedishembedded/brain/commit/1b0c357148e79d59d157f7277b8e9d44305c362b))

- First VLM serving measurements in the ledger ([37c77d4c](https://github.com/swedishembedded/brain/commit/37c77d4c7086cc1c471d4f0e9f66a5595a9e8392))

- LFM2.5-Encoder guide + measured status ledger; AGENTS routing ([61ca5fb7](https://github.com/swedishembedded/brain/commit/61ca5fb7fe33f93696f8b77f3df150a4e6ac3a9d))

- Optimization-pass results in the LFM status ledger ([b73369d1](https://github.com/swedishembedded/brain/commit/b73369d1a63193698f466f15994c54a8479547e9))

- Porting playbook - the goldens-first parity methodology ([997c646b](https://github.com/swedishembedded/brain/commit/997c646bfdd910ac5a03da81d42122883ab80612))

- Kernel checklist + performance ladder, routed from AGENTS.md ([edd21bb4](https://github.com/swedishembedded/brain/commit/edd21bb4e052b6fda73b801542a47c909c24ca2d))

- *(lfm)* Record measured NPU numbers (Intel AI Boost, f16/int8/int4) ([9e343f62](https://github.com/swedishembedded/brain/commit/9e343f62ed96e2e72d5ccd5518f9988bd080ef1b))

- *(forecast)* Status ledger for forecasting over D-Bus + NPU placement ([937ac467](https://github.com/swedishembedded/brain/commit/937ac467c051348e8d6b32cc6170bb20dad18078))

- *(forecast)* IGPU serve path is registry-budgeted (post-rebase) ([7eac3aa8](https://github.com/swedishembedded/brain/commit/7eac3aa8cb79acc0051b3eb395c6699a28d5c139))

- *(forecast)* Record the training-batching optimization pass + numbers ([09087e16](https://github.com/swedishembedded/brain/commit/09087e161a551b531e93fcf8a396d2b9862be1f4))

- *(forecast)* Record the perf-target + regression-gate integration ([0fa01252](https://github.com/swedishembedded/brain/commit/0fa01252211bd4e1cb2e604e23bb0863471d7b06))

- *(forecast)* Record measured NPU-cached vs CPU kronos timing (2.1x) ([81e270f0](https://github.com/swedishembedded/brain/commit/81e270f045f44ffe7221aed2bfbe2d6ba13a4f4f))

- *(flux2)* Measured prompt-format effect on the edit path ([9eb8b57c](https://github.com/swedishembedded/brain/commit/9eb8b57c05f63fa7998de1e779455b453d283563))

- *(imaging)* Plan for SAM2 + identity conditioning + face restore + FLUX.1 ([c016ea5f](https://github.com/swedishembedded/brain/commit/c016ea5f6cadb8af45c9c22ccca1e8718786f5f7))

- *(imaging)* Add the UNet diffusion family + a backbone-agnostic ControlNet ([4167ba6a](https://github.com/swedishembedded/brain/commit/4167ba6af337fefbf701f91bf547b87ffc56f047))

- Require zero compile warnings; correct the "fp32 only" invariant ([eb76de08](https://github.com/swedishembedded/brain/commit/eb76de084b9cd64d47fbe88436e8a7c5c24279c3))

- *(imaging)* Mark phase 0 done and record both measurements ([2c210337](https://github.com/swedishembedded/brain/commit/2c2103376a4331b9111c3d6f92b38d796e4681c0))

- *(kernels)* A timed region without poll_wait measures the host, not the GPU ([883b40b0](https://github.com/swedishembedded/brain/commit/883b40b0c40cad7f517701a624b2e97aa100695c))

- *(imaging)* Record the phase 1-3b gate with the numbers that were measured ([b85cf41c](https://github.com/swedishembedded/brain/commit/b85cf41cc1ea38d4a80eebb469b77d131baf056c))

- Mandatory API security-audit policy + two-sources-of-truth spec rule (P17/P18 policy) ([3e05207d](https://github.com/swedishembedded/brain/commit/3e05207d6f7098dda35bbe82c669630d3867235e))

- *(examples)* Claude-with-brain.sh - run Claude Code on a local qwen3 via brain ([65943941](https://github.com/swedishembedded/brain/commit/65943941791f4fa2a1512cf60063a09412ba13b2))

- *(apis)* HTTP inference API guide (endpoints, catalog, model-card, testing, security) ([26798922](https://github.com/swedishembedded/brain/commit/26798922685f641a1f5758de1c93160c3ceb3d69))

- Observability guide - stats model + braintop (P13) ([a546c0b6](https://github.com/swedishembedded/brain/commit/a546c0b6cdbb832b48350d3814422024f7f8aba7))

- Move federated-moe.md under docs/ ([0a9b59e7](https://github.com/swedishembedded/brain/commit/0a9b59e7192bb62da47ce22372ecc6113234c96c))

- *(models)* Add readmes + status ledgers for the remaining models ([63fbcca9](https://github.com/swedishembedded/brain/commit/63fbcca9ce6be80a7fa4d132913b861cbb8e1259))

- Add serving & runtime stack prose map ([69b553e6](https://github.com/swedishembedded/brain/commit/69b553e649f1daeae77a4dcb6e47dbcbcadbb1bb))

- *(agents)* Fix routing-guide text and refresh stale references ([c1a7f476](https://github.com/swedishembedded/brain/commit/c1a7f476e54b0e40f7afa2b1df426a0e159d473d))

- Describe model seam architecture ([e46e98fc](https://github.com/swedishembedded/brain/commit/e46e98fc446b0591202602d193f06b962371513f))

- *(agents)* Move completed .todo tasks to .todo/completed/, not delete ([21b0cfcf](https://github.com/swedishembedded/brain/commit/21b0cfcf9c4295b1cff99e68796577b20d135861))

- *(agents)* Document the scripts/ vs tools/ distinction and check/scripts ([536d2a46](https://github.com/swedishembedded/brain/commit/536d2a46eb621b4b01a8722d7996d93549d7c428))

- *(imaging)* Record the training gate, and what finite differences cannot see ([9c938508](https://github.com/swedishembedded/brain/commit/9c93850807d523cd16cce29c59e29a4b836855d3))

- *(imaging)* Record the phase 3c/4 gate with the numbers that were measured ([ec787ebd](https://github.com/swedishembedded/brain/commit/ec787ebd913a650b6b432b85c22548ab5ab0e64c))

- Record phase 4b and the imaging serving contract, with measured numbers ([7b2bdf52](https://github.com/swedishembedded/brain/commit/7b2bdf526b9c62ca4f6139bebcb762c135ce1e4a))

- Record phases 4c and 5 and the first imaging backwards, with measured numbers ([0bad9c97](https://github.com/swedishembedded/brain/commit/0bad9c974c2f3557552b8b44931a6f90e5a56745))

- Route the new crates, and record what the workstream did NOT finish ([c0bfe6d3](https://github.com/swedishembedded/brain/commit/c0bfe6d3e246f2654448d7802d49988c253adb8c))

- Collect the cross-cutting lessons, and make recording them a rule ([3e2cbfb0](https://github.com/swedishembedded/brain/commit/3e2cbfb013ab5335a014e6ce1b66fa91161d9d86))

- *(lessons)* #19 - registration split across N lists is an unexploded defect ([9f2ec839](https://github.com/swedishembedded/brain/commit/9f2ec8393b3e18c8fb8cba8cb9473251b4344c10))

- *(upscale)* Blending the tile overlap is WORSE than cropping it - measured ([bcda0de6](https://github.com/swedishembedded/brain/commit/bcda0de6066d87d4c7afccf7c5344672ffcce7b9))

- *(yolo)* Fix loss.rs's doc lists by reading them, not by reindenting ([982457b2](https://github.com/swedishembedded/brain/commit/982457b251d1715cd708573f7a6b09840c323f19))

- *(imaging)* Scope BiRefNet from the checkpoint, not the paper ([4b8b0741](https://github.com/swedishembedded/brain/commit/4b8b0741cc1859b3b857645dd4f697a341664e98))

- *(todo)* BiRefNet matting as a standalone .todo task ([1dcde0f8](https://github.com/swedishembedded/brain/commit/1dcde0f88234d7a28ed29fb58507616b1b507762))

- *(lessons)* #20 - a fallback path is a path, measure it too ([84fa1ae1](https://github.com/swedishembedded/brain/commit/84fa1ae1b0eedd666ea86873b48716ca53329786))

- *(kernel-checklist)* §F - the optimisation loop, so the next kernel is fast by construction ([2ebe0370](https://github.com/swedishembedded/brain/commit/2ebe037031c9a07dd1305e0490096be16f8dee4a))

- *(training)* Record the semantic-validation work in the ledger ([aa6e80c9](https://github.com/swedishembedded/brain/commit/aa6e80c92d09bc8addc7354ad9abdfe60036191e))

- *(qwen)* Close out the named-LoRA-adapter workstream (M7-M9) in the ledgers ([455e4ce6](https://github.com/swedishembedded/brain/commit/455e4ce67863f9e5e00131c9f78912485e1a0660))

- *(lessons)* #21 - the per-kernel table is an upper bound, the whole pass is the truth ([ce28bdf9](https://github.com/swedishembedded/brain/commit/ce28bdf99883758fa28fe1b8649038cdbc11a7bf))

- Kill the "matmul_gemv in unet is free" hypothesis with the number ([635c2268](https://github.com/swedishembedded/brain/commit/635c22680071eafa939b9cc5e46c077556b9addb))

- Re-measure the per-dispatch floor at the current dispatch count ([e32367ad](https://github.com/swedishembedded/brain/commit/e32367adddf106ea10301dabae994cf17a86696b))

- Record the roofline, coverage and regen findings with their numbers ([04e2db5d](https://github.com/swedishembedded/brain/commit/04e2db5d8d621ddc98200dee30eb51a0ee51fa08))

- Record the Qwen3-0.6B decode baseline and what its ceiling means ([5dce9dbc](https://github.com/swedishembedded/brain/commit/5dce9dbce1b431cabf4c5bba6c1c876a36ce7b64))

- Correct the serving-GEMM entry - the submit-counter test does not skip ([3fba9829](https://github.com/swedishembedded/brain/commit/3fba9829e6b6b2e6e1c4cda20e34dc7f95b8adfe))

- Generate a kernel catalogue in README.md, gated against drift ([71011b8e](https://github.com/swedishembedded/brain/commit/71011b8e44f922628b770b822fe0bca302636ab4))

- *(kernels)* Declare the catalogue metadata in each kernel, not infer it ([3e1c6b2e](https://github.com/swedishembedded/brain/commit/3e1c6b2e118cab89e6b02a4dcf8c00d96bbc4679))

- *(AGENTS)* The kernel catalogue is declared, not derived ([201ae305](https://github.com/swedishembedded/brain/commit/201ae305ec92be8af585808f79cebcbc02ebd8f3))

- Correct the next-target call, and record what gqa_scores actually costs ([e4f65ec9](https://github.com/swedishembedded/brain/commit/e4f65ec986aa909ddc3f5611e063f437c2f8444c))

- *(qwen)* Doc-truth pass for the int8 paged KV default (W3.5) ([5ae0a1b0](https://github.com/swedishembedded/brain/commit/5ae0a1b0988c718518d2a0e71d112ebfe5274c24))

- *(cli)* Model_dir doc-truth pass; file the multi-file manifest follow-up ([f31c3c19](https://github.com/swedishembedded/brain/commit/f31c3c1964437ecaf8b8c0e12701629c43e008cc))

- *(omni)* Stand up the Qwen3-Omni-30B-A3B docs skeleton and goldens dumper ([20714ed0](https://github.com/swedishembedded/brain/commit/20714ed00a86670cbb0a0dc56227144d3fec790a))

- *(omni)* M6 design note -- M-RoPE collapses to plain RoPE for text/audio ([872db0d0](https://github.com/swedishembedded/brain/commit/872db0d0df1aa7cae8e74952b6c273a0dd4cc39b))

- *(omni)* M9b -- KV-cache decode + real multimodal input write-up ([3d29e232](https://github.com/swedishembedded/brain/commit/3d29e232b562569c1371cca94c1dd987b3f90124))

- M17 -- testdata audit finds most of testdata/ is not actually restorable here ([ead0c470](https://github.com/swedishembedded/brain/commit/ead0c470975f25050548f23395aeeda9b3ecee8a))

- Status ledgers + AGENTS.md + README.md for everything landed this arc ([574e681c](https://github.com/swedishembedded/brain/commit/574e681c70fd71535fb12c9ae866d9e54a7a98bc))

- *(performance)* Add the Intel Arc iGPU measurement ledger ([3175730a](https://github.com/swedishembedded/brain/commit/3175730a582593ebc0eaa1fe0ff158b92bc609d6))

- *(performance)* Resolve the wgpu-vs-vulkan DP4A/int8_dot contradiction ([44635fa3](https://github.com/swedishembedded/brain/commit/44635fa319cb5995c6d237ce9899f97f8984507b))

- *(models)* Add readme.md reference cards for 14 served models ([94893602](https://github.com/swedishembedded/brain/commit/948936020550ee0d42909c70fd6cae194aa75c15))

- *(readme)* Add a Model support table covering every servable model ([28a476f9](https://github.com/swedishembedded/brain/commit/28a476f9c501e9aed7f2eea9d46ce7fd11741204))

- Document auto-fetch for Z-Image and YOLO, extend the model-table drift guard ([e832460e](https://github.com/swedishembedded/brain/commit/e832460ec215d181aa370fc9da0c88f70de1cf92))

- Remove .todo/ path references from every git-tracked file ([65d24282](https://github.com/swedishembedded/brain/commit/65d24282872bc89a3a41cc0d93af35778ee1a796))

- *(readme)* Add the rest of the model catalog ([5bc2f5f2](https://github.com/swedishembedded/brain/commit/5bc2f5f22d16ca6da06d388438b015a3d94345bd))

- *(agents)* Add the three model-registry gaps the audit found ([753b78b4](https://github.com/swedishembedded/brain/commit/753b78b431d8f7d74c629ebf5f089c9bbd77ef40))

- *(qwen35)* Start the workstream ledger ([4bbe8f51](https://github.com/swedishembedded/brain/commit/4bbe8f5164949aa25d120d44544a26f45caceb67))

- *(agents)* Make full backward+gradcheck the default, not an opt-out ([17e30bdb](https://github.com/swedishembedded/brain/commit/17e30bdb38414f2437d6c6b0982529508214c87b))

- *(qwen35)* Update ledger for GDN forward + INT4 landing ([996a0aa8](https://github.com/swedishembedded/brain/commit/996a0aa85d120284e769f599f3560b0f71faf4b3))

- *(qwen35)* Record P8 completion and the F32-import storage constraint ([e424f877](https://github.com/swedishembedded/brain/commit/e424f8773b47e5397fc98ac2c8090c41d467034f))

- *(qwen35)* Record P10a (single-GPU int8) completion ([3340b6bc](https://github.com/swedishembedded/brain/commit/3340b6bc79a4807362f01ade2e1b02db7afc391b))

- *(qwen35)* Record P2b (GDN backward) completion ([294f69fe](https://github.com/swedishembedded/brain/commit/294f69fe772f958a88325e6de475ddb1f31a788e))

- *(qwen35)* Record P11a (decode primitives) + lesson on offset-direction bugs ([c7a984aa](https://github.com/swedishembedded/brain/commit/c7a984aada136c1f8120ec7cf1581065f4ca7bf0))

- *(qwen35)* Mark P11b done ([e8982c3c](https://github.com/swedishembedded/brain/commit/e8982c3cec053b3cb75b2859481520c115d84174))

- *(qwen35)* Record P11c, P14, and the CLI entry point ([a5f219b6](https://github.com/swedishembedded/brain/commit/a5f219b6fb4780254597f75d5adf01a8ce51dfad))

- *(qwen35)* Record P13 (full serving contract) as done ([b9eb0fc3](https://github.com/swedishembedded/brain/commit/b9eb0fc324d3175a13d30fbba663ec17de2c3560))

- *(qwen35)* Record P12 (LoRA + cross-GPU sharding) as done ([bf7bc757](https://github.com/swedishembedded/brain/commit/bf7bc7571ceffd90612baf6533a46355c42c76fd))

- *(AGENTS.md)* Add qwen35moe to the model index, fix stale crates/qwen refs ([6d14ea07](https://github.com/swedishembedded/brain/commit/6d14ea07102d2962c3fe9713f33ac917b344bf31))

- *(omni)* Add Qwen3-Omni data-flow infographic ([c143c13b](https://github.com/swedishembedded/brain/commit/c143c13b7a8b98ea72b8eb36a28fd4c0164fe189))

- *(zimage)* Record the fp32 stress-case result (clean structural failure) ([b9d4a070](https://github.com/swedishembedded/brain/commit/b9d4a0701434769e911ac6e9b9336a39476e8056))

- *(zimage)* Fp32 stress case now runs -- 144s/2 steps via the weight window ([d82f19e6](https://github.com/swedishembedded/brain/commit/d82f19e635e2c876cc5aca78a05ebeb21a9c29be))

- *(model)* Record shared_expert_fwd's missing adjoint (audit F18) ([7d4dedef](https://github.com/swedishembedded/brain/commit/7d4dedefbfd5bc81599caf47c46d9eb57b7300f2))

- *(serving)* Reference every serving BRAIN_* env var; gate it; fail loud on bad numeric flags ([7f80e52f](https://github.com/swedishembedded/brain/commit/7f80e52f0558e4d1596e9b4233fbe64798497268))

- *(models)* Qwenvl IS servable - fix four stale documents and un-blind the drift guard ([e45b1b9c](https://github.com/swedishembedded/brain/commit/e45b1b9c43cc91112157fce45529b7f1c5e3276e))

- *(omni)* Add the missing README servable row; delete two invented HTTP routes ([583d0130](https://github.com/swedishembedded/brain/commit/583d0130e309635a2f5cb9274e95d4201878f09f))

- *(agents)* AGENTS.md matches landed code - T5/vqgan backward, int8-Thinker state, VLM routing, dead .todo refs ([d29ebded](https://github.com/swedishembedded/brain/commit/d29ebded40b7c00e8fcbde3de34e3559ccd889cb))

- *(kernels)* Regenerate README kernel catalogue after core's header edits ([62499d48](https://github.com/swedishembedded/brain/commit/62499d4830fe22ae1419b3aa79b6181759af980a))

- *(models)* Renumber the duplicate lessons #35; no machine-local paths in copy-paste commands ([4a1c7dd6](https://github.com/swedishembedded/brain/commit/4a1c7dd6afcb0b01eb58461baad4f24489894336))

- *(qwen35)* Record P7 (sparse MoE decode dispatch) as done ([36440792](https://github.com/swedishembedded/brain/commit/36440792064fcf5eec200312c620410465039b0d))

- Add brain banner ([eb0af2fa](https://github.com/swedishembedded/brain/commit/eb0af2fa34d5a3bfaa12d0fa2a5fe734e200e9f6))

- Move contributor rules and per-model roadmaps to .agents/ ([fc647ba1](https://github.com/swedishembedded/brain/commit/fc647ba1b48a8346d01220744dae0d80d95d4235))

- Rewrite serving/API docs as docs/using/ ([99a07247](https://github.com/swedishembedded/brain/commit/99a072476b249483d19f8c255f7eb6e05312fc53))

- Replace docs/engine/ with docs/introduction/ ([74e13bbb](https://github.com/swedishembedded/brain/commit/74e13bbb081c51c85b5f0ee54d6a008dc7d55ad0))

- Remove ADR, raw LLM transcript, and unreferenced research essay ([d87beae4](https://github.com/swedishembedded/brain/commit/d87beae420193154712ff267814da8d3bda18475))

- Rewrite the model catalog as user-facing pages ([533eb429](https://github.com/swedishembedded/brain/commit/533eb429aa5317c6a9dc056c19c1e692e82d352d))

- Add task-oriented "what brain can do" overview pages ([3d950d59](https://github.com/swedishembedded/brain/commit/3d950d596d2375623b0aad3c971fa9db4916bb61))

- Rewrite training guides as docs/training/ ([94480148](https://github.com/swedishembedded/brain/commit/944801485a52356e653d5f545f7c16bc12c62b28))

- Rewrite performance and scaling docs, correcting stale claims ([9e4b2438](https://github.com/swedishembedded/brain/commit/9e4b2438f11b7069df28c6e6e500bb2cfbf04cd2))

- Add doc-tree landing page, rewrite manifest, support PDF parts ([fe20fad1](https://github.com/swedishembedded/brain/commit/fe20fad1a0f7b0e349d527825b96a28ed4e59943))

- Turn README.md into a product brief, move the kernel catalogue out ([65da75a6](https://github.com/swedishembedded/brain/commit/65da75a694bdc3cc9436b2010842ca18827ecb79))

- Repoint AGENTS.md at the new doc tree, stop routing findings to docs/ ([ab557794](https://github.com/swedishembedded/brain/commit/ab5577947843689ed55368af581eeacefa64fa4b))

- Remove docs/imaging/plan.md ([66bc7174](https://github.com/swedishembedded/brain/commit/66bc717461533e750221d255e30c4ab68bcd10d0))

- Repoint and repair doc references in code comments ([cadabf53](https://github.com/swedishembedded/brain/commit/cadabf53af9de5ac9f5e343deb67ce27c8750e21))

- Remove all docs/ and .agents/ file-path citations from code ([808d8109](https://github.com/swedishembedded/brain/commit/808d81095e68610a48cf961260e8e447df9832bf))

- Document deepseek-ocr (model page, routing, env var, example) ([378ef715](https://github.com/swedishembedded/brain/commit/378ef7155fec83cfd36e0340ad52223167c94bb3))

- *(deepseek-ocr)* Document the LoRA training story, close the roadmap item ([d9767028](https://github.com/swedishembedded/brain/commit/d97670286cc4221afe05e3bfd87e43988ec08e82))

- *(deepseek-ocr)* Record the CPU AVX2/AVX-512 fast-path pass's measured impact ([9da3f224](https://github.com/swedishembedded/brain/commit/9da3f2249ef34947aeb7af5ec55394074986d11c))

- Docs(deepseek-ocr): vision-encoder perf pass -- CPU-vs-GPU gap, per-kernel
breakdown, both fixes' before/after, updated head-to-head

Documents this session's follow-up to the KV-cache decode pass: the CPU pin
sam1 runs under (crates/sam1's wgpu backend corrupts output at production
scale, a separate open defect, not touched here) costs ~3.6x measured via a
new weight-free bench; attn_apply_cross was found at 70-71% of the tower's
own CPU forward via BRAIN_PROFILE, traced to a cache-hostile V-transpose;
two fixes landed (sam1's pick_gemm wiring, a tiled transpose in
backend-cpu) with honest before/after including where the whole-pass
magnitude could NOT be pinned down this session due to a busy shared
machine; new per-stage BRAIN_PROFILE instrumentation inside the vision
encoder itself; and an updated, honestly-caveated head-to-head against
llama-mtmd-cli (~6-7x -> ~1.4-1.9x under matched same-session conditions,
explicitly NOT claimed as an absolute number since llama.cpp's own runs
were equally slowed by the same machine load).

Updates docs/performance/overview.md's DeepSeek-OCR case study,
.agents/roadmap/deepseek-ocr.md's Phase 8 entry, and
docs/models/deepseek-ocr.md's hardware/limits section. ([eafbf70f](https://github.com/swedishembedded/brain/commit/eafbf70fbcf74fcc14636faf3d9d8519cc2f5465))

- *(deepseek-ocr)* Clean single-tenant head-to-head vs llama-mtmd-cli ([fbba2840](https://github.com/swedishembedded/brain/commit/fbba28407a78987a98d029ba20d330a34b7b0676))

- *(deepseek-ocr)* Drop llama.cpp naming from the performance/roadmap docs ([6b6a8d96](https://github.com/swedishembedded/brain/commit/6b6a8d9644cf8e6b88ccff5bc32eb26a9ade4dc3))

- *(deepseek-ocr)* Name llama.cpp explicitly in the oracle/provenance notes ([f88da499](https://github.com/swedishembedded/brain/commit/f88da499b2b389a3ec5b38e396bc8cdef2e97908))

- *(deepseek-ocr)* Model-construction profiling findings, JIT ruled out ([6d338677](https://github.com/swedishembedded/brain/commit/6d338677e662ee04e639b6368248e6899e5f0d1f))

- *(deepseek-ocr)* Real 50-page document throughput + concurrent-request measurement ([94e03999](https://github.com/swedishembedded/brain/commit/94e039997303fb3d4f635e5ee85098554f5f9c76))

- *(deepseek-ocr)* Confirm the wgpu sam1 correctness fix, document why the CPU pin still stays ([8ec3b458](https://github.com/swedishembedded/brain/commit/8ec3b458a66566197ad1eea5e5e7d52dd7aefba2))

- *(deepseek-ocr)* Phase 8 conclusion, real numbers for the split-device pass ([66a13874](https://github.com/swedishembedded/brain/commit/66a13874f6a56e0d13d3a1873320a3f4a7c36440))

- *(deepseek-ocr)* Correct the max_new=32 head-to-head, 1.8x was not reproducible ([f182d91d](https://github.com/swedishembedded/brain/commit/f182d91d016c9f09788f8595648599c1530c7fc3))

- Sweep bare perf numbers to zero, fix stale hardware/NPU claims ([acc432a6](https://github.com/swedishembedded/brain/commit/acc432a697d3d46230bd101c82a8d556e231b26a))

- *(roadmap)* Record the no-raw-git-plumbing rule for concurrent phases ([9437cc8f](https://github.com/swedishembedded/brain/commit/9437cc8f1c5fc444d7ae71ce64e08b82991061ff))

- *(omni)* Drop bare perf numbers inherited from the omni rebase ([fa31cd96](https://github.com/swedishembedded/brain/commit/fa31cd96d77191cd2b5a394bfd3e0fdc144d34f6))

- *(models)* Restructure docs/models/ to canonical arch ids ([f265bb94](https://github.com/swedishembedded/brain/commit/f265bb944e86fc4b883b507ef724ddd16c18b530))

- *(agents)* Sweep AGENTS.md, .agents/rules/, .agents/roadmap/ for arch renames ([11de825b](https://github.com/swedishembedded/brain/commit/11de825bb4fad55e1bc2903099b1dca52455a5b8))

- *(examples)* Sweep examples/ for arch renames and CLI grammar ([e094a074](https://github.com/swedishembedded/brain/commit/e094a0741e54c170f18ec38fbd0ccff528972051))

- *(readme)* Flagless quick start + Model support table ([b65d8a1b](https://github.com/swedishembedded/brain/commit/b65d8a1bfb9fd2b91f6b820c207a084fb1023ca6))

- *(roadmap)* Self-improve initiative status, P1-P5 done, P6 scoped ([e8c45b09](https://github.com/swedishembedded/brain/commit/e8c45b097614fc207e0265cfdbce401dc57571ea))

- Docs, Makefile: fix dead CLI syntax examples across docs/models and the Makefile

A sweep of every tracked `brain ...` invocation in docs/ and the Makefile
against what the CLI resolver actually accepts today, after finding the
first few by hand while validating a working quickstart. Three recurring
kinds of drift:

- `brain qwen ...` (Makefile's train/qwen/lora target and its own comment,
  docs/training/lora.md, docs/using/models-and-weights.md) -- brain_arch's
  id is qwen3, exact-match only; `make train/qwen/lora` failed outright.
- `brain tts ...` (all of docs/models/qwen3tts.md) -- the id is qwen3tts.
- "CLI (`brain do`)" as a capability-table header across 23 docs/models/
  pages -- `brain do` was replaced by `brain <arch> <action>` grammar;
  the header cell kept the old spelling.

Also: `brain import-gguf` corrected to `brain import`
(docs/using/models-and-weights.md), a stray reference to the gitignored
.todo/ tree removed from the Makefile's own comment, and assorted smaller
staleness (dead flags, renamed ids) fixed page-by-page where found during
the same pass. ([a099c7f8](https://github.com/swedishembedded/brain/commit/a099c7f8f5ea0752596b25f31b5240e29f319e8d))

- Correct fp32-only framing -- int8 DP4A is a shipped, first-class path ([5cb887c5](https://github.com/swedishembedded/brain/commit/5cb887c59b61d1a5665ffb1ffee67304f3bf0322))

- Update model index auto-fetch (⤓) markers to match reality ([a40e7a79](https://github.com/swedishembedded/brain/commit/a40e7a79bd251ce93fa38ab0aa9cb4b732eefd8b))

- *(roadmap)* Record P6a's hot-swap cycle glue and the remaining gap to P6 ([8776b4a7](https://github.com/swedishembedded/brain/commit/8776b4a767b8700a1035c3f6da7aabde1aa93cfc))

- *(roadmap)* Defer P6 to sven's side, link the task brief ([7def1e28](https://github.com/swedishembedded/brain/commit/7def1e285c9f4906cbd04557554cb358dedd3877))

- *(roadmap)* Document real test-suite hang findings (backend-vulkan, npu, modelstore) ([5ff07388](https://github.com/swedishembedded/brain/commit/5ff073889647c7c4de67773f5d8dba8e4c1e8085))

- *(readme)* Rewrite Quick start with real, current output and plain markdown images ([5ca4f97a](https://github.com/swedishembedded/brain/commit/5ca4f97a5ff4d5c072d1f8679d840f73deddfea4))

- *(roadmap)* Mark s3dit and qwen3's real bugs fixed, with root cause and verification ([aa606767](https://github.com/swedishembedded/brain/commit/aa606767dbc54f8331e2b4d03229c8fcfba2e04e))

- *(roadmap)* Root-cause the intermittent SIGSEGV as an NVIDIA driver defect, unify it with fastvlm's ([e31ea175](https://github.com/swedishembedded/brain/commit/e31ea175b37798b6dbb430ec65f0ef792b5da0ce))


### Features

- *(bench)* Toolcall benchmark - train+score only the assistant tool-call span ([ebf33f99](https://github.com/swedishembedded/brain/commit/ebf33f992a07f394402f914f3df16e27d3bc0a0c))

- *(bench)* Formal-language benchmarks - parity, mod_add, dyck ([f6829e01](https://github.com/swedishembedded/brain/commit/f6829e011b26b5ba824e6d4a8640e81f43a9dd82))

- *(kernels)* Bidirectional attention kernels + FD tests (PR-7) ([99a00f19](https://github.com/swedishembedded/brain/commit/99a00f190645c6832fa90da3d614e71697dc5876))

- *(kernels)* Cross-attention kernels + FD tests (PR-8) ([2214b380](https://github.com/swedishembedded/brain/commit/2214b3802236deb7a82d7dc8f6e169a88ed46bf8))

- *(seq2seq)* Encoder-decoder Transformer Model (gradient-checked, PR-9) ([86083315](https://github.com/swedishembedded/brain/commit/86083315b534e2e8fbbad7d0243435b24b4cf207))

- *(bench)* Multi-scale scaling-law sweep harness (L(N)=E+A*N^-alpha) ([f6069a02](https://github.com/swedishembedded/brain/commit/f6069a024c20d6474cb90f68c4703638edae1f5a))

- Regression head + MSE kernels + autoencoder; register mad_compress (PR-10) ([306ef347](https://github.com/swedishembedded/brain/commit/306ef3471849fe1e6c3b6675b7d1b5dc4c42ef46))

- *(bench)* Turn-key architecture eval harness - battery + axes + results + compare (#18a) ([739ec51b](https://github.com/swedishembedded/brain/commit/739ec51b6f39cd6737c5421e3573e135d6b916b9))

- *(bench)* Predictive per-capability scaling + tuning advisor (#18b) ([611660f7](https://github.com/swedishembedded/brain/commit/611660f7488b752c3da1a608f1188f1d2d5e243a))

- *(bench)* MoE is benchmark-capable (MoeDecoder + Engine per-position logits); add moe arch ([2d0deb1e](https://github.com/swedishembedded/brain/commit/2d0deb1e4dfc7d0a70703413ef0d951be98f001e))

- *(kernels)* Add conv-net + detection WGSL kernels ([0f169964](https://github.com/swedishembedded/brain/commit/0f169964acf24f8f1724fb1fd2cbf1907495f020))

- *(events)* JSONL event protocol crate ([0b9bb241](https://github.com/swedishembedded/brain/commit/0b9bb2419889d7753fd17b1b5ae76675ced97273))

- *(hfsm)* Generic hierarchical state machine engine ([2a0d20a3](https://github.com/swedishembedded/brain/commit/2a0d20a3e4eb32ce5eea5389d832ae12a66d4647))

- *(data)* Synthetic object-detection dataset generator ([473f67d8](https://github.com/swedishembedded/brain/commit/473f67d8a076766eb4e97ee2ff2d864cb19cf962))

- *(eval)* Detection metrics (IoU / precision / recall / mAP@0.5) ([06dfd873](https://github.com/swedishembedded/brain/commit/06dfd873eeeb2c7349f34367fc0181038fddc826))

- *(yolo)* From-scratch YOLOv8 anchor-free detector ([a835e669](https://github.com/swedishembedded/brain/commit/a835e66917b37b97f81da2884b320b65f406cc23))

- *(runtime)* Event-driven multimodel controller ([d44e7ef6](https://github.com/swedishembedded/brain/commit/d44e7ef6d41c9d3431185366295a12f3acab91ce))

- *(cli)* Brain yolo subcommands + brain run event loop ([581dd75c](https://github.com/swedishembedded/brain/commit/581dd75c5926716e655adadb67c86da5290b2b94))

- *(events,runtime)* Req_id request/response correlation ([78e9b8ce](https://github.com/swedishembedded/brain/commit/78e9b8cee09213cf4b436259ddeeb8729c982c0b))

- *(cli)* --conf/BRAIN_CONF detection threshold for brain run ([6c8499df](https://github.com/swedishembedded/brain/commit/6c8499df6c95f0f20be3e23108ad866843721043))

- *(brain-py)* Event-driven Python client + annotated-image example ([8210443a](https://github.com/swedishembedded/brain/commit/8210443aae8e275245beb63d26977b59f1ec5f32))

- *(brain-py)* COCO labels + --timeout for real yolov8n inference ([8b2954b6](https://github.com/swedishembedded/brain/commit/8b2954b692c04b4a7d89ccf2c5c2f46341f61684))

- *(wgsl-cpu)* Work-group execution model - workgroup memory + barriers ([00f7cece](https://github.com/swedishembedded/brain/commit/00f7cece6a076cc4b588f5e611fa861b88a54bde))

- *(yolo)* Weight-tiled conv on GPU via single-source WGSL (B2+B3) ([5effd451](https://github.com/swedishembedded/brain/commit/5effd451480fdea57362b4469884fcaa50b92d9f))

- *(onnx)* Pure-Rust ONNX graph model + serializer crate ([f47f50ee](https://github.com/swedishembedded/brain/commit/f47f50ee950a6541cf1c6236d81868995a3fcf75))

- *(npu)* YOLO->ONNX export + brain-native INT8 PTQ + OpenVINO NPU runtime ([0aab1e42](https://github.com/swedishembedded/brain/commit/0aab1e42e46ca3d93d4d9ba29889246b14544e6f))

- *(cli)* Brain npu commands + --device npu; Makefile targets + docs ([a42798eb](https://github.com/swedishembedded/brain/commit/a42798eb0b4ca78cacb12f343222ea0d3b0974c6))

- *(sched)* Make the integrated GPU schedulable for forecasting ([699b6ffa](https://github.com/swedishembedded/brain/commit/699b6ffadf4bfa661efe6cc3997989ffc60c188b))

- *(kronos)* Batched decoder training (b sequences/step) - gradcheck + parity gated ([c5da17b9](https://github.com/swedishembedded/brain/commit/c5da17b99a972f1714bed6f9543c21a9a16ab300))

- *(perf)* Kronos forecaster as a `brain perf` target (regression-gate-ready) ([077eca95](https://github.com/swedishembedded/brain/commit/077eca9573c1cf957b05a34fdebfe1403596dd9c))

- *(perf)* Chronos2 + fincast forecast targets (DRY shared executor) ([e7610fbe](https://github.com/swedishembedded/brain/commit/e7610fbe2d8d83e74d6f0fa063ac39ad70f5845a))

- *(forecast)* Consolidated parity gate + latency regression gate (#40) ([ed329348](https://github.com/swedishembedded/brain/commit/ed329348c03f7ab0e28cf7d72f67633d4bf073cb))

- *(npu)* Nemotron FastConformer encoder ONNX - Conformer blocks + projectors ([a97c313e](https://github.com/swedishembedded/brain/commit/a97c313eef3fbe55dba80a75346d6995454642b9))

- *(npu)* Nemotron encoder ONNX export (external-data) + full-graph gate ([a8c1c7a6](https://github.com/swedishembedded/brain/commit/a8c1c7a660138d4a6c61add88d3a62ef86f33406))

- *(npu)* Qwen3-ASR audio-encoder head ONNX (windowed ViT + projector) ([79f13c83](https://github.com/swedishembedded/brain/commit/79f13c83ac6713039443e2c49bb85c3301f7193f))

- *(npu)* Kronos s1 KV-cache graphs (prefill + single-token decode) ([0d2740ba](https://github.com/swedishembedded/brain/commit/0d2740bac261eb6ab9450c93060af279f3aafd81))

- *(npu)* Kronos dep (s2) KV-cache graphs + parity ([cf671213](https://github.com/swedishembedded/brain/commit/cf67121378bc29d925f3c939ff4395653c81f5fc))

- *(kronos/npu)* Cached NPU rollout - KronosNpuInstance uses KV-cache (O(cap)/step) ([edac5dc9](https://github.com/swedishembedded/brain/commit/edac5dc96284325673e679f228da40dc217e6fb0))

- *(kronos/npu)* Shared-prefill for the cached NPU rollout (samples=N) ([f5aeeb67](https://github.com/swedishembedded/brain/commit/f5aeeb67a27c97049339cf5002d5699616073d5c))

- *(kernels)* Maxpool2d + maxpool2d_dx - generic K, stride, pad max-pool ([3a565d60](https://github.com/swedishembedded/brain/commit/3a565d60997a15cc69b623db78c6e650f682d9a4))

- *(kernels)* Convtr2d + _dx + _dw - 2D transposed convolution ([3656ea62](https://github.com/swedishembedded/brain/commit/3656ea626821ece67086cce2d4d601a156a9c588))

- *(kernels)* Grid_sample + _dx + _dgrid - bilinear resample at data-dependent coords ([17b01141](https://github.com/swedishembedded/brain/commit/17b01141535a84296cf80f9b17866a4e10cd5b3e))

- *(kernels)* Resize_bicubic + _dx - Catmull-Rom (a = -0.75) 2D resample ([440f12cb](https://github.com/swedishembedded/brain/commit/440f12cbe869f581a13da7de8ee902b21445699d))

- *(kernels)* Prelu + _bwd + _bwd_wg - learned per-channel PReLU ([9d249735](https://github.com/swedishembedded/brain/commit/9d2497356206fb258821d55f887d2048fcfce4d3))

- *(vision)* ConvTranspose, generic MaxPool, LayerNorm2d and the ConvNeXt CXBlock ([ec9fa671](https://github.com/swedishembedded/brain/commit/ec9fa6712d178a4025e345410f5cb5b451d20ca4))

- *(imaging)* Add crates/imaging as the one home for image handling ([82ed7f95](https://github.com/swedishembedded/brain/commit/82ed7f959461fe2de9562b338b6a5c9666cbf65b))

- *(capability)* Add Blob::with_media so an encoder can retag its output ([e5e23f73](https://github.com/swedishembedded/brain/commit/e5e23f73480a8a6705b1250bcbff3a173effa77c))

- *(model)* Windowed spans and q_pool in the shared ViT builder ([80178df9](https://github.com/swedishembedded/brain/commit/80178df9afd9dca534cdb49cadce23a5d826811c))

- *(model)* Cosine and l2_normalize get a home in hostmath ([f94c9dcd](https://github.com/swedishembedded/brain/commit/f94c9dcda2ddc6e6e7e22488d660ddec5913deba))

- *(vision)* AvgPool and PReLU join the shared conv-net blocks ([1bf23f18](https://github.com/swedishembedded/brain/commit/1bf23f1886a770bfd6f2d78056a6d51a7b133446))

- *(onnx)* An import-side reader, so nothing hand-rolls a second protobuf decoder ([c8861e4b](https://github.com/swedishembedded/brain/commit/c8861e4bd65c5d56f462609bed29530debeddb11))

- *(kernels)* Quick_gelu, the activation CLIP-L actually uses ([f1b4fdc7](https://github.com/swedishembedded/brain/commit/f1b4fdc728e0b87804f72ad87bc76cca1212a6f2))

- *(sam2)* SAM 2.1 image path, forward-parity-gated on both released variants ([fba93e5e](https://github.com/swedishembedded/brain/commit/fba93e5ebb1fd516019d7628befd07f86c13c99d))

- *(facenet)* The insightface antelopev2 stack, ONNX-imported and parity-gated ([0337795a](https://github.com/swedishembedded/brain/commit/0337795a83ffeaa49fca2e7b1d5fbfe6fbe679f4))

- *(vqgan)* The CodeFormer VQ autoencoder, on the hoisted vae blocks ([4426b60a](https://github.com/swedishembedded/brain/commit/4426b60a92fdf88e53d46a8a57644724079e7193))

- *(clip)* CLIP-L, OpenCLIP-bigG and EVA-CLIP-L/336 behind one graph ([8034b57a](https://github.com/swedishembedded/brain/commit/8034b57a1ef9035a7e186a756e64fcd419b4cfab))

- *(asr)* Wire Nemotron + Qwen3-ASR encoders to the NPU (#24) ([925e1fa4](https://github.com/swedishembedded/brain/commit/925e1fa4691f7722b46245bc4b15e8dfa604d3ec))

- *(asr)* OpenVINO model-cache for the NPU encoders + status doc ([acbb5a6b](https://github.com/swedishembedded/brain/commit/acbb5a6bb2334a55777ec57d7ff058adf7b81c9c))

- *(checkpoint)* Safetensors read/write + ModelCard (P1a) ([cb817bda](https://github.com/swedishembedded/brain/commit/cb817bda5add250ccbf26d597582448f9276817b))

- *(checkpoint)* GGUF full KV metadata + per-tensor dequant (P1c) ([b06707e6](https://github.com/swedishembedded/brain/commit/b06707e65b8aec65ad9e036aa788a5937e7c33b9))

- *(checkpoint)* Back checkpoint on safetensors; retire PyTorch parity (P1b-core) ([19c7604a](https://github.com/swedishembedded/brain/commit/19c7604a7ec40b2720f56d6840bd50d1df69aa46))

- *(checkpoint)* Streaming mmap reader + incremental writer (P1d-core) ([955101b2](https://github.com/swedishembedded/brain/commit/955101b2b60e28fa4a22e30e93e4684a25805b76))

- Stream served-model loads via TensorSource (P1d-consumer) ([4dc1499d](https://github.com/swedishembedded/brain/commit/4dc1499d5de708f74a9ef40708c6afbcc7f3e66f))

- *(cli)* Global model directory + per-file carded residents (P2) ([6fbfdc45](https://github.com/swedishembedded/brain/commit/6fbfdc455d08d2ff200cc5a8dc536ad8efd69b6c))

- *(apiserve)* HTTP API scaffold + 3 provider skeletons (P4) ([28fd2820](https://github.com/swedishembedded/brain/commit/28fd28203e29f001e5babcd7618859b272c60825))

- *(cli)* Wire apiserve into `brain serve` - one shared executor (P4) ([c13db9af](https://github.com/swedishembedded/brain/commit/c13db9affaba4238c9c48d99062b2f3a4202576d))

- *(qwen)* Chat `generate` contract - messages/system/top_p/stop + per-token stream (P5/P7) ([b67f1815](https://github.com/swedishembedded/brain/commit/b67f1815dd4f45ad63f68084b12d825d496a9016))

- *(apiserve)* Real chat - non-stream + SSE for all 3 providers (P5/P7) ([88808920](https://github.com/swedishembedded/brain/commit/88808920194b255d8278742ac24b9a800de7b542))

- *(apiserve)* Admission control - 429 within deadline + edge shed 503 (P6) ([8849c86b](https://github.com/swedishembedded/brain/commit/8849c86b9c921f0379648e81ebd54543dc889f75))

- *(apiserve)* Embeddings endpoint (OpenAI + OpenRouter) (P9) ([f66445eb](https://github.com/swedishembedded/brain/commit/f66445eb851995b9b078ce529b681d0935dd766b))

- `api-sync` command - refresh vendored upstream specs, report drift (P18) ([d07c1f1c](https://github.com/swedishembedded/brain/commit/d07c1f1c97bf7b090d2eb5eb31ac183343a7e71c))

- *(apiserve)* Images/generations (OpenAI + OpenRouter) + denoise SSE (P10) ([555a4927](https://github.com/swedishembedded/brain/commit/555a4927e9cb33db2488834c337005f341e5f19a))

- *(apiserve)* OpenRouter model prefix-strip + models fallback (P11) ([71e526dd](https://github.com/swedishembedded/brain/commit/71e526dd9c62f470da2b268bbc4e119b14874251))

- *(stats)* Self-describing stats subsystem + D-Bus stream (P14) ([cef75f45](https://github.com/swedishembedded/brain/commit/cef75f450ca0dc355a3e8e59be2ae8ca46f374cd))

- *(braintop)* Btop-like TUI over the D-Bus stats stream (P15) ([5f6aea1a](https://github.com/swedishembedded/brain/commit/5f6aea1a18e5b32e75707ec37cd79e8b5e2cdda7))

- *(stats)* Populate live requests[] from executor in-flight jobs (P20) ([a0f63e22](https://github.com/swedishembedded/brain/commit/a0f63e227407575e887fcc487cfa58a59ba82b1e))

- *(brain-py)* D-Bus default + clean multi-transport capability client (P19) ([b1da1b6a](https://github.com/swedishembedded/brain/commit/b1da1b6ac913891aff51cb6002ee2ea4e6d846d8))

- GGUF embedded tokenizer -> brain BPE, so .gguf chat models serve (P21) ([79c61211](https://github.com/swedishembedded/brain/commit/79c612119557acb318f0d559d2701cd7cdba0150))

- Stream ::load inference paths off mmap WeightReader (P22a) ([915da257](https://github.com/swedishembedded/brain/commit/915da2578b5ea961a5ef5a29c66917992b00116c))

- Brain fetch -- generic model downloader + real-GGUF family alias fix ([c4a13525](https://github.com/swedishembedded/brain/commit/c4a13525ac93f85902bd5a1b90353597e7925378))

- *(checkpoint)* WeightReader::open_hf_dir -- streaming sharded HF reader (P22-full 1/5) ([a52467c5](https://github.com/swedishembedded/brain/commit/a52467c51b495dd3e9ea77cf9294afac95af827a))

- *(checkpoint)* StWriter allows any-order writes (P22-full 2/5) ([761ddf88](https://github.com/swedishembedded/brain/commit/761ddf88aedb041322903b3a41f605a831eb3afe))

- *(federated)* Stream MoE expert sharding, never load the whole base (P22-full 3/5) ([15316324](https://github.com/swedishembedded/brain/commit/15316324d418bbde30b54481a518ff4ca53d7a81))

- Stream qwen/glm/lfm HF imports -- never build the whole model in RAM (P22-full 4/5) ([b89ff00e](https://github.com/swedishembedded/brain/commit/b89ff00efc76348b7080e6f1e693102710ef3e25))

- Stream chronos2/fincast/mirror/speaker/tts/codec HF imports (P22-full 5/6) ([595d0112](https://github.com/swedishembedded/brain/commit/595d011233358182bce030b77481ff96d7ef38a2))

- *(npu)* Stream ONNX export source reads; fix nemotron double-materialization (P22-full 6/6) ([fcc9bace](https://github.com/swedishembedded/brain/commit/fcc9baced85b9ac75524d47a95be80075c5a9f6e))

- *(cli)* Add version flags ([f6ce5280](https://github.com/swedishembedded/brain/commit/f6ce52804086889acd9f3018c71d97216d83f651))

- *(cli)* Extend the mock model so more examples run for real ([26dfab89](https://github.com/swedishembedded/brain/commit/26dfab89c03cfc8bf402b87af2c0917a2f01f84b))

- *(modelref)* The fully-qualified model reference grammar ([67284efd](https://github.com/swedishembedded/brain/commit/67284efd894777eb64104dcdc5e0f9f93b292a6f))

- *(modelstore)* New crate for the model store layout and fetch resolution ladder ([f3643c84](https://github.com/swedishembedded/brain/commit/f3643c84a88a37764aaf1e03fbb49220eb3367f6))

- *(models)* Rename every served model to its fully-qualified <vendor>/<repo> id ([bc081042](https://github.com/swedishembedded/brain/commit/bc081042d51251e6fbdf88e79bc057b182b7c8ed))

- *(checkpoint)* From-scratch GGUF quantizers and writer ([b698ce35](https://github.com/swedishembedded/brain/commit/b698ce353170494cdc6f3322089373a588e6304e))

- *(backend-api)* DType/promote and NumericSupport bf16 fields ([53f72691](https://github.com/swedishembedded/brain/commit/53f7269114d15d63458bc69be4f2c54c4f9d24ea))

- *(residency)* Dynamic model registration on a running Executor ([51aa3469](https://github.com/swedishembedded/brain/commit/51aa3469c5244731f93608c766b226ad96f6f264))

- *(residency,cli)* The model-supplier seam (auto-fetch's front half) ([e4ff936b](https://github.com/swedishembedded/brain/commit/e4ff936b83b06b22ed8dd13518ed6f23c44de59c))

- *(modelstore,testutil)* Migrate testdata checkpoints into the model store ([6d922d16](https://github.com/swedishembedded/brain/commit/6d922d16b1a78f776fc7c8ffeed45579622ed6d9))

- *(modelstore,qwen,glm,lfm)* Dispatch a fetched checkpoint's Convert step by architecture ([4f167cf3](https://github.com/swedishembedded/brain/commit/4f167cf39d8283049b96f19b790dd14b4fe58233))

- *(cli,apiserve,dbus)* Wire transparent auto-fetch into HTTP and D-Bus, with live progress ([abb564ae](https://github.com/swedishembedded/brain/commit/abb564ae0fbfb79c4175d24cc3308813e406afb6))

- *(vae)* Backward for the shared conv/attn/resnet blocks ([3ead5995](https://github.com/swedishembedded/brain/commit/3ead5995f84b77cb792867f248bd797ff1446f4a))

- *(sam2)* Mask-decoder backward, with the trunk and neck frozen ([428853e8](https://github.com/swedishembedded/brain/commit/428853e8af5c1410a88fc7753d4cc6a120438381))

- *(facenet)* ArcFace training graph for the IResNet embedding ([5523e9d1](https://github.com/swedishembedded/brain/commit/5523e9d125a5d51f1295524b900d26085ca797d7))

- *(vqgan)* The VQ straight-through estimator, so the encoder actually trains ([7b131088](https://github.com/swedishembedded/brain/commit/7b1310889984f57f46ff90d235613f640d654b83))

- *(clip)* SSA training forward and backward for the text tower ([3b15554a](https://github.com/swedishembedded/brain/commit/3b15554a1dd791f87990c30f9ee5d9bae165b620))

- *(model)* Hoist the four dispatch rules FLUX.1 would otherwise have copied ([80628b70](https://github.com/swedishembedded/brain/commit/80628b70977ce30c673005222c826bd91536d538))

- *(t5)* The T5-XXL encoder FLUX.1 conditions on, parity-gated per stage ([a3a47980](https://github.com/swedishembedded/brain/commit/a3a4798009a2237ea6e17c32a8ba94c9f069bc6a))

- *(flux1)* The FLUX.1 / Kontext transformer forward, gated at two depths ([43ce71e1](https://github.com/swedishembedded/brain/commit/43ce71e16cefeda10e1840a01d00d3b137005d1a))

- *(restore)* CodeFormer face restoration - the transformer, the CFT and the w dial ([9ac7480e](https://github.com/swedishembedded/brain/commit/9ac7480e0043cc100e2db617bad0f698687fea84))

- *(data)* The CLIP BPE tokenizer, id-exact vs HF on both SDXL towers ([d22dd50d](https://github.com/swedishembedded/brain/commit/d22dd50d0ea5bf5261671aa74c5bc3e5267c6d29))

- *(diffusion)* The discrete DDIM/Euler/Euler-a/DPM++ schedulers, parity-gated ([cb0b83f6](https://github.com/swedishembedded/brain/commit/cb0b83f622c3c5d7633d5ea6fe1b6ff90fc26854))

- *(unet)* The SDXL UNet2DConditionModel forward, parity-gated per stage ([c7d97a70](https://github.com/swedishembedded/brain/commit/c7d97a708c585c1dc71502f001137c8773684c03))

- *(sam2,facenet)* The serving contract - capability, residency, D-Bus, examples ([ceb4bde0](https://github.com/swedishembedded/brain/commit/ceb4bde0f2adabfbd728d967291751868d5283cf))

- *(vqgan,restore)* The serving contract for the CodeFormer stack ([7fa170a7](https://github.com/swedishembedded/brain/commit/7fa170a7c98919f0f8a6a8ec6aa292cbbf3483d9))

- *(t5)* The T5-XXL encoder backward ([05980828](https://github.com/swedishembedded/brain/commit/05980828551d9004c13f3e4e879601126db58bb3))

- *(restore)* The CodeFormer code-transformer backward ([afcc61ac](https://github.com/swedishembedded/brain/commit/afcc61aceec621f5ab4c5ce5d18a79a4ca01718e))

- *(gradcheck)* Check_t5 and check_codeformer, and a per-ENTRY check for folded parameters ([4171bccb](https://github.com/swedishembedded/brain/commit/4171bccb03dbcaac0515cac4b09162f14d73117f))

- *(unet)* Consume control residuals, and migrate onto Gpu::write_f32 ([8f3e60bc](https://github.com/swedishembedded/brain/commit/8f3e60bc0614e2d25fee6af9ca8a48f30520a00e))

- *(controlnet)* The backbone-agnostic control seam and the SDXL ControlNet ([e76c839a](https://github.com/swedishembedded/brain/commit/e76c839a00b419f52855f58683ccceec32e3427e))

- *(flux1)* A device-side per-block injection seam for identity adapters ([ccd1a952](https://github.com/swedishembedded/brain/commit/ccd1a9528646bb5f3440b912ea3a532b75960c0e))

- *(pulid)* PuLID-FLUX identity conditioning on the FLUX.1 backbone ([60fcd5bf](https://github.com/swedishembedded/brain/commit/60fcd5bf401e9c1e212d8cdccf03fd69ebea2d95))

- *(clip)* The serving contract - capability, residency, genuine batching ([3c853a5e](https://github.com/swedishembedded/brain/commit/3c853a5ee4d937799061deabbb1a6f98e027da15))

- *(pulid)* Build id_cond from a photograph, with the reference's asymmetry ([6a9da997](https://github.com/swedishembedded/brain/commit/6a9da997d66e67835afc52a4bffb495bf2e45e89))

- *(instantid)* The released shapes, derived rather than assumed, plus goldens ([ba452d36](https://github.com/swedishembedded/brain/commit/ba452d36fc513c9488c98be5428c6f26e0888495))

- *(imgpipe)* The composed pipeline, with "change only X" as an exact contract ([f76d4397](https://github.com/swedishembedded/brain/commit/f76d43975fb78f2b0a8b3679225c7766b196d948))

- *(imgpipe)* Expose the pipeline as a capability - one Run, not four ([fad914bd](https://github.com/swedishembedded/brain/commit/fad914bd405180b5f690849193d69e78ad46bda5))

- *(instantid)* The Resampler forward, on the hoisted emitter, at cosine 1.0 ([0a70abf5](https://github.com/swedishembedded/brain/commit/0a70abf5f9cb421bca2fb6089eaebfc7a23f1f34))

- *(instantid)* The decoupled cross-attention, gated at both SDXL widths ([944b2c85](https://github.com/swedishembedded/brain/commit/944b2c8514ff5a42406737e47b9be83b65b31142))

- *(model,unet,instantid)* A cross-attention injection seam, and InstantID on it ([c85cc2e4](https://github.com/swedishembedded/brain/commit/c85cc2e4f39caefcfaa7f189da4cb70f74970d76))

- *(unet)* The SDXL text-to-image pipeline - brain generates an image ([0a61de02](https://github.com/swedishembedded/brain/commit/0a61de023e2f789f673c8f5c9d42a6b53e2a8220))

- *(upscale)* Real-ESRGAN RRDBNet - the imaging pipeline's upscale tail ([5926aaad](https://github.com/swedishembedded/brain/commit/5926aaad76571dbe2ac72f3e0019b48af1a19a1a))

- *(imgpipe)* The upscale tail - a size-changing stage, handled as one ([d58cd95f](https://github.com/swedishembedded/brain/commit/d58cd95f0f5bc5d7b55f2fc37246f5a53bc9a775))

- *(upscale)* A real tiling path, and the halo the released net actually needs ([af2a490c](https://github.com/swedishembedded/brain/commit/af2a490c487ac49bc4763a90f1095ebdcd71bd98))

- *(npu)* The Real-ESRGAN topology, gated for what can be checked without one ([911b8447](https://github.com/swedishembedded/brain/commit/911b84476bf77f75a4dd35460faa8384a64b1773))

- *(npu)* Facenet needs no topology - and SCRFD's blocker, named ([b9377db0](https://github.com/swedishembedded/brain/commit/b9377db0b7eeecc48d470cb136ea829d375bf4ea))

- *(examples)* The imaging pipeline end to end - and four examples the harness never saw ([03fd5947](https://github.com/swedishembedded/brain/commit/03fd59473079e5fb039a138dfce668c2b0a12511))

- *(npu)* GroupNorm for the conv-autoencoder topologies, checked NUMERICALLY ([35417add](https://github.com/swedishembedded/brain/commit/35417add6975db47c5940ae2e10ef039bc7c7ecb))

- *(npu)* The AutoencoderKL decoder topology - the VAE every latent model ends with ([8639d38a](https://github.com/swedishembedded/brain/commit/8639d38aa81b61b44e2345c9ab39e505371338f3))

- *(gpu-core)* Cost formulas for the conv2d and GroupNorm families ([6d5c9f56](https://github.com/swedishembedded/brain/commit/6d5c9f564c008f605cdc49f72ec74565147455ea))

- *(qwen,checkpoint)* Adapter-only LoRA save, and folding into a base for zero-overhead serving ([5cd2f936](https://github.com/swedishembedded/brain/commit/5cd2f936761064ce7c7db97051d1ab2b8316f1e9))

- *(data)* Multi-turn ChatSample with per-message loss masking, and a JSONL loader ([6cc28242](https://github.com/swedishembedded/brain/commit/6cc28242a5f1113682adef38c1380385f6f9821b))

- *(modelref,modelstore)* A model reference grammar and store layout for named LoRA adapters ([325d4419](https://github.com/swedishembedded/brain/commit/325d44199876cfc617a3c658f823e7890a730d02))

- *(data)* Generic chat-template rendering -- execute a checkpoint's OWN Jinja, not a per-model Rust port ([a0750f67](https://github.com/swedishembedded/brain/commit/a0750f67cef4bf87f26b7d24cb344d49e8d5db78))

- *(data)* Qwen3 chat-template renderer + streaming tool-call scanner ([47a2ab5b](https://github.com/swedishembedded/brain/commit/47a2ab5b4a3ad957d5a47c94bbfb48e06627f927))

- *(capability)* Additive Progress::event for structured streaming ([c4f98a66](https://github.com/swedishembedded/brain/commit/c4f98a6683af6cf33422d71af38b949033dd4b7d))

- *(qwen)* Decode-only serving build, multi-EOS stop, cached LM head ([ba5508f2](https://github.com/swedishembedded/brain/commit/ba5508f2b85ccf9d54c4d961989870064fb83808))

- *(apiserve)* OpenAI/OpenRouter tool calling, end to end ([007aa3ae](https://github.com/swedishembedded/brain/commit/007aa3ae2265aa721727a7c045ef5da8220e6cb0))

- *(data)* ChatSample renders through the checkpoint's real chat template, not an approximation ([ffaa05fa](https://github.com/swedishembedded/brain/commit/ffaa05fa9582d15721cf8541c8327617bcabad0d))

- *(data)* ChatTemplate::from_model_dir -- load the real template with zero import-time wiring ([9d9b994e](https://github.com/swedishembedded/brain/commit/9d9b994e47620702d475db643537c46c3a80e968))

- *(agents,data)* A hard rule -- validate everything crossing into brain, structurally AND semantically ([7748e93a](https://github.com/swedishembedded/brain/commit/7748e93a78a10d4d045c26e04c7342ca0e488a33))

- *(qwen,cli,modelstore)* Brain qwen finetune --lora trains a named adapter end to end ([3ced153d](https://github.com/swedishembedded/brain/commit/3ced153d7442ae4f15088c8080069863274bdd92))

- *(qwen,cli)* Brain qwen eval -- Gate B, held-out loss/accuracy for a real checkpoint + adapter ([52bdcae3](https://github.com/swedishembedded/brain/commit/52bdcae3e34dad0e1be70fcb0f5dca7675ca40f1))

- *(qwen,cli)* Serve a named LoRA adapter as its own catalog entry ([efc55564](https://github.com/swedishembedded/brain/commit/efc555644794bc3cb6916851b955e0b386c87f62))

- *(qwen,residency,apiserve,dbus)* Concurrent serving performance overhaul (W0-W8) ([0bf5ab03](https://github.com/swedishembedded/brain/commit/0bf5ab035fe1316507ef28a3e8167632431483ed))

- *(npu)* Parameterise the VAE topology by leaf name - vqgan reuses it, not a copy ([b90a4010](https://github.com/swedishembedded/brain/commit/b90a401082c43e42fbd489acca282b4dd36983bf))

- *(vae/blocks)* Multi-head, pre-fused qkv, and normed-residual attention ([6b25b9e9](https://github.com/swedishembedded/brain/commit/6b25b9e964e66703b15adec0b674620492a1eda7))

- *(cli)* Brain serve --help, and unknown flags are a hard error ([0fb75b4c](https://github.com/swedishembedded/brain/commit/0fb75b4ccdfbccfdd2a6f7d91dd9923baed65fd4))

- *(serve)* --ready-file, touched once every requested listener is bound ([5bf26079](https://github.com/swedishembedded/brain/commit/5bf26079a5281f6604638875dee263108ee47941))

- *(qwen)* Brain qwen calib -- per-(layer,kv-head) K/V outlier report ([c4aa5d12](https://github.com/swedishembedded/brain/commit/c4aa5d12127801b9e9d240435cfbc65667232469))

- *(qwen)* Calibrated int8 KV scales via a percentile clip ceiling ([28428516](https://github.com/swedishembedded/brain/commit/28428516e731e22eaf789d701ece7ea3cb1f3c1c))

- *(qwen)* Brain qwen eval scores through the paged engine, per KV dtype ([2c5cf6ea](https://github.com/swedishembedded/brain/commit/2c5cf6ea5ff54545fdb3098be1536d37f47511d8))

- *(gpu-core)* Measure the device's roofline instead of hardcoding a P40's ([73466fee](https://github.com/swedishembedded/brain/commit/73466fee05ec0712092b5cc58cf61d8091295d66))

- *(gpu-core)* One shared pass profiler; a partly covered pass reports no rate ([fc902a81](https://github.com/swedishembedded/brain/commit/fc902a817b6920743681d9c8c3b8f3be49d59a0e))

- *(gpu-core/cost)* Cover the whole VQGAN training step, and ratchet coverage ([c7c519db](https://github.com/swedishembedded/brain/commit/c7c519db4114eceeb506496580298aa20f8e0202))

- *(qwen)* Qwen_bench - the first per-kernel profiler for the LLM datapath ([2485ed08](https://github.com/swedishembedded/brain/commit/2485ed08bbb7464d3b3667d60f23c7c1490bdb7c))

- *(gpu-core)* Refuse rates above the device roof; cover the paged serving tape ([f3263ed8](https://github.com/swedishembedded/brain/commit/f3263ed813b28db1b5ea46d392920ffaf375d55e))

- *(gpu-core,backend-wgpu)* Time kernels inside the production single pass ([a8cadf8c](https://github.com/swedishembedded/brain/commit/a8cadf8c2eeebddb4c8b761f4d7dcf01d5c9cc5a))

- *(kernels)* Split-K forward GEMM for skinny-M shapes ([dcfd8bd7](https://github.com/swedishembedded/brain/commit/dcfd8bd70263b09a527953da160faafa04fe3e40))

- *(gpu-core)* Measure the int8 roof - it re-grades the int8 GEMM 4.15x worse ([b500a809](https://github.com/swedishembedded/brain/commit/b500a8094506a416c20966cc6a3230fc2676268a))

- *(qwen)* Kv_pool_bytes -- the paged KV pool's byte cost, measured not asserted ([3c292fbd](https://github.com/swedishembedded/brain/commit/3c292fbdc61140b67f98c2bf276c40ecf6790909))

- *(model,qwen,perf)* Kv_pool_bytes reaches the perf artifact's memory block ([3adb7ef7](https://github.com/swedishembedded/brain/commit/3adb7ef7aa93648d75fbf09383f290a59a90c402))

- *(qwen,cli)* Int8 paged KV is now the serving default (W3.5) ([6a797c49](https://github.com/swedishembedded/brain/commit/6a797c49d8a5efffd65fe5b49baab4e2ae089078))

- *(qwen,cli)* Wire KvCalib into serving, opt-in (W3.3 finishes W3.5) ([4d768075](https://github.com/swedishembedded/brain/commit/4d76807592dce5b031d86deea714f8ca3da9287b))

- *(cli)* Raise BRAIN_QWEN_CTX's default to 24576, guard the fp32 opt-out ([9f3cfce7](https://github.com/swedishembedded/brain/commit/9f3cfce774bee460e81140334ce181821c16832e))

- *(checkpoint)* Add save_carded, an additive card-carrying save sibling ([4c54e2b3](https://github.com/swedishembedded/brain/commit/4c54e2b30f13d25b534a0e0dc30ad8796b46c48e))

- *(gpt)* Attach a ModelCard to saved checkpoints ([d3ab5c61](https://github.com/swedishembedded/brain/commit/d3ab5c61ad9327b0fa11d3dc8f15f5819d479f46))

- *(glm)* Attach a ModelCard to saved checkpoints ([c3f93ee3](https://github.com/swedishembedded/brain/commit/c3f93ee36f5cfc5f1ea1ce3d2618a1588f9c20f4))

- *(yolo)* Wire into model-store auto-discovery ([71f19348](https://github.com/swedishembedded/brain/commit/71f19348339bd1dc9e6a9e84ea13ba8189b425af))

- *(depth)* Wire into model-store auto-discovery, fix a stale metadata field ([4b218abf](https://github.com/swedishembedded/brain/commit/4b218abfe0dab002d566c5ca04dcc092d5fa6dca))

- *(gpu-core,backend)* Measure the "2x resident" cost - it's wgpu's, not the hardware's ([a858441b](https://github.com/swedishembedded/brain/commit/a858441b34c70343d35addef90afef04c409d751))

- *(model,kernels)* Sparse top-k MoE forward - model::moe, no dense-eval waste ([0976631b](https://github.com/swedishembedded/brain/commit/0976631bf197e7a8922c6c18a3d40665caccbcc8))

- *(model,kernels)* Int8 tier for sparse MoE - expert_fwd_i8 ([d070bec7](https://github.com/swedishembedded/brain/commit/d070bec7b5397a423253ca1a53ddc10fbd375c81))

- *(omni)* New crate + config parser for Qwen3-Omni-30B-A3B ([78978647](https://github.com/swedishembedded/brain/commit/789786475c232b06aedf8e01f1c1668e243ae0df))

- *(omni)* HF -> brain tensor name mapper, validated against all 28010 real names ([dc06f479](https://github.com/swedishembedded/brain/commit/dc06f47924614ae0393f1b97b583815fcc0d61d3))

- *(checkpoint)* Int8-native StWriter -- create_mixed + write_u32 ([8da86aca](https://github.com/swedishembedded/brain/commit/8da86aca22d7282c743813e202df2788f8583214))

- *(omni)* Streaming int8-native import_as -- ~35GB on-disk checkpoint ([879e7e0f](https://github.com/swedishembedded/brain/commit/879e7e0f20daf8fe7d23c18a260473a38b907551))

- *(omni,qwen-asr)* Audio tower parity -- exact, zero new encoder code ([87f1a90f](https://github.com/swedishembedded/brain/commit/87f1a90fb2a021a19981ca39dc948fa4b778176e))

- *(omni,qwenvl)* Vision tower parity -- exact, image path ([a8cfdc20](https://github.com/swedishembedded/brain/commit/a8cfdc20183ea468bdad05b72dc1d83fde9396b5))

- *(omni)* Thinker decoder layer forward, exact real-weight parity (M6a) ([1f702164](https://github.com/swedishembedded/brain/commit/1f702164335ea6ce3f0429e5d58b699b23fc5e86))

- *(model)* Hoist rope2d_fwd -- table-driven M-RoPE, shared dispatch ([9646a5e1](https://github.com/swedishembedded/brain/commit/9646a5e1e9c9a19b655c5227d54bd6cb726dfdec))

- *(omni)* Real 3-axis M-RoPE + full decoder composition (M6b) ([9058da69](https://github.com/swedishembedded/brain/commit/9058da699feeee225d9e4308150b101697e0e174))

- *(model)* Shared_expert_fwd -- always-active dense SwiGLU for MoE blocks ([434fa560](https://github.com/swedishembedded/brain/commit/434fa560cb5b536783a915df7095faf7f41d10f1))

- *(omni)* Talker decoder layer, exact real-weight parity (M7a) ([72164fe5](https://github.com/swedishembedded/brain/commit/72164fe51a930b7691b8c1484bf89fb5d2d50d6a))

- *(tts)* Expose MtpModel::build_on for tests that bypass ParamStore I/O ([4e7550a1](https://github.com/swedishembedded/brain/commit/4e7550a1412678385ac2a1e3a695c3c7350c0e23))

- *(omni)* Code predictor validated against real weights (M7b) ([f0d970a6](https://github.com/swedishembedded/brain/commit/f0d970a6f10e7308357df24cb57a1721995d1f63))

- *(codec)* Decode_omni -- Qwen3-Omni Code2Wav vocoder support ([5fb2d2ce](https://github.com/swedishembedded/brain/commit/5fb2d2ced861cfdcc4f74099f02bb9216d486802))

- *(omni)* Code2Wav vocoder validated against real weights (M8) ([ca7a6b81](https://github.com/swedishembedded/brain/commit/ca7a6b814a48bed5c0376690b5d42b0d0f00fbac))

- *(omni)* Streaming greedy generation loop for the Thinker decoder ([48cc361e](https://github.com/swedishembedded/brain/commit/48cc361e214a6a6fe3c1845caf487f10834f5425))

- *(omni)* Caps.rs -- generate action, OmniProvider ([c97be54d](https://github.com/swedishembedded/brain/commit/c97be54dad10bc6945c2037015fbaf4e73a23d4f))

- *(cli)* OmniResident + CLI dispatch wiring (M9a) ([2bb98899](https://github.com/swedishembedded/brain/commit/2bb98899735e4ee48857399311ba57ae3a8d51e1))

- *(brain-py)* OpenAI/Anthropic HTTP clients + examples/omni.py (M13/M14) ([f6c6c683](https://github.com/swedishembedded/brain/commit/f6c6c683ec6f3a141a7bd85dfe7851d72c44d7bc))

- *(model,omni)* KV-cache decode -- O(cached length), not O(cached length)^2 ([4d61547a](https://github.com/swedishembedded/brain/commit/4d61547a30b1ffb19c27ed54a628c08253890ba6))

- *(qwenvl,omni)* Real audio/image input -- encoder splice + real M-RoPE ([5f840e4e](https://github.com/swedishembedded/brain/commit/5f840e4e3c1248f9cfc65e837ec01cb704f139c6))

- *(omni)* Wire audio/image blobs into the generate action ([fa5e7a8a](https://github.com/swedishembedded/brain/commit/fa5e7a8a380c73512fde0a7588ba63d9a2ea1262))

- *(brain-py,examples)* --in-speech/--in-image are real over D-Bus ([17e99d38](https://github.com/swedishembedded/brain/commit/17e99d3809ba7689aa0ca4c3b2e94ebd192699a5))

- *(omni)* Talker KV-cache decode -- foundation for the speech-output loop ([a9c25f7d](https://github.com/swedishembedded/brain/commit/a9c25f7d8a70e79a30c83c7b8a68e30f5e2b9ef2))

- *(omni)* Thinker->Talker prefill assembly (talker_prompt) ([32a4ab51](https://github.com/swedishembedded/brain/commit/32a4ab517104d76698f78e6588a69c9fee92349f))

- *(omni)* Talker codec-id generation loop (talker_generate) ([a1d2c582](https://github.com/swedishembedded/brain/commit/a1d2c582d49fb3cbcf6619a9208a5f80ae82d7ab))

- *(omni)* Wire speak -- real speech output as a served action ([39f0ee31](https://github.com/swedishembedded/brain/commit/39f0ee31da8dfaad2fc61ba38de2a59be4ab80c5))

- *(omni)* M15 -- NPU export wiring for audio tower, vision tower, Code2Wav ([4fecf949](https://github.com/swedishembedded/brain/commit/4fecf9494bdefb2d782cca943b5fc3fa5a1cde91))

- *(omni)* M16 -- omni_bench profiler for Thinker/Talker MoE layers + towers ([ac0670ef](https://github.com/swedishembedded/brain/commit/ac0670ef07c97db7d6f0a07b5e3dfbdaa2154180))

- *(backend-vulkan,vulkan)* Real device sharing + fix a real ERROR_DEVICE_LOST ([e4531d73](https://github.com/swedishembedded/brain/commit/e4531d73da10ba939e6cf4f138ab0bc61cf11f54))

- *(backend-wgpu)* Bound every GPU wait with a timeout, report device-lost ([c7a33d00](https://github.com/swedishembedded/brain/commit/c7a33d0030d6db88604d4077a3086be8a0d90b21))

- *(checkpoint)* U32 packed-tensor read-back + a real full-import coverage test ([d28130ca](https://github.com/swedishembedded/brain/commit/d28130ca3cc3f0a2fd7cb4e97de7b31f4ca6f553))

- *(model,kernels)* MoE sparse dispatch backward pass ([7fbbfce6](https://github.com/swedishembedded/brain/commit/7fbbfce60b76a092df08b753e7e9d570d02badc4))

- *(model,kernels,glm)* Row-compacted MoE forward, 7x faster than GLM's dense path ([5c816dfe](https://github.com/swedishembedded/brain/commit/5c816dfe1c10244ad0c3df93a101caca2f460e1a))

- *(residency)* Multi-GPU placement foundation ([cc3995e9](https://github.com/swedishembedded/brain/commit/cc3995e9ec549e7c6adc21cec0c54a4e71e4e8f8))

- *(omni)* Int8 dual-GPU resident Thinker -- weight store, forward branch, real generate() ([b08cd448](https://github.com/swedishembedded/brain/commit/b08cd448cce7b5140bab5786195fe8b1e8ce5d38))

- *(glm)* Row-compacted MoE inference forward, wired into sample::generate ([ce69227f](https://github.com/swedishembedded/brain/commit/ce69227fb970aa04fa8ff648263e947b2511ad5a))

- *(qwenvl,cli)* Full serving registration + M-RoPE/DeepStack incremental decode ([24203c8a](https://github.com/swedishembedded/brain/commit/24203c8a765621fad968a149aea21eca0037a32a))

- *(capability,omni,cli)* Real video input for omni generate ([79e97d47](https://github.com/swedishembedded/brain/commit/79e97d47929f4bfd2b2d4e280bc2251e0008a793))

- *(dbus)* Print how to connect braintop to the bus just created ([cc48fec9](https://github.com/swedishembedded/brain/commit/cc48fec9865500f35c4ac3925dd33d4396f8aa12))

- *(perf)* Add GPU device telemetry, wired into Env and every artifact ([4dce625d](https://github.com/swedishembedded/brain/commit/4dce625d4a5bb45e459a348f12a69756be9b10e3))

- *(perf)* Add unit-appropriate workload presets for non-token targets ([a2e5f20c](https://github.com/swedishembedded/brain/commit/a2e5f20c8c32dd0ac3da9a674a9ebcff924175bf))

- *(backend-vulkan)* Add per-kernel device timestamp support ([640d2fd2](https://github.com/swedishembedded/brain/commit/640d2fd2e051c085a25654e486e86c8e0176b23c))

- *(perf)* Add perf targets for 13 model families ([26cd4e1c](https://github.com/swedishembedded/brain/commit/26cd4e1c05411cb411bce78991dfef18b65ec4ae))

- *(stats)* Surface GPU frequency/throttle state in Accelerator.extra ([5499eb1e](https://github.com/swedishembedded/brain/commit/5499eb1eb38159fc57b9e3ed809c2d9137ffeaea))

- *(gpu-core)* Bound the roofline probe's per-dispatch wait ([dad2931a](https://github.com/swedishembedded/brain/commit/dad2931adadf27127315d6fd447b93d124a4261a))

- *(examples)* Add an OpenAI HTTP surface example client ([4808f43a](https://github.com/swedishembedded/brain/commit/4808f43a16487703f9b72de9abbeade0767b126a))

- *(zimage)* Auto-fetch Z-Image from Tongyi-MAI/Z-Image-Turbo ([c3d1bc25](https://github.com/swedishembedded/brain/commit/c3d1bc25bcb52aa1d4e5fc07c5c0ddd0f9552bea))

- *(yolo)* Auto-fetch YOLOv8 from Ultralytics/YOLOv8 via a pure-Rust .pt reader ([a2be3e10](https://github.com/swedishembedded/brain/commit/a2be3e10b0834d2c90a04a90495fd2ea2a4b7c85))

- *(npu)* Add read-only npu-diagnose script, fix OpenVINO symlink check and container-aware firmware guidance ([0655273a](https://github.com/swedishembedded/brain/commit/0655273a71bf78b6fa749e239b0ff2736550f426))

- *(glm)* Persistent compact-MoE scratch, no more per-call GPU allocation ([f9d414be](https://github.com/swedishembedded/brain/commit/f9d414be55874426c68a4e9c506dc5b042a0ef2e))

- *(residency,omni,cli)* Real multi-device dispatch through the Executor ([76f3be75](https://github.com/swedishembedded/brain/commit/76f3be75ea87287ee7e600008e541efb23fbc9e4))

- *(qwen35)* Scaffold brain-qwen35 crate with hybrid decoder config ([ce695453](https://github.com/swedishembedded/brain/commit/ce69545377f4ade3691c1deae62fe7c752eeddf8))

- *(kernels,model)* Add rope2d_partial for partial-rotary M-RoPE ([adfecdd7](https://github.com/swedishembedded/brain/commit/adfecdd721f117435408b717e3692677f46580ba))

- *(checkpoint)* Implement TensorSource for MmapGguf ([991c0f7f](https://github.com/swedishembedded/brain/commit/991c0f7f3b27d510f17504a9b71a20c10cb322f3))

- *(qwen35)* GGUF import with empirically-derived llama.cpp tensor mapping ([fc178997](https://github.com/swedishembedded/brain/commit/fc1789972b832f53a835427e0027042f6adf35ce))

- *(model,kernels)* Add INT4 (q4) weight quantization at the shared level ([f43daefc](https://github.com/swedishembedded/brain/commit/f43daefc3e28dea32493a89ea017793aa595896d))

- *(model,kernels)* Gated DeltaNet chunked-recurrence forward pass ([48185ddd](https://github.com/swedishembedded/brain/commit/48185ddd80e00d4988e10d37f3cafad35e278418))

- *(qwen35)* Assemble the hybrid decoder forward pass (text-only) ([76fee3ac](https://github.com/swedishembedded/brain/commit/76fee3ace8cc14ff3f391d3d92bb1a628873bbbf))

- *(qwen35)* Add INT8 (DP4A) quantized inference path ([f3f94979](https://github.com/swedishembedded/brain/commit/f3f94979dad18c4ac74cc4328cb4c744e394e3db))

- *(model,kernels)* Gated DeltaNet backward pass (full reverse-mode) ([d8aadf6d](https://github.com/swedishembedded/brain/commit/d8aadf6d9df5c3878ae0c0963a5268e2a1b17285))

- *(checkpoint)* Add zero-copy and bounded-chunk accessors to TensorSource ([5fc29fb6](https://github.com/swedishembedded/brain/commit/5fc29fb66168b4e883f6f45c9425547807f863e5))

- *(paramstore)* Bound weight-upload host scratch to one chunk, not one tensor ([04f85744](https://github.com/swedishembedded/brain/commit/04f85744fb68992247a4da2e69e6b9a87dee1ba3))

- *(memauth)* Add the process-wide memory authority, fix unified-memory double-counting ([898c84c7](https://github.com/swedishembedded/brain/commit/898c84c746504b7ae4dc942c70ab802b30eb2bbe))

- *(checkpoint)* Add RemapSource, a streaming rename/reslice TensorSource ([dfaa7bef](https://github.com/swedishembedded/brain/commit/dfaa7bef7eb20b0034adbc1735cf3ec984d04998))

- *(weightset)* Within-instance weight window (Bélády CyclicScan, Lru, AllResident) ([6bb31918](https://github.com/swedishembedded/brain/commit/6bb3191810ae4a22c22815f16f08f82f9c5f752d))

- *(residency)* Instance::demote/promote + MemCost.mapped contract for Tier::Warm/Cold ([76b16510](https://github.com/swedishembedded/brain/commit/76b16510ebc40ed2f7d10501db975c4e8c53cd1b))

- *(zimage)* Fp32 single-GPU DiT via a weight window (ZImageDitWindowed) ([b5f67beb](https://github.com/swedishembedded/brain/commit/b5f67beb55662159cad56fc3e7f3d69e19fb4e1f))

- *(residency)* Wire ResidencyManager's eviction/claim to Instance::demote/promote ([021ad251](https://github.com/swedishembedded/brain/commit/021ad2515643d953850b2a85d8924ff40bacf81c))

- *(zimage,residency)* Real demote/promote for the int8 DiT (host weight cache) ([d9311c6d](https://github.com/swedishembedded/brain/commit/d9311c6d08e5a182c07a5d572c1aa83541476d16))

- *(perf)* A real "weights" scenario -- CyclicScan vs Lru vs AllResident ([54501750](https://github.com/swedishembedded/brain/commit/54501750177c0957ac1e835714620304b6b15c13))

- *(residency,cli)* --verbose/-v <0-3> logging, model residency lifecycle visibility ([24deee3c](https://github.com/swedishembedded/brain/commit/24deee3c6f131db1e5f30f1962b6b70645d4f5bc))

- *(cli,apiserve)* Info-level visibility into download/load/request/token lifecycle ([9a16bc80](https://github.com/swedishembedded/brain/commit/9a16bc80efca43be8e7997324c199806d5e7bb5c))

- *(omni)* Converse -- real audio/image/video input fused with real speech output ([62ad38f9](https://github.com/swedishembedded/brain/commit/62ad38f9d07790c6e6fed90a2a05b226f6f16af9))

- *(omni,imaging,qwenvl,capability)* Real video file decoding + real temporal patching ([691049a5](https://github.com/swedishembedded/brain/commit/691049a5a2928362e57b1a9193e6df7cf3c4f0f8))

- *(codec)* StreamConvTr1dSym -- the symmetric-crop streaming transposed-conv primitive Omni's Code2Wav chunking needs ([08370d25](https://github.com/swedishembedded/brain/commit/08370d2547f517deef85ff4076fd86df866a599e))

- *(codec)* Codec::decode_omni_chunked -- real chunked Code2Wav decode, bit-exact vs decode_omni ([b36827d5](https://github.com/swedishembedded/brain/commit/b36827d5e21fe32aac79786f4a5e27cd001eed62))

- *(omni)* Generate_codes_streaming -- per-frame callback for overlapping code generation with vocoding ([149ca746](https://github.com/swedishembedded/brain/commit/149ca746638f46abb4428fa4bed8c8da6059bf7f))

- *(omni,dbus,capability)* Speak/converse stream real audio to a real client -- Gap 3 closed ([8a5424b4](https://github.com/swedishembedded/brain/commit/8a5424b4e6320aaf8184d6f3cd2227fe94eed5a9))

- *(model)* Shared_expert_fwd_i8 -- int8 for MoE's always-active shared expert ([33b93915](https://github.com/swedishembedded/brain/commit/33b939159a0f682c81005dad82fde1fc87c6f9b9))

- *(omni)* Int8 for Talker's routed AND shared experts ([94758286](https://github.com/swedishembedded/brain/commit/9475828608617871dd35a063474d270e947bdf96))

- *(omni)* Int8 for the Thinker's lm_head projection ([55c53f43](https://github.com/swedishembedded/brain/commit/55c53f439bb3e9ea983404e644df8b8cf0231d28))

- *(omni)* Wire lm_head_fwd_i8 into the Thinker's actual serving path ([b2a273f0](https://github.com/swedishembedded/brain/commit/b2a273f080a578e5231dad52ae434f05708afcb4))

- *(gguf)* Add crates/gguf -- generic architecture-dispatched GGUF loader ([831bf150](https://github.com/swedishembedded/brain/commit/831bf1501096d65e93723ee28ac436cc03ebbcc9))

- *(model,kernels)* SAM decomposed relative-position bias + zero-pad windowing ([8295a9a1](https://github.com/swedishembedded/brain/commit/8295a9a100639062426f6122c7abb6e65b70bca5))

- *(model,moe,imaging)* Core hoists for DeepSeek-OCR (rename, router, backward, tiling) ([e64a495a](https://github.com/swedishembedded/brain/commit/e64a495a739e3d9069ae572f2f87639f85b646e2))

- *(sam1)* Add crates/sam1 -- SAM-1 ViT-B tower for DeepSeek-OCR ([8c773951](https://github.com/swedishembedded/brain/commit/8c7739511f7a9eeb4c477ec89850c9d2bbf88428))

- *(clip)* Add ClipVision -- vanilla CLIP-L/14 tower beside EvaVision ([f55bbaa1](https://github.com/swedishembedded/brain/commit/f55bbaa15c47fe930f0c45f0287643f093d62554))

- *(deepseekv2)* Add crates/deepseekv2 -- MHA-only DeepSeek-V2 decoder ([dd092ec8](https://github.com/swedishembedded/brain/commit/dd092ec89d1e37e4c45a98ea0eee385f458d3753))

- *(deepseekv2)* Wire LoRA forward/backward into the decoder (GREEN) ([250d340a](https://github.com/swedishembedded/brain/commit/250d340a5b7dc1d332d055b4ac927b5cb772b0ac))

- *(deepseekv2)* Incremental KV-cache decode (O(T) per token, not O(T^2)) ([32717e81](https://github.com/swedishembedded/brain/commit/32717e81d07e261c964efe6df5117f87ebb4a10a))

- *(deepseekocr)* Wire the serving path onto KV-cached decode ([3edcc75d](https://github.com/swedishembedded/brain/commit/3edcc75d2240d0d0ea118d975068dcccded0cb49))

- *(backend-cpu)* AVX2+AVX-512 fast paths for moe_linear_gated and GQA self-attention ([d77f4b86](https://github.com/swedishembedded/brain/commit/d77f4b866dd8b2612d6c1db27b51edeb3f2d3ee6))

- *(omni)* Wire real image/audio/video input into the int8 GPU-resident Thinker ([883800f4](https://github.com/swedishembedded/brain/commit/883800f40c356bb4841fbe6ba8e3b558459099c3))

- *(omni)* Real Jinja chat templates on both Thinker paths; rename int8 model to W8A16 ([d6c9a562](https://github.com/swedishembedded/brain/commit/d6c9a5624a686b3ab4b5c6ece207952dcb4b608a))

- *(backend-api)* Unify DType, make kernel capability gating data (B1) ([6dd418ce](https://github.com/swedishembedded/brain/commit/6dd418ce7ef2a01a9bb5bfd67bf0850057b53469))

- *(dtype)* B11 - native f16 compute, capability-gated and measured ([6b838517](https://github.com/swedishembedded/brain/commit/6b838517f77d1e768162097a9caaa120badad114))

- *(arch)* Add crates/arch, the canonical model-architecture registry ([e2879ae1](https://github.com/swedishembedded/brain/commit/e2879ae1418902187d28299d218a0f061ccf457b))

- *(arch)* Rewire modelstore + gguf_import onto the arch registry ([88010a6a](https://github.com/swedishembedded/brain/commit/88010a6aae286c0a0eb20cb1b1e22e8c5cf3b8bf))

- *(cli)* Unify the CLI into one architecture-namespace resolver ([4c1c9c26](https://github.com/swedishembedded/brain/commit/4c1c9c26ea899e15909bcbe42a3597a3df748904))

- *(cli)* Auto-fetch default weights for infer with no --weights ([b1198c29](https://github.com/swedishembedded/brain/commit/b1198c292b41435c362a7d9a172883ce301a87e4))

- *(gates)* Add check-arch-names.sh, wired into make check/scripts ([3df1cfee](https://github.com/swedishembedded/brain/commit/3df1cfee9c50869327fe2eccb66b22b08fed958d))

- *(rl)* Mirror sven's ATIF trajectory crate as crates/atif (self-improve P1) ([2c794ad3](https://github.com/swedishembedded/brain/commit/2c794ad30bf7061ea13bb01e33f7d22e37956788))

- *(model,qwen3,gradcheck)* Generic weighted-loss Batch contract (self-improve P2) ([2fd973ea](https://github.com/swedishembedded/brain/commit/2fd973eab98b92250372c6b6c5ecbd9563d95ddd))

- *(rl)* Generic weighted-training driver + ATIF ingestion (self-improve P3/P5) ([113db259](https://github.com/swedishembedded/brain/commit/113db2596c61931f3600a22d5262f661065be6cf))

- *(cli)* Continuous-training hot-swap cycle glue (self-improve P6a) ([9a1b8058](https://github.com/swedishembedded/brain/commit/9a1b805814b6d69606e20c22fae7ad56ff401aba))

- *(docs)* A real, end-to-end validated Quick start harness ([e1449e81](https://github.com/swedishembedded/brain/commit/e1449e81639ff6313674223f5c2bdd5e9e2914d9))

- *(docs)* Add monocular depth (zipdepth) to the Quick start; fix docs/models/zipdepth.md's wrong CLI verb ([13aed6a7](https://github.com/swedishembedded/brain/commit/13aed6a7f2ee226b05c808742ee5ec59146644ae))

- *(docs)* Add lfm2 embedding, qwen3asr, and deepseek2ocr to the Quick start ([cb4622d7](https://github.com/swedishembedded/brain/commit/cb4622d7245086da18f27100f890fbe9d032f1f2))

- *(wan)* Register the Wan video architecture and its variant configs ([e17639b7](https://github.com/swedishembedded/brain/commit/e17639b7e0f5aab4c18d070b85dd7106a77ea304))

- *(kernels)* Causal conv3d forward and backward ([ab39cf73](https://github.com/swedishembedded/brain/commit/ab39cf735bde963bfa7d80d95779d086acc5702a))

- *(diffusion)* Flow-matching UniPC and DPM++ solvers for Wan ([303c4d4f](https://github.com/swedishembedded/brain/commit/303c4d4ff1a4657705d55465b68b96a0b8264dc3))

- *(gates)* Ban merge commits on main ([ff8d2f76](https://github.com/swedishembedded/brain/commit/ff8d2f76f70899790dbee1c3d0ca11c34fe89db5))

- *(build)* Full-parity Debian packaging ([54136ad8](https://github.com/swedishembedded/brain/commit/54136ad8cf3ede6285ea543a042b504b10217b5a))

- *(release)* Bump2version + git-cliff release automation ([fbaa26c2](https://github.com/swedishembedded/brain/commit/fbaa26c2524373d5147f2d956b8a2e98360a0098))


### Miscellaneous

- Fix the two standing test warnings (unused const, needless mut) ([dae76753](https://github.com/swedishembedded/brain/commit/dae76753b566f3ff97fa50838554ad2021b6e962))

- Cargo.lock for brain-qwen-asr npu dev-dependency ([442a78fd](https://github.com/swedishembedded/brain/commit/442a78fde5b8e2d66c1a22f61f9841527096476b))

- Untrack generated report artifacts under out/ ([25ab4e70](https://github.com/swedishembedded/brain/commit/25ab4e70db35d549bcac5246773618614702e18b))

- Skeletons for the four phase-1..3b imaging crates ([78a105e3](https://github.com/swedishembedded/brain/commit/78a105e3e125054155ec55955c84afbc55de9cd7))

- Rename brain checkpoint extension .weights -> .safetensors (P1b) ([d4091383](https://github.com/swedishembedded/brain/commit/d409138316cd93526d7b563d6dda801380168e00))

- Stop committing bench result example artifacts ([9f768e8d](https://github.com/swedishembedded/brain/commit/9f768e8d25c1ef4a4f6ff087e395a7ea6d34120e))

- *(scripts,tools)* Delete dead scripts, repair broken paths, add check/scripts gate ([bb66e9e8](https://github.com/swedishembedded/brain/commit/bb66e9e839cc99cffa5cdd76a700016def16f07c))

- Skeletons for the phase 3/4 crates - t5, flux1, restore ([00e946c9](https://github.com/swedishembedded/brain/commit/00e946c917f2cff302be22719001a8a78c1da03b))

- Skeletons for crates/unet and crates/controlnet ([085445a7](https://github.com/swedishembedded/brain/commit/085445a7e0f7db4feee737e8f118f572ab57672d))

- Skeleton for crates/pulid ([1942b254](https://github.com/swedishembedded/brain/commit/1942b254a98e8863aec388a0eda204b1437dc687))

- Skeletons for crates/instantid and crates/imgpipe ([62a5ccd3](https://github.com/swedishembedded/brain/commit/62a5ccd3f1da3defaa09124c84249a3cb732ae9b))

- Land the imaging tools in upstream's scripts/ and tools/ tree ([8a45b038](https://github.com/swedishembedded/brain/commit/8a45b038e5198a1e9e47f1141325311187cb85a5))

- Integrate origin/main - seed the new kernel's catalogue block and cost it ([2da986e4](https://github.com/swedishembedded/brain/commit/2da986e43d0934e02b402f0a9dfe658f96def7df))

- *(qwen)* Re-capture the serving perf baseline for int8 KV (W3.5) ([0f163b76](https://github.com/swedishembedded/brain/commit/0f163b76b7e232861c9bb004460da56295df51ae))

- *(omni)* Fix clippy warnings in audio/vision parity tests ([fc7e90cd](https://github.com/swedishembedded/brain/commit/fc7e90cdecf7ae5b95fbb328fd8fe65ae278ff0f))

- *(cli)* Remove the old brain fetch command, superseded by auto-fetch ([86759c50](https://github.com/swedishembedded/brain/commit/86759c50e69677652d174ae9b392589b0be5bd24))

- *(workspace)* Omni/upscale/wgsl-cpu into default-members; unify wgpu/pollster/image/ureq/zbus/sha2 pins; fix memauth honesty + race ([0fbb2b48](https://github.com/swedishembedded/brain/commit/0fbb2b48c3415f45e857d9dfd99090320a7ce018))

- *(goldens)* No machine-specific absolute paths; Makefile help/.PHONY parity; real gradcheck gate ([8344f00e](https://github.com/swedishembedded/brain/commit/8344f00e71b6fb7589debf8e27fc714a7230c213))

- Cargo.lock update for the fastvlm/qwenvl brain-imaging dependency ([b51c89be](https://github.com/swedishembedded/brain/commit/b51c89bef3ab55a2aa018da0b2a10564f646f9b7))

- *(kernels)* Delete the unclipped paged_kv_append_i8_batched twin - one i8 append kernel ([eddf52ce](https://github.com/swedishembedded/brain/commit/eddf52cedab659b93f058a8d7b061baf54e20c1d))

- *(wm)* Diamond parity dump records upstream repo identity, not a machine-local path ([2bd8744c](https://github.com/swedishembedded/brain/commit/2bd8744cf09d7ef94306d484b36590b3d7d31f52))

- Remove stale moe-rs naming, brain's predecessor project ([8d19f647](https://github.com/swedishembedded/brain/commit/8d19f647ea14a0e5086ab4bf2a839ce568a19197))


### Other

- *(bench)* Head-to-head YOLOv8n inference benchmark (brain vs Ultralytics) ([66938be6](https://github.com/swedishembedded/brain/commit/66938be6385286c4f6cd0fa89d16c2dffd479f91))

- Unify device selection under one DEVICE env (both sides) ([1345fc1a](https://github.com/swedishembedded/brain/commit/1345fc1a1b663c3cbfdc8c3bd215de88ca050e31))

- Torch GPU fallback chain (cuda -> vulkan -> mps -> cpu) ([092ce0d0](https://github.com/swedishembedded/brain/commit/092ce0d082cf2a831517beff799b835cd580f870))

- Surface engine per-stage timing under BRAIN_PROFILE ([d2db9d92](https://github.com/swedishembedded/brain/commit/d2db9d9239525e9d2c05674e89dc1e8f472da33a))

- Update yolo inference benchmark ([749864b6](https://github.com/swedishembedded/brain/commit/749864b62326dfa6da911f5e57e4b4c67bba1551))

- Widen token ids u16 -> u32 for large vocabularies ([5605f369](https://github.com/swedishembedded/brain/commit/5605f369a5c7b84a19bf6ad868846d82282d7994))

- Qwen byte-level BPE tokenizer (tokenizer.json) ([d9daea7c](https://github.com/swedishembedded/brain/commit/d9daea7c2d0f6fb1f645153d868c19108828a9bc))

- Parameter roles (Trainable/Frozen) ([ee94ccc7](https://github.com/swedishembedded/brain/commit/ee94ccc79c8e8f7eda92d12ba90204115a79c1de))

- Qwen decoder WGSL kernels (GQA, RoPE-base, axpy) ([3bb6a8f7](https://github.com/swedishembedded/brain/commit/3bb6a8f77616349548425f9396c2100f40dbb87f))

- Minimal safetensors reader (bf16/f16/f32 -> f32) ([e6eec22b](https://github.com/swedishembedded/brain/commit/e6eec22be4e100d5e016d2f1af24d8ea978b1016))

- Qwen3 dense decoder crate (forward + backprop + import) ([2a79b9ec](https://github.com/swedishembedded/brain/commit/2a79b9ecbf5f272cf2b80f6b03154edc719e0736))

- Finite-difference checks for Qwen (full + LoRA) ([be9cd352](https://github.com/swedishembedded/brain/commit/be9cd3527300fce6726cbd23715efc7a4d117bb5))

- Brain qwen {import,infer,train,finetune} ([e263ab14](https://github.com/swedishembedded/brain/commit/e263ab144571b96e2008ef565d4657582c93ec92))

- Qwen as a DecoderLm arch + learnability test ([bab76ca3](https://github.com/swedishembedded/brain/commit/bab76ca3305a1b5ac0120a4fbf95c5fd41f535ee))

- Tile Qwen embedding/lm_head over vocab to fit binding limit ([be3bff1d](https://github.com/swedishembedded/brain/commit/be3bff1db89a2dadd3636e109ccbf588e980fcb6))

- External-data serialization + int64 graph inputs ([a3dbaeb4](https://github.com/swedishembedded/brain/commit/a3dbaeb4762481e8134c9efa360faa74890302e6))

- Qwen3 ONNX decoder export + OpenVINO decode (CPU/GPU/NPU) ([7169faa2](https://github.com/swedishembedded/brain/commit/7169faa2f97aeb12e3b9f620bb11440f4800246e))

- Route brain qwen to NPU + add qwen export ([43c23681](https://github.com/swedishembedded/brain/commit/43c23681cc6fbea5bb8401e936ea354a5b12f1f2))

- Brain-codebase QA recall test (Qwen) ([898937b1](https://github.com/swedishembedded/brain/commit/898937b1c4d8dd4f9f9d2c7745d4a0ad25d29ec0))

- Shared transformer block-builders; refactor Qwen onto them ([42bb161e](https://github.com/swedishembedded/brain/commit/42bb161e11bf8e3c24d97dac684cc634b7b4749d))

- Report load/gen timing from infer (npu + cpu/gpu) ([d851931d](https://github.com/swedishembedded/brain/commit/d851931d64f03eb2d2254a623b6681df503b41d3))

- Bench_qwen_inference.py - CPU/GPU/NPU comparison ([088f5ad8](https://github.com/swedishembedded/brain/commit/088f5ad877c7fb6e73678b1eaed01b09e2ec716c))

- Cache the exported ONNX + compiled blob (precompile) ([94af7fff](https://github.com/swedishembedded/brain/commit/94af7fffb85f09516e29e40444e1257feebd9ab8))

- Add 'requirements' target + requirements.txt ([29390744](https://github.com/swedishembedded/brain/commit/293907446f75e018f41d9b33a68435a0a490e156))

- Auto-discover the OpenVINO pip wheel (no env dance) ([8ff85663](https://github.com/swedishembedded/brain/commit/8ff85663041c1fd5e102d537a59fd10837e9cce0))

- Print export/compile progress so a cold run isn't silent ([97633845](https://github.com/swedishembedded/brain/commit/97633845101713f35f61ce332463cf5926bf01e9))

- *(phase1)* 1D conv/transposed-conv + activation kernels + audio crate ([538615b1](https://github.com/swedishembedded/brain/commit/538615b1d9444123461560626a13effa7bdfda34))

- *(phase2/4)* Snake_beta+scale_chan kernels, codec+tts crate scaffolds ([513ea77f](https://github.com/swedishembedded/brain/commit/513ea77f26fd221dd876e453b02042cf48a65de6))

- *(phase3)* ECAPA speaker-encoder crate scaffold + config ([c83a954e](https://github.com/swedishembedded/brain/commit/c83a954e8462b2e9bca5face4c382340d4725b98))

- *(phase2,3)* Codec decode + ECAPA speaker encoder, parity-verified ([26faef78](https://github.com/swedishembedded/brain/commit/26faef787b2ec7d600ffb68e629427247062b6fc))

- *(phase4)* Talker + MTP code predictor, logit-parity verified ([0e4c3914](https://github.com/swedishembedded/brain/commit/0e4c39148e69149c9a7cbce871862c56338db245))

- *(phase5)* End-to-end voice-clone/synth pipeline + brain tts CLI ([79f30e24](https://github.com/swedishembedded/brain/commit/79f30e24c83a16e673e87fff36541b62f66a4a10))

- *(phase8)* Eval metrics + streaming serving + docs ([fa912e1b](https://github.com/swedishembedded/brain/commit/fa912e1b0b93bb78a6cc33fb16cc6d6e79967833))

- *(phase6)* KV-cache incremental decode + ONNX/NPU export ([e3105ae0](https://github.com/swedishembedded/brain/commit/e3105ae07132a8acafd77549ac608ebf9d7ed222))

- *(phase7)* Codec encoder + training/SFT + cached-path CLI wiring ([5a134fab](https://github.com/swedishembedded/brain/commit/5a134fab0dbddc4a752dead12eb53b6c61958a58))

- Fix GPU buffer usage (GL backend), wire cached path + NPU codec export ([93c94973](https://github.com/swedishembedded/brain/commit/93c94973f37c71eb237c333fa644be1772ae2d09))

- Add encode_wav example (wav -> [T,16] codes dump) ([09dae784](https://github.com/swedishembedded/brain/commit/09dae7847dddbd805f920e8ce7c41280ab4d2d8a))

- *(phase10)* Rayon-parallelize the cached CpuTalker matvec/head ([520bc6aa](https://github.com/swedishembedded/brain/commit/520bc6aa502e30a9cbb1c5d79bd929539406abb0))

- *(phase10)* Cached CpuMtp + codec host-loop parallelization ([1579a999](https://github.com/swedishembedded/brain/commit/1579a99955dde2b68aadbcf3bf4e0673507c46a7))

- Log the adapter line once per process, not per engine instance ([84c9595b](https://github.com/swedishembedded/brain/commit/84c9595b8a5247ace65980e063645481771f9632))

- *(wip)* Cached-vs-cachefree diagnostic + decode/stage profiling timers ([d4b3d880](https://github.com/swedishembedded/brain/commit/d4b3d880a9e1a534583b201b2b5955ca6d2f933f))

- Add input-embedding Talker hidden-state ONNX graph ([1aee31a3](https://github.com/swedishembedded/brain/commit/1aee31a3a2139b6633c499a6266a36a0ec694632))

- Add EmbedSession + CodecSession OpenVINO runtimes ([5df0b653](https://github.com/swedishembedded/brain/commit/5df0b65349671028d00f13f442686d725180bdec))

- Decouple prompt assembly via TalkerHost/MtpHost traits ([7007d3a2](https://github.com/swedishembedded/brain/commit/7007d3a201d048ac33b25fcc40493bdf877221be))

- Run Talker+MTP+codec on the Intel NPU (--device npu) ([4c6d169c](https://github.com/swedishembedded/brain/commit/4c6d169cff9bd96863889184fd00b2bb9d86c71b))

- AVX2/FMA SIMD dot in CPU matvec + attention ([864f6c03](https://github.com/swedishembedded/brain/commit/864f6c03307e25eeae5d772947e3a8bdafafd4ac))

- Add first-class native Vulkan compute backend ([fffeba44](https://github.com/swedishembedded/brain/commit/fffeba448d76d59574fd9fe8f4a78fc3a930bd65))

- Support the 1.7B MTP (separate embedding/decoder width + projection) ([6a795714](https://github.com/swedishembedded/brain/commit/6a795714288a271c73fd6c8028b72af08d337ea8))

- Weight-only INT8 Talker hidden graph ([091e91ad](https://github.com/swedishembedded/brain/commit/091e91ad8a19335e8e7481f3aa8e2267c753cfc3))

- VoiceDesign/CustomVoice instruct synthesis + fast NPU placement ([33ad3642](https://github.com/swedishembedded/brain/commit/33ad364282ce34a49fd9b2d4cd5e0b92af9d7899))

- Per-stage timing for the NPU generation path (TTS_PROFILE) ([4e212c20](https://github.com/swedishembedded/brain/commit/4e212c2018327cf18392c2dba4025ab0cb4b694c))

- Resident KV-cache Talker graphs (decode + prefill) + sessions ([e383aa82](https://github.com/swedishembedded/brain/commit/e383aa829e5acf530f4c08fc3d791c772ba74e7d))

- KV-cache resident Talker generation (default for NPU) ([45e3e654](https://github.com/swedishembedded/brain/commit/45e3e6547cd65bddf699d5273438b09904cfc3f0))

- MTP-on-NPU (resident KV-cache decode) - correct but opt-in ([13515774](https://github.com/swedishembedded/brain/commit/1351577423dec60d06a4fb67f25c29908cce823a))

- Resident TtsEngine for server mode (load graphs once, stream PCM) ([d6513eab](https://github.com/swedishembedded/brain/commit/d6513eab78050593bd409de01e9146e50d30e051))

- `brain tts serve` - event-driven resident TTS server ([e4237bb3](https://github.com/swedishembedded/brain/commit/e4237bb331de0884afcfc3ed2eb50c1ad46d9c82))

- Streaming codec - emit audio per chunk while generating ([a18fef6c](https://github.com/swedishembedded/brain/commit/a18fef6c7679231fd450f961216a47a92f9eae77))

- Python TTS clients for `brain tts serve` ([e275b525](https://github.com/swedishembedded/brain/commit/e275b525734f885be7d4b81a3520ba55c0780eb4))

- Exact streaming-conv state primitives (stateful decoder core) ([45535b12](https://github.com/swedishembedded/brain/commit/45535b1298bccf66879aaba72d03dcc7e1cc51e1))

- Full stateful streaming decoder + wire into the TTS engine ([d05d34c5](https://github.com/swedishembedded/brain/commit/d05d34c559c2fe09416cd7af254ca8c82ecf696c))

- Parallelize streaming conv primitives (rayon) + decode bench ([c6d118a8](https://github.com/swedishembedded/brain/commit/c6d118a8e8976ae9f12ed1666ef374fa29095472))

- Stateful streaming codec graphs (front + streaming-back) ([7b0c990d](https://github.com/swedishembedded/brain/commit/7b0c990db4a3a10b9d49d2df9fe8369adc8dbf5b))

- Fix slow design-engine load + server robustness ([1f9cb3a4](https://github.com/swedishembedded/brain/commit/1f9cb3a4ed4e2a46ebe8750759797d522615997c))

- BackStreamSession for the streaming-back codec graph ([be97ac29](https://github.com/swedishembedded/brain/commit/be97ac2936842b75766a33aeb08879fdccd14a9c))

- NPU streaming codec host loop + engine wiring (npu-stream) ([1a88d271](https://github.com/swedishembedded/brain/commit/1a88d271bd7d0b8ea81ed4657b4ab3a08c636e1e))

- Validate NPU streaming codec vs CPU reference (9.8e-6) ([66e1935f](https://github.com/swedishembedded/brain/commit/66e1935f95dd2f1a8eec743844e3edd2d9f746b1))

- Default the server codec to the NPU stateful streaming decoder ([8da7d267](https://github.com/swedishembedded/brain/commit/8da7d267e560fee6351f0ef8de1961ea7d3baf2d))

- BRAIN_TILE_BUDGET_WORDS override to reproduce/test vocab tiling ([13b6baad](https://github.com/swedishembedded/brain/commit/13b6baad378aa3a1887ab1fbf3abe102099263af))

- Fix flaky tiled (sliced-binding) corruption on Intel ANV (#12) ([01e047d2](https://github.com/swedishembedded/brain/commit/01e047d24f25190d6f0b2f656fddf10ff5f39298))

- Device-local storage + staging buffers (fixes #20) ([fae96442](https://github.com/swedishembedded/brain/commit/fae964421739c179aea5fec6f383a064c0075a4d))

- Unified cross-backend parity gate (#10) ([bd086fba](https://github.com/swedishembedded/brain/commit/bd086fba341d224394f4c60ca7ea773cf00a5530))

- MTP on NPU by default for 1.7B + cb0 head rayon (real-profiled wins) ([1994d0ba](https://github.com/swedishembedded/brain/commit/1994d0ba6bbb7478471544b6f9160943e1a45b36))

- INT4 weight-only Talker (opt-in) - i4 ONNX + NPU decode path ([ab63c0ee](https://github.com/swedishembedded/brain/commit/ab63c0ee4f49c313e26f61f9cb1ab73fc0503c48))

- Serve --talker-quant, startup path print, HW capability query, spk-sim ([56822c6d](https://github.com/swedishembedded/brain/commit/56822c6d545912b0ea25f08af3825271238ecc64))

- Fused single-infer MTP graph (EXPERIMENTAL - compiles+runs, has a correctness bug) ([27fe0112](https://github.com/swedishembedded/brain/commit/27fe0112c38531fefc4553378ac5c182eda50ad5))

- Fused MTP is topology-CORRECT - NPU loss is fp16 in-graph argmax, not a bug ([2c0636dd](https://github.com/swedishembedded/brain/commit/2c0636dd7ce4e1273905ca86d52d20d8c77cec6b))

- Wire fastest efficient path + explicit READY summary ([547ee9d6](https://github.com/swedishembedded/brain/commit/547ee9d6936656bd98587568eed6d1c50504673a))

- Commit agents md ([1ac8a5a7](https://github.com/swedishembedded/brain/commit/1ac8a5a7a2035f03a4e6bbbabce09e7f73905f33))

- Extract trait-based Backend/GraphBackend seam over a facade ([7c8c9704](https://github.com/swedishembedded/brain/commit/7c8c970454549e6c4ee8f732902103b3a0f11650))

- Read sharded safetensors model directories ([9a1f2bc3](https://github.com/swedishembedded/brain/commit/9a1f2bc3581669fa2f04f34756045d08c066fcdf))

- Add MLA attention + sigmoid noaux_tc router WGSL ([12d2ea47](https://github.com/swedishembedded/brain/commit/12d2ea47ac9b17e98f9e0ea6685d831809c2e7cc))

- GLM-5.2 (glm_moe_dsa) MLA-MoE decoder crate ([21b12430](https://github.com/swedishembedded/brain/commit/21b124302664e8acf3f6cd561be127f8c16c32dd))

- Add check_glm (MLA + sigmoid-router backprop gate) ([6476a85f](https://github.com/swedishembedded/brain/commit/6476a85f917d9e65529a023c6297deeba23439a0))

- Register the glm architecture in the eval battery ([e40d6ef0](https://github.com/swedishembedded/brain/commit/e40d6ef050cd4e26cb00d98f2183aff4fe480d8e))

- Add `brain glm` (train/finetune/infer/eval/import) ([0bff1210](https://github.com/swedishembedded/brain/commit/0bff1210d874226f9d047175807a98f6ce67f1cd))

- Add DSA indexer forward kernels ([7177d8fd](https://github.com/swedishembedded/brain/commit/7177d8fd305c4e116018709c90fee693941ec53c))

- DSA sparse indexer + IndexShare (Phase 2) ([94317f4c](https://github.com/swedishembedded/brain/commit/94317f4c149a7ef1da21eac9c18100eac1f5806d))

- Multi-Token Prediction head (Phase 3) ([182dd461](https://github.com/swedishembedded/brain/commit/182dd46158a4552a0e05f2ac30b0c36c70f0c09a))

- Dense-expert ONNX export (Phase 4) ([f465502f](https://github.com/swedishembedded/brain/commit/f465502fc698a5808e9d7a171a8ab4de8fbd90e5))

- Unify the model CLIs on a shared arg grammar + canonical verbs ([432385dc](https://github.com/swedishembedded/brain/commit/432385dcb9037f4bb04f53a4cdc84d25c676ea6e))

- Validate the ONNX export on OpenVINO CPU + Intel NPU ([7a6bac4d](https://github.com/swedishembedded/brain/commit/7a6bac4d459d3e1cff8f84d2b744718b515f73a8))

- INT8 weight-only export + `brain glm infer --device npu` ([2e7abe41](https://github.com/swedishembedded/brain/commit/2e7abe41f9dae2fda831a1c4e030a7641befa805))

- DSA indexer distillation training (host-side) ([d24534e7](https://github.com/swedishembedded/brain/commit/d24534e71dd76741af79a5dfb32ae230a0eb28a7))

- Matched-activation arch variants (qwen-cmp / glm-cmp) ([74f02b10](https://github.com/swedishembedded/brain/commit/74f02b1097cd40b0695856a6783dfcb66ae2b1e1))

- World-model workstream docs, fixture regeneration, and registry tooling ([9b5b7cea](https://github.com/swedishembedded/brain/commit/9b5b7cea66f19d93f60fc552aec72b9200597aa8))

- GroupNorm, FiLM/adaLN, and diffusion-glue families (fwd+bwd) ([28b75136](https://github.com/swedishembedded/brain/commit/28b75136d2c6f17e32ca4ff2db598ab2ee6eea14))

- WorldModel trait, FakeWorldModel, and kernel-family dispatch ([fcc9e01e](https://github.com/swedishembedded/brain/commit/fcc9e01e23f6805e58894e4a4a9f267a3f988b59))

- SDL2 window, chord input, pacing, and brain wm play/bench ([04788ba3](https://github.com/swedishembedded/brain/commit/04788ba3321a59419d138c1c874a9f127f879bf4))

- Pure-Rust torch .pt reader (zip + pickle state_dicts) ([693cbb9a](https://github.com/swedishembedded/brain/commit/693cbb9aac7c10304bb92aab2ea958c3a3a706ec))

- Playable DIAMOND world model - import, parity-exact UNet, brain wm play ([a20b8314](https://github.com/swedishembedded/brain/commit/a20b8314f46c6750694fdadd8782fe085d9f8a95))

- Ledger - DIAMOND playable, measured fps, next steps ([27424761](https://github.com/swedishembedded/brain/commit/2742476108586a1356b7fa631d8deab6da42d064))

- Refresh brain wm module docs (diamond is live) ([b51d11b4](https://github.com/swedishembedded/brain/commit/b51d11b4fe6dcfc4f76ec00075e0bb40995899a8))

- Fix SDL_PIXELFORMAT_RGB24 + pixel-faithful round-trip test ([5d590614](https://github.com/swedishembedded/brain/commit/5d590614f65772fbe1cfcdeb54f39fd84544fcd0))

- Measurement-driven CPU/GPU optimization of the DIAMOND path ([8396be7f](https://github.com/swedishembedded/brain/commit/8396be7f7da1bdf0d526cd77158fbcc498e21cbc))

- Wm-perf-gate - fps regression floor for the DIAMOND path ([72befd4b](https://github.com/swedishembedded/brain/commit/72befd4b096a44ae801f4289c8266041458670ae))

- Intel NPU path for DIAMOND - fp32 ONNX export + OpenVINO playback (16 fps) ([e66ecbf7](https://github.com/swedishembedded/brain/commit/e66ecbf7aae7029faa44008eeba617ed99260535))

- Training - full-UNet backward, EDM loss, gradcheck, fine-tune loop ([6cf61559](https://github.com/swedishembedded/brain/commit/6cf615596b1f03cae249122d334a175d697611a8))

- Episode datasets, pong env, record/replay, and brain wm finetune ([558cdaa2](https://github.com/swedishembedded/brain/commit/558cdaa244b3bba2acdcc4ac2fee23d06291d576))

- Ledger - P3 training + data layer landed ([233a0084](https://github.com/swedishembedded/brain/commit/233a00843063c898740eaf31d4c446e8eb4ebcf8))

- Stop fine-tuning on divergence and never save NaN weights ([5d44946d](https://github.com/swedishembedded/brain/commit/5d44946db2a3bbcf09e6ccbe0d6e391e9723cd5a))

- Fix training-graph OOB zeros buffer (gradients now correct at scale) ([b1e0531e](https://github.com/swedishembedded/brain/commit/b1e0531e3dec267db2d312d215f8ee507ae2abd8))

- Verified end-to-end fine-tune (record -> gen pong -> finetune -> play) ([17015d5b](https://github.com/swedishembedded/brain/commit/17015d5b7eb2765c56235eac3098c0671ab97855))

- VQ nearest-codebook + depthwise conv3d (P1 remainder for tokenizers) ([387350c1](https://github.com/swedishembedded/brain/commit/387350c1ff8c3dbc5ec683df715a4ef14cb973aa))

- SDL window always compiled (drop wm-sdl feature + build/wm) ([98be454e](https://github.com/swedishembedded/brain/commit/98be454e7bed4e595f5f715bb233b5e2de4f25c8))

- Ledger - P1 remainder kernels done, P4 GenieRedux spec + backlog ([468031fe](https://github.com/swedishembedded/brain/commit/468031fe24d2698bb7785b7c0fc15143ce2ceab0))

- Biased / configurable-scale attention (GenieRedux ST primitive) ([fd1804cd](https://github.com/swedishembedded/brain/commit/fd1804cdeff23661676ec574ce37ae0f48400906))

- L2norm_scale - per-row L2-norm + learnable per-dim scale (QK-norm) ([eccdb6ef](https://github.com/swedishembedded/brain/commit/eccdb6ef504a06e29f3c2bf70305dbebf2c35ff6))

- Ledger - both GenieRedux attention gaps closed; STBlock is next ([24cb0fdd](https://github.com/swedishembedded/brain/commit/24cb0fdd64a22b40cd40576bcfc4ef147f8b6ced))

- Bump registry count assertion 170 -> 178 ([3b12aff3](https://github.com/swedishembedded/brain/commit/3b12aff3e55d9f6473905e16d35d5a4c98134165))

- Drop the brittle all_kernels_present_and_nonempty count test ([5f4a0f48](https://github.com/swedishembedded/brain/commit/5f4a0f489d204a3d92d786c87fa8a0ceab80d0e3))

- Enter fully resets to the initial seed context (fix random-dream reset) ([64a4e9da](https://github.com/swedishembedded/brain/commit/64a4e9da1f4f6266014454dbd1d03a9403a4eb72))

- BiasedAttn helper - reusable QK-norm + biased attention seam ([8661daa3](https://github.com/swedishembedded/brain/commit/8661daa3c7345c11985edb636d870638d27ab238))

- GenieRedux STBlock sub-modules (QK-norm biased attention + GEGLU) ([57f2d5fd](https://github.com/swedishembedded/brain/commit/57f2d5fd1615345339b2fb280074c6b1a243b2e7))

- Ledger - STBlock sub-modules landed; P4 remaining (assembly/tokenizer/dynamics) ([aa57dc7c](https://github.com/swedishembedded/brain/commit/aa57dc7cbfe82a62c80936ab8e06e66d71fa4f32))

- Full STBlock assembly (PEG + spatial/temporal attn + GEGLU) ([9c3b2a68](https://github.com/swedishembedded/brain/commit/9c3b2a6819d6cb4c553c2d562b95d7a22a527711))

- STTransformer stack (N STBlocks + final LayerNorm) ([257da648](https://github.com/swedishembedded/brain/commit/257da64818172e47e7e3dba73be174937317e4a7))

- Ledger - STBlock + STTransformer landed; tokenizer/dynamics next ([fc40a7ae](https://github.com/swedishembedded/brain/commit/fc40a7aeb04e6d3eec07266eea11a67aff3699c3))

- STBlock/STTransformer support both st and ts order (decoder path) ([33b43b5e](https://github.com/swedishembedded/brain/commit/33b43b5e70d6c27eba7a33c055742026c6b3b724))

- Tokenizer boundary - patch-embed, to_pixels, cosine VQ ([bae86281](https://github.com/swedishembedded/brain/commit/bae86281607e914265839f309cbed764c33f13eb))

- Position-bias helpers (temporal ALiBi + spatial ContinuousPositionBias) ([21c57e00](https://github.com/swedishembedded/brain/commit/21c57e00c5309616df08c037b46acdd67581da46))

- Full ST-ViViT tokenizer forward assembly ([1c840e58](https://github.com/swedishembedded/brain/commit/1c840e58bdd020af792ab9d0432839bc41ece341))

- Ledger - tokenizer forward path complete; import/parity/dynamics next ([3ae87992](https://github.com/swedishembedded/brain/commit/3ae87992f9134da46886ab553d38d48a214109fd))

- Exact erf GELU; wm-genie GEGLU uses it (GenieRedux parity) ([c76a7813](https://github.com/swedishembedded/brain/commit/c76a78130555d63b8612ef82c04d8550b1efae1f))

- Attention k,v from un-normed x (GenieRedux parity detail) ([e2752914](https://github.com/swedishembedded/brain/commit/e2752914a74b9792bdae8bbc11d114ca8a7a7cc0))

- FeedForward LayerNorm carries bias (parity); checkpoint reader verified ([e9bc8590](https://github.com/swedishembedded/brain/commit/e9bc85907eacb4cd85e7d1bcd79bb6640840ddfe))

- GenieRedux tokenizer checkpoint import (full coverage, verified) ([0b123ca5](https://github.com/swedishembedded/brain/commit/0b123ca5ef30d1ed6a04ccd8afe7863598295415))

- Ledger - tokenizer import verified against real checkpoint; parity/dynamics next ([87a85165](https://github.com/swedishembedded/brain/commit/87a851659d9bb2cbeb4de2aeeb15c9e04458d422))

- Tokenizer PARITY-EXACT vs GenieRedux reference (+VQ raw-codebook fix) ([d17f3f65](https://github.com/swedishembedded/brain/commit/d17f3f65a8a0ead7bed9e0deb343970f55c7aea7))

- Ledger - tokenizer PARITY-EXACT; dynamics + perf next ([20fc9476](https://github.com/swedishembedded/brain/commit/20fc947620b5fb83e99316507d716b407f9fec51))

- Guided-dynamics (MaskGIT) forward + import - PARITY-EXACT ([55f9af46](https://github.com/swedishembedded/brain/commit/55f9af46103099539e6f9319537f1850ce5844b6))

- Ledger - dynamics PARITY-EXACT; both models done; sampler/ingest/perf next ([93a4882b](https://github.com/swedishembedded/brain/commit/93a4882b4eecb6dcb50e576b5963c81b829b2e76))

- MaskGIT sampler + closed generative loop (parity-exact) ([bb3d6d68](https://github.com/swedishembedded/brain/commit/bb3d6d689296d3d1c2f47ad4946cc409735f7bc2))

- Ledger - MaskGIT sampler + closed loop parity-exact; ingest/wrap/perf next ([4d0e9bc1](https://github.com/swedishembedded/brain/commit/4d0e9bc1123c4ae64b47c379334ab8d0bf8d3878))

- Lower the rounding/clamp/mix math builtins ([4701bd94](https://github.com/swedishembedded/brain/commit/4701bd94f7e2305ae03d338860538f2222e943a0))

- Torchpt reads int64 storages ([dffd0a89](https://github.com/swedishembedded/brain/commit/dffd0a89557d153cf92e761fecc3d0693183b5e7))

- Gelu_erf_bwd - the exact-GELU derivative ([afb291e1](https://github.com/swedishembedded/brain/commit/afb291e1a97d91369fc730f532226f061a225c6d))

- Ledger - P0 engine unblocks done; crates/vision next ([9d0a9e2e](https://github.com/swedishembedded/brain/commit/9d0a9e2ebb4473b5b3411dd394c1291c34c65064))

- Pin the tiny detector's forward output bitwise (P1) ([c6ac01c0](https://github.com/swedishembedded/brain/commit/c6ac01c0510985bfc2ccb286718fab1e919e726c))

- New crate - Shape/Ctx/ActTap + name-resolved ConvKernelIds ([29b24620](https://github.com/swedishembedded/brain/commit/29b2462050b69815883356d5ed5140815b3d99f8))

- Net.rs -> re-export shim over crates/vision ([1d233f54](https://github.com/swedishembedded/brain/commit/1d233f54beea5e1b485bfd9c0ed13ba942dd62d7))

- Move the conv blocks out of yolo ([25f7fb05](https://github.com/swedishembedded/brain/commit/25f7fb05ebb876cebb190645ad6815f7a1a9f041))

- Move the neck plumbing (Up/Cat/Acc) out of yolo ([89d2eefb](https://github.com/swedishembedded/brain/commit/89d2eefb885d64d64074724818b43b0ffdebe3fe))

- The depth family (P2) - grouped/dilated conv, bilinear, pooling, convex upsample ([c817c8ff](https://github.com/swedishembedded/brain/commit/c817c8ffa6ee2336a4db3271051a8c5101f3b8d7))

- Ledger - P1 vision layer + P2 kernels done; ZipDepth model next ([c36b1244](https://github.com/swedishembedded/brain/commit/c36b12448aee0ab49b19e6ade132e5f9abdf0039))

- Close the ZipDepth kernel set + fix a SIGPIPE bug in kernels-regen ([81908ab4](https://github.com/swedishembedded/brain/commit/81908ab4f6b2e5bd488a4c8cc2b04fb50efc9ad1))

- Ledger - kernel layer complete for ZipDepth; the model is next ([7cdc747b](https://github.com/swedishembedded/brain/commit/7cdc747bfd1a065760e30a76608912d10f4872a3))

- Delete 4 redundant kernels found by a reuse audit ([a5cf7d24](https://github.com/swedishembedded/brain/commit/a5cf7d24c29a6998a4d6d216513cf18cc72ed576))

- Ledger - record the reuse audit and the standing rule it establishes ([9063ba20](https://github.com/swedishembedded/brain/commit/9063ba20318cc92f80c2e505d696fe7a5987281f))

- Crates/depth - ZipDepth's config and parameter layout ([103f4e89](https://github.com/swedishembedded/brain/commit/103f4e890a14b7a6268179277c0bb616928f580d))

- RepVGG fuse; move fold_bn from npu to vision ([c93484b9](https://github.com/swedishembedded/brain/commit/c93484b913f2bbcb8df4e96928af984627c92443))

- Ledger - P3 config+fuse verified against the real checkpoints ([fe2d360a](https://github.com/swedishembedded/brain/commit/fe2d360aab74eff7c3fe12a7b0581fe92fbc5ad5))

- ZipDepth kernel registry + deterministic weight init ([215a498e](https://github.com/swedishembedded/brain/commit/215a498eeec7fc2aad41056290b3146c01676038))

- Make Conv spec-driven (grouped/dilated/ReLU); depth uses it ([cb1ff295](https://github.com/swedishembedded/brain/commit/cb1ff295aa0ba446b479237f14521e0464ef76d9))

- Ledger - net/init/ConvSpec done; record the bn_eval and dense-conv traps ([2a942088](https://github.com/swedishembedded/brain/commit/2a942088f20b3823cacf427f7959dc58f67ffe12))

- QARepBlock + configurable Conv tensor names ([d9a2d6df](https://github.com/swedishembedded/brain/commit/d9a2d6df96fa89cb86430ee8f4c50ddcf6051966))

- Ledger - QARepBlock done; record the bias_add/NCHW trap ([a78701a8](https://github.com/swedishembedded/brain/commit/a78701a8f2ec82a65e287634e55b80dbd514ca7d))

- ChannelAttention (SE) + widen ConvKernelIds for the depth blocks ([027671b0](https://github.com/swedishembedded/brain/commit/027671b0c811d0151dce2fd056c06d812d48751d))

- Ledger - SE done; MinimalMultiScale needs a separable BN unit ([9b14d944](https://github.com/swedishembedded/brain/commit/9b14d944e43a38c2e45ca0cee6b9cbe08f09c89c))

- Standalone BatchNorm + ConvSpec::norm; depth: MinimalMultiScale ([8276df71](https://github.com/swedishembedded/brain/commit/8276df716ad7fe2161ab810b6b3ad2450f0e66f8))

- Ledger - BatchNorm/Norm landed, MinimalMultiScale done; 3 of 10 blocks ([1eab6496](https://github.com/swedishembedded/brain/commit/1eab6496f1daecf68ccc3d074c97f8b0c45ef9bd))

- StripPoolingAttention; vision: Act::Sigmoid ([775068a2](https://github.com/swedishembedded/brain/commit/775068a233f06da2eb07408504b1bc7ce2c84135))

- GlobalContextBlock; vision: ConvSpec::bias ([a2e68e56](https://github.com/swedishembedded/brain/commit/a2e68e56cddb632a08179e885325b56cc4550272))

- SPPF::with_spec + NameStyle/SppfSpec; ids: axpy ([1a62c8a7](https://github.com/swedishembedded/brain/commit/1a62c8a7335ad04388a9cfd92749ea649695d0e5))

- The last five blocks - SPPF, CrossScale, Fusion, ConvexUpsample x2 ([42c22830](https://github.com/swedishembedded/brain/commit/42c2283014bb10d4033e27ab48d5cb9debbf16b5))

- ZipDepth model - the encoder/decoder assembly ([f3eca767](https://github.com/swedishembedded/brain/commit/f3eca767fa44ace945a80cce55b6af0b155a837e))

- ZipDepth master gradcheck - the whole backward, end to end ([5393e883](https://github.com/swedishembedded/brain/commit/5393e88374c0d5d0a17eab4811b4f8577ecaecd4))

- Import a released ZipDepth checkpoint; P3 complete ([b606a751](https://github.com/swedishembedded/brain/commit/b606a7514569218c6763511730fb9b6883aec524))

- Viz - colormap, robust bounds, side-by-side composite (P5) ([b5d7e650](https://github.com/swedishembedded/brain/commit/b5d7e650393d3a192571dc6993f5edb15794fbfa))

- Brain depth --image - the demo runs on real weights (P5) ([b334d001](https://github.com/swedishembedded/brain/commit/b334d0018e1391ebb21c453aa24bda101ef460ad))

- Device-aware CLI (--device cpu|vulkan); ledger - P5 image path done ([bd08c656](https://github.com/swedishembedded/brain/commit/bd08c6569dccce17b54564e3889dfc3bae32a426))

- V4L2 webcam + brain depth --camera (realtime path) ([34b30052](https://github.com/swedishembedded/brain/commit/34b300526a813fba4d90102af7f390952e386321))

- Per-layer INT8 outlier report - measure, then decide (P6) ([34e77fd9](https://github.com/swedishembedded/brain/commit/34e77fd933e32cb76bc365c42889ba5b908878e0))

- Ledger - P6 measurement done, finding recorded ([880680a4](https://github.com/swedishembedded/brain/commit/880680a4aa1b489407d5776893db6ce6ce1a76dc))

- Live smoke test - brain-built ONNX runs on the Intel NPU ([f6cb8f59](https://github.com/swedishembedded/brain/commit/f6cb8f59c361b8611a695931689796367e151a8b))

- ZipDepth -> ONNX -> Intel NPU, exact parity ([6ffd4543](https://github.com/swedishembedded/brain/commit/6ffd45430e7cd435f87c38509c22687f64963126))

- Brain depth --infer npu - the demo runs on the Intel NPU ([80b7cb23](https://github.com/swedishembedded/brain/commit/80b7cb236d851ca64c958197ff0b8bd3dd66407d))

- Ledger - fp32 NPU deployment done (parity 0.99998 on the real NPU) ([716821b3](https://github.com/swedishembedded/brain/commit/716821b36394ff9704da0590096b91b033919a0e))

- Brain depth --camera --infer npu - realtime webcam depth on the NPU ([50133ec9](https://github.com/swedishembedded/brain/commit/50133ec9492ef799c42b35a7d3e94f9ac05788a4))

- In-frame HUD text - fps/latency/drops read ON the image ([99ba42bc](https://github.com/swedishembedded/brain/commit/99ba42bc157bf935d6f946b1be36fa436a8218b1))

- FIX accuracy - aspect-preserving preprocessing to match the reference ([d0de1d03](https://github.com/swedishembedded/brain/commit/d0de1d031d94ee01cb91a333f26e6d86ff06ccd7))

- Untrack photo.ppm + gitignore demo artifacts ([79c9a5f3](https://github.com/swedishembedded/brain/commit/79c9a5f307192d5a23f14bc406c93a979ffe3108))

- Magic-Eye autostereogram view + a clean view-toggle UX ([03de3f15](https://github.com/swedishembedded/brain/commit/03de3f156aeeb5e78e1508f840585697d2de06b3))

- Textured autostereogram + per-view window sizing (no more 2x stretch) ([fd1499c7](https://github.com/swedishembedded/brain/commit/fd1499c7f8a8ee7cfd012af6848284e62f4119a4))

- Fix textured stereogram (was garbage), stronger depth, add stereo-dual ([52cab3aa](https://github.com/swedishembedded/brain/commit/52cab3aab75e559285f46115ed02288c9c2a2a43))

- --stripes knob for the stereograms (default 5, wider slices) ([65a10979](https://github.com/swedishembedded/brain/commit/65a109796ddb4ec6d93d7b3b850cec881f0bccd1))

- Fix static-centre stereogram bug (TDD); add fog + blur views ([5d9ff3f5](https://github.com/swedishembedded/brain/commit/5d9ff3f59901b94242796a1cbbe17843a7dfc4aa))

- Kill the 2-blocking-submits-per-dispatch uniform path ([f309a005](https://github.com/swedishembedded/brain/commit/f309a005a74f249dcb40b08762c59be40fd43d58))

- Brain depth --bench N - steady-state frame timing ([0da74dce](https://github.com/swedishembedded/brain/commit/0da74dcef2b6778d4e82c2159f821bb375fcf2cb))

- Act-selector fused conv + grouped register tiling - ZipDepth eval fuses on every backend ([06e7c493](https://github.com/swedishembedded/brain/commit/06e7c49329139cc6ad822c3f7ea21cb9e7eae002))

- Brain depth train - the end-to-end training loop LEARNS (P4 placeholder-grade) ([b4724c44](https://github.com/swedishembedded/brain/commit/b4724c44909a51ea166f067970a1e5d51c5be48c))

- Brain depth --input N - trade depth sharpness for frame rate ([51a43c6b](https://github.com/swedishembedded/brain/commit/51a43c6bc17a61f69fa1f5f6eb332330b4ec5981))

- Per-kernel GPU timestamp profiling + trusted shaders + conv microbench ([c6801bd4](https://github.com/swedishembedded/brain/commit/c6801bd4f90561462aceb9db4712c3fc2ebc63b9))

- Collapse the eval graph - 165 -> 86 dispatches/frame ([9ce5530a](https://github.com/swedishembedded/brain/commit/9ce5530a9bbd3821190939e47bd6bf433b5f6cbb))

- Row-parallel host pre/post-processing (rayon) ([da01881b](https://github.com/swedishembedded/brain/commit/da01881bf09a7f41c2127c72cac438e64eeb9124))

- Pipelined camera inference - overlap host work with device compute ([ce7dd801](https://github.com/swedishembedded/brain/commit/ce7dd80105682cc8fd5d0cbe179520c0eb4c7615))

- Bn_eval act word is optional for direct callers too ([db633fb0](https://github.com/swedishembedded/brain/commit/db633fb0e7b4e084fe09c8af6e0e32fe7d119180))

- Parameterize LayerNorm epsilon (was hard-coded 1e-5) ([5055b1eb](https://github.com/swedishembedded/brain/commit/5055b1eb12ca83ada7da41cf29c7edfc9d4c6118))

- Generic device prefix-scan and stable LSD radix sort ([bd714622](https://github.com/swedishembedded/brain/commit/bd714622f4b85657aad1adf4c77d8ef769d826b3))

- Shared ViT block builder + the kernels vision transformers need ([b79fc6f5](https://github.com/swedishembedded/brain/commit/b79fc6f548c7576021bebe9ce3ad80c560ee5d55))

- From-scratch 3D Gaussian Splatting - tiled rasterizer, backward, fit ([db67daa6](https://github.com/swedishembedded/brain/commit/db67daa6d6b80d0ed69235f1244f696e6f50bdc0))

- Relative mouse-look + fly-camera keys ([7a7e15f4](https://github.com/swedishembedded/brain/commit/7a7e15f4543b164384d31ab8280dc0f36043f909))

- WorldMirror-2 on the brain engine - import, encode, reconstruct ([4f4256c6](https://github.com/swedishembedded/brain/commit/4f4256c6bd18b8d66115951af92a2665d2d1206b))

- WorldMirror-2 DINOv2 encoder as an OpenVINO whole-graph export ([735d9167](https://github.com/swedishembedded/brain/commit/735d91677e62bd5d471d19b5eb95f84c59a5f428))

- Brain mirror + brain splat - photos to a navigable 3DGS world ([94e89631](https://github.com/swedishembedded/brain/commit/94e89631bce5b14ab4ba48b0528d249f5ac3fcbc))

- ViT block backward - the full training path, autograd-verified ([1effe5e1](https://github.com/swedishembedded/brain/commit/1effe5e17b805504188ce64c7034eec5c1c559da))

- Row-blocked matmul for the transformer forward hot path ([dc4fdeed](https://github.com/swedishembedded/brain/commit/dc4fdeedc19bd73a6209839bb656a4fbf4c21691))

- Pos-embed interpolation - any aspect ratio, T7 rect parity ([91745355](https://github.com/swedishembedded/brain/commit/9174535583bbae3919d9ffc28c1d1a967263b6ff))

- Fix the multi-frame (S>1) forward - NaN scenes at S=3 ([57fb64f5](https://github.com/swedishembedded/brain/commit/57fb64f5fdcb713e059cfa49752560ac07d77abc))

- Reference-exact voxel-merge prune, wired as mirror --prune ([3b474c00](https://github.com/swedishembedded/brain/commit/3b474c00a97e1d0fe00e4f8416e95ed016f84f1f))

- WorldMirror trunk + DPT-head ONNX export (stages 6b/6c) ([5de75c7f](https://github.com/swedishembedded/brain/commit/5de75c7f56675451f1a8e13f7f1e1c6028c3f0cc))

- Run the mirror ONNX exports for real - trunk fails on NPU ([450f35f6](https://github.com/swedishembedded/brain/commit/450f35f63d1e15eb1022ea88356fa748b646bd9d))

- Fix the trunk on NPU - a Concat writing into a graph output ([fdd050c6](https://github.com/swedishembedded/brain/commit/fdd050c655a010998bcce5f49d28fc82482dbba8))

- Reject dispatches that bind their output as an input ([7120ce24](https://github.com/swedishembedded/brain/commit/7120ce24c490c82a651af52a50755ccc0fd81b94))

- Pass the LayerNorm eps - a call site my eps refactor missed ([87206959](https://github.com/swedishembedded/brain/commit/872069599778907d86343e22a886ac93ff3c3c7d))

- Subsystem + weekly finetune P0/P1 (differentiable Kronos decoder) ([5ce44879](https://github.com/swedishembedded/brain/commit/5ce448795b03283cd18bbb20b97ac925e1cf7d46))

- LoRA fine-tune surface (P1-D) ([8da5da08](https://github.com/swedishembedded/brain/commit/8da5da08b90155228f8ceea4acc0d4f1b547131e))

- Fit loop + walk-forward promotion gate + checkpoint save (P3) ([39fb392a](https://github.com/swedishembedded/brain/commit/39fb392a49a109cee887d803659358918848bdea))

- End-to-end universe fine-tuning over leak-safe windows (P4) ([c5bdcab0](https://github.com/swedishembedded/brain/commit/c5bdcab0c5f77c0a389f4447c681ad28d73f59f9))

- `brain forecast finetune` - runnable weekly Kronos fine-tune (P4) ([d937842d](https://github.com/swedishembedded/brain/commit/d937842d10405a02069e520b77f52044c8c4b367))

- Fold LoRA adapters into the saved checkpoint + load .weights everywhere ([fa7a6d96](https://github.com/swedishembedded/brain/commit/fa7a6d96e3414143112ee74c7d543c6f52570b85))

- Progress display, held-out-names generalization eval, weekly machinery ([52b63dc3](https://github.com/swedishembedded/brain/commit/52b63dc30ee98b6105dace5864964dd75769e20d))

- Resumable per-origin backtest checkpoint + honest RankIC proof ([691721d8](https://github.com/swedishembedded/brain/commit/691721d8fad342ec217173fea786144f4519467e))

- Real streaming + recoverable cancellation (P0 blockers #1/#2) ([400d1ea4](https://github.com/swedishembedded/brain/commit/400d1ea4bea4bf02c5a09123d36f4291eaf6a49b))

- Stream the backtest (backtest_chunk), the streaming payoff ([2a4383b4](https://github.com/swedishembedded/brain/commit/2a4383b4732d58eb0530b82a301d5e413b92d1fc))

- Register the forecasting battery as a 'forecasting' axis ([785f811e](https://github.com/swedishembedded/brain/commit/785f811e766aea2f6dd302d9d4b54b953f23cd78))

- Pinball/quantile-loss gradient (Chronos-2 backprop milestone 1) ([9cd5f9ed](https://github.com/swedishembedded/brain/commit/9cd5f9edfcab74b399b3de1fe89854da14821e76))

- Differentiable head path, gradchecked (milestone 2) ([1ade631c](https://github.com/swedishembedded/brain/commit/1ade631c85b0079e42880754a5c9cab77a98558c))

- Differentiable encoder block, gradchecked (milestone 3) ([3fdcb384](https://github.com/swedishembedded/brain/commit/3fdcb384a61019c1e8c7d5556dca8b8c88f28684))

- Full differentiable backbone + from-scratch learning (milestone 4) ([dfbab746](https://github.com/swedishembedded/brain/commit/dfbab746b0712d8c5eca3781b18791371b1e19b2))

- Pluggable transformer core in forecast_quantiles (NPU/M5 seam) ([cb0dce70](https://github.com/swedishembedded/brain/commit/cb0dce70cc139882d258526c8ea72e6c0e23e5be))

- Brain npu chronos2 driver - transformer core on the NPU (validated) ([0dbec641](https://github.com/swedishembedded/brain/commit/0dbec6419da36c5f6cfe14addc413adb0d0eda26))

- Brain npu kronos driver - dual-graph AR loop on the NPU (validated) ([43a59626](https://github.com/swedishembedded/brain/commit/43a596269c595aefb83a1a404c6915d4bf71364d))

- P1-P2 config + param_list + T0 layout gate (verified vs real 991.4M v1.pth) ([2409b103](https://github.com/swedishembedded/brain/commit/2409b103d8cd1eb609f1eb5a222cb1958c6e0c5f))

- P3-P5 import + preprocessing + device forward (patched decoder + top-2 MoE + PQ head) ([3db8b16d](https://github.com/swedishembedded/brain/commit/3db8b16df9cfbf9785ff2840b15030ba453b0967))

- P6-P7,P9 parity gate + forecaster adapter + host-differentiable training ([57ff543e](https://github.com/swedishembedded/brain/commit/57ff543ed52278257f64b5659a36a065d06d640c))

- P7 CLI - brain forecast import/serve/compare --fincast ([878056c5](https://github.com/swedishembedded/brain/commit/878056c59394f9cd8fbbd65e2f41de0dc6f0420f))

- P8 NPU - ONNX core export + FincastSession + brain npu fincast ([3fb96f99](https://github.com/swedishembedded/brain/commit/3fb96f99f30f949a0325873ba2a2f60829862a3b))

- NPU parity fix - opset-13 ReduceSum axes-as-input (cosine 1.0 on real NPU) ([b021a493](https://github.com/swedishembedded/brain/commit/b021a49332e79884c191f1b5c34aa5f43584c9a1))

- MoE gather/scatter - compute only routed tokens (~1.36x CPU) ([3d6a5fdf](https://github.com/swedishembedded/brain/commit/3d6a5fdff7b3615def2b1e85a0a9560048838886))

- Model-agnostic out-of-sample cross-sectional skill eval + report ([0bd8c415](https://github.com/swedishembedded/brain/commit/0bd8c4155238aba01cfbe59e39548b9d4b3cd0ea))

- Rendered out-of-sample model comparison report (real 2026 data) ([247f09e0](https://github.com/swedishembedded/brain/commit/247f09e0f6acdcd5b60a7b333d611e618cfda1bf))

- P40 GPU training: register GEMMs, two-pass CE, optimizer offload, tool-call finetune

Make brain train and infer efficiently on 2x Tesla P40, and unblock full
fine-tuning of Qwen3-0.6B on the two cards.

Kernels / compute
- Register-tiled + software-pipelined GEMMs (matmul_reg, matmul_reg2) and
  tiled backward GEMMs (matmul_dx_reg, matmul_dw_reg): bit-exact, ~34% of FP32
  peak vs ~0.7% for the naive 0.25-FLOP/byte matmul.
- INT8 DP4A GEMM (matmul_i8) and conv-as-GEMM path (im2col, conv_epilogue).
- Per-kernel workgroup sizing instead of a hard-locked @workgroup_size(64).

Training bottleneck fix (the real one)
- Profiling showed GEMMs <1% of a 0.6B step; cost was two reductions.
- Two-pass cross-entropy gradient (ce_stats + ce_grad_stats): per-row softmax
  stats computed once -> O(rows*vocab) not O(rows*vocab^2). Wired into gpt and
  qwen; gradcheck-green (bit-exact). Full 0.6B step 89s -> 7s.

Optimizer-state offload to system RAM (ZeRO-Offload)
- Role::Offload: weight+grad on GPU, AdamW moments (m/v) held in host RAM;
  OffloadAdam runs exact AdamW on CPU (rayon), host-side grad-norm clip.
- Parity test bit-exact vs GPU AdamW (rel 5.9e-8). Fits block-512 full-trainable
  where the all-on-GPU path OOMs at 24 GB.

Fine-tuning + tool-call pipeline
- qwen::finetune: FullOffload vs LoRA over masked chat/tool-call datasets.
- data::chat + data::toolcall: chat/tool-call example generation, token-level
  assistant-span masking, teacher-forced exact-match eval (toolcall_eval).
- Integration tests (qwen3): inference coherence, training validity, tool-call
  and reasoning finetune, full-vs-LoRA comparison. Full offload FT generalizes
  0% -> 41.7% held-out tool-call exact-match.

Multi-GPU foundation
- BRAIN_GPU_INDEX device selection over enumerated discrete adapters; verified
  gradcheck runs on the second P40.

Misc
- Time-based checkpointing (checkpoint_secs, default 600s) reporting save time.
- CLI: qwen import --block, qwen toolcall gen|eval, --save-secs.
- Docs: P40.md (roofline, offload, CE fix), PERFORMANCE/TTS accel notes. ([68db4859](https://github.com/swedishembedded/brain/commit/68db485974af833631a7177f13c9244b541aa797))

- Pipeline-parallel model sharding across GPUs (bit-exact) ([4eac8266](https://github.com/swedishembedded/brain/commit/4eac82664cb3185b5d1656f551a9790295a8fe43))

- Data-parallel training across GPUs (1.34-1.58x on 2 P40s) ([e632cb5f](https://github.com/swedishembedded/brain/commit/e632cb5f52cac9839ac46ea35d692adf4c9b82a0))

- Make data-parallel training generic over all models ([77e17c43](https://github.com/swedishembedded/brain/commit/77e17c43bf9431f1da3fa1aec56a5808fecfc34a))

- Generic pipeline sharding with automatic cut placement ([f65cb6ab](https://github.com/swedishembedded/brain/commit/f65cb6abd0d6ad06a1171a0ba04209635ca1f6dd))

- Concurrent micro-batched pipeline (GPipe + recompute), 1.26x ([ac008f32](https://github.com/swedishembedded/brain/commit/ac008f32993ffe9ec27050d0432f1311a1dd1e4e))

- Transport-agnostic collectives layer (all-reduce/gather/scatter/broadcast) ([8c26b3c4](https://github.com/swedishembedded/brain/commit/8c26b3c4adc14a1b25a8a56fe6752068e26c558c))

- Tensor-parallel MLP validated bit-exact across 2 GPUs (Megatron) ([9fa9a135](https://github.com/swedishembedded/brain/commit/9fa9a1357ef9ccc7c585066a0ad873e6429179b0))

- Tensor-parallel planner (T_local+T_comm+T_sync cost model) ([3c28b704](https://github.com/swedishembedded/brain/commit/3c28b704b68d596d4d02b646f62640bacb086017))

- 3D process grid (tensor x pipeline x data rank mapping) ([e0b03ebf](https://github.com/swedishembedded/brain/commit/e0b03ebfd0306d024794c3c89624ee3a0244a00a))

- LocalGroups -- per-group collectives from a 3D grid ([95a182eb](https://github.com/swedishembedded/brain/commit/95a182eb2359e835936a8ae67e533134a8649f36))

- Tensor-parallel attention (head-split) validated bit-exact ([417e4bb0](https://github.com/swedishembedded/brain/commit/417e4bb0512e322ec4e834098488cd811b3aa43e))

- Tensor-parallel training (backward) validated bit-exact ([aea1b268](https://github.com/swedishembedded/brain/commit/aea1b26867cbde0f297a4569de1c7d07e30a9c8c))

- Add flow-matching Euler scheduler (Z-Image/FLUX.2 rectified flow) ([5f7ae6dc](https://github.com/swedishembedded/brain/commit/5f7ae6dc23bac0303b3b1b43110bee511919ca41))

- Add AutoencoderKL decoder with bit-exact Z-Image parity ([effb30b5](https://github.com/swedishembedded/brain/commit/effb30b5e4d97dd1b98298991292b1fe9ecb8e99))

- Encoder hidden-state extraction + sharded import (Qwen3-4B text encoder) ([f4484e12](https://github.com/swedishembedded/brain/commit/f4484e12bab2c66ef739dac3d016561bac78cc39))

- Add multi-axis RoPE freqs_cis builder (Z-Image S³-DiT positional encoding) ([ed42b913](https://github.com/swedishembedded/brain/commit/ed42b913ae3af78c1f51e0fbb4df9082ee0ed32a))

- ZImageTransformerBlock forward with bit-exact diffusers parity ([8885cd07](https://github.com/swedishembedded/brain/commit/8885cd0779280edc6a4c0cb62e853dc9cc8f1e38))

- Full S³-DiT forward with bit-exact diffusers parity ([4ff58ae3](https://github.com/swedishembedded/brain/commit/4ff58ae333cc8ea44a68888a9dbaad8edc31d977))

- Import original/Comfy checkpoint layout (fused-qkv split + key map) ([4f044c49](https://github.com/swedishembedded/brain/commit/4f044c49512a01690b11d724557d087ce5c0c78c))

- Real 6B Z-Image-Turbo DiT forward parity (end-to-end, trained weights) ([08cda446](https://github.com/swedishembedded/brain/commit/08cda446ac480b5dc76df37555889fd09782bb03))

- Device-resident forward (ZImageDit) - weights resident, no per-block round-trips ([f9c796e8](https://github.com/swedishembedded/brain/commit/f9c796e811cda7f6d95c6590c2fe371e2fb463c9))

- Fix BRAIN_GPU_INDEX selecting the same card twice ([4b61e9ee](https://github.com/swedishembedded/brain/commit/4b61e9ee0b23bd617bfb3a822930a256822ec15c))

- Register-tiled GEMM + 2-GPU sharding + profiler ([a7fb7789](https://github.com/swedishembedded/brain/commit/a7fb7789f9da715833a4d4c0583a975394708d14))

- Per-block flush during resident build + GPU probe mode ([5c5113cf](https://github.com/swedishembedded/brain/commit/5c5113cf8381bba7d6f2f579adfdbb94fac86255))

- BRAIN_ZIMAGE_LAYERS override for single-card profiling ([3307e2ca](https://github.com/swedishembedded/brain/commit/3307e2ca4f7d4ec28a1bd1d26efa96c4a8184521))

- Make the 6B Z-Image DiT fit and run across both P40s ([800bdac2](https://github.com/swedishembedded/brain/commit/800bdac257329a979cbe2f6f92849c0c35af5169))

- DP4A GEMM pipeline with on-device dynamic activation quantization ([f9f6cae7](https://github.com/swedishembedded/brain/commit/f9f6cae79220fa68f6bd4b6cecd85514cd785a5a))

- Full DP4A DiT on ONE card - per-token + per-channel quant, cosine 0.99 ([28884bf1](https://github.com/swedishembedded/brain/commit/28884bf1ae6cdd850cb4364faf74f38d76bdae1a))

- Hand-written S³-DiT block backward + finite-difference gradcheck ([0827b0eb](https://github.com/swedishembedded/brain/commit/0827b0ebf0dfd0642f1d9c8894097432bf5c66cb))

- Training-step profiler - real fwd+bwd GEMM sweep across the DiT ([369ba3c1](https://github.com/swedishembedded/brain/commit/369ba3c1d7206d36f12f079fabe9d0b70521973d))

- Device (GPU) S³-DiT block backward - matches gradchecked host at fp32 ([15e3a3c8](https://github.com/swedishembedded/brain/commit/15e3a3c89bedc526dda8c51282fb570043fd8438))

- Overfit-one-batch gate - block gradients drive Adam to convergence ([a27ec76d](https://github.com/swedishembedded/brain/commit/a27ec76d14f88a8d84a554c9ec2da7740713b044))

- Full-model training loop - end-to-end backward + flow-matching, gradchecked ([54380692](https://github.com/swedishembedded/brain/commit/54380692a68ab40920bb4e53e900aeef79c16e58))

- Drop superseded per-tensor int8 kernels from the block pipeline ([298686b3](https://github.com/swedishembedded/brain/commit/298686b369231d667f9beeeb48b8c686713fe629))

- Device (GPU) full-model training loop - matches host reference, overfits ([1eebb628](https://github.com/swedishembedded/brain/commit/1eebb62809786227d9ed8c191805442b391d6cf6))

- Unify distributed training on the Collective seam - cluster + federated ([b198aa73](https://github.com/swedishembedded/brain/commit/b198aa73e27d018a57b390d4e928c828046c4f9f))

- End-to-end distributed-training test over both transports + federated ([8796f8ef](https://github.com/swedishembedded/brain/commit/8796f8ef41492a6fd1b5f01c8248ca0704f206d5))

- Impl Model - Z-Image joins brain's generic distributed training ([31cd2d0a](https://github.com/swedishembedded/brain/commit/31cd2d0a66a482f2d712f508fd36ad7f947da050))

- Pipeline-shardable phases + grad-parity - memory-safe full-6B training ([8cbfd997](https://github.com/swedishembedded/brain/commit/8cbfd997a5e7c2b256535eea55fba7e1a7242118))

- ShardTrainer - pipeline-parallel training across both P40s, no OOM ([e044e0fc](https://github.com/swedishembedded/brain/commit/e044e0fc378ac3327ea15af1b6efc350dc884430))

- GPipe overlap - both P40s run concurrently (the efficiency win) ([0acd2040](https://github.com/swedishembedded/brain/commit/0acd204063d6292e377955fef0361045cb9037f2))

- F32 runtime training storage - halves RAM, keeps the f64 gradcheck ([84fc042c](https://github.com/swedishembedded/brain/commit/84fc042c8c5e2261b0e0ff47d4eb84a8b3c37cce))

- Generalized model-capability interface + `brain caps`/`brain do` ([da5bdf25](https://github.com/swedishembedded/brain/commit/da5bdf257845c74640f5d2e46de32f9f60f3201a))

- Expose actions over the event API (manifest_request / action_request) ([edc10e29](https://github.com/swedishembedded/brain/commit/edc10e29f5f70bc6c0db7939dd315f2d7b973d2d))

- Drive `brain do` param parsing with clap, spec-driven (no custom parsing) ([d209523f](https://github.com/swedishembedded/brain/commit/d209523f8a57c9b87864084be84dd6082f1f292c))

- Imageops capability - deterministic mask + procedural renders, viewable via `brain do` ([5dbc89ac](https://github.com/swedishembedded/brain/commit/5dbc89aca7ff6c843462242802e0cd3fc47c8f9c))

- Assemble the real text-to-image pipeline; wire it into the text2image action ([4ddaaf94](https://github.com/swedishembedded/brain/commit/4ddaaf946d2e7d2920f681a3727d5879a3c35566))

- Run the generation pipeline on CPU (fp32) - fits RAM, avoids P40 limits ([cb35e053](https://github.com/swedishembedded/brain/commit/cb35e053cfe51a501cd5a0311c4b3ba8a223ea1e))

- Keep heavy compute on the GPU - inference-only encoder + int8 DiT ([e36477ba](https://github.com/swedishembedded/brain/commit/e36477ba2ed6da3312d4ba194f9435688d9bb0a4))

- Fix end-to-end text2image - VAE latent dims + velocity sign ([4b095242](https://github.com/swedishembedded/brain/commit/4b095242c607cda5a43cdc90a290f464d1233004))

- Wire image2image / inpaint / outpaint + add a VAE encoder ([a609d527](https://github.com/swedishembedded/brain/commit/a609d5274b0f34db0b8b3c5c0fd156453c57d6b6))

- LoRA training core (low-rank adapters over the frozen base) ([c30343a2](https://github.com/swedishembedded/brain/commit/c30343a290203f15a0a0434c7e55e7f0c15cbf68))

- Mask feathering for inpaint/outpaint + accurate lora_train status ([f2efc6ae](https://github.com/swedishembedded/brain/commit/f2efc6aea186e4374f37ea394ab6631c49bae5c4))

- Higher-fidelity fp32 DiT path (2×P40 shard) via --precision ([7a9f8f5c](https://github.com/swedishembedded/brain/commit/7a9f8f5c8d6eafb210b6bbfc06aaf7fe88446d56))

- Flash attention: tiled online-softmax attention backend (Pascal-friendly)

Adds flash_attn_bidir.wgsl - a fused scores→softmax→apply that NEVER materialises
the [B,H,T,T] scores/probs, so peak attention memory is O(T·head_dim) instead of
O(T²). One workgroup owns 64 query rows of an (b,h) and streams K/V in 8-row tiles
THROUGH SHARED MEMORY (40 KiB), so all 64 queries reuse each loaded tile - ≈64×
less global K/V traffic than a per-query kernel, the difference between
memory-bound and compute-bound on a P40 (sm_61: no subgroups/f16, only shared mem
+ workgroupBarrier). q cached in shared so the output accumulator stays in
registers. Output layout matches attn_apply_bidir → drop-in for the trio.

Wired into both the fp32 and int8 Z-Image blocks (block.rs `push_attention`);
Scratch skips the [nh·t·t] buffers under flash. Auto-enabled once the materialised
scores would exceed ~512 MiB (well before the 2 GiB per-binding limit that OOMs
high-res latents); BRAIN_ZIMAGE_FLASH=1|0 overrides. GPU-only (the CPU JIT can't
compile the barrier); CPU keeps the materialised path.

Verified: with flash forced on, the real 6B int8 DiT still matches the diffusers
golden (cosine 0.99123 vs 0.99022 baseline) - online softmax is numerically exact.

Benefits: lower peak attention memory, faster high-resolution inference, lower
training activation memory, and generation at resolutions that otherwise OOM. ([c655f60f](https://github.com/swedishembedded/brain/commit/c655f60f628ce72f5dde99f8e9b439484dc35d4e))

- Flash attention: higher-occupancy kernel + memory-escape-hatch heuristic

Kernel: cache q_i and the output accumulator in REGISTERS (q reused across all
tiles - minimal traffic) and keep only the K/V tiles in shared memory (16 KiB, was
40 KiB). At 16 KiB a P40 SM runs ~3 workgroups instead of 1, raising sustained
power 88 W→120 W - more warps to hide the global-load latency this kernel is bound
by. (Larger BC=16 tile too.)

Heuristic: the materialised path uses a tuned register-tiled GEMM, so where it FITS
it beats the hand-written fused flash loops on Pascal (no tensor cores). Flash is
therefore a MEMORY escape hatch, not a blanket speedup - auto-enable only once the
[nh·t·t] scores buffer approaches the 2 GiB per-binding limit (~1800 MiB, ~1024²).
Below that, materialised stays (faster); above it, flash is the only thing that
runs. BRAIN_ZIMAGE_FLASH=1|0 forces it. ([d538e95b](https://github.com/swedishembedded/brain/commit/d538e95bac1d39254c5a832cca93fb710b873536))

- Hot-weights server path + card-agnostic flash + brain-py image gen ([2d244f6a](https://github.com/swedishembedded/brain/commit/2d244f6a6bc3262e15879d783b0e2fd15933ae4c))

- Tight weight upload + staging reclaim; zimage: encoder-GPU option ([e368b9eb](https://github.com/swedishembedded/brain/commit/e368b9eb79f536ce8c979d82696e572f77e19dbe))

- 2-card encoder shard (Encoder::Split) - correct architecture, opt-in ([9bc1a348](https://github.com/swedishembedded/brain/commit/9bc1a348b1f397192e8d51d727c76c0c7c52ed5b))

- Int8 GPU encoder for superfast hot generations ([b3da48c8](https://github.com/swedishembedded/brain/commit/b3da48c8a698083bbeb357e2dd7683decbbe74ef))

- Place VAE decoder on the encoder card (int8 GPU encoder) ([6bff6de6](https://github.com/swedishembedded/brain/commit/6bff6de67a035d0ca0bb0e1c2e715158cd964675))

- Pool activation buffers in the decode/encode graph ([e69ef2c6](https://github.com/swedishembedded/brain/commit/e69ef2c62d478a3ac987fb5b3031c7ea42a96301))

- LoRA-training foundations - dataset loader + weight-format bridge ([60a1dc31](https://github.com/swedishembedded/brain/commit/60a1dc313f3dc3dcac13bc007ea064ba23186e18))

- Flow-matching batch builder for LoRA training ([5b7922dd](https://github.com/swedishembedded/brain/commit/5b7922dd5392255e71b81bf8d13d500b4854d554))

- End-to-end LoRA fine-tuning on a folder + generate-with-adapter ([8ab53235](https://github.com/swedishembedded/brain/commit/8ab532351cd848c62ba94f22370aafe0da6c26a2))

- Optional D-Bus control surface (com.swedishembedded.Brain1) ([fcc98eea](https://github.com/swedishembedded/brain/commit/fcc98eea719bfaa600c08edca952b8ee8a3364a8))

- Compile into the default build, serve only with --dbus ([80e11aac](https://github.com/swedishembedded/brain/commit/80e11aac03073fe302dabc0922c66bae01e13bb2))

- P0 - memory model (budgets, LRU, placement/eviction) ([3094aba5](https://github.com/swedishembedded/brain/commit/3094aba5500bc88ba34ee4f1ee4aac5cbd71c842))

- Memory-mapped safetensors (on-demand cold tier) ([4c657adf](https://github.com/swedishembedded/brain/commit/4c657adf81b3e6931870902d1d8373f9ffea9112))

- P1 - ResidentModel/Instance traits (the residency unit) ([3a7fc097](https://github.com/swedishembedded/brain/commit/3a7fc09791dff99615c55bc43401a6c9117727d3))

- P2 core - ResidencyManager (automatic promote/evict/LRU swap) ([8e49db4d](https://github.com/swedishembedded/brain/commit/8e49db4da4a34be4c94ff6bf0eae45df3f6f6f2d))

- P3 - general Executor + smart scheduler policy ([abafc890](https://github.com/swedishembedded/brain/commit/abafc8906bade9e69de023061583b66b4c7ec6af))

- Scheduler metrics (builds/evictions/batches) for validation ([3713a414](https://github.com/swedishembedded/brain/commit/3713a4142d2e53d44ffb28554320114f2611db26))

- Serve through the residency Executor (scheduler is live) ([516adef0](https://github.com/swedishembedded/brain/commit/516adef0b64bb05aa1f860fc3d1451065af43e1c))

- Yolo detection resident model (over the scheduler) ([23c94a52](https://github.com/swedishembedded/brain/commit/23c94a5279d607c5f6a448e33b25d2b494677d3d))

- Idiomatic, reusable Python client (brain_dbus_client) ([01329280](https://github.com/swedishembedded/brain/commit/01329280a46b47785f1d480e46f091c9f0bea9f1))

- Consolidate the reusable D-Bus Python into the package ([4bf08592](https://github.com/swedishembedded/brain/commit/4bf0859299e484f29ac4055dd3898b9bbe84f0c3))

- Fetch-yolov8.sh - pretrained YOLOv8 -> brain weights ([1c385180](https://github.com/swedishembedded/brain/commit/1c385180d6365cacc082014ffece60d8aa44b8ab))

- Full generate→detect→boxes pipeline over D-Bus ([deafdb1d](https://github.com/swedishembedded/brain/commit/deafdb1d2f18ee384af4fe269113d14c54aae0bd))

- Multi-device parallel lanes (dispatcher + per-device lanes) ([127fe684](https://github.com/swedishembedded/brain/commit/127fe68489de12783da832c840dee68596238bfb))

- True batched forward + pipeline emits every step ([09a32c19](https://github.com/swedishembedded/brain/commit/09a32c1920f7adc1b582629160a3aeff24fb6a68))

- Wrap gpt/glm/qwen/depth/tts as ResidentModels (P5) ([a816073a](https://github.com/swedishembedded/brain/commit/a816073a511f7fa4541bdbccd098f7e78d15d9b4))

- Portable GPU KV-cache decode for the Qwen3 block (P6 foundation) ([744318f5](https://github.com/swedishembedded/brain/commit/744318f59b8a9ce506318a91f42a9655b6d60d61))

- KV-decode tok/s benchmark (O(T) vs O(T2) recompute) ([b49e529e](https://github.com/swedishembedded/brain/commit/b49e529eb7dff4c06a0a3373c8490b486d24213c))

- Incremental GPU/CPU KV-cache decode step() (P6) ([dccdd7a2](https://github.com/swedishembedded/brain/commit/dccdd7a20485adec88e6339f70f92501bd67110f))

- Incremental GPU/CPU KV-cache decode step() (P6) ([14cb7d95](https://github.com/swedishembedded/brain/commit/14cb7d95f4b5508ac779d48562bb9e66ff404bf8))

- Incremental GPU/CPU KV-cache decode step() (MLA + MoE) (P6) ([54ebd795](https://github.com/swedishembedded/brain/commit/54ebd79526cfdf6701ada00a3182ca9a34f2c6d6))

- Incremental GPU/CPU KV decode (S1+S2, sliding window) (P6) ([eaa5cff5](https://github.com/swedishembedded/brain/commit/eaa5cff5dea147746627669c8603df9fd72c4d7c))

- KV step vs CpuTalker oracle parity + decode profiling (P6) ([f3fdfd91](https://github.com/swedishembedded/brain/commit/f3fdfd914757b5d752b3905814a4155c60ff2ed0))

- Persistent cached decode tape - ~2x faster KV decode (P6 superfast) ([7634cf1c](https://github.com/swedishembedded/brain/commit/7634cf1cc198583f953c75c97edd3f23ef22a699))

- Generate_kv - O(T) KV-cache generation via step() (P6) ([d4e65b8e](https://github.com/swedishembedded/brain/commit/d4e65b8efde908be882070e041cfbd3002847c54))

- Gpt, glm: generate_kv - O(T) KV-cache generation via step() (P6)

Mirror qwen's generate_kv: feed the prompt through step() then sample one token
per step (untied lm_head applied host-side to each hidden). Both validated
identical greedy tokens vs the O(T2) recompute generate
(generate_kv_matches_recompute_greedy). ([79de1743](https://github.com/swedishembedded/brain/commit/79de1743b88613d34c689fb5f66b9e17acdad1d2))

- Brain qwen/gpt/glm infer use the O(T) KV-cache generate_kv (P6) ([e5c94ca9](https://github.com/swedishembedded/brain/commit/e5c94ca949aed9bcd163b9fa230c05badd31fd8e))

- Collapse the double-engine - main generation uses the unified step() (P6) ([4f938ee0](https://github.com/swedishembedded/brain/commit/4f938ee09370502fb12ea144e734d5ab95b6addb))

- Paged KV-cache foundation - block allocator + tables + paged kernels (P7.1) ([dde2dc0f](https://github.com/swedishembedded/brain/commit/dde2dc0fbc2bbdee185b29ea5e6f5efacb008308))

- Batched ragged paged decode kernels (P7.2 core) ([50d8b5ea](https://github.com/swedishembedded/brain/commit/50d8b5ea14b4890be9a2a2e5e38029a69a95b1e7))

- Batched paged serving engine - concurrent multi-sequence decode (P7.2) ([0e58e10c](https://github.com/swedishembedded/brain/commit/0e58e10cc934c0523c227f98b4c757dc92b0eaf4))

- Continuous-batching scheduler + throughput benchmark (P7.3) ([3ebcc2bf](https://github.com/swedishembedded/brain/commit/3ebcc2bf6a2faa21e59e4b3be2a43029c7c122c8))

- Batched prefill - whole prompt in one causal forward (P7.4) ([881a7235](https://github.com/swedishembedded/brain/commit/881a72357b7c648270935e15d2072b66d7c77817))

- Int8 paged KV cache - ~4x smaller KV pool (P7.4) ([eb0978bc](https://github.com/swedishembedded/brain/commit/eb0978bc496c22a6cff1651f7847b96b9e5dd475))

- Chunked prefill via run_batched (P7.4) ([4b3d58fc](https://github.com/swedishembedded/brain/commit/4b3d58fc107aec8f226140b9cf2e997da9f202a8))

- Speculative decoding + BlockTable::truncate (P7.4) ([174b1909](https://github.com/swedishembedded/brain/commit/174b1909ab461e483c9f5281a88309ecd23bcf80))

- Embedding-input path - tts Talker multi-stream paged decode (P7.4) ([00cf610b](https://github.com/swedishembedded/brain/commit/00cf610ba634212b1ecb7c4abddf76a0b6f89cb9))

- Engine::load from checkpoint + brain qwen serve command (P7.3 tail) ([0e639dec](https://github.com/swedishembedded/brain/commit/0e639decc867e1ba65a9f9e03cbbaec52df88e54))

- Opt-in Vulkan validation + record the adapter ([0e8c773f](https://github.com/swedishembedded/brain/commit/0e8c773ff4dba9824b7874d6ea119ffdecee8151))

- --device selects schedulable compute, not just a backend

`--device` used to pick one process-wide backend from a fixed set of four
names. It now declares WHICH COMPUTE IS SCHEDULABLE, and everything else
(which Gpu a model builds, which residency budgets exist, how many CPU
threads run, whether the NPU path is allowed) follows from that set.

  (omitted)     every device present - all GPUs + CPU + NPU, together
  cpu | gpu | npu | vulkan       one class only
  gpu,cpu       comma-separated union
  gpu0          only that physical card (pins BRAIN_GPU_INDEX)
  cpu21         only that core        cpu0-7   inclusive core range
  gpu1,cpu0-3   mixed

An indexed CPU selection pins process affinity via sched_setaffinity and
sizes the rayon pool to match, so `cpu21` is genuinely one core rather
than one core's worth of threads spread over the machine. Out-of-range
indices are errors, never silent clamps.

This bounds where work EXECUTES; host RAM and disk stay available as
cache/spill tiers, so `--device gpu` still uses RAM for weight caching.
`brain serve --dbus` now budgets only the devices the spec allows, so the
residency scheduler cannot place work on excluded hardware.

Parsing is pure and total, resolution probes the machine, and the two are
separate so the grammar is testable without hardware (24 tests).

The NPU is inventoried and honoured when named, but a plain run will not
silently route to it: OpenVINO is a whole-graph compiler, so reaching it
needs a per-model export path that does not exist yet. ([6f0d2190](https://github.com/swedishembedded/brain/commit/6f0d2190f4498d6f59eef19f6286412512b1a093))

- Report admissions and per-step tokens (Scheduler::step_report) ([f242d27c](https://github.com/swedishembedded/brain/commit/f242d27c39bef017befb99beedb111aa54aa98b6))

- Run the LM head on the device - 2.0-6.4x decode throughput ([cc23d417](https://github.com/swedishembedded/brain/commit/cc23d417f7474a911cbbecc17f5503c6023bb58f))

- One implementation of the host math (model::hostmath) ([8aec6bec](https://github.com/swedishembedded/brain/commit/8aec6bec281370c032ea4dca23ab7451aa345590))

- Fix the test-suite deadlock AND the exit-time driver crash ([061202f6](https://github.com/swedishembedded/brain/commit/061202f6c0bd517e780aaec958d8b6c36afab00e))

- Rayon lives ONLY in the CPU scheduler (backend_cpu::par) ([564420b8](https://github.com/swedishembedded/brain/commit/564420b85ab7db917cc86d881e61e305b140e3bc))

- One graph-emission base (crate::topo::TopoBase) across topologies ([ab2edb87](https://github.com/swedishembedded/brain/commit/ab2edb873640254e1e318366c5a8b3b7a167ae4a))

- Make test: build without the timeout, run with it

The deadlock guard is a statement about RUNNING tests. A cold rebuild
after an engine change takes minutes on its own; letting it eat the
budget turned 'compiling' into a false 'TIMED OUT' that read like a hang
- which is exactly the confusion the guard exists to prevent. ([55bb0af9](https://github.com/swedishembedded/brain/commit/55bb0af946101e707a6508084fca51b8495b36e6))

- Size the stblock tolerance to its accumulation depth ([40b83dd6](https://github.com/swedishembedded/brain/commit/40b83dd6ad7015f0829da3579c93fa33d81005a7))

- Per-iteration prefill budget - decode runs between admissions ([4d756f58](https://github.com/swedishembedded/brain/commit/4d756f585d9dde2d06aa0ee842beab5dd9bcc4e7))

- Decode-regime matmul/rmsnorm/argmax + the fidelity gate - 19x ([e60ec815](https://github.com/swedishembedded/brain/commit/e60ec81515bd45c2d1a8664098ae2713b777a1ee))

- EvictionPolicy seam - CostAware (GDSF) alongside Lru ([dab2cdfd](https://github.com/swedishembedded/brain/commit/dab2cdfd700b81225f9fa4360b427b7df68234ef))

- AdmissionPolicy seam - the server can refuse impossible work ([57fb1c7b](https://github.com/swedishembedded/brain/commit/57fb1c7b74a1c2ac1ff99334882c174361cede12))

- DeviceCaps - a queryable device capability model (S1) ([2a3aff23](https://github.com/swedishembedded/brain/commit/2a3aff23316f43b2ab6fabab6e1193642f7f8408))

- KernelSelector - one policy for which kernel runs (S2) ([4111347f](https://github.com/swedishembedded/brain/commit/4111347f934d85375a5e004781a73dc1c70e6bd1))

- Int8 weights (A0) + warm start (F1/F2) + policy-comparing benchmarks ([c09be473](https://github.com/swedishembedded/brain/commit/c09be4736528689271799c65e3fd278f39ea2e61))

- On-device decode window - one readback per k tokens (A4) ([346917da](https://github.com/swedishembedded/brain/commit/346917dacba7dc655b3928797d97a1352c7ec1bf))

- Prompt-prefix cache - prefill computes only the unmatched tail (D) ([354ba0f1](https://github.com/swedishembedded/brain/commit/354ba0f1a6cc51c85697b0e3591a0f0d2b5ebd04))

- DeviceStats - queryable device-op accounting (K) ([b9515d83](https://github.com/swedishembedded/brain/commit/b9515d838c89c096837d8bb6ee4ab8057ab2ac74))

- Template specialisation - one source, tunable constants (S3) ([3534f37f](https://github.com/swedishembedded/brain/commit/3534f37f064992c85d38da0a85d81e485d82046c))

- AutoTuner - measure once per device, remember, persist (S5) ([d10c0957](https://github.com/swedishembedded/brain/commit/d10c09576c16c6bd8acf840e24cea149b74aac4a))

- Add Batch::Multimodal for vision-language inputs ([397e2d1c](https://github.com/swedishembedded/brain/commit/397e2d1c6f233a4f60f66793c1cb7f0f52c868a6))

- Residual embedding-splice seam for VLMs ([d0ebba79](https://github.com/swedishembedded/brain/commit/d0ebba7987757f12ba38aeea157e994e8d955850))

- Optional image-embedding splice in the decoder ([326e62b3](https://github.com/swedishembedded/brain/commit/326e62b3895baf76386065e4e8b56f13b727b678))

- Check_vlm_splice validates the VLM embedding-splice gradient ([7ff042c2](https://github.com/swedishembedded/brain/commit/7ff042c223442ed864554846603ebc9f317c1001))

- Qwen3-VL-4B config (ViT vision + reused Qwen3 text) with HF parser ([da8c61d3](https://github.com/swedishembedded/brain/commit/da8c61d3b15bec31fb89b95b7ab5f158c52f3eca))

- Interleaved M-RoPE host table builder (no new kernel) ([a311ae02](https://github.com/swedishembedded/brain/commit/a311ae02817cd82fdef0d6352bd927f97b41c9a3))

- Get_rope_index - 3-axis M-RoPE position bookkeeping ([8fdb088e](https://github.com/swedishembedded/brain/commit/8fdb088e7ec0fe6dd159e15af6f4ab236c11c014))

- Optional table-driven M-RoPE on q/k (Qwen3-VL) ([c2018891](https://github.com/swedishembedded/brain/commit/c20188915e3580bd0b19f9220ed51b1feff185bf))

- Check_qwen_mrope validates the table-driven M-RoPE decoder path ([039a24e9](https://github.com/swedishembedded/brain/commit/039a24e97052f8c19a0baa718a67b1d1331aac40))

- ViT vision positions + 2-D vision-RoPE tables ([e30f2225](https://github.com/swedishembedded/brain/commit/e30f2225f7edb172b0fef37aa7cdc4147373e37a))

- Bilinear pos-embed resampling indices/weights (parity-exact) ([05deac09](https://github.com/swedishembedded/brain/commit/05deac09df655e0829d396368669026903b497ef))

- ViT vision encoder GPU forward (reuses model::vit block builder) ([c52ebe67](https://github.com/swedishembedded/brain/commit/c52ebe67d97e30b156dcd6d383bc8fe38d380c2d))

- PatchMerger (2×2 merge → LN → Linear → GELU-erf → Linear) ([49300e08](https://github.com/swedishembedded/brain/commit/49300e08f5d4a6f9e6cb16b1cb57b9d67c006f9b))

- End-to-end composite forward (encoder → merger → spliced M-RoPE decoder) ([49768b3f](https://github.com/swedishembedded/brain/commit/49768b3f3c8b44a65507338584364854d1279568))

- DeepStack tap capture in the ViT encoder ([a8d4c6c9](https://github.com/swedishembedded/brain/commit/a8d4c6c930333116034f65b1805ae240880e0358))

- DeepStack residual add hook (Qwen3-VL) ([ed984953](https://github.com/swedishembedded/brain/commit/ed984953731534f0a49a718d8f0c816ffb76355a))

- Wire DeepStack into the composite forward ([6b490c1d](https://github.com/swedishembedded/brain/commit/6b490c1d9d69cb2f83aeec320ba0df58537fee47))

- Smart-resize + token-count preprocessing (parity-exact) ([094aba1e](https://github.com/swedishembedded/brain/commit/094aba1e41d349b048d2fe3728c96db3d609d584))

- Im2col patch packing + [-1,1] normalize (parity-exact) ([42c918dd](https://github.com/swedishembedded/brain/commit/42c918dd76905690a3734e7376f2a671f8d9a55b))

- Drop the ViT encoder's post-block norm (HF parity) ([be43c689](https://github.com/swedishembedded/brain/commit/be43c6894959eb8af6d92c648da2898e108a6471))

- HF checkpoint name mapping + partitioning ([ae6b739b](https://github.com/swedishembedded/brain/commit/ae6b739bdaabe3027cfd276ded95f3cd04741ec9))

- From_hf/from_tensors loader + real-checkpoint coverage ([de1e97f4](https://github.com/swedishembedded/brain/commit/de1e97f47a32fcb6e95149e2e90010627fd8056f))

- Crate scaffold + config (FastViTHD + Qwen2 decoder) with HF parser ([9b3ec521](https://github.com/swedishembedded/brain/commit/9b3ec5212877923f8e19b81909f685794acacf1e))

- Qwen2 decoder toggles (QK-norm off / qkv bias on) + check_qwen2 ([af6cb4cd](https://github.com/swedishembedded/brain/commit/af6cb4cd398227fe4b9ba1fe004175deec69a2ee))

- HF checkpoint name mapping (decoder + projector) + coverage ([83515d7b](https://github.com/swedishembedded/brain/commit/83515d7b3b026a888885d1a05ca3923a35d9b123))

- FastViTHD conv scaffolding (pipeline + Ctx + smoke) ([92c1d43c](https://github.com/swedishembedded/brain/commit/92c1d43c5484a4074b4eb880711022df5ba3b621))

- ConvUnit - the atomic FastViTHD conv primitive ([0e2ae356](https://github.com/swedishembedded/brain/commit/0e2ae35664597be539efe63695fbc9d822bbca56))

- ConvFFN + RepMixerBlock (FastViTHD stage-0–2 block) ([636783cc](https://github.com/swedishembedded/brain/commit/636783cc461c9a836e51653f7e01c9168091191d))

- PatchEmbed downsample + RepCPE (FastViTHD conv-stage blocks) ([c3921978](https://github.com/swedishembedded/brain/commit/c3921978302de5831b7690c063f1e4a15ea46808))

- AttentionBlock (FastViTHD stage-3/4, reuses vit attention core) ([124b021f](https://github.com/swedishembedded/brain/commit/124b021fe0f0e2f6069501f36866682da6d6d1b1))

- Assemble the FastViTHD encoder forward (stem→stages→conv_exp→tokens) ([774c3911](https://github.com/swedishembedded/brain/commit/774c391161d6110368d6b2cbb9154e63b8f6afe1))

- End-to-end composite (FastViTHD → projector → Qwen2 splice) ([8c454f71](https://github.com/swedishembedded/brain/commit/8c454f71c264e8aa66b190e4e41d1f6e9c310bb7))

- Geglu_shift MoE expert activation (+2 backward) gradchecked ([60288b24](https://github.com/swedishembedded/brain/commit/60288b2490b3123f2143f5c1d13fe5a81fc55472))

- Tau_scale attention-temperature kernels gradchecked ([86605197](https://github.com/swedishembedded/brain/commit/866051975a8931709134698ff3aec487d1c1360a))

- Adaptive_avgpool2d (+dx) gradchecked ([8edc4d6d](https://github.com/swedishembedded/brain/commit/8edc4d6daeb46e439bbd526781d3d8b49dbe558d))

- Prefix-LM mask + partial-RoPE (last new kernels) ([5922df13](https://github.com/swedishembedded/brain/commit/5922df1339648e56907db81499094e6c633c0c9a))

- Config (text + SigLIP vision + MoE) from config.py ([024bc4e4](https://github.com/swedishembedded/brain/commit/024bc4e41b7487edd2fd6360d6eb2fd06f82d48c))

- SigLIP ViT vision encoder (reuses model::vit) ([342db3c6](https://github.com/swedishembedded/brain/commit/342db3c6ba52a50fc2ab9ab8b4b21c2609d5f41a))

- Connector projection MLP (2304→8192→2048) ([38808e15](https://github.com/swedishembedded/brain/commit/38808e15dc8e99cae4c677a14dfae0aebe7c3699))

- Sparse-MoE GeGLU FFN (router + geglu_shift experts) ([6ba131d2](https://github.com/swedishembedded/brain/commit/6ba131d278a080f661e969f09febd7f427a55392))

- Parallel attn+MLP decoder block forward ([f5d188ec](https://github.com/swedishembedded/brain/commit/f5d188ec8192f88a0ae366177b026d23b53616c4))

- Wire the MoE FFN into the parallel block (dense | MoE) ([31533176](https://github.com/swedishembedded/brain/commit/315331766ace0b94b167afdef041773e607c8d28))

- Full decoder stack forward (embed → blocks → head → CE) ([7d7002f2](https://github.com/swedishembedded/brain/commit/7d7002f260a92feb70414c0488d2d961cb4775dd))

- End-to-end composite (SigLIP ViT → connector → spliced MoE decoder) ([80f26c9d](https://github.com/swedishembedded/brain/commit/80f26c9dab4b003ca87ed90bbfba71ba598d0d55))

- Overlap multi-crop reconstruct + adaptive-pool → global‖local concat ([87650390](https://github.com/swedishembedded/brain/commit/876503906d03f2d153b628aebfd6765d4bc7c83b))

- Per-head attention temperature (tau) on q,v in the block ([d5b467ee](https://github.com/swedishembedded/brain/commit/d5b467ee2b6bd355e0af3ca43febb2d13f54f3fe))

- HF checkpoint importer (662-tensor map + MoE expert split) ([f32255a8](https://github.com/swedishembedded/brain/commit/f32255a80fb3a81bdd63fca460d014c7301fbc24))

- Faithful multi-crop composite forward (global‖local → tau MoE decoder) ([514cd293](https://github.com/swedishembedded/brain/commit/514cd29388d89f894cc220b97ee818dcedbc17aa))

- Move test-only BLOCK_LEAVES into the test module (clean build) ([3364dac3](https://github.com/swedishembedded/brain/commit/3364dac388d34d52727d91011c5d6f1a09f1ae2f))

- Dense parallel-block backward + finite-diff gradcheck ([e4112a6c](https://github.com/swedishembedded/brain/commit/e4112a6ceb1e5f3ec0ed2b1bd8cdccbdb9456d91))

- Dense decoder backward + end-to-end check_moondream gradcheck ([8f12c85e](https://github.com/swedishembedded/brain/commit/8f12c85e9b66ed26d04b3bc79c9300889cbb1dde))

- Tau attention-temperature backward + gradcheck ([95e647f6](https://github.com/swedishembedded/brain/commit/95e647f6f09fd75b0b452624aaad5db3200ec70b))

- MoE-expert backward + gradcheck (completes the block backward) ([fdb48e41](https://github.com/swedishembedded/brain/commit/fdb48e41385edf709f513e04aaaf56b8636c7ff7))

- Full-architecture decoder gradcheck (tau + MoE) - gradient-faithful ([126a3dcb](https://github.com/swedishembedded/brain/commit/126a3dcbcd59cef354a5654c84ad25ba6e7b029b))

- SigLIP ViT training fwd/bwd + gradcheck (vision-tower backward) ([d0670a5f](https://github.com/swedishembedded/brain/commit/d0670a5f7cb5ce7d377a78353a16786b88549712))

- Qwen3-VL ViT training fwd/bwd + gradcheck (2-D vision RoPE) ([0ba52a44](https://github.com/swedishembedded/brain/commit/0ba52a444cc1dc599b856731543ba5cd6e2fd5d3))

- ConvUnit backward + gradcheck (FastViTHD conv-tower foundation) ([e2b49d12](https://github.com/swedishembedded/brain/commit/e2b49d12574ec347afc7379af5db54bec58e921f))

- ConvUnit conv+BN+bias+GELU backward gradcheck (training BN) ([4f2e15f8](https://github.com/swedishembedded/brain/commit/4f2e15f8087887789f091cbe0def2646b91bc60a))

- ConvFFN backward + gradcheck (FastViTHD channel-mixer) ([a3238db9](https://github.com/swedishembedded/brain/commit/a3238db92004db6f4720b7c05202f6d7313c886a))

- RepMixerBlock backward + gradcheck (FastViTHD stage-0-2 block) ([9231d009](https://github.com/swedishembedded/brain/commit/9231d009298c4f761e2de3f15870983088cb85db))

- PatchEmbed backward + gradcheck (FastViTHD downsample) ([0e36ed74](https://github.com/swedishembedded/brain/commit/0e36ed746e563c0f07d5b62240e200849ad1fc61))

- MoE expert-sharding weave + split/assemble round-trip test ([217c5ef9](https://github.com/swedishembedded/brain/commit/217c5ef97c397bc4f507d8e71a907bc8531e69c7))

- AttentionBlock + Encoder backward + gradchecks (FastViTHD tower complete) ([7e1b1267](https://github.com/swedishembedded/brain/commit/7e1b1267ddead733426b7c1738106bbbf7ac21d7))

- Real-weight decoder parity vs HuggingFace (mean|Δ|~7e-6) ([a3bfbdff](https://github.com/swedishembedded/brain/commit/a3bfbdfff251132969beaf9fe4c78c7547aedb29))

- Greedy-generation parity - brain decodes 'Red, Blue, and Yellow.' matching HF ([6c33eab0](https://github.com/swedishembedded/brain/commit/6c33eab04b6a88d6747909930e0ebce3bf3bef67))

- Image-caption parity - brain captions a real image matching HF token-for-token ([2ad24165](https://github.com/swedishembedded/brain/commit/2ad241651db20b4bca27478192a3d51c492ad9e3))

- VLM training-convergence smoke (overfit → loss 3.12→0.01) ([c7a20c0d](https://github.com/swedishembedded/brain/commit/c7a20c0d55e1faf230225c3ded9c6da0066f24dd))

- Conv_exp SE-gating + mobileclip_l config (FastViTHD-L structure) ([724e23a0](https://github.com/swedishembedded/brain/commit/724e23a044c0b46d014717a68721c1cf2a5ea7ed))

- Mobileclip_l vision import + full param coverage vs real checkpoint ([b41d34bc](https://github.com/swedishembedded/brain/commit/b41d34bc180d57704617055d6dd494765933d5f1))

- Brain's own FastViTHD vision tower matches HF (cosine=1.000000) ([79a005bd](https://github.com/swedishembedded/brain/commit/79a005bd74d7110bc6fdefaa7ac1a321e68d1b8c))

- Qwen3-VL partial-depth decoder parity harness (streamed real weights) ([55b41422](https://github.com/swedishembedded/brain/commit/55b414220e95f53eb7aa2bed0815548832f789f6))

- Fully-in-brain image caption matches HF (vision + projector + decoder) ([86d3a0c2](https://github.com/swedishembedded/brain/commit/86d3a0c2843433ef9236c859d7a5f36d0013b984))

- Qwen3-VL 4-layer decoder parity passes (mean|Δ|~5e-6) + moondream logits accessor ([c266d8df](https://github.com/swedishembedded/brain/commit/c266d8dfd9a72c9322c2a3eb846b899d93ab201b))

- Partial-depth decoder parity harness (streamed real weights) ([5a9546f0](https://github.com/swedishembedded/brain/commit/5a9546f08e46f03faabdd32d171e40da37b75c2e))

- Parity diagnostic - report per-position max|Δ| before argmax check ([bbce86d9](https://github.com/swedishembedded/brain/commit/bbce86d972fbb1e14d4924fd4609851721186d5f))

- Fix the mtp_bench example without breaking the rayon invariant ([10f82800](https://github.com/swedishembedded/brain/commit/10f82800b148290e4de6bbc37d16aef79a310b8b))

- Provider for qwen, yolo, depth, tts (L) ([03a49ed2](https://github.com/swedishembedded/brain/commit/03a49ed281cd5277eeec84717210f8f69d0c3170))

- Drop an import left dead by the DeviceStats rework ([e26fd62c](https://github.com/swedishembedded/brain/commit/e26fd62c59d037143a5ba7745748cae7f37874d4))

- Make test: default TEST_THREADS to 8 - the pooled-device payoff

Every GPU-test crate now either shares the per-binary pooled device
(gpu_core::testgpu), pins the CPU backend, or serializes its
deliberately-multi-device tests behind a per-binary lock - so the
reason the default sat at 1 (concurrent tests stacking real devices on
the card until the driver deadlocked) no longer exists. 8 is the
proven point: qwen, gradcheck and the six heaviest migrated crates are
clean there on the two-P40 box. TEST_THREADS=1 make test restores the
serial lane; the timeout guard still turns any regression into a fast,
loud failure rather than an hour of silence. ([526786b3](https://github.com/swedishembedded/brain/commit/526786b3ff3e3b44ae680f5c3afe3c518a4a9c26))

- Fix missing fused-qkv bias - real-weight parity now passes (mean|Δ|~5e-6) ([0581928e](https://github.com/swedishembedded/brain/commit/0581928e9a744b516b27be2e66318ec5bc0d8b86))

- Fused-qkv bias backward + gradcheck (gradient-faithful finetune of the real model) ([9dae4ce3](https://github.com/swedishembedded/brain/commit/9dae4ce30b6e07f217efd7020bd4bfa5c037894f))

- Capability Provider - image captioning, fully in brain ([663d6526](https://github.com/swedishembedded/brain/commit/663d652682b372864c023a61a65f7e0cf040e544))

- Fastvlm behind the scheduler + dbus - validated end to end ([91f083b3](https://github.com/swedishembedded/brain/commit/91f083b308898b0a1b4280d9992c7b47a439d0e7))

- Generic KV decode for VLMs + int8 KV decode; fastvlm 3x ([33c59982](https://github.com/swedishembedded/brain/commit/33c59982be118b025109e12fbc6bbbc09b92dbe8))

- Profile the resident path, then three simple decode fixes ([f6b1e5e1](https://github.com/swedishembedded/brain/commit/f6b1e5e1a33ae359735c3485fd6e205c445e4d6f))

- Log-mel front ends for Nemotron & Qwen3-ASR (parity-gated) ([91a20b96](https://github.com/swedishembedded/brain/commit/91a20b96322cfc0fff22873c8e09cfb2d06fa81c))

- Audio encoder + projector (parity-gated vs HF) ([6c070940](https://github.com/swedishembedded/brain/commit/6c070940b94f22b9d6fa9b5e15172bdd80082dbc))

- End-to-end transcription matches HF exactly ([13669985](https://github.com/swedishembedded/brain/commit/13669985f0e47a73c867c6ee297a45b657f4ed4f))

- GLU + LSTM-gate kernels (fwd+bwd, gradchecked) ([93967f3e](https://github.com/swedishembedded/brain/commit/93967f3ea712594164254aae1a4dd8cb48a9725b))

- Rel_shift kernel (Transformer-XL rel-pos), fwd+bwd ([d9d591a9](https://github.com/swedishembedded/brain/commit/d9d591a9e7c440bcd055c1798c9e3c77c3c572a8))

- FastConformer subsampling (parity-gated) + config + goldens ([a548ed31](https://github.com/swedishembedded/brain/commit/a548ed3141c2b4edb94c2aad25f7b430a1dc4ccb))

- FastConformer encoder reference (parity-exact) ([575f97f3](https://github.com/swedishembedded/brain/commit/575f97f3745cccaf2f29edb03a984e9f261983c0))

- RNN-T predictor + joint + greedy decode (HF-exact) ([26d3e011](https://github.com/swedishembedded/brain/commit/26d3e01179059bcf95fb6e3a59c6b3ab864858a8))

- Device FastConformer encoder (160x faster, parity-exact) ([1717f0ef](https://github.com/swedishembedded/brain/commit/1717f0ef1366a570dd6bf10d0eb108d86676e67e))

- End-to-end model (frontend → device encoder → RNN-T) ([5da2addd](https://github.com/swedishembedded/brain/commit/5da2adddad724b32683eb0c6efb8f88c79cb9063))

- Device RNN-T joint head + tidy greedy loop ([a40b2983](https://github.com/swedishembedded/brain/commit/a40b2983bc23f2772a78e9a2e890bf8a330424b7))

- Parallelize per-head attention scoring + stage timing ([fc69ee06](https://github.com/swedishembedded/brain/commit/fc69ee0600f575a7d4f2b276d241f40049cea001))

- Batch macaron FF on device (encoder ~25% faster) ([a4cd56e4](https://github.com/swedishembedded/brain/commit/a4cd56e40c546be2cb45d3641410fba72b6995c5))

- KV-cache prefill+decode (3.6x faster, still HF-exact) ([b6377a98](https://github.com/swedishembedded/brain/commit/b6377a98cebb98b5572960d538bffc5f2e0db978))

- Run encoder on Intel Arc iGPU via Vulkan (parity-exact) ([87c7ce0d](https://github.com/swedishembedded/brain/commit/87c7ce0d0e301fe7ac240fc154411a8367304c5b))

- RNN-T joint-network backward + gradcheck ([ef14eaae](https://github.com/swedishembedded/brain/commit/ef14eaae779e399fc7de23b880a10bbfb8db1eeb))

- Rel-pos attention backward + gradcheck (the crux) ([204af6ad](https://github.com/swedishembedded/brain/commit/204af6ad33398b0cc4203816d3a9f392ed27d5fe))

- Conv-module backward + gradcheck ([99b0f626](https://github.com/swedishembedded/brain/commit/99b0f626c667d888f63e8edb5b25e52fcba84f25))

- Full Conformer-block backward + gradcheck (the repeating unit) ([65e350aa](https://github.com/swedishembedded/brain/commit/65e350aad22f0f1b89cefeea667d7d740fb724fd))

- Full encoder backward (model-level gradcheck) ([44ed7144](https://github.com/swedishembedded/brain/commit/44ed7144999b235f672e073df852ab4c5716f565))

- RNN-T transducer loss + LSTM predictor BPTT (gradchecked) ([58369834](https://github.com/swedishembedded/brain/commit/5836983479a36b9be12fc9efc8f1e08d811f6c0b))

- On-device FF backward gradcheck (device training path) ([f4a6eec2](https://github.com/swedishembedded/brain/commit/f4a6eec25dfd11ef24d95359c37ce00dd912def2))

- Trainable RNN-T Transducer as model::Model (gradchecked) ([4972b475](https://github.com/swedishembedded/brain/commit/4972b475672febe25f41615f77f28e875a0a09c0))

- Full trainable AcousticModel (Conformer+RNN-T) as model::Model ([04c6da65](https://github.com/swedishembedded/brain/commit/04c6da65728fbb2a56a48d28860d612bab821f6c))

- GQA head expansion, accumulating chunked-backward, row scatter ([79025ad0](https://github.com/swedishembedded/brain/commit/79025ad099a5e7c764dd036988f9382a1a97276a))

- :block: shared bidirectional-attention builders; migrate seq2seq/qwen/vit ([091a0686](https://github.com/swedishembedded/brain/commit/091a06868fdc6913ada3c23e2d8a105354cc2bdf))

- HF tokenizer digit-run autodetect + template prefix; MLM corruption ([c808c1c5](https://github.com/swedishembedded/brain/commit/c808c1c5b770122bd08da5ba2c64f9eba091d834))

- First-class NPU device in budgets and placement ([498456d5](https://github.com/swedishembedded/brain/commit/498456d5220c99892cd8e687da108e728059d158))

- LFM2.5-Encoder (230M/350M) - parity-exact import, 8k inference + training ([a168502d](https://github.com/swedishembedded/brain/commit/a168502deb829d42d843791e0787b0af72b941db))

- Check_lfm - the LFM encoder joins the backprop gate ([e9f4578f](https://github.com/swedishembedded/brain/commit/e9f4578f24633e783106a3b1a5a110cbba62315c))

- MLM pseudo-perplexity + masked-token accuracy ([028caff6](https://github.com/swedishembedded/brain/commit/028caff6b630172e99db19af7cca0cadd66fd2f9))

- Serve the LFM encoder - CLI verbs, capability surface, D-Bus residency ([04ab52fb](https://github.com/swedishembedded/brain/commit/04ab52fb70d0688d06d8b80f49719365443c1488))

- LFM2.5-Encoder ONNX export + LfmSession; hoist the linear emitter ([1618e4b7](https://github.com/swedishembedded/brain/commit/1618e4b7356305b3f95b804085b140a28c778609))

- Native fast paths for the cross-attention trio (6.9x forward) ([fdacd85f](https://github.com/swedishembedded/brain/commit/fdacd85f469962aad500b712eef91e5a3e6248bb))

- :block: GEMM attention + workgroup-row softmax (8k GPU 22s -> 7.5s) ([2f92d37d](https://github.com/swedishembedded/brain/commit/2f92d37d1b7c040e8600808f49050e6d7524bdf8))

- Step carries a neutral StepMeta (kernel, params, threads) ([4c79303d](https://github.com/swedishembedded/brain/commit/4c79303d0d34ea3f05ae61cb3c13cfd1a2bc1939))

- Per-kernel FLOP/OPS cost registry + offline/online accounting ([4ea3a73b](https://github.com/swedishembedded/brain/commit/4ea3a73bf7eb2bc4818a8496593b81b56e7506cb))

- Expose recorded-graph costs; pin coverage + agreement ([e72cdc3d](https://github.com/swedishembedded/brain/commit/e72cdc3dbcfdf7d597b314ace34645080b13a228))

- Brain flops - offline + online FLOP/OPS reports; docs ([84709b7a](https://github.com/swedishembedded/brain/commit/84709b7a6a1da273215a753cbeabbff984b2c39b))

- Build-once + batched encoder; migrate test assets to testdata/ ([68f80d89](https://github.com/swedishembedded/brain/commit/68f80d89bfe8c9c1e14930470f3e114f75431102))

- Remove all absolute paths from source (repo-wide); testdata/ + env-driven

Enforces the new AGENTS.md invariant across every crate - no machine-specific
absolute path may appear in source (code, tests, defaults, or doc comments).

Test/parity fixtures now resolve at runtime from $BRAIN_TESTDATA (default
<repo>/testdata) via per-crate `testdata()`/`repo_path()` helpers; each test
still skips itself when its fixture is absent. `make fetch/testdata` populates
the gitignored, tree-organized testdata/ from local mirrors (asr/vl/tts) -
hard-linked, only what's missing.

Runtime paths now come from env vars, never baked-in literals:
  * tts_cli   --ckpt / $BRAIN_TTS_CKPT
  * tts_serve $BRAIN_TTS_RES, socket via std::env::temp_dir()
  * fastvlm   $BRAIN_FASTVLM_WEIGHTS
Stray /tmp/ literals (tts test scratch, events sample, tts socket) now use
std::env::temp_dir(). Doc-comment examples (qwen QWEN3_DIR, wm-diamond ref)
genericized.

Touched: audio, fastvlm, moondream, qwenvl, tts, speaker, codec, wm-genie,
events, cli. Grep gate `grep -rnE '"/(data|home|tmp|opt|mnt|root)/' crates`
is empty; all touched crates' test targets compile. ([a7de6538](https://github.com/swedishembedded/brain/commit/a7de653842d9a1317ecf3fe6bc17f80f0f10a76b))

- ASR serving: capability + residency + batched run_batch + streaming D-Bus + example

Wires Nemotron 3.5 ASR and Qwen3-ASR through brain's full serving stack, end-to-end
validated (mic/wav -> pipe fd -> D-Bus StreamTranscribe -> Executor -> segment frames).

Capability (shared contract in audio::asr_caps - one implementation, both models):
  * transcribe action: audio blob (raw mono f32 LE 16 kHz) in, text out, streaming.
  * nemotron::caps NemotronProvider + a metaspace-BPE detokenizer (nemotron::tokenizer)
    from tokenizer.json.
  * qwen_asr::caps QwenAsrProvider: fixed-window (probes the encoder's n_audio at
    load - the chunked packing is non-analytic - then assembles the decoder and
    splices the fixed chat template; from_hf_windowed). Detok via the Qwen BPE.

Residency (cli::resident_asr, registered in build_executor, env-gated
BRAIN_NEMOTRON / BRAIN_QWEN_ASR): both build the model ONCE in activate. Nemotron's
run_batch is a TRUE batched forward - concurrent same-prompt stream-windows encode in
one FastConformer pass (Encoder::transcribe_batch). Qwen3-ASR is offline/autoregressive
(sequential run_batch on a build-once fixed-window instance), documented as such.

D-Bus (crates/dbus): new StreamTranscribe(model, params, pcm_fd) -> (job, event_fd).
A server thread reads continuous f32 PCM from the pipe, slices it into window_ms
windows, submits each as a transcribe Job (so concurrent streams batch + schedule
uniformly), and streams `segment` frames back over the SEQPACKET, ending with `done`
+ the full transcript. New StreamTx::segment frame; client method in brain_py.dbus.

examples/asr: transcribe_mic.py (live mic or --wav), bench_streams.py (N concurrent
streams -> RTF/latency/throughput + scheduler batch counters), README.

Every-new-model serving contract now mandated in AGENTS.md and docs/serving-contract.md
(capability + residency + batching + D-Bus wiring + example; fit the surface or extend
it, never a side channel). ([62234b47](https://github.com/swedishembedded/brain/commit/62234b47684d1bbc4b29b5d7a1e70edc3c19a1af))

- End-to-end perf report + Qwen3-ASR transcription-only detok ([ced7c6bf](https://github.com/swedishembedded/brain/commit/ced7c6bf5d115c088bb719da04db6c08c7878302))

- NPU as a first-class schedulable compute target (residency + generic seam)

Makes the Intel NPU a device the residency scheduler AUTO-PLACES models on, like a
GPU - closing the gap devices.rs names ("transparent NPU scheduling needs the
per-model export path first"). Design in docs/npu-residency.md.

Residency:
  * Device::Npu(u32) + MemCost.npu (a model advertises an NPU path with npu > 0;
    MemCost::new keeps npu = 0, so non-NPU models are never placed there).
  * place::pick_device / plan_eviction: device-class preference NPU → GPU → CPU,
    generalised over the cost field (Budgets::npus()). Unit-tested
    (npu_capable_model_prefers_the_npu).
  * The Executor auto-spawns an NPU lane once the NPU has a budget - no executor
    change needed.

Generic reuse seam (crates/npu):
  * NpuModel trait - the ONE per-model NPU contract: build the OpenVINO graph (via
    the shared topo/onnx blocks) + a cache_key; default onnx_bytes/compile.
  * openvino::NpuGraph - a generic named-tensor runner (f32/i64 in by name, f32 out
    by name) generalising the bespoke per-model sessions, so compile/cache/infer is
    model-agnostic. Hardware smoke (tests/npugraph.rs): a MatMul+Relu graph compiles
    and runs on the real NPU (maxdiff 3e-5). Confirms the Rust openvino 0.11 binding
    is ABI-compatible with OpenVINO 2026.2.
  * ensure_openvino_on_path now auto-creates the pip wheel's missing UNVERSIONED
    lib*.so symlinks (2026.2 ships only versioned files) so `--device npu` works out
    of the box.

Wiring + proof:
  * build_executor + `brain serve` discover/budget the NPU and pass it through.
  * DepthResident is the first NpuModel: activate(Device::Npu) compiles ZipDepth via
    NpuGraph; the only depth-specific NPU code is one `build` method (~5 lines) +
    image resize glue. PROVEN end to end: `brain serve --dbus --device npu` auto-
    places depth on the NPU (`depth: compiled ZipDepth 384x384 on NPU`; result
    device=npu; builds:1). Every future model that ships an NpuModel gets the same. ([8ab4250d](https://github.com/swedishembedded/brain/commit/8ab4250d5827d6c35ecf10791a35a0bfe35e7e68))

- Nemotron NPU: FastConformer subsampling ONNX topology (golden-parity)

First stage of the Nemotron encoder as an OpenVINO-compilable ONNX graph
(crates/npu/nemotron_topology.rs), so the streaming ASR encoder can run on the NPU
via the generic NpuModel/NpuGraph seam. Depthwise-separable causal subsampling (×8):
causal Conv2d ((k-1,s-1) asymmetric pads matching NeMo) → per-stage time-mask + ReLU
→ transpose/reshape → linear. The macaron Conformer blocks + rel-pos attention +
projectors land next (each parity-gated against its golden).

Parity: compiled on OpenVINO CPU (fp32) vs the dumped HF golden - maxdiff 7.3e-4 over
the valid frames (tests/nemotron_subsampling.rs; skips without OpenVINO/testdata).

Gotcha fixed during bring-up: the depthwise conv of each separable stage must NOT
apply mask/ReLU - those come only after the pointwise conv (verified against a torch
reference that matches the golden bit-for-bit on valid frames). ([30ae91ae](https://github.com/swedishembedded/brain/commit/30ae91ae8511c025e925d2198c816ecbb658b8ce))

- Nemotron NPU: verified full-encoder python-onnx prototype (reference)

Golden-parity reference for the FastConformer encoder ONNX port: subsampling +
24 macaron Conformer blocks (rel-pos attention with rel_shift + banded mask, GLU
conv) + projectors. block0 validated vs golden at 2.67e-5 on OpenVINO CPU; the
Rust topology (crates/npu/nemotron_topology.rs) ports this stage by stage. ([9f34ef56](https://github.com/swedishembedded/brain/commit/9f34ef562a06f1b7df26d574fa80253cf6df0314))

- Regenerate after rebase onto origin/main ([4b79c6a6](https://github.com/swedishembedded/brain/commit/4b79c6a6c4012dc7a2f76d0e292b79d17432a0a0))

- Stop tracking binary .safetensors goldens; gitignore model blobs ([b05dbc22](https://github.com/swedishembedded/brain/commit/b05dbc222eb54085bfd6d318fbab68ae809d3722))

- Cost formulas for the fused-attention and head-packing kernels ([a38acbb7](https://github.com/swedishembedded/brain/commit/a38acbb727130cbd174000bef1ca4fbbac117410))

- Record the P40 chunked-attention bind-offset failure in the ledger ([829655de](https://github.com/swedishembedded/brain/commit/829655dea08625f44b9238a03ea5e5138e910d2f))

- Streaming Nemotron mel front end, bit-identical to offline ([45bc9ce6](https://github.com/swedishembedded/brain/commit/45bc9ce6abcd7165e7d07ec6d46f3b129820fc8f))

- Frame-synchronous streaming encoder + stateful RNN-T decode ([a9b0021e](https://github.com/swedishembedded/brain/commit/a9b0021e5c665959c114b4dcf41aea2b6d31e238))

- ASR serving: live transcribe_stream sessions (caps + residency + D-Bus)

Streaming transcription joins the generalized serving surface as a
transcribe_stream action (schema shared in audio::asr_caps): a session id
names per-stream state created on first use, eos flushes and closes it, and
each call returns the session's newly emitted text/tokens -- segments are
exact deltas of one growing transcription, not independent per-window
guesses. nemotron::caps::StreamSessions is the one implementation behind
both the direct Provider (event API) and the resident instance, whose
run_batch steps every concurrent stream's window through one batched
encoder pass; idle sessions are reaped after 10 min, and eviction drops
live sessions (a restarted id starts fresh).

D-Bus StreamTranscribe auto-upgrades: a model whose manifest advertises
transcribe_stream is driven as one live session (no per-window re-encode,
EOF sends the flushing eos step) while offline models keep the independent
per-window fallback -- qwen-asr unchanged. Verified live over the bus:
serve + transcribe_mic.py emit incremental sub-word deltas that concatenate
to exactly the offline transcription; a checkpoint-gated test asserts the
delta contract. Status ledger and example README updated. ([f9766d3d](https://github.com/swedishembedded/brain/commit/f9766d3d940d10c90e52f24979b2e1506fd41723))

- GGUF v3 reader for unquantized tensors (F32/F16/BF16) ([8ee440e4](https://github.com/swedishembedded/brain/commit/8ee440e49aa887f288bb26d790b2a7d0292d61f3))

- FLUX.2 Klein reference golden dumper ([18dcbf88](https://github.com/swedishembedded/brain/commit/18dcbf88749a8ab89cfd235dcb85ecbd1a22319a))

- Additive key-mask GQA scores for padded encoders ([1cca9235](https://github.com/swedishembedded/brain/commit/1cca9235d5ce2a847c4966315e2016b0be5eb476))

- Qwen3_8b preset, one-pass multi-tap encoder, masked-pad path ([2c8a8235](https://github.com/swedishembedded/brain/commit/2c8a823596d8eebe8e0de88647b73505826bda7c))

- FLUX.2 autoencoder - quant convs, latent pack/unpack, eval BatchNorm ([c9e61e38](https://github.com/swedishembedded/brain/commit/c9e61e3815140f82ffbd8d6d91fd67af2ba36529))

- Crate scaffolding, config manifest, BFL/diffusers/GGUF import ([39566d30](https://github.com/swedishembedded/brain/commit/39566d302b06192574d07b78705b9fea23dc5660))

- Parity-proven DiT forward (cosine 1.000000 vs reference) ([c1070bc3](https://github.com/swedishembedded/brain/commit/c1070bc3a6e5544f9caa399d6de8295ae160e956))

- FLUX.2 empirical-mu exponential shift; model: shared randn ([7ee9cb20](https://github.com/swedishembedded/brain/commit/7ee9cb205cf8ee8b27423e7706014c19a054860a))

- Sampling pipeline, ref-image editing, brain flux2 CLI ([a7b6ce5e](https://github.com/swedishembedded/brain/commit/a7b6ce5e27ed529d9bef0b30b79d312e401eafc1))

- Training - FD-gated backward, sliced LoRA, finetune; data: imageset ([84a2e485](https://github.com/swedishembedded/brain/commit/84a2e4858cf3eb7701c54f1b15e0b3acc8268d32))

- Cooperative cancel, artifact blobs, blob hoist ([c5128eea](https://github.com/swedishembedded/brain/commit/c5128eeafb2afa9c9a78fb51daa8cd68ec225aa0))

- Serving surface - caps, residency, D-Bus examples ([14187ebc](https://github.com/swedishembedded/brain/commit/14187ebc668ea63d5555a9e9db92c9aaa350ab68))

- Canonical device registry - identity-keyed GPU selection ([fef7724b](https://github.com/swedishembedded/brain/commit/fef7724bc748ce23fca196d34b91a5fbb729e6cb))

- Int8 (DP4A) DiT - single-card serving; quantizer hoisted to model::int8 ([8365757d](https://github.com/swedishembedded/brain/commit/8365757dfa130079a99c31fe4270585754372c9b))

- Coalesced workgroup-per-row LayerNorm (layernorm/ln_stats/layernorm_dx) ([6559bc7f](https://github.com/swedishembedded/brain/commit/6559bc7fe570937cbaf8de80ea4a64aefce3b988))

- Cooperative grad-norm - 87.2% of GPT training GPU time, 2122x ([dba35181](https://github.com/swedishembedded/brain/commit/dba3518161834c9313b6b0b80b0ef0fe4029c00e))

- Cooperative max_abs_row - 6.5% of the FLUX.2 int8 forward, 9.4x ([11a4bd21](https://github.com/swedishembedded/brain/commit/11a4bd2114e96c1315e3a8e3a336fb97e553d936))

- True batched inference - bit-identical, and worth 4.4% ([adba36da](https://github.com/swedishembedded/brain/commit/adba36da4ef5ff05655123a16a11c7641c7a69d5))

- Brain npu lfm-bench - LFM2.5-Encoder NPU timing + parity across quants ([b360ec0d](https://github.com/swedishembedded/brain/commit/b360ec0d5fc5fcea57c3969663244784eb55211d))

- Serve chronos2/fincast/kronos over D-Bus + scheduler (NPU-placed) ([4df0c7f3](https://github.com/swedishembedded/brain/commit/4df0c7f39b8fa4718428dd45130ad4e9a037f141))

- Run all three models on the NPU (kronos rollout + fincast external export) ([4e59f1d3](https://github.com/swedishembedded/brain/commit/4e59f1d342b05c529bd9cf74f9812a3914f70213))

- Oos eval: full-universe sharded walk-forward with verdict + subsets

Scales the date-keyed oos_skill_eval harness to the full ~480-name SP500
protocol and retires the old rankic_eval/backtest.sh orchestration (index-
keyed origins silently misalign calendars once names' bar counts differ):

- oos_skill_eval.rs: OOS_SHARD="i/n" name-sharding (deterministic index
  modulo after the sort; date-keyed records merge back into full cross-
  sections); KRONOS_FT loads a fine-tuned decoder as model "kronos_ft"
  evaluated in the SAME sweep as base kronos (identical origins/panels =>
  honest paired comparison); JSON checkpoint every 10 origins instead of
  every origin (the full-list rewrite was O(origins^2) at 480 names).
- tools/oos_shard.py: build-once orchestrator - cargo builds the test
  binary, exports ^gspc from stocks.db, fans N pinned shard processes,
  merges, scores. Turns the ~45h serial full-SP500 sweep into hours.
- tools/merge_records.py --concat: same-name models across inputs are
  shards - records append, mase/latency recombine n-weighted. Default
  cross-model common-universe mode unchanged.
- tools/prep_backtest_data.py: full universe by default (--names 0);
  ft/holdout split is now liquidity-STRATIFIED seeded random (median 20-bar
  dollar volume as of T0, half of each decile each way) instead of
  alphabetical; writes split_manifest.json (seed, T0, per-name decile +
  assignment); protocol-sized defaults (bt 400 bars, embargo 10 = 2x
  horizon, not the old arbitrary 60 that burned a quarter of the window).
- tools/oos_skill_report.py: per-subset RankIC (ft-names vs holdout-names
  from the manifest - a fine-tune is graded on names it never saw),
  --k-frac (K as a fraction of the week's cross-section), cost stress
  column, Newey-West stderr when step<horizon overlaps labels, paired
  ft-base promotion gate, a mechanical VERDICT block over the
  pre-registered criteria, and --summary-out writing the trademiner-
  compatible backtest_summary.json superset.
- tools/full_backtest.sh: prep -> gated fine-tune (<= T0) -> sharded sweep
  -> report, one command.

Verified: 16-name/4-origin run sharded 2 ways merges to a record set
IDENTICAL to the unsharded run (64/64 records, bit-equal preds); shuffled
negative control sits at +0.015 +/- 0.125 = 0 as required; stratified
manifest round-trips through the report's subset scoring. Measured
~3.7 s/forecast (ctx 120, nsamples 1, cpu) => full 52-week base+ft sweep
projects to ~7-8 h on 20 shards. ([20021c24](https://github.com/swedishembedded/brain/commit/20021c24c58348cc9a65f823fbf52bb18bbedf6f))

- Reserve the full graded OOS window after T0, not just the embargo ([bf222916](https://github.com/swedishembedded/brain/commit/bf2229167e2954f3b6127d6e8f010ce5c4657a39))

- *(kronos)* Report prefill/decode split (metrics for batching work) ([960ed9aa](https://github.com/swedishembedded/brain/commit/960ed9aa05207671d9354cca3bddd4c5eaf2d7e4))

- Kronos resident: per-request checkpoint selection (stateless runtime registration)

Run("kronos","forecast") gains a `checkpoint` param: the decoder path per
request, defaulting to the boot decoder. Instances are keyed on
(path, mtime, size), so:

- base + any number of fine-tunes stay warm side by side (A/B without
  restarts; verified live: interleaved base/ft requests on one server
  produced distinct forecasts with both instances staying warm);
- an overwritten checkpoint file hot-activates a fresh instance on the
  next request (the stale one ages out via normal eviction);
- there is no registry to manage or desync - checkpoint selection is
  request state, not server state. Boot env only sets the default.

estimate() sizes RAM from the requested path (file size or HF-dir sum,
same +30% margin as before); activate() fails fast on a missing path.

Known gap surfaced while validating (left for the executor track): an
activate() error never reaches the Run reply - the failed job wedges its
lane and, with Cancel a stub, every later Run queues forever (observed
as 32 parked brain-lane-Npu threads with ListModels still answering).
Callers currently validate paths client-side. ([34a0d470](https://github.com/swedishembedded/brain/commit/34a0d4708920387cb8a49d4c1a5300e22de887e0))

- Fix three scheduler bugs that wedged or killed the executor ([d17e6c24](https://github.com/swedishembedded/brain/commit/d17e6c24b32dc4d52ed74c393822bc196a3fcb3a))

- *(export_ohlcv)* Freshness ignores NULL-price and partial intraday bars ([b573e90e](https://github.com/swedishembedded/brain/commit/b573e90e50244d6357afde1aea822f8e6b1676a5))

- *(kronos)* Finetune-step batch-scaling - the training throughput numbers ([c288cb79](https://github.com/swedishembedded/brain/commit/c288cb7972fbb3bb72d67ea23c666d2a2ec87c7d))

- Expose --strength on the CLI and the edit capability ([ef9926aa](https://github.com/swedishembedded/brain/commit/ef9926aaabbfca79c4707a801e69bb5e014526e7))

- Add prep_finetune_data script ([f8a78b00](https://github.com/swedishembedded/brain/commit/f8a78b00a8fdbe6d50942f3d6c77ff91c5752755))

- Clear the whole clippy backlog, and fix the one kernel it exposed ([b9e8ab02](https://github.com/swedishembedded/brain/commit/b9e8ab0214e8b472f2ff2b86274f272e80aa1890))

- Clear the warnings the aborted clippy gate had been hiding ([a59300f5](https://github.com/swedishembedded/brain/commit/a59300f58d7253e944ef4de79e5e7b01e5e9a83b))

- Tune the arity threshold to the domain, and drop redundant parens ([dca0870f](https://github.com/swedishembedded/brain/commit/dca0870fff32b0c50f11c678138e054b41d446a7))

- Clear 11 clippy warnings, and four doc comments that documented the wrong item ([df830c56](https://github.com/swedishembedded/brain/commit/df830c56aa4681f92536c177df52d06d78f6bd67))

- Clippy's machine-applicable fixes, reviewed one by one ([91e7ea46](https://github.com/swedishembedded/brain/commit/91e7ea4657da5f9f4a021a0f0dadcb5cfbba1b5f))

- Support cross-vendor GGUF quant publishers ([f52f8ce0](https://github.com/swedishembedded/brain/commit/f52f8ce010bf51a3d4c5283eb0c7bc078d54ba15))

- Add gdn_decay_gate_bwd (Gated DeltaNet decay-gate backward) ([3360ed51](https://github.com/swedishembedded/brain/commit/3360ed51aa05412708f210dbf9bcc3ac2dc9fbe5))

- Wire model-level backward (GDN+GQA+MoE) + gradcheck ([e42651d6](https://github.com/swedishembedded/brain/commit/e42651d6bb55f63a32bdd0af0509882ab83bddfb))

- Brain-qwen -> brain-qwen3, brain-qwen35 -> brain-qwen35moe ([2918222d](https://github.com/swedishembedded/brain/commit/2918222d6623ae301af294f52fce8ce309b0d7d8))

- :gdn: add single-token decode-step primitives (recurrent state + causal conv) ([68d04baf](https://github.com/swedishembedded/brain/commit/68d04bafdf3321b4ef8ac6a360171fd7c80d3737))

- Wire single-sequence incremental decode (P11b) ([6fcd362f](https://github.com/swedishembedded/brain/commit/6fcd362f72a1fc7d156a93c3967466d45ea64a8c))

- Implement single-GPU PagedDecoder serve::Engine (P11) ([736fb9ca](https://github.com/swedishembedded/brain/commit/736fb9ca9535ab92a1e9aa256196937fe5b86e6b))

- Add sample::generate_kv + brain qwen35moe CLI (import/infer) ([0f0447f1](https://github.com/swedishembedded/brain/commit/0f0447f119e96ff3e8361fbc1bf226f038b14104))

- Best-effort ONNX/OpenVINO export for qwen35moe (P14) ([5b941ef9](https://github.com/swedishembedded/brain/commit/5b941ef914e09fa7a6bb8c6863f19a5d936886ca))

- Add int8/int4 GEMM throughput to the matmul benchmark, with explicit ms

bench_matmul only reported fp32 GFLOP/s; extend it with a second test,
bench_matmul_quant, covering the int8 (DP4A, matmul_i8_dyn/_gemv) and
int4/W4A8 (matmul_q4_dyn/_gemv) GEMM kernels already in the tree, on
both the CPU (Cranelift-JIT, rayon-threaded) and GPU (wgpu/Vulkan)
backends, with an exact-integer host reference for parity. Also add
explicit millisecond timings alongside the existing GFLOP/s columns in
both tables, since GFLOP/s alone doesn't answer "how many ms".

int8's fast tiled kernel (matmul_i8_dyn) has no CPU-JIT lowering, so
the CPU side is only measurable at the decode-regime GEMV kernels
(matmul_i8_gemv/matmul_q4_gemv, M<=32) - reported as such rather than
silently picking one shape. ([17093817](https://github.com/swedishembedded/brain/commit/17093817622e4ea7515705530ed7f813b85355df))

- Wire vision-language embedding splice (Qwen35Vl) ([c34cee9b](https://github.com/swedishembedded/brain/commit/c34cee9be50418fa5b5bada96e9791268f8a56b5))

- Full serving contract (caps, residency, D-Bus/HTTP, example) ([edc662ad](https://github.com/swedishembedded/brain/commit/edc662ad4553f13f6795e8581bb386b5dd5c8af3))

- LoRA + cross-GPU pipeline sharding, real-weight smoke test ([5de4f0de](https://github.com/swedishembedded/brain/commit/5de4f0de10e1ba32e143cf30e8c508fba4a93a83))

- Device-side sparse MoE decode dispatch (24.8x fewer dispatches) ([1e6ef265](https://github.com/swedishembedded/brain/commit/1e6ef265dc5fbf3d72661fa86922db428ae5ca46))

- Device-side sparse MoE decode dispatch (24.8x fewer dispatches) ([07c1be25](https://github.com/swedishembedded/brain/commit/07c1be259bc84a7476468db7f0186d9b504e1a69))

- Add Apache-2.0 LICENSE ([2367691b](https://github.com/swedishembedded/brain/commit/2367691b2cde1d4b49250c2f9cd1dbfd0217b81d))

- Add SPDX/copyright header + commit-trailer git hooks

SPDX gate: every Rust/C/Python/shell/Makefile/WGSL/proto/... source file
must carry exactly one "SPDX-License-Identifier: Apache-2.0" line followed
immediately by the project copyright line (scripts/spdx/rules.py has the
file-selection/comment-style rules; scripts/spdx/check.py validates them,
with a --fix mode).

Commit-trailer cleanup: scripts/hooks/commit-msg silently strips
(never fails the commit); scripts/hooks/pre-push is the actual gate - it
fails the push if one survived anyway (a commit that bypassed commit-msg, or
arrived via fetch/merge/cherry-pick). scripts/hooks/trailers.py holds the
shared stripping logic.

Enforced two ways: `.pre-commit-config.yaml` for the pre-commit framework
(`pre-commit install` wires all three hook types via
default_install_hook_types), and `make hooks/install` for a plain git hook
that needs no extra dependency. check/spdx also folded into `make
test/full`. ([0e7f750f](https://github.com/swedishembedded/brain/commit/0e7f750f059a2d520ca39b723afa56aa812c84ac))

- Serving contract (caps/resident/catalog) + composite crate ([4a9a2730](https://github.com/swedishembedded/brain/commit/4a9a2730f9a32833e35db0990f6aa466ff89d81a))

- *(docs)* Add check-no-perf-numbers, deny bare perf claims in docs/ ([8430aa76](https://github.com/swedishembedded/brain/commit/8430aa76307e6a45253fb1e6201877daf3176537))

- Add KernelVariant::RegisterTiled, unify the three GEMM pickers ([3902aa68](https://github.com/swedishembedded/brain/commit/3902aa6841aab1dbc677f50022a0517e21a986a6))

- Migrate chronos2/fincast NPU sessions onto the NpuModel seam (C2) ([6148294f](https://github.com/swedishembedded/brain/commit/6148294ff122500fe63cd09a275a667ff20ae204))

- Delete NPU_REQUESTED sidecar, route through ambient_compute_set (C3) ([598f27a7](https://github.com/swedishembedded/brain/commit/598f27a77b8557dbd5c7e163c0f4d050aef6c9f5))

- Add Ops/Weight kernel-selection facade (f32+I8+Q4) (B3) ([3a444b45](https://github.com/swedishembedded/brain/commit/3a444b4531039a03be12c8fa12002de2ebe6d419))

- Collapse 9 bespoke OpenVINO sessions onto NpuGraph (C4) ([dd3186e6](https://github.com/swedishembedded/brain/commit/dd3186e6a038917971fca4057c71f0d60121196f))

- Update readme ([e2f3cbf1](https://github.com/swedishembedded/brain/commit/e2f3cbf12a219ac689c2b11c8ac3d6524f97f749))

- Dtype_variant templater + bf16 storage tier (B4) ([4b8230da](https://github.com/swedishembedded/brain/commit/4b8230daea2a2f1658e9384a4ac19a03546f5105))

- F16 storage tier (B5) ([3ee8a158](https://github.com/swedishembedded/brain/commit/3ee8a158bf8ece2169c6e68ef7b98df19fccc6f9))

- Add @dtype header field + CI matrix across all 400 kernels (B6) ([d58e5e96](https://github.com/swedishembedded/brain/commit/d58e5e965e95621f6e5e7bdf4cbc93507f520ec0))

- Migrate model.rs's forward-inference linears onto Ops/Weight (B7) ([96bcfe6a](https://github.com/swedishembedded/brain/commit/96bcfe6ab423e81305b5d213c925446c8842a169))

- Migrate serve.rs's per-layer weight storage onto Weight (B7) ([c8fc0ab3](https://github.com/swedishembedded/brain/commit/c8fc0ab35429445ea4cd8cdd92cc8e699267f3d5))

- Add B7 TDD gate + fix a real int8-capability test gap ([1202645d](https://github.com/swedishembedded/brain/commit/1202645d188b4c980bc47d2b29b41d41114c900f))

- Full float-kernel dtype coverage sweep (B8) ([e7f4c270](https://github.com/swedishembedded/brain/commit/e7f4c270b999c34f846a73acc9b2d3ff740b308b))

- Bf16 paged-KV-cache storage tier (B9) ([68902e8a](https://github.com/swedishembedded/brain/commit/68902e8a2b69aa83c94f3abccc34144e43a60556))

- Bf16 training tier, default off (B10) ([42269a16](https://github.com/swedishembedded/brain/commit/42269a1680a592f181ba5ef2e4c079315b1d7ead))

- *(vulkan)* Optional allocation-size + backtrace log behind BRAIN_VK_ALLOC_DEBUG ([f611e6cc](https://github.com/swedishembedded/brain/commit/f611e6cc5b5b49de56e2c1df38a18a10c8338634))

- Forward typed multimodal content and tool schemas into the real chat template ([5bead09f](https://github.com/swedishembedded/brain/commit/5bead09f7338f4b519e7895f522fd7397b2a08ca))

- Add a lane-split causal GQA flash-attention kernel for the Thinker ([3837fe35](https://github.com/swedishembedded/brain/commit/3837fe35945099c5c3374350f3766280eedf8394))

- Fix smart-resize pixel-area bounds to match the real released config ([70665529](https://github.com/swedishembedded/brain/commit/70665529a447d29840e36bc81b8e445b3eba9bc3))

- Real PNG encoder; move zipdepth's viz module in for reuse ([86a0f53b](https://github.com/swedishembedded/brain/commit/86a0f53b2efd83724ab9dad14edeb3b1646e5681))

- Write real image files instead of always raw P6 PPM ([7379aec7](https://github.com/swedishembedded/brain/commit/7379aec7a88b7bf8cf5f6bee92a2a842e3b8a047))

- Add draw_boxes and colorize actions ([4b1838ef](https://github.com/swedishembedded/brain/commit/4b1838ef5b0a29a03911f41c8ab935ef60c0fbf3))

- Fix brain zipdepth infer hard-exiting on the bare infer verb ([5d74c3e5](https://github.com/swedishembedded/brain/commit/5d74c3e5250f1b0290a8bf24873f7b1a02e3bc53))

- Fix flux2 --help falsely advertising text2image/edit/lora_train as CLI actions ([53f28b46](https://github.com/swedishembedded/brain/commit/53f28b46cc05ae75b6d3a16ac93c702070e80c12))

- Default --weights-dir from \$BRAIN_QWEN3TTS_WEIGHTS ([36052588](https://github.com/swedishembedded/brain/commit/360525889762c6116e5264a560360665a43de2e7))

- Make flux2-klein, imageops, demo, and imgpipe reachable from the CLI ([bb2948d2](https://github.com/swedishembedded/brain/commit/bb2948d2d984e5c130fa9a006db81b53ae4798ec))

- HF_TOKEN support for authenticated fetches ([7c8cdaf4](https://github.com/swedishembedded/brain/commit/7c8cdaf4b544c1f4c89976f66d423a11fcf4ffb7))

- Generalize FilesRecipe into a declarative whole/partial-repo fetch ([8d03abae](https://github.com/swedishembedded/brain/commit/8d03abaee1d8358de386e208655e1fab5dc7f15b))

- Add weights_env, wire auto-fetch coverage for 8 more architectures ([015ee0a6](https://github.com/swedishembedded/brain/commit/015ee0a6e7721d89810277e631c50a7456d8d22e))

- Generic convert_files + qwen3tts assembly + capability-path auto-fetch ([f801db21](https://github.com/swedishembedded/brain/commit/f801db2112bdc8e51736604e0f59d8d91a273db6))

- Wire auto-fetch into brain <arch> dispatch, skip for --help ([6af5fe86](https://github.com/swedishembedded/brain/commit/6af5fe8627415eed098bc73c6f3a783d0d20b9f9))

- Q8_0 GGUF import for the DiT (register S3ditImporter) ([4820b874](https://github.com/swedishembedded/brain/commit/4820b8745dccfd372e3cb8442bcc54d6b16fbb84))

- Correct clippy warning-count baseline (279 -> 294) ([7a0b364a](https://github.com/swedishembedded/brain/commit/7a0b364af40509dc2d491243a67752cbc0db4503))

- Fix real-checkpoint OOM by using the decode-only Qwen constructor ([d5933f1d](https://github.com/swedishembedded/brain/commit/d5933f1dfbdbe7abaa5663b1c39e64acbcb5ea55))

- Gpu-core, s3dit: respect --device restriction; VAE-on-CPU when the fp32 shard is full

Two related bugs against the real Tongyi-MAI/Z-Image-Turbo checkpoint's
fp32 ("hifi") path:

1. --device gpu0 did not restrict s3dit's single-vs-multi-GPU placement
   decision. hifi_needs_window(gpu_count) -- meant to fall back to a
   small-footprint single-GPU streaming window when fewer than 2 GPUs are
   available -- read gpu_core::devices::gpus().len(), the machine's raw
   physical card count, not what --device actually restricted the process
   to. brain --device gpu0 s3dit text2image --precision fp32 still built
   the 2-GPU ZImageDitShard and hit the OOM below regardless.

   Adds gpu_core::devices::schedulable_gpu_count(), reading the SAME
   ambient_compute_set() every other --device-aware decision resolves
   through, and switches every gpus().len() call feeding a machine-shape
   decision (hifi_needs_window, default_bulk_gpu's two call sites, the
   VRAM-cost estimate resident.rs budgets against) to it.

2. brain s3dit text2image --precision fp32 OOM'd building the VAE decoder
   immediately after the 2-GPU-sharded fp32 DiT finished building, 100%
   reproducible. Real capacity exhaustion, not a false alarm: the ~33 GB
   fp32 checkpoint's DiT half lands within half a GB of a 24 GB P40's
   ceiling on ITS OWN, once this backend's real ~2.00x VRAM-per-uploaded-
   byte cost on this non-ReBAR card is accounted for -- there is no
   headroom left on either card for the VAE decoder's own weight upload,
   independent of how the block cut between cards is chosen.

   Decodes the VAE on the CPU when the fp32 DiT is the 2-GPU Shard engine
   (vae_on_cpu = matches!(dit, DitEngine::Shard(_))) -- the same fallback
   the Qwen-4B encoder already takes for itself in this same pipeline. ([a5d9dee9](https://github.com/swedishembedded/brain/commit/a5d9dee9a5dd0b59ba5264389e8175e7dbbf3a8b))

- Paramstore, cli/qwen: avoid a 2x real-VRAM cost from zero-init writes; wire BRAIN_OFFLOAD_ADAM into full finetune

brain qwen3 finetune OOM'd on GPU at real Qwen/Qwen3-0.6B scale (block
2048) even at --batch 1. Root cause: ParamStore::new's Trainable/Offload
branches zero-initialized the grad/Adam moment buffers via
gpu.storage_init(name, &z) with a full zero vector. Measured precisely on
this backend: a buffer that is allocated and never written costs its
real 1.00x on this non-ReBAR card; ANY upload into it -- storage_init or
a plain storage() + write_f32, chunked or not -- costs 2.00x, because the
resident cost tracks cumulative bytes ever WRITTEN, not the call shape.
wgpu already guarantees a freshly-created buffer reads as zero, so
writing a vector of zeros into it was paying that 2x for a write whose
only content was what the buffer already had -- three such buffers per
trainable tensor (grad, adam_m, adam_v) made this the dominant real VRAM
cost of a full (non-LoRA) finetune, invisible from the nominal parameter
count. Switches both branches to gpu.storage(numel), which allocates but
never writes.

Separately: qwen_cli's train() (what a plain, non---lora brain qwen3
finetune actually calls) never set BRAIN_OFFLOAD_ADAM, so
qwen3::finetune::Mode::FullOffload -- which keeps the AdamW moments in
system RAM instead of on the GPU, trading weight+grad+m+v (4x) for
weight+grad (2x) GPU-resident -- was dead code: only finetune_lora's
Mode::Lora path ever got exercised. Sets BRAIN_OFFLOAD_ADAM=1 around the
model::fit call when base is Some (a real checkpoint finetune; a
from-scratch train has no checkpoint-scale weights to offload for),
restoring any prior value afterward. ([37980d2b](https://github.com/swedishembedded/brain/commit/37980d2b4afc31b7ffe71b5367bb83fa6f4ef0a2))

- Add missing cost formulas for scale_row, moe_linear_gated, paged_kv_append_batched_word ([f5067f3c](https://github.com/swedishembedded/brain/commit/f5067f3c627ef20243808053099f4a08449fdb84))

- Qwen3, model: fix serve.rs's Ops kernel list, add a shared completeness check

`qwen3::serve::Engine::from_map_with_gpu` builds a `model::ops::Ops` façade
from `ops_kernel_list()`, a hand-maintained `(name, wgsl_source)` list that
is supposed to mirror `model::ops::REQUIRED_KERNELS` exactly (`Ops::new`
requires every required kernel to already be registered on the `Gpu` it is
built from). It had silently drifted 15 kernels short: missing `embed`,
`moe_linear_gated`, every `paged_*_batched` bf16 storage tier, and
`matmul_dx`/`matmul_dw` (plus their own bf16/f16 variants), relative to
`qwen3::model::pipelines()` (already correct) and `model::ops`'s own
test-only `kernel_list()`. Reproduced on real hardware (Tesla P40): `Ops::new`
returned `Err("kernel 'embed' is not registered on this Gpu")`, which
`from_map_with_gpu` turns into a panic.

The gap was never caught by `cargo test` because `Engine::from_map_with_gpu`
is only reached lazily, via the residency pool's `activate()` (GPU upload
on-demand so many resident models can share one card) - never eagerly at
`brain serve` startup. That is a deliberate, correct design (forcing eager
activation would defeat sharing one GPU across many resident models), so the
right fix is not "activate eagerly" but "catch the drift before it ships":
`ops_kernel_list()` now registers the full required set, matching
`qwen3::model::pipelines()` and `model::ops`'s `kernel_list()` exactly.

To stop this class of bug recurring, `model::ops::assert_kernel_list_complete`
is a new, reusable, non-test-gated helper: a plain name-set comparison of a
caller's kernel list against `REQUIRED_KERNELS`, needing no `Gpu` at all. It
is now wired into a no-GPU-needed test in each of the three real `Ops::new`
call sites (`qwen3::serve`, `qwen3::model`, `gradcheck::bf16_train`) plus
`model::ops`'s own test module, so any future drift in any of them fails at
`cargo test -p <crate>` time instead of on a live server's first request.

Verified: `qwen3::serve::tests::batched_serving_matches_reference` (which
exercises this exact path on real GPU hardware) now passes, along with the
rest of qwen3's serve/model test modules, model::ops's test module, and
gradcheck::bf16_train's tests (including its real gradient check). `make
gradcheck` passes with zero failures. ([593f1bc0](https://github.com/swedishembedded/brain/commit/593f1bc00e3a0dff31c78bc3ae701ce2c1240074))

- Cleanup and update banner ([b47c7318](https://github.com/swedishembedded/brain/commit/b47c7318ee02b318d4c8386aa637d51b9607d68c))


### Performance

- *(cpu)* Add BRAIN_PROFILE per-kernel timing to CPU backend ([5a75c4ce](https://github.com/swedishembedded/brain/commit/5a75c4ce4181ed42d916f86fce63a3a069c0f2ec))

- *(cpu)* Native AVX2/FMA GEMM conv2d fast path (7.7x e2e) ([6d527d8a](https://github.com/swedishembedded/brain/commit/6d527d8a5cd6fb1283bbf1134d806aaccf3f501b))

- *(yolo)* BRAIN_PROFILE per-frame detect stage timing ([3149736e](https://github.com/swedishembedded/brain/commit/3149736e7699292ed5a8d459c3d18febadaed06d))

- *(cpu)* L2 cache-blocking for conv GEMM + conv GFLOP/s microbench ([aae7732f](https://github.com/swedishembedded/brain/commit/aae7732fc6c293eab0b94b6845dc48996d8fc9cd))

- *(cpu)* Per-panel hot im2col + unified microkernel (58->107 GFLOP/s) ([74adb210](https://github.com/swedishembedded/brain/commit/74adb210156055e2bedf6c81212e668b94f0a088))

- *(cpu)* Native fast paths for concat/bn_eval/silu/upsample ([3f926952](https://github.com/swedishembedded/brain/commit/3f926952f635fcdf2bec45244209045c9edee2b1))

- *(cpu)* Coarse-grained rayon chunking for concat/bn/upsample ([5ba0323e](https://github.com/swedishembedded/brain/commit/5ba0323ecca8fab5c9e91b0a751e71309c2e1304))

- *(yolo)* Fuse conv->BN(eval)->SiLU into one conv_act kernel (forward -26%) ([b2d1b379](https://github.com/swedishembedded/brain/commit/b2d1b379e04451c5333379663f2b3c48992ffdf9))

- *(cpu)* Segmented bulk-memcpy for concat2/concat_split ([1d5d76c4](https://github.com/swedishembedded/brain/commit/1d5d76c414d85b6b73b1a8dba1fec2823cb81202))

- *(yolo)* Argmax class scores on raw logits (postprocess ~2x) ([50a3b3ae](https://github.com/swedishembedded/brain/commit/50a3b3aea7c981691de128eac1074f2e6c8972f1))

- *(yolo)* Single-pass C2f channel concat (drop O(n^2) left-fold) ([82f71720](https://github.com/swedishembedded/brain/commit/82f71720803e9d65db03685e22971d4038990594))

- *(cpu)* Winograd F(2,3) conv path (opt-in scaffolding, Phase 7) ([1e7f398e](https://github.com/swedishembedded/brain/commit/1e7f398e52ac5dc2ff18d725d60201023f45f2b4))

- *(gpu)* BRAIN_PROFILE op counters for the wgpu backend ([c8bf58c0](https://github.com/swedishembedded/brain/commit/c8bf58c090554d0d18398ca55f2aa52fd8af1c63))

- *(gpu)* Kill per-frame host readbacks in yolo inference (241 -> 7/frame) ([7efb798d](https://github.com/swedishembedded/brain/commit/7efb798df16906120a2a72b977f24eeefff186f9))

- *(gpu)* Lazy-submit batching - coalesce a forward into one queue.submit ([b6580398](https://github.com/swedishembedded/brain/commit/b6580398ebc739148191d9794d2d44a3521c571f))

- *(gpu)* Coalesce a forward into ONE compute pass (not ~130) ([975a1800](https://github.com/swedishembedded/brain/commit/975a1800594b9ed0851eac71a5c269738b5d7647))

- *(gpu)* Default to naive fused conv; tiled is opt-in + honest fwd timing ([ad7b085f](https://github.com/swedishembedded/brain/commit/ad7b085f7f20eee6de0591d1ea586e15e777b3b7))

- *(gpu)* Register-tiled fused conv (4 channels/invocation) as default ([eee968ee](https://github.com/swedishembedded/brain/commit/eee968ee1255a4afc0de092ecd073a77c0efc3d2))

- *(gpu)* 4x4 register tile (4 channels x 4 positions) for the fused conv ([23e32341](https://github.com/swedishembedded/brain/commit/23e32341b74f754d933b816bae05e6776b3a6e9c))

- *(gpu)* Fully unroll the 4x4 register tile into scalar accumulators ([30c6067f](https://github.com/swedishembedded/brain/commit/30c6067f281e09412414ce9a660425cf88d23ee4))

- *(gpu)* Coalesce the register-tiled conv's memory access ([b0e48a20](https://github.com/swedishembedded/brain/commit/b0e48a209b223d544a1588fd16994f518b1f6003))

- *(yolo)* Fuse detection-head conv + bias into one conv_bias kernel ([1bdb1a75](https://github.com/swedishembedded/brain/commit/1bdb1a75a5ffa8052e8a109599ec02095d77c795))

- *(gpu)* Hoist conv boundary checks out of the channel loop ([035bba73](https://github.com/swedishembedded/brain/commit/035bba73238f7052d851916677da955a0d798e68))

- *(gpu)* Widen register tile to 8 channels x 4 positions ([efe72a2e](https://github.com/swedishembedded/brain/commit/efe72a2eeed78d0a204c13be2ed7e61cd37a4323))

- Performance benchmarking suite (crates/perf + `brain perf`) ([43b968c6](https://github.com/swedishembedded/brain/commit/43b968c6548dd7399a1bb002534935ede8b993ed))

- --input/--output workload overrides; record first P40 findings ([9ee60772](https://github.com/swedishembedded/brain/commit/9ee60772333ae55a6fdacac419c1b4773c52cb1a))

- The remaining Tier-2 benchmarks - all 14 scenarios now run ([252eabea](https://github.com/swedishembedded/brain/commit/252eabea44c6afb9e79684b382090f472f250022))

- Hard-floor regression gate (J2) ([030cdfbb](https://github.com/swedishembedded/brain/commit/030cdfbbb77a9c3d83506a3c0f8ff5911f996ee4))

- Real BPE stages in frontend (I) + honest placement scope (H) ([1a636b6d](https://github.com/swedishembedded/brain/commit/1a636b6d5ba8129a4fb7d114ff959dfad281a6f9))

- Real single-process fault injection (G) ([c31ef31d](https://github.com/swedishembedded/brain/commit/c31ef31de4dd48366a82b9afaff57228815e3519))

- ExecutorTarget - honest concurrency benchmarks through the scheduler ([dff98364](https://github.com/swedishembedded/brain/commit/dff98364ddd8bdfff9d839abc95d737bfebf2351))

- Resident-backed flux2 target (streaming denoise_step artifacts) ([4cde57f3](https://github.com/swedishembedded/brain/commit/4cde57f326592fe62938f8aadc9abe48b819b55f))

- FLUX.2 DiT forward 4.67x - the limiter was attention, not the GEMMs ([455f8054](https://github.com/swedishembedded/brain/commit/455f8054a093446d604487092f1bae128d12a9ba))

- FLUX.2 text-encoder + VAE 7.3s -> 2.5s - the VAE decode was 88% of it ([8dd49f0f](https://github.com/swedishembedded/brain/commit/8dd49f0fbd8f752fa7734eda2cdf9b2e3839b311))

- *(cpu)* AVX2+FMA matvec on the host KV path (kronos 1.47x forecast) ([5cb2c701](https://github.com/swedishembedded/brain/commit/5cb2c70196663dd3533f4e2c64d479f78cc9dbb4))

- *(kronos)* KV-cache the S2 dependency head (decode 2.3x+ faster) ([22a62020](https://github.com/swedishembedded/brain/commit/22a62020a3009dc2f6948b777c28d1de31a3d0ca))

- *(kronos)* Shared-prefill sampling - one prefill, forked KV decode (3.4x) ([f8eb5b4e](https://github.com/swedishembedded/brain/commit/f8eb5b4e529934a12f5c70fe79fda22be6cc5b03))

- *(forecast)* Kronos D-Bus resident uses the fast KV-cached path + samples ([b9644345](https://github.com/swedishembedded/brain/commit/b9644345373f797e5627521e4f6e5559d0920146))

- *(kronos)* Cross-sectional batch - rayon over names (parity-exact) ([0c192804](https://github.com/swedishembedded/brain/commit/0c1928046139bfe0472a7e73a552f6124c8992de))

- *(lfm)* Dispatch fused flash attention when the split kernel is selectable ([75c77e8a](https://github.com/swedishembedded/brain/commit/75c77e8aeaff56a2e8fa830d6243ad465f64591d))

- *(model)* The GEMM guard was the bottleneck - 1.93x on the SDXL forward ([6cfc1586](https://github.com/swedishembedded/brain/commit/6cfc15861775e380c5960e1c445ba97e29b17dbe))

- *(unet,vae)* Matmul_reg3 supersedes matmul_reg2 - unet held both, used the slow one ([9e03b601](https://github.com/swedishembedded/brain/commit/9e03b601711b8e164b1ba590f45ec70565327925))

- Retire matmul_reg2 as a model's tiled GEMM - eleven crates were on it ([9b6415a6](https://github.com/swedishembedded/brain/commit/9b6415a6cb175aacd40239ae9ea3cc4e421e7e79))

- *(vqgan)* The first BACKWARD profiler, and what it says about training ([00fe6e81](https://github.com/swedishembedded/brain/commit/00fe6e8184f275a5e9f9b20c8ca5c84542e89df7))

- *(vae)* The CPU GroupNorm fallback was the serial kernel - 3x, and more accurate ([96f9783f](https://github.com/swedishembedded/brain/commit/96f9783f5e09d91d40750fce2a3109bada397749))

- *(vae)* Lower the conv input gradient to a GEMM - 5.2x on it, 1.48x on the step ([3cbeb525](https://github.com/swedishembedded/brain/commit/3cbeb52585ac1df9bdd80a11485727be959c153d))

- *(vae)* Lower the conv WEIGHT gradient too, and report GFLOP/s against the roof ([a3cb2475](https://github.com/swedishembedded/brain/commit/a3cb2475f9ffe0aeef14b559d612f5896ac3e691))

- *(vae)* Two-stage GroupNorm backward reduction - gn_dsum 229 ms -> 21 ms (10.8x) ([1c99375e](https://github.com/swedishembedded/brain/commit/1c99375ef59bc4ccac548b15be67f0c0eea71379))

- *(qwen)* Prefill via one batched Qwen::prefill call, not step()-per-token ([b6a87375](https://github.com/swedishembedded/brain/commit/b6a87375620f4223d6989f1363267bc2fa1b009a))

- *(vae)* Fuse the GroupNorm affine gradients - 170 ms -> 16 ms (10.6x) ([d2310227](https://github.com/swedishembedded/brain/commit/d2310227f03b2a74b0e2ee38c69322a563d16ae5))

- *(kernels)* Fused layernorm2d - §E's open question, settled at 4.38x ([2c8eb0da](https://github.com/swedishembedded/brain/commit/2c8eb0daa3c050d7132963e86ddbd6676f1d5b06))

- *(vae/blocks)* Split-K weight gradient - backward 572.54 -> 457.66 ms ([ec04fb7a](https://github.com/swedishembedded/brain/commit/ec04fb7a34a392dc3b74212cc668aa5afec74afd))

- *(vae/blocks)* Re-derive GEMM_CONV_MIN_COUT - 128 was inherited, 32 is measured ([863016d2](https://github.com/swedishembedded/brain/commit/863016d2cf7bb5dfd092b6d4816a6fd8c477d906))

- *(model/block)* Size the vocab tiling from the device, not from a constant ([78e96690](https://github.com/swedishembedded/brain/commit/78e9669045fb0161312085ff8217bbb5be65404e))

- *(qwen/serve)* Register the tiled GEMM - prefill was 98% naive matmul ([3d438068](https://github.com/swedishembedded/brain/commit/3d438068acd5c08fbfdafa64ffde9e50beda64e1))

- *(gpt,glm)* Route the GEMM pickers through block::pick_gemm ([ba3a03ca](https://github.com/swedishembedded/brain/commit/ba3a03caf9ed5e1c9888cafb534e0994c29432fe))

- *(qwen/serve)* Coalesce the paged decode scores - 1.94x on the top row ([6b6f6b87](https://github.com/swedishembedded/brain/commit/6b6f6b877b04ff44b76a676d7e7ba4158634d926))

- *(qwen/serve)* Dispatch the split-K forward GEMM - 1.52x on the GEMM path ([de855312](https://github.com/swedishembedded/brain/commit/de855312ec208a03e9990944b735c784c6a68ffb))

- *(kernels)* Sweep lanes-per-score - paged decode scores to 77.5% of roof ([d52a52e0](https://github.com/swedishembedded/brain/commit/d52a52e060ba357ddf976eadca46fa42ea629603))

- *(gpu-core)* Gate cost tally off the dispatch hot path; fix profile attribution; roofline race + test hygiene ([38cdea7b](https://github.com/swedishembedded/brain/commit/38cdea7bd7cfb99ef2cd179227f509176213036d))

- *(residency)* Executor::manifests() returns an Arc snapshot instead of deep-cloning the catalog ([65e5e4db](https://github.com/swedishembedded/brain/commit/65e5e4db1219258d793f3a01013149366cfd9777))

- *(deepseekocr)* Stage-level BRAIN_PROFILE timing in Session::load/generate ([d5c12e57](https://github.com/swedishembedded/brain/commit/d5c12e5786dd93545d3d8547e625bf1e35040dca))

- *(deepseekocr)* Per-stage BRAIN_PROFILE timing inside the vision encoder ([89387215](https://github.com/swedishembedded/brain/commit/8938721574b837714044141005d8a538e1652be1))

- *(sam1)* Dispatch matmul_reg3 via block::pick_gemm, like clip/deepseekv2 ([fcab2d36](https://github.com/swedishembedded/brain/commit/fcab2d36d4c05d48c5bf3dec40047a12d0965f4c))

- *(backend-cpu)* Tile attn_apply_cross's V transpose for cache locality ([47439c20](https://github.com/swedishembedded/brain/commit/47439c20f311598541de72b9c65e7a0dbba3b049))

- *(deepseekocr)* Fine-grained BRAIN_PROFILE brackets for model construction ([a7dee637](https://github.com/swedishembedded/brain/commit/a7dee637d0c9f3e48b0fef62fbe93b84335cc4e6))

- *(backend-cpu)* AVX2 fast paths for silu_mul and scale_add ([b45225ec](https://github.com/swedishembedded/brain/commit/b45225ecbf30b01e719bc4511d27e5e8eb4e0966))

- *(deepseekocr)* Move the vision encoder onto wgpu, decoder stays CPU ([39efb68d](https://github.com/swedishembedded/brain/commit/39efb68d586c166295c6ce21fbceeef853b96944))

- *(omni)* Fix Vulkan buffer leak, wire int8+KV-cache Thinker path, fix profiling ([2ebd7159](https://github.com/swedishembedded/brain/commit/2ebd7159a4aa5eaedd79ab7aef265c7fc2cc5242))


### Refactor

- *(yolo)* Share detection post-processing + add calibration tap seam ([117fe081](https://github.com/swedishembedded/brain/commit/117fe081c78a718be2ff6c5ed2265cfd7bdfe1ae))

- *(cpu)* Route kronos cross-section parallelism through backend_cpu::par ([bb80fc39](https://github.com/swedishembedded/brain/commit/bb80fc39ef884d46018ec3eb82c2bba7cbb78cc5))

- *(kernels)* Migrate SPPF to maxpool2d and delete maxpool5{,_dx} ([d09a799e](https://github.com/swedishembedded/brain/commit/d09a799e197961c0370e48f8c38a5c42158dfbf9))

- *(npu)* Hoist sub_t onto TopoBase; add Chronos2Session::head_out ([1be625fe](https://github.com/swedishembedded/brain/commit/1be625fee73edf49ab5a346bc87398960648ad9a))

- *(yolo)* Move the letterbox into imaging; keep boxmath's names ([f96b40b5](https://github.com/swedishembedded/brain/commit/f96b40b51009c366c5de4e295510b58bd8a9ae8f))

- *(capture)* Move yuyv_to_rgb into imaging; V4L2 only again ([7302a0bf](https://github.com/swedishembedded/brain/commit/7302a0bf1d856055923cb82c99cdc589476f32a5))

- *(npu)* Drop the two private chw_to_hwc copies ([6806f3ce](https://github.com/swedishembedded/brain/commit/6806f3ced704c1c0be978227bbbb630085976478))

- *(mirror)* Delete the second P6 parser, RgbImage and the ImageNet constants ([f669b174](https://github.com/swedishembedded/brain/commit/f669b174df320ed75229f02aed52ba5354188b13))

- *(depth)* Six host bilinear resizes become one ([c7a5b733](https://github.com/swedishembedded/brain/commit/c7a5b733c2be71885f435387b651b2b16f7b6591))

- *(wm-display)* Call imaging's converters directly ([c00a755f](https://github.com/swedishembedded/brain/commit/c00a755fb98583a8d7c83b237770aff3f8ee37bc))

- *(cli)* Route every image path through imaging ([86327225](https://github.com/swedishembedded/brain/commit/86327225603c1eeddd80be1451a4725b239bad85))

- *(vae)* The private conv/attn/resnet builder becomes vae::blocks ([cca1c341](https://github.com/swedishembedded/brain/commit/cca1c341dce5ce86f3534166f3133c77eb1b8964))

- *(capability,residency)* Streaming + admission seams (P3) ([ea747ae8](https://github.com/swedishembedded/brain/commit/ea747ae8182416c6b52d7466cb6f09d43376980b))

- *(testdata)* Unify fixture-path resolution, narrow fetch-testdata.sh ([7de4f8a8](https://github.com/swedishembedded/brain/commit/7de4f8a80e0173655334489339d09bcd758b311d))

- *(scripts,tools)* Organize into a subfolder tree by function ([74700dca](https://github.com/swedishembedded/brain/commit/74700dca20f240b073d94100f5320febb7a11370))

- Refactor(flux2): dispatch through the hoisted helpers, and record why it keeps
the tiled GEMM

Migrates the first user of the four rules hoisted in the previous commit:
`qknorm_rows` -> `block::rms_variant`, the timestep embedding ->
`hostmath::timestep_embedding`, `quant_rows` -> `int8::quant_rows_steps`, and
`mm`/`mm_rows_at`/`mm8` -> `block::gemm_variant`.

Every one is behaviour-identical by construction, and `gemm_variant` is passed
`gemv: None` so the GEMM tier reproduces exactly what this model dispatched
before. That is deliberate and it is MEASURED, not assumed. The flux1 review
reported that flux2 "wastes 127/128 of a tile on every skinny-M int8 GEMV"; it
does not, because it has none. Instrumenting every GEMM dispatch on the real
klein-4B forward gives a minimum M of 512:

  fp32  M in {512, 1024, 1536, 2048, 2560}   (320 dispatches)
  int8  M in {512, 1024, 1536}               (800 dispatches)

Not one would reach the `m <= 32` arm. That follows from this file's own
header: FLUX.2's modulation is global and folded into LayerNorm affine params on
the host, so unlike flux1 - whose per-block modulation issues 77 `m = 1`
mat-vecs per forward - there is no skinny-M work on the device at all.
Registering the two GEMV kernels would add dead pipelines, and an M-dependent
kernel choice would put a hazard under tests/batch_parity.rs, whose bit-identity
claim rests on every dispatch being independent of M. The measurement is in the
comment so the next reader does not re-derive it.

`modelgrad`'s generic-T timestep embedding stays as the FD oracle (AGENTS.md
exception 1); its doc comment now points at hostmath rather than at a method
that no longer exists.

Re-gated on one Tesla P40, `--release`, against the real klein-4B checkpoint:
26 tests, 0 failed - dit_parity, e2e_parity, int8_parity, host_forward_parity,
import_real, model_grad, model_smoke, lora_train, block_grad, and all four
batch_parity cases still bit-identical. ([573f9ec3](https://github.com/swedishembedded/brain/commit/573f9ec3f5d2642a39681eff3b2ecf47caab8d51))

- *(data)* One two-sided LCG for test fixtures, and migrate the copies ([55eb7618](https://github.com/swedishembedded/brain/commit/55eb76184bf3505550228c424dad63e18150b471))

- *(model)* One timestep sinusoid - hoist diffusers' two knobs into hostmath ([fbf64052](https://github.com/swedishembedded/brain/commit/fbf640526c72e4568bf9cb50e00bf5dad53bac24))

- *(vae)* Switchable GroupNorm epsilon and a staged weight upload in the shared builder ([fa91f8d8](https://github.com/swedishembedded/brain/commit/fa91f8d88be4c69be8bcc400396c2ac71a624d53))

- *(examples)* One P6 reader, in brain_py.image ([dda214d8](https://github.com/swedishembedded/brain/commit/dda214d80efb26ddc89dd03394bec025faea3a37))

- *(gpu-core)* One host-f32 buffer upload - Gpu::write_f32 ([bbdf44e0](https://github.com/swedishembedded/brain/commit/bbdf44e08cf3d20d3691cc8212965c2b3be6dfbe))

- *(model)* One LayerNorm selection rule - make block::ln_variant public ([7e8dfb3e](https://github.com/swedishembedded/brain/commit/7e8dfb3ea4365c6b0bcd12c7aaa948fa24e52546))

- *(model)* Hoist the row-range dispatch emitter out of pulid ([8a23fdac](https://github.com/swedishembedded/brain/commit/8a23fdac54b8152144083975b7f588cf91982b15))

- *(cli)* One model catalog, so a model cannot be half-registered ([5075a67e](https://github.com/swedishembedded/brain/commit/5075a67e70bb0e42c55acb0348aba11c1081f2f6))

- Name the four worst type signatures ([f3dfee60](https://github.com/swedishembedded/brain/commit/f3dfee60a43d5ed32acbad0c0f2572a26334b083))

- *(vae)* Hoist concat, adaptive GroupNorm, and per-level groups into the shared builder ([0c376234](https://github.com/swedishembedded/brain/commit/0c376234fa63e5abd2a62aa2905f5ace7a1f9a70))

- *(wm-diamond)* Drop two helpers that already existed elsewhere ([a3af728b](https://github.com/swedishembedded/brain/commit/a3af728b94e8999a69912acae38be9e96ca62436))

- *(wm-diamond)* Record the inference UNet with the shared block builder ([a23c7e15](https://github.com/swedishembedded/brain/commit/a23c7e1516e891345d3af1f8b76d2505445d5568))

- *(apiserve,dbus)* Collapse the serve_all*/serve_with_* ladders to one ServeOpts ([86841f38](https://github.com/swedishembedded/brain/commit/86841f382c705f8464d039d7c58ae60596e8b8fd))

- *(model)* Extract the activation-percentile reservoir out of depth::quant ([40c2e0f5](https://github.com/swedishembedded/brain/commit/40c2e0f58def276580ff6319383536ba815a1a06))

- *(vqgan,unet)* Drive both benches from the shared profiler and measured roofs ([16998819](https://github.com/swedishembedded/brain/commit/169988190fa36479f8fbc6b6f4e724f5db678d9d))

- *(qwen/serve)* Drop three pipelines the engine compiled and never dispatched ([12ff1a29](https://github.com/swedishembedded/brain/commit/12ff1a294ad81f211a317dcbbf47837c42c3930d))

- *(qwen)* Migrate KV-cache decode onto model::block's hoisted primitive ([97ed4a34](https://github.com/swedishembedded/brain/commit/97ed4a34947a83d4e3c503b4c435653c8de860d9))

- *(modelstore)* Generalize plan_base into a pluggable recipe registry ([2c10a364](https://github.com/swedishembedded/brain/commit/2c10a364ce88acb159abea172758a66aeb717047))

- *(kernels,residency,gpu-core)* Catalogue metadata tells the truth; bounded audit log; tune-store merge-on-save ([4009499e](https://github.com/swedishembedded/brain/commit/4009499eb79d9294c64fa223f9c704470d7c8ecb))

- *(capability)* One shared last_user_text; delete the three hand-synced copies ([9b358cd8](https://github.com/swedishembedded/brain/commit/9b358cd87c11ed91024171a5e0f67755feca5076))

- *(vlm)* Fastvlm/qwenvl preprocessors use crates/imaging, not hand-rolled bilinear loops ([0b5d8934](https://github.com/swedishembedded/brain/commit/0b5d893429d901f3d986f796e92b96ede472693d))

- *(model)* Hoist shared host LoRA pair math out of flux2/zimage into model::lora ([4fa58dab](https://github.com/swedishembedded/brain/commit/4fa58dabb38eb52fa96abb9914858f8202f9c1d6))

- *(flux)* Hoist the fp32/int8 linear-dispatch scaffolding into model::dispatch ([f2477f54](https://github.com/swedishembedded/brain/commit/f2477f5415bd0a333f2279d3c2fcb1df0aa36665))

- *(zimage)* One dit_config seam instead of three hardcoded ZImageConfig::turbo() build sites ([2efd8e72](https://github.com/swedishembedded/brain/commit/2efd8e7261563255b806cd491a48b9da8f4667f6))

- *(models)* Cosine and timestep_embedding call sites use model::hostmath, not local copies ([78352200](https://github.com/swedishembedded/brain/commit/78352200d2ca86eb130a1e97aa05427b7bef5a27))

- *(models)* Kronos/npu/tts LCG copies use the unified data::rng::Lcg ([8db58df5](https://github.com/swedishembedded/brain/commit/8db58df562a5087ac5af2be48ca5511e7766c672))

- *(backend-api,npu)* Delete dead GraphBackend trait (C1) ([4990bb99](https://github.com/swedishembedded/brain/commit/4990bb9984fd25cd2186b239978e5bfdd64fae9e))

- *(arch)* Rename text-decoder crates to their canonical arch ids ([b4e016aa](https://github.com/swedishembedded/brain/commit/b4e016aa47027532ec53b9c176bd899f3cc2beef))

- *(arch)* Rename multimodal/ASR crates to their canonical arch ids ([d0ec8539](https://github.com/swedishembedded/brain/commit/d0ec853952c9af954e82db48838b0e9023cd6918))

- *(arch)* Rename audio-stack crates to their canonical arch ids ([5a8140cd](https://github.com/swedishembedded/brain/commit/5a8140cd8aee19b9c57c388e0a6e91396b5d8518))

- *(arch)* Rename yolo/depth/t5 to their canonical arch ids ([11644b1c](https://github.com/swedishembedded/brain/commit/11644b1cef34f4831f3ebb0bb6e84c63b2949909))

- *(arch)* Rename diffusion/image crates to their canonical arch ids ([a6a47346](https://github.com/swedishembedded/brain/commit/a6a4734606ab5902a562ec37bc9bcc160d858655))

- *(arch)* Rename 3D/world-model crates to their canonical arch ids ([73db8304](https://github.com/swedishembedded/brain/commit/73db83040a685172378f5a87e9c8a48e6c148dfe))

- *(arch)* Rename brain's own toy architectures to their canonical ids ([a126e9b2](https://github.com/swedishembedded/brain/commit/a126e9b2203970fff225c961123156e66c1349db))

- *(env)* Rename BRAIN_<MODEL>_* env vars to their canonical arch id ([e2e25d87](https://github.com/swedishembedded/brain/commit/e2e25d8768ecc8589460687d64c477326d48a53f))

- *(bench)* Rename internal arch registry to canonical arch ids ([c6340334](https://github.com/swedishembedded/brain/commit/c63403341f5c9ae02c7fb0b8039ae4c81ee62d2c))

- *(facenet)* Split into crates/scrfd and crates/arcface ([d3ec333c](https://github.com/swedishembedded/brain/commit/d3ec333c7123b4621251bc18aec222257cb68678))

- *(facenet)* Remove the remaining crates/facenet files ([bc3ff7aa](https://github.com/swedishembedded/brain/commit/bc3ff7aaaaa1434978272a197cadb633c8520d67))

- *(catalog)* Rename served model ids to their canonical arch id ([bbe4ad38](https://github.com/swedishembedded/brain/commit/bbe4ad385bb59e4a83977fd2b5f7c7844b4a3f3e))


### Testing

- Require weight paths via env/CLI, never hardcode resource paths ([7793627b](https://github.com/swedishembedded/brain/commit/7793627b73feaf4e5d9d5508cb4f8e321b198546))

- End-to-end scheduler validation over D-Bus (BATS) ([435923d2](https://github.com/swedishembedded/brain/commit/435923d274b0e53007b5dcea61c02333d266a10d))

- Skip multi-GPU tests without the hardware; fix two stale fixtures ([46312951](https://github.com/swedishembedded/brain/commit/46312951cd652c4144bec177c83efc63479d16d3))

- Test lanes: move the 60s detection FD-gradcheck to the slow lane; size the guard above measured runtime

The run-phase timeout fired mid detection_loss_gradcheck_frozen_assignment
- not a hang: it passes in 60s standalone. A finite-difference gradcheck
is 'make gradcheck' coverage, not fast feedback. The deadlock guard now
sits above the measured ~950s serial run so a completing suite never
reads as a hang again. ([2ccb1b0c](https://github.com/swedishembedded/brain/commit/2ccb1b0cdfcfa5458bedb321a931c74e9f88d75b))

- Qwen/gpt/tts/speaker onto the shared testgpu device pool ([30ce626b](https://github.com/swedishembedded/brain/commit/30ce626b00943ba22a8f02dbba81980b45e9df5f))

- Finish the testgpu migration - every GPU test shares the pool ([89d8611f](https://github.com/swedishembedded/brain/commit/89d8611f918bbaa1b7caa2fec33309c0009faae7))

- Model-parity goldens join the fetched tree ([a4ff40ed](https://github.com/swedishembedded/brain/commit/a4ff40ed834818e548417888113ab28d07c33c23))

- *(api)* Mock model + HTTP conformance harness + claude e2e (P16) ([6ea0653d](https://github.com/swedishembedded/brain/commit/6ea0653d655017bf4ffc1320ef1deb89b8b63d2c))

- *(api)* Security conformance over the real socket + mock modes (P16 security) ([31c6b411](https://github.com/swedishembedded/brain/commit/31c6b4118c68d5017757717eb840b5beda5b56cb))

- *(e2e)* A regression harness that actually runs every example ([cb484b53](https://github.com/swedishembedded/brain/commit/cb484b53da3e0592a3a9c0623f7dc3e5e4a3bf81))

- *(gradcheck)* Finite-difference gates for sam2, facenet, vqgan and clip ([5ef0e339](https://github.com/swedishembedded/brain/commit/5ef0e3393d7ba1d3d4318b7969face9adfae45f3))

- *(qwen)* Gate A -- an always-run CPU proof that a LoRA adapter learned from its data ([77186190](https://github.com/swedishembedded/brain/commit/771861904bb4313e84d0354b17079471ab286cb0))

- *(perf)* Commit a qwen serving perf-gate baseline; bisect the M3->M4 gap ([c240310f](https://github.com/swedishembedded/brain/commit/c240310f52ff1be7eeeea579009d6ec45bca009a))

- *(wm-diamond)* Cover the attention path where the two graphs could drift ([f4654747](https://github.com/swedishembedded/brain/commit/f46547477e711e57726d7a6bcf65b36e971e6e45))

- *(model,qwen)* Gate paged attention at cap > max(seqlen) and cover the decode-window path ([c2b214e9](https://github.com/swedishembedded/brain/commit/c2b214e92c7c8e0b2426cfdebc9683391a418760))

- *(qwen)* Int8 KV scale/byte quantization gated against a same-engine oracle ([18e0fc11](https://github.com/swedishembedded/brain/commit/18e0fc1139471b4961c3e9eda4bb01535edd6ac4))

- *(qwen)* G4 -- the 8 int8-affecting invariant tests now run at both KV dtypes ([3f3ac7e9](https://github.com/swedishembedded/brain/commit/3f3ac7e9582b4240e33a637983255d1f1320bfe9))

- *(qwen)* Gate the paged-serve suite on the CPU backend (G6) ([17ead664](https://github.com/swedishembedded/brain/commit/17ead6644488714e019b01d1ab08123cd17de3da))

- *(cli)* Discovery gates for glm/yolo/depth checkpoints ([34ae2246](https://github.com/swedishembedded/brain/commit/34ae2246329b8ee6da93db244edb4b911e5669e4))

- *(omni)* Real end-to-end generation vs HF, on the now-complete checkpoint ([b16cdb96](https://github.com/swedishembedded/brain/commit/b16cdb96787513be376d41b1c6364c240e3f7f0f))

- *(omni)* Real end-to-end speak validation -- found and fixed a real bug ([528cbde5](https://github.com/swedishembedded/brain/commit/528cbde5fe386377eaf472e916ecf820ef04d399))

- *(gpu-core)* A Vulkan device-churn delay experiment narrows the residual theory ([1b90afb6](https://github.com/swedishembedded/brain/commit/1b90afb60e85c53b1135283a8c80e0817b613ee0))

- *(e2e)* Wire the OpenAI transport into the shared example-server harness ([364a01e2](https://github.com/swedishembedded/brain/commit/364a01e2dd4a08e78987d41c2732057b437c6eb6))

- *(e2e)* Guard README's Model support table against catalog drift ([84b3670a](https://github.com/swedishembedded/brain/commit/84b3670a54c2dda8f92fb6aca15397bff419a89b))

- *(qwen35)* Validate config_from_gguf + tokenizer against the real header ([6331cc58](https://github.com/swedishembedded/brain/commit/6331cc588d09200552aed587846203f3b23fefb5))

- *(gradcheck)* Split-K asserting oracle; GEMM-lowered conv backward under FD; dead-gradient guards ([83990db0](https://github.com/swedishembedded/brain/commit/83990db0082242a33865014d31dab97bb9d52d9e))

- *(models)* Moondream/nemotron/vision suites use the shared testgpu device, not a hard-pinned CPU handle ([551c43e3](https://github.com/swedishembedded/brain/commit/551c43e37b7141ed028ca97fb1bdc469f41d8a68))

- *(vlm)* Shared read_f32/read_i32 fixture readers; delete the triplicated dead repo_path ([02ab480a](https://github.com/swedishembedded/brain/commit/02ab480ad50439fed34fae85c4cb89c1f3504387))

- *(deepseekocr)* Checkpoint-free golden dumper + llama.cpp real-weight converter ([246a9d18](https://github.com/swedishembedded/brain/commit/246a9d1896da59a3206b73b9a73c530d1067ef03))

- *(deepseekocr)* Add LoRA descent smoke test (RED) ([0eab51ac](https://github.com/swedishembedded/brain/commit/0eab51ac51786f61de9cdc399ea14cb23bdae541))

- *(sam1)* Minimal checkpoint-free repro of the wgpu 3+-block corruption ([656cb5cc](https://github.com/swedishembedded/brain/commit/656cb5ccfbac75e5496344afc041156e508a5395))

- *(goldens)* Rename tools/goldens/*.py to their canonical arch id ([1e28972d](https://github.com/swedishembedded/brain/commit/1e28972d887862c0c351e97ddbe95325b9569623))

- Adopt cargo-nextest for per-test hang isolation and timeout ([4f4c3c33](https://github.com/swedishembedded/brain/commit/4f4c3c3376562b0cba0ea99e37f298098e28c958))

- *(deb)* Assert control metadata, md5sums and payload of the package ([88a9fbee](https://github.com/swedishembedded/brain/commit/88a9fbee0fd706820c7070c241ad8c06c297b493))


