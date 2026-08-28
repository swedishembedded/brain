# Changelog

All notable changes to brain (https://github.com/swedishembedded/brain) are
documented here. Generated with git-cliff from conventional-commit history;
see CONTRIBUTING or AGENTS.md for the commit-message convention.
## [1.1.0] - 2026-08-28

### Bug Fixes

- *(clippy)* Rephrase a doc line clippy read as an unmarked quote ([cd396773](https://github.com/swedishembedded/brain/commit/cd3967730423c74782855a22d86462ea9a2611f8))

- *(gates)* Enforce the no-machine-paths rule that only existed as prose ([a573b9bd](https://github.com/swedishembedded/brain/commit/a573b9bd7769a20f81c3ee493a5ef02385ab490f))

- *(checkpoint)* Read ZIP64 archives, so .pth files over 4 GB load at all ([ad7e55dd](https://github.com/swedishembedded/brain/commit/ad7e55dd17646ba43bede00b8f8e9ddc9aaf8402))

- *(modelstore)* Fetch Wan's native checkpoint instead of rejecting it ([573156f3](https://github.com/swedishembedded/brain/commit/573156f3948fbc5b7fbdc8e649bc82a4f4915d47))

- *(testutil)* Let a run demand that fixtures not be skipped ([64bcd5d6](https://github.com/swedishembedded/brain/commit/64bcd5d69da34a04f7b2bc63c91d9d1b62e4d92b))

- *(gates)* Refuse video and model weights in git, and drop the one that got in ([22c9e2ea](https://github.com/swedishembedded/brain/commit/22c9e2ea3e8203d93e3bbbea65cd20c58f2a02e6))

- *(perf)* Wire a fidelity check into ExecutorTarget ([0a070749](https://github.com/swedishembedded/brain/commit/0a0707491a4f171940fb4588b8a4c76ed0ed6e17))

- *(perf)* Wire a fidelity check into HttpTarget ([1989a8c2](https://github.com/swedishembedded/brain/commit/1989a8c244612e77a657cbe05230b86744551398))

- *(backend-cpu)* Size the transpose tile from the byte budget, not a head width ([7e28c7af](https://github.com/swedishembedded/brain/commit/7e28c7affb8e5e63c1161db2ff8ba5155b7d37eb))

- *(clippy)* Reflow two doc blocks and annotate parked scaffolding, 291 -> 283 ([03fab7fa](https://github.com/swedishembedded/brain/commit/03fab7fa0a12c1443f8dd1aa931156cc4a22ac04))

- *(worldmirror2)* Size the camera refine-net from the config, not a hardcode ([d26dd7bf](https://github.com/swedishembedded/brain/commit/d26dd7bf1b23a9c8a79668738f45d82f97a9364c))

- *(docs)* Reattach doc comments orphaned when helpers moved to hostmath ([413ab970](https://github.com/swedishembedded/brain/commit/413ab9708ea21ba63d6c2cc396afa9abd78f59ff))

- *(residency)* Write GB rather than 1 * GB in the budget test tables ([8147e963](https://github.com/swedishembedded/brain/commit/8147e9637b225c21cf3da297f798c9013f50d720))

- *(clippy)* The remaining lints, and set the ratchet to zero ([ac463c72](https://github.com/swedishembedded/brain/commit/ac463c721c504b7b1ed127f9d1660e56ef337d3c))

- *(gates)* Run the two kronos parity tests separately, as cargo requires ([f109a4b4](https://github.com/swedishembedded/brain/commit/f109a4b429e2b7139ec320d50a58f5d2fa07fcbe))

- *(kronos)* Use the shared prefill on the ForecastModel path, and report the real checkpoint ([a4071b66](https://github.com/swedishembedded/brain/commit/a4071b6676db11ea10b8d8d76d28bc949eaaf13a))

- *(kronos)* Detokenize the rollout inside its context window ([56140879](https://github.com/swedishembedded/brain/commit/561408796a5c40a9bf8da3654dd997ec689221c1))

- *(kronos)* Drop the rotary from the dependency layer at inference ([b64ac4b0](https://github.com/swedishembedded/brain/commit/b64ac4b00742783afd1b9137a26f0a7df1528258))

- *(kronos)* Default the context so the cache stays exact, and score the distribution ([74f5dc13](https://github.com/swedishembedded/brain/commit/74f5dc132944b2aec3d167b05d8968fb163bee95))

- *(gates)* Scope the machine-path check to Rust in both of its modes ([4131c89f](https://github.com/swedishembedded/brain/commit/4131c89f9515b3b81775d6e71ef661ca5ec29fc5))

- *(wan)* Stream the GGUF import instead of materialising it three times ([a5405597](https://github.com/swedishembedded/brain/commit/a54055975971be9b2474313eafecac6f358dd576))

- Stop three test paths reporting a real failure as a skip ([86beeeb9](https://github.com/swedishembedded/brain/commit/86beeeb9dc4dffd988e30aa6ec33882ab17fd2ec))

- Move kronos and chronos2's golden dumps out of git, into testdata/ ([b4e825c0](https://github.com/swedishembedded/brain/commit/b4e825c0e2065d4f31cb422790b1e7384fcda584))

- *(clippy)* Correct two mistyped literal suffixes in ltxv ([f740e4d4](https://github.com/swedishembedded/brain/commit/f740e4d448e867395edf28b368c5c4da2d14f07e))

- *(kronos)* Make the dependency layer causal in training, matching upstream ([29e21264](https://github.com/swedishembedded/brain/commit/29e21264fc45c7f5db4fc3838d8e9a968ef792ac))

- *(s3dit)* Stream the GGUF import, the same defect Wan's importer had ([40a21531](https://github.com/swedishembedded/brain/commit/40a215313acb1bd17c16141c516b134118b14462))

- *(forecast)* Validate finetune's input the same way predict already does ([57b8b282](https://github.com/swedishembedded/brain/commit/57b8b282cecf59005c70d59528ce9746c014f210))

- *(fetch-testdata)* Every mirror default named a path that does not exist ([6ec36139](https://github.com/swedishembedded/brain/commit/6ec36139518edb8ceb0943227557756934a5f49f))

- Remove upstream's own dead tracked golden binaries, kronos and chronos2 ([60117f56](https://github.com/swedishembedded/brain/commit/60117f56e9ef4c138879a9b7730b6ffcdbd5ab3b))

- *(clippy)* Is_multiple_of and div_ceil, upstream's ltxv/gemma4/vae drift ([9afcf725](https://github.com/swedishembedded/brain/commit/9afcf72543822d9459df1704f7ba39f6cb88ddf8))

- *(docs)* Reflow doc-list warnings, one of which was rendering wrong ([e1f0541e](https://github.com/swedishembedded/brain/commit/e1f0541ea04e7a86672494c07e0ec04e20bf6313))

- *(clippy)* Literal hygiene in ltxv - hex grouping, excessive precision ([884e84e5](https://github.com/swedishembedded/brain/commit/884e84e54aa389672e5104e6ce97ba987b511ba3))

- *(clippy)* Derive DfrOpts's Default instead of hand-writing it ([4f5b2057](https://github.com/swedishembedded/brain/commit/4f5b205744ae5bf6e50f0a2e6f5c57bdfd9c85da))

- *(clippy)* Doc_lazy_continuation in qwen35 config's tiny() doc comment ([b6d8ec17](https://github.com/swedishembedded/brain/commit/b6d8ec17b4f9902a8117ffd4e7f4511340a2e393))

- *(clip)* Give embed_image's shared Gpu the vision + resize kernels it needs ([bd41e0e6](https://github.com/swedishembedded/brain/commit/bd41e0e6e63df146fec81b5f3f0b53e39a858d70))

- *(ltxv)* Use the real distilled sigma schedule, not a generic formula ([0aeef320](https://github.com/swedishembedded/brain/commit/0aeef3203a8e1ced81bc6132ce9b86ee04e16010))

- *(cli)* Ltxv dispatch test fixture was missing connector_apply_gated_attention ([79aa9c73](https://github.com/swedishembedded/brain/commit/79aa9c732e3e004c82d332b29a2af226283cc665))

- Strip baked-in absolute machine paths from qwen35 docs, widen the gate ([3e77ad4e](https://github.com/swedishembedded/brain/commit/3e77ad4ee928cd6baf1d07476620789fa7601943))

- *(ltxv)* Clear all clippy warnings in crates/ltxv ([d03a1a2e](https://github.com/swedishembedded/brain/commit/d03a1a2eb81bc84e8e25d92c0fc4796ea9212fcb))

- *(ltxv)* Ltxv_bench streamed was profiling with the connector disabled ([bdeff8bc](https://github.com/swedishembedded/brain/commit/bdeff8bc81b9124fefaee0de767759dda858ea31))

- *(vulkan)* Make the Vulkan instance a process-lifetime singleton ([c6326c14](https://github.com/swedishembedded/brain/commit/c6326c146f346bcff7b7da8f2ce4e95eb5b21035))

- *(ltxv)* Pick the DiT fixture by architecture, not by filename ([0c2299c0](https://github.com/swedishembedded/brain/commit/0c2299c0b9a150e55da72c4ecf1737ae82f42fa7))

- *(ltxv)* The real Gemma-4 encoder pads context to 1024, not to the connector's own register count ([2fec1382](https://github.com/swedishembedded/brain/commit/2fec1382350b80e0e04499c360e9c03b087eedbc))

- *(ltxv)* Build RoPE positions in the reference's real pixel-space units, and add start/end-frame image conditioning ([091c3f9e](https://github.com/swedishembedded/brain/commit/091c3f9efa64e5c6fb1e3bd4fbc74471f2f4a322))

- *(ltxv)* Give frozen conditioning tokens their own zero timestep ([bf00210e](https://github.com/swedishembedded/brain/commit/bf00210e56ccce28a080bf682859df32af1a0246))

- *(ltxv)* Image conditioning was refusing the checkpoint's own sampler ([3eb16256](https://github.com/swedishembedded/brain/commit/3eb162569e456d752ac6bcc3c946ecafa99b36fd))

- *(cli)* Make -v/--verbose a global flag, not brain-serve-only ([8c49a417](https://github.com/swedishembedded/brain/commit/8c49a417afb56eda5b8b86c4b212cd3d6bb4c47c))

- *(model)* Use iter_mut instead of range-loop indexing in test fixture ([390d9994](https://github.com/swedishembedded/brain/commit/390d9994ba4ee4f68da12a09466144121617aa14))

- *(checkpoint)* Resolve diffusers-format single-file checkpoint dirs too ([1016dab8](https://github.com/swedishembedded/brain/commit/1016dab8e3836a24501f92d3443cfd2e2e3c4471))

- *(audio)* Needless_range_loop in bias_ncl_kernels' fwd_ref ([4a3d4172](https://github.com/swedishembedded/brain/commit/4a3d4172667791fcef2fb98e1eec1439ba52fd67))

- *(examples)* Musicgen README's CLI comparison used the wrong flag spelling ([45aac58e](https://github.com/swedishembedded/brain/commit/45aac58e9ffa14ef69f8e70a82fbd4ddf312a276))

- *(model)* Kernel_list() returns a slice, so its callers no longer borrow one ([d22b0251](https://github.com/swedishembedded/brain/commit/d22b0251b68598068d3a4333bb02afd5313f1a35))

- *(gpu-core)* The roofline probe measured the idle clock and cached it forever ([1bbcfc77](https://github.com/swedishembedded/brain/commit/1bbcfc77ed868ea01d52627f457b556f69750bb7))


### Build

- A seam for consuming a fixed wgpu, and the fix worth consuming ([1b033a34](https://github.com/swedishembedded/brain/commit/1b033a34870df9afd2eba979235a5990fee7e969))

- Point the wgpu patch fallback at the fork's future main branch ([6985cdc0](https://github.com/swedishembedded/brain/commit/6985cdc0fbb60d2056c8da44419fb5dee329a920))

- Name the manifest that actually broke the workspace ([31a6ca9f](https://github.com/swedishembedded/brain/commit/31a6ca9f1cfc5d3e29db9635931f034ea215d236))


### Documentation

- *(wan)* Set the bar for the CLI at one command, one playable file ([da9a2024](https://github.com/swedishembedded/brain/commit/da9a2024b9349f79a596a359012978d8c42c463d))

- *(wan)* Record the training state and what it does not cover ([232f000f](https://github.com/swedishembedded/brain/commit/232f000fe1ed1f2c248c9f56ce97b60dc0f66b8b))

- *(wan)* Quick start, runnable demo and the model page ([10e9bc81](https://github.com/swedishembedded/brain/commit/10e9bc819a46d38d495981f5841ff6789db45542))

- *(wan)* Mark the measured perf figures as reviewed exceptions ([be965fbf](https://github.com/swedishembedded/brain/commit/be965fbf803bd51cf47046e950055b866af94df2))

- *(build)* Say why `make build` is the dev profile, and leave it alone ([22435c72](https://github.com/swedishembedded/brain/commit/22435c728e586fc1446df668781fcede9c519452))

- List ltxv in the README model table and the model catalog ([03d5d208](https://github.com/swedishembedded/brain/commit/03d5d2080680c153bca258179c7c21e40c45dbcb))

- *(ltxv)* Bring the model page current through M9 ([e35cfe27](https://github.com/swedishembedded/brain/commit/e35cfe2739e346577ade8561843cb633ec1a4ba1))

- Reflow doc comments so wrapped prose is not parsed as a list ([0cc21ef1](https://github.com/swedishembedded/brain/commit/0cc21ef184e980715e2c0b9911a8cfff494042ae))

- *(quickstart)* Kronos forecasting, with a chart and an honest caption ([9a2b9396](https://github.com/swedishembedded/brain/commit/9a2b9396c3751583e7af3d1607bea86224fa15e2))

- *(quickstart)* Regenerate the kronos chart from the fixed model ([8254db9d](https://github.com/swedishembedded/brain/commit/8254db9dc173c1ba3105259c9ee5a5510936d90b))

- *(quickstart)* Regenerate the kronos chart on realistic data ([6cbbd2ae](https://github.com/swedishembedded/brain/commit/6cbbd2aef64413edf1729521f56309315da3aa48))

- Regenerate the kernel table after the rebase ([5ecf375a](https://github.com/swedishembedded/brain/commit/5ecf375ae3e9a42492060e1d47a24826e493acb2))

- *(kernels)* Stop stating kernel counts, total or per-tier ([a8b1186b](https://github.com/swedishembedded/brain/commit/a8b1186b0e8e0b7ecfcc9d199ff37fd4f316b52e))

- *(gemma4)* The missing model page, and a gate to stop it recurring ([c7acaaa0](https://github.com/swedishembedded/brain/commit/c7acaaa0fc93f65a66301ecc8fe614cf33c5346e))

- *(agents)* Distinguish FLUX.1's measured int8 cosine from its enforced floor ([e08a47b6](https://github.com/swedishembedded/brain/commit/e08a47b629bb3eb1cb413afd141f7500129ad3c1))

- *(ltxv)* Ledger the int8/AV-sharding milestone; fix the now-stale gap ([f4a6e58c](https://github.com/swedishembedded/brain/commit/f4a6e58c7a451166476bff9637d920a1c2925886))

- *(qwen35)* Real LoRA training example, fix stale hardware/limits claims ([9d2e58f6](https://github.com/swedishembedded/brain/commit/9d2e58f64a5c627efa9da3bc4ed021c4b30ad0df))

- *(qwen35)* Use printf for the example corpus, not a for-loop ([c455be7b](https://github.com/swedishembedded/brain/commit/c455be7b098d6d6abc0508ff194a124566863892))

- *(ltxv)* Strip milestone-number tags from comments and doc ([9d148cb2](https://github.com/swedishembedded/brain/commit/9d148cb2993f2bca7e7e2e14caa47503a654b1ea))

- Strip milestone-number tags from comments across apiserve/chronos2/cli/gemma4/gradcheck/mimi/qwen3/qwen3omnimoe ([533bd9ae](https://github.com/swedishembedded/brain/commit/533bd9ae1298f2a34b68d2433bc78a5341bfb178))

- Point quickstart's Where to go next at the model catalog ([c3a3d297](https://github.com/swedishembedded/brain/commit/c3a3d297aed28c1cd850ceff787ed9b65d56a90f))

- *(qwen35)* Strip milestone-number tags from comments ([f0eec509](https://github.com/swedishembedded/brain/commit/f0eec5095fd01a38e04fbf56cf0b6a1b5cf869f4))

- *(gpu-core)* Strip milestone-number tags from comments ([de670cb2](https://github.com/swedishembedded/brain/commit/de670cb2bf5a804762b026f5d1424cda0eb19be1))

- *(qwen3omnimoe)* Strip the last two milestone-number tags from comments ([5ee29ede](https://github.com/swedishembedded/brain/commit/5ee29ede53590857b801ac0c29a5e07f24063522))

- *(qwen35)* Record why the resident lm_head stays on the CPU backend ([80bcbddf](https://github.com/swedishembedded/brain/commit/80bcbddf11dbfe1c82e3ac217c4c6c7f3b6f2cb2))

- *(ltxv)* Ledger the ada_layer_norm_single parallelization (Phase 9) ([b160d752](https://github.com/swedishembedded/brain/commit/b160d752c265e1ec1a8fc6a5793ab91f9f3b138c))

- *(ltxv)* Ledger the host-side block-weight cache (Phase 9) ([4c503f7c](https://github.com/swedishembedded/brain/commit/4c503f7cc60af186b0971086673886eb402f735f))

- *(ltxv)* TeaCache-style temporal caching killed by measurement (Phase 9) ([b4c258a2](https://github.com/swedishembedded/brain/commit/b4c258a2ca0987176ff0dc6fe547dbdca553622b))

- *(ltxv)* Spatial masking (STA) scoped out on an analytic crossover (Phase 9) ([ef9b6124](https://github.com/swedishembedded/brain/commit/ef9b61242545d6553aeae40808e3fb8e7ecaa784))

- *(ltxv)* Cross-modal optimization scoped out, no real AV forward exists yet (Phase 9) ([ca34154f](https://github.com/swedishembedded/brain/commit/ca34154f3d8beca402c45b6e487e89fed3666afe))

- *(ltxv)* Close the Phase 7 "review: base vs finetuned clips" gate ([9974b893](https://github.com/swedishembedded/brain/commit/9974b893c6f2dd88ef4b3febaa07c118af5c03bb))

- *(ltxv)* Ledger the --trace-ltxv observability milestone ([15b19ee0](https://github.com/swedishembedded/brain/commit/15b19ee0c665db62e8cbc306ec6dbe3c2cc520b8))

- *(ltxv)* Ledger the whole-generation profile and four exact wins (Phase 10) ([33d8c681](https://github.com/swedishembedded/brain/commit/33d8c681043b883252d309e345965604e236760f))

- *(lessons)* A stage missing from the timing struct, and a bench warmed by page cache ([eaa2a630](https://github.com/swedishembedded/brain/commit/eaa2a630def386fa166496046cf92a344e941fdb))

- Two absolute machine paths the no-machine-paths gate was refusing ([6f3d1621](https://github.com/swedishembedded/brain/commit/6f3d16216f4ce01bead09977dd84aee5a96ff541))

- *(ltxv)* Ledger the text-encoder phase, and close two Phase 10 gaps ([bc6534fc](https://github.com/swedishembedded/brain/commit/bc6534fca7d31c11a877de63d16d9ba9f72feaf2))

- *(ltxv)* Ledger the connector, streamed-path, and Gemma-4 real-weight parity gates ([fd2a00d4](https://github.com/swedishembedded/brain/commit/fd2a00d44611019bf951b28832fd580e2cbf9666))

- *(lessons)* A parity gate built from the pipeline's own simplified positions can't catch the pipeline getting positions wrong (#48) ([36aa3ded](https://github.com/swedishembedded/brain/commit/36aa3ded74cdf5c022bc473d8d8ee4d2dadd6767))

- *(ltxv)* Ledger Phase 18 - the forward stops being a PCIe benchmark ([63670013](https://github.com/swedishembedded/brain/commit/63670013bd93105bdf4b0ee6a75621a72cbe7619))

- *(backend-vulkan)* Ledger the device-lost root cause, the 2x SPIR-V deficit, and wgpu's staging heap ([2a027c21](https://github.com/swedishembedded/brain/commit/2a027c21d765c4abe90a0127bb65995ad826d129))

- *(backend-vulkan)* The measured roofline table, both backends, on an idle box ([bd965438](https://github.com/swedishembedded/brain/commit/bd9654387c85887cb0a49c6fd551c977a31d5548))

- *(qwen35)* Record the wgpu buffer-size wall and native-Vulkan findings ([6635b76e](https://github.com/swedishembedded/brain/commit/6635b76e454e5dedb61e5e4220a9eb7c0b0613e8))

- *(minimaxmusic3)* Record the flow-matching DiT milestone ([fad3bc0c](https://github.com/swedishembedded/brain/commit/fad3bc0c5454938dd25c63d804eabb5590b130bf))

- *(minimaxmusic3)* Diagnose and record the real e2e RAM/buffer gap ([49398c8c](https://github.com/swedishembedded/brain/commit/49398c8c35c3ed0c2d718942fb00693d208c0738))

- *(examples)* Musicgen/generate_song.py - MiniMax Music 3 over D-Bus ([63a53fc8](https://github.com/swedishembedded/brain/commit/63a53fc8f915954a731dac939246679c5be1aedb))

- *(minimaxmusic3)* Record M8 (serving contract), update user-facing docs ([a526cdb6](https://github.com/swedishembedded/brain/commit/a526cdb6646b1ceb4ba892851de40ba3a58b27db))

- *(minimaxmusic3)* Update module doc for the serving milestone ([e98b4e8a](https://github.com/swedishembedded/brain/commit/e98b4e8a9ab78865b0a6eb32db8aca5f2c04500c))

- *(minimaxmusic3)* Correct a stale "single-chunk only" gap entry ([cbe45be3](https://github.com/swedishembedded/brain/commit/cbe45be37bb321164287a35e71462c8c75bdfc1c))

- *(cli)* List minimaxmusic3 among the generic-dispatch architectures ([23a366f4](https://github.com/swedishembedded/brain/commit/23a366f466d38401cf60a09eee8e780f50172503))

- *(audio)* Consolidate crate-level module doc for the new front-end/back-end pieces ([60cd7231](https://github.com/swedishembedded/brain/commit/60cd7231051e8bc2a55ba5fe1a6b2f20adf60bc7))

- *(backend-vulkan)* Say which state the box was left in ([3f576422](https://github.com/swedishembedded/brain/commit/3f57642235d4be5fbfaf27f99e2e539a7bb1632c))

- *(backend-vulkan)* The upload regression was two bugs, and both are fixed ([3ab1337b](https://github.com/swedishembedded/brain/commit/3ab1337b1527fd2b50297613ff9bf7d05edb005e))

- *(ltxv)* The bit-identity claim, measured rather than argued ([291549cc](https://github.com/swedishembedded/brain/commit/291549cc4a160c469701d8398c9581e2158f48a1))

- *(ltxv)* Ledger Phase 20 - the prompt starts reaching the picture ([a8f414e5](https://github.com/swedishembedded/brain/commit/a8f414e59c1b9d8f22241d674c7079ededab36af))

- *(cosyvoice)* Record Phase 7a (CosyVoice 3 golden dumper) in the roadmap ([b07475d3](https://github.com/swedishembedded/brain/commit/b07475d356a4d61ea9296d8ab0cf5ca7539b732c))

- *(cosyvoice)* Record Phase 7b (CosyVoice 3 LM/flow/HiFT) in the roadmap ([abeb8046](https://github.com/swedishembedded/brain/commit/abeb8046ac4d604e3cc5d0bd4fce316812dd92f8))

- *(cosyvoice)* Record Phase 10 (partial) LM training in the roadmap ([3a56e00d](https://github.com/swedishembedded/brain/commit/3a56e00dee9a8091ebf5ef485d6296e0dfe0addf))

- *(cosyvoice)* Record Phase 11 (serving contract) in the roadmap ([c5aeb058](https://github.com/swedishembedded/brain/commit/c5aeb0582dbc2031441f7d4d2a5b8714b402f35a))

- *(cosyvoice)* Add CosyVoice to AGENTS.md, README.md, and the model index ([1014b1e9](https://github.com/swedishembedded/brain/commit/1014b1e9e93957c55428da48826326a05da84902))

- *(cosyvoice)* Record Phase 13 (profiling + NPU export) in the roadmap ([f416dfe6](https://github.com/swedishembedded/brain/commit/f416dfe68a164516204a02ad45a9658a3495ad99))

- *(ltxv)* Ledger Phase 21 - the upscaler stops being generation-only ([d1d25585](https://github.com/swedishembedded/brain/commit/d1d255853d2e129baebcbaace115f792cfda1226))

- *(ltxv-cli)* The module doc names all three subcommands, and stops being wrong ([fc03df10](https://github.com/swedishembedded/brain/commit/fc03df1081ba8e78b648e0218104f8fe2faa9d0d))

- *(ltxv)* Ledger Phase 22 - a clip stops being one window long ([700f793a](https://github.com/swedishembedded/brain/commit/700f793a168d144d3ee20a11697e979e1971f21a))

- *(ltxv)* The seam gate's cost is measured, and names the pin that works ([d23e1816](https://github.com/swedishembedded/brain/commit/d23e18169310233590e0401bc040c06578f9a76c))

- *(ltxv)* Ledger Phase 23 - the long-form seam, re-derived and measured ([eff182d5](https://github.com/swedishembedded/brain/commit/eff182d52021734a16d430d6a2ea05a90d7f9b04))

- *(ltxv)* Ledger Phase 24 - a clip stops being one scene long ([0c9c34ad](https://github.com/swedishembedded/brain/commit/0c9c34ad8c953a222c0d583a1e42fdcd8d5898e2))

- The published manual names its publisher ([8ff79f57](https://github.com/swedishembedded/brain/commit/8ff79f578d8951780bba3bfcf16ed92a8e2284ea))

- The two entry pages point at who builds brain ([599a24ec](https://github.com/swedishembedded/brain/commit/599a24ec50604f69a0b5a2ba276c6c1543242a62))

- The ledgers stop claiming work that is already done ([704cfc64](https://github.com/swedishembedded/brain/commit/704cfc64ce74da09d35f688d26f425d359e69c6d))

- Document 19 BRAIN_* variables, and stop the perf gate tripping on identifiers ([97fd989a](https://github.com/swedishembedded/brain/commit/97fd989a6065b89eec9fae81673085bcb39a827d))

- Fix 18 broken links, and gate them so they cannot rot again ([766b441e](https://github.com/swedishembedded/brain/commit/766b441e55ec7ac49bd0887cb0325a50d4ddb45c))

- Three support tables disagreed with the code that derives HTTP exposure ([e2478153](https://github.com/swedishembedded/brain/commit/e247815352f6bec3764d1c452875bb6b4082bec4))

- Six support tables said nothing about HTTP; now they say why not ([7a878906](https://github.com/swedishembedded/brain/commit/7a878906ba5727b60acf4256e773fac0257bfb09))

- *(ltxv)* Ledger Phase 25 - an upscaled clip stops being several clips ([67b67eb8](https://github.com/swedishembedded/brain/commit/67b67eb8414ed724f365cd39667d4b4b6d3f91ba))

- *(ltxv)* Ledger Phase 26 - a clip stops being anchored only at its ends ([87a87ebf](https://github.com/swedishembedded/brain/commit/87a87ebfb8ea291d5610b488e90612e5f3a7df25))

- *(minimaxmusic3)* Ledger Phase 12 - the blockers belonged to the old box ([a68d59da](https://github.com/swedishembedded/brain/commit/a68d59da28902202d43277e2fd98ed9317d60a42))

- *(minimaxmusic3)* Ledger Phase 13 - the first real song ([e2e01145](https://github.com/swedishembedded/brain/commit/e2e0114599b01099b86cca402280c20d34913df0))

- *(minimaxmusic3)* Correct a mislabelled denoise time in Phase 13 ([948502aa](https://github.com/swedishembedded/brain/commit/948502aaa5bfe862ca80d04323997f95665ff665))

- *(minimaxmusic3)* Ledger the resident hoist, the device head, and two-card AR ([a215e1b1](https://github.com/swedishembedded/brain/commit/a215e1b1a4c14b4b8f11be215348e2179bd7fc59))

- *(kernels)* Ledger the decode-GEMV occupancy work and its dead ends ([efdffdf5](https://github.com/swedishembedded/brain/commit/efdffdf54aff3aa224ba25376f12da78805780d0))

- Ledger the conv lowering, the sentinel bug class, and the measurement traps ([f14a0843](https://github.com/swedishembedded/brain/commit/f14a08439630737f288d90b2b0e832e2edd2d31a))

- Document every BRAIN_* variable the crates actually read ([89c51e57](https://github.com/swedishembedded/brain/commit/89c51e574a57c547b1559630591d12067d681e38))

- Ledger the adaLN dedup, and two lessons about what a measurement means ([98112491](https://github.com/swedishembedded/brain/commit/981124919bdc475c6b24cf443208cdb4346164b8))

- *(wan)* Describe the cost shape, not a measured run ([4c6f99da](https://github.com/swedishembedded/brain/commit/4c6f99daabcd25da2ca6eed38e9baf18c017a5f7))

- Ledger the flash cross-attention, the audio wiring, and three rules ([82c1839f](https://github.com/swedishembedded/brain/commit/82c1839fdce1a621f1ba22376bb46cda408c5876))

- The caption file is YAML, described as such ([8e34734d](https://github.com/swedishembedded/brain/commit/8e34734dd6e028bb731b5d67a0a5124dccfee479))

- Show what --strength does, on the room the README generates ([0bf4966d](https://github.com/swedishembedded/brain/commit/0bf4966deafb368ef2fbcb4835989b139e40d50a))

- Correct served-model status and stale crate-path citations ([bcbfeb39](https://github.com/swedishembedded/brain/commit/bcbfeb390afdf1ee30d73df27eb84b26ef37eb09))

- What an idle GPU costs a measurement, and what the coopmat kernel is not ([9bdbec6c](https://github.com/swedishembedded/brain/commit/9bdbec6c00292ba1631a96d0748fb2bdefff84bd))

- *(probe)* The end-to-end numbers, and what the chassis does to them ([3a94ab57](https://github.com/swedishembedded/brain/commit/3a94ab577236426d2721753f41f9743647aae1b6))

- Where a qwen3vl caption's time goes, and what is left ([2c91b981](https://github.com/swedishembedded/brain/commit/2c91b9811b540cb781f93dab3b8f79db302b39ac))

- The int8 tier's real cost, and the batching arithmetic ([53c0ed7e](https://github.com/swedishembedded/brain/commit/53c0ed7ece84c3b8d74db93cfd976cda21eb8a33))


### Features

- *(wan)* The Wan-VAE, a causal 3D autoencoder at (4, 8, 8) stride ([1985fe2a](https://github.com/swedishembedded/brain/commit/1985fe2a7072ec50a8d921d94c207b1f54bd2cf0))

- *(data)* SentencePiece unigram tokenizer ([5ac489e9](https://github.com/swedishembedded/brain/commit/5ac489e9bdeaf64e8a2d9e7912512970d0c46d85))

- *(kernels)* Attn_keypad_mask, the bidirectional twin of attn_prefix_mask ([6f98f6db](https://github.com/swedishembedded/brain/commit/6f98f6db234444a75f04b676ca9f4e8b60811f1b))

- *(t5encoder)* UmT5-XXL, the text encoder Wan conditions on ([c5d5e7da](https://github.com/swedishembedded/brain/commit/c5d5e7da4c61eab251bce403044c970137e79c8a))

- *(wan)* The diffusion transformer, at parity on the real 1.3B weights ([2c78af10](https://github.com/swedishembedded/brain/commit/2c78af10f83470a71672d013a9b7e83736e27bad))

- *(wan)* Text to video from one command, ending in a playable mp4 ([8542058b](https://github.com/swedishembedded/brain/commit/8542058b2857ac44594cd9f9dd422ce0725a05eb))

- *(capability)* A Video media type, replacing the Bytes workaround ([ca572b42](https://github.com/swedishembedded/brain/commit/ca572b421ab4b70edce4400388cd6a79967cfa6e))

- *(wan)* Serve text-to-video over the capability surface ([af5277e8](https://github.com/swedishembedded/brain/commit/af5277e8f8ee4cb7ad0cfaf87278c2f6634a4eed))

- *(wan)* GGUF importer for the wan architecture tag ([bc429139](https://github.com/swedishembedded/brain/commit/bc4291397ff417770b6eb78ebf54ba6d062a9618))

- *(wan)* Host training reference, generic over the float type ([3cb7cad2](https://github.com/swedishembedded/brain/commit/3cb7cad24d5bdd956c5a810163a91bd0ea030ce1))

- *(gradcheck)* Check_wan, alongside the other model gradient checks ([6963183f](https://github.com/swedishembedded/brain/commit/6963183f77a21ecba50722d132903d61378bd237))

- *(wan)* LoRA adapters and video-clip fine-tuning ([bfeb29c9](https://github.com/swedishembedded/brain/commit/bfeb29c9e1c9af9235a674e3f4f78269c74e285b))

- *(perf)* A byte-exact fidelity comparator alongside the greedy one ([1a17d805](https://github.com/swedishembedded/brain/commit/1a17d805a2e36f7b3bab3d5ed2948b6813ed5b0d))

- *(perf)* Say out loud when a compared result was never fidelity-checked ([fddf866a](https://github.com/swedishembedded/brain/commit/fddf866a848012f75576a983c4dd9c9e8364e2df))

- *(arch)* Register ltxv, the LTX-2.5 audio+video diffusion transformer ([e31ff6da](https://github.com/swedishembedded/brain/commit/e31ff6da8dfe94b834e9804bacb99cbbf27fc037))

- *(ltxv)* Dump reference goldens for the video VAE, tiny DiT, audio VAE, schedule ([43938fdd](https://github.com/swedishembedded/brain/commit/43938fdde18ad8eb00677c5a40bbfa340f16e43d))

- *(ltxv)* Port the LTX-2.5 causal 3D video VAE (encoder + conv decoder) ([2bf69a4c](https://github.com/swedishembedded/brain/commit/2bf69a4c3b6566b9e2d34252739267a3bf3671f6))

- *(ltxv)* Port the LTX-2.5 video-only DiT stream (M3) ([eb5d5685](https://github.com/swedishembedded/brain/commit/eb5d5685205680e2ce34260ef8cd10c7af632c97))

- *(ltxv)* Pipeline, CLI, and serving contract (M4) ([ab2d4709](https://github.com/swedishembedded/brain/commit/ab2d470952b3ca39f6a7bc9847c00ecedb8e6517))

- *(gemma4)* Port LTX-2.5's Gemma-4 text encoder (M5) ([948a196c](https://github.com/swedishembedded/brain/commit/948a196c441e2a5f121ea72bc40c76e15f1b8c0e))

- *(ltxv)* Port the LTX-2.5 audio VAE and base vocoder ([3e37207b](https://github.com/swedishembedded/brain/commit/3e37207b820cb8c573c471793a3a74f50be1336c))

- *(ltxv)* Add the audio DiT stream and A<->V cross-attention (M6b) ([faa341dc](https://github.com/swedishembedded/brain/commit/faa341dc1dd2c4a539d98106915ee56dbad71db1))

- *(ltxv)* Add training support for the video-only DiT (M7) ([1b51711e](https://github.com/swedishembedded/brain/commit/1b51711ee4f62a6354a6fedd4f0afb27532c3b83))

- *(ltxv)* Port the LTX-2.5 latent upscalers and duration head (M8a) ([0e8ae281](https://github.com/swedishembedded/brain/commit/0e8ae28136833570f58779c9efd0f044c4e7a6a7))

- *(ltxv)* Port the NA diffusion video decoder (M8b) ([a310eda8](https://github.com/swedishembedded/brain/commit/a310eda8796c10a4dd0e53d5a810a528ac406755))

- *(ltxv)* Add the DFR multi-stage pipeline (M8c) ([ade67c86](https://github.com/swedishembedded/brain/commit/ade67c86060c0a849cd8ec0c700b3ba6065c5ca5))

- *(ltxv)* Add INT8 storage format for the DiT's weights (M9 slice) ([c02e5d3e](https://github.com/swedishembedded/brain/commit/c02e5d3ec3a823d968728c9f31599ca2fcf43ed3))

- *(ltxv)* Add pipeline-parallel sharding for the video-only DiT (M9 slice) ([2e73bbda](https://github.com/swedishembedded/brain/commit/2e73bbda7c13307ba56fb24c8d5e1b5282b52182))

- *(ltxv)* Add a performance profiling pass and record the NPU scope decision (M9 slice) ([a2e63b89](https://github.com/swedishembedded/brain/commit/a2e63b89b8b3dd9aa5352829ca692ec7226ebd2b))

- *(kronos)* Auto-fetch both checkpoints, and settle on one env-var spelling ([cd6f12ae](https://github.com/swedishembedded/brain/commit/cd6f12ae73b9afda12ef50738378d52cdbf0d742))

- *(forecast)* OHLCV CSV in, forecast chart out ([61ae72bd](https://github.com/swedishembedded/brain/commit/61ae72bd6c49272e165f7f46cb1d106eac847efe))

- *(forecast)* Generate a series with the character Kronos was trained on ([defc95a3](https://github.com/swedishembedded/brain/commit/defc95a3226edd396b525fe5d070578f562072e5))

- *(qwen35)* Register the qwen35 arch id and disambiguate names (M0) ([3e89273d](https://github.com/swedishembedded/brain/commit/3e89273d39e43fe1ccd9b4cf948f89ed426f9d64))

- *(qwen35)* Dump tiny-dims reference goldens for the dense hybrid decoder (M2) ([86e5e72d](https://github.com/swedishembedded/brain/commit/86e5e72d6a5f1c8f6aa98b5e61695f2cd5aba46f))

- *(qwen35)* New crate - config, param manifest, fresh-weight init (M3) ([401a8055](https://github.com/swedishembedded/brain/commit/401a80551ffaa786e0d2083e54d9f4633f2f25cb))

- *(qwen35)* FP8 blockwise import with two-way coverage (M4) ([26b234b3](https://github.com/swedishembedded/brain/commit/26b234b33ae67fc19d3f59e7ed9f9f528113b915))

- *(qwen35)* Text-only forward at tiny dims, full parity vs the golden (M5) ([1ad2ad4d](https://github.com/swedishembedded/brain/commit/1ad2ad4df30beb070edbf54624587ce705c14bf7))

- *(qwen35)* Backward + gradcheck::check_qwen35 (M6) ([46001e75](https://github.com/swedishembedded/brain/commit/46001e75ddeeb8c82ff70ef4fbc89a0d51a40905))

- *(qwen35)* MTP head (M7) ([1d29dd57](https://github.com/swedishembedded/brain/commit/1d29dd57b31721d50c5dd82f09ceb8c64a06148b))

- *(qwen35)* LoRA + full finetune (M8) ([47b31a12](https://github.com/swedishembedded/brain/commit/47b31a12669c70e891c2aed22f679bba01542029))

- *(qwen35)* Vision tower splice, real-dims parity (M9) ([279bd90a](https://github.com/swedishembedded/brain/commit/279bd90ab46d30087d73b2186bf046547e71dcc4))

- *(qwen35)* Incremental decode + pipeline sharding (M11 core) ([e8aa8580](https://github.com/swedishembedded/brain/commit/e8aa85803ebacb80170b823749b5ab5d9918dbe2))

- *(qwen35)* Sampling, paged serving engine, capability provider (M11) ([be1c304f](https://github.com/swedishembedded/brain/commit/be1c304f77547293f75594e3fd771e6cc263a1b1))

- *(qwen35)* CLI, residency, catalog, docs (M11 finish) ([7e2faff5](https://github.com/swedishembedded/brain/commit/7e2faff59b716bf6c2997262aad939ef61f716fd))

- *(wan)* LoRA finetuning - gradient-checked CPU/GPU trainer, G0-G3 gates ([b1205a7e](https://github.com/swedishembedded/brain/commit/b1205a7ec85542e56d5bbc0b5ee16c7f455048ce))

- *(wan)* Direct GGUF loading (f32/f16/int8/int4) + adapter/dtype CLI ([75bf364d](https://github.com/swedishembedded/brain/commit/75bf364d1f1689b75abcc60ace3fe26e35664e32))

- *(ltxv)* Real 22B config, AV tensor manifest, direct GGUF import ([6e3df2b9](https://github.com/swedishembedded/brain/commit/6e3df2b90f75047b7768f9ca2e6cb6a686ea7538))

- *(ltxv)* Gated attention + embeddings connectors, parity-proven ([edc87525](https://github.com/swedishembedded/brain/commit/edc875251498ce4cccfe9b158fe05dc04d18981c))

- *(gemma4)* Real 12B config, two-way import, real tokenizer ([addcf02e](https://github.com/swedishembedded/brain/commit/addcf02ec367237ac8df685cbacaae18293afadb))

- *(ltxv)* Real-weight parity ladder for the 22B DiT, reduced depth ([d39c8cfa](https://github.com/swedishembedded/brain/commit/d39c8cfa699909790f12ef19ad422c65ad38aa9e))

- *(ltxv)* Int8/int4 compute path + AV sharding, run for real on both P40s ([8d4767e3](https://github.com/swedishembedded/brain/commit/8d4767e33fc8100ac497e8da52993c8f51796b66))

- *(ltxv)* Wire real weights into brain ltxv t2v, first real clip ([22ce8de9](https://github.com/swedishembedded/brain/commit/22ce8de92ecadbfee497a804f65f633d1c1ad282))

- *(ltxv)* Fine-tune validation for the AV DiT (Phase 7) ([1b431009](https://github.com/swedishembedded/brain/commit/1b4310091839fa250f9839c8891ad7b6be7c839a))

- *(ltxv)* Brain perf integration, committed baseline gate, fix real dit_config gap ([73436038](https://github.com/swedishembedded/brain/commit/7343603841a77b5fbeb10626b0bdd1003023957c))

- *(trace)* Workspace-wide structured tracing behind a --trace-<family> registry ([c83ccd06](https://github.com/swedishembedded/brain/commit/c83ccd06f034cfb5e810827ddcf8890715120c95))

- *(ltxv)* Instrument the video pipeline, streamed DiT and served action ([fbfe6ad3](https://github.com/swedishembedded/brain/commit/fbfe6ad3c0480f2197222c148604b8cfe1956fdf))

- *(gpu)* Instrument the four GPU crates for --trace-gpu ([bf998830](https://github.com/swedishembedded/brain/commit/bf998830641d2f1e6e6e8e8cf15d66145f5e5071))

- *(checkpoint)* A generic source-to-quantized-GGUF converter ([11a7fc9b](https://github.com/swedishembedded/brain/commit/11a7fc9b59932c7b6304d444aaa894d8357b9285))

- *(cli)* Brain quantize, the export sibling of brain import ([4625966d](https://github.com/swedishembedded/brain/commit/4625966d27d9da2958e4960638fe33c5d449b058))

- *(gemma4)* A streamed, capability-gated int8 tier over a quantized GGUF ([972519cc](https://github.com/swedishembedded/brain/commit/972519cc0693c544a1ebf5dcbd7119ffb3b4a557))

- *(flux2)* Wire the real FLUX.2 Klein 4B repo into auto-fetch ([cf966c11](https://github.com/swedishembedded/brain/commit/cf966c117c5258d47b383b0416fae26d77bd9617))

- *(qwen35)* Let stream_train_step pick cpu/gpu/vulkan explicitly ([6e499240](https://github.com/swedishembedded/brain/commit/6e499240ee6646bba5ad9089f06b575ac0e40df6))

- *(diffusion)* Let FlowMatchEulerScheduler invert its sigma schedule ([c0468958](https://github.com/swedishembedded/brain/commit/c0468958f3ec819a349dd4fb19be9da4ce5ffbe7))

- *(minimaxmusic3)* Register the architecture and add the crate skeleton ([dedb7e6d](https://github.com/swedishembedded/brain/commit/dedb7e6d40dadab734962d88d2422a748b74ae7c))

- *(minimaxmusic3)* Golden dumper for the four diffusers-PR components ([f3375e40](https://github.com/swedishembedded/brain/commit/f3375e400456dff819b129aa4a15a39ca1039ee0))

- *(minimaxmusic3)* Condition encoder import + forward, real-weight parity ([8d1f68ac](https://github.com/swedishembedded/brain/commit/8d1f68ac69eaf52cb48c37db1ed78a17dddb9afd))

- *(kernels)* Add the Snake1d activation (fwd + backward) ([1c0cbc08](https://github.com/swedishembedded/brain/commit/1c0cbc08e8547aa429e316884e477dc71ce61196))

- *(minimaxmusic3)* Vocoder import + device forward, real-weight parity ([a595874c](https://github.com/swedishembedded/brain/commit/a595874c5770e22a34c2330012cb4d04f6e094aa))

- *(kernels)* Add bias_grad_ncl and tanh_act_bwd ([ac9761fa](https://github.com/swedishembedded/brain/commit/ac9761fa3e22a6685fb9ac0ce98b4092dd914b9c))

- *(minimaxmusic3)* Vocoder backward + gradcheck ([9cbdb87b](https://github.com/swedishembedded/brain/commit/9cbdb87b402dbac046b134812246652d8ba945eb))

- *(minimaxmusic3)* Vocoder LoRA fine-tuning ([fc1513df](https://github.com/swedishembedded/brain/commit/fc1513dff193330c465619be458461ca86212f07))

- *(minimaxmusic3)* STFT-magnitude adversarial discriminator training ([205d2c48](https://github.com/swedishembedded/brain/commit/205d2c48741dcae5efc3be8e9e730c8b5032c55e))

- *(minimaxmusic3)* RVQ depth decoder import + forward/backward + LoRA ([ba1a223b](https://github.com/swedishembedded/brain/commit/ba1a223b94e088c647c85ac3795a5c1987f1d261))

- *(minimaxmusic3)* Flow-matching DiT import + device forward, real-weight parity ([10ade4d2](https://github.com/swedishembedded/brain/commit/10ade4d2e06a231d190deacd85256e368abc63bc))

- *(minimaxmusic3)* DiT backward + gradcheck ([21adb384](https://github.com/swedishembedded/brain/commit/21adb384f72d0ea78336609cbedd04581eeb93a4))

- *(minimaxmusic3)* DiT LoRA fine-tuning ([36f96360](https://github.com/swedishembedded/brain/commit/36f96360b5a89c9e98e00f3b27f4a88621ce39d8))

- *(minimaxmusic3)* DiT int8 storage tier ([00caa7e6](https://github.com/swedishembedded/brain/commit/00caa7e65a774ef3afcca9a9e0baaf080df91bee))

- *(minimaxmusic3)* DiT model::Shardable pipeline-parallel sharding ([2f5ce320](https://github.com/swedishembedded/brain/commit/2f5ce32020a7e99f0a8be479f1f663dc93364064))

- *(minimaxmusic3)* Global LLM streamed import + audio-code CE training ([00d417f6](https://github.com/swedishembedded/brain/commit/00d417f6101143904217ba6512d0b2601c9144b9))

- *(minimaxmusic3)* Port the caption/lyrics text-cleaning contract ([7a73e5e2](https://github.com/swedishembedded/brain/commit/7a73e5e2fce329d65ca32df0afc0d7f312f20de5))

- *(qwen3)* Expose a standalone embedding-row lookup (Qwen::embed_row) ([68a323f7](https://github.com/swedishembedded/brain/commit/68a323f7e6e77a9149164c4c0bb02d6b4177c429))

- *(minimaxmusic3)* CFG-guided AR generation loop (pipeline::generate_frames) ([6401676f](https://github.com/swedishembedded/brain/commit/6401676f56bd6eb5563b68c9691ad9b87ecc5e99))

- *(minimaxmusic3)* Chunked DiT denoise (denoise::denoise_chunk) ([a766c26a](https://github.com/swedishembedded/brain/commit/a766c26a2667ad66397a80aa6bea2590cb04de98))

- *(audio)* Multi-channel WAV write (wav::encode_multi/write_multi) ([59fa2a34](https://github.com/swedishembedded/brain/commit/59fa2a34e73385eeb2103b37d9b5d32d821c09de))

- *(minimaxmusic3)* Vocoder crop-and-stitch (stitch::Stitcher) ([e45a52a6](https://github.com/swedishembedded/brain/commit/e45a52a6e59c650dd75649f35814d79d338b32af))

- *(minimaxmusic3)* AR prompt assembly (global_llm::assemble_prompt) ([090afd67](https://github.com/swedishembedded/brain/commit/090afd67ded848c9018ec8540de34f28fa240f69))

- *(minimaxmusic3)* Serving contract - generate::generate + caps::Provider ([88f08a8c](https://github.com/swedishembedded/brain/commit/88f08a8c1f2a49c5d996ee8be4838328e9db73b4))

- *(cli)* Wire minimaxmusic3 into catalog/resolve/residency/D-Bus ([c7335d65](https://github.com/swedishembedded/brain/commit/c7335d655f8cc26d5b5f3f00629c19a3de058591))

- *(cosyvoice)* Reserve architecture names for CosyVoice 2/3 port ([09031225](https://github.com/swedishembedded/brain/commit/0903122556968608c2f38976f7c7e29f5bed2d87))

- *(audio)* Mixed-radix FFT + center=False mel + CosyVoice mel preset ([dc9315c4](https://github.com/swedishembedded/brain/commit/dc9315c44d500f9269b9f790d35277d9bbb228c2))

- *(audio)* Audio::istft - inverse STFT via overlap-add + NOLA norm ([22059417](https://github.com/swedishembedded/brain/commit/2205941746c130be62af04ba09293bf147beac8f))

- *(audio)* Audio::resample::rational - Kaiser-windowed-sinc resampler ([e532b63d](https://github.com/swedishembedded/brain/commit/e532b63da7ae21e109a218ba565dddddf94e198b))

- *(kernels,audio)* Elu/elu_bwd kernels + audio::act - CosyVoice's f0 predictor ([1fecac85](https://github.com/swedishembedded/brain/commit/1fecac85dda4ffd8577d1cd8aef892f012773c34))

- *(campplus)* CAM++ speaker encoder - import + forward, real-weight parity ([c3e369bb](https://github.com/swedishembedded/brain/commit/c3e369bbdf4c27581354f0eceddc685782b8315a))

- *(s3tokenizer)* S3Tokenizer v2 - FSQ speech tokenizer, exact token-id parity ([6be2a2d4](https://github.com/swedishembedded/brain/commit/6be2a2d4e06a099cb72512f32e685d41c5e86d3b))

- *(cosyvoice)* CosyVoice 2 speech-token LM - Qwen2.5-0.5B on qwen3, forward parity ([f8b19b5f](https://github.com/swedishembedded/brain/commit/f8b19b5f29af1ace941fe33ccbfa112a3d3b79a4))

- *(cosyvoice)* Flow decoder CosyVoice 2 - UpsampleConformerEncoder + UNet CFM, real-weight parity ([3fa5471e](https://github.com/swedishembedded/brain/commit/3fa5471e43826b015e9c320b6863ea4dd8064ed4))

- *(cosyvoice)* HiFT vocoder - conv trunk + NSF source + ISTFT, real-weight parity ([b8a8cfcd](https://github.com/swedishembedded/brain/commit/b8a8cfcdaeddcb4b5c403bce45da05062595fd7b))

- *(cosyvoice)* Non-streaming end-to-end pipeline - text + reference clip to real WAV ([5139936e](https://github.com/swedishembedded/brain/commit/5139936e9e6ac13efef7567ca0918f53a56a3624))

- *(cosyvoice)* Add CosyVoice 3 speech-token LM (CosyVoice3LM) ([c9748bdb](https://github.com/swedishembedded/brain/commit/c9748bdb4950c2bba1d237a77316c871cc2a2cd1))

- *(cosyvoice)* Add CosyVoice 3 DiT flow decoder (CausalMaskedDiffWithDiT) ([279aafc2](https://github.com/swedishembedded/brain/commit/279aafc2e15f364f7d471cf23fad289b541bf59d))

- *(cosyvoice)* Add CosyVoice 3 causal HiFT vocoder (CausalHiFTGenerator) ([aa0332ad](https://github.com/swedishembedded/brain/commit/aa0332ad1ecd3723b126c87fa6a4a5ba9124154d))

- *(cosyvoice)* Add gradient-checked LM training (gradcheck, LoRA, overfit) ([5ad11b07](https://github.com/swedishembedded/brain/commit/5ad11b0790434fde2626534d7bcb747d129a648b))

- *(cosyvoice)* Wire pipeline::generate into the serving contract (M11) ([90d91df7](https://github.com/swedishembedded/brain/commit/90d91df742714cbecf44003a2e5020f3db675988))

- *(cosyvoice)* Profile pipeline::generate per kernel-kind (M13 step 1) ([97b66902](https://github.com/swedishembedded/brain/commit/97b66902c886bbb4cb1cabf8cfa38e1603f8bed2))

- *(npu)* Export CosyVoice 2's HiFT vocoder and LM backbone to ONNX ([a8bc0fd5](https://github.com/swedishembedded/brain/commit/a8bc0fd56c32bd00b320c8ee9169fb9ad5ed0755))

- *(model)* One canonical Ops kernel list, and a real per-dtype GEMM probe ([765e586a](https://github.com/swedishembedded/brain/commit/765e586a4191fb95eb60341eedf53a31cdfe8ba1))

- *(model)* The capability probe reports the operating point, not the idle clock ([1ed2d17b](https://github.com/swedishembedded/brain/commit/1ed2d17b5b97039411326abc083e8ce287f35853))


### Miscellaneous

- Stop tracking results/.gitkeep ([fe0ca465](https://github.com/swedishembedded/brain/commit/fe0ca4658e3e625fa806a18b3c5d7545338ebfbd))

- Drop stale and gitignored path citations from source ([0a060a5d](https://github.com/swedishembedded/brain/commit/0a060a5d6e93580138ca8f8cc4ffab53a847ccfc))

- *(ltxv)* Remove machine-specific "this host" language; resolve audio golden gap ([1ad7a69f](https://github.com/swedishembedded/brain/commit/1ad7a69f9ab064121316a30788733bbe383cbb13))

- *(perf)* Stop committing brain-perf baselines to git, repo-wide ([71b41140](https://github.com/swedishembedded/brain/commit/71b411402d17676fcd733d55919d4340115be0ff))

- *(cli)* Derive Debug on quantize_cli's Args ([c5e7d005](https://github.com/swedishembedded/brain/commit/c5e7d00590e313789a6fde331bf2fc45575f2493))

- Untrack a qwen35 perf baseline that slipped back in via a rebase ([02652179](https://github.com/swedishembedded/brain/commit/026521796b33776b195a7714bc9cc0e2950735a0))

- *(cosyvoice)* Clean up clippy warnings in torch_rng/hift_import/hift_parity ([19c76ca8](https://github.com/swedishembedded/brain/commit/19c76ca8cda7c63a5170ac13c745c439f1363cad))


### Other

- Migrate mixer int8 dispatch onto model::ops::{Ops,Act,Weight} ([185f902a](https://github.com/swedishembedded/brain/commit/185f902a1e3794eb4a1d69e8099d89118e7c0cca))

- Hoist GDN/GQA mixer orchestration from qwen35/qwen35moe ([de6a113b](https://github.com/swedishembedded/brain/commit/de6a113b46bd5830578fe1a9475e9a76ddd766a4))

- Fix mmap streaming reader rejecting F8_E4M3 tensors ([285aa58f](https://github.com/swedishembedded/brain/commit/285aa58f783f5c5936a2fe6fa1ee5d14e835f6cb))

- Real-weight streaming parity against Qwen/Qwen3.8-27B-FP8 (M10) ([d740e3ed](https://github.com/swedishembedded/brain/commit/d740e3edf392fde3d04e55e0cff70056bdc3da39))

- M13 performance pass - profile-first, FP8 GEMM ruled out ([0e39417d](https://github.com/swedishembedded/brain/commit/0e39417d724cd6bd083919b0a3ca273cff368402))

- M14 int8 (DP4A) weight tier, real-weight sanity verified ([258f43b7](https://github.com/swedishembedded/brain/commit/258f43b7f2d28c3623df2c4e50c221bc030d6c9b))

- ParamSpec min/max/step + brain-py cancellable subscribe ([efc5420d](https://github.com/swedishembedded/brain/commit/efc5420d304818d28e8dd9a3d0f1c49258dfe359))

- Raise on a dead peer mid-Subscribe instead of returning empty ([e809bf40](https://github.com/swedishembedded/brain/commit/e809bf40f625b085d70b2507ef3b86cda1030d46))

- M15 sliding-window streaming forward over all 64 real layers ([d80295e0](https://github.com/swedishembedded/brain/commit/d80295e071f09efdc9b6757f6c2b1c4b46a7e7cc))

- Drop milestone-number references from stream.rs doc comments ([1026e2b5](https://github.com/swedishembedded/brain/commit/1026e2b581b4f49a389b73954ad143e273f058e9))

- Finish documenting BrainError.name's transport-vs-action contract ([1835932a](https://github.com/swedishembedded/brain/commit/1835932aba7bd7262c08997d102e0f25b663e9eb))

- Add the missing D-Bus example, fix AGENTS.md's stale status ([6000b9b9](https://github.com/swedishembedded/brain/commit/6000b9b94e50d850c13008bda44d0afc18611ee6))

- The serving contract - capability, residency, genuine batching ([d67887b5](https://github.com/swedishembedded/brain/commit/d67887b5d30a0d1c9b0242aebf0a7812baf204c2))

- The serving contract - capability, residency, D-Bus example ([29383bda](https://github.com/swedishembedded/brain/commit/29383bdaaa5f351bef9168287204cd45a6139db3))

- The serving contract - its own sampler loop, residency, D-Bus ([5436720c](https://github.com/swedishembedded/brain/commit/5436720c8e3145770735024a3638d35eca3bf046))

- A text2image sampling pipeline + the serving contract ([c85f05ab](https://github.com/swedishembedded/brain/commit/c85f05abc9be5e30e556d1adabfa269c77644a39))

- Identity-conditioned FLUX.1 - the image path + the serving contract ([99ae2e4b](https://github.com/swedishembedded/brain/commit/99ae2e4b4f2be306e7b0958eae9ee09f114d80e4))

- Add MmapSafetensors::tensor_f32_range for bounded row reads ([7ac89ebe](https://github.com/swedishembedded/brain/commit/7ac89ebe5c5cd3a50d023dd77c16a56226d06451))

- M16 real end-to-end streaming generation ([9fb68686](https://github.com/swedishembedded/brain/commit/9fb686865cd66eaff8b8f44fc561875b968daa7e))

- ResidentModels - list what's warm without parsing StatsSnapshot ([44c0dedd](https://github.com/swedishembedded/brain/commit/44c0dedd2f76a191f829f2df53117deb040acf9a))

- Memoize E4M3 byte decode, ~90% of real FP8 tensor import cost ([068d2ff5](https://github.com/swedishembedded/brain/commit/068d2ff53e4f6970e36d72b832405d0951bb5ef7))

- M17 phase 1 - real-weight MTP import ([a7eb4d41](https://github.com/swedishembedded/brain/commit/a7eb4d41ac3cc88870b1097131990bf3291ab72c))

- M17 phase 2-4 - MTP-accelerated greedy streaming decode ([86b271c2](https://github.com/swedishembedded/brain/commit/86b271c2ff1607dac712e2f657eabcabfb7c15c0))

- M18 streaming LoRA fine-tuning through the real 27B checkpoint ([686be490](https://github.com/swedishembedded/brain/commit/686be4902da8b4a8f5ae7157568696230142accc))

- M19 part 1 - measure the streaming residency-window policy for real ([960b4234](https://github.com/swedishembedded/brain/commit/960b4234458bb12fcc053a84def5d1288bc85d55))

- Parallelize the FP8 (E4M3) byte decode across cores ([cda3003f](https://github.com/swedishembedded/brain/commit/cda3003fc7b43631d7c5867c339be6c97c397331))

- M19 part 2 - wire the streaming decode path into crates/perf ([5545f83b](https://github.com/swedishembedded/brain/commit/5545f83bb35c0ba14060c2b660634baecdc02076))

- Sampling must not crash or silently misfire on a NaN logit ([07214869](https://github.com/swedishembedded/brain/commit/0721486922b25df51a67c96ef6238fc9724d0905))

- One process-wide --limit-vram-total/--limit-ram-total ceiling ([0b25aaeb](https://github.com/swedishembedded/brain/commit/0b25aaebe1ce86636a3fd821cc4e2af54af53a2f))

- Flash attention for self-attention (7.6x on attn1, 1080p now possible) ([0c1eb10c](https://github.com/swedishembedded/brain/commit/0c1eb10c65c9b13ef90dae2062aa09e5af72b86b))

- The block-weight cache stops dying with the generation ([ef9a0151](https://github.com/swedishembedded/brain/commit/ef9a015150fd3800739a88fe8d98e2593233a15a))

- The adaLN-single host stage, and a correction to Phase 12's attribution ([1ee64780](https://github.com/swedishembedded/brain/commit/1ee64780e997bbda4296c9446e9052320fe69baa))

- The two CFG branches run on two cards, bit-identically ([83850b15](https://github.com/swedishembedded/brain/commit/83850b15a759b2d55e238797ca14cce6ad55ba24))

- A device pool, and LTX batches stop running one at a time ([c6c23eaa](https://github.com/swedishembedded/brain/commit/c6c23eaa0a8f5736dd8f59f2f7031feaf1111a19))

- Ledger Phase 15 - the second card starts existing ([99a5e139](https://github.com/swedishembedded/brain/commit/99a5e139aa4227042e233f2367687d5fef1cf66e))

- Overlapping-tile geometry for 3D causal video autoencoders ([b3bb10d5](https://github.com/swedishembedded/brain/commit/b3bb10d5b761daa7b2176e4f746251e656c5871b))

- The VAE decoder stops being the 1080p ceiling ([676c223a](https://github.com/swedishembedded/brain/commit/676c223a617e97e4263191b9ca07cfbb0ec7f649))

- The x0 conversion runs at the token's own timestep ([739ad8fa](https://github.com/swedishembedded/brain/commit/739ad8fa12ede69832e32b26ff3e6bed32885c36))

- Adaln_row, a per-token adaLN table row extract + block-table add ([92f55824](https://github.com/swedishembedded/brain/commit/92f5582428f7751cfff88cb154e98159e07d6ee2))

- The DiT forward stops being a PCIe benchmark (2.22x at 720p) ([aba8dfbf](https://github.com/swedishembedded/brain/commit/aba8dfbf25bc8e8cc989f5f1f3870c42312396b6))

- The resident window must leave room for the rest of the generation ([2cfb0383](https://github.com/swedishembedded/brain/commit/2cfb038380b9e766963fd468de98c4ef4e2ac4c9))

- Deferred reclaim freed buffers live descriptor sets still named ([5caee512](https://github.com/swedishembedded/brain/commit/5caee5120390e86fc5c7363e8c4e57ad18121a46))

- Stop compiling every kernel with runtime checks wgpu turns off ([ed11bf2b](https://github.com/swedishembedded/brain/commit/ed11bf2bdf22af0f20f553cb0b3f31924c89bd9e))

- *(model)* Fix needless_range_loop in the quantize-invariance fixture ([a9703a8f](https://github.com/swedishembedded/brain/commit/a9703a8f0a4245648ad29af60e6e95022f9fa558))

- Split a generation's denoise half from its decode half, for bisecting ([ef2af76f](https://github.com/swedishembedded/brain/commit/ef2af76f72a2bb821cbd8962e7be5bd710a37956))

- Uploads are work too, so flush has to submit them ([4c80c054](https://github.com/swedishembedded/brain/commit/4c80c0542ce8e3ad19ae0db0e1a6c3cdb71e179b))

- The clip stops falling apart before it ends (1080p) ([d679b3ea](https://github.com/swedishembedded/brain/commit/d679b3eae5aa92c3acd68be49245ef2570a9749a))

- The 49-state text projection is not a bare Linear ([7d86e4a7](https://github.com/swedishembedded/brain/commit/7d86e4a79bf70012afdc8c12e7df4685a2c299ee))

- The text context stops being a constant ([7af34a0e](https://github.com/swedishembedded/brain/commit/7af34a0e6c33b5b0d9cd9c8f5a6d1cebfb573128))

- A video decode that keeps its bytes, and one that reports its fps ([904da75c](https://github.com/swedishembedded/brain/commit/904da75c05b2196ebad3c37f5cc487abfb614517))

- A finished clip can go back through the upscaler that made it ([97b49bd9](https://github.com/swedishembedded/brain/commit/97b49bd9255c547672414fde48f171b4e7463f53))

- The per-segment GenOpts stops disagreeing with the stage it accompanies ([5059b3d8](https://github.com/swedishembedded/brain/commit/5059b3d89e34fa46b82228aac10fa952e14e1900))

- A clip stops being one window long ([b25b4ad6](https://github.com/swedishembedded/brain/commit/b25b4ad6ca5440fe5646bd0dcea4c97b64ae5b9a))

- --frames stops having a ceiling ([03fbd10f](https://github.com/swedishembedded/brain/commit/03fbd10f3c68e6afe0d8ca9b91fb8076539831e3))

- The seam gate stops rewarding a stall that undercuts a near-perfect seam ([1e3d8578](https://github.com/swedishembedded/brain/commit/1e3d8578817487abe8e71e6bb623c56924bc3b85))

- A half-conditioned token stops stepping from an unblended estimate ([e4b1d8d6](https://github.com/swedishembedded/brain/commit/e4b1d8d6af972749994c4aae91af3e877d0e6891))

- A window loop stops holding the card it is about to need ([4c053a83](https://github.com/swedishembedded/brain/commit/4c053a83208c066ad69e922cbea640f2ea69b322))

- A clip stops being one scene long ([f4c0c33d](https://github.com/swedishembedded/brain/commit/f4c0c33dcd47dfad14816481a2fa9abe4af4228c))

- --scene puts several scenes in one file ([bc104415](https://github.com/swedishembedded/brain/commit/bc104415e88be9602586bf23216661d23cab532c))

- The workspace license stops contradicting every file in it ([b6e73b8f](https://github.com/swedishembedded/brain/commit/b6e73b8f47b161caa11ff50b9a0205f933c8a243))

- Name Swedish Embedded AB as the author of brain ([e08d8fdf](https://github.com/swedishembedded/brain/commit/e08d8fdf2ac37ce8253be4b13dbba966fca7cc69))

- Fill in the copyright owner ([2ae27243](https://github.com/swedishembedded/brain/commit/2ae272437de3f1ba9eeaad4b166a995b5d9023c6))

- Say who builds brain, and what we can be hired for ([23533e39](https://github.com/swedishembedded/brain/commit/23533e390357bf2bb58c65bad50ac46616f30886))

- Name what we do on the front page of the 14 core crates ([e5518780](https://github.com/swedishembedded/brain/commit/e551878036547094f5926e0963f89eb612ac919d))

- Each integration demo says who can build this for you ([6ea0bdd6](https://github.com/swedishembedded/brain/commit/6ea0bdd6f196d64d427ba289b037dc63587567ea))

- A killed fetch resumes instead of re-downloading ([d6a67a03](https://github.com/swedishembedded/brain/commit/d6a67a0324ed3236338dfa8e6fce3ac1d1a57d7a))

- GLM is discoverable without a checkpoint on the box ([76d0bfb8](https://github.com/swedishembedded/brain/commit/76d0bfb837006cbcb5a5f69c627a6ca09fae1890))

- The 16 newest dumpers record which checkpoint they dumped from ([fbce80c2](https://github.com/swedishembedded/brain/commit/fbce80c2fb6e552f66110257a6ee96f2a754bffb))

- Record what Phase 0/1 actually found, not what it predicted ([173884fe](https://github.com/swedishembedded/brain/commit/173884fe477a4d675acd539009ce3fc281743964))

- A Cargo.toml edited mid-build misdiagnoses itself ([24b2e3fc](https://github.com/swedishembedded/brain/commit/24b2e3fc66de83f39e777d3713b5a904f46a24a5))

- The link check covers .agents/ too, and finds one there ([8b00e3c2](https://github.com/swedishembedded/brain/commit/8b00e3c26441604a77f67c92871637daed265330))

- Clear the 23-warning backlog the gate's zero baseline was already failing ([09f96330](https://github.com/swedishembedded/brain/commit/09f96330aa3ece1c83a6bd4849f78be98cd316e2))

- The vision tower moved to wgpu; the budget never heard about it ([894ebd1e](https://github.com/swedishembedded/brain/commit/894ebd1e5b838f685ad4d2708b4cff3cfb0d704c))

- The composite rebuilt all 8.8 B params on every forward ([ad25543a](https://github.com/swedishembedded/brain/commit/ad25543a3fcc9686e6ecf760815c5bffafca1831))

- Variant='cosyvoice3' was discoverable but always refused ([f6ea9255](https://github.com/swedishembedded/brain/commit/f6ea9255e7ce01b9dad3954c702dbd605a438d32))

- The only code that could load a real checkpoint was in a test ([96c69992](https://github.com/swedishembedded/brain/commit/96c6999216053092f0401eb6941287d308e0140e))

- The composite could compute a loss but not emit a token ([4d9c59b6](https://github.com/swedishembedded/brain/commit/4d9c59b6d363279749189c3892d782b8877393a4))

- Close check_unet -- the transformer half was not differentiable at all ([7537e04c](https://github.com/swedishembedded/brain/commit/7537e04ca6d9a226f6b68253ff338ffcea6e2020))

- Say what the serving surface is actually blocked on ([ad95952e](https://github.com/swedishembedded/brain/commit/ad95952edac22ac53c3cdc608e848933e1575b6f))

- Make it fit -- int8 experts and one shared activation set ([06cba7b9](https://github.com/swedishembedded/brain/commit/06cba7b9899766721d77685b55a2411ab37a96e2))

- The serving contract -- caption over caps, residency, D-Bus ([434b399d](https://github.com/swedishembedded/brain/commit/434b399d58ffb906fc10ca779b35873b770a389d))

- KV-cached decode -- O(pos) per token instead of a full recompute ([8c1aeff9](https://github.com/swedishembedded/brain/commit/8c1aeff9068a455299ac35e660f622b30ee9a031))

- Real run_batch on the axis this architecture actually has ([9023bfb8](https://github.com/swedishembedded/brain/commit/9023bfb8ee8efd682daa40a357956cdef791a0e4))

- GPU placement, with the plumbing actually checked ([cf2d03ae](https://github.com/swedishembedded/brain/commit/cf2d03ae0598c60adc0a30690b86b1f9dab5ccef))

- Make the region-head port discoverable instead of guessing at it ([e4a0faa3](https://github.com/swedishembedded/brain/commit/e4a0faa3ca853b628a063a6397f8beab596cee60))

- The KV decode loop round-tripped the hidden state per layer ([fb211f97](https://github.com/swedishembedded/brain/commit/fb211f97d610b4e57e43041174b8386505b03c9f))

- An upscaled clip stops being several clips ([cb357b04](https://github.com/swedishembedded/brain/commit/cb357b04ad4daaa98b670a946aa23365374272ea))

- A clip stops being anchored only at its ends ([2dfc45b4](https://github.com/swedishembedded/brain/commit/2dfc45b418bca6a304af3c8c729668010507b4d2))

- --mid-frame anchors the middle of a clip ([55b3280e](https://github.com/swedishembedded/brain/commit/55b3280e19a4e1e8526aac14bec618583e75b925))

- One device-token -> Gpu mapping, and it understands an indexed card ([8174865f](https://github.com/swedishembedded/brain/commit/8174865fda0421f40545df86c4d0591179099bc1))

- The DiT stops re-uploading its weights every denoise step ([015d3500](https://github.com/swedishembedded/brain/commit/015d3500f920b451271b3748eae043bc2d2789a8))

- The DiT and vocoder stop being pinned to the CPU backend ([387667a4](https://github.com/swedishembedded/brain/commit/387667a4d4ab4b2d0ee45be5c38c90a1d2b8216e))

- A multi-minute generate stops reporting nothing ([978c6d65](https://github.com/swedishembedded/brain/commit/978c6d65dda7a1ae14cf14d3ce4c66777c4c28ad))

- The e2e gate compiles again, and narrates itself ([3d3af921](https://github.com/swedishembedded/brain/commit/3d3af9215abcf2b3be01bebb3044f2da68d2ccd3))

- Vae, diamond, s3dit, wan, gemma4, ltxv: use the shared device-token mapping

Replaces all 17 private copies of the `Some("cpu") => new_cpu / Some("gpu") =>
new_wgpu / _ => new` match with `Gpu::open`. Seven were inline in a public
constructor; ten were `open_device`/`new_gpu` wrappers, now deleted rather
than left forwarding - a local alias for a shared function is how the private
copy comes back at the next edit.

Five of those wrappers were `pub`, so 14 further call sites across wan,
gemma4 and ltxv (including six integration tests) now call `Gpu::open`
directly.

Behavioural change, stated plainly: every one of these 17 sites previously
dropped a `gpu<i>` token through the `_` arm into the ambient selection, so a
caller asking for the second card silently got the first. They now honour it,
and an out-of-range index panics naming the token instead of landing on card
0. That is the point of the shared helper, but `Some("gpu1")` reaching any of
these constructors changes outcome.

Two observability labels in ltxv/src/dit.rs named the deleted function
(`stage_time("... open_device ...")`, `stage = "open_device"`) and now
describe the operation instead. Nothing in crates/perf or scripts/ parses
them - checked, because BRAIN_PROFILE stage totals ARE parsed by perf gates
elsewhere. Historical measurement tables in .agents/roadmap/ltxv.md keep the
old label: they record what was measured then. ([30907a40](https://github.com/swedishembedded/brain/commit/30907a40a033e1c757ac0942c883b625f0ee03e2))

- The reference is a pip release, not an unmerged PR ([215acdcf](https://github.com/swedishembedded/brain/commit/215acdcf4b31d7484c7a3b7ba4c39d737b2787ca))

- Parity gates stop being CPU-only ([6f0f4f4a](https://github.com/swedishembedded/brain/commit/6f0f4f4a03feb0636139126b6345a5142b525dda))

- Clear the 22 pre-existing clippy warnings ([b03471fa](https://github.com/swedishembedded/brain/commit/b03471fa388e9405827e520204ac68366d3509df))

- KV-cache the depth decoder, 3.75x for identical output ([f111e6cb](https://github.com/swedishembedded/brain/commit/f111e6cb373a1677e9224797aa8c7b6721629b76))

- The f16/bf16 storage tier becomes requestable, and is proven to dispatch ([1d707a09](https://github.com/swedishembedded/brain/commit/1d707a0975706e7a8b891d790aec1b9574d05cf5))

- The first real song, and the DiT stops using the naive matmul ([594a3db3](https://github.com/swedishembedded/brain/commit/594a3db3cb164308e4028adf6bea763cfd7e0bb7))

- The vocoder stops allocating 4x the device memory it needs ([07063173](https://github.com/swedishembedded/brain/commit/070631735d5bdb0aa441a3a32a58ede53d580fe4))

- The training paths stop allocating 4x their device memory too ([0df12c05](https://github.com/swedishembedded/brain/commit/0df12c05453d8d1099e04fcb7dddfd068890cff4))

- A multi-chunk song survives the vocoder, on the second card ([7525154b](https://github.com/swedishembedded/brain/commit/7525154b55cddfb754a3d260a9674ff885e5f081))

- Retract a wrong claim about wgpu's memory behaviour ([52b76be6](https://github.com/swedishembedded/brain/commit/52b76be6bb128cffde4559c3ba44b1a02a7a0d3d))

- A per-kernel-kind bench, and it kills the stated hypothesis ([3c5f30d7](https://github.com/swedishembedded/brain/commit/3c5f30d778854b487d77af7fdadb6f69ac02b677))

- The DiT stops copying 3.26 GB per forward, and gets flash attention ([57d48265](https://github.com/swedishembedded/brain/commit/57d482653dafd0b03349b2644fbdb8e3b8b5d476))

- The two CFG branches run on two cards, 1.69x ([b05faa55](https://github.com/swedishembedded/brain/commit/b05faa55083af4b9f14de8501f108b06ef5ba1a1))

- The depth decoder moves to the device, 4.28x ([83d7e28d](https://github.com/swedishembedded/brain/commit/83d7e28df93c12bbbef3af62497ead26c0f8214d))

- Honest VRAM, warm weights, and a run_batch that says why it is serial ([e2157e58](https://github.com/swedishembedded/brain/commit/e2157e58e09aae50c51af07ab02cb7cbb0b2574f))

- The DiT weights upload once per generation, not once per chunk ([9c84952b](https://github.com/swedishembedded/brain/commit/9c84952b2b2db8dfd23d603523c078d2d3497df4))

- Qwen3, minimaxmusic3: a device LM head, and the AR branches step concurrently

Two changes to the AR stage, which after the DiT work is the largest cost in a
generation - and where both P40s read 0% utilisation while 30 GB of weights sat
resident, because the stage was host-bound and serial.

1. THE LM HEAD MOVES TO THE DEVICE. qwen3's incremental decode returns the
   hidden state and leaves the head to the caller - its own doc says so - so
   EVERY decoder in the workspace applies a multi-GB head with
   hostmath::matvec_par. For minimaxmusic3 that is a [200000, 4096] fp32 table
   applied twice per 25 Hz frame, streaming 6.56 GB of host memory per frame,
   plus a 3.28 GB duplicate held in host RAM by read_weight.

   Qwen::decode_logits applies it on device, vocab-tiled, into a lazy [vocab]
   buffer (800 KB) - the batched [n, vocab] slab is deliberately not
   resurrected on a decode build.

   Tiling is mandatory, not an optimisation: wgpu clamps
   max_storage_buffer_binding_size to i32::MAX on every card, and this box
   measures max_buffer_size 4094 MiB against a 2047 MiB binding limit, so a
   3.28 GB table cannot be bound whole even on a 24 GB P40.

   The head needs its OWN tiling rather than vocab_tiles(). The GEMV writes
   out[row*n + col] with col local to the dispatch, so the OUTPUT binding must
   start at v0, and wgpu requires that offset be 256-byte aligned;
   tiles_with_budget does not guarantee it (P40: stride 65503 rows, odd).
   align_head_tiles keeps that fix local to qwen3 rather than perturbing the
   shared vocab_tiles* that lfm2 and t5encoder also use for embeddings.

   Not matmul_tile, which forward_steps uses: one thread per output element
   with adjacent threads reading weight rows d_model apart is uncoalesced, on
   a dispatch that is pure traffic over 3.28 GB. The GEMV arm is offered only
   when self.coop, because backend-cpu's FastIdx routes matmul/matmul_reg* to
   native AVX2 but does not know matmul_gemv - otherwise it is a GPU win paid
   for on CPU.

   The head stays fp32: Dtype::I8 only ever populates self.weights with the 7
   per-layer linears, so tok.weight/lm_head.weight remain f32 in the
   ParamStore.

2. THE TWO CFG BRANCHES STEP CONCURRENTLY, one per card. ar_branch_devices
   already placed them on different cards, and pipeline.rs then stepped them
   one after the other, so each card idled while the other worked.

   qwen3::Qwen is Send but genuinely !Sync - probed, not assumed: Cell<f32>,
   Cell<bool>, Cell<Option<..>> and RefCell<Option<optim::*>> reached through
   optim::Optim. So ltxv::dispatch_cfg_pair's shape - share &D across
   thread::scope with D: Sync as the compiler-checked safety argument - is NOT
   available here, and was not forced.

   ArBranches instead holds the unconditional branch by unique borrow
   (&mut Qwen, Send) and reborrows it into the spawned thread for one step. A
   &mut cannot alias, so the compiler still proves the moved branch is
   untouchable from the orchestrating thread while the scope is open. Same
   guarantee, different route. One private generic `pair` serves both the
   prefill and the per-frame step, so there is no second copy.

   Gated on the existing BRAIN_MINIMAXMUSIC3_CFG_PARALLEL knob rather than a
   second one; one card, the CPU backend, or a pinned device keeps the
   byte-for-byte sequential path with no thread spawned.

   embed_row still runs on lm_cond only and is hoisted out of the loop, the
   feedback embedding is still sampled once and broadcast, and the host fold,
   RNG order and progress call after the join are unchanged. The depth decoder
   is untouched - it already runs its CFG pair batched at b=2 in one step.

Bit-identical, gated by the_concurrent_ar_branches_are_bit_identical_to_the_
sequential_ones over a whole 4-frame AR loop with differing prompts per branch,
assert_eq! on the f32 vectors, at tiny dims with no checkpoint.

Also fixes a test that referenced model::block::vocab_tiles_with_budget_for_test,
which does not exist. align_head_tiles is a pure function of a base tiling, so
the test now builds one to the same rule locally rather than widening
model::block's public API for a single caller. ([dd9250f0](https://github.com/swedishembedded/brain/commit/dd9250f078e8c1621645491da66d3639cac23363))

- Kernels, gpu-core: a register-accumulator decode GEMV, upgraded per shape

`matmul_gemv` holds its accumulators in workgroup memory, which costs a GPU
twice: `partial` is sized for the worst case (m = 32), so every workgroup
reserves 8 KB of shared memory at every m, and the inner loop is a
read-modify-write per (k, m), so each accumulator carries a dependency chain
through shared-memory latency. On a GP102 that is ~37.5% occupancy and 36% of
the card's measured memory roof.

`matmul_gemv_reg` is the same arithmetic in registers. Both limiters come off
one `kernels::template` knob, `MREG`, which sizes the accumulator array and
`partial` together. It is a second file rather than a template variant of the
first because a function-local accumulator array is a different body AND the
CPU JIT rejects one outright, which is the constraint that makes the workgroup
accumulators mandatory over there; each header points at the other.

Selection goes in `gpu_core::upgrade` rather than a new `gemm_variant` slot -
same Params, same bindings, same n*64 thread count, so it is a drop-in, and
every decode path in the tree inherits it with no edits. Rows may now be
shape-specialised: a knob plus a bucket ladder, the bucket chosen from the
caller's own params. The ladder is measured, not assumed - a single MREG=32
build is a 0.44x REGRESSION at m=1, while power-of-two buckets win >= 1.7x
across m in 1..=32.

Bit-identical, and gated as such: same k-stride, own accumulator per output,
same 64-partial fold in the same order, so nothing reassociates and the gate
asserts raw bits rather than a tolerance. Mutation-verified - reversing the
fold order, pure reassociation with no arithmetic error, fails the bit
assertion while passing an independent f64 oracle.

Two seams cleaned up on the way: `Gpu::physical_kernel_name` becomes
`physical_kernel_names`, since one caller slot now maps to several physical
pipelines and taking the first silently zeroed profile rows; and
`kernelmeta.py::cpu` now derives `@cpu` from both structural facts the JIT
actually checks (barrier count and work-group-kernel local arrays), with
`wgsl-cpu`'s skip list matching, so the declaration and the compiler cannot
drift. No existing kernel's `@cpu` cell changes.

Measured, mm3_bench depth 8 3 on an idle P40, A/B via BRAIN_NO_KERNEL_UPGRADE:

  matmul_gemv   21.9 ms -> 8.3 ms   2.64x   36.0% -> 96.6% of the memory roof
  whole pass    24.01 ms -> 10.53 ms   2.28x

The top row now sits essentially at its roof, so the remaining lever there is a
narrower weight tier, not a kernel. ([0135ee5d](https://github.com/swedishembedded/brain/commit/0135ee5df1f7bef6a754705d16d1e61033c0f919))

- Gate the vocoder on relative L2, not cosine alone ([8835c941](https://github.com/swedishembedded/brain/commit/8835c9410a01c2ec602d4031a742c3f9b10f2787))

- Audio, kernels: lower the 1-D convolutions to GEMM

The DAV vocoder ran its 1-D convolutions on one-thread-per-output kernels:
`conv1d` at 2.2% of this card's compute roof, `convtr1d` at 0.4%, together 99%
of the stage. Unlike the DiT's two misses there was no fast sibling to
register - `conv1d`/`convtr1d` (+`_dx`/`_dw`) are the only 1-D conv kernels in
the tree and `audio::conv` had no selector at all, so all ~12 crates that
convolve in 1-D ran the naive kernel. The 2-D side already had the pattern to
port (`vae::blocks::conv_s`).

The algebra needs no transposes and no weight permutes, which is what makes
this cheap: the native `[Cout, Cin/G, K]` weight IS `[Cout, Cin*K]` row-major,
so `matmul_reg3` takes it directly; `K == 1` skips im2col entirely via the NN
form and covers `dec_in_proj` plus twelve residual convs; and the transposed
conv's TN form takes the native NCL input and native weight as-is, doing `L`
rows rather than `Lo` for the same FLOP.

Two new kernels (`im2col1d_at`, `col2im1d_bias`), both barrier-free, and a new
additive selector in `audio::conv` gated through `backend_api::select`.
`ConvKernels`, both backward builders and `train.rs` are untouched, so the
trainer and gradcheck stay bit-identical. Chunking is mandatory rather than an
optimisation: the unchunked stage-3 `col` is 1.9 GB against a 2047 MiB binding
limit.

Deduplicated rather than copied: `gpu_core::lower` now owns the budget and
chunk arithmetic, replacing two byte-identical copies in `vae::blocks` and
`blocks3d`, and the existing `nlc_bias_nchw` is reused instead of a second
epilogue kernel.

Thresholds are measured per kernel pair, not inherited: `GEMM_CONV1D_MIN_COUT`
is 16 and the transposed one is 4, where the 2-D lowering uses 32. Its baseline
is a fast kernel at ~700 GFLOP/s; ours is a naive one at 2.2% of roof, and a
weaker baseline crosses earlier. Copying 32 would have cost 1.9-6.0x across
16 <= Cout < 32.

Measured, mm3_bench vocoder 689, one P40, best-of-3, A/B in the same binary via
BRAIN_CONV1D_GEMM=0:

  whole pass    16642 ms -> 1424 ms   11.7x    0.48x -> 5.62x realtime
  device time   16060 ms ->  797 ms   20.2x
  convtr1d       7668 ms ->   96 ms   80.9x

No kernel in the pass is below its roof floor any more. Parity holds at cosine
1.000000000, rel_l2 1.676e-6 against a 1e-4 ceiling.

Mutation-verified. One mutation is worth recording: dropping the chunk offset
in `im2col1d_at` was NOT caught at first, because every shape in the suite fit
a single chunk. The suite gained a three-chunk case, and the fix is what the
gate now rests on - a chunked path whose tests never cross a chunk boundary is
not tested. ([3ac9d3ea](https://github.com/swedishembedded/brain/commit/3ac9d3eacaa7176f2d4961b4bc38a144dfa7ba2d))

- Model, minimaxmusic3: a real unregistered-slot sentinel, and the training GEMM

Two defects, one file in common.

`model::block::UNREGISTERED` did not exist, so models filled unused `KernelIds`
slots with `0` - a real, registered kernel in every PIPELINES list in this
workspace. A builder reading such a slot runs that kernel against another
kernel's bindings and uniform. On a GPU backend the binding check makes it a
panic; on `backend-cpu`, the backend every unit test in these crates uses,
there is no buffer-count or uniform-size check at dispatch, so it is an
out-of-bounds read that no test on that backend can see.

Thirteen sites, not the two originally identified. Seven held a literal `0`;
six more wore a better disguise - a live index for a DIFFERENT kernel,
commented as a harmless placeholder, invariably a backward-pass slot holding a
forward-pass kernel in a crate that never runs backward. Each slot was
confirmed genuinely undispatched by reading which builder consumes it.

The gate is dispatch-linked rather than static: it runs the real pass, reads
the device's own per-kernel counters, and fails only if an unused slot names a
kernel that pass actually dispatched - so it reports whether getting it wrong
would have mattered, not merely that a field changed. It also asserts the pass
dispatched something, so it cannot pass vacuously. `KernelIds::slots()` lives
next to the struct so adding a field cannot leave the gate checking 15 of 16.

Separately, `dit_train.rs` registered only `matmul`/`matmul_dx`/`matmul_dw` and
dispatched the naive kernel at every site - the same defect the inference path
had. It now shares `dit.rs`'s `linear_step` (hoisted, not copied) and routes
the backward through `model::block::pick_gemm`. New kernels are appended to
PIPELINES so no existing index moves.

Read the training numbers with their caveat. Per-kernel the forward gains
111-342x and the backward 10-26x at real dims, but this changed nothing that
runs today: `Trainer::new` hardcoded `Gpu::new_cpu`, and `backend-cpu`
intercepts these kernels by identity and routes them to the same AVX2 GEMMs as
their `_reg` siblings, so on the only device this trainer could reach the two
tiers were the same code. `Trainer::new_on` is what makes the fast tier
reachable and therefore checkable; without it the fix would have been untested
dead code. `Trainer::new` is unchanged.

Split-K was measured and rejected in both families - it lost at every shape and
slice count, its tile grids here being 256-2048 workgroups on a 30-SM card, the
opposite of the starved-grid case it exists for. Both split-K kernels are also
GPU-only with no `backend-cpu` native path, which is the all-zero-gradient trap
this repo gates against.

Gradcheck green on both backends; lib tests 93 -> 96.

Drive-by, because staging the files surfaced it: a qwen35moe doc comment cited
an absolute machine path for the reference implementation and pointed at a
report that is not in the repo. Both replaced with the checkpoint-relative
filename and the env var that resolves it. ([a58ffb60](https://github.com/swedishembedded/brain/commit/a58ffb6022d6f0e5e55c230b12d0cc860e41cc5f))

- Do not print a profile banner for a run that never profiled ([1ee1d945](https://github.com/swedishembedded/brain/commit/1ee1d945cca8d9631ddb385929f58da4b0eff187))

- Ltxv, dit: one adaLN row per distinct timestep, not one per token

`ada_layer_norm_single` computed a row per token. Every row is a function of
that token's scalar timestep and nothing else, and `pipeline::denoise` builds
those timesteps two ways: `vec![sigma; t]` with no conditioning, and
`mask * sigma` with it, where the mask is 1.0 (generated), 0.0 (frozen - an
image anchor, a long-form window's carried context) or `1 - strength`. So the
distinct count is ONE for plain text-to-video and TWO once anything is frozen.
At the real 720p token count that is 3520 rows of which 1-2 are distinct and
the rest are recomputed copies.

Tuning was already exhausted: the `[3520,4096]x[36864,4096]^T` table GEMM ran
at 120 GFLOP/s against this host's own measured 127.8 GFLOP/s scalar-MAC
ceiling. The only lever left was fewer rows.

Deduplication is GENERIC, not a "uniform" fast path. A uniform special case
would drop every anchored and every long-form shape back to full cost, and
those are precisely the shapes a fallback exists to serve. Measured at 8
layers, T=3520: no dedup 10302 ms, two distinct 609 ms, one distinct 199 ms,
and `distinct == t` degrades continuously to exactly the old cost.

`adaln_row.wgsl` takes a fourth binding, the row map, and gathers
`tab[map[r]*NR*D + off]`. It also gained the `gpu_core::cost` row it never had
- it was UNCOVERED, so the harness could not have reported it as a defect.

Bit-identical, gated on raw bits rather than a tolerance: same table, same
arithmetic, only the row lookup moved. Mutation-verified on this change's real
failure mode, which is a wrong scatter rather than wrong arithmetic - rotating
the gather by one token PASSES the uniform case and fails only the
two-interleaved one, on both backends. A suite whose every case was uniform
would have shipped it.

Measured, 48 layers, T=3520, ctx 1024, one idle P40, best-of-4, warm-up
excluded:

  adaLN stage      10243.4 ms ->  222.0 ms   46x
  adaLN upload         519 MB ->  161 KB
  warm forward       47.91 s  -> 36.41 s     1.32x

A real CLI generation came in at 393.2 s against a 547.0 s reference, and the
11.25 s per-forward CLI delta matches the controlled bench delta of 11.50 s on
two independent harnesses. Only ~90 s of the 153.8 s whole-run difference is
attributable; the rest is not, and is recorded as unattributed rather than
claimed.

`ltxv_bench` also now prints the DEFECT block that the other five benches in
this tree already print. It was the only one that did not, which is why this
model's roof-floor defects had never appeared in its own harness output. ([b13cda56](https://github.com/swedishembedded/brain/commit/b13cda5617b08cbd976ea20436d3e5e29f43680b))

- Ltxv, cli: bound the text cache, and stop refusing a clip that fits one window

Three defects found while profiling, plus one cosmetic.

**The text-context cache had no bound.** No budget, no eviction, no prune -
`store()` wrote and returned. Measured: 86 entries at 33.5 MB each = 2.7 GB,
growing by one entry per distinct prompt forever, on a filesystem at 96%.

Eviction is least-recently-USED, and that choice is load-bearing rather than a
coin flip: this cache exists for the same prompt re-encoded on every iteration
of a change-something-else loop, and under oldest-written eviction the entry
being iterated on is exactly the one thrown away. Recency rides on mtime,
rewritten on every hit, because the default `relatime` does not update atime on
a read within the day. The most-recently-used entry is never evicted: a budget
below one entry is not honoured by deleting what the caller is about to read -
`BRAIN_LTXV_TEXT_CACHE=0` is the switch for caching nothing.

There is no existing disk-cache budget convention in this workspace to follow,
so the default reuses the only comparable number in the tree rather than
inventing one, and the env-var shape copies `longform::max_window_tokens_from_env`.

**`window_plan` refused a request that fits a single window.** The
`max_lat < context + 1` guard sat before the early return it should follow. At
1920x1088, `max_lat` is 6 and a 25-frame clip has `k_total = 3`, so the early
return emits `k_total + 1 <= max_lat` frames carrying zero context - that
window always fits on its own, and the guard protected nothing on that path.
Past the early return `k_total >= max_lat`, a continuation window genuinely
exists and the guard is genuinely required, so it moves rather than goes. The
refinement caller is unaffected: it passes `fitted_context`, which already caps
at `max_lat - 1`. `max_lat == 0` is still refused.

The regression test was watched failing with the exact reported error before
the fix, and it asserts up front that the grid really is too dense for a
context so it cannot silently stop testing anything.

**`--help` claimed a specific CFG-parallel multiplier.** It advertised 1.94x
where a fresh reading showed 1.46-1.48x. Rather than bake in another figure
that the adaLN work in the previous commit was about to move again, the claim
is now the mechanism plus how to measure it on your own box - matching this
repo's convention for perf claims elsewhere in the docs.

Also: `brain flops --help` warned about an unrecognised argument and exited 2
with usage on stderr. It now exits 0 with usage on stdout, from a single shared
const so the two paths cannot drift, and a genuinely unusable invocation still
fails.

Drive-by: two `text_cache` tests mutate process-wide env vars while `make test`
runs at 48 threads. They now share a module-local mutex, since adding a third
would have made an existing race worse. ([1004e4b7](https://github.com/swedishembedded/brain/commit/1004e4b704603b6dae705830982e5655a609aad3))

- Mimi, ecapatdnn: make three parity gates actually run, and gate on two metrics

All three skipped, and a skipped test reports as a pass here. So a served TTS
codec decoder and a speaker encoder had no executable numerical parity
coverage at all - the same state the MiniMax-Music3 gates were in before their
goldens existed. Both checkpoints were already on the box; only the goldens
were missing.

Qwen3-TTS is in no release of `transformers`, and the published checkpoints
carry no remote modelling code, so the only reference is the `qwen-tts` package
and the transformers version it pins. The dumper installs that pinned stack
into a private directory on `sys.path` for its own process, leaving the shared
environment untouched, and registers the two namespace packages bare so their
`__init__` bodies do not drag in a different tokenizer and an audio stack.
**Not one line of the reference is patched** - a golden that came from
modified reference code proves nothing about the port.

Reproducibility was verified rather than assumed: deleting the reference tree
and re-running from empty re-fetches everything and produces byte-identical
dumps. Both dumpers carry checkpoint provenance read from the checkpoints they
actually loaded, so a mismatched fixture becomes a named skip instead of a
wrong number.

Now measured, with zero skips under `BRAIN_REQUIRE_FIXTURES=1`:

  mimi decode      max-abs 5.721e-4, log-mel L1 7.691e-4
  mimi encode      code match 100.00% (400/400)
  ecapatdnn        cosine 1.0000000000, rel_l2 7.089e-7

The mimi decode figure is ~65x better than the comment that stood in its place,
which claimed 3.7e-2 and blamed fp accumulation order. That number was dumped
from a bf16 reference forward, so it recorded the REFERENCE's precision while
reading as a measurement of this port. Corrected here and in the README.

`ecapatdnn` asserted a cosine floor and nothing else. Cosine is scale
invariant, so a uniformly mis-scaled embedding scores a perfect 1.0 and passes.
It now asserts a relative-L2 ceiling too, set two orders of magnitude off the
measured clean value rather than fitted to one run.

Every gate mutation-verified, applied then reverted, and each mutation was
caught by a different metric: a uniform scale on the embedding by rel_l2 alone
with cosine printing 1.0000000000 unchanged; a uniform scale on the waveform by
log-mel L1 alone, having slipped UNDER the max-abs ceiling; and a scaled RVQ
input by code match, with the damage concentrated in the residual codebooks.
Each single-metric gate would have missed at least one of them.

All three added to the strict parity suites so they are certified rather than
merely present.

Pre-existing defects fixed on the way: all three mimi tests deleted the shared
fixture they memoize, which is a hard failure run serially and a race run in
parallel; the intermediate checkpoints were named per-pid, so every run leaked
one and a 646 MB stale file from the previous day was sitting on a 96%-full
disk; and three test docs named packages that do not exist.

campplus is deliberately left skipping: its gate is already well formed, it
needs ~8-9 GB of checkpoint and scratch environment, and its dumper is
all-or-nothing so the 28 MB piece it needs cannot be fetched alone. ([d1605046](https://github.com/swedishembedded/brain/commit/d16050463930effb8bc77161e835611771ee6068))

- Certify the mimi and ecapatdnn parity suites ([5e73953d](https://github.com/swedishembedded/brain/commit/5e73953da775902a2ca8d3fc9d5c17e94ccb3418))

- Crates, tools: no hardcoded performance numbers in source narration

A perf number written into a comment, a doc-comment or a string literal is a
claim that outlives the hardware, driver and code that produced it, and nothing
revalidates it. Two failures in this repo inside one day: a comment recording a
stage at "~76 s per forward" described the tree BEFORE the commit that
introduced it, and a later reader took it as current, costing an hour of
reconciliation against a fresh measurement of the same stage at 10.2 s; and
`ltxv t2v --help` advertised a two-card speedup a third larger than a fresh
reading showed. Users read help text as fact.

727 flagged lines across 285 files. The transformation everywhere keeps the
claim and its reasoning and drops the figure, because most of these numbers are
load-bearing to the argument around them - deleting the digits and leaving the
sentence produces something worse than either. A benchmark table becomes a
pointer to the harness that reproduces it plus the SHAPE of the result: which
kernel dominates, what is memory- versus compute-bound, which way the crossover
goes. A threshold the code actually depends on keeps its value, and the prose
now says it is a measured threshold rather than a reported result.

No escape hatch is used anywhere. The strongest candidate was a datasheet table
in the Vulkan peak-flops test, and even that rewrote: the values already live in
the constants the test computes against, so the header keeps the derivation and
points at them.

Two things found while reading rather than merely stale: a five-line comment
block duplicated verbatim in `qwen3::serve`, and `matmul_i8_gemv`'s rationale,
whose subject was ambiguous enough to read as an argument against the kernel it
introduces. It is not - the kernel measured slower there is the TILED int8 GEMM
run in the decode regime, which is precisely why this one exists. Said
explicitly now.

Drive-by, because staging these files surfaced it: two more absolute machine
paths in a qwen35moe doc comment, replaced with the checkpoint-relative name
and the env var that resolves it.

Numbers keep their home in dated, immutable records - commit messages, the
per-model roadmap ledger, and the rules ledger where the value often IS the
lesson. What changes is that they no longer sit in code, where they read as
current truth. ([564c3ad1](https://github.com/swedishembedded/brain/commit/564c3ad1e30fccf5ce0865af17923446bda00910))

- Extend the no-perf-numbers gate to source narration ([0cbb4673](https://github.com/swedishembedded/brain/commit/0cbb46735f3dca7fd23f37c025b41af88c0319cb))

- Ltxv, kernels: a flash text cross-attention, and the RMSNorm sibling ltxv never registered

Two kernels, both found by profiling at the width the model is actually run at
rather than the width that is cheap to profile.

**Text cross-attention had no flash kernel anywhere in the tree.** It ran as a
materialized trio - scores, softmax, apply - at roughly 5% of BOTH the compute
and the memory roof, which this repo's own rule calls a bug rather than a
ceiling, while self-attention in the same block did more arithmetic at 36% of
roof. `flash_attn_cross_reg2` is a port of `flash_attn_bidir_reg2` rather than
of the causal GQA kernel: the bidirectional rung is the closer template and the
port is two changes, three operand buffers instead of one fused slab and an
`nq`/`nk` split. It needs no `pack_qkv` because ltxv's attention already
produces q/k/v as three plain buffers. Selection goes through a shared
`model::block` seam, so any model with a cross trio adopts it in one call, and
`BRAIN_NO_FLASH_CROSS` is the A/B switch - without one, an A/B on a capable
device compares the fused path against itself and reports a parity that looks
like evidence and is not.

**RMSNorm was the recurring defect class**: `rmsnorm_rows` already existed and
`wan` and `flux2` already registered it. ltxv had simply never learned about
it, so every norm in every block ran one thread per row. One registration, one
call site.

Measured on one idle Tesla P40 (GP102, measured roofline 10517 GFLOP/s fp32,
287.5 GB/s DRAM), via `BRAIN_PROFILE=1 ltxv_bench streamed 8 13200 1024 1 1 1`,
8 layers at 13200 tokens and 1024 context, cache-hit arm, cards confirmed idle
before each run and never sampled during. `BRAIN_PROFILE`'s tables are
cumulative, so every figure is the difference between the tables at the end of
call 2 and call 1:

  text cross-attention   3688.5 ms -> 486.8 ms   7.58x   (5% -> 34.6% of roof)
  RMSNorm                1060.2 ms -> 108.6 ms   9.76x
  GPU kernel time         14471.6 -> 10308.8 ms  1.404x
  wall, 8 layers            27.88 s -> 23.20 s   1.202x

Control: across three separate runs the rows NOT touched reproduce to under
0.4%, so the deltas are the change rather than run-to-run noise. Wall moves
less than device time because a forward also spends time on activation upload
and a host RoPE table build, neither touched here. The fused kernel also stops
allocating a multi-gigabyte score-plus-probabilities slab per layer.

Parity on real Q8_0 weights, from printed output: `dit_parity` including the
exact tap replaced at cosine 1.000000000, `host_forward_parity`,
`shard_parity`, `streamed_vs_eager_real`, `connector_real_parity`,
`int8_compute`, and `av_dit_parity` - the A<->V cross-attentions are fused by
the same seam.

New gates assert max_abs AND rel_l2 alongside cosine over six shapes with both
`nq>nk` and `nq<nk` and both tile tails, plus an analytic V-index test.
Mutation-verified, and one mutation is the reason the gate is not cosine-only:
a score-scale error scores cosine 0.999979 while rel_l2 reads 6.864e-3 - four
orders less alarming on the metric most people would gate on, and the more
plausible bug of the set.

One hypothesis was refuted rather than implemented. A recorded note claimed
`head_dim = 64` against a 128-wide tile wasted half of every tile. This model
is `inner_dim 4096 / 32 heads`, so head_dim is 128 and matches the tile
exactly. There is no zero-fill and a template knob would compile to the
identical kernel, so none was built. ([50569233](https://github.com/swedishembedded/brain/commit/505692337a9ad383c8ae51423435fa6ba6f0b50c))

- Native audio-visual generation, wired end to end ([0c4eedb2](https://github.com/swedishembedded/brain/commit/0c4eedb29a6bf0f69217848e055ce3771c8cccd3))

- A quantized, device-resident audio-visual block ([5dc2ce08](https://github.com/swedishembedded/brain/commit/5dc2ce08ee1ede20d199c60692010f2a550947e9))

- Audio crosses a window seam, so a long clip can have sound ([5c5d3b3b](https://github.com/swedishembedded/brain/commit/5c5d3b3be63ca66ee50996dfa68ca6f0a5fb62b3))

- Capability-mock, catalog: an in-process weight-free Provider, and the served-model catalog as a library

Groundwork for whale/Loom linking brain's capability crate in-process:
whale must never download or run real weights during development, and
wants to enumerate the full ~70-model catalog without going through the
CLI/D-Bus/HTTP transports.

brain-capability-mock (capability_mock) depends on nothing but
brain-capability + serde_json. MockProvider advertises a Manifest built
by hand (new + action) or mirrored 1:1 from a real one (from_manifest,
inferring each action's synthetic shape from its declared output Media
kind), and every action generates deterministic content by pure wrapping
arithmetic - no RNG crate, no file I/O, no GPU: a seeded HWC gradient for
Image/Mask (through blob::image_blob, mask re-tagged via
Blob::with_media), a moving-gradient clip for Video (through
blob::video_blob, meta.fps set), a sine tone as raw f32-LE PCM for Audio
(meta.sample_rate/meta.channels, no WAV container - matching
audio::asr_caps's raw-PCM convention), a prompt-derived string for Text,
and a counter byte pattern for Bytes. Every action ticks a handful of
Progress::steps and polls the invocation's CancelToken between them,
aborting promptly on cancel - the discipline every real model action is
required to follow, demonstrated rather than merely documented.
Round-tripped through capability::blob's own decoders.

Note: the builder MockProvider::action(self, spec, output) and the
Provider::action(&self, name) dispatch method share a name by design
(matching the requested shape); method-call syntax always resolves to the
inherent builder, so a caller dispatching by name needs
Provider::action(&provider, name) - documented on the type and used
throughout its own tests.

brain-catalog (catalog) is crates/cli/src/catalog.rs promoted to a
library: the LazyProvider machinery, ModelEntry, the always!/from_env!/
resident!/resident_multi! macros, and models()/manifests()/provider() -
one manifest + weight-free provider constructor per registered model
crate, in ONE list, so an in-process consumer (this workspace's
brain-cli, or a separate binary like whale) enumerates and constructs
every model with no CLI/D-Bus/HTTP transport in the loop.

Deviation from a purely mechanical move: about twenty models' residency
adapters (Sam2Resident and siblings) are crate::resident_* types defined
in crates/cli, which - per this workspace's crate-graph layer rule (cli
aggregates everything and sits at the top of the stack; catalog sits in
the same layer as capability/residency/dbus, all of which cli depends on,
never the reverse) - brain-catalog must not depend on. Splitting the
manifest/provider pair out per model would have reintroduced exactly the
multi-list drift this file was built to kill, so instead brain-catalog
stays the single source of truth for manifest + provider (every model, no
exceptions), and crates/cli/src/catalog.rs becomes a thin extension: it
patches the CLI-local residency adapters back onto catalog::models()'s
entries by the model crate's own caps::MODEL constant (a rename now fails
to compile instead of quietly un-serving a model), and appends the four
entries whose MANIFEST itself is CLI-local (imageops/demo, and the three
forecasters, whose manifest fns are pub(crate) in resident_forecast.rs
and were never reachable from outside crates/cli regardless of this
split). residents()/multi_residents() move to this CLI-local file
entirely, since they are inherently about CLI-local ResidentModel impls.

crates/cli gains a brain-catalog dependency and loses brain-qwen3vl (no
longer used anywhere in crates/cli now that the old catalog.rs's one
reference moved into the new crate - confirmed via cargo build --tests,
no other CLI file names it). brain caps still lists the exact same 38
models (verified with a throwaway count test before removing it);
ltxv_cli::tests::every_upscale_flag_the_parser_accepts_is_documented
fails identically on main before this change, a pre-existing, unrelated
drift in that file's own --help text.

Both new crates build with zero warnings (cargo build/clippy
--all-targets); 15 new unit tests in capability-mock (round-trip through
capability::blob's real decoders, from_manifest inference, two
cancellation tests) and 4 in catalog pass; crates/cli's existing catalog
tests (uniqueness, constructibility, imgpipe stage ids, the two
residency-specific invariants, a new one pinning every patched id) all
pass, plus one new test asserting the patch table's ids are real. Root
Cargo.toml: both crates added to default-members and
[workspace.dependencies]. .agents/rules/architecture.md and AGENTS.md
document both under the serving and front-ends layer. ([859427cf](https://github.com/swedishembedded/brain/commit/859427cf0a4c0ef724cd26c5992421ce200affb1))

- Route audio-visual generation through the quantized path ([db675eb2](https://github.com/swedishembedded/brain/commit/db675eb20d5e8d04f6ba7adc51b2c058a41a2e99))

- Give the int8 GEMM a shared tile shaped for DP4A, not for FFMA ([a9e3e5f6](https://github.com/swedishembedded/brain/commit/a9e3e5f6ec5c356e51c01d69b8f9561502573c12))

- Gpu-core, ltxv: a replay scratch arena, because the host half was the allocator

A warm forward at real width spent a third of its wall clock with the card
idle. Profiling the host half - in-process spans first, then `perf record -g
--call-graph dwarf` from outside, agreeing - found it is not host MATH at all.
It is the Vulkan allocator, and half of it was somewhere nothing was timing.

Attribution of a 48-layer warm forward's 30.24 s of host time, every row a
span overlapping no other, with **zero unaccounted**:

  device buffer allocation   12.04 s  39.8%   (~74 temporaries per block x 48)
  device buffer destruction  10.53 s  34.8%   (the same set, dropped per block)
  submit + poll residual      1.84 s   6.1%
  block weight upload         1.67 s   5.5%
  graph recording             1.65 s   5.4%
  output stage (host)         0.88 s   2.9%
  ... nine smaller rows       1.63 s   5.5%

**Destruction had never appeared in any table.** The recorded figure for this
cost covered allocation only, because every existing timing span closes before
the function returns and dropping the buffers is the last thing it does. A
span ending at a closing brace cannot see its own scope's destruction. The two
halves together were 75% of the host time.

`perf` confirms it from outside the process: the allocator's block-new and
free paths, `ioctl` and `munmap` together dominate, and all four vanish or
halve in the fixed arm. Note the frame that dominates a flat profile here is
the process blocked in `device.poll(Wait)`, which is device time rather than
host stall - its share RISES after this change, which is the correct direction.

`gpu_core::scratch::Arena` + `Gpu::scratch_scope` is a replay arena: a repeated
pass asks for the same temporaries in the same order every iteration, so the
arena hands the same buffers back and nothing is created or destroyed after
the first block. It lives in the shared facade rather than in one model, so
opting in is one line per loop body.

Aliasing safety is three separate parts, and they were verified separately
because they mask each other: the cursor only advances, so no collision within
a scope; a slot is refused while any caller still names its buffer, which also
makes the chained activation correct with no special case; and the caller must
drain, which the existing blocking one-word read already does. A first draft
of the doc claimed the refcount also covered submitted dispatches - checked in
the backend and it does not, a recorded `Step` holds a bind group rather than a
buffer clone, so that argument would have been circular.

Measured on one idle P40, best-of-2 warm, no concurrent build, 48 layers at
13200 tokens, A/B through `BRAIN_LTXV_NO_SCRATCH_POOL`:

  video  warm forward  87.72 s -> 64.56 s  1.36x   host -78.5%, device +0.3%
  audio+video          101.60 -> 72.98 s   1.39x   host -76.0%, device +0.4%
  device share of wall   66.1% -> 90.1% (video), 62.6% -> 87.5% (AV)

Device time unchanged is the control; output mean/std/min/max are identical to
six decimals on both streams, and the residency grant and peak VRAM did not
move. Independently reproduced at 8 layers: 16.01 -> 12.30 s wall with device
9.66 s in BOTH arms and identical output stats.

The before arm is a fresh run rather than a citation, and it does not
reproduce an earlier phase's absolutes - both halves measure about 6% faster
here, box or toolchain. The host SHARE reproduces to the decimal, 33.9% then
and now, which is the figure the conclusion rests on.

Five mutations, each restored and re-confirmed. The finding is that the guards
mask each other: deleting the size check first PASSED, because the uniqueness
guard refused the slot before the size ever mattered. That test now releases
the buffer so the size check is what it actually exercises. Dropping
uniqueness AND handing back the previous slot turns the real-weight bit gate
red at 2048 of 2048 words, which is what proves that gate can see aliasing at
all. ([c2a442c8](https://github.com/swedishembedded/brain/commit/c2a442c86955dc636d50aacb6ee3bd75f4b0d93b))

- Vae, kernels: one fused channel norm, and one device for the whole tiled decode

The VAE decode is a fifth of a real generation and every optimisation this
model has had went into denoise. Profiled at the real clip geometry it is
85.7% device-bound, and three of the top five kernels turn out to be ONE
operation.

`nchw_nlc` -> `l2norm_scale` -> `nlc_nchw` is a channel-wise L2/RMS norm
written as a permute sandwich. That form pays the strided channel walk TWICE,
once per permute, to spare the middle kernel from paying it once - and the
middle kernel then redoes each position's whole sum of squares once per
channel. Both permutes tripped this repo's roof-floor rule at a real tile
shape. The argument against the sandwich was already settled for LayerNorm by
an existing fused kernel; the L2/RMS twin had simply never been written.

`l2norm_scale2d` is that twin: one invocation per spatial position, one pass
over the channel axis, barrier-free. Bit-identical - the permutes are exact
and both arms fold over ascending channels - so the gate asserts the bits.

It went into the single shared call site both norms already route through, so
`crates/wan`'s VAE inherited it with no change in that crate at all, and a
private byte-identical copy of the same trio inside this crate's audio VAE now
uses the shared kernel instead.

Second fix: each tile SHAPE was opening its own device and re-uploading the
whole decoder. One device and one weight memo now thread through every build.

Measured on one P40 at the real geometry, same binary, A/B through
`BRAIN_VAE3D_SPLIT_NORM` and `BRAIN_LTXV_VAE_NO_SHARED_WEIGHTS`:

  norm kernels        61.82 s -> 2.28 s   27.1x
  VAE device         332.49 s -> 273.27 s  1.217x
  host                 25.8 s -> 27.0 s    unchanged, the control
  graph build, 8 shapes 31.96 s -> 5.83 s
  graph drop, 8 shapes   7.03 s -> 1.86 s

The two convolution rows agree to 0.2% and 1% across the matched pair, so they
did not move; both roof-floor rows are gone and the pass went from 26.5% to
30.3% of its roof at one tile shape. The stage is now 91% device-bound and its
device half is 95% two kernels.

Two recorded numbers are corrected. Tile overlap waste at this geometry is
1.502x, not the 1.192x on record - that figure came from a 25-frame clip whose
temporal axis never split, where this clip plans 16 tiles in 8 shapes. And
this harness measures the pre-change decode at 357.9 s where the pipeline
reports 645.7 s for the same geometry; the gap is not chased here, so read the
ratios rather than the absolutes.

Mutation-verified, each separately: a REVERSED FOLD ORDER turns the gate red
at 2709 of 5040 words, which is what proves it sees a pure re-association
rather than only gross error; a dropped per-channel gain turns it red; a wrong
NLC indexing assumption segfaults the CPU JIT, which catches an indexing class
the bounds-checked GPU arm only shows as garbage; a mis-keyed weight memo turns
the sharing gate red.

The sharing gate's own shape-count guard fired on the first geometry chosen for
it: an 8x8 latent under a 4-cell tile splits into three EQUAL tiles, so that
geometry would have compared the shared arm against itself and passed forever.

Left undone deliberately: at this geometry a third of the stage's device time
decodes pixels the blend averages away, and a tile-size search bottoms out near
1.31x overlap - about 12% of the stage with no kernel written. Taking it needs
a per-request VRAM estimate the residency path still lacks, and guessing one
here would give the workspace a second answer to how much fits.

Measurements were taken on a shared box: the other card ran a concurrent
workstream and this card ran thermally throttled for part of the session, which
is why the headline is a matched-thermal pair rather than a first-and-last
comparison. ([7caf167c](https://github.com/swedishembedded/brain/commit/7caf167c8b41b957bacf39fc789b2844499d9047))

- Ltxv, backend-wgpu: record the next block while this one runs

The scratch arena left the block loop 89% device-bound, and the remaining host
time was serialised rather than large. A block body is RECORD, SUBMIT, WAIT,
and recording touches no device memory - a bind group NAMES a buffer, it does
not write one. So the per-block drain never had to sit between this block's
submit and the next block's recording. It only has to sit between the previous
block's submit and this block's SUBMIT:

  serial     record(l) submit(l) wait(l)   record(l+1) submit(l+1) wait(l+1)
  pipelined  record(l) wait(l-1) submit(l) record(l+1) wait(l)     submit(l+1)

Same drains, same submissions in the same order, at most one block in flight
either way. `BRAIN_LTXV_NO_PIPELINE` is the A/B arm.

The alternating-arena design this was dispatched with was unnecessary and is
not built. It assumed the aliasing window is "recording block N+1 while N
runs"; it is not, because N+1 only takes the arena's HANDLES while recording
and every dispatch that writes one is submitted after the wait that completes
N. One arena, no extra VRAM. `gpu_core::scratch`'s contract is restated to the
condition actually required: drained before the new scope's dispatches are
SUBMITTED, not before the scope is entered.

The instrument had to be fixed first, and this is the part worth reading.
`flush_timed` resolved its query sets and then MAPPED them, which blocks until
the submission completes - so the harness that reports "device share of wall"
was itself a per-flush drain, and the pipelined arm measured identical to the
serial one. The timestamp readback is now deferred and folded in on a
deliberate read.

The buffer-lifetime question was checked in the backend rather than inherited.
A recorded step holds a bind group, not a buffer clone, as previously
established - but the conclusion drawn from that was incomplete: this wgpu
prepends a transit pass built from the DEVICE-GLOBAL tracker on every submit,
and a storage buffer is never in the ordered-uses mask, so a recycled buffer's
old and new users are ordered by the backend across submissions. The drain is
load-bearing for MEMORY, not for arithmetic. That is why deleting it turns no
bit gate red, and why the control mutation below matters.

Measured, 48 layers at 13200 tokens, int8, resident, best warm call with the
first excluded, same binary A/B:

  warm forward   65.44 s -> 62.49 s   1.047x
  device          58.22 s -> 58.24 s   +0.03%, the control
  host             7.22 s ->  4.24 s   -41%
  device share      89.0% -> 93.2%
  cold forward   138.87 s -> 106.21 s  1.31x
  peak VRAM        18092 -> 18555 MiB  bounded, flat over six forwards

One block of lookahead, with the number that decides it: deleting the wait
entirely is 1.1% faster and costs 23222 MiB of 24576, which does not fit the
audio-visual path at all since it already peaks at 21082. At one block the card
is busy 1.21 s per block against 0.09 s of host work, so the host is already
idle 93% of the loop.

Mutation-verified, each separately. Deleting the drain turns NOTHING red, which
is not a blind gate: the control - reusing the previous arena slot with the
uniqueness guard removed - turns both bit gates red under the pipelined arm, so
the gate can see aliasing and the backend really does order the other case.

A bug shipped and then found by measuring, which is the same shape as the one
this workstream started from. Resolving timestamps in the profile dump made
EVERY handle drop wait for the queue, including the shared handle an evicted
resident block holds, dropped once per streamed block mid-forward. A destructor
sits outside every span the caller times, so it appeared nowhere: the stage
table blamed the weight upload, 2.3 s to 34.4 s, and it read as the pipelining
not working. A temporary probe around the loop's phases found it at 96 ms to
9801 ms per block drop. Only the last handle resolves now, and a new gate pins
it.

The audio-visual path takes the same reordering but was not re-measured; the
bit gate covers the video stream. Measurements were taken while another
workstream shared the CPU and occasionally gpu0 - one before-run whose device
time landed outside the 58.0-58.3 s band was discarded, and every pair above is
same-binary. ([b55bff84](https://github.com/swedishembedded/brain/commit/b55bff84c3ec708479883cd0eaa9035809cdf0bd))

- Reuse the readback staging buffer, and stop casting mapped bytes in place ([70a91b01](https://github.com/swedishembedded/brain/commit/70a91b0184d107135054f9a994a846607ea1192a))

- Model, flux2: load and fold third-party LoRA adapters

brain could TRAIN a FLUX.2 LoRA but not LOAD someone else's: `lora::load_adapter`
accepted only brain's own checkpoint container, and `LoraAdapter::from_tensors`
holds per-slice pairs (separate A/B for q, k, v) that a third-party file's FUSED
qkv pair cannot be read into at all. So every adapter published for FLUX.2 Klein
was unusable here.

`model::lora::read_external_adapter` parses the ai-toolkit / ComfyUI / diffusers
convention into per-linear pairs, resolving each to the base tensor it targets by
ComfyUI's own rule: strip a leading `diffusion_model.`, strip the
`.lora_{A,B}.weight` suffix, and the stem plus `.weight` is the base key. The
three common A/B spellings and `.alpha` are accepted. `flux2::fold_external_adapter`
then validates the WHOLE adapter against the tensor map before writing anything and
folds `W += strength*(alpha/r)*B*A`.

The fused form is a simpler fold than brain's own, not a harder one: every target is
a whole tensor at offset 0, so `Pair::delta` is the exact operation and no second
`B*A` is introduced (`Pair::from_ab` supplies a moment-free pair for it).

Semantics are taken from the reference implementations rather than inferred:

  * ComfyUI `comfy/weight_adapter/lora.py` - `weight += (strength * alpha) *
    mm(mat1, mat2)`, `mat1` the up/`lora_B`, `mat2` the down/`lora_A`, and
    `alpha = v[2]/rank` or `1.0` when no `.alpha` tensor is present.
  * ai-toolkit `toolkit/network_mixins.py` - `scale = alpha / lora_dim`, with alpha
    initialised to the rank and `.alpha` stripped from PEFT-format saves.

Both therefore resolve an alpha-less file to a multiplier of exactly 1.0, by two
different routes. `B*A` needs no transpose: both store PyTorch `nn.Linear` weights
`[out, in]`, already brain's row-major manifest layout.

Failure is loud by design. An adapter key matching no base tensor, a half pair, a
rank disagreement or an unrecognised name is an error naming the tensor, and the
base weights are left untouched. A loader that quietly skips a key returns
base-model output from a run the user believes is adapted, which is the one failure
mode here that looks exactly like success; `tests/lora_external.rs` gates each of
those cases, plus a `BRAIN_FLUX2_LORA` test that folds a real published adapter over
the real klein-9b manifest and asserts all 112 of its linears are reached.

clippy-gate: exit 0, 0 warnings (baseline 0). ([58eee7af](https://github.com/swedishembedded/brain/commit/58eee7afd7d610f9418833f425bd83a7c37a7bc8))

- Flux2, cli: apply a LoRA adapter from the CLI, with a strength dial

The adapter loader landed in the previous commit but nothing could reach it.
`brain flux2 generate` passed a literal `None` where the pipeline takes an
adapter and had no flag to pass one, so the only route to a folded LoRA was
D-Bus or HTTP, and even that route could load only brain's own checkpoints.

`Pipeline::build_*` now takes an `AdapterSpec { path, scale }` instead of a bare
path, and picks the family by extension: a `.safetensors` goes to
`fold_external_adapter`, anything else to brain's own `load_adapter`. Both fold
into the same f32 tensor map before quantization, so an adapter works at fp32 or
int8 - the same order ComfyUI uses, patch the weights then run.

`scale` exists because a third-party adapter file carries no alpha to read. Both
ai-toolkit and ComfyUI resolve an alpha-less file to a multiplier of exactly 1.0,
so 1.0 is the default and this is ComfyUI's `strength_model`, a user dial rather
than a value recovered from the file. `--lora-scale 0` reproduces the base model
exactly, which is how to see what an adapter actually contributes.

Surfaces:

  * `brain flux2 generate --adapter <path> [--lora-scale S]`
  * the capability `adapter` param now accepts either family, plus a `lora_scale`
    param, so HTTP and D-Bus get the same dial
  * the resident instance key carries the strength next to the path, since a
    different strength is different folded weights and must not reuse a cached
    pipeline. The path stays last in the key because it is the only field that
    may contain ':'

Folding is announced on stderr with the linear count, rank and strength. A run
that claims to be adapted should say how much of the model it moved, so a silent
no-op cannot hide behind a clean exit.

clippy-gate: exit 0, 0 warnings (baseline 0). ([b8d12fe6](https://github.com/swedishembedded/brain/commit/b8d12fe665fdddab7d74702f88cbb7f267d3b189))

- Stop discarding BRAIN_GPU_INDEX on a run that narrows nothing ([f25c2763](https://github.com/swedishembedded/brain/commit/f25c2763af881692acc9845f68d5008bae336f1c))

- Imaging, cli: report where a clip's sound actually went, not where it was sent

`encode_frames` filters a generated audio track against the requested
container: a path whose extension carries no audio stream drops the track
and says so. The no-ffmpeg fallback then printed, from `ltxv_cli`:

  imaging::video: . carries no audio stream; the clip is written silent
  ltxv: the generated sound is <dir>/audio.wav - it is NOT lost, the
        command below muxes it in

Three lines apart, and the second is false. The file was never written and
the printed ffmpeg command has no second input. `ltxv_cli` derived that
reassurance from `video.audio.is_some()` - whether the PIPELINE made sound
- so it fired precisely when the encoder had discarded it. "It is NOT
lost" is also the one sentence that makes a user stop looking.

`write_frame_dir_with_audio` now returns `(command, the WAV it wrote)` and
`Encoded::Frames` carries that as `audio`, so the encoder's own answer is
the only thing a caller can report. The false statement is no longer
representable: `has_audio` cannot reach the message. When a track was
generated and then dropped, the CLI now names the remedy instead:

  ltxv: the generated sound was DROPPED - re-run with an --output-path
        ending in .mp4/.mkv/.mov/.webm to keep it

The four call sites that never mentioned audio (`ltxv upscale`, `ltxv
dfr`, `wan`, `caps`) bind the new field as `_`.

Also: an extensionless path rendered the container as a bare "." in that
first message, which reads as a typo rather than as the cause.

Gated by a test asserting the fallback reports the WAV it wrote and
reports nothing when silent, mutation-verified by making the function
return `Some(dir.join("audio.wav"))` unconditionally - the exact defect
shape - which fails it on "no sound track means there is nothing to
report". Verified end to end on this 2x Tesla P40 host by generating with
ffmpeg hidden from PATH: an extensionless `--output-path` now prints the
DROPPED line, and a `.mp4` one writes a real `audio.wav` beside the PPMs.

brain-imaging lib 59 passed, clippy-gate: exit 0, 0 warnings (baseline 0). ([568f377f](https://github.com/swedishembedded/brain/commit/568f377fd260558b8dec6f5d335ffbe146e1d1f3))

- Stop `ltxv_bench vae` reporting a path a generation would not take ([5f67f67d](https://github.com/swedishembedded/brain/commit/5f67f67d1aec12d627f4397e87fe59fb2da182a7))

- Flux2 roadmap: record the whole-checkpoint text-encoder import

`pipeline.rs` builds the text encoder as `Shard { start: 0, end: deepest
tap, embed: true, head: false }`, so the layers past the deepest tap and
the LM head are never read. The import does not know that: it runs first
and demands the whole checkpoint.

`checkpoint::safetensors::read_model_dir` reads every shard named in the
index's `weight_map` and takes no parameter describing what the caller
wants, and `qwen3::import::brain_init_from_hf` enforces two-way coverage
against the full `param_list()` of a config carrying the untruncated
`n_layers` with `tie_embeddings: false`. Both must be satisfied before the
`Shard` is ever consulted.

For the Qwen3-8B encoder that is roughly 4.2 GB of 15.6 GB fetched,
dequantised and validated to be discarded, the LM head among it. Found
while waiting on that download at this host's measured ~2.6 MB/s uplink,
where it is most of an hour before the first image.

Recorded rather than fixed: the shape of the fix is a shard-aware import
(derive the required names from the `Shard`, and let `read_model_dir` take
that set so it can skip whole shard files), which is not a change to make
underneath a run in progress. The note also pins two things a later
attempt would otherwise get wrong: `hf_source`'s streaming path is not
this fix (it lowers the ~32 GB host-RAM import peak but validates against
the same full list, so it saves memory and not bytes), and the two-way
coverage check must stay exact against whatever set is genuinely required
rather than be relaxed, since it is what catches a wrong checkpoint. ([6f75c29f](https://github.com/swedishembedded/brain/commit/6f75c29f4dccaaddbf2eb11ad6f284642e9598f0))

- Stop force-fetching a default checkpoint a fully configured run cannot use ([7efd8e25](https://github.com/swedishembedded/brain/commit/7efd8e2581e8e18ec0c042ca32d9c2f12650d444))

- Size a pipeline for the reference tokens it will actually attend to ([959e8c34](https://github.com/swedishembedded/brain/commit/959e8c346f0cc18b6f7f4a7b09d08561cc54bd59))

- Give the denoise loop a spatial dial, not just a global one ([6813f003](https://github.com/swedishembedded/brain/commit/6813f00367efcfd944ef4adf9b6f95ed451649f3))

- Cli, docs: expose the FLUX.2 preservation mask as `--mask`

`brain flux2 generate --mask <image>`: white regenerates, black preserves the
first `--ref` exactly, greys blend. Any resolution - the pipeline area-averages
it to the latent grid - and the load path is the same PPM/PNG/JPEG decoder the
`--ref` images already use, so a mask is just an image.

The run prints what it loaded and how much of the canvas the mask actually
frees ("Mask(1024x768, 53.4% regenerate)"), because a mask that silently came
out all-black would otherwise look exactly like a model that refuses to stage.

The docs say plainly that there is no automatic mask generator and why the two
obvious ones were tried and rejected on the real photographs: a monocular-depth
near-field threshold marks the CEILING as foreground on a living room (directly
above the camera it genuinely is the nearest surface) while leaving a sofa
against the far wall as background, and a depth top-hat fixes the ceiling but
misses any object larger than its structuring element - a bed filling half the
frame is entirely missed. "Near" is not "furniture". Also recorded: staging has
to ADD furniture where there is none, so a mask covering only what is already
in the room cannot stage, and the floor is at once architecture and the place
new objects go, which is what the grey levels are for.

The roadmap carries the measured ladder for both real photographs, on 2x Tesla
P40, and the note that the whole-frame edge correlation of a genuinely restaged
room cannot exceed roughly 0.4 - new furniture has new edges. The number worth
gating is edge correlation on the PRESERVED region, which reaches 0.980 and
0.991 against VAE-round-trip ceilings of 0.988 and 0.996. ([479c8575](https://github.com/swedishembedded/brain/commit/479c8575e19a8d2ec579ff44ff353d8947b0dea5))

- Ltxv cli: say in `upscale --help` that it reads no audio VAE and emits no sound

`BRAIN_LTXV_AUDIO_VAE` joined `pipeline::OPTIONAL_PATH_VARS` without reaching
`UPSCALE_HELP`, which `every_upscale_flag_the_parser_accepts_is_documented`
catches - the suite has been red on it, unrelated to whatever change happens
to be in flight.

Documenting it "not used here" is the honest resolution rather than relaxing
the test: `upscale` runs no audio-visual DiT. But the interesting half is the
consequence, which nothing in the help said either - `upscale` builds its
`Video` with `audio: None`, so the upscaled clip comes out SILENT and the
input's sound track stays in the input file. A user upscaling a clip they
generated with `--audio` would otherwise find out by watching the result. ([09834032](https://github.com/swedishembedded/brain/commit/0983403254091f54a0eaef6726ea4b79491e0363))

- Read a GGUF through a mapping instead of slurping it ([63f79d0b](https://github.com/swedishembedded/brain/commit/63f79d0b27735b68deda5a0ffc56fa99232c946d))

- Split a fused linear's columns in parallel, once, for everyone ([7a83a747](https://github.com/swedishembedded/brain/commit/7a83a74791265fa188b41592916c3ee01a029081))

- Time the model build, and stop splitting linear2 serially ([2dd1cea6](https://github.com/swedishembedded/brain/commit/2dd1cea69c7d77c8ccf9578220b96205f2d8ab73))

- A `load` mode, because the weight load is not free ([2761892a](https://github.com/swedishembedded/brain/commit/2761892ab080a34858b64b6e3b8ea14173fbbfd1))

- Flux2 roadmap: record where a klein-9b int8 generation's time goes

Ranked, measured, at BOTH token counts, host and device, with the roofline
each kernel is actually against. Numbers live here rather than in code or
docs, per the meta-rule in .agents/rules/kernels.md.

The three findings that change what to do next:

1. A single-image run is a one-off weight LOAD plus a generation, and the
   load was 41% of process wall with zero instrumentation on it. The older
   table's "~61 s" was the pipeline stage total, not the process wall; the
   wall was ~105 s. Both are now recorded so they cannot be confused again.

2. The denoise top row CHANGES IDENTITY with sequence length. Attention
   scales 3.43x against a 3.45x quadratic prediction while everything else
   is linear, so GEMMs dominate at 3584 joint tokens (60% combined) and
   attention alone dominates at 6656 (46%). A profile at one size is not
   evidence about the other.

3. Device kernel time is 97% of the denoise stage wall, so there is no host
   bubble in the denoise loop and the replay-arena work that paid off on
   ltxv has nothing to recover here.

Also corrects three stale or wrong entries: the flash-attention "second
query row per thread" item was already implemented (`flash_attn_bidir_reg2`
is that kernel, and the real remaining gap is a ~2x shared-memory-bandwidth
bound that a third query row cannot fix inside the 48 KiB limit); the
staging-reclaim readbacks cost 0.14 s, not seconds; and `gpu.write` runs at
3.0-3.7 GB/s, not the page-fault-bound rate.

Records the largest remaining load lever - a direct Q8_0 to packed-int8
requantisation - together with the argument that it can be made
BIT-identical, and the two non-numerical things blocking it. ([c5ff59af](https://github.com/swedishembedded/brain/commit/c5ff59afb2c0233bef2cde2e56ad0830d2effe86))

- A two-command virtual-staging example with a third-party LoRA ([a9c5221b](https://github.com/swedishembedded/brain/commit/a9c5221b48ab59052fff4049a5176b5e2950d7c2))

- Backend-cpu, model, checkpoint: the primitives a direct requantiser needs

Three small additions, inert until the next commit uses them. Each is
deliberately a SHARED primitive rather than something the caller
reimplements, because the caller's whole claim is going to be bit-identity,
and bit-identity against a reimplementation is a coincidence waiting to
break.

`par::chunks_mut_with` - `chunks_mut` over two disjoint outputs at once. The
shape of a loop producing two different things per row and needing both:
packed weight words plus that row's scale. Two passes would either read the
input twice or recompute half the work.

`int8::row_scale` / `int8::pack_row` - `quantize_weight`'s per-row
arithmetic, lifted out and made public, with `quantize_weight` now calling
them. A caller that obtains a row's f32 by some other route (decoding it from
a quantized checkpoint block rather than indexing a materialized fp32 tensor)
can now reach the identical packed bytes by CONSTRUCTION - running the same
code - rather than by reimplementing the scale/round/clamp and hoping. The
left-to-right fold in `row_scale` is load-bearing and says so: `f32::max`
propagates the non-NaN operand, so with a NaN present the order is
observable.

`gguf::q8_0_expand` - decode a block-aligned element range of a Q8_0 tensor
without expanding the whole tensor, through the same `deq_q8_0` every other
read path uses. Refuses an unaligned range rather than rounding it, since a
Q8_0 block is the smallest independently decodable unit. ([2329f5b8](https://github.com/swedishembedded/brain/commit/2329f5b8d42c8306b82c69e5fbfabc83d16cb63d))

- Requantise Q8_0 straight to int8, never building the fp32 model ([5f100912](https://github.com/swedishembedded/brain/commit/5f1009129215ea7e63d80cf1328dc6ee0ff32cff))

- Flux2 roadmap: record what the direct Q8_0 path actually bought

Replaces the estimate with the measurement, and keeps the estimate's ERROR
on the record because it is the more useful half: it predicted ~11.5 s on
the assumption that the quantize term would go to zero once the checkpoint
"already holds int8". It does not - the block-scale to row-scale conversion
still touches every weight - so only the whole-model dequant and the free
disappear, and the measured saving is 7 s.

Also records the two things the measurement changed about what to do next.
The process host peak is now bounded by the TEXT ENCODER's fp32 import
rather than the DiT (the DiT phase peaks at 10.4 GB; the 32.9 GB figure
arrives afterwards), which promotes the shard-aware TE import to the next
real lever. And with a full-coverage LoRA the streamed path is a trade -
slower, but holding 10 GB less - not a win, and is written down as a trade.

Drops a bare percentage from `weights.rs` prose that
check-no-perf-numbers.sh flagged; the claim reads the same without it. ([386176dc](https://github.com/swedishembedded/brain/commit/386176dcf589f8745db39941889ea8bd4a965ea6))

- A shard-aware streaming import, so a truncated encoder needs only its own tensors ([cbda83d7](https://github.com/swedishembedded/brain/commit/cbda83d7e14bcfd1671902ac558af325da652bd2))

- Stream the text encoder instead of importing it whole ([517a52d8](https://github.com/swedishembedded/brain/commit/517a52d80662ed2126592f715077bc803e829d52))

- Flux2 roadmap: record what streaming the text encoder actually bought

Closes the shard-aware-import item with the measured numbers, and keeps the
two things that are easy to lose: the whole-process figure is weaker evidence
than the import figure (contended cards, a run that OOMed in VAE decode rather
than completing), and the load got SLOWER - bounded footprint is paid for in
page-cache misses. Also records what deliberately stayed eager. ([d8a2092c](https://github.com/swedishembedded/brain/commit/d8a2092cc5289a376246d427c91dc5cf9f890f12))

- A VAE-only latent laboratory, so latent-space claims can be measured ([b59bd3e2](https://github.com/swedishembedded/brain/commit/b59bd3e20f50026d2bdf92e2dfd8763bc0f2fe0b))

- Flux2 roadmap: what the VAE latent actually is, measured

Twelve experiments through the new `flux2_latent`, all VAE-only. The three
that change what we should say out loud:

Reflection and rotation are equivariant ONLY at latent resolution - MAD 15.6
at full resolution, 2.7 after box-downsampling to the latent's own 8x grid,
with the knee exactly at the cell size. The latent does not store fine
texture; the decoder synthesizes it, and the synthesis is direction-dependent.

A region splice decodes with no artifact whether or not it is cell-aligned.
Misaligning by half a cell, so boundary cells hold a genuine fractional mix of
two unrelated latents, moves the seam-band MAD by 4-12% and is invisible at
1:1. This refutes the explanation previously given for masked-generation
artifacts, and the roadmap now says so.

At matched displacement (0.711 sigma-units) a latent blend decodes to a clean
photograph and Gaussian noise decodes to confetti. Three candidate
explanations for the difference - spatial coherence, cross-channel subspace,
convexity - were each tested and each refuted. What holds is that
differences of real latents are safe directions at any magnitude tried; that
was not reduced to a simpler statistic, and the entry says that too.

Measured on a Tesla P40 (gpu1): 5.5 s to load the VAE and encode three
1024x768 images, ~2 s per decode, bit-identical run to run and across cards. ([f1b5d6df](https://github.com/swedishembedded/brain/commit/f1b5d6df1598a3faa246f0482e9a1cac745362e4))

- Give the streaming decoder the parallelism the eager one already had ([2fac869c](https://github.com/swedishembedded/brain/commit/2fac869c540effdd094a48d5f143033d0fcdd034))

- Flux2 roadmap: streaming the encoder is no longer a memory-for-time trade

The earlier entry recorded a 13.8 s wall cost as the price of a bounded host
footprint. That price turned out to be a serial decoder in the streaming path,
not an inherent cost of streaming, and it is gone.

Keeps the refuted hypothesis alongside the confirmed one: `advise_dontneed` was
the obvious suspect and was innocent, which is worth remembering the next time
a streaming path looks slow. Also replaces the contended, OOMed whole-process
numbers with a completed best-of-2 pair on idle cards, and records that the
output PNG md5 is identical before and after. ([d432c67d](https://github.com/swedishembedded/brain/commit/d432c67d667b94c652eb608afae23e9bd6dc64b4))

- A valve for the text-encoder route, and the device budgets to justify it ([1a7bac2c](https://github.com/swedishembedded/brain/commit/1a7bac2c2c6d41b1a0cc1fece4ae3d451d545f04))

- Read a checkpoint once, and give its config a cheap door ([39a54371](https://github.com/swedishembedded/brain/commit/39a54371bf2b222d1216d6250eba5c3c538fb1e9))

- The blk.N name shape belongs to llama.cpp, not to one model ([24be7b82](https://github.com/swedishembedded/brain/commit/24be7b8236d1859b4f5519e89c28a0158fdb7f33))

- One load path, two container formats ([df3cd48c](https://github.com/swedishembedded/brain/commit/df3cd48cd3c225dd26f0597a933760b88dbb7316))

- A reference the model can actually see at every --strength ([acef1073](https://github.com/swedishembedded/brain/commit/acef10735178871923ca7469a5e56dbf677d3c5e))

- Release each tensor's pages once it has been decoded ([a77e8931](https://github.com/swedishembedded/brain/commit/a77e8931f45c5161d0b863b5fbf9c82a77c2ed67))

- Dequantize a GGUF tensor in bounded chunks, like every other source ([6a408a56](https://github.com/swedishembedded/brain/commit/6a408a568a5813b59180cf045a887f1371e60bdb))

- --strength is one sampler at every setting, not two ([62e5886d](https://github.com/swedishembedded/brain/commit/62e5886db2eee397aa21a8d739f217453cbdff6e))

- `--out photo.jpg` writes a JPEG, and an extension nobody supports is an error ([cf85f21b](https://github.com/swedishembedded/brain/commit/cf85f21bc67579f51d0b648bda59aa489dc832de))

- Say the memory invariant in words, not multipliers ([9dd9a743](https://github.com/swedishembedded/brain/commit/9dd9a74339b194281a4f0a25fda853785a470efa))

- Flux2 roadmap: the bottom rung of the --strength ladder

`--strength 0` returning the source was gated in the unit tests and asserted
in prose; it was never rendered on the real weights. It is now: edge corr
0.985, MAD 3.4 against the source photograph, i.e. the VAE round trip, which
is the floor for anything that edits in latent space. Rendered from both
trees, and the two agree to every digit recorded - once the trajectory has
almost no distance to travel the shape of the schedule stops mattering, which
is the reason the discontinuity showed at the TOP of the dial and nowhere
else.

Measured on the same 2x Tesla P40 placement, klein-9b Q8_0 int8, image-02,
seed 7, 12 steps, LoRA 1.0. ([168457c5](https://github.com/swedishembedded/brain/commit/168457c561370ce9fa57a713a281b9dc016b5ef0))

- A caption can be a paragraph, not a line ([02f2e4fe](https://github.com/swedishembedded/brain/commit/02f2e4fef9b50456b40da4adc627eb74b434dad8))

- Labeling a dataset is a capability, not a script ([c1c3e113](https://github.com/swedishembedded/brain/commit/c1c3e113a161a9a213f390cd25a4a79449d21d5e))

- The trainer the CLI could not reach ([79354e8c](https://github.com/swedishembedded/brain/commit/79354e8cfd16ff37dd8c9a0f0211a1a2577bd02f))

- Route to the labeler and the flux2 finetune verb ([994b998b](https://github.com/swedishembedded/brain/commit/994b998b69951e1651dbbbc79f730fa809b3f3c6))

- --ref-size, so an unprepared photograph is a valid reference ([71055ff0](https://github.com/swedishembedded/brain/commit/71055ff076784b663f562c9e15704a039eb192cd))

- One folder in, that person in the target pose out ([00f6b74e](https://github.com/swedishembedded/brain/commit/00f6b74e7e1a3b81a99f2a2e5bc3b81c7a2c5436))

- Parse captions.yaml with a YAML parser, not with our own ([20e749e3](https://github.com/swedishembedded/brain/commit/20e749e30355218e7c46bdcb6669ce789c1b7120))

- Hold the training weights once, not twice ([a71b4281](https://github.com/swedishembedded/brain/commit/a71b4281cb57c4aee1fb0233d06bfaff71537b32))

- The LoRA trainer that runs on the card ([6276bdb6](https://github.com/swedishembedded/brain/commit/6276bdb67dc6487609a0530af11fb44183af4e6d))

- `finetune --trainer device|host`, said out loud ([0c879bd9](https://github.com/swedishembedded/brain/commit/0c879bd9ee714c350c90dae6bd8183209c158cbf))

- Flux2 roadmap: what a training step costs, and what it costs it

`tests/dev_step_time.rs` is the step-cost harness the optimisation pass was
measured with (warm-up discarded, best-of-N with N printed, nothing polling
nvidia-smi while the clock runs) and `.agents/roadmap/flux2.md` records the
before/after and the per-kernel profile ranked by share of the step, so the
next pass has a baseline instead of an anecdote. On one Tesla P40, klein-4b at
512 px, rank 16: about 98 s per step became 11.74 s, and a 1500-step run went
from roughly 41 hours to 4.9. 79.9% of the step is now in the three
register-tiled GEMMs; the next two targets, with their reasons, are written
down.

`tests/int8_base_grads.rs` asks the open question with a measurement rather
than an assumption: an int8 frozen base would put klein-9b on one card instead
of two, so it takes one REAL double block out of the released checkpoint,
round-trips it through brain's own per-output-row int8 grid, and reports what
that does to the adapter gradients. It reports rather than asserts a
threshold, and it says which term of the error it does NOT cover (the
per-token activation quantization a real int8 kernel would add). ([47020643](https://github.com/swedishembedded/brain/commit/47020643d0600ba7248189d8a263adb68bfa8aaf))

- Flux2 roadmap: the int8 frozen base, measured on real klein-9b weights

One real double block out of the released Q8_0 checkpoint, round-tripped
through brain's own per-output-row int8 grid, backward run on both bases: the
weights move by about 1% rel_l2 and the adapter gradients come back at worst
cosine 0.999530 / rel_l2 3.08e-2, with dx at 0.999938 / 1.11e-2. A 1.8-degree
direction error and 3% of magnitude is not what decides whether an adapter
trains, so the weight-quantization term is affordable.

What that does not settle is written down next to it: the per-token activation
quantization a real dp4a kernel adds is unmeasured, and the saving is 2x
rather than 4x because `dx = dy.W` contracts over W's row axis, where a
per-row scale will not factor out - so a dp4a backward needs a second,
transposed int8 copy. 18 GB still does not leave room on a 24 GiB card, and
the two-card fp32 split already gets there with no fidelity question. ([be8c7697](https://github.com/swedishembedded/brain/commit/be8c769786ddf0c9d106412b63161fb6f230861c))

- Flux2 roadmap: the int8 memory arithmetic, the right way round

An earlier draft concluded that two int8 copies of klein-9b "still does not
fit one 24 GiB card". That is wrong: 9.05 G parameters are about 18.1 GB as
two int8 copies, and next to roughly 2.4 GB of activations at 1536 joint
tokens the total is about 20.5 GB, which fits with a few GB to spare. int8
collapses the two-card split back to one card.

It also under-weighted the real prize. The released klein-9b DiT is Q8_0 and
both trainers reach it through `read_dit_tensors`, which materialises the
whole model as host fp32 before a single step runs - that expansion, not the
training, is what puts the first step an hour away. `DitWeights::try_i8_rect`
already goes Q8_0 to packed int8 with no fp32 intermediate for inference; a
trainer taking its frozen base the same way inherits it.

What is missing to build it is written down next to the claim: the transposed
copy the backward needs cannot reuse `try_i8_rect`'s block-aligned fast path
(Q8_0 blocks run along rows), and the per-token activation term is still
unmeasured. ([06a1415d](https://github.com/swedishembedded/brain/commit/06a1415d614ee23f5c2539855dfa6d67140908fd))

- Roofline the training step, then close the gaps it names ([c80b8459](https://github.com/swedishembedded/brain/commit/c80b84593f59ec2b6e66aa4ea1925e13086eac3a))

- Flux2 roadmap: the after-profile, and the size of what is left

Records the step cost this pass reached (10.53 s, 4.39 h for 1500 steps, 28.4%
of the 2.99 s roofline floor) and what each change bought, then prices the
three remaining items so the next pass can choose instead of guess:

* the two register-tiled GEMMs are 85.8% of the step at 37.5% / 38.5% of the
  fp32 roof - about 2.9 s if they reached 60%. The diagnosis (an 8x8
  accumulator tile caps Pascal at ~3 workgroups per SM, which is about 37%
  occupancy) is written down as a hypothesis to test, not a finding, and the
  blast radius is named: that kernel is shared by every model here.
* the recompute is 32.5% of the arithmetic; stashing the eight per-block
  tensors that cost a GEMM to recreate needs 5.7 GB against roughly 5 GB
  spare, so it is a partial win on klein-4b and needs per-block buffers.
* `softmax_k_dx` and `rmsnorm_dx_eps` are the last one-thread-per-row kernels,
  0% and 4.6% of the DRAM roof, ~0.55 s between them, and each needs new WGSL.

Also fixes the one clippy warning in `int8_base_grads.rs` (a wrapped sentence
whose continuation dash started a line, which reads as an unindented markdown
list item). ([5bf268c2](https://github.com/swedishembedded/brain/commit/5bf268c220e4bd52a80cb0400ec890c960c9c02a))

- Price pack_qkv, sigmoid and chan_place ([6c0b1c46](https://github.com/swedishembedded/brain/commit/6c0b1c468286ec07b8ba8bffa9eb499aa880bf5c))

- Cost::Recording, so a graph can be priced without running it ([a0390b58](https://github.com/swedishembedded/brain/commit/a0390b58fcc261bf8971ca9f04835763f873c089))

- Three shell examples for the tasks people actually arrive with ([7f1459ca](https://github.com/swedishembedded/brain/commit/7f1459ca6f037b842de1604d43790ea51dc6c0a6))

- What a user may type, and where models live ([b0dd9654](https://github.com/swedishembedded/brain/commit/b0dd96545a36d6ce06415ba8d88b7e90f8073773))

- Price a whole image or video generation, offline, by stage ([d405aa7c](https://github.com/swedishembedded/brain/commit/d405aa7ccd90ecf66cb383b8ecf04bb0c7c5dbd1))

- Brain pull, said out loud ([cf8a25e0](https://github.com/swedishembedded/brain/commit/cf8a25e0a1f95d52bb8b0e0f7d977d34f7b8ab34))

- Measure the predictor against the machine, and document it ([3a37d971](https://github.com/swedishembedded/brain/commit/3a37d971f5070d17d9758893943c40125e4e921e))

- Gate the trainer at each variant's real widths, not just tiny dims ([6caeaf29](https://github.com/swedishembedded/brain/commit/6caeaf295ce518cc00a3d8e5c3533e1f5970daf7))

- Price the LTX 3D VAE decode, as the tiled decode that really runs ([5ea9df09](https://github.com/swedishembedded/brain/commit/5ea9df098465e74e4066247ff2d0acb20143eea7))

- An aggregate is not one kernel, and cannot be graded like one ([2c4daefd](https://github.com/swedishembedded/brain/commit/2c4daefd781d6ba768fed49640d2180fb000dd5f))

- Say what the ltxv measurement can and cannot attribute ([48a00bd8](https://github.com/swedishembedded/brain/commit/48a00bd8402134a0df8af0923301389d6c85dbdb))

- Drop the bare hardware ratio from the mixed-roof docs ([d838f6a1](https://github.com/swedishembedded/brain/commit/d838f6a1b0473c19f15972474071529bbac18173))

- One capacity-aware seam every model inherits, instead of card 0 ([0703d325](https://github.com/swedishembedded/brain/commit/0703d3257f37de3ab35699785a3e95471f24615d))

- Declare the pipeline's parts, let the engine place them ([36c06391](https://github.com/swedishembedded/brain/commit/36c0639142809bb0e8215f770a2d9b56139614df))

- Price the decode from the graph, and stop building a device per image ([7da4f718](https://github.com/swedishembedded/brain/commit/7da4f718846fc94472f8c314dc877af9837653af))

- Run a trained adapter, and make the portrait mask opt-in ([cfb7854b](https://github.com/swedishembedded/brain/commit/cfb7854bf051f368647c8e85f1ca7aabdea89287))

- Port the video memory bank, so a mask follows the moving object ([6bcc0e79](https://github.com/swedishembedded/brain/commit/6bcc0e79fd3d9dd99d5e6a12c9a8c504106368c3))

- Brain sam2 track - a clip and a click in, a mask sequence out ([00cb9c26](https://github.com/swedishembedded/brain/commit/00cb9c26ca4897cb07e4d810d021b5211b62f33c))

- Port IC-LoRA reference-video conditioning, and say what it cannot do ([cc817f86](https://github.com/swedishembedded/brain/commit/cc817f860890799c00777daac62fd23cc01594b3))

- Build LTX-2.3 too, because the release really is a config ([4db8e2eb](https://github.com/swedishembedded/brain/commit/4db8e2eb8a7759afa43e144f5f965dce0aad7d70))

- Masked conditioning, so a character swap keeps the set bit-exactly ([8cdbca2d](https://github.com/swedishembedded/brain/commit/8cdbca2df9a7d4ad8ca5b8d970b1a64427503796))

- Record Phase 39 - masked conditioning, and the resample-gate lesson ([58108d15](https://github.com/swedishembedded/brain/commit/58108d15c747dd0d81ff033b88929f3aff63e934))

- Stream the block gradients, and let a run resume where it stopped ([1efcf226](https://github.com/swedishembedded/brain/commit/1efcf226e5f7817a8c4dee3dfa5c92c110d98a38))

- Default the portrait script to the variant that is on the box ([797a61c2](https://github.com/swedishembedded/brain/commit/797a61c277b62b89539729a52a2c26a2f3ec2baf))

- Make --lora-scale reach a brain-trained adapter, not just a foreign one ([5557ea48](https://github.com/swedishembedded/brain/commit/5557ea481fe1a4caac89d099eec43ae479e925fa))

- Let the caller say what a trigger phrase NAMES, instead of assuming a style ([e7511e3f](https://github.com/swedishembedded/brain/commit/e7511e3f5f41fa07d7d8ac1c9228752d3afe7b3e))

- Train a face onto klein-4b, and grade it with a number ([f66b7920](https://github.com/swedishembedded/brain/commit/f66b7920168d40e82097c83f51adbbd309f69c7a))

- Size reference images in brain, instead of making the caller do it ([069304f2](https://github.com/swedishembedded/brain/commit/069304f28443bf5cbedff6ebdc70b558cab19ce0))

- Take the output size from the reference that IS the canvas ([52544385](https://github.com/swedishembedded/brain/commit/525443859bb082be94520f241e2e713fcf3f92f8))

- Face_swap.sh, and stop resampling images the CLI can size itself ([360553e6](https://github.com/swedishembedded/brain/commit/360553e61164a6286f9bfbc25a9daf52554fcf10))

- Let an arch keep both table rows when its handler forwards the rest ([94121f93](https://github.com/swedishembedded/brain/commit/94121f93867cf00076e7b4372b173f236a8127b1))

- Arch, supir, llava: reserve the two architecture ids for SUPIR restoration

Registers supir and llava in crates/arch's canonical table before any of
their code exists - this fixes the crate directory names, package names,
CLI words and docs filenames in one place up front. Both crates are
placeholders (module docs only, no implementation yet); llava is brought
in as supir's optional captioner, never a hard dependency. supir ships no
default_ref/auto-fetch - the released weights are under a non-commercial
license with no official HF repo.

.agents/roadmap/{supir,llava}.md carry the full architecture spec (verified
against upstream source and the real checkpoint headers) and the staged
implementation plan, so neither has to be re-derived later. ([efc44106](https://github.com/swedishembedded/brain/commit/efc44106fe9f9dfae012fefc3728d129d86fbb57))

- Sdxlunet, controlnet: factor the duplicated SDXL sampling loop

sdxlunet::pipeline::Sdxl::generate and controlnet::caps::Controlled::
generate were near-identical copies (byte-identical gaussian/encode_with,
same CLIP-tower plumbing, same Euler loop, same CFG expression, same
VAE-decode tail) because Sdxl built its Unet with the plain constructor
and had no seam for a per-step residual. A SUPIR pipeline would have been
a third copy, and SUPIR's loop differs from both anyway (a per-step LQ
latent, its own scheduler), so this had to happen before that port
started, not after.

New: sdxlunet::sampler (a Denoiser trait - one forward per step - plus
the shared seed/CFG/scheduler loop) and sdxlunet::textenc (the dual CLIP
conditioning). Sdxl and Controlled are now a ~15-line Denoiser impl each
over the shared loop. controlnet no longer depends on brain-clip/
brain-diffusion directly.

model::hostmath::gaussian is the Box-Muller helper both crates duplicated,
moved to the one place host math belongs and pointed at from flux1's
identical third copy too. It is deliberately a distinct function from the
existing hostmath::randn (Z-Image/FLUX.2's own noise source) and
data::rng::Rng::next_gaussian (minimaxmusic3's) - unifying those would
silently change what an existing --seed reproduces for the model family
that already uses it, which is a behaviour change, not a refactor.

No behaviour change: controlnet's real-weight parity tests
(sdxl_controlnet_residuals_match_diffusers) and sdxlunet's
(sdxl_unet_forward_matches_diffusers) pass unchanged. ([0f180937](https://github.com/swedishembedded/brain/commit/0f1809375f00a7fa461990605e12aee9c9d233af))

- Dump real SUPIR reference activations for the parity ladder ([22e65816](https://github.com/swedishembedded/brain/commit/22e65816a49aa3f8e86c129fc41df44540e2f0d2))

- Debt-first prerequisites for the SUPIR seams ([9053d6b3](https://github.com/swedishembedded/brain/commit/9053d6b37d888454b986c2d7b4d8e34e98b59a9f))

- The SkipFuse seam that replaces the up path's skip concat ([cff35b74](https://github.com/swedishembedded/brain/commit/cff35b745b5efb67ee3fe06e9e6e0c71495f04d8))

- Op::Mix, the ZeroSFT/ZeroCrossAttn lerp, reusing edm_mix.wgsl ([93bc2fe5](https://github.com/swedishembedded/brain/commit/93bc2fe57800d392e93ae4dd3c2d99021c0c0817))

- Restore, SUPIR's RestoreEDMSampler scalar math ([a96c52ec](https://github.com/swedishembedded/brain/commit/a96c52ec237f4661f9e96da7a83116348d865dae))

- Imaging, kernels: a blended TilePlan variant, for SUPIR's tiled sampler

imaging::tiling's own module doc already pre-authorised this: halo
tiling gives every tile a disjoint core and no blend, which is right
for a model that only degrades near a tile's border - wrong for a
diffusion tail, where each tile's CONTENT differs, not just its edge
error. SUPIR's tiled sampler and tiled VAE both need real blended
overlap.

The weight math is vae::tiling3d's separable trapezoidal construction
(W(h,w) = Wh*Ww), dropped to 2D. One new kernel, blend_accumulate
(acc[c,h,w] += x[c,h,w] * weight[h,w]) - checked against
.agents/rules/kernels.md's existing-kernel survey first; nothing already
expressed a per-pixel weighted accumulate into a live canvas. Division by
the summed weight is a separate pass, so the kernel itself stays a plain
accumulate with no reduction.

Gated: a tiled identity transform (no actual model in the loop) round-trips
to fp32 round-off, and a single-tile plan blends to the identity - the
cheapest possible partition-of-unity check. ([f2bbae29](https://github.com/swedishembedded/brain/commit/f2bbae2983f15df05ef4edcbd4317addc4079767))

- Widen record_into and add Rec::set_fuse for external callers ([6c2108c3](https://github.com/swedishembedded/brain/commit/6c2108c38dee70c304306627e7ad0e7894a6cd06))

- Implement config/import/trunk/adaptors/model (weight-free gated) ([83ca8cb0](https://github.com/swedishembedded/brain/commit/83ca8cb046a5f6088e156178e6d570cd3fe38c5b))

- Real-checkpoint import coverage + restore schedule parity (step 5, partial) ([57664c99](https://github.com/swedishembedded/brain/commit/57664c997f4fb646e1b1a98de60d649400846ce0))

- Update Cargo.lock for the brain-diffusion dev-dependency ([fd7d1307](https://github.com/swedishembedded/brain/commit/fd7d1307553923bb299c655b852c0fab54fed759))

- *(supir)* Thread an --s-churn override through the reference dumper ([cd17ff09](https://github.com/swedishembedded/brain/commit/cd17ff094d5fab84792b3a230e6203b7c265ec1c))

- Restore::DiscreteDenoiserWithControl::index, snap's missing half ([ad9c05c2](https://github.com/swedishembedded/brain/commit/ad9c05c2e0a0e1791925a32515ef70a973095446))

- Import a single-file CompVis/LDM SDXL checkpoint (load_ldm) ([c33fc131](https://github.com/swedishembedded/brain/commit/c33fc131e0d4d46ff7eabc5d101627c2c615ce16))

- Tap every trunk/adaptor stage, and real-checkpoint forward parity ([a3e5dc24](https://github.com/swedishembedded/brain/commit/a3e5dc2417b6793d5308b657e8344bd61861ad67))

- Sdxlunet, supir: int8 host-memory quantization, honestly gated

sdxlunet::int8 (group-wise, QUANT_GROUP=32) and supir::int8 pack eligible
weights to int8 in host RAM, verified against the real checkpoints: sdxlunet
alone drops 8.90 GB fp32 -> 2.23 GB packed (cosine 0.999990698, rel_l2
4.315e-3); the combined SUPIR trunk+adaptors+backbone drops 15.60 GB ->
5.62 GB host-resident.

That host-side win does not close the device-memory ceiling found in Phase
3: vae::blocks::Builder::set_packed dequantizes each packed tensor to fp32
at upload, so device-resident buffers are still fp32-sized. On this box's
Intel iGPU (2047 MiB per-buffer cap, no discrete card), recording the full
SUPIR graph still hits a wgpu Out-of-Memory - reproduced with per-tap
buffer pinning both on and off, ruling that out as the cause. Genuine
device-side int8 storage (a dequantizing GEMM, the shape crates/flux1 and
crates/s3dit already have) is real, scoped follow-up work, not done here.

supir_full_forward_int8_fits_this_machine and its taps-off sibling are
gated behind BRAIN_SUPIR_ALLOW_FULL_MEMORY=1, matching the fp32 sibling
test, and skip themselves with the measured numbers rather than claim a
false pass. .agents/roadmap/supir.md records the gap and reflects actual
Phase 0-4 progress instead of the stale "everything unstarted" placeholder. ([5c7eef7e](https://github.com/swedishembedded/brain/commit/5c7eef7ed6069f0cf12dd3ddab7f4f8654105027))

- Training - trunk+adaptors backward, gradcheck, LoRA, adaptor-only/full-backbone finetune ([438c1b1a](https://github.com/swedishembedded/brain/commit/438c1b1af6c6e80e0d9a7bea6b8d5845fe451029))

- Clip, qwen3: LLaVA-1.5 config presets (clip_l336, llama2_13b)

ClipVisionConfig::clip_l336() - the openai/clip-vit-large-patch14-336
vision tower (24x1024, 16 heads, MLP 4096, patch 14, 577 positions,
quick-GELU), reusing the deepseek_ocr() topology unchanged since the
two towers differ only in image_size/n_positions.

QwenConfig::llama2_13b() - Vicuna-1.5-13B's decoder half, a LLaMA-2-13B
fine-tune with no architecture changes (40 layers, d_model 5120, 40
heads, plain MHA via n_kv_heads == n_heads, d_ff 13824, rope_theta
10000, rms_eps 1e-5, untied lm_head). Verified against the real
meta-llama/Llama-2-13b-hf config.json (mirrored at
NousResearch/Llama-2-13b-hf) rather than trusted from the port plan.

Both are config-preset additions with no new capability; first step of
Phase 6 (crates/llava) per .agents/roadmap/llava.md. ([0d2c6500](https://github.com/swedishembedded/brain/commit/0d2c65008f2fb7714fff42f72833d00ff01909be))

- Llama_bpe - LLaMA-2/Vicuna SentencePiece byte-fallback BPE ([a55114f7](https://github.com/swedishembedded/brain/commit/a55114f71038ba9ad11de3c9dfc3bcdcb44b6376))

- The crate - vision->projector->decoder splice, template, INT8, serving ([4d31e9d1](https://github.com/swedishembedded/brain/commit/4d31e9d1700955610d0c39336ce4b3c1b077a98e))

- Supir, imaging, npu: restoration pipeline, served restore action, ZeroCrossAttn NPU export

crates/supir/src/pipeline.rs closes the gap the crate's own doc had left
open: the full restoration loop (dual encode - denoise_encoder's CompVis
weights renamed to the diffusers keys vae::VaeEncoder reads, merged with
the frozen backbone's own quant_conv - dual-CLIP conditioning reused
unmodified from sdxlunet::textenc, RestoreEDMSampler driven directly off
diffusion::restore's primitives, CFG combined in eps-space) and colour fix
(a new imaging::colorfix module: wavelet_reconstruction, the real 5-level
a-trous decomposition upstream's own default uses, plus adain for its
other supported mode).

crates/supir/src/caps.rs is the served restore action: capability::blob
image I/O, cancellable per denoise step (inv.cancel, the wan::caps
contract), and optional LLaVA auto-captioning through a capability::Registry
the caller supplies - this crate links no VLM.

crates/npu/src/supir_topology.rs + supir_export.rs export ZeroCrossAttn
(the one SUPIR adaptor with linear projections) through the shared
topo::linear_quant emitter, structurally tested. ZeroSFT and the 1.24B
GLVControl trunk have no export path yet - no cross-attention UNet has
ever been exported from this tree, so there is no existing block walk to
adapt - recorded honestly in the roadmap rather than implied.

All weight-free and gated; a real end-to-end run is expected to hit the
device-memory ceiling crates/supir/tests/parity.rs already documents. ([23915066](https://github.com/swedishembedded/brain/commit/2391506609b1146be0dfb5d4ead82c8e1c119bd3))

- Cli, catalog, imgpipe, docs: serve SUPIR (CLI verb, D-Bus, GGUF stub, imgpipe stage)

crates/cli/src/resident_supir.rs is the residency adapter over
supir::caps::Session, run_batch serial for the stated reason
resident_sdxl.rs/resident_controlnet.rs give: every restore call is its
own multi-step sample. crates/catalog registers brain/supir (with an
LLaVA-carrying capability::Registry for the caption auto-fill, the same
"registry supplied by the caller" precedent crates/imgpipe's own
PipelineProvider set); crates/cli's own catalog.rs patches the residency
adapter back in and adds a cross-check that supir::caps::LLAVA_MODEL still
names the real llava::caps::MODEL. One ARCH_TO_MODEL row makes `brain
supir restore ...` work - sdxlunet/controlnet ship no such shortcut at
all, so this is one line ahead of that precedent, not a new supir_cli.rs.

crates/cli/src/gguf_import.rs registers supir::import::GGUF_ARCHITECTURE
("sdxl", a borrowed spelling - the frozen backbone genuinely is SDXL, the
same reasoning s3dit used for "lumina2") as a second documented
ambiguous-tag exception, with a stub import_gguf that states plainly no
real file has ever been observed. D-Bus Run needed no new code - it
dispatches generically over the residency Executor once a model is
registered.

crates/imgpipe gets a new Stage::SupirRestore variant (not
Stage::Restore{w}, whose fidelity dial has no SUPIR meaning): a second
size-changing tail alongside Stage::Upscale, mutually exclusive with it
since this crate defines no combined order for two tails that each change
the resolution.

Docs: supir.md rewritten from its placeholder, index.md/README.md move
supir and llava out of "reserved, not started" into their real tables,
imgpipe.md documents the new stage, examples/restore gets a worked SUPIR
script alongside restore_face.py's. Roadmap and lessons.md updated with
this phase's honest scope: NPU export covers ZeroCrossAttn only, the
per-step control-scale ramp and tiled sampling are not wired into the
pipeline, and a real end-to-end run needs more device memory than this
port's own hardware has. ([1cb9d63d](https://github.com/swedishembedded/brain/commit/1cb9d63d9701995b72d7fe9869f60f2c6daab369))

- Flux2, gradcheck, ltxv, sam2, sdxlunet, supir, vae: mark reviewed perf numbers

check-no-perf-numbers flags a bare number next to a performance
unit/claim unless a human has reviewed it with a `perf-number:`
comment. Thirteen hits across these crates were unreviewed: mostly
architecture/config constants the gate's context-gating cannot tell
apart from a measured claim (int8's 4x byte-width ratio, a VAE's
32x32/8x downsample factors, a checkpoint's 128 latent channels, a
tolerance-headroom multiplier), plus two genuine dated hardware
measurements (flux2's roofline baseline, supir's OOM investigation)
that are legitimately reviewed exceptions, not live promises.

supir::finetune's one true drift risk - a specific loss-reduction
percentage from a single real training run, cited only as narrative
rationale for an unrelated ratio-based assertion - is reworded to
drop the bare number instead, pointing at `--nocapture` to see the
current trajectory on your own hardware. ([6758b197](https://github.com/swedishembedded/brain/commit/6758b197692b6633b8beca103c8ca31399f845a2))

- Add fetch/label/pull to check-arch-names' infra-verb list ([db280a7f](https://github.com/swedishembedded/brain/commit/db280a7f38ac747c48542413ab006b49828b43c1))

- *(ltxv, sam2)* Add the missing source block to three dumpers ([0dc5dfff](https://github.com/swedishembedded/brain/commit/0dc5dfff1f496edee4df23f79a77b2da42bc7705))

- Read a component that ships as a file, not only as a directory ([1a94acec](https://github.com/swedishembedded/brain/commit/1a94acec53f1e12b7a8fd2a871eec03ef929a9f5))

- --text-encoder, so the encoder can be swapped per run ([21c5a250](https://github.com/swedishembedded/brain/commit/21c5a250ae28fc1a719ef3ba02b72a7d587b1772))

- Make the adapter and the encoder flags, not just env vars ([5e955117](https://github.com/swedishembedded/brain/commit/5e955117e6ca845bc337710113e9654f0b5c726b))

- Name one file, or let a GGUF repo name one quantization ([7dcdafd9](https://github.com/swedishembedded/brain/commit/7dcdafd93471dc496b24ac4f8cd555d78854e75c))

- A missing mask or adapter warns, instead of ending the run ([4f99de1f](https://github.com/swedishembedded/brain/commit/4f99de1f1aa54722ea6d38ee84905633e225c5a2))

- List an image that could not be captioned, with an empty caption ([51d83cb0](https://github.com/swedishembedded/brain/commit/51d83cb0be7ed5d2148578776f78729048a71b53))

- Say that resume is per file, not per byte ([45a98fac](https://github.com/swedishembedded/brain/commit/45a98facd13219636003f4e2a7373eb442ffe781))

- One seam answers which model a GGUF is ([1f6825be](https://github.com/swedishembedded/brain/commit/1f6825be118a44482df5e168dec1899555d170ae))

- Load a two-file GGUF checkpoint ([ee2d5ae3](https://github.com/swedishembedded/brain/commit/ee2d5ae39c1c7092338078bceb5b3a1e804516c3))

- One architecture table, with a direct-load column ([b90bdc2c](https://github.com/swedishembedded/brain/commit/b90bdc2c83c14dbee3f669ac5b3992a7797553d9))

- Pick the captioner from the checkpoint's own architecture ([161340f6](https://github.com/swedishembedded/brain/commit/161340f6a7b81e4f9aa8a76d50c13eee1a68abdb))

- One M-RoPE decode step that leaves its result on the device ([9f7da65c](https://github.com/swedishembedded/brain/commit/9f7da65c2ed4bece263bbc839c37cbe80e15875b))

- Dispatch the register-tiled GEMM where a model registers one ([f8b0f6e1](https://github.com/swedishembedded/brain/commit/f8b0f6e12ce8a8b5f30fe79562b2feb47af9fee0))

- Chunk the tower's attention, and give it the tiled GEMM ([4350812a](https://github.com/swedishembedded/brain/commit/4350812a64f62ab20c18e3dd0fb08623bb4c201b))

- Run the vision tower where the decoder runs ([acb65142](https://github.com/swedishembedded/brain/commit/acb6514298fbacc3c9e9e907eabdc1562488cef4))

- Apply the LM head on the device, and stop reading back prefill ([99e957ff](https://github.com/swedishembedded/brain/commit/99e957ffaafeb60fba4f38425acf5f0a17f4de3f))

- A bench that says where a caption's time goes ([a084f5c2](https://github.com/swedishembedded/brain/commit/a084f5c231da7e2c0efe264a646dfb4c58fee25d))

- Fuse the bidirectional attention where a model registers flash ([7e37a158](https://github.com/swedishembedded/brain/commit/7e37a15869873ca995b888148c2bf858d9ca0659))

- Hold the vision tower resident instead of rebuilding it per image ([449af5d2](https://github.com/swedishembedded/brain/commit/449af5d2d7b5ea0cb0651ff46c0eca376e940f2e))

- Stop packing int8 activations an fp32 model never reads ([2a6d8efb](https://github.com/swedishembedded/brain/commit/2a6d8efb204651e630aeb4a811392758b742bad4))

- --profile, so a resident model's kernel table is reachable ([7f932443](https://github.com/swedishembedded/brain/commit/7f932443e26a2cb16a5d707d66c278c2be08c55f))

- Charge prefill only for the weights a prefill step reads ([98cb4958](https://github.com/swedishembedded/brain/commit/98cb49589f958ed6e49185dd05c939b62a805147))

- Final qwen3vl numbers, the dead int8 packing, and three meter bugs ([1d68f45c](https://github.com/swedishembedded/brain/commit/1d68f45c3943b9d7fad876c91b842535eeb6ddaf))

- Give the int8 decode GEMV the register accumulators fp32 got ([8b6678d5](https://github.com/swedishembedded/brain/commit/8b6678d56449e736dceba4da907658916b189afd))

- An opt-in int8 decoder tier, gated as the lossy thing it is ([246ce63c](https://github.com/swedishembedded/brain/commit/246ce63c5e5a4a0ef5ed0b665c23b1d75cf3556a))

- --precision, and a compare mode that prices quality too ([4a2d57ed](https://github.com/swedishembedded/brain/commit/4a2d57ed850ad93edb3da97dbcf13f97622ac63c))

- Unwrap a prose dash that clippy reads as a markdown list ([71c543a7](https://github.com/swedishembedded/brain/commit/71c543a7be93cf8c760d6fcacb6d45c163c538f1))

- Optimisation pass - per-kernel profile, no low-hanging fruit found ([69e4e44a](https://github.com/swedishembedded/brain/commit/69e4e44aea15d70c6351819ce200727d53ebc322))


### Performance

- *(build)* Switch the release profile to thin LTO ([7e84ec16](https://github.com/swedishembedded/brain/commit/7e84ec169fbae53eea90659c53cbf849ccf54d8a))

- *(wan)* First optimization pass, 1.63x end to end ([fad075e2](https://github.com/swedishembedded/brain/commit/fad075e2393caf6c01519770505c038d9691684e))

- *(attn)* Coalesce cross-attention scores for the ViT and cross callers ([f451e547](https://github.com/swedishembedded/brain/commit/f451e5473e0d67c1aca0819c94026c85b38d348e))

- *(attn)* Migrate the last four models to coalesced cross-attention ([69a796ee](https://github.com/swedishembedded/brain/commit/69a796ee0fd1319b4be86c98dbe08a948bc14772))

- *(attn)* Register-tiled bidirectional flash attention, 1.98x ([a34a4ef9](https://github.com/swedishembedded/brain/commit/a34a4ef91fa8098a676d91fa0a295fd1e90e7e8b))

- *(wan)* Rewrite wan_bench onto the shared device-timed profiler ([3e3ec27c](https://github.com/swedishembedded/brain/commit/3e3ec27cf05abeb8e576f60ac0acd7b35ae04c0f))

- *(gpu-core)* Cost formulas for ltxv's 3D conv/attention/rope/elementwise kernels ([a4b153da](https://github.com/swedishembedded/brain/commit/a4b153dafcde2949acb761728c355876548cdf8d))

- *(ltxv)* Fix fp32 GEMM and cross-attention kernel-selector regressions ([7bf03d2e](https://github.com/swedishembedded/brain/commit/7bf03d2edbfe14b19ae57fb2471ca55a67e07b34))

- *(qwen35 import)* Parallelize+fix fp8 dequant and int8 pack (M15 follow-up) ([b52fced1](https://github.com/swedishembedded/brain/commit/b52fced19e6c7efd504299113ce425d1d6dadb81))

- *(ltxv)* Parallelize the host linear() behind ada_layer_norm_single ([5ca1c006](https://github.com/swedishembedded/brain/commit/5ca1c006c7596bb4d5921e28aa62cec2c653d73d))

- *(ltxv)* Host-side per-generation block-weight cache ([1279b295](https://github.com/swedishembedded/brain/commit/1279b2952aa755c2b19d573b645370b2031fa700))

- *(model)* Row-parallel int8/int4 weight quantization ([8ccba57f](https://github.com/swedishembedded/brain/commit/8ccba57f8efc87ba0af9787f45253b734a09939f))

- *(checkpoint)* Block-parallel GGUF dequantization, and gate the block-to-output mapping ([310bc77f](https://github.com/swedishembedded/brain/commit/310bc77fb471f61260f2c93ec4ebd61d05036aec))

- *(checkpoint)* Map safetensors instead of slurping, and decode dtypes in parallel ([c4c62992](https://github.com/swedishembedded/brain/commit/c4c6299210d47065a7f4723a9195b9a2f3d9a828))

- *(ltxv)* Cache the embeddings-connector routing for a generation ([4e74cf3d](https://github.com/swedishembedded/brain/commit/4e74cf3d61be54fcb1e628e744c48443df098be7))

- *(ltxv)* Stream the text encoder from a quantized GGUF, and cache its output ([8da5c89d](https://github.com/swedishembedded/brain/commit/8da5c89d0d7296bdb4465ee1fe7b87040c6b4bcd))


### Refactor

- *(dit)* Hoist timestep embedding, patchify and adaLN-table into crates/dit ([7b78e8b2](https://github.com/swedishembedded/brain/commit/7b78e8b2d7a482e5d9d1670d8aa805e456b79473))

- Walk slices with enumerate() where the loop steps with them ([7f742690](https://github.com/swedishembedded/brain/commit/7f742690be284468b06b3c013b3548d7eac6fbf3))

- Walk two more slices with enumerate() where they step together ([f875e2a3](https://github.com/swedishembedded/brain/commit/f875e2a39bbc43e9a0659e3c96f2de1995370bd2))

- *(model)* Hoist GDN scratch allocators out of qwen35moe (M1) ([e24cdcec](https://github.com/swedishembedded/brain/commit/e24cdcec564ef70f84f781a19222b7551c09aa83))

- *(model)* Hoist linear_rows/linear_rows_bwd into hostmath, add dsilu ([ba0b0ae1](https://github.com/swedishembedded/brain/commit/ba0b0ae1aae79216874148d9d4324cc96434fe78))

- *(audio)* Hoist fold_weight_norm out of minimaxmusic3 into audio::conv ([266c6365](https://github.com/swedishembedded/brain/commit/266c636563a1d970c8e153735f8bdb84e6474665))


### Testing

- One skip helper and one parity report, across the workspace ([d2a34538](https://github.com/swedishembedded/brain/commit/d2a3453886537df3653701f89616d452957e97de))

- *(kronos)* Gate the CSV-to-forecast path, and read the checkpoint's own config ([b10f6600](https://github.com/swedishembedded/brain/commit/b10f6600d8fcc9087d68eafda2ff0b319f570e71))

- *(kronos)* Gate the whole ladder against upstream, on the shipped checkpoint ([5345cb24](https://github.com/swedishembedded/brain/commit/5345cb24622093f1a66cca4efbb5c739659dd5ec))

- *(kronos)* Assert calibration, not a point-accuracy claim the data cannot support ([2112bee2](https://github.com/swedishembedded/brain/commit/2112bee2acf778d33a34e9598707750f4dba96bd))

- Make a golden name the checkpoint it was dumped from ([c6d9b4f9](https://github.com/swedishembedded/brain/commit/c6d9b4f9b2393a8af0edd8208a0365a64a28d787))

- Actually run the parity for clip, pulid, instantid and sam2 ([f9d6d5b6](https://github.com/swedishembedded/brain/commit/f9d6d5b6eb857e660b004c0ab843d408e0760724))

- Route every remaining skip through a helper that says which kind it is ([1f7d7333](https://github.com/swedishembedded/brain/commit/1f7d7333ef313a1e3c50a9c30b72cc15de2bd921))

- *(ltxv)* Close gate holes so real-weight parity suites can actually fail ([e1df72fd](https://github.com/swedishembedded/brain/commit/e1df72fd05473c5edb318f045ca66ad43e2cc25d))

- *(kernels)* Validate matmul-family CPU native fast paths and cross-backend dispatch ([2f2d04e4](https://github.com/swedishembedded/brain/commit/2f2d04e4c30c52c3490cb35abda9e826a2f8d761))

- *(checkpoint)* Cover more than one decode group in the dequant gate ([12f9e3a1](https://github.com/swedishembedded/brain/commit/12f9e3a1659a662cb10c5a2c867659bf045480ce))

- *(ltxv)* Real-weight parity for the embeddings connector and the streamed forward path ([7ae61ed3](https://github.com/swedishembedded/brain/commit/7ae61ed350cefc4161df0ca413d70043ac4dc8c2))

- *(gemma4)* Real-weight, real-width port correctness parity ([9bbdc0ed](https://github.com/swedishembedded/brain/commit/9bbdc0ede16ca137f9d5af08cbe6ff9cae1b5c91))

- *(ltxv)* Gate that a generated clip actually moves, on a metric a frozen clip cannot pass ([c73cde5e](https://github.com/swedishembedded/brain/commit/c73cde5e204ca9acbb61a48494afd2e73225adb1))

- *(minimaxmusic3)* Vocoder overfits a single batch ([0905ac6c](https://github.com/swedishembedded/brain/commit/0905ac6cadef2fb0ddd026e0f82c48042ec123a0))

- *(minimaxmusic3)* Real short end-to-end generation gate ([2f42370a](https://github.com/swedishembedded/brain/commit/2f42370aee58da0d083cbfe0a9ace724e085f9c3))

- *(cosyvoice)* Golden reference dumper against real CosyVoice2-0.5B weights ([def79e4a](https://github.com/swedishembedded/brain/commit/def79e4ab622d9175b3f606c7289d184aea92740))

- *(gpu-core)* Price the other half of the staging trade, not just the half that improved ([33805bf3](https://github.com/swedishembedded/brain/commit/33805bf3b937b31dca35db9281883a4525199a11))

- *(gpu-core)* The staging trade stopped being a trade ([8bedddb0](https://github.com/swedishembedded/brain/commit/8bedddb02d2f166597f7b42c17da5cc448972ca1))

- *(cosyvoice)* Golden reference dumper against real CosyVoice3-0.5B-2512 weights ([177a35be](https://github.com/swedishembedded/brain/commit/177a35bee602f21f78350d446f7da4420352930e))

- *(model)* Serialise the probe tests, which now hold the GPU for seconds ([5e9f357a](https://github.com/swedishembedded/brain/commit/5e9f357a32652d85c2c41c019af1d9355f7788c4))


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


