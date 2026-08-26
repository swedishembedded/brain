# flux2 — roadmap

FLUX.2 Klein text-to-image / image-editing diffusion transformer (Qwen3 text
encoder + VAE + MMDiT), ported to brain's own kernels and serving stack.

## Not yet done

- [ ] Mixed-progress admission — a new request joining an already-running
      batch. The scheduler hands a lane a fixed job slice and marks the
      instance "running" for the whole call, so no job can join a batch
      already in flight; this needs an executor-level change (an admission
      channel a lane can drain between denoise steps), not a model-side one.
- [ ] Batched text encoder forward — the text encoder's graph is built for
      one sequence at a time; prompt batching is not implemented.
- [ ] Batched VAE decode — the VAE decoder runs per request; batching it is
      not implemented.
- [ ] The VAE decoder's graph and weights are rebuilt and re-uploaded on
      every decode call instead of being cached per output resolution.
- [ ] An implicit-GEMM convolution for the VAE decode, so the convolution's
      im2col gather is folded into the GEMM's tile load instead of
      materializing a separate scratch buffer.
- [ ] A fused, tiled causal+key-mask attention kernel for the text encoder's
      score computation — the current per-element masked kernel is far off
      peak throughput and neither a repack nor a coalescing-only fix closes
      the gap.
- [ ] Several smaller kernel-efficiency gaps identified but not yet closed:
      a workgroup-per-row LayerNorm kernel (RMSNorm and softmax already got
      this treatment), a workgroup-per-row reduction for the int8 path's
      row-max step, wider (vec4) shared-memory tile loads for the core GEMM
      kernel, and a second query row per thread in the flash-attention
      kernel.
- [ ] Performance sweeps do not record GPU thermal state (temperature,
      clock, throttle reason) per concurrency level, so a multi-level sweep
      on passively-cooled cards can be dominated by thermal throttling
      rather than the effect it's meant to measure.
- [ ] A GPU-backend test drives the model at dimensions that violate the
      device's minimum storage-buffer offset alignment and fails as a result;
      it needs either aligned test dimensions or an explicit alignment
      assertion so the failure is readable instead of a raw driver error.
- [ ] Klein-9B's cached-reference-attention variant is out of scope: it needs
      per-token modulation blending, which is incompatible with the current
      approach of folding modulation into the LayerNorm.
- [ ] The text encoder is imported whole and then truncated, so a large
      fraction of it is fetched, dequantised and validated only to be
      discarded. `pipeline.rs` builds `Shard { start: 0, end: deepest tap,
      embed: true, head: false }`, but that truncation happens at BUILD time,
      after the import has already insisted on the whole checkpoint:
      `checkpoint::safetensors::read_model_dir` reads every shard named in
      the index's `weight_map` unconditionally (it takes no parameter saying
      what the caller wants), and `qwen3::import::brain_init_from_hf`
      enforces two-way coverage against the full `param_list()` of a config
      whose `n_layers` is the untruncated count with `tie_embeddings: false`.
      For the Qwen3-8B encoder that means the layers past the deepest tap and
      the LM head - about 4.2 GB of 15.6 GB - are downloaded and checked, and
      the LM head is never read by any shard the pipeline builds. On a
      bandwidth-limited box that is most of an hour before the first image.
      The fix is a shard-aware import: derive the required `param_list()`
      from the `Shard` the caller will build, and let `read_model_dir` take
      the resulting name set so it can skip whole shard files. Note
      `hf_source`'s streaming path is NOT this fix - it lowers the ~32 GB
      host-RAM import peak but validates against the same full list, so it
      saves memory and not bytes. Keep the two-way coverage check: it is what
      catches a wrong checkpoint, and it must stay exact against whatever set
      is genuinely required.

The core GEMM kernel already runs near a structural throughput ceiling for
its current shared-memory tiling scheme, and batching the diffusion
transformer's forward pass has a small, bounded payoff because its GEMMs are
already near their row-count-independent plateau at a single sample; most of
a served image's latency lives in the (currently unbatched) text encoder and
VAE decode rather than in the transformer itself.
