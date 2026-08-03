Yes, but the obvious architecture depends on whether you need **one dense model** or are willing to design a **modular sparse model**.

## The central problem

You cannot take a dense Transformer, independently train arbitrary groups of weights on disconnected GPUs, and later concatenate them. The weights are coupled through forward activations and backward gradients. Ordinary tensor parallelism and pipeline parallelism therefore require communication during essentially every training step.

An 8 GB GPU can hold only the currently active fraction of a large model. It does not magically make the rest of the model independent.

## Best architecture: shard by independently testable modules

For a system made from many small, potentially unreliable GPUs, I would build:

**A relatively small shared backbone + a very large number of sparse experts/adapters.**

Each worker trains one of:

* one Mixture-of-Experts feed-forward expert;
* a group of small experts;
* one domain-specific LoRA adapter;
* one branch derived from a known base checkpoint.

Sparse MoE models are specifically designed so that only a subset of the parameters are active for each token. Switch Transformer demonstrated this architecture at up to trillion-parameter scale, while keeping per-token computation far below the total parameter count. ([arXiv][1])

This gives you clean integration boundaries:

```text
shared attention/backbone
        |
        +-- expert 0001
        +-- expert 0002
        +-- expert 0003
        ...
        +-- expert 100000
```

A worker can train expert `N` without being allowed to modify experts `N+1`, the tokenizer, the router or the shared backbone.

That is dramatically easier to audit than accepting an opaque update to the entire model.

### An even easier initial version

Freeze the base model and let workers train LoRA-style deltas. LoRA keeps the original model frozen and trains small low-rank update matrices, reducing the number of trainable parameters enormously. QLoRA additionally stores the frozen base in 4-bit form while backpropagating into adapters. ([arXiv][2])

The integration artifact becomes:

```text
base_model_hash
adapter_target_layers
adapter_rank
adapter_weights
training_manifest
validation_results
```

You can reject one bad adapter without corrupting the base model.

The limitation is important: this is primarily **adaptation or modular capability growth**, not a perfect substitute for end-to-end dense pretraining from random initialization.

---

## For a genuinely dense model

For dense pretraining, there are three realistic approaches.

### 1. Tightly coupled ZeRO/FSDP

Use:

* ZeRO-3 or PyTorch FSDP;
* tensor parallelism;
* pipeline parallelism;
* activation checkpointing;
* CPU and NVMe offload.

ZeRO-3 partitions parameters, gradients and optimizer state. DeepSpeed currently supports parameter and optimizer-state offload to CPU or NVMe, including tiled processing for models whose complete state cannot fit in memory. ([deepspeed.readthedocs.io][3])

This works well only when workers have a reasonably fast and reliable interconnect. Trying to do all-reduce or fine-grained tensor parallelism over arbitrary Internet-connected 8 GB machines would be brutal.

### 2. Local training followed by model-delta aggregation

A better low-bandwidth design is similar to DiLoCo:

1. Workers start from checkpoint (W_t).
2. Each worker performs many local optimization steps.
3. It returns a delta:

[
\Delta_i = W_i - W_t
]

4. A central optimizer aggregates the deltas.
5. A new checkpoint (W_{t+1}) is published.

DiLoCo reported comparable results to synchronous training in its experiments while communicating much less frequently. Its newer asynchronous successor, Decoupled DiLoCo, allows independent learners to submit parameter fragments to a central synchronizer without globally stalling on failed or slow workers. ([arXiv][4])

However, each learner still has to execute the full model, or a legitimate model-parallel partition of it. With an 8 GB GPU, this means aggressive CPU/NVMe streaming.

### 3. Resilient pipeline parallelism

SWARM parallelism was developed specifically for heterogeneous, unreliable and poorly connected devices. It constructs temporary pipelines and reroutes around failed workers. ([arXiv][5])

This is closer to “each GPU owns a few layers,” but integrity is harder because a malicious pipeline stage can alter activations or gradients flowing through every downstream stage. Recent work on stagewise integrity verification addresses this exact problem, but it remains a significantly harder security model than verifying independent adapters or experts. ([arXiv][6])

## My recommendation

For loosely connected 8 GB GPUs:

```text
Do not begin with arbitrary tensor or pipeline shards.

Use:
    immutable shared base
  + independently trained experts/adapters
  + periodic trusted backbone training
  + central router training
```

Then, once the system works, optionally add DiLoCo-style backbone updates from larger or more trusted learner groups.

---

# Making an untrusted training shard auditable

You need to distinguish two failure modes:

1. **Accidental errors**: unstable GPU, bad overclock, driver error, silent data corruption.
2. **Adversarial output**: worker fabricates results, skips steps or poisons the model.

Checksums alone only detect transmission or storage corruption. They do not prove that the computation was correct.

## Every work unit should be deterministic and immutable

Give the worker a signed manifest:

```yaml
job_id: ...
base_checkpoint_hash: ...
trainable_parameter_names: ...
dataset_merkle_root: ...
dataset_range: ...
tokenizer_hash: ...
container_image_hash: ...
source_code_commit: ...
optimizer: AdamW
optimizer_parameters: ...
random_seed: ...
step_start: ...
step_count: ...
precision_policy: ...
gradient_clipping: ...
checkpoint_interval: ...
```

The worker returns:

```yaml
delta_hash: ...
intermediate_checkpoint_hashes: ...
loss_curve: ...
gradient_norms: ...
update_norms: ...
tokens_processed: ...
hardware_info: ...
kernel_versions: ...
```

And the actual artifact should contain only the permitted parameter delta.

## Validation pipeline

### 1. Structural validation

Immediately reject updates containing:

* NaN or infinity;
* incorrect shapes;
* modifications to forbidden parameters;
* implausible update norms;
* extreme per-layer norm ratios;
* malformed or missing checkpoints;
* invalid dataset or base-model hashes.

### 2. Hidden replay challenges

Have the worker checkpoint every (N) steps.

After it commits its final result, randomly select one or more training intervals and require enough state to replay them:

```text
checkpoint at step 12,000
replay steps 12,001–12,050
compare produced checkpoint/delta
```

The verifier must choose the challenged interval **after** the worker commits, otherwise the worker can compute only the known audited steps.

GPU training is normally not bit-for-bit reproducible across arbitrary hardware because floating-point reductions and kernel scheduling can differ. One optimistic-verification approach demonstrated exact replay by controlling precision and rounding decisions, including experiments on GPT-2-sized models, but this remains specialized rather than a solved trillion-scale verification method. ([arXiv][7])

Practically, use either:

* identical hardware and deterministic kernels for exact comparison;
* quantized/rounded intermediate checkpoints;
* bounded numerical tolerances;
* comparison of resulting deltas and hidden validation behavior.

### 3. Redundant computation

The strongest practical mechanism is replication.

Assign some work units to two or three unrelated workers:

```text
worker A -> delta A
worker B -> delta B
worker C -> delta C
```

Accept if two results agree within the expected numerical tolerance.

You do not necessarily need to duplicate everything. Duplicate:

* all work from new or low-reputation workers;
* a random fraction of established workers;
* high-impact shards;
* suspicious outliers.

### 4. Canary minibatches

Insert hidden minibatches whose expected behavior is already known.

The worker does not know which batches are canaries. Check:

* loss before and after;
* gradient norm;
* selected gradient projections;
* update hash after controlled rounding.

A worker that simply invents plausible-looking loss numbers is unlikely to reproduce hidden gradient projections.

### 5. Random projections instead of full gradient comparison

A full gradient can be enormous. The verifier can choose random vectors (r_j) and compare:

[
s_j = r_j^T \Delta
]

or per-layer projections:

[
s_{j,l} = r_{j,l}^T \Delta_l
]

These are compact fingerprints of the update. The random vectors must be revealed only after the worker has committed its output.

This does not alone prove training correctness, but it makes post-hoc alteration and inconsistent reporting easier to detect.

### 6. Peer and robust aggregation

For several workers starting from the same checkpoint, compare:

* update norm;
* cosine similarity;
* coordinate distributions;
* validation-loss change;
* distance from the median update.

Robust aggregators such as trimmed means, coordinate medians and Krum-style methods are intended to tolerate Byzantine workers under certain assumptions. But they generally rely on an honest-enough majority, and research has demonstrated attacks against several supposedly robust aggregation mechanisms. Treat robust aggregation as another filter, not as the complete security system. ([arXiv][8])

### 7. Integration gates

Never merge directly into the production checkpoint.

Use:

```text
submitted
    -> structurally_valid
    -> replay_valid
    -> peer_valid
    -> hidden_eval_valid
    -> staging_model
    -> extended_evaluation
    -> accepted
```

For an expert or adapter, evaluate it both:

* in isolation;
* through the router;
* on its intended domain;
* on general capability and safety regression sets.

A failed shard is simply discarded.

---

# Can weights and optimizer updates live on disk?

Yes.

More precisely:

* the full weights can reside on NVMe;
* optimizer moments can reside on NVMe;
* gradients can be written to NVMe;
* only the active parameter tile needs to be in CPU RAM or GPU VRAM;
* the optimizer can sweep over parameter tiles sequentially.

DeepSpeed ZeRO-Infinity explicitly supports NVMe-backed model parameters and optimizer state. PyTorch FSDP supports CPU offloading, though direct NVMe-tier management is more characteristic of systems such as DeepSpeed. ([deepspeed.readthedocs.io][3])

A disk-backed optimizer loop can look like:

```python
for shard in parameter_shards:
    weight = read_weight(shard)
    master = read_master_weight(shard)
    grad = read_gradient(shard)
    m = read_first_moment(shard)
    v = read_second_moment(shard)

    master, m, v = adamw_update(master, grad, m, v)
    weight = cast_to_bf16(master)

    atomic_write(shard, weight, master, m, v)
```

In practice you would use:

* pinned CPU buffers;
* asynchronous prefetch;
* double or triple buffering;
* large sequential reads;
* aligned extents;
* separate disks or namespaces for weights and optimizer state;
* per-shard checksums;
* copy-on-write or generation-numbered checkpoints;
* atomic manifest replacement after all shard writes complete.

## But disk bandwidth becomes the wall

For Adam-style training, you may have approximately:

```text
BF16 model weight        2 bytes
FP32 master weight       4 bytes
gradient                 2–4 bytes
first moment             4 bytes
second moment            4 bytes
```

The optimizer must read and rewrite much of that state. A rough end-to-end I/O budget can easily be around **24–32 bytameter per optimizer update**, depending on representation and caching.

At an idealized 7 GB/s sequential NVMe rate:

* 100 billion parameters × 28 bytes ≈ 2.8 TB, or about **400 seconds** of raw I/O;
* 1 trillion parameters × 28 bytes ≈ 28 TB, or about **4,000 seconds**, roughly **67 minutes**.

That is before filesystem overhead, contention, forward/backward streaming and synchronization.

So disk-backed training is technically valid, but a single SSD does not make trillion-parameter training practical. You need to shard the disk traffic across many machines and many SSDs.

## Layer streaming

A full out-of-core iteration usually looks like:

```text
Forward:
    load layer 0 -> GPU
    compute
    save/checkpoint activation
    evict layer 0

    load layer 1 -> GPU
    compute
    ...

Backward:
    reload last layer
    recompute/load activation
    compute gradient
    write gradient shard to disk
    evict layer

Optimizer:
    stream weight + gradient + moments
    update tile
    write new generation
```

There is always some active tile in memory. “Entirely on disk” means the complete model state does not need to be resident simultaneously—not that the GPU can calculate directly against SSD blocks.

---

# A concrete architecture I would build

```text
                           Coordinator
                               |
                signed manifests + model roots
                               |
        ------------------------------------------------
        |              |              |                |
     worker A        worker B       worker C         worker D
     expert 17       expert 81      LoRA math       LoRA code
        |              |              |                |
        ---------------- artifact store ----------------
                               |
                    verification workers
              replay / redundancy / hidden eval
                               |
                         staging registry
                               |
                  router + integration training
                               |
                      accepted checkpoint
```

Each worker has:

```text
8 GB GPU
32–128 GB CPU RAM where possible
1–4 TB local NVMe
local immutable base-model cache
small active GPU tile
```

The coordinator stores:

```text
content-addressed model shards
content-addressed datasets
signed task manifests
worker reputation
redundancy assignments
hidden verification seeds
accepted/rejected artifact history
```

The fundamental design principle is:

> **Make every externally trained contribution small, replaceable, independently evaluable and reversible.**

That strongly favors experts and adapters over arbitrary slices of a dense model.

For a genuinely enormous model, the most viable combination would be:

```text
Sparse MoE architecture
+ expert-level sharding
+ quantized frozen/shared backbone on workers
+ NVMe layer streaming
+ local multi-step training
+ signed delta artifacts
+ random redundant execution
+ post-commit replay challenges
+ robust aggregation
+ trusted router/backbone integration
```

That does not eliminate the compute cost, but it turns a monolithic, unverifiable distributed training job into a set of auditable components.

[1]: https://arxiv.org/abs/2101.03961?utm_source=chatgpt.com "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
[2]: https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com "LoRA: Low-Rank Adaptation of Large Language Models"
[3]: https://deepspeed.readthedocs.io/en/latest/zero3.html?utm_source=chatgpt.com "ZeRO — DeepSpeed 0.19.3 documentation"
[4]: https://arxiv.org/abs/2311.08105?utm_source=chatgpt.com "DiLoCo: Distributed Low-Communication Training of Language Models"
[5]: https://arxiv.org/html/2301.11913?utm_source=chatgpt.com "SWARM Parallelism: Training Large Models Can Be ..."
[6]: https://arxiv.org/html/2603.03592v1?utm_source=chatgpt.com "Sentinel: Stagewise Integrity Verification for Pipeline ..."
[7]: https://arxiv.org/abs/2403.09603?utm_source=chatgpt.com "Optimistic Verifiable Training by Controlling Hardware Nondeterminism"
[8]: https://arxiv.org/abs/1703.02757?utm_source=chatgpt.com "[1703.02757] Byzantine-Tolerant Machine Learning"
## Complete runnable project

[Download the full project ZIP](sandbox:/mnt/data/sharded_moe_example.zip)

Individual files:

* [Main model and CLI — sharded_moe_lm.py](sandbox:/mnt/data/sharded_moe_example/sharded_moe_lm.py)
* [Complete usage guide](sandbox:/mnt/data/sharded_moe_example/README.md)
* [End-to-end sharding workflow](sandbox:/mnt/data/sharded_moe_example/example_workflow.sh)
* [Smoke test](sandbox:/mnt/data/sharded_moe_example/test_smoke.py)
* [Requirements](sandbox:/mnt/data/sharded_moe_example/requirements.txt)

### Implemented

* Actual sparse gather → expert execution → scatter dispatch. Unselected experts are not evaluated.
* Router softmax, top-k selection, and combine-weight renormalization after overflow dropping.
* Per-expert capacity limits.
* Load-balancing auxiliary loss and router z-loss.
* Optional noisy routing.
* RoPE causal attention with automatic SDPA or portable manual attention.
* RMSNorm, SwiGLU experts, residual Transformer blocks.
* FP32, FP16, and supported BF16 execution.
* Tiny Shakespeare downloading and character tokenization from Karpathy’s published corpus. ([GitHub][1])
* Synthetic structured corpus with a 64-token vocabulary.
* Memory-mapped `.npy` and raw `.bin` datasets.
* Training, evaluation, generation, checkpoint inspection, hardware diagnostics, and assembly commands.
* SHA-256 checkpoint verification.
* CPU expert offload for models larger than GPU VRAM.
* Gradient accumulation and AMP scaling. PyTorch documents FP16 autocast together with gradient scaling, which the implementation uses on CUDA. ([docs.pytorch.org][2])
* Portable causal SDPA with a manual fallback; PyTorch’s SDPA interface supports causal attention directly. ([docs.pytorch.org][3])

### Shard structure

Each expert ID is a vertical shard spanning every layer:

```text
expert_0003.pt
  blocks.0.moe.experts.3.*
  blocks.1.moe.experts.3.*
  blocks.2.moe.experts.3.*
  ...
```

A worker can therefore:

1. Load the immutable common checkpoint.
2. Freeze the backbone, routers, and all other experts.
3. Force routing through expert `N`.
4. Train only expert `N`.
5. Return only `expert_N.pt`.
6. Have the coordinator verify and overlay it.
7. Train only the routers after expert integration.

The final assembled checkpoint supports last-overlay-wins replacement for both expert shards and shared/router state. Shards carrying a different base-manifest identity are rejected.

### P40 support

The Tesla P40 is CUDA compute capability 6.1. ([NVIDIA Developer][4]) The code uses no Triton, FlashAttention, fused grouped-GEMM, or architecture-specific extension. Its `auto` precision mode queries PyTorch’s BF16 support and selects FP16 when BF16 is unavailable.

A suitable starting command is:

```bash
python sharded_moe_lm.py train \
  --data-dir data/tinyshakespeare \
  --out-dir checkpoints/base \
  --device cuda \
  --precision fp16 \
  --attention-impl auto \
  --seq-len 256 \
  --n-layers 8 \
  --n-heads 8 \
  --d-model 512 \
  --d-ff 1536 \
  --n-experts 16 \
  --moe-top-k 2 \
  --capacity-factor 1.25 \
  --router-noise-std 0.05 \
  --batch-size 2 \
  --grad-accum-steps 16 \
  --max-steps 10000
```

I tested the following successfully:

* Model forward and backward.
* Sparse dispatch with inactive experts proving zero calls.
* Full checkpoint save and reload.
* Independent single-expert shard save.
* Base-manifest validation.
* Router-only integration checkpoint.
* Expert and shared-overlay assembly.
* Numerically matching inference after assembly.
* Two-step CLI training, evaluation, and generation.
* External checkpoint inspection and hash verification.

The execution environment had CPU-only PyTorch, so the CUDA/P40 path could not be physically exercised here.

[1]: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt "char-rnn/data/tinyshakespeare/input.txt at master · karpathy/char-rnn · GitHub"
[2]: https://docs.pytorch.org/docs/stable/amp.html?utm_source=chatgpt.com "Automatic Mixed Precision package - torch.amp"
[3]: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html?utm_source=chatgpt.com "torch.nn.functional.scaled_dot_product_attention"
[4]: https://developer.nvidia.com/cuda/gpus/legacy "Legacy CUDA GPU Compute Capability | NVIDIA Developer"
No. **The total MoE model does not need to be smaller. The portion resident and trainable on each node must be smaller.**

With the existing implementation, a 4 GB GPU node can already train a moderate model one expert at a time, but a strict 4 GB guarantee requires several additional changes.

## What currently consumes GPU memory

For one worker, approximate memory is:

[
M_{\text{GPU}} \approx
4P_{\text{shared}}
+16P_{\text{trainable}}
+M_{\text{activations}}
+M_{\text{temporary}}
+M_{\text{CUDA}}
]

where parameter counts are measured in individual scalar parameters.

The approximate factors come from:

| Item                 | FP32 bytes/parameter |
| -------------------- | -------------------: |
| Frozen shared weight |                    4 |
| Trainable weight     |                    4 |
| Gradient             |                    4 |
| Adam first moment    |                    4 |
| Adam second moment   |                    4 |
| Trainable total      |     approximately 16 |

The current `--precision fp16` uses autocast. Autocast converts selected operations to FP16, but the program never explicitly converts model parameter storage to FP16, so `model.move_shared_to(device)` still puts FP32 parameters on the GPU. PyTorch describes autocast as operation-level casting rather than permanent conversion of model parameters. ([docs.pytorch.org][1])

Activations are additional and depend mostly on:

[
B \times T \times L \times d_{\text{model}}
]

plus attention and MoE routing intermediates.

## What already works without code changes

Train exactly one expert shard:

```bash
python sharded_moe_lm.py train \
  --data-dir data/tinyshakespeare \
  --resume checkpoints/base \
  --out-dir checkpoints/expert_0 \
  --device cuda \
  --precision fp16 \
  --attention-impl sdpa \
  --train-scope experts \
  --train-experts 0 \
  --active-experts 0 \
  --expert-placement active \
  --resident-experts 0 \
  --batch-size 1 \
  --seq-len 64 \
  --grad-accum-steps 64 \
  --max-steps 10000
```

This does the important parts correctly:

* Only expert `0` is trainable.
* Only expert `0` is resident on the GPU.
* Other experts stay on CPU.
* Effective routing becomes top-1 because only one expert is active.
* Gradient accumulation supplies a larger effective batch without increasing microbatch memory.
* SDPA avoids the explicit full attention matrix used by the manual fallback. PyTorch’s SDPA API supports causal attention directly. ([docs.pytorch.org][2])

For the earlier example configuration:

```text
layers      = 8
d_model     = 512
d_ff        = 1536
experts     = 16
```

The approximate counts are:

```text
Shared parameters:              8.50 million
One vertical expert shard:     18.87 million
All expert parameters:        301.99 million
```

One expert with ordinary FP32 Adam requires roughly:

```text
shared weights:       8.50M × 4  =   34 MB
expert training:     18.87M × 16 =  302 MB
```

That is easily below 4 GB before activations.

Training all 16 experts simultaneously would need approximately:

```text
301.99M × 16 = 4.83 GB
```

for expert weights, gradients, and Adam state alone. That cannot fit in 4 GB.

---

# Required changes for a hard 4 GB guarantee

## 1. Add activation checkpointing

This is the most important missing change.

Currently, the forward loop keeps intermediate activations for every Transformer block until backward finishes:

```python
for block in self.blocks:
    x, stats = block(x)
```

Change it to checkpoint each block. Checkpointing stores fewer activations and recomputes the block during backward. PyTorch explicitly provides `torch.utils.checkpoint` for exchanging additional computation for lower activation memory and recommends the non-reentrant implementation. ([docs.pytorch.org][3])

Because `RouterStats` is a dataclass, return ordinary tensors from the checkpoint wrapper:

```python
from torch.utils.checkpoint import checkpoint


def run_checkpointed_block(
    block: TransformerBlock,
    x: torch.Tensor,
) -> tuple[torch.Tensor, RouterStats]:

    def block_forward(inp: torch.Tensor):
        out, stats = block(inp)
        return (
            out,
            stats.load_balance_loss,
            stats.z_loss,
            stats.dropped_fraction,
            stats.expert_load,
            stats.router_entropy,
        )

    (
        output,
        load_balance_loss,
        z_loss,
        dropped_fraction,
        expert_load,
        router_entropy,
    ) = checkpoint(
        block_forward,
        x,
        use_reentrant=False,
        preserve_rng_state=True,
    )

    return output, RouterStats(
        load_balance_loss=load_balance_loss,
        z_loss=z_loss,
        dropped_fraction=dropped_fraction,
        expert_load=expert_load,
        router_entropy=router_entropy,
    )
```

Then:

```python
for block in self.blocks:
    if self.training and self.config.gradient_checkpointing:
        x, stats = run_checkpointed_block(block, x)
    else:
        x, stats = block(x)

    layer_stats.append(stats)
```

Add:

```python
gradient_checkpointing: bool = True
```

to `ModelConfig`.

This is especially important as `n_layers`, sequence length, and hidden width increase.

## 2. Store frozen GPU parameters in FP16

During expert-only training, the shared backbone is frozen. It does not need FP32 parameter storage on the GPU.

After setting `requires_grad`, explicitly cast frozen GPU parameters:

```python
@torch.no_grad()
def cast_frozen_parameters(
    model: nn.Module,
    dtype: torch.dtype,
) -> None:
    for parameter in model.parameters():
        if not parameter.requires_grad and parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype=dtype)
```

Call it after device placement:

```python
model.move_shared_to(device)
model.place_experts(
    device,
    args.expert_placement,
    resident_ids=resident,
)

if device.type == "cuda" and precision_name == "fp16":
    cast_frozen_parameters(model, torch.float16)
```

This changes frozen shared parameter storage from:

```text
4 bytes/parameter
```

to:

```text
2 bytes/parameter
```

Keep the selected trainable experts in FP32 initially. That is more numerically conservative on a P40 than turning the actual optimizer parameters into FP16.

## 3. Offload Adam state to CPU

The current code uses:

```python
torch.optim.AdamW(...)
```

Adam’s two moment tensors stay beside the trainable parameters. This means the optimizer state alone consumes eight bytes per trainable FP32 parameter.

For a hard 4 GB target, put the following on CPU:

* FP32 master weights;
* Adam first moment;
* Adam second moment;
* optionally accumulated gradients.

The GPU then holds primarily:

* the active model weights;
* gradients;
* current activations.

DeepSpeed ZeRO-Offload supports CPU offloading of optimizer state and optimizer computation. ZeRO-3 additionally supports model parameter offload to CPU or NVMe. ([deepspeed.readthedocs.io][4])

A basic custom implementation would operate like this:

```python
# CPU state
cpu_master = parameter.detach().float().cpu().clone()
cpu_m = torch.zeros_like(cpu_master)
cpu_v = torch.zeros_like(cpu_master)

# After backward
cpu_grad = parameter.grad.detach().float().cpu()

# Perform Adam update on CPU
cpu_m.mul_(beta1).add_(cpu_grad, alpha=1.0 - beta1)
cpu_v.mul_(beta2).addcmul_(
    cpu_grad,
    cpu_grad,
    value=1.0 - beta2,
)

update = cpu_m / cpu_v.sqrt().add_(eps)
cpu_master.add_(update, alpha=-learning_rate)

# Copy updated parameter back
parameter.data.copy_(
    cpu_master.to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
)
```

A real implementation also needs:

* bias correction;
* decoupled weight decay;
* asynchronous pinned-memory transfers;
* state serialization;
* gradient scaling overflow handling.

Using an established offload implementation is safer than writing a production optimizer from scratch.

## 4. Change shard granularity for genuinely large models

The current shard boundary is:

```text
expert 7 across every Transformer layer
```

That is a **vertical expert shard**.

Its parameter count is:

[
P_{\text{vertical expert}}
==========================

L \times 3d_{\text{model}}d_{\text{ff}}
]

For a larger configuration:

```text
layers  = 24
d_model = 1024
d_ff    = 4096
```

one vertical expert contains:

[
24 \times 3 \times 1024 \times 4096
===================================

301,989,888
]

parameters.

Ordinary FP32 Adam memory is approximately:

[
301,989,888 \times 16
\approx 4.5\ \text{GiB}
]

That one expert alone already exceeds the GPU.

The checkpoint boundary should therefore become:

```text
(layer_id, expert_id)
```

rather than only:

```text
expert_id
```

For example:

```text
layer_000/expert_007.pt
layer_001/expert_007.pt
layer_002/expert_007.pt
```

One layer expert contains:

[
3 \times 1024 \times 4096
=========================

12,582,912
]

parameters, requiring roughly 192 MiB for FP32 parameter, gradient, and Adam tensors.

The parameter extraction function should change from:

```python
expert_id_from_parameter_name(name)
```

to something returning both IDs:

```python
LAYER_EXPERT_RE = re.compile(
    r"^blocks\.(\d+)\.moe\.experts\.(\d+)\."
)


def layer_expert_from_parameter_name(
    name: str,
) -> tuple[int, int] | None:
    match = LAYER_EXPERT_RE.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))
```

Then support:

```bash
--train-layer-experts 0:7,1:7,2:7
```

or, for the smallest working set:

```bash
--train-layer-experts 12:7
```

The assembler overlays those precise tensors into the final model.

### Important limitation

Training each `(layer, expert)` entirely independently is not equivalent to simultaneous end-to-end pretraining.

A sensible sequence is:

1. Train a stable shared backbone.
2. Freeze the backbone.
3. Train individual layer experts or small expert groups.
4. Assemble them.
5. Train routers.
6. Run a final low-rate integration phase across groups of experts.
7. Validate and reject regressions.

You get composability, but some integration training remains necessary.

## 5. Stop constructing every expert in host RAM

This matters if “4 GB total memory per node” includes system RAM, not merely GPU VRAM.

Currently the program executes:

```python
model = SparseMoELanguageModel(config)
```

That allocates **all experts in CPU memory**, even when only one expert is later placed on the GPU.

For an arbitrarily large model, change construction to use lazy or metadata-only modules:

```python
with torch.device("meta"):
    model = SparseMoELanguageModel(config)
```

Then materialize only:

* shared layers required by this worker;
* the selected expert or layer-expert shard;
* routing metadata.

Alternatively, replace `ModuleList` of concrete experts with lazy handles:

```python
class LazyExpertHandle:
    def __init__(self, layer_id: int, expert_id: int, checkpoint_path: Path):
        self.layer_id = layer_id
        self.expert_id = expert_id
        self.checkpoint_path = checkpoint_path
        self.module = None

    def load(self, device: torch.device) -> SwiGLUExpert:
        if self.module is None:
            module = SwiGLUExpert(...)
            state = load_expert_state(self.checkpoint_path)
            module.load_state_dict(state)
            self.module = module.to(device)
        return self.module

    def unload(self) -> None:
        self.module = None
```

For a strict total-RAM limit, expert and optimizer tensors must be memory-mapped or streamed from NVMe rather than fully loaded into CPU RAM. DeepSpeed’s ZeRO-3 documentation supports CPU and NVMe parameter offloading for this class of use case. ([deepspeed.readthedoc 6. Add a runtime memory budget

A fixed configuration is less reliable than measuring actual peaks.

Add:

```python
torch.cuda.reset_peak_memory_stats()
```

before training and log:

```python
allocated = torch.cuda.max_memory_allocated()
reserved = torch.cuda.max_memory_reserved()

print(
    f"peak allocated={allocated / 2**20:.1f} MiB, "
    f"peak reserved={reserved / 2**20:.1f} MiB"
)
```

Use a conservative target:

```text
maximum allocated: 3.0–3.3 GiB
```

rather than trying to reach exactly 4.0 GiB. CUDA context, allocator fragmentation, library workspaces, and temporary tensors need headroom.

---

# Recommended 4 GB node architecture

Each node should hold:

```text
GPU:
  one small shared-backbone partition
  one layer-expert or small expert group
  current microbatch activations
  gradients for that shard

CPU:
  optimizer master state where RAM permits
  staging buffers

NVMe:
  complete expert checkpoint
  Adam moments when CPU RAM is constrained
  immutable base checkpoint
```

Use:

```text
microbatch size       = 1
sequence length       = 64 initially
gradient accumulation = 32–256
active experts        = 1 during isolated training
resident experts      = 1
activation checkpoint = enabled
frozen weights        = FP16
trainable weights     = FP32 initially
optimizer state       = CPU or NVMe
attention             = SDPA
```

## Bottom line

You can make the **assembled model enormous** by adding more experts without increasing each node’s GPU memory proportionally.

But these three things must fit or be streamed:

1. The shared backbone portion needed by the node.
2. The currently trained expert shard.
3. Activations and gradients for one microbatch.

For moderate models, the existing vertical-expert design plus batch size 1 will fit 4 GB.

For truly large models, the required structural changes are:

```text
vertical expert shards
        ↓
per-layer expert shards

full FP32 shared backbone on GPU
        ↓
FP16 frozen weights or layer streaming

GPU Adam state
        ↓
CPU/NVMe optimizer state

normal forward activation storage
        ↓
activation checkpointing

all experts allocated in host RAM
        ↓
lazy/meta-device checkpoint loading
```

[1]: https://docs.pytorch.org/docs/stable/amp.html?utm_source=chatgpt.com "Automatic Mixed Precision package - torch.amp"
[2]: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html?utm_source=chatgpt.com "torch.nn.functional.scaled_dot_product_attention"
[3]: https://docs.pytorch.org/docs/stable/checkpoint.html?utm_source=chatgpt.com "torch.utils.checkpoint"
[4]: https://deepspeed.readthedocs.io/en/stable/zero3.html?utm_source=chatgpt.com "ZeRO — DeepSpeed 0.19.0 documentation"
## Direct answer

For the independently trainable architecture we discussed, **each specialist expert should receive a deliberately different—but partially overlapping—training distribution**.

Do not give every expert the same data. They will tend to become redundant.

Do not give experts completely isolated datasets either. They may become brittle, incompatible with the shared backbone, and catastrophic when the router chooses the wrong one.

The strongest practical design is:

```text
always-on shared expert
        +
one or more routed specialist experts
```

The shared expert learns general language structure and common knowledge. Specialist experts learn domains such as code, mathematics, legal text, electronics, fiction, Swedish, and so on. DeepSeekMoE uses the same broad principle: shared experts capture common capabilities while routed experts provide additional specialized capacity. ([arXiv][1])

---

# There are three different ways to train MoE experts

## 1. Conventional end-to-end MoE

In a standard MoE pretraining run:

* every worker draws from the same global mixed corpus;
* the router decides which tokens visit each expert;
* experts and router train simultaneously;
* specialization emerges automatically.

You do **not** manually construct a separate dataset for each expert.

This is how architectures such as Switch Transformer, OLMoE and conventional DeepSeekMoE training generally work. However, emergent specialization is not guaranteed to correspond cleanly to human concepts such as “mathematics expert” or “C++ expert.” OpenMoE found that routing could become strongly associated with token identity and token-level semantics rather than contextual domain understanding, and that those routing patterns formed early during training. ([arXiv][2])

So standard MoE experts may become specialists in things like:

```text
punctuation patterns
number-like hidden states
proper nouns
source-code syntax tokens
sentence boundaries
particular lexical groups
formatting patterns
```

That can improve the model without producing interpretable experts.

## 2. Explicit domain experts

Here you assign separate datasets:

```text
expert 0 -> general web and books
expert 1 -> source code
expert 2 -> mathematics
expert 3 -> scientific papers
expert 4 -> embedded systems documentation
expert 5 -> legal documents
expert 6 -> Swedish text
expert 7 -> fiction and dialogue
```

Each expert begins from the same seed model or identical expert initialization and is then continued on its assigned distribution.

Branch-Train-MiX follows this general pattern: start from a seed model, branch into separately trained domain models, place their feed-forward networks into MoE layers, and then train a router during an integration stage. ([arXiv][3])

This approach best matches your goal of:

* asynchronous training;
* small independent nodes;
* independently verifiable contributions;
* removable experts;
* understandable data provenance.

## 3. Automatically discovered experts

Instead of inventing domains manually:

1. Embed complete documents or document chunks.
2. Cluster them semantically.
3. Balance clusters by token count.
4. Train one expert per cluster.
5. Train a router to recognize the clusters.

Research on unsupervised domain discovery has shown that clustering corpora into meaningful document groups and independently training expert language models can work well while avoiding synchronous expert training. ([arXiv][4])

This can discover categories you did not anticipate, but the resulting clusters may be difficult to name or uneven in quality.

---

# What I recommend for your model

Use a **hybrid supervised-domain architecture**.

## Shared data pool

Create a high-quality common dataset containing:

* general prose;
* dialogue;
* basic reasoning;
* common programming discussions;
* general factual material;
* instruction formatting;
* broad linguistic coverage.

The shared expert and shared backbone see this entire pool.

## Specialist data pools

Each specialist has a manifest such as:

```yaml
expert_id: 7
name: embedded_systems
base_checkpoint: sha256:...
specialist_sources:
  - Zephyr documentation
  - Linux device-driver documentation
  - MCU reference manuals
  - embedded C repositories
  - RTOS discussions
common_replay_fraction: 0.25
specialist_fraction: 0.70
contrastive_fraction: 0.05
tokens: 12_000_000_000
```

A reasonable starting mixture—not a universal rule—is:

```text
65–80% specialist data
15–30% shared general replay
5–10% neighboring or contrastive domains
```

For example, a C/embedded expert should not see only register definitions. It should also see enough:

* natural language;
* general programming;
* build systems;
* operating systems;
* mathematics;
* documentation prose;

to remain compatible with the rest of the model.

## Why the common replay portion matters

Suppose the code expert sees only source files. It may become excellent at predicting syntax but poor at answering:

> Explain why this DMA ring buffer has a race condition.

That task combines:

* prose;
* concurrency;
* code;
* hardware knowledge;
* reasoning.

The common-data portion prevents the expert from drifting too far away from the shared representation space.

---

# Assign documents, not individual random tokens

For manually specialized experts, perform domain assignment at the:

* document;
* repository;
* book;
* article;
* conversation;
* long sequence;

level.

Do not randomly send individual tokens to separate training corpora.

A code repository should generally remain together because understanding one file may depend on:

* headers;
* definitions;
* comments;
* build configuration;
* neighboring source files.

For mixed documents, you can assign multiple labels:

```json
{
  "document": "linux_dma_driver_tutorial",
  "domains": {
    "embedded": 0.55,
    "linux": 0.30,
    "code": 0.15
  }
}
```

You can either:

* sample it into several experts;
* assign it to the highest-scoring expert;
* use probabilistic assignment according to the scores.

Some duplication is desirable. Massive uncontrolled duplication is not.

---

# A good routing structure

For your model, I would modify routing conceptually to:

```text
output =
    shared_expert(x)
  + specialist_router(x)
```

With:

```text
1 shared expert always active
1 or 2 specialists selected by top-k
```

For example:

```text
token hidden state
      |
      +---- shared expert
      |
      +---- top-1 specialist
```

This makes failures less catastrophic. Even if the specialist router makes a poor choice, the token still receives the general expert transformation.

DeepSeekMoE reported that disabling its shared expert caused a large loss increase, supporting the idea that the shared component carries broadly essential functionality rather than merely duplicating routed experts. ([arXiv][1])

---

# How to train one expert

Assume expert `7` is the embedded-systems expert.

## Step 1: Start from a common base

Every expert must begin from the same:

* tokenizer;
* shared backbone;
* attention weights;
* normalization parameters;
* base expert initialization;
* checkpoint hash.

Do not initialize each specialist completely independently unless you intend to perform much more integration training.

## Step 2: Freeze the shared model

Train only:

```text
blocks.0.moe.experts.7
blocks.1.moe.experts.7
...
blocks.N.moe.experts.7
```

Force all specialist-path tokens through expert `7` during this stage.

The router should not be trained independently by that node because the node sees only its own data distribution. It cannot learn how its domain compares with the other experts.

## Step 3: Use a mixed specialist curriculum

For every batch, sample approximately:

```python
r = random.random()

if r < 0.70:
    batch = embedded_dataset.sample()
elif r < 0.95:
    batch = common_dataset.sample()
else:
    batch = neighboring_domains.sample()
```

## Step 4: Save the expert shard

The result should contain:

```text
expert weights
base checkpoint hash
dataset manifest hash
token counts
training steps
domain labels
validation metrics
```

## Step 5: Integrate all experts

Once the experts are assembled:

1. Freeze all experts and the backbone.
2. Train only the router on a balanced mixture of every domain.
3. Evaluate routing.
4. Optionally unfreeze experts for a brief low-learning-rate integration phase.

BTX uses an MoE fine-tuning stage after independently trained branches are assembled specifically to learn token-level routing. ([arXiv][3])

FlexOlmo demonstrates an even more modular design in which independently trained experts can be integrated using domain-informed routing without sharing the original private datasets during joint training. ([arXiv][5])

---

# How do you know what an expert knows?

You cannot determine this from its training-data name alone.

You need at least four types of evidence.

## 1. Data provenance

This tells you what the expert **was exposed to**.

Record:

```text
source datasets
document categories
languages
token distributions
dates
licenses
deduplication hashes
training token counts
```

Exposure does not prove capability, but without it you have no auditable basis for the expert’s intended function.

## 2. Routing specialization

For domain (D) and expert (e), calculate:

[
R(e,D)=
\frac{
\text{tokens from domain }D\text{ routed to expert }e
}{
\text{all tokens from domain }D
}
]

Build a matrix:

| Expert | General | Code | Math | Science | Legal |
| ------ | ------: | ---: | ---: | ------: | ----: |
| E0     |     41% |   8% |  10% |     15% |   12% |
| E1     |      6% |  72% |  19% |     11% |    3% |
| E2     |      8% |  17% |  68% |     22% |    4% |
| E3     |     15% |   5% |  14% |     61% |    7% |

OLMoE uses essentially this type of domain-routing analysis, measuring the fraction of domain tokens routed to each expert. It found meaningful domain specialization for some experts and substantially weaker domain specialization in another evaluated MoE, demonstrating that specialization is architecture- and training-dependent rather than automatic. ([arXiv][6])

But routing alone only says:

> The router uses this expert on code.

It does not prove:

> This expert improves code performance.

## 3. Causal ablation

Disable expert (e), reroute its tokens to alternatives, and measure the loss increase:

[
A(e,D)=
\operatorname{NLL}_{-e}(D)
--------------------------

\operatorname{NLL}_{normal}(D)
]

Interpretation:

```text
large positive A(e, code)
small A(e, general)
```

means expert (e) is causally important for code and less important for general prose.

This is one of the strongest simple tests.

DeepSeekMoE similarly evaluated expert redundancy by disabling high-ranked routed experts and observing the loss degradation. Greater degradation indicated that the selected experts were less replaceable and therefore less redundant. ([arXiv][1])

## 4. Forced-expert evaluation

Force every eligible token through expert (e) and measure domain loss:

[
L_{\text{forced}}(e,D)
]

Then compare experts:

```text
                    General NLL    Code NLL    Math NLL
expert 0                2.10          3.44         3.72
expert 1                2.55          1.81         3.20
expert 2                2.62          3.01         1.74
```

That gives a direct capability profile.

However, forcing one expert everywhere is unnatural, so use it as a diagnostic—not as the final model score.

---

# The best metric: marginal utility

The question is not merely:

> Does the router activate expert 7?

The real question is:

> Does activating expert 7 improve the prediction compared with the alternatives?

For each token or evaluation sequence, calculxt{best alternative experts})
---------------------------------------

L(x\mid e\text{ available})
]

Positive (U) means the expert helps.

Aggregate by domain:

[
U(e,D)=\mathbb{E}_{x\in D}[U(e,x)]
]

This produces an expert capability table:

| Expert | Intended domain | Routing rate | Loss improvement | Removal damage |
| ------ | --------------- | -----------: | ---------------: | -------------: |
| E3     | Embedded C      |          64% |             0.41 |           0.36 |
| E4     | Mathematics     |           7% |             0.02 |           0.01 |
| E5     | General         |          18% |             0.08 |           0.06 |

If an expert routes heavily but has near-zero marginal utility, the router may be wasting capacity.

---

# Evaluate at multiple levels

Experts in different layers do different things.

An early-layer expert may specialize in:

* character patterns;
* token classes;
* language identification;
* formatting;
* lexical structure.

A middle-layer expert may specialize in:

* syntax;
* code constructs;
* sentence structure;
* local semantic relationships.

A late-layer expert may affect:

* factual prediction;
* reasoning continuations;
* style;
* answer formation.

Therefore, it is dangerous to say:

> Expert 7 knows mathematics.

A more accurate statement is:

> The vertical expert-7 pathway provides measurable marginal improvement on mathematical sequences, especially in layers 8–16.

Knowledge remains distributed across:

* embeddings;
* attention;
* shared backbone;
* normalization;
* multiple experts;
* router decisions.

---

# Expert card format

Every integrated expert should have a machine-readable report:

```yaml
expert_id: 7
name: embedded_systems
base_checkpoint: sha256:abc123
training:
  specialist_tokens: 8_400_000_000
  common_tokens: 2_500_000_000
  neighboring_tokens: 600_000_000
  domains:
    embedded: 0.61
    code: 0.24
    linux: 0.10
    general: 0.05

routing:
  embedded_activation_rate: 0.67
  general_activation_rate: 0.09
  average_router_probability: 0.71
  overflow_drop_rate: 0.003

causal_evaluation:
  embedded_nll_improvement: 0.38
  code_nll_improvement: 0.17
  general_nll_improvement: 0.02
  removal_damage_embedded: 0.34

benchmarks:
  embedded_qa_delta: 8.4
  code_completion_delta: 3.1
  general_language_delta: -0.2

overlap:
  nearest_expert: 11
  routing_coactivation: 0.23
  output_similarity: 0.31
```

This becomes the evidence for what the expert actually contributes.

---

# Tiny Shakespeare specifically

Tiny Shakespeare is too small and homogeneous to produce meaningful experts such as “math,” “science,” or “programming.”

You can still use it to test the mechanism.

A reasonable four-expert experiment would be:

```text
shared expert:
    all Tiny Shakespeare text

expert 0:
    dialogue-heavy chunks

expert 1:
    stage directions and speaker labels

expert 2:
    verse-like or high-rhythm chunks

expert 3:
    prose-like or narrative chunks
```

Alternatively, divide 256–1024 character chunks using:

1. TF-IDF or embedding vectors.
2. Balanced k-means with four clusters.
3. Inspect top words and representative samples.
4. Train one expert per cluster.
5. Give every specialist 20–30% samples from the complete corpus.
6. Train the router on all clusters after assembly.

With a character-level tokenizer, expect specialization around:

* capitalization;
* speaker-name patterns;
* punctuation;
* newlines;
* archaic spelling;
* dialogue structure;

rather than high-level literary understanding.

---

# Recommended training lifecycle

```text
1. Train shared backbone/general expert on all data

2. Clone a common expert initialization

3. Define expert datasets:
      code
      math
      science
      embedded
      legal
      language-specific
      general

4. Independently train each expert:
      mostly specialist data
      some shared replay
      backbone frozen
      routing forced

5. Assemble expert shards

6. Train router on balanced union:
      experts frozen
      backbone frozen

7. Measure:
      routing distributions
      domain perplexity
      forced-expert loss
      expert ablation
      co-activation
      overflow rates

8. Run a short integration phase:
      low learning rate
      router + experts
      mixed corpus

9. Accept/reject each expert based on marginal utility
```

The central rule is:

> **The dataset defines the expert’s intended specialty; causal evaluation defines what it actually contributes.**

[1]: https://arxiv.org/html/2401.06066v1 "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"
[2]: https://arxiv.org/html/2402.01739v2 "OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models"
[3]: https://arxiv.org/html/2403.07816v1 "Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM"
[4]: https://arxiv.org/abs/2303.14177?utm_source=chatgpt.com "Scaling Expert Language Models with Unsupervised Domain Discovery"
[5]: https://arxiv.org/html/2507.07024v1 "FlexOlmo: Open Language Models for Flexible Data Use"
[6]: https://arxiv.org/html/2409.02060v1 "OLMoE: Open Mixture-of-Experts Language Models"
The clean practical architecture is:

```text
immutable shared backbone
+ immutable public anchor expert
+ independently trainable client experts
+ append-only expert/router registry
+ tenant-specific retrieval index
```

A client should not fine-tune a complete copy of your large model and return the entire checkpoint. They should train a narrowly defined expert package against an exact, versioned anchor model.

FlexOlmo is almost exactly this design: data owners independently train their feed-forward experts and router embeddings against a frozen public anchor, then the resulting expert modules are combined without requiring access to all private datasets or joint training. ([arXiv][1])

# 1. Decide what belongs in weights and what belongs in RAG

Internal company knowledge divides into two categories.

### Train into the expert

Use expert training for:

* company terminology;
* reasoning patterns;
* document structure;
* internal coding conventions;
* style and communication format;
* domain-specific interpretation;
* common procedures;
* how to use internal tools;
* how to reason over retrieved internal documents.

### Keep in retrieval

Use RAG for:

* customer records;
* current product inventory;
* prices;
* source-of-truth procedures;
* frequently changing policies;
* credentials and secrets;
* employee information;
* exact document contents;
* anything requiring citations or access control.

Parametric model weights are difficult to update precisely and cannot provide reliable provenance. Retrieval keeps knowledge explicit and replaceable, which is the original motivation behind RAG for knowledge-intensive tasks. ([arXiv][2])

The best client product is therefore:

```text
client expert:
    understands the client's domain

client retrieval index:
    contains the client's actual current knowledge
```

Do **not** train passwords, API keys, personal records or proprietary source code verbatim into an expert you intend to collect centrally.

Model weights can leak training examples. FlexOlmo itself found nonzero extraction from independently trained experts and recommends differential privacy when private or sensitive data is involved. More general research has demonstrated extraction of training examples from language models. ([arXiv][1])

# 2. Give every client an immutable expert SDK

You distribute a signed training package:

```text
client-expert-sdk/
├── base/
│   ├── config.json
│   ├── tokenizer.json
│   ├── backbone.safetensors
│   ├── anchor_experts.safetensors
│   └── anchor_router.safetensors
├── trainer/
│   ├── train_expert.py
│   ├── preprocess.py
│   ├── evaluate.py
│   └── export.py
├── calibration/
│   ├── public_replay.bin
│   ├── negative_examples.bin
│   └── evaluation_prompts.jsonl
├── container/
│   └── locked-container-digest.txt
└── manifest.json
```

The manifest must identify the exact model ABI:

```json
{
  "model_family": "vroom-moe",
  "expert_abi": 3,
  "base_model_version": "2026.06",
  "base_checkpoint_sha256": "abc123...",
  "tokenizer_sha256": "def456...",
  "architecture_sha256": "789abc...",
  "num_layers": 32,
  "hidden_size": 4096,
  "expert_intermediate_size": 11008,
  "router_type": "anchor-key-v1",
  "training_code_sha256": "..."
}
```

Every accepted expert must have been trained from that exact anchor.

An expert trained against backbone version `2026.06` must not silently be loaded into backbone `2027.01`. The hidden-state coordinate system may have changed. Treat the backbone and tokenizer hashes as an ABI.

# 3. Change the router from fixed-size output to expert keys

Your current example likely has something equivalent to:

```python
self.router = nn.Linear(d_model, n_experts)
```

That permanently fixes the number of experts because the router has exactly `n_experts` output rows.

Instead, represent each expert using an independent router key and bias:

[
s_e(x) = x^\top k_e + b_e
]

where:

* (x) is the token representation;
* (k_e) is the expert’s router key;
* (b_e) is its selectivity bias.

Adding an expert means appending:

```text
expert FFN weights
router key
router bias
metadata
```

The router becomes:

```python
class ComposableRouter(nn.Module):
    def __init__(
        self,
        expert_keys: torch.Tensor,
        expert_biases: torch.Tensor,
        top_k: int,
    ) -> None:
        super().__init__()

        if expert_keys.ndim != 2:
            raise ValueError("expert_keys must be [experts, hidden_size]")

        if expert_biases.shape != (expert_keys.shape[0],):
            raise ValueError("expert_biases must be [experts]")

        self.expert_keys = nn.Parameter(expert_keys)
        self.expert_biases = nn.Parameter(expert_biases)
        self.top_k = top_k

    def forward(
        self,
        hidden: torch.Tensor,
        allowed_experts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden: [tokens, hidden_size]
        logits = hidden @ self.expert_keys.T
        logits = logits + self.expert_biases

        if allowed_experts is not None:
            if allowed_experts.shape != (logits.shape[-1],):
                raise ValueError("Incorrect allowed_experts shape")

            logits = logits.masked_fill(
                ~allowed_experts.unsqueeze(0),
                float("-inf"),
            )

        k = min(self.top_k, logits.shape[-1])
        top_logits, top_indices = torch.topk(logits, k=k, dim=-1)

        combine_weights = torch.softmax(top_logits, dim=-1)

        return top_indices, combine_weights
```

FlexOlmo uses independent per-expert router embeddings and later concatenates those embeddings into a unified router. Keeping the public router embedding fixed gives independently trained experts a common coordinate system. ([arXiv][1])

# 4. Train every client expert against a public anchor

At each MoE layer, create a two-expert local training model:

```text
expert 0: public anchor expert — frozen
expert 1: client expert        — trainable
```

Both start from the same FFN weights.

For layer (l):

```text
anchor FFN E₀,l  = frozen
client FFN Ec,l  = initialized from E₀,l, then trained

anchor key k₀,l  = frozen
client key kc,l  = trainable

anchor bias b₀,l = frozen
client bias bc,l = trainable
```

Freeze:

* token embeddings;
* attention;
* RMSNorm;
* output head;
* public anchor experts;
* public router keys.

Train only:

* client FFNs;
* client router keys;
* client router biases.

This common frozen anchor prevents every client from inventing an incompatible hidden-state coordinate system. FlexOlmo found that naive isolated continued training allowed experts to diverge excessively, while frozen-anchor training made independent composition practical. ([arXiv][1])

## Local routing during client training

For each token:

```python
anchor_score = hidden @ anchor_key
client_score = hidden @ client_key + client_bias

scores = torch.stack(
    [anchor_score, client_score],
    dim=-1,
)
```

The client expert should not automatically process every token.

Give it a negative initial bias:

```python
client_bias = -2.0
```

That forces it to demonstrate that it improves the language-model objective before replacing the anchor. FlexOlmo uses a negative bias for similar selectivity: pairwise training against the anchor would otherwise produce overly broad expert regions when independently composed. ([arXiv][1])

# 5. Local client training recipe

The client prepares three data pools:

```text
70–85% internal domain material
10–25% supplied public replay data
 5–10% negative or neighboring-domain data
```

A batch sampler could use:

```python
def choose_dataset(rng):
    value = rng.random()

    if value < 0.75:
        return internal_dataset

    if value < 0.95:
        return public_replay_dataset

    return negative_domain_dataset
```

The public replay data helps prevent the specialist from becoming incompatible or broadly overriding the anchor.

## Use three loss components

[
L =
L_{\mathrm{LM}}

* \lambda_{\mathrm{anchor}}L_{\mathrm{anchor}}
* \lambda_{\mathrm{router}}L_{\mathrm{router}}
  ]

### Language-model loss

Ordinary causal next-token prediction on internal data.

### Anchor-preservation loss

On public replay data, penalize divergence from the original model:

[
L_{\mathrm{anchor}}
===================

D_{\mathrm{KL}}
\left(
p_{\mathrm{anchor}}
\parallel
p_{\mathrm{client}}
\right)
]

This keeps the expert from changing unrelated behavior excessively.

### Router selectivity loss

Encourage the client expert on internal examples and discourage it on unrelated examples:

```python
internal_router_loss = binary_cross_entropy_with_logits(
    client_score - anchor_score,
    torch.ones_like(client_score),
)

negative_router_loss = binary_cross_entropy_with_logits(
    client_score - anchor_score,
    torch.zeros_like(client_score),
)
```

Do not force the expert label on every internal token. Some ordinary language tokens should still use the public expert.

# 6. Train by layer-expert shards on 4 GB nodes

For a constrained node, do not require the entire vertical expert to be trainable simultaneously.

Train:

```text
(layer 0, client expert)
(layer 1, client expert)
...
(layer N, client expert)
```

or small consecutive groups:

```text
layers 0–3
layers 4–7
layers 8–11
```

Each training artifact is identified by:

```text
client_uuid
expert_uuid
base_model_hash
layer_start
layer_end
revision
```

The node uses:

* one resident layer-expert at a time;
* frozen backbone streaming;
* activation checkpointing;
* CPU/NVMe optimizer state;
* batch size 1;
* gradient accumulation.

The complete client expert is then the collection:

```text
client_expert/
├── layer_000.safetensors
├── layer_001.safetensors
├── ...
├── layer_031.safetensors
├── router_keys.safetensors
└── manifest.json
```

# 7. What the client returns

Use `safetensors`, not Python-pickled `.pt` files, for externally supplied artifacts.

A package should look like:

```text
expert-7f49c2/
├── manifest.json
├── signature.ed25519
├── layers/
│   ├── 000.safetensors
│   ├── 001.safetensors
│   └── ...
├── router/
│   ├── keys.safetensors
│   └── biases.safetensors
├── reports/
│   ├── private-evaluation.json
│   ├── public-evaluation.json
│   ├── training-metrics.jsonl
│   └── privacy-report.json
└── hashes.sha256
```

Example manifest:

```json
{
  "expert_id": "7f49c2",
  "owner_id": "customer-1049",
  "base_checkpoint_sha256": "abc123...",
  "tokenizer_sha256": "def456...",
  "expert_abi": 3,
  "domain": "industrial-control-systems",
  "visibility": "tenant-private",
  "allowed_tenants": ["customer-1049"],
  "training": {
    "tokens": 483201187,
    "steps": 11797,
    "internal_fraction": 0.78,
    "public_replay_fraction": 0.17,
    "negative_fraction": 0.05,
    "differential_privacy": false
  },
  "layer_files": {
    "0": "layers/000.safetensors",
    "1": "layers/001.safetensors"
  }
}
```

The client does not need to send:

* raw documents;
* tokenized datasets;
* validation examples;
* internal document names.

However, claims about training quality cannot be independently proven from metrics alone. For high-assurance onboarding, use a signed training container and either:

* an auditor inside the client environment;
* confidential-compute attestation;
* reproducible reruns on a permitted sample;
* a non-sensitive proxy evaluation set supplied by the client.

# 8. Server-side intake and validation

Never immediately add an ibutions;
* router-key norm;
* router-bias range.

Reject packages containing arbitrary executable files.

## Capability verification

Run:

1. Public regression evaluation.
2. Domain proxy evaluation.
3. Forced-client-expert evaluation.
4. Expert-disabled evaluation.
5. Router activation analysis.
6. Backdoor and trigger screening.
7. Memorization/extraction probes.
8. Load and capacity tests.
9. Adversarial prompt testing.
10. Comparison with the anchor-only model.

Measure:

[
\Delta L_D
==========

## L_{\text{anchor}}(D)

L_{\text{with expert}}(D)
]

and:

[
\Delta L_G
==========

## L_{\text{with expert}}(G)

L_{\text{anchor}}(G)
]

You want:

```text
large positive improvement on domain D
near-zero regression on general set G
```

No finite validation suite can prove an externally trained expert contains no backdoor. Keep experts revocable, tenant-scoped and canary-deployed.

# 9. Integration into the large model

Do not necessarily create one giant physical checkpoint.

Create a logical composition manifest:

```json
{
  "backbone": "vroom-moe-2026.06",
  "anchor_expert": "general-v12",
  "experts": [
    "math-v4",
    "code-v7",
    "embedded-v3",
    "customer-1049-private-v1"
  ]
}
```

At service startup:

1. Load the shared backbone.
2. Load the public anchor expert.
3. Load the registered expert modules.
4. Stack their router keys per layer.
5. Construct the expert-index mapping.
6. Apply tenant authorization masks.
7. Run top-k routing over only authorized experts.

Adding a new expert becomes:

```python
registry.install(package)
registry.validate(package)
deployment.add_expert(package.expert_id)
deployment.canary(package.expert_id)
deployment.promote(package.expert_id)
```

FlexOlmo supports adding or removing an expert by adding or removing its expert module and router embedding, without retraining the other experts. Nexus similarly investigates adding independently trained experts to an existing MoE using domain-informed expert embeddings. ([arXiv][1])

# 10. Tenant authorization must happen before top-k

For a client-private expert:

```python
allowed_experts = registry.allowed_for(
    tenant_id=request.tenant_id,
    user_id=request.user_id,
)
```

Then mask unauthorized experts before softmax and top-k:

```python
logits[:, ~allowed_experts] = float("-inf")
```

Do not route globally and discard unauthorized selections afterward. That would distort routing probabilities and could create side channels.

A user from another tenant must not be able to:

* select the expert;
* inspect its router score;
* infer its presence;
* load its weights;
* query it indirectly.

This gives you:

```text
same shared backbone
different authorized expert sets per customer
```

# 11. Router integration choices

There are three practical levels.

## Level A: zero central client-data access

Use anchor-pair training and concatenate client router keys.

```text
client data never leaves
no joint router training
instant expert installation
```

This is closest to FlexOlmo. ([arXiv][3])

## Level B: client supplies redacted proxy examples

The client sends a small non-confidential proxy dataset or synthetic examples representing its domain.

You freeze all experts and train only:

* router keys;
* router biases;
* optional domain classifier.

This normally produces better competition between overlapping experts.

## Level C: centrally train on combined domain data

This is the BTX approach:

1. Independently train experts.
2. Assemble their FFNs.
3. Train the unified router on the combined data.
4. Optionally perform short MoE integration training.

BTX demonstrated asynchronously trained branches being combined into MoE feed-forward layers followed by a shorter joint routing phase. ([arXiv][4])

For confidential client data, Level C is generally unavailable unless the client permits central use.

# 12. Adding versus upgrading an expert

## Adding a new domain

```text
existing:
    general
    code
    math

add:
    embedded systems
```

Append its FFNs and router keys. Existing experts stay unchanged.

## Upgrading an existing expert

```text
embedded-v1
        ↓
embedded-v2
```

Keep the same logical domain ID but deploy a new revision:

```json
{
  "logical_expert": "embedded",
  "revision": 2,
  "replaces": "embedded-v1"
}
```

Canary `v2`, compare it against `v1`, then atomically update the registry pointer.

BAR demonstrated independently adding and replacing domain experts while holding previous experts fixed, using lightweight router composition. It also found that modularity avoided cross-domain degradation seen in monolithic continual training. ([arXiv][5])

# 13. Important limitation: post-training behavior is harder

The frozen-backbone expert method is best for:

* continued pretraining;
* domain terminology;
* domain knowledge patterns;
* specialized prediction;
* document and code understanding.

It may be insufficient for major behavioral changes such as:

* a new reasoning format;
* new special tokens;
* tool-call syntax;
* safety policy;
* complex instruction following.

BAR found that freezing all shared non-FFN parameters significantly degraded some post-training scenarios because behavior changes can require attention and other shared components to adapt. Its solution progressively unfroze shared layers and later merged them. ([arXiv][5])

For such clients, give the package two parts:

```text
client expert FFNs
+ small client-specific shared LoRA adapter
```

At inference:

```text
shared backbone
+ tenant LoRA
+ tenant expert
+ tenant RAG
```

The LoRA adapter must also be tenant-scoped. It should not be averaged blindly into the global backbone.

# 14. Recommended production architecture

```text
                         Model registry
                               |
            ----------------------------------------
            |                  |                   |
      backbone v3        public experts       private experts
            |                  |                   |
            ---------------- composition -----------
                               |
                         inference service
                               |
                    tenant authorization mask
                               |
                top-k over authorized experts only
                               |
             -----------------------------------
             |                                 |
       shared/public expert              client expert
             |                                 |
             -------------- combine ------------
                               |
                        retrieval context
                               |
                           final output
```

The practical product contract should be:

> The client keeps its source documents locally, trains a versioned expert against our immutable anchor, and returns a signed expert artifact. We validate it, register it as a removable module, and expose it only to authorized tenants. Exact and changing company knowledge remains in a client-controlled retrieval index.

That gives you asynchronous training, no raw data pooling, independent expert upgrades, revocation and controlled integration into the larger model.

[1]: https://arxiv.org/html/2507.07024v1 "FlexOlmo: Open Language Models for Flexible Data Use"
[2]: https://arxiv.org/abs/2005.11401?utm_source=chatgpt.com "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
[3]: https://arxiv.org/abs/2507.07024?utm_source=chatgpt.com "FlexOlmo: Open Language Models for Flexible Data Use"
[4]: https://arxiv.org/html/2403.07816v1 "Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM"
[5]: https://arxiv.org/html/2604.18473v1 "Train Separately, Merge Together: Modular Post-Training with Mixture-of-Experts"
It does **not** scale to literally unlimited experts or unlimited capability. It can scale to an **effectively unbounded expert catalog** while keeping per-request compute bounded, but storage, routing quality, validation cost, and the shared backbone remain hard limits.

The right mental model is not one giant MoE checkpoint. It is:

> **A shared model runtime plus a searchable registry of expert modules.**

## The scaling equation

Let:

* (E) = total registered experts
* (P_e) = parameters per expert
* (k) = experts activated per token
* (d) = router-key dimension

Then:

[
\text{total storage} \propto E P_e
]

but with sparse routing:

[
\text{expert compute per token} \propto k P_e
]

So total capacity can grow while active compute stays approximately constant, provided (k) remains fixed.

The trap is routing. A naïve router comparing every token against every expert costs:

[
O(E d)
]

That becomes impossible at large (E).

PEER demonstrated learned product-key retrieval over more than one million very small experts, replacing exhaustive routing with a sublinear lookup structure. However, its experts are essentially tiny single-neuron MLP units, not million-parameter client experts, so it proves the routing principle rather than directly solving your full client-expert problem. ([arXiv][1])

# The architecture that scales

Use two-stage routing.

```text
request
   |
tenant and permission filter
   |
coarse expert retrieval
   |
candidate set: perhaps 8–64 experts
   |
load or locate candidate experts
   |
token-level top-k routing
   |
activate perhaps 1–2 experts per token
```

## Stage 1: request-level expert retrieval

Before model execution, derive a request representation from:

* the prompt;
* conversation summary;
* tenant ID;
* task type;
* retrieved documents;
* requested tools;
* language;
* explicit capability hints.

Search an expert registry using:

* approximate nearest-neighbor lookup;
* product-key lookup;
* hierarchical domain routing;
* explicit tenant/domain filters.

This returns a small candidate set.

For example:

```text
Total registry:          2,000,000 experts
Authorized for tenant:         47 experts
Retrieved candidates:           8 experts
Token-level active:             2 experts
```

The model never compares every token against all two million experts.

## Stage 2: token-level routing

Normal top-k routing operates only over the candidates:

```python
candidate_keys = expert_index.lookup(request_embedding, limit=16)

logits = hidden @ candidate_keys.T
selected = logits.topk(k=2)
```

This gives high-resolution token routing without making token cost depend on the entire catalog.

# Experts must live outside the main checkpoint

The deployment manifest should contain expert references, not necessarily their full weights:

```json
{
  "backbone": "foundation-v12",
  "shared_experts": [
    "general-v8"
  ],
  "expert_registry": "registry-2026-06",
  "default_candidate_limit": 16,
  "token_top_k": 2
}
```

Store expert weights in tiers:

```text
GPU:
    hot experts currently serving requests

CPU RAM:
    warm experts likely to be used soon

local NVMe:
    colder experts for the current node or tenant

distributed object store:
    complete expert registry
```

When a request selects an expert:

1. Check GPU cache.
2. Check CPU cache.
3. Read from local NVMe.
4. Fetch from distributed storage if absent.
5. Pin the expert for the session or request batch.

That makes storage horizontally scalable. Adding more expert-storage nodes increases catalog capacity without increasing every inference node’s memory.

# Batch by expert affinity

Millions of experts create a batching problem.

Suppose 64 requests each select different experts. You would get poor GPU utilization because every expert sees very few tokens.

The scheduler should group requests by:

* selected expert;
* tenant;
* model generation;
* sequence-length bucket;
* cached expert availability.

For example:

```text
queue:
    expert embedded-v4:  31 requests
    expert math-v9:      22 requests
    expert legal-se-v2:  14 requests
```

Dispatch batches when:

* enough tokens are ready;
* a latency deadline is reached;
* the expert is already resident.

The expert registry therefore needs both a semantic router and a serving scheduler.

# Three scales of experts

Do not treat all experts identically.

## 1. Micro-experts

These are tiny units, perhaps a neuron or small low-rank block.

Suitable for:

* millions of experts;
* fine-grained pattern storage;
* token-level retrieval;
* PEER-style product-key routing.

PEER’s result supports this scale and shows that very large pools of tiny learned experts can remain computationally sparse. ([arXiv][1])

## 2. Domain experts

These are complete FFN expert paths spanning many Transformer layers.

Suitable for:

* code;
* mathematics;
* embedded systems;
* medicine;
* individual languages;
* industrial domains.

These may contain hundreds of millions of parameters. You could keep thousands or tens of thousands in an external registry, but only a few should be loaded for a given request.

## 3. Tenant experts

These are private client modules.

Suitable for:

* company terminology;
* internal reasoning styles;
* tool use;
* coding conventions;
* document interpretation.

Tenant filtering should reduce the expert search space before semantic routing.

A user in tenant A should never compete against or even expose routing scores from tenant B’s private experts.

# Append-only experts require calibrated router scores

There is a subtle problem when independently trained experts are added over time.

Expert A may emit scores around:

```text
1.0 to 3.0
```

while expert B emits:

```text
8.0 to 12.0
```

Expert B would dominate softmax even if it were irrelevant.

Every expert therefore needs calibration against a common frozen anchor.

Store:

```text
router key
router bias
router temperature
score mean
score standard deviation
anchor-relative margin
domain prototypes
```

Instead of trusting raw scores, normalize them:

[
z_e(x)=
\frac{s_e(x)-\mu_e}{\sigma_e}
]

or train each expert to produce an anchor-relative score:

[
m_e(x)=s_e(x)-s_{\text{anchor}}(x)
]

Then require a positive threshold before selection:

```python
if expert_margin < activation_threshold:
    route_to_anchor()
```

This allows the model to abstain rather than forcing an irrelevant specialist.

FlexOlmo shows that independently trained experts can be composed using a common frozen public anchor and domain-informed routing, including selectively including or excluding experts at inference. ([arXiv][2])

# Why capability does not improve linearly forever

Adding an expert only helps when all of these are true:

1. It sees useful, nonredundant training data.
2. It learns something the existing system does not already do.
3. The shared backbone can represent and use that capability.
4. The router reliably selects it.
5. It does not degrade other experts when composed.
6. The evaluation suite can detect whether it actually helps.

Adding expert number 10,001 with data nearly identical to experts 2,103 and 7,942 may add essentially nothing.

Research on MoE scaling reports both performance improvements from increasing expert count and diminishing returns at very large counts; more experts also increase model-memory requirements and generally require more training data to use effectively. ([arXiv][3])

Limited data per expert also causes undertraining, overfitting, routing sparsity and redundant experts. Work on expert clustering specifically identifies sparse data allocation and overfitting as scaling problems when expert count grows faster than useful data. ([arXiv][4])

# The shared backbone becomes the ceiling

An expert usually replaces or augments feed-forward layers. It does not independently replace:

* attention;
* tokenizer;
* embeddings;
* context length;
* positional encoding;
* global reasoning depth;
* tool protocol handling;
* multimodal encoders;
* output vocabulary.

If the backbone cannot represent some new concept or behavior, another FFN expert may not solve it.

For example, adding an expert may help with:

* Rust syntax;
* Swedish legal terminology;
* RF engineering;
* internal API conventions.

It may not be enough to add:

* vision understanding to a text-only backbone;
* reliable million-token memory;
* a new reasoning token protocol;
* entirely new tool-control behavior;
* stronger global planning.

Recent modular post-training work found that freezing all non-FFN shared parameters was insufficient for some behavioral capabilities; attention and other shared components had to be progressively adapted. ([arXiv][5])

So the architecture needs **backbone generations**.

```text
Foundation v1
    experts v1.x

Foundation v2
    larger context
    improved attention
    better tokenizer
    stronger reasoning
    experts v2.x
```

Experts are ABI-bound to a backbone generation.

# Migrating experts to a new backbone

An old expert generally cannot be loaded into a new backbone unless dimensions and hidden representations remain compatible.

Use one of these methods.

## Retraining from original data

Best quality, but the client may need to rerun training.

## Teacher distillation

Run the old backbone plus expert as a teacher and train a new expert against the new backbone:

[
L =
L_{\mathrm{task}}
+
\lambda D_{\mathrm{KL}}
\left(
p_{\mathrm{old}}
\parallel
p_{\mathrm{new}}
\right)
]

This may use:

* original client data locally;
* synthetic prompts;
* logged non-sensitive examples;
* generated challenge sets.

## Compatibility projection

Learn an adapter between old and new hidden spaces:

[
h_{\mathrm{old}} \approx A h_{\mathrm{new}}
]

Then:

```text
new hidden state
     |
compatibility adapter
     |
old expert
     |
inverse/output adapter
```

This is convenient but normally inferior to native migration.

## Keep old runtimes

For rarely used legacy experts, retain an old inference pool rather than migrating immediately.

# Expert lifecycle management is mandatory

An unlimited append-only registry would eventually become garbage.

Every expert should periodically be classified as:

```text
active
candidate
canary
redundant
superseded
unsafe
deprecated
archived
```

Measure:

* routing frequency;
* marginal loss improvement;
* benchmark delta;
* latency impact;
* cache hit rate;
* overlap with other experts;
* tenant usage;
* security incidents;
* age and backbone compatibility.

## Merge redundant experts

When several experts are very similar:

1. Collect examples on which they activate.
2. Train one consolidated expert.
3. Distill outputs from the originals.
4. Validate it.
5. Replace the cluster with one expert.

## Archive rarely used experts

Keep metadata and weights in cheap object storage but exclude them from the default retrieval index.

## Replace, do not endlessly append

Maintain logical expert identities:

```text
embedded-systems:
    v1 archived
    v2 deprecated
    v3 current
```

Modular post-training research has demonstrated adding and replacing domain experts without retraining prior experts, but that does not remove the need for validation and version management. ([arXiv][5])

# Capability should be measured as marginal utility

Do not count experts. Measure what they add.

For expert (e) and evaluation domain (D):

[
U(e,D)=
L_{\text{without }e}(D)
-----------------------

L_{\text{with }e}(D)
]

Positive utility means the expert improves prediction.

Also measure benchmark difference:

[
\Delta B(e,D)=
B_{\text{with }e}(D)
--------------------

B_{\text{without }e}(D)
]

An expert should be admitted only when it passes thresholds such as:

```text
domain improvement       > required mral regression       < allowed maximum
security tests           passed
router precision         > required minimum
router recall            > required minimum
duplicate overlap        < allowed maximum
```

An expert that exists but is never correctly selected adds storage, not capability.

# Use RAG and tools for the truly unlimited part

Weights should not carry every fact.

A scalable capability stack is:

```text
shared backbone:
    general language and reasoning

experts:
    specialized transformations and domain behavior

retrieval:
    current factual knowledge and private documents

tools:
    calculators, compilers, databases, APIs, simulators

memory:
    session and user-specific state
```

This distinction matters.

Adding every customer document as a new expert is wasteful. A client expert should learn how to interpret that client’s documents, while the documents themselves remain in retrieval storage.

Then information capacity scales with the document store rather than model parameters.

# Recommended system

```text
                         Expert registry
                    millions of expert records
                               |
         tenant filter + domain/capability retrieval
                               |
                    8–64 candidate experts
                               |
                  expert cache and load manager
                               |
                  per-layer top-k token router
                               |
                anchor + 1–2 selected specialists
                               |
              retrieval context and external tools
                               |
                            output
```

And operationally:

```text
Every request:
    filter by authorization
    retrieve candidate experts
    load/pin selected modules
    execute sparse routing
    record routing and utility metrics

Every new expert:
    verify ABI
    validate capability
    calibrate router score
    canary
    register
    monitor
    replace or archive later

Every backbone generation:
    retrain or distill important experts
    retain compatibility where useful
    retire obsolete generations gradually
```

## Bottom line

You can scale toward:

* millions of tiny experts;
* thousands of full domain experts;
* huge numbers of tenant-specific adapters;
* unbounded external documents and tools.

But you cannot obtain unlimited intelligence by endlessly appending experts.

The scalable promise is narrower and still powerful:

> **The system can keep acquiring specialized capabilities without retraining or loading the entire accumulated model, while active compute per request remains bounded.**

General capability eventually requires a better shared backbone, better routing, new modalities, stronger reasoning training, better tools and better data—not merely more experts.

[1]: https://arxiv.org/html/2407.04153v1 "Mixture of A Million Experts"
[2]: https://arxiv.org/abs/2507.07024 "[2507.07024] FlexOlmo: Open Language Models for Flexible Data Use"
[3]: https://arxiv.org/html/2502.05172v1 "Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient"
[4]: https://arxiv.org/abs/2207.09094?utm_source=chatgpt.com "MoEC: Mixture of Expert Clusters"
[5]: https://arxiv.org/pdf/2604.18473 "Train Separately, Merge Together: Modular Post-Training with Mixture-of-Experts"
