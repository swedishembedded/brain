# Mathematical Seam Architecture for Brain

## Reusing, composing, distributing and distilling heterogeneous AI models

### Abstract

A model seam is any internal or external variable through which one computation can be separated from another. Examples include transformer hidden states, encoder outputs, KV caches, object detections, depth maps, codec tokens, forecast distributions, world-model states and 3D Gaussian parameters.

The common mistake is to treat these seams as interchangeable tensors. They are not. A useful seam has a mathematical meaning:

* a sufficient statistic of an input;
* a predictive distribution over future observations;
* a compressed code with a known decoder;
* a coordinate-covariant geometric field;
* an estimate of a latent physical state;
* a routing distribution over experts;
* a cached state of a sequential computation;
* or an action-conditioned transition model.

Once that meaning is explicit, Brain can use the seam to:

1. expose valuable information as a service;
2. partition a pipeline between machines or accelerators;
3. attach new heads or decoders;
4. align independently trained models;
5. distil expensive models into small deployable models;
6. train controllers from predictive models;
7. construct multimodal world models;
8. preserve uncertainty through a composed pipeline;
9. validate whether a proposed composition is mathematically legitimate.

The central design recommendation is that Brain should standardize **typed statistical contracts**, not generic hidden tensors.

---

# 1. A mathematical definition of a model seam

Consider a model expressed as a composition:

[
f(x)=f_n\circ f_{n-1}\circ \cdots \circ f_1(x).
]

At boundary (k),

[
z_k=f_k\circ \cdots \circ f_1(x)
]

is a seam.

The obvious software interpretation is that (z_k) can be serialized and passed to another component. The more important interpretation is statistical:

[
X \rightarrow Z_k \rightarrow Y.
]

Here:

* (X) is the original observation;
* (Z_k) is the seam representation;
* (Y) is a downstream variable of interest.

A seam is sufficient for (Y) when:

[
p(Y\mid X)=p(Y\mid Z_k),
]

or equivalently,

[
I(Y;X\mid Z_k)=0.
]

This means that once (Z_k) is known, the original input (X) provides no additional information needed to predict (Y).

Most neural seams are only approximately sufficient:

[
I(Y;X\mid Z_k)\approx 0
]

for the tasks represented in the model’s training distribution.

This immediately explains why a language-model hidden state may be sufficient for predicting the next word but insufficient for reconstructing an exact image. The state retained information useful to the language objective, not necessarily all information present in the source.

The information-bottleneck formulation makes the trade-off explicit:

[
\min_{p(z\mid x)}
I(X;Z)-\beta I(Z;Y).
]

The first term rewards compression. The second rewards retention of task-relevant information. Neural encoders approximately implement different points on this trade-off curve.

## 1.1 Five distinct classes of seam

Brain should distinguish at least five classes.

### Semantic seams

These have explicit meaning outside the originating model:

[
z =
\text{boxes},\ \text{depth},\ \text{transcript},\ \text{forecast},
\text{speaker embedding},\ \text{point cloud}.
]

They are the safest public APIs because their semantics can remain stable while the implementation changes.

### Codec seams

A codec seam has meaning through a paired decoder:

[
z=E(x), \qquad \hat{x}=D(z).
]

Examples include image VAE latents, Mimi tokens and VQ video tokens.

The representation is not independently meaningful, but it has a defined reconstruction contract.

### Predictive-state seams

A predictive state summarizes the history in terms of its implications for the future:

[
z_t = \phi(o_{\leq t},a_{<t}),
]

such that:

[
p(o_{t+1:t+H}\mid o_{\leq t},a_{<t},a_{t:t+H-1})
\approx
p(o_{t+1:t+H}\mid z_t,a_{t:t+H-1}).
]

Forecasting encoders, world-model states and streaming ASR states belong here.

### Task-specific learned latents

These include transformer residual streams and encoder token sequences. They are useful, but their coordinate systems are not inherently identifiable.

Suppose:

[
z'=Az
]

for any invertible matrix (A). A downstream layer can preserve exactly the same model output by replacing its weights (W) with:

[
W'=WA^{-1}.
]

The two models compute the same function even though their hidden states are numerically unrelated. Therefore, “same dimension” does not imply “same semantics.”

### Runtime-state seams

KV caches, convolution caches, diffusion solver states and expert-dispatch buffers describe a partially completed computation.

They should normally be represented by opaque handles rather than stable tensor formats.

---

# 2. What makes two seams composable?

Suppose model (A) produces:

[
z_A \in \mathbb{R}^{n\times d_A}
]

and model (B) expects:

[
z_B \in \mathbb{R}^{m\times d_B}.
]

The problem is not merely to transform the tensor shape. A bridge must reconcile:

* feature coordinates;
* token granularity;
* temporal or spatial resolution;
* positional encoding;
* probability calibration;
* normalization;
* causality;
* missing information;
* invariances;
* decoder expectations.

A general adapter can be written as:

[
z_B = g_\phi(z_A,c),
]

where (c) contains metadata such as timestamps, coordinates, masks or modality identity.

## 2.1 Linear projection

The simplest bridge is:

[
z_B=z_AW+b.
]

This works when the representations are already strongly aligned and only their bases or dimensionalities differ.

It is usually insufficient when one model emits 1,000 image tokens and another expects 77 text-conditioning tokens.

## 2.2 Resampling

A learned resampler transforms variable-length representations:

[
q_j^{(0)} = q_j,\qquad
q_j^{(l+1)}
===========

\operatorname{CrossAttention}
\left(q_j^{(l)},z_A,z_A\right).
]

A fixed collection of learned queries extracts (m) destination tokens from (n) source tokens.

This is appropriate for:

* vision encoder to language decoder;
* long text encoder to image generator;
* event history to forecasting model;
* multi-view features to a scene representation.

## 2.3 Distribution matching

When both seams represent distributions, the adapter can minimize:

[
\mathcal{L}_{KL}
================

D_{\mathrm{KL}}
\left(
p_A(z\mid x);|;p_B(z\mid x)
\right).
]

For categorical outputs:

[
\mathcal{L}_{KD}
================

T^2
D_{\mathrm{KL}}
\left(
\operatorname{softmax}(l_A/T)
;|;
\operatorname{softmax}(l_B/T)
\right).
]

The temperature (T) exposes relative probabilities that would be hidden by a hard label. This is the classical knowledge-distillation mechanism.

## 2.4 Feature matching

For paired samples:

[
\mathcal{L}_{feat}
==================

\left|
P_\phi(z_A)-z_B
\right|_2^2.
]

This assumes that direct coordinate matching is meaningful. A safer alternative matches relations:

[
G_A=z_Az_A^\top,\qquad
G_B=z_Bz_B^\top,
]

[
\mathcal{L}_{rel}
=================

|G_A-G_B|_F^2.
]

This preserves similarities between tokens without requiring identical hidden coordinates.

## 2.5 Contrastive alignment

For paired examples ((x_i^A,x_i^B)), normalized embeddings are trained using:

[
\mathcal{L}_{NCE}
=================

-\sum_i
\log
\frac{
\exp(s(z_i^A,z_i^B)/\tau)
}{
\sum_j \exp(s(z_i^A,z_j^B)/\tau)
}.
]

This creates a shared semantic geometry suitable for retrieval and coarse conditioning.

It does not guarantee that either embedding contains enough information to drive a high-fidelity decoder.

## 2.6 Task-level alignment

The strongest test is whether the downstream task works:

[
\mathcal{L}
===========

\mathcal{L}*{task}
\left(
D_B(g*\phi(E_A(x))),y
\right).
]

An adapter that minimizes hidden-state distance but performs poorly on the real task is useless. Brain should therefore treat representation losses as auxiliary, not authoritative.

---

# 3. Distributed execution as a seam optimization problem

Let a seam transfer (S) bytes, incur network latency (L), bandwidth (B), serialization cost (T_s), and occur (R) times per request.

The communication cost is approximately:

[
T_{\text{comm}}
===============

R\left(
L+\frac{S}{B}+T_s
\right).
]

This equation explains most good and bad model partitions.

## 3.1 Encoder-decoder split

An encoder output crosses once:

[
R=1.
]

Even if the representation is moderately large, this can be practical over ordinary networks.

## 3.2 Decoder layer split

For autoregressive generation, an intermediate hidden state may cross once for every generated token:

[
R=T_{\text{generated}}.
]

Pipeline latency then accumulates directly into token latency. Such splits generally require NVLink, InfiniBand or a similarly low-latency fabric.

## 3.3 MoE expert split

An MoE may communicate at every MoE layer:

[
R=T_{\text{tokens}}\times N_{\text{MoE layers}}.
]

Expert parallelism is therefore highly sensitive to all-to-all latency.

## 3.4 Codec split

A low-rate codec may produce only tens of tokens per second. It is an excellent cross-machine seam because:

[
S_{\text{codec}}\ll S_{\text{waveform}}.
]

## 3.5 Diffusion and flow loops

If a latent is transferred at every solver step:

[
R=N_{\text{generation steps}}.
]

The denoising or flow-integration loop should normally remain on one accelerator group. Encoders and final decoders are much better remote boundaries.

---

# 4. Language and sequence-model seams

## 4.1 GPT decoder

A GPT-style decoder models:

[
p(x_{1:T})
==========

\prod_{t=1}^{T}
p(x_t\mid x_{<t}).
]

Each transformer layer computes approximately:

[
H^{(l+1)}
=========

H^{(l)}
+
\operatorname{Attention}
\left(
\operatorname{Norm}(H^{(l)})
\right)
+
\operatorname{MLP}(\cdot).
]

The final logits are:

[
\ell_t=W_o h_t^{(L)}+b_o,
]

[
p(x_{t+1}\mid x_{\leq t})
=========================

\operatorname{softmax}(\ell_t).
]

GPT models are decoder-only autoregressive transformers.

### Residual-stream seam

The residual stream (h_t^{(l)}) is a distributed representation of everything the first (l) layers have computed about token (t).

It can support:

* task heads;
* reward or value heads;
* action heads;
* anomaly probes;
* feature steering;
* early-exit decoders;
* cross-model distillation.

It should not be interpreted as a human-readable semantic vector. Its coordinate system is checkpoint-specific.

### Logit seam

Logits have a precise meaning:

[
\ell_i-\ell_j
=============

\log
\frac{p_i}{p_j}.
]

This makes logits highly valuable for:

* uncertainty estimation;
* distillation;
* constrained decoding;
* ensemble combination;
* token-level confidence;
* speculative verification.

### KV-cache seam

For layer (l), the cache stores:

[
K_{\leq t}^{(l)},V_{\leq t}^{(l)}.
]

This is a computational sufficient state for continuing the exact same decoder:

[
p(x_{t+1}\mid x_{\leq t})
=========================

F(x_t,C_t).
]

It is not a general semantic representation. A cache is tied to:

* exact model weights;
* layer layout;
* RoPE convention;
* cache precision;
* token history;
* batch position.

Brain should expose it as an opaque session state.

### Valuable compositions

A GPT hidden sequence can condition another model through:

[
c=g_\phi(H^{(L)}).
]

Potential products include:

* language-to-action controller;
* language-conditioned image decoder;
* language-conditioned audio generator;
* code model with a learned execution-state head;
* structured event extractor;
* reward model distilled from a larger reasoner.

The adapter should use the whole sequence where fine token alignment matters. A pooled embedding generally loses order and detail.

---

## 4.2 Qwen3 decoder

Qwen3 includes dense and sparse-MoE decoder models and unifies thinking and non-thinking behavior in one family.

Mathematically, its core seams are the same as the GPT seams, but additional control variables become useful.

### Reasoning-budget seam

Let (b) represent a reasoning budget or mode:

[
p(y\mid x,b).
]

This creates a controllable compute-quality frontier. Brain can learn a router:

[
b^*=r_\phi(x,c_{\text{latency}},c_{\text{risk}}).
]

A small model can choose whether a query requires:

* direct decoding;
* extended reasoning;
* tool execution;
* a larger remote model.

This router can be distilled from observed marginal utility:

[
\Delta U(b)
===========

U(y_b)-U(y_{b-\Delta b}).
]

The system can stop reasoning when:

[
\mathbb{E}[\Delta U(b)] < \lambda_{\text{compute}}.
]

### Qwen semantic conditioning

Qwen hidden states can be adapted into other modalities, but a useful objective should combine several terms:

[
\mathcal{L}
===========

\lambda_1\mathcal{L}*{target}
+
\lambda_2\mathcal{L}*{contrastive}
+
\lambda_3\mathcal{L}*{feature}
+
\lambda_4\mathcal{L}*{preservation}.
]

The preservation term prevents a new adapter from damaging the original language model.

---

## 4.3 Sparse MoE transformer

An MoE layer computes:

[
r(x)=\operatorname{softmax}(W_rx),
]

selects:

[
\mathcal{K}(x)=\operatorname{TopK}(r(x)),
]

and produces:

[
y
=

\sum_{i\in\mathcal{K}(x)}
\alpha_i(x)E_i(x).
]

Qwen3’s MoE models use fine-grained routed experts, demonstrating this general pattern.

### Router-probability seam

The router output is a conditional distribution over computational specialists:

[
p(e=i\mid x).
]

It can be exploited for:

* load balancing;
* domain analysis;
* expert pruning;
* anomaly detection;
* expert caching;
* adaptive precision;
* routing requests to physical accelerators.

The entropy:

[
H(E\mid x)
==========

-\sum_i p_i(x)\log p_i(x)
]

measures routing uncertainty.

Low entropy means a token strongly activates a particular specialization. High entropy may indicate ambiguity or poor expert separation.

### Expert activation as a representation

A sparse expert-usage vector:

[
s(x)\in\mathbb{R}^{N_e}
]

can serve as a coarse representation of which learned skills the input invokes.

This is potentially useful for:

* request classification;
* capability discovery;
* dataset clustering;
* selecting LoRA adapters;
* detecting distribution shift.

It is not proof that an expert has a specific interpretable function.

### Adding experts

A domain expert (E_{new}) can be added while freezing the backbone:

[
y
=

\sum_i \alpha_iE_i(x)+\alpha_{new}E_{new}(x).
]

Training can be restricted to:

* the new expert;
* router parameters;
* a load-balancing regularizer;
* a retention loss over old data.

This provides a natural path for continuous specialization.

### Distributed execution

Expert dispatch is valuable only when the reduction in active compute exceeds communication cost. Brain should model:

[
T_{\text{MoE}}
==============

T_{\text{route}}
+
T_{\text{dispatch}}
+
\max_i T_{E_i}
+
T_{\text{combine}}.
]

The router should be scheduler-aware so that its objective includes real accelerator cost, not just model loss.

---

## 4.4 GLM-5.2 decoder

GLM-5.2 is presented by Z.ai as a long-horizon, long-context model. Detailed public architectural documentation is currently less complete than for Qwen3, so Brain should not hard-code unverified internal assumptions about its exact attention, cache or MoE structure.

It should initially expose the same reliable decoder contracts:

[
\text{tokens}
\rightarrow
\text{hidden sequence}
\rightarrow
\text{logits}
\rightarrow
\text{generated actions}.
]

Potential seams include:

* input embeddings;
* layer-group boundaries;
* final hidden states;
* logits;
* tool-call outputs;
* long-context cache handles;
* speculative draft and verification interfaces, where supported by the implementation.

The key distinction is between a **model-family API** and a **checkpoint-derived execution description**. Brain should inspect the actual graph before exposing internal GLM-specific seams.

---

## 4.5 PID event/effect transformer

No public model with this exact name was identified. PIDformer is a different architecture that introduces proportional, integral and derivative feedback concepts into transformer dynamics.

For Brain, a PID event/effect transformer can be formally defined as an event-conditioned predictive-state model.

Let each event be:

[
e_i=(\tau_i,c_i,m_i,v_i),
]

where:

* (\tau_i) is time;
* (c_i) is event type;
* (m_i) is magnitude;
* (v_i) contains contextual variables.

Define a latent state:

[
z_t=F_\theta(e_{\leq t}).
]

A useful PID-inspired decomposition is:

[
P_t = \psi_P(e_t),
]

[
I_t=\lambda I_{t-1}+\psi_I(e_t),
]

[
D_t=\psi_D(e_t-e_{t-1}),
]

[
z_t=\operatorname{Fuse}(P_t,I_t,D_t).
]

The model then predicts an effect distribution:

[
p(\Delta y,\Delta t,c_{\text{effect}}\mid z_t).
]

### Mathematical meaning of the seams

The proportional component estimates immediate effect.

The integral component estimates accumulated exposure or persistent influence.

The derivative component estimates change in event intensity or direction.

These outputs can drive:

* anomaly detection;
* forecast covariates;
* regime detection;
* causal hypothesis generation;
* adaptive control;
* experiment selection.

The model must not equate attention or correlation with causation. A causal effect requires intervention assumptions:

[
p(Y\mid \operatorname{do}(A=a))
\neq
p(Y\mid A=a)
]

in general.

A more defensible event/effect model should emit:

[
\hat{\tau}(a)
=============

## \mathbb{E}[Y\mid \operatorname{do}(A=a)]

\mathbb{E}[Y\mid \operatorname{do}(A=a_0)],
]

together with uncertainty and a statement of the identification assumptions.

---

## 4.6 Seq2seq encoder-decoder transformer

A sequence-to-sequence model factorizes:

[
H_x=E_\theta(x_{1:N}),
]

[
p(y_{1:T}\mid x)
================

\prod_{t=1}^{T}
p(y_t\mid y_{<t},H_x).
]

The transformer was originally introduced in this encoder-decoder form.

### Encoder seam

The encoder sequence:

[
H_x\in\mathbb{R}^{N\times d}
]

is one of the strongest reusable neural seams because it is computed once and can be consumed many times.

It can feed:

* multiple language decoders;
* classification heads;
* image generators;
* TTS models;
* retrieval models;
* structured extractors.

### Cross-attention seam

The decoder computes:

[
Q=H_yW_Q,\qquad
K=H_xW_K,\qquad
V=H_xW_V,
]

[
A=\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right),
]

[
C=AV.
]

Here (A) describes how decoder positions query the encoded input. It can support alignment and diagnostics, although raw attention is not a guaranteed explanation.

### Multi-decoder training

A shared encoder can be trained with several decoder losses:

[
\mathcal{L}
===========

\sum_m \lambda_m
\mathcal{L}_m(D_m(E(x)),y_m).
]

This is the cleanest architecture for Brain’s proposed universal semantic encoder concept.

---

## 4.7 LFM2.5-Encoder

LFM2.5-Encoder is a bidirectional encoder based on the LFM2 hybrid architecture, interleaving gated short-convolution blocks with grouped-query attention. The causal mask is replaced by bidirectional attention, and the released encoder supports masked-language-model training and downstream fine-tuning.

Its mathematical output is:

[
H=E_\theta(x_{1:N})
\in\mathbb{R}^{N\times d}.
]

### Token-level states

Each (h_i) is conditioned on both left and right context:

[
h_i=f(x_{1:N},i).
]

This is superior to a causal decoder state for tasks where the complete input is available:

* classification;
* named-entity recognition;
* event extraction;
* semantic tagging;
* reranking;
* dense retrieval.

### Pooled embedding

A pooled representation may be:

[
z=\operatorname{Pool}(H).
]

It can be used for routing or coarse semantic conditioning.

However:

[
\operatorname{Pool}(H)
]

is generally not sufficient for tasks requiring exact token-level detail.

### ColBERT-style late interaction

For query tokens (q_i) and document tokens (d_j):

[
s(q,d)
======

\sum_i\max_j
q_i^\top d_j.
]

This retains token-level semantic structure and is stronger than a single pooled vector for retrieval.

### Potential Brain uses

LFM2.5-Encoder is a good candidate for:

* a shared text-conditioning service;
* model routing;
* event extraction;
* semantic retrieval;
* instruction embeddings for world models;
* distillation into small domain encoders.

Its output should be versioned by checkpoint because an upgrade can rotate or reorganize the hidden space even if downstream benchmark quality improves.

---

# 5. Compression and latent-code seams

## 5.1 Bottleneck autoencoder

A deterministic autoencoder computes:

[
z=E_\phi(x),\qquad
\hat{x}=D_\theta(z).
]

Training minimizes:

[
\mathcal{L}_{rec}
=================

d(x,\hat{x}).
]

A variational autoencoder instead learns:

[
q_\phi(z\mid x)
]

and:

[
p_\theta(x\mid z),
]

using the evidence lower bound:

[
\mathcal{L}_{ELBO}
==================

\mathbb{E}*{q*\phi(z\mid x)}
[\log p_\theta(x\mid z)]
------------------------

D_{KL}
\left(
q_\phi(z\mid x)|p(z)
\right).
]

### What the bottleneck means

The latent (z) is an economical description of (x) under the distortion metric used during training.

If the distortion is pixel MSE, the latent prioritizes average pixel reconstruction.

If the distortion includes perceptual losses, it prioritizes perceptual similarity.

If it includes semantic losses, it may discard pixel details while preserving identity or content.

There is no universally meaningful autoencoder latent.

### Rate-distortion interpretation

A codec solves approximately:

[
\min
D+\lambda R,
]

where:

* (D) is reconstruction distortion;
* (R) is latent bitrate.

The seam therefore represents a rate-distortion operating point.

### Exploitation

A bottleneck latent can support:

* edge-to-cloud compression;
* latent forecasting;
* anomaly detection through reconstruction error;
* generative modeling;
* new decoder training;
* domain conversion;
* latent control policies.

An anomaly score can be:

[
s(x)=d(x,D(E(x))).
]

A stronger score also measures latent likelihood:

[
s(x)
====

-\log p(z)+\lambda d(x,\hat{x}).
]

### Cross-decoder reuse

A new decoder (D_\psi') can be trained:

[
\hat y=D_\psi'(E_\phi(x)).
]

This is most useful when the encoder latent is sufficient for (y). The sufficiency must be tested rather than assumed.

---

# 6. Vision and geometry seams

## 6.1 Qwen-VL and Moondream

A VLM typically computes:

[
H_v=E_v(I),
]

[
Z_v=P(H_v),
]

[
p(y\mid I,x)
============

D([Z_v,E_t(x)]).
]

Qwen3-VL uses multi-level vision features and spatial-temporal positional mechanisms to integrate image and video content with its language model. Moondream follows the broad compact vision-encoder-plus-language-decoder pattern.

### Vision-feature seam

The raw visual feature sequence retains:

* local appearance;
* spatial layout;
* object parts;
* texture;
* some geometry.

Its tokens are usually arranged according to an image patch grid.

### Projected visual-token seam

After projection, tokens are aligned with the language-model residual space:

[
Z_v=H_vW_p+b.
]

These tokens are useful for language reasoning but may have lost reconstruction-critical information.

### Multi-level features

Let (H_v^{(l)}) be features from vision layer (l). A multi-level seam may be:

[
Z_v
===

\sum_{l\in\mathcal{S}}
\alpha_lP_l(H_v^{(l)}).
]

Earlier layers retain fine local structure. Later layers retain more semantic abstraction.

This enables separate heads for:

* detection;
* OCR;
* segmentation;
* scene description;
* spatial reasoning.

### Valuable combinations

VLM features can be fused with:

* YOLO objects;
* depth estimates;
* 3D points;
* world-model states;
* image-generation conditioning;
* action policies.

For example, an object embedding can be constructed from a mask (M_j):

[
z_j
===

\frac{
\sum_{u,v}M_j(u,v)H_v(u,v)
}{
\sum_{u,v}M_j(u,v)
}.
]

This gives each detected object a semantic representation.

---

## 6.2 YOLOv8-style detector

YOLOv8 uses a backbone, multi-scale feature-fusion neck and anchor-free detection head.

Let the backbone produce:

[
F_s=B_s(I)
]

at different scales (s). The neck produces fused features:

[
P_s=N_s(F_1,\ldots,F_S).
]

The head estimates a set:

[
\mathcal{O}
===========

{(b_i,c_i,p_i)}_{i=1}^{N},
]

where:

* (b_i) is a box;
* (c_i) is a class;
* (p_i) is confidence.

### Feature-pyramid seam

The pyramid is approximately equivariant to image translation and separates spatial scales.

Fine features are useful for small objects. Coarse features capture larger receptive fields.

Potential reuse includes:

* segmentation heads;
* pose heads;
* tracking embeddings;
* anomaly detection;
* object depth estimation;
* visual servoing.

### Detection seam

Post-processed detections are a stable semantic interface.

They can be modeled as a random finite set:

[
p(\mathcal{O}\mid I).
]

This is more accurate than pretending the output is a fixed-length vector.

### Object-centric state

For control, create an object state:

[
s_i
===

[
x_i,y_i,w_i,h_i,
\dot{x}_i,\dot{y}_i,
c_i,p_i,z_i
].
]

A tracker updates:

[
p(s_{i,t}\mid s_{i,t-1},\mathcal{O}_t).
]

This is often much more useful to a controller than raw image features.

---

## 6.3 ZipDepth

ZipDepth is a compact zero-shot monocular-depth model using a lightweight encoder-decoder and knowledge distillation from a larger foundation model.

It estimates:

[
\hat d(u,v)=F_\theta(I).
]

### Relative-depth ambiguity

From a single uncalibrated image, absolute metric depth is generally not identifiable without priors.

A relative depth prediction may be valid only up to scale:

[
d'(u,v)=a,d(u,v),
]

or for some representations, scale and shift:

[
d'(u,v)=a,d(u,v)+b.
]

Brain must therefore attach metadata identifying:

* metric or relative depth;
* depth or inverse depth;
* scale;
* shift;
* valid range;
* uncertainty;
* camera calibration.

### Geometric lifting

For camera intrinsics (K), pixel:

[
\tilde p=[u,v,1]^\top
]

and metric depth (d), the camera-frame point is:

[
P_c=dK^{-1}\tilde p.
]

Combining depth with a YOLO mask (M_i) gives object distance:

[
\bar d_i
========

\operatorname{median}
{d(u,v):M_i(u,v)=1}.
]

This can drive collision avoidance, manipulation and measurement.

### Distillation seam

ZipDepth itself demonstrates a valuable Brain pattern:

[
\text{large geometric teacher}
\rightarrow
\text{compact embedded student}.
]

The student can be trained with:

[
\mathcal{L}
===========

\mathcal{L}*{depth}
+
\lambda_1\mathcal{L}*{feature}
+
\lambda_2\mathcal{L}*{gradient}
+
\lambda_3\mathcal{L}*{normal}.
]

Depth gradients preserve boundaries, while normal consistency preserves local geometry.

---

## 6.4 WorldMirror-2

WorldMirror-2 is a feed-forward multi-view geometry model that can output world-coordinate point clouds, camera-frame depth, surface normals, camera parameters and 3D Gaussian attributes.

This model exposes unusually rich mathematical seams.

### Camera seam

A camera can be described by:

[
K=
\begin{bmatrix}
f_x & 0 & c_x\
0 & f_y & c_y\
0 & 0 & 1
\end{bmatrix},
]

and pose:

[
T_{cw}
======

\begin{bmatrix}
R_{cw} & t_{cw}\
0 & 1
\end{bmatrix}.
]

These are stable geometric interfaces if coordinate conventions are explicit.

### Depth seam

Per-view depth lifts image pixels into 3D.

### Surface-normal seam

A normal:

[
n(u,v)\in S^2
]

describes local surface orientation.

It enables:

* lighting estimation;
* contact reasoning;
* planar segmentation;
* material processing;
* mesh reconstruction.

### Point-cloud seam

A point cloud is:

[
\mathcal P
==========

{(p_i,f_i,\sigma_i)},
]

where (f_i) may contain colour or semantics and (\sigma_i) uncertainty.

### Multi-view correspondence

If feature (z_i^a) in view (a) corresponds to (z_j^b) in view (b), Brain can establish persistent object and surface identities.

### Valuable exploitation

WorldMirror outputs can initialize:

* 3D Gaussian scenes;
* occupancy maps;
* SLAM backends;
* collision models;
* robotic scene graphs;
* semantic digital twins.

A semantic 3D point can be formed by fusing VLM features:

[
f(P)
====

\frac{
\sum_v w_v(P)f_v(\pi_v(P))
}{
\sum_v w_v(P)
}.
]

The weights (w_v) can depend on visibility, angle and model confidence.

---

## 6.5 3D Gaussian Splatting

A 3D Gaussian primitive has mean:

[
\mu_i\in\mathbb{R}^3,
]

covariance:

[
\Sigma_i
========

R_iS_iS_i^\top R_i^\top,
]

opacity:

[
\alpha_i,
]

and view-dependent colour parameters (c_i).

The spatial density is:

[
G_i(x)
======

\exp
\left(
-\frac{1}{2}
(x-\mu_i)^\top
\Sigma_i^{-1}
(x-\mu_i)
\right).
]

3DGS optimizes anisotropic Gaussians and uses visibility-aware splatting for real-time rendering.

### Rendering seam

After projection into the image, sorted Gaussian contributions are alpha-composited:

[
C(p)
====

\sum_i
T_i(p)\alpha_i(p)c_i,
]

where:

[
T_i(p)
======

\prod_{j<i}
(1-\alpha_j(p)).
]

### Why this is a strong seam

Unlike an anonymous hidden state, Gaussian parameters have explicit scene meaning.

They can be:

* spatially indexed;
* edited;
* pruned;
* transmitted;
* assigned object IDs;
* decorated with semantic features;
* rendered by independent implementations.

### Semantic Gaussians

Attach an embedding:

[
s_i\in\mathbb{R}^{d_s}
]

to each Gaussian.

A rendered semantic feature is:

[
S(p)
====

\sum_iT_i(p)\alpha_i(p)s_i.
]

This enables pixel queries and 3D queries against the same scene.

### Distributed rendering

Partition Gaussians spatially:

[
\mathcal G
==========

\bigcup_k\mathcal G_k.
]

Each worker renders local contributions. Results are depth-sorted and composited.

Large scenes can also use hierarchical levels of detail, a direction already explored for scalable Gaussian representations.

---

# 7. Image-generation seams

## 7.1 Z-Image

Z-Image is a 6B image-generation model based on a scalable single-stream diffusion transformer. It processes text, visual-semantic and image-latent tokens in a unified transformer stream.

A latent image generator can be represented as:

[
z_0=E_{\mathrm{VAE}}(I),
]

with a noised or interpolated state (z_t), and a model:

[
v_\theta(z_t,t,c)
]

that predicts a denoising or flow direction.

### Unified-token seam

Let:

[
X=
[
X_{\text{text}};
X_{\text{visual}};
X_{\text{latent}}
].
]

Self-attention operates over all token types:

[
H^{l+1}
=======

F_l(H^l,t,\text{type embeddings}).
]

This creates a natural insertion seam for new modalities:

* depth tokens;
* pose tokens;
* segmentation tokens;
* 3D scene tokens;
* sound-event tokens.

### Adapter training

A new conditioning encoder (E_m) can be attached through:

[
X_m=P_\phi(E_m(x_m)).
]

Training can begin with the generator frozen:

[
\min_\phi
\mathbb{E}
[
|v-v_\theta(z_t,t,c,X_m)|^2
].
]

Then LoRA modules can be unfrozen to increase capacity.

### Stream interference

Because token types share transformer parameters, modifying one modality may damage another.

Training should include a preservation objective:

[
\mathcal{L}_{pres}
==================

\mathbb{E}*{x\sim D*{old}}
|
v_{\theta'}(x)-v_\theta(x)
|^2.
]

---

## 7.2 FLUX.2 Klein

FLUX.2 Klein unifies image generation and editing in compact 4B and 9B variants. The official repository also exposes model variants and an editing-oriented KV-cache path.

Its useful mathematical seams are:

* text embedding sequence;
* reference-image encoding;
* image latent;
* flow state;
* transformer state;
* predicted velocity;
* final decoder latent.

### Flow-matching interpretation

A simple rectified-flow path is:

[
z_t=(1-t)z_0+t\epsilon.
]

The target velocity is:

[
v^*=\epsilon-z_0.
]

The model minimizes:

[
\mathcal{L}_{flow}
==================

\mathbb{E}
|
v_\theta(z_t,t,c)-v^*
|^2.
]

Inference integrates:

[
\frac{dz}{dt}=v_\theta(z,t,c).
]

### Reference-image seam

For editing:

[
c=
[c_{\text{text}},c_{\text{image 1}},\ldots,c_{\text{image N}}].
]

This supports:

* identity transfer;
* style transfer;
* local editing;
* multi-reference composition;
* product visualization.

### Valuable Brain use

The Base variants are appropriate teachers for training:

* tiny domain-specific image generators;
* control adapters;
* subject-preservation modules;
* depth-guided generation;
* industrial synthetic-data generators.

The student need not reproduce the complete image distribution. It can be distilled for one narrow operating domain.

---

# 8. Audio and speech seams

## 8.1 Qwen3-TTS

Qwen3-TTS uses a dual-track language-model architecture and discrete speech tokenizers, including a 12.5 Hz multi-codebook representation and a 25 Hz semantic-oriented representation.

A speech model can factorize:

[
p(c\mid x,s,p)
==============

\prod_t
p(c_t\mid c_{<t},x,s,p),
]

where:

* (x) is text;
* (s) is speaker identity;
* (p) is prosody or style;
* (c_t) is a collection of codec codes.

### Semantic-token seam

Semantic speech tokens primarily encode linguistic content and coarse prosody.

They are useful for:

* speech-language modeling;
* speech translation;
* semantic editing;
* low-bitrate transmission.

### Acoustic-code seam

Residual codebooks progressively add acoustic detail:

[
\hat z_t
========

\sum_{q=1}^{Q}
e_{q,c_{t,q}}.
]

Early codebooks may carry broad speech structure. Later codebooks refine timbre and waveform detail.

### Speaker-conditioning seam

A speaker representation (s) conditions generation:

[
p(c_t\mid c_{<t},x,s).
]

This can be supplied by:

* a learned TTS speaker encoder;
* ECAPA-TDNN;
* a voice-description encoder;
* an enrollment centroid.

An adapter is required because the geometry of an ECAPA embedding is not necessarily the geometry expected by the TTS model.

### Distributed speech synthesis

A practical split is:

[
\text{text/speaker}
\rightarrow
\text{codec-token generator}
\rightarrow
\text{network}
\rightarrow
\text{local codec decoder}.
]

This lowers bandwidth and allows the final waveform decoder to run close to the user.

---

## 8.2 Mimi-style neural codec

Mimi is a streaming neural audio codec that maps 24 kHz audio to a 12.5 Hz low-bitrate token representation. It uses residual vector quantization and semantic distillation.

Let:

[
z_t=E(x)_t.
]

Residual quantization proceeds as:

[
r_t^{(0)}=z_t,
]

[
k_{t,q}
=======

\arg\min_j
|r_t^{(q-1)}-e_{q,j}|^2,
]

[
r_t^{(q)}
=========

r_t^{(q-1)}-e_{q,k_{t,q}}.
]

The quantized latent is:

[
\hat z_t
========

\sum_{q=1}^{Q}
e_{q,k_{t,q}}.
]

The waveform is:

[
\hat x=D(\hat z).
]

### Bitrate

For codebook sizes (K_q) and frame rate (f):

[
R
=

f\sum_q\log_2K_q
\quad \text{bits/s}.
]

### Progressive quality

The decoder can use only the first (q<Q) codebooks:

[
\hat z_t^{(q)}
==============

\sum_{j=1}^{q}e_{j,k_{t,j}}.
]

This enables adaptive bitrate and graceful degradation.

### Codec tokens as model input

A language model can model:

[
p(k_{1:T,1:Q}).
]

This supports:

* TTS;
* speech continuation;
* speech translation;
* voice conversion;
* acoustic-event prediction;
* full-duplex dialogue.

### Compatibility warning

Mimi tokens and Qwen3-TTS tokens are not compatible merely because they have similar frame rates.

Compatibility requires identical or aligned:

* codebooks;
* codebook order;
* residual semantics;
* encoder distribution;
* decoder distribution;
* frame alignment.

A translator would require:

[
T_\phi:
k^{Mimi}\rightarrow k^{QwenTTS}
]

trained against paired audio, ideally with both token and waveform losses.

---

## 8.3 ECAPA-TDNN speaker encoder

ECAPA-TDNN extracts frame-level features and uses channel-dependent attentive statistical pooling to produce a fixed-length speaker embedding.

Let frame features be:

[
h_t=F(x)_t.
]

Attention weights are:

[
a_t
===

\frac{\exp(g(h_t))}
{\sum_j\exp(g(h_j))}.
]

The weighted mean is:

[
\mu
===

\sum_ta_th_t,
]

and variance:

[
\sigma^2
========

\sum_ta_t(h_t-\mu)^2.
]

The speaker embedding is:

[
e=W[\mu,\sigma]+b.
]

### Metric meaning

Speaker verification commonly uses cosine similarity:

[
s(e_1,e_2)
==========

\frac{e_1^\top e_2}
{|e_1||e_2|}.
]

The embedding geometry is trained so that same-speaker samples are close and different-speaker samples are separated.

### Valuable uses

The embedding can support:

* speaker verification;
* diarization;
* speaker-conditioned TTS;
* speaker-aware ASR;
* conversation indexing;
* voice clustering.

### Limitations

The embedding may contain nuisance information:

* microphone;
* room acoustics;
* language;
* emotion;
* speaking style.

Brain should optionally disentangle:

[
e=
[e_{\text{speaker}},
e_{\text{channel}},
e_{\text{style}}].
]

Adversarial losses can reduce nuisance leakage.

---

## 8.4 Nemotron 3.5 ASR Streaming

Nemotron 3.5 ASR Streaming uses a cache-aware FastConformer-RNNT architecture with configurable streaming chunk sizes.

An RNN-T decomposes recognition into:

* acoustic encoder;
* prediction network;
* joint network.

Let:

[
h_t=E(x_{1:t}),
]

[
g_u=P(y_{<u}),
]

[
\ell_{t,u}
==========

J(h_t,g_u).
]

The transcript probability sums over valid alignments (\pi):

[
p(y\mid x)
==========

\sum_{\pi\in\mathcal A(x,y)}
\prod_{(t,u)\in\pi}
p(\pi_{t,u}\mid h_t,g_u).
]

### Streaming encoder state

The cache summarizes previous audio context needed for future chunks:

[
c_t=U(c_{t-1},x_{t:t+\Delta}).
]

This is a predictive computational state.

### Acoustic-state seam

Encoder features can support:

* keyword detection;
* emotion estimation;
* speaker change detection;
* acoustic-event classification;
* confidence prediction.

### Transcript seam

Partial transcripts should include stability:

[
p(y_{1:k}\ \text{will remain unchanged}\mid x_{\leq t}).
]

This lets downstream language models begin work before finalization without repeatedly invalidating large contexts.

---

## 8.5 Qwen3-ASR

Qwen3-ASR provides multilingual recognition, language identification and forced alignment models.

A useful decomposition is:

[
H_a=E_a(x),
]

[
p(y\mid x)
==========

\prod_t
p(y_t\mid y_{<t},H_a).
]

### Audio-encoder seam

The encoder sequence contains linguistic and non-linguistic acoustic information.

It can be reused for:

* audio understanding;
* emotion;
* speaker properties;
* event recognition;
* direct multimodal reasoning.

### Forced-alignment seam

Given audio (x) and transcript (y_{1:N}), forced alignment estimates:

[
p(t_i^{start},t_i^{end}\mid x,y).
]

This is valuable for:

* subtitle generation;
* TTS dataset preparation;
* speech editing;
* word-level retrieval;
* training temporal audio models.

### ASR-to-TTS closed loop

Qwen3-ASR and Qwen3-TTS can form:

[
x_{\text{speech}}
\rightarrow
H_a
\rightarrow
y_{\text{text}}
\rightarrow
c_{\text{speech}}
\rightarrow
\hat x_{\text{speech}}.
]

A richer speech-to-speech model should preserve selected acoustic attributes outside the transcript:

[
s=
g(H_a)
]

for speaker, emotion, rhythm or emphasis.

---

# 9. Forecasting and financial-model seams

## 9.1 General predictive-model interpretation

A forecasting model represents:

[
p(y_{t+1:t+H}\mid h_t,c_{t+1:t+H}),
]

where:

* (h_t) is historical information;
* (c) contains known future covariates;
* (H) is the prediction horizon.

A point prediction alone returns:

[
\hat y=\mathbb{E}[Y].
]

A valuable seam should preserve uncertainty:

* samples;
* quantiles;
* covariance;
* mixture components;
* regime probabilities.

For quantile (\tau), the prediction is:

[
q_\tau(x)
=========

\inf{y:F(y\mid x)\geq\tau}.
]

It can be trained using pinball loss:

[
\mathcal{L}_\tau(y,\hat q)
==========================

\max
\left(
\tau(y-\hat q),
(\tau-1)(y-\hat q)
\right).
]

---

## 9.2 Chronos-2

Chronos-2 supports univariate, multivariate and covariate-informed forecasting through group attention.

Let related series form a group:

[
X
=

{x^{(1)},\ldots,x^{(M)}}.
]

Group attention permits information sharing:

[
H_i'
====

\operatorname{Attention}
\left(
H_i,
[H_1;\ldots;H_M]
\right).
]

### Encoder-state seam

The hidden sequence is an estimate of predictive state:

[
z_t=\phi(x_{\leq t},c_{\leq t}).
]

It can support:

* forecasting heads;
* anomaly detection;
* regime classification;
* maintenance prediction;
* controller scheduling;
* causal event analysis.

### Forecast-distribution seam

This is the most valuable stable output:

[
\mathcal F
==========

{q_{\tau,h}},
]

for horizons (h) and quantiles (\tau).

It supports risk-sensitive decisions:

[
a^*
===

\arg\min_a
\mathbb{E}_{Y\sim\mathcal F}
[C(a,Y)].
]

### Covariate seam

Event embeddings can be supplied as future or historical covariates:

[
c_t
===

[
c_t^{calendar},
c_t^{weather},
c_t^{event},
c_t^{control}
].
]

Temporal alignment is essential. A semantically strong embedding with the wrong timestamp is destructive.

---

## 9.3 Kronos

Kronos tokenizes candlestick or K-line data and trains an autoregressive financial transformer over more than one market and task.

Let continuous market observations be:

[
x_t=[O_t,H_t,L_t,C_t,V_t,\ldots].
]

A tokenizer produces:

[
k_t=Q(x_t;\eta),
]

and the model learns:

[
p(k_{t+1:t+H}\mid k_{\leq t}).
]

### Token seam

The discrete token has meaning through the financial tokenizer:

[
\hat x_t=Q^{-1}(k_t).
]

Tokenization can improve modelling of multimodal future distributions because the model predicts a categorical distribution rather than a single regression value.

### Scenario seam

Sampled trajectories:

[
\tilde x_{t+1:t+H}^{(s)}
\sim
p_\theta(\cdot\mid x_{\leq t})
]

are more useful than one point forecast.

They enable:

* stress testing;
* portfolio simulation;
* execution-policy evaluation;
* volatility estimation;
* controller or policy distillation.

### Hidden-state seam

The hidden state may encode:

* volatility regime;
* trend;
* liquidity;
* recurring temporal pattern;
* cross-feature dependence.

These meanings should be tested using probes, not assumed.

### Caution

A financial predictive model does not directly yield an optimal trading or control action. The action depends on:

* objective;
* cost;
* risk preference;
* constraints;
* transaction costs;
* uncertainty calibration.

The proper chain is:

[
\text{predictive distribution}
\rightarrow
\text{decision optimization}
\rightarrow
\text{policy distillation}.
]

---

## 9.4 FinCast

FinCast is designed for financial forecasting under non-stationarity, multiple financial domains and multiple temporal resolutions.

A patching model maps windows:

[
p_j
===

[x_{jP},\ldots,x_{jP+P-1}]
]

into tokens:

[
z_j=W_pp_j+b_p.
]

### Patch seam

A patch token summarizes local dynamics over a finite interval.

It is analogous to a local system-identification feature and can encode:

* slope;
* volatility;
* periodicity;
* jumps;
* covariance.

### Frequency embedding

A resolution variable (f) conditions the model:

[
z_j'=z_j+e_f.
]

This is essential because identical numerical patterns can mean different things at millisecond and weekly scales.

### MoE regime seam

If FinCast uses sparse experts, the routing distribution can be interpreted as a learned regime assignment:

[
p(r_t\mid x_{\leq t}).
]

This can drive:

* gain scheduling;
* risk-model switching;
* model selection;
* adaptive filtering.

---

# 10. World-model seams

## 10.1 DIAMOND

DIAMOND trains an agent inside a diffusion world model that predicts visual observations.

A world model approximates:

[
p(o_{t+1},r_t,d_t\mid o_{\leq t},a_{\leq t}),
]

where:

* (o_t) is observation;
* (a_t) is action;
* (r_t) is reward;
* (d_t) is termination.

### Pixel-state seam

DIAMOND deliberately preserves visual detail in pixel space instead of relying exclusively on heavily compressed discrete states.

This is useful when small visual differences affect action selection.

### Action-conditioned diffusion seam

The model represents:

[
p(o_{t+1}\mid h_t,a_t)
]

through iterative denoising.

The score or velocity field provides local information about the conditional observation distribution.

### Uncertainty seam

Multiple world-model samples:

[
o_{t+1}^{(s)}
\sim
p_\theta(o_{t+1}\mid h_t,a_t)
]

allow epistemic and aleatoric uncertainty analysis.

An exploration score can be:

[
u(h_t,a_t)
==========

\operatorname{Var}*s
[\phi(o*{t+1}^{(s)})].
]

The agent can select:

[
a_t
===

\arg\max_a u(h_t,a)
]

to collect informative experience.

### Distilling a controller

A large policy trained in DIAMOND can supervise a small controller:

[
\pi_S(a\mid s)
\approx
\pi_T(a\mid o).
]

The student may consume a small state vector extracted by YOLO, depth or a learned encoder.

---

## 10.2 GenieRedux-G

GenieRedux-G uses a spatiotemporal VQ tokenizer and an action-conditioned dynamics model with explicit one-hot actions.

Let:

[
z_t=Q(E(o_t)).
]

The dynamics model learns:

[
p(z_{t+1}\mid z_{\leq t},a_{\leq t}).
]

The decoder reconstructs:

[
\hat o_t=D(z_t).
]

### Discrete world-state seam

The VQ code is:

* compact;
* serializable;
* autoregressively modelled;
* indexable;
* suitable for replay storage.

### Action seam

Explicit action conditioning removes uncertainty about which action caused a transition:

[
p(z_{t+1}\mid z_t,a_t).
]

This is a cleaner control interface than latent-action inference when true actions are available.

### World-model planning

For candidate action sequences (A^{(j)}):

[
Z^{(j)}
\sim
p_\theta(Z_{t+1:t+H}\mid z_t,A^{(j)}).
]

Evaluate:

[
J(A^{(j)})
==========

\mathbb{E}
\left[
\sum_{h=0}^{H}
\gamma^h r(z_{t+h},a_{t+h})
\right].
]

Choose:

[
A^*=\arg\max_AJ(A).
]

A small policy can then imitate the first action:

[
\pi_\phi(z_t)\approx a_t^*.
]

---

# 11. Distilling predictive models into PID gain schedulers

This is one of the most valuable concrete compositions for Brain.

The crucial point is that a predictive model should not usually be trained to output PID gains directly without an intermediate control objective.

The mathematically correct chain is:

[
\boxed{
\text{measurement history}
\rightarrow
\text{predictive plant model}
\rightarrow
\text{optimal-control solver}
\rightarrow
\text{PID gains}
\rightarrow
\text{small gain-scheduling student}
}
]

## 11.1 Plant model

Let a nonlinear plant be:

[
x_{t+1}
=======

f(x_t,u_t,w_t),
]

[
y_t=h(x_t)+v_t,
]

where (w_t) and (v_t) are disturbances and measurement noise.

The system may change regime:

[
r_t\in{1,\ldots,R},
]

[
x_{t+1}
=======

f_{r_t}(x_t,u_t,w_t).
]

A forecasting model such as Chronos-2, Kronos or FinCast can approximate:

[
p_\theta
(y_{t+1:t+H}\mid
y_{t-L:t},
u_{t-L:t+H-1},
c_t).
]

The model must include candidate future controls (u) if it is to model intervention effects. A model trained only on passive observations estimates correlation-driven forecasts, not controlled dynamics.

## 11.2 PID controller

For error:

[
e_t=r_t^{set}-y_t,
]

a discrete PID controller is:

[
I_t=I_{t-1}+T_se_t,
]

[
D_t=\frac{e_t-e_{t-1}}{T_s},
]

[
u_t
===

K_pe_t+K_iI_t+K_dD_t.
]

A practical controller also includes:

* derivative filtering;
* saturation;
* anti-windup;
* bumpless gain transfer;
* rate limits.

## 11.3 Teacher optimization

For the current regime history (h_t), solve:

[
K^*(h_t)
========

\arg\min_{K_p,K_i,K_d}
\mathbb{E}*{\theta}
\left[
\sum*{k=1}^{H}
e_{t+k}^\top Qe_{t+k}
+
u_{t+k}^\top Ru_{t+k}
+
\Delta u_{t+k}^\top S\Delta u_{t+k}
\right].
]

Add penalties for:

* overshoot;
* settling time;
* actuator saturation;
* integral windup;
* constraint violations;
* uncertainty.

A risk-sensitive objective is:

[
J_{\mathrm{risk}}
=================

\mathbb{E}[J]
+
\lambda\operatorname{CVaR}_\alpha(J).
]

The optimizer uses sampled trajectories from the predictive model.

## 11.4 Student gain scheduler

Train a small student:

[
g_\phi(m_t)
===========

[K_p,K_i,K_d],
]

where (m_t) may include:

* recent measurements;
* derivatives;
* setpoint;
* operating point;
* load estimate;
* forecast hidden state;
* regime probability.

Bound the gains using:

[
K_p
===

K_p^{min}
+
(K_p^{max}-K_p^{min})
\sigma(a_p),
]

and similarly for (K_i,K_d).

This guarantees gain bounds regardless of student output.

## 11.5 Distillation objective

A basic supervised objective is:

[
\mathcal{L}_{gain}
==================

|g_\phi(m_t)-K^*(m_t)|_W^2.
]

But matching gains is not enough. Several gain combinations may produce similar behavior, and a small gain error may cause a large trajectory error.

Add a rollout loss:

[
\mathcal{L}_{roll}
==================

\sum_{k=1}^{H}
|
y_{t+k}^{student}
-----------------

y_{t+k}^{teacher}
|^2.
]

Add a control loss:

[
\mathcal{L}_{u}
===============

\sum_{k=1}^{H}
|
u_{t+k}^{student}
-----------------

u_{t+k}^{teacher}
|^2.
]

The total objective becomes:

[
\mathcal{L}
===========

\lambda_g\mathcal{L}*{gain}
+
\lambda_r\mathcal{L}*{roll}
+
\lambda_u\mathcal{L}*u
+
\lambda_s\mathcal{L}*{stability}.
]

## 11.6 Stability constraints

For a locally linearized discrete system:

[
x_{t+1}=Ax_t+Bu_t,
]

the closed-loop system is:

[
x_{t+1}=A_{cl}(K)x_t.
]

Local discrete stability requires:

[
\rho(A_{cl})<1,
]

where (\rho) is spectral radius.

A Lyapunov condition is the existence of (P\succ0) such that:

[
A_{cl}^\top PA_{cl}-P\prec0.
]

A penalty can be:

[
\mathcal{L}_{stability}
=======================

\max(0,\rho(A_{cl})-1+\epsilon)^2.
]

For nonlinear or uncertain systems, Brain should use:

* robust linearization over operating regions;
* common or parameter-dependent Lyapunov functions;
* reachability analysis;
* adversarial disturbance simulation;
* a fallback certified controller.

Differentiable MPC and Riccati-based control provide a way to backpropagate through the control solution while retaining explicit stability structure.

## 11.7 Regime-conditioned gain scheduling

The predictive model can estimate:

[
p(r_t=j\mid h_t).
]

One approach blends gain sets:

[
K_t
===

\sum_j
p(r_t=j\mid h_t)K_j.
]

This can be dangerous because a convex combination of individually stable controllers is not necessarily stable.

A safer design is:

* train one continuous scheduler with stability constraints;
* use hysteresis when switching discrete gain sets;
* validate transition regions explicitly;
* retain a fallback controller.

## 11.8 Online adaptation

Prediction residual:

[
\epsilon_t
==========

y_t-\hat y_t
]

can update:

* regime estimates;
* uncertainty;
* last-layer model parameters;
* gain-scheduler confidence.

The controller should not freely retrain itself online without constraints. Instead:

[
K_t
===

\begin{cases}
g_\phi(m_t), & c_t\geq c_{min},\
K_{safe}, & c_t<c_{min}.
\end{cases}
]

## 11.9 Generalization beyond PID

The same process can distil into:

* PI controllers;
* feed-forward tables;
* state-feedback matrices;
* finite-state controllers;
* gain-scheduled filters;
* small neural policies;
* explicit rule sets.

A large model discovers or optimizes behavior. The deployed model is the smallest representation that preserves the required closed-loop behavior.

---

# 12. High-value model compositions

## 12.1 Predictive maintenance and adaptive control

[
\text{sensors}
\rightarrow
\text{Chronos-2}
\rightarrow
\begin{cases}
\text{failure probability}\
\text{future load}\
\text{regime embedding}
\end{cases}
\rightarrow
\text{PID gain scheduler}.
]

The forecast model provides future uncertainty, while the controller student provides deterministic low-latency gains.

The teacher can optimize gains under predicted disturbances rather than only the current operating point.

---

## 12.2 Event-aware industrial controller

[
\text{measurements}
+
\text{maintenance events}
+
\text{operator actions}
\rightarrow
\text{event/effect transformer}
\rightarrow
\text{Chronos-2}
\rightarrow
\text{robust optimizer}
\rightarrow
\text{embedded controller}.
]

This can account for:

* component replacement;
* environmental change;
* load transitions;
* calibration;
* operator interventions.

The event representation should remain time-aligned and distinguish observation from intervention.

---

## 12.3 Semantic 3D world model

[
\text{images/video}
\rightarrow
\begin{cases}
\text{WorldMirror-2 geometry}\
\text{YOLO objects}\
\text{Qwen-VL semantics}
\end{cases}
\rightarrow
\text{semantic 3D Gaussians}.
]

For each Gaussian:

[
g_i=
(\mu_i,\Sigma_i,\alpha_i,c_i,s_i,o_i),
]

where (s_i) is a semantic embedding and (o_i) an object ID.

This supports:

* spatial natural-language queries;
* persistent object tracking;
* world editing;
* simulation;
* robot navigation;
* digital-twin updates.

---

## 12.4 Low-bandwidth multimodal agent

[
\text{camera}
\rightarrow
\text{Moondream/YOLO}
\rightarrow
\text{semantic events},
]

[
\text{microphone}
\rightarrow
\text{Mimi}
\rightarrow
\text{audio tokens},
]

[
\text{edge semantic state}
\rightarrow
\text{remote Qwen3}.
]

Only compact semantic or codec seams cross the network.

The edge device retains raw private data unless an escalation policy requires it.

---

## 12.5 Full-duplex speech agent

[
\text{audio}
\rightarrow
\text{Nemotron/Qwen3-ASR states}
\rightarrow
\text{Qwen3}
\rightarrow
\text{Qwen3-TTS tokens}
\rightarrow
\text{local decoder}.
]

ECAPA provides persistent speaker identity:

[
e_s=\operatorname{ECAPA}(x).
]

The system should carry several parallel streams:

* transcript;
* speaker identity;
* prosody;
* acoustic events;
* dialogue state;
* output codec tokens.

Text alone discards too much of the original speech signal.

---

## 12.6 World-model-to-embedded-policy distillation

[
\text{DIAMOND or GenieRedux-G}
\rightarrow
\text{large planner}
\rightarrow
\text{teacher actions}
\rightarrow
\text{small policy}.
]

The student may consume:

* low-resolution image;
* YOLO object state;
* depth;
* compact latent code;
* sensor vector.

Training should include both imitation and real-environment correction:

[
\mathcal L
==========

\mathcal L_{\text{imitation}}
+
\lambda\mathcal L_{\text{real rollout}}.
]

This prevents the student from inheriting world-model errors uncritically.

---

## 12.7 Synthetic-data factory

Z-Image or FLUX.2 Klein can generate controlled scenes:

[
I\sim p(I\mid c),
]

while WorldMirror, ZipDepth or YOLO create pseudo-labels.

A stronger pipeline renders from an explicit scene state:

[
S
\rightarrow
\text{renderer/generator}
\rightarrow
(I,Y),
]

where (Y) contains exact labels.

This can train:

* detectors;
* depth estimators;
* pose models;
* segmentation models;
* anomaly detectors.

The synthetic distribution must be validated against real data. More synthetic samples do not help if the generator systematically omits real-world failure modes.

---

# 13. Brain’s seam type system

Brain should not expose:

```text
Tensor hidden_state
```

as the primary abstraction.

It should expose a descriptor such as:

```text
SeamDescriptor {
    model_family
    checkpoint_hash
    seam_name
    seam_version

    semantic_class
    task_contract
    approximate_sufficiency_for[]

    shape
    axis_semantics
    dtype
    quantization
    normalization

    tokenizer_id
    vocabulary_id
    codebook_id
    codebook_order

    causal
    streaming
    frame_rate
    temporal_stride
    timestamps

    coordinate_frame
    units
    camera_intrinsics
    scale_ambiguity

    probability_type
    calibration_method
    uncertainty_representation

    differentiable
    gradient_owner
    loss_contract

    privacy_class
    retention_policy
}
```

## 13.1 Stable public types

Brain should standardize semantic interfaces such as:

* `TextTokenSequence`
* `SemanticTokenSequence`
* `DenseEmbedding`
* `LateInteractionEmbedding`
* `DetectionSet`
* `InstanceMaskSet`
* `DepthField`
* `NormalField`
* `CameraModel`
* `PointCloud`
* `GaussianScene`
* `AudioCodecStream`
* `SpeakerEmbedding`
* `TranscriptDistribution`
* `ForecastDistribution`
* `EventEffectDistribution`
* `WorldTransitionDistribution`
* `ActionDistribution`
* `ControllerParameters`

## 13.2 Versioned latent types

Examples:

* `Qwen3ResidualStream`
* `LFM25EncoderSequence`
* `ZImageLatent`
* `MimiCodeStream`
* `GenieReduxVQState`

The checkpoint hash must be part of the type identity.

## 13.3 Opaque runtime types

Examples:

* `DecoderCacheHandle`
* `StreamingASRStateHandle`
* `DiffusionSessionHandle`
* `MoEDispatchHandle`

These should not imply portable tensor compatibility.

---

# 14. Mathematical operators Brain should support

Brain can turn model composition into a small algebra of operators.

## Encode

[
z=E(x).
]

## Decode

[
\hat x=D(z).
]

## Project

[
z'=zW+b.
]

## Resample

[
z'=\operatorname{CrossAttention}(Q,z,z).
]

## Align

[
\min_\phi
d(g_\phi(z_A),z_B).
]

## Fuse

[
z=F(z_1,\ldots,z_n).
]

## Condition

[
p(y\mid x,c).
]

## Predict

[
p(x_{t+1:t+H}\mid h_t,a_{t:t+H-1}).
]

## Optimize

[
a^*=\arg\min_a\mathbb{E}[C(a,Y)].
]

## Distil

[
\min_\phi
D(p_T|p_S)
+
\lambda\mathcal L_{task}.
]

## Control

[
u_t=\pi(z_t,r_t).
]

## Calibrate

[
\hat p
======

C_\phi(p).
]

## Verify

[
V(f,\mathcal D,\mathcal C)
\rightarrow
{\text{pass},\text{fail},\text{bounds}}.
]

This algebra is more important than trying to define one universal latent tensor.

---

# 15. How Brain should evaluate a seam

## 15.1 Sufficiency test

Train a strong probe:

[
\hat y=g(z).
]

Compare it with a model using the original input:

[
\Delta
======

## \operatorname{Perf}(g(z))

\operatorname{Perf}(f(x)).
]

A small gap indicates approximate sufficiency for that task.

## 15.2 Reconstruction test

Train:

[
\hat x=D(z)
]

and measure which information is recoverable.

This can reveal privacy leakage as well as utility.

## 15.3 Invariance test

For a transformation (T) that should not change meaning:

[
d(E(Tx),E(x))
]

should be small.

Examples include:

* text paraphrase;
* image brightness;
* irrelevant background noise;
* sensor-unit conversion after normalization.

## 15.4 Equivariance test

For transformations that should transform the output:

[
E(Tx)\approx\rho(T)E(x).
]

Geometry seams should be tested under camera and coordinate transformations.

## 15.5 Intervention test

Modify one semantic variable while holding others fixed:

[
z'=z+\delta_i.
]

Observe whether downstream behavior changes in the intended way.

This is more informative than visualizing attention weights.

## 15.6 Calibration test

For a prediction with confidence (p):

[
P(Y=\hat Y\mid \hat p=p)\approx p.
]

Forecast and controller pipelines should evaluate:

* expected calibration error;
* interval coverage;
* quantile calibration;
* out-of-distribution confidence.

## 15.7 Version-compatibility test

Given old producer (E_1), new producer (E_2), and consumer (D):

[
\Delta_{compat}
===============

\mathbb{E}
[
\mathcal L(D(E_2(x)),y)
-----------------------

\mathcal L(D(E_1(x)),y)
].
]

No latent seam should be declared compatible based only on shape.

## 15.8 Closed-loop test

For control compositions, open-loop prediction accuracy is insufficient.

Evaluate:

* stability;
* tracking error;
* overshoot;
* settling time;
* constraint violations;
* actuator wear;
* recovery from disturbance;
* model error;
* sensor failure;
* adversarial operating points.

The controller is the complete closed-loop system, not merely the gain-prediction network.

---

# 16. Recommended implementation priority

## Highest-value stable seams

Brain should prioritize:

1. forecast distributions;
2. audio codec tokens;
3. encoder sequences;
4. detections and masks;
5. depth and camera geometry;
6. point clouds and 3D Gaussians;
7. speaker embeddings;
8. ASR partial/final transcripts;
9. controller parameters with bounds;
10. world-transition distributions.

These have clear mathematical contracts and support both model composition and distributed deployment.

## Valuable but checkpoint-specific seams

Next:

* VLM projected tokens;
* image-generation conditioning tokens;
* time-series hidden states;
* world-model VQ states;
* transformer final hidden sequences;
* autoencoder latents.

These should require adapters and explicit checkpoint versions.

## Primarily internal execution seams

Keep these internal or opaque:

* raw KV-cache layouts;
* arbitrary transformer layer states;
* MoE dispatch tensors;
* diffusion intermediate solver states;
* convolution caches;
* sparse-attention index structures.

They are valuable for scheduling but poor long-term public interfaces.

---

# 17. Final conclusion

The models supported by Brain do not merely form a list of unrelated architectures. They define a network of mathematical transformations:

[
\text{observation}
\rightarrow
\text{representation}
\rightarrow
\text{prediction}
\rightarrow
\text{optimization}
\rightarrow
\text{action}
\rightarrow
\text{new observation}.
]

The most valuable reusable objects are not necessarily the largest hidden states. They are the representations that have the strongest mathematical contract:

* predictive distributions retain uncertainty;
* codec tokens retain reconstructable information;
* geometric fields retain coordinate meaning;
* object sets retain scene structure;
* encoder sequences retain task-relevant context;
* world states retain action-conditioned future information;
* controller parameters retain a compact deployable policy.

The Kronos-to-PID example generalizes into a broad engineering principle:

[
\boxed{
\text{Use large models to infer, predict, simulate and optimize.}
}
]

[
\boxed{
\text{Distil the resulting behavior into the smallest verifiable model that can safely execute it.}
}
]

A forecasting model should not be treated as a controller. An image encoder should not be treated as an image codec. A language hidden state should not be treated as a universal semantic representation. A KV cache should not be treated as a portable latent format.

Brain becomes strategically valuable when it knows these distinctions and can automatically construct, train, validate and deploy bridges between them.

Its fundamental abstraction should therefore be:

[
\boxed{
\text{typed seam}
=================

\text{tensor}
+
\text{statistical meaning}
+
\text{coordinate contract}
+
\text{uncertainty}
+
\text{training contract}
+
\text{validation evidence}.
}
]
