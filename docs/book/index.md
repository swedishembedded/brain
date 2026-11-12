<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# From First Principles to Intelligent Machines

*The mathematics, algorithms and engineering of machine learning, told
through the building of brain.*

This is a textbook. It starts from arithmetic and ends with a complete
machine learning engine: one that trains and serves language, vision,
audio, video, forecasting and decision models on CPUs, GPUs and dedicated
accelerators. Every idea is derived before it is used, written as
language-neutral pseudocode, checked with worked numbers you can reproduce,
and then followed into a real implementation in brain, including the
mistakes that implementation made on the way.

You can read it without installing anything, and you can implement every
algorithm in it in the programming language of your choice.

Start with the [Preface](preface.md) and [How to Read This Book](how-to-read.md).
The symbols are collected in [Notation](notation.md), the algorithm language
in [Pseudocode Conventions](pseudocode.md), and the engineering timeline the
historical sections draw on in the [Chronology](history/chronology.md).

## Contents

<!-- contents -->
### [Volume I: Thinking in Mathematics](v01-mathematics/index.md)

1. [Numbers, Quantities and Notation](v01-mathematics/01-numbers.md): What are the numbers a machine learns with, and how do we write about them precisely?
2. [Functions and Their Shapes](v01-mathematics/02-functions.md): How do we describe a rule that turns inputs into outputs, and what shapes can such rules take?
3. [Vectors: Lists of Numbers with Geometry](v01-mathematics/03-vectors.md): How can a list of numbers have a length and a direction, and how do we measure how alike two lists are?
4. [Matrices as Linear Maps](v01-mathematics/04-matrices.md): What does it mean to multiply a matrix by a vector, and why is that the central operation of machine learning?
5. [Arrays, Shapes, Strides and Broadcasting](v01-mathematics/05-arrays.md): How are vectors, matrices and higher-dimensional arrays actually laid out in memory and indexed?
6. [Eigenvalues, Rotations and Low-Rank Structure](v01-mathematics/06-decompositions.md): What hidden structure does a matrix have, and how can a big matrix be approximated by small ones?
7. [Derivatives: The Mathematics of Change](v01-mathematics/07-derivatives.md): How do we measure how fast a function's output changes when its input moves a tiny bit?
8. [Gradients, Jacobians and Many Variables](v01-mathematics/08-multivariable-calculus.md): When a quantity depends on millions of numbers, which way should we move them all to make it smaller?
9. [Integrals and Differential Equations](v01-mathematics/09-integrals-and-odes.md): How do we add up infinitely many small pieces, and how do we follow a system that changes continuously?
10. [Probability: Reasoning Under Uncertainty](v01-mathematics/10-probability.md): How do we put numbers on uncertainty and update them when we see evidence?
11. [The Distributions Machine Learning Uses](v01-mathematics/11-distributions.md): Which few probability distributions keep appearing, and why?
12. [Statistics: Learning Facts from Samples](v01-mathematics/12-statistics.md): How much can a finite sample tell us, and how do we know whether a difference is real?
13. [Information, Entropy and Surprise](v01-mathematics/13-information-theory.md): How many bits does it take to describe what we expect to see, and what does it cost to expect the wrong thing?
14. [Complex Numbers, Frequencies and the Fourier Transform](v01-mathematics/14-fourier.md): How can any signal be written as a sum of waves, and why does that make convolution cheap?
15. [Floating Point and Numerical Error](v01-mathematics/15-floating-point.md): Why does a computer get arithmetic slightly wrong, and when does slightly wrong become completely wrong?
16. [Algorithms, Cost and Complexity](v01-mathematics/16-algorithms.md): How do we predict how long a computation will take and how much memory it needs before we run it?
17. [Randomness on a Deterministic Machine](v01-mathematics/17-randomness.md): How does a deterministic computer produce random numbers, and how do we make random experiments repeatable?
18. [Project: A Numerical Library from Nothing](v01-mathematics/18-project-numerical-library.md): Can you build the numerical toolkit every later chapter relies on, and prove each piece correct?

### [Volume II: What Does It Mean to Learn?](v02-learning/index.md)

1. [The Learning Problem](v02-learning/01-learning-problem.md): What exactly is a model, what is it learning, and from what?
2. [Linear Regression and the First Descent](v02-learning/02-linear-regression.md): How do we find the straight line that best explains our measurements, and what does 'best' mean?
3. [Classification and Logistic Regression](v02-learning/03-logistic-regression.md): How does a model output a probability for a yes-or-no question, and how do we train it?
4. [Maximum Likelihood, Priors and Regularization](v02-learning/04-likelihood.md): Why are the losses we use the right ones, and what does it mean to prefer simple explanations?
5. [Generalization, Overfitting and Honest Evaluation](v02-learning/05-generalization.md): How do we know a model has learned the pattern rather than memorized the examples?
6. [Measuring Models: Metrics and Calibration](v02-learning/06-metrics.md): Which number tells us whether a model is good, and why is accuracy so often the wrong one?
7. [Machine Learning Before Deep Learning](v02-learning/07-classical-methods.md): What did the field know before neural networks took over, and when are those methods still the right tool?
8. [Experiments, Baselines and Claims](v02-learning/08-experiments.md): How do we run an experiment whose result we can believe and someone else can repeat?
9. [Project: A Predictor You Can Trust](v02-learning/09-project-first-predictor.md): Can you train a model on real measurements, evaluate it honestly and explain exactly where it fails?

### [Volume III: Building Neural Networks from Nothing](v03-neural-networks/index.md)

1. [The Neuron and the Perceptron](v03-neural-networks/01-perceptron.md): What is an artificial neuron, and what can a single one never learn?
2. [Multilayer Networks](v03-neural-networks/02-mlp.md): Why does stacking layers with a nonlinearity between them let a network represent almost anything?
3. [Softmax, Logits and Cross-Entropy](v03-neural-networks/03-softmax.md): How does a network choose among many classes, and why is its gradient so simple?
4. [Backpropagation Derived from Scratch](v03-neural-networks/04-backpropagation.md): How do we compute the gradient of one loss with respect to millions of parameters in about the cost of one forward pass?
5. [The Backward Pass of Every Layer](v03-neural-networks/05-layer-gradients.md): What is the backward pass of a matrix multiply, a sum, a broadcast and an elementwise function, written out exactly?
6. [Proving Gradients Correct](v03-neural-networks/06-gradient-checking.md): How do we know a backward pass is right without trusting the person who wrote it?
7. [Optimizers: From SGD to AdamW](v03-neural-networks/07-optimizers.md): Once we have a gradient, what is the smartest way to use it?
8. [Learning-Rate Schedules and Gradient Clipping](v03-neural-networks/08-schedules-and-clipping.md): Why should the step size change during training, and what do we do when a gradient explodes?
9. [Initialization: Where Training Starts](v03-neural-networks/09-initialization.md): Why does the random starting point decide whether a deep network trains at all?
10. [Normalization Layers](v03-neural-networks/10-normalization.md): How does rescaling activations inside a network make it train faster and more stably?
11. [Regularization in Neural Networks](v03-neural-networks/11-regularization.md): How do we stop a network with millions of parameters from memorizing its training set?
12. [Training Dynamics and Debugging](v03-neural-networks/12-training-dynamics.md): When a loss curve looks wrong, how do we find out why?
13. [First Complete Models](v03-neural-networks/13-first-models.md): What does it take to build, train and verify a small network end to end?
14. [Project: A Neural Network Library with Proven Gradients](v03-neural-networks/14-project-neural-network.md): Can you build a small neural network library whose every gradient is checked, and train it to solve a problem?

### [Volume IV: How Computers Actually Execute Machine Learning](v04-computation/index.md)

1. [Bits and Number Formats](v04-computation/01-number-formats.md): How many bits does a number need, and what do we lose with each bit we remove?
2. [The CPU: Cores, Caches and Vectors](v04-computation/02-cpu.md): Why does the same arithmetic run ten times faster when the data is arranged differently?
3. [Memory Bandwidth, Compute and the Roofline](v04-computation/03-roofline.md): Is a computation limited by arithmetic or by moving data, and how do we tell?
4. [The GPU: Thousands of Small Workers](v04-computation/04-gpu.md): How does a GPU organize its thousands of threads, and what does that demand of our algorithms?
5. [Writing Compute Kernels](v04-computation/05-compute-kernels.md): How do we express one parallel computation so a GPU can run it?
6. [Making Matrix Multiplication Fast](v04-computation/06-matmul.md): How does a matrix multiply go from a three-line loop to running near the hardware's peak?
7. [Parallel Reductions, Scans and Sorting](v04-computation/07-reductions.md): How do thousands of threads add up one array, and why is the obvious way slow?
8. [Kernel Fusion and Memory Traffic](v04-computation/08-fusion.md): Why is the cheapest computation the one whose intermediate result never touches memory?
9. [One Program, Many Devices](v04-computation/09-backends.md): How can one set of kernels run on CPUs, Vulkan GPUs and CUDA GPUs with the same results?
10. [Choosing the Fastest Correct Kernel](v04-computation/10-kernel-selection.md): How does an engine run near the roofline on hardware nobody tuned it for?
11. [Measuring Performance Honestly](v04-computation/11-profiling.md): Why do most performance numbers lie, and how do we produce ones that do not?
12. [Quantization: Fewer Bits per Weight](v04-computation/12-quantization.md): How can a model's weights be stored in four bits each and still work?
13. [Neural Processing Units and Graph Compilers](v04-computation/13-npu.md): How does a whole network become one compiled program for a fixed-function accelerator?
14. [Project: A Device-Independent Tensor Engine](v04-computation/14-project-tensor-engine.md): Can you make the same matrix multiply run correctly and fast on more than one kind of hardware?

### [Volume V: Language, Sequences and Transformers](v05-sequences/index.md)

1. [Text as Data](v05-sequences/01-text.md): What is text to a computer, byte by byte?
2. [Tokenization](v05-sequences/02-tokenization.md): How do we cut text into pieces a model can count, and why does the choice matter so much?
3. [Embeddings](v05-sequences/03-embeddings.md): How does a model turn a token id into numbers it can compute with, and how do those numbers come to mean something?
4. [Language Modeling](v05-sequences/04-language-modeling.md): How can predicting the next token teach a model about language?
5. [Recurrent Networks](v05-sequences/05-recurrent-networks.md): How can a network carry memory from one step of a sequence to the next, and why does that memory fade?
6. [Encoder-Decoder Models and the Birth of Attention](v05-sequences/06-seq2seq.md): How can a model read one sequence and write another, and why did a fixed-size summary fail?
7. [Self-Attention](v05-sequences/07-self-attention.md): How can every position in a sequence look at every other position in parallel?
8. [Positional Information](v05-sequences/08-positions.md): Attention ignores order; how do we tell it where each token is?
9. [The Transformer Block](v05-sequences/09-transformer-block.md): How do attention, a feed-forward network, residuals and normalization fit together into one repeating unit?
10. [Training a GPT from Scratch](v05-sequences/10-training-gpt.md): What does the full pipeline from raw text to a trained generative model look like?
11. [Decoding: Turning Probabilities into Text](v05-sequences/11-decoding.md): Given next-token probabilities, how do we choose what to say?
12. [The KV Cache](v05-sequences/12-kv-cache.md): Why recompute the past at every step when we could remember it, and what does remembering cost?
13. [Attention Without the Quadratic Memory](v05-sequences/13-flash-attention.md): How can attention over long sequences be computed exactly without ever storing the full attention matrix?
14. [Bidirectional Encoders and Embedding Models](v05-sequences/14-encoders.md): How do models that read a whole text at once learn representations for search and classification?
15. [Project: Your Own Small Language Model](v05-sequences/15-project-gpt.md): Can you train a small GPT from raw text and generate from it with a KV cache that provably matches uncached generation?

### [Volume VI: Advanced Language Model Architectures](v06-language-models/index.md)

1. [Anatomy of a Modern Decoder](v06-language-models/01-modern-decoder.md): What changed between the first GPT and today's open-weight language models, and why?
2. [Importing Pretrained Models and Proving Parity](v06-language-models/02-importing-models.md): How do we load someone else's trained weights into our own implementation and prove it computes the same thing?
3. [Attention Variants: Sharing and Compressing Keys](v06-language-models/03-attention-variants.md): How do we shrink the KV cache without losing what attention needs?
4. [Mixture of Experts](v06-language-models/04-mixture-of-experts.md): How can a model have a trillion parameters yet use only a few billion per token?
5. [Linear Attention, Recurrence and Hybrid Models](v06-language-models/05-linear-attention.md): Can attention be computed in time linear in the sequence length, and what is lost?
6. [Sparse Attention](v06-language-models/06-sparse-attention.md): If each token only needs a few others, how do we find them cheaply?
7. [Looped and Recurrent-Depth Transformers](v06-language-models/07-recurrent-depth.md): What if a model could think longer by reusing the same layers more times?
8. [Long Context](v06-language-models/08-long-context.md): How do models trained on short texts read long ones?
9. [A Research Architecture: The Spectral Language Model](v06-language-models/09-spectral-lm.md): How do you test a new architectural idea honestly, using brain's spectral language model as the example?
10. [Counting Parameters, FLOPs and Memory](v06-language-models/10-model-cost.md): Before running a model, how do we know what it will cost?
11. [Project: Comparing Language Model Blocks](v06-language-models/11-project-architecture-comparison.md): Can you implement several alternative language-model blocks and compare their quality, cost and memory fairly?

### [Volume VII: Vision and Understanding Images](v07-vision/index.md)

1. [Images as Data](v07-vision/01-images.md): What is an image to a model, and how many ways can preprocessing quietly go wrong?
2. [Convolution](v07-vision/02-convolution.md): How does a small filter slid across an image find edges, textures and eventually objects?
3. [Convolutional Networks and Residual Learning](v07-vision/03-cnns.md): How do stacks of convolutions become image classifiers, and why did residual connections let them go deep?
4. [Object Detection](v07-vision/04-detection.md): How does a network say not just what is in an image but where?
5. [Dense Prediction: Segmentation and Depth](v07-vision/05-dense-prediction.md): How does a network label every pixel or estimate how far away it is?
6. [Vision Transformers](v07-vision/06-vision-transformers.md): Can attention replace convolution for images?
7. [Promptable Segmentation and Video Tracking](v07-vision/07-segment-anything.md): How can one model segment any object you point at, and follow it through a video?
8. [Contrastive Vision-Language Learning](v07-vision/08-clip.md): How can images and texts be placed in one space where matching pairs land close together?
9. [Vision-Language Models](v07-vision/09-vlms.md): How do we give a language model eyes?
10. [Reading Documents](v07-vision/10-ocr.md): How does a model read a page of text from pixels, and how few tokens can a page be squeezed into?
11. [Faces: Detection, Identity and Parsing](v07-vision/11-faces.md): How does a model find faces and recognize the same person across photos?
12. [Project: A Vision Pipeline](v07-vision/12-project-vision-pipeline.md): Can you build a pipeline that detects, segments and tracks objects through video frames?

### [Volume VIII: Generating Images](v08-image-generation/index.md)

1. [What It Means to Generate](v08-image-generation/01-generative-modeling.md): What does it mean for a model to learn a distribution over images, and how can we sample from it?
2. [Variational Autoencoders](v08-image-generation/02-vae.md): How can an autoencoder be made into a generator, and why does that need a bound on likelihood?
3. [Discrete Latents and Vector Quantization](v08-image-generation/03-vector-quantization.md): How can an image be turned into a short sequence of discrete codes?
4. [Adversarial Training](v08-image-generation/04-gans.md): What happens when the loss itself is a network trained to catch the generator?
5. [Diffusion Models](v08-image-generation/05-diffusion.md): How can a model learn to create an image by learning to remove noise?
6. [Flow Matching and Rectified Flow](v08-image-generation/06-flow-matching.md): Can generation be a straight path from noise to data, followed by an ordinary differential equation?
7. [Samplers and Guidance](v08-image-generation/07-samplers-and-guidance.md): How many steps does sampling need, and how do we make a model follow its prompt more closely?
8. [Denoiser Architectures: U-Nets and Diffusion Transformers](v08-image-generation/08-denoiser-architectures.md): What network shapes are good at predicting noise, and how do they take in the prompt?
9. [Conditioning on Text](v08-image-generation/09-text-conditioning.md): How does a sentence become the signal that steers image generation?
10. [Latent Diffusion Systems End to End](v08-image-generation/10-latent-diffusion-systems.md): How do the encoder, denoiser, sampler and decoder combine into a working image generator?
11. [Control, Identity and Adapters](v08-image-generation/11-control-and-identity.md): How do we steer a generator with a pose, an edge map or a particular person's face?
12. [Restoration and Super-Resolution](v08-image-generation/12-restoration.md): How can a network recover detail that is not in the input?
13. [Project: A Small Image Generator](v08-image-generation/13-project-image-generator.md): Can you train a small generator and grow it into a conditioned latent generation system?

### [Volume IX: Audio, Speech and Music](v09-audio/index.md)

1. [Sound and Sampling](v09-audio/01-sound.md): How is a sound wave turned into numbers without losing what we hear?
2. [Spectrograms and the Mel Scale](v09-audio/02-spectrograms.md): How do we see which frequencies are present at each moment, the way an ear does?
3. [Neural Audio Codecs](v09-audio/03-audio-codecs.md): How can audio become a stream of discrete tokens a language model can predict?
4. [Speech Recognition](v09-audio/04-speech-recognition.md): How does a model turn speech into text when it does not know where each word starts?
5. [Who Is Speaking](v09-audio/05-speaker-embeddings.md): How does a model capture the identity of a voice in a single vector?
6. [Speech Synthesis](v09-audio/06-speech-synthesis.md): How does a model speak a sentence in a given voice?
7. [Music Generation](v09-audio/07-music.md): What changes when the target is minutes of structured music rather than a sentence of speech?
8. [Models That Listen, See and Speak](v09-audio/08-omni-models.md): How is one model built to take in audio, images and text and answer in speech?
9. [Project: From Waveform to Speech and Back](v09-audio/09-project-speech-pipeline.md): Can you build a pipeline from a waveform to a learned representation and back to generated audio?

### [Volume X: Video, Space and World Models](v10-video-and-3d/index.md)

1. [Video as Data](v10-video-and-3d/01-video.md): How do we represent and compress a sequence of frames?
2. [Generating Video](v10-video-and-3d/02-video-generation.md): How do diffusion models keep motion consistent across hundreds of frames?
3. [Cameras and Projective Geometry](v10-video-and-3d/03-camera-geometry.md): How does a 3D point become a pixel, and how do we undo that?
4. [Structure from Motion and Multi-View Stereo](v10-video-and-3d/04-structure-from-motion.md): How can we recover a 3D scene and the camera positions from ordinary photographs?
5. [Learning 3D Directly](v10-video-and-3d/05-feed-forward-3d.md): Can one network look at a few images and output the scene's geometry in a single pass?
<!-- /contents -->
