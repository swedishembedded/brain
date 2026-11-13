<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Volume VII: Vision and Understanding Images

Every earlier volume handed the machine a tensor and took its shape for granted. A picture is the first input whose tensor is made out of a measurement: the numbers are codes of light that someone chose a curve for, the grid has a size that no network wants, the colours were packed by a camera for transmission rather than for arithmetic, and the model on the far side was trained on pictures that went through a particular sequence of resizing and scaling that its weights have never forgotten. Get any one of those wrong by a half pixel, a channel or a constant and the program still runs, the output still looks like an output, and the accuracy is quietly lost. This volume is about seeing in two senses: what an image is as data, and what networks do with it.

The volume stands on three earlier ones. Volume III built the network and its training, and every vision model is that network with a different first layer. Volume IV taught the computation underneath, kernels, memory traffic and the way a quantized model is calibrated on sample inputs, and images are the inputs for which a calibration set most easily goes wrong. Volume V and Volume VI built the transformer and the language models around it, which the second half of this volume reuses: an image is cut into patches and becomes a sequence, and a language model with a vision tower attached becomes a model that reads pictures. What the volume adds is the thing none of them had to face, the input that is not already numbers.

It starts, therefore, with the data. The first chapter treats an image as a tensor with a history: the codes, the layouts, the transfer curve, the camera formats, the resampling conventions and the constants of normalization, each checked against a reference. Every later chapter relies on it, because every later model begins with a preprocessing function, and a model whose preprocessing differs from its reference is a different model. The chapters after it follow the history of the field: the convolution and the networks built from it, detection and dense prediction, the transformer applied to images, promptable segmentation, the joint embedding of images and text, vision-language models, documents and faces. The volume ends with a pipeline that detects, segments and tracks objects through video.

You can say what the numbers of an image mean and convert between the layouts and colour formats a decoder, a camera and a network use. You can resample an image under any of the usual conventions and say which source pixels each output reads, and you know why a reference's resizer has to be reproduced rather than approximated. You can derive the preprocessing of a given model from its reference and verify, to a stated tolerance, that your pipeline gives the same input. Later chapters add the filters that find edges and textures, the networks that stack them, the heads that say where objects are and which pixel belongs to which, the transformers that replace the filters, the models that segment whatever you point at, that put images and sentences in one space, that describe pictures in words and read pages of text, and the models that find a face and know whose it is.

1. [Images as Data](01-images.md): what an image is to a model, and how many ways preprocessing can quietly go wrong.
2. [Convolution](02-convolution.md): how a small filter slid across an image finds edges, textures and eventually objects.
3. [Convolutional Networks and Residual Learning](03-cnns.md): how stacks of convolutions become image classifiers, and why residual connections let them go deep.
4. [Object Detection](04-detection.md): how a network says not just what is in an image but where.
5. [Dense Prediction: Segmentation and Depth](05-dense-prediction.md): how a network labels every pixel or estimates how far away it is.
6. [Vision Transformers](06-vision-transformers.md): whether attention can replace convolution for images.
7. [Promptable Segmentation and Video Tracking](07-segment-anything.md): how one model can segment any object you point at, and follow it through a video.
8. [Contrastive Vision-Language Learning](08-clip.md): how images and texts can be placed in one space where matching pairs land close together.
9. [Vision-Language Models](09-vlms.md): how to give a language model eyes.
10. [Reading Documents](10-ocr.md): how a model reads a page of text from pixels, and how few tokens a page can be squeezed into.
11. [Faces: Detection, Identity and Parsing](11-faces.md): how a model finds faces and recognizes the same person across photos.
12. [Project: A Vision Pipeline](12-project-vision-pipeline.md): build a pipeline that detects, segments and tracks objects through video frames.
