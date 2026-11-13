<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Volume V: Language, Sequences and Transformers

Everything so far has fed a network numbers that were already numbers: the
measurements of Volume II, the pixels and feature vectors of Volume III. Text
is not like that. A sentence arrives as a string of characters, its meaning
depends on the order of its words, its length is not fixed, and what a model
must do with it is usually to say what comes next. This volume is about
turning that into something a network can learn from, and about the one
architecture that learned it best. It follows one question from the bytes of a
file to a generative model that can talk: how does a machine come to read and
write language?

A language model is the most expensive object this book describes, and the
reasons are the ones Volume IV taught. Its cost is set by the arithmetic of
matrix products and by the bytes those products must move; it is fast or slow
by the roofline, quantized by the formats of its first chapter and run on the
kernels of its middle chapters. A reader who has met those ideas can read a
transformer as a computation with a known price, which is how the later chapters
of this volume treat it: the quadratic memory of attention, the cache that avoids
recomputing the past, the tiling that avoids storing an attention matrix at all.
The volume also uses the vectors and matrices of Volume I, the softmax and
cross-entropy of Volume III and the training loop built there.

The language itself comes first, because every step after it depends on what
the model is given. A model never sees a character: it sees a list of integer
ids, and which integers, how many, and what a mistake in producing them costs is
settled before the first matrix is multiplied. The volume therefore starts with
text as bytes and the tokenizers built on them, moves to the vectors those ids
become and the prediction task that trains them, goes through the recurrent
networks and encoder-decoder models that were the answer before the
transformer, and then builds the transformer itself, one part at a time.

You can say exactly how text becomes a sequence of integers and what is lost or
fixed along the way: which bytes a string occupies, how two spellings of the
same word are made one, how a vocabulary of pieces is learned and how special
tokens and chat templates frame a conversation. You can derive an embedding
table and its gradient, state what predicting the next token optimizes, and
explain why a recurrent network forgets and what attention does instead. You
can write self-attention with its backward pass, give a network a sense of
position, assemble the transformer block with residuals and normalization, and
train a small GPT from raw text. You can turn next-token probabilities into text
with several decoding rules, say what the key-value cache stores and what it
costs, compute attention exactly without ever holding the full attention matrix,
and say how the bidirectional encoders used for search and classification
differ. The project that ends the volume trains a small language model from raw
text and generates from it with a cache that provably matches uncached
generation.

1. [Text as Data](01-text.md): what text is to a computer, byte by byte.
2. [Tokenization](02-tokenization.md): how text is cut into pieces a model can count,
   and why the choice matters so much.
3. [Embeddings](03-embeddings.md): how a token id becomes numbers a model can compute with, and how
   those numbers come to mean something.
4. [Language Modeling](04-language-modeling.md): how predicting the next token can teach a model about
   language.
5. [Recurrent Networks](05-recurrent-networks.md): how a network carries memory from one step of a
   sequence to the next, and why that memory fades.
6. [Encoder-Decoder Models and the Birth of Attention](06-seq2seq.md): how a model can read one
   sequence and write another, and why a fixed-size summary failed.
7. [Self-Attention](07-self-attention.md): how every position in a sequence can look at every
   other position in parallel.
8. [Positional Information](08-positions.md): how attention, which ignores order, is told
   where each token is.
9. [The Transformer Block](09-transformer-block.md): how attention, a feed-forward network, residuals and
   normalization fit into one repeating unit.
10. [Training a GPT from Scratch](10-training-gpt.md): the full pipeline from raw text to a trained
    generative model.
11. [Decoding: Turning Probabilities into Text](11-decoding.md): how to choose what to say, given
    next-token probabilities.
12. [The KV Cache](12-kv-cache.md): why the past is remembered instead of recomputed, and what
    remembering costs.
13. [Attention Without the Quadratic Memory](13-flash-attention.md): how attention over long
    sequences is computed exactly without storing the full attention matrix.
14. [Bidirectional Encoders and Embedding Models](14-encoders.md): how models that read a whole text
    at once learn representations for search and classification.
15. [Project: Your Own Small Language Model](15-project-gpt.md): train a small GPT from raw
    text and generate from it with a KV cache that provably matches uncached generation.
