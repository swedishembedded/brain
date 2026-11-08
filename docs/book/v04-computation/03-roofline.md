<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Memory Bandwidth, Compute and the Roofline

*Is a computation limited by arithmetic or by moving data, and how do we tell?*

A language model with 7 billion weights generates one token. For every weight the machine does two operations, a
multiplication and an addition, so a token costs 14 billion floating-point operations. The weights are stored in a
16-bit format, 2 bytes each, so a token also needs 14 billion bytes of weights to be read from main memory,
once. A graphics processor can do 40 trillion operations in a second and read 800 billion bytes in a second.
How long does the token take?

Two answers come to mind at once, one from each number, and they disagree:

<!-- worked: roofline.decode_two_times -->
One token of a dense model with P = 7,000,000,000 weights stored as fp16 (2 bytes), batch size 1: every weight is read once and used in one multiplication and one addition. The machine is the illustrative GPU of the next section: F_pk = 40,000 (GFLOP/s) and BW = 800 (GB/s). Both times are the work divided by the stated rate, a model and not a measurement.

| quantity | value |
|---|---:|
| FLOPs for one token (2 per weight) | 14,000,000,000 |
| bytes of weights read for one token (2 per weight) | 14,000,000,000 |
| seconds the arithmetic needs at the peak rate | 0.000350 |
| seconds the weights need to arrive at the bandwidth | 0.017500 |
| the second divided by the first | 50.0 |
<!-- /worked -->

The arithmetic alone would take 0.35 milliseconds. The weights alone need 17.5 milliseconds to arrive, fifty times as
long. A computation cannot finish before both of its needs are met, so it cannot finish before the larger of the two
times, and here the larger one has nothing to do with arithmetic: this machine spends almost all of the token waiting for
bytes, and its arithmetic units are idle for 49 of every 50 moments. Making the multiplications twice as fast
would change nothing. Storing the weights in half the bytes would halve the time.

This chapter turns that comparison into a method. A computation has two hardware-independent costs, the operations it
performs and the bytes it moves, and a machine has two rates, the operations it can perform per second and the bytes it can move per
second. Comparing the two ratios says which resource limits the computation and gives a number it cannot beat. The picture that
shows it is called the roofline. Volume IV, Chapter 2 ended with the count of operations and bytes; this chapter says what to do with it.

The method leans on six earlier ideas, so bring them back before the counting starts. Answer from memory; the answers are at the end of the chapter.

1. (Volume I, Chapter 4, matrix multiplication.) How many floating-point operations does the product of an $[m, k]$
   matrix and a $[k, n]$ matrix cost, counting a multiplication and an addition as one operation each?
2. (Volume I, Chapter 16, asymptotic complexity.) One quantity grows as $n^3$ and another as $n^2$. How does their ratio
   grow, and in the notation of that chapter what is it?
3. (Volume IV, Chapter 2, cache hierarchy.) A program reads one byte that is not in any cache. How many bytes travel
   from main memory, and why?
4. (Volume IV, Chapter 2, tiling.) What does tiling a loop change about a computation: its operation count, or the number of
   cache lines it fetches?
5. (Volume IV, Chapter 1, number formats.) How many bytes per weight does fp16 take, and how many does int8 with one
   32-bit scale for every 32 values take?
6. (Volume III, Chapters 5 and 7.) The backward pass of a linear layer does about how many times the arithmetic of its forward
   pass? And an AdamW update of one parameter does 16 operations on how many bytes of traffic?

## Two rates of a machine

The time a computation needs depends on the machine through only two numbers, both of which can be read off its design, so we can state them for a machine we have never met. A machine can be asked two different questions. How many floating-point operations can it complete in a second? That is its
**peak arithmetic rate**, written $F_{\mathrm{pk}}$, in operations per second (FLOP/s). It comes from the design: the
number of arithmetic units times the operations each does per cycle times the clock rate. And how many bytes can it bring from main
memory in a second? That is its **memory bandwidth**, written $\mathrm{BW}$, in bytes per second: the number of bytes that main memory
can deliver to the processor in one second when the processor asks for them in the best possible way. It too comes from the design: the
bytes transferred at once times the transfers per second.

Both are rates of work, and they are unrelated: a machine can have a great deal of one and little of the other. The book
uses two machines, both **model parameters**: round numbers invented for the arithmetic and not claims about any
product. The reader can rebuild them from a design. The CPU has 10 cores, each with 2 vector units of 8 lanes that do one fused
multiply-add (2 operations) per lane per cycle, at a clock of 2.5 billion cycles per second: $10 \times 2 \times 8 \times 2 \times 2.5 \times 10^9 = 800 \times 10^9$
operations per second (Chapter 2 built these three factors: cores, lanes and issue rate). Its memory moves 8 bytes per transfer on each of
8 channels at 1.25 billion transfers per second: $8 \times 8 \times 1.25 \times 10^9 = 80 \times 10^9$ bytes per second. The GPU of the next
chapter has about 10,000 lanes doing one fused multiply-add per cycle at 2 billion cycles per second, $10{,}000 \times 2 \times 2 \times 10^9 = 40{,}000 \times 10^9$, and a
memory bus of 32 bytes at 25 billion transfers per second, $32 \times 25 \times 10^9 = 800 \times 10^9$.

<!-- worked: roofline.machines -->
Model parameters, round numbers invented for the book (not any product). Peak is the rate of fp32 arithmetic in GFLOP/s (a multiply and an add count 2); bandwidth is bytes from main memory per second in GB/s (1 GB = 10^9 bytes); the ridge is the peak divided by the bandwidth.

| machine | peak F_pk (GFLOP/s) | bandwidth BW (GB/s) | ridge F_pk / BW (FLOP/byte) |
|---|---:|---:|---:|
| CPU | 800 | 80 | 10.0 |
| GPU | 40,000 | 800 | 50.0 |
<!-- /worked -->

The last column is the quotient $F_{\mathrm{pk}} / \mathrm{BW}$, operations per byte, and it is the number this chapter is
about. For the CPU it is 10: for every byte that arrives, the machine could do 10 operations if it had them to do. For the GPU it is 50. A machine
with a high quotient is one with a lot of arithmetic compared with its pipes to memory, and it needs a computation with a lot of arithmetic per
byte to keep its arithmetic busy. That requirement is a property of the computation, and the next section counts it.

Memory has levels (Volume IV, Chapter 2), and every level has its own bandwidth, higher for the small levels near the core. Unless a sentence says
otherwise, $\mathrm{BW}$ is the bandwidth of main memory, the largest and slowest level, because that is where a working set larger than the caches lives.
The section on where the roofline stops being true returns to the others.

## Counting a computation: FLOPs and bytes

The other half of the comparison belongs to the computation and its sizes alone: its cost in the two currencies a machine charges. Start with the smallest case. Add two vectors of 4 numbers of 4 bytes, $c = a + b$. That is 4 additions. The processor must read the 4 numbers of $a$ (16
bytes), read the 4 numbers of $b$ (16 bytes) and write the 4 numbers of $c$ (16 bytes): 48 bytes for 4 operations. Whatever the vector length $n$, the ratio is the same: $n$
operations on $12n$ bytes, one twelfth of an operation per byte. No cleverness in the program changes it, because every
output needs its two inputs and each input is used once.

Define the two counts for any computation. $N_{\mathrm{flop}}$ is the number of floating-point operations, with a multiplication and an addition counting one each and
a fused multiply-add two (the convention of Volume IV, Chapter 2). $N_{\mathrm{byte}}$ is the number of bytes that must cross between main memory and the
arithmetic: at least every input read once and every output written once. It is a lower bound, because a program can be so careless
that it reads a number several times (the tiling section shows when), but no program can read less than the inputs it
needs. The ratio is the **arithmetic intensity**:

$$
\mathcal{I} = \frac{N_{\mathrm{flop}}}{N_{\mathrm{byte}}} \quad \text{(floating-point operations per byte)}.
$$

Intensity says how much arithmetic every byte moved is used for. It does not depend on the machine at all, only on the algorithm, the sizes and the
formats of the numbers.

Now the three operations. Volume I, Chapter 4 counted $2mnk$ operations for an $[m, k]$ matrix times a $[k, n]$ matrix. The matrix-vector product
$\mathbf{y} = \mathbf{W}\mathbf{x}$ with $\mathbf{W}$ of shape $[m, k]$ is the case $n = 1$: $2mk$ operations. It reads all of $\mathbf{W}$, which is $mk$ numbers, reads $\mathbf{x}$ ($k$ numbers) and writes $\mathbf{y}$
($m$ numbers). With weights of $b$ bytes and 4-byte activations that is $b\,mk + 4(m + k)$ bytes. For large $m$ and $k$ the matrix dominates the traffic, and

$$
\mathcal{I}_{\text{matrix-vector}} \to \frac{2mk}{b\,mk} = \frac{2}{b}.
$$

One weight, used for one multiplication and one addition and never again, costs $b$ bytes: half an operation per byte for 4-byte weights, 2 for
1-byte weights. The matrix-matrix product $\mathbf{C} = \mathbf{A}\mathbf{B}$ with all three of side $n$ and 4-byte numbers does $2n^3$ operations
and moves $3n^2$ numbers, $12n^2$ bytes:

$$
\mathcal{I}_{\text{matrix-matrix}} = \frac{2n^3}{12n^2} = \frac{n}{6}.
$$

The operation count grows as $n^3$ and the byte count as $n^2$ (Volume I, Chapter 16): $\Theta(n^3)$ against $\Theta(n^2)$, so the intensity is $\Theta(n)$. Every number
read takes part in about $n$ multiplications. The vector addition and the matrix-vector product have $\Theta(1)$ intensity at any size, because each number takes part in a
constant number of operations. Here they are for concrete sizes:

<!-- worked: roofline.three_operations -->
FLOPs count each multiplication and each addition as one. Bytes are the least that must cross between main memory and the arithmetic: every input read once, every output written once; vectors and matrices of activations are fp32 (4 bytes); weights are fp32 or int8 (1 byte, scales ignored) as stated. The matrix product moves A, B and C once each.

| operation | FLOPs | bytes | intensity (FLOP/byte) |
|---|---:|---:|---:|
| c = a + b, vectors of 1,048,576 | 1,048,576 | 12,582,912 | 0.0833 |
| y = W x, [4096, 4096], fp32 weights | 33,554,432 | 67,141,632 | 0.4998 |
| y = W x, [4096, 4096], int8 weights | 33,554,432 | 16,809,984 | 1.9961 |
| C = A B, [4096, 4096] by [4096, 4096], fp32 | 137,438,953,472 | 201,326,592 | 682.6667 |
<!-- /worked -->

Four rows, a span of about 8,200: from 0.0833 for the addition to 682.67 for the product. The int8 row shows the effect of storage format on the
matrix-vector product: the same 33,554,432 operations over a quarter of the weight bytes, so four times the intensity, nearly (the activations are still
4-byte numbers, so the ratio is 1.996 and not 2). Volume IV, Chapter 2 counted the same three operations at $n = 1024$ in its closing table; the counts here follow the
same rule and differ only in the size.

## Two lower bounds on time: the roofline

With the cost of a computation and the rates of a machine in hand, we can state the shortest time the machine could possibly take, and which of the two resources sets it. A computation with $N_{\mathrm{flop}}$ operations and $N_{\mathrm{byte}}$ bytes runs on a machine with $F_{\mathrm{pk}}$ and $\mathrm{BW}$. Its time $t$ satisfies two inequalities, and neither can be broken:

$$
t \ge \frac{N_{\mathrm{flop}}}{F_{\mathrm{pk}}} \qquad\text{(the arithmetic cannot go faster than the peak)},
\qquad
t \ge \frac{N_{\mathrm{byte}}}{\mathrm{BW}} \qquad\text{(the bytes cannot arrive faster than the bandwidth)}.
$$

Both hold at once, so $t \ge \max(N_{\mathrm{flop}}/F_{\mathrm{pk}},\ N_{\mathrm{byte}}/\mathrm{BW})$. The rate at which the operations are completed is $N_{\mathrm{flop}}/t$, and dividing the
operations by this lower bound on $t$ gives an upper bound on the rate. Write $F_{\mathrm{att}}$ for it:

$$
F_{\mathrm{att}} = \frac{N_{\mathrm{flop}}}{\max(N_{\mathrm{flop}}/F_{\mathrm{pk}},\; N_{\mathrm{byte}}/\mathrm{BW})}
= \min\!\left(F_{\mathrm{pk}},\; \frac{N_{\mathrm{flop}}}{N_{\mathrm{byte}}}\,\mathrm{BW}\right)
= \min\!\left(F_{\mathrm{pk}},\; \mathcal{I}\,\mathrm{BW}\right).
$$

The step to the second form divides the numerator into each term of the maximum: $N_{\mathrm{flop}}$ over $N_{\mathrm{flop}}/F_{\mathrm{pk}}$ is $F_{\mathrm{pk}}$, and $N_{\mathrm{flop}}$ over $N_{\mathrm{byte}}/\mathrm{BW}$ is
$\mathcal{I}\,\mathrm{BW}$, and a maximum in the denominator becomes a minimum of the quotients. The **roofline model** is this formula: the rate a computation can attain on a machine is the smaller of the machine's
peak arithmetic rate and its intensity times its memory bandwidth. It depends on the computation only through $\mathcal{I}$. Drawn against $\mathcal{I}$ on logarithmic axes it is a line of slope 1 that rises from the left (the
slanted part, $\mathcal{I}\,\mathrm{BW}$) and meets a horizontal line (the flat part, $F_{\mathrm{pk}}$), the shape of a roof, which is how it got its name.

The two parts meet where $\mathcal{I}\,\mathrm{BW} = F_{\mathrm{pk}}$, at

$$
\mathcal{I}_{\mathrm{ridge}} = \frac{F_{\mathrm{pk}}}{\mathrm{BW}},
$$

the **ridge point**: the last column of the machine table. A computation with $\mathcal{I} < \mathcal{I}_{\mathrm{ridge}}$ is **bandwidth-bound**: its attainable rate is on the slanted part, and what limits
it is the bytes, so making them fewer (smaller formats, reuse) raises the bound in proportion and making the arithmetic cheaper does nothing. A computation with $\mathcal{I} \ge \mathcal{I}_{\mathrm{ridge}}$ is **compute-bound**: the bound is the
flat part, and only fewer or cheaper operations, or a machine with more of them, help. The same computation can be on different sides of two machines' ridge points, as the next section shows. The test is one comparison:

```text
function roofline_bound(flops: int, bytes: int, F_pk: real, BW: real) -> (attainable: real, limit: text):
    require flops >= 0 and bytes >= 1 and F_pk > 0 and BW > 0
    intensity = flops / bytes
    attainable = min(F_pk, intensity * BW)
    if intensity < F_pk / BW:
        return (attainable, "bandwidth")
    return (attainable, "compute")
```

Two remarks on what the formula claims. First, it is a bound and not a prediction: it assumes that arithmetic and memory traffic overlap perfectly, so the time is the maximum of the two. If they did not overlap at all, the time would be their
sum. Since a sum of two positive numbers is between their maximum and twice their maximum, the sum can be at most twice the bound, so the bound is never off by more than a factor of two in this direction. The real distance
from the bound is everything else a program does badly, and measuring it is the subject of the section on the achieved fraction. Second, a machine has several $F_{\mathrm{pk}}$ (fp32 vector arithmetic, int8 dot products, matrix units) and several $\mathrm{BW}$ (one per level of memory); the
formula holds for each pair, and the question is always which pair applies to the computation at hand.

## Where the operations land

The formula earns its keep when the first sight of an operation tells where it stands. Apply $F_{\mathrm{att}} = \min(F_{\mathrm{pk}}, \mathcal{I}\,\mathrm{BW})$ to the four rows of the three-operations table on both machines:

<!-- worked: roofline.roofline_positions -->
Attainable GFLOP/s = min(F_pk, intensity x BW) on the two machines of `roofline.machines` (CPU: F_pk = 800, BW = 80, ridge 10; GPU: F_pk = 40,000, BW = 800, ridge 50, in GFLOP/s and GB/s), for the four operations of `roofline.three_operations`. The fraction is of the machine's peak arithmetic rate.

| operation | intensity | CPU attainable | CPU fraction of peak | CPU bound | GPU attainable | GPU fraction of peak | GPU bound |
|---|---:|---:|---:|---|---:|---:|---|
| c = a + b, vectors of 1,048,576 | 0.0833 | 6.7 | 0.0083 | bandwidth | 66.7 | 0.0017 | bandwidth |
| y = W x, [4096, 4096], fp32 weights | 0.4998 | 40.0 | 0.0500 | bandwidth | 399.8 | 0.0100 | bandwidth |
| y = W x, [4096, 4096], int8 weights | 1.9961 | 159.7 | 0.1996 | bandwidth | 1,596.9 | 0.0399 | bandwidth |
| C = A B, [4096, 4096] by [4096, 4096], fp32 | 682.6667 | 800.0 | 1.0000 | compute | 40,000.0 | 1.0000 | compute |
<!-- /worked -->

![Log-log roofline plot. The horizontal axis is the base-10 logarithm of arithmetic intensity from -2 to 3.4, the vertical axis the base-10 logarithm of attainable GFLOP/s. The GPU roof rises with slope 1 and flattens at the ridge at log 1.7; the lower CPU roof rises in parallel and flattens at the ridge at log 1. Green dots on the GPU roof mark vector addition, the fp32 matrix-vector product, the int8 matrix-vector product and the 4096 by 4096 matrix product; red dots mark fp16 decode of a 7-billion-weight model at batch sizes 1, 8, 64 and 512.](img/roofline.svg)

*Figure 3.1: the roofline of the two machines. The dashed verticals are the ridge points, 10 and 50 FLOP per byte. The dots sit on the GPU roof at their intensities.*

Read the table with the picture. The vector addition, the matrix-vector products and every operation of intensity below the ridge are on the slanted part: on the GPU
the fp32 matrix-vector product can attain at most a fraction 0.0100 of the machine's peak, the int8 one 0.0399, the addition 0.0017. These are not failures. They are what the
machine allows an operation that has so little arithmetic per byte, and a program that reached them would be perfect. The matrix product is far to the right of both ridges, on the flat part,
and can attain the whole peak on either machine. The CPU's lower roof reaches the flat part sooner (ridge 10), so the int8 matrix-vector product, whose intensity is 1.996, is still on the slanted part of both machines but gets a fraction 0.1996 of the CPU's peak against 0.0399 of the GPU's. The
machine with the bigger peak pays for it by being further from reaching it.

## Matrix size changes the intensity

Of the three operations only the matrix product has an intensity that can be raised, by making it larger, and the question is how large it must be to reach a ridge. For the square product the count is exact: $2n^3$ operations on $12n^2$ bytes. The table lists it for six sizes, and the last column is the closed form $n/6$, which agrees to
every digit shown:

<!-- worked: roofline.intensity_by_size -->
C = A B for square fp32 matrices of side n: FLOPs 2 n^3, bytes 4 (3 n^2) = 12 n^2 (A, B read once, C written once). The last column is the closed form n / 6.

| n | FLOPs | bytes | intensity | n / 6 |
|---|---:|---:|---:|---:|
| 16 | 8,192 | 3,072 | 2.667 | 2.667 |
| 64 | 524,288 | 49,152 | 10.667 | 10.667 |
| 256 | 33,554,432 | 786,432 | 42.667 | 42.667 |
| 1,024 | 2,147,483,648 | 12,582,912 | 170.667 | 170.667 |
| 4,096 | 137,438,953,472 | 201,326,592 | 682.667 | 682.667 |
| 16,384 | 8,796,093,022,208 | 3,221,225,472 | 2730.667 | 2730.667 |

The smallest side n for which the square product reaches the ridge: CPU (ridge 10) n = 60; GPU (ridge 50) n = 300.
<!-- /worked -->

The product reaches the ridge when $n/6 \ge \mathcal{I}_{\mathrm{ridge}}$, that is $n \ge 6\,\mathcal{I}_{\mathrm{ridge}}$: 60 for the CPU and 300 for the GPU, the values in the table's last line. Below that size even the matrix product is bandwidth-bound,
which is why a matrix multiplication of a small size is a different kind of computation from one of a large size, and why the shapes in a network matter as much as their count. The figure puts the four curves together.

![Log-log plot of arithmetic intensity against matrix side n from 10 to 100,000. The matrix product rises with slope 1 from about 1.7 to 16,000; the fp32 matrix-vector curve stays near 0.5, the int8 matrix-vector curve near 2, and the vector addition at 1/12. Dashed horizontal lines mark the CPU ridge at 10 and the GPU ridge at 50; the matrix product crosses them at n = 60 and n = 300.](img/intensity-vs-size.svg)

*Figure 3.2: intensity against the side $n$ of the matrices (for the vector addition, vectors of $n^2$ numbers). Only the matrix product climbs.*

Only one curve crosses the ridges. For the other three the problem is not size: no size makes a matrix-vector product reuse a weight, because each weight is used for one output. They can only be sped up
by moving fewer bytes, and the next two sections look at two ways: a different format and more vectors per weight.

## Decoding a token reads the model

The operation that dominates language-model generation is the matrix-vector product, so the roofline tells how fast generation can be, which is the question the chapter opened with. A decoder-only language model (Volume V builds one) generates one token by passing one new vector through every layer. A layer is a matrix product with the weights, and for one token it is a
matrix-vector product, so the whole model does $2P$ operations for $P$ weights and reads all $P$ weights once; if a weight occupies $b$ bytes, a token moves $Pb$ bytes and has intensity $2P/(Pb) = 2/b$ (activations and the memory of earlier tokens are small by comparison, and ignored here; an exercise adds them).
For a model of $P = 7 \times 10^9$ weights in the four storage formats of Chapter 1:

<!-- worked: roofline.decode_per_token -->
A dense model of P = 7,000,000,000 weights at batch size 1: one token is 2 P FLOPs and reads every weight once, so the bytes per token are P b for b bytes per weight (scales charged, as in Volume IV, Chapter 1) and the intensity is 2 / b. Activations and the cache of earlier tokens are ignored. Arithmetic is fp32 at the machine's peak whatever the storage format. The last column of each machine is the roofline's bound on tokens per second, BW / (P b): a model, not a measurement. GB = 10^9 bytes.

| weights stored as | bytes per weight | GB read per token | intensity | CPU fraction of peak | CPU tokens/s bound | GPU fraction of peak | GPU tokens/s bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 4.0000 | 28.000 | 0.50 | 0.0500 | 2.9 | 0.0100 | 28.6 |
| fp16 | 2.0000 | 14.000 | 1.00 | 0.1000 | 5.7 | 0.0200 | 57.1 |
| int8, f32 scale per 32 | 1.1250 | 7.875 | 1.78 | 0.1778 | 10.2 | 0.0356 | 101.6 |
| int4, f32 scale per 32 | 0.6250 | 4.375 | 3.20 | 0.3200 | 18.3 | 0.0640 | 182.9 |
<!-- /worked -->

Every row is bandwidth-bound on both machines: the intensities, 0.5 to 3.2, are far below the ridges of 10 and 50. The fraction of peak that a token attains is 1 percent of the GPU's compute in 32-bit weights and 6.4 percent in 4-bit ones, but that is the wrong
way to read the table. The right reading is the bound on tokens per second, $\mathrm{BW}/(Pb)$, which is the bandwidth divided by the bytes of weights: it is inversely proportional to the bytes per weight. Going from fp16 to the 4-bit
format with its scales divides the bytes by $2/0.625 = 3.2$ and multiplies the bound by 3.2: 57 tokens per second to 183 on the GPU, 5.7 to 18.3 on the CPU. The scales are charged in full, as
in Chapter 1, and they are why 4 bits is a factor of 3.2 and not 4.

![Bytes read per generated token, in GB, against the bytes per weight, for a model of 7 billion weights: a straight line through the origin with four points: int4 with scales 0.625 bytes per weight and 4.38 GB, int8 with scales 1.125 and 7.88 GB, fp16 2 and 14 GB, fp32 4 and 28 GB.](img/bytes-per-token.svg)

*Figure 3.3: bytes per token are the number of weights times the bytes per weight. For a bandwidth-bound computation the line is the whole story.*

The bound does not depend on how well the arithmetic is written. A decoder whose matrix-vector products are perfect still cannot beat it, and one that is beneath it by a factor of three has a gap that
is entirely in how it moves bytes. That is what the word "bound" is for.

## Batching moves decoding toward compute

The weights of a token are read once per token. If several tokens used them during a single read, the bytes per token would fall, and the question is how many tokens it takes. Generate for $B$ sequences at once. The matrix-vector product becomes a product of a $[B, D]$ matrix of activations with the $[D, O]$ weights. The operation count is $2BDO$, $B$ times the single-token count. The weights are read **once** for all $B$ rows, so the bytes are
$bDO$ for the weights plus $4B(D + O)$ for the activations in and out. The intensity is

$$
\mathcal{I}(B) = \frac{2BDO}{b\,DO + 4B(D + O)}.
$$

For small $B$ the weights dominate the denominator and $\mathcal{I} \approx 2B/b$: it grows in proportion to the batch, since the same bytes now support $B$ times the arithmetic. For very large $B$ the activations take over and $\mathcal{I}$ tends to $DO/(2(D + O))$, the intensity of a square-ish matrix product, which is large.
The table is for one layer with $D = O = 4096$ and fp32 activations, with the weights in three formats:

<!-- worked: roofline.batch_ladder -->
One layer [B, 4096] by [4096, 4096]: B rows of activations (fp32) times a weight matrix stored at b bytes per weight (fp32 b = 4; int8 and int4 with an f32 scale per 32 weights, b = 1.125 and 0.625). FLOPs 2 B (4096)^2; bytes b (4096)^2 + 4 B (4096 + 4096). Percent of the GPU's peak at the attainable rate min(F_pk, intensity x BW).

| B | I fp32 | I int8 | I int4 | GPU fraction of peak fp32 | GPU fraction of peak int8 | GPU fraction of peak int4 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.50 | 1.77 | 3.19 | 0.0100 | 0.0355 | 0.0638 |
| 2 | 1.00 | 3.54 | 6.36 | 0.0200 | 0.0709 | 0.1272 |
| 4 | 2.00 | 7.06 | 12.64 | 0.0399 | 0.1412 | 0.2528 |
| 8 | 3.98 | 14.03 | 24.98 | 0.0797 | 0.2805 | 0.4995 |
| 16 | 7.94 | 27.68 | 48.76 | 0.1588 | 0.5535 | 0.9752 |
| 32 | 15.75 | 53.89 | 93.09 | 0.3151 | 1.0000 | 1.0000 |
| 64 | 31.03 | 102.40 | 170.67 | 0.6206 | 1.0000 | 1.0000 |
| 128 | 60.24 | 186.18 | 292.57 | 1.0000 | 1.0000 | 1.0000 |
| 256 | 113.78 | 315.08 | 455.11 | 1.0000 | 1.0000 | 1.0000 |
| 512 | 204.80 | 481.88 | 630.15 | 1.0000 | 1.0000 | 1.0000 |
| 1024 | 341.33 | 655.36 | 780.19 | 1.0000 | 1.0000 | 1.0000 |

The smallest batch B whose intensity reaches the ridge:

| weights | bytes per weight b | CPU (ridge 10) | GPU (ridge 50) |
|---|---:|---:|---:|
| fp32 | 4.000 | 21 | 106 |
| int8 | 1.125 | 6 | 30 |
| int4 | 0.625 | 4 | 17 |
<!-- /worked -->

On the GPU the fp32 layer needs a batch of 106 before it leaves the slanted part, the int8 layer 30 and the 4-bit layer 17. The count for the CPU is much smaller, 21, 6 and 4, because its ridge is lower. Two things happen as the batch grows. First, the throughput in operations per second rises in proportion
(compare the fractions of peak in the fp32 column: 0.0100 at $B = 1$, 0.0200 at 2, 0.0400 at 4): doubling the batch doubles the work done per byte read, so the machine finishes twice as many tokens in about the same time. Second, past the ridge the intensity continues to rise but the attainable rate does not: the flat part has been reached, and more
batch only adds time in proportion.

The simplification $\mathcal{I} \approx 2B/b$, with the activations dropped, gives the batch at which a whole model reaches the ridge: $2B/b \ge \mathcal{I}_{\mathrm{ridge}}$, so $B^\ast = \lceil \mathcal{I}_{\mathrm{ridge}}\,b/2 \rceil$. On the GPU, 50 for fp16 weights and 16 for 4-bit ones. This simplification is what the model-wide table below uses, so its $B = 1$ intensity for 4 bits is exactly 3.2, while the layer table, which charges the activations, has 3.19.

<!-- worked: roofline.whole_model_batch -->
The 7,000,000,000-weight model on the GPU (F_pk = 40,000, BW = 800, in GFLOP/s and GB/s), batch B tokens at once: FLOPs 2 P B, weight bytes P b (read once for the whole batch; activations and the cache of earlier tokens ignored), intensity 2 B / b, tokens per second = B / max(2 P B / F_pk, P b / BW) from the roofline's two bounds (a model).

| B | I fp16 | fp16 bound | fp16 tokens/s | I int4 | int4 bound | int4 tokens/s | int4 / fp16 |
|---:|---:|---|---:|---:|---|---:|---:|
| 1 | 1.0 | bandwidth | 57 | 3.2 | bandwidth | 183 | 3.20 |
| 4 | 4.0 | bandwidth | 229 | 12.8 | bandwidth | 731 | 3.20 |
| 16 | 16.0 | bandwidth | 914 | 51.2 | compute | 2,857 | 3.13 |
| 32 | 32.0 | bandwidth | 1,829 | 102.4 | compute | 2,857 | 1.56 |
| 64 | 64.0 | compute | 2,857 | 204.8 | compute | 2,857 | 1.00 |
| 128 | 128.0 | compute | 2,857 | 409.6 | compute | 2,857 | 1.00 |
| 256 | 256.0 | compute | 2,857 | 819.2 | compute | 2,857 | 1.00 |

Smallest B reaching the ridge, ceil(ridge x b / 2): fp16 50, int4 16.
<!-- /worked -->

Batching is the lever behind every serving system, and it is limited by something else the roofline leaves out: each sequence has its own memory of earlier tokens, so a bigger batch needs more memory and brings more bytes of its own (Volume XIV treats serving). The roofline says
only that the weights are worth sharing, and how many it takes before sharing stops paying.

## Quantization helps where bytes limit

The quantization chapters later in this volume will cost accuracy for bytes, and the roofline says when bytes are worth buying. The last column of the model-wide table is the speed of the 4-bit model divided by that of the fp16 model. It is 3.20 at $B = 1$ and 4 and then falls: 3.13 at 16, 1.56 at 32 and 1.00 from 64 on. This is the roofline in one column. While both models are bandwidth-bound, the time is bytes divided by bandwidth and the ratio is
the ratio of bytes per weight, $2/0.625 = 3.2$. At $B = 16$ the 4-bit model has crossed its ridge and the fp16 model has not, so the 4-bit model's time stops falling while the fp16 model's still does, and the ratio narrows. Once both are at the flat part, the time is
operations divided by peak, which is the same for both because the arithmetic is the same (the weights are decoded and the arithmetic done in fp32, "store narrow, compute wide" of Chapter 1). Fewer bytes buy nothing there.

So a format with fewer bytes per weight is a speedup for bandwidth-bound work, in proportion to the bytes it removes, and no speedup for compute-bound work. It lowers the batch at which the ridge is reached (from 50 to 16 for fp16 against 4-bit on the GPU), which is also a gain: the machine reaches its peak with a smaller batch, so a single user waits less for it. Quantization also reduces the memory a model occupies, which is a different benefit that the roofline does not measure: 14 GB against 4.4 GB is whether the model fits at all.

A different arithmetic helps compute-bound work. Many machines have faster units for 8-bit integer dot products or for small matrices (Chapter 4 describes them), a second and higher flat part with its own $F_{\mathrm{pk}}$, and a computation can use them only if its operands are in the matching format. That is a faster roof and a different ridge, not fewer bytes, and Chapter 12 treats the formats that can use it.

## Training operations on the roofline

The roofline classifies every operation, not only inference, and a training step is made of the same three kinds, whose counts Volume III already gave in the form we need. Volume III, Chapter 5 counted the three matrix products of a linear layer, one forward and two backward, as $2BDO$ operations each over the same bytes, and Chapter 7 counted the operations and bytes of an optimizer's update per parameter. Their intensities, placed on the machines:

<!-- worked: roofline.training_operations -->
FLOPs and bytes as counted in Volume III, Chapters 5 and 7 (fp32; a layer reads X, W and writes Y once; an update reads w, g and the state and writes w and the state). Percent of peak at min(F_pk, intensity x BW) on the two machines of `roofline.machines`.

| operation | FLOPs | bytes | intensity | CPU fraction of peak | CPU bound | GPU fraction of peak | GPU bound |
|---|---:|---:|---:|---:|---|---:|---|
| linear layer forward, B = 64, D = 784, O = 256 | 25,690,112 | 1,069,056 | 24.03 | 1.0000 | compute | 0.4806 | bandwidth |
| linear layer forward, B = 8192, D = 784, O = 256 | 3,288,334,336 | 34,881,536 | 94.27 | 1.0000 | compute | 1.0000 | compute |
| linear layer backward dX or dW, B = 64 | 25,690,112 | 1,069,056 | 24.03 | 1.0000 | compute | 0.4806 | bandwidth |
| AdamW update, per parameter | 16 | 28 | 0.57 | 0.0571 | bandwidth | 0.0114 | bandwidth |
| SGD update, per parameter | 2 | 12 | 0.17 | 0.0167 | bandwidth | 0.0033 | bandwidth |
<!-- /worked -->

Three observations. The forward and the two backward products of a layer have the same intensity, 24.03 at a batch of 64: so they have the same bound on a given machine. That intensity is above the CPU's ridge of 10, so on the CPU the layer is compute-bound; and below the GPU's ridge of 50, so on the GPU the same layer is
bandwidth-bound, with an attainable fraction of 0.4806 of the peak. The classification belongs to the pair of computation and machine, never to the computation alone. Raising the batch to 8192 moves the layer past both ridges, as the batching argument says: weights are read once and used for more rows.

Second, the optimizer updates are bandwidth-bound everywhere, at 0.57 and 0.17 operations per byte for AdamW and SGD, and nothing that happens to the arithmetic changes that: the way to make an update cheaper is to move fewer bytes. Volume III, Chapter 7 gave one way, storing the moments in 8 bits, which moves 16.25 bytes in place of 28 for the same 16 operations and so raises the attainable rate in the ratio $28/16.25$.
That raises the intensity to 0.98 and leaves the update still far below any ridge.

## Data reuse: caches change the bytes

The byte count above was a lower bound that assumes each number is read from main memory once, and when a program does not manage that the count is larger. Tiling brings it back down, and we can say by how much. The matrix product reads each of $A$ and $B$ once if the processor can keep them somewhere near, but they do not fit. $A$ and $B$ at $n = 4096$ are 64 MiB each. Computing one entry $C_{ij}$ needs a row of $A$ and a column of $B$, and if the loops compute entries
one after another and nothing is kept, then each of the $n^2$ entries re-reads $2n$ numbers: $2n^3$ numbers, or $8n^3$ bytes, for $2n^3$ operations. The intensity is $1/4$, the intensity of a dot product, however large $n$ is.

Tiling (Volume IV, Chapter 2) brings numbers back by computing a $T \times T$ block of $C$ at a time. Keep the block in the cache. For each $k$, bring in the $T$ numbers of $A$ in the block's rows and the $T$ numbers of $B$ in its columns,
and update all $T^2$ entries: $T^2$ multiply-adds from $2T$ loaded numbers. The block's working set is $T^2 + 2T$ numbers, and the traffic is

$$
N_{\mathrm{byte}} = 4\left(\frac{n^2}{T^2}\cdot n \cdot 2T + n^2\right) = 4\left(\frac{2n^3}{T} + n^2\right),
$$

where $n^2/T^2$ is the number of blocks, each reading $2T$ numbers for each of the $n$ values of $k$, and the last term writes $C$ once. The intensity is

$$
\mathcal{I}(T) = \frac{2n^3}{4(2n^3/T + n^2)} = \frac{1}{4/T + 2/n}.
$$

At $T = 1$ it is the $1/4$ of the loop with no reuse. At $T = n$, the whole matrix as one block, it is $n/6$, the lower bound. In between it grows almost in proportion to $T$ (it is $T/4$ when $n \gg T$): every doubling of the tile halves the traffic.
What stops it is the cache: the tile must fit, $4(T^2 + 2T) \le C$ for a cache of $C$ bytes.

<!-- worked: roofline.tiling_traffic -->
C = A B for n = 4096, fp32. The cache holds a T by T tile of C and, for each k, the T numbers of A and the T numbers of B that update it: a working set of 4 (T^2 + 2 T) bytes. Main-memory traffic is 4 (2 n^3 / T + n^2) bytes (A and B re-read for every tile, C written once); FLOPs 2 n^3. Caches of Volume IV, Chapter 2: L1 32 KiB, L2 1 MiB, L3 32 MiB. The last two columns are percent of peak at min(F_pk, intensity x BW) for the two machines of `roofline.machines`; the model gives each machine the same cache, to isolate the effect of reuse.

| T | working set (bytes) | smallest level that holds it | DRAM bytes | intensity | CPU fraction of peak | GPU fraction of peak |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 12 | L1 (32 KiB) | 549,822,922,752 | 0.25 | 0.0250 | 0.0050 |
| 8 | 320 | L1 (32 KiB) | 68,786,585,600 | 2.00 | 0.1998 | 0.0400 |
| 64 | 16,896 | L1 (32 KiB) | 8,657,043,456 | 15.88 | 1.0000 | 0.3175 |
| 256 | 264,192 | L2 (1 MiB) | 2,214,592,512 | 62.06 | 1.0000 | 1.0000 |
| 1024 | 4,202,496 | L3 (32 MiB) | 603,979,776 | 227.56 | 1.0000 | 1.0000 |
| 2048 | 16,793,600 | L3 (32 MiB) | 335,544,320 | 409.60 | 1.0000 | 1.0000 |
| whole matrices | - | - | 201,326,592 | 682.67 | 1.0000 | 1.0000 |
<!-- /worked -->

On the CPU of Chapter 2's cache sizes, a tile of 64 fits in the first-level cache (16,896 bytes of 32 KiB) and already brings the intensity to 15.88, above the CPU's ridge of 10: the matrix product has been moved from bandwidth-bound to compute-bound by a loop transformation alone, with the operation count unchanged.
A tile of 256 fits the second level, and gives 62.06, above the GPU's ridge as well. That is the whole logic of a fast matrix multiplication: choose the tile so that the working set fits the nearest memory that can hold it, and the intensity measured against the farther memory rises to
the ridge. Chapter 6 builds the real thing, with the levels of a GPU, and shows how close to $n/6$ the best tilings come.

The effect is general. Any computation that uses a number several times gains from keeping the number near: fusing two operations so that the intermediate result never goes to main memory (Chapter 8) is the same idea with the roles of reuse and of the consumer exchanged.
Reading a number again after another has been read, within the cache, costs none of the bandwidth of main memory. The cache hierarchy is, in this chapter's terms, a way to lower $N_{\mathrm{byte}}$ at the level that is slow.

## The achieved fraction of the roofline

A stopwatch reading is a fact about one machine. To say how well a program uses a machine, or to compare across machines, we need a number with the machine divided out. Time a computation and count its operations and bytes. Its achieved rate is $N_{\mathrm{flop}}/t$ and its achieved bandwidth $N_{\mathrm{byte}}/t$. The **achieved fraction of the roofline** is the achieved rate divided by the rate the roofline allows at its intensity:

$$
\text{fraction} = \frac{N_{\mathrm{flop}}/t}{\min(F_{\mathrm{pk}},\ \mathcal{I}\,\mathrm{BW})} = \frac{t_{\text{bound}}}{t},
\qquad t_{\text{bound}} = \max\!\left(\frac{N_{\mathrm{flop}}}{F_{\mathrm{pk}}},\frac{N_{\mathrm{byte}}}{\mathrm{BW}}\right).
$$

It is the ratio of the best time the machine could do to the time taken, between 0 and 1 if the counts are right. For a bandwidth-bound computation it is the achieved bandwidth over the machine's bandwidth; for a compute-bound one it is the achieved rate over peak. Here are four runs. The stopwatch readings are
invented for the exercise, not measurements:

<!-- worked: roofline.achieved_fraction -->
HYPOTHETICAL stopwatch readings, invented for the example and not measurements: fp32 y = W x with W [4096, 4096] (33,554,432 FLOPs, 67,141,632 bytes, intensity 0.4998) and fp32 C = A B with all sides 4096 (137,438,953,472 FLOPs, 201,326,592 bytes, intensity 682.67). The machines are those of `roofline.machines`. Achieved rate = FLOPs / time; achieved bandwidth = bytes / time; roofline bound = min(F_pk, intensity x BW); fraction of the roofline = achieved rate / bound.

| operation | machine | reading (ms) | achieved GFLOP/s | achieved GB/s | roofline bound (GFLOP/s) | fraction of the roofline | fraction of peak compute |
|---|---|---:|---:|---:|---:|---:|---:|
| GEMV | CPU | 1.00 | 33.6 | 67.1 | 40.0 | 0.8393 | 0.0419 |
| GEMV | GPU | 0.30 | 111.8 | 223.8 | 399.8 | 0.2798 | 0.0028 |
| GEMM | CPU | 400.00 | 343.6 | 0.5 | 800.0 | 0.4295 | 0.4295 |
| GEMM | GPU | 5.00 | 27,487.8 | 40.3 | 40,000.0 | 0.6872 | 0.6872 |
<!-- /worked -->

Look at the matrix-vector rows. On the stopwatch the GPU wins, by more than three times (the reading is 0.30 against 1.00 milliseconds). On the roofline the CPU is at a fraction of 0.8393 of what it could do and the GPU at 0.2798 of what it could do: the GPU is the one with the larger gap, and a tuner would
work on it. The last column shows why the fraction of peak compute is not the measure for a bandwidth-bound operation. It says the CPU reached 0.0419 of its peak and the GPU 0.0028, which would suggest that both are doing awfully, when the CPU in fact is doing nearly the best the machine allows. In the matrix-product rows
the two fractions coincide, because the bound is the peak. The achieved fraction is the same measure for both kinds.

Three things the number must be taken with. It needs a denominator that is the machine's own, measured on it, because the peak of a data sheet is not what the machine delivers (the section on brain below has its example). It needs the numerator's counts to be complete: an operation count that leaves out
part of the work divides a partial number by the whole time and reports a low fraction that is false (brain's example is below). And a fraction above 1 is a diagnosis, never a result: the bound is a lower bound on time, so no correct count and timing can beat it. Either the time is wrong (a clock that stopped before the device finished), or the bytes are wrong (the data were already in a cache, and the relevant bandwidth is that
cache's, which is higher), or the machine's numbers are.

## Where the roofline stops being true

A bound is only as good as its assumptions. The model has two lines and the real machine has more, and each of the following puts the true time above the bound; the first of them is cheap to measure.

**Fixed costs.** Starting a computation on a GPU takes time that does not depend on its size. Take 5 microseconds for it (a model parameter) and add it to the roofline time of a vector addition:

<!-- worked: roofline.small_problems -->
c = a + b on the GPU of `roofline.machines` (BW = 800, in GB/s). Roofline time = 12 n bytes / BW. The model adds a fixed cost of 5 microseconds to start the dispatch (a parameter, not a measurement), so the fraction of the roofline reached is roofline time / (roofline time + 5 microseconds).

| n | elements | bytes | roofline time (microseconds) | with the 5 microsecond start (microseconds) | fraction of the roofline |
|---|---:|---:|---:|---:|---:|
| 2^10 | 1,024 | 12,288 | 0.02 | 5.02 | 0.0031 |
| 2^14 | 16,384 | 196,608 | 0.25 | 5.25 | 0.0468 |
| 2^18 | 262,144 | 3,145,728 | 3.93 | 8.93 | 0.4402 |
| 2^20 | 1,048,576 | 12,582,912 | 15.73 | 20.73 | 0.7588 |
| 2^22 | 4,194,304 | 50,331,648 | 62.91 | 67.91 | 0.9264 |
| 2^24 | 16,777,216 | 201,326,592 | 251.66 | 256.66 | 0.9805 |
| 2^26 | 67,108,864 | 805,306,368 | 1006.63 | 1011.63 | 0.9951 |

The fraction is one half when the roofline time equals the start cost: n = 5e-6 x 800e9 / 12 = 333,333 elements.
<!-- /worked -->

Below about 300 thousand elements the time is mostly the start, and the fraction of the roofline is small however good the kernel is. The size at which it is one half is where the roofline time equals the fixed cost, $n = t_0\,\mathrm{BW}/12$. A small problem is limited by neither of the two rates, but by latency,
and a chain of dependent steps (Chapter 2) is the same: its time is the length of the chain times the delay of a step, and neither peak nor bandwidth enters. The fix is the program's, never the machine's: do more work per start, which is what fusing steps and batching do.

**Access patterns.** The bandwidth is quoted for streaming in whole lines. A program that reads one 4-byte number from each 64-byte line, a strided read (Volume I, Chapter 5), uses a sixteenth of the bytes it moves, so its useful bandwidth is a sixteenth of $\mathrm{BW}$.
The byte count in the intensity should be the bytes moved, lines and all, and the count of the chapter's operations is the useful bytes, which is why the sections above call them a lower bound.

**Which roof.** A working set that fits in a cache is served at the cache's bandwidth, several times that of main memory. Grading it against the main-memory $\mathrm{BW}$ gives a fraction above 1. The roofline is a family, one slanted line per memory level.

**Overlap and the other units.** The bound assumes arithmetic and memory work at the same time, and that the arithmetic can use the units its peak counts. A computation made of int8 and fp32 work has two flat parts, and the time of the two kinds of arithmetic adds when they share the same units, which brain's profiler below shows to be a real mistake in practice.

## Habits that keep the roofline honest

Count first. Before any timing, write down $N_{\mathrm{flop}}$ and $N_{\mathrm{byte}}$ of the operation for the sizes at hand, and the intensity. Below the machine's ridge the work is on bytes; above it, on arithmetic or on getting reuse. Then take $F_{\mathrm{pk}}$ and $\mathrm{BW}$ from the machine and not from a table: measure the peak with a loop that does nothing but arithmetic, and the bandwidth with a loop that does nothing but stream a buffer much larger than any cache (a copy or a scaled add, 12 bytes per element for $a + s b$). A data sheet's value is not what the machine achieves.

Report the fraction together with its two counts, because "x percent of peak" without the intensity and the count it divides by cannot be checked, and treat a fraction above 1 as a bug in the measurement. When the numbers say the work is on bytes, move fewer bytes before computing less: smaller formats, fusion, reuse by tiling, batching. When they say it is on arithmetic, the remedies are fewer operations, a faster unit or a better schedule.

## How brain holds itself to its roofline

brain's hard rule (`AGENTS.md`) is this chapter's method turned into an engineering promise: brain runs at its roofline on every device, by selection and not by hand-tuning. It measures the device's roofline by microbenchmark (not from a specification), times the candidate kernels per operation and shape class,
records each chosen kernel's achieved fraction and lists the operations below threshold as the device's gap list. Four places carry it.

### The measured roofline in `crates/gpu-core/src/roof.rs`

The `Roofs` record holds what the chapter calls $F_{\mathrm{pk}}$ (`gflops`), $\mathrm{BW}$ (`gbs`), a second bandwidth for a working set that stays in cache (`cache_gbs`: the roofline is hierarchical, as the section on where the roofline stops being true says), and an int8 rate and a native-f16 rate where the device has such a path.
`Roofs::ridge` is $F_{\mathrm{pk}}/\mathrm{BW}$, `Roofs::classify` compares a kernel's intensity with it and returns `Bound::Compute` or `Bound::Memory`, and `Roofs::utilisation` is the achieved fraction of this chapter, graded against the roof of its own class. No compute interface reports either peak, so both are
measured. The arithmetic probe is a chain of fused multiply-adds held in registers with no memory traffic in its loop (`ROOF_FMA`: eight independent accumulators, 16 operations per thread per iteration, so that the dependency chain of Chapter 2 does not limit it); the bandwidth probe is the classic triad shape over two buffers of 64 Mi numbers
(`BW_BYTES_PER_ELEM = 12`, the vector-addition count of this chapter), far past any cache. Both calibrate their trip count until the timed region lasts at least `MIN_PROBE_SECONDS` and are bracketed by waiting for the device to finish, because a timing that only submitted the work measured the host. The result is persisted per device and keyed by a fingerprint of the probe sources, so editing a probe
invalidates old numbers. The probe deliberately measures the silicon and not "the best matrix product brain has written": grading kernels against brain's own best kernel would hide the gap that the method exists to close. The CPU backend has its own probe, `crates/backend-cpu/src/roofline.rs`, whose arithmetic rate comes from a representative mix of convolutions and not from a register-only chain, so it is the rate of a real kernel mix, a weaker bound than the GPU's.

Two stories from the history of that file are worth keeping. Commit 73466fee0 (August 2026) replaced the literals that every benchmark divided by (one card's datasheet numbers, copied into several benchmarks) with the measured roof. It reports, for the card it was developed on:

<!-- perf-number: historical measurement, commit 73466fee0 -->
In August 2026, on one older discrete GPU, the measured arithmetic roof came to a fraction of 0.898 of the datasheet value and the measured bandwidth roof to 0.831 of it.

Its own conclusion is the point: a datasheet peak is not an achievable peak, and grading kernels against a roof that nothing reaches builds a permanent pessimism into every row. Knowledge #27 states the lesson as the title says: a percentage of peak divided by a hard-coded peak is not a measurement, and the whole method (rank by distance from the roof, fix the top row, measure again) rests on that denominator. The commit also adds tests that are independent of the device on purpose: one that pinned a number would pass on exactly one card.
The second story is commit 1bbcfc77e (August 2026). A device that has been idle runs at a low clock and needs seconds of continuous work to reach the clock it will run a job at; the probe started timing at once, so on an integrated GPU it recorded the idle clock as the silicon's ceiling, and because the result is cached and persisted, that one cold probe became the denominator of every later fraction on the machine. The symptom was in the numbers: the measured arithmetic roof came out below the rate that a real matrix product reached on the same device,
which cannot happen if the roof is a bound. The fix runs the register-only chain for a warm-up period before the first rung (`ROOF_WARMUP`), and a test, `real_work_can_never_beat_the_roof` in `crates/gpu-core/tests/it/roofline.rs`, states the invariant of the previous section directly: real work against the measured roof, on the same device, must not exceed it. A fraction above 1 caught the bug, as the chapter says it should.

### The counts in `crates/gpu-core/src/cost.rs`

The achieved fraction needs the numerator, so brain keeps one registry that maps each dispatched kernel to its operation count, integer-operation count and bytes. Its header states the conventions of this chapter: one multiply-accumulate is two operations, and bytes are "best-effort streaming traffic" with each operand read or written once, "a roofline denominator, not a cache model", the lower bound of the
byte-count section. A kernel the registry does not know yields no count at all and is reported as uncovered, never as zero.

### The per-kernel table in `crates/gpu-core/src/profile.rs`

The profiler classifies each kernel as compute-bound or bandwidth-bound from the measured ridge and grades it against its own class's roof: reporting a streaming kernel as a fraction of an arithmetic peak is the mistake of the achieved-fraction table above. `Bound::target_pct` and `Bound::defect_pct` (in `roof.rs`) state the bands brain holds each class to: a target of 60 for compute-bound kernels and 70 for bandwidth-bound ones, and a defect line at 30 and 35, below which a kernel is a defect and not untuned.
Two lessons about the denominator and the numerator came out of it. Commit fc902a817 (August 2026), recorded as knowledge #28, found that a benchmark summed the operations of every kernel it had a formula for and divided by the time of the whole pass: ten of the 26 kinds of a backward pass had no formula, so the quoted rate was computed from a numerator missing about a third of the graph.

<!-- perf-number: historical measurement, commit fc902a817 -->
The rate quoted for that backward pass was 5.4 percent of peak, an under-report by construction.

The profiler now reports no rate at all for a partly covered pass and names the kinds it lacks, because the omitted kernels are usually cheap enough to fall outside the printed rows, which is where a missing formula hides. Knowledge #21 is the other half. A profiler that times groups of kernels must wait between them, so each group's number loses the overlap that kernel would have had in the real submission: the table is good for ranking and unsafe for credit. A change counts when the time of the whole pass moves, so `profile.rs` prints both the single-submission total and the sum of the groups.
Commit 2c4daefd7 (August 2026) is a related defect in grading an aggregate: taking the larger of an integer-operation count and a floating-point count against one roof made a whole int8 image-model stage come out cheaper than its own fp32 attention alone, so `Roofs::seconds_at_roof` adds the two arithmetic classes' times (they share the same units) and takes the maximum of that sum and the memory time (memory overlaps with arithmetic), the two kinds of combination described under overlap above.

### Selection by roofline in `crates/gpu-core/src/native_tune.rs`

The hard rule's other half is implemented by commit 72b363135 (October 2026). The decode linear layer for int8 weights is a set of candidates (a GEMV with 4-way dot products, a wide GEMV on matrix units, and tiled matrix-unit products of 16 and 32 rows), each declaring the device capability it needs and the shapes it serves. On the first use of a device, `native_tune` runs every candidate a device can run
on a grid of shapes (rows, $K$, bytes of weight: the classes of a decode step, so the batching ladder of this chapter), rejects any candidate that disagrees with a reference computed in 64-bit floating point, times the survivors with the weights cycled past the cache so that the stream is bound by main memory as in real decoding, picks the fastest and records its achieved bandwidth as a fraction of the device's measured roof. The table is cached per device, build and candidate source. Operations whose fraction is below
`GAP_THRESHOLD` (0.6) form the device's gap list, which `brain roofline gpu --kernels` prints (`crates/cli/src/roofline_cli.rs`). Models just call the operation. That is the chapter's batching and bandwidth arguments built into the dispatch: at one row the candidates are judged by the bytes of weights per second, at more rows by operations per second.

### The performance suite asks a different question

`crates/perf` is brain's performance suite, and it answers a different question from the roofline's: how much correct work, meeting a latency target, an engine delivers over time (requests, first-result time, goodput), with the device, backend and build recorded as axes so that two runs are compared only when they are comparable (`crates/perf/src/gate.rs`
refuses a comparison across different hardware or after a failed correctness check, and skips and reports a metric that one side never measured; unmeasured fields are null and never zero). The suite records the device's peak bandwidth in its environment record when it is known, but it does not itself grade a run against a roofline; that grading is in `gpu-core` and the profiling chapter (Chapter 11) joins the two.
A number from `crates/perf` is therefore a dated observation of one setup, and the quantity this chapter calls the achieved fraction comes from the roofline side.

## Check your understanding

1. A student says: "The GPU has 50 times the peak arithmetic rate of the CPU, so decoding a token is 50 times faster on it." Find the error, using the two machines of this chapter, and say by what factor the roofline bounds the two.
2. A matrix product of two $[4096, 4096]$ matrices in fp32 is compute-bound on both machines. The weights are converted to int8 (1 byte, scale ignored) and decoded to fp32 as they are read. Predict what happens to the roofline bound for the time, and give the reason.
3. For which $B$ does an fp16 model on the CPU (ridge 10) reach the ridge, using $B^\ast = \lceil \mathcal{I}_{\mathrm{ridge}}\,b/2 \rceil$? What does the answer say about batching on a CPU compared with a GPU?
4. Find the error: "An optimizer step reaches only 0.0113 of the GPU's peak arithmetic rate, so its code must be badly written."
5. Explain from the counts of operations and bytes why the intensity of the vector addition is the same for every length and that of the matrix product is not.
6. A benchmark reports that a kernel achieved a bandwidth above the machine's measured main-memory bandwidth. List two explanations, and say which one the measured cache bandwidth in brain's `Roofs` is there for.
7. With the tiling table in mind: a tile of 64 and a tile of 256 both give a compute-bound product on the CPU. Why would one still prefer the second on the GPU?
8. In the table of small problems the vector addition of 16,384 elements reaches a fraction of 0.0467 of its roofline. Give the explanation and the remedy.

## Summary

A computation has two costs that do not depend on the machine: the floating-point operations $N_{\mathrm{flop}}$ and the bytes $N_{\mathrm{byte}}$ it must move, and their ratio is its arithmetic intensity. A machine has two rates, its peak arithmetic rate $F_{\mathrm{pk}}$ and its memory bandwidth $\mathrm{BW}$.
No computation finishes before the arithmetic or the bytes allow, so the attainable rate is $\min(F_{\mathrm{pk}}, \mathcal{I}\,\mathrm{BW})$: the roofline model. Below the ridge point $F_{\mathrm{pk}}/\mathrm{BW}$ a computation is bandwidth-bound and only fewer bytes help; above it, compute-bound, and only less arithmetic or a faster unit does. The vector addition (intensity 1/12), the matrix-vector product (about 2 divided by the bytes per weight) and the
matrix product ($n/6$) lie at very different places, and so does decoding a token, which is a matrix-vector product per layer and is bandwidth-bound on any machine: its speed is set by the bytes of weights, and quantization speeds it in proportion to the bytes removed and not at all once batching has made it compute-bound.
Batching raises the intensity in proportion to the batch, and tiling raises that of a matrix product by keeping blocks in the cache; both lower the bytes that come from the slow memory. The honest measure of efficiency is the achieved fraction of the roofline, the best time the machine allows divided by the time taken, because it divides the machine out. It needs
the machine's own measured roofs, complete counts, and a suspicion of any value above 1. The roofline is a bound: fixed costs, strided access, caches with their own bandwidths and units with their own peaks put real time above it.

What this chapter could not say is what the machine is on the inside: how a GPU organizes the tens of thousands of lanes behind its peak, and what it demands of an algorithm to keep them all busy and its memory streaming. Chapter 4 opens the GPU.

## References

- Samuel Williams, Andrew Waterman and David Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", 2009.
- John D. McCalpin, "Memory Bandwidth and Machine Balance in Current High Performance Computers", 1995.

## Exercises

1. **[Calculate]** A feed-forward layer in a decoder has weights of shape $[4096, 11008]$ stored in fp16 (2 bytes per weight). For one token (one row of fp32 activations), compute the operations, the bytes (weights and activations) and the intensity. Is it bandwidth-bound on the CPU and on the GPU of this chapter?
2. **[Calculate]** A third machine has a peak arithmetic rate of 100,000 (GFLOP/s) and a bandwidth of 2,000 (GB/s). Find its ridge point and the batch at which an int8 model (1.125 bytes per weight with scales) reaches it, using $B^\ast = \lceil \mathcal{I}_{\mathrm{ridge}}\,b/2 \rceil$.
3. **[Calculate]** Add two vectors of $2^{28}$ elements in fp32 on the GPU of this chapter. Give the operations, the bytes and the roofline time in milliseconds, and the same for the CPU. What fraction of the GPU's peak arithmetic rate can it reach?
4. **[Calculate]** For $n = 4096$ and fp32 find the largest power-of-two tile $T$ whose working set $4(T^2 + 2T)$ bytes fits in 1 MiB, and the intensity $1/(4/T + 2/n)$ for it. Is it above the ridge of the GPU?
5. **[Calculate]** A run of the 4096 matrix-vector product ($N_{\mathrm{flop}}$ = 33,554,432, $N_{\mathrm{byte}}$ = 67,141,632) takes 0.2 milliseconds on the GPU of this chapter. Compute its achieved bandwidth and its achieved fraction. Is the answer possible?
6. **[Implement]** Reproduce `roofline.three_operations` and `roofline.roofline_positions` from the counts of this chapter: the operation and byte counts of the vector addition, the matrix-vector product and the matrix product, and $\min(F_{\mathrm{pk}}, \mathcal{I}\,\mathrm{BW})$ for the two machines.
7. **[Implement]** Reproduce `roofline.batch_ladder`: the intensity of the $[B, 4096]$ by $[4096, 4096]$ layer for the three formats at the eleven batch sizes, the fractions of the GPU's peak, and the search for the smallest batch that reaches each machine's ridge.
8. **[Implement]** Reproduce `roofline.tiling_traffic`. First write the blocked loop for a small $n$ (for example $n = 64$, $T = 8$) with a counter that adds 4 bytes for every number it loads from $A$ or $B$ and every number it stores to $C$, and check the count against $4(2n^3/T + n^2)$; then evaluate the formula at the table's tiles.
9. **[Implement]** Reproduce `roofline.small_problems` and `roofline.whole_model_batch` from the parameters in their headers, and extend the second with the 7-billion-weight model at 8-bit weights with scales.
10. **[Investigate]** Add the memory of earlier tokens to the decode model. A model has 32 layers and 8 key-value heads of dimension 128, stored in fp16, so that each token of context adds $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes that every later token reads. At what context length are the bytes of the memory equal to the 14 GB of fp16 weights? What does that do to the bound on tokens per second at batch 1 beyond that length, and what does it do to the benefit of batching?
11. **[Investigate]** On your own machine measure the bandwidth with a loop that computes $a_i = a_i + s\,b_i$ on arrays of 256 MiB (counting 12 bytes per element) and the arithmetic rate with a loop of independent fused multiply-adds in registers; then time an $[n, n]$ matrix product and a matrix-vector product at a few sizes and report each as a fraction of its own roofline. State your machine, compiler, number of repetitions and the minimum you took, and explain any fraction above 1.
12. **[Investigate]** Extend the model of the table of small problems to a CPU with a fixed cost of 1 microsecond: at what vector length is the fraction one half? Compare it with the GPU's 333,333 elements and say why a GPU needs a much larger problem to be efficient.

## Key terms

- **memory bandwidth**: the number of bytes per second that main memory can deliver to the processor when the program asks for them in the best possible way (long sequential reads), a property of the machine and not of the program. <!-- concept: memory-bandwidth -->
- **arithmetic intensity**: the number of floating-point operations a computation performs per byte it must move between main memory and the arithmetic, a property of the algorithm and the sizes and formats and not of the machine. <!-- concept: arithmetic-intensity -->
- **roofline model**: the bound that the attainable rate of a computation on a machine is the smaller of the machine's peak arithmetic rate and its arithmetic intensity times its memory bandwidth, drawn as a slanted line that meets a flat one at the ridge point. <!-- concept: roofline-model -->
- **bandwidth-bound**: limited by the bytes it moves: the arithmetic intensity is below the machine's ridge point, so the attainable rate is intensity times bandwidth and only moving fewer bytes makes it faster. <!-- concept: bandwidth-bound -->
- **compute-bound**: limited by arithmetic: the arithmetic intensity is at or above the machine's ridge point, so the attainable rate is the peak arithmetic rate and only fewer or cheaper operations, or a faster unit, make it faster. <!-- concept: compute-bound -->

## Answers

**Recall questions.**

1. $2mnk$ operations: $mn$ outputs, each a dot product of length $k$ with $k$ multiplications and about $k$ additions. (Volume I, Chapter 4.)
2. As $n^3 / n^2 = n$: the ratio is $\Theta(n)$. (Volume I, Chapter 16.)
3. The whole 64-byte line that contains it, because memory is answered in aligned lines. (Volume IV, Chapter 2.)
4. The number of cache lines fetched (the traffic), not the number of operations: the same additions are done in a different order. (Volume IV, Chapter 2.)
5. fp16 takes 2 bytes; int8 with a 32-bit scale per 32 values takes $1 + 4/32 = 1.125$ bytes. (Volume IV, Chapter 1.)
6. About twice (the backward pass has two products, $d\mathbf{X}$ and $d\mathbf{W}$, each the cost of the forward one). An AdamW update does 16 operations on 28 bytes of traffic, 0.57 per byte. (Volume III, Chapters 5 and 7.)

**Check your understanding**

1. The peak is not what limits decoding: the bandwidth is. The GPU has 800 (GB/s) and the CPU 80 (GB/s), a factor of 10, so the roofline bounds the GPU's speed at 10 times the CPU's, for example 57.1 and 5.7 tokens per second for fp16 weights. The factor of 50 in peak arithmetic is not available to an operation of intensity 1, which attains a fraction of 0.02 of the GPU's peak.
2. The bound does not change. The product is compute-bound, so its time is $N_{\mathrm{flop}}/F_{\mathrm{pk}}$, and the operations are the same: the weights are decoded to fp32 and the arithmetic is unchanged. Fewer weight bytes lower only the bandwidth term, which is not the maximum. (The model's memory is smaller, which is a different benefit.)
3. $\lceil 10 \times 2 / 2 \rceil = 10$. A CPU reaches the flat part with a batch of 10 where the GPU needs 50 for the same weights, so batching pays off sooner on the CPU and stops paying off sooner, since the machine is saturated with fewer tokens.
4. The optimizer step is bandwidth-bound: its intensity is 0.57, so the roofline allows a fraction of $0.57 \times 800 / 40000 \approx 0.0114$ of the peak, and 0.0113 is about all there is to get. The fraction of peak is the wrong denominator; the measure is the achieved bandwidth over the machine's bandwidth.
5. The addition does 1 operation per element on 12 bytes per element at every length: operations and bytes both grow as $n$. The matrix product does $2n^3$ operations on $12n^2$ bytes, $\Theta(n^3)$ against $\Theta(n^2)$, so each number read takes part in about $n$ operations.
6. The bytes may have been served from a cache (the working set fits in a cache, whose bandwidth is higher) or the counting or the timing is wrong (for example a clock stopped before the device finished). The measured cache bandwidth is there for the first: a kernel above the main-memory roof but below the cache roof is graded against the cache roof, and one above both is flagged as an accounting or timing error.
7. The GPU's ridge is 50, and a tile of 64 gives 15.88, below it; the tile of 256 gives 62.06, above. On the CPU 15.88 already clears the ridge of 10.
8. Its roofline time is 0.25 microseconds and the fixed cost of starting is 5: almost all the time is the start. Do more per start, with larger vectors or by fusing it with neighbouring operations.
