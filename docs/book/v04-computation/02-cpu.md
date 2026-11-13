<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# The CPU: Cores, Caches and Vectors

*Why does the same arithmetic run ten times faster when the data is arranged differently?*

Take a table of 16 rows and 16 columns of 4-byte numbers and add up every entry. There are 256 additions, and
nothing about them depends on the order, so you may add the table row by row or column by column. A
mathematician would say the two programs do the same work. A processor does not agree, and the reason is
not the additions but where the numbers are.

A processor does not read memory one number at a time. Main memory is slow compared with the arithmetic units
and is answered in blocks: a request for any one byte brings in the whole aligned block of 64 bytes that
contains it, called a **cache line**, and keeps a copy in a small, fast memory next to the arithmetic called a
**cache**. A line of 64 bytes holds 16 numbers of 4 bytes, so if the next 15 numbers are wanted next, they are
already there. The cache is small, so a new line pushes an old one out; the rule used here is that the line
that has gone longest without being used leaves first.

In a row-major table (Volume I, Chapter 5) a row of 16 numbers is exactly one line. Read down column 0 with a
cache that holds 4 lines, and watch the cache:

<!-- worked: cpu.lru_trace -->
A 16 by 16 matrix of 4-byte numbers, row-major from address 0 (a row is exactly one 64-byte line, so element A[i, j] is at address 4(16i + j) and is in line i). A cache of 4 lines, fully associative, least recently used first. The first ten reads of column 0, read down the column:

| read | element | address | line | result | lines held afterwards (oldest first) |
|---|---:|---:|---:|---|---|
| 0 | A[0, 0] | 0 | 0 | miss | 0 |
| 1 | A[1, 0] | 64 | 1 | miss | 0, 1 |
| 2 | A[2, 0] | 128 | 2 | miss | 0, 1, 2 |
| 3 | A[3, 0] | 192 | 3 | miss | 0, 1, 2, 3 |
| 4 | A[4, 0] | 256 | 4 | miss | 1, 2, 3, 4 |
| 5 | A[5, 0] | 320 | 5 | miss | 2, 3, 4, 5 |
| 6 | A[6, 0] | 384 | 6 | miss | 3, 4, 5, 6 |
| 7 | A[7, 0] | 448 | 7 | miss | 4, 5, 6, 7 |
| 8 | A[8, 0] | 512 | 8 | miss | 5, 6, 7, 8 |
| 9 | A[9, 0] | 576 | 9 | miss | 6, 7, 8, 9 |
<!-- /worked -->

Every read of column 0 asks for a different line, and the cache holds only 4, so by the time the column ends
the first lines are gone. Column 1 then starts, needs line 0 again, and finds it missing: the line was fetched
once and thrown away before its other 15 numbers were used. Counting only the lines fetched, for the whole
table and for caches of different sizes:

<!-- worked: cpu.lru_cliff -->
A 16 by 16 matrix of 4-byte numbers (256 reads, 16 lines), cache fully associative with least recently used replacement, all reads. Lines fetched when the matrix is read by rows and by columns:

| cache size in lines | by rows | by columns | columns / rows |
|---|---:|---:|---:|
| 4 | 16 | 256 | 16.0 |
| 8 | 16 | 256 | 16.0 |
| 15 | 16 | 256 | 16.0 |
| 16 | 16 | 16 | 1.0 |
| 17 | 16 | 16 | 1.0 |
| 32 | 16 | 16 | 1.0 |
<!-- /worked -->

Row by row, each line is fetched once, 16 lines in all, whatever the cache size. Column by column the count
is 256, one fetch per read and sixteen times as many, until the cache is big enough to hold a whole column
(16 lines); then it falls to the 16 of the row order, because the lines read for column 0 are still there for
columns 1 to 15. The additions are the same 256 in both cases. The data moved is not.

This chapter is about the machine that makes the difference. It starts with the core, the part that executes
instructions, and why even a plain sum can run slowly there. Then it follows the data: the levels of memory, a
small cache simulator that counts the fetches above, and what loop orders and tilings do to the count. Then it
turns to the two ways a core does many additions at once, vector instructions and several cores, and to the
traps in each. It closes by counting the work of a few operations, which is the first step of Chapter 3, and
reads the CPU backend of brain.

A note on how speed is stated. Seconds belong to one machine on one day, and the same program changes by a
factor of several between two machines. Counts do not: the lines fetched, the instructions issued and the
length of the longest chain of dependent operations are the same wherever you run the program. This chapter
reasons with counts. Where it needs a cost in cycles it takes it from a small model whose assumptions are
printed beside the result, and says so; it never reports a stopwatch reading as a fact about your computer.

The cache arguments below lean on layout and strides from Volume I and on its treatment of floating-point sums.

1. (Volume I, Chapter 5, row-major layout.) A $[m, n]$ array of 4-byte numbers is stored row-major starting at
   address 0. At what address is element $(i, j)$? What is the stride of the first axis, in elements?
2. (Volume I, Chapter 5, strides.) The transpose of a $[3, 4]$ row-major array is a view of shape $[4, 3]$.
   What are its strides, and what does reading it row by row do to the storage positions it visits?
3. (Volume I, Chapter 16, algorithms and reductions.) A sum of $n$ numbers takes $n - 1$ additions. In a
   tree reduction, how long is the longest chain of additions that depend on each other, for $n = 1024$?
4. (Volume I, Chapter 15, floating point.) Is addition of 32-bit floating-point numbers associative? What
   happens to the result of a long sum if the additions are done in a different order?
5. (Volume IV, Chapter 1.) How many bytes does an f32 take, and how many of them fit in a 64-byte line?

## What one core does

### Instructions, registers and a pipeline

Before we can say why an arrangement of data matters, we need a picture of the machine that consumes it, small
enough to count with. A program, to the processor, is a list of **instructions**: load the number at this address into a register, add
two registers, store a register to an address, jump. A **register** is a storage slot inside the arithmetic
unit itself, a few bytes wide, named in the instruction, and reading it costs nothing extra. A **CPU core** is
one such executing unit: it fetches instructions from memory, decodes them, finds the registers and numbers
they name, executes them in its arithmetic units, and writes the results back, while owning its own registers
and its own first caches. A chip holds several cores that share the rest of the memory system. Everything in
the first half of this chapter concerns one core.

A core does not finish one instruction before it starts the next. It is a **pipeline**: while one instruction
is being executed, the next is being decoded and the one after that fetched, as in an assembly line. A modern
core goes further. It looks ahead in the instruction list and starts any instruction whose inputs are ready,
in any order, and it has several execution units, so it can start a few instructions in the same cycle (a
**cycle** is one tick of the core's clock, the unit in which hardware latencies are quoted). Two numbers
describe an arithmetic instruction on such a core:

- its **latency**: the number of cycles from when it starts to when its result can be used;
- its **issue rate**: how many instructions of that kind the core can start per cycle.

For floating-point addition, typical values are a latency of a few cycles (the model below uses 4) and an issue
rate of one or two per cycle (the model uses 2). These are typical orders of magnitude, not properties of any
particular chip. What matters is the relationship between them: an instruction that needs the result of
another must wait out that instruction's latency, while an independent instruction can start at once.

### The dependency chain

The simplest loop in this book, a sum, already shows the cost of waiting, and it does so without any cache or
memory in the picture. Take eight numbers and add them with one running total, the way every textbook writes it:
$s \leftarrow s + x_i$ for $i = 1, \dots, 7$ after starting from $x_0$. Each addition needs the previous
total, so each must wait for the one before: a **dependency chain** of seven additions in a row, however many
execution units the core has. With a latency of 4 cycles, seven dependent additions take 28 cycles, while the
core could have started two additions every cycle.

Now use two running totals, one over the even-numbered elements and one over the odd, and add the two at the
end. The two chains are independent and can run side by side. With four totals the chains are two long; with
eight, one addition long and the work is almost all in the final combination. The table counts additions and the
longest chain, and schedules the additions on the model core (each cycle, start up to two additions whose
inputs are ready, in program order), for $L$ running totals combined at the end by a tree:

<!-- worked: cpu.dependency_small -->
Summing n = 8 numbers into L accumulators, then adding the L totals as a tree. Model: an addition takes 4 cycles from issue to result, the core can start 2 additions per cycle, operands are already in registers. Program: element i belongs to total i mod L; the first addition of a total adds its first two elements, each later one adds the next element of that total to its running value, so a total of m elements is a chain of m - 1 additions; then the L totals are combined in layers, adding adjacent pairs (total 0 with 1, 2 with 3, ...) and then adjacent pairs of those results, until one value is left. Program order is all the additions of total 0, then of total 1, and so on, then the layers from first to last, left to right. Schedule: cycles are numbered 0, 1, 2, ...; in each cycle the not yet issued additions are scanned in program order and the first two whose operands are available at the start of that cycle are issued (an addition that is not ready does not block the ones after it); an addition issued in cycle c delivers its result at the start of cycle c + 4. The modelled cycles are the cycle at which the last result is available; the dependent chain counts additions along the longest path. Here x0 ... x7 are the inputs and o0, o1, ... are the results of the additions in program order; the programs are:

- L = 1: o0 = x0 + x1; o1 = o0 + x2; o2 = o1 + x3; o3 = o2 + x4; o4 = o3 + x5; o5 = o4 + x6; o6 = o5 + x7
- L = 2: o0 = x0 + x2; o1 = o0 + x4; o2 = o1 + x6; o3 = x1 + x3; o4 = o3 + x5; o5 = o4 + x7; o6 = o2 + o5
- L = 4: o0 = x0 + x4; o1 = x1 + x5; o2 = x2 + x6; o3 = x3 + x7; o4 = o0 + o1; o5 = o2 + o3; o6 = o4 + o5
- L = 8: o0 = x0 + x1; o1 = x2 + x3; o2 = x4 + x5; o3 = x6 + x7; o4 = o0 + o1; o5 = o2 + o3; o6 = o4 + o5

| L | additions | longest chain of dependent additions | modelled cycles to the sum |
|---|---:|---:|---:|
| 1 | 7 | 7 | 28 |
| 2 | 7 | 4 | 16 |
| 4 | 7 | 3 | 13 |
| 8 | 7 | 3 | 13 |
<!-- /worked -->

The number of additions is the same for every $L$, 7 for 8 numbers, as Chapter 16 of Volume I says a
reduction takes $n - 1$ of them. What changes is the longest chain: Chapter 16 showed that regrouping the
combinations of a reduction into a tree cuts the longest dependent chain from $n - 1$ to $\lceil \log_2 n
\rceil$, and here the same regrouping is used inside one core, to give its execution units something to do.

For a long sum the effect is larger, and it saturates:

<!-- worked: cpu.dependency_chain -->
The same model with n = 1024: an addition takes 4 cycles from issue to result, the core starts 2 additions per cycle, so 8 independent additions are needed to keep it busy. A MODEL of a core, not a measurement of one. Program: element i belongs to total i mod L; the first addition of a total adds its first two elements, each later one adds the next element of that total to its running value, so a total of m elements is a chain of m - 1 additions; then the L totals are combined in layers, adding adjacent pairs (total 0 with 1, 2 with 3, ...) and then adjacent pairs of those results, until one value is left. Program order is all the additions of total 0, then of total 1, and so on, then the layers from first to last, left to right. Schedule: cycles are numbered 0, 1, 2, ...; in each cycle the not yet issued additions are scanned in program order and the first two whose operands are available at the start of that cycle are issued (an addition that is not ready does not block the ones after it); an addition issued in cycle c delivers its result at the start of cycle c + 4. The modelled cycles are the cycle at which the last result is available; the dependent chain counts additions along the longest path. The issue-limited cycles are the number of additions divided by 2, rounded up; the last column is that divided by the modelled cycles.

| L | additions | longest dependent chain | modelled cycles | issue-limited cycles | fraction of the issue rate |
|---|---:|---:|---:|---:|---:|
| 1 | 1023 | 1023 | 4092 | 512 | 0.125 |
| 2 | 1023 | 512 | 2048 | 512 | 0.250 |
| 4 | 1023 | 257 | 1029 | 512 | 0.498 |
| 8 | 1023 | 130 | 523 | 512 | 0.979 |
| 16 | 1023 | 67 | 523 | 512 | 0.979 |
| 32 | 1023 | 36 | 523 | 512 | 0.979 |
<!-- /worked -->

With one total the core completes an addition every 4 cycles, about one eighth of its modelled issue rate. With
$L = 2$ it is a quarter, with $L = 4$ a half, and with 8 totals the chain is no longer the limit: the count
is within two percent of the issue-limited number, and 16 or 32 totals add nothing. The rule behind the
table is that the core needs
$t_{\mathrm{lat}} \cdot r_{\mathrm{iss}}$ independent operations in flight to run at its issue rate, where
$t_{\mathrm{lat}}$ is the latency in cycles and $r_{\mathrm{iss}}$ the issue rate; here $4 \times 2 = 8$. That
is a statement about the model, and the count of independent chains is the portable part of it. A
compiler may not make the change for you, because changing how a floating-point sum is grouped changes its
result (Chapter 15 and Chapter 16 of Volume I); the section on vectors comes back to this.

What makes the picture a model is everything it leaves out: operands are assumed to be in registers, there is
no branch, and the core is assumed to find all its parallelism in the instruction list. Real cores are more
complicated in each respect, and the next sections are about the first one. Getting an operand into a register
is itself a request to the memory system, and that is where the table at the head of this chapter was won and
lost.

## The memory hierarchy

To count the cost of a loop we need to know what the levels of memory are, how large each is, and how a program's
accesses move data between them. No memory can be both large and near: fast memory is small, because it has to sit close to the arithmetic and be wired to it directly; large memory
is slow, because it is far away and shared. Hardware answers by stacking several memories of increasing size
and decreasing speed between the arithmetic and main memory. The stack is the **cache hierarchy**: each level
keeps copies of the lines most recently used from the level below it, so that a number needed again soon is
found near the core. The figure below shows one core's view of it.

![A column of five boxes from registers at the top to DRAM at the bottom, each with a typical capacity, a typical latency in cycles and who shares it: registers 640 B, 1 cycle, one core; L1 data cache 32 KiB, 4 cycles, one core; L2 cache 1 MiB, 14 cycles, one core; L3 cache 32 MiB, 45 cycles, all cores; DRAM 16 GiB, 250 cycles, all cores. Arrows between the boxes carry 64-byte lines.](img/memory-hierarchy.svg)

The same numbers as a table, with the capacity expressed in lines and the latency relative to the first level
of cache:

<!-- worked: cpu.hierarchy_table -->
TYPICAL orders of magnitude for one core of a recent server or desktop chip, rounded for arithmetic; they vary by chip and are not measurements of any machine. The register row counts 16 vector registers of 32 bytes (512 bytes) and 16 scalar registers of 8 bytes (128 bytes): 640 bytes.

| level | typical capacity | lines of 64 bytes | shared by | typical latency, cycles | relative to L1 |
|---|---:|---:|---|---:|---:|
| registers | 640 B | - | one core | 1 | 0.25 |
| L1 data cache | 32 KiB | 512 | one core | 4 | 1.00 |
| L2 cache | 1 MiB | 16,384 | one core | 14 | 3.50 |
| L3 cache | 32 MiB | 524,288 | all cores | 45 | 11.25 |
| DRAM | 16 GiB | 268,435,456 | all cores | 250 | 62.50 |
<!-- /worked -->

The values are rounded orders of magnitude for a recent desktop or server core, chosen so that the arithmetic
in this chapter is easy to follow. They are typical, not measured on any machine, and a particular chip
differs in every entry; what is stable across chips is the shape: each cache level is about thirty times
larger than the one above and three to four times slower, with main memory a larger step in both.

The unit that moves between levels is the cache line. The 64 bytes is a property of the hardware that has
been the same on most chips for years, and the book assumes it throughout: every cache in the worked tables has 64-byte lines.
Two habits of programs make this design pay.

**Spatial locality** is the habit of using numbers near one another in memory soon after one another. A
program that reads an array from start to end uses every number of every line it fetches: one fetch serves 16
reads of 4-byte numbers.

**Temporal locality** is the habit of using the same number again soon. A number used twice in quick
succession is fetched once, because the second use finds it in the cache. A program with neither habit, one that
touches a different line on every read and does not come back to it, pays a fetch per read.

A line travels as a whole, so two numbers that merely share a line also share its fate: if one thread
writes one of them, the other thread's cached copy of the line is out of date, a fact the section on cores
returns to.

## A cache you can run

"The cache is small, so lines get evicted" is not something you can count with. A precise, small rule is, and the
same rule gives every table in this chapter. The model is a **set-associative cache with least-recently-used replacement**. A cache of $C$ bytes is made of
lines of $s_{\mathrm{line}} = 64$ bytes, grouped into $n_{\mathrm{sets}}$ sets of $n_{\mathrm{ways}}$ lines each,
so $C = n_{\mathrm{sets}} \cdot n_{\mathrm{ways}} \cdot s_{\mathrm{line}}$. Memory is cut into line numbers
$\ell = \lfloor a / 64 \rfloor$ for address $a$, and a line may live only in the set $\ell \bmod
n_{\mathrm{sets}}$; inside the set it may sit in any of the $n_{\mathrm{ways}}$ places. An access to address $a$
is a **hit** if line $\ell$ is in its set; otherwise it is a **miss**: the line is fetched, and if the set is
already full the line unused for longest in it is evicted. The count of misses is the number of lines fetched.

When $n_{\mathrm{sets}} = 1$ the cache is **fully associative**: any line may be anywhere, and only the total
capacity limits it. Real first-level caches are 8-way associative or thereabouts, which is a compromise between
the cost of searching every line and the freedom of placement; the book uses both, a fully associative cache as
the clean case and an 8-way cache with 64 sets as the realistic one, each of 32 KiB (512 lines), a typical size
for a first-level data cache. Writes are treated as reads: a write to a line that is not present fetches it
first. The model has one level, no prefetching and ignores the order in which fetches overlap in time. It
counts fetches and nothing else.

The rule as an algorithm, with each way holding a line number (or $-1$ when empty) and the value of a counter at
the last time it was used:

```text
record Cache:
    sets: int
    ways: int
    tag: array<int>[sets, ways]      # line number held, or -1 when the way is empty
    stamp: array<int>[sets, ways]    # clock value when the way was last used (0 if never)
    clock: int
    misses: int

function access(address: int, inout cache: Cache) -> bool:
    require address >= 0
    line = address // 64
    s = line mod cache.sets
    cache.clock = cache.clock + 1
    victim = 0
    for w in 0 .. cache.ways:
        if cache.tag[s, w] == line:
            cache.stamp[s, w] = cache.clock      # a hit: now the most recently used
            return true
        if cache.stamp[s, w] < cache.stamp[s, victim]:
            victim = w                           # the least recently used way so far
    cache.tag[s, victim] = line                  # a miss: evict the victim, fetch the line
    cache.stamp[s, victim] = cache.clock
    cache.misses = cache.misses + 1
    return false
```

Empty ways have stamp 0, below any used way, so they are filled before anything is evicted, and ties among them
go to the lowest way. The 16 by 16 example of the first page is this function run on the addresses
$4(16 i + j)$ of the elements in the order of the loop, with $n_{\mathrm{sets}} = 1$ and $n_{\mathrm{ways}}$ equal to
4, 8, 15, 16, 17 and 32.

## Reading a matrix by rows and by columns

The first page showed a cliff at a small size. Real matrices are big, and the cliff, together with the extra cliffs
that associativity adds, decides whether a loop nest is cheap or expensive.

### The cliff

Why does the row order fetch each line once and the column order fetch every line again for each column? Read
the matrix by columns, with $n$ rows of 4-byte numbers per column. The $n$ reads of one column touch $n$
different lines (when a row is a multiple of 16 numbers long, so rows start on line boundaries). The next column
uses the same $n$ lines again: column $j + 1$ of row $i$ is in the same line as column $j$ of row $i$ unless
$j$ is the last element of its line. So the lines of a column are reused 15 times in a row, provided they are
still in the cache when the next column asks for them, and that needs the cache to hold the $n$ lines. If it
holds fewer, least-recently-used replacement makes it hold the wrong ones: when the column ends, the lines it
read first were used longest ago and have just been evicted, which is exactly the ones wanted next. Every read
then misses, and the fetch count is $n^2$ against the $n^2 / 16$ of the row order.

So the model says the cost of the column order depends on one comparison: the number of lines in a column
against the number of lines in the cache. The next table shows it for matrices up to 1024 by 1024 with the 32 KiB cache
of 512 lines, in both associativities:

<!-- worked: cpu.traversal_sizes -->
An n by n row-major matrix of 4-byte numbers, every element read once, row by row or column by column. Cache model: 32 KiB (512 lines of 64 bytes), least recently used replacement; "full" is fully associative (one set of 512 ways), "8-way" has 64 sets of 8 ways. Lines fetched:

| n | elements | matrix size | rows | columns, full | columns, 8-way | full / rows | 8-way / rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 1 KiB | 16 | 16 | 16 | 1.0 | 1.0 |
| 32 | 1,024 | 4 KiB | 64 | 64 | 64 | 1.0 | 1.0 |
| 64 | 4,096 | 16 KiB | 256 | 256 | 256 | 1.0 | 1.0 |
| 96 | 9,216 | 36 KiB | 576 | 576 | 576 | 1.0 | 1.0 |
| 128 | 16,384 | 64 KiB | 1,024 | 1,024 | 16,384 | 1.0 | 16.0 |
| 192 | 36,864 | 144 KiB | 2,304 | 2,304 | 36,864 | 1.0 | 16.0 |
| 256 | 65,536 | 256 KiB | 4,096 | 4,096 | 65,536 | 1.0 | 16.0 |
| 384 | 147,456 | 576 KiB | 9,216 | 9,216 | 147,456 | 1.0 | 16.0 |
| 512 | 262,144 | 1,024 KiB | 16,384 | 16,384 | 262,144 | 1.0 | 16.0 |
| 640 | 409,600 | 1,600 KiB | 25,600 | 409,600 | 409,600 | 16.0 | 16.0 |
| 768 | 589,824 | 2,304 KiB | 36,864 | 589,824 | 589,824 | 16.0 | 16.0 |
| 1024 | 1,048,576 | 4,096 KiB | 65,536 | 1,048,576 | 1,048,576 | 16.0 | 16.0 |
<!-- /worked -->

![Lines fetched per 16 elements read against the matrix side n, from 16 to 1024. Reading by rows costs 1 for every n. Reading by columns with a fully associative cache costs 1 up to n = 512 and 16 from n = 640 on. With an 8-way cache it costs 1 up to n = 96 and 16 from n = 128 on.](img/cache-lines-traversal.svg)

The figure plots the same rows as lines fetched per 16 numbers read, so that the row order is a flat line at 1 (one
fetch serves 16 numbers) and the worst case is 16. The fully associative column order stays at 1 up to
$n = 512$, where a column is 512 lines, exactly the cache, and goes to 16 beyond it. The 8-way cache goes
to 16 much sooner, at $n = 128$, a matrix of only 64 KiB, though its capacity is the same.

### Why the 8-way cache fails early

The set a line lives in is $\ell \bmod 64$. In a matrix with $n = 128$ a row is 128 numbers, 512 bytes, or 8
lines; the lines of a column are $8 i$ lines apart, so their set numbers are $8 i \bmod 64$, and the 128 lines
of the column fall into only 8 different sets, 16 of them in each. A set holds 8 lines, so only 64 of the 128 fit
at once, and LRU then evicts as before. The cache has plenty of room but not in the places where this access
pattern puts its data. The same happens at $n = 192$ (a row is 12 lines, so the column falls into 16 sets), 256,
384 and 512. It does not happen at $n = 96$ (a row is 6 lines, 32 sets are used, $32 \times 8 = 256$ lines fit
in a column of 96). This is a **conflict**, as opposed to a capacity shortage, and the practical rule is that
strides that are multiples of a power of two (the row length of a matrix whose side is a power of two) are the
worst case: many lines, few sets. The usual remedy is padding, which a later table demonstrates: give each row
a few unused elements at the end so that the row length is not a power of two.

### What it costs, as a model

A fetch is not free, and the cost of a fetch depends on where the line comes from. The following is a model
and not a measurement: a 1024 by 1024 matrix (4 MiB) sits in the L3 cache, each read costs the latency of L1
when the line is in the first-level cache, and a line fetched from L3 adds the difference of the two typical
latencies of the hierarchy table above. Nothing overlaps and nothing is prefetched.

<!-- worked: cpu.hierarchy_cost_model -->
A MODEL, not a measurement. A 1024 by 1024 matrix of 4-byte numbers (4 MiB) that sits in L3, read once in each order, 8-way 32 KiB cache. Each read costs 4 cycles when its line is in L1, and a line fetched from L3 adds the difference 41 cycles (the typical latencies of the hierarchy table; no prefetching, no overlap between fetches). Expected cycles per element = 4 + (lines fetched / elements) * 41:

| order | lines fetched | fetches per element | modelled cycles per element |
|---|---:|---:|---:|
| by rows | 65,536 | 0.0625 | 6.6 |
| by columns | 1,048,576 | 1.0000 | 45.0 |

Modelled ratio: 6.9.
<!-- /worked -->

The modelled ratio is about seven, not sixteen: the 4 cycles of the L1 hit are paid in both orders, and
16 times as many fetches multiplies only the part of the cost that is the fetch. Real machines differ in two
ways that go in opposite directions. A core detects that a program is reading consecutive lines and fetches
the next ones early (a **prefetcher**), which makes the row order cheaper than the model says; and several
fetches can be in flight at once, which makes the column order cheaper than the model says. Neither can be
derived from the counts here, and a figure for your machine has to be measured on it, with the measuring
discipline of Chapter 11. The count of lines fetched is the part that holds everywhere, and it already says
which order has more to do.

## Tiling: making the loop fit the cache

Sometimes both orders are needed. The transpose of a matrix reads one by rows and writes the other by columns
whatever order you pick, and a tiling changes the order of visits so that the working set fits. Copy a matrix into its transpose: $B_{ji} = A_{ij}$ for all $i, j$. Visiting $(i, j)$ in row order reads $A$ by
rows and writes $B$ by columns, or the other way round; one of the two is the column order of the last section.
The matrices here are 256 by 256 (256 KiB each), and the cache of this experiment is smaller, 8 KiB or 128 lines, to
keep the arithmetic short; the story is the same at 32 KiB with larger matrices.

The remedy is to cut the index space into $T \times T$ tiles and finish one tile before starting the next. The
visit order of a small example is in the figure below: the left panel is the row order of an 8 by 8 grid, the right
panel the same grid in 4 by 4 tiles. Cells with consecutive numbers are visited one after the other. In the row
order, a run of 16 consecutive steps spans two whole rows of the grid; in the tile order, 16 consecutive steps are
a square block that touches four rows and four columns.

![Two 8 by 8 grids of cells numbered by the order of visiting. Left, row by row: 0 to 7 in the first row, 8 to 15 in the second, and so on to 63. Right, tile by tile with 4 by 4 tiles: the top-left tile holds 0 to 15, the top-right tile 16 to 31, the bottom-left 32 to 47 and the bottom-right 48 to 63, each tile filled row by row.](img/tile-order.svg)

Why does this reduce fetches? Inside one tile the transpose reads $T$ rows of $A$, each $T$ numbers
long, and writes $T$ rows of $B$ of the same length. The lines of $A$ it reads are each used in full, and never
again; they are streamed through. The lines of $B$ are the issue: row $j$ of the tile, for $i$ going over the
tile's rows, is written at $B_{j, i_0}, B_{j, i_0 + 1}, \dots$, so as $i$ advances by one, each of the $T$ lines
being filled receives one more number and the $T$ lines are all in use at the same time. A line is complete after
16 values of $i$; during those 16 steps the loop needs the $T$ lines of $B$ it is filling, plus the one line of
$A$ it is reading, to stay in the cache. If $T + 1$ lines do not fit, they are evicted before they are complete
and fetched again; if they fit, each line of $B$ is fetched once. The working set of the tiled loop is about
$T + 1$ lines, against $n$ for the untiled one, and $T$ is chosen. The table counts the fetches:

<!-- worked: cpu.tiled_transpose -->
B[j][i] = A[i][j] for a 256 by 256 matrix of 4-byte numbers (256 KiB each; B starts right after A), visited tile by tile: tiles of T by T elements in row-major order of tiles, each tile in row-major order. Cache model: 8 KiB (128 lines), least recently used; "full" is fully associative, "8-way" has 16 sets of 8 ways, "8-way, padded" lays both matrices out with rows of 272 elements instead of 256 (the data is the same). Lines fetched; the least possible is every line once, 2 * 256 * 256 / 16 = 8,192.

| T | full | 8-way | 8-way, padded |
|---|---:|---:|---:|
| 1 | 69,632 | 69,632 | 69,632 |
| 2 | 36,864 | 36,864 | 36,864 |
| 4 | 20,480 | 21,056 | 20,480 |
| 8 | 12,288 | 16,128 | 12,288 |
| 16 | 8,192 | 69,632 | 8,192 |
| 32 | 8,192 | 69,632 | 8,192 |
| 64 | 8,192 | 69,632 | 8,192 |
| 128 | 69,632 | 69,632 | 36,002 |
<!-- /worked -->

Read the "full" column first. With no tiling ($T = 1$ is the plain row order) the count is 69,632, about 8.5 times
the least possible, 8,192, which is every line of both matrices once. As $T$ grows the count falls (36,864,
20,480, 12,288) and reaches the least possible at $T = 16$, the size at which a tile row is a whole line,
and stays there for $T = 32$ and $T = 64$. At $T = 128$ it jumps back to 69,632: the 129 lines of the working
set do not fit in 128, and the whole benefit is lost at once, as at the cliff of the first page. A tile has to be small enough
to fit and large enough that its rows fill lines; between those two bounds the choice does not matter in this
model, which is why a tile size is usually picked from a table like this one and then left alone.

The "8-way" column is the same experiment in a realistic cache, and it fails earlier for the reason of the previous
section. The rows of both matrices are 256 numbers, 16 lines, apart, and with 16 sets of 8 ways every
row begins in set 0 modulo 16, so the lines of a tile's column all compete for the same few places. A tile of
8 is not yet hurt much (16,128), but from $T = 16$ the count is back at 69,632 as if there was no
tiling at all. The "8-way, padded" column is the remedy: lay the two matrices out with rows of 272 numbers
instead of 256 (one line of padding, the data is the same) and the lines of a tile's column spread over 16
different sets. The count is then the same as in the fully associative case for all $T$ up to 64.

Two lessons carry forward. A tiling is a loop transformation: it changes the order of the
iterations and leaves every individual operation unchanged, so it can be done without changing the result of a
transpose, which has no sums to reorder. And a tile is sized by a byte budget (the lines it keeps live, against the cache's
lines), not by a favourite number: a number that is right for one row width is wrong for another, which brain's
code learned the hard way, as the last section tells. Chapter 6 uses the same idea for the three-line matrix
multiply, where the tile is chosen so that the pieces of two matrices it multiplies stay in cache while every
number in them is used many times.

## Vectors: one instruction on many numbers

The previous sections kept the arithmetic fixed and changed which memory it touched. The next lever changes how
many numbers one instruction handles. The registers of a modern core are wider than one number. A 256-bit register holds eight f32 numbers, and an
instruction that works on all of them at once adds eight pairs, or multiplies eight pairs, in a single
instruction with the latency and issue rate of a single one. This is **SIMD** (single instruction, multiple
data), and the groups are **vectors** in the machine sense, distinct from the mathematical vectors of Volume I;
the number of numbers per register is its number of **lanes**. The instruction sets of the x86 family offer 4
lanes (128-bit, called SSE), 8 lanes (256-bit, AVX2) and 16 lanes (512-bit, AVX-512) of f32; AArch64 has 4
lanes (NEON). A core that supports a wider set is detected at run time, and a well-built program carries a
version of each hot loop for each width and picks one when it starts.

### Instruction counts

Consider the dot product of two f32 arrays of length $n$. A scalar loop issues, per element, two loads, one
multiplication and one addition. Most cores have a **fused multiply-add**, a single instruction computing
$a b + c$ with one rounding; then it is two loads and one multiply-add per element, 3 instructions. In
a vector loop, each step loads $n_{\mathrm{lane}}$ numbers of each array and does the multiply-add on all lanes
at once, so the same 3 instructions serve $n_{\mathrm{lane}}$ elements. At the end the $n_{\mathrm{lane}}$ partial
sums have to be added together, a **horizontal reduction**: halve the lanes by adding the upper half of the
register to the lower half, and repeat. Counting the instructions of the loop and of the reduction, for $n = 1024$:

<!-- worked: cpu.simd_instructions -->
A dot product of two f32 arrays of n = 1024 elements, one fused multiply-add per step. Counting model: each step issues two loads and one multiply-add (3 instructions) for `lanes` elements; the horizontal sum at the end costs one shuffle and one add per halving of the lanes; loop control and the address arithmetic are not counted.

| width | lanes | bytes per load | steps | instructions in the loop | horizontal sum | total | scalar total / total |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar | 1 | 4 | 1,024 | 3,072 | 0 | 3,072 | 1.00 |
| SSE (128-bit) | 4 | 16 | 256 | 768 | 4 | 772 | 3.98 |
| AVX2 (256-bit) | 8 | 32 | 128 | 384 | 6 | 390 | 7.88 |
| AVX-512 (512-bit) | 16 | 64 | 64 | 192 | 8 | 200 | 15.36 |
<!-- /worked -->

The loop's instruction count falls by the number of lanes: 3,072 become 390 with 8 lanes and 200 with 16,
and the reduction at the end is a fixed cost of a few instructions. The instruction count is the portable
statement. How fast the instructions run depends on whether the loads keep up, which is the subject of Chapter 3,
and the 16-lane machine is not a factor of two faster than the 8-lane one for a loop that is already limited
by the memory it reads. But a core that is limited by its instructions, as the single-total sum was, is
helped directly.

Written as an algorithm, with $n_{\mathrm{lane}}$ lanes, the vectorized dot product is:

```text
function dot_lanes(x: array<f32>[n], y: array<f32>[n], w: int) -> f32:
    require w == 1 or w == 2 or w == 4 or w == 8 or w == 16 or w == 32
    acc = zeros(f32, [w])                       # one running total per lane
    blocks = n // w
    for b in 0 .. blocks:                       # one vector instruction per b
        for j in 0 .. w:                        # all w lanes at the same time
            acc[j] = acc[j] + x[b * w + j] * y[b * w + j]
    h = w
    while h > 4:                                # fold the upper half onto the lower half
        h = h // 2
        for j in 0 .. h:
            acc[j] = acc[j] + acc[j + h]
    if h == 4: s = (acc[0] + acc[1]) + (acc[2] + acc[3])
    else if h == 2: s = acc[0] + acc[1]
    else: s = acc[0]
    for i in blocks * w .. n:                   # the tail: fewer than w elements left
        s = s + x[i] * y[i]
    return s
```

The inner `for j` loop is a single instruction on a real core, not a loop; the algorithm shows the lanes
explicitly because the order of the additions is the whole story of the next paragraphs. The line
`acc[j] + x[..] * y[..]` rounds the product and then the sum, as the pseudocode's operations do, and that is how the
worked sums of this section compute it. The instruction count above is for a core that fuses the two into one
multiply-add with a single rounding; without fusion each step is four instructions (two loads, a multiplication, an
addition) and the counts become $4n/n_{\mathrm{lane}}$ plus the same reduction. The grouping of the additions, which
is what this section studies, is identical either way, though the last digits are not. The condition allows $w = 32$, which stands for two 16-lane registers
used as one group of accumulators, so that the worked examples can show a width beyond one register. Lane $j$ adds the
products $j, j + w, j + 2w, \dots$ in order. The fold at the end is the one the 256-bit kernels of brain's
`hsum256` use: add the high 128 bits to the low (lanes $j$ and $j + 4$), then two pair additions. The tail
handles the $n \bmod w$ elements that do not fill a vector.

### Alignment

A vector load of 32 bytes can come from any address. It is cheapest when it does not straddle two cache lines,
because a load that does needs two lines. A load whose start address is a multiple of its width is **aligned**,
and cannot straddle (a 64-byte line is a multiple of 32 bytes). Counting, for an array of 1024 f32 that starts at
a given number of bytes after a line boundary:

<!-- worked: cpu.alignment -->
Loads of consecutive vectors from an array of 1024 f32 (4096 bytes) that starts `offset` bytes after a 64-byte line boundary. A load is split when its bytes lie in two lines.

| offset in bytes | 32-byte loads split | 64-byte loads split |
|---|---:|---:|
| 0 | 0 of 128 | 0 of 64 |
| 4 | 64 of 128 | 64 of 64 |
| 16 | 64 of 128 | 64 of 64 |
| 32 | 0 of 128 | 64 of 64 |
| 60 | 64 of 128 | 64 of 64 |
<!-- /worked -->

An array that starts on a line boundary, or any multiple of 32 for the 32-byte loads, never straddles; one that
starts at any other multiple of 4 straddles on every other 32-byte load and on every 64-byte load. This is why
allocators that serve numerical code return aligned blocks, and why a program that loads vectors from the middle
of an array (the tail of a row, a slice that starts at a column) pays this. The cost on a real core is small and
varies by chip, which is why the count of straddling loads is what the book gives.

### The order of the additions changes

The lane structure changes the *sum*, not only the speed. A scalar loop adds the products in the order $0, 1, 2,
\dots$, one running total; an 8-lane loop adds eight separate chains and combines them. Volume I, Chapter 15
showed that floating-point addition is not associative, and Chapter 16 that a reduction regrouped into a tree gives
a different, usually slightly better, answer. Here it is concretely, on 20 products with the lane structure
above (no fused multiply-add, to keep the arithmetic to one rounding per operation):

<!-- worked: cpu.hsum_trace -->
Generator: the book's SplitMix64, seed 0x123899b4aebbbb06 (the seed of the name `cpu.hsum_trace`). n = 20. Draw order: the 20 values of x, each `2 * uniform() - 1` rounded to f32, then the 20 values of y the same way. Every product and every sum is rounded to f32 (no fused multiply-add). Lane j of the 8 accumulators adds the products j, j + 8 (elements 0..15 only). The horizontal sum is s[j] = lane j + lane j + 4, then (s[0] + s[1]) + (s[2] + s[3]); the last four products are then added to it one at a time. The last row is the exact (unrounded) products of the stored f32 inputs summed in exact arithmetic, then shown to 9 digits.

| step | value |
|---|---:|
| lane accumulators 0 | 0.230359614 |
| lane accumulators 1 | 0.229517639 |
| lane accumulators 2 | -0.151933283 |
| lane accumulators 3 | 0.201494679 |
| lane accumulators 4 | -0.235686466 |
| lane accumulators 5 | 0.385827333 |
| lane accumulators 6 | -0.0897886157 |
| lane accumulators 7 | -0.471316755 |
| lanes j and j + 4: s[0] | -0.0053268522 |
| lanes j and j + 4: s[1] | 0.615345001 |
| lanes j and j + 4: s[2] | -0.241721898 |
| lanes j and j + 4: s[3] | -0.269822061 |
| s[0] + s[1] | 0.610018134 |
| s[2] + s[3] | -0.511543989 |
| horizontal sum | 0.0984741449 |
| after the 4 tail elements (16 .. 19), one at a time | 0.641020656 |
| left to right, one accumulator | 0.641020656 |
| exact sum of the exact products of the stored x and y | 0.641020649 |
<!-- /worked -->

The vector sum and the one-accumulator sum happen to agree to all 9 printed digits on this small example (both
differ from the exact sum in the ninth digit), which is luck: eight chains of two or three products are too
short to lose much. On 4096 products the orders separate:

<!-- worked: cpu.simd_sum_order -->
Generator: the book's SplitMix64, seed 0x02074311ab966a02 (the seed of the name `cpu.simd_sum_order`). n = 4096. Draw order: the 4096 values of x, each `2 * uniform() - 1` rounded to f32, then the 4096 of y. The products are rounded to f32 and then added in f32, one addition at a time. With w lanes, lane j adds the products j, j + w, j + 2w, ... in order, and the lanes are then combined by the horizontal sum (while more than 4 lanes remain, lane j + h is added to lane j for h = half the lanes; then (a0 + a1) + (a2 + a3)); w = 1 is the plain left-to-right sum. The reference is the exact sum, in exact arithmetic, of the exact (unrounded) products of the stored f32 inputs: 7.77823008. (The exact sum of the f32-rounded products is 7.77822936; the two differ by the rounding of the products, which no order of additions can remove.) The error column is |f32 sum - reference| / |reference|.

| lanes w | f32 sum | relative error against the reference |
|---|---:|---:|
| 1 | 7.77820683 | 2.99e-06 |
| 4 | 7.77820969 | 2.62e-06 |
| 8 | 7.77821827 | 1.52e-06 |
| 16 | 7.77821922 | 1.40e-06 |
| 32 | 7.77823257 | 3.21e-07 |

5 distinct results from 5 orders.
<!-- /worked -->

Five lane counts give five different results. All are within the error bound of Chapter 15, and the sums
with more lanes are closer to the reference in this run, because each chain accumulates fewer terms and a short
sum rounds less (Chapter 16's pairwise argument again), though that is a property of the run and not a guarantee.
What is certain is that a vectorized kernel and a scalar kernel are different computations. Two consequences
follow. A compiler that is asked to keep IEEE semantics will not vectorize a floating-point reduction by itself,
because it would change the answer; a programmer who writes the lanes explicitly has decided that the change is
acceptable. And a test that compares a vectorized kernel with its scalar reference has to use a tolerance
(Volume I, Chapter 15), unless the kernel is built to preserve the order. A way to do that is to vectorize
across independent outputs instead of across the reduction: eight different output elements in the eight lanes,
each lane summing its own products in the same order as the scalar code. That is bit-identical to the scalar
loop, which the CPU backend uses where it needs that guarantee, as the last section explains.

## Many cores: data-parallel loops

A chip has several cores, and the cores are the third lever, after the data layout and the instructions. To use
them, a loop must be split so that the pieces can run at once without disturbing each other. A **thread** is a sequence of instructions that the operating system schedules onto a core; several threads run
at once on several cores. **Thread parallelism** is using several threads, each on its own core with its
own registers and first caches, to do the pieces of one computation at the same time. The cores share the
last-level cache and main memory and have no other common state unless a program makes one.

The structure that fits is a **data-parallel loop**: a loop in which no iteration reads anything another
iteration writes, so the iterations can run in any order, or at the same time, with the same result. Volume I's
pseudocode already has the statement for it, `parallel for i in 0 .. n`, whose contract is stated in
`docs/book/pseudocode.md`: no iteration reads what another writes, and an algorithm whose iterations must
combine results uses an explicit reduction, never a shared variable. That contract is what lets a runtime hand
the iterations to threads in any way it likes.

### A sum across workers

A parallel sum is the first case where the contract bites. A single shared total is not allowed, since every
iteration both reads and writes it. The correct structure gives each worker its own partial sum and combines
them after:

```text
function parallel_sum(x: array<f32>[n], workers: int) -> f32:
    require n >= 1 and workers >= 1
    partial = zeros(f32, [workers])             # one slot per worker, written only by that worker
    parallel for t in 0 .. workers:
        lo = (t * n) // workers
        hi = ((t + 1) * n) // workers
        s = 0.0
        for i in lo .. hi:
            s = s + x[i]
        partial[t] = s
    total = partial[0]
    for t in 1 .. workers:                      # a fixed order, on one thread
        total = total + partial[t]
    return total
```

It has two properties to take from it. First, each worker owns a contiguous block of the input, so its reads
are consecutive lines: the data-parallel loop keeps the spatial locality of the serial one. Second, the result
depends on `workers`: a sum over 4 blocks and a sum over 8 are different groupings, exactly as lane counts were.
A program whose answer must not change with the number of cores has to fix the grouping independently of it, for
example by cutting the input into blocks of a fixed length and summing those in a fixed order. brain does this
for the gradient norm, as the last section describes.

### Load balance

Splitting into equal blocks assumes equal iterations. When the iterations differ in cost, a worker with the
heavy ones is the last to finish and the others wait. Take 1000 iterations, with iteration $i$ costing
$i + 1$ units of work (a loop over a triangle, like the rows of a causal attention matrix), among 8 workers.
The best any split could do is to give each worker the total divided by 8; the cost of a split is its
**makespan**, the load of the most loaded worker, and its efficiency is the best possible divided by the makespan:

<!-- worked: cpu.parallel_schedules -->
1000 iterations, iteration i costs i + 1 units of work (total 500,500), 8 workers, so a perfectly balanced split gives each worker 62,562.5. The makespan is the largest load; efficiency = total work / (workers * makespan). Dynamic scheduling hands out the chunks in order of index, each to the worker with the smallest load so far.

| split | largest load | smallest load | efficiency |
|---|---:|---:|---:|
| one block of 125 per worker | 117,250 | 7,875 | 0.534 |
| round robin, one iteration at a time | 63,000 | 62,125 | 0.993 |
| chunks of 1, next chunk to the least loaded worker | 63,000 | 62,125 | 0.993 |
| chunks of 16, next chunk to the least loaded worker | 68,672 | 56,504 | 0.911 |
| chunks of 125, next chunk to the least loaded worker | 117,250 | 7,875 | 0.534 |
<!-- /worked -->

One equal block per worker leaves the first worker with 7,875 units and the last with 117,250, an efficiency of about 0.53:
almost half the machine idle. Dealing the iterations round robin, or giving out chunks of one iteration to
whichever worker is free, is almost perfect, and chunks of 16 are in between; chunks as big as a whole block
are no better than blocks. The choice of a chunk size is a trade: small chunks balance better but there are more
of them, and each one costs the runtime something to hand out and the worker something to start (a few
microseconds on typical hardware, enough to be more than a small iteration's whole work).

The last point is what the **grain** of a parallel loop is about: the smallest piece of work worth giving a
worker. brain's CPU backend fixes one for its generated kernels: it cuts a loop of $N$ invocations (one invocation is one run of the loop body for one index) into
about 8 chunks per thread, but never into chunks of fewer than 16,384 invocations, and a loop that fits in one
chunk runs on the calling thread with no tasks at all.

<!-- worked: cpu.dispatch_grain -->
The splitting rule: a loop of N invocations on T threads is cut into chunks of max(ceil(N / 8T), 16384) invocations (about 8 chunks per thread, but never fewer than 16,384 per chunk); one chunk or fewer means no tasks at all. An invocation is one run of the loop body for one index. The tasks column is ceil(N / chunk size) when that is more than one, and 0 otherwise.

| threads T | invocations N | chunk size | tasks |
|---|---:|---:|---:|
| 8 | 2,048 | 16,384 | 0 (runs on the caller) |
| 8 | 100,000 | 16,384 | 7 |
| 8 | 1,048,576 | 16,384 | 64 |
| 8 | 42,000,000 | 656,250 | 64 |
| 144 | 2,048 | 16,384 | 0 (runs on the caller) |
| 144 | 100,000 | 16,384 | 7 |
| 144 | 1,048,576 | 16,384 | 64 |
| 144 | 42,000,000 | 36,459 | 1,152 |
<!-- /worked -->

A small kernel therefore costs nothing extra on a wide machine, and a large one is cut into enough chunks to
balance. The row with 144 threads and 42 million invocations shows the target of 8 per thread, 1,152 tasks; every
smaller row is held at the floor. The story in the last section is what the floor is for.

### False sharing

Splitting a loop into parts that write only their own results sounds safe, and for correctness it is. For cost it
is not, because the unit of ownership in a cache is the line and not the number. A core that wants to write a line
must hold it exclusively: any other core's copy is invalidated, and the next time that core uses the line it
must fetch it again from wherever the writer has it. If two threads write two different numbers that happen
to sit in one line, the line moves between the cores on every alternation, though the program never shares a
single variable. This is **false sharing**.

The model: four workers each add 1 to a counter of their own 8 bytes, 1000 times; the counters sit next to each
other in one line, or each in a line of its own. A line is written only by the core that owns it, and a write by
another core takes it over at the price of one transfer. The schedule is the interleaving of the writes: a
`burst` of writes by worker 0, then by worker 1, and so on.

<!-- worked: cpu.false_sharing -->
4 workers each add 1 to their own 8-byte counter 1,000 times. A line may be written only by the core that owns it; a write by another core takes the line away (one invalidation, one transfer) and the line then belongs to the writer. Counter t sits at byte 8t (adjacent) or 64t (padded). The schedule is `burst` writes by worker 0, then `burst` by worker 1, and so on round after round: burst 1 is the worst case, burst 1000 is no overlap at all. Line transfers:

| layout | lines holding counters | burst 1 | burst 8 | burst 100 | burst 1000 |
|---|---:|---:|---:|---:|---:|
| adjacent 8-byte counters | 1 | 3,999 | 499 | 39 | 3 |
| one counter per line | 4 | 0 | 0 | 0 | 0 |

With each worker keeping its count in a register and writing it once at the end, there are at most 3 transfers whatever the schedule.
<!-- /worked -->

With counters in separate lines there is no transfer whatever the interleaving. With four counters in one line,
the count of transfers is the number of times the next writer differs from the last: 3,999 of 4,000 writes
when the workers alternate on every write, and as few as 3 when each worker's 1,000 writes come in one run. The
real interleaving is not under the program's control, and the model's worst case is a good guide to what the
structure can cost. The remedy has two forms: pad each worker's slot to a full line, or, better, keep the running
value in a register and write it once at the end (a value that is written once is written once). The `partial`
array of the sum above is exactly this hazard: eight 4-byte partial sums are half a line. A loop that writes
its own slots must separate them by a line.

The cost of the transfers is time, and the time depends on the chip; the count is what the book gives.

### Where the parallelism stops

Splitting a loop among workers speeds up the part that is split and nothing else; the serial part of a program
and the cost of starting and joining the workers set a ceiling. This is Amdahl's argument, and Chapter 11 of
this volume, on measuring performance, derives it and shows how to find the serial part of a real program with
a profile. The practical form is a question: what fraction of the work runs inside a parallel loop, and what
does a worker cost to start? A sampled profile of a program of this kind can show most of its CPU samples in the worker pool spinning while the
average core use stays low; that pattern means the work is sitting in a serial step and not in the loops that are
parallel, because idle workers spin while one thread works.

## Counting the work

The three levers of this chapter, layout, vectors and cores, help different limits, and to tell which limit a
computation is at, one counts what it does and what it moves. For an operation, count the arithmetic operations it needs (**FLOPs**: one multiplication or one addition is
one, a fused multiply-add is two) and the bytes it must at least move into or out of the core's caches (every
input once, every output once). The ratio, FLOPs per byte, says how much arithmetic each byte supports.

<!-- worked: cpu.work_counts -->
n = 1024, 4-byte numbers. FLOPs count each multiplication and each addition (a fused multiply-add is two). Bytes are the least that must cross the boundary of the core's cache: every input read once, every output written once.

| computation | FLOPs | bytes at least | FLOPs per byte |
|---|---:|---:|---:|
| dot product of two vectors of n | 2,048 | 8,192 | 0.250 |
| elementwise sum c = a + b, vectors of n | 1,024 | 12,288 | 0.083 |
| matrix-vector product, [n, n] by [n] | 2,097,152 | 4,202,496 | 0.499 |
| matrix product, [n, n] by [n, n] | 2,147,483,648 | 12,582,912 | 170.667 |
<!-- /worked -->

The dot product does a quarter of a FLOP per byte, and the elementwise sum a twelfth, because every byte it moves
supports at most one operation. The matrix-vector product is about a half. The matrix product does 2n^3
FLOPs on 3n^2 numbers: the ratio grows with $n$, here to about 171. A computation whose ratio is low has little
arithmetic to do per byte, and on a core that can do many operations in the time of one fetch from main memory it
waits for data whatever vector width it has; a computation with a high ratio has work to hide the fetches behind.
The next chapter turns this ratio into a criterion that says, for a given machine, which side of the line a
computation is on, and into a bound on how fast it can run. Until then it is a count, like the lines of this
chapter, and it is the reason that the layout and tiling work above is the first thing to do for a loop with a
low ratio, and the vectors and cores the first thing for one with a high ratio.

## The CPU backend in brain's crates

`crates/backend-cpu` is brain's CPU backend, and everything in this chapter has a counterpart in it.

**The kernels are generated, then run across cores.** brain's compute kernels are written once, in a shading
language, and the CPU backend does not interpret them: `crates/wgsl-cpu/src/lib.rs` translates each kernel to
native code through a code generator (Cranelift) when it is first needed, producing a function that loops over a
contiguous range of invocation numbers. The backend (`crates/backend-cpu/src/lib.rs`) runs the recorded steps
one after the other and splits the invocations of each step among a pool of threads; each invocation writes its
own output element, so the workers never write the same number, which is the data-parallel loop contract. The
splitting rule of the grain table above is in that file: about 8 chunks per thread (`threads * 8`) with a floor of
`MIN_CHUNK_INVOCATIONS`, 16K invocations, and a dispatch that fits in one chunk runs on the calling thread.

**One pool, one policy.** `crates/backend-cpu/src/par.rs` is the CPU-parallel primitives of the whole repository.
`AGENTS.md` states the rule: CPU parallelism is `backend_cpu::par`, and no other crate depends on the thread-pool
library. Commit 564420b85 (July 2026) moved six crates off direct uses, because each ad-hoc parallel loop had
run in a pool no one owned, so a policy such as `--device cpu0-7` (pin the process to eight cores and size the pool to
them) held only for the loops that went through the scheduler that honoured it. The functions of `par`
are named for the shape of the loop and not for the library: `rows_mut` and `chunks_mut` give each
worker a disjoint block of an output buffer, `map` and `map_f32` an index-ordered result per iteration, and
`zip_each` an elementwise step over two slices, so a caller states which part is parallel and the scheduler
decides how. The doc of `chunks_mut` states the contract of this chapter in other words: chunk $i$ covers the same
range whatever the number of threads, so a body that reads only its chunk and shared immutable data is
bit-identical to the serial loop.

**Vector paths, chosen once.** `crates/backend-cpu/src/fast_ops.rs` and `crates/backend-cpu/src/fast_conv.rs`
hold hand-written vector versions of the hot loops. `fast_conv::isa_tier` detects, once per process, whether the
processor offers AVX-512, AVX2 with fused multiply-add, or neither, and each kernel branches on the answer to
the widest version it can run, with a plain scalar loop as the fallback. The horizontal sum of this chapter is
`hsum256` in `fast_ops.rs`: the high 128 bits added to the low, then two pairwise additions, and `dot_avx2`
is the three-part structure above (a loop of 8 lanes, the horizontal sum, a scalar tail). Its tests compare the
vector paths against scalar references with a tolerance.

### The transpose that touched 64 lines at a time

Commit 47439c20f (August 2026) is the cache story of this chapter in brain's own code. A profile of an image
encoder on the CPU backend found one attention step taking most of the whole tower's time, against another step
of identical FLOP count that took a small fraction. The only structural difference was a loop that
transposed a matrix of values, $4096 \times 64$ numbers, into $64 \times 4096$ before a matrix multiplication,
writing `vt[d * tk + j]` with $d$ innermost: each write went to a different row of the destination, 16 KiB
apart. The commit tiled it: read 16 source rows, then write the destination row by row, 16
consecutive numbers (one line) at a time, with a 4 KiB staging buffer. The function is
`transpose_rows_tiled` in `crates/backend-cpu/src/fast_ops.rs`, and its test checks it against the plain
definition.

The commit says the destination lines were "evicted from L1" between revisits. Run through the model of this
chapter, that is true for one reason and not the other:

<!-- worked: cpu.commit_transpose -->
VT[d][j] = V[j][d] with j in 0 .. 4096, d in 0 .. 64: V is 4096 by 64 (row of 64 elements, 256 B), VT is 64 by 4096 (row of 4096 elements, 16 KiB), VT starts right after V. "naive" is j outer, d inner, one element at a time. "tiled" reads 16 rows of V, then writes VT d by d, 16 consecutive elements (one whole line) each; the 4 KiB staging buffer is not counted. Cache: 32 KiB, 64-byte lines, least recently used. Every line of V and VT once is 2 * 4096 * 64 / 16 = 32,768 fetches.

| cache | loop | lines fetched |
|---|---|---:|
| fully associative | naive | 32,768 |
| fully associative | tiled | 32,768 |
| 8-way | naive | 278,528 |
| 8-way | tiled | 32,768 |
<!-- /worked -->

With a fully associative 32 KiB cache the naive loop is already optimal: its 64 destination lines are revisited
and fit, so capacity is not what is wrong. With an 8-way cache the 16 KiB row stride puts all 64 destination lines
of one step into the same set (16 KiB is 256 lines, and $256 \bmod 64 = 0$), eight of them fit, and every write
misses: 278,528 fetches against 32,768, about 8.5 times as many. The tiling cures the conflict by making the 16
writes to a line consecutive. That is the model's reading, and it rests on an assumption the commit does
not state, an 8-way first-level cache; this chapter did not measure it. The commit itself reports that the
measured gain was not pinned down on the heavily loaded machine used, and says so. Line fetches are not time:
a conflict in a first-level cache is usually served by the second level, so how much of the fetch count shows up as
time is a measurement, and the model cannot give it.

The story has a second half that is the lesson of the size rule. Commit 7e28c7aff (August 2026) records
that the first version staged the tile in a fixed 64-wide buffer and policed the width with a debug assertion,
which release builds remove; a model with a head width of 128 then indexed out of bounds and a parity test failed
on the CPU backend. The fix, visible in the function today, derives the tile from a byte budget
(`TILE = 16 * 64` numbers, 4 KiB) and blocks both axes, so the staged tile stays small whatever the row width. A
fixed row count is wrong for somebody; a byte budget is the right unit, which is what the tiling table says.

### One row on one core, and the grain

Commit 585de524a (September 2026) is a load-balance story. The steady state of a language model's decoder
is one new token against weights that are read once: a matrix-vector product with a single row. The backend's two
matrix-product entry points split their work over output rows, so with one row there was exactly one non-empty
chunk, and the whole product ran on one core of the pool whatever its size. The commit added `gemv_cols` in
`crates/backend-cpu/src/fast_ops.rs`, which splits a one-row product over output columns, each task getting
a contiguous slab of the weights, and sizes its tasks by bytes streamed (`MIN_TASK_BYTES`, 256 KiB in the code
today) instead of by rows. Each task runs the same microkernel over its own columns, so the split changes which
core computes a column and never the order of the sum along $k$; the commit gates that with exact equality tests
(`gemv_column_split_is_bit_identical`). That is the data-parallel contract again: the iteration space is
the output columns, the iterations are independent, and a block of consecutive columns keeps the weights
contiguous.

Commit 50f04e6b7 (October 2026) is the grain story. The splitting rule then made a chunk per $8T$ pieces without
a floor, so on a 144-core node a kernel of a few thousand invocations became one-invocation tasks. The commit
reports, for one gradient check of a tiny model:

<!-- perf-number: historical measurement, commit 50f04e6b7 -->
In October 2026, on one node of 144 cores, the finite-difference gradient check of a tiny adapter model took 1858 s with the default pool and 32 s on one thread; with the floor of 16K invocations a chunk it took 26 s.

That is one setup on one day, and the commit's own explanation is the mechanism rather than the number: the cost
of waking and stealing across a wide pool exceeds the work in a task a few thousand invocations big. The floor
makes small work run where it is, and large work still fans out, which is the table above.

### One implementation of the arithmetic, and the order that must not change

Two more pieces of `backend-cpu` show the chapter's last themes. `crates/backend-cpu/src/host_gemm.rs`, from commit
1ee64780e (August 2026), is a blocked matrix-multiply loop whose documentation names both of the first
section's ideas. Its naive form reads the whole weight matrix once per output row. The new form computes
`MR = 8` output rows against one weight row, so that row is read once per eight outputs, and it gives the core
eight independent accumulator chains: the module doc says a single accumulator is a loop-carried
dependency on a latency of about four cycles, so a core retires roughly one multiply-add per four cycles
whatever the bandwidth. It also states why it does not simply use the vector kernel in `fast_ops.rs`, which is faster
still: that kernel splits $k$ across eight lanes and uses fused multiply-add, so it is not bit-identical to the
loop the surrounding code is gated on. The test `blocked_linear_is_bit_identical_to_the_naive_loop_it_replaces`
in `crates/backend-cpu/tests/it/host_gemm.rs` compares bit patterns across tile sizes that divide the rows and ones
that do not. The register block changes which element is computed when and never the order along $k$, which is
the same observation as the vectorize-across-outputs remark above.

The reduction order is the second. `par::sum_sq_f64` is the global gradient-norm reduction. Commit 4b56d9428
(October 2026) changed it because it had been adding per-tensor partial sums with the thread library's
adaptive sum, whose last bit depended on how many threads were free; now every tensor is cut into chunks of a
fixed size, each chunk summed in sequence, and the partials added in order. A test in `par.rs`,
`sum_sq_is_bitwise_the_same_on_any_number_of_threads`, compares the bit patterns on different numbers of threads. The sum
accumulates in 64-bit floating point on purpose (the doc says an f32 running total loses the low bits the clipping
threshold needs). Every worker count gives the same bits, which is the "fix the grouping independently of the
workers" rule of the parallel-sum section, in production.

## Check your understanding

1. A student says: "Adding a matrix by columns does 16 times as many additions as adding it by rows, so
   the column order is 16 times slower." Find the error, and say what is 16 times as large.
2. Using the table of lines fetched by matrix size, predict the lines fetched when a $192 \times 192$ matrix is read by columns, in the fully
   associative and the 8-way model. Why do they differ although the cache has the same capacity?
3. A sum is vectorized with 8 lanes and, in addition, a single vector accumulator is used. The instruction count
   fell by a factor of about 8, yet the core still runs the loop far below its issue rate. Using the dependency
   chain of the first section, explain why, and say what to change.
4. Two programmers each write a parallel sum: one gives each of 8 threads a slot `partial[t]` of 4 bytes in one
   array, the other gives each thread a local variable and writes `partial[t]` once at the end. They agree on the
   result. Which has the false-sharing hazard and what does the count of line transfers look like?
5. A loop of 3000 invocations is dispatched on a 64-thread pool by the rule of the grain table. How many tasks run and why
   is that the right answer for work this small?
6. In the transpose of the tiling table with a fully associative cache of 128 lines, the count is at its minimum for $T =
   64$ and not for $T = 128$. Derive the working set of a tile in lines and use it to explain the jump.
7. Find the error: "A vectorized sum and a scalar sum of the same f32 numbers give the same answer, because
   addition is commutative and the vector sum only reorders the additions."

## Summary

A core executes instructions that name registers; an instruction's result is ready a latency after it starts, so
a loop whose every step depends on the last, like a sum with one running total, is limited by the length of
its dependency chain and not by the core's issue rate, and independent accumulators shorten the chain. Memory is
a hierarchy of levels, each larger and slower than the one above, and data moves between the levels in cache lines
of 64 bytes, so a program is cheap when it uses all of a line before the line leaves the cache (spatial locality)
and returns to a line while it is still there (temporal locality). With a cache model that can be run, the same
256 additions touch 16 lines or 256 depending on loop order; the penalty is a cliff set by the
comparison of a working set with the cache, with extra conflict cliffs when strides are powers of two, and a
tiling moves the cliff by shrinking the working set. A vector instruction does the work of several scalar ones
in one issue, cutting the instruction count by the lane count, and changes the grouping of a floating-point sum,
so vectorized and scalar results differ in the last digits. Several cores run a data-parallel loop, one whose
iterations are independent, with each worker writing only its own results, balanced by splitting into chunks that
are neither one equal block nor so small that starting them dominates, and without two workers writing one cache
line. Counting FLOPs against bytes tells which of these levers will help.

What this chapter could not do is say, for a given machine, whether a computation is limited by its arithmetic or by
the bytes it moves, and how to compare it with the best the machine could do. The ratio of FLOPs to bytes counted at
the end is half of that answer; the other half is the machine's own rates of arithmetic and of memory traffic.
Chapter 3 defines them and combines them into the roofline.

## Exercises

1. **[Calculate]** A row-major $[100, 160]$ matrix of 4-byte numbers starts at a line boundary. How many lines does
   one row occupy, and how many distinct lines does one column touch? How many of the 64 sets of the 8-way
   cache do those column lines fall into, and does reading the matrix by columns then exceed the cache's capacity in
   either associativity?
2. **[Calculate]** Using the model of the dependency section (latency 4, 2 issues a cycle), find by hand the
   modelled cycles for summing 16 numbers with $L = 2$ totals combined at the end: first the chain length of each
   total, then the combination. Compare with a single total.
3. **[Calculate]** An array of 256 f32 starts 12 bytes after a line boundary. Count the 32-byte vector loads over
   the array that straddle two lines, and the 64-byte ones.
4. **[Calculate]** With 3 workers and a burst of 2, write out the first 12 writes of the false-sharing schedule of
   one-line counters and count the transfers they cause.
5. **[Calculate]** A dispatch of 500,000 invocations on 16 threads: by the grain rule, what is the chunk size and how
   many tasks are there? Repeat for 20 threads.
6. **[Implement]** Implement the cache of this chapter (the function `access`) in your own language and reproduce the
   16 by 16 tables of `cpu.lru_trace` and `cpu.lru_cliff`, then the first five rows of `cpu.traversal_sizes` (the
   fully associative and the 8-way columns).
7. **[Implement]** Reproduce `cpu.tiled_transpose`: the tile-by-tile visit order, the two matrices at the stated
   addresses, the 8 KiB cache, and the three columns (fully associative, 8-way, 8-way with the row pitch 272).
8. **[Implement]** Reproduce `cpu.dependency_chain`: build the dependency graph of the $L$-total sum with a final
   pairwise tree, schedule it with latency 4 and 2 issues per cycle by the scheduling rule printed with
   `cpu.dependency_small` (which also lists the additions of the small cases to check your program against), and print
   the longest chain and the cycle at which the last result is ready.
9. **[Implement]** Reproduce `cpu.simd_sum_order` and `cpu.hsum_trace` from the generator seeds printed beside them,
   rounding every product and every sum to f32. Make the horizontal sum a function of the lane count.
10. **[Investigate]** In the model of the tiling table, find the smallest row padding (in elements) for which the 8-way
    transposition with $T = 16$ reaches the least possible count, by trying 0, 4, 8, 12, 16 and 20 elements. Explain
    the result in terms of the sets a tile's lines fall into.
11. **[Investigate]** Time on your own machine a sum of the entries of a $4096 \times 4096$ f32 matrix by rows and by
    columns, and a $1024 \times 1024$ one. Compare the ratio of the times with the ratio of line fetches the model
    predicts for each size, and say which differences a prefetcher or overlapping fetches could explain. State your
    machine, compiler and the number of repetitions.
12. **[Investigate]** Add a one-megabyte second level (16,384 lines, 8-way) behind the first-level cache in your
    simulator, so that a miss in the first level is looked up in the second, and recount `cpu.traversal_sizes` for
    $n$ up to 1024. For which sizes does the second level absorb the column order's misses, and what is the
    matrix size at which it stops doing so?

## Key terms

- **CPU core**: one unit of a processor that fetches, decodes and executes instructions in a pipeline, with its own registers and first-level caches, sharing the rest of the memory system with the other cores of the chip. <!-- concept: cpu-core -->
- **cache hierarchy**: the stack of memories between a core's registers and main memory, each level larger and slower than the one above it and holding copies of the lines most recently used from the level below. <!-- concept: cache-hierarchy -->
- **cache line**: the aligned block of memory, 64 bytes on most chips, that is fetched and kept in a cache as a unit when any of its bytes is needed. <!-- concept: cache-line -->
- **SIMD**: single instruction, multiple data: an instruction that applies the same operation to all the lanes of a wide register at once, for example eight f32 additions in one 256-bit instruction. <!-- concept: simd -->
- **thread parallelism**: using several threads, each running on its own core with its own registers and first caches and sharing the later caches and memory, to carry out the pieces of one computation at the same time. <!-- concept: thread-parallelism -->
- **data-parallel loop**: a loop in which no iteration reads anything another iteration writes, so its iterations can be run in any order or at the same time with the same result; the `parallel for` of the book's pseudocode. <!-- concept: data-parallel-loop -->

## Answers

**Recall questions.**

1. The address is $4(i n + j)$ bytes; the stride of the first axis is $n$ elements. (Volume I, Chapter 5.)
2. The strides are $(1, 4)$. Reading the transpose row by row moves the storage position by 4 elements at each step: it reads
   the original by columns, so it touches a new line on nearly every read of a long column.
3. $\lceil \log_2 1024 \rceil = 10$ additions in the longest chain, against 1023 for the left-to-right sum.
4. No. $(0.1 + 0.2) + 0.3 \neq 0.1 + (0.2 + 0.3)$ in f32. A different order of additions gives a slightly different
   result, within the error bound of the format.
5. 4 bytes; 16 of them fit in a 64-byte line.

**Check your understanding**

1. The number of additions is the same, 256 for 16 by 16, in both orders. What is 16 times as large is the number
   of cache lines fetched when the cache cannot hold a column (256 against 16 in the first table). Whether that
   makes the program 16 times slower depends on the machine; the model of the hierarchy-cost table gives about seven,
   and a real core can be better or worse.
2. For $n = 192$ in that table: 2,304 lines in the fully associative model (the same as the row order, since a column of 192
   lines fits in 512) and 36,864 in the 8-way model (every read misses). A row is 12 lines, so the lines of a column
   fall into $64 / \gcd(12, 64) = 16$ sets, which hold $16 \times 8 = 128$ lines, fewer than the 192 of a column.
3. With one vector accumulator each step still adds into the same register, so the 128 steps for 1024 numbers
   form a chain of 128 dependent multiply-adds, each waiting out the latency. The chain per number has been divided
   by 8, but the chain is still one per register. Use several vector accumulators (4 or 8 registers) over interleaved
   blocks and add them at the end: the same remedy as the independent totals, now with vectors. This changes
   the grouping of the sum.
4. The first. Eight 4-byte slots are 32 bytes, half a line, so all eight workers write the same line and every write by
   a different worker takes the line over: with an alternating schedule almost every write is a transfer
   (about 8 times the number of rounds, minus one). The second writes the array once per thread, so there are at most
   $8 - 1$ transfers whatever the schedule.
5. None: $\max(\lceil 3000 / (8 \cdot 64) \rceil, 16384) = 16384 \ge 3000$, so the whole loop is one chunk and runs
   on the calling thread. Waking threads for 3000 invocations costs more than the invocations.
6. In the transposition, while one line of $B$'s tile is being filled, the loop touches the $T$ lines of the tile's
   rows of $B$ (one per tile row) and one line of $A$, so $T + 1$ lines. At $T = 64$ that is 65 of 128, at $T =
   128$ it is 129, one more than the cache, and least recently used replacement then evicts exactly the line needed
   next, the cliff of the first page.
7. Commutativity is not what is needed. The vector sum does not only reorder additions; it regroups them into
   lane chains, and addition of f32 numbers is not associative (Volume I, Chapter 15). The two sums are different
   computations whose results differ in the low digits (the 4096-product table has five different results from five groupings).
