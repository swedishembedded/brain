<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Security of Machine Learning Services

*What can a hostile request or a malicious model file do to a server, and how do we stop it?*

Forty-seven bytes are enough to ask a vision server for 51 gigabytes. The bytes are a PNG (Portable Network Graphics) image file: an 8-byte signature, a header chunk that says the picture is 65,535 pixels wide and 65,535 pixels high, and the first two bytes of a compressed pixel stream that never continues. Encoded in base64, the text form in which a chat request carries an image, they are 64 characters, less than one line of the JSON (JavaScript Object Notation) text that carries them. A server that believes the header and prepares the picture the way a vision model reads it, three 32-bit floats per pixel, sets out to allocate $65{,}535^2 \times 12 = 51{,}538{,}034{,}700$ bytes before it has read a single pixel. The request cost its sender nothing; the server either fails the allocation, which in some languages ends the process outright and every other conversation in it, or succeeds and has given one stranger a machine's worth of memory.

Nothing in that exchange is a bug in the usual sense. The decoder did what PNG decoders do, the model would have read what it was given, and the request was well formed. What went wrong is that a number chosen by someone else decided how much the server spent, and nobody compared it with anything first. A server for machine learning models is unusually exposed to this, for three reasons that the earlier chapters of this volume established. Its requests carry media and long texts whose decoded size is many times their encoded size. Its unit of work, a forward pass on a device, is so expensive that a few requests can occupy everything. And it reads files of tens of gigabytes whose layout is described by headers written by whoever made the file. Every one of those is a place where the size of the work is a claim made by the input.

Four ideas from earlier chapters carry the argument, and each is worth bringing back before reading on. What must a checkpoint format say about every tensor so that a reader that has never seen the file can find and decode its bytes, and how does a reader of a safetensors file locate tensor $x$ (Volume VI, Chapter 2, "Importing Pretrained Models and Proving Parity")? A safetensors header lists a tensor of type F32 with shape $[3, 5]$: how many bytes must the range of its data span? When a server cannot start a request now, what does admission control do with it, and why is a wait with a deadline better than a queue without one (Chapter 4, "Batching and Scheduling Requests")? And what does a request to an HTTP (Hypertext Transfer Protocol) inference API (application programming interface) carry, and where in it does the client's key travel (Chapter 6, "Capabilities, Providers and APIs")?

## What a request costs its sender and what it costs the server

Call **hostile input** any byte a server receives from a party whose aim may be to harm it: the body of a request, the headers around it, the timing of its packets, a file downloaded on a request's behalf, a file descriptor handed over by a local program. The definition is about the source, not the content. Most input from most sources is harmless; the point is that a server cannot tell which, so every quantity that such input declares (a width, a count, an offset, a length, a nesting depth) is a claim to be checked, never a fact to be acted on.

The most common harm such input does is not to read a secret or run a command but to make the server spend. **Resource exhaustion** is an attack, or an accident, in which an input makes a server consume a resource it shares among all its clients (memory, threads, open connections, slots on a device, disk) until the server can no longer serve anyone. Its measure is the **amplification**, what the input costs the server divided by what it cost the sender: an amplification of a few is a nuisance, one of millions is a weapon, because one sender with a modest connection can then outspend any machine.

The opening's file can be written out byte by byte, which makes the ratio exact.

<!-- worked: security.png_bomb -->
A PNG that declares w x h 8-bit RGB (red, green, blue) pixels and holds no pixel data is 47 bytes: the 8-byte signature, a 25-byte IHDR (image header) chunk (4 bytes of length, 4 of type, 13 of data: width and height as big-endian 32-bit integers, bit depth 8, colour type 2, compression, filter and interlace 0, and a 4-byte CRC-32, the chunk's cyclic redundancy check) and a 14-byte IDAT (image data) chunk whose data are the 2 bytes 0x78 0x01 of a zlib stream header. Base64 makes 4 characters of every 3 bytes. A decoded pixel is three 32-bit floats, 12 bytes. The ratio is the float bytes over the file bytes, rounded to the nearest integer. The header check refuses more than 36,000,000 pixels per image; exactly that many pass it.

The 65,535 x 65,535 file, in hexadecimal: `89 50 4e 47 0d 0a 1a 0a 00 00 00 0d 49 48 44 52 00 00 ff ff 00 00 ff ff 08 02 00 00 00 39 67 4e 07 00 00 00 02 49 44 41 54 78 01 ec 1a 7e d2`.

| declared size | file bytes | base64 characters | pixels | bytes as 8-bit RGB | bytes as f32 RGB | f32 bytes per file byte | header check |
|---|---:|---:|---:|---:|---:|---:|---|
| 6,000 x 6,000 | 47 | 64 | 36,000,000 | 108,000,000 | 432,000,000 | 9,191,489 | passes the header check |
| 6,001 x 6,000 | 47 | 64 | 36,006,000 | 108,018,000 | 432,072,000 | 9,193,021 | refused from the header |
| 65,535 x 65,535 | 47 | 64 | 4,294,836,225 | 12,884,508,675 | 51,538,034,700 | 1,096,553,930 | refused from the header |
<!-- /worked -->

The 47-byte file costs the same 47 bytes whatever size it declares, so the ratio is the declared size alone: 9,191,489 float bytes per file byte for a 6,000 x 6,000 picture, 1,096,553,930 for the 65,535 x 65,535 picture of the opening. Neither file holds a pixel. A decoder that reads the header, allocates, and only then discovers that the data stop has already done the damage.

A cautious reader might object that a real image must carry its pixels, and pixels are not free. They are nearly free when they are all the same. A PNG's pixel stream is compressed with deflate (the format of RFC 1951, an Internet Request for Comments), which encodes a run of repeated bytes as back-references: "copy 258 bytes from 1 byte back". The format lets an encoder choose a Huffman code (a code that gives frequent symbols short bit strings) in which the length code for 258 bytes has 1 bit, and RFC 1951 states that a block that uses only one distance code still spends 1 bit on it, so a run costs 2 bits per 258 bytes, a ratio of $258 \times 8 / 2 = 1{,}032$. No deflate stream of a flat picture can be shorter than its pixel stream divided by 1,032.

<!-- worked: security.deflate_floor -->
The pixel stream of an 8-bit RGB PNG with no interlacing is h x (1 + 3 w) bytes, one filter byte before every row. A deflate stream that encodes a run of equal bytes as matches of length 258 at distance 1 spends at least 2 bits per match under a dynamic Huffman code (a 1-bit length code and a 1-bit distance code), so no stream is shorter than the pixel stream divided by 1,032, rounded up; the few bytes of headers and the first literal are left out. The last column is the decoded f32 bytes over that least compressed size.

| image | pixels | pixel stream bytes | least deflate bytes | base64 characters | bytes as f32 RGB | f32 bytes per compressed byte |
|---|---:|---:|---:|---:|---:|---:|
| 1,000 x 1,000 | 1,000,000 | 3,001,000 | 2,908 | 3,880 | 12,000,000 | 4,126.5 |
| 6,000 x 6,000 | 36,000,000 | 108,006,000 | 104,657 | 139,544 | 432,000,000 | 4,127.8 |
| 8,000 x 4,500 | 36,000,000 | 108,004,500 | 104,656 | 139,544 | 432,000,000 | 4,127.8 |
<!-- /worked -->

A real 6,000 x 6,000 picture of a single colour therefore fits in about 104,657 bytes of deflate, 139,544 characters of base64, and decodes to 432,000,000 bytes of floats: 4,127.8 bytes of server memory for every byte the sender paid for. That number is the honest version of the opening's: it holds even against a decoder that refuses files whose data are missing. And the 8 MiB (mebibytes, $2^{20}$ bytes each) body that brain's chat routes accept holds 6,291,456 bytes of decoded base64, room for 60 such images by bytes alone; only a limit on their number or their pixels stops a request at fewer (the request's pixel budget, below, admits one).

Memory is the resource the opening counted, but it is not the only one. A thread that decodes an image for a second is a thread that serves nobody else for that second, and a server built on a small pool of asynchronous workers has few of them. A connection held open costs a file descriptor whether or not a request ever arrives on it. A request admitted to a device lane holds the lane for as long as its generation runs. A model fetched on a request's behalf fills the disk. Each of these is a shared budget, and each needs the same treatment.

## Refusing from the header

The treatment has one shape. An **input bound** is a ceiling on a quantity that hostile input controls and that sizes the server's work, checked before the work is done (before the allocation, the decode, the recursion or the wait), and refused with an error that names the bound. Three words in that sentence carry weight. *Before*: a check made after the decode is a report of the damage, not a defence. *Names*: a client that hits a bound should learn which one and by how much, because a legitimate client will hit it too, and silent truncation (decoding the first million pixels, keeping the first 128 tools) turns a refusal into a wrong answer. *Sizes the work*: the bound is on the quantity that decides the cost, which for an image is its declared pixel count and not the bytes of its file.

brain bounds images at two levels (`crates/capability/src/blob.rs`): `MAX_IMAGE_PIXELS`, 36,000,000 pixels for one image (a 6,000 x 6,000 photograph), and `MAX_REQUEST_PIXELS`, 64,000,000 pixels for all the images of one request, which may carry at most `MAX_IMAGES`, 8 of them. The first is checked against the header. `decode_bounded` in `crates/imaging/src/codec.rs` reads the dimensions of a PNG or JPEG (Joint Photographic Experts Group) file from its header without decoding the pixels, refuses a product over the limit, and only then decodes, with the decoder's own allocation limit set to 8 bytes per allowed pixel, enough for the widest source format it converts from. A PPM (portable pixmap) file, which stores its pixels raw, is refused when its data are shorter than its header claims, before anything is copied. The test `an_image_declaring_billions_of_pixels_is_a_400_not_an_allocation` in `crates/apiserve/tests/it/api.rs` sends the 65,535 x 65,535 header of the table above and expects a 400 (bad request), and `an_image_declaring_more_pixels_than_the_limit_is_refused_before_it_is_decoded` in `crates/apiserve/src/media.rs` builds the same 47-byte file.

The second level is subtler, because a budget shared by several items has to be checked in the right order. The obvious code decodes each image, adds its pixels to a running total and refuses the request once the total is over the budget. Follow it with a sender who chooses the sizes.

<!-- worked: security.pixel_budget -->
Per-image limit 36,000,000 pixels, request budget 64,000,000 pixels, at most 8 images, 12 bytes per decoded pixel. Images are taken in the order listed. 'Check after' decodes an image within the per-image limit, adds it and refuses the request when the total is over the budget; 'check before' reads each image's declared size from its header and refuses it, before decoding, when it is over the smaller of the per-image limit and the budget left. 'Decoded' counts the images whose pixels were decoded before the outcome. The count of images is checked first, before the size of the image, so image 9 is refused whatever it declares; under 'check after' an image over the per-image limit is also refused from its header without being decoded, so only the budget is checked late.

| images declared (pixels) | rule | decoded | pixels decoded | f32 bytes held | outcome |
|---|---|---:|---:|---:|---|
| 36,000,000 then 28,000,000 then 36,000,000 | check the total after each decode | 3 | 100,000,000 | 1,200,000,000 | refused after decoding image 3 |
| 36,000,000 then 28,000,000 then 36,000,000 | check each header against what is left | 2 | 64,000,000 | 768,000,000 | refused at the header of image 3 |
| 36,000,000 twice | check the total after each decode | 2 | 72,000,000 | 864,000,000 | refused after decoding image 2 |
| 36,000,000 twice | check each header against what is left | 1 | 36,000,000 | 432,000,000 | refused at the header of image 2 |
| 8 images of 8,000,000 | check the total after each decode | 8 | 64,000,000 | 768,000,000 | served |
| 8 images of 8,000,000 | check each header against what is left | 8 | 64,000,000 | 768,000,000 | served |
| 9 images of 1,000 | check the total after each decode | 8 | 8,000 | 96,000 | refused at image 9: more than 8 images |
| 9 images of 1,000 | check each header against what is left | 8 | 8,000 | 96,000 | refused at image 9: more than 8 images |
<!-- /worked -->

Checked after the decode, the first sequence decodes 100,000,000 pixels, 1,200,000,000 bytes of floats, before the total crosses the budget: a total exactly at the budget still passes, so the most that can be decoded is the budget plus a whole image of the per-image size. The refusal arrives, but the memory was spent. Checked against the header with what is left of the budget, the same sequence stops at the third header, having decoded 64,000,000 pixels, and no sequence of images can make the server hold more than the budget. The general rule is that a shared budget bounds the next item by what is left, $\min(\text{per item}, \text{budget} - \text{used})$, not by the per-item limit alone. Written as a routine:

```text
function admit_image(w: int, h: int, used: int, per_image: int, per_request: int) -> (bool, int):
    require w >= 0 and h >= 0 and 0 <= used and used <= per_request
    if w * h > min(per_image, per_request - used):
        return (false, used)              # refused from the header; nothing decoded
    return (true, used + w * h)           # decode it; the budget now counts it
```

`ExtractedMedia::push_image` in `crates/apiserve/src/media.rs` is exactly this: it passes `MAX_IMAGE_PIXELS.min(left)` to `decode_bounded` as the image's limit, and the test `the_images_of_one_request_share_a_pixel_budget_checked_before_decoding` gives a request 1,000 pixels of budget and a 100 x 100 header with no pixel data behind it, so the only way the test can pass is by refusing from the header. The history runs the other way. Commit `78f2d7908` (1 November 2026) introduced the per-image and per-request limits with the per-image check on the header and the request total checked after each image was decoded, and five days later commit `071961b05` records what that allowed: one 8 MiB request "of flat PNGs decoded about 100 Mpx (some 1.2 GB of f32) before it was refused" (Mpx being millions of pixels), the first row of the table, and moved the request check onto the header.

## A ceiling for every number that sizes work

Images are the vivid case. Every other quantity a client sends gets the same treatment, and the useful habit is to ask of each field of a request: if this number were a billion, what would the server do? For the size of the request itself, the answer is a body limit, checked before any handler buffers the body: brain's dialect routes (chat, embeddings, images) take at most `MAX_BODY_BYTES`, 8 MiB (`crates/apiserve/src/lib.rs`, a 413, "payload too large", past it), the generic `/v1/run` and `/v1/jobs` routes and the video routes take 64 MiB, because media travels inline as base64, and the route for typed decisions takes 2 MiB. For the counts inside a body, each has its own ceiling, chosen from the largest legitimate use: an embeddings request embeds at most 2,048 strings, each one a job, so that "a body of a million one-letter strings is not a million jobs" (the comment in `parse_embedding_request`, `crates/apiserve/src/openai.rs`); a request declares at most 128 tools of at most 256 KiB (kibibytes, $2^{10}$ bytes each) together, each named in at most 64 characters (`validate_tools`); `stop` lists hold at most 16 strings of 256 bytes, since every one is matched against every generated token (`MAX_STOPS` and `MAX_STOP_BYTES` in `crates/qwen3/src/chat.rs`). The JSON parser has its own bound on nesting, 128 levels by default in `serde_json`, the library the surface uses (version 1.0.151 in the lockfile), so a body of ten thousand opening brackets is a parse error and not a stack overflow.

Numbers that size work inside a model are the largest class, and the one most easily forgotten, because each model declares its own parameters. brain's answer is a rule enforced by a test rather than by review. Every parameter of every served action declares its type and, for a number, its range, and `ActionSpec::validate` in `crates/capability/src/lib.rs` refuses a value outside it by name before anything is submitted. Commit `c68efb167` (6 November 2026) records why the rule had to become a gate: most numeric parameters declared no maximum, so the generic routes accepted frames, widths, heights, steps and token budgets of any size, "one call to imageops gradient with a large width and height aborted the process on its allocation", and a negative value cast to an unsigned size became a huge one. The commit gave every number that sizes work a ceiling, shared across models where two parameters mean the same thing (`crates/capability/src/bounds.rs`: 262,144 tokens, 8,192 pixels on a side, 2,048 video frames, 1,000 denoising steps and so on), capped every string parameter at 16 MiB (`MAX_STR_PARAM_BYTES`), and added `crates/catalog/tests/it/served_bounds.rs`, which fails on any served integer or float without a maximum unless `scripts/gates/unbounded-params.allow` lists it. That list holds only knobs that size nothing (seeds, token ids, sampling and guidance scales, learning rates), and the test also fails on an entry that is no longer needed, so the list can only shrink.

A parameter can be dangerous without being large. Some actions need a path on the serving machine: where the weights are, where a dataset lives. If a client can set that parameter, it can point a model at any file the server can read. Commit `2139a4889` (2 October 2026) records that `/v1/capabilities` listed such parameters and `/v1/run` accepted them, "so any caller holding a key could point a model at a file of their choosing". The fix is a second rule of the same validator: a parameter the host resolves for itself is marked so (`ParamSpec::host_env` or `host_resolved`), a served manifest omits it, and `ActionSpec::validate_served` validates a remote call in two passes, first against the manifest with those parameters removed, so that naming one is an unknown parameter, then against the full manifest, where the host fills in its own answer. The D-Bus surface (the desktop message bus of Linux, through which local programs call brain) calls the same function in `served_invocation` (`crates/dbus/src/service.rs`), so neither transport lets a caller name a file.

The constants of one surface, gathered in one place, make the bounds composable: each one limits a cost, and the products and sums of them limit the worst case of the whole server.

<!-- worked: security.surface_ceilings -->
The ceilings are brain's constants (named in the text); the derived column is arithmetic on them. Base64 carries 3 bytes in 4 characters, so a body of b bytes holds at most floor(3 b / 4) bytes of media; a bit rate is the bytes times 8 divided by the seconds, rounded down.

| what is bounded | ceiling | what follows from it |
|---|---:|---|
| request body, chat, embeddings and image routes | 8,388,608 bytes | base64 media in it: at most 6,291,456 bytes |
| request body, `/v1/run`, `/v1/jobs`, `/videos` | 67,108,864 bytes | base64 media in it: at most 50,331,648 bytes |
| request body, `/v1/systemone` | 2,097,152 bytes |  |
| decoded image pixels, one request | 64,000,000 pixels | 768,000,000 bytes of f32 |
| media decodes at once, whole process | 4 | 3,072,000,000 bytes of f32 at most |
| request head after connecting or between requests | 10 seconds |  |
| whole request body | 300 seconds | 67,108,864 bytes in 300 seconds is 1,789,569 bits per second |
| connections held, one surface | 1,024 | to keep all filled without a key: 102.4 new connections per second |
| requests in flight at the edge | 256 | one more is answered 503 at once |
| one input fd over D-Bus | 268,435,456 bytes |  |
| input fds of one D-Bus call | 16 fds, 1,073,741,824 bytes together | 16 x 268,435,456 = 4,294,967,296, so the total binds first |
<!-- /worked -->

The fifth row is the one to read twice. A media decode may hold up to 768,000,000 bytes of floats, and at most 4 run at once across the whole process (`MAX_CONCURRENT_DECODES` in `crates/apiserve/src/preprocess.rs`), so decode memory has a worst case of 3,072,000,000 bytes, however many requests arrive. Without the concurrency bound each request is bounded and the sum is not. The D-Bus rows show the same arithmetic one level down: 16 file descriptors of up to 268,435,456 bytes each could carry 4,294,967,296 bytes, so the per-call total of 1,073,741,824 bytes is the bound that decides.

## Holding a socket without sending a byte

Memory is spent by what a request says. Connections are spent by what it does not say. A server reads a request in two parts, the head (the request line and the headers) and the body, and a server that waits patiently for each can be held by a peer that sends neither. Open a connection and send nothing: the server keeps a file descriptor and a little state for it. Send one header byte every few seconds: the server keeps waiting for the end of the head. Do this a thousand times and the process runs out of descriptors, at which point it cannot accept its legitimate clients, all without a key, because the key is in a header that never finishes. Attacks of this shape are known as slow-request attacks. The cost to the sender is a few bytes per connection per minute; the cost to the server is everything it reserves per connection.

The cure is a deadline on every wait and a cap on what one surface holds. brain's HTTP surfaces are served by their own accept loop (`crates/apiserve/src/listener.rs`), whose module comment states the three bounds and the reason for each. A connection must deliver its request head within `ConnectionLimits::header_read`, 10 seconds by default; the timer starts whenever the server waits for a head, so it also closes a kept-alive connection that sits idle between requests. A body must arrive whole within `ConnectionLimits::body`, 300 seconds, a deadline on the entire body rather than on the gap between pieces of it, since a gap deadline is reset by every byte and a body trickled one byte at a time never trips it. And a surface holds at most `max_connections`, 1,024, connections; one more is closed at accept, before a byte of it is read. The defaults are sized from legitimate use: 300 seconds for a 64 MiB body is 1,789,569 bits per second, the slowest link that can still upload the largest accepted request (the table above). Only HTTP/1.1 is served, because HTTP/2 has no header deadline, so a peer that answers the protocol's keep-alive pings could hold a connection without ever sending a request.

Commit `6c57f6408` (6 November 2026) is where these came from. Its message records that the framework's default serving function "sets no timer, so hyper's header deadline never ran, and it accepted without limit: any host that reached a surface could open sockets, send nothing or a header byte at a time, and hold them until the process ran out of descriptors, with no key at all". The tests in `crates/apiserve/tests/it/connections.rs` pin each bound over a real socket: `an_idle_connection_is_closed_at_the_header_deadline`, `a_connection_that_never_finishes_its_headers_is_closed`, `a_body_that_trickles_in_is_cut_at_the_body_deadline` (a 4,096-byte body sent one byte every 50 milliseconds, well inside any gap deadline, must be cut within 5 seconds when the body deadline is 0.5 seconds), and `connections_over_the_cap_are_refused_and_a_freed_slot_is_reused`.

The bounds do not make the surface impossible to fill. A sender who opens 1,024 connections and replaces each one as the 10-second deadline closes it keeps the surface full with 102.4 new connections per second. What changes is the price: holding the surface now costs a steady stream of connections, each a round trip the sender must make, instead of one burst followed by silence, and the process itself, its descriptors and its memory, is never at risk. A bound turns an attack that costs nothing into one that costs in proportion to how long it lasts, and that is the most a single server can do; stopping a sender who is willing to pay is the job of the network in front of it.

The same reasoning applies to the threads that do a request's work before the device does. Decoding a recording or a picture costs seconds of processor time. An asynchronous server multiplexes all its connections over a few worker threads, so a decode run on one of them stalls every connection that worker owns, keyed or not. `apiserve::preprocess::off_runtime` moves every request's media decode to a pool of blocking threads, at most four at once process-wide, and a request that cannot get a decode slot within `DECODE_QUEUE_DEADLINE`, 30 seconds, is answered 503 ("service unavailable") rather than queued without end (commit `071961b05`). A decoder that panics frees its slot and becomes a 500 with no detail (the test `a_decode_that_panics_is_an_error_and_frees_its_slot`).

And it applies to the device. The admission control of Chapter 4 already has the right shape: a request that cannot start now waits, but with a deadline, and the server sheds what it cannot start rather than letting a queue grow without bound. brain applies it at two points that are worth telling apart (`crates/residency/src/admission.rs`, shared by the HTTP and D-Bus surfaces so that both shed the same way). At the edge, at most `EDGE_CONCURRENCY`, 256, requests are in flight on the routes that do work, and one more is shed at once with 503: the server is saturated, retry elsewhere. Behind the edge, a request that has been let in waits for a device lane for at most the admission deadline, by default 15 minutes because an agent's turn legitimately runs that long, and is answered 429 ("too many requests") when it expires: accepted, but nothing could start it in time. The two codes tell a client two different things, and the deadline, set by the operator's `--admit-deadline-ms`, is what keeps a wedged lane from becoming an unbounded queue.

The order in which these checks run is itself a defence: each one is cheaper than the work it protects, and the cheapest run first, so that a hostile peer is turned away before it costs anything larger.

![The gates one HTTP request passes in brain, in order: accept and connection cap, TLS (Transport Layer Security) handshake and request head within 10 seconds, key check, edge permit, body limit and deadline, parse and bounds, media decode with a bounded pool, the dispatch gate for training and the caller's budget, admission to a lane, and the executor.](img/request-gates.svg)

The first row needs no key: a peer that cannot finish its head, or has no valid key, never causes the server to read a body. The key check wraps every route, including the fallback for paths that do not exist, so a peer without a key gets 401 ("unauthorized") for `/no/such/route` and learns nothing about which routes exist (the test `unauthenticated_unknown_path_is_401_not_404`). Only a caller with a key reaches the body, and only a body that parsed within its bounds reaches a decoder.

## A model file is a program for its reader

Requests are the obvious input. Model files are the overlooked one, because they feel like part of the server rather than something sent to it. They are not, in two ways. A server that fetches models on demand parses a file named, indirectly, by a request: the client names a repository, the server downloads it and reads it. And a model store on disk can be written by more than its owner, a shared directory, a mounted volume, a mirror. A **model file attack surface** is everything a reader of a model file does on the file's word: the code it may run, the memory it reserves from declared counts, the offsets it follows, the depth it recurses to and the arithmetic it does on sizes the file chose. Each of the checkpoint formats of Volume VI exposes some of these, and one exposes the worst.

PyTorch's `torch.save` writes a checkpoint as a ZIP archive holding the raw bytes of each tensor's storage and a description of the object tree in Python's pickle format. Pickle is not a data format in the sense safetensors is. It is a small program for a stack machine, whose instructions build values, and two of its instructions are a function call: GLOBAL names a module and an attribute, and REDUCE calls whatever is on the stack with the arguments below it. A general unpickler imports the module and makes the call. Written out, a pickle that calls a function looks like this.

<!-- worked: security.pickle_program -->
Two pickles of protocol 2, byte offsets from 0. Opcodes: PROTO (0x80) and its version byte; GLOBAL (the byte `c`, then a module name and an attribute name, each ended by a newline byte); BINUNICODE (`X`, a 4-byte little-endian length, that many bytes of UTF-8); TUPLE1 (0x85, the top value becomes a 1-tuple); EMPTY_TUPLE (`)`); REDUCE (`R`, pop the arguments and the callable, push the result of the call); STOP (`.`, the value on top of the stack is the result).

`builtins.print` with the argument `hello`: 31 bytes, `80 02 63 62 75 69 6c 74 69 6e 73 0a 70 72 69 6e 74 0a 58 05 00 00 00 68 65 6c 6c 6f 85 52 2e`.

| offset | opcode | operand |
|---:|---|---|
| 0 | PROTO | protocol 2 |
| 2 | GLOBAL | `builtins` `print` |
| 18 | BINUNICODE | 5 bytes: `hello` |
| 28 | TUPLE1 |  |
| 29 | REDUCE |  |
| 30 | STOP |  |

`collections.OrderedDict` with no arguments: 30 bytes, `80 02 63 63 6f 6c 6c 65 63 74 69 6f 6e 73 0a 4f 72 64 65 72 65 64 44 69 63 74 0a 29 52 2e`.

| offset | opcode | operand |
|---:|---|---|
| 0 | PROTO | protocol 2 |
| 2 | GLOBAL | `collections` `OrderedDict` |
| 27 | EMPTY_TUPLE |  |
| 28 | REDUCE |  |
| 29 | STOP |  |

| global | a general unpickler | the allow-list reader |
|---|---|---|
| `builtins.print` | imports `builtins` and calls `print`('hello') | an empty record (nothing imported, nothing called) |
| `collections.OrderedDict` | imports `collections` and calls `OrderedDict`() | an empty ordered mapping |
<!-- /worked -->

The first program is 31 bytes, and a general reader of it imports `builtins` and calls `print('hello')`. Nothing restricts the module or the function to anything so harmless: the same 31 bytes with another module name and another argument run a command of the file author's choosing with the rights of the process that loads it. That is why loading an untrusted pickle is running untrusted code, whatever the file's extension says.

The defence is not to inspect the program for danger, which is a contest the inspector loses, but to give the program a machine on which danger cannot be expressed. brain's reader, `crates/checkpoint/src/torchpt.rs`, interprets the subset of pickle instructions that `torch.save` emits for state dictionaries and never imports or calls anything. GLOBAL pushes the pair of names as a value. REDUCE (and NEWOBJ) consult a closed list: `collections.OrderedDict` builds an ordered mapping, `torch._utils._rebuild_tensor_v2` and `_rebuild_tensor` build a tensor view from a storage, an offset, a size and a stride, `_rebuild_parameter` unwraps the tensor it holds, and any other pair, which in a real checkpoint is a module class whose object was pickled whole, becomes an empty mapping that the BUILD instruction may fill with the object's saved fields. An unknown instruction is an error naming its offset. The worst the 31-byte program can do to this reader is produce an empty record with no tensors in it, the last column of the table. The reader of the table, reduced to the seven opcodes listed, is this routine over the list of disassembled instructions:

```text
function read_pickle(ops: list<op>) -> value:
    stack = []                                   # values: a pair of names, a text, or a tuple of a given length
    for op in ops:
        if op is PROTO:       continue           # the version byte changes nothing here
        if op is GLOBAL:      append (op.module, op.name) to stack
        if op is BINUNICODE:  append a text to stack
        if op is EMPTY_TUPLE: append a tuple of length 0 to stack
        if op is TUPLE1:      pop the top; append a tuple of length 1 to stack
        if op is REDUCE:
            args = pop the top;  callee = pop the top
            if callee is (collections, OrderedDict):  append an empty ordered mapping to stack   # arguments ignored
            else if callee is another listed pair:    refuse, naming the pair and the argument count it got (needs at least four)
            else:                                     append an empty record to stack            # nothing imported or called
        if op is STOP:        return pop the top
    refuse: no STOP
```

The ZIP container is read the same way (`crates/checkpoint/src/zipread.rs`): `torch.save` writes every entry uncompressed, so a compressed entry is refused rather than inflated, which closes the decompression bomb before it opens.

Refusing to execute is necessary and not sufficient, because a data-only reader still allocates on the file's word. A tensor in a pickle is a view: a size, a stride per axis and an offset into a storage, all three chosen by the file. A stride of zero is legal, it repeats one element, and with it a file of a few bytes can describe a tensor of $2^{40}$ elements over a storage of one. A reader that materialises the view as a contiguous array of 4-byte floats then allocates four tebibytes.

<!-- worked: security.stride_views -->
A tensor of a pickled checkpoint is a view of a storage: a size, a stride per axis and an offset, all chosen by the file. The element count is the product of the sizes, formed axis by axis, and a partial product of 2^64 or more refuses the view; the last element read is offset + sum over the axes of (size - 1) x stride. A view is accepted when its count is at most the storage's elements and its last element is inside the storage; the count is checked first, so an overflowing or oversized view is refused before anything is allocated.

| size | stride | offset | storage elements | elements declared | last element read | outcome |
|---|---|---|---:|---:|---:|---|
| [2, 3] | [3, 1] | 0 | 6 | 6 | 5 | accepted |
| [3] | [2] | 0 | 5 | 3 | 4 | accepted |
| [4] | [0] | 0 | 4 | 4 | 0 | accepted |
| [3] | [3] | 0 | 6 | 3 | 6 | refused: reads element 6, past the storage |
| [2, 2] | [2, 1] | 2 | 6 | 4 | 5 | accepted |
| [1048576, 1048576] | [0, 0] | 0 | 1 | 1,099,511,627,776 | - | refused: 1,099,511,627,776 elements over a storage of 1 |
| [4294967296, 4294967296, 0] | [0, 0, 0] | 0 | 1 | over 2^64 | - | refused: the element count overflows 64 bits |
<!-- /worked -->

The rule in `gather` (`crates/checkpoint/src/torchpt.rs`) is the table's: the element count is a checked product of the sizes, and a view of more elements than its storage holds is refused before anything is allocated; every element is then read with a bounds check, so a stride that walks off the end is an error and not a read past the buffer. A legal broadcast such as size $[4]$ with stride $[0]$ passes, since it holds no more elements than its storage. As a routine, with the checks in the order the table applies them:

```text
function view_fits(size: list<int>, stride: list<int>, offset: int, S: int) -> bool:
    require len(size) == len(stride) and offset >= 0 and S >= 0
    count = 1
    for d in size:
        count = count * d
        if count >= 2^64:
            return false                  # the product does not fit 64 bits, even if a later size is 0
    if count > S:
        return false                      # more elements than the storage holds: refused before allocating
    if count == 0:
        return true
    last = offset
    for i in 0 .. len(size):
        last = last + (size[i] - 1) * stride[i]
    return last < S                       # the last element read lies inside the storage
```

Commit `00d6b03f6` (7 November 2026) added the element-count check, and its message names the failure it closes: "a pickled tensor with zero strides expanded a small storage into an allocation of any size".

The same commit closed the declared-count problem in the GGUF reader, the single-file quantized format of Volume IV, Chapter 12. A GGUF header declares how many tensors and metadata entries follow, and a reader naturally reserves room for that many before reading them. In Rust a reservation the allocator cannot satisfy ends the process rather than returning an error a request could catch (the commit's own words: "a failed reservation aborts the process, no request isolation catches it"), so a header count is a direct line to the process's life. The bound is that no declared count can claim more items than the bytes left in the file could hold, given the fewest bytes one item can occupy:

```text
function reserve_items(n: int, left: int, least: int) -> int:
    require n >= 0 and left >= 0 and least >= 1
    return min(n, left // least)          # never more items than the remaining bytes can encode
```

<!-- worked: security.count_reservation -->
A reader reserves room for the n items a header declares before it reads them. 'Bytes left' is the file length minus the position where the declared items begin (byte 24 for the tensor records of a GGUF file with no metadata entries, after its 4-byte magic, 4-byte version and two 8-byte counts; byte 4,096 for the array); every declared item takes at least the given number of bytes of the file (a tensor record of a GGUF file at least 24: an 8-byte name length, a 4-byte dimension count, a 4-byte type and an 8-byte offset; a metadata array item at least 1). The naive reservation is n records of 24 bytes each, the size a reader gives every item it reserves, whatever the item occupies in the file (so the array row is priced as 2^62 records); the bounded one is min(n, floor(bytes left / least bytes per item)) records.

| header | bytes left | least bytes per item | items declared | naive reservation (bytes) | bounded reservation (items) |
|---|---:|---:|---:|---:|---:|
| 1,024-byte file declaring 2^40 tensors | 1,000 | 24 | 1,099,511,627,776 | 26,388,279,066,624 | 41 |
| 1,024-byte file declaring 3 tensors | 1,000 | 24 | 3 | 72 | 3 |
| 1 MiB file declaring a metadata array of 2^62 items | 1,044,480 | 1 | 4,611,686,018,427,387,904 | 110,680,464,442,257,309,696 | 1,044,480 |

A nested metadata array costs the file 12 bytes per level (a 4-byte element type and an 8-byte count), so a file of L bytes can spell floor(L / 12) levels; a reader that recurses once per level needs that many stack frames unless it stops at a fixed depth.

| file bytes | levels the file can spell | levels read before refusing |
|---|---:|---:|
| 1,024 | 85 | 8 |
| 1,048,576 | 87,381 | 8 |
| 1,073,741,824 | 89,478,485 | 8 |
<!-- /worked -->

A 1,024-byte file that declares $2^{40}$ tensors asks a naive reader for 26,388,279,066,624 bytes and the bounded one for 41 records; reading then fails cleanly at the end of the file, long before the 42nd. `Cursor::reserve` in `crates/checkpoint/src/gguf.rs` is this routine, called with 24 bytes for a tensor record, 8 for each dimension of a record and 1 for an item of a metadata array. Its sibling is depth. A GGUF metadata array may hold arrays, each level costing the file only 12 bytes, so a 1 MiB file can spell 87,381 levels, and a reader that recurses once per level runs out of stack. brain's reader stops at `MAX_ARRAY_DEPTH`, 8 levels (real files nest one). The safetensors reader got the remaining checks of its header in the same commit: `data_offsets` must be exactly two integers, in order, inside the data, and the extent must equal the bytes of the declared shape and type, so the $[3, 5]$ F32 tensor of the recall question must span exactly 60 bytes or the file is refused by name. Chapter 1 of this volume showed the truncation checks of the mapped reader; these are their counterpart for a header that lies rather than one that stops.

One class of mistake hides inside correct-looking bounds: arithmetic on sizes the file chose. A check `start + size <= end` is right in mathematics and wrong in 64-bit integers, where a size near $2^{64}$ makes the sum wrap around to a small number that passes. Commit `e73f050b3` (6 November 2026) records this exact failure in the reader of the MP4 video container (the file format of the MPEG-4 standards of the Moving Picture Experts Group) that an edit request's input clip goes through: a box (MP4's length-prefixed record) with a 64-bit size near $2^{64}$ "wrapped the bound check and sent the box walk back to the start of the file forever, growing a Vec until the process died; a 40-byte file was enough". The walk is short enough to run by hand.

<!-- worked: security.wrapping_walk -->
A 40-byte file of two length-prefixed records: at byte 0 a record of 24 bytes, at byte 24 a record whose 64-bit size is 2^64 - 24 = 18,446,744,073,709,551,592. The walk starts at byte 0, records each position it visits and moves by the declared size; a record must lie inside the file. Eight steps are shown.

| arithmetic | positions visited | outcome |
|---|---|---|
| sum wraps modulo 2^64 | 0, 24, 0, 24, 0, 24, 0, 24 | still walking after 8 records |
| sum checked | 0, 24 | refused: the record at byte 24 overruns the file |
<!-- /worked -->

With wrapping arithmetic the second record's end is $24 + (2^{64} - 24) = 2^{64}$, which wraps to 0, a position inside the file, and the walk returns to the start forever. With the sum checked, the same record is refused as overrunning the file. `children` in `crates/videocodec/src/mp4.rs` now computes `at.checked_add(size)` and refuses a box whose end is past its parent. brain also makes this class of mistake fail loudly everywhere it parses hostile bytes: the release profile in the workspace `Cargo.toml` turns on overflow checks for exactly the crates that read requests and files (the HTTP and D-Bus surfaces, the media codecs, the checkpoint, model store and data readers), with a comment that states the trade: a size that wraps "would otherwise slip past the bound it was checked against; with the checks it panics, inside the request that sent it". A panic in one request's task ends that request; a wrapped bound can end anything.

The last surface of a model file is where its names point. A model store keyed by names from a network must never let a name climb out of it, and Chapter 1 showed both halves of that guarantee: the grammar of a model reference refuses `.` and `..` segments, and a manifest whose role path is absolute or contains `..` is refused. The rule is the same as for parameters that name paths: the file is allowed to choose bytes, never places.

## When a local caller hands over a file descriptor

The D-Bus surface moves bulk data differently from HTTP. A local program passes a file descriptor for each input blob, a handle to memory or a file, so that a large image travels without being copied through the bus. That is efficient, and it hands the server something whose contents the caller still controls. Chapter 1 noted the contract that every file mapping rests on: if the file shrinks while it is mapped, reading a page past the new end raises SIGBUS, the bus-error signal, and the default action of that signal ends the process. A loader can promise itself that its weight files are not modified while mapped. A server cannot promise that about a descriptor a caller still holds.

Commit `2be2e79b7` (6 November 2026) records that the first version mapped the caller's descriptor to read it, "so a caller that shrank its own unsealed fd during the copy killed the server with SIGBUS". The fix (`read_fd_to_vec` in `crates/dbus/src/fd.rs`) reads a regular file, which an anonymous memory file is, with `pread`, a read at an explicit offset that returns fewer bytes, not a signal, when the file has shrunk; only a DMA (direct memory access) buffer, memory a device reads directly and whose size its exporter fixes, is still mapped. The size is checked against `MAX_FD_BLOB_BYTES`, 256 MiB, before anything is read, and a call carries at most 16 descriptors and 1 GiB (gibibyte, $2^{30}$ bytes) together. The test `a_caller_shrinking_its_fd_mid_read_cannot_kill_the_server` truncates and regrows an 8 MiB descriptor in a loop on another thread while the server reads it 300 times, and asserts only that every read ends in data or an error. The outputs go the other way as sealed anonymous memory files (`memfd_seal`): the seals forbid shrinking, growing and writing, so a client that maps a result can trust its size, which is the same contract seen from the receiving side.

## Who is calling, and whose job is this

Bounds stop one request from costing too much. They say nothing about who may make requests, or about what one client may see of another's. Those are questions of identity, and they begin with the key.

An HTTP inference surface identifies its caller by a secret in a header: `Authorization: Bearer <key>` for the OpenAI and OpenRouter dialects, `x-api-key` for the Anthropic one (`presented_key` in `crates/apiserve/src/auth.rs`). brain generates a key per surface as `sk-brain-` and 32 hexadecimal digits drawn from two random 64-bit numbers, 128 random bits (`random_key` in `crates/apiserve/src/surface.rs`). That many bits put guessing out of reach, provided the server does not help. A comparison that stops at the first differing character helps: the time it takes grows with the number of leading characters that are right, and a sender who can measure it learns the key one character at a time.

<!-- worked: security.key_guessing -->
A key is `sk-brain-` followed by 32 hexadecimal digits (128 random bits), 41 characters in all. An early-exit comparison stops at the first character that differs, so it compares min(k + 1, length) characters of a guess whose first k characters are right; a constant-time comparison compares every character whatever the guess.

| leading characters right | compared, early exit | compared, constant time |
|---|---:|---:|
| 0 | 1 | 41 |
| 9 | 10 | 41 |
| 20 | 21 | 41 |
| 40 | 41 | 41 |
| 41 | 41 | 41 |

With a perfect timing oracle (the comparison count is observed exactly) the digits are found one at a time: at most 16 guesses per digit, 512 in all, and 272 on average when the right digit is equally likely to be any of the 16 tried in a fixed order: a digit costs 1 to 16 guesses, counting the guess that matches, which is (1 + 2 + ... + 16) / 16 = 8.5 on average, times 32 digits (the 16th guess of a digit could be deduced untested; the maximum counts it). Without the oracle a guess is one of 16^32 = 2^128 = 340,282,366,920,938,463,463,374,607,431,768,211,456 keys, and 170,141,183,460,469,231,731,687,303,715,884,105,728 guesses are needed on average.
<!-- /worked -->

With a perfect timing oracle, 512 guesses at most recover a key that would otherwise need $2^{127}$ on average. A real oracle is noisy and each guess must be repeated many times to read it, which multiplies the 512 by a large factor; the product is still nowhere near $2^{127}$. A constant-time comparison compares every byte whatever the guess, accumulating the differences instead of returning at the first:

```text
function ct_equal(a: bytes, b: bytes) -> bool:
    if len(a) != len(b):
        return false                       # the length is not a secret for keys of a fixed format
    acc = u8(0)
    for i in 0 .. len(a):
        acc = acc or (a[i] xor b[i])       # no branch depends on where the first difference is
    return acc == 0
```

`key_matches` in `crates/apiserve/src/auth.rs` uses the `subtle` crate's constant-time equality for exactly this, and the authenticator for several named callers compares the presented key with every caller's key, keeping the match without stopping, so the time also says nothing about which caller a key belongs to. Commit `ea82ba1b6` (31 July 2026), the record of an audit of the HTTP surface, made the change from a plain `==`, and its test `wrong_key_of_equal_length_is_401_correct_is_200` sends a wrong key of the right length. The same audit made two other rules about the key that matter as much as the comparison. The file that receives generated keys is created readable by its owner only, written beside its final name and renamed over it so no reader ever sees it with looser permissions (`write_private`); and once the key is in such a file, the startup line on standard output names the file instead of repeating the key, because standard output ends up in logs that other users of a machine can read (`Surface::announcement`).

A key answers "is this caller allowed in". A server with several clients also needs "which one is it", because the things a server keeps between requests, background jobs, streaming sessions, results waiting to be fetched, belong to someone. **Tenant isolation** is the property that one client (a tenant of the shared server) can neither see, change nor exhaust what belongs to another: its jobs and their results, its sessions, its share of the server. It needs three things: an identity for every request, a scope that every piece of kept state is looked up through, and a budget per identity.

With one static key per surface, every holder of the key is the same identity. brain's `StaticKey` authenticator says so in its job scope, the empty string for everyone, and that was a real weakness while the keys were shared among applications. Commit `bb0e0f93b` (6 November 2026) states it: "One key per dialect was the whole security identity: every holder could run every served action, saw every background job, and had no share of its own, so one compromised application could occupy every lane." Its answer is `Callers` (`crates/apiserve/src/auth.rs`), declared by the operator as `--caller NAME:ROLE:KEYFILE[:MAX]`. Each caller has its own key, a role, a name that becomes its job scope, and an in-flight budget, 16 calls by default. The roles are ordered, each including the ones before it: `inference` runs actions that do not train, `training` may also run training actions when the server serves them at all (`--training`), and `admin` may also make the server fetch a model it does not hold. The role is checked at the one place every HTTP call is dispatched, `Pending::begin` in `crates/apiserve/src/bridge.rs`, so no route can reach a model around it; training actions are refused there with 403 ("forbidden") unless the server was started with `--training`, and the same commit found that several training actions of the catalogue had been served to any key-holder until they were marked as training and the check covered them all. The budget is taken at the same place and held until the call is settled, a background job's whole run included; past it, the caller gets 429 and the other callers are untouched.

<!-- worked: security.caller_shares -->
Three named callers and their in-flight budgets. A call holds one slot of its caller's budget from the moment it is let in until it is settled, a background job's whole run included. The edge admits 256 requests at once across all callers. The share is the budget over 256 as a percentage, printed exactly.

| caller | role | in-flight budget | share of the edge (%) |
|---|---|---:|---:|
| assistant | inference | 16 | 6.25 |
| batch | inference | 4 | 1.5625 |
| operator | admin | 2 | 0.78125 |

The budgets add to 22, below 256, so the calls these budgets let run cannot fill the edge by themselves. A request over its caller's budget is refused with 429 at once, and holds an edge permit only for the moment that takes.
<!-- /worked -->

The scope is what makes kept state private. A background job of the HTTP surface is found by the pair of the caller's scope and the job's identifier, so another caller asking for an identifier that exists gets "no such job", the same answer as for one that does not, and learns nothing (`AsyncJobs::view` and `cancel` in `crates/apiserve/src/capabilities.rs`; the test `a_caller_sees_and_cancels_only_its_own_jobs` in `crates/apiserve/tests/it/callers.rs`). The identifiers are random version-4 UUIDs (universally unique identifiers, 122 random bits), but the isolation does not rest on that.

<!-- worked: security.job_ids -->
A caller tries to name another caller's job. 1,024 jobs are live (the most one HTTP surface holds). The first column is how the server numbers jobs and finds them; the second is the chance that one guess names a live job; the third is whether naming it is enough to read or cancel it.

| identifiers | chance a guess names a live job | a named job can be touched |
|---|---|---|
| a counter from 1, no owner | 1 for every number below its own | yes |
| random, 122 bits (a version 4 UUID), one scope for every key-holder | 1,024 / 2^122 = 1.926e-34 per guess | yes |
| random, 122 bits, looked up with the caller's scope | 1.926e-34 per guess | no: another scope's job is 'no such job' |
<!-- /worked -->

The three rows are three designs. A counter makes every other job nameable by anyone who has seen one number. Random identifiers make a guess hopeless, $1.926 \times 10^{-34}$ per try with 1,024 live jobs, but an identifier that leaks (in a log, a shared link, a screenshot) is then a capability anyone can use. A scoped lookup makes the identifier useless outside its owner's scope whether it leaks or not. Commit `2be2e79b7` extended the same scope to state that models keep: the streaming transcription sessions of one model were named by the client, so another caller who guessed a name could read or advance the stream; they are now keyed by the caller's scope, which the HTTP dispatch stamps on every invocation (`Invocation::scope`), and at most 256 are open. The D-Bus surface draws its boundary elsewhere. Its jobs are numbered by a counter and `Cancel` takes the number, because on the bus the unit of trust is whoever the bus policy lets call the service at all: on a session bus that is the user's own programs, and on the system bus the policy brain ships (`scripts/build/com.swedishembedded.Brain1.conf`) lets only root and members of the `brain` group call it, with the service itself owned by an unprivileged `brain` user so that the code that parses media and model files never runs with root's rights.

Isolation also covers what a refusal says. An error message is output to the caller, and an internal error carries what a stranger should not see: a weight file's path, a panic message with a source location, the text of an operating-system error. brain sorts every failure through `capability::ReplyError` (`crates/capability/src/failure.rs`) for both transports: the caller gets a class and a fixed message, the operator's log gets the detail. A failed automatic fetch is reported with exactly the error of a model that does not exist (`ensure_and_recheck` in `crates/apiserve/src/bridge.rs`), so the response is not an oracle for which repositories the server tried to reach and why they failed.

One more boundary runs through the text itself, and Volume V drew it. A chat request's messages are rendered into a prompt between the model's control tokens, and a client who can type a control token can forge a turn of the conversation. Volume V, Chapter 2, "Tokenization", showed how brain escapes such literals in client text so that only the template can emit a control token (commit `226777cc3`). It is the same principle as the stores and the parameters: data from a caller may choose bytes, never the structure that interprets them.

## Keeping the wire private

The key travels in a header, the prompt and the answer in bodies. On the loopback interface of one machine they never leave it. On any network, a private one included, every host that can see the traffic can read them, and a key that has been read is a key that has been given away. **Transport security** is the protection of what crosses the network between a client and a server: confidentiality (no one else can read it), integrity (no one can alter it unnoticed) and authentication of the server (the client is talking to the machine it meant), in practice provided by TLS (Transport Layer Security), with the server proving its identity by a certificate the client verifies.

brain's rule is stated in `crates/apiserve/src/tls.rs`: a surface bound to any address other than loopback is refused without TLS, unless the operator states by flag that something else encrypts the path. The check, `check_exposure`, runs before any model loads, so a misconfigured server fails at once with a message that names the two remedies, not after an hour of warm-up.

<!-- worked: security.exposure -->
An IPv4 (Internet Protocol version 4) address is loopback when its first byte is 127; an IPv6 (version 6) address is loopback only when it is `::1` (an IPv4 address written in IPv6 form, `::ffff:127.0.0.1`, is not). The surfaces are served when TLS is configured, or the operator states that something else encrypts the path, or every address is loopback.

| addresses bound | TLS | plaintext stated | outcome |
|---|---|---|---|
| `127.0.0.1` | no | no | serves |
| `127.0.0.1`, `::1` | no | no | serves |
| `127.8.9.10` | no | no | serves |
| `0.0.0.0` | no | no | refused before anything loads |
| `::` | no | no | refused before anything loads |
| `::ffff:127.0.0.1` | no | no | refused before anything loads |
| `127.0.0.1`, `192.168.1.5` | no | no | refused before anything loads |
| `192.168.1.5` | yes | no | serves |
| `192.168.1.5` | no | yes | serves |
<!-- /worked -->

Two rows deserve a second look. Binding to `0.0.0.0` or `::`, "every interface", is refused in plaintext even though the operator may only mean to reach it locally: an unspecified address listens on every network the machine has. And `::ffff:127.0.0.1`, the loopback address written in IPv6 form, is refused too, because the standard library's test for IPv6 loopback accepts only `::1`; the rule errs toward refusing, the safe side for a rule like this. With a certificate (`--tls-cert` and `--tls-key`, both PEM files, the Privacy-Enhanced Mail text encoding that certificates and keys are stored in), every surface is served over TLS with the `rustls` library on its `ring` cryptography provider, the protocol versions it considers safe, and HTTP/1.1 as the only protocol offered in the negotiation (ALPN, Application-Layer Protocol Negotiation), matching the listener. The handshake must finish within the 10-second header deadline, so a peer that opens a TLS session and stalls is closed like one that stalls in its headers. A certificate and key that do not match, or a file that holds none, are refused at startup with the file's name and never its contents (`Tls::from_pem_files`). The test `a_tls_surface_answers_a_verifying_client_and_nothing_in_plaintext` in `crates/apiserve/tests/it/tls.rs` sends a request through a client that verifies the certificate and gets its 200, sends the same request in plaintext to the same port and gets no HTTP answer at all, and checks that a client trusting a different certificate refuses the server. Commit `c7d4ae176` (6 November 2026) introduced all of it, and its message is plain about the cost: "Breaking for an operator who binds a non-loopback address today: add the certificate, or --insecure-plaintext."

Transport security also has a local half: the server's secrets at rest. The connection record a running server publishes so that local clients can find it (its addresses and, without named callers, its keys) is written with `write_private` into a directory only its owner may enter, and `private_dir` refuses a directory that others may write, because a client trusts the address a record names and a record someone else could replace would send it elsewhere (`crates/apiserve/src/surface.rs`, test `a_record_directory_is_one_only_this_user_writes`).

## How brain keeps the defences from decaying

Security of a server is not a feature that is finished; every new route, parameter and model is a new place for a bound to be missing. brain keeps three kinds of evidence against that. The constants live next to the code they bound and are named after what they bound, as the tables above show. The checks are tests that run against the real entry points, the router in-process (`crates/apiserve/tests/it/api.rs`) and a served socket (`tests/e2e/api_conformance.bats`, which runs the server with a mock model and checks the 401 for unknown paths, the 413, the absence of keys in every response body and the 429 of admission). And the rules that a review cannot hold in its head are gates that fail on a missing bound: the served-bounds test with its shrink-only allow-list, the catalogue-wide test that fails on a training action not marked as one (`crates/catalog/tests/it/served_training.rs`) and the one that fails on a served action that lets a caller name a path (`crates/catalog/tests/it/served_paths.rs`).

The history shows why the third kind matters. The July audit, commit `ea82ba1b6` (31 July 2026), fixed the key comparison, the error bodies, the body limit and the key file, and recorded what it had checked and found sound. The commits of 6 and 7 November 2026 cited in this chapter found that the server had no header deadline, decoded media on its asynchronous workers, mapped caller descriptors, parsed model files on their word, served training to any key-holder and accepted parameters of any size. Some of those places did not exist in July: image parts in chat requests arrived in August (commit `c0e56d374`), the generic `/v1/run` and `/v1/jobs` routes in October (commit `3d6989846`). Others existed and were not on the checklist, the connection deadlines among them. A bound is a statement about one quantity; what keeps a server safe is the habit, and the test, that asks the question of every new quantity as it arrives.

## Summary

**Hostile input** is input from a source that may intend harm, and its first danger is **resource exhaustion**: a quantity chosen by the sender (a pixel count, a tensor count, a stride, a nesting depth, a silence on a socket) decides how much memory, how many threads, connections or device lanes the server spends, and the ratio of the server's cost to the sender's can reach millions, as a 47-byte PNG header that declares 51 gigabytes of floats shows. An **input bound** is the cure: a ceiling on each such quantity, sized from the largest legitimate use, checked before the work and refused by name, with a shared budget bounding the next item by what is left of it, with sums of declared sizes checked against overflow, and with concurrency bounds so that the worst case of the whole server is the product of its constants. A **model file attack surface** is the same problem reached through a file: a pickle is a program and must be read on a machine that cannot call anything, and a data-only reader still has to bound counts by the bytes left, views by their storage, recursion by a depth and arithmetic by checked sums. **Tenant isolation** gives every request an identity, every kept item a scope and every identity a budget, so that one compromised client holds its own share and sees its own jobs; and **transport security** keeps keys and data off any network in the clear, with plaintext refused off loopback before the server loads anything.

The limits are those of a single server. Bounds make an attack cost the attacker in proportion to its duration but cannot stop one who is willing to pay; a rule about who may call is only as good as the keys and the bus policy behind it; and every new route or model needs its quantities bounded in turn, which is why the rules are tests. The next chapter turns from what a server must refuse to how an operator sees what it is doing: its traces, its statistics, its state and a shutdown that leaves nothing half-done.

## References

- P. Deutsch. *DEFLATE Compressed Data Format Specification version 1.3*. RFC 1951, 1996.

## Check your understanding

1. A colleague proposes to protect an image endpoint by limiting the request body to 1 MiB. Using the deflate bound, explain why this does not bound the server's decoded memory, and name the quantity that must be bounded instead.
2. A server checks a request's total pixel count after decoding each image, with a per-image limit of 36,000,000 and a budget of 64,000,000 pixels. What is the most pixels it can decode before refusing, and what single change brings that down to the budget?
3. Find the error: "Our reader checks `offset + size <= file_len` before following each record, so a record can never point outside the file."
4. A pickle-reading library offers a "safe mode" that scans the instruction stream and refuses any GLOBAL whose module is on a list of dangerous modules. Why is an allow-list of constructors, applied by an interpreter that never imports, a stronger design?
5. Two clients share one static key. Client A starts a background job and receives a random 122-bit identifier, which later appears in a shared log. What can client B do with it under a design with one scope for all key-holders, and what under brain's named callers?
6. An operator binds a surface to `192.168.1.5` behind a firewall that admits only the office network, without TLS. brain refuses to start. Which assumption of the operator does the rule reject, and what are the two ways to proceed?
7. Predict what happens to a server, keyed or not, when a peer opens 2,000 connections to one surface and sends nothing, under brain's default limits.

## Exercises

1. **[Calculate]** A PNG declares 12,000 x 8,000 pixels of 8-bit RGB with no interlacing. Find its pixel stream bytes, the least deflate bytes by the 1,032 bound, the base64 characters of that, and the bytes of its decode into 32-bit floats. Is it accepted by brain's per-image limit?
2. **[Calculate]** With a per-image limit of 36,000,000 pixels and a request budget of 64,000,000, find a sequence of at most 8 images that makes a check-after-decode server hold the most pixels, and give that number. Repeat for a per-image limit of 10,000,000 and the same budget.
3. **[Calculate]** A GGUF file of 4,096 bytes declares $2^{50}$ tensor records of at least 24 bytes each, to begin at byte 24. How many records does the bounded reservation allow? How many levels of nested metadata arrays can a file of 64 KiB spell at 12 bytes a level?
4. **[Calculate]** A surface accepts bodies of 64 MiB within 300 seconds. What is the slowest link, in bits per second, that can upload the largest body in time? If the body deadline were 60 seconds, what would it be? Which clients does a shorter deadline turn away?
5. **[Calculate]** Three callers have budgets of 32, 32 and 8 calls. What share of the 256 edge permits can each hold at most, and how many requests can be in flight from callers not on the list?
6. **[Implement]** Reproduce `security.pixel_budget`: the two rules, the per-image limit, the request budget and the image cap, applied to the four sequences listed. Your decoded counts, pixel totals and outcomes must equal the table.
7. **[Implement]** Reproduce `security.stride_views` with `view_fits(size, stride, offset, S)` as the chapter states it (partial products of $2^{64}$ or more refuse, the count is compared with the storage before the last element is computed). Every row of your table must agree.
8. **[Implement]** Reproduce `security.pickle_program`: write the two pickles byte by byte, disassemble them into the printed tables, and implement the allow-list reader for the seven opcodes listed. Your bytes, offsets and outcomes must equal the tables. Then add a third pickle that calls `collections.OrderedDict` with a 1-tuple argument holding a text, and decide what your reader must do with it. Where this exercise leaves a case open, follow the reader sketched in the chapter: REDUCE of a pair outside the list gives an empty record whatever its arguments, and STOP returns the top of the stack.
9. **[Implement]** Reproduce `security.wrapping_walk` for the 40-byte file, then construct a different file of at most 48 bytes whose wrapping walk visits three distinct positions forever, and show that the checked walk refuses it.
10. **[Investigate]** Read `crates/apiserve/src/listener.rs` and `crates/apiserve/src/lib.rs` and list, in the order a request meets them, every check between the accept call and the handler's first read of the body, naming the code that makes each and the status a client receives when it fails.
11. **[Investigate]** Read `crates/dbus/src/fd.rs` and the call path of an input descriptor in `crates/dbus/src/service.rs`. Which checks run before any byte of a descriptor is read, and which bound decides for a call of 16 descriptors of 200 MiB each?

## Key terms

- **hostile input**: any input a server receives from a source whose aim may be to harm it (request bodies and headers, the timing of a connection, files fetched for a request, descriptors from local programs), so that every quantity it declares is a claim to check rather than a fact. <!-- concept: hostile-input -->
- **resource exhaustion**: an attack or accident in which input makes a server spend a resource shared by all its clients (memory, threads, connections, device lanes or disk) until it cannot serve anyone; measured by how much the server spends per unit the sender spends. <!-- concept: resource-exhaustion -->
- **input bound**: a ceiling on a quantity that input controls and that sizes the server's work, sized from the largest legitimate use, checked before the work is done and refused with an error that names it. <!-- concept: input-bounds -->
- **model file attack surface**: everything a reader of a model file does on the file's word: the code it may run, the memory it reserves from declared counts, the offsets it follows, the depth it recurses to and the arithmetic on sizes the file chose. <!-- concept: model-file-attack-surface -->
- **tenant isolation**: the property that one client of a shared server can neither see, change nor exhaust what belongs to another, provided by an identity for every request, a scope for every piece of kept state and a budget for every identity. <!-- concept: tenant-isolation -->
- **transport security**: protection of what crosses the network between client and server (confidentiality, integrity and authentication of the server), in practice TLS, together with keeping the server's secrets out of logs and readable files. <!-- concept: transport-security -->

## Answers

**Recall.**

1. Each tensor's name, number format (type), shape and the byte range of its data, in a header that can be parsed without the data. A safetensors file begins with an unsigned 64-bit little-endian length $n$, then $n$ bytes of JSON mapping each name to its type, shape and `data_offsets` (start and end counted from the first byte after the header); tensor $x$ is the bytes $[8 + n + \text{start}, 8 + n + \text{end})$ of the file.
2. $3 \times 5 = 15$ elements of 4 bytes: 60 bytes, so `end - start` must be 60.
3. At the door, before any work is spent on a request, it decides whether to take the request in now, let it wait or refuse it, from what the request needs and what the server has. A queue without a deadline grows without bound when requests arrive faster than the device serves them, and a request at its back is answered after its sender has stopped waiting while the device still does the work; a wait with a deadline refuses what cannot start in time, so the device spends its work on answers that will be read.
4. The model's name, the input (messages or a prompt, possibly with media), and parameters such as the token budget and the sampling settings, as a JSON body; the key travels in a header, `Authorization: Bearer <key>` in the OpenAI dialect and `x-api-key` in the Anthropic one.

**Check your understanding.**

1. A flat picture compresses at up to 1,032 to 1, so 1 MiB of PNG can declare and actually contain over a billion bytes of pixel stream, and each 3 bytes of it become 12 bytes of floats. The body limit bounds the encoded size; the decoded memory is set by the declared pixel count, which must be bounded per image from the header and per request by what is left of the budget.
2. The budget, plus one full image: $64{,}000{,}000 + 36{,}000{,}000 = 100{,}000{,}000$ pixels (a total exactly at the budget is not over it). Checking each image's header against $\min(36{,}000{,}000, \text{budget} - \text{used})$ before decoding it caps the total at the budget.
3. In 64-bit integers `offset + size` wraps when `size` is near $2^{64}$, so the sum can be small and pass the check while the record's real end is far past the file. The check must be done without overflow (a checked addition, or `size <= file_len - offset` after confirming `offset <= file_len`).
4. A list of dangerous modules is a guess about everything an attacker might use and fails on the first one it omits; an allow-list of constructors names the few things a checkpoint legitimately needs, and an interpreter that never imports cannot call anything else even when the list is wrong, since an unknown name becomes inert data.
5. With one scope, the identifier is all B needs: B can read the job's progress and result and cancel it. With named callers, B's lookups use B's scope, so B is told "no such job" and can do nothing, whether or not B has the identifier.
6. It rejects the assumption that a private network is a trust boundary: every host on the office network can read the key and the traffic. The operator can configure TLS with `--tls-cert` and `--tls-key`, or state with `--insecure-plaintext` that a proxy or a VPN (virtual private network) already encrypts the path.
7. The first 1,024 connections are held, and each is closed when 10 seconds pass without a request head; the other 976 are closed at accept. No key is checked because no head arrives, the process keeps at most 1,024 descriptors for the surface, and other clients can connect again as slots free; holding the surface full takes 102.4 new connections per second.

**Exercises.**

1. Pixel stream: $8{,}000 \times (1 + 36{,}000) = 288{,}008{,}000$ bytes. Least deflate: $\lceil 288{,}008{,}000 / 1{,}032 \rceil = 279{,}078$ bytes, base64 $4 \times \lceil 279{,}078 / 3 \rceil = 372{,}104$ characters. Floats: $96{,}000{,}000 \times 12 = 1{,}152{,}000{,}000$ bytes. It is refused: 96,000,000 pixels are over the 36,000,000 limit, and the refusal comes from the header.
2. With 36,000,000 per image: as in the table, an image of 36,000,000, one of 28,000,000 and one of 36,000,000 make 100,000,000 pixels; no sequence does better, since the total before the last decode is at most 64,000,000 (a total at the budget is not over it) and the last adds at most 36,000,000. With 10,000,000 per image: six images of 10,000,000 and one of 4,000,000 reach 64,000,000 with seven images, and the eighth adds 10,000,000: 74,000,000 pixels.
3. Bytes left $4{,}096 - 24 = 4{,}072$, so $\min(2^{50}, \lfloor 4{,}072 / 24 \rfloor) = 169$ records. A 64 KiB file spells $\lfloor 65{,}536 / 12 \rfloor = 5{,}461$ levels; a reader that stops at 8 reads 8.
4. $67{,}108{,}864 \times 8 / 300 = 1{,}789{,}569$ bits per second (rounded down). With 60 seconds: $8{,}947{,}848$ bits per second. A shorter deadline turns away clients on links between those rates that send the largest legitimate bodies, so the deadline is set from the slowest link the server means to serve.
5. 32/256 = 12.5 percent, 12.5 percent and 8/256 = 3.125 percent. The named callers' budgets add to 72; with named callers declared, a request without a valid caller key is refused 401 at the key check, so no other request can be in flight.
6. The printed table is the check.
7. The printed table is the check.
8. The printed tables are the check. For the third pickle, `collections.OrderedDict` with the argument `('hello',)` must not import or call anything, and a text is not a sequence of key-value pairs, so the reader should refuse it with an error rather than guess a mapping; brain's reader does exactly that (`OrderedDict arg is not a pair sequence`). The reader sketched in the chapter, which ignores the arguments of `OrderedDict`, would need that check added.
9. One answer: records at byte 0 of size 16, at byte 16 of size 16 and at byte 32 of 64-bit size $2^{64} - 32$, in a file of 48 bytes. The wrapping walk visits 0, 16, 32, 0, 16, 32, ...; the checked walk refuses the record at byte 32 because $32 + (2^{64} - 32)$ is not representable.
10. The accept loop's connection slot (`serve`, excess closed with no response); the TLS handshake within the header deadline when configured (closed); hyper's header deadline (closed); the body deadline is armed as a wrapper on the body (a 4xx when it expires during the read); trace and cross-origin layers; the key check, `auth::authenticate` (401, also for unknown routes); on the work routes, the edge's concurrency limit with load shedding (503); then the handler's body extractor with the route's body limit (413).
11. The count of descriptors against 16 and the sum of their sizes from `fstat` against 1 GiB in `build_inv`, then, per descriptor, its size against 256 MiB in `read_fd_to_vec`, all before any read. For 16 descriptors of 200 MiB, each is under 256 MiB but the total is 3,200 MiB, over the 1 GiB per call, so the call is refused before any descriptor is read.
