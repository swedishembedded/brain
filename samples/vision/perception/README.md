# sample: vision/perception

An image in, structured perception out, through the public `brain` SDK:
objects with boxes and scores, text read from the image, scene attributes and
relations, and the answer to a form about the picture.

```bash
brain pull Cactus-Compute/mirage
make samples/vision/perception/build
make samples/vision/perception/run ARGS="--image photo.jpg"
```

## What it demonstrates

* `brain::PerceptionPipeline::from_pretrained` resolves Mirage out of the local
  model store with the resolver the CLI uses, and puts it on the device once.
* `describe_with` returns the engine-shaped JSON plus typed accessors
  (`objects()`, `text_lines()`); `extract` answers a form of choice, count and
  text fields with the probability of every candidate.
* Usage accounting rides on the result: decoder tokens and the latency split of
  the batch the request ran in.
* No CLI process and no capability-dispatch server in the loop.

## What it needs

The released checkpoint (`config.json`, `mirage.safetensors`, `mirage.cact`,
about 240 MB). If it is not in the local store the sample prints the
`brain pull` command and exits non-zero instead of downloading behind your
back.

## Options

| flag | default |
|---|---|
| `--image PATH` | *required* |
| `--model ID` | `Cactus-Compute/mirage` |
| `--form JSON` | a one-field animal form |
| `--max-tokens N` | model default |
