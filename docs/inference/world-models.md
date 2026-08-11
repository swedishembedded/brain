# World models: playable, action-conditioned simulation

A world model lets you "play" inside a learned simulation of a game or
environment: reset it with a few context frames, step it forward one
discrete action at a time, and it predicts what happens next — rendered
live in a window so you can watch and steer it like a game.

## Capabilities

### Interactive play — world models

Take an action, watch the model predict the next frame, and repeat — a way
to interactively probe what a world model has actually learned, rather than
just reading a loss curve. Frames render live in a window as you steer, so
you can directly see whether the simulation stays coherent as you push it
away from its training distribution.

DIAMOND, an Atari-100k-style world model, is the architecture that's
playable end to end today. You can also record an episode as you play and
replay it later — optionally re-running it live against the model to verify
the two match — or fine-tune the model on a set of your own recorded
episodes. See [the world models page](../models/world-models.md).
