# Reproducibility and run state

`experiment.toml` is immutable experiment intent. `state.json` is mutable progress. `train init` refuses to overwrite an existing experiment; `train run` accepts only runtime controls such as device, iteration count, and `--forever`.

Self-play identifies a game by global game ID and model generation. Every RNG purpose derives from experiment seed, generation, game ID, and a fixed purpose tag—not worker ID—so one- and many-thread execution commit the same replay order and metadata. Replay sampling similarly uses a fixed purpose plus global training step.

Weights are written to a safetensors temporary path whose extension remains `.safetensors`, atomically renamed, then referenced by state. A successful batcher reload advances the evaluation-cache namespace. On restart, the current implementation restores weights only and records replay/optimizer reset instead of overstating continuation fidelity.

